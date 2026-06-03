# Strategy A Faithfulness Audit

Date: 2026-04-20

## Scope

Assess current Strategy A against `docs/plans/active/deep-research-report.md` as the scientific source of truth.

## Report contract

- Strategy A target:
  - recover two invariant fields as the smooth near-nullspace of `D = v·∇`
  - solve the smallest nontrivial eigenpairs of `A = D^T W D + mu L`
  - use diagnostics `r_i`, `e_x`, `s`
  - treat Strategy C as a refinement only after Strategy A/B provides a valid foliation-level initialization
  - keep Strategy B available as a characteristic baseline/oracle
- Operator contract:
  - `D` is the discrete transport operator
  - `D^T` is the algebraic transpose / adjoint role used in the least-squares energy
  - `L` is the regularizing Laplacian from `||∇psi||^2`
  - `A` should be symmetric positive semidefinite if built consistently

## Implementation verdict

- Matrix-free operator path is close to the report contract:
  - `TransportOperator3D` implements `D = v·∇`
  - `apply_DT()` is the algebraic transpose of the discrete stencil
  - `LaplacianOperator3D` implements `L = -∇²`
  - `CombinedOperatorA` implements `A = D^T D + mu L` with `W = I`
- Important deviations:
  - the production SLEPc path does not solve the raw matrix-free operator; it solves a 45-color probed assembled matrix `A_pre`
  - exact-object quality and `PsptaEngine` still use legacy `Ly/Lz` self-period semantics when differentiating/interpolating consumed invariants

## Evidence

- PETSc/SLEPc contract audit:
  - SLEPc EPS explicitly supports shell matrices through `EPSSetOperators()`, but `EPS_HEP` is only valid when the operator is actually Hermitian/symmetric.
  - Shift-and-invert targets eigenvalues near the requested shift by solving with `(A-\sigma I)^{-1}`.
  - `EPSComputeError()` reports residual-based error.
  - For eigenvalues near the origin, SLEPc recommends absolute residual convergence rather than the default relative-to-eigenvalue test; the production path now sets `EPS_CONV_ABS`.
  - PETSc `MatComputeOperator()` forms an explicit matrix by applying the operator to columns of the identity, and is only recommended on small problems.
- V100 operator validation:
  - `run_operator_tests`: `D(constant)=0`, adjoint error `0`, `A` symmetry error `0`
  - new PSD probe: minimum random Rayleigh quotient stayed positive on all tested cases
    - `uniform_x`: `171.365 .. 172.851`
    - `layered_x`: `179.157 .. 180.642`
    - `darcy_small`: `106.881 .. 107.707`
- Control foliation recovery:
  - `uniform_x` and `layered_x` exact-object consumed pairs had `mean_abs_alignment = 1.0`
  - mismatch stayed `~1.000` / `~1.020`, so the cross-product direction is right while amplitude/parameterization is wrong
  - expected control `yz` foliation capture:
    - full six-mode Strategy A subspace: `0.863761`
    - best two-mode pair plane: `0.363761 .. 0.431881`
  - interpretation: the recovered eigenspace contains the right foliation information more strongly than any single transported 2-mode pair
- `darcy_small`:
  - best transported pairs kept high directional alignment `0.906 .. 0.987`
  - mismatch remained `1.071 .. 1.074`
  - drift remained `~8.2e-6`
  - failures remained substantial: best `801/158` at `mu=3e-5`
  - subspace similarity vs `mu=1e-5` collapsed with `mu`: `1.0, 0.484, 0.028, 0.043, 0.008`
- Critical fidelity failure in the implemented eigensolver path:
  - the previous production-like path solved a 45-color probed surrogate `A_pre`, not the raw shell-derived operator
  - that surrogate hard-coded a `5x3x3` stencil envelope (`±2` in `x`, `±1` in `y,z`)
  - this is mathematically suspect for `darcy_small`: because `D` uses centered periodic differences in `y,z`, the composed operator `D^T D` can couple `±2` in `y,z` and mixed offsets once `v_y,v_z` are nonzero
  - on `darcy_small`, the probed surrogate shows a measurable symmetry defect `~2.03e-2`
  - the same probed path previously reported negative eigenvalues and very large residuals at every tested `mu`
  - examples:
    - `mu=1e-5`: `eig=[-2.21e-3, -7.24e-3, 9.87e-3]`
    - `mu=1e-4`: `eig=[6.48e-3, -1.57e-2, -2.12e-2]`
    - `mu=1e-3`: `eig=[-1.03e-2, -1.08e-2, 1.28e-2]`
  - this contradicts the positive matrix-free Rayleigh evidence above
  - conclusion: the probed assembled eigensolver realization is not faithfully preserving the intended PSD Strategy A operator on `darcy_small`
- Minimal corrective path now implemented for small/control grids:
  - `SLEPcProductionBackend` uses exact `MatComputeOperator()` assembly for `n <= 50000`
  - the near-zero targeted Hermitian solve now uses `EPS_CONV_ABS`, matching SLEPc's documented recommendation for origin-near spectra
  - the 45-color probed surrogate remains only for larger grids, and remains not proven yet
- Root cause of the apparent small-grid zero cluster:
  - the `MATSHELL` never declared its vector type, so PETSc utilities such as `MatComputeOperator()` were free to create non-CUDA vectors while our `MATOP_MULT` callback only handled `VECCUDA`
  - after setting `MatShellSetVecType(A_shell,VECSEQCUDA)`, the shell/exact explicit audit on `darcy_small` becomes machine-precision tight:
    - exact-action mismatch `~2e-16`
    - exact-Rayleigh mismatch `~0`
    - exact symmetry defect `~1e-15`
  - this proves the previous broad zero cluster was an operator-realization bug, not a scientific feature of Strategy A
- Corrected small-grid exact solve on `darcy_small`:
  - the spurious zero cluster disappears along with the negative-spectrum artifact
  - the corrected first six eigenvalues are strictly positive and form a structured low cluster:
    - `mu=1e-5`: `[5.72e-4, 5.83e-4, 6.20e-4, 7.00e-4, 1.12e-3, 1.19e-3]`
    - `mu=3e-5`: `[1.41e-3, 1.43e-3, 1.46e-3, 1.55e-3, 2.76e-3, 2.85e-3]`
    - `mu=1e-4`: `[4.21e-3, 4.24e-3, 4.27e-3, 4.36e-3, 8.33e-3, 8.41e-3]`
  - among the computed modes, the first four stay in a tight low cluster (`<=1.3*λ0 -> 4`) while the first 4D subspace remains much more stable under `mu` than the leading 2D pair:
    - `prefix2(mu_ref)` drops from `1.0` to `0.357`
    - `prefix4(mu_ref)` stays `0.968 .. 0.993`
  - transport remains scientifically unacceptable as a final gauge, but much less pathological than before:
    - best transported pairs now reach `94/17` failures at `mu=3e-5` and `8/2` at `mu=1e-3`
    - `rel_mismatch` still stays `~1.068 .. 1.072`
    - `drift_max` still stays `~8.3e-6`
  - interpretation: on `darcy_small`, the corrected Strategy A object is no longer a fake zero eigenspace. The useful scientific object is a low invariant subspace (at least four modes among the computed spectrum), not a fixed 2-mode pair.
- 4D-subspace-to-gauge audit on the exact consumed path:
  - host-side linear gauge extraction from the first four low modes can reduce host mismatch below the best pair-plane gauges:
    - `layered_x`: host-best `rel_mismatch ~ 0.9997..1.0000`
    - `darcy_small`: host-best `rel_mismatch ~ 0.9979..0.9985`
  - but those apparent gains do not survive realization as the exact consumed scalar field:
    - `layered_x`: consumed mismatch returns to `~1.021` before transport
    - `darcy_small`: consumed mismatch returns to `~1.0707..1.0717` before transport
  - `prepare()` is not the dominant source of degradation:
    - all controls keep `prepare_drift_max ~ 3e-8`
    - `darcy_small` only grows to `final_drift_max ~ 8.29e-6 .. 8.33e-6` after stepping
  - implication: the current mismatch is primarily a representation / consumed-semantics issue, not a transport-stepping issue
  - the first-4-mode subspace remains the right Strategy A object, but ranking gauges by host linear-combination gradients is not aligned with `PsptaInvariantField::compute_quality()` on the actual consumed object
- consumed-ranked 4D-subspace gauge audit:
  - the search now ranks candidate 2-field gauges by `PsptaInvariantField::compute_quality()` immediately after upload, before `prepare()`
  - this removes the host-ranking false positives:
    - `uniform_x`: consumed-best remains `~1.000`
    - `layered_x`: consumed-best is already `~1.021`
    - `darcy_small`: consumed-best remains `~1.068`
  - representative `darcy_small` transported consumed-ranked gauges:
    - `mu=1e-5`: best `282/54`, `rel_mismatch ~ 1.068`
    - `mu=3e-5`: best `340/68`, `rel_mismatch ~ 1.069`
    - `mu=1e-4`: best `492/81`, `rel_mismatch ~ 1.068`
    - `mu=3e-4`: best `241/50`, `rel_mismatch ~ 1.068`
    - `mu=1e-3`: best `415/78`, `rel_mismatch ~ 1.068`
  - conclusion: consumed-object ranking is the scientifically correct selector, but it still does not expose a usable linear 2-field gauge inside the faithful first-4-mode Strategy A subspace on `darcy_small`

## Updated A->C handoff

- On `darcy_small`, the corrected first-4-mode low Strategy A subspace is now the authoritative scientific output of Strategy A.
- Strategy A should therefore hand off a low invariant subspace, not a privileged fixed pair.
- Strategy C should start from a consumed-object-aware initialization built from that first-4-mode subspace.
- Fixed pair extraction remains a diagnostic / comparison tool only; it is no longer the authoritative Strategy A interface.

## Scientific conclusion

- `uniform_x` / `layered_x`:
  - Strategy A appears to recover the correct invariant foliation / near-nullspace structure
  - current 2-mode consumed parameterization is not yet a usable final streamfunction gauge
- `darcy_small`:
  - not proven yet that the current implemented Strategy A path recovers a final 2-field usable representation
  - the previous blocker was fidelity of the actual eigensolver operator realization (`A_pre`) relative to the intended matrix-free `A`
  - after correcting the shell/exact realization, the remaining blocker is no longer a fake zero cluster; it is that the corrected low spectrum is a stable low-dimensional invariant subspace whose useful gauge is not captured by the current host extraction and exact consumed scalar-field semantics

## Next move

1. Iterate on Strategy A fidelity first.
2. Treat the small-grid exact assembly as the authoritative Strategy A eigensolver path on control problems.
3. Treat the corrected first-4-mode `darcy_small` eigenspace as the scientific handoff object and keep gauge search on the exact consumed object itself.
4. The first consumed-object-aware Strategy C test is now complete:
   - an alternating local fit + Poisson projection refiner can lower mismatch in trial states,
   - but on `darcy_small` every mismatch-reducing trial is rejected because invariance growth is two to three orders of magnitude too large,
   - so the current unconstrained correction-based Strategy C is scientifically insufficient.
5. That corrective step is now partially validated:
   - a first subspace-constrained Strategy C (`SubspaceQuadraticMap`) can lower consumed mismatch on `darcy_small` without the catastrophic invariance blow-up of the unconstrained variant,
   - but the gain is modest (`~1.0685 -> ~1.0673` best) and transport failures remain mixed.
6. The next corrective step is therefore a stronger coefficient-space Strategy C that still preserves the recovered first-4-mode foliation while refining the gauge on the consumed object.
7. Use Strategy B as a baseline/oracle only after the faithful Strategy A-to-C handoff is understood well enough to separate foliation-vs-gauge questions cleanly.
