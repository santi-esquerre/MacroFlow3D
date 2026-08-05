# SF-05 — Multigrid reuse

- State: `active`
- Goal: `Validar y adaptar el multigrilla cell-centered como precondicionador de A(q).`
- Depends on: `SF-04`
- Unlocks: `SF-06`
- Branch: `science/lester-sf05-multigrid-reuse`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf05-multigrid-reuse`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-05T15:11Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Test the priority reuse hypothesis rather than assuming the current flow
multigrid remains a valid symmetric preconditioner for periodic `q=1/K`.

## Preconditions

- SF-04 projected PCG converges with a simple preconditioner.

## In scope

- Generic coefficient naming/coarsening, sign adapter, per-level zero-mean
  projection, and quantitative PCG/MG tests.

## Out of scope

- Replacing multigrid, changing transfer order, or optimizing kernels.

## Files and symbols

- Extend `src/multigrid/MGHierarchy*`, V-cycle, GSRB, residual, coefficient
  coarsening, restriction, and prolongation only where tests require it.
- Add a projected, sign-correct preconditioner adapter.

## Implementation specification

1. Build the hierarchy from cell-centered `q` once and reuse it.
2. Use geometric 2x2x2 coefficient coarsening initially; record that for
   `q=1/K` it equals the inverse geometric coarsening of `K`.
3. Project level RHS, residuals, and corrections without replacing the
   physical smoother.
4. Check preconditioner symmetry and compare PCG iteration counts against
   unpreconditioned projected PCG.

## Expected numerical effect

The preconditioner reduces iteration count while the outer PCG retains the
same converged zero-mean solution.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Relative residual `<=1e-10` on constant and smooth positive `q`.
- At most 100 PCG iterations on the fixed `32^3` and `64^3` suite.
- Iteration growth from `32^3` to `64^3` is no more than 50%.
- MG-preconditioned and reference solutions agree within solver tolerance.

## Regression surface

- Flow MG hierarchy construction, negative-operator preconditioning, transfer
  kernels, and memory ownership.

## Failure and rollback policy

- If reuse fails symmetry or convergence criteria, document the failed
  hypothesis and make the smallest local correction; do not replace MG in this
  increment.
- A need for a new multigrid design requires a new decision record and plan
  revision.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] q hierarchy and sign adapter are explicit.
- [ ] Level projection and nullspace behavior are tested.
- [ ] Symmetry, residual, iteration, and mesh-growth thresholds pass.
- [ ] Existing flow MG results remain unchanged.
- [ ] Human review and evidence are recorded.
- [ ] Dashboard marks SF-05 complete and selects SF-06.
<!-- completion-checklist:end -->

## Advancement rule

SF-06 may assemble affine-periodic right-hand sides using the validated
operator and preconditioner.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-05T15:11Z | `c4d77c3`, active | Verified SF-05 is the dashboard `NEXT`, created the exact persistent runtime Goal, and created the canonical SF-05 branch/worktree after completing the required scientific, numerical, validation, workflow, architecture, and code preflight. | `master=origin/master=c4d77c3`; increment checker passed with `next=SF-05`; SF-04 is `done`; Goal is `Validar y adaptar el multigrilla cell-centered como precondicionador de A(q).`; existing MG uses the legacy negative-sign operator, coefficient buffers named `K`, `PinSpec`, and no per-level mean-zero projection. | Build the explicit SF-05 task DAG around coefficient hierarchy, projected sign adapter, quantitative controls, and legacy-flow regression evidence. |
| 2026-08-05T15:17Z | `edee517`, research `SF-05-T01` | Independently audited the current MG sign, coefficient, coarsening, periodic coloring, transfers, nullspace handling, ownership, and legacy-flow surface without modifying the tree. | Kernels consistently solve the legacy negative `L(c)=div(c grad)` despite a contradictory positive-sign comment; harmonic face coefficients accept `q` directly; geometric coarsening gives `geom(q)=1/geom(K)`; the current red-black pre/post sequence does not establish a symmetric PCG preconditioner; periodic GSRB requires even dimensions at every active level. | Generalize coefficient storage/coarsening in T04, then implement an explicitly positive projected symmetric-cycle adapter in T05. |
| 2026-08-05T15:17Z | `edee517`, research `SF-05-T02` | Defined deterministic SF-05 acceptance controls for constant and smooth positive `q` on `32^3` and `64^3`, true CPU/GPU residuals, identity-versus-MG solution agreement, mesh iteration growth, geometric coarsening, level gauge, reuse, bilinear symmetry/positivity, and legacy-flow regression. | Tests must enforce relative residual `<=1e-10`, MG iterations `<=100`, `64^3/32^3` iteration growth `<=1.5`, zero-mean gauge, and a symmetric positive preconditioner before interpreting iteration reduction. The test scope will omit unrelated generic error-policy expansion. | Implement T04/T05 in topological order and then materialize the accepted controls in T06. |
| 2026-08-05T15:17Z | `c4d77c3`, baseline `SF-05-T03` | Established the read-only local baseline before MG adaptation: checker, configure, build, CTest, direct operator suites, and legacy PSPTA smoke all passed after confirming the Ninja build had completed and rerunning the initially premature checks. | `CTest 2/2`; existing projected-PCG controls took 5 iterations for constant `q` and 35 for smooth `q`, with relative residuals `5.6663e-13` and `2.2663e-13`; legacy head MG-CG took 10 iterations and reduced `1.02e+01 -> 1.77e-13`; smoke ended with active/exited `387/113` and zero Newton failures. Local RTX 3050, CUDA 13.3.73, Debug arch 86. | Preserve these operator and legacy-flow results while adding the SF-05 quantitative `32^3/64^3` controls. |
| 2026-08-05T15:26Z | `4ad6f70`, implementation `SF-05-T04` | Generalized `MGLevel::coefficient`, moved the single geometric 2x2x2 coarsening kernel into `src/multigrid`, added `populate_coefficient_hierarchy`, and migrated the sole Darcy caller without changing the kernel operation order or any smoother, residual, V-cycle, transfer, pin, or PCG formula. | Root diff audit confirmed the old and new log/exp order match and V-cycle/preconditioner edits are naming-only. Checker, configure/build, both test targets, full CTest, and smoke passed; flow MG-CG remained 10 iterations with `1.02e+01 -> 1.77e-13`. Two accidentally overlapping Ninja sessions were detected; the newer group was terminated, the remaining build was allowed to finish, and a single quiescent rebuild plus the full validation passed. | Base T05 on `4ad6f70`; add the separate symmetric projected positive adapter without altering the validated legacy defaults. |
| 2026-08-05T15:37Z | `c1e8f4b`, implementation `SF-05-T05` | Added a separate projected-positive MG adapter over a pre-populated `q` hierarchy, using `b=-P r` for the legacy negative operator, triply periodic BC, no pin, preallocated mean-zero workspaces per level, projected RHS/residual/corrections, forward/backward GSRB ordering, and a symmetric coarse composition. | Existing checker/build/CTest/smoke passed and legacy head remained 10 iterations with `1.02e+01 -> 1.77e-13`; quantitative symmetry and positivity remain intentionally unclaimed until T06. The subagent again launched overlapping Ninja sessions; root terminated the newer group and required a single quiescent rebuild before accepting its reported validation. | Close root-audit contract gaps in C01, then measure the adapter independently in T06. |
| 2026-08-05T15:37Z | `c1e8f4b`, audit `SF-05-C01` | Root audit found the adapter validation still accepted zero pre/post smoothing, omitted periodic even-parity enforcement on the coarsest level, and did not independently verify isotropic finite spacing or 2:1 spacing between levels. | With zero smoothing the coarse-only map can be singular on fine modes; odd coarsest dimensions break periodic red-black bipartition; anisotropic metadata is invalid because current kernels use `dx` only. These are concrete in-scope preconditioner contract defects, not reasons to redesign MG. | Enforce the missing host contracts in a separate corrective commit without changing valid-path numerics. |
| 2026-08-05T15:45Z | `3456542`, corrective `SF-05-C01` | Enforced matching positive pre/post sweeps, positive coarse iterations, even dimensions at every periodic level, finite positive isotropic spacing, exact relative-tolerance 2:1 spacing/dimensions, matching config level count, and matching buffers. | Root diff audit confirmed valid `32^3/64^3` numerics and kernels are unchanged. A single Ninja session, targeted/full CTest, and smoke passed; flow remained 10 iterations with `1.02e+01 -> 1.77e-13`. | Run T06 quantitative acceptance; correct the remaining sign-scaling prose independently in C02. |
| 2026-08-05T15:45Z | `3456542`, audit `SF-05-C02` | Root audit found the sign overview now names the legacy operator correctly but its scaling paragraph still calls the positive unscaled face-difference sum `Lx`, contradicting `L=-D/h^2` and the actual residual. | This is documentation-only but directly concerns the sign adapter contract: define the unscaled positive difference as `D(x)`, state `Lx=-D/h^2`, and state `r=b-Lx=b+D/h^2`. | Correct only the mathematical comment in parallel with T06; do not alter executable formulas. |
| 2026-08-05T15:48Z | `b8a6f6a`, corrective `SF-05-C02` | Corrected only the legacy MG sign/scaling explanation: `D(x)` is the positive unscaled face-difference sum, `Lx=-D(x)/dx^2`, and `r=b-Lx=b+D(x)/dx^2`, with the matching GSRB update and `div(K grad h)=rhs`. | Root diff audit, `git diff --check`, the increment checker, and targeted sign-symbol inspection passed. No build/test was claimed for this comment-only commit and there is no numerical effect. | Integrate C02 with the production and quantitative-test commits after T06 reports. |
| 2026-08-05T15:58Z | `73e964a`, validation `SF-05-T06` | Added independent CPU-long-double/GPU acceptance controls for geometric `q`/inverse-`K` coarsening, hierarchy reuse, per-level gauge, invalid construction, preconditioner symmetry/positivity, projected-PCG convergence and mesh growth on constant/smooth `q` at `32^3/64^3`, and exact preservation of the legacy default red-black smoother order. | All cases passed: coarsening max relative error `1.9958e-16`; bilinear symmetry defect `1.6259e-16`; positive forms `87.3665/87.4431`; max output gauge `1.7683e-19`; MG iterations constant `10/10` versus Identity `14/27`; smooth `10/10` versus `107/217`; true MG residuals `1.551e-14..7.429e-14`; mesh growth `1.0`; solution RMS difference `<=5.923e-14`; legacy default/explicit red-black RMS `0`. Targeted/full CTest and baseline smoke passed. | Integrate commits topologically and independently rerun Gate 1+2. |
| 2026-08-05T15:58Z | `73e964a`, failed hypothesis during `SF-05-T06` | A proposed direct legacy `solve_head` CG-versus-PCG_MG comparison on a deterministic `16^3` case produced solution RMS difference `1.29433062194` despite reported PCG relative residual `9.39944659943e-13`. | Production was not changed and the failure was not used as SF-05 evidence; the increment instead retains the exact legacy GSRB equivalence control plus the unchanged canonical PSPTA smoke baseline. The discrepancy is outside the adapter path and remains a separate legacy investigation, not a claim that residual alone proves solution equivalence. | Preserve the failed observation in integration/PR risks; do not expand SF-05 into a legacy flow-solver repair. |
| 2026-08-05T16:03Z | `7fc1e55`, integration validation | Independently inspected and cherry-picked T04 `4ad6f70`, T05 `c1e8f4b`, C01 `3456542`, C02 `b8a6f6a`, then T06 `73e964a`, verifying each parent and changed-file scope first; the integrated equivalents are `521f012`, `7449115`, `decefad`, `a327f59`, and `7fc1e55`. Audit found one generic `coefficient` hierarchy with geometric `q` coarsening (and inverse geometric `K` equivalence), harmonic faces unchanged, the explicit `b=-P r` adapter over legacy `L=-A`, 3D periodic/no-pin execution, level projection, forward/backward GSRB, and preserved legacy default RedBlack order. | Checker and configure/build passed in one monitored Ninja session. Direct controls passed: coarsening `1.99580752804e-16` (CPU long-double), symmetry `1.62586613971e-16`, positive forms `87.3665452453/87.4431236651`, gauge `1.76828053907e-19`, constant MG `10/10` vs Identity `14/27` with true residuals `1.55131283884e-14/4.81887817892e-14`, smooth MG `10/10` vs Identity `107/217` with `2.21254689354e-14/7.42934989387e-14`, growth `1`, RMS `9.87519238975e-15/5.92254690611e-14`, and legacy RedBlack RMS `0`. CTest focal `1/1`, operator `2/2`, full `2/2`; RTX 3050 Laptop 4096 MiB, CUDA 13.3.73, Debug sm_86, GCC 16.1.1. `config_pspta_small.yaml`: seed 42, `64x32x32`, head 10 `1.02e+01 -> 1.77e-13`, particles active/exited `387/113`, zero stalls/failures. `diff --check` and final checker passed. | Gate 1+2 evidence is reproducible; Gate 3A/4/V100 do not apply to this linear preconditioner increment with no nonlinear streamfunction, transport, or remote-performance claim. Preserve the unrelated failed `solve_head` CG-vs-PCG_MG RMS `1.29433062194` at relative residual `9.39944659943e-13`; do not use it as acceptance or repair it here. Next: required human review and normal SF-05 closeout; do not advance state/checklist/NEXT/Goal. |
