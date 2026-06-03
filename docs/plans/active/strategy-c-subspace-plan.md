# Strategy C Subspace-Constrained Plan

## Scientific starting point

- On `darcy_small`, the corrected first-4-mode low Strategy A subspace is now the authoritative A->C handoff object.
- Consumed-object scoring has already ruled out host-ranked linear gauge slices as scientific authority.
- The first consumed-object-aware Strategy C iteration was scientifically useful because it ruled out free voxelwise correction:
  - it could lower mismatch,
  - but only by increasing invariance residuals by roughly `O(10^2..10^3)`,
  - so every mismatch-reducing update left the recovered Strategy A foliation.

## Hypothesis

The remaining `rel_mismatch ~ 1.068` floor is not evidence that Strategy A recovered the wrong foliation. It is evidence that the current gauge representation is too weak and that free corrections are too unconstrained. A subspace-constrained gauge map on the first 4 Strategy A coordinates can reduce mismatch without blowing up invariance.

## Strategy C object

- **Input:** the first 4 low Strategy A modes on the corrected exact small-grid path.
- **Authoritative evaluation space:** the realized `PsptaInvariantField`, scored by `compute_quality()` before `prepare()`.
- **Optimization variables:** coefficients of two scalar gauge maps defined on the 4 modal coordinates.

## Minimal constrained variant to test

Represent the refined gauge as two low-order polynomial maps of the 4 modal coordinates:

- `phi = [phi0, phi1, phi2, phi3]`
- `psi1 = a1·phi + q1(phi)`
- `psi2 = a2·phi + q2(phi)`

where `q1`, `q2` are quadratic polynomials in `phi`.

This keeps refinement tied to the recovered Strategy A foliation because the realized scalars depend only on the recovered modal coordinates, not on free voxelwise corrections.

## Why this follows from the evidence

- The corrected 4D Strategy A subspace is much more stable in `mu` than any fixed 2D prefix.
- Consumed-object ranking already showed that linear 2-field slices saturate around `rel_mismatch ~ 1.068`.
- The first unconstrained refiner showed that mismatch can be reduced only by stepping out of the foliation.
- Therefore the minimal next test is not another linear slice and not another free correction. It is a richer but subspace-locked gauge representation.

## Success criteria

The constrained Strategy C is successful only if, on `darcy_small`:

1. consumed-object `rel_mismatch` drops materially below the current `~1.068` floor,
2. invariance residuals stay near the Strategy A baseline,
3. degeneracy does not worsen materially,
4. prepare drift stays negligible,
5. transport drift / Newton failures do not regress badly,
6. no improvement is caused by gradient collapse or field trivialization.

## Failure modes to rule out

- reducing mismatch only by increasing invariance residuals,
- reducing mismatch only by shrinking gradients / field ranges,
- a candidate that looks good on host algebra but not after realization in `PsptaInvariantField`,
- transport becoming the first-order limiter after consumed-object selection.

## Validation order

1. `compute_quality()` immediately after upload / refinement
2. quality after `prepare()`
3. transport drift / Newton failures after stepping

Controls:

- `uniform_x`
- `layered_x`
- main target: `darcy_small`

## Minimal implementation scope

- Keep the corrected exact small-grid path authoritative.
- Keep the old probed surrogate path demoted and untrusted.
- Do not revisit eigensolver fidelity, fixed-pair extraction, Strategy B, or legacy marching.
- Prefer extending the existing exact-object harness and `RefinementAC` reporting path.

## First iteration result

- Implemented as `RefinementACStrategy::SubspaceQuadraticMap`.
- Representation:
  - keep the consumed-ranked linear initialization from the first 4 Strategy A modes,
  - add quadratic corrections in those 4 modal coordinates,
  - score every trial inside the exact consumed object before `prepare()`.
- Scientific outcome:
  - `uniform_x`: no fake gain is accepted; mismatch remains `~1.00004`.
  - `layered_x`: small admissible improvements appear (`~1.02107 -> ~1.01884` best) with invariance still essentially zero.
  - `darcy_small`: first admissible foliation-preserving mismatch reductions appear, but they are modest:
    - `mu=3e-5`: `1.06860 -> 1.06837`
    - `mu=1e-4`: `1.06846 -> 1.06775`
    - `mu=3e-4`: `1.06857 -> 1.06770`
    - `mu=1e-3`: `1.06857 -> 1.06732`
  - invariance only grows mildly (`~10–25%`), not by `10^2..10^3` as in the unconstrained variant.
  - `prepare_drift_max` stays `~3e-8`, so transport is still not the first-order limiter.
  - transport failures remain mixed relative to the best consumed-ranked baseline candidates.

## Interpretation

- The first subspace-constrained Strategy C variant is scientifically useful because it proves the A->C handoff is not empty: mismatch can be reduced without leaving the recovered Strategy A foliation.
- The current quadratic gauge-map representation is still too weak to make `darcy_small` scientifically usable.
- The next refinement should stay subspace-constrained, but move beyond fixed-step coordinate search on quadratic coefficients toward a stronger coefficient-space optimization and/or richer subspace map.
