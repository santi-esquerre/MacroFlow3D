# Strategy C Coefficient-Space Gauss-Newton Plan

## Scientific starting point

- On `darcy_small`, the corrected first-4-mode low Strategy A subspace is the authoritative A->C handoff object.
- The unconstrained Strategy C variant was ruled out because it reduced mismatch only by leaving the recovered foliation.
- The first subspace-constrained variant (`SubspaceQuadraticMap`) proved that foliation-preserving refinement can reduce mismatch, but only modestly, and transport robustness did not consistently improve.

## Hypothesis

The current mismatch floor is partly a representation issue, but also an optimizer issue:

- the current refiner keeps the linear coefficients frozen,
- and it uses fixed-step coordinate search over quadratic coefficients only.

If Strategy C instead optimizes **linear and quadratic coefficients together** with a small trust-region / Gauss-Newton style solve in coefficient space, it should find materially better consumed-object gauges while staying inside the recovered 4D Strategy A foliation.

## Refined Strategy C object

- **Input:** the first 4 low Strategy A modes on the corrected exact small-grid path.
- **Representation:** two scalar gauge maps on those 4 modal coordinates
  - `psi1(phi) = a1·phi + q1(phi)`
  - `psi2(phi) = a2·phi + q2(phi)`
- **Unknowns:** all linear and quadratic coefficients for both fields.
- **Constraint:** the realized scalars depend only on the recovered modal coordinates, so refinement remains foliation-preserving by construction.

## Optimizer to test first

Use a small Levenberg-Marquardt / trust-region Gauss-Newton step in coefficient space:

1. build the current realized fields from the coefficient vector;
2. evaluate a host-side residual vector on the realized fields using the exact consumed semantics:
   - cross-product residual `v - grad(psi1) x grad(psi2)`
   - invariance residuals `v·grad(psi1)`, `v·grad(psi2)`
3. estimate the Jacobian of that residual vector with finite differences in coefficient space;
4. solve the damped normal equations for a trial coefficient update;
5. enforce a trust radius / backtracking on the coefficient update;
6. upload the trial fields and use `PsptaInvariantField::compute_quality()` as the authoritative acceptance gate before `prepare()`.

This is not replacing consumed-object scoring. It is only a stronger proposal generator.

## Success criteria

On `darcy_small`, the stronger Strategy C is successful only if:

1. consumed-object mismatch drops materially below the current constrained floor,
2. invariance remains near the Strategy A baseline,
3. degeneracy and anti-collapse gates remain satisfied,
4. `prepare_drift_max` stays negligible,
5. transport drift / Newton failures do not regress badly.

## Failure modes to rule out

- gains that appear only on host residual algebra but not after upload,
- gains that come from invariance growth or collapse,
- gains that disappear already at `prepare()`,
- improvements in mismatch that remain disconnected from transport robustness.

## Minimal implementation scope

- extend `RefinementAC` with a new coefficient-space Strategy C mode;
- keep the current quadratic coordinate-search mode as baseline for comparison;
- update the exact-object harness to run the stronger Strategy C path on the same consumed-ranked initializations;
- add a focused regression test that requires linear-coefficient repair, which the old frozen-linear strategy cannot satisfy.

## First iteration result

- Implemented as `RefinementACStrategy::SubspaceQuadraticGaussNewton`.
- Representation:
  - optimize linear and quadratic coefficients together for both scalar fields,
  - remain strictly inside the first-4-mode Strategy A subspace,
  - generate proposals with a damped coefficient-space Gauss-Newton step,
  - accept or reject only with exact consumed-object quality gates before `prepare()`.

### Control behavior

- `uniform_x`:
  - no fake improvement,
  - best consumed mismatch `1.00004 -> 1.00001`,
  - invariance remains zero, failures remain zero.
- `layered_x`:
  - best consumed mismatch `1.02107 -> 1.02049`,
  - invariance remains zero, failures remain zero.

### `darcy_small`

- The GN refiner improves mismatch more than the earlier quadratic coordinate search:
  - `mu=1e-5`: best `1.06755`, `258/52` fails
  - `mu=3e-5`: best mismatch `1.06790`, but best-fail candidate is different (`1.06846`, `253/50`)
  - `mu=1e-4`: best mismatch `1.06745`, but best-fail candidate is different (`1.06777`, `309/58`)
  - `mu=3e-4`: best mismatch `1.06745`, but best-fail candidate is different (`1.06776`, `221/43`)
  - `mu=1e-3`: best `1.06673`, `271/53` fails
- Accepted invariance remains in the same `O(1e-4)` regime as the earlier constrained refinements, not the `O(1e-1)` blow-up of the unconstrained variant.
- `prepare_drift_max` stays `~3e-8`, so transport is still not the first-order limiter.

## Interpretation

- The stronger coefficient-space optimizer validates the next part of the Strategy C story:
  - the current floor was not purely “optimizer weakness” in the earlier quadratic search,
  - but better coefficient-space optimization does continue to improve the consumed-object gauge inside the recovered foliation.
- The remaining blocker is also clearer:
  - mismatch improvement and transport robustness are still only partially coupled,
  - so a stronger optimizer alone is not enough.
- The next Strategy C step should therefore stay subspace-constrained, but start addressing the coupling between consumed-object mismatch and downstream transport robustness explicitly.
