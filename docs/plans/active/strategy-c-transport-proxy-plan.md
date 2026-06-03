# Strategy C Projection-Aware Plan

## Scientific starting point

- The corrected first-4-mode low Strategy A subspace on `darcy_small` remains the authoritative A->C handoff object.
- The stronger coefficient-space Gauss-Newton refiner improves consumed-object mismatch inside that foliation.
- Transport robustness still does not follow mismatch monotonically, especially at mid-`mu`.

## Hypothesis

The missing upstream signal is **projection robustness**, not another generic smoothness or mismatch term.

`PsptaEngine` does not transport by integrating the full cross-product field directly. It advances in `x` and then solves a 2×2 Newton problem in `(y,z)` using the local Jacobian

`J_yz = [[dpsi1/dy, dpsi1/dz], [dpsi2/dy, dpsi2/dz]]`

with determinant

`det(J_yz) = dpsi1/dy*dpsi2/dz - dpsi1/dz*dpsi2/dy = (grad psi1 x grad psi2)_x`.

So a Strategy C objective that ignores `J_yz` conditioning is missing the main downstream robustness mechanism.

## New proxy to test

Use a **projection-aware proxy** built from the exact consumed semantics:

1. **x-component mismatch**
   - `vx - det(J_yz)`
   - because the engine advances in `x`, and the Newton solve is consistent only if the `x` component of the Euler-potential reconstruction matches the actual throughflow.

2. **Jacobian conditioning proxy**
   - use a dimensionless reciprocal-condition surrogate
     - `rho = 2*|det(J_yz)| / (||J_yz||_F^2 + eps)`
   - `rho -> 0` means nearly singular / projection-fragile
   - `rho -> 1` means well-conditioned and well-balanced

3. **Barrier residual**
   - penalize `max(0, rho_floor - rho)` inside the coefficient-space residual vector.

## Strategy C variant

- Stay strictly inside the first-4-mode Strategy A subspace.
- Keep exact consumed-object quality before `prepare()` as an authority.
- Replace the GN proposal residual vector with a projection-aware one:
  - strongly weight `vx - det(J_yz)`
  - weakly keep `vy,vz` cross-product mismatch
  - keep invariance residuals
  - add the conditioning barrier on `rho`
- Use the same trust-region / damped GN step in coefficient space.
- Among admissible candidates, select by a combined score that includes both:
  - consumed-object mismatch
  - projection proxy score

## Success criteria

On `darcy_small`, the new variant is only successful if it:

1. improves consumed-object mismatch,
2. improves or at least stabilizes the projection proxy,
3. keeps invariance / degeneracy / collapse gates passing,
4. and makes transport failures track the better refined candidates more coherently.

## Failure modes to rule out

- improving the proxy but not consumed-object mismatch,
- improving mismatch but worsening the projection proxy enough that transport still chooses different candidates,
- control-case corruption (`uniform_x`, `layered_x`),
- or proving that even a projection-aware upstream objective still cannot align transport robustness.

## First iteration result

- Implemented as `RefinementACStrategy::SubspaceQuadraticGaussNewtonProjectionProxy`.
- The proposal residual now combines:
  - `vx - det(J_yz)`,
  - lightly weighted `vy,vz` mismatch,
  - invariance residuals,
  - and a `rho` barrier for near-singular `J_yz`.
- Admissible candidates are still filtered by exact consumed-object quality before `prepare()`, but accepted coefficient updates now also require improvement in a combined mismatch + projection-proxy selector.

### Controls

- `uniform_x` stays clean:
  - `rel_mismatch` remains `~1.000`,
  - invariance remains zero,
  - failures remain zero.
- `layered_x` stays interpretable:
  - `rel_mismatch` remains `~1.021`,
  - invariance remains zero,
  - failures remain zero.

### `darcy_small`

- Best accepted candidates by `mu`:
  - `1e-5`: `1.06761`, `234/50`
  - `3e-5`: `1.06827`, `382/66`
  - `1e-4`: `1.06771`, `220/42`
  - `3e-4`: best mismatch `1.06837`, but best fail `214/41` occurs at `1.06853`
  - `1e-3`: `1.06768`, `220/46`
- Compared with the previous non-projection GN:
  - low/high `mu` transport failures improve (`1e-5`, `1e-4`, `1e-3`),
  - mid-`mu=3e-5` regresses badly,
  - and `3e-4` still splits best-mismatch from best-fail candidates.
- Compared with the best unrefined consumed-ranked 4D-subspace gauges, the projection-aware refinements still lose on transport at every tested `mu` even when they improve mismatch:
  - `1e-5`: Strategy C best `234/50` vs unrefined subspace `223/37`
  - `3e-5`: `382/66` vs `166/28`
  - `1e-4`: `220/42` vs `98/19`
  - `3e-4`: `214/41` vs `123/28`
  - `1e-3`: `220/46` vs `128/29`
- Best-mismatch and best-fail candidate identity now aligns on `4/5` tested `mu` values, versus `2/5` for the previous GN variant.
- Raw final projection-proxy score alone is **not** a reliable global selector across initializations:
  - the lowest final proxy candidate is often not the lowest-fail transported candidate.

## Interpretation

- The projection-aware objective is scientifically meaningful because it improves coherence between mismatch improvement and downstream transport on most tested `mu` values without leaving the recovered Strategy A foliation.
- It is still insufficient:
  - the proxy does not rank all candidate initializations reliably,
  - the refined gauges still underperform the best unrefined consumed-object subspace gauges on transport,
  - mismatch remains `~1.068`,
  - and transport robustness still regresses at `mu=3e-5`.
- Conclusion: the remaining blocker is no longer just “include a projection-aware term”.
  The current projection proxy captures part of the downstream physics, but not enough to replace a more engine-faithful robustness signal.
