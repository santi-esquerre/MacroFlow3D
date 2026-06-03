# Strategy C Engine-Sampled Surrogate Plan

## Scientific starting point

- The corrected first-4-mode low Strategy A subspace on `darcy_small` remains the authoritative A->C handoff object.
- The current projection-aware GN is scientifically meaningful, but its static `vx-det(J_yz)` plus conditioning proxy is still too indirect:
  - it improves best-mismatch / best-fail alignment relative to plain GN,
  - but it still loses to the best unrefined consumed-ranked 4D gauges on transport,
  - and the lowest final proxy candidate is often not the lowest-fail transported candidate.

## Hypothesis

The missing upstream signal is a **sampled emulation of the engine's own Newton projection path**, not another static field diagnostic.

The transport probe uses:

1. deterministic injection at `x0 = 0.25*Lx`,
2. `psi_const` captured from the exact consumed object,
3. three `newton_solve_yz()` calls per step (`x`, `x_mid`, `x_new`),
4. fail-count accumulation under the engine's exact `Ly/Lz` self-period semantics.

So the next Strategy C selector should score trial gauges with a cheap host-side surrogate that follows that same logic on a deterministic sample of particles, instead of inferring robustness only from static Jacobian algebra.

## Strategy C variant to test

- Keep the current first-4-mode coefficient-space representation.
- Keep consumed-object quality before `prepare()` as the first scientific gate.
- Keep the projection-aware residual as the proposal generator for now.
- Replace the previous acceptance selector with a combined score that includes a **sampled one-step engine proxy**:
  - deterministic sample particles on the same injection plane used by the transport probe,
  - same `dt = 0.25*dx/vmax`,
  - same host-side trilinear `psi` sampling, periodic lifting, and Newton solve semantics,
  - same stage structure: project at `x`, `x_mid`, `x_new`.

## Engine-sampled proxy metrics

For each sampled gauge candidate, compute:

1. `sample_fail_fraction`
   - fraction of sampled particles that fail at any stage of the one-step emulation.

2. Stage-resolved fail fractions
   - `sample_fail_x_fraction`
   - `sample_fail_mid_fraction`
   - `sample_fail_new_fraction`
   - to identify which Newton stage correlates best with downstream failures.

3. `sample_mean_newton_iters`
   - mean successful Newton iterations across all attempted stage solves.

4. `sample_mean_final_residual`
   - mean terminal residual of successful Newton solves, normalized by the engine tolerance scale.

5. `sample_low_recip_condition_fraction`
   - fraction of attempted stage solves whose minimum reciprocal-condition surrogate drops below the chosen floor.

6. `combined_score`
   - initial version:
     - `rel_mismatch`
     - `+ w_fail * sample_fail_fraction`
     - `+ w_iter * normalized_mean_iters`
     - `+ w_res * normalized_mean_final_residual`
   - stage-resolved fractions are recorded for diagnosis even if not all are used in the first selector.

## Success criteria

This variant is only successful on `darcy_small` if it:

1. keeps admissible refinements inside the recovered 4D Strategy A foliation,
2. improves consumed-object mismatch,
3. improves or at least stabilizes the sampled engine proxy,
4. and makes downstream transport failures track the refined candidates more coherently than the projection-aware GN.

The key scientific comparison is not only against the previous projection-aware GN, but also against the best unrefined consumed-ranked 4D gauges.

## Failure modes to rule out

- sampled proxy improves but full transport does not,
- best sampled-proxy candidate still differs systematically from best-fail candidate,
- refined gauges still lose to unrefined consumed-ranked subspace gauges on transport,
- or control cases stop being interpretable.

## Results

- V100 regression coverage passed after implementing the sampled engine surrogate:
  - `validate_slepc_eigensolver`
  - `invariant_pair_search`
  - `refinement_ac`
  - `refinement_ac_subspace`
  - `refinement_ac_subspace_gn`
  - `refinement_ac_transport_proxy`
  - `refinement_ac_engine_proxy`
- First engine-sampled selector (`rel_mismatch + 0.25 * engine.combined_score`) was scientifically useful but insufficient:
  - it improved best-mismatch / best-fail coherence relative to the projection-aware GN,
  - but still lost to the best unrefined consumed-ranked 4D gauges on transport at every tested `darcy_small` `mu`.
- Offline audit of the accepted `darcy_small` candidates showed the selector itself was the weak point:
  - among those accepted candidates, `final_eng_fail_fraction` matched the lowest real `total_fail` row at all five tested `mu`,
  - `final_eng_combined_score` matched none of them.
- A second iteration replaced the selector with a fail-fraction-first lexicographic order (`fail_fraction`, then normalized final Newton residual, then `rel_mismatch`).
- That second iteration ruled out binary sampled fail fraction as a sufficient optimization target:
  - accepted `darcy_small` candidates drove sampled `final_eng_fail_fraction` to `0` across the board,
  - but full transport still remained poor (`246/50`, `266/51`, `179/44`, `345/69`, `245/51` across `mu=1e-5..1e-3`),
  - and the best refined candidates still lost to the best unrefined consumed-ranked 4D gauges (`223/37`, `166/28`, `98/19`, `123/28`, `128/29`).
- Current signal ranking on the final fail-fraction run:
  - `final_eng_fail_fraction`: too coarse after optimization; it saturated to zero and no longer discriminated candidates.
  - `final_eng_mean_normalized_final_residual`: still carries some information (`3/5` agreement with best-fail row across `mu`).
  - `final_eng_combined_score`: not scientifically trustworthy as a selector (`0/5` agreement in the earlier run).

## Current conclusion

- The sampled engine surrogate is closer to the real PSPTA failure mechanics than the static projection proxy.
- But the present sampled selectors are still too weak:
  - combined score is misweighted,
  - fail fraction alone is gameable and saturates.
- The remaining blocker is therefore **surrogate weakness**, not loss of the recovered Strategy A foliation.
- The next Strategy C variant should use a harder engine-faithful signal than binary sampled fail fraction:
  - more sampled steps / points,
  - or a richer near-failure Newton residual/conditioning aggregate that cannot be driven to zero so easily.
