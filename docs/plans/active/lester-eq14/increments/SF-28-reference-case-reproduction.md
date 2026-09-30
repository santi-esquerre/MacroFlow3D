# SF-28 — Reference case reproduction (256^3, sigma^2 = 4)

- State: `pending`
- Goal: `Reproducir el caso de referencia de Lester et al. (256^3, ell=1/16, sigma_Y^2=4) con iteración anidada de malla y registrar métricas Gate 3A con convergencia bajo refinamiento en la V100.`
- Depends on: `SF-27`
- Unlocks: `SF-29`
- Branch: `science/lester-sf28-reference-case`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reach the paper's Table 1 case (parametric reproduction per the overview:
same domain, periodicity, lognormal statistics, `ell/h = 16`, mean velocity 1;
project-fixed Gaussian covariance and seed) and show that the streamfunctions
reconstruct an independent Darcy solve with errors converging under
refinement — the "physical reproduction" level. Both solvers are exercised;
their cost at 256^3 is measured (the "eventual performance" evaluation of
pseudo-time).

## Preconditions

- SF-26 and SF-27 `done`; V100 free memory `>= 8 GiB` verified (`nvidia-smi`).

## In scope

- Nested iteration 64^3 -> 128^3 -> 256^3: the SF-18 field is generated on
  256^3 and restricted downward (exact low-pass), unless a diagnostic proves
  the same-seed hash-per-mode generator already yields the identical continuum
  realization across grids (print `||Y_256 restricted - Y_64|| / ||Y_64||`).
- Prolongation of the accepted fluctuations with the SF-05 MG transfer +
  mean-zero projection as the warm start of the next grid; `q`/MG hierarchy
  rebuilt per grid; independent SF-19 Darcy solve per grid.
- Continuation policy: `lambda = 1` directly at each grid from the prolonged
  state (the SF-21 lambda axis available as rescue only, recorded).
- Runs with the implicit stack and with pseudo-time; per-grid record of
  `r_F`, `e_v` (L2/Linf/angular), invariance, `e_div`, `|c|` percentiles,
  iterations/steps, wall time, peak memory; `e_v(h)` order; 3 extra seeds at
  128^3.
- Experiment note comparing to the paper's reported figures (residual
  `1e-16` recorded as reference, not tolerance).

## Out of scope

- Kernel fusion / optimization beyond what is needed to fit memory; tracking;
  exponential covariance; `sigma^2 = 6.25`.

## Files and symbols

- `apps/config_streamfunctions_reference_256_var4.yaml` (+ 64/128 variants);
  `ContinuationController.*` (grid leg) or a dedicated `GridSequencer` in
  `src/physics/streamfunctions/`; `src/multigrid/transfer/{prolong,restrict}_3d.cuh`;
  `physics::generate_periodic_gaussian_field` (restriction path);
  `docs/experiments/2026-XX-XX-sf28-reference-case.md`.

## Implementation specification

1. Add the grid leg (restrict field, prolong fluctuations, rebuild, solve)
   with structured per-grid records; no allocation inside solver loops.
2. Verify memory at 256^3 against the workspace accounting before running.
3. Run 64/128/256 with both solvers; record everything in the experiment note
   with exact commands and build hashes.

## Expected numerical effect

- `e_v`, invariance and `e_div` decrease ~4x per refinement level at fixed
  realization; `r_F <= 1e-8` at 256^3 within budget; pseudo-time fine-grid
  steps `O(1e3-1e4)` after prolongation.

## Validation commands

```bash
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote run sf28-ref64  -- "./build/v100-release/macroflow3d_pipeline apps/config_streamfunctions_reference_64_var4.yaml"
scripts/remote run sf28-ref128 -- "./build/v100-release/macroflow3d_pipeline apps/config_streamfunctions_reference_128_var4.yaml"
scripts/remote run sf28-ref256 -- "./build/v100-release/macroflow3d_pipeline apps/config_streamfunctions_reference_256_var4.yaml"
```

## Acceptance thresholds

- 256^3 accepted state with `r_F <= 1e-8` (implicit or pseudo-time; both
  recorded), no unexplained degenerate cells, `|c|` 0.1 % percentile `> 0`.
- `e_v(h)` observed order `>= 1.5` over 64 -> 128 -> 256.
- Peak memory and wall time recorded for both solvers.

## Regression surface

- Continuation controller records/exports; SF-18 generator (restriction path
  must not change the existing same-grid generation bitwise).

## Failure and rollback policy

- Memory overflow at 256^3: record and stop at 128^3 (the increment remains
  `blocked` with evidence; SF-28 optimization work is then re-planned).
- Non-convergence at a grid: record, do not tune gates.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Field restriction / fluctuation prolongation grid leg implemented and unit-tested.
- [ ] 64/128/256 runs recorded for both solvers with Gate 3A metrics.
- [ ] `e_v(h)` convergence and seed robustness recorded.
- [ ] Memory/time evidence recorded; Gate 4 interpretation and human review recorded.
- [ ] Experiment note, PR, and commit are recorded.
- [ ] Dashboard marks SF-28 complete and selects SF-29.
<!-- completion-checklist:end -->

## Advancement rule

SF-29 may consume the accepted 256^3 streamfunctions.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-09-30T00:00Z | created by the owner-directed re-sequencing (decision 2026-09-30) | Replaces the former "SF-28 — GPU optimization"; absorbs the former grid-continuation increment. | `docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md` | Activate only when named by `NEXT`. |
