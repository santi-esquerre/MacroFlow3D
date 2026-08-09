# SF-16 — Pipeline, configuration, and output

- State: `active`
- Goal: `Integrar configuración, logging, exportación y pipeline sin alterar el comportamiento por defecto.`
- Depends on: `SF-15`
- Unlocks: `SF-17`
- Branch: `science/lester-sf16-pipeline-io-config`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 3A integration`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-09T09:01Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Make the accepted Picard solver reproducible and observable in the existing
pipeline while preserving all disabled-path behavior.

## Preconditions

- SF-15 adaptive Picard is accepted through its library API.

## In scope

- Strict YAML config, validation/serialization/manifest, post-flow invocation,
  CSV/JSON histories, optional field output, and stage profiling.

## Out of scope

- Grid/heterogeneity continuation, PSPTA consumption, and future-method config.

## Files and symbols

- Extend `src/io/Config*`, `ConfigYaml.cpp`, manifest/effective config,
  `OutputLayout`, and writers.
- Invoke from `src/runtime/ensemble/EnsembleRunner.cu` after Darcy velocity and
  before transport, only when enabled.

## Implementation specification

1. Add `streamfunction_solver.enabled`, affine mean-velocity mode/value,
   epsilon, Picard, linear solver, and diagnostic/export settings.
2. Default `enabled=false`; preserve strict unknown-key rejection.  Do not add
   Anderson/Newton/grid-continuation keys yet.
3. Write under `output/streamfunctions/r_NNNN/grid_NNNN/`:
   `iteration_history.csv`, `summary.json`, and optional raw double fields.
4. Avoid CPU/GPU field transfers except at configured export points.

## Expected numerical effect

Disabled runs are unchanged.  Enabled runs reproduce the library solver and
record sufficient configuration and metrics.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
./build/wsl-debug/macroflow3d_pipeline apps/config_pipeline_par2.yaml
```

## Acceptance thresholds

- Disabled configs produce the prior pipeline behavior and artifacts.
- Unknown and invalid streamfunction keys fail with actionable messages.
- Enabled homogeneous run matches SF-13 metrics and output schema.

## Regression surface

- Strict config parsing, output paths, ensemble lifetime, transport memory, and
  pipeline stage ordering.

## Failure and rollback policy

- Do not add a compatibility fallback for malformed keys.
- If peak coexistence with transport exceeds memory, release optional solver
  diagnostics before transport rather than changing transport ownership.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Minimal strict config and validation are implemented.
- [ ] Disabled baseline is unchanged.
- [ ] Enabled homogeneous pipeline run matches library results.
- [ ] Histories, summary, manifest, and optional fields are reproducible.
- [ ] Par2/PSPTA smokes and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-16 complete and selects SF-17.
<!-- completion-checklist:end -->

## Advancement rule

SF-17 may add eta/epsilon continuation and extend config only for those landed
features.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-09T09:01Z | activation on `master=ead6322` (SF-15 closure merged via PR #27) | SF-16 activated after verifying `NEXT: SF-16`, SF-15 `done`, and checker `OK (29 increments, next=SF-16)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) the pipeline's compact `VelocityField` (U `(nx+1,ny,nz)`, V `(nx,ny+1,nz)`, W `(nx,ny,nz+1)`) has exactly the solver's CompactMAC layout, so the Darcy velocity is adapted by zero-copy spans; the runner sizes/computes `vel_compact` also when the streamfunction solver is enabled. (2) The streamfunction `BCSpec` is constructed triply periodic by the runner (the v1 solver requirement), independent of the flow BCs; enabled runs inherit the library's isotropic-spacing/MG-coarsenability validation and fail fast with its messages. (3) Minimal strict config surface `streamfunction_solver{enabled(false), affine_mean_velocity{mode: fixed|measured, value(1.0)}, epsilon, eta, picard{max_iter, tolerance, omega}, adaptive{enabled}, linear{rtol, max_iter, check_every}, mg{num_levels}, export{iteration_history(true), summary(true), fields(false)}}`; no Anderson/Newton/continuation keys; unknown keys rejected by the existing strict mechanism; effective-config serialization and manifest updated; `measured` mode takes vbar as the arithmetic mean of the unique Darcy U-faces. (4) Output under `output/streamfunctions/r_NNNN/grid_NNNN/` following the existing `r_NNNN` convention with `grid_NNNN` = zero-padded fine-grid `nx`; `iteration_history.csv` carries the accepted-state history plus the SF-15 trial history (omega/outcome per trial); `summary.json` carries exit reason, iterations, final residual and physical diagnostics, memory report, and a config echo. (5) Solver fields/workspace are constructed per realization after the velocity stage and destroyed before transport (the spec's release-before-transport policy); stage profiling via the existing profiler. (6) Enabled homogeneous acceptance thresholds PRESPECIFIED before any run: `picard_iterations=0`, `r_F <= 1e-13`, `RMS(u1/u2) <= 1e-13`, gauge at the SF-03 bound, and `e_v <= 1e-8` — the e_v bound is limited by the pipeline Darcy head-solve relative tolerance (1e-10), not by the streamfunction path, and this rationale is recorded rather than silently reusing the SF-13 1e-13 figure. (7) Disabled-baseline invariance verified by comparing the `config_pspta_small.yaml` and `config_pipeline_par2.yaml` smoke artifacts against references generated from the increment base build with identical seeds/configs. | Base commit is this activation commit on `master=ead6322`. Gate 1 + Gate 3A(integration) apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
