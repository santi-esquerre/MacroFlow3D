# SF-16 — Pipeline, configuration, and output

- State: `pending`
- Goal: `Integrar configuración, logging, exportación y pipeline sin alterar el comportamiento por defecto.`
- Depends on: `SF-15`
- Unlocks: `SF-17`
- Branch: `science/lester-sf16-pipeline-io-config`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf16-pipeline-io-config`
- Acceptance gate: `Gate 1 + Gate 3A integration`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
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
