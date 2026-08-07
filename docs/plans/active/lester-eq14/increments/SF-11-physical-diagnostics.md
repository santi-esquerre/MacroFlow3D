# SF-11 — Physical diagnostics

- State: `active`
- Goal: `Reconstruir v_psi sobre CompactMAC y calcular los diagnósticos físicos obligatorios.`
- Depends on: `SF-10`
- Unlocks: `SF-12`
- Branch: `science/lester-sf11-physical-diagnostics`
- Worktree: `Claude-managed per-node isolated worktrees (native isolation: worktree)`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-07T19:02Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Prevent acceptance based solely on algebraic residual by measuring velocity
reconstruction, invariance, divergence, angle, magnitude, and nondegeneracy.

## Preconditions

- SF-10 provides total cell gradients, `c`, and reduction infrastructure.

## In scope

- CompactMAC face reconstruction and all Gate 3A physical metrics.

## Out of scope

- Exact discrete curl formulations, output writers, and trajectory invariance.

## Files and symbols

- Add `src/physics/streamfunctions/Diagnostics.cuh/.cu`.
- Reuse `VelocityField` CompactMAC layout and natural MAC divergence.

## Implementation specification

1. At each normal face, use the normal centered derivative and consistently
   interpolated tangential derivatives before forming the cross product.
2. Compare directly with Darcy CompactMAC components.
3. Report L2/Linf by component, magnitude error, correlation, and robust angular
   error; exclude only explicitly counted near-zero pairs from angles.
4. Compute Darcy invariance at one documented common location and split
   cross-gradient degeneracy by Darcy-speed threshold.

## Expected numerical effect

The homogeneous flow reconstructs exactly; manufactured fields exhibit
second-order physical-metric convergence.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Uniform velocity reconstruction relative error `<=1e-13`.
- Manufactured velocity and divergence errors have L2 order at least 1.8.
- Invariance metrics agree with independent CPU calculations within `1e-12`
  relative plus spatial truncation error.

## Regression surface

- CompactMAC indexing, interpolation placement, periodic boundary faces, and
  treatment of low-speed cells.

## Failure and rollback policy

- Do not claim algebraic divergence freedom for this initial reconstruction.
- If face interpolation fails second-order convergence, stop before adding
  output or solver acceptance logic.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] CompactMAC reconstruction is implemented and documented.
- [ ] Required velocity, invariance, divergence, and degeneracy metrics exist.
- [ ] Uniform and manufactured thresholds pass.
- [ ] Low-speed exclusions are explicit and counted.
- [ ] Full regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-11 complete and selects SF-12.
<!-- completion-checklist:end -->

## Advancement rule

SF-12 may define the stable public ownership/API around the accepted numerical
primitives and reports.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-07T19:02Z | active; base `master=origin/master=a3a5718a18c2546f06a4a75545b31eb4e64cc612` | Activated SF-11 documentation state under the Claude Code orchestration harness. | Preflight verified on the default branch: SF-10 `done` via PR #21 (`dd83caa`, tree identical to audited `54a2720`) and closure PR #22 (`a3a5718`); checker PASS (`29 increments, next=SF-11`); clean tree. Reuse surface inspected in code: `VelocityField` CompactMAC layout (U-face `i` between cells `i-1`,`i`; periodic duplicate boundary planes), `enqueue_total_streamfunction_gradients` (SF-07, independent spacings), SF-09 `c=g1×g2` convention and `kMaxDegeneracyThresholds`, SF-10 workspace/enqueue/synchronize pattern with `blas` reductions, and the CPU oracle library in `tests/streamfunctions/reference_operators.*`. Interpretive decisions recorded for human review: (1) reconstruction is interpolate-then-cross — face gradients use the natural compact normal derivative plus arithmetic two-cell averages of the SF-07 cell-centered tangential derivatives, then the cross product's face-normal component is stored (the normal compact derivative cancels algebraically from that component; documented); (2) the documented common location for invariance/magnitude/angle/degeneracy metrics is cell centers with per-component two-face averaging of the Darcy MAC field; (3) all normalizations use the measured cell-centered `v_D,rms` with no hidden floors (degenerate normalizations surface as NaN/Inf); (4) face reductions run over unique faces only; (5) the module supports independent spacings (its dependencies do), diverging deliberately from the SF-10 chain's isotropic fail-fast — documented; (6) `\|c\|` percentiles stay in SF-10's evaluator, SF-11 adds exact min/max/mean and the Darcy-speed-split degeneracy counts. No algebraic divergence-freedom claim will be made (spec rollback rule). sccache remains disabled locally (documented in SF-09 activation). Persistent Goal `Reconstruir v_psi sobre CompactMAC y calcular los diagnósticos físicos obligatorios.`; delivery branch `science/lester-sf11-physical-diagnostics`. | Build the SF-11 intra-increment DAG and delegate implementation to isolated workers. |
