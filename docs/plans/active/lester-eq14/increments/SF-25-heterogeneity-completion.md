# SF-25 — Heterogeneity completion

- State: `active`
- Goal: `Completar la continuación de heterogeneidad hasta lambda uno con convergencia terminal Newton para todas las varianzas objetivo.`
- Depends on: `SF-24`
- Unlocks: `SF-26`
- Branch: `science/lester-sf25-heterogeneity-completion`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `Claude Code orchestrator (Fable) + delegated workers`
- Started: `2026-08-13`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Re-impose, UNCHANGED, the heterogeneity acceptance set that SF-21 closed
partially (owner option (a), 2026-08-11): the damped Picard/Anderson
fixed-point map proved non-contractive exactly at `eta=1` for
`sigma_Y^2 >= 1` (r_F plateau ~1e-3 at `lambda~0.5`, 32^3), which is the
regime the plan reserves for Newton terminal convergence. With the SF-22..24
Newton-Krylov stack accepted, drive every stage to tolerance and reach the
full lognormal target conductivity for all benchmark variances.

## Preconditions

- SF-21 (partial) provides the audited lambda/eta-rescue continuation
  machinery, the Anderson-wired stage solver, the pipeline surface, and the
  recorded partial evidence (32^3: sigma^2=0.25 reached lambda=1;
  sigma^2=1 floor-exhausted at lambda=0.5 with clean physics).
- SF-24 provides globalized Newton-Krylov with reproducible Picard fallback.

## In scope

- Wire Newton terminal convergence into the heterogeneity stage solver
  (activation from the accepted Picard/Anderson state per the SF-24
  activation-threshold policy; rescue/rollback semantics preserved).
- The MOVED, UNCHANGED SF-21 gates:
  - fixed-seed `32^3` smoke reaches `lambda=1` for `sigma_Y^2=1`
    (seed 12345, ell=8, epsilon fixed 1e-2, every accepted stage
    `r_F <= 1e-6`);
  - the `64^3` suite reaches `lambda=1` for `sigma_Y^2={0.25,1,2.25,4}`
    with `ell/h=8` (seed 12345), run as a recorded experiment through the
    pipeline binary; epsilon legs toward `1e-6` recorded where budget
    allows.

## Out of scope

- Grid transfer, `256^3`, exponential covariance, `sigma_Y^2=6.25`
  (prohibited until the `sigma_Y^2=4` suite is accepted).

## Files and symbols

- Extend the SF-21 stage solver path (`ContinuationController`,
  `StreamfunctionSolver`) with the Newton terminal phase; reuse the SF-21
  benchmark configs; extend stage records with the Newton/Anderson counters
  needed for per-stage attribution.

## Implementation specification

1. Preserve the SF-21 lambda axis, eta-rescue ordering, epsilon staging,
   coefficient-reuse lifecycle, and rollback semantics exactly.
2. Newton activates per stage only from the plan-locked threshold
   (`r_F < 1e-2` initially) and falls back to Picard/Anderson per SF-24.
3. No gate, seed, budget, tolerance, or fixture value may change from the
   SF-21 prespecification; any conflict is a recorded failure.

## Expected numerical effect

Stages that plateau at `eta=1` under the fixed-point map converge
quadratically once inside the Newton basin, making `lambda=1` reachable for
all benchmark variances with controlled cost.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/v100-release --output-on-failure -R streamfunction_heterogeneity
./build/v100-release/macroflow3d_pipeline apps/config_streamfunctions_gaussian_smoke32.yaml
./build/v100-release/macroflow3d_pipeline apps/config_streamfunctions_gaussian_64_var4.yaml
```

(Validation executes on the remote V100 per the standing venue policy.)

## Acceptance thresholds

- `32^3` smoke reaches `lambda=1` for `sigma_Y^2=1` (the SF-21 sigma^2=0.25
  pass stands as recorded evidence and must not regress).
- The `64^3` suite reaches `lambda=1` for `0.25, 1, 2.25, 4` with `ell/h=8`.
- Every accepted stage reports `r_F`, velocity, invariance, divergence,
  degeneracy, eta, epsilon, MG rebuild count, and solver-phase attribution.

## Regression surface

- SF-21 accepted behavior (sigma^2=0.25 smoke, default paths byte-stable),
  Newton/Picard handoff correctness, continuation rollback, total runtime.

## Failure and rollback policy

- Minimum lambda-step failure remains a recorded physical/numerical failure;
  do not skip lambda intervals.
- `sigma_Y^2=6.25` is prohibited in this increment.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Newton terminal convergence is wired into the heterogeneity stage solver.
- [ ] The moved 32^3 sigma^2=1 gate passes unchanged.
- [ ] The 64^3 suite meets the moved acceptance set unchanged.
- [ ] Full Gate 3A metrics and experiment notes are recorded.
- [ ] Gate 4 interpretation and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-25 complete and selects SF-26.
<!-- completion-checklist:end -->

## Advancement rule

SF-26 may prolong accepted lambda-one solutions to finer versions of the same
periodic realization.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-13T16:10Z | activation on `master=23c5555` (SF-24 merged; checker OK next=SF-25) | UNDERSTAND: full re-read of `ContinuationController.hpp` (stage-solver contract: `base_config` supplies every non-overridden field for EVERY stage — baseline, lambda attempts, eta-rescue ramp, epsilon leg — so `config.newton` flows into all stages; the rescue/rollback machinery already consumes the SF-24 structured `newton_exhausted`/`newton_budget_exhausted` exits as `solver_not_converged`), the smoke fixture (`run_heterogeneity_smoke`: R5 Anderson block + degenerate epsilon 1e-2 + verbatim gates reached_target/final_lambda==1/every-accepted-r_F<=1e-6), and the five benchmark YAMLs (smoke32 degenerate-epsilon; the 64^3 quartet with epsilon leg to 1e-6, whose own SF-21 comment fixes "the GATE is reaching lambda=1", epsilon legs recorded where budget allows). PRESPECIFIED decisions (recorded verbatim in `.claude/orchestration/SF-25-heterogeneity-completion/understanding.md` BEFORE implementation): E1 the ONLY sanctioned fixture change is `newton.enabled=true` with SF-24 defaults in the smoke base_config and the five YAMLs (exact analogue of SF-21 R5 Anderson enablement); controller control flow untouched; E2 per-stage attribution = additive HeterogeneityStageRecord Anderson/Newton counters (closing the recorded SF-21 caveat and the SF-24 Anderson-export backlog at stage granularity), additive export/prints, SF-17 record untouched; E3 64^3 budget policy: sequential tmux jobs (025->1->225->4), no artificial kill, 12 h per-variance stop-and-record bound, acceptance gate = lambda=1 per variance, epsilon-leg outcomes recorded as-is; E4 regression: full ctest on the final head expected 16/16 (the known-fail heterogeneity smoke becomes PASSING — the increment's headline), byte-compares vs fresh 23c5555 base refs fully identical mod manifest timestamp/stdout timing; E5 all claims are invariant-CONSTRUCTION claims, no transport claim. Recorded risks: near-singular-Jacobian GMRES budget pressure at sigma^2>=2.25/64^3 (honest fallback chain ends in a recorded lambda-floor failure, not tuning); 64^3 epsilon legs at 1e-6 may be the hard part (gate is lambda=1). | DAG: T01 stage-record counters -> {T02 export+YAML enablement, T03 smoke wiring} (parallel) -> I01 -> F01 (Gate A smoke + full suite + byte-compares, then Gate B 64^3 experiment suite). | Delegate T01. |
| 2026-08-13T18:40Z | T01 `3dabf8b`, T02 `b3812a9`, T03 `61e66f8` all ACCEPTED (compile-gate-only workers; full diffs personally audited) | T01: eight attribution counters added to `HeterogeneityStageRecord` (types exact to `StreamfunctionSolveReport`) and harvested at the single record-builder every one of the five stage-kind call sites routes through; per-stage semantics guaranteed because `newton_jv_evaluations` is a per-solve delta by SF-24 design and each stage is one solve call; SF-17 record and all controller control flow untouched. (First T01 worker was interrupted mid-task by the owner; its partial header edit was verified, and a fresh worker independently implemented the full scope, correctly flagging the stale resume precondition.) T02: the eight counters appended as trailing CSV columns in the heterogeneity `write_stage_history` overload ONLY (the three default byte-compare configs never reach that path); `newton: enabled: true` + one SF-25 comment added to the five benchmark YAMLs and NOTHING else (adaptive strict-parser default `enabled=true` verified in all five, satisfying the C01 validation dependency); the summary.json schema intentionally untouched (per-stage attribution lives in the CSV, consistent with the recorded SF-24 export precedent). T03: `base_config.newton.enabled = true` in `run_heterogeneity_smoke` — the ONLY sanctioned fixture change (E1a, exact analogue of SF-21 R5) — plus attribution counters in the stage-history printer and one ADDITIVE non-gate sanity check (`attribution_counters_populated`); the three moved gates byte-verbatim. | Audits recorded in `.claude/orchestration/SF-25-heterogeneity-completion/` (dag.json node notes). SF-25 base byte-compare refs regenerated on the V100 at `23c5555` (`~/sf25_base_refs/`, now including the SF-24 `newton:` effective-config block, so final byte-compares must be FULLY identical mod manifest timestamp/stdout timing). Integration order: `3dabf8b -> b3812a9 -> 61e66f8`. | I01 integrator, then F01: Gate A smoke + full suite (expected 16/16) + byte-compares, then Gate B 64^3 suite per E3. |
| 2026-08-11T23:30Z | created by the owner-directed option (a) restructuring | This increment inherits, UNCHANGED, the sigma^2>=1 heterogeneity gates that SF-21 closed partially: 32^3 sigma^2=1 (floor-exhausted at lambda=0.5 under Picard/Anderson, non-contractive eta=1 plateau r_F~1e-3, zero degeneracy) and the full 64^3 suite. See `docs/decisions/2026-08-11-newton-before-heterogeneity-completion.md` and the SF-21 bitácora/audit records for the complete evidence. | Gates moved verbatim; no tuning. | Activate only when named by `NEXT` after SF-24 is done. |
