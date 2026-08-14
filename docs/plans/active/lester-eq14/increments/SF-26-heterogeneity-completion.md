# SF-26 — Heterogeneity completion

- State: `pending`
- Goal: `Completar la continuación de heterogeneidad hasta lambda uno con convergencia terminal Newton para todas las varianzas objetivo.`
- Depends on: `SF-25`
- Unlocks: `SF-27`
- Branch: `science/lester-sf26-heterogeneity-completion` (the pre-restructure work-in-progress merged via the science/lester-sf25-heterogeneity-completion restructuring PR)
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `unassigned (re-blocked on SF-25 by the 2026-08-14 restructuring)`
- Started: `2026-08-13 (paused 2026-08-14; re-blocked on SF-25)`
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
- SF-25 provides the manifold-robust terminal schedule (shifted Newton) that the eta=1 plateau requires (2026-08-14 restructuring; see `docs/decisions/2026-08-14-manifold-robust-terminal-solver.md`).

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
   (`r_F < 1e-2` initially) and falls back to Picard/Anderson per SF-24;
   the SF-25 terminal schedule is the sanctioned wiring addition of this
   increment (2026-08-14 restructuring).
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
- [ ] Dashboard marks SF-26 complete and selects SF-27.
<!-- completion-checklist:end -->

## Advancement rule

SF-27 may prolong accepted lambda-one solutions to finer versions of the same
periodic realization.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-13T16:10Z | activation on `master=23c5555` (SF-24 merged; checker OK next=SF-25) | UNDERSTAND: full re-read of `ContinuationController.hpp` (stage-solver contract: `base_config` supplies every non-overridden field for EVERY stage — baseline, lambda attempts, eta-rescue ramp, epsilon leg — so `config.newton` flows into all stages; the rescue/rollback machinery already consumes the SF-24 structured `newton_exhausted`/`newton_budget_exhausted` exits as `solver_not_converged`), the smoke fixture (`run_heterogeneity_smoke`: R5 Anderson block + degenerate epsilon 1e-2 + verbatim gates reached_target/final_lambda==1/every-accepted-r_F<=1e-6), and the five benchmark YAMLs (smoke32 degenerate-epsilon; the 64^3 quartet with epsilon leg to 1e-6, whose own SF-21 comment fixes "the GATE is reaching lambda=1", epsilon legs recorded where budget allows). PRESPECIFIED decisions (recorded verbatim in `.claude/orchestration/SF-25-heterogeneity-completion/understanding.md` BEFORE implementation): E1 the ONLY sanctioned fixture change is `newton.enabled=true` with SF-24 defaults in the smoke base_config and the five YAMLs (exact analogue of SF-21 R5 Anderson enablement); controller control flow untouched; E2 per-stage attribution = additive HeterogeneityStageRecord Anderson/Newton counters (closing the recorded SF-21 caveat and the SF-24 Anderson-export backlog at stage granularity), additive export/prints, SF-17 record untouched; E3 64^3 budget policy: sequential tmux jobs (025->1->225->4), no artificial kill, 12 h per-variance stop-and-record bound, acceptance gate = lambda=1 per variance, epsilon-leg outcomes recorded as-is; E4 regression: full ctest on the final head expected 16/16 (the known-fail heterogeneity smoke becomes PASSING — the increment's headline), byte-compares vs fresh 23c5555 base refs fully identical mod manifest timestamp/stdout timing; E5 all claims are invariant-CONSTRUCTION claims, no transport claim. Recorded risks: near-singular-Jacobian GMRES budget pressure at sigma^2>=2.25/64^3 (honest fallback chain ends in a recorded lambda-floor failure, not tuning); 64^3 epsilon legs at 1e-6 may be the hard part (gate is lambda=1). | DAG: T01 stage-record counters -> {T02 export+YAML enablement, T03 smoke wiring} (parallel) -> I01 -> F01 (Gate A smoke + full suite + byte-compares, then Gate B 64^3 experiment suite). | Delegate T01. |
| 2026-08-13T18:40Z | T01 `3dabf8b`, T02 `b3812a9`, T03 `61e66f8` all ACCEPTED (compile-gate-only workers; full diffs personally audited) | T01: eight attribution counters added to `HeterogeneityStageRecord` (types exact to `StreamfunctionSolveReport`) and harvested at the single record-builder every one of the five stage-kind call sites routes through; per-stage semantics guaranteed because `newton_jv_evaluations` is a per-solve delta by SF-24 design and each stage is one solve call; SF-17 record and all controller control flow untouched. (First T01 worker was interrupted mid-task by the owner; its partial header edit was verified, and a fresh worker independently implemented the full scope, correctly flagging the stale resume precondition.) T02: the eight counters appended as trailing CSV columns in the heterogeneity `write_stage_history` overload ONLY (the three default byte-compare configs never reach that path); `newton: enabled: true` + one SF-25 comment added to the five benchmark YAMLs and NOTHING else (adaptive strict-parser default `enabled=true` verified in all five, satisfying the C01 validation dependency); the summary.json schema intentionally untouched (per-stage attribution lives in the CSV, consistent with the recorded SF-24 export precedent). T03: `base_config.newton.enabled = true` in `run_heterogeneity_smoke` — the ONLY sanctioned fixture change (E1a, exact analogue of SF-21 R5) — plus attribution counters in the stage-history printer and one ADDITIVE non-gate sanity check (`attribution_counters_populated`); the three moved gates byte-verbatim. | Audits recorded in `.claude/orchestration/SF-25-heterogeneity-completion/` (dag.json node notes). SF-25 base byte-compare refs regenerated on the V100 at `23c5555` (`~/sf25_base_refs/`, now including the SF-24 `newton:` effective-config block, so final byte-compares must be FULLY identical mod manifest timestamp/stdout timing). Integration order: `3dabf8b -> b3812a9 -> 61e66f8`. | I01 integrator, then F01: Gate A smoke + full suite (expected 16/16) + byte-compares, then Gate B 64^3 suite per E3. |
| 2026-08-13T22:20Z | Gate A first run on integrated head `18b3a59` (V100 job `sf25-gateA`; raw logs `/tmp/sf25_smoke.log`, `/tmp/sf25_full_suite.log`) | HONEST SPLIT. sigma^2=0.25: PASS and 2.4x FASTER than SF-21 (205.6 s vs 488.1 s; 8/8 stages, r_F<=1e-6 everywhere, attribution populated) — Newton terminal acceleration works where the map contracts. sigma^2=1: **STILL FAILS** at `lambda_floor_exhausted` lambda=0.5 with the SAME stage trajectory as SF-21 (48/83 accepted, 73 rescue stages) at 4.3x the wall (6834.6 s). Full suite 15/16 (only the sigma^2=1 sub-case); byte-compares vs the 23c5555 refs FULLY clean (stdout identical x3; artifacts differ only by the manifest timestamp — the newton block is now in the base refs as predicted). **Per-stage attribution forensics (the counters wired this increment did their job):** off the eta=1 plateau Newton finishes stages quadratically (newton_acc 2-4, jv 9-228); exactly AT eta=1 for lambda>=0.5 Newton activates, accepts 18-27 monotone steps (r_F ~5e-3 -> ~1e-3) at ~112 Jv PER ACCEPTED STEP — i.e. EVERY inner GMRES solve exhausts its 100-iteration budget (restart 10) and returns partial directions accepted at small alpha — then one step failure ends the activation and the rescue window's Picard steps cannot move on the non-contractive plateau (newton_rescue=0: the stage dies before the retry). Mechanism: the linear solver, not a proven Jacobian singularity — the small-eigenvalue mode the block-MG preconditioner does not damp needs more Krylov space than restart-10/budget-100 provide. This is the SF-24-recorded structural risk materializing, and it is METHOD-parameter territory (existing config surface: newton.gmres.max_iterations unbounded, restart <= 15), not gate territory. **PRESPECIFIED DIAGNOSTIC PROBE (recorded here BEFORE running; explicitly NOT a gate run):** a /tmp-generated variant of the smoke32 pipeline config (repo tree untouched; remote never edited) with sigma2=1.0, newton.gmres.restart=15, newton.gmres.max_iterations=1000, run once through the pipeline binary with the per-stage CSV captured (which also yields the missing per-stage exit_reason forensics). Outcome decides: converges lambda->1 => methodology amendment (raise the two knobs in the five YAMLs + smoke fixture, recorded pre-corrective, gates untouched) then the REAL gate rerun; fails => escalate to the owner as a structured finding with full evidence (candidate follow-up: stronger linear solver — deflation/FGMRES/restart-cap raise — outside SF-25 scope). Probe wall bound: 4 h. | Raw stage tables preserved in the remote logs; stage[57..82] excerpts recorded in the orchestration audit dir. | Run the probe; decide amendment-vs-escalation on its evidence. |
| 2026-08-14T09:30Z | Diagnostic probe result (job `sf25-probe`, /tmp variant, restart=15 + max_iterations=1000, NOT a gate run) | TIMEOUT at the prespecified 4 h bound (exit 124), stdout unflushed, no export point reached — the entire budget was consumed inside the continuation. Inference: a 10x linear budget that actually SOLVED the plateau systems would have made the run FASTER than the 1.9 h failing baseline (quadratic stage finishes), not slower; instead it did not even complete. Together with the Gate-A forensics (~112 Jv per accepted Newton step == every GMRES solve budget-exhausted at the plateau) this is strong evidence that the eta=1/sigma^2>=1 difficulty is NOT curable by Krylov budget within the accepted SF-23 structure (restart cap 15, block-diag(A,A) MG preconditioner blind to the coupling mode). SCIENTIFIC HYPOTHESIS (recorded for the owner): the Clebsch/streamfunction representation is non-unique (functional recombinations of (psi1,psi2) preserve v — Lester et al. 2021); at eta=1 EXACTLY the system inherits that gauge freedom, giving a solution MANIFOLD with nontrivial tangent space and hence a genuinely singular Jacobian at the solution (the mean-zero gauge fixes only additive constants). At eta<1 the homotopy blend breaks the degeneracy — matching the observed cliff (Newton superb at eta=0.98..0.997, grinding exactly at eta=1) — and the epsilon regularization + coupling strength modulate how singular the regularized system is, matching the sigma^2 dependence (0.25 converges at eta=1; >=1 beyond lambda~0.5 does not). This would ALSO explain the paper's own method choice: explicit pseudo-time flows onto the manifold without inverting J, indifferent to tangential singularity. STATUS: the sigma^2=1 Gate-A target is NOT met by the accepted SF-22..24 stack as wired; per the pre-recorded decision tree this is now ESCALATED to the owner as a structured finding rather than further blind probing or any gate relaxation. Options prepared for the owner: (A) grow SF-25 scope with a singular-mode-robust terminal method (pseudo-transient/SER Newton, Levenberg-type shift, or nullspace deflation) — increment-sized; (B) restructure: insert a dedicated increment (terminal-solver robustness for the eta=1 manifold) before SF-25 completion, mirroring the accepted Newton-pull-forward precedent, ideally activation-gated by a matrix-free spectral diagnostic (smallest eigenvalues of J at a frozen plateau state) to choose the method on evidence; (C) an overnight >=19 h brute-budget probe to close the budget question conclusively (high cost, likely confirmatory); (D) accept partial closure of SF-25 (sigma^2=0.25 improvement + attribution + wiring) and move the sigma^2>=1 gates again — NOT recommended without (B). Orchestrator recommendation: (B). | Everything above is method/diagnosis; no gate, seed, tolerance, or budget value was changed at any point. sigma^2=0.25 evidence (PASS, 2.4x faster) and the clean byte-compares stand. | AWAIT OWNER DECISION; SF-25 remains active/incomplete. |
| 2026-08-14T13:30Z | owner decision: option (B); increment PAUSED and renumbered SF-25 -> SF-26 by the restructuring | The owner directed an exhaustive scientific/numerical investigation and the insertion of a dedicated manifold-robust terminal-solver increment (new SF-25) before this completion. Research record: `docs/decisions/2026-08-14-manifold-robust-terminal-solver.md` (gauge-manifold hypothesis, Ψtc/LM survey, shifted-system wiring, falsifiable D-gate). This increment's ACCEPTED work ships with the restructuring PR because SF-25's D-gate depends on it: T01 `3dabf8b` (per-stage attribution counters — the forensic instrument that localized the failure), T02 `b3812a9` (CSV export + sanctioned newton enablement in the five configs), T03 `61e66f8` (smoke wiring + attribution prints), integrated as `747010b/0b6f0c9/18b3a59` (patch-equal). Its GATES are explicitly NOT claimed: sigma^2=1 remains unmet (that is the point of SF-25). Standing evidence that carries: sigma^2=0.25 PASS 2.4x faster (205.6 s vs 488.1 s, 8/8 stages), clean byte-compares, and the full forensic record above. State: `pending`, re-blocked on SF-25. | Restructuring decision + dashboard renumbering in the same PR; checker expected_count 30 -> 31. | Reactivate when named by `NEXT` after SF-25 is done and merged. |
| 2026-08-11T23:30Z | created by the owner-directed option (a) restructuring | This increment inherits, UNCHANGED, the sigma^2>=1 heterogeneity gates that SF-21 closed partially: 32^3 sigma^2=1 (floor-exhausted at lambda=0.5 under Picard/Anderson, non-contractive eta=1 plateau r_F~1e-3, zero degeneracy) and the full 64^3 suite. See `docs/decisions/2026-08-11-newton-before-heterogeneity-completion.md` and the SF-21 bitácora/audit records for the complete evidence. | Gates moved verbatim; no tuning. | Activate only when named by `NEXT` after SF-24 is done. |
