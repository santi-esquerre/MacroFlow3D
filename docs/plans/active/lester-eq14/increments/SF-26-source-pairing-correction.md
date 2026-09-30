# SF-26 — Equation (14) source pairing correction

- State: `active`
- Goal: `Corregir el emparejamiento de las fuentes S1/S2 en el residuo acoplado al sistema derivado (mismo índice), demostrarlo con un par exacto y re-imponer sin cambios los gates de heterogeneidad sigma_Y^2>=1.`
- Depends on: `SF-25`
- Unlocks: `SF-27`
- Branch: `science/lester-sf26-source-pairing-correction`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Code orchestrator (Fable) + delegated workers`
- Started: `2026-09-30`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

`docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md` derives that the
coupled system is `L_i = lap(psi_i) - grad(ln k).grad(psi_i) = S_i` (SAME
index), while the code (`ResidualEvaluator.cu:45`, following the paper's
printed eq. 14) pairs `G1 <- S2`, `G2 <- S1`. An exact Darcy pair
(`k = k(x1)`, `psi1 = x2 + Phi(x3)`, `psi2 = x3`, `v = e1`) gives residual
`2e-14` under the same-index form and `O(1)` under the crossed form. The
increment fixes the pairing, adds the contract tests that would have caught
it, and tests the prespecified prediction that the SF-21/SF-25 wall was a
consequence of solving an unsolvable system: the UNCHANGED `sigma^2 = 1`
gates must now pass with the accepted implicit stack.

## Preconditions

- SF-25 `done` and merged (root-cause decision record on the default branch).
- The accepted implicit stack (SF-14..SF-24) and the heterogeneity fixtures
  (`heterogeneity_continuation_gpu_cases.cu`, the five benchmark YAMLs) are
  unchanged on the default branch.

## In scope

- The pairing fix in `combine_residual_rhs_kernel` (`ResidualEvaluator.cu`)
  and in any CPU reference mirror of the residual assembly
  (`tests/streamfunctions/reference_operators.cpp`); doc comments in
  `ResidualEvaluator.cuh`, `NonlinearSources.cuh` stating the same-index form.
- New Gate 2 contract cases (registered in `streamfunction_operator_tests`):
  1. `coupled_residual_exact_pair_k1`: `k = 1`, `u1 = Phi(x3)`, `u2 = 0`, with
     `Phi` a smooth periodic trigonometric polynomial; `F` converges to zero
     with order 2 on 16/32/64 (normalized `r_F` and `Linf`).
  2. `coupled_residual_exact_pair_kx1`: same pair with `k = exp(0.7 sin 2πx1 +
     0.3 cos 4πx1)` (harmonic-mean face coefficients) — same acceptance.
  3. `coupled_residual_pairing_mutant`: evaluating the same pair with the
     crossed pairing (test-local helper, not a production switch) yields
     `r_F = O(1)` that does NOT decrease under refinement — the contract
     mutant.
  4. `coupled_residual_gauge_recombination`: on a converged `sigma^2 = 0.25`
     state, `u1 <- u1 + alpha * Phi(x3 + u2)` (small periodic `Phi`) keeps
     `r_F` within a factor 3 of the truncation-level residual measured by
     case 1 at that grid.
- Re-run of the moved-verbatim heterogeneity gates (SF-21 prespecification,
  values unchanged): `heterogeneity_smoke_sigma1` (32^3, seed 12345, ell=8,
  epsilon 1e-2, every accepted stage `r_F <= 1e-6`, `lambda = 1`) and the
  64^3 suite `sigma^2 in {0.25, 1, 2.25, 4}` through the pipeline binary,
  recorded as an experiment note with `e_v(h)` for 32^3 vs 64^3.
- Re-classification of the SF-25 instrument: `streamfunction_terminal_dgate`
  and the heavy `terminal_*` cases assert on a frozen plateau that no longer
  exists; convert their E2 hard asserts into recorded-evidence prints or
  retire the ctest entry, documented in the bitácora (no silent deletion).

## Out of scope

- Any solver-parameter, tolerance, seed, or fixture change (a prediction that
  fails is recorded, not tuned away).
- The pseudo-time solver (SF-27), grids beyond 64^3, tracking.

## Files and symbols

- `src/physics/streamfunctions/ResidualEvaluator.cu` (`combine_residual_rhs_kernel`),
  `ResidualEvaluator.cuh`, `NonlinearSources.cuh` (comments only).
- `tests/streamfunctions/reference_operators.{hpp,cpp}` (residual reference
  pairing, if mirrored), `tests/streamfunctions/coupled_residual_gpu_cases.cu`,
  `streamfunction_operator_test_cases.hpp`, `CMakeLists.txt` (ctest entries).
- `tests/streamfunctions/terminal_solver_gpu_cases.cu` (assert re-classification).
- `docs/experiments/` note for the gate re-run.

## Implementation specification

1. Change the two lines of `combine_residual_rhs_kernel` to
   `g1 = rhs1 - eta*q*s1`, `g2 = rhs2 - eta*q*s2`; mirror in the CPU
   reference; update comments to the derived form and cite the decision.
2. Implement the four contract cases with a shared exact-pair fixture; the
   mutant uses a test-local kernel/host recombination, never a config flag.
3. Rebuild locally (WSL debug) and run the fast targeted operator/coupled-
   residual cases — the SF-09/SF-10 GPU-vs-CPU agreement cases must remain
   green (both sides change together). The full suite is long-duration
   computation and runs on V100 as a detached job (step 4), not locally.
4. On the V100, as detached jobs (`scripts/remote run` + `scripts/remote
   wait`, one at a time — the remote mirror is shared state, do not overlap
   jobs): full ctest, then `heterogeneity_smoke_sigma1`, then the 64^3 quartet
   sequentially (12 h per-variance bound, stop-and-record).
5. Record Gate 3A metrics per accepted stage and the `e_v`, invariance, and
   `e_div` comparison 32^3 vs 64^3 on the same dimensionless problem.

## Expected numerical effect

- Exact-pair residuals: `O(h^2)` instead of `O(1)`.
- `sigma^2 = 0.25` results change at second order in the field amplitude
  (`e_v` expected to DROP from ~2.5 %); byte-compares against pre-fix
  references are expected to differ and are not an acceptance criterion.
- Prespecified prediction: `sigma^2 = 1` 32^3 reaches `lambda = 1`,
  `r_F <= 1e-6` per stage; 64^3 quartet reaches `lambda = 1`; `e_v(h)` decreases
  by ~4x from 32^3 to 64^3 at fixed `L/ell`.

## Validation commands

Local (fast dev loop):

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
./build/wsl-debug/streamfunction_operator_tests --case coupled_residual_exact_pair_k1
./build/wsl-debug/streamfunction_operator_tests --case coupled_residual_pairing_mutant
```

Remote V100 (long-duration computation — detached jobs, run one at a time;
the remote mirror is shared state, do not overlap jobs — see
`docs/runbooks/remote-v100.md` Section 0):

```bash
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote run ctest-full -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote wait ctest-full
scripts/remote run sf26-smoke32 -- "./build/v100-release/macroflow3d_pipeline apps/config_streamfunctions_gaussian_smoke32.yaml"
scripts/remote wait sf26-smoke32
scripts/remote run sf26-64var1 -- "./build/v100-release/macroflow3d_pipeline apps/config_streamfunctions_gaussian_64_var1.yaml"
scripts/remote wait sf26-64var1
```

## Acceptance thresholds

- Cases 1-2: observed order `>= 1.9` on 16->32->64; case 3: mutant `r_F > 1e-2`
  at every grid; case 4: ratio `<= 3`.
- Full suite green on WSL and V100 (after the instrument re-classification).
- `sigma^2 = 1` 32^3 smoke: `lambda = 1`, all accepted stages `r_F <= 1e-6`.
- 64^3 suite: `lambda = 1` for all four variances (epsilon legs recorded as-is).
- Every accepted stage reports `r_F`, `e_v`, invariance, `e_div`, `|c|`
  percentiles; `e_v` 32^3 vs 64^3 recorded.

## Regression surface

- Every Picard/Anderson/Newton fixture value that encoded the crossed system
  (e.g. recorded `r_F` histories, the `lambda* = 0.1125` stall fixture, the
  frozen plateau `1.1204722529922055e-3`): numbers will change; fixtures that
  assert bitwise values must be re-baselined with a bitácora entry each.
- SF-16 default-pipeline byte-compares (streamfunction stage disabled by
  default: expected unchanged).

## Failure and rollback policy

- If the prediction fails (wall persists after the fix), keep the fix
  (it is correct by derivation), record the evidence, and re-open the
  campaign report's H1/H2 hypotheses; SF-27 then becomes the discriminating
  instrument. No gate is weakened.
- Any need to change a gate value is a recorded failure of this increment.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Pairing corrected in the residual evaluator and every CPU mirror; comments cite the decision.
- [ ] Exact-pair, mutant, and gauge-recombination contract cases pass with recorded orders.
- [ ] Full suite green (WSL + V100) after the SF-25 instrument re-classification.
- [ ] The moved 32^3 sigma^2=1 gate passes unchanged.
- [ ] The 64^3 suite meets the moved acceptance set unchanged; `e_v(h)` recorded.
- [ ] Gate 3A metrics and experiment note recorded; human review recorded.
- [ ] Evidence, PR, and commit are recorded; dashboard marks SF-26 complete and selects SF-27.
<!-- completion-checklist:end -->

## Advancement rule

SF-27 may implement the pseudo-time solver on the corrected residual only after
this increment is merged and its prediction outcome is recorded either way.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-09-30T00:00Z | created by the owner-directed re-sequencing (decision 2026-09-30) | This increment replaces the former "SF-26 — Heterogeneity completion": its sigma^2>=1 gates are re-imposed here verbatim, now downstream of the pairing correction that the root-cause analysis identifies as the reason those gates were unreachable. The former increment's bitácora rows are preserved below for continuity. | `docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md` | Activate only when named by `NEXT` after SF-25 closure is merged. |
| 2026-08-13T16:10Z | activation on `master=23c5555` (SF-24 merged; checker OK next=SF-25) | UNDERSTAND: full re-read of `ContinuationController.hpp` (stage-solver contract: `base_config` supplies every non-overridden field for EVERY stage — baseline, lambda attempts, eta-rescue ramp, epsilon leg — so `config.newton` flows into all stages; the rescue/rollback machinery already consumes the SF-24 structured `newton_exhausted`/`newton_budget_exhausted` exits as `solver_not_converged`), the smoke fixture (`run_heterogeneity_smoke`: R5 Anderson block + degenerate epsilon 1e-2 + verbatim gates reached_target/final_lambda==1/every-accepted-r_F<=1e-6), and the five benchmark YAMLs (smoke32 degenerate-epsilon; the 64^3 quartet with epsilon leg to 1e-6, whose own SF-21 comment fixes "the GATE is reaching lambda=1", epsilon legs recorded where budget allows). PRESPECIFIED decisions (recorded verbatim in `.claude/orchestration/SF-25-heterogeneity-completion/understanding.md` BEFORE implementation): E1 the ONLY sanctioned fixture change is `newton.enabled=true` with SF-24 defaults in the smoke base_config and the five YAMLs (exact analogue of SF-21 R5 Anderson enablement); controller control flow untouched; E2 per-stage attribution = additive HeterogeneityStageRecord Anderson/Newton counters (closing the recorded SF-21 caveat and the SF-24 Anderson-export backlog at stage granularity), additive export/prints, SF-17 record untouched; E3 64^3 budget policy: sequential tmux jobs (025->1->225->4), no artificial kill, 12 h per-variance stop-and-record bound, acceptance gate = lambda=1 per variance, epsilon-leg outcomes recorded as-is; E4 regression: full ctest on the final head expected 16/16 (the known-fail heterogeneity smoke becomes PASSING — the increment's headline), byte-compares vs fresh 23c5555 base refs fully identical mod manifest timestamp/stdout timing; E5 all claims are invariant-CONSTRUCTION claims, no transport claim. Recorded risks: near-singular-Jacobian GMRES budget pressure at sigma^2>=2.25/64^3 (honest fallback chain ends in a recorded lambda-floor failure, not tuning); 64^3 epsilon legs at 1e-6 may be the hard part (gate is lambda=1). | DAG: T01 stage-record counters -> {T02 export+YAML enablement, T03 smoke wiring} (parallel) -> I01 -> F01 (Gate A smoke + full suite + byte-compares, then Gate B 64^3 experiment suite). | Delegate T01. |
| 2026-08-13T18:40Z | T01 `3dabf8b`, T02 `b3812a9`, T03 `61e66f8` all ACCEPTED (compile-gate-only workers; full diffs personally audited) | T01: eight attribution counters added to `HeterogeneityStageRecord` (types exact to `StreamfunctionSolveReport`) and harvested at the single record-builder every one of the five stage-kind call sites routes through; per-stage semantics guaranteed because `newton_jv_evaluations` is a per-solve delta by SF-24 design and each stage is one solve call; SF-17 record and all controller control flow untouched. (First T01 worker was interrupted mid-task by the owner; its partial header edit was verified, and a fresh worker independently implemented the full scope, correctly flagging the stale resume precondition.) T02: the eight counters appended as trailing CSV columns in the heterogeneity `write_stage_history` overload ONLY (the three default byte-compare configs never reach that path); `newton: enabled: true` + one SF-25 comment added to the five benchmark YAMLs and NOTHING else (adaptive strict-parser default `enabled=true` verified in all five, satisfying the C01 validation dependency); the summary.json schema intentionally untouched (per-stage attribution lives in the CSV, consistent with the recorded SF-24 export precedent). T03: `base_config.newton.enabled = true` in `run_heterogeneity_smoke` — the ONLY sanctioned fixture change (E1a, exact analogue of SF-21 R5) — plus attribution counters in the stage-history printer and one ADDITIVE non-gate sanity check (`attribution_counters_populated`); the three moved gates byte-verbatim. | Audits recorded in `.claude/orchestration/SF-25-heterogeneity-completion/` (dag.json node notes). SF-25 base byte-compare refs regenerated on the V100 at `23c5555` (`~/sf25_base_refs/`, now including the SF-24 `newton:` effective-config block, so final byte-compares must be FULLY identical mod manifest timestamp/stdout timing). Integration order: `3dabf8b -> b3812a9 -> 61e66f8`. | I01 integrator, then F01: Gate A smoke + full suite (expected 16/16) + byte-compares, then Gate B 64^3 suite per E3. |
| 2026-08-13T22:20Z | Gate A first run on integrated head `18b3a59` (V100 job `sf25-gateA`; raw logs `/tmp/sf25_smoke.log`, `/tmp/sf25_full_suite.log`) | HONEST SPLIT. sigma^2=0.25: PASS and 2.4x FASTER than SF-21 (205.6 s vs 488.1 s; 8/8 stages, r_F<=1e-6 everywhere, attribution populated) — Newton terminal acceleration works where the map contracts. sigma^2=1: **STILL FAILS** at `lambda_floor_exhausted` lambda=0.5 with the SAME stage trajectory as SF-21 (48/83 accepted, 73 rescue stages) at 4.3x the wall (6834.6 s). Full suite 15/16 (only the sigma^2=1 sub-case); byte-compares vs the 23c5555 refs FULLY clean (stdout identical x3; artifacts differ only by the manifest timestamp — the newton block is now in the base refs as predicted). **Per-stage attribution forensics (the counters wired this increment did their job):** off the eta=1 plateau Newton finishes stages quadratically (newton_acc 2-4, jv 9-228); exactly AT eta=1 for lambda>=0.5 Newton activates, accepts 18-27 monotone steps (r_F ~5e-3 -> ~1e-3) at ~112 Jv PER ACCEPTED STEP — i.e. EVERY inner GMRES solve exhausts its 100-iteration budget (restart 10) and returns partial directions accepted at small alpha — then one step failure ends the activation and the rescue window's Picard steps cannot move on the non-contractive plateau (newton_rescue=0: the stage dies before the retry). Mechanism: the linear solver, not a proven Jacobian singularity — the small-eigenvalue mode the block-MG preconditioner does not damp needs more Krylov space than restart-10/budget-100 provide. This is the SF-24-recorded structural risk materializing, and it is METHOD-parameter territory (existing config surface: newton.gmres.max_iterations unbounded, restart <= 15), not gate territory. **PRESPECIFIED DIAGNOSTIC PROBE (recorded here BEFORE running; explicitly NOT a gate run):** a /tmp-generated variant of the smoke32 pipeline config (repo tree untouched; remote never edited) with sigma2=1.0, newton.gmres.restart=15, newton.gmres.max_iterations=1000, run once through the pipeline binary with the per-stage CSV captured (which also yields the missing per-stage exit_reason forensics). Outcome decides: converges lambda->1 => methodology amendment (raise the two knobs in the five YAMLs + smoke fixture, recorded pre-corrective, gates untouched) then the REAL gate rerun; fails => escalate to the owner as a structured finding with full evidence (candidate follow-up: stronger linear solver — deflation/FGMRES/restart-cap raise — outside SF-25 scope). Probe wall bound: 4 h. | Raw stage tables preserved in the remote logs; stage[57..82] excerpts recorded in the orchestration audit dir. | Run the probe; decide amendment-vs-escalation on its evidence. |
| 2026-08-14T09:30Z | Diagnostic probe result (job `sf25-probe`, /tmp variant, restart=15 + max_iterations=1000, NOT a gate run) | TIMEOUT at the prespecified 4 h bound (exit 124), stdout unflushed, no export point reached — the entire budget was consumed inside the continuation. Inference: a 10x linear budget that actually SOLVED the plateau systems would have made the run FASTER than the 1.9 h failing baseline (quadratic stage finishes), not slower; instead it did not even complete. Together with the Gate-A forensics (~112 Jv per accepted Newton step == every GMRES solve budget-exhausted at the plateau) this is strong evidence that the eta=1/sigma^2>=1 difficulty is NOT curable by Krylov budget within the accepted SF-23 structure (restart cap 15, block-diag(A,A) MG preconditioner blind to the coupling mode). SCIENTIFIC HYPOTHESIS (recorded for the owner): the Clebsch/streamfunction representation is non-unique (functional recombinations of (psi1,psi2) preserve v — Lester et al. 2021); at eta=1 EXACTLY the system inherits that gauge freedom, giving a solution MANIFOLD with nontrivial tangent space and hence a genuinely singular Jacobian at the solution (the mean-zero gauge fixes only additive constants). At eta<1 the homotopy blend breaks the degeneracy — matching the observed cliff (Newton superb at eta=0.98..0.997, grinding exactly at eta=1) — and the epsilon regularization + coupling strength modulate how singular the regularized system is, matching the sigma^2 dependence (0.25 converges at eta=1; >=1 beyond lambda~0.5 does not). This would ALSO explain the paper's own method choice: explicit pseudo-time flows onto the manifold without inverting J, indifferent to tangential singularity. STATUS: the sigma^2=1 Gate-A target is NOT met by the accepted SF-22..24 stack as wired; per the pre-recorded decision tree this is now ESCALATED to the owner as a structured finding rather than further blind probing or any gate relaxation. Options prepared for the owner: (A) grow SF-25 scope with a singular-mode-robust terminal method (pseudo-transient/SER Newton, Levenberg-type shift, or nullspace deflation) — increment-sized; (B) restructure: insert a dedicated increment (terminal-solver robustness for the eta=1 manifold) before SF-25 completion, mirroring the accepted Newton-pull-forward precedent, ideally activation-gated by a matrix-free spectral diagnostic (smallest eigenvalues of J at a frozen plateau state) to choose the method on evidence; (C) an overnight >=19 h brute-budget probe to close the budget question conclusively (high cost, likely confirmatory); (D) accept partial closure of SF-25 (sigma^2=0.25 improvement + attribution + wiring) and move the sigma^2>=1 gates again — NOT recommended without (B). Orchestrator recommendation: (B). | Everything above is method/diagnosis; no gate, seed, tolerance, or budget value was changed at any point. sigma^2=0.25 evidence (PASS, 2.4x faster) and the clean byte-compares stand. | AWAIT OWNER DECISION; SF-25 remains active/incomplete. |
| 2026-08-14T13:30Z | owner decision: option (B); increment PAUSED and renumbered SF-25 -> SF-26 by the restructuring | The owner directed an exhaustive scientific/numerical investigation and the insertion of a dedicated manifold-robust terminal-solver increment (new SF-25) before this completion. Research record: `docs/decisions/2026-08-14-manifold-robust-terminal-solver.md` (gauge-manifold hypothesis, Ψtc/LM survey, shifted-system wiring, falsifiable D-gate). This increment's ACCEPTED work ships with the restructuring PR because SF-25's D-gate depends on it: T01 `3dabf8b` (per-stage attribution counters — the forensic instrument that localized the failure), T02 `b3812a9` (CSV export + sanctioned newton enablement in the five configs), T03 `61e66f8` (smoke wiring + attribution prints), integrated as `747010b/0b6f0c9/18b3a59` (patch-equal). Its GATES are explicitly NOT claimed: sigma^2=1 remains unmet (that is the point of SF-25). Standing evidence that carries: sigma^2=0.25 PASS 2.4x faster (205.6 s vs 488.1 s, 8/8 stages), clean byte-compares, and the full forensic record above. State: `pending`, re-blocked on SF-25. | Restructuring decision + dashboard renumbering in the same PR; checker expected_count 30 -> 31. | Reactivate when named by `NEXT` after SF-25 is done and merged. |
| 2026-08-11T23:30Z | created by the owner-directed option (a) restructuring | This increment inherits, UNCHANGED, the sigma^2>=1 heterogeneity gates that SF-21 closed partially: 32^3 sigma^2=1 (floor-exhausted at lambda=0.5 under Picard/Anderson, non-contractive eta=1 plateau r_F~1e-3, zero degeneracy) and the full 64^3 suite. See `docs/decisions/2026-08-11-newton-before-heterogeneity-completion.md` and the SF-21 bitácora/audit records for the complete evidence. | Gates moved verbatim; no tuning. | Activate only when named by `NEXT` after SF-24 is done. |
| 2026-09-30T19:30Z | activation on `master=b86602b` (PR #39 merged 15:35Z; checker OK next=SF-26); delivery branch `science/lester-sf26-source-pairing-correction` | UNDERSTAND: full read of the harness docs, Lester 2023 (all 24 pp.) and Lester 2021 (all 31 pp.), the SF-21..SF-25 bitácoras/campaign report, and the real code. The same-index derivation was re-done independently by hand from the paper's own definitions of `B`, `S_i` on the exact pair (`c = e1`, `B = (0,0,-Phi'')`, `B x grad psi1 = (Phi'',0,0)` => `S1 = Phi'' = L1`, `S2 = 0 = L2`) and cross-checked against Lester 2021: its eq. (2.20) implies `L2 = a1`, `L1 = a2` (2023: `S1 == a2`, `S2 == a1`), i.e. the printed (2.22)/(2.23) and the printed 2023 eq. (14) both carry the same index swap. Code confirmed: `ResidualEvaluator.cu` combine kernel (crossed), CPU mirror `reference_operators.cpp:943`, and `coupled_residual_mutation_sensitivity` whose mutant (a) is literally the correct form. **PRESPECIFIED amendments (recorded BEFORE any run, no gate value changed):** (A1) discrete analysis shows the spec's exact-pair cases 1-2 (`u1 = Phi(x3)`, `u2 = 0`, `k = 1` or `k(x1)`) are EXACT DISCRETE solutions of the same-index system (the HVP `d3d3` 3-point stencil equals the operator's; z-face harmonic means of equal `q` are exact; `mean(q(x1) Phi''_h(x3)) = 0` by separability), so their residual is roundoff, not O(h^2), and the spec's `order >= 1.9` is unmeasurable on them. Resolution: gate cases 1-2 at roundoff (`r_F <= 1e-12`, `Linf <= 1e-10 max|Phi''| q_max`, epsilon = 0, grids 16/32/64) and ADD case 2b `coupled_residual_exact_pair_general`: `psi1 = x2 + Phi(x3)`, `psi2 = x3 + Theta(x2)`, `k = k1(x1)(1 - Phi' Theta')` (`|Phi' Theta'| < 1`), an exact Darcy pair with `v = (1 - Phi' Theta') e1`, `S1 = (Phi'' + Phi' Theta'')/f`, `S2 = (Theta'' + Phi'' Theta')/f`, on which O(h^2) is genuine: order `>= 1.9` gated there. (A2) case 4's criterion "within a factor 3 of the truncation-level residual of case 1" is ill-posed for the same reason; replaced by: on the SAME recombined converged sigma^2=0.25 32^3 state, `r_F(same-index) <= (1/20) r_F(crossed, test-local)` (expected ~1/300) and `r_F(same-index) <= 1e-1`, plus a cheap analytic recombination of the case-2b pair gated at ratio `<= 3` to its own base residual and order `>= 1.9`. (A3) instrument re-classification option chosen: E2 hard asserts of `terminal_dgate_diagnostic`, `terminal_shelf_probe_phase1`, `terminal_explicit_flow_probe` become recorded-evidence prints with an `E2_PLATEAU_ABSENT` branch; the three cases become always-pass evidence recorders; the `streamfunction_terminal_dgate` ctest entry is KEPT so the V100 suite log records whether the plateau still exists. (A4) V100 logistics: two V100-32GB GPUs are available; the 64^3 quartet may run two-at-a-time on separate GPUs (budget policy, not a gate). Regression watch: `anderson_stall_fixture_a/b` and `newton_difficult_case` assert that Picard-only stalls at `lambda* = 0.1125`; if the control now converges they are re-baselined by a corrective node with a bitácora entry each. | DAG: T01 (fix + CPU mirror + existing mutant) ∥ T03 (instrument re-classification) -> T02 (contract cases) -> I01 -> F01 (WSL cheap suite; V100 full ctest, smoke32 sigma^2=1, 64^3 quartet, Gate 1 smoke byte-compare vs base refs generated on the untouched remote mirror before sync). Runtime record: `.claude/orchestration/SF-26-source-pairing-correction/{understanding.md,dag.json}`. | Delegate T01 and T03 in parallel. |
| 2026-09-30T22:40Z | T01 `b9bd251` ACCEPTED (fix + CPU mirror + mutant inverted), T03 `ca1b8b9` ACCEPTED (instrument re-classified); T02 and corrective C01 delegated | T01's cheap-tier run exposed three fixtures that encoded crossed-system behaviour: `coupled_residual_direct_agreement` (manual host mirror still crossed, `eta1_norm_linf = 17.09`; folded into T02), `picard_adaptive_fixed_mode_equivalence` (hard-coded 40 fixed-omega Picard iterations; the same-index system converges in 131, `r_F = 9.7e-7`), and the two SF-23 GMRES contracts at `eta = 1` (`gmres_dense_lu_oracle` rel. error 0.109 vs gate 1e-4; `gmres_iteration_reduction` both solves pinned at 250). **SCIENTIFIC FINDING (independent numpy FD-Jacobian diagnostic, recorded BEFORE any gate run; script + outputs in `.claude/orchestration/SF-26-source-pairing-correction/`):** on the exact non-separable Darcy pair (`psi1 = x2 + 0.1 sin 2πx3`, `psi2 = x3 + 0.1 cos 2πx2`, `k = k1(x1)(1 - Phi'Theta')`), the same-index discrete residual converges at order ≈1.9 (`r_F` = 8.15e-2 / 3.80e-2 / 2.20e-2 at 8/12/16^3) while the crossed residual stays O(1) (4.2-4.5). The same-index Jacobian at that solution carries, beyond the two constant modes, a NEAR-NULL CLUSTER that sharpens with refinement (smallest non-trivial relative singular value 1.7e-4 / 7.8e-6 / 4.8e-6; modes below 1e-3: 10 / 24 / 44), and a constructed gauge-tangent direction (`delta = (dH/dpsi2, -dH/dpsi1)`, `H = sin 2πpsi1 sin 2πpsi2`) satisfies `|J t|/|t|` = 4.6 % / 1.0 % / 0.35 % of a random direction's. The crossed system's Jacobian has no such cluster (smallest non-trivial 3.0e-3 / 6.7e-4 / 2.2e-4, i.e. the ordinary elliptic low-mode scaling). Interpretation: the corrected equation is exactly equivariant under label recombination (`F' = M F`, `M = [[1, Phi'],[0,1]]` in the continuum), so every solution is a point of a manifold and `J` is singular there in the continuum limit — the SF-25 gauge-manifold hypothesis (H4) is TRUE for the correct system, and false for the crossed one it was tested on. Consequences prespecified: (i) the GMRES contract tests are re-characterized with a CONSISTENT rhs (`b = J x_true`; residual-based gate at eta = 1; singular-value readout printed as evidence) — no gate value of any heterogeneity fixture changes; (ii) the σ² ≥ 1 gates run with the UNCHANGED stack (Newton enabled per SF-25 E1); if Newton's inner GMRES budget-exhausts on the gauge cluster, Picard/Anderson (neutral along gauge directions, contractive elsewhere) still carry the stage, and the outcome is recorded either way; (iii) all Gate 3A acceptance metrics are gauge-invariant. The Lester 2021 §3 manufactured field (`f = sin2πx1 cos2πx2 sin2πx3 + 0.4 sin2πx1 sin8πx3`, N = 40/80/160) is noted as a candidate paper-parity control for SF-27/28 (out of scope here). | Audits: `.claude/orchestration/SF-26-source-pairing-correction/audits/{T01,T03}.md`; DAG updated with C01. | Await T02 and C01; then I01 integration; then V100 gates. |
| 2026-10-01T13:00Z | delivery branch rebased onto `master=abd873c` (PR #40: harness mandates detached V100 jobs for long-duration computation; remote mirror is single-flight shared state) | Session interrupted mid-T02/C02 (both uncommitted; partial diffs preserved and handed to fresh workers). Rebase clean (docs-only upstream change; source base for the increment remains `b86602b`, worker commits `b9bd251`, `ca1b8b9`, `3653880` cherry-pick unchanged). Execution policy from here on, per the updated `docs/runbooks/remote-v100.md` §0: workers validate locally ONLY with build + fast targeted `--case` runs (seconds to ~1 min); every heavy run (full ctest, `gmres_iteration_reduction`, the heavy gauge-recombination case, the terminal instrument, heterogeneity smokes, the 64^3 quartet, byte-compare smokes) is an orchestrator-owned detached `scripts/remote run` job on V100, serialized (one job at a time, `scripts/remote status` checked before every sync/run), executed on the integrated head. Base references for the byte-compares were already generated on the untouched mirror (`~/sf26_base_refs`, job `sf26-base-refs`, all three runs exit 0) before any sync. The earlier `jvp_converged_picard_base` discrepancy (1.05e6) seen by C01 under concurrent GPU use was re-run by the orchestrator on an idle GPU: PASS (3.75e-8) — contention artifact, no code issue. | `git log`: 7cde5e0; `.claude/orchestration/SF-26-source-pairing-correction/dag.json` updated (C02 added; validation venue re-assigned). | Relaunch C02 and T02 (local fast validation only); then I01; then serialized V100 gates. |
| 2026-10-01T15:30Z | T02 `64fbd9a` ACCEPTED WITH CORRECTIVE; C01 `3653880` accepted (intermediate); C02 `19bccae` ACCEPTED; C03 delegated | T02 (fast local runs, idle GPU): pair A (`k = 1` and `k = k1(x1)`) residual at roundoff at 16/32/64 (`r_F` 2e-15..9e-15, prespecified A1 confirmed); crossed mutant `r_F` 9.8-10.0 (pair A) and 4.45-4.55 (pair B), ratio(64/16) 1.02 (O(1), non-decreasing); analytic recombination ratio 1.70 at every grid with orders 1.952 / 1.988; general pair `r_F` 2.197e-2 / 5.658e-3 / 1.425e-3 (orders 1.957 / 1.989), `Linf` 0.1245 / 0.03404 / 0.008719 (orders **1.871** / 1.965); `direct_agreement` manual mirror fixed (2e-15). **PRESPECIFIED amendment A5 (before re-run):** the 16->32 `Linf` order is pre-asymptotic (the `cos 4πx1` mode of `k1` has 8 cells per wavelength at 16^3); the threshold 1.9 is unchanged and the ladder is extended to 16/32/64/128 with the gate on the two finest transitions, 16->32 recorded as evidence (C03). GMRES contracts: C01's consistent rhs left `gmres_dense_lu_oracle` at 2.905e-3 (eta = 1, restart 10, 400 its) and both `gmres_iteration_reduction` solves unconverged at 250 (true residuals pc/id 4.87e-4 / 1.40e-3 at 16^3, 6.49e-5 / 6.30e-4 at 32^3); orchestrator numpy reproduction on an independent J (8^3): restart-10 GMRES 2.45e-3 vs full GMRES 1.9e-14, crossed J restart-10 5.5e-11 ⇒ structural stagnation of restarted GMRES on the gauge cluster of the CORRECT equation, not an implementation defect. C02 re-states the eta = 1 gates well-posedly without touching any tolerance/budget: GPU restarted residual ≤ 3·max(host textbook restarted-GMRES(10) residual, 1e-6) [2.905e-3 vs host 2.931e-3, bound 8.79e-3] and host full-recurrence GMRES ≤ 1e-8 [8.84e-11]; `gmres_iteration_reduction` gates on preconditioner efficacy at fixed budget (pc ≤ 0.5·id true residual, pc ≤ 1e-2; iteration comparison only if converged); the non-converging Jacobi singular-value readout was removed. `picard_adaptive_fixed_mode_equivalence` re-baselined 40 -> 131 iterations (converged, `r_F` 9.696e-7; second-order change of the fixed-omega Picard map). Design implication recorded for SF-27+: at eta = 1 Newton-Krylov's inner restart-10 GMRES cannot resolve the gauge cluster; the pseudo-time solver (SF-27) flows onto the manifold without inverting J. | Audits: `.claude/orchestration/SF-26-source-pairing-correction/audits/{T02,C01,C02}.md`; numpy GMRES probe `gmres_probe.py` in the orchestration dir. | C03; then I01 (chains: T01 -> T02 -> C03; T01 -> C01 -> C02; T03); then serialized V100 gates. |
