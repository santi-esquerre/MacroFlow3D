# SF-43 — Confirmatory Monte Carlo study of macrodispersion with quantified error budgets

- State: `awaiting_review`
- Goal: `Medir alpha_L y alpha_T a sigma^2 en {0.25, 1, 2.25, 4} con un Monte Carlo confirmatorio de semillas independientes y estimadores por planos de control sobre una ventana común de recorrido físico, con cada estimación acompañada de un presupuesto de error cuantificado (muestreo, discretización con orden observado, ancho transversal e integración; la representación como contraste reportado), el contraste de alpha_L con Par2 pareado sobre el mismo campo y atribuido por un factorial, y el signo de alpha_T decidido por una prueba robusta a esos sesgos con control de multiplicidad.`
- Depends on: `SF-39, SF-42`
- Unlocks: `none`
- Branch: `science/lester-sf43-confirmatory-macrodispersion`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 4 + Gate 5 as specified by decision 2026-10-29-sf39-remediation-sf42-sf43 (G5.4-G5.7: robust sign with error budget and Holm; matched paired ratio equivalence with power and factorial attribution)`
- Human review: `required`
- Owner: `Claude Opus 5.5 orchestrator (session 2026-11-02)`
- Started: `2026-11-02T18:20Z on base 0755a008`
- Completed: `not completed`
- PR: `#87 https://github.com/santi-esquerre/MacroFlow3D/pull/87`
- Commit: `source-bearing audited head fd3741bdb407ce9afbcaf045c5d05ae70c7cf3ca` (results, experiment note and citations; pre-registration `cbbd69da`; production tree `e85ec5a1`; freeze row `40135561`)

## Scientific or engineering intent

Decision record `docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md`. SF-39 is the exploratory pilot: its transverse result is
plausible but rests on one production-domain refinement with an assumed order, unconverged widths, amplitude-dependent time windows that
select surviving particles, and a Gate 5 that could neither resolve nor explain an offset (W1-W13). SF-43 re-measures the macrodispersivities
with independent seeds and a design fixed before any run, so that every reported number carries a quantified error budget and the few
primary claims are confirmatory at a stated family-wise error rate. Model: purely advective transport in smooth, locally isotropic Gaussian
log-conductivity fields (`C(r) = sigma^2 exp(-(r/lambda)^2)`, B&dD eq. 1), medium (b), periodic transverse boundaries; `alpha_T` is not
presupposed (`docs/theory/lester-2023-key-claims.md`, `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md`).

## Preconditions

- SF-39 and SF-42 `done` on the default branch; SF-42's acceptance D (MMS) and E (spectral) met, or owner decisions recorded on budgeting
  them.
- A pre-activation review (docs only, no data, no GPU) merged on the default branch, as for SF-39 (decision 2026-10-27); its merge commit is
  the increment base.
- The pre-registration (artifact `preregistration.md`) committed and pushed to a draft PR of the SF-43 branch before the first run (no
  force-push), its sha256 in a new bitácora row; it fixes every number left open below from SF-39 and SF-42 data only.
- The owner budget of decision G10 (up to about 150 GPU-h, one GPU).

## In scope

1. **Estimands (G4), on control planes in both codes.** For each `sigma^2` in {0.25, 1, 2.25, 4}: `alpha_TP`, `alpha_TAP` (transverse) and
   `alpha_LP` (travel-time longitudinal), per-realization least-squares slopes over planes `x0 + d`, `d` in `[25, 94] lambda` from the
   injection plane, in boxes of 104 `lambda` with 5-`lambda` buffers (definitions in decision G4; transverse crossing variance `(var x2 +
   var x3) / 2`; for `alpha_LP` the estimand is the mean of per-realization values); moments centred per realization; the stationarity
   reading (the two halves `[25, 59.5]` and `[59.5, 94]`) reported as an interval, no claim without a pre-registered equivalence margin.
   Secondary (descriptive): SF-39-style time-window estimators ending at each realization's first exit; the B&dD-style window; the
   displacement-based transverse variants (if SF-42's estimator-noise pilot makes one of them primary, the pre-registration says so); the
   `exp(-I)` weight-sensitivity check (same paths, not a comparator); Par2's transverse dispersivity in the matched configuration (window
   injection, displacement-based).
2. **Production configurations within the V100 cap (G3)** (planning; the pre-registration fixes them). The source is a centred square of
   fixed size `S` in `lambda` per amplitude, the same in every arm of that amplitude (decision G4), flux weights, lattice `M` admissible by
   the SF-42 seed-lattice rule:

   | `sigma^2` | width (`lambda`) | `lambda/h` | grid | cells | source `S` |
   |---|---|---|---|---|---|
   | 0.25, 1 | 24 | 10 | 1040x240x240 | 6.0e7 | 19.2 `lambda` |
   | 2.25 | 24 | 16 | 1664x384x384 | 2.45e8 | 19.2 `lambda` |
   | 4 | 12 | 32 | 3328x384x384 | 4.9e8 (cap) | 9.6 `lambda` |

3. **Discretization budget `E_h` (G5.4).** Grid sequences on the production box that contain the production level, ratio-2 steps where they
   fit, paired by seed: planning `sigma^2` 0.25 and 1 at `lambda/h` 5 / 10 / 20 (production in the middle; the finest 2080x480x480, 4.8e8
   cells); 2.25 at 8 / 16 plus 4 or 20 (chosen in the pre-registration from the SF-42 grid pilot); 4 at 8 / 16 / 32 (production finest, at the
   cap; the asymptotic range is a stated assumption). Sized for a target `E_h` relative to the planning `alpha_T`.
4. **Width budget `E_W` (G5.4).** Unpaired width arms with the amplitude's fixed source (no arm narrower than the source): widths 24 and 48
   `lambda` at `lambda/h` 10 for `sigma^2` 0.25 and 1 (the production `lambda/h`; 48 at 10 is 2.4e8 cells); at 2.25, widths 24 and 48 at
   `lambda/h` 8 and 10 (a 2x2 interaction test; the transfer to 16 is a stated assumption); at 4, widths 12 and 24 at `lambda/h` 8 and 16
   (2x2; the transfer to 32 is a stated assumption). `E_W` bounds `|alpha(2 W_prod) - alpha(W_prod)|`. Sized jointly with the production
   arms for the Gate 4 power target.
5. **Integration and representation.** `h_max` passes 8 / 16 / 32 on a subset, with the binding step control recorded; `E_int` as decision
   G5.4; the Pollock bridge on every realization (reported, not in the budget).
6. **Sampling (G5.2, G5.3).** `N` fixed in the pre-registration from absolute targets, with SF-39 / SF-42 standard deviations at their 80 %
   upper bound and means at their 80 % lower bound; one blinded variance re-estimation allowed. Planning (from SF-39 plane data on
   `[25, 94]`): 90 % half-width of `alpha_TP` at most about 6e-4 / 2.8e-3 / 6.6e-3 `lambda` at `sigma^2` 1 / 2.25 / 4 (about 10 % of the
   planning means) and 2.5e-4 at 0.25, giving `N` of the order of 1000-1650 / 90-110 / 30 / 1100. The interval procedure per estimand is the
   one the SF-42 coverage simulation qualifies (G5.3); BCa is reported only.
7. **Gate 5 (G5.5).** Matched and paired: both codes on the same SF-18 field and grid per seed, window injection, uniform weights (consumer
   twin series), Par2 `cg_rtol` 1e-10, Par2 time step fixed a priori in the pre-registration, `alpha_LP` on the common planes in both. Grids
   Par2 can hold: 1040x240x240 at 0.25 and 1 (`lambda/h` 10), 1248x288x288 at 2.25 (12), 1664x192x192 at 4 (16, width 12), each with the
   amplitude's source. `agree`, `disagree` and `inconclusive` as decision G5.5. Paired log-ratio TOST within `+-log(1.10)`, sized at the
   Holm worst level `0.05 / 7` (planning, unpaired worst case: about 170 / 60 / 30 / 100 per arm; fewer with the pilot's pairing
   correlation). The SF-39-style unmatched comparison is reported for continuity.
8. **Gate 5 attribution (G7).** At `sigma^2` 0.25 and 1, Par2 cells of a `2^(4-1)` resolution-IV fraction of generator, injection, time step
   and solver tolerance, analysed as a split-plot; plus the free decomposition consumer-twin / Pollock-twin / Par2.
9. **Distance and B&dD (W6).** At `sigma^2` 1 and 2.25, boxes of about 208 `lambda` at the production `lambda/h` and width 24 (planning 60-100
   realizations): `alpha_LP`, `alpha_TP` over `[25, 94]` and `[94, 194]`; a secondary fit of B&dD eq. 6 to the time-window `alpha_L(t)` of
   these arms and of Par2; the remaining gap recorded (decision G9b).
10. **Particles.** `M` per configuration from SF-42's particle pilot; the split-half particle share of each primary estimator's variance at
    most a pre-registered fraction, else `M` is raised before the first run.
11. **Reproducibility (G8).** Per-realization reduced tables committed; host-side reduction and pruning with sha256 and a retained subset;
    the independent recomputation; the solver-verdict reconciliation; `exec` logging during the blinded phase; the human review's
    independent reduction of at least three realizations.

## Out of scope

- Asymptotic `alpha_T` and Lester et al.'s `t -> infinity` claim; finite Péclet; other covariance models; anisotropic `K`; `sigma^2 > 4`.
- The model restrictions of the decision record (periodic transverse boundaries, Dirichlet `x` boundaries, generator class) as factors.
- The open questions G9a and G9b (only recorded); memory engineering (G3).
- Changing SF-39's verdicts.

## Files and symbols

- Runners: `lagrangian_ensemble` (medium (b); window and lattice by the SF-42 seed-lattice rule), `macroflow3d_pipeline` (Par2 with SF-42's
  SF-18 field, window, tolerance and plane-crossing options).
- Tooling: SF-42's artifact scripts (reader and reduction, pruning, `mcstat/`, recomputation, reconciliation); a new campaign script modelled
  on `docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/scripts/run_sf39.sh` (parts, bookkeeping, restart, md5 gate, host
  reduction and pruning, no `git -C`).
- New artifact root `docs/experiments/artifacts/<date>-sf43-confirmatory-macrodispersion/` and note
  `docs/experiments/<date>-sf43-confirmatory-macrodispersion.md`.

## Implementation specification

1. **Pre-registration before any run** fixes: seeds (disjoint ranges per amplitude and arm family, reserved replacements), every arm (grid,
   width, `lambda/h`, `--ell`, `--eps`, window, `M`, planes, `--hmax-div`), the travel window, `N` per arm with the sizing computation and the
   joint allocation, the interval procedure per estimand with its coverage evidence, the budget rules of G5.4 (order range, safety factors,
   bootstrap scheme), the Gate 4 classes of G5.7 (with the `numerical` fraction), the Gate 5 rule of G5.5, the primary family and Holm, the
   factorial design and analysis, every reading rule, the retained-array subset and the planned cost.
2. **Blinded phase (G8).** From the first production run to the analysis freeze only costs, solver verdicts and the blinded variance
   re-estimation may be read; `exec` commands are logged. No outcome-dependent decision.
3. **Gate 4.** For each transverse estimator and amplitude, the robust lower limit and p-value of G5.4; the amplitude p-value
   `max(p_TP, p_TAP)`; the class of G5.7 in its order of precedence (`physical (restricted)` only when Holm rejects, at `sigma^2` 1, 2.25, 4;
   at 0.25 `numerical`, `unresolved (bounded)` or `unresolved`); always qualified "within this model, medium, width range and travel window,
   under the stated transfer assumptions".
4. **Gate 5.** The paired TOST of G5.5 at every amplitude, its p-value in the Holm family; the factorial decomposition at 0.25 and 1.
5. **Failure rules.** Failure criteria fixed in advance and applied blind to `alpha`; failure rate per arm reported; an unusable solve is
   replaced by the next reserved seed (disclosed); above 1 % failures in an arm, a worst-case imputation at the 5th / 95th percentiles of
   `alpha` is reported; a sensitivity without fallback-accepted solves; a halted arm stops and goes to the owner.

## Expected numerical effect

No result is predicted. The SF-39 pilot suggests `alpha_TP` of order 0.006 / 0.028 / 0.066 `lambda` on `[25, 94] lambda` at `sigma^2` 1 /
2.25 / 4 and near 0 at 0.25; these size the study and are not hypotheses.

## Validation commands

```bash
# local: tooling tests and dry runs of the campaign script
python3 -W error -m unittest discover -s <SF-42 scripts> -t <SF-42 scripts>
bash <A>/scripts/run_sf43.sh build/v100-release <part> --dry-run
# remote, detached, one job at a time (REMOTE_GPU_WAIT=7200; REMOTE_GPU_MIN_FREE_MIB=30000 for CUDA parts)
scripts/remote --increment SF-43 run sf43-<part> -- "bash <A>/scripts/run_sf43.sh build/v100-release <part>"
scripts/remote --increment SF-43 wait sf43-<part>
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- **A. Design fidelity.** Every arm, `N`, window, procedure and rule as pre-registered (pre-registration commit pushed before the first
  run); every deviation recorded with its reason; no threshold or rule changed after data.
- **B. Budgets.** Every primary and secondary estimand at every amplitude reported with its sampling interval, `E_h` (observed order and its
  interval, or the fallback), `E_W` (and the interaction tests), `E_int`, the Pollock contrast and the stated transfer assumptions.
- **C. Gate 4.** The robust-sign p-values, the Holm decisions and the G5.7 class per amplitude, with the achieved Gate 4 power reported.
- **D. Gate 5.** The paired ratio-equivalence decision per amplitude with achieved power; the factorial decomposition at 0.25 and 1. An
  unexplained `disagree` keeps SF-43 active until attributed or decided by the owner.
- **E. Precision.** Achieved half-widths against the targets (reported, not extended after the fact).
- **F. Independent checks.** The recomputation from raw moments reproduces every reduced value it covers to 1e-10 relative; the solver
  reconciliation matches the job logs; the human review records its independent reduction of at least three realizations.

## Regression surface

- No runner change is expected; any change is a disclosed corrective with byte-identity checks of existing paths.
- Claims: SF-39's estimates are superseded where SF-43 overlaps; the dashboard and the theory notes cite SF-43 for confirmatory values.

## Failure and rollback policy

- An unmet SF-42 precondition without an owner decision blocks activation.
- A projected cost above the owner budget stops before the extension and goes to the owner, `N` unchanged until decided.
- Gate 5 `disagree` without attribution, or a coverage requirement no procedure meets, keeps SF-43 active.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Implementation matches the scope and contains no unrelated changes.
- [x] Targeted validation passes and its evidence is recorded.
- [x] Required regression tests pass.
- [x] Scientific or engineering findings are appended to the bitácora.
- [ ] Required human review is recorded.
- [x] PR and commit identifiers are recorded.
- [ ] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
<!-- completion-checklist:end -->

## Advancement rule

Acceptance of SF-43 provides the project's confirmatory finite-distance macrodispersivities for this model and enables the specification of
the finite-Péclet phase and, if the owner schedules them, of the open questions G9a and G9b.

## Bitácora

Append entries; do not rewrite prior observations. Store large raw outputs as artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-29T21:00Z | `pending`; governance PR (docs only) | Specification created on the governance PR of decision `2026-10-29-sf39-remediation-sf42-sf43.md`, after the SF-39 post-review evaluation and three adversarial reviews of the draft. Planning numbers (geometry within the cap, `N`, cost) from SF-39 data; not commitments. | Owner decisions G1-G4, G10, G11 of 2026-10-29. | SF-39 and SF-42 `done`; pre-activation review; pre-registration. |
| 2026-11-02T15:04Z | `pending`; pre-activation PR (docs only) | Pre-activation review `docs/experiments/2026-11-02-sf43-pre-activation-review.md` (verified bibliography; eleven analyses of the committed SF-39/SF-42 data; five adversarial reviews) and decision record `docs/decisions/2026-11-02-sf43-pre-activation-decisions.md` (H1-H21, proposed). Findings: the specified design does not fit the owner's deadline (analysed results by 2026-11-09 00:00 local); displacement-basis estimands, a geometry-robust null, a Fieller-form Gate 5, gated `E_h`, a two-GPU first stage (owner exception of 2026-11-02, recorded as H14) and variant A (about 60 GPU-h planning) recommended. No GPU, no SF-43 data. Authorized process deviation (owner, 2026-11-02): the campaign tooling is prepared in isolated worktrees from `master` before activation, without GPU, host or data (H20). | Owner brief of 2026-11-02 (Phases 1-4, deadline, two-GPU exception). | Single owner consultation; record the outcome in the decision record; merge (increment base); activation. |
| 2026-11-02T15:12Z | `pending`; pre-activation PR #86 (docs only) | Single owner consultation answered: variant A complete (with the `sigma^2 = 4` distance extension and the `sigma^2 = 0.04` positive control); displacement basis primary, null at the unbounded first-order value `c = 7.992e-5 sigma^2 lambda`, `m = 8` with `sigma^2 = 0.25`; Gate 5 in Fieller form with coverage qualified before the pre-registration and the pipeline-level qualification route; latitude items accepted as written (H6-H8, H10-H13, H16, H21). Recorded in the decision record (status accepted). No GPU, no SF-43 data. | Owner consultation of 2026-11-02 (`docs/decisions/2026-11-02-sf43-pre-activation-decisions.md`, section "Owner decisions of the consultation"). | Human merge of PR #86 (increment base); activation. |
| 2026-11-02T18:20Z | `active`; base `0755a008ea2008e57f09198c39abdc7de8d4ea7e` | Activated in the orchestrator session after the owner merged the pre-activation PR #86 (squash merge `0755a008`, 2026-11-02T18:17:49Z; tree identical to the audited PR head `0b515fb7`). Preconditions verified on that tree: SF-39 and SF-42 `done`; pre-activation review and decision record `2026-11-02-sf43-pre-activation-decisions.md` (H1-H21, status accepted, owner consultation of 2026-11-02 recorded) merged, so `0755a008` is the increment base; `check-lester-increments.sh`: `OK (44 increments, ready=SF-34 SF-43, nonterminal=none)`, so activating SF-43 leaves one nonterminal increment. Runtime goal = the Goal above, verbatim. Owner constraints for this run (brief of 2026-11-02): analysed results available by the start of 2026-11-09 (local, UTC-3); one remote job at a time on mirror `scripts/remote --increment SF-43`; always `REMOTE_GPU_WAIT=7200`, CUDA jobs also `REMOTE_GPU_MIN_FREE_MIB=30000` unless the pre-registration justifies a lower value for a stage running only small grids; never `--gpu`, `CUDA_VISIBLE_DEVICES` or GPU work under `exec`; two GPUs only in SF-43's stage 1 (H14: at most about 30 h, ends when its queue empties); `$HOME/sf39_campaign` and `$HOME/sf42_campaign` not to be deleted or modified, the owner consulted before any disk action; budget about 150 GPU-h (G10), GPU-h of every CUDA job and CPU-only slot-hours recorded separately here; stop cleanly and consult if the projection exceeds it; no agent polls jobs in loops (one background watcher); human review required, no agent merges. Orchestration record (runtime, gitignored): `.claude/orchestration/SF-43/` (understanding, DAG, audits). | Owner brief and consultation of 2026-11-02; merge of PR #86. | Integration of the approved tooling; host build; smoke test with the coverage run; pre-registration on a draft PR; production job. |
| 2026-11-02T18:25Z | `active`; owner-authorized process deviation (H20) | The campaign tooling was prepared BEFORE activation, in isolated worktrees from `master` `ef87a1bc`, without GPU, host or data, and audited by the orchestrator (records in the runtime directory): remote two-GPU request `0910c61f`; engine `35f996ac`; runner executor `04afcbbb`; recompute/reconcile `c18d9b34`; analysis Gate 4 core `e302cb77`; Gate 5/G7/secondaries/failures + coverage harness `29f0903f`; corrective nodes of the executability review: engine/executor `ad213064`, plan generator `1bba2c0a`, analysis `9af32591` and coverage `c0cb3afc`, Pollock plane rows `0db2a4c1`, alignment of the whole chain `c4e1d86e`, unattended-run robustness `696c3f51`, pre-registration gaps (achieved Gate 4 power, Par2 completeness, tracker counts, basis contrasts, qualification at the realized sizes) `3a9b8275`, H19 theory-note corrections `0ddf6b47`, test fix `a73dd85c`. Every suite green on the assembled chain (SF-42 scripts 392, sf43ana 168, sf43camp 72 + executor 49, sf43plan 36, sf43cov 16, recompute43 19, reconcile43 21, end-to-end chain 15; remote helper tests ALL PASS). Nothing of it ran on a GPU or the host; no SF-43 seed exists. | Owner authorization of 2026-11-02 (H20). | One integrator brings exactly this chain onto the increment base; orchestrator audit. |
| 2026-11-02T21:16Z | `active`; pre-registration `cbbd69da` pushed to draft PR #87 | Pre-registration `docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/preregistration.md` sha256 `3c5e05b27d5edb71734f99b0ae7c507360c2434ce3c42e5e2ddeabc709ccc43e`, committed and pushed (no force-push) before the first production run; companions hashed in it (production plan with binary md5 `d8b673b7...` / `5f8158b5...` and the coverage files). Host work so far (mirror `--increment SF-43`, one job at a time): builds `sf43-build` (failed: host /tmp 2 GB full), `sf43-build2`, `sf43-build3` (clean rebuild, TMPDIR on /home, 305/305, 0 assembler messages); smoke `sf43-smoke1` (failed at spawn, fixed by C-PATH), `sf43-smoke1b` (two GPUs; cancelled under the owner's GPU-sharing policy of 2026-11-02, also the kill test), `sf43-smoke2` (resume on one GPU: 40/40 arms, determinism PASS across GPUs, analysis STRUCTURE_OK m = 8); `sf43-coverage` (H15, CPU-only: 76/76 cells; pipeline qualified at every profile; Gate 5 Hall from n = 200, J3B gating qualifies n = 250; sampling limit alone a diagnostic, R13); `sf43-smoke-post` (RECOMPUTE PASS, RECONCILE OK). GPU-h of CUDA jobs (GPUs held x wall from the grant): smoke1 0.32, smoke1b 0.24, smoke2 0.86, total 1.42; CPU-only slot-hours (not G10): builds 0.21, coverage 0.58, smoke-post 0.12, total 0.90. Corrective nodes after the first integration (audited, re-integrated): C-PATH, P-GPUPOL, C-RECOMP, C-QUAL. Departures recorded before data: D1-D4 (pre-registration section 14). | Pre-registration sections 1.5, 11, 14; runtime ledger `.claude/orchestration/SF-43/jobs.tsv`. | `blinded start`; production job `sf43-campaign`. |
| 2026-11-02T21:20Z | `active`; production job `sf43-campaign` running (blinded phase since 2026-11-02T21:18:01Z) | Launched 2026-11-02T21:18:05Z on mirror `--increment SF-43` at the tree of `e85ec5a1` (synced; binary md5 and `plan.json` sha256 verified on the host): `REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_MIN_FREE_MIB_2=17000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1 scripts/remote --increment SF-43 run sf43-campaign -- "bash docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/run_sf43.sh build/v100-release all"`. Helper snapshot: both GPUs free (no compute process, 32510 MiB free), so two GPUs were granted under the owner's GPU-sharing policy: lane A GPU 0, lane B GPU 1. **Stage 1 start 2026-11-02T21:18:58Z** (H14 cap: 2026-11-04T03:18:58Z; stage-1 GPU-h = 2 x its wall). Engine start gate (binary md5, analysis sha256) passed; first projection 57.83 GPU-h vs budget 150, cutoff 2026-11-08T19:00:00Z. During the blinded phase only costs, solver verdicts, blinded re-estimation lines and counts are read; every `exec`/`run` is logged on the host. | Job log `sf43-campaign.log` (CAMPAIGN, PROJECTION lines). | One background watcher; record LANE_B_RELEASED and every failure event; final audit after the job ends. |
| 2026-11-03T21:30Z | `active`; production `sf43-campaign` running (blinded) | Progress read from the job log only (costs, verdicts, re-estimation, counts): 377 units ended; **stage 1 ended 2026-11-03T14:04:00Z** (`LANE_B_RELEASED gpu=1`, normal release when no unit of its class remained; first host observation of that path), stage-1 wall 16.75 h -> 33.5 GPU-h held, within the 30-h cap; since then one GPU (GPU 0). Two unusable-solve replacements at `sigma^2 = 4`, group-wide (`s4_prod` seed 434153 -> 434401; `s4_l16` 434222 -> 434402); no crash, halt, OOM, spawn failure or disk event. Blinded re-estimation (H9) on 9 of the 10 listed arms: N unchanged everywhere (N_req 1-24). Engine projection at 21:23Z: 40.8 GPU-h used, 51.6 projected (budget 150), completion about 2026-11-04T08:12Z (cutoff 2026-11-08T19:00Z). Host /home 484 GB free. A colleague's process now runs on GPU 1, which the job no longer holds. | Job log `sf43-campaign.log`. | Watcher re-armed; Phase 4 after the job ends. |
| 2026-11-04T13:20Z | `active`; production `sf43-campaign` ended (exit 0, `DONE`); **analysis freeze declared before `summary.json` is read** | Job ended 2026-11-04T08:50:49Z, exit 0, `FINAL status=DONE analysis=ok gpu_h=51.14`; unit loop ended 07:38:31Z; in-job analysis by the frozen command (`ANALYSIS_STEP analysis rc=0`, `recompute rc=0`, `reconcile rc=0`). Read during the blinded phase only (G8): `REPORT.txt` counts: 44 arms, each complete at its planned size (5294 solves: consumer 3483, Par2 1811), 0 halted, 0 OOM, 0 lane crashes, 0 unreduced segments; third unusable-solve replacement `g_d_s4` 436437 -> 436501 (`s4_dist`, 2026-11-04T00:57Z), so three in all, all at `sigma^2 = 4` (`s4_prod` 1 of 201, `s4_l16` 1 of 251, `s4_dist` 1 of 101 attempted); blinded re-estimation on all 10 listed arms, `N` unchanged (N_req 1-24); `check-finite`: STRUCTURE_OK, m = 8, non-finite means 0, warnings_count 0, errors_count 0 (64 non-finite leaves in section `budgets`, structure only; matplotlib mathtext UserWarnings at figure time, outside `process.warnings`); `recompute43`: RECOMPUTE PASS (22 289 174 values compared, 0 beyond 1e-10 relative, 0 problems, max relative 8.97e-11); `reconcile43`: RECONCILE OK (5294 solves = 5294 ledger seed records; 3 `not_converged_beyond_fallback` = 3 replacements; 0 conflicts). GPU-h of the job (GPUs held x wall from the grant 21:18:57Z): lane A 35.53 h (to the job end, incl. about 1.2 h of CPU-only analysis), lane B 16.75 h (to `LANE_B_RELEASED`), 52.28 GPU-h held (engine count 51.14, to the end of the unit loop); SF-43 total 53.70 GPU-h of the 150 guard (smoke 1.42). **Freeze (G8, workflow 8A), host hashes at 2026-11-04T13:15:14Z:** analysis = the frozen files of the pre-registration commit `cbbd69da` (mirror at the tree of `e85ec5a1`, analysis files identical; engine sha256 gate passed at start and before the analysis); `analysis/summary.json` sha256 `c5c543d5c0b4614a4f9d081e6bfaa368b2e0a03d11dc724df8ad128b200a718c`; reduced tables 2409 files, 1 269 865 886 bytes, sorted manifest `~/sf43_campaign.freeze.reduced.sha256` sha256 `250395ed5319052ddf1e5338e0d081d60610a41e85e1f66252b828daeb492dd6`; `ledger.jsonl` (7985 records) `940326fd4d37da7bdd4cb2ecc05245685a591527b168dba128ac96426eaaa04c`; `REPORT.txt` `cda77410400492e4...`; recompute report `073a86a667533957...`; reconcile report `62249e0d52b29090...`; job log `83690c4f34c53031...`; exec log `exec-blinded.log` (10 lines) `c286f75129bf415bd71b58a6c0cf4b6d373e5f3a54cf628d80ddc3fa69a076f4`. No estimate, contrast, p-value, decision or class has been read. | Job log `sf43-campaign.log`, `REPORT.txt`, recompute and reconcile reports, host hashes. | `blinded stop`; read `summary.json`; Phase 4 audit (completeness, failures, GPU-h, recompute and reconciliation, analysis as pre-registered); experiment note; PR #87 `awaiting_review`. |
| 2026-11-04T13:49Z | `awaiting_review`; source-bearing audited head `fd3741bd` on PR #87 | **Correction of the previous row:** its UTC label 13:20Z is wrong; the freeze commit `40135561` was made and pushed at 13:15:43Z, before `blinded stop` (13:15:57Z, host log), so the order of G8 / workflow 8A holds. **Phase 4 audit (orchestrator):** results copied from the host after the freeze and verified against it (`summary.json` sha256 equal; 2409 reduced files extracted from the per-arm archives equal to the freeze manifest; job log equal; first 10 lines of `exec-blinded.log` equal to the frozen hash, line 11 is `blinded-stop`); analysis ran as pre-registered (engine sha256 gate at start and before the analysis; `process.prereg_sha256` = `prereg.json`; coverage and qualification documents as hashed; 0 errors, 0 warnings; 68 analysis entries `ok`, each a complete prefix at its planned size; replacements applied group-wide); `recompute43` PASS covers every production realization (no `PROBLEM`; 2738 non-production rows outside the retained subset not covered, as pre-registered); `reconcile43` OK. **Confirmatory (Holm, m = 8):** 7 of 8 rejected. `robust:<a>` rejected at 0.25, 1, 2.25, 4 (`p` = 5.96e-5 each; `RL` at `0.05/8` = 0.75-0.88 of the mean): class `physical (restricted)` at all four amplitudes, both estimators; `alpha_TP` (displacement) = 5.26e-4, 7.22e-3, 2.87e-2, 6.84e-2 `lambda`, budgets 13 / 7 / 10 / 16 % of the estimate. `gate5:<a>` equivalence established at 0.25, 1, 2.25 (ratios 1.004, 1.007, 1.017) and NOT at 4 (ratio 1.188, Fieller 90 % [1.159, 1.228], descriptive verdict `disagree`, pre-registered as plausible). **Acceptance D open:** pre-registered decomposition at 4: consumer/Pollock 1.021 [0.987, 1.056], Pollock/Par2 1.164 [1.131, 1.198], so the offset is on the Par2 side; the Par2 mechanism (Euler step, horizon truncation of the slow tail, injection) is not separable with these data; owner decision required. Descriptive: three trackers agree on the transverse growth within 0.5 %; far/main transverse ratio 0.99-1.04 (no decay to 194 `lambda`); positive control `R` = 1.145 [1.122, 1.168], ok; achieved Gate 4 power 1.00; G7: the generator factor gives `alpha_TP` 1.48 [1.11, 2.10] times larger on SF-18 than on random-spectral fields at `sigma^2 = 1` (model-scope caveat); backflow 15.5 % of particles at 4. **Deviations found after data** (none changes a decision): D5 B&dD eq. 6 fit and Par2 time-window estimators not computed (inputs absent from the reduced tables; not recoverable after pruning); D6 `E_h`/`E_W`/`E_int` computed for the transverse estimands only (text says per estimand; the companion ran); D7 recompute report counters undercount realizations (reporting only). Experiment note `docs/experiments/2026-11-03-sf43-confirmatory-macrodispersion.md`; theory notes and dashboard cite SF-43. Artifact `A/results/` (181 MB: H21 per-arm archives). Ledger: GPU-h of CUDA jobs smoke 1.42 + production 52.28 = 53.70 of 150; CPU-only slot-hours 0.90; host staging `~/sf43_artifact_stage` (177 MB, archives only). | Audit record `.claude/orchestration/SF-43/audits/PHASE4-RESULTS.md`; `A/results/`; experiment note. | Human review on PR #87: acceptance D at `sigma^2 = 4`; independent reduction of at least three retained realizations (acceptance F); deviations D5-D7; generator caveat. No `done`, no merge by an agent. |
