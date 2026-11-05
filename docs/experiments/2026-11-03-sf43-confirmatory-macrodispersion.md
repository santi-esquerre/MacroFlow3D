# SF-43: confirmatory Monte Carlo study of macrodispersion at `sigma_Y^2` in {0.25, 1, 2.25, 4} with quantified error budgets

- What this is: the run record and the pre-registered result of increment SF-43
  (`docs/plans/active/lester-eq14/increments/SF-43-confirmatory-macrodispersion-study.md`): plane-based `alpha_TP`, `alpha_TAP` and
  `alpha_LP` over the common travel window `[25, 94] lambda`, each with a sampling interval and the error budget of decision G5.4, the
  robust sign of the transverse growth (Gate 4) and the matched, paired comparison of `alpha_LP` with Par2 RWPT (Gate 5), both inside one
  Holm family of eight hypotheses.
- What it is not: an asymptotic statement, a test of the Lester et al. (2023) zero-transverse claim at `t -> infinity`, or a statement
  about other covariance models, anisotropic `K`, finite Péclet, other media or `sigma^2 > 4`.
- Status: **results frozen and audited by the orchestrator; awaiting human review** (State `awaiting_review`). The confirmatory outcomes
  below are the frozen analysis's output; nothing in this note changes a pre-registered rule. On 2026-11-04 the owner reopened SF-43
  (`active`) for addendum 1 (exploratory follow-up arms J1, J3, J9 and the conditional J8, plus post hoc analyses, outside the family;
  `docs/decisions/2026-11-04-sf43-addendum-1.md`); this note's text was corrected accordingly (end of section 4). The addendum-1
  results are now **frozen and audited** (section 3.5); the increment returns to `awaiting_review`, and acceptance D at `sigma^2 = 4`
  is the owner's decision (the J1 class is an input to it). Where SF-43 overlaps SF-39
  (`docs/experiments/2026-10-27-sf39-macrodispersion-study.md`, the exploratory pilot), SF-43 supersedes it.
- Normative contract: `A/preregistration.md` (sha256 `3c5e05b27d5edb71734f99b0ae7c507360c2434ce3c42e5e2ddeabc709ccc43e`, commit
  `cbbd69da`, pushed to draft PR #87 before the first production run) with its companions `A/plan.json` (`a9e599a1...`) and
  `A/prereg.json` (`9b92a524...`); governance `docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md` (G1-G11) and
  `docs/decisions/2026-11-02-sf43-pre-activation-decisions.md` (H1-H21, owner consultation of 2026-11-02); GPU use
  `docs/decisions/2026-11-02-sf43-gpu-sharing-policy.md`. Theory: `docs/theory/lester-2023-key-claims.md` (helicity-free regime, the
  zero-transverse claim and its limits, first-order context), `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` (classical 3D
  baseline, context only).
- Artifact root `A = docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/`; results under `A/results/` (layout at
  the end of this note).

Reading key. **Confirmatory** = the eight family hypotheses and their Holm decisions (section 3.1). **Descriptive (pre-registered)** =
every other block the frozen analysis computes (budgets, classes, verdicts, contrasts, controls, secondaries); reported, never deciding.
**Exploratory (post hoc)** = observations made by the orchestrator after the freeze, from the frozen outputs only, flagged as such
(section 3.3); they suggest, they do not test. **Post hoc (2026-11-04)** = the analyses N1, N3, N5, N6, N8 and S of addendum 1
(`docs/decisions/2026-11-04-sf43-addendum-1.md`, item 4), recomputed from the frozen data by scripts that are themselves frozen
(section 3.4); outside the Holm family, none changes a pre-registered result. The text corrections of 2026-11-04 are listed at the end
of section 4.

## 1. Question

Within the model of the decision record (smooth Gaussian log-conductivity `C(r) = sigma^2 exp(-(r/lambda)^2)`, scalar isotropic `K`,
medium (b) with Dirichlet `x` faces and periodic transverse faces, SF-18 periodic normalized generator, purely advective transport):

1. Is the transverse growth of the first-passage distribution over `[25, 94] lambda` larger than the first-order window value
   `c = 7.992e-5 sigma^2 lambda` once sampling, discretization (`E_h`), width (`E_W`, `E_WxH`) and integration (`E_int`) errors are
   bounded, at each `sigma^2` in {0.25, 1, 2.25, 4} (robust sign, G5.4; class G5.7)?
2. Is the consumer's `alpha_LP` equivalent within a factor 1.10 to Par2's on the same field, grid and window (Gate 5, G5.5)?
3. With what error budget are `alpha_TP`, `alpha_TAP`, `alpha_LP` measured (acceptance B, E)?

## 2. Hypotheses, design and execution (summary; the pre-registration is normative)

**Family** (m = 8, Holm at family-wise 0.05) [prereg 5.1]: `robust:<a>` -- null `alpha_e <= c` for both `e` in {`alpha_TP`,
`alpha_TAP`} (displacement basis; intersection-union), rejected iff `RL_e(a) = L_e(a/2) - sum_k B_k(a/2K) > c` with `L_e` the one-sided
bootstrap-t lower limit of the production mean (`B` = 33599); `gate5:<a>` -- TOST in Fieller form of `mu_L / mu_P` within
`[1/1.10, 1.10]`, Hall's skewness-corrected `t` (the procedure the coverage run qualified at the planned pair counts). Pre-registered
expectation [prereg 6]: at `sigma^2 = 4` the plane-based `alpha_LP` is a slow-tail, non-Fickian quantity and a Gate 5 `disagree` is
plausible. No `alpha_T` value was presupposed (sizing used the SF-42 pilot means as design inputs only).

**Design** (variant A, owner-accepted) [prereg 2]: 44 engine arms (68 analysis entries), production `N = 200` per amplitude
(1040x240x240 at 0.25 and 1, `lambda/h` 10, width 24 `lambda`; 1664x384x384 at 2.25, `lambda/h` 16; 3328x384x384 at 4, `lambda/h` 32,
width 12), `E_h` ladders paired by seed (5/10/20; 4/8/12/16; 4/8/16/32), unpaired width cells (48 vs 24 `lambda`; 24 vs 12 at 4, with a
2x2 interaction at 2.25 and 4), `h_max` passes `h/8, h/16, h/32` on the first 30 production seeds, Gate 5 Par2 arms (200 / 200 / 200 /
250 pairs), distance arms (L = 208 `lambda`, 100 realizations at 1, 2.25, 4), the G7 Par2 factorial at 0.25 and 1 (8 cells x 60), and the
positive control `sigma^2 = 0.04` (30 realizations). Source: centred square of fixed size per amplitude (19.2 / 19.2 / 19.2 / 9.6
`lambda`), flux weights, seed lattice `M` = 128 / 128 / 100 / 100 (`N_p = M^2`). Planes `x0 + d`, `d` = 25..94 (70; 25..194 on the
distance arms). Seeds 431001-436510 / 437001-437363 in disjoint blocks with reserves [prereg 3].

**Builds** [prereg 1.4]: host clean rebuild (`cmake --preset v100-release`, target clean, `-j 32`, `TMPDIR` on /home; 305/305 steps, 0
assembler messages); `lagrangian_ensemble` md5 `d8b673b769aabfb7662a2cd0cb34b3fe`, `macroflow3d_pipeline` md5
`5f8158b55cd54598d3267bc6f7219552`, verified by the engine at start (`A/binaries.json`). Mirror `scripts/remote --increment SF-43` at the
tree of `e85ec5a1` (analysis files identical to the pre-registration commit `cbbd69da`; sha256 of every analysis file verified by the
engine at start and again before the analysis).

**Commands** (exact; owner rules: one remote job at a time, `REMOTE_GPU_WAIT=7200`, CUDA jobs `REMOTE_GPU_MIN_FREE_MIB=30000`, never
`--gpu` or `CUDA_VISIBLE_DEVICES`):

```bash
scripts/remote --increment SF-43 blinded start --label "SF-43 production campaign (pre-registration cbbd69da)"
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_MIN_FREE_MIB_2=17000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1 \
  scripts/remote --increment SF-43 run sf43-campaign -- \
  "bash docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/run_sf43.sh build/v100-release all"
# in-job, after the unit loop (plan.json `analysis`, `recompute`, `reconcile`):
#   bash A/scripts/analyse_sf43.sh ~/sf43_campaign build/v100-release      (python3 -W default -m sf43ana run ...; check-finite)
#   python3 -W error -m recompute43 run --campaign ~/sf43_campaign --reduced ~/sf43_campaign/reduced --out ~/sf43_campaign.recompute.txt \
#       --primary-arm s025_prod --primary-arm s1_prod --primary-arm s225_prod --primary-arm s4_prod --single-pass-hmax 8
#   python3 -W error -m reconcile43 run --campaign ~/sf43_campaign --out ~/sf43_campaign.reconcile.txt
# freeze (bitácora row, commit 40135561, 2026-11-04T13:15:43Z), then:
scripts/remote --increment SF-43 blinded stop        # 2026-11-04T13:15:57Z
```

**Execution record** (job log `A/results/logs/sf43-campaign.log`, `A/results/REPORT.txt`):

- `sf43-campaign` launched 2026-11-02T21:18:05Z, GPUs granted 21:18:57Z (both free under the owner's GPU-sharing policy: lane A GPU 0,
  lane B GPU 1); stage 1 (two GPUs, H14) from 21:18:58Z to `LANE_B_RELEASED gpu=1` at 2026-11-03T14:04:00Z (normal release: no unit of
  its class left; 16.75 h, inside the 30-h cap); then one GPU. Unit loop ended 2026-11-04T07:38:31Z; analysis, recompute and reconcile
  `rc=0`; `FINAL status=DONE analysis=ok exit=0` at 08:50:49Z. The analysis was **frozen** (declared before `summary.json` was read) at
  2026-11-04T13:15Z, commit `40135561` (13:15:43Z, below).
- **GPU-h** (GPUs held x wall from the grant): lane A 35.53 h, lane B 16.75 h, **52.28 GPU-h of production** (the engine's own count,
  `FINAL ... gpu_h=51.14`, stops at the end of the unit loop; the job held GPU 0 during the ~1.2 h CPU-only analysis). SF-43 total
  with the smoke jobs (1.42): **53.70 GPU-h** (= 52.28 + 1.42) of the 150 GPU-h guard (G10); planning was 59.70. CPU-only slot-hours (not G10): builds 0.21, coverage 0.58, smoke post-processing 0.12.
- **Completeness**: every arm complete at its planned size; 5294 solves (consumer 3483, Par2 1811); 0 halted arms, 0 OOM, 0 lane
  crashes, 0 unreduced segments; 68 analysis entries `ok`, each a complete prefix equal to its planned `n`.
- **Failures** (all at `sigma^2 = 4`, `not_converged_beyond_fallback`, replaced group-wide by the next reserve [prereg 3]): `s4_prod`
  seed 434153 -> 434401, `s4_l16` 434222 -> 434402 (group `g_s4`: applied to every arm that planned the seed, including the Gate 5
  twins), `s4_dist` 436437 -> 436501. Rate 3 / 5294 = 5.7e-4 (largest per arm 1 / 101 = 0.99 %, below the 1 % imputation trigger); no
  `fallback`-accepted solve anywhere, so every fallback sensitivity equals the main analysis.
- **Blinded re-estimation** (H9): all 10 listed arms, `N` unchanged (`N_req` 1-24 against `n_conf` 150-200).
- **Blinded phase** (G8): from 2026-11-02T21:18:01Z; only costs, verdicts, re-estimation lines and counts were read; the host log
  `exec-blinded.log` records the `run` and 8 `exec` commands of the phase (one GPU-process query on 2026-11-03, seven reads of counts,
  reports and hashes on 2026-11-04); the freeze row (UTC, analysis commit, sha256 of `summary.json`, of the 2409 reduced tables via a sorted manifest, of the
  ledger and of the exec log) was committed and pushed at 13:15:43Z, before `blinded stop` (13:15:57Z) and before `summary.json` was
  read.

## 3. Results

Units: `lambda`. Intervals 90 % unless stated. Estimands per realization over the window, ensemble means over the analysis set
[prereg 4]. Tables T1-T11 are generated from the frozen `A/results/analysis/summary.json` (no recomputation).

### 3.1 Confirmatory (the Holm family)

#### T1. Confirmatory family (Holm, m = 8, FWER 0.05)

| hypothesis | p | Holm rank | Holm level | rejected | Holm-adjusted p |
|---|---|---|---|---|---|
| `gate5:s025` | 2.89e-128 | 1 | 6.25e-03 | yes | 2.31e-127 |
| `gate5:s1` | 1.84e-112 | 2 | 7.14e-03 | yes | 1.29e-111 |
| `gate5:s225` | 3.30e-61 | 3 | 8.33e-03 | yes | 1.98e-60 |
| `robust:s025` | 5.96e-05 | 4 | 0.01 | yes | 2.98e-04 |
| `robust:s1` | 5.96e-05 | 5 | 0.0125 | yes | 2.98e-04 |
| `robust:s225` | 5.96e-05 | 6 | 0.01667 | yes | 2.98e-04 |
| `robust:s4` | 5.96e-05 | 7 | 0.025 | yes | 2.98e-04 |
| `gate5:s4` | 1 | 8 | 0.05 | **no** | 1 |

**Robust sign (Gate 4).** All four `robust:<a>` hypotheses are rejected: at every amplitude both transverse estimators exceed `c` after
the budget is subtracted, with `p` at the lattice minimum (5.96e-5) and `RL` at the strictest Holm level `0.05/8` between 0.75 and
0.88 of the production mean (T2). The pre-registered class is therefore **`physical (restricted)` at `sigma^2` = 0.25, 1, 2.25 and 4**,
for `alpha_TP` and `alpha_TAP` alike: *within this model, medium, width range (24-48 `lambda`; 12-24 at `sigma^2 = 4`) and travel
window `[25, 94] lambda`, under the stated transfer assumptions* [prereg 5.2, 6]. Qualification held at the realized sizes (pipeline
profile `half` of `K = 3` at 0.25 and 1, `full` of `K = 4` at 2.25 and 4; all floors met; see 4.2 for why `half` was the largest
eligible profile at 0.25 and 1). By H16, only an `E_h` or `E_int` artifact would support the attribution of `alpha_T` to numerics; both
are bounded far below the estimates (T3).

#### T2. Robust sign (Gate 4), displacement basis, window [25, 94] `lambda`; units of `lambda`

| `sigma^2` | N | estimator | mean | 90 % bootstrap-t | `c` | mean / `c` | `L(a/2)` at a = 0.05 / 0.00625 | `RL` at a = 0.05 / 0.00625 | p | class |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 200 | `alpha_TP` | 5.263e-04 | [5.111e-04, 5.410e-04] | 2.00e-05 | 26 | 5.08e-04 / 5.01e-04 | 4.38e-04 / 4.19e-04 | 5.96e-05 | physical (restricted) |
| 0.25 | 200 | `alpha_TAP` | 5.263e-04 | [5.112e-04, 5.410e-04] | 2.00e-05 | 26 | 5.08e-04 / 5.00e-04 | 4.38e-04 / 4.18e-04 | 5.96e-05 | physical (restricted) |
| 1 | 200 | `alpha_TP` | 7.219e-03 | [7.061e-03, 7.377e-03] | 7.99e-05 | 90 | 7.03e-03 / 6.96e-03 | 6.53e-03 / 6.33e-03 | 5.96e-05 | physical (restricted) |
| 1 | 200 | `alpha_TAP` | 7.219e-03 | [7.062e-03, 7.376e-03] | 7.99e-05 | 90 | 7.03e-03 / 6.96e-03 | 6.54e-03 / 6.32e-03 | 5.96e-05 | physical (restricted) |
| 2.25 | 200 | `alpha_TP` | 0.028691 | [0.028213, 0.029173] | 1.80e-04 | 160 | 0.02812 / 0.02789 | 0.02539 / 0.02463 | 5.96e-05 | physical (restricted) |
| 2.25 | 200 | `alpha_TAP` | 0.02869 | [0.028207, 0.029174] | 1.80e-04 | 160 | 0.02812 / 0.0279 | 0.02539 / 0.02464 | 5.96e-05 | physical (restricted) |
| 4 | 200 | `alpha_TP` | 0.068402 | [0.066463, 0.070422] | 3.20e-04 | 214 | 0.06612 / 0.06521 | 0.05492 / 0.05162 | 5.96e-05 | physical (restricted) |
| 4 | 200 | `alpha_TAP` | 0.068381 | [0.066471, 0.070412] | 3.20e-04 | 214 | 0.06609 / 0.06522 | 0.05488 / 0.05162 | 5.96e-05 | physical (restricted) |

**Gate 5.** Equivalence of `alpha_LP` within `[1/1.10, 1.10]` is **established at 0.25, 1 and 2.25** and **not established at 4**
(`p_tost` = 1; the 90 % Fieller interval of the ratio, [1.159, 1.228], lies entirely above 1.10). The descriptive G5.5 verdict is
`agree` / `agree` / `agree` / **`disagree`** (T5). Per the specification (acceptance D) and prereg 6, a `disagree` keeps SF-43 open until
it is attributed or decided by the owner; the pre-registered attribution is in 3.2 (T6).

#### T5. Gate 5: `alpha_LP` consumer twin (uniform weights) / Par2, paired by seed

| `sigma^2` | pairs | consumer | Par2 | ratio | 90 % Fieller (Hall) | `p_tost` | Holm level | equivalence (confirmatory) | verdict (descriptive) | pair corr. | Par2 completeness < 0.999 | power at ratio 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 200 | 0.23297 | 0.23212 | 1.0037 | [1.0014, 1.006] | 2.89e-128 | 6.25e-03 | established | agree | 0.9855 | 0 of 200 | 1 |
| 1 | 200 | 1.0894 | 1.0814 | 1.0073 | [1.0046, 1.0102] | 1.84e-112 | 7.14e-03 | established | agree | 0.9762 | 0 of 200 | 1 |
| 2.25 | 200 | 3.2662 | 3.2104 | 1.0174 | [1.0121, 1.0228] | 3.30e-61 | 8.33e-03 | established | agree | 0.8833 | 0 of 200 | 1 |
| 4 | 250 | 9.4 | 7.9114 | 1.1882 | [1.159, 1.2285] | 1 | 0.05 | **not established** | disagree | 0.3088 | 40 of 250 | 0.9774 |

### 3.2 Descriptive (pre-registered, outside the family)

**Error budgets** (T3). The width term dominates at 0.25 and 1 (`E_W` bound 13 % and 6 % of the mean, with measured width differences
`dW` of 5 % and 0.4 %); at 2.25 and 4 the `E_h` ladders are non-monotone (level differences change sign), so no real order exists and
the pre-registered fallback `3 UCB(max |a_i - a_j|)/(r_min - 1)` applies (gate failed by construction: `E_h = max(order, fallback)`).
`E_int` is negligible at every amplitude (passes `h/8` and `h/16` agree to about 1e-10 relative): the integration error at `h/8` is not
material, consistent with SF-37's finding that tolerances 1e-6 to 1e-10 give bitwise-identical results at `h/8`. Budgets sum to 7-16 %
of the estimates; even the full sum at the strictest level leaves `RL` far above `c` (T2).

#### T3. Error-budget components `B_k` of `alpha_TP` (displacement basis) at a = 0.05 / 0.00625; units of `lambda`

| `sigma^2` | `E_h` | `E_h` branch (order `p_use`) | `E_W` (est. `dW`) | `E_WxH` | `E_int` | sum of `B_k` at 0.05, % of mean |
|---|---|---|---|---|---|---|
| 0.25 | 2.42e-06 / 2.51e-06 | order (1.598) | 6.79e-05 / 7.98e-05 (2.71e-05) | - | 1.25e-13 / 1.61e-13 | 13.4 % |
| 1 | 5.85e-05 / 7.25e-05 | order (0.8339) | 4.40e-04 / 5.59e-04 (3.14e-05) | - | 6.18e-12 / 7.56e-12 | 6.9 % |
| 2.25 | 8.94e-04 / 1.05e-03 | fallback (order undefined) | 1.73e-03 / 2.07e-03 (4.67e-04) | 1.12e-04 / 1.38e-04 | 5.12e-10 / 6.37e-10 | 9.5 % |
| 4 | 5.12e-03 / 6.05e-03 | fallback (order undefined) | 5.28e-03 / 6.56e-03 (5.65e-04) | 8.04e-04 / 9.82e-04 | 1.95e-07 / 2.38e-07 | 16.4 % |

**Longitudinal `alpha_LP`, halves and distance** (T4). `alpha_LP` = 0.233 / 1.092 / 3.261 / 9.44 `lambda`. The halves are reported as
intervals, without a claim (no equivalence margin was pre-registered). The far window `[94, 194]` of the distance arms gives
`alpha_TP(far) / alpha_TP(main)` = 1.02 / 1.04 / 0.99 at 1 / 2.25 / 4: no decay of the transverse growth rate is visible up to 194
`lambda` (descriptive; a plateau here separates "decay toward zero" from "plateau", not physics from numerics [prereg 12]). The
distance arms have one width per amplitude (24 `lambda`; 12 at `sigma^2 = 4`), so this far/main plateau rests on that width only; the
width dependence at distance is the subject of the addendum-1 arm J9 (`A/preregistration-addendum-1.md`).
`alpha_LP(far)/alpha_LP(main)` = 1.06 [1.02, 1.10] at 4: still growing, the expected non-Fickian behaviour.

#### T4. `alpha_LP` (travel-time, plane-based) and the transverse windows; units of `lambda`

| `sigma^2` | `alpha_LP` mean | 90 % bootstrap-t | `U_r` | h1 [25, 59.5] `alpha_LP` | h2 [59.5, 94] `alpha_LP` | h1 `alpha_TP` | h2 `alpha_TP` | far/main `alpha_TP` (L = 208) | far/main `alpha_LP` |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.23341 | [0.23046, 0.23646] | 1.0441 | 0.2306 [0.2266, 0.2346] | 0.2319 [0.2264, 0.2374] | 5.09e-04 [4.81e-04, 5.37e-04] | 5.42e-04 [5.10e-04, 5.73e-04] | - | - |
| 1 | 1.0918 | [1.0802, 1.1037] | 1.1835 | 1.083 [1.069, 1.098] | 1.108 [1.089, 1.127] | 7.18e-03 [6.96e-03, 7.41e-03] | 7.07e-03 [6.81e-03, 7.32e-03] | 1.021 [0.9756, 1.068] | 1.002 [0.9761, 1.03] |
| 2.25 | 3.2611 | [3.2279, 3.2967] | 1.4463 | 3.266 [3.229, 3.303] | 3.259 [3.211, 3.307] | 0.02893 [0.02823, 0.02962] | 0.02839 [0.02752, 0.02926] | 1.036 [1.001, 1.072] | 0.9908 [0.973, 1.009] |
| 4 | 9.4392 | [9.0163, 10.841] | 1.8761 | 9.651 [8.38, 10.92] | 9.414 [8.999, 9.829] | 0.06983 [0.0672, 0.07246] | 0.06717 [0.06434, 0.07] | 0.9877 [0.9354, 1.044] | 1.062 [1.023, 1.102] |

**Attribution of the `sigma^2 = 4` Gate 5 offset** (T6; decomposition fixed in prereg 7, same pairs, uniform weights). The total
consumer/Par2 ratio 1.188 splits into consumer/Pollock **1.021 [0.987, 1.056]** (velocity representation: spline `v_s` against Pollock's
face-flux interpolation on the same realizations, lattice and integrator-free cell integration; interval contains 1) and Pollock/Par2
**1.164 [1.131, 1.198]** (Euler stepping with fixed `dt`, random against lattice injection, Par2's own crossing bookkeeping and horizon).
The offset therefore sits on the Par2 side of the bridge, not in the consumer's representation. The pre-registered decomposition does
not separate the Par2-side mechanisms at `sigma^2 = 4` (the G7 factorial runs only at 0.25 and 1). Par2 crossing completeness:
40 of 250 Par2 realizations have `n_last / n_traced < 0.999` (minimum 0.9914); removing those pairs leaves ratio 1.154 [1.126, 1.198]
(`disagree`). At 0.25 to 2.25 the same decomposition shows consumer/Pollock offsets of 0.4-1.0 % and Pollock/Par2 within 0.7 %.

#### T6. Attribution (pre-registered decomposition on the Gate 5 pairs, uniform weights): ratio of means, 90 % paired interval

| `sigma^2` | response | consumer / Pollock (velocity representation) | Pollock / Par2 (Euler stepping, injection) | consumer / Par2 (total) |
|---|---|---|---|---|
| 0.25 | `alpha_LP` | 1.0042 [1.004, 1.0043] | 0.99952 [0.99721, 1.0018] | 1.0037 [1.0014, 1.006] |
| 0.25 | `alpha_TP:disp` | 1.0019 [1.0017, 1.0022] | 0.9976 [0.99513, 1.0001] | 0.99953 [0.99703, 1.002] |
| 1 | `alpha_LP` | 1.0078 [1.0072, 1.0085] | 0.99952 [0.99683, 1.0022] | 1.0073 [1.0046, 1.0101] |
| 1 | `alpha_TP:disp` | 1.004 [1.0036, 1.0044] | 0.99852 [0.99667, 1.0004] | 1.0025 [1.0006, 1.0044] |
| 2.25 | `alpha_LP` | 1.0104 [1.0062, 1.0147] | 1.0069 [1.0013, 1.0124] | 1.0174 [1.0121, 1.0227] |
| 2.25 | `alpha_TP:disp` | 1.0018 [1.0009, 1.0028] | 1.0001 [0.99811, 1.0022] | 1.002 [0.99993, 1.004] |
| 4 | `alpha_LP` | 1.0207 [0.98695, 1.0556] | 1.1641 [1.1313, 1.1979] | 1.1882 [1.1551, 1.2222] |
| 4 | `alpha_TP:disp` | 1.0015 [0.99989, 1.003] | 0.99579 [0.9917, 0.9999] | 0.99724 [0.99311, 1.0014] |

**Pollock contrast on the production set** (T7). The Pollock contrast is a pre-registered reported quantity outside the budget and the
family (prereg 7: paired difference and ratio of means, "outside the budget and the family"); no equivalence margin was pre-registered
for it, so no verdict `indistinguishable` or `agree` is attached to it. The consumer/Pollock ratio of the transverse estimands
(displacement basis) is 1.002 [1.0017, 1.0022] at 0.25 and 1.0039 [1.0035, 1.0043] at 1 (T7; on the Gate 5 pairs, T6: 1.0019
[1.0017, 1.0022] and 1.004 [1.0036, 1.0044]): +0.2 % and +0.4 %, small but resolved, with intervals that exclude 1. At 2.25 and 4 the
intervals contain 1 (1.0007 [0.9995, 1.0019]; 0.99898 [0.9969, 1.001]). `alpha_LP` differs by 0.4 % / 0.8 % / 1.5 % / 5.4 %
[0.0 %, 12.3 %]. Together with T6 and T9 (Par2 / consumer transverse 0.998-1.003; its interval excludes 1 at 1,
0.99752 [0.99563, 0.99941]), **three trackers that share the field and the Darcy solution -- the spline consumer (DP5(4)), the
Pollock face-flux tracker and Par2 (Euler, separate code) -- give transverse growth that differs by at most 0.5 % in the ratio of
means at every amplitude**. They share the SF-18 field and the finite-volume Darcy solution (they are not independent of it); that
shared part is what the `E_h` ladders refine.

#### T7. Pollock contrast on the production set (flux weights): consumer / Pollock ratio of means, 90 % paired seed-bootstrap

| `sigma^2` | `alpha_TP` disp | `alpha_TAP` disp | `alpha_TP` pos | `alpha_LP` | `U_r` |
|---|---|---|---|---|---|
| 0.25 | 1.002 [1.0017, 1.0022] | 1.002 [1.0017, 1.0022] | 1.0014 [0.9991, 1.0045] | 1.0042 [1.004, 1.0043] | 1.0008 [1.0008, 1.0008] |
| 1 | 1.0039 [1.0035, 1.0043] | 1.0039 [1.0035, 1.0043] | 1.0036 [1.0017, 1.0055] | 1.0076 [1.0069, 1.0084] | 1.0022 [1.0022, 1.0023] |
| 2.25 | 1.0007 [0.99951, 1.0019] | 1.0007 [0.99953, 1.0019] | 0.99867 [0.99532, 1.0022] | 1.0145 [1.008, 1.0211] | 1.0001 [0.99966, 1.0006] |
| 4 | 0.99898 [0.99688, 1.001] | 0.99898 [0.9969, 1.001] | 0.99836 [0.99547, 1.0013] | 1.0537 [0.99979, 1.123] | 0.99814 [0.99687, 0.99949] |

**Basis contrast and precision** (T8). No basis flag: the position-minus-displacement difference and its uniform-weight control (zero
by transverse stationarity) contain 0 at every amplitude. The primary (displacement) half-widths meet their targets with a wide margin
(1.5e-5 / 1.6e-4 / 4.8e-4 / 2.0e-3 against 2.5e-4 / 6e-4 / 2.8e-3 / 6.6e-3); the position-basis secondary does not meet them at 0.25 and
1 (reported, not extended).

#### T8. Basis contrast `pos - disp` (flux; uniform-weight control) and precision of the primary estimand

| `sigma^2` | `alpha_TP` pos - disp, 90 % bootstrap-t | uniform control, 90 % bootstrap-t | flag | target half-width | achieved (disp) | achieved (pos) |
|---|---|---|---|---|---|---|
| 0.25 | 3.09e-04 [-1.71e-04, 7.69e-04] | 2.63e-04 [-2.23e-04, 7.32e-04] | False | 2.50e-04 | 1.50e-05 | 4.79e-04 (not met) |
| 1 | 5.39e-04 [-4.02e-04, 1.46e-03] | 4.44e-04 [-5.78e-04, 1.41e-03] | False | 6.00e-04 | 1.58e-04 | 9.65e-04 (not met) |
| 2.25 | -1.97e-04 [-1.75e-03, 1.37e-03] | -5.27e-04 [-2.22e-03, 1.19e-03] | False | 2.80e-03 | 4.80e-04 | 1.63e-03 |
| 4 | 2.82e-04 [-7.60e-04, 1.42e-03] | -4.93e-04 [-1.66e-03, 7.22e-04] | False | 6.60e-03 | 1.98e-03 | 2.20e-03 |

**Achieved Gate 4 power** (block `gate4_power`; Student sampling limit, normal draws, realized sizes and observed SDs, R = 1000): 1.00
for both estimators at `fwer/m` and at `fwer`, in both scenarios (`no_bias`, `observed`), at every amplitude.

**Positive control** (T10). `R` = 1.145 [1.122, 1.168], inside the pre-registered `[0.8, 1.25]`: status `ok`. It is above the "few per
cent" anticipated in the rationale (prereg 7); the window slope of the ensemble transverse second moment, 1.5e-5 `lambda`
(`alpha_T`-equivalent), exceeds the box first-order slope (-1.4e-6), difference [2.5e-5, 4.2e-5]. Read with 3.3 (item 1) and with the plane-by-plane analysis N3 of 3.4.

#### T10. Positive control `sigma^2 = 0.04` (`pc_s004`, N = 30)

| statistic | value | 90 % interval | bounds / reference | status |
|---|---|---|---|---|
| `R` (flux, displacement second moment / first-order box) | 1.1447 | [1.1221, 1.1681] | [0.8, 1.25] | ok |
| window slope, `alpha_T`-equivalent (measured vs box first order) | 1.53e-05 | difference [2.52e-05, 4.22e-05] | -1.40e-06 | descriptive |

**G7 factorial at 0.25 and 1** (Par2 cells, 60 seeds each, `2^(4-1)` with `I = GIDR`; effects as ratios of cell means at the `+1`
and `-1` levels, from the log-scale block `log_effects`, 90 % bootstrap; the responses are Par2's uniform-weight position-basis plane
estimands; `alpha_TP` at 0.25 not analysed because a cell mean was not positive). **The basis of the G7 `alpha_TP` was not fixed by
the pre-registration: the frozen block used the position basis (the `alpha_TP` column without suffix), whereas the primary basis of
SF-43 is the displacement basis (deviation D8, section 4.3; recomputed in the primary basis in 3.4, N1).** Time step (D, `dt/2` against
`dt`) and solver tolerance (R, 1e-10 against 1e-6): ratios within 0.2 % of 1 everywhere (resolved at 0.25, practically nil). Injection
(I, window against full plane): `alpha_LP` ratio 0.973 [0.962, 0.984] at 0.25 (with its alias `GI = DR` 1.029: about -5.5 % with the
random-spectral generator, +0.2 % with SF-18), 0.999 [0.989, 1.010] at 1; `alpha_TP` (position basis) 1.06 [0.97, 1.16] at 1.
**Generator** (G, SF-18 periodic normalized against Par2's random-spectral with 10 000 modes; whole plot, unpaired): `alpha_LP`
0.965 [0.935, 0.999] at 0.25 and 1.002 [0.968, 1.038] at 1. For `alpha_TP` the frozen block gives **1.48 [1.11, 2.10] at 1 in the
position basis** (cell means 0.0086-0.0089 against 0.0057-0.0062 `lambda`); this is a secondary figure. **In the primary
(displacement) basis, N1 of 3.4 gives 1.05 [0.997, 1.10] (unadjusted p = 0.13) at 1 and 1.05 [0.986, 1.11] (p = 0.23) at 0.25**
(cell means at 1: 0.00706-0.00718 with SF-18, 0.00677-0.00685 `lambda` with random-spectral; all far above `c`): no generator effect on
the transverse magnitude is distinguished from 1 at these sizes. The generator class remains a model restriction of the decision
record, not a budget component. Full tables in `summary.json` (`g7.factorial`, position basis) and `A/posthoc/n1/` (both bases). None of
these factors was varied at `sigma^2 = 4`.

**Secondary estimators** (T9). SF-39-style time-window `alpha_LAM` and `alpha_TAM` on the production and distance arms; B&dD eq. 6
averaged over the same time window with their Table 1 parameters gives consumer/B&dD = 0.96 / 0.86 / 0.67 / 0.47 (production arms) and
0.86 / 0.73 / 0.54 (distance arms at 1, 2.25, 4): the gap to B&dD widens with `sigma^2` and shrinks with travel time (G9b, recorded, not
scheduled). Par2's matched transverse dispersivity equals the consumer's within 0.3 %. The `exp(-I)` weight check changes no estimate
by more than 0.2 %.

#### T9. Secondary (descriptive) estimators; units of `lambda`

| arm | `alpha_LAM` (time window) | `alpha_TAM` (time window) | `t_end` (adv. times) | B&dD eq. 6 over the window (Table 1) | consumer / B&dD |
|---|---|---|---|---|---|
| `s025_prod` | 0.2303 [0.2254, 0.2352] | 2.61e-03 [1.33e-03, 3.89e-03] | 78.47 | 0.24 | 0.9596 |
| `s1_dist` | 1.093 [1.071, 1.114] | 7.98e-03 [6.60e-03, 9.36e-03] | 139.1 | 1.27 | 0.8603 |
| `s1_prod` | 1.094 [1.078, 1.111] | 6.49e-03 [4.67e-03, 8.31e-03] | 59.12 | 1.27 | 0.8618 |
| `s225_dist` | 3.163 [3.125, 3.202] | 0.02964 [0.02801, 0.03128] | 111.3 | 4.35 | 0.7272 |
| `s225_prod` | 2.934 [2.906, 2.963] | 0.03057 [0.02817, 0.03297] | 43.41 | 4.349 | 0.6748 |
| `s4_dist` | 6.749 [6.656, 6.842] | 0.0677 [0.06516, 0.07024] | 86.51 | 12.53 | 0.5386 |
| `s4_prod` | 5.207 [5.136, 5.279] | 0.0749 [0.07168, 0.07813] | 32.83 | 11.02 | 0.4725 |

| `sigma^2` | Par2 / consumer transverse (displacement, matched) | `expI` / flux, `alpha_TP` | `expI` / flux, `alpha_LP` |
|---|---|---|---|
| 0.25 | 1.0005 [0.99797, 1.003] | 0.99844 [0.9946, 1.0023] | 1 [0.9999, 1.0001] |
| 1 | 0.99752 [0.99563, 0.99941] | 0.99883 [0.99627, 1.0014] | 0.99996 [0.99964, 1.0003] |
| 2.25 | 0.99805 [0.99604, 1.0001] | 1.0005 [0.99949, 1.0016] | 0.99963 [0.99923, 1] |
| 4 | 1.0028 [0.99862, 1.0069] | 0.99977 [0.99947, 1.0001] | 0.99975 [0.99885, 1.0007] |

**Failures and tracker counts** (T11). Two different quantities must not be confused. `frac_backflow` is the **fraction of the traced
particles** that meet backflow (`v1 < 0` along their path) at least once: `backflow_encounters / n_traced`, a 0/1 flag per particle,
per realization (`sf42red reduce --tracker-counts`; `A/scripts/sf43ana/README.md`, H17). It is not the fraction of the **domain volume**
where `v1 < 0`, which SF-43 did not compute. By particles, backflow is absent at 0.25 and 1, rare at 2.25 (0.29 % of particles on
average in the production arm, T11) and frequent at 4 (15.5 % of particles on average, up to 38 % in one realization; recrossings
0.6 % of traced particles; seed backflow in 23 of 200 production realizations). The estimand is the first-passage distribution at each plane in both codes (Par2's recorder
takes the first step with `x_n < X <= x_{n+1}`, `src/runtime/stats/PlaneCrossingRecorder.cuh`). The H17 `alpha_LP` sensitivity without
backflow-encounter particles is not computable (departure D2).

#### T11. Failures and tracker counts

Total 5294 attempted, 3 unusable (rate 5.67e-04, Clopper-Pearson 95 % [1.17e-04, 1.66e-03]); halted arms: none; unreduced segments: none.

| arm | attempted | unusable | rate | CP 95 % |
|---|---|---|---|---|
| `s4_dist` | 101 | 1 | 9.90e-03 | [2.51e-04, 0.05393] |
| `s4_l16` | 251 | 1 | 3.98e-03 | [1.01e-04, 0.022] |
| `s4_prod` | 201 | 1 | 4.98e-03 | [1.26e-04, 0.02741] |

| amplitude | family | realizations | `recrossings_per_traced` mean (max) | `frac_backflow` mean (max) | `seed_backflow` positive |
|---|---|---|---|---|---|
| s004 | positive_control | 30 | 0 (0) | 0 (0) | 0 |
| s025 | E_W | 150 | 0 (0) | 0 (0) | 0 |
| s025 | E_h | 200 | 0 (0) | 0 (0) | 0 |
| s025 | h_max_pass | 60 | 0 (0) | 0 (0) | 0 |
| s025 | production | 200 | 0 (0) | 0 (0) | 0 |
| s1 | E_W | 150 | 0 (0) | 0 (0) | 0 |
| s1 | E_h | 200 | 0 (0) | 0 (0) | 0 |
| s1 | distance | 100 | 0 (0) | 0 (0) | 0 |
| s1 | h_max_pass | 60 | 0 (0) | 0 (0) | 0 |
| s1 | production | 200 | 0 (0) | 0 (0) | 0 |
| s225 | E_W | 300 | 5.67e-05 (1.00e-03) | 2.89e-03 (0.0163) | 0 |
| s225 | E_h | 300 | 3.20e-05 (8.00e-04) | 2.50e-03 (0.022) | 2 |
| s225 | distance | 100 | 1.50e-04 (1.80e-03) | 6.67e-03 (0.0284) | 3 |
| s225 | gate5_consumer | 200 | 2.60e-05 (7.00e-04) | 2.84e-03 (0.0212) | 2 |
| s225 | h_max_pass | 60 | 1.33e-05 (1.00e-04) | 2.09e-03 (9.20e-03) | 0 |
| s225 | production | 200 | 3.25e-05 (8.00e-04) | 2.93e-03 (0.0203) | 2 |
| s4 | E_W | 300 | 6.10e-03 (0.0347) | 0.1666 (0.3236) | 40 |
| s4 | E_h | 300 | 5.08e-03 (0.0329) | 0.133 (0.3692) | 28 |
| s4 | distance | 100 | 0.0132 (0.0444) | 0.2797 (0.4783) | 15 |
| s4 | gate5_consumer | 250 | 6.29e-03 (0.0393) | 0.154 (0.3846) | 25 |
| s4 | h_max_pass | 60 | 4.86e-03 (0.0227) | 0.1463 (0.2258) | 6 |
| s4 | production | 200 | 6.44e-03 (0.0427) | 0.1554 (0.3833) | 23 |

### 3.3 Exploratory (post hoc; not pre-registered, not tested)

1. **Amplitude scaling of the transverse growth.** `alpha_TP / sigma^4` = 8.4e-3 / 7.2e-3 / 5.7e-3 / 4.3e-3 at 0.25 / 1 / 2.25 / 4
   (displacement basis, production means), and about 9.5e-3 at 0.04 if the positive control's ensemble window slope (a related but not
   identical estimator) is used. The ratio falls by about half between 0.25 and 4, so there is **no `sigma^4` law over this range**:
   post hoc S (3.4) gives the local exponents `d ln alpha_TP / d ln sigma^2` of 1.93 [1.77, 2.09] (0.04 -> 0.25), 1.89 [1.86, 1.92],
   1.70 [1.67, 1.74] and 1.51 [1.45, 1.57] (2.25 -> 4), decreasing steadily from about 2 to about 1.5 (a `sigma^4` law would be an
   exponent of 2 throughout); only the first interval, with 30 realizations at 0.04, contains 2. B&dD's eq. 7 states `alpha_TAM ~ alpha_TAM(1) sigma^4` for their asymptotic
   time-based estimator (`docs/theory/beaudoin-de-dreuzy-2013-key-claims.md`); SF-43's plane-based values are 0.6-0.7 of B&dD's tabulated
   `alpha_TA` (0.01 at 1, 0.044 at 2.25; about 0.11 read from their Fig. 4b at 4); different estimands, order of magnitude only [prereg 12].
2. **`alpha_LP` across grids and widths** (per-arm means of the frozen `failures` block, unpaired view): at 0.25 and 1 the `lambda/h`
   5 / 10 / 20 values differ by at most 0.9 %; at 2.25 the 4 / 8 / 12 / 16 sequence is 3.405 / 3.281 / 3.265 / 3.261 (converging); at 4 the
   4 / 8 / 16 / 32 sequence is 11.25 / 9.32 / 9.17 / 9.44 (the finest level with a heavy right tail, [8.86, 10.02]). Width doubling moves
   `alpha_LP` by at most 1 %. The frozen analysis computes no `E_h`/`E_W` bound for `alpha_LP` (deviation D6 below).
3. **Box length at `sigma^2 = 4`.** On the same window, grid and width, the L = 208 arm gives `alpha_LP` = 8.63 [8.41, 8.86] against
   9.17 [8.95, 9.39] on the L = 104 rung (`s4_l16`), about -6 %, and `alpha_TP` about -2 %; at 1 and 2.25 the differences are below 1 %.
   The outlet boundary layer or the generator's period in `x` (equal to the box length) may matter for the slow tail at `sigma^2 = 4`;
   no SF-43 budget covers the box length.
4. **Candidate Par2-side mechanisms for the `sigma^2 = 4` offset** (not separable with these data): (a) explicit Euler with fixed `dt`
   (`U dt = 0.0244 dx`) in fast channels, where `|v|` reaches tens of `U` at `sigma^2 = 4`, so a step approaches a cell; D was nil at 0.25
   and 1 but was not varied at 4; (b) truncation of the slow tail by Par2's finite horizon (`3.6 (L1 - 10 ell) / dt` steps; 40 realizations
   below 0.999 completeness, and `alpha_LP` is a variance of travel times dominated by the slow tail); (c) random against lattice
   injection (the G7 injection factor moved `alpha_LP` by up to about 3 % at 0.25, mostly with the random-spectral generator; unknown at
   4). A discriminating follow-up would rerun a subset of `s4_g5p` with `dt/2` and a longer
   horizon. The owner decided on 2026-11-04 to run it as addendum 1 of SF-43 (`docs/decisions/2026-11-04-sf43-addendum-1.md`); its
   results are in section 3.5 of this note (J1: class `horizonte dominante` at n = 60 and n = 100; Par2's ratio for `dt/2` alone is `r_D` = 1.017 [1.000, 1.037] at n = 60 and 1.008 [0.9962, 1.022] at n = 100).

### 3.4 Post hoc (2026-11-04)

Everything in this section is **post hoc**: the analyses were decided after the frozen results were known (owner decisions of
2026-11-04, `docs/decisions/2026-11-04-sf43-addendum-1.md`, item 4). They are descriptive, outside the Holm family (`m = 8`), use only
the committed frozen data (no GPU, no host) and change no pre-registered result, interval, class or decision. Method, commands, the
scripts (`A/scripts/posthoc/`) and the full-precision outputs are in `A/posthoc/RESULTS.md` and `A/posthoc/{n1,n3,n5,n6,n8,scaling}/`
(accepted commit `94f75e8d`; sha256 in `A/posthoc/SHA256SUMS`). The archives were extracted outside the repository and the 2409 files
verified against `freeze.reduced.sha256` (0 mismatched). Numbers below are copied from `A/posthoc/RESULTS.md` (3 digits).

**N1. G7 in the primary (displacement) basis (deviation D8).** The frozen `sf43ana` was run unchanged with a copy of the
pre-registration that differs only in `g7.responses` (`alpha_TP:disp`, `alpha_TAP:disp`, `alpha_TP`, `alpha_LP`). Fidelity to the
committed `summary.json` (max relative deviation per block): every block within 1e-9 except two leaves of `secondaries`
(`time_window/{s1_prod,s225_dist}/bdd_window/predicted/sd`, 7.0e-9 and 1.9e-9 relative, i.e. 5.2e-18 and 4.7e-17 absolute on values of
7.33e-10 and 2.48e-8: the SD of a near-constant deterministic prediction), reported as found and not adjusted; the Gate 5 power entries
(Monte Carlo) within 1.3e-5 (limit 1e-4); no string, flag, class or decision differs; the position-basis G7 entries equal the committed
ones to 3.3e-12. Generator effect on Par2 (G: SF-18 over random spectral; ratio, 90 % interval, unadjusted p):

| `sigma^2` | basis | `alpha_TP` G | `alpha_LP` G |
|---|---|---|---|
| 1 | displacement (primary) | 1.05 [0.997, 1.10], p = 0.130 | 1.00 [0.968, 1.04], p = 0.928 |
| 0.25 | displacement (primary) | 1.05 [0.986, 1.11], p = 0.225 | 0.965 [0.935, 0.999], p = 0.086 |
| 1 | position (frozen G7; secondary) | 1.48 [1.11, 2.10], p = 0.041 | - |
| 0.25 | position | not analysed (non-positive cell mean) | - |

(`alpha_TAP:disp` coincides with `alpha_TP:disp` to 5 digits. The displacement-basis I and GI effects at 0.25 are 1.04 [1.02, 1.06]
(p = 0.0017) and 0.975 [0.956, 0.994] (p = 0.043), at 1: 1.01 [1.00, 1.03] and 1.00 [0.988, 1.02]; they are listed because they are in
the output, with p unadjusted and belonging to no family.) The displacement-basis consumer/Par2 and Pollock/Par2 contrasts of
`alpha_TP` at 0.25 and 1 are those of T6; at 4 they are 0.997 [0.993, 1.001] and 0.996 [0.992, 1.000]. Outputs: `A/posthoc/n1/`.

**N3. Positive control by plane, `pc_s004` (sigma^2 = 0.04, N = 30, 70 planes, consumer, displacement basis).** With the frozen
`sf43ana.positive_control` code (first-order box prediction), the measured transverse second moment relative to the box prediction
(flux weights, 90 % realization-bootstrap, B 2000, seed 20261104) is, for the centred moment (the quantity `alpha_TP` is the slope of):
1.02 [0.977, 1.07] at d = 25 and 1.27 [1.23, 1.33] at d = 94 (raw ratio; 1.03 -> 1.24 for the OLS line of the excess plus the
prediction); for the second moment (what `R` is): 1.04 [0.991, 1.08] -> 1.29 [1.24, 1.34] (OLS line 1.05 -> 1.25). `R` is reproduced
exactly (1.14474). The excess (measured - predicted) grows about linearly: slope over [25, 94] of 3.33e-5 [2.53e-5, 4.27e-5]
`lambda^2` per `lambda` (centred; 3.36e-5 [2.54e-5, 4.24e-5] second moment), positive in all bootstrap resamples and stable over 8
trimmed windows (2.7e-5 to 3.5e-5; the 90 % interval excludes zero in all of them). The box prediction itself falls by 4.55 % between
d = 80 and 94 (window slope -2.80e-6 against the measured 3.05e-5). **The zero crossing of the excess is poorly determined**: 8.5
(second moment) / 12.9 `lambda` (centred) on the full window, 0.5-15.6 over the trimmed windows, bootstrap 90 % interval roughly
[-9, 24]. Reading, post hoc and descriptive: the measured transverse displacement variance grows faster with distance than the first-order
box prediction, which is flat to slightly decreasing in the window; the excess is **consistent with a second-order `alpha_T` visible
already at `sigma^2 = 0.04`**. With two amplitudes only, the data cannot separate a `sigma^2` from a `sigma^4` amplitude. Output:
`A/posthoc/n3/`.

**N5. `alpha_LP` (a) budget rules and (b) `lambda/h` 16 against 32 at `sigma^2 = 4`.** (a) The frozen G5.4 budget machinery
(`sf43ana.gate4.RobustTest`, `design_from_arms`; reproduces the committed `budgets` of `alpha_TP`/`alpha_TAP` to 1.7e-13) applied to
`alpha_LP` (displacement basis; enforced bounds as % of the production mean, `a` = 0.05 / 0.00625):

| `sigma^2` | mean `alpha_LP` | `E_h` | `E_W` | `E_WxH` | sum of the bounds | dominant |
|---|---|---|---|---|---|---|
| 0.25 | 0.233 | 0.085 / 0.090 % | 3.49 / 4.49 % | n/a | 3.57 / 4.58 % | `E_W` |
| 1 | 1.09 | 0.457 / 0.537 % | 3.22 / 4.05 % | n/a | 3.68 / 4.59 % | `E_W` |
| 2.25 | 3.26 | 19.6 / 20.8 % (fallback) | 2.65 / 3.36 % | 1.97 / 2.48 % | 24.3 / 26.6 % | `E_h` |
| 4 | 9.44 | 107 / 116 % (fallback) | 5.79 / 7.34 % | 12.5 / 14.8 % | 125 / 138 % | `E_h` |

At 4 the ladder (`lambda/h` 4, 8, 16, 32: 11.2, 9.32, 9.09, 9.40) is non-monotone, no order is estimable and the order-agreement gate
fails; at 2.25 the lower order limit is not positive and the fallback applies (order-branch value 11.3 % of the mean against 19.6 %).
**`alpha_LP` at `sigma^2 = 4` is not grid-converged by this rule and is not presented as a value**; at 2.25 the bound is dominated by
the fallback, not by an observed non-convergence. (b) Paired by seed (n = 200), `alpha_LP(s4_l16) / alpha_LP(s4_prod)` = 0.961,
Student 90 % [0.902, 1.02], bootstrap-t [0.847, 1.01] (means 9.07 / 9.44; SD 1.85 / 4.98): both intervals contain 1, but seed 434111
alone (production value 72.0 against 9.21 at `l16`) moves the ratio by +3.4 %, more than the next four seeds together, so this does not
establish convergence between `lambda/h` 16 and 32. Output: `A/posthoc/n5/`.

**N6. Local `alpha_T` against distance (`s1_dist`, `s225_dist`, `s4_dist`, 100 realizations each, planes d = 25..194).** `alpha_TP`
of nine windows of 17 `lambda` ([25, 42] ... [161, 178]) over the main-window value (the window [25, 94] reproduces the frozen
per-seed values to 6.6e-16): range 0.941-1.08 (1), 0.971-1.09 (2.25), 0.906-1.06 (4); half-widths of the 90 % intervals of the
ratios about 0.05-0.12; 4 of the 27 intervals exclude 1 (about the number expected from 27 unadjusted 90 % intervals; no common pattern
with distance across the amplitudes). Linear trend of the local values, in % of the main value per 100 `lambda` [90 % interval], by two
definitions that give different magnitudes: windows +1.25 [-4.5, +7.2] / +0.72 [-3.3, +5.1] / -2.18 [-8.0, +3.9] % and quadratic
coefficient of the global fit over d >= 25 +4.25 [-0.82, +9.6] / +3.14 [-0.58, +7.1] / -2.01 [-7.8, +3.9] % at 1 / 2.25 / 4; **all
six intervals contain 0**. A descriptive saturation fit `s0^2 + s_psi^2 (1 - exp(-d/zeta))` of the ensemble-mean curve does not show a
saturation scale within 170 `lambda` at any amplitude; `zeta` is not identifiable upward, so only one-sided 95 % lower bounds are
reported (percentile bootstrap, 2000 refits): 1.2e4, 1.8e4 and 1.3e3 `lambda` at 1 / 2.25 / 4 (the share of refits at the grid end
1e5: 0.92 / 0.92 / 0.29). The bound measures how often resamples look linear. **No statement about Lester's asymptotic claim is
made.** Output: `A/posthoc/n6/`.

**N8. Par2 completeness strata at `sigma^2 = 4` (selection, not causal).** On the 250 Gate 5 pairs (uniform weights), stratified by the
number of Par2 particles that did not cross the last window plane: Pollock/Par2 `alpha_LP` = 1.05 [1.02, 1.08] (0 missing, 25 pairs),
1.13 [1.07, 1.20] (1-2, 97), 1.15 [1.12, 1.19] (3-9, 86), 1.36 [1.26, 1.48] (>= 10, 42); correlation of log(Pollock/Par2) with the
missing fraction 0.490 (Pearson), 0.349 (Spearman). The strata are defined by a Par2 property that correlates with the outcome, so
this is a **selection, not a causal estimate**; the residual of about 5 % in the complete stratum is not explained by the horizon.
Output: `A/posthoc/n8/`.

**S. Scaling with sigma^2.** `alpha_TP / sigma^4` (displacement, production means, 90 % Student): 9.54e-3 [6.71e-3, 1.24e-2] at 0.04
(N = 30), 8.42e-3 [8.18e-3, 8.66e-3] at 0.25, 7.22e-3 [7.06e-3, 7.38e-3] at 1, 5.67e-3 [5.57e-3, 5.76e-3] at 2.25, 4.28e-3
[4.15e-3, 4.40e-3] at 4; local exponents between consecutive amplitudes 1.93, 1.89, 1.70, 1.51 (intervals in 3.3, item 1). Descriptive;
no functional form is fitted and nothing here concerns the asymptotic `alpha_T`. Output: `A/posthoc/scaling/`.

### 3.5 Addendum 1 (exploratory; pre-registered 2026-11-04, outside the Holm family)

Scope. Everything in this section is **descriptive and outside the Holm family** (`m = 8` unchanged). The rules were pre-registered in
`A/preregistration-addendum-1.md` (sha256 of the machine-readable companion `A/prereg_addendum.json` recorded in the summary:
`2305529c...`) with its pre-run Amendment A1, before any addendum script ran on data; the owner's text is in its Appendix A. The
confirmatory `summary.json` (`c5c543d5...`) is unchanged; the confirmatory verdict `gate5:4` stays `disagree` ("equivalence not
established") and is **not recomputed**. Acceptance D at `sigma^2 = 4` remains the owner's decision; the J1 class below is an input to
it, not a verdict. Numbers are copied from `A/addendum1-results/analysis/addendum_summary.{md,json}` (4 significant digits), the frozen
output of the pre-registered analysis (`sf43add.ana run`, host sha256 `aa85b41c...`); intervals are IC90 by bootstrap-t (Welch on the
log for unpaired ratios), the Student interval of the same function beside it, never used by a rule. Artifacts: `A/addendum1-results/`
(section 8).

**What ran.** Seven arms, each an exact copy of its origin arm's runner and reduce blocks except the manipulated factor (the plan
generator asserts it); J1 and J3 reuse the field seeds of their origin arms with the same transport seeds (common random numbers, by
design), the J9 block 438001-438150 is new and disjoint.

| arm | origin | manipulated factor (only change) | seeds | n |
|---|---|---|---|---|
| `j1_base10` | `s4_g5p` | none (reproduction check) | 434001-434010 | 10 |
| `j1_H3` | `s4_g5p` | horizon factor `hz` (dimensionless multiplier of the number of steps) 3.6 -> 10.8 | 434001-434060 | 60 |
| `j1_D` | `s4_g5p` | `dt/2` (`2 x n_steps`) | 434001-434060 | 60 |
| `j1_DH3` | `s4_g5p` | both | 434001-434060 | 60 |
| `j1_H2` | `s4_g5p` | `hz` 3.6 -> 7.2 (dose-response) | 434001-434060 | 60 |
| `j3_l5` | `s1_dist` | `lambda/h` 10 -> 5 (grid 1040x120x120; `ell` 1/24, L = 208, W = 24 `lambda`) | 436001-436100 | 100 |
| `j9_w48` | `j3_l5` | W = 48 `lambda` (grid 1040x240x240, `ell` 1/48, window `0.3,0.7,0.3,0.7`; `lambda/h` 5) | 438001-438150 | 150 |

Commands (one detached job each, one GPU, granted by the helper; never `--gpu` or `CUDA_VISIBLE_DEVICES`):
`REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-43 run sf43add-prod -- "bash A/scripts/run_sf43.sh
build/v100-release all --plan A/plan_addendum.json"`; S0 (smoke, structure and cost only) `sf43add-s0` with `--smoke` and plan
`5a1c1101`. Production plan `A/plan_addendum.json` sha256 `c4eba9cdb08da079b763fcf44fcc646a2dd5fe388b2bcf02a2b3172d0f09dec1`
(finalized from S0 by the plan generator); binaries md5 in `A/binaries_addendum.json` (`macroflow3d_pipeline`
`de66d6a20e15ccbc1c429a51f5a06a87`, `lagrangian_ensemble` `4099aa1aa7ce1ac5cdb0482552fdcb0f`; deviation AD1). Blinded phase
2026-11-04T18:39:14Z -> freeze (bitácora row, commit `786448b9`, pushed 2026-11-05T09:15:08Z, before `addendum_summary.json` was read)
-> `blinded stop` 2026-11-05T09:15:16Z. Production result: 50 units, all `cls=ok`, completed = planned in every arm (10, 60, 60, 60, 60,
100, 150), 0 replaced, 0 failed, 0 OOM; `recompute43` PASS and `reconcile43` OK on the addendum campaign; analysis `errors = 0`,
`warnings = 0`, 77 frozen files verified.

**J8 (conditional extension, triggered after the freeze).** Job `sf43add-j8`, same command form with `--plan A/plan_addendum_j8.json`
(sha256 `40ef528d9e9e9c3c84b641d66e89758bed78c05cb2dc0d577c198a0d1713200a`, finalized from the production-measured costs
`A/j8_costs.json` `021727c4...`, budget 3.10 GPU-h, disk guard 378 GB); arms `j8_H3`, `j8_D`, `j8_DH3` (seeds 434061-434100, identical
settings; no `j8_H2` arm, so `r_H2` and `q_H2` are not evaluable at n = 100 by design). Own blinded phase 2026-11-05T09:31:33Z -> freeze
(commit `188054d4`) -> `blinded stop` 2026-11-05T10:34:41Z; GPU granted 09:32:22Z, `DONE` 10:33:01Z. Result: 12 units, all `cls=ok`,
40/40 realizations per arm, 0 replaced, 0 failed, 0 OOM; `RECOMPUTE PASS`, `RECONCILE OK`, `STRUCTURE_OK`. Artifacts:
`A/addendum1-results/j8/`.

**GPU-h per job** (held = GPUs x wall from the grant):

| job | held GPU-h | note |
|---|---|---|
| `sf43add-s0` (S0 smoke) | 0.084 | engine 0.07 |
| `sf43add-prod` (production) | 4.307 | engine `gpu_h_used` 3.81 of 7.41; the rest is the in-job analysis, recompute and reconcile |
| `sf43add-j8` (conditional J8) | 1.011 | engine 1.00 of 3.10 |
| addendum total | 5.402 | 0.084 + 4.307 + 1.011, of the 8 GPU-h cap (4.391 before J8) |
| SF-43 total | 59.10 | of 150 (cap 61.7 for the addendum, decision 2026-11-04; 58.09 before J8) |

#### T12. Every rule of section 3 of the pre-registration (n = 60 for J1; class conditions in the order of the first match)

| rule | condition | value (IC90) | threshold | truth | outcome |
|---|---|---|---|---|---|
| 3.1.1 reproduction | `j1_base10` = `s4_g5p`, seeds 434001-434010: max relative difference over 30 rows; Par2 crossing CSV sha256 equal | 0; 10/10 equal (bitwise) | <= 1e-12 | true | **PASS** (J1 classified) |
| 3.1.2 manipulation (`j1_H3`) | mean completeness | 0.99999 | >= 0.999 | true | PASS |
| 3.1.2 manipulation (`j1_H3`) | fraction of complete realizations | 0.9167 (55/60) | >= 0.90 | true | PASS (horizon saturated) |
| 3.1.3 class 1 `horizonte dominante` | `r_H` | 1.104 [1.066, 1.16] | >= 1.10 | true | |
| 3.1.3 class 1 | `abs(r_D - 1)` | 0.01667 (`r_D` = 1.017 [1.000, 1.037]) | <= 0.03 | true | |
| 3.1.3 class 1 | IC90(`q_H3`) within | [1.011, 1.096] | [0.9091, 1.10] | true | **MATCH (first match)** |
| 3.1.3 class 2 `paso dominante` | `r_D` | 1.017 | >= 1.05 | false | |
| 3.1.3 class 2 | `r_H` | 1.104 | < 1.05 | false | not met |
| 3.1.3 class 3 `mixto` | `r_H` | 1.104 | >= 1.03 | true | |
| 3.1.3 class 3 | `r_D` | 1.017 | >= 1.03 | false | |
| 3.1.3 class 3 | IC90(`q_DH3`) within | [0.9725, 1.041] | [0.9091, 1.10] | true | not met (`r_D`) |
| 3.1.3 class 4 `abierto` | any other case | - | - | - | not reached |
| Amendment A1 override | lower limit of IC90(`q_DH3`) above 1.10 | 0.9725 | > 1.10 | false | not applied; class before the override: `horizonte dominante` |
| **J1 class (n = 60, PRIMARY)** | first match of the pre-registered order | | | | **`horizonte dominante`** |
| J8 trigger (a) | J1 passed 3.1.1 | PASS | | true | |
| J8 trigger (b), `r_H` | IC90 contains in its interior a threshold of {1.03, 1.05, 1.10} | [1.066, 1.16] | 1.10 | true | contained: 1.10 |
| J8 trigger (b), `r_D` | threshold of {0.97, 1.03, 1.05} | [1.000, 1.037] | 1.03 | true | contained: 1.03 |
| J8 trigger (b), `q_H3` | threshold of {0.9091, 1.10} | [1.011, 1.096] | none | false | none contained |
| J8 trigger (b), `q_DH3` | threshold of {0.9091, 1.10} | [0.9725, 1.041] | none | false | none contained |
| J8 trigger (c) | addendum GPU-h used + 1.2 x projected J8 cost <= 8 | 4.391 + 1.2 x 1.4 = 6.071 | <= 8 | true | |
| J8 trigger (d) | J8 `cutoff_utc` + grace and projected end <= 2026-11-07T03:00Z | end 2026-11-05T13:00Z | <= 2026-11-07T03:00Z | true | **J8 RUN** (bitácora row, commit `e019df38`) |
| 3.2 J3 | IC90(`m_main`) within | [0.9942, 0.9969] | [0.97, 1.03] | true | |
| 3.2 J3 | IC90(`m_far`) within | [0.9927, 0.9976] | [0.97, 1.03] | true | |
| 3.2 J3 | IC90(`m_far/m_main`) within | [0.9967, 1.002] | [0.98, 1.02] | true | |
| **J3 class** | all three | | | | **`lambda/h = 5 equivalente`** |
| 3.3 J9 class 0 | fewer than 100 `j9_w48` realizations completed | 150 completed | < 100 | false | not applied |
| 3.3 J9 class 1 `meseta robusta al ancho` | IC90(`F48`) within | [0.9572, 1.042] | [0.90, 1.10] | true | |
| 3.3 J9 class 1 | IC90(`F48/F5`) within (unpaired) | [0.9192, 1.040] | [0.90, 1.10] | true | **MATCH (first match)** |
| 3.3 J9 class 2 `decae con el ancho` | upper limit of IC90(`F48`) | 1.042 | < 0.95 | false | not met |
| **J9 class** | first match; J3 qualification: none (J3 equivalent) | | | | **`meseta robusta al ancho`** |

#### T12b. J1 rules at n = 100 (SECONDARY; same rule, +40 seeds 434061-434100 in H3, D and DH3; n = 60 above stays PRIMARY)

| rule | condition | value (IC90) | threshold | truth | outcome |
|---|---|---|---|---|---|
| 3.1.2 manipulation (`j1_H3`) | mean completeness | 0.99997 | >= 0.999 | true | PASS |
| 3.1.2 manipulation (`j1_H3`) | fraction of complete realizations | 0.92 | >= 0.90 | true | PASS |
| 3.1.3 class 1 `horizonte dominante` | `r_H` | 1.105 [1.079, 1.14] | >= 1.10 | true | |
| 3.1.3 class 1 | `abs(r_D - 1)` | 0.008484 (`r_D` = 1.008 [0.9962, 1.022]) | <= 0.03 | true | |
| 3.1.3 class 1 | IC90(`q_H3`) within | [1.004, 1.062] | [0.9091, 1.10] | true | **MATCH (first match)** |
| 3.1.3 class 2 `paso dominante` | `r_D` | 1.008 | >= 1.05 | false | |
| 3.1.3 class 2 | `r_H` | 1.105 | < 1.05 | false | not met |
| 3.1.3 class 3 `mixto` | `r_H` | 1.105 | >= 1.03 | true | |
| 3.1.3 class 3 | `r_D` | 1.008 | >= 1.03 | false | |
| 3.1.3 class 3 | IC90(`q_DH3`) within | [0.9764, 1.024] | [0.9091, 1.10] | true | not met (`r_D`) |
| Amendment A1 override | lower limit of IC90(`q_DH3`) above 1.10 | 0.9764 | > 1.10 | false | not applied |
| **J1 class (n = 100, SECONDARY)** | first match | | | | **`horizonte dominante`** |

#### T13. J1 (Par2 at `sigma^2 = 4`: horizon against step), paired by seed, n = 60 common seeds

`r_X = Par2(X)/Par2(s4_g5p)`; `q_X = Pollock(s4_g5pol)/Par2(X)` (`q_base` on the same 60 seeds).

| quantity | n | ratio | IC90 bootstrap-t | IC90 Student |
|---|---|---|---|---|
| `r_H` (`hz` 10.8) | 60 | 1.104 | [1.066, 1.16] | [1.061, 1.15] |
| `r_D` (`dt/2`) | 60 | 1.017 | [1.000, 1.037] | [0.9987, 1.035] |
| `r_H2` (`hz` 7.2; descriptive) | 60 | 1.104 | [1.073, 1.166] | [1.066, 1.144] |
| `r_DH3` (both; descriptive) | 60 | 1.150 | [1.099, 1.229] | [1.092, 1.211] |
| `q_base` | 60 | 1.154 | [1.104, 1.232] | [1.095, 1.217] |
| `q_H2` | 60 | 1.045 | [1.013, 1.097] | [1.006, 1.086] |
| `q_H3` (residual; not attributed by the rule) | 60 | 1.045 | [1.011, 1.096] | [1.004, 1.088] |
| `q_D` | 60 | 1.135 | [1.090, 1.200] | [1.083, 1.190] |
| `q_DH3` (residual; not attributed by the rule) | 60 | 1.004 | [0.9725, 1.041] | [0.9702, 1.039] |

Crossing completeness (`n_last / n_traced`; the frozen `s4_g5p` on the same 60 seeds is the base):

| arm | n | mean | fraction complete | quantiles 0 / 5 / 25 / 50 / 75 / 95 / 100 % |
|---|---|---|---|---|
| `s4_g5p` (base, `hz` 3.6) | 60 | 0.9991 | 0.1167 | 0.9914, 0.9952, 0.9995, 0.9997, 0.9999, 1, 1 |
| `j1_base10` | 10 | 0.9994 | 0.2000 | 0.9954, 0.9973, 0.9997, 0.9999, 0.9999, 1, 1 |
| `j1_H2` | 60 | 1.000 | 0.8333 | 0.9989, 0.9998, 1, 1, 1, 1, 1 |
| `j1_H3` | 60 | 1.000 (0.99999) | 0.9167 | 0.9998, 0.9999, 1, 1, 1, 1, 1 |
| `j1_D` | 60 | 0.9991 | 0.1833 | 0.9911, 0.9949, 0.9994, 0.9997, 0.9999, 1, 1 |
| `j1_DH3` | 60 | 1.000 | 0.9000 | 0.9998, 0.9999, 1, 1, 1, 1, 1 |

Representativity: `q_base` on the 60 seeds is 1.154 [1.104, 1.232] (Student [1.095, 1.217]) against the frozen 1.164 [1.131, 1.198]
(250 seeds, Student): the frozen ratio lies in both intervals and the Student intervals overlap, so the 60 seeds are consistent with
the 250 for this quantity.

**Class at n = 60 (PRIMARY): `horizonte dominante`** (first match). Plainly: `r_H` = 1.104 clears the 1.10 threshold by **0.004 on the
point estimate**, while its IC90 [1.066, 1.16] **contains 1.10**; the class is therefore fragile on `r_H`, which is why the
pre-registered J8 extension was triggered (T12). For the step alone, Par2's ratio is `r_D` = 1.017 [1.000, 1.037] (Student [0.9987, 1.035]): its bootstrap-t interval reaches 1, and no
effect of the step alone is claimed (the class 1 condition `abs(r_D - 1) <= 0.03` holds on the point estimate); at the long
horizon 0.92 of the realizations are complete against 0.12 at the base horizon. The class at n = 100 (SECONDARY; same rule, +40 seeds in H3, D and DH3) is given in T13b.

Descriptive observations (not rule results): (i) `r_H2` (`hz` 7.2) equals `r_H` (`hz` 10.8) to 5e-5, so the horizon effect on `alpha_LP`
is saturated by `hz` 7.2; (ii) with both factors `q_DH3` = 1.004 [0.9725, 1.041], i.e. Pollock/Par2 is not distinguished from 1 when
the horizon is long and the step is halved; (iii) descriptive arithmetic on the point estimates: `r_DH3 / r_H` = 1.041 with the long horizon, against `r_D` = 1.017 alone; (iv) the residual `q_H3` = 1.045 [1.011, 1.096] is **not attributed by the rule** (about 5 %
was expected from the post hoc completeness strata, N8) and is not called a step effect here.

#### T13b. J1 at n = 100 (SECONDARY): `j8_*` seeds 434061-434100 added to 434001-434060 in H3, D and DH3

`j1_H2` and `j1_base10` have no seeds 434061-434100, so `r_H2` and `q_H2` are not evaluable at n = 100 by design (no `j8_H2` arm).

| quantity | n | ratio | IC90 bootstrap-t | IC90 Student |
|---|---|---|---|---|
| `r_H` | 100 | 1.105 | [1.079, 1.14] | [1.076, 1.135] |
| `r_D` | 100 | 1.008 | [0.9962, 1.022] | [0.9957, 1.021] |
| `r_DH3` (descriptive) | 100 | 1.138 | [1.105, 1.185] | [1.099, 1.178] |
| `q_base` | 100 | 1.137 | [1.101, 1.19] | [1.096, 1.179] |
| `q_H3` (residual; not attributed by the rule) | 100 | 1.029 | [1.004, 1.062] | [1.002, 1.057] |
| `q_D` | 100 | 1.128 | [1.093, 1.171] | [1.092, 1.165] |
| `q_DH3` (residual; not attributed by the rule) | 100 | 0.9995 | [0.9764, 1.024] | [0.9762, 1.023] |

| arm | n | mean | fraction complete | quantiles 0 / 5 / 25 / 50 / 75 / 95 / 100 % |
|---|---|---|---|---|
| `s4_g5p` (base) | 100 | 0.9992 | 0.13 | 0.9914, 0.9954, 0.9996, 0.9998, 0.9999, 1, 1 |
| `j1_H3` | 100 | 1.000 (0.99997) | 0.92 | 0.998, 0.9999, 1, 1, 1, 1, 1 |
| `j1_D` | 100 | 0.9992 | 0.13 | 0.9911, 0.9956, 0.9994, 0.9998, 0.9999, 1, 1 |
| `j1_DH3` | 100 | 1.000 | 0.92 | 0.9981, 0.9999, 1, 1, 1, 1, 1 |

Manipulation (`j1_H3`): passed (mean 0.99997 >= 0.999, fraction 0.92 >= 0.90). Representativity: `q_base` on the 100 seeds is 1.137
[1.101, 1.19] (Student [1.096, 1.179]) against the frozen 1.164 [1.131, 1.198]; the frozen ratio lies in both intervals and the Student
intervals overlap. Conditions: T12b.

**Class at n = 100 (SECONDARY): `horizonte dominante`, the same as at n = 60 (PRIMARY).** Plainly: `r_H` = 1.105 [1.079, 1.14] **still
contains 1.10**; the class rests on the point estimate clearing the threshold by **0.005**, as at n = 60 (0.004), although the interval
is narrower. `r_D` = 1.008 [0.9962, 1.022] (n = 60: 1.017 [1.000, 1.037]); the residual `q_H3` = 1.029 [1.004, 1.062] (n = 60:
1.045 [1.011, 1.096]) is **not attributed by the rule**; `q_DH3` = 0.9995 [0.9764, 1.024] (descriptive: Pollock/Par2 not distinguished
from 1 with the long horizon and the half step, as at n = 60). Descriptive arithmetic on the point estimates: `r_DH3 / r_H` = 1.029
at n = 100 (1.041 at n = 60). Adding the 40 seeds does not change the class or any rule outcome.

#### T14. J3 (`lambda/h = 5` twin of `s1_dist`; `alpha_TP:disp`, flux weights; paired, n = 100) and J9 (W = 48 `lambda`)

| quantity | n | ratio | IC90 bootstrap-t / Welch | IC90 Student |
|---|---|---|---|---|
| `m_main` = J3 / `s1_dist`, main [25, 94] | 100 | 0.9956 | [0.9942, 0.9969] | [0.9942, 0.9969] |
| `m_far` = J3 / `s1_dist`, far [94, 194] | 100 | 0.9951 | [0.9927, 0.9976] | [0.9926, 0.9975] |
| `F5` = far / main within `j3_l5` | 100 | 1.020 | [0.9731, 1.068] | [0.9747, 1.068] |
| `m_far / m_main` (paired ratio of ratios) | 100 | 0.9995 | [0.9967, 1.002] | - |
| `F48` = far / main within `j9_w48` | 150 | 0.9978 | [0.9572, 1.042] | [0.9568, 1.041] |
| `F24` = far / main of `s1_dist` (recomputed, same method) | 100 | 1.021 | [0.9741, 1.068] | [0.9755, 1.068] |
| `F48 / F5` (unpaired, Welch on the log) | 150, 100 | 0.9779 | [0.9192, 1.040] | - |
| `F48 / F24` (unpaired) | 150, 100 | 0.9774 | [0.919, 1.040] | - |
| `W48/W24` main, `lambda/h` 5 (descriptive; no class) | 150, 100 | 0.9851 | [0.9502, 1.021] | - |
| `W48/W24` far, `lambda/h` 5 (descriptive; no class) | 150, 100 | 0.9633 | [0.9217, 1.007] | - |

The frozen `F24` (secondary of the confirmatory summary) is 1.021 with percentile IC90 [0.9756, 1.068]. J3 class: **`lambda/h = 5`
equivalente** (all three intervals within their margins; the grid at `lambda/h` 5 reproduces the `lambda/h` 10 values of `s1_dist` to
0.5 %, and the far/main ratio to 0.05 %). J9 class: **`meseta robusta al ancho`**: the far/main ratio at W = 48 `lambda` is 0.9978
[0.9572, 1.042], within 10 % of 1 and of `F5`, so the plateau to 194 `lambda` is robust to doubling the width from 24 to 48 `lambda` at
`lambda/h` = 5 within +-10 %; no J3 qualification applies. These are descriptive statements about `alpha_TP` over the two windows of this
model; they say nothing about `t -> infinity`.

**Reproduction check (3.1.1).** `j1_base10` against `s4_g5p` on seeds 434001-434010 (10 seeds, 30 rows compared): maximum relative
difference 0; the Par2 crossing CSVs agree by sha256 in 10/10 (bitwise). The S0 pre-check on seed 434001 had already passed
(`REPRO PASS`, crossing CSV sha256 equal 1/1) before the production job.

**Audit and independent checks (orchestrator).** (1) The frozen analysis was re-run locally on the copied data with python 3.13 /
numpy 2.5 / scipy 1.18 (host: 3.11 / 1.26 / 1.11): maximum relative difference 4.0e-15, every interval identical, the markdown
identical except the count of frozen files verified (78 against 77: the host ran with `--frozen-segments`, the local run without; no
number depends on it). (2) An independent csv-level recomputation of every point estimate (own csv reader; no project analysis module
imported) matched the summary. (3) The acceptance-F script (`A/scripts/review_f/`) was validated by the orchestrator on two **non-F**
realizations, 432100 (single pass; maximum relative difference 2.1e-16) and 432002 (multi-pass `hmax_8`; 6.0e-16); the four F
realizations themselves are left to the owner (section 7).

**Addendum deviations and findings** (disclosed in the bitácora; none changes a pre-registered rule or result):

- **AD1. Binaries md5.** The md5 of the host binaries differ from `A/binaries.json` because the rebuild embeds the git revision
  (`cmake/git_rev.cmake`); the sources have not changed since `e85ec5a1`. Found pre-run; numerical identity is tested by the
  reproduction check (bitwise PASS).
- **AD2. Amendment A1.** Pre-run tightening of the class-4 override of J1 (3.1.3; stated in `A/preregistration-addendum-1.md`), applied
  before any script ran on data; not applied in this result (T12).
- **AD3. Retention.** The engine runs no final prune, so `j9_w48` kept the raw Pollock arrays of 122 realizations (about 54 GB on the
  host) instead of at most 2; the reduced tables are complete and the analysis is unaffected. No deletion without the owner (open item,
  section 7). The J8 campaign shows the same engine behaviour (retention
  `j8_H3` 22/2, `j8_D` 22/2, `j8_DH3` 32/2; 13 GB on the host in `~/sf43_addendum_j8`); nothing was deleted (the owner's answer of
  2026-11-05 authorized running, not deleting).
- **AD4. Ledger hash in the summary.** The summary records the campaign-ledger sha256 at analysis time (`8520abbd...`), before the
  engine appended its final records (final ledger `9c736b2f...`, in the freeze row); informational.

## 4. Independent checks, audit and deviations

### 4.1 Independent checks (acceptance F)

- `recompute43` (independent of `sf43ana`, from raw per-realization moments and the retained plane arrays): **RECOMPUTE PASS**, 22 289
  174 values compared, 0 beyond 1e-10 relative (1e-15 absolute), 0 problems, maximum relative difference 8.97e-11. Coverage: every
  realization-window row of the four production arms from the plane arrays (displacement and position bases, `perplane` and `common`,
  flux and uniform weights; a production realization without arrays would have been a `PROBLEM`, none); the position-basis `perplane`
  rows of every consumer realization and the Par2 plane rows from the moment files. Not covered, by design [prereg 10]: 2738
  non-production realization-window rows outside the retained subset for the variants that need per-particle arrays, and the time
  family. (The report's `realizations=` and `PRIMARY_ARM ... passes=` counters count distinct source names, which repeat across
  segments; they undercount and do not measure coverage.)
- `reconcile43`: **RECONCILE OK**, 5294 solves = 5294 ledger seed records, 3 unusable = 3 replacements, 0 conflicts / unmapped /
  unparsed findings.
- Determinism across GPUs: PASS in the smoke test (72 consumer and 4 Par2 raw files) [prereg 1.5]; production lanes used the same
  binaries.
- Frozen analysis: `process.prereg_sha256` = `prereg.json`; coverage summary and qualification document hashes as pre-registered; 0
  errors, 0 warnings recorded; `check-finite` STRUCTURE_OK (64 non-finite leaves in `budgets` are the undefined orders of the non-monotone
  ladders, `p_L = -inf`, `p_use = nan`, by design).
- Human-review independent reduction of at least three realizations: **pending** (retained arrays on the host, below).

### 4.2 Audit findings (orchestrator, `PHASE4` audit)

- Gate 5 at `sigma^2 = 4`: `disagree`, attributed by the pre-registered decomposition to the Pollock -> Par2 step (3.2); the mechanism
  inside Par2 is not identified (3.3, item 4). **Owner decision required** (acceptance D).
- Gate 5 qualification at `sigma^2 = 4`: the observed moments of `D_hi`, `D_lo` (skewness 3.8-4.1, excess kurtosis 21-23) exceed the
  J3B-bound family (2.73, 7.59) used to qualify Hall's procedure. The decision is insensitive to this: the point ratio (1.188) is outside
  the margin and the bootstrap-t also gives `p` = 1.
- Qualification profile at 0.25 and 1: the frozen code counts the production-width cell at the ladder's paired prefix (100 seeds,
  `gate4.design_from_arms`), so the realized width size is 100 and the largest eligible profile is `half` (qualified; `full` was
  qualified too). `E_W` there uses 100 production seeds against 150 wide ones (conservative).
- `E_h` at 2.25 and 4: order undefined (non-monotone ladders), fallback applied; the gate intervals are reported but have no role.
- Positive control: inside its bounds but 14 % above first order (3.2).
- Process: the freeze row is labelled 13:20Z; its commit time is 13:15:43Z, before `blinded stop` (13:15:57Z); the order of G8 is
  respected (corrected in the next bitácora row). Matplotlib mathtext `UserWarning`s in `analysis.log` come from figure rendering and are
  outside `process.warnings`.

### 4.3 Deviations (section 13 of the pre-registration; all recorded in the bitácora)

Recorded before data: D1-D4 (prereg 14). Found after data, none changes a decision:

- **D5. B&dD eq. 6 fit and Par2 time-window estimators not computed.** The reduced tables carry no snapshot-time curves (`curve_t`,
  `curve_alpha`) and no Par2 `alpha_*:time` columns; the frozen analysis records `input absent` for every `bdd_fit` entry and only
  `t_end_adv` for the Gate 5 Par2 arms. The `sf43ana` README listed the dependency; prereg 7 listed the fit without flagging it. The
  B&dD-window comparison for the consumer is computed (T9). Not recoverable post hoc: Par2 snapshots and consumer time arrays were pruned
  after reduction except the retained subset [H21].
- **D6. `E_h`/`E_W`/`E_int` computed for the transverse estimands only.** Prereg 7 says "budgets per estimand"; the frozen companion
  computes them for `alpha_TP` and `alpha_TAP` (both bases). `alpha_LP` carries its sampling interval, the Pollock contrast, the halves,
  the distance ratio and the per-arm means (3.3, item 2), but no budget bound. Under the rule of section 13 the companion ran.
- **D7. Recompute coverage counters** (4.1): reporting only; the comparisons are per row and the coverage is as pre-registered.
- **D8. Basis of the G7 responses not fixed** (found 2026-11-04, after data). The pre-registration did not fix the basis of the G7
  responses (`prereg.json`: `g7.responses` = `alpha_LP`, `alpha_TP`, without a basis suffix), so the frozen G7 block used the position
  basis (the column without suffix; `alpha_TP` at 0.25 not analysable in it), whereas the primary basis of SF-43 is the displacement
  basis (prereg 4). The first version of this note therefore reported a generator effect of 1.48 [1.11, 2.10] at `sigma^2 = 1`, which is
  a position-basis figure. Recomputed in the primary basis with the unchanged frozen analysis and a copy of the pre-registration that
  only changes `g7.responses` (post hoc N1, 3.4): 1.05 [0.997, 1.10] at 1 and 1.05 [0.986, 1.11] at 0.25. G7 is descriptive and
  outside the family, so no decision, class or confirmatory result changes.

**Corrections of 2026-11-04 (addendum 1)** to the text of this note (no pre-registered result or frozen output changed):

1. Generator: conclusion 5.3, caveat 6, the G7 paragraph (T10 block), open item 7.4 and the Lester theory note no longer conclude
   "about 1.5 times"; they report the primary-basis factor (N1, with D8), the position-basis 1.48 only as a labelled secondary
   figure; the caveat wording is marked "proposed wording; owner decision". The sentence calling the generator the "largest single
   sensitivity" of the transverse magnitude was removed (not supported in the primary basis).
2. T7 discussion: "three independent trackers" -> "three trackers that share the field and the Darcy solution".
3. Scaling: 3.3 item 1 and the B&dD theory note no longer say "close to `sigma^4`" or "close to quadratic at small amplitude"; they give
   `alpha_TP/sigma^4` and the local exponents (S, 3.4).
4. Pollock contrast: no verdict `agree`/`indistinguishable` (no equivalence margin was pre-registered; prereg 7 lists it as reported,
   outside budget and family); +0.2 % and +0.4 % at 0.25 and 1 stated as resolved (intervals exclude 1); "contain or nearly contain 1"
   removed.
5. Dates: the analysis freeze is 2026-11-04T13:15Z (commit `40135561`, 13:15:43Z) and is now stated in the execution record; the
   job's `FINAL DONE` (2026-11-04T08:50:49Z) is the end of the analysis run, not the freeze. No other freeze date occurs in this
   note or in the two theory notes.
6. Backflow: `frac_backflow` (fraction of traced particles meeting `v1 < 0`; 0.29 % at 2.25, 15.5 % at 4) is defined and kept apart from
   the fraction of the domain volume with `v1 < 0` (not computed in SF-43). A volume fraction of "0.12 %" does not occur in this note
   or in the two theory notes, nor does it have a source in the frozen evidence read for this correction.
7. Per-step error bound of the integrator ("about 2.4e-7 lambda"): the number does not occur in this note or in the two theory notes
   and no derivation or source exists in the frozen evidence (`A/results/**`, `A/posthoc/RESULTS.md`); none is added. (The
   `4e-7` at `tol = 1e-8` of the Lester theory note is that paper's figure reading for its own scheme, not a SF-43 bound.)
8. Cost: 52.28 GPU-h is production (GPUs held x wall); 53.70 GPU-h is the SF-43 total with the 1.42 GPU-h of smoke jobs; the engine's
   51.14 is labelled as its own count.
9. 3.3 item 4 and open item 7.1: the Par2 follow-up is no longer "an owner decision, not part of this increment"; the owner decided it
   on 2026-11-04 as addendum 1 (pointer only; the results are in section 3.5, "Addendum 1 (exploratory)").
10. New section 3.4 (post hoc N1, N3, N5, N6, N8, S) and the pointers to it.
11. T4: the far/main plateau to 194 `lambda` rests on one width per amplitude (24 `lambda`; 12 at 4); width dependence at distance is
    addendum J9.

## 5. Conclusion

1. **Confirmatory.** At every `sigma^2` in {0.25, 1, 2.25, 4}, the plane-based transverse growth over `[25, 94] lambda` (displacement
   basis, `alpha_TP` and `alpha_TAP`) exceeds the first-order window value `7.992e-5 sigma^2 lambda` after the discretization, width and
   integration errors are bounded; Holm rejects all four robust-sign hypotheses at family-wise 5 %. Class `physical (restricted)`,
   within this model, medium, width range and travel window, under the stated transfer assumptions. Estimates: `alpha_TP` = 5.26e-4
   [5.11e-4, 5.41e-4], 7.22e-3 [7.06e-3, 7.38e-3], 2.87e-2 [2.82e-2, 2.92e-2], 6.84e-2 [6.65e-2, 7.04e-2] `lambda`, with budgets of
   13 %, 7 %, 10 % and 16 % of the estimate.
2. **Confirmatory.** `alpha_LP` of the consumer and of Par2 are equivalent within 10 % at 0.25, 1 and 2.25 (ratios 1.004, 1.007,
   1.017). At 4 equivalence is not established (ratio 1.188 [1.159, 1.228]); the offset is on the Par2 side of the Pollock bridge, not in
   the consumer's velocity representation. Until the owner decides on this attribution, SF-43 is not `done`.
3. **Descriptive.** The transverse growth is the same, within 0.5 %, for three trackers that share the field and the Darcy solution, so
   it is not an artefact of the consumer's spline representation or integrator; the shared discretization is bounded by `E_h` (at most
   7.5 % of the estimate). No decay of the transverse growth rate is seen between `[25, 94]` and `[94, 194] lambda` (one width per
   amplitude on the distance arms). Generator class: in the primary (displacement) basis, Par2 on SF-18 fields gives 1.05 [0.997, 1.10]
   (unadjusted p = 0.13) times the transverse growth it gives on random-spectral fields at `sigma^2 = 1`, and 1.05 [0.986, 1.11]
   (p = 0.23) at 0.25 (G7, post hoc N1 of 3.4): not distinguished from 1, both far above `c`. The frozen G7 block's 1.48 [1.11, 2.10]
   at 1 is a position-basis figure (deviation D8), kept as a secondary one. The confirmed values are values of this model with the
   SF-18 generator.
4. Nothing here bears on `t -> infinity`. A positive finite-window `alpha_T` beyond first order is now a confirmed property of this
   model's discrete Darcy flow at these widths and distances; it is not a demonstration that the asymptotic transverse dispersivity is
   positive, and it does not refute Lester et al.'s asymptotic claim, which concerns `t -> infinity` and bounded label fluctuations
   (`docs/theory/lester-2023-key-claims.md`).
5. **Exploratory (addendum 1; descriptive, outside the Holm family; 3.5).** Par2 at `sigma^2 = 4`, J1 at n = 60 (PRIMARY): class
   `horizonte dominante` (`r_H` = 1.104 [1.066, 1.16] for the horizon factor `hz` 3.6 -> 10.8; `r_D` = 1.017 [1.000, 1.037] for
   `dt/2` alone; residual `q_H3` = 1.045 [1.011, 1.096], not attributed by the rule; `q_DH3` = 1.004 [0.9725, 1.041]). The class is
   fragile: `r_H` clears the 1.10 threshold by 0.004 on the point estimate and its interval contains 1.10, which triggered the
   pre-registered J8 extension. At n = 100 (SECONDARY; +40 seeds) the class is the same, `horizonte dominante`, with the same fragility
   (`r_H` = 1.105 [1.079, 1.14], 0.005 above the threshold; `r_D` = 1.008 [0.9962, 1.022]; `q_H3` = 1.029 [1.004, 1.062], not
   attributed; `q_DH3` = 0.9995 [0.9764, 1.024]). J3: `lambda/h = 5` equivalent to `lambda/h` 10 (`m_main` 0.9956, `m_far` 0.9951).
   J9: the far/main plateau is robust to the width at `lambda/h` = 5 (`F48` = 0.9978 [0.9572, 1.042], `F48/F5` = 0.9779
   [0.9192, 1.040]: width 24 -> 48 `lambda` within +-10 %). All descriptive; acceptance D at `sigma^2 = 4` is the owner's decision,
   with the J1 class as an input.

## 6. Caveats

- **Model**: Gaussian covariance, scalar isotropic `K`, medium (b) (Dirichlet `x` faces, periodic transverse faces), SF-18 periodic
  normalized generator (periodic in `x` with the box length), purely advective transport; widths 24-48 `lambda` (12-24 at 4); window
  `[25, 94] lambda` from the injection plane.
- **Generator dependence** (proposed wording; owner decision): the G7 generator factor compares SF-18 fields with Par2's
  random-spectral fields (10 000 modes), descriptive, 60 seeds per generator, Par2 only. In the primary (displacement) basis the factor
  on `alpha_TP` is 1.05 [0.997, 1.10] at `sigma^2 = 1` and 1.05 [0.986, 1.11] at 0.25: no generator effect is distinguished from 1,
  and an effect of up to about 10 % is not excluded. In the position basis, which the frozen G7 block used (D8), the factor at 1 is
  1.48 [1.11, 2.10]; the dependence of that figure on the basis is not explained here. Which generator better represents the target
  Gaussian field at these box sizes is not established here (open question for the owner).
- **Transfer assumptions** [prereg 5.2]: `E_W` measured at `lambda/h` 12 (2.25) and 16 (4) and transferred to production 16 and 32; the
  residual beyond the widest width; the outlet layer (about `W/2 pi`) inside the 5-`lambda` buffer. The box length is not budgeted (3.3,
  item 3).
- **Qualification assumptions**: coverage qualified at a smaller size carries over to larger sizes; the Gate 5 `sigma^2 = 4` moments
  exceed the qualified family (4.2).
- **Shared components**: the three trackers share the SF-18 field and the Darcy discretization; their agreement tests tracking, not
  the field. The `E_h` ladders refine the field sampling, the Darcy solve and the spline together.
- **`sigma^2 = 4`**: `alpha_LP` is a slow-tail, non-Fickian, pre-asymptotic quantity (far/main 1.06), with frequent backflow (15.5 %
  of particles) and no backflow-free sensitivity (D2); its Gate 5 comparison is open (3.2).
- **Not computed**: the B&dD eq. 6 fit (D5), `alpha_LP` budgets (D6).
- **Pending**: the human review's independent reduction of at least three realizations.

## 7. Open items for the human review

1. **Acceptance D at `sigma^2 = 4`**: accept the pre-registered decomposition (offset on the Par2 side; consumer representation not
   implicated) as the attribution, or use the discriminating Par2 follow-up (`dt/2`, longer horizon on a subset of `s4_g5p`), which the
   owner scheduled on 2026-11-04 as addendum 1 (`docs/decisions/2026-11-04-sf43-addendum-1.md`); acceptance D stays the owner's
   decision, with the follow-up as input. The follow-up has run: the J1 class is `horizonte dominante` at n = 60 (primary; fragile on
   `r_H`, 3.5) and also at n = 100 (secondary, with the same fragility: `r_H` = 1.105 [1.079, 1.14]); `gate5:4` is not recomputed.
2. **Acceptance F**: independent reduction of at least three realizations from the retained arrays, without importing the analysis
   package. Retained on the host (`~/sf43_campaign/segments/<arm>/...`): for each production arm the first five seeds and the seeds at
   0.4, 0.7 and 1.0 of `n_conf` (e.g. `s1_prod`: 432001-432005, 432080, 432140, 432200; `s4_prod`: 434001-434005, 434080, 434140, 434200);
   the full plane arrays of every production realization are kept.
3. Deviations D5-D8 and the audit findings (4.2).
4. The generator caveat (G7, 3.2 and 3.4 N1; wording proposed in section 6): record it as a model-scope caveat in the primary-basis
   form, or schedule a comparison of generators (not in SF-43's scope; the decision record lists the generator class as a model
   restriction).
5. **Addendum retention (AD3, 3.5).** The `j9_w48` arm kept the raw Pollock arrays of 122 realizations (about 54 GB on the host) and
   the J8 arms kept 22, 22 and 32 realizations (`j8_H3`, `j8_D`, `j8_DH3`; 13 GB in `~/sf43_addendum_j8`), because the engine runs no
   final prune: prune them with the engine's verified pruner (reduced tables are complete) or keep them. No deletion has been done
   without the owner.

## 8. Artifact layout (`A/results/`)

- `analysis/summary.json` (sha256 `c5c543d5c0b4614a4f9d081e6bfaa368b2e0a03d11dc724df8ad128b200a718c`) and `analysis/figures/`
  (`estimates.png`, `budget.png`, `gate5.png`, `positive_control.png`; made in the job, H18).
- `REPORT.txt`, `ledger.jsonl` (solver-verdict and failure ledger, 7985 records), `DONE`.
- `recompute.txt`, `reconcile.txt` (the independent checks), `logs/` (`sf43-campaign.log` job log, `analysis.log`, `recompute.log`,
  `reconcile.log`, `exec-blinded.log`).
- `reduced/tables/<arm>.tar.xz` (per-realization estimator tables and reduction manifests of every arm, both codes) and
  `reduced/planes/<arm>.tar.xz` (per-plane moment tables), one archive per engine arm [H21]; `reduced/freeze.reduced.sha256` (sha256 of
  every reduced file at the freeze; manifest sha256 `250395ed5319052ddf1e5338e0d081d60610a41e85e1f66252b828daeb492dd6`) and
  `SHA256SUMS` (the archives). Extract with `tar xJf <archive>`; `sha256sum -c` against the freeze manifest from the extraction root.
- On the host only: segment raw outputs (about 70 GB: plane arrays of every production realization, the retained subsets).

Addendum 1 (`A/addendum1-results/`, section 3.5; its own `README.md` and `SHA256SUMS`): `analysis/addendum_summary.{json,md}`
(sha256 `aa85b41ca0d968d36d80d4a9c0711a36e94eb6f04e632f953351e107c60d2dee`, `56f860c5...`) and `analysis/reproduction.json`;
`REPORT.txt`, `ledger.jsonl`, `DONE`; `logs/` (`sf43add-prod.log`, `sf43add-s0.log`, `exec-blinded.log`, `analysis.log`,
`recompute.log`, `reconcile.log`); `reduced/tables/<arm>.tar.xz`, `reduced/planes/<arm>.tar.xz` (one per addendum arm, same member layout
as above) and `reduced/freeze.reduced.sha256` (225 files; manifest sha256 `63bb2def4238224aae9d460ba2a75aad3b05ea8769e0819a2c3b0db9d07f0dd6`;
its paths are relative to the campaign root, so run `sha256sum -c` from the parent of the extracted `reduced/`). `j8/` has the same layout
for the J8 extension (job `sf43add-j8`; its `analysis/addendum_summary.{json,md}` adds the `J1 (n100)` block and has sha256
`ab5b816f857dac710a7b7c38b696892515e1cb18c81c2dace4f387187c939201` / `0616d047...`; 36 reduced files, freeze manifest sha256
`9e0d3d037c202cace16ed522c3a6a39b3c5f76941cd76807557c0a5f25d516e3`; `recompute.txt`, `reconcile.txt`, `logs/exec-blinded.log` with 29
lines of which the first 27 are the frozen version). One `addendum1-results/SHA256SUMS` covers every file including `j8/`. The raw
segment outputs (about 54 GB of unpruned `j9_w48` arrays and 13 GB of J8 arrays among them) stay on the host.

Re-running the analysis from the committed tables (CPU; not re-run for this note): extract the archives into `<C>/reduced/` (the
archives contain `<arm>/<segment>/...`), copy `ledger.jsonl` to `<C>/`, then `bash A/scripts/analyse_sf43.sh <C> build/v100-release`
from the repository root; every bootstrap stream is seeded with 20261103, so the same inputs and library versions should give the same
`summary.json`.
