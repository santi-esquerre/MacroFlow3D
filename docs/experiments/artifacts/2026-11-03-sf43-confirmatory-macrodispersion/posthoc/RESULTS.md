# SF-43 addendum 1: post hoc analyses N1, N3, N5, N6, N8 and the sigma^2 scaling table (node A1-POSTHOC)

**Everything in this file is post hoc (2026-11-04)**: the analyses were decided after the frozen SF-43 results were known (decision
`docs/decisions/2026-11-04-sf43-addendum-1.md`, item 4). They are descriptive, outside the confirmatory Holm family (`m = 8`) and change no
pre-registered result, interval, class or decision. Base commit `33acc6fa`. No GPU, no remote host: only the committed frozen data.
`A` = `docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion`. Scripts: `A/scripts/posthoc/`; outputs: `A/posthoc/<item>/`.
Numbers have 3 significant digits; full precision is in the JSON/CSV of each item. The "expected" values are those of an unaudited preview
(`.claude/orchestration/SF-43/preview-2026-11-04/`); each disagreement is stated, none was adjusted.

## Frozen inputs (all read-only; their sha256 are in `SHA256SUMS`)

`A/prereg.json`, `A/results/ledger.jsonl` (7985 records), `A/results/reduced/tables/*.tar.xz` and `A/results/reduced/planes/*.tar.xz` (44 arms
each), `A/results/reduced/freeze.reduced.sha256`, `A/results/reduced/SHA256SUMS`, `A/results/analysis/summary.json` (sha256 `c5c543d5...`),
`A/coverage/summary.json`, `A/coverage/qualification.json`; the frozen packages `A/scripts/sf43ana`, `sf43cov` and SF-42 `mcstat`, `sf42red`
imported unmodified. The archives were extracted to a scratch directory outside the repository (`scripts/posthoc/build_scratch.py`), and the
2409 extracted files were checked against `freeze.reduced.sha256`: 2409 present, 0 mismatched, 0 extra.

Commands (from the repository root unless noted; `SCR` = scratch directory outside the repository):

```text
python3 A/scripts/posthoc/build_scratch.py SCR                    # extract + verify against freeze.reduced.sha256
bash    A/scripts/posthoc/run_n1.sh SCR                           # N1: frozen sf43ana with prereg_posthoc_g7.json (about 3 min)
python3 A/scripts/posthoc/n1_compare.py SCR; python3 A/scripts/posthoc/n1_effects.py
python3 A/scripts/posthoc/n3_positive_control.py SCR
python3 A/scripts/posthoc/n5_alpha_LP.py SCR
python3 A/scripts/posthoc/n6_local_alphaT.py SCR
python3 A/scripts/posthoc/n8_par2_completeness.py SCR
python3 A/scripts/posthoc/s_scaling.py SCR
(cd A/scripts/posthoc && python3 -W error -m unittest test_posthoc)   # 4 tests
python3 A/scripts/posthoc/make_sha256sums.py                      # SHA256SUMS
```

## N1. G7 in the primary (displacement) basis (post hoc (2026-11-04); deviation D8)

**Method.** `A/scripts/posthoc/prereg_posthoc_g7.json` is `A/prereg.json` with only `g7.responses` changed to
`["alpha_TP:disp", "alpha_TAP:disp", "alpha_TP", "alpha_LP"]` (asserted by `test_posthoc`: the only differing top-level key is `g7`, the only
differing key inside it is `responses`). The frozen `sf43ana` was run exactly as `A/scripts/analyse_sf43.sh` runs it in production
(`python3 -W default -m sf43ana run --prereg ... --campaign C --reduced C/reduced --gate5 auto --out C/analysis`; coverage summary and
qualification as named in the pre-registration; then `check-finite`: `STRUCTURE_OK`, `warnings_count=0 errors_count=0`) on the scratch campaign
layout (ledger + tables + planes). Output: `A/posthoc/n1/summary.json`; `compare.json`; effects in `g7_effects.csv` (all seven effects) and
`g7_twins.csv`.

**Fidelity to the committed `results/analysis/summary.json`** (max relative deviation over the numeric leaves of each block; non-numeric
leaves, i.e. strings, flags, classes, decisions: 0 mismatches in every block):

| block | max rel. deviation | block | max rel. deviation |
|---|---|---|---|
| basis | 8.6e-16 | gate5 (all leaves) | 1.3e-5 |
| budgets | 3.1e-13 | gate5 power entries (Monte Carlo, 48 leaves) | 1.3e-5 (limit 1e-4) |
| estimands | 1.1e-15 | gate5 without the power entries | 5.8e-14 |
| failures | 8.2e-10 | pollock | 2.3e-15 |
| family | 2.2e-14 | positive_control | 1.1e-15 (R identical) |
| gate4 | 3.0e-15 | precision | 1.1e-15 |
| gate4_power | 0 | secondaries | 7.0e-9 |

Acceptance (<= 1e-9 relative; Monte-Carlo Gate 5 power entries <= 1e-4): met by every block except two leaves of `secondaries`:
`time_window/s1_prod/bdd_window/predicted/sd` (7.0e-9 relative) and `time_window/s225_dist/bdd_window/predicted/sd` (1.9e-9). Both are
absolute values of order 1e-9 (7.33e-10 against 7.33e-10 to 8 digits: floating-point noise of a quantity that is zero to that precision), so
they exceed the relative tolerance only because the quantity is itself ~1e-9; no other leaf of the block exceeds 1e-9. Reported, not
adjusted. (`failures` 8.2e-10 relative is on a value of 1.8e-4.)
`process` differs only in `loader.ledger_records` (7985 now; 7983 when the analysis ran: the ledger gained its `analysis ok` and `final DONE`
records afterwards), `prereg_effective.g7.responses` and `prereg_sha256` (`411d504c...` here, `9b92a524...` committed). `g7` block:
the position-basis entries that exist in the committed summary (`twins` for `alpha_LP`, `alpha_TP` at the four amplitudes; `factorial` for
`alpha_LP`, `alpha_TP`) equal the committed ones to <= 3.3e-12 relative (the frozen run used the position column). The position-basis
`alpha_TP` factorial at `sigma^2 = 0.25` is "not analysed: cell means must be positive for effects on the log scale" in both runs.

**Effects (ratio, 90 % interval, unadjusted p; factor G generator SF-18 vs random spectral, I injection plane vs window, GI interaction).**
Displacement basis (`:disp`) is the primary basis of the pre-registration; the position basis is the one the frozen G7 used (D8).

| sigma^2 | response (basis) | G | I | GI |
|---|---|---|---|---|
| 0.25 | `alpha_TP` (disp) | 1.05 [0.986, 1.11] p=0.225 | 1.04 [1.02, 1.06] p=0.0017 | 0.975 [0.956, 0.994] p=0.043 |
| 0.25 | `alpha_TAP` (disp) | 1.05 [0.986, 1.11] p=0.225 | 1.04 [1.02, 1.06] p=0.0017 | 0.975 [0.956, 0.994] p=0.043 |
| 0.25 | `alpha_TP` (position) | not analysed (non-positive cell mean) | | |
| 0.25 | `alpha_LP` | 0.965 [0.935, 0.999] p=0.086 | 0.973 [0.962, 0.984] p=1.1e-4 | 1.03 [1.02, 1.04] p=4.5e-5 |
| 1 | `alpha_TP` (disp) | 1.05 [0.997, 1.10] p=0.130 | 1.01 [1.00, 1.03] p=0.120 | 1.00 [0.988, 1.02] p=0.766 |
| 1 | `alpha_TAP` (disp) | 1.05 [0.997, 1.10] p=0.130 | 1.01 [1.00, 1.03] p=0.120 | 1.00 [0.988, 1.02] p=0.766 |
| 1 | `alpha_TP` (position) | 1.48 [1.11, 2.10] p=0.041 | 1.06 [0.966, 1.16] p=0.295 | 0.975 [0.889, 1.07] p=0.786 |
| 1 | `alpha_LP` | 1.00 [0.968, 1.04] p=0.928 | 0.999 [0.989, 1.01] p=0.933 | 1.00 [0.994, 1.02] p=0.464 |

(`alpha_TP:disp` and `alpha_TAP:disp` of Par2 coincide to 5 digits in G7; the intervals are the frozen `mcstat.splitplot.analyze` ones; the
p-values are unadjusted and belong to no family; the ones of I and GI at 0.25 are listed because they are in the output, not because a
claim rests on them. The alias structure `I = GIDR` is in the JSON.) The twin decomposition (consumer / Pollock / Par2) with the new
responses is in `g7_twins.csv`; the displacement-basis `alpha_TP` contrasts at `sigma^2 = 4` are consumer/Par2 0.997 [0.993, 1.001],
Pollock/Par2 0.996 [0.992, 1.000].

**Agreement with the preview.** All the expected values reproduce: G at `sigma^2 = 1` displacement 1.046 [0.997, 1.098] p=0.13; at 0.25
1.049 [0.986, 1.114]; position at 1: 1.482 [1.106, 2.105] p=0.041; `alpha_LP` at 0.25 G 0.965 [0.935, 0.999], GI 1.029. No disagreement.

## N3. Positive control by plane, `pc_s004`, sigma^2 = 0.04 (post hoc (2026-11-04))

**Method.** Frozen code of `sf43ana.positive_control` (`read_planes`, `x22_box`: box first-order prediction for W = 24 lambda, L = 104 lambda,
`kmax` 12) on the planes tables of `pc_s004` (30 realizations, 70 planes d = 25..94, consumer, displacement basis, the only basis for which a
prediction exists). Variants: weights flux (primary) / unif, quantity `second_moment` (what `R` and X22 are) / `centred` (what `alpha_TP`
is the slope of). Excess = measured - predicted; OLS over the window planes; 90 % realization-bootstrap percentile intervals (B 2000, seed
20261104; the mean profile of each resample is re-fitted). Reproduction: `R` (flux, second moment) = 1.14474 identical to the frozen value;
interval here [1.122, 1.168] (frozen [1.122, 1.168]; another bootstrap seed). Output `n3/n3_positive_control.json`.

| flux weights | second moment | centred |
|---|---|---|
| ratio measured / predicted at d = 25 [90 %] | 1.04 [0.991, 1.08] | 1.02 [0.977, 1.07] |
| ratio at d = 94 [90 %] | 1.29 [1.24, 1.34] | 1.27 [1.23, 1.33] |
| ratio of the OLS line of the excess (+ prediction), d = 25 -> 94 | 1.05 -> 1.25 | 1.03 -> 1.24 |
| excess slope over [25, 94] (lambda^2 per lambda) | 3.36e-5 [2.54e-5, 4.24e-5] | 3.33e-5 [2.53e-5, 4.27e-5] |
| share of bootstrap slopes > 0 | 1.00 | 1.00 |
| d where the OLS excess line crosses zero (window [25, 94]) | 8.5 | 12.9 |
| same over 8 trimmed windows (25-94, 30-90, 35-85, 40-80, 25-60, 60-94, 25-80, 40-94) | 0.5 to 11.5 | 6.6 to 15.6 |
| bootstrap percentile 90 % interval of that d (full window) | [-8.8, 19.9] | [-2.8, 23.9] |

Window slopes of the measured (centred) profile 3.05e-5 and of the prediction -2.80e-6, hence the excess slope 3.33e-5; `alpha_T`
equivalent (half the slope) of the measured profile 1.53e-5. The box prediction falls by 4.55 % between d = 80 and d = 94. Stability under
trimming: the excess slope stays between 2.67e-5 and 3.55e-5 (second moment) and 2.68e-5 and 3.52e-5 (centred) over the 8 windows; the 90 %
interval excludes zero in all of them (shortest window [25, 60]: [3.9e-6, 5.0e-5] and [5.2e-6, 4.9e-5]). Unif weights give the same to 1 %
(`n3_positive_control.json`).

**Reading (wording fixed by the brief).** The measured transverse displacement variance grows faster with distance than the first-order box
prediction, which is flat to slightly decreasing in the window; the excess grows about linearly with d, consistent with a second-order
`alpha_T` visible already at `sigma^2 = 0.04`. The data cannot separate a `sigma^2` from a `sigma^4` amplitude (two amplitudes only, the
scaling table below).

**Disagreement with the preview.** Slope 3.33e-5 [2.53e-5, 4.27e-5] (centred) agrees with 3.33e-5 [2.52e-5, 4.22e-5]; zero crossing 8.5-12.9
(window 25-94) is inside "8-13 lambda", but over the trimmed windows it ranges 0.5-15.6 and its bootstrap interval spans roughly -9 to +24
lambda: the extrapolation to zero is not well determined. The ratio from first to last plane is 1.02 -> 1.27 (raw) or 1.03 -> 1.24
(OLS line), not "1.02 -> about 1.22": I cannot reproduce 1.22 with any quantity computed here, so it is not reported. The box-prediction
fall of 4.6 % between 80 and 94 lambda is confirmed (4.55 %).

## N5. `alpha_LP` (post hoc (2026-11-04))

### (a) G5.4 budget rules applied to `alpha_LP`

**Method.** The frozen `sf43ana.gate4.RobustTest` / `design_from_arms` (E_h with its fallback and gate, E_W, E_WxH, E_int; levels 0.05 and
0.05/8 = 0.00625 as the `report_levels` of the pre-registration; labels and seeds of the frozen run), applied to `alpha_LP` at the four
amplitudes, displacement basis (the position basis table agrees within 0.01 percentage points). Entries: enforced bound as % of the
production mean (N = 200 at each amplitude). Check: the same code on `alpha_TP` and `alpha_TAP` reproduces the committed `budgets` of
`summary.json` to <= 1.7e-13 relative (L(a/2), RL, every enforced component). Output `n5/n5a_budget_table.json`.

| sigma^2 | mean `alpha_LP` | E_h (a=.05 / .00625) | E_W | E_WxH | E_int | sum | dominant |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.233 | 0.0850 / 0.0903 % | 3.49 / 4.49 % | n/a | 0.00 / 0.00 % | 3.57 / 4.58 % | E_W |
| 1 | 1.09 | 0.457 / 0.537 % | 3.22 / 4.05 % | n/a | 0.00 / 0.00 % | 3.68 / 4.59 % | E_W |
| 2.25 | 3.26 | 19.6 / 20.8 % | 2.65 / 3.36 % | 1.97 / 2.48 % | 0.00 / 0.00 % | 24.3 / 26.6 % | E_h |
| 4 | 9.44 | 107 / 116 % | 5.79 / 7.34 % | 12.5 / 14.8 % | 0.01 / 0.01 % | 125 / 138 % | E_h |

**When the fallback fires (E_h).** Not at 0.25 and 1: the order branch is used (order estimate 1.97 and 2.25; `p_use` 1.94/1.93 and
1.63/1.45) and the gate passes. At 2.25: the lower order limit of the ladder (lambda/h 4, 8, 16 means 3.41, 3.28, 3.25) is not positive
(`p_L = -inf`), so the fallback applies: 0.641 (19.6 % of the mean) instead of an order-branch value of 0.368 (11.3 %). At 4: the ladder
(lambda/h 4, 8, 16, 32: 11.2, 9.32, 9.09, 9.40) is non-monotone, no order is estimable (`order_hat` NaN), the `orders_agree` gate fails, and
E_h is the fallback 10.1 (107 %; order-branch value 4.77, 50.5 %). Expected (preview) 3.6 / 3.7 / 24 / 125 % of the estimate: the "sum"
column at a = 0.05 gives 3.57 / 3.68 / 24.3 / 125 %, agreement.

**Wording.** `alpha_LP` at `sigma^2 = 4` is not grid-converged by this rule; it is not presented as a value. At 2.25 the bound is dominated by the
fallback of E_h, not by an observed non-convergence.

### (b) `alpha_LP(s4_l16) / alpha_LP(s4_prod)` (lambda/h 16 vs 32, sigma^2 = 4)

**Method.** Production selection (consumer, plane, flux weights, per-plane set), common seeds (n = 200; `s4_l16` has 250, `s4_prod` 200),
ratio of means, `mcstat.gate5.gate5_paired` delta-method Student 90 % interval and bootstrap-t (B 2000, seed 20261104), leave-one-out change
of the ratio per seed. `alpha_LP` is identical in both bases. Output `n5/n5b_l16_vs_prod.json`.

| quantity | value |
|---|---|
| mean `alpha_LP` l16 / prod | 9.07 / 9.44 (sd 1.85 / 4.98; correlation 0.102) |
| ratio of means l16 / prod | 0.961 |
| Student 90 % | [0.902, 1.02] |
| bootstrap-t 90 % | [0.847, 1.01] |
| per-seed ratio L_r/P_r (min / median / max) | 0.128 / 0.996 / 2.21 |
| leave-one-out ratio range | 0.956 to 0.994 |

Top 5 influential seeds (ratio without the seed; change of the ratio): 434111 (l16 9.21, prod 72.0): 0.994, +0.0330 (+3.4 %); 434151
(8.29, 30.1): 0.972, +0.0111; 434077 (18.4, 8.32): 0.956, -0.00552; 434144 (21.0, 11.9): 0.956, -0.00511; 434109 (14.7, 7.25): 0.957,
-0.00409. One production seed (434111) moves the ratio by 3.4 %, more than the next four together; the production sample has a heavy upper
tail (sd 4.98 against 1.85) that the l16 sample does not show at the same seeds. Both intervals contain 1; given that tail and the budget of
(a), this does not establish convergence between lambda/h 16 and 32.

## N6. Local `alpha_T` against distance (post hoc (2026-11-04))

**Method.** `s1_dist`, `s225_dist`, `s4_dist` (100 realizations each), consumer, flux weights, displacement basis, per-plane moments of the
170 planes d = 25..194 (`*.planes.csv`); sigma_T^2(d) = (var_x2 + var_x3)/2 / ell^2 in lambda^2 per realization; `alpha_TP` of a window = half the
OLS slope of sigma_T^2 on d. Reproduction: the window [25, 94] reproduces the frozen table `alpha_TP` per seed to <= 6.6e-16 relative for the
three arms. Windows: 9 consecutive windows of 17 lambda, [25,42] to [161,178], end planes included in both neighbours (as the frozen main window
[25, 94]); the main reference is [25, 94] (7.19e-3, 2.86e-2, 6.72e-2). 90 % realization-bootstrap percentile intervals (B 2000, seed
20261104); the ratio local/main is re-formed in each resample. Outputs `n6/n6_<arm>.json` and `n6_<arm>_curve.csv` (ensemble sigma_T^2(d) with
SE for every plane).

Local `alpha_TP` / main [25, 94] (90 % interval of the ratio in `n6_<arm>.json`):

| window | s1_dist | s225_dist | s4_dist |
|---|---|---|---|
| 25-42 | 1.02 | 1.06 | 0.931 |
| 42-59 | 0.943 | 0.971 | 1.00 |
| 59-76 | 1.08 | 1.02 | 1.03 |
| 76-93 | 0.941 | 1.05 | 1.03 |
| 93-110 | 0.960 | 1.02 | 0.913 |
| 110-127 | 0.982 | 1.00 | 1.05 |
| 127-144 | 1.05 | 1.03 | 1.06 |
| 144-161 | 1.01 | 1.09 | 0.936 |
| 161-178 | 1.00 | 0.996 | 0.906 |
| range | 0.941-1.08 | 0.971-1.09 | 0.906-1.06 |

The 90 % interval half-widths of the ratios are about 0.05-0.12. Four of the 27 intervals exclude 1 (s1 59-76 [1.01, 1.15]; s225 25-42
[1.01, 1.12] and 144-161 [1.01, 1.17]; s4 161-178 [0.832, 0.986]); four of 27 is about the number expected from 27 unadjusted 90 %
intervals, none was selected, and no pattern with distance is present in the same direction at the three amplitudes.

Linear trend of the 9 local values against the window centres, % of the main value per 100 lambda [90 % interval] ("windows", primary of
this item) and the trend of the global quadratic fit over d >= 25 (the preview's definition: d alpha_T/dd = quadratic coefficient):

| arm | windows | quadratic (preview definition) | expected (preview) |
|---|---|---|---|
| s1_dist | +1.25 % [-4.5, +7.2] | +4.25 % [-0.82, +9.6] | +4.2 % [-0.8, +9.3] |
| s225_dist | +0.72 % [-3.3, +5.1] | +3.14 % [-0.58, +7.1] | +3.1 % [-0.7, +7.0] |
| s4_dist | -2.18 % [-8.0, +3.9] | -2.01 % [-7.8, +3.9] | -2.0 % [-8.0, +4.0] |

All six intervals contain 0. The quadratic definition reproduces the preview (differences in the upper limits from the bootstrap seed); the
window-based definition gives smaller trends for the two lower amplitudes because it ignores the shape within windows and the planes
beyond 178. **Disagreement with the preview:** the local/main range "0.94-1.08" holds for s1 (0.941-1.08) and s225 (0.971-1.09, upper
1.09), but not for s4 (0.906-1.06; windows 144-161 and 161-178 are 0.936 and 0.906); the preview window set is unknown.

**Ensemble sigma_T^2(d)** (lambda^2, mean +- SE): s1 0.652 +- 0.008 (d=25), 1.63 +- 0.016 (94), 3.08 +- 0.030 (194); s225 2.06 +- 0.022,
6.04 +- 0.051, 12.0 +- 0.094; s4 4.08 +- 0.093, 13.4 +- 0.21, 26.6 +- 0.34.

**Descriptive saturation fit** sigma_T^2(d) = s0^2 + s_psi^2 (1 - exp(-d/zeta)) of the ensemble-mean curve over all 170 planes (variable
projection over a zeta grid [1, 1e5] lambda with non-negative linear coefficients, unweighted: plane values are strongly correlated; rms
residual 0.011, 0.031, 0.067 lambda^2). Lower bound on zeta: one-sided 95 % = 5 % quantile of zeta from 2000 realization-bootstrap
refits.

| arm | zeta of the fit | zeta lower bound (95 %, one-sided) | share of refits at the grid end 1e5 |
|---|---|---|---|
| s1_dist | at the grid end (1e5; linear in the data) | 1.2e4 lambda | 0.92 |
| s225_dist | at the grid end (1e5) | 1.8e4 lambda | 0.92 |
| s4_dist | 5.0e3 lambda | 1.3e3 lambda | 0.29 |

Over the 170 lambda of data, the growth of sigma_T^2 does not show a saturation scale within the window at any amplitude; zeta is bounded
below by values far above the plane range. The bound is a percentile bootstrap bound of a parameter that is non-identifiable towards
infinity (it measures how often resamples look linear), not a statement about any asymptotic regime, and **no statement about Lester's
asymptotic claim is made**.

## N8. Par2 completeness strata at sigma^2 = 4 (post hoc (2026-11-04); selection, not causal)

**Method.** Gate 5 pairs of the pre-registration at sigma^2 = 4: Pollock twin `s4_g5pol` (source `s4_l16`, uniform weights) against Par2
`s4_g5p`, `alpha_LP` (column `alpha_LP`, main window), paired by seed (250 pairs). Stratum = `n_traced - n_last` of the Par2 realization
(Par2 particles that did not cross the last window plane; `n_traced` is 10000 in every realization). Per stratum: ratio of means
Pollock/Par2, `gate5_paired` Student 90 % interval and bootstrap-t (B 2000, seed 20261104). Reproduction: all 250 pairs give 1.164
[1.131, 1.198], identical to the frozen decomposition. Output `n8/n8_par2_completeness.json`.

| missing Par2 particles | n pairs | mean Par2 completeness | Pollock/Par2 | Student 90 % | bootstrap-t 90 % |
|---|---|---|---|---|---|
| 0 | 25 | 1 | 1.05 | [1.02, 1.08] | [1.02, 1.08] |
| 1-2 | 97 | 0.99985 | 1.13 | [1.07, 1.20] | [1.08, 1.30] |
| 3-9 | 86 | 0.99956 | 1.15 | [1.12, 1.19] | [1.12, 1.20] |
| >= 10 | 42 | 0.997 | 1.36 | [1.26, 1.48] | [1.26, 1.48] |

Across the 250 pairs corr(log(Q/P), 1 - c) = 0.490 (Pearson, p 1.8e-16) and 0.349 (Spearman, p 1.5e-8), with c = n_last/n_traced. This is a
selection by a Par2 property that correlates with the outcome (a slow-tail realization both loses crossings and has a larger `alpha_LP`), **not
a causal estimate**: the strata are not randomized and the residual in the complete stratum (about 5 %, interval [1.02, 1.08]) is not
explained by the horizon. Agreement with the preview: 1.048 [1.018, 1.079] (here Student [1.016, 1.082]), 1.133, 1.153, 1.362 [1.263, 1.469]
(here [1.258, 1.475]), corr 0.49; the interval limits differ in the third digit (the preview's interval method is not stated; mine is the
frozen `gate5_paired`).

## S. Scaling with sigma^2 (post hoc (2026-11-04))

`alpha_TP` (displacement, flux, main window) of the production arms (N = 200, the frozen analysis sets) and of `pc_s004` (N = 30); 90 %
Student intervals; exponents d ln `alpha_TP` / d ln sigma^2 between consecutive amplitudes with delta-method standard errors from the arm
standard errors (independent arms; normal 90 %) and a realization-bootstrap percentile interval (B 20000, seed 20261104). Output
`scaling/scaling.json`.

| sigma^2 | arm | `alpha_TP` | `alpha_TP / sigma^4` [90 %] |
|---|---|---|---|
| 0.04 | `pc_s004` | 1.53e-5 | 9.54e-3 [6.71e-3, 1.24e-2] |
| 0.25 | `s025_prod` | 5.26e-4 | 8.42e-3 [8.18e-3, 8.66e-3] |
| 1 | `s1_prod` | 7.22e-3 | 7.22e-3 [7.06e-3, 7.38e-3] |
| 2.25 | `s225_prod` | 2.87e-2 | 5.67e-3 [5.57e-3, 5.76e-3] |
| 4 | `s4_prod` | 6.84e-2 | 4.28e-3 [4.15e-3, 4.40e-3] |

| from -> to | exponent | se (delta) | 90 % delta | 90 % bootstrap |
|---|---|---|---|---|
| 0.04 -> 0.25 | 1.93 | 0.096 | [1.77, 2.09] | [1.79, 2.10] |
| 0.25 -> 1 | 1.89 | 0.016 | [1.86, 1.92] | [1.86, 1.92] |
| 1 -> 2.25 | 1.70 | 0.021 | [1.67, 1.74] | [1.67, 1.74] |
| 2.25 -> 4 | 1.51 | 0.035 | [1.45, 1.57] | [1.45, 1.57] |

Agreement with the expected 9.5e-3, 8.42e-3, 7.22e-3, 5.67e-3, 4.28e-3 and 1.93, 1.89, 1.70, 1.51. The local exponent decreases steadily
from about 2 at small amplitude to about 1.5 at `sigma^2 = 4`; the 0.04 -> 0.25 exponent is compatible with 2 but its interval is wide (the
control arm has 30 realizations). Descriptive only; no functional form is fitted and nothing here is a claim about the asymptotic `alpha_T`.

## Limits and notes

- Two ~1e-9-sized leaves of `secondaries` exceed the 1e-9 relative figure of the brief (see N1); all other leaves of all blocks are within
  it. No decision, class or string differs.
- The weights/quantity variants of N3 (second moment vs centred) stand in for "per basis": the box prediction exists only for the
  displacement basis, so no position-basis excess was computed.
- N6 windows overlap by their end plane; the 17 lambda width gives 9 windows over 25-178 as specified. The planes 178-194 enter only the
  curve, the quadratic trend and the saturation fit.
- None of the scripts imports anything beyond the frozen packages, numpy and scipy; every random stream is seeded (20261104 in this node).
