# sf43ana: frozen analysis package of SF-43

Purpose: from the campaign ledger, the reduced tables of the engine and the pre-registration JSON, produce `summary.json`: estimands,
error budgets, robust-sign tests, the fixed-family Holm decisions, the Gate 4 classes (decision `2026-10-29-sf39-remediation-sf42-sf43`
G4, G5.1-G5.7, G8 with the pre-activation decisions H2-H6, H13, H15-H17), Gate 5 in Fieller form (H5), the descriptive blocks (G7,
secondaries, failures, positive control) and the standard figures. It imports `mcstat` and `sf42red` of the SF-42 artifact unchanged
(located relative to this directory, or `SF42_SCRIPTS`). stdlib + numpy + scipy (+ optional matplotlib).

```bash
cd docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts
python3 -W error -m unittest discover -s sf43ana/tests -t .        # about 2 min (the null simulations dominate)
python3 -m sf43ana run --prereg P.json --campaign C --reduced R [--gate5 auto|none|G.json] --out O      # O/summary.json (+ O/figures/)
python3 -m sf43ana check-finite --summary O/summary.json     # smoke test: structure and COUNTS only, STRUCTURE_OK | STRUCTURE_FAIL <reasons>
```

`--coverage-summary PATH|none` overrides `gate5.coverage_summary` (the smoke run reads its own coverage output; `none`: Gate 5 descriptive).
`--gate5`: `auto` (default) computes Gate 5 from the pre-registration's `gate5` block (no block: no p-values); `none` supplies no p-value; a
path supplies `{"p": {"<amplitude>": p}}`. A hypothesis without a p-value enters the Holm family with `p = 1` (untestable).
`check-finite` prints per section the numbers of finite / non-finite / other leaves and `STRUCTURE_OK` or `STRUCTURE_FAIL`; it never prints an
estimate, bound, p-value, decision or class name (failures: missing section, an `error` recorded in a descriptive block, the family not
listing `m` hypotheses, a configured amplitude without both transverse estimators, no finite number in a required section, a non-finite
production mean).

## Data contract (one structure for every module)

Inputs are exactly what the campaign engine (`sf43camp`) writes (`loader.py` docstring is the reference):

* **Ledger** `<campaign>/ledger.jsonl` (the engine's records, `sf43camp/lanes.py`): per solve a `seed` record (`arm`, `segment` = segment
  directory name, `seed`, `seed0` = first seed of the unit, `verdict`, `class` ok | unusable_solve, `iterations`, `rel`, `max_div`; the LATEST
  record of (arm, seed) counts), `replacement` (`group`, `seed`, `replaced_by`), `arm_halt`, `reduced` / `unreduced` (`arm`, `segment`, `seeds`: a
  segment listed `unreduced` and not reduced afterwards has its seeds MISSING, reported explicitly: the arm's prefix ends there with the reason
  `unreduced segment ...`, and `process.loader.unreduced` / `failures.unreduced_segments` list them) and `reest` (`arm`, `n_new`). A torn last
  line is ignored. A completed seed of a `recovered` unit has its verdict in its `seed` record as well; a `None` verdict is counted in
  `process.loader.verdict_unknown`. Verdict vocabulary: `ok_verdicts` [converged, fallback]; the fallback sensitivity uses `fallback`.
* **Reduced tables** `<reduced>/<arm>/<segdir>/<arm>__<segdir>__d<lo>-<hi>.csv` (executor naming). THE window rule: the file stem is `d%g-%g` of the
  window bounds (`loader.window_stem`), whatever the semantic label of the window in the pre-registration (`main`, `h1`, `h2`, `far`);
  `reduced_pattern` has the fields `{arm} {segment} {stem} {window} {lo} {hi}`. Rows: `code` consumer | par2 | pollock (the Pollock-bridge plane
  moments of the consumer's realizations, same seed and `hmax_div` label, always the production label for the three-pass arms). Consumer rows
  carry their `seed`; Par2 rows have `seed = nan` and the realization index `r`: the seed is `seed0 + r` (`seed0` from the `seed` records).
  Planes tables `<table minus .csv>.planes.csv` (positive control).

`loader.load_all` returns `({arm id: ArmData}, process info, Ledger)` for EVERY pre-registered arm (both codes); `loader.build_data` gives the
`data` dictionary that `gate5`, `g7`, `secondaries`, `failures`, `positive_control` read (each has `analyse(ctx)`; `ctx.data` is built lazily):

```text
data = {"arms":   {arm id: {"seed": int array (N,) IN PREFIX ORDER, "verdict": str array (N,), "N", "status", "segment": [segdir per seed],
                            "cols": {name: float array (N,)}}},
        "ledger": [{"arm","segment","seed","verdict","class",["replaces"]}],     # class: ok | unusable_solve | halted
        "halted": [arm], "sources": {ledger arm: [arm ids reading its tables]}, "coverage_summary": dict|None}
columns: <est> = position basis, <est>:<basis> any other (alpha_TP:disp), est in alpha_TP, alpha_TAP, alpha_LP, U_r (primary window);
         alpha_LAM:time alpha_Tfixed:time alpha_TAM:time t_end_adv (time family); alpha_TP:expI alpha_LP:expI (consumer expI weights);
         frac_recross frac_backflow alpha_LP:nobackflow (H17, only if the table carries them)
```

**Analysis set** of an arm = the longest COMPLETE prefix of its pre-registered seed order after the group-wide replacements, cut at the arm's
PLANNED `n_conf`, or at `n_new` of the engine's blinded re-estimation (ledger `reest`, never below `n_conf`, never beyond the seeds listed) for an
entry with `reest: true` whose engine arm has such a record (the production arm and its Pollock partner; the Gate 5 twins, the `h_max` passes and
the ladders keep their planned size);
the first incomplete position ends it. Row selection: `selection` (code, family, weights, set), the arm's `code` (Par2: weights `unif`), the arm's
own `selection` override, `hmax_div`. **Production arm and `h_max` passes** (engine contract): the production arm of an amplitude is ONE engine
arm; its first 30 seeds run three passes (rows with `hmax_div` 8 / 16 / 32). In the pre-registration this is two entries sharing
`"source": "<production arm id>"` with `"hmax_div": 8` and `16` (`h8`: its first 30 seeds, `h16`); the production entry itself (same `source`, all `n_conf` seeds) reads
`hmax_div` 8, because the engine's reduced rows carry `hmax_div` 8 for every seed and 16 / 32 only for the first 30. A test covers it.

## Pre-registration schema (`config.validate`; STRICT: unknown keys, unknown arm references and bad shapes fail with exit code 2)

Defaults (`config.DEFAULTS`, echoed in `process.prereg_effective`): `windows` main [25,94], h1 [25,59.5], h2 [59.5,94], far [94,194];
`bases` {primary: disp, secondary: pos}; `levels` {fwer 0.05, interval 0.90, ucl 0.05}; `null_per_sigma2` 7.992e-5 (H3: `c = null_value(a) =
7.992e-5 sigma^2`, or `amplitudes.<a>.null_value`, 0 allowed); `tie_rtol` 1e-9; `numerical_fraction` 0.5; `ok_verdicts` [converged, fallback];
`fallback_verdicts` [fallback]; `reduced_pattern`; `sampling_method` bootstrap-t; `bootstrap` {B_sampling 33599 (B+1 = 33600: exact ranks for
every Holm level 0.05/j and half), B_budget 4000, B_ucl 4000, B_bca 5000, seed};
`eh` {p_range [0.5,2.1], p_cap 2, safety 1.25, fallback_factor 3, gate_level 0.90}; `floors` {production 100, ladder 30, width 30, hmax 20,
gate5 200}. Top-level keys: the defaults, `amplitudes`, `arms`, `family`, `targets`, `first_order`, `report_levels` and the blocks `gate5`,
`g7`, `secondaries`, `failures`, `positive_control`.

```text
arms[]        id, amplitude, group, seeds (confirmatory order: the first n_max seeds of the group's block), reserves, n_conf (planned), n_max
              (default len(seeds)), floor (2 <= floor <= n_conf <= n_max <= len(seeds)), source (default id), hmax_div (-1),
              code (consumer|par2|pollock), reest (false), role, family, selection {code,family,weights,set}
amplitudes.<a>  sigma2, control (true: a control amplitude such as the positive control's 0.04: arms may refer to it, no Gate 4, not in the family), production, components (subset of E_h E_W E_WxH E_int; K = their number), primary (false: out of the family), null_value,
              floors {production,ladder,width,hmax,gate5} (amplitude override), distance_arm,
              ladder {arms: coarse->fine, resolutions, production_index, triples, fallback_levels, gates},
              width {wide_fine, prod_fine[, wide_coarse, prod_coarse]}, hmax {h16, h8}
ladder.gates[]  {"type": "extra_level", "triple": [0,1,2], "arm": "<arm id on the ladder's seeds>", "resolution": 12}     (sigma^2 2.25)
                {"type": "orders_agree", "triples": [[0,1,2],[1,2,3]]}                                                   (sigma^2 4)
```

Example (sigma^2 2.25 with the gate on the lambda/h = 12 Gate 5 consumer arm; sigma^2 4 with the two triples):

```json
"2.25": {"production": "p225", "components": ["E_h","E_W","E_WxH","E_int"],
         "ladder": {"arms": ["l4","l8","p225"], "resolutions": [4,8,16], "production_index": 2,
                    "gates": [{"type": "extra_level", "triple": [0,1,2], "arm": "g5c225", "resolution": 12}], "fallback_levels": [0,1,2]},
         "width": {"wide_fine": "w48_12", "prod_fine": "g5c225", "wide_coarse": "w48_8", "prod_coarse": "l8"}, "hmax": {"h16": "p225_h16", "h8": "p225_h8"}},
"4":    {"production": "p4", "components": ["E_h","E_W","E_WxH","E_int"],
         "ladder": {"arms": ["l4_4","l8_4","l16_4","p4"], "resolutions": [4,8,16,32], "production_index": 3,
                    "gates": [{"type": "orders_agree", "triples": [[0,1,2],[1,2,3]]}]}, "floors": {"production": 100}}
```

### Floors (H13)

A hypothesis is untestable (`p = 1`, estimates descriptive) unless every arm it needs reaches its floor: Gate 4 at an amplitude needs production
>= `floors.production`, the ladder >= `floors.ladder` COMPLETE PAIRED seeds (common prefix of all ladder arms, the gate arm and the
production-width arm of the group), every width cell >= `floors.width`, the h_max passes >= `floors.hmax` paired seeds, plus each arm's own `floor`;
Gate 5 needs >= `gate5.floor` pairs (default `floors.gate5` = 200; 250 at sigma^2 4 via `gate5.amplitudes.<a>.floor`).

### `gate5` (H5; Fieller form; `gate5.py`)

```json
"gate5": {"family_m": 8, "fwer": 0.05, "margin_ratio": 1.10, "level": 0.90, "B": 33599, "seed": 20261103, "key": "alpha_LP", "ref": "t",
          "coverage_summary": "<out of sf43cov>/summary.json", "floor": 200, "fallback_verdicts": ["fallback"],
          "amplitudes": {"0.25": {"consumer": "g5c025", "par2": "g5p025"},
                         "4": {"consumer": "g5c4", "par2": "g5p4", "floor": 250, "j3b": true,
                               "pilot": {"consumer": "pilot_c", "par2": "pilot_p"}, "unmatched": {"consumer": "p4", "par2": "g5p4"}}}}
```

Pairs = the first `n` positions of the two arms' analysis sets in prefix order (the seed orders must agree; the pair prefix stops at the first
non-finite pair). Below `floor` pairs: `p = 1`, descriptive. Procedure = the first of Hall, bootstrap-t, bootstrap-t at half the level that the
coverage summary qualifies at the LARGEST coverage grid point <= the realized `n` (all pairs are used); `j3b: true` (sigma^2 4) also requires the
J3B-bound family (skewness 2.73, excess kurtosis 7.59, both signs). Ratio interval by inverting the tests (Fieller); verdict `agree` /
`disagree` / `inconclusive`; achieved power = Fieller-form power at the realized `n` with the observed SD of `D_hi = L - 1.10 P`, `D_lo = L - P/1.10`
and their correlation (also the delta-method power for continuity); `pair_exclusion_sensitivity` (pairs with a `fallback` verdict in either code
removed); `d_moments` of `D_r` at the margins, and `pilot` moments if pilot arms are passed (they must exist in `data`, e.g. as extra arms).

### `positive_control` (H9; `positive_control.py`)

```json
"positive_control": {"arm": "pc_s004", "sigma2": 0.04, "width_lambda": 24, "length_lambda": 104, "window": [25, 94],
                     "weights": ["flux","unif"], "level": 0.90, "B": 2000, "seed": 20261103, "kmax": 12.0,
                     "R_bounds": [0.8, 1.25], "R_weights": "flux", "expectation": "..."}
```

Compares the per-plane ensemble displacement variance profile of the `sigma^2 = 0.04` arm (planes tables, displacement basis; second moment and
centred) with the first-order box prediction `X22(d)` (port of `firstorder_box.py`, times `sigma^2`): ratios per plane with seed-bootstrap
intervals, the window slope against the box value (difference and ratio with intervals). Descriptive.

**Pre-registered statistic `R`** (block `R` of the output): the mean over the window planes of the measured FLUX-weighted displacement-basis second
moment divided by the same mean of the box prediction, with a 90 % seed-bootstrap interval. `positive_control_ok` = the interval lies within
`R_bounds` [0.8, 1.25]; `status` `ok` | `inconclusive` (the interval straddles a bound) | `failed` (entirely outside). EXPECTATION (pre-registered,
also in the `expectation` string of the pre-registration): `R` within [0.8, 1.25]. Rationale: the prediction is first order in `sigma^2`; the
O(`sigma^2`) nonlinear corrections are about 4 % at 0.04; the discretization (lambda/h = 10) and the Dirichlet inflow / outflow faces of medium (b),
which the prediction neglects, are expected to move the window mean by a few per cent at most. A value outside the bounds indicts the estimand pipeline
or the runner, not the physics; the control is descriptive and never gates the family.

### `pollock` (acceptance B, item 8; `pollock.py`)

```json
"pollock": {"level": 0.90, "B": 2000, "seed": 20261103,
            "contrast": {"s1": {"consumer": "s1_prod", "pollock": "s1_prodpol"}},
            "decomposition": {"s1": {"consumer": "s1_g5c", "pollock": "s1_g5pol", "par2": "s1_g5p"}}}
```

`contrast`: for every amplitude, over the production analysis set (flux weights), every plane estimand (`alpha_TP`, `alpha_TAP`, `alpha_LP`, `U_r`) in both
bases: the paired difference consumer - Pollock (mean, 90 % Student and bootstrap-t intervals) and the ratio of means (paired seed-bootstrap
percentile interval). `decomposition`: on the Gate 5 pairs (uniform weights) the consumer/Pollock, Pollock/Par2 and consumer/Par2 log-ratio
contrasts (`g7.twins`). Reported, outside the error budget and the Holm family. The Pollock entries are arms with `code: pollock`, `source` = the engine
arm, the production label `hmax_div` 8 (the multi-pass realizations carry identical Pollock rows under each label).

### `g7`, `secondaries`, `failures`

```json
"g7": {"level": 0.90, "log_bootstrap": 2000, "responses": ["alpha_LP","alpha_TP"],
       "twins": {"0.25": {"consumer": "g7c","pollock": "g7pol","par2": "g5p025"}},
       "factorial": {"0.25": [{"g": 1, "i": 1, "d": 1, "r": 1, "arm": "g7_cell1"}]}},
"secondaries": {"level": 0.90, "B": 1000, "time_window": {"s4_c": {"amplitude": "s4"}}, "expI": {"s4": {"arm": "p4"}},
                "par2_transverse": {"1": {"par2": "g5p1", "consumer": "g5c1"}}, "bdd_fit": [{"name": "s4", "arm": "snap4", "amplitude": "s4"}],
                "recrossings": {"s4": {"arm": "p4"}}},
"failures": {"max_rate": 0.01, "failure_classes": ["unusable_solve"], "fallback_verdicts": ["fallback"], "q_low": 0.05, "q_high": 0.95}
```

`failures` runs with its defaults when absent. A failure inside a descriptive block is recorded as `{"error": ...}` (stderr `BLOCK_ERROR`), never
fatal; Gate 5 errors propagate. Inputs not written by `sf42red` today and what they need: (the G7 Pollock twin is an arm with `code: pollock`, whose rows `sf42red reduce --pollock` writes), **`bdd_fit`** (columns `curve_t`,
`curve_alpha` of snapshot-time curves, `unavailable` without), **`recrossings`** (H17: columns `frac_recross`, `frac_backflow` and
`alpha_LP:nobackflow` in the reduced table; the reducer does not write them, so the block reports `input absent` and names the columns).

## Statistics (decisions in `gate4.py`, `eh.py`, `holm.py`)

* Robust limit `RL(a) = L(a/2) - sum_k B_k(a/(2K))` (`mcstat.RobustLimit`); `L` = BOOTSTRAP-t lower limit of the production mean (not Student);
  `B_E_W, B_E_WxH, B_E_int` Student contrast bounds; `B_E_h` = `eh.EhBound`.
* Threshold `c` explicit: reject iff `RL(a) > c`, p-value `inf{a : RL(a) > c}` on the mcstat lattice; the data are not shifted. Ties: relative
  tolerance 1e-9 (`gate4.exceeds`, `holm`). Holm decisions evaluate `RL(level) > c` at each Holm level exactly.
* `E_h` (H6): per-triple order range test [0.5, 2.1], `p_use = min(p_L over triples, 2)`, Giles finest branch, fallback over the configured
  levels, and the **gate on the asymptotic range** (`ladder.gates`): a paired seed bootstrap of the ladder means; the gate passes iff 0 lies in the
  `gate_level` (90 %) interval of `a(12) - a_fit(12)` (fit `a_inf + C h^p` on the triple) or of the difference of two orders; if it fails, `E_h =
  max(order branch, fallback)`. Always at least `mcstat.LadderBound` (tested).
* Classes (H16, `gate4.py` docstring): `physical (restricted)` (Holm rejects) > `numerical` (production 90 % bootstrap-t interval above `c` AND the
  one-sided 95 % upper limit of the bias-corrected estimate below 0.5 x the production mean; Richardson order `p_use` if in range else 1) >
  `unresolved (bounded)` (`U(0.05) + sum_k B_k(0.05/K)`) > `unresolved`; descriptive, outside the family.
* Bootstrap size: a tail `t` is resolvable by `B` resamples only if `t (B + 1) >= 1`; `process.bootstrap_resolution` reports the check.

## Gate 5 reporting (acceptance D)

Every amplitude of `gate5.amplitudes.<a>` carries `reporting`: `confirmatory_outcome` = the Holm TOST decision of the family hypothesis `gate5:<a>`
(`rejected`, `p_tost`, `holm_level`) -- THE confirmatory outcome -- and `descriptive_verdict` = the G5.5 verdict (agree | disagree | inconclusive from the
90 % Fieller interval of the ratio), which is descriptive and never decides the hypothesis.

## `summary.json`

Sections `estimands`, `gate4`, `family`, `budgets`, `precision`, `process`, and `gate5`, `g7`, `secondaries`, `failures`, `positive_control`, `pollock` if
configured; non-finite numbers are the strings `"nan"`, `"inf"`, `"-inf"`. Figures (H18) in `figures/` if matplotlib imports, else
`FIGURES_SKIPPED` on stderr; never fatal.

## Tests (`sf43ana/tests`)

`test_eh` (known answers, per-triple range, fallback levels, the gate on an exact power law and a pre-asymptotic law, `>= LadderBound` on random
ladders with active gates), `test_gate4`, `test_holm`, `test_loader` (prefix, replacements, executor naming, Par2 seed map, torn ledger, data
contract), `test_gate5` (Hall, Fieller inversion, selection incl. J3B, prefix pairs, floors, Fieller power vs simulation, exclusion sensitivity),
`test_pollock`, `test_alignment` (Gate 5 reporting labels, coverage override, unreduced block), `test_g7`, `test_secondaries` (incl. recrossings hook), `test_failures`, `test_positive_control`, `test_nullsim` (family-wise rate <= 0.05 under
normal/skewed data with BOTH the Student and the bootstrap-t sampling limit), `test_e2e` (synthetic campaign: Gate 5 from the campaign, floors,
strict validation, `check-finite`, figures, the two-entries-share-the-production-arm contract).
