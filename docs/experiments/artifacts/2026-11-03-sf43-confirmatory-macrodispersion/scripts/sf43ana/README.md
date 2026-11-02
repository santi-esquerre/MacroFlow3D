# sf43ana: frozen analysis package of SF-43 (Gate 4 core)

Purpose: from the per-segment `sf42red` tables, the campaign ledger and the pre-registration JSON, produce `summary.json` with the
estimands, the error budgets, the robust-sign tests, the fixed-family Holm decisions and the Gate 4 classes (decision
`2026-10-29-sf39-remediation-sf42-sf43` G4, G5.1-G5.4, G5.6, G5.7, G8 with the pre-activation refinements H2-H6, H16). It imports
`mcstat` and `sf42red` of the SF-42 artifact unchanged (located relative to this directory, or `SF42_SCRIPTS`). Not in this
module: Gate 5 (its p-values are an input), G7, secondaries, failure statistics (sibling modules, interface below).

```bash
cd docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts
python3 -W error -m unittest discover -s sf43ana/tests -t .        # about 70 s (the null simulation dominates)
python3 -m sf43ana run --prereg P.json --campaign C --reduced R --gate5 G.json|none --out O      # O/summary.json
```

## Inputs

* **Reduced tables** `R/<reduced_pattern>` with the default `{arm}/{segment}/{window}.csv` (`arm` = the arm's `source`, `segment` from the
  ledger record of the seed, `window` a key of `windows`): `python3 -m sf42red reduce` output, one table per d-window. Rows used:
  `code`, `family`, `weights`, `set` equal to `selection` (default consumer / plane / flux / perplane), `hmax_div` equal to the arm's,
  both bases; columns `seed`, `basis`, `alpha_TP`, `alpha_TAP`, `alpha_LP`, `U_r`.
* **Ledger** `C/<ledger>` (JSON lines): seed records `{"arm","segment","seed","verdict","class"}` (the LAST record of an
  (arm, seed) counts) and replacement records `{"kind":"replacement","seed","replaced_by","group"}` (group-wide).
* **Gate 5 p-values** `--gate5` JSON `{"p": {"<amplitude>": p}}` (wrapper optional); `none` or a missing amplitude: `p = 1`, untestable.

## Analysis set (loader)

Per arm: its pre-registered `seeds` order with the group's replacements applied (a replaced seed's position holds its reserve; chains
followed), cut at `n_conf`, then the longest COMPLETE prefix: a position is complete iff the ledger verdict is in `ok_verdicts` and the
segment table has a finite row (both bases, primary window). The first incomplete position ends the prefix (nothing later is used).
`N < floor` -> status `below_floor`: every hypothesis using the arm is untestable (`p = 1`), estimates are descriptive. Arms that share
seeds (same `group`) are cut to the common prefix wherever they are used together (ladder with its width-production arm, the two
resolutions of a width, the two `h_max` passes); the production arm keeps its own prefix for its sampling limit.

## Pre-registration schema (`config.py` validates; defaults in `config.DEFAULTS` are echoed into `summary.json`)

```text
windows            {"main":[25,94], "h1":[25,59.5], "h2":[59.5,94], "far":[94,194]}    primary_window "main"
selection          {"code":"consumer","family":"plane","weights":"flux","set":"perplane"}
bases              {"primary":"disp","secondary":"pos"}                                    (H2: displacement primary)
ledger, reduced_pattern, ok_verdicts
levels             {"fwer":0.05, "interval":0.90, "ucl":0.05}
null_per_sigma2    7.992e-5                       null_value(amplitude) = null_per_sigma2 * sigma^2, or amplitudes.<a>.null_value (0 allowed)
numerical_fraction 0.5
eh                 {"p_range":[0.5,2.1], "p_cap":2.0, "safety":1.25, "fallback_factor":3.0}
bootstrap          {"B_sampling","B_budget","B_ucl","B_bca","seed"}   all bootstrap streams derive from seed + a label (crc32)
sampling_method    "bootstrap-t" (default) | "student"
family             optional explicit list ["robust:<a>", "gate5:<a>", ...]; default robust:<a> for every amplitude without
                   "primary": false, plus gate5:<a> for every amplitude  (H4: a FIXED family, m never shrinks)
report_levels, targets {<a>: {"alpha_TP": half-width}}, first_order {<a>: {"alpha_TP","alpha_TAP","alpha_LP"}}
amplitudes.<a>     sigma2 (default float(a)), production (arm id), components (subset of E_h, E_W, E_WxH, E_int; K = their number),
                   ladder {arms: coarse->fine, resolutions, production_index, triples?}, width {wide_fine, prod_fine[, wide_coarse,
                   prod_coarse]}, hmax {h16, h8}, distance_arm, primary
arms[]             id, amplitude, group (arms with the same seeds), seeds (confirmatory order), reserves, n_conf, floor (>= 2),
                   source (ledger/table arm; default id), hmax_div (default -1), code (default consumer), role (informational)
```

## Statistics (all decisions in `gate4.py`, `eh.py`, `holm.py`)

* Robust limit `RL(a) = L(a/2) - sum_k B_k(a/(2K))` is `mcstat.RobustLimit`; `L` = bootstrap-t (or Student) lower limit of the production
  mean; `B_E_W, B_E_WxH, B_E_int` = `mcstat` contrast bounds with `method='student'` (the method SF-42 qualified for unpaired /
  interaction contrasts); `B_E_h` = `eh.EhBound` (docstring: triples, `p_use = min(p_L, p_cap)`, finest-branch second difference,
  fallback; at least `LadderBound` in every branch, tested).
* The null is shifted: reject iff `RL(a) > null_value`. Lattice p-value = smallest lattice `a` with `RL > null`; amplitude p =
  `max(p_TP, p_TAP)`. Holm decisions evaluate `RL(level) > null_value` at each Holm level exactly (both estimators); lattice p-values
  and Holm-adjusted p-values are reported.
* Classes (module docstring of `gate4.py`): `physical (restricted)` > `numerical` > `unresolved (bounded)` > `unresolved`; TP and TAP
  must agree, else `unresolved`.
* **Bootstrap size.** A tail `t` is resolvable by `B` resamples only if `t (B + 1) >= 1`. With the fixed family of `m` hypotheses,
  the sampling limit needs `B_sampling + 1 >= 2 m / fwer` and `E_h` needs `B_budget + 1 >= 4 K m / fwer` (`m = 8`, `K = 4`: 2560);
  otherwise the order lower limit is unresolved and `E_h` silently takes its conservative fallback. `process.bootstrap_resolution`
  reports this check.

## `summary.json`

Sections `estimands`, `gate4`, `family`, `budgets`, `precision`, `process`; non-finite numbers are the strings `"nan"`, `"inf"`,
`"-inf"`. Optional sibling blocks: if `sf43ana/<name>.py` exists for `name` in `gate5`, `g7`, `secondaries`, `failures` and defines
`analyse(ctx) -> dict` (`ctx.prereg`, `ctx.arms` (loaded `ArmData`), `ctx.campaign`, `ctx.reduced`, `ctx.gate5_p`), the dictionary is
stored under `summary[name]`.

## Tests (`sf43ana/tests`)

`test_eh` (known answers, `>= LadderBound` on random ladders in every branch), `test_gate4` (robust limit by hand, shifted null,
classes and precedence), `test_holm` (untestable hypotheses, exact decisions), `test_loader` (prefix, group-wide and chained
replacements, missing tables, `source`/`hmax_div`, a table written by the real `sf42red` reducer), `test_nullsim` (family-wise
rejection rate under a true alpha at the null / zero / biased / skewed, plus a power check; prints the rates), `test_e2e` (synthetic
campaign in the `sf42red` format, CLI, determinism, sibling-block merge).
