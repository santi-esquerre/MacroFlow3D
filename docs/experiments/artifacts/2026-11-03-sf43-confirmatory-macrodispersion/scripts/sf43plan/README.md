# sf43plan: generator of the SF-43 plan, pre-registration and seed registry (variant A)

One design table (`design.py`, data) -> five files, deterministically (stdlib only; Python 3.11 compatible; runs under `python3 -W error`).
Not a pre-registration itself: the generated files are what the pre-registration commit freezes.

```bash
cd <A>/scripts          # A = docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion
python3 -m sf43plan generate --variant A --campaign-dir sf43_campaign --binaries <md5.json|placeholder> \
    (--coverage-summary <summary.json> --coverage-qualification <qualification.json> | --smoke-only) --out <A>
python3 -W error -m unittest discover -s sf43plan/tests -t .        # ~4 s, no GPU
```

`--binaries`: JSON `{"lagrangian_ensemble": "<md5>", "macroflow3d_pipeline": "<md5>"}`; `placeholder` writes all-zero md5, adds
`binaries_note` to the plan and a banner to `design_table.md`, and prints a warning: the engine's md5 gate then refuses to start (fails safe).
`--campaign-dir` is the plan's `campaign_dir` (relative to the repository root); the smoke directory is the same plus `_smoke`.

| file | content |
|---|---|
| `plan.json` | campaign-engine plan (`sf43camp/plan.py` schema): 44 arms, groups with reserves, blinded re-estimation, analysis / recompute / reconcile commands, `analysis.sha256_list` (analysis files, pre-registration, coverage files: regenerate last), and the `smoke` overlay (`run_sf43.sh ... --smoke`) |
| `prereg.json` | analysis pre-registration (`sf43ana/config.py` schema, strict; every key emitted is accepted) with the sections `gate5` (incl. `completeness_min`), `g7`, `pollock`, `secondaries` (incl. `recrossings` of every consumer arm that is not a duplicate), `failures`, `positive_control`, `gate4_power` (achieved power, planning alternatives = the SF-42 R3 displacement-basis means, note lines 1025-1032), `basis` (H2) and the top-level `qualification` (the committed `coverage/qualification.json` the robust sign's qualification gate reads; production only) |
| `prereg_smoke.json` | the same for the smoke arms (`sm_` ids), every floor 2 and B reduced, so that every analysis code path runs |
| `seeds_registry.json` | every seed block (group, purpose, start, count, end, reserves, arms), the smoke blocks, the forbidden ranges, checked disjoint (`build.check_seeds`) |
| `design_table.md` | arms, GPU-h per arm and total, seed blocks |

## Design (decision 2026-11-02-sf43-pre-activation-decisions, H5-H13, H21; owner-accepted variant A)

* One CRN group (`g_s025`, `g_s1`, `g_s225`, `g_s4`) per amplitude: production, `E_h` ladder, width cells, Gate 5 consumer twin and Par2 arm, all
  with the same `seeds.start` and priority 1. Distance arms (`g_d_*`, priority 3), the positive control (`g_pc`, priority 2) and the G7 cells
  (`g7_<amp>_S|R`, priority 3: the four cells of a generator share seeds, the two generators have their own) are separate groups. Engine
  priority = H13 priority (the engine runs smaller numbers first). Seeds `43xxxx`; the smoke uses `439xxx`.
* Re-estimation (`blinded_reestimation.arms`): production arms and the wider width cells; never a Gate 5 arm, a ladder rung or a distance arm.
  `n_max = 2 n_conf` there. `hw_target {arm: {alpha_TP|alpha_TAP: half-width}}` is the spec's item-6 target (90 %); the engine computes
  `N_req = ceil((t_0.95 SD_UB80 / hw)^2)` (H9: inert unless the SD is several times the planning value; the tests check N_req at the SF-42 R3
  displacement-basis planning SD against `n_conf / 4` and at 20 times that SD against `n_conf`). The smoke scales the targets by 0.1 so that
  its re-estimation runs.
* Memory: `near` 30000 MiB (2080x480x480, 3328x384x384, 1248x576x576), `half` 17000 (1040x480x480, 1664x384x384), others measured own-PID peak
  of SF-42 x 1.1 + 500 MiB rounded up to 500 (`design.PEAK_MIB`; 416x48x48 and 3328x192x192 are estimates: the smoke measures them).
* `est_seconds` = `design_cost.py` `SEC` (planning). Total 57.77 GPU-h over `n_conf` plus 1.88 for the three-pass extras on the first 30
  production seeds (`hmax_div_first.est_seconds_extra`) = 59.65 (design_cost variant A: 59.6; the 0.05 is the 2.25 rung 416x96 at 150).
* Runner configurations are the SF-42 pilots' (`pilots.design`): tests compare window, planes, `--ell`, Par2 `dt` / steps / `mg_levels`.

## Engine mapping (one place)

`engine_map.py`: `executor`; per arm `primary` (production arms), `no_reestimation` (arms outside the re-estimation list), smoke `lane`; in the
consumer dict of the production arms `hmax_div_first` (`{n: 30, divs: [8, 16, 32]}`, `unit_size` divides n); smoke top-level
`determinism_pairs` `[[a, b]]`, `background` `[{name, argv, nice}]`, `background_wait_minutes`; `build` sets top-level `lane_b_min_free_mib` (17000).
The planning extras of the three-pass realizations are in `arms[].hmax_extra` (ignored by the engine).

## Alignments (all closed; the toolchain is tested as a whole by `tests_integration`)

1. Window labels: `prereg.json` keeps the semantic labels `main` [25, 94], `h1` [25, 59.5], `h2` [59.5, 94], `far` [94, 194]; `sf43ana.loader.window_stem` derives the
   executor's file stem `d%g-%g` from the bounds (one rule; no `reduced_pattern` in the pre-registration).
2. Arm entries: `n_conf` = the PLANNED size and `n_max` = the seeds the engine may run (`seeds` has `n_max` entries); `reest: true` on the entries that follow the
   engine's blinded re-estimation (`ledger` record `reest`, `n_new`): the production arms, their Pollock partners and the width cells. Gate 5 twins (`selection: {weights: unif}`,
   `hmax_div` 8 for a production source, -1 otherwise) and Par2 arms read the uniform-weight rows.
3. Pollock: `<amp>_prodpol` (acceptance B: the production set, flux, label 8) and `<amp>_g5pol` (the Gate 5 twin, uniform weights) are arms with `code: pollock` and
   the consumer's engine arm as `source`; `g7.twins.pollock` points at `<amp>_g5pol`; the `pollock` block lists the contrast and the decomposition.
   Basis control (H2): `<amp>_produnif` is the production set read with uniform weights (`source` = the production arm, `selection: {weights: unif}`, label 8, `reest: true` like
   `<amp>_prodpol`, same seeds / `n_conf` / `n_max` / floor); the `basis` block pairs `<amp>_prod` (flux) with it. These entries are analysis-only (no engine arm).
4. Par2 time-window parameters (a priori, `design.py`): `U_time = exp(sigma^2 / 6)`, `par2_exit_x` = the outlet plane (L / W in box units), `snapshots_every` about
   100 snapshots per realization (a multiple of 1000).
5. Positive control: arm `pc_s004` under the control amplitude `s004`; the pre-registered statistic `R` with its expectation [0.8, 1.25] and rationale is in
   `positive_control.expectation` (see `sf43ana/README.md`).
6. Analysis: `analysis.command` = `bash <A>/scripts/analyse_sf43.sh {campaign_dir} {build_dir}` (the smoke overlay adds `--smoke` and its own `sha256_list`).
   `analysis.sha256_list` covers the wrapper, `sf43ana`, `sf43cov`, `recompute43`, `reconcile43`, `mcstat`, `sf42red`, `seedlattice`, the pre-registration and, for
   the production plan, the coverage summary and qualification: the generator takes `--coverage-summary` and `--coverage-qualification` (the smoke job's outputs,
   committed as `<A>/coverage/summary.json` and `<A>/coverage/qualification.json`) and REFUSES a production plan without them; `--smoke-only` writes a plan that
   is good for the smoke run only (the coverage hashes are all-zero placeholders: the analysis gate refuses a production run). The smoke prereg has no
   `gate5.coverage_summary`: `analyse_sf43.sh --smoke` passes the smoke job's own `{campaign_dir}/coverage/summary.json` when it exists.
7. `background` = `bash <A>/scripts/sf43cov/run_cov.sh {campaign_dir}/coverage --workers 32 --profile sf43` (nice 19; ends with `sf43cov qualify`).
8. `recompute`: `--primary-arm` for each production arm and `--single-pass-hmax 8`.
9. `failure.halt_rate` 0.25 (systematic-failure detector; the expected rates per amplitude are pre-registered in `failures.expected`); `s4_l4` has 150 realizations (H9);
   the determinism-pair arms are not `heavy` (the lane pins place them).
10. The engine's `--dry-run` builds lane B without its class threshold and with unreadable memory, so lane B takes near-empty units there (stage 1 20.0 h, 61.4 GPU-h);
    with the class (17000 MiB) and live memory it leaves after its 11.8 h of half-class units (stage 1 12.2 h).
