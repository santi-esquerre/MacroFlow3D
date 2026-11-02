# sf43plan: generator of the SF-43 plan, pre-registration and seed registry (variant A)

One design table (`design.py`, data) -> five files, deterministically (stdlib only; Python 3.11 compatible; runs under `python3 -W error`).
Not a pre-registration itself: the generated files are what the pre-registration commit freezes.

```bash
cd <A>/scripts          # A = docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion
python3 -m sf43plan generate --variant A --campaign-dir sf43_campaign --binaries <md5.json|placeholder> --out <A>
python3 -W error -m unittest discover -s sf43plan/tests -t .        # ~4 s, no GPU
```

`--binaries`: JSON `{"lagrangian_ensemble": "<md5>", "macroflow3d_pipeline": "<md5>"}`; `placeholder` writes all-zero md5, adds
`binaries_note` to the plan and a banner to `design_table.md`, and prints a warning: the engine's md5 gate then refuses to start (fails safe).
`--campaign-dir` is the plan's `campaign_dir` (relative to the repository root); the smoke directory is the same plus `_smoke`.

| file | content |
|---|---|
| `plan.json` | campaign-engine plan (`sf43camp/plan.py` schema): 44 arms, groups with reserves, blinded re-estimation, analysis / recompute / reconcile commands, `analysis.sha256_list` of the analysis files PRESENT at generation time (regenerate last, after the analysis nodes are integrated), and the `smoke` overlay (`run_sf43.sh ... --smoke`) |
| `prereg.json` | analysis pre-registration (`sf43ana/config.py` schema) plus the sections `gate5`, `g7`, `secondaries`, `failures`, `positive_control`, `engine_alignment` (config.py accepts unknown keys, so there is no separate `prereg_extra.json`) |
| `prereg_smoke.json` | the same for the smoke arms (`sm_` ids), floors lowered to 2, so that every analysis code path runs |
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

## Alignments still open (see also `engine_alignment` in `prereg.json`)

1. Window labels: the executor's tables are `<arm>__<segment>__d<lo>-<hi>.csv`, so `prereg.json` uses the stems as window labels
   (`d25-94`, `d25-59.5`, `d59.5-94`, `d94-194`) with `reduced_pattern` `{arm}/{segment}/{arm}__{segment}__{window}.csv`. `sf43ana.estimands`
   reads the halves and the far window under the labels `h1`, `h2`, `far`, which therefore find no table until the analysis accepts aliases.
2. Gate 5 twin and Par2 entries carry `selection: {weights: unif}`; `config.py` / `loader.py` ignore it at the base.
3. `g7.twins.pollock` refers to arms `<amp>_g5pollock` that do not exist (no Pollock-bridge reduced table yet).
4. Provisional numbers: Par2 `snapshots_every` (about 100 snapshots, a multiple of 1000), `U_time` = exp(sigma^2 / 6), `par2_exit_x` = box length.
5. The positive control's acceptance rule is not fixed in the sources read; `positive_control` only names the arm, the estimand and the reference.
6. `background` runs `sf43cov/run_cov.sh` (the real harness; already `nice -n 19`), not the placeholder `python3 -m sf43cov run --profile sf43`.
7. `analysis.command` calls `<A>/scripts/analyse_sf43.sh {campaign_dir} {build_dir}` (another node; absent at this base, so not hashed).
8. The engine's `--dry-run` builds lane B without its class threshold and with unreadable memory, so lane B takes near-empty units there
   (stage 1 20.0 h, 61.3 GPU-h); with the class (17000 MiB) and live memory it leaves after its 11.8 h of half-class units (stage 1 12.2 h).
