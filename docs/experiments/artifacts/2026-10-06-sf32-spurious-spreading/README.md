# SF-32 artifacts — spurious transverse spreading of conventional trackers on surrogate flows with exact invariants

Experiment note: `docs/experiments/2026-10-06-sf32-spurious-spreading.md`. Increment:
`docs/plans/active/lester-eq14/increments/SF-32-reference-trackers-and-scalings.md`. Integrated source head `0f5b916`
(base `master = 37bfb25`).

## Provenance (V100: Tesla V100, nvcc 11.4, preset `v100-release`, detached `scripts/remote` jobs; logs in `logs/`)

| job | mirror | tree | content |
|---|---|---|---|
| `sf32-labels` | `SF-32-labels` | N2a `ef56372` | `analytic-labels --pair G --amplitude 0.05 --n 128`; `solve-labels --field lester2021 --eps 0.25 --n 128` and `--n 256` (`--max-iter 600`) |
| `sf32-int-build` | `SF-32` | `0f5b916` | configure + full build + `ctest -N` (23) + `ctest -R "tracker\|spurious"` (5/5) |
| `sf32-ctest-full` | `SF-32` | `0f5b916` | full `ctest` (23/23, 2752.88 s) |
| `sf32-ladders` | `SF-32` | `0f5b916` | `macroflow3d_pipeline apps/config_pspta_small.yaml` smoke; `run_ladders.sh` on the three label prefixes (8192 seeds, seed 20261006); `analyze.py` |
| `sf32-brk` | `SF-32` | `0f5b916` | post-hoc field (disclosed): `solve-labels --field lester_brk --eps 0.25 --n 128/256`; ladders; `analyze.py` over the five fields |
| `sf32-labels-equiv` | `SF-32` | `0f5b916` | bytewise equivalence of the labels produced by `ef56372` and `0f5b916` (analytic G 128^3, stack lester2021 128^3) |

Each ladder = Pollock `Delta/Delta0 = 1, 2, 4, 8`; RK `tol = 1e-4 .. 1e-10` (`dt_max = 0.25`); pseudo-symplectic
`tol_psi = 1e-8, 1e-10, 1e-12` (`ds = h/2`); 14 runs per field, 42 + 28 runs in total, all exit 0.

## Layout

- `ladders/<field>/<tracker>_<level>/summary.json`, `run.log` — every run (the analysis recomputes every statistic
  from the per-seed CSVs; `tables.md` reports the differences);
- `ladders/<field>/<tracker>_<level>/seeds.csv.gz` — per-seed outputs, kept for the two exponent-bearing fields
  `analytic_G_a0.05_n128` and `stack_lester_brk_e0.25_n256` (gzip; the other fields' CSVs stay on the mirror);
- `ladders/<field>/labels.json`, `labels/*.json` — label metadata (route, `r_F`, `e_v`, `min|c|`, exit reason);
  the raw `.bin` label fields (16-128 MiB each) stay on the mirror and in the orchestration record;
- `ladders/analysis/` — `tables.md`, `exponents.json`, `histograms/`, figures (`fig3b_*`, `fig3c_*`, `fig4a_*`,
  `fig4b_*`, `ps_drift_*`, `exponents_crossfield`) as PNG and PDF, produced by `apps/spurious_spreading/analyze.py`
  with the pre-registered constants in its single rule block;
- `preregistration-understanding-record.md`, `dag.json` — the orchestrator's UNDERSTAND/PLAN record (readings T1-T4,
  predictions P1-P6, decisions D-1..D-3, the corrections made before the runs and the post-hoc addition, with
  timestamps); `orchestrator_prototype_pollock.{py,log}` — the independent numpy prototype used for the audits.

Field name convention: `<route>_<field|pair>_<e eps|a amplitude>_n<N>`.
