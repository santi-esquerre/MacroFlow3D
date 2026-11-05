# sf43cov: restricted coverage harness of SF-43 (decision H15, H5)

CPU only. Qualifies, with the SF-42 machinery (`mcstat.coverage`: plan hash, deterministic streams independent of the number of workers,
restartable chunks, Clopper-Pearson verdicts of G5.3: 95 % one-sided UCL of the miss rate <= 1.25 x tail, >= 175 expected misses at the tail):

| block (proc) | what | sizes | R per cell |
|---|---|---|---|
| `fieller` | Gate 5 Fieller-form procedures (Hall, bootstrap-t, bootstrap-t at half the level) at the 8-hypothesis Holm levels and their halves, `normal` / `stress` / `stress-` | n 100..500 (7) | 56 000 |
| `fieller`, J3B | the same under the J3B-bound family (skewness 2.73, excess kurtosis 7.59, both signs), JUDGED (gating at sigma^2 = 4) | n 250 | 56 000 |
| `sf43_sampling` | the bootstrap-t sampling limit of the production mean (rows `lower`, `upper`) at the m = 8 Holm tails and their halves (smallest 3.125e-3) | n 100, 150, 200 (the grid of the qualification), B 33599 | 56 000 |
| `sf43_pipeline` | the whole robust-sign pipeline `RL(a_j) > 0` at the m = 8 Holm levels at two PROFILES of arm sizes per structure: `full` = variant A (production 200; ladder 100 at K = 3 / 150 at K = 4; width cells 150; h_max 30) and `half` = the truncated campaign (K = 3: production 100, ladder 50, width 75, h_max 20; K = 4: production 100, ladder 75, width 75, h_max 20); bootstrap-t production limit, Student components, biases 0 and +2 s.e. per component (order 1, 2); the same distributions and R in both profiles | 2 structures x 2 profiles x 3 biases | 28 000 |

```bash
cd docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts
python3 -W error -m unittest discover -s sf43cov/tests -t .                           # about 1 min
python3 -m sf43cov {plan,bench,run,summarize,qualify} --out DIR --profile sf43 [options]
bash sf43cov/run_cov.sh <out_dir> --workers 32 [--profile sf43]                         # smoke check, run, summary.json, qualification.json
```

**As a background CPU command** (inside the smoke-test job): argv `["nice", "-n", "19", "bash", "<A>/scripts/sf43cov/run_cov.sh", "<out_dir>", "--workers", "32"]`
(or without the outer `nice`: the study itself already runs at `nice -n 19`); the `--workers` value is capped at 32; exit code 0 = complete; the plan
is one file, finished chunks are skipped on a rerun (resumable, idempotent; the same command line must be reused: another plan in the same
directory is refused by hash). A tiny end-to-end check of every procedure runs first (warnings are errors) and aborts the job before the long run.

**Why two profiles and three sampling sizes (H13 floors against H15).** The floors of H13 (production 100, ladder 30, width 30, h_max 20) are smaller than the sizes
the coverage was first simulated at, and the campaign may be truncated or its re-estimation shrink an arm; a robust sign whose realized sizes the simulation did not cover
has no demonstrated coverage. `sf43ana` therefore declares the robust sign testable only at sizes covered by a qualified profile (`sf43ana/README.md`, "Qualification gate"):
the pipeline of its structure K at the largest profile with every arm size at most the realized one, and the sampling limit at the largest grid n at most the realized
production N, otherwise `p = 1`, status `not qualified at the realized sizes`. **Assumption (stated, not tested): qualification at a smaller size carries over to larger sizes**
(the coverage error of the studentized procedures decreases with n); the half profile is the smallest one simulated, so a campaign below it (or an arm of it below the profile)
is untestable by construction.

**Outputs** in `<out_dir>`: `plan.json`, `chunks/`, `summary.json` (the file `sf43ana` reads as `gate5.coverage_summary`), `summary.md`,
**`qualification.json`** (`python3 -m sf43cov qualify --results <dir>`, the document the pre-registration quotes): per Gate 5 procedure and `n` qualified
yes/no for the base families and with the J3B family (`gate5.families`, `gate5.first_qualified`), the bootstrap-t sampling limit on the grid
(`sampling`, grid n = 100, 150, 200), every pipeline cell (`pipeline`), the cells grouped by PROFILE (`pipeline_profiles`: structure K and the arm sizes production / ladder /
width / h_max, qualified iff every cell of the profile passed), the sampling result per grid size (`sampling_grid`) and `overall` (`null` = not all cells complete; including
`pipeline_by_profile`). The grouping is `sf43ana.qualification.pipeline_profiles`, the function the analysis applies. `env.txt`, `run.log`.

**Cost** (this workstation, one core, measured by `python3 -m sf43cov bench --profile sf43`, one batch per cost group; the bench of the whole plan takes a few minutes).
Before the truncation profiles: 52 cells, 2.41 M replicates, 8.0 core-hours (fieller 3.9, pipeline 3.9, sampling 0.2). The plan now has **76 cells and 3.25 M replicates**
(`python3 -m sf43cov plan --profile sf43`): the half-profile pipeline cells (+18 cells, 0.50 M replicates) and the sampling sizes n = 100, 150 (+6 cells, 0.34 M replicates).
Bench of those groups, 12 batches per group, three runs on this (loaded) workstation: **+2.0 / +2.5 / +2.7 core-hours here** (half pipeline 1.7-2.5, 0.008-0.022 s per
replicate, K = 4 about 1.6 times K = 3; sampling n = 100, 150 0.24-0.29, about 0.003 s per replicate); the same bench gives 3.0-3.3 core-hours for the full pipeline cells (the
8.0 above has 3.9) and 0.4-0.5 for the sampling cells, so the whole plan is about **10.5 core-hours here**. On the host (4 to 9 times slower per core; SF-42 measured its
coverage at about 6 times its docstring projection, and 11.5 times on its largest arrays): the addition is 10-25 host core-hours and the plan 42-95 host core-hours,
1.5-3.3 h wall on 32 workers at 90 % efficiency (the 6x reading of SF-42: about 63 core-hours, 2.2 h). The pipeline cells at K = 4 with a bias are the most expensive.

Not re-simulated (reasons recorded): the explicit threshold of H3 (the coverage of `RL > c` for a true alpha at or below `c` is the coverage of
`RL > 0` for the shifted truth: shift-invariant), the `E_h` changes of H6 (the bound is pointwise larger or equal to `LadderBound`'s; tested in
`sf43ana/tests/test_eh.py`). The pipeline cells use `mcstat.LadderBound` (Student) for `E_h`, not the gated `sf43ana.eh.EhBound`.
