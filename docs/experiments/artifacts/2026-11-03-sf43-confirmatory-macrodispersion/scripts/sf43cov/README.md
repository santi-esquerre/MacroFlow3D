# sf43cov: restricted coverage harness of SF-43 (decision H15, H5)

CPU only. Qualifies, with the SF-42 machinery (`mcstat.coverage`: plan hash, deterministic streams independent of the number of workers,
restartable chunks, Clopper-Pearson verdicts of G5.3: 95 % one-sided UCL of the miss rate <= 1.25 x tail, >= 175 expected misses at the tail):

| block (proc) | what | sizes | R per cell |
|---|---|---|---|
| `fieller` | Gate 5 Fieller-form procedures (Hall, bootstrap-t, bootstrap-t at half the level) at the 8-hypothesis Holm levels and their halves, `normal` / `stress` / `stress-` | n 100..500 (7) | 56 000 |
| `fieller`, J3B | the same under the J3B-bound family (skewness 2.73, excess kurtosis 7.59, both signs), JUDGED (gating at sigma^2 = 4) | n 250 | 56 000 |
| `sf43_sampling` | the bootstrap-t sampling limit of the production mean (rows `lower`, `upper`) at the m = 8 Holm tails and their halves (smallest 3.125e-3) | n 200, B 33599 | 56 000 |
| `sf43_pipeline` | the whole robust-sign pipeline `RL(a_j) > 0` at the m = 8 Holm levels, variant A's arm sizes (production 200; ladder 100 at K = 3 / 150 at K = 4; width cells 150; h_max 30), bootstrap-t production limit, Student components, biases 0 and +2 s.e. per component (order 1, 2) | 2 structures x 3 biases | 28 000 |

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

**Outputs** in `<out_dir>`: `plan.json`, `chunks/`, `summary.json` (the file `sf43ana` reads as `gate5.coverage_summary`), `summary.md`,
**`qualification.json`** (`python3 -m sf43cov qualify --results <dir>`, the document the pre-registration quotes): per Gate 5 procedure and `n` qualified
yes/no for the base families and with the J3B family (`gate5.families`, `gate5.first_qualified`), the bootstrap-t sampling limit at n = 200
(`sampling`), every pipeline cell (`pipeline`), and `overall` (`null` = not all cells complete). `env.txt`, `run.log`.

**Cost** (this workstation, one core, measured by `python3 -m sf43cov bench --profile sf43`, one batch per cost group; the bench of the full
plan takes about 4 minutes): 52 cells, 2.41 M replicates, 8.0 core-hours (fieller 3.9, pipeline 3.9, sampling 0.2); on the host (4 to 9 times
slower per core, SF-42 measured 11.5 times on its largest arrays): 32-72 host core-hours, 1.1-2.5 h wall on 32 workers at 90 % efficiency.
The pipeline cells at K = 4 with a bias are the most expensive (0.05-0.06 s per replicate here).

Not re-simulated (reasons recorded): the explicit threshold of H3 (the coverage of `RL > c` for a true alpha at or below `c` is the coverage of
`RL > 0` for the shifted truth: shift-invariant), the `E_h` changes of H6 (the bound is pointwise larger or equal to `LadderBound`'s; tested in
`sf43ana/tests/test_eh.py`). The pipeline cells use `mcstat.LadderBound` (Student) for `E_h`, not the gated `sf43ana.eh.EhBound`.
