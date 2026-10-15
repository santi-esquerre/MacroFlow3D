# SF-33 N6 — preconditioner gate at 16^3-32^3 (spec item 5; "Preconditioner not validated ... the 64^3-128^3 runs do not start")

## VERDICT: FAIL

The per-mode preconditioner P-A with restarted GMRES does **not** reproduce the prototype's direct-solve Newton
iterates on the full step-9a matrix. Linear failures (GMRES `max_iterations` or `stagnation` before
`lin_tol = 1e-12`) occur in 11 of 19 cases with the defaults (`--restart 100 --max-inner 6000`): 36 linear
failures; in 7 of 19 with `--restart 200 --max-inner 12000`: 28 linear failures. In two step-9a cases
(`gauss:0.5:24`, `gauss_ch:0.5:24`) and in `generic3d` eps 1 (16 / 20 / 24) they hit continuation stages that the
prototype accepts with direct solves, so the continuation ends at `continuation_floor` (status and metrics in
`proto_comparison.md`). No 32^3 step hits the inner cap (0 with both settings), but at 32^3 the first `eps = 0.25`
stage of `gauss` and `gauss_ch` fails for linear reasons with both settings (a stage the prototype also fails, by
Newton stagnation, so the PATH still matches). Neither setting passes; no setting tried makes every stage of every
step-9a case free of linear failures. Per the spec, the 64^3-128^3 runs must not start with P-A as is.

What does pass: every **accepted** Newton step of every case reached a true relative residual <= `lin_tol`
(LINEAR lines; maximum 9.9e-13 with the defaults, 1.0e-12 printed with r200); there is no accepted inexact step,
and where the continuation converges the GPU state equals the prototype's to ~1e-14 (`proto_comparison.md`).

## Method

Source: the `SOLVER`, `STAGE`, `LINEAR`, `NEWTON`, `STAGE_END` lines of `logs/proto/*.log` (defaults; job
`sf33-proto`) and `logs/proto_r200/*.log` (additional run, job `sf33-proto-r200`). A `LINEAR` line followed by a
`NEWTON ... lambda=` line is an accepted step; followed by `NEWTON ... linear_failure` it is a linear failure (step
not taken, stage ends `linear_failure`). "Steps at inner cap": LINEAR lines with `its >= max_inner`. Stage list:
`eps:status` in order (`ok` = converged; `None` = the final attempt at the target amplitude, which prints no
`STAGE_END`; its outcome is the case STATUS). The digest is a ~70-line parser of those lines run locally on the
retrieved logs (not committed: a reading aid, every number is re-derivable from the logs; the per-stage summary is
also in the `GMRES_STATS` lines of each log and in the `GMRES stage` lines of `logs/jobs/sf33-compare.log`).

## Linear failure vs the prototype's stage outcome

Prototype `STAGE_END` statuses (sweep2 cell logs, direct solves) for the stages the GPU failed for linear reasons:

| case | prototype stage outcomes | GPU (defaults) | GPU (r200) |
|---|---|---|---|
| gauss:0.25:24 / 32 | 0.25:stagnation(5 its) 0.125:ok 0.25:ok | 0.25:linear_failure (24: its 6000 rel 1.3e-7; 32: stagnation rel 0.27) | 24: 0.25:stagnation (= prototype); 32: 0.25:linear_failure |
| gauss_ch:0.25:24 / 32 | 0.25:stagnation(5) 0.125:ok 0.25:ok | 0.25:linear_failure | 24: = prototype; 32: 0.25:linear_failure |
| gauss:0.5:16 | 0.25:ok 0.5:stagnation(5) 0.375:ok 0.5:ok | 0.5:linear_failure (its 6000, rel 6.0e-10) | = prototype |
| gauss_ch:0.5:16 | 0.25:stag 0.125:ok 0.25:ok 0.5:stagnation(9) 0.375:ok 0.5:ok | 0.5:linear_failure | = prototype |
| gauss:0.5:24 | ... 0.5:stagnation(5) **0.375:ok 0.5:ok (8 its)** | 0.375->0.5 stage: linear_failure (its 6000, rel 4.0e-5) | 0.375->0.5 stage: linear_failure (its 12000, rel 6.3e-6) |
| gauss_ch:0.5:24 | ... 0.5:stagnation(5) **0.375:ok 0.5:ok (8 its)** | 0.375 stage linear_failure, then 0.5 linear_failure | 0.375->0.5 stage: linear_failure (its 12000, rel 1.2e-5) |
| generic3d:1:16 | 0.25 0.5 1:stag **0.75:ok** 1:stag 0.875:ok 1:stag 0.9375:ok 1:ok | 0.75 stage: linear_failure (its 6000, rel 7.8e-6) | 0.75 stage: linear_failure (stagnation rel 2.1e-2) |
| generic3d:1:20 | 0.25 0.5 1:linesearch-fail **0.75:ok** ... 1:ok | 0.75 stage: linear_failure (stagnation rel 0.27) | 0.75 stage: linear_failure |
| generic3d:1:24 | 0.25 **0.5:ok** 1:stag 0.75:stag 0.625:ok 0.75:ok ... 1:ok | 0.25->0.5 stage: linear_failure (stagnation rel 0.13) | 0.5 ok; 0.75 / 0.625 stages: linear_failure |

Reading: in the first five rows the GPU fails, for linear reasons, a stage that the prototype also fails (Newton
stagnation after 5-9 iterations), so PATH and final state agree; this is still a linear failure in the sense of
the gate. In the last five rows (bold) the GPU fails a stage the prototype **accepts**: the decisive gate failures.
The hard linear systems are the Jacobians far from the solution at larger amplitude (`eps >= 0.375` at 24^3;
`eps >= 0.5` for generic3d), where P-A — the per-mode `k = 1` operator, documented in the spec as a baseline that
"fails beyond eps = 0.25 at N >= 32" (SF-29 R7) — already fails at 24^3 (and at 16^3 for generic3d eps >= 0.75).

## GMRES statistics

Accepted-step iterations grow with N and eps: defaults median 166 / 300 / 357 at 16 / 24 / 32 (all stages, all
cases); the converged 32^3 cases need max 888 (gauss 0.25) and 1900 (gauss_ch 0.25), median ~355; target-stage
max / median per case are also in `raw/proto/compare_proto_fullref.md` (column "GMRES max / median (target)").
Restart used: 100 (defaults; `cycles` = its/100 in the LINEAR lines) and 200 (r200). Wall time per case:
`README.md`.

r_F history agreement where a full-precision solution exists (converged cases): 5.6-7.1 correct digits
(gauss / gauss_ch / generic3d), 15.7-17.0 (control2d), minimum at the last pre-converged entry (see
`proto_comparison.md`); failed cases: 0.4 (gauss 0.5 24), 8.3 (gauss_ch 0.5 24, agreement only until the GPU
diverges from the prototype path), -0.7 (generic3d 1 16).

## Per-case digest — defaults (`--restart 100 --max-inner 6000`, `logs/proto/`)

| case | restart / max_inner / lin_tol | stages `eps:status` | accepted steps | max rel (accepted) | all accepted <= lin_tol | linear failures (stage eps: GMRES status, its, rel) | GMRES its (accepted): max / median | steps at inner cap |
|---|---|---|---|---|---|---|---|---|
| gauss_0.25_16 | 100 / 6000 / 1e-12 | 0.25:ok | 10 | 6.6e-13 | yes | none | 167 / 124 | 0 |
| gauss_0.25_24 | 100 / 6000 / 1e-12 | 0.25:linear_failure 0.125:ok 0.25:ok | 17 | 6.6e-13 | yes | 0.25: max_iterations its=6000 rel=1.3e-07 | 3000 / 197 | 1 |
| gauss_0.25_32 | 100 / 6000 / 1e-12 | 0.25:linear_failure 0.125:ok 0.25:ok | 17 | 4.6e-13 | yes | 0.25: stagnation its=400 rel=2.7e-01 | 888 / 355 | 0 |
| gauss_0.5_16 | 100 / 6000 / 1e-12 | 0.25:ok 0.5:linear_failure 0.375:ok 0.5:ok | 25 | 6.6e-13 | yes | 0.5: max_iterations its=6000 rel=6.0e-10 | 900 / 200 | 1 |
| gauss_0.5_24 | 100 / 6000 / 1e-12 | 0.25:linear_failure 0.125:ok 0.25:ok 0.5:linear_failure 0.375:ok 0.5:linear_failure 0.4375:linear_failure 0.40625:ok 0.4375:linear_failure 0.5:None | 37 | 9.9e-13 | yes | 0.25: max_iterations its=6000 rel=8.3e-08; 0.5: stagnation its=800 rel=1.3e-01; 0.5: max_iterations its=6000 rel=4.0e-05; 0.4375: max_iterations its=6000 rel=6.2e-08; 0.4375: max_iterations its=6000 rel=1.5e-10; 0.5: stagnation its=400 rel=2.8e-01 | 5900 / 789 | 4 |
| gauss_ch_0.25_16 | 100 / 6000 / 1e-12 | 0.25:stagnation 0.125:ok 0.25:ok | 19 | 1.1e-13 | yes | none | 500 / 94 | 0 |
| gauss_ch_0.25_24 | 100 / 6000 / 1e-12 | 0.25:linear_failure 0.125:ok 0.25:ok | 15 | 2.3e-13 | yes | 0.25: stagnation its=400 rel=1.4e-01 | 363 / 199 | 0 |
| gauss_ch_0.25_32 | 100 / 6000 / 1e-12 | 0.25:linear_failure 0.125:ok 0.25:ok | 17 | 7.7e-13 | yes | 0.25: stagnation its=300 rel=4.0e-01 | 1900 / 359 | 0 |
| gauss_ch_0.5_16 | 100 / 6000 / 1e-12 | 0.25:stagnation 0.125:ok 0.25:ok 0.5:linear_failure 0.375:ok 0.5:ok | 34 | 9.6e-13 | yes | 0.5: stagnation its=400 rel=1.1e-01 | 1800 / 250 | 0 |
| gauss_ch_0.5_24 | 100 / 6000 / 1e-12 | 0.25:linear_failure 0.125:ok 0.25:ok 0.5:linear_failure 0.375:linear_failure 0.3125:ok 0.375:ok 0.5:linear_failure 0.4375:linear_failure 0.5:None | 32 | 9.8e-13 | yes | 0.25: stagnation its=400 rel=1.4e-01; 0.5: stagnation its=300 rel=5.2e-01; 0.375: max_iterations its=6000 rel=2.0e-12; 0.5: stagnation its=400 rel=6.1e-01; 0.4375: stagnation its=600 rel=3.0e-01; 0.5: stagnation its=400 rel=6.1e-01 | 4100 / 450 | 1 |
| control2d_0.25_16 | 100 / 6000 / 1e-12 | 0.25:ok | 2 | 9.6e-14 | yes | none | 48 / 38 | 0 |
| control2d_0.25_24 | 100 / 6000 / 1e-12 | 0.25:ok | 2 | 7.9e-14 | yes | none | 62 / 50 | 0 |
| control2d_0.5_16 | 100 / 6000 / 1e-12 | 0.25:ok 0.5:ok | 4 | 9.6e-14 | yes | none | 89 / 38 | 0 |
| control2d_0.5_24 | 100 / 6000 / 1e-12 | 0.25:ok 0.5:ok | 4 | 9.5e-14 | yes | none | 154 / 50 | 0 |
| generic3d_1_16 | 100 / 6000 / 1e-12 | 0.25:ok 0.5:ok 1:linear_failure 0.75:linear_failure 0.625:linear_failure 0.5625:ok 0.625:linear_failure 0.59375:ok 0.625:linear_failure 1:None | 34 | 9.7e-13 | yes | 1: stagnation its=400 rel=3.8e-01; 0.75: max_iterations its=6000 rel=7.8e-06; 0.625: max_iterations its=6000 rel=4.7e-06; 0.625: max_iterations its=6000 rel=2.1e-09; 0.625: max_iterations its=6000 rel=1.8e-11; 1: stagnation its=300 rel=6.0e-01 | 5600 / 896 | 4 |
| generic3d_1_20 | 100 / 6000 / 1e-12 | 0.25:ok 0.5:ok 1:linear_failure 0.75:linear_failure 0.625:linear_failure 0.5625:linear_failure 0.53125:ok 0.5625:linear_failure 1:None | 26 | 9.6e-13 | yes | 1: stagnation its=300 rel=4.6e-01; 0.75: stagnation its=400 rel=2.7e-01; 0.625: stagnation its=5500 rel=1.0e-09; 0.5625: stagnation its=700 rel=3.7e-02; 0.5625: max_iterations its=6000 rel=2.2e-12; 1: stagnation its=300 rel=4.7e-01 | 5600 / 1600 | 1 |
| generic3d_1_24 | 100 / 6000 / 1e-12 | 0.25:ok 0.5:linear_failure 0.375:ok 0.5:linear_failure 0.4375:ok 0.5:linear_failure 0.46875:ok 0.5:linear_failure 0.484375:ok 0.5:linear_failure 1:None | 36 | 9.0e-13 | yes | 0.5: stagnation its=400 rel=1.3e-01; 0.5: stagnation its=500 rel=7.4e-02; 0.5: max_iterations its=6000 rel=4.8e-08; 0.5: max_iterations its=6000 rel=2.2e-11; 0.5: max_iterations its=6000 rel=2.4e-10; 1: stagnation its=300 rel=4.8e-01 | 5300 / 1000 | 3 |
| generic3d_0.25_16 | 100 / 6000 / 1e-12 | 0.25:ok | 7 | 9.8e-14 | yes | none | 99 / 84 | 0 |
| generic3d_0.5_16 | 100 / 6000 / 1e-12 | 0.25:ok 0.5:ok | 14 | 7.7e-13 | yes | none | 600 / 132 | 0 |

GMRES iterations over all accepted Newton steps of all stages, per grid (default):

| N | accepted steps | max | median |
|---|---|---|---|
| 16 | 149 | 5600 | 166 |
| 20 | 26 | 5600 | 1600 |
| 24 | 143 | 5900 | 300 |
| 32 | 34 | 1900 | 357 |

Totals (default): linear failures 36 in 11 cases (gauss_0.25_24, gauss_0.25_32, gauss_0.5_16, gauss_0.5_24, gauss_ch_0.25_24, gauss_ch_0.25_32, gauss_ch_0.5_16, gauss_ch_0.5_24, generic3d_1_16, generic3d_1_20, generic3d_1_24); cases with an accepted step above lin_tol: 0; steps at the inner cap at 32^3: 0.

## Per-case digest — additional run (`--restart 200 --max-inner 12000`, `logs/proto_r200/`)

| case | restart / max_inner / lin_tol | stages `eps:status` | accepted steps | max rel (accepted) | all accepted <= lin_tol | linear failures (stage eps: GMRES status, its, rel) | GMRES its (accepted): max / median | steps at inner cap |
|---|---|---|---|---|---|---|---|---|
| gauss_0.25_16 | 200 / 12000 / 1e-12 | 0.25:ok | 10 | 2.8e-13 | yes | none | 122 / 110 | 0 |
| gauss_0.25_24 | 200 / 12000 / 1e-12 | 0.25:stagnation 0.125:ok 0.25:ok | 19 | 7.1e-13 | yes | none | 600 / 164 | 0 |
| gauss_0.25_32 | 200 / 12000 / 1e-12 | 0.25:linear_failure 0.125:ok 0.25:ok | 17 | 1.1e-13 | yes | 0.25: stagnation its=800 rel=1.4e-01 | 568 / 264 | 0 |
| gauss_0.5_16 | 200 / 12000 / 1e-12 | 0.25:ok 0.5:stagnation 0.375:ok 0.5:ok | 29 | 7.5e-13 | yes | none | 400 / 182 | 0 |
| gauss_0.5_24 | 200 / 12000 / 1e-12 | 0.25:stagnation 0.125:ok 0.25:ok 0.5:stagnation 0.375:ok 0.5:linear_failure 0.4375:ok 0.5:linear_failure 0.46875:ok 0.5:linear_failure | 49 | 8.7e-13 | yes | 0.5: max_iterations its=12000 rel=6.3e-06; 0.5: max_iterations its=12000 rel=2.5e-09; 0.5: max_iterations its=12000 rel=9.3e-07 | 11000 / 742 | 3 |
| gauss_ch_0.25_16 | 200 / 12000 / 1e-12 | 0.25:stagnation 0.125:ok 0.25:ok | 19 | 1.5e-13 | yes | none | 172 / 94 | 0 |
| gauss_ch_0.25_24 | 200 / 12000 / 1e-12 | 0.25:stagnation 0.125:ok 0.25:ok | 19 | 9.8e-13 | yes | none | 1600 / 177 | 0 |
| gauss_ch_0.25_32 | 200 / 12000 / 1e-12 | 0.25:linear_failure 0.125:ok 0.25:ok | 17 | 9.9e-13 | yes | 0.25: stagnation its=5800 rel=1.4e-06 | 556 / 270 | 0 |
| gauss_ch_0.5_16 | 200 / 12000 / 1e-12 | 0.25:stagnation 0.125:ok 0.25:ok 0.5:stagnation 0.375:ok 0.5:ok | 42 | 4.7e-13 | yes | none | 779 / 172 | 0 |
| gauss_ch_0.5_24 | 200 / 12000 / 1e-12 | 0.25:stagnation 0.125:ok 0.25:ok 0.5:linear_failure 0.375:ok 0.5:linear_failure 0.4375:ok 0.5:linear_failure 0.46875:linear_failure 0.5:None | 38 | 9.8e-13 | yes | 0.5: stagnation its=600 rel=2.0e-01; 0.5: max_iterations its=12000 rel=1.2e-05; 0.5: max_iterations its=12000 rel=1.4e-04; 0.46875: max_iterations its=12000 rel=2.9e-09; 0.5: max_iterations its=12000 rel=1.4e-04 | 9600 / 677 | 4 |
| control2d_0.25_16 | 200 / 12000 / 1e-12 | 0.25:ok | 2 | 9.6e-14 | yes | none | 48 / 38 | 0 |
| control2d_0.25_24 | 200 / 12000 / 1e-12 | 0.25:ok | 2 | 7.9e-14 | yes | none | 62 / 50 | 0 |
| control2d_0.5_16 | 200 / 12000 / 1e-12 | 0.25:ok 0.5:ok | 4 | 9.6e-14 | yes | none | 89 / 38 | 0 |
| control2d_0.5_24 | 200 / 12000 / 1e-12 | 0.25:ok 0.5:ok | 4 | 9.5e-14 | yes | none | 135 / 50 | 0 |
| generic3d_1_16 | 200 / 12000 / 1e-12 | 0.25:ok 0.5:ok 1:linear_failure 0.75:linear_failure 0.625:ok 0.75:linear_failure 0.6875:ok 0.75:linear_failure 0.71875:ok 0.75:linear_failure 1:None | 44 | 1.0e-12 | yes | 1: stagnation its=1000 rel=1.4e-01; 0.75: stagnation its=3000 rel=2.1e-02; 0.75: max_iterations its=12000 rel=2.0e-12; 0.75: stagnation its=1400 rel=1.2e-02; 0.75: stagnation its=1800 rel=3.6e-03; 1: stagnation its=5800 rel=1.5e-07 | 10000 / 1158 | 1 |
| generic3d_1_20 | 200 / 12000 / 1e-12 | 0.25:ok 0.5:ok 1:linear_failure 0.75:linear_failure 0.625:ok 0.75:linear_failure 0.6875:linear_failure 0.65625:linear_failure 1:None | 25 | 9.7e-13 | yes | 1: stagnation its=600 rel=3.7e-01; 0.75: stagnation its=800 rel=3.7e-02; 0.75: stagnation its=800 rel=2.4e-01; 0.6875: stagnation its=600 rel=2.8e-01; 0.65625: stagnation its=1400 rel=1.1e-01; 1: max_iterations its=12000 rel=1.1e-12 | 11200 / 745 | 1 |
| generic3d_1_24 | 200 / 12000 / 1e-12 | 0.25:ok 0.5:ok 1:linear_failure 0.75:linear_failure 0.625:linear_failure 0.5625:linear_failure 0.53125:ok 0.5625:linear_failure 1:None | 28 | 9.3e-13 | yes | 1: stagnation its=600 rel=4.1e-01; 0.75: stagnation its=1200 rel=1.9e-01; 0.625: max_iterations its=12000 rel=8.5e-10; 0.5625: max_iterations its=12000 rel=1.4e-06; 0.5625: max_iterations its=12000 rel=4.4e-12; 1: stagnation its=600 rel=4.3e-01 | 5400 / 1700 | 3 |
| generic3d_0.25_16 | 200 / 12000 / 1e-12 | 0.25:ok | 7 | 9.8e-14 | yes | none | 99 / 84 | 0 |
| generic3d_0.5_16 | 200 / 12000 / 1e-12 | 0.25:ok 0.5:ok | 14 | 1.0e-13 | yes | none | 356 / 115 | 0 |

GMRES iterations over all accepted Newton steps of all stages, per grid (r200):

| N | accepted steps | max | median |
|---|---|---|---|
| 16 | 171 | 10000 | 148 |
| 20 | 25 | 11200 | 745 |
| 24 | 159 | 11000 | 378 |
| 32 | 34 | 568 | 264 |

Totals (r200): linear failures 28 in 7 cases (gauss_0.25_32, gauss_0.5_24, gauss_ch_0.25_32, gauss_ch_0.5_24, generic3d_1_16, generic3d_1_20, generic3d_1_24); cases with an accepted step above lin_tol: 0; steps at the inner cap at 32^3: 0.
