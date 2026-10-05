# Streamline closure of the production Darcy flow (SF-30 closure gate)

- Date: 2026-10-05
- Status: complete (human review pending); section "e_v(h)" completed separately — see below
- Theory: [`docs/theory/lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) §2-§4
  (the claims whose project-verified limits this note extends)
- Predecessor: [`2026-10-02-streamline-closure-and-eq14-vs-darcy.md`](2026-10-02-streamline-closure-and-eq14-vs-darcy.md)
  (CPU probes; this note repeats its Question 1 on the production stack)
- Pre-registration: the bitácora of
  [`SF-30-streamline-closure-gate.md`](../plans/active/lester-eq14/increments/SF-30-streamline-closure-gate.md),
  rows of 2026-10-05, readings D-1..D-12 (fixed before any matrix run)
- Artifacts: [`artifacts/2026-10-05-sf30-closure-gate/`](artifacts/2026-10-05-sf30-closure-gate/README.md)
  (`raw/`, `raw_followup/`, `logs/`, `analysis/`, `scripts/`)

Every number below is read from a committed file. Sources are named per table:
`tables.md` = [`analysis/tables.md`](artifacts/2026-10-05-sf30-closure-gate/analysis/tables.md)
(output of `scripts/analyze.py`, numbered tables 1-12),
`classification.json` = [`analysis/classification.json`](artifacts/2026-10-05-sf30-closure-gate/analysis/classification.json),
`followup_tables.md` = [`analysis/followup_tables.md`](artifacts/2026-10-05-sf30-closure-gate/analysis/followup_tables.md)
(output of `scripts/followup_checks.py`, tables F1-F4), `summary.json` = a run's
`raw/<group>/<run-id>/summary.json`, `logs/` = the job logs. Numbers are rounded to 3-4
significant digits.

## Question

Does the Darcy flow of the project's own triply periodic Gaussian fields (SF-18
generator, SF-19 affine-periodic Darcy solve), at `sigma_Y^2` in {0.25, 1, 4} and `ell`
in {1/8, 1/16} on the unit cell (the paper's case is `(4, 1/16, 256^3)`), have closed
streamlines, i.e. is the first-return map of the face `x1 = 0` the identity?

A closed-streamline flow is a necessary condition for a nondegenerate affine + triply
periodic invariant pair `(psi1, psi2)` (2026-10-02 note, Question 1). The measurement
uses only the Darcy solve and a streamline integrator; it does not use any label, so it
is an oracle independent of the streamfunction solvers.

## Hypothesis

Predictions recorded in the bitácora before any run (row 2026-10-05T11:30Z, D-3 and
"Predictions to confront"):

- **P-A**: the production instrument reproduces the 2026-10-02 spectral-probe values on
  `lester_brk` (8.69e-3), `generic3d` (5.402e-2) and `two_mode` (2.85e-2), within the
  grid-convergence error.
- **P-B**: at fixed realization and `ell`, `R(sigma^2 = 1)/R(sigma^2 = 0.25)` is about 4
  (probes: 4.3).
- **P-C**: `R` changes by less than 10 % from 128^3 to 256^3 at `sigma^2 <= 1`.
- **Control prediction 1** (derived, D-3): `lester2021` closes to the PCG/roundoff level
  at every resolution (the mirror `x1 -> 1/2 - x1` maps cell centres onto cell centres).
- **Control prediction 2** (derived, D-3): `control2d` is exactly planar but closes
  in-plane only at `O(h^2)`.
- **Not predicted**: the class at `(4, 1/16)`, the backflow fraction at `sigma^2 = 4`,
  the many-period growth law.

Decision rule, as pre-registered (D-5, with D-11 and D-12; thresholds in the
"do not edit" block of `scripts/analyze.py`). Per case `(sigma^2, ell)` on realization
3001, `N_f = 256`, `N_c = 128`, `R(N)` = unweighted non-uniform RMS displacement at
period 1 at the working tolerance 1e-8, `E_ctrl(N)` = max(`R` of the matched 2-D control
of the same seed, `R` of `lester2021` with 1024 seeds) at `N`:

- validity (else `ambiguous`): Darcy PCG converged for the three correctors on every run
  used; integrator error `e_int` (D-11: max distance between the 1e-8 and the 1e-12
  return points) `<= 0.01 R(N_f)` or `<= E_ctrl(N_f)`; non-`ok` streamlines `<= 1 %` of
  the seeds in the domain of the map;
- `does_not_close`: `abs(R(N_f) - R(N_c))/R(N_f) < 0.10` AND `R(N_f) > 10 E_ctrl(N_f)`
  AND every 128^3 realization of the case has `R > 10 E_ctrl(128)` (with the control of
  seed 3001, D-12);
- `closes`: `R(N_f) <= 10 E_ctrl(N_f)` AND `R(N_f) < R(N_c)`;
- anything else: `ambiguous` -> owner. `(4, 1/16)` is evaluated on realizations 3001,
  3002, 3003 (D-12: 3002 and 3003 use the matched control of their own seed as
  `E_ctrl(N_f)`, `E_ctrl(N_c)`) and is classified only if the three agree.

The only numerical thresholds are 10 % and 10x (plus the two validity fractions, 0.01).

## Build / environment

- Source head: `947a523` (integrated; orchestrator-audited, record `FINAL_AUDIT`
  rounds 1-2). Executable `closure_gate` (`apps/closure_gate/`).
- The matrix ran on the synced tree of `531c73e` (the integrated commit before corrective
  C3, which only touched `apps/closure_gate/ev_ladder_main.cu` and a test file). The gate
  sources are byte-identical between `531c73e` and `947a523` (sha256 prefixes recorded in
  the orchestrator's final audit): `closure_gate.cuh` 8854f4eb7eab7b03,
  `closure_gate_main.cu` 334d6f46f65ac6ce, `streamline_integrator.hpp` bb9288e383a3f1ac,
  `closure_fields.hpp` 9c4116f25ae3074c, `closure_statistics.hpp` cec95b678a9213c4,
  `AffinePeriodicFlowSolver.cu` ac58658dfa46b540, `run_matrix.sh` 29635d85133d8912.
- Host: remote `v100` (2x Tesla V100-PCIE-32GB, as in the 2026-10-01 SF-26 note), CUDA
  11.4 (nvcc 11.4.152; orchestrator record), preset `v100-release`, per-increment
  mirror `scripts/remote --increment SF-30` (`cwd=/home/sesquerre/MacroFlow3D-SF-30` in
  every log header).
- Jobs:
  - `sf30-q1-build`, `sf30-q1-controls` (`logs/sf30-q1-controls.log`, 2026-10-05T12:06Z,
    GPU 0): instrument qualification on the N2 candidate `f9ca080`, before the
    pre-registration refinements D-10..D-12;
  - `sf30-matrix` (`logs/sf30-matrix.log`, 2026-10-05T12:35:03Z-12:48:38Z, GPU 1,
    `finished exit=0`): the 103 pre-registered runs. `raw/*/status.tsv`: controls 30,
    matched 22, matrix128 30, ladder 14, manyperiod 7 lines, all exit code 0; summed
    wall 220.4 + 162.5 + 190.1 + 119.9 + 120.7 s;
  - `sf30-post` (`logs/sf30-post.log`, 13:06:02Z-13:09:01Z, GPU 1, `finished exit=0`):
    `config_pspta_small` smoke (`smoke_exit=0`, `logs/sf30-smoke.log`) and the 18
    exploratory follow-up runs (`raw_followup/sensitivity/`).
- Double precision. Streamline integration on the host with 32 threads (`threads=32` in
  every run's log line and `timing.json`); results are independent of the thread count
  (each streamline is independent and stored by id; contract test of N2).

## Config(s)

- Unit cell `L = 1`, mean flux `qbar = e1`, `K = exp(Y)`, porosity 1.
- SF-18 Gaussian covariance `C_Y(r) = sigma^2 exp(-(r/ell)^2)`,
  `normalize_variance = true` (sample variance exactly `sigma^2`: `Y var` column of
  `tables.md` table 3), realizations (seeds) 3001-3005; the stateless mode hash gives the
  same continuum field on every grid.
- SF-19: three corrector problems, projected PCG + MG (levels auto, coarsest 4^3),
  `pcg_rtol = 1e-10` for the Gaussian runs, `1e-12` for the analytic controls.
- Seeds: 1024 points on `x1 = 0` from the stateless hash with `seed_rng = 20261005`, the
  same points in every run (grid comparisons are paired); the 16 probe points of the
  2026-10-02 note (`scripts/probe_seeds16.csv`) for the `_p16` controls.
- Integrator ladder `tol` in {1e-6, 1e-8, 1e-10, 1e-12}, working tolerance 1e-8 (1e-10
  for the `_p16` controls), maximum step `h` (D-11).
- Instrument (D-1): Dormand-Prince 5(4) on the arclength form `dx/ds = g/abs(g)` with
  `g = G + grad s_h` (`s_h` the SF-28 tricubic spline of the SF-19 potential fluctuation
  `h_tilde`), travel time from `K = exp(s_Y)`; the return point is landed on `x1 + 1`
  with `x1` as the independent variable (Hénon's device). The spec's wording
  "parametrized by `x1`" was not used for the whole streamline because that ODE is
  singular at `v1 = 0` and would discard exactly the streamlines with backflow whose
  count the spec asks for; statuses (`ok`, `seed_backflow`, `cap_exceeded`,
  `stagnation`, `step_underflow`, `landing_failed`) and backflow encounters are counted,
  nothing is clamped.
- Statistic: `R` = sqrt of the population variance about the mean of the displacement
  `d = (x2, x3)(x1 + 1) - (x2, x3)(x1)` over `ok` streamlines (the probes'
  "non-uniform RMS"; insensitive to a uniform drift).
- Controls: the five analytic probe fields (`lester2021`, `lester_brk`, `control2d` at
  eps 1; `two_mode`, `generic3d` at eps 0.5) against the spectral probe values; and, per
  matrix point, the matched 2-D control (D-3): the `x3`-average of the same realization,
  zero mean, rescaled to variance `sigma^2`, extruded along `x3` (field `gaussian2d`).

Run matrix (`scripts/run_matrix.sh all --list`; `tables.md` table 12: 103 found, none
missing, no configuration deviation, no Darcy failure):

| group | runs | content |
|---|---|---|
| `controls` | 30 | five analytic fields x {64, 128, 256}^3 x {16 probe seeds, 1024 seeds} |
| `matched` | 22 | `gaussian2d`, six cases x seed 3001 x {64, 128, 256}^3; `(4, 1/16)` seeds 3002, 3003 x {128, 256}^3 |
| `matrix128` | 30 | `gaussian`, six cases x seeds 3001-3005 at 128^3 |
| `ladder` | 14 | `gaussian`, six cases x seed 3001 at {64, 256}^3; `(4, 1/16)` seeds 3002, 3003 at 256^3 |
| `manyperiod` | 7 | `gaussian` seed 3001 at 128^3 with 64 periods (six cases); `(4, 1/16)` at 256^3 with 16 periods |

## Commands

From the repository root (detached V100 jobs; nothing ran locally except the analysis):

```bash
scripts/remote --increment SF-30 sync
scripts/remote --increment SF-30 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j --target closure_gate"
scripts/remote --increment SF-30 run sf30-matrix -- "bash docs/experiments/artifacts/2026-10-05-sf30-closure-gate/scripts/run_matrix.sh all"
scripts/remote --increment SF-30 wait sf30-matrix
```

One underlying command per group (from `run_matrix.sh all --list`):

```bash
# controls
build/v100-release/closure_gate --field lester_brk --n 256 --eps 1 --pcg-rtol 1e-12 --seeds-file docs/experiments/artifacts/2026-10-05-sf30-closure-gate/scripts/probe_seeds16.csv --tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-10 --out output_sf30/controls/a_lester_brk_n256_p16
build/v100-release/closure_gate --field lester2021 --n 256 --eps 1 --pcg-rtol 1e-12 --tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-8 --out output_sf30/controls/a_lester2021_n256
# matched
build/v100-release/closure_gate --field gaussian2d --n 256 --sigma2 4 --ell 0.0625 --seed 3001 --tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-8 --out output_sf30/matched/m_s4_l16_r3001_n256
# matrix128
build/v100-release/closure_gate --field gaussian --n 128 --sigma2 4 --ell 0.0625 --seed 3001 --tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-8 --out output_sf30/matrix128/g_s4_l16_r3001_n128
# ladder
build/v100-release/closure_gate --field gaussian --n 256 --sigma2 4 --ell 0.0625 --seed 3001 --tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-8 --out output_sf30/ladder/g_s4_l16_r3001_n256
# manyperiod
build/v100-release/closure_gate --field gaussian --n 128 --sigma2 4 --ell 0.0625 --seed 3001 --periods 64 --tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-8 --out output_sf30/manyperiod/p_s4_l16_r3001_n128_p64
```

Follow-up job `sf30-post` (exploratory, NOT pre-registered; command line copied from the
header of `logs/sf30-post.log`; the `exit=$?` it prints is the exit status of `grep`, the
job itself finished with exit 0):

```bash
G=./build/v100-release/closure_gate; O=output_sf30/sensitivity
for C in "s025_l16 0.25 0.0625" "s1_l16 1 0.0625" "s4_l16 4 0.0625"; do set -- $C
  for P in 2 4 8 16 32; do
    $G --field gaussian --sigma2 $2 --ell $3 --seed 3001 --n 128 --periods $P --tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-8 --out $O/g_$1_r3001_n128_p$P
  done
done
for P in 2 4 8; do
  $G --field gaussian --sigma2 4 --ell 0.0625 --seed 3001 --n 256 --periods $P --tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-8 --out $O/g_s4_l16_r3001_n256_p$P
done
```

Analysis (local, Python 3 standard library; regenerates the committed `analysis/` files):

```bash
python3 docs/experiments/artifacts/2026-10-05-sf30-closure-gate/scripts/analyze.py docs/experiments/artifacts/2026-10-05-sf30-closure-gate/raw \
  --tables docs/experiments/artifacts/2026-10-05-sf30-closure-gate/analysis/tables.md \
  --json docs/experiments/artifacts/2026-10-05-sf30-closure-gate/analysis/classification.json
python3 docs/experiments/artifacts/2026-10-05-sf30-closure-gate/scripts/analyze.py --self-test
cd docs/experiments/artifacts/2026-10-05-sf30-closure-gate
python3 scripts/followup_checks.py . --out analysis/followup_tables.md
```

## Outputs inspected

### 1. Instrument qualification

Analytic controls with the 16 probe seeds, tolerance 1e-12 (`tables.md` table 1; the
same values were measured in the qualification job `sf30-q1-controls` on `f9ca080`):

| field | eps | R(64) | R(128) | R(256) | Richardson 128/256 | spectral probe |
|---|---|---|---|---|---|---|
| `lester_brk` | 1 | 8.789e-3 | 8.713e-3 | 8.696e-3 | 8.690e-3 | 8.690e-3 |
| `two_mode` | 0.5 | 2.842e-2 | 2.846e-2 | 2.847e-2 | 2.847e-2 | 2.847e-2 |
| `generic3d` | 0.5 | 5.390e-2 | 5.399e-2 | 5.401e-2 | 5.402e-2 | 5.402e-2 |
| `control2d` | 1 | 7.103e-4 | 1.790e-4 | 4.483e-5 | (order 1.989, 1.997) | closes |
| `lester2021` | 1 | 3.251e-10 | 3.454e-10 | 3.067e-11 | | closes |

- With 1024 seeds at 1e-8 (`tables.md` table 2): `lester2021` `R` = 7.208e-9, 5.019e-10,
  3.452e-11 at 64/128/256 (integrator level at the step cap `h`); `control2d` in-plane
  `R` = 6.250e-4, 1.573e-4, 3.940e-5; its mean `d3` is at most 2.5e-16 in magnitude
  (table 1): the flow is planar.
- What is independent between the two instruments: the Darcy solver (spectral in the
  probes, second-order finite-volume MG-PCG here), the interpolant (exact trigonometric
  sum vs SF-28 tricubic spline of the potential), the integrator (scipy DOP853 in `x1`
  vs DP5(4) in arclength with a landing) and the language (Python vs C++/CUDA). The
  common elements are the field formulas and the 16 seed points.

### 2. Matrix and grid ladders

Per case, seed 3001 (`tables.md` tables 3, 4, 5 and 6; PCG iterations per corrector from
table 3, the same on every grid of the case):

| case | R(64) | R(128) | R(256) | R(128), seeds 3001-3005 | E_ctrl(128) | E_ctrl(256) | R(256)/E_ctrl(256) | rel. change 128->256 | PCG it. | e_int(256) |
|---|---|---|---|---|---|---|---|---|---|---|
| (0.25, 1/8) | 1.280e-2 | 1.277e-2 | 1.276e-2 | 1.277e-2 .. 1.938e-2 | 7.964e-5 | 1.990e-5 | 641.4 | 5.49e-4 | 10 | 8.63e-9 |
| (0.25, 1/16) | 1.160e-2 | 1.154e-2 | 1.153e-2 | 1.154e-2 .. 1.331e-2 | 2.087e-4 | 5.213e-5 | 221.2 | 6.95e-4 | 10 | 7.04e-8 |
| (1, 1/8) | 5.325e-2 | 5.314e-2 | 5.312e-2 | 5.314e-2 .. 8.194e-2 | 2.409e-4 | 6.012e-5 | 883.6 | 4.17e-4 | 20 | 4.88e-8 |
| (1, 1/16) | 5.092e-2 | 5.019e-2 | 5.008e-2 | 4.865e-2 .. 5.629e-2 | 1.320e-3 | 3.301e-4 | 151.7 | 2.31e-3 | 20 | 6.36e-7 |
| (4, 1/8) | 0.1605 | 0.1615 | 0.1618 | 0.1615 .. 0.2372 | 8.054e-4 | 2.023e-4 | 799.7 | 1.98e-3 | 20 | 5.39e-7 |
| (4, 1/16) | 0.1396 | 0.1352 | 0.1354 | 0.1352 .. 0.1608 | 1.178e-2 | 3.350e-3 | 40.43 | 1.70e-3 | 30 | 2.26e-5 |
| (4, 1/16) seed 3002 | | 0.1608 | 0.1611 | | 1.634e-2 | 4.722e-3 | 34.12 | 2.07e-3 | 30 | 1.79e-5 |
| (4, 1/16) seed 3003 | | 0.1548 | 0.1553 | | 4.761e-3 | 1.285e-3 | 120.8 | 2.86e-3 | 30 | 3.08e-5 |

`E_ctrl` is the matched 2-D control in every row (the `lester2021` value is smaller by
four orders or more). Matched controls (`tables.md` table 4): `R` falls with observed
order 2.00 +- 0.02 at `sigma^2 <= 1` and at `(4, 1/8)` (1.981-2.011); at `(4, 1/16)` the
orders are 1.407 (64->128) and 1.815 (128->256) for seed 3001, 1.791 and 1.889 for seeds
3002, 3003 (128->256): the control is not yet in its asymptotic range there.
`max abs d3` of the matched controls is at most 7.1e-13 (planar).

### 3. Classification

`tables.md` table 6, `classification.json` (`cases[*].class`): every case is
`does_not_close`.

| case | rel. change < 0.10 | R(256) > 10 E_ctrl(256) | min over 3001-3005 of R(128)/E_ctrl(128; 3001) | validity (Darcy, non-ok, e_int) | class |
|---|---|---|---|---|---|
| (0.25, 1/8) | 5.49e-4 | 641.4 | 160.4 | yes, 0, 8.63e-9 | `does_not_close` |
| (0.25, 1/16) | 6.95e-4 | 221.2 | 55.29 | yes, 0, 7.04e-8 | `does_not_close` |
| (1, 1/8) | 4.17e-4 | 883.6 | 220.6 | yes, 0, 4.88e-8 | `does_not_close` |
| (1, 1/16) | 2.31e-3 | 151.7 | 36.84 | yes, 0, 6.36e-7 | `does_not_close` |
| (4, 1/8) | 1.98e-3 | 799.7 | 200.5 | yes, 0, 5.39e-7 | `does_not_close` |
| (4, 1/16) r3001 / r3002 / r3003 | 1.70e-3 / 2.07e-3 / 2.86e-3 | 40.43 / 34.12 / 120.8 | 11.47 | yes, 0, <= 3.08e-5 | `does_not_close` (three agree) |

Margins, stated as they are:

- At `(4, 1/16)` the condition on the five 128^3 realizations is met with ratio 11.47
  (realization 3001's `R(128) = 0.1352` against the seed-3001 control
  `E_ctrl(128) = 1.178e-2`), as pre-registered (D-12).
- With seed 3002's OWN matched control at 128^3 (`E = 1.634e-2`, `tables.md` table 4),
  its `R(128) = 0.1608` is 9.84x, i.e. below 10x. The pre-registered rule does not use
  that comparison (D-12: the five-realization condition uses the seed-3001 control);
  the 256^3 comparisons with each seed's own control (34.1x, 40.4x, 120.8x) carry the
  case. The matched control at `(4, 1/16)` is itself not asymptotic at 128^3 (order
  1.41 from 64^3).
- The spec-literal reading (`tables.md` table 7: `E_lit` = max of the analytic
  `control2d` and `lester2021` with 1024 seeds; NOT the rule) gives the same six classes
  with ratios 292.8-4106 at 256^3.
- The integrator error is far inside its bound: the largest `e_int(256)/R(256)` is
  2.0e-4 (`(4, 1/16)` seed 3003; bound 0.01).
- `R(256) < R(128)` holds at `sigma^2 <= 1` and not at `sigma^2 = 4` (`R` increases by
  0.17-0.29 %); this criterion enters only the `closes` branch.

### 4. Pointwise convergence of the return map

`followup_tables.md` F2 (seeds paired by id, 1024 pairs except 1022 for `(4, 1/16)`
seed 3003, whose two `seed_backflow` seeds are not in the domain):

| case | RMS 64-128 | RMS 128-256 | ratio | p99 128-256 | max 128-256 | R(256) | RMS(128-256)/R(256) |
|---|---|---|---|---|---|---|---|
| (0.25, 1/8) | 2.238e-4 | 5.592e-5 | 4.00 | 1.307e-4 | 1.460e-4 | 1.276e-2 | 4.38e-3 |
| (0.25, 1/16) | 6.695e-4 | 1.646e-4 | 4.07 | 4.098e-4 | 5.929e-4 | 1.153e-2 | 1.43e-2 |
| (1, 1/8) | 1.324e-3 | 3.313e-4 | 4.00 | 9.652e-4 | 1.482e-3 | 5.312e-2 | 6.24e-3 |
| (1, 1/16) | 4.816e-3 | 1.215e-3 | 3.96 | 4.444e-3 | 9.536e-3 | 5.008e-2 | 2.43e-2 |
| (4, 1/8) | 1.469e-2 | 4.845e-3 | 3.03 | 1.562e-2 | 7.334e-2 | 0.1618 | 3.00e-2 |
| (4, 1/16) | 4.862e-2 | 1.884e-2 | 2.58 | 9.421e-2 | 0.1729 | 0.1354 | 0.139 |
| (4, 1/16) seed 3002 | | 1.731e-2 | | 7.501e-2 | 0.1606 | 0.1611 | 0.107 |
| (4, 1/16) seed 3003 | | 1.941e-2 | | 8.461e-2 | 0.2664 | 0.1553 | 0.125 |

At `sigma^2 <= 1` the return points converge pointwise at second order (ratio 3.96-4.07).
At `sigma^2 = 4` the pointwise map is not yet in the asymptotic range (ratios 3.03 and
2.58): at `(4, 1/16)` the RMS 128-256 difference is 1.9e-2, 14 % of `R`, while `R`
itself changes by 0.17 % (table 5). `R` is a variance-type statistic over the seeds; the
classification concerns `R`, not the individual return points.

Recomputation (`followup_tables.md` F1): `R` recomputed from the 103 per-seed CSVs equals
the summaries to a maximum absolute difference of 1.11e-16, with no `ok`-count mismatch.

### 5. Amplitude law

`tables.md` table 8, 128^3, same realization and `ell` (the fields at different
`sigma^2` are the same field scaled); range and mean of the five realizations' ratios:

| ell | R(1)/R(0.25) range | mean | R(4)/R(1) range | mean |
|---|---|---|---|---|
| 1/8 | 4.11-4.23 | 4.16 | 2.84-3.38 | 3.01 |
| 1/16 | 3.84-4.35 | 4.15 | 2.69-2.97 | 2.85 |

### 6. Backflow and statuses

From `tables.md` table 3 (and `summary.json` `direction_field_diagnostics`, `counts`):

- `sigma^2 <= 1`: no cell with `g1 <= 0`; the smallest `min g1` is 6.646e-2
  (`(1, 1/16)` seed 3001, 256^3). No seed or streamline meets backflow.
- `sigma^2 = 4`, `ell = 1/8`: backflow volume fraction 0 (seeds 3001, 3002) to 5.87e-4;
  `min g1` down to -9.558e-2; `ok` streamlines with a backflow encounter in period 1:
  0-11 of 1024; seeds on backflow (`seed_backflow`, outside the domain of the map): 14 for
  seed 3005, 0 otherwise.
- `sigma^2 = 4`, `ell = 1/16`: backflow volume fraction 2.28e-4 to 1.24e-3; `min g1` down
  to -0.2134; streamlines with a backflow encounter in period 1: 18-47 of 1024 (21 at
  64^3); seeds on backflow: 2 (seed 3003, both grids), 3 (seed 3004), 0 otherwise.
- `non_ok_fraction = 0` in every run: every seed in the domain of the map returned to
  `x1 + 1` (no `cap_exceeded`, `stagnation`, `step_underflow` or `landing_failed`).
- Many periods at `(4, 1/16)`: after 64 periods at 128^3, 906 of 1024 streamlines have
  met backflow at least once; after 16 periods at 256^3, 403 (table 10).

Consequence flagged for the owner (not decided here): a construction that assumes
`v1 > 0` everywhere, such as labels carried from the inlet face along `x1`, does not
hold for 1.8-4.6 % of the streamlines per period at `sigma^2 = 4`, `ell = 1/16`, and
for most streamlines over 64 periods.

### 7. Balance checks

Derived identities of the continuum flow with mean flux `e1`, porosity 1, unit cell:
flux-weighted mean travel time `<tau>_fw` = 1 (volume/flux) and flux-weighted mean
displacement = 0. Diagnostics, not gates.

- Analytic controls, 1024 seeds (`tables.md` table 2; `mean_tau_flux_weighted` of the
  `raw/controls/a_<field>_n<N>/summary.json`): `<tau>_fw` = 0.9951-1.005; the change from
  128^3 to 256^3 is at most 1.06e-4 for four fields and 3.6e-4 for `control2d`, and the
  offsets from 1 (up to 5e-3) do not shrink with the grid (the bitácora reads them as
  seed sampling). Flux-weighted mean displacement magnitude <= 7.84e-4.
- Gaussian cases, mean and sample standard deviation over the five 128^3 realizations
  (`tables.md` table 9, last block):

| case | `<tau>_fw` mean | sd |
|---|---|---|
| (0.25, 1/8) | 0.9959 | 0.0091 |
| (0.25, 1/16) | 0.9959 | 0.0105 |
| (1, 1/8) | 0.9898 | 0.0175 |
| (1, 1/16) | 0.9892 | 0.0156 |
| (4, 1/8) | 0.9662 | 0.0315 |
| (4, 1/16) | 0.9863 | 0.0414 |

At `sigma^2 = 4` the mean is 0.966 (`ell = 1/8`) and 0.986 (`ell = 1/16`) with sample
standard deviations 0.03-0.04 over five realizations; the deviation from 1 is 2.4 and
0.7 standard errors of the mean. It is not resolved from 1 at this sampling, and its
origin (seed sampling of strongly varying flux weights, or an inconsistency between the
spline streamlines and the FV flux) is not determined here.

### 8. Re-injection estimate (Lester 2023 eq. 36)

`D_ii = var(d_i)/(2 <tau>)` from the one-period return map, in units `qbar L`; mean and
sample standard deviation over the five 128^3 realizations (`tables.md` table 9, last
block). This is the number the re-injection protocol yields when applied to the return
map of the actual Darcy flow of the periodic cell; it is not presented as, nor accepted
as, a macrodispersion coefficient (owner decision O3).

| case | D22 (uniform) | D33 (uniform) | D22 (flux-weighted) | D33 (flux-weighted) |
|---|---|---|---|---|
| (0.25, 1/8) | 6.24e-5 +- 2.61e-5 | 6.88e-5 +- 2.95e-5 | 5.81e-5 +- 2.74e-5 | 6.23e-5 +- 2.77e-5 |
| (0.25, 1/16) | 3.40e-5 +- 4.01e-6 | 4.43e-5 +- 7.97e-6 | 3.04e-5 +- 3.05e-6 | 4.02e-5 +- 6.09e-6 |
| (1, 1/8) | 9.39e-4 +- 3.92e-4 | 1.06e-3 +- 3.71e-4 | 8.54e-4 +- 4.86e-4 | 8.87e-4 +- 3.41e-4 |
| (1, 1/16) | 5.51e-4 +- 9.66e-5 | 7.06e-4 +- 1.24e-4 | 4.81e-4 +- 8.15e-5 | 6.25e-4 +- 9.23e-5 |
| (4, 1/8) | 4.86e-3 +- 1.64e-3 | 4.72e-3 +- 2.72e-3 | 8.98e-3 +- 5.78e-3 | 8.05e-3 +- 3.29e-3 |
| (4, 1/16) | 2.91e-3 +- 3.66e-4 | 3.84e-3 +- 5.76e-4 | 4.51e-3 +- 5.46e-4 | 5.51e-3 +- 9.67e-4 |

The uniform-seed `<tau>` that enters the uniform `D_ii` has per-case means 1.02-2.36
(table 9); the flux-weighted one has per-case means 0.966-0.996 (section 7).

### 9. Many-period iteration

Pre-registered runs (`tables.md` table 10, working tolerance 1e-8, seed 3001):

| run | R(1) | R(64) (R(16) at 256^3) | var/(n var(1)) at last n, d2 / d3 | local exponent `log2(var(2n)/var(n))`, d2 | d3 |
|---|---|---|---|---|---|
| (0.25, 1/8), 128^3 | 1.277e-2 | 0.4106 | 21.36 / 6.849 | 1.977 -> 1.423 | 1.965 -> 0.489 |
| (0.25, 1/16), 128^3 | 1.154e-2 | 0.3317 | 11.39 / 14.55 | 1.993 -> 1.285 | 1.962 -> 1.239 |
| (1, 1/8), 128^3 | 5.314e-2 | 1.130 | 11.75 / 0.3121 | 1.690 -> 1.798 | 1.504 -> -0.360 |
| (1, 1/16), 128^3 | 5.019e-2 | 0.5718 | 1.273 / 2.890 | 1.470 -> 0.169 | 1.554 -> 0.805 |
| (4, 1/8), 128^3 | 0.1615 | 1.729 | 2.483 / 0.5057 | 1.487 -> 1.089 | 0.920 -> 0.816 |
| (4, 1/16), 128^3 | 0.1352 | 1.146 | 1.109 / 1.139 | 1.024 -> 1.042 | 1.116 -> 1.076 |
| (4, 1/16), 256^3, 16 periods | 0.1354 | 0.5689 | 0.9812 / 1.235 | 1.013 -> 0.833 | 1.164 -> 0.807 |

(Exponents: first value from `n = 1 -> 2`, last value from `n = 32 -> 64`, or
`n = 8 -> 16` at 256^3.)

Validity of the individual trajectories (`followup_tables.md` F3; the follow-up runs of
`raw_followup/` are exploratory, NOT pre-registered): the RMS / max distance between the
return points at tolerance 1e-8 and at 1e-12 after `n` periods:

| case, N | n = 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| (0.25, 1/16), 128 | 1.647e-7 / 6.716e-7 | 2.32e-7 | 4.237e-7 | 8.245e-7 | 2.238e-6 | 7.48e-6 | 4.507e-5 / 8.832e-4 |
| (1, 1/16), 128 | 7.179e-7 / 4.329e-6 | 1.149e-6 | 2.651e-6 | 1.593e-5 | 2.602e-3 / 7.054e-2 | 3.472e-2 / 0.4645 | 0.1683 / 1.266 |
| (4, 1/16), 128 | 7.196e-6 / 9.321e-5 | 6.099e-5 | 4.626e-3 / 0.1124 | 6.955e-2 / 0.4896 | 0.3665 / 1.294 | 0.8315 / 2.476 | 1.408 / 4.096 |
| (4, 1/16), 256 | 1.377e-6 / 2.258e-5 | 7.643e-6 | 3.732e-4 / 9.094e-3 | 3.906e-2 / 0.5033 | 0.2728 / 1.154 | | |

(RMS, or RMS / max. Cases with only `n = 1` and 64: `(0.25, 1/8)` 4.83e-4 max at
`n = 64`; `(1, 1/8)` 0.113; `(4, 1/8)` 5.94; F3 and `tables.md` table 11.) The `n = 1`
entries are bitwise identical across the several runs that contain them (F3
consistency table).

- At `sigma^2 = 0.25` the trajectories are resolved over 64 periods: the separation is
  at most 8.83e-4 (`ell = 1/16`) and 4.83e-4 (`ell = 1/8`), against `R(64)` = 0.33-0.41.
- At `sigma^2 >= 1` the tolerance-induced separation grows by large factors per
  doubling of `n` (up to 163x at `(1, 1/16)` from `n = 8` to 16, and 75.8x at
  `(4, 1/16)` from 2 to 4) and reaches order one. Individual trajectories are NOT
  resolved beyond about `n = 16-32` at `(1, 1/16)` and `n = 4-8` at `sigma^2 = 4`
  (`ell = 1/16`); at `(1, 1/8)` and `(4, 1/8)` only `n = 64` was measured (max 0.113
  and 5.94).
- The ensemble statistics agree across the four tolerances (`followup_tables.md` F4):
  relative spread `(max - min)/mean` of `R` <= 3.5 % at every `n` <= 64 (largest at
  `(4, 1/16)`, `n = 64`), of `var(d2)`, `var(d3)` <= 12.8 % (largest: `var(d2)` at
  `(4, 1/16)`, `n = 64`; `var(d3)` at `(4, 1/8)`, `n = 64`: 10.9 %); at `n <= 4` all
  spreads are <= 0.12 %.

Observations only:

- At `(4, 1/16)` `var(n)/(n var(1))` stays between 1.0 and 1.4 up to `n = 64` at 128^3
  (d2 1.017-1.134, d3 1.084-1.362) and between 0.98 and 1.41 up to `n = 16` at 256^3:
  about linear growth, i.e. compatible at that level with the independence assumption
  behind eq. 36.
- At `sigma^2 = 0.25` the growth starts quadratic (local exponent 1.96-1.99 at
  `n = 1 -> 2`) and slows: at `ell = 1/16` to 1.2-1.3 for both components; at
  `ell = 1/8` to 1.42 for `d2` and 0.49 for `d3`.
- At `(1, 1/8)` and `(4, 1/8)` the two components behave differently (`d3` exponent
  falls below 1, negative at `(1, 1/8)`), in runs whose individual trajectories are not
  resolved beyond the ranges above.

Open question (`unresolved`, not a finding): the growth of the tolerance-induced
separation is what a return map with sensitive dependence on initial conditions would
show. Whether that sensitivity is a property of the continuum periodic-cell Darcy flow
or of the discretized one (second-order finite-volume potential, `C^1` spline gradient)
is not established here. The pattern is similar on 128^3 and 256^3 at `(4, 1/16)` (F3).

### 10. `e_v(h)` of the frozen periodic stack

Pre-registered design (D-7; non-gating): driver `streamfunction_ev_ladder`
(`apps/closure_gate/ev_ladder_main.cu`, public API of the frozen stack only), direct solve
at `lambda = eta = 1`, `epsilon = 1e-6`, Anderson on, Newton off, tolerance 1e-8, SF-19
Darcy reference with `qbar = e1`, SF-11 diagnostics on the final accepted iterate, `r_F`
reported next to every `e_v`. Fixtures and predictions:

- Lester (2021) field, `eps = 0.25`, 32^3 / 64^3 / 128^3: converges (`e_v` decreasing
  with observed order about 2 while `r_F` reaches the tolerance).
- Gaussian fixture, SF-18 seed 3001, `ell = L/4`, `sigma^2` = 0.0625 and 0.25, 32^3 /
  64^3 / 128^3 on the same continuum field: `e_v` plateaus (change < 20 % from 64^3 to
  128^3) at a value scaling as amplitude squared (ratio about 4; probe field 4.9e-3 and
  2.0e-2), while `r_F` stalls at a floor that falls with `h`. Fixed in advance: the
  Gaussian states are not converged solutions of the discrete system; "plateau" requires
  `e_v` to stay while `r_F` falls.

| field | parameter | N | Picard iterations | `r_F` | `e_v` | wall (s) | status |
|---|---|---|---|---|---|---|---|
| `lester2021` | eps 0.25 | 32 | 1158 | 9.999e-9 | 4.121e-3 | 702 | converged (job `sf30-ev`) |
| `lester2021` | eps 0.25 | 64 | | | | | pending (job `sf30-ev2`) |
| `lester2021` | eps 0.25 | 128 | | | | | pending (job `sf30-ev2`) |
| `gaussian` | sigma^2 0.0625 | 32 | | | | | pending (job `sf30-ev2`) |
| `gaussian` | sigma^2 0.0625 | 64 | | | | | pending (job `sf30-ev2`) |
| `gaussian` | sigma^2 0.0625 | 128 | | | | | pending (job `sf30-ev2`) |
| `gaussian` | sigma^2 0.25 | 32 | | | | | pending (job `sf30-ev2`) |
| `gaussian` | sigma^2 0.25 | 64 | | | | | pending (job `sf30-ev2`) |
| `gaussian` | sigma^2 0.25 | 128 | | | | | pending (job `sf30-ev2`) |

The completed row ran on V100 as job `sf30-ev`, which was cancelled after this run to
free the GPU queue; the remaining eight runs are job `sf30-ev2` with an iteration budget
of 1000. Source of the completed row: the orchestrator's record of job `sf30-ev`; its
JSON record and log are to be committed with those of `sf30-ev2`. The rows, the exact
command lines and the reading against the predictions are filled in by the orchestrator
from the job records.

## Result

- **R1 — The production instrument reproduces the spectral probes.** With the 16 probe
  seeds, the Richardson values at 128/256 are 8.690e-3 (`lester_brk`), 2.847e-2
  (`two_mode`) and 5.402e-2 (`generic3d`), equal to the probe values to four digits;
  `control2d` closes at second order (observed orders 1.99, 2.00) and stays planar;
  `lester2021` closes to 3.1e-11 at 256^3 (16 seeds, tol 1e-12) and 3.5e-11 (1024 seeds,
  tol 1e-8).
- **R2 — On the production stack the return map is not the identity** for the six cases
  and the realizations run. `R(256)` ranges from 1.153e-2 to 0.1618 per period, changes
  by 0.04-0.29 % from 128^3 to 256^3, and is 34.1x-883.6x the matched 2-D control at
  256^3. At the paper's parameters `(4, 1/16, 256^3)`, `R` = 0.1354, 0.1611, 0.1553 on
  realizations 3001, 3002, 3003. Classification under the pre-registered rule: all six
  cases `does_not_close` (`classification.json`), with the margins of section 3.
- **R3 — Hence no nondegenerate affine + triply periodic invariant pair represents
  those Darcy flows.** The 2026-10-02 R1/R2 extend from the CPU probes (amplitude <= 1,
  `L/ell = 4`) to the production stack (SF-18, SF-19, SF-28) and to `sigma^2 = 4`,
  `ell = 1/16`.
- **R4 — Amplitude law.** `R(1)/R(0.25)` = 3.84-4.35 (means 4.16 and 4.15 for `ell` 1/8
  and 1/16): second order in the amplitude, as in the probes. From `sigma^2` = 1 to 4,
  `R` grows by 2.69-3.38 (means 3.01, 2.85), less than the factor 4 of the
  small-amplitude law.
- **R5 — Backflow exists at `sigma^2 = 4`.** Volume fraction up to 5.87e-4 (`ell` 1/8)
  and 1.24e-3 (`ell` 1/16); 18-47 of 1024 streamlines per period meet backflow at
  `ell = 1/16`, 906 of 1024 within 64 periods; every streamline still returns
  (`non_ok_fraction = 0`). None at `sigma^2 <= 1` (`min g1` >= 6.6e-2).
- **R6 — Many periods.** The displacement variance grows in every case. At
  `sigma^2 = 0.25` the trajectories are resolved over 64 periods and the growth slows
  from quadratic to exponents 0.5-1.4; at `(4, 1/16)` the growth is close to linear
  (`var(n)/(n var(1))` 1.0-1.4 to `n = 64`). At `sigma^2 >= 1` individual trajectories
  are not resolved beyond `n` about 4-32 (case dependent); the ensemble statistics agree
  across tolerances within 3.5 % (`R`) and 12.8 % (variances). Whether the trajectory
  sensitivity belongs to the continuum or to the discretized flow is open.
- **R7 — Re-injection numbers.** Section 8 gives `D22`, `D33` per case as protocol
  numbers of the periodic cell, from 3.0e-5 (`(0.25, 1/16)`) to 9.0e-3
  (`(4, 1/8)`, flux-weighted) in units `qbar L`; nothing in this note accepts them as
  macrodispersion coefficients.

Scorecard of the pre-registered predictions:

| prediction | outcome | numbers |
|---|---|---|
| P-A probe values reproduced | confirmed | Richardson 8.690e-3 / 2.847e-2 / 5.402e-2 vs probe 8.690e-3 / 2.847e-2 / 5.402e-2 |
| P-B `R(1)/R(0.25)` about 4 | confirmed | 3.84-4.35 over ten realization-`ell` pairs (probes 4.3) |
| P-C `R` stable within 10 % from 128^3 to 256^3 at `sigma^2 <= 1` | confirmed | 0.04-0.23 %; also 0.17-0.29 % at `sigma^2 = 4` |
| `lester2021` closes at every resolution | confirmed | `R` <= 3.5e-10 (16 seeds, 1e-12) and <= 7.2e-9 (1024 seeds, 1e-8) on every grid: the integrator level |
| `control2d` planar, in-plane closure at `O(h^2)` | confirmed | `abs(mean d3)` <= 2.5e-16; orders 1.989, 1.997 |
| class at `(4, 1/16)` | not predicted | `does_not_close` on 3001, 3002, 3003 |
| backflow at `sigma^2 = 4` | not predicted | volume fraction <= 1.24e-3, 18-47 streamlines per period at `ell = 1/16` |
| many-period growth law | not predicted | section 9; open question |

## Caveats

- Periodic cell only. Nothing is inferred about the transverse macrodispersion of a
  random, non-periodic medium, and no value of `alpha_T` is presupposed or concluded.
- Five realizations per case at 128^3; one grid ladder per case (realization 3001), three
  at `(4, 1/16)` (128/256 for 3002, 3003).
- `L/ell` = 8 and 16 only.
- The Darcy solve is second-order finite volume. At `(4, 1/16)` neither the pointwise
  return map (F2 ratio 2.58) nor the matched control (orders 1.41, 1.82) is in its
  asymptotic range at 256^3; `R` itself changes by 0.17 %.
- The matched control is a 2-D flow with the same amplitude and correlation length,
  used as a proxy for the instrument error on the 3-D field; it is not the instrument
  error of the 3-D field itself.
- First-return definition with backflow: one arclength integrator for every streamline,
  landing on `x1 + 1` (D-1); with backflow the backward first return is not the inverse
  of the forward one, so the round-trip check uses only streamlines without a backflow
  encounter (D-10). Seeds with `g1 <= 0` are outside the domain of the map (up to 14 of
  1024).
- The integrator runs on a `C^1` right-hand side (the spline gradient); its global error
  there is 30-300x `tol` (contract test C1, worst observed ratio 305). The `1e3 x tol`
  bound used by two contract tests is that empirical magnitude, not a derivation.
- Seeds are uniform on the face; flux-weighted statistics are reported next to the
  uniform ones.
- The matrix ran on the tree of `531c73e`; the gate sources are byte-identical to the
  final head `947a523` (section "Build / environment").
- One matrix point (`(1, 1/8)`, seed 3001, 64^3) had been run locally by the N2 worker
  as a prescribed smoke before D-11 was fixed (`R` = 5.32e-2, no backflow; disclosed in
  the bitácora). D-11 changed no threshold.
- Many-period trajectories are not tolerance-resolved at `sigma^2 >= 1` beyond the
  ranges of section 9; the follow-up runs that show it are exploratory, not
  pre-registered.
- The flux-weighted travel-time balance at `sigma^2 = 4` (0.966 and 0.986, five
  realizations) is not resolved from 1 and not explained (section 7).
- The `e_v(h)` section is incomplete at the time of writing (one of nine rows); the
  completed row's numbers come from the orchestrator's job record, not yet from a
  committed file.

## Next step

Inputs to owner decisions, not decisions:

1. Human review of SF-30 (spec: `Human review: required`), including the margins of
   section 3 at `(4, 1/16)`.
2. The later-phases gate "the labels' return map agreeing with SF-30" (decision record
   2026-10-02, R4; spec "Regression surface") now has reference numbers:
   `analysis/tables.md` tables 3, 5, 6 (per-case `R` and classification),
   `analysis/followup_tables.md` F2 (pointwise grid differences of the return map), and
   the per-seed return points `raw/<group>/<run-id>/streamlines.csv.gz` (1024 fixed
   seeds, `seed_rng = 20261005`).
3. The `v1 > 0` assumption at `sigma^2 = 4` (section 6): a label construction carried
   from the inlet face along `x1` needs a stated treatment of the 1.8-4.6 % of
   streamlines per period that meet backflow at `ell = 1/16`.
4. The open question of section 9 (sensitive dependence of the return map: continuum or
   discretization). The most useful discriminating checks: (a) repeat the
   tolerance-separation measurement with the 2026-10-02 spectral probe (exact
   trigonometric gradient, no spline, no finite volumes) on its Gaussian field at
   amplitudes 1 and 2; (b) estimate the Jacobian of the return map from nearby seed pairs
   at fixed tolerance.
5. Proposal (not applied here): when SF-30 is accepted, the "production-stack repeat
   pending in SF-30" qualifiers of theory note §2 and §4 and of `ARCHITECTURE.md` §4.3 /
   `AGENTS.md` can point to this note, with its limits (periodic cell, `L/ell` 8 and 16,
   five realizations).
6. Complete section 10 from the `sf30-ev2` records.
