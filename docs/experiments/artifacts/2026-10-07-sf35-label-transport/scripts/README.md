# SF-35 campaign scripts

Purpose: run the SF-35 campaign of `inlet_slab --construct trace` unattended as detached V100 jobs and compute the
pre-registered verdicts by code. Run everything from the repository root. Dependencies: bash, `/usr/bin/time`,
python3 with numpy (digests: standard library + numpy only). The scripts never edit the repository outside this
artifact directory.

Driver outputs parsed: the `--summary` JSON of `--construct trace` is the primary source (keys `status`, `exit_code`,
`metrics.trace`, `residual`, `backflow`, `growth`, `transport` (`roundtrip_per_plane`, `status_hist(_per_plane)`,
`non_ok`), `planes`, `ladder[].labeldiff`, `timing`, `memory`; with `--solve`: `metrics.i1o4`, `continuation`,
`xcons`, `solve_status`). The runner echoes the log lines `CASE|BACKFLOW|GROWTH|RESIDUAL|TRANSPORT_STATUS|LABELDIFF|
TIMING|MEMORY|STATUS` of every run for the job log. JSON `null` means NaN/inf.

## `run_campaign.sh <build_dir> <part> [args]`

Common: `BIN=<build_dir>/inlet_slab`; `THREADS` (default 32); `PYTHON` (default `python3`); `NS` overrides the grid list of
a part; field `--sigma2 1 --ell 0.25 --seed 3001`; integration settings are the driver defaults (never passed).
Every run: `--summary raw/<dir>/<name>.json`, log `logs/<dir>/<name>.log`, `/usr/bin/time -v` file
`logs/<dir>/<name>.time`; prints the command, then `CASE_RESULT log=.. rc=.. STATUS .. wall=.. max_rss_kB=..` and
the key log lines; appends `<name>\t<rc>` to `logs/<dir>/exit_codes.tsv`; writes `logs/<dir>/run.env`
(`COMMIT` from the file `<artifact>/COMMIT` written by N7 before syncing, else `unknown`; host; UTC date; GPU;
threads). Nothing is retried; a return code outside the part's allowed set prints `UNEXPECTED_EXIT_CODE` and makes the
script exit 1 at the end (all parts still run to completion).

| part | runs | allowed rc | outputs |
|---|---|---|---|
| `controls [NS]` (default `16 32 64`) | `ctest --output-on-failure -R 'inlet_slab_(controls\|trace16)'` (timed; A4 + fast tests); per N: `--cells exports/controls/cos_x2_N<N>.npy --eps 0.5 --save-labels` (A2); `k = 1` (`--cells` of zeros) at the largest N; `--analytic control2d --n 32 --eps 0.5` (largest N if 32 is not in NS) | 0 | `logs/controls/`, `raw/controls/`, `analysis/controls.md` |
| `xcons` | `--construct trace --solve` at (eps, N) = (0.25, 32), (0.25, 64), (0.5, 32) | 0 | `logs/xcons/eps<E>_N<N>.*`, `analysis/xcons.md` |
| `regress` | `--production --n 32 --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder --threads $THREADS` | 0 | `logs/regress/prod32.*`, `analysis/regress.md` |
| `matrix <05\|10\|15> [NS]` (default `32 64 128 256`) | eps = 0.5 / 1 / 1.5; per N: `--construct trace --n N --eps E <field> --threads T --save-labels exports/matrix_eps<L>/N<N>`; `--ladder` only at N in `LADDER_NS` (default `32 64`) | 0 21 19 17 18 | `logs/matrix_eps<L>/N<N>.*`, `raw/matrix_eps<L>/N<N>.json`, `raw/matrix_eps<L>/planes_eps<E>.csv`, `analysis/matrix_eps<L>.md`, `analysis/cost_256.md` (rebuilt from the N256 points present) |
| `info` | `--sigma2 2.25 --eps 1` and `--sigma2 1 --eps 1.5` at `INFO_N` (default 32), `--save-labels`, then `labels_diff.py` | 0 21 19 17 18 | `logs/info/`, `analysis/info_sigma_eps.md` |
| `gpu [NS]` (default `128 256`) | `--tracer gpu` then `--tracer host` at eps 0.5 per N with `--save-labels`, then `gpu_equiv.py`. If the binary answers `--tracer gpu` with exit 1 and "not built", records it, writes `analysis/gpu.md` (not evaluable) and exits 0 | 0 21 19 17 18 | `logs/gpu/`, `raw/gpu/`, `analysis/gpu.md` |
| `smoke` | LOCAL ONLY: controls (NS 16), xcons (0.25, 16), matrix 05 (NS 16, ladder at 16), info (16), gpu (16) | as above | `logs/smoke/...`, `raw/smoke/...`, `analysis/smoke_*.md`, `exports/smoke/` (all gitignored) |

## Python tools (each has `--help`; deterministic; output printed and optionally written with `--out`)

| script | usage | role |
|---|---|---|
| `make_cos_x2_cells.py` | `make_cos_x2_cells.py N out.npy [--zeros]` | `Y_cell = cos(2 pi (j + 1/2) h)` (or 0) in the Grid3D layout `i + N (j + N k)` (x1 fastest), float64 `.npy`, for `--cells` |
| `controls_digest.py` | `--raw raw/controls --exports exports/controls [--ctest-log ..] [--out analysis/controls.md]` | A1 (`<= 1e-13`), A2 (straightness `<= 1e-10`; `e_v`, `r_F` orders on both pairs, `>= 3.5`, pre-asymptotic reading), A3 (`<= 1e-10`), ctest result and times (`<= 60 s`) |
| `xcons_digest.py` | `raw/xcons/eps*_N*.json [--sf33 DIR] [--out ..]` | D: `r_F <= 1e-10`; `e_psi`, `e_psi1`, `e_psi2` relative differences `<= 1e-6` against the SF-33 FULL-PRECISION JSON (`metrics.i1o4`, printed as `REFERENCE ...` lines); `a_psi1/2` order 32->64 `>= 2.5`; XCONS per-plane table |
| `regress_diff.py` | `NEW.log REF.log [--out ..]` | G: CASE/ORACLE/ORACLE_SUMMARY/EXTRA lines, only timing fields masked; lists every differing line; informational ORACLE_CONFIG/RUN/LABELDIFF comparison |
| `orders_digest.py` | `--eps E raw/matrix_eps<L>/N*.json [--out ..] [--planes-csv ..] [--expected-ns 32,64,128,256]`; `--selftest` | per-amplitude tables, observed orders, per-plane growth, ladder, verdict B and the literal verdict C (rule text printed at the top); missing stages -> `not_evaluable`; `--selftest` checks the verdict logic on synthetic series |
| `labels_diff.py` | `DIR_A DIR_B [--out ..] [--expect 1e-12]` | max `|dpsi|` between two `--save-labels` exports |
| `gpu_equiv.py` | `--case N HOST.json GPU.json HOST_DIR GPU_DIR ... [--out ..]` or `--not-built` | F: max `|dpsi|` host vs GPU `<= 1e-12`, per-plane nfev/status histograms, wall times |
| `cost_digest.py` | `raw/matrix_eps*/N256.json logs/matrix_eps*/N256.time [--out ..]` | H: wall, host max RSS, peak device memory per amplitude |
| `_common.py` | (library) | JSON/null handling, orders, tables, `/usr/bin/time -v` parsing |

## Export layout (`--save-labels DIR`, gitignored)

`DIR/psi1.npy`, `DIR/psi2.npy` shape `(N+1, N, N)` float64 `[plane j][m2][m3]` (`psi1 = x2 + U1`, `psi2 = x3 + U2`,
`x = m/N`), `DIR/labels.json` (field, eps, N, h, h_max, tol, backend, status, transport statistics, commit); ladder
entries in `DIR/ladder_h<div>_tol<tol>/`.
