# SF-33 N6 — V100 campaign A: index

Prototype reproduction (step 9a), preconditioner gate at 16^3-32^3 (item 5), SF-19 cross-check (step 8).

| file | content | verdict |
|---|---|---|
| `proto_comparison.md` | per-case GPU vs prototype table (both sides), PATHs, FIELDDIFF, r_F history, reading rule | 9a: 12/14 PASS, 2 FAIL (`gauss`, `gauss_ch` eps 0.5 at 24^3); `generic3d` eps 1 at 16/20/24: FAIL |
| `preconditioner_gate.md` | linear failures per case and stage, GMRES its at 16/24/32, accepted-step residuals, defaults and `--restart 200 --max-inner 12000` | **FAIL** (both settings) |
| `sf19_crosscheck.md` | SF-19 inlet `v1` / spline `v_perp` vs the spectral reference, 16/24/32, observed orders | recorded: orders 1.92-2.04 |

## Execution facts

- Host `v100` (two V100, 32 GB each per `MEMORY total_device`; 80 cores), per-increment mirror `~/MacroFlow3D-SF-33`, state root
  `~/.macroflow3d-remote/macroflow3d-SF-33`; one `scripts/remote --increment SF-33 sync` at 2026-10-06T20:59Z from
  the worktree at commit `681b039` (SF-33 C2b head; chain N0..N5, C1, C2, C2b). Build `build/v100-release`
  (preset `v100-release`).
- Every job ran with `REMOTE_GPU_WAIT=7200`; none waited (no exit 75). Job logs (copied): `logs/jobs/<job>.log`.
  The logs of the earlier SF-33 build jobs (`sf33-base-build-refs`, `sf33-n0-build`, `sf33-n134-build`,
  `sf33-n01234-build`, other nodes) are copied there too for completeness; they are not part of N6.

| job | kind | GPU | start (UTC) | end (UTC) | exit | command / outputs |
|---|---|---|---|---|---|---|
| `sf33-proto-ref24` | CPU (prototype) | 0 (lock only) | 2026-10-06T21:00:02Z | 22:01:28Z | 0 | `candidate_i.py <case>:i1o4 --direct-max 32 --reuse-lu 0 --save ~/sf33_ref24` for gauss/gauss_ch 0.25, control2d 0.5, gauss/gauss_ch 0.5 at 24^3 (five concurrent processes, 3 BLAS threads each), then `export_proto.py --solutions ~/sf33_ref24/*.npz --out ../exports/solutions`; logs `logs/jobs/candidate_i_24*.log` |
| `sf33-build` | build + ctest `-R inlet_slab_` | 1 | 21:00:54Z | 21:02:01Z | 0 | `BUILD_EXIT=0`; 5/5 inlet_slab tests passed (39 s) |
| `sf33-export` | CPU | 1 (lock only) | 21:02:57Z | 21:25:57Z | 0 | 19 case exports, 12 16^3 solution conversions, 3 cross-check exports (host-only `exports/`, gitignored) |
| `sf33-proto` | GPU | 1 | 21:26:53Z | 21:36:58Z | 0 | `scripts/run_proto.sh build/v100-release <artifact>`; `logs/proto/`, `raw/proto/` (per-case driver exits in the job log: 0 converged, 15 continuation_floor) |
| `sf33-proto-r200` | GPU (additional gate run) | 1 | 21:38:27Z | 22:00:59Z | 0 | same matrix, `--restart 200 --max-inner 12000`; `logs/proto_r200/`, `raw/proto_r200/` |
| `sf33-crosscheck` | GPU | 0 | 22:01:32Z | 22:01:46Z | 0 | `scripts/run_crosscheck.sh build/v100-release`; `logs/crosscheck/`, `raw/crosscheck/` |
| `sf33-proto-24ref` | GPU | 0 | 22:03:12Z | 22:06:52Z | 0 | five 24^3 cases with `--solution <exports>/solutions/<case>_i1o4` (defaults); `logs/proto_24ref/`, `raw/proto_24ref/` |
| `sf33-compare` | CPU | 0 (lock only) | 22:07:50Z | 22:07:52Z | 0 | `compare_proto.py` over `logs/proto`, `logs/proto_24ref`, `logs/proto_r200` (exit 1 per table = some case FAIL, by design) -> `raw/proto/compare_proto_fullref.md`, `raw/proto_24ref/compare_proto.md`, `raw/proto_r200/compare_proto.md` |

Ordering note: `sf33-export` was launched only after the five 24^3 oracle caches of `sf33-proto-ref24` were on
disk (21:00:16-21:00:32Z; `cases.load_case` writes them non-atomically), so no two processes wrote the same cache.
`sf33-proto` ran before the 24^3 references existed and therefore without `--solution` at 24^3; those five cases
were re-run as `sf33-proto-24ref` (metrics bitwise identical to `sf33-proto`, checked on all 44 metric values of
each JSON summary).

## Wall time per case

Driver `TIMING solve=` [s] (GPU, defaults / r200), job `CASE_RESULT wall=` [s] including load (defaults), peak device
memory (`MEMORY peak_device_bytes`, defaults, device-wide):

| case | solve defaults | solve r200 | wall defaults | peak device [MB] |
|---|---|---|---|---|
| gauss_0.25_16 | 1.32 | 1.38 | 5 | 497 |
| gauss_0.25_24 | 20.19 | 7.59 | 24 | 525 |
| gauss_0.25_32 | 13.38 | 14.44 | 17 | 569 |
| gauss_0.5_16 | 15.36 | 10.81 | 19 | 499 |
| gauss_0.5_24 | 111.86 | 266.52 | 116 | 529 |
| gauss_ch_0.25_16 | 2.75 | 2.17 | 6 | 497 |
| gauss_ch_0.25_24 | 4.59 | 14.17 | 9 | 525 |
| gauss_ch_0.25_32 | 15.32 | 28.43 | 19 | 569 |
| gauss_ch_0.5_16 | 16.37 | 19.36 | 20 | 499 |
| gauss_ch_0.5_24 | 64.06 | 269.50 | 68 | 529 |
| control2d_0.25_16 | 0.05 | 0.05 | 4 | 497 |
| control2d_0.25_24 | 0.12 | 0.12 | 3 | 525 |
| control2d_0.5_16 | 0.17 | 0.17 | 4 | 497 |
| control2d_0.5_24 | 0.36 | 0.40 | 4 | 525 |
| generic3d_1_16 | 77.99 | 254.28 | 82 | 499 |
| generic3d_1_20 | 76.32 | 174.55 | 80 | 513 |
| generic3d_1_24 | 108.38 | 213.47 | 112 | 529 |
| generic3d_0.25_16 | 0.59 | 0.58 | 4 | 497 |
| generic3d_0.5_16 | 4.13 | 4.06 | 8 | 497 |

Prototype 24^3 references (`sf33-proto-ref24`, CASE `t=` [s], sparse direct solves, five concurrent processes):
gauss 0.25 1875, gauss_ch 0.25 1948, control2d 0.5 435, gauss 0.5 3683, gauss_ch 0.5 3573.

Cross-check: ~1 s per grid (`TIMING total` 0.94 / 1.01 / 1.03 s at 16 / 24 / 32).

## Script changes

None. The campaign scripts ran as committed; `run_proto.sh` has no option for extra driver arguments or for
solution directories other than the SF-29 16^3 ones, so the additional runs (`sf33-proto-r200`,
`sf33-proto-24ref`) were plain driver loops written inline in the job command (recorded in the job logs).

---

# SF-33 N6b — V100 re-run of the 9a matrix with the coarse-corrected solver (C3 defaults): index

| file | content | verdict |
|---|---|---|
| `proto_comparison_b.md` | per-case GPU vs prototype table (both sides), the `compare_proto.py` table verbatim, continuation outcome of the non-converged cases, reading rule (`PASS` vs `PASS(4dig)`; 32^3 eps 0.5 rows: no prototype reference) | the 14 N6 9a cases: 14/14 PASS (0 bisections); `gauss:0.5:32`, `gauss_ch:0.5:32`: FAIL (`continuation_floor`); `generic3d` eps 1 at 16/20/24: `continuation_floor` (informative) |
| `preconditioner_gate_b.md` | verdict rule (verbatim), per-criterion evidence, GMRES its per accepted step per case, per case/stage linear failures, GMRES max/median, coarse K, build/apply times, rcond, curves of every failed linear solve at 32^3 eps 0.5, wall times | **FAIL** (32^3, eps 0.5) |

## Execution facts (N6b)

- Worktree commit `bfe4efb` (SF-33 C3 on the N0..N7c chain head `ea53d09`). One `scripts/remote --increment SF-33
  sync` at 2026-10-07T00:33Z, after checking that no SF-33 job was `running` / `waiting_gpu` (all 12 earlier SF-33
  jobs `succeeded`). `rsync --delete` would remove the host-only (gitignored) `exports/` and the SF-29 24^3/32^3
  oracle caches; both were copied aside to `~/sf33b_keep/` before the sync and restored with `rsync -a
  --ignore-existing` right after it (short `exec` steps; 22 case exports + 17 solution directories + 10 untracked
  caches back in place), so only the two new 32^3 eps 0.5 exports had to be computed.
- Every job ran with `REMOTE_GPU_WAIT=7200`; none waited. Job logs (copied): `logs/jobs/sf33b-*.log`.

| job | kind | GPU | start (UTC) | end (UTC) | exit | command / outputs |
|---|---|---|---|---|---|---|
| `sf33b-build` | configure + build + ctest `-R inlet_slab_` | 0 | 2026-10-07T00:34:21Z | 00:35:32Z | 0 | `BUILD_EXIT=0` (CUDA 11.4, preset `v100-release`; first V100 build of the N7c coarse code), 6/6 inlet_slab tests passed (43 s, incl. `inlet_slab_coarse`), `CTEST_EXIT=0` |
| `sf33b-export` | CPU | 1 (lock only) | 00:34:46Z | 00:40:43Z | 0 | `export_proto.py gauss:0.5:32 gauss_ch:0.5:32 --out ../exports` (32 stage amplitudes each; oracle caches computed and written, `cache_written=True`) |
| `sf33b-proto` | GPU | 0 | 00:36:36Z | 00:52:18Z | 0 | inline loop (recorded in the job log): `inlet_slab --proto <exports>/<case> [--solution <exports>/solutions/<case>_i1o4] --summary raw/proto_b/<case>.json > logs/proto_b/<case>.log`, driver defaults, 21 cases; the two 32^3 eps 0.5 cases ran after waiting for `sf33b-export` to finish (status file polled); per-case `CASE_RESULT ... rc= STATUS ... wall=` lines |
| `sf33b-compare` | CPU | 0 (lock only) | 00:53:58Z | 00:53:59Z | 0 | `compare_proto.py ../logs/proto_b/*.log --exports ../exports --out ../raw/proto_b/compare_proto_b.md` (exit 1 = some case FAIL, by design); `digest_newton.py ../raw/proto_b/*.json > ../raw/proto_b/digest_b.txt` (exit 0) |

Retrieval: `tar czf ~/sf33b_evidence.tgz logs/proto_b raw/proto_b jobs/` on the host, `scp` to the worktree,
extracted into this artifact (text only; no binaries, no exports).

Script changes in N6b: none. `run_proto.sh` writes to `logs/proto/` only and passes `--solution` only for the
SF-29 16^3 solutions, so the run used an inline loop (as N6's `sf33-proto-24ref`).
