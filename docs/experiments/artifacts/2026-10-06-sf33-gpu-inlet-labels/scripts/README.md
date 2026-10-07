# SF-33 prototype-input exporter (`export_proto.py`)

Purpose: give the GPU inlet-label solver of SF-33 (`src/physics/streamfunctions/inlet_slab/`) the
**same inputs** as the SF-29 CPU prototype, so that claim (a) of SF-33 ("the GPU solves the same discrete
problem: identical inputs give identical metrics") is tested on identical data (step 9a), and provide the
spectral reference data of the SF-19 cross-check (step 8). Not for production fields (step 9b uses the
SF-18/SF-19 stack, not this script).

## Provenance and read-only rule

- Source: `docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/scripts/` (`cases.load_case`,
  `cases.build_reference`, `candidate_i.light_case`, `candidate_i.amplitude_path`, `candidate_i._nphi_light`,
  `metrics.fd_metrics`, `metrics.rms_vec`), imported **read-only** by `sys.path` insertion, exactly as the
  SF-29 `reference.py` imports the 2026-10-02 closure probes. No SF-29 file is modified.
- The only writes into the SF-29 tree are the oracle caches that `cases.load_case` writes by design into
  `raw/cache/`, and only when that file is gitignored there (`*.npz` except `!*_N16_*`): a missing **N = 16**
  cache (e.g. `gauss:0.5:16`) is computed but **not** written, so the SF-29 tree stays clean in git. 24^3 / 32^3
  caches (minutes of CPU each) are written and reused (`case.json: cache_written`).
- numpy/scipy only; numpy 1.26-compatible (V100 host: Python 3.11.7 / numpy 1.26.4 / scipy 1.11.4; local:
  Python 3.13 / numpy 2.x / scipy 1.18). Versions are recorded in every manifest.
- Outputs go to `../exports/` (gitignored, regenerated deterministically); only small text outputs are ever
  committed.

## Usage (run from this directory)

```bash
python3 export_proto.py --selftest                                   # < 1 min locally (committed 16^3 cache)
python3 export_proto.py --list-amplitudes 0.25 0.5 1.0               # reachable continuation amplitudes + rule
python3 export_proto.py gauss:0.25:16 [field:eps:N ...] --out ../exports
python3 export_proto.py --solutions ../../2026-10-02-sf29-inlet-labels/raw/sweep2/solutions/gauss_0.25_16_i1o4.npz \
        --out ../exports/solutions
python3 export_proto.py --crosscheck gauss:0.25:16 gauss:0.25:24 gauss:0.25:32 --out ../exports
```

Options: `--bisect K` (bisection budget of the enumerated continuation, default 4 = `candidate_i.py` default),
`--no-stages` (main files only; the driver then fails with `missing_stage_input` on any bisection).

C++ consumers: `src/physics/streamfunctions/inlet_slab/NpyIo.hpp` (reader/writer),
`ProtoCase.cuh` (`load_proto_case`, `ProtoStageProvider`, `load_solution`), and the executable
`inlet_slab_proto_check` (`--metrics-of-oracle`, `--residual-of-solution`, `--metrics-of-solution`, `--stages`).

## Driver `inlet_slab` and campaign scripts (SF-33 N5)

The executable `inlet_slab` (`apps/inlet_slab/`, built with the other targets; documented-experiment instrument,
not a ctest entry) consumes these exports. Run from the repository root:

```bash
# claim (a): prototype reproduction (step 9a); --solution adds the field-wise FIELDDIFF vs the saved 16^3 solution
./build/wsl-debug/inlet_slab --proto <exports>/gauss_0.25_16 --solution <exports>/solutions/gauss_0.25_16_i1o4 \
        --summary /tmp/gauss16.json
# claim (b): production stack (SF-18 field or --analytic <closure field>), N3 stages, continuation, oracle
./build/wsl-debug/inlet_slab --production --n 32 --eps 0.5 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder
./build/wsl-debug/inlet_slab --production --n 16 --eps 0.25 --analytic generic3d --oracle-ladder
# step 8: SF-19 inlet-face v1 / spline-flow v_perp vs the spectral reference of the gauss field
./build/wsl-debug/inlet_slab --sf19-crosscheck <exports>/crosscheck_gauss_0.25_16
```

Solver options (all modes that solve): `--lin-tol 1e-12 --restart 100 --max-inner 6000 --newton-tol 1e-13
--max-newton 120 (40 with --psitc off) --bisect 4 --prec pa` (P-A is the only preconditioner); linear forcing (SF-33 N7a) `--forcing ew`
(default: inexact Newton, Eisenstat-Walker choice 2, gamma 0.9, alpha 2, `--ew-eta0 0.1`, `--ew-eta-max 0.1`,
eta_min = `--lin-tol`, oversolving guard 0.5 newton_tol / merit; every NEWTON line prints `eta=`, the SOLVER line and
`GMRES_STATS` record the policy and the per-stage eta sequence) or `--forcing fixed` (every Newton system to
`--lin-tol`: the N2-N6 behaviour, bitwise; use it for iterate-history reproduction checks); pseudo-transient
continuation (SF-33 N7b) `--psitc on` (default: every Newton system shifted, `(J + mu_k D) p = -E`, `D = q_v / h^2`
on the equation rows, 0 on the outlet rows, SER `mu_k = clamp(mu0 m_k / m_0, 0, mu_max)` on the merit norm of the
stage's start state, `--psitc-mu0 1`, `--psitc-mu-max 100`; P-A factored for the shifted operator; line-search
failure -> mu x4 (clamped) and re-solve, at most 4 times; every LINEAR / NEWTON line prints `mu=`, the SOLVER line,
`solver_config` and `GMRES_STATS` (`mus=`, `^` = retry) record it) or `--psitc off` (mu = 0: the N7a iteration
bitwise); `--save-solution <dir>` writes `u1.npy`,
`u2.npy`, `solution.json` in the `--solutions` layout above; `--summary <json>` writes every reported number at full
precision. Production oracle: `--oracle-hmax-div 8` (`h_max = h/8`, orchestrator decision: the SF-30 default `h`
gives step-limited round trips ~1e-6 at 16^3), `--oracle-tol 1e-8`, `--oracle-max-roundtrip 1e-8` (acceptance (d):
a plane above it, or any non-ok streamline, is `oracle_roundtrip_fail`), `--oracle-ladder` (adds `(h/16, tol)` and
`(h/16, 1e-10)` runs, per-plane `ORACLE` tables, `ORACLE_LABELDIFF` vs the primary run), `--threads T`
(default min(cores, 32)), `--no-oracle`.

Output lines (prototype formats where the prototype has one): `STAGE` / `NEWTON` / `LINEAR` / `STAGE_END` /
`CONTINUATION` / `PATH` (N2), `GMRES_STATS` (per stage), `CASE ... cand=i1o4` and the ceiling `cand=oracle_fd4`,
`EXTRA`, `HISTORY` (`%.2e`) and `HISTORY_FULL` (`%.17e`), `FIELDDIFF` (per-field and joint max relative
difference, absolute differences), production `SETUP` (SF-18 / SF-19 / inlet report of every stage), `ORACLE*`,
`TIMING`, `MEMORY` (`cudaMemGetInfo` polls around every phase, device-wide, plus the workspaces' bytes),
`STATUS <name>`. Exit codes (distinct, nothing clamped): 0 converged (crosscheck: ok), 1 exception, 2 usage,
10 linesearch-fail, 11 stagnation, 12 maxit, 13 linear_failure, 14 nan_inf, 15 continuation_floor,
16 missing_stage_input, 17 inlet_backflow, 18 darcy_failed, 19 oracle_roundtrip_fail (production: reported only
when the solver converged; `STATUS_DETAIL` gives both).

Campaign scripts (plain bash, parameterised by the build dir, print every command; detached V100 jobs):

```bash
scripts/remote --increment SF-33 run sf33-proto -- \
  "bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_proto.sh build/v100-release"
scripts/remote --increment SF-33 run sf33-crosscheck -- \
  "bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_crosscheck.sh build/v100-release"
scripts/remote --increment SF-33 run sf33-ladder-0.5 -- \
  "bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_ladder.sh build/v100-release 0.5"
```

- `run_proto.sh <build_dir> [<out_root>]`: the step-9a matrix (`gauss`, `gauss_ch`, `control2d` x eps 0.25, 0.5 x
  N 16, 24; N 32 for `gauss:0.25`, `gauss_ch:0.25`; `generic3d` eps 1 at N 16, 20, 24 and eps 0.25 / 0.5 at 16);
  exports when absent, `--solution` for every case with a saved 16^3 prototype solution; logs
  `<out_root>/logs/proto/<case>.log`, JSON `<out_root>/raw/proto/<case>.json`, table
  `<out_root>/raw/proto/compare_proto.md`. `CASES="field:eps:N ..."` overrides the matrix. Extra driver options
  (e.g. `--forcing fixed`) via `EXTRA_ARGS="..."` or after a literal `--`: `run_proto.sh <build> <out> -- --forcing
  fixed` (appended to every driver call; use a distinct `<out_root>` per policy, the log names do not include it).
- `run_crosscheck.sh <build_dir> [<out_root>]`: `--crosscheck gauss:0.25:{16,24,32}` + `--sf19-crosscheck`; logs
  `logs/crosscheck/N<N>.log`, table with observed orders `raw/crosscheck/crosscheck.md`.
- `run_ladder.sh <build_dir> <eps> [<out_root>]`: production `N = 32, 64, 128` (`NS` overrides), `--sigma2 1
  --ell 0.25 --seed 3001 --oracle-ladder --threads 32`; logs `logs/ladder_<eps>/N<N>.log`, JSON
  `raw/ladder_<eps>/`, table `raw/ladder_<eps>/ladder_orders.md`.
- `compare_proto.py LOG... [--summary <SF-29 sweep2 summary.md>] [--exports ../exports] [--out md]`: GPU vs
  prototype per case (full precision from `solution.json` / `ref_metrics.json` gated at 1e-6 relative; 4-digit
  prototype values gated at their rounding bound and labelled `PASS(4dig)`), STATUS (converged) and r_F gates,
  PATH equality (informational since SF-33 N7b: never part of the verdict), FIELDDIFF, r_F history agreement,
  GMRES statistics per stage.
- `digest_newton.py JSON...` (SF-33 N7b): per-stage Newton / GMRES digest of the driver's `--summary` JSON
  (status, steps, r_F history, eta and mu sequences, GMRES total / max / median per solve, per-restart curves of
  every failed linear solve). `--crosscheck LOG...`: the step-8 table and orders.
- SF-33 N7c (probe, `SlabCoarseCorrection.cuh`): driver options `--coarse off|add|mult` (default off = the N7b
  solver bitwise), `--coarse-profiles 1|2` (x1-constant, + x1-linear column profiles; K = 2 P N^2),
  `--coarse-assembly direct|colored` (K vs p^2 2 P operator applications, bitwise-equal E),
  `--coarse-factor dense|banded` (host dense LU vs banded LU in the folded m2 ordering; banded needs colored); `COARSE
  build` / `COARSE apply` lines per Newton step. Mode `--linear-probe <case_dir> --eps-stage E --newton-steps k
  [--eps-from A] [--probe-ladder 0.25,0.375] [--probe-precs pa,mult1,mult2,add1] [--probe-tol 1e-8] [--probe-mu
  both|ser|zero] [--probe-stagnation 1]`: continuation to A (driver policy), k Newton steps of E, then the FROZEN
  Jacobian solved with every listed preconditioner at mu_SER and mu = 0 (`PROBE` / `PROBE_CURVE` lines, JSON).
  `run_linear_probe.sh <build_dir> [<out_root>]` runs the N7c probe matrix (`PROBES=...` overrides; logs
  `logs/n7c_local/probe/`, JSON `raw/n7c_local/probe/`); `digest_probe.py JSON... [--curves]` tabulates them.
- `ladder_orders.py LOG...`: per-grid table (status, r_F, PATH, metrics, ceiling, oracle round trips, GMRES,
  timing, memory), observed orders, the acceptance-(b) reading, SF-18 applied-scale consistency.

Default `<out_root>` is this artifact directory (`exports/` is gitignored; `logs/` and `raw/` are the outputs to
inspect and, after review, commit as small text).

## Output layout

All arrays are float64, C order, NumPy `.npy` (version 1.0 header).

### Case directory `<field>_<eps:%g>_<N>/` (from `cases.load_case(field, eps, N)`)

| file | shape | content |
|---|---|---|
| `case.json` | - | field, eps, N, nphi, nf, Q0, inlet_vmin, L, a, pcg_res, pcg_its, max_d1phi, cache_hit / cache_file / cache_written, roundtrip per plane + max, v_rms, amplitude rule, `stage_amplitudes`, per-stage source / nphi / v_rms / time, versions, export time, source artifact |
| `lnk.npy` | (N+1, N, N) | analytic ln k at the vertices |
| `grad_lnk_{1,2,3}.npy` | (N+1, N, N) | analytic grad ln k (complex step) at the vertices |
| `vD_{1,2,3}.npy` | (N+1, N, N) | spectral reference Darcy velocity at the vertices |
| `psi0_{1,2}.npy` | (N, N) | inlet labels, **full** labels (affine part included) |
| `vperp_{2,3}.npy` | (N, N) | (v2, v3) of the Darcy velocity at the inlet vertices |
| `psi_or_{1,2}.npy` | (N+1, N, N) | DOP853 oracle labels, **full** labels; `psi_or_i[0] == psi0_i` |
| `v_rms.npy` | () | `metrics.rms_vec(vD)` (= `candidate_i.Ctx.v_rms`) |
| `ref_metrics.json` | - | `metrics.fd_metrics(psi_or, ..., order=4)` (`cand=oracle_fd4`) at full precision + its CASE line |
| `stage/<amp:%g>/` | | `lnk`, `grad_lnk_{1,2,3}`, `psi0_{1,2}`, `vperp_{2,3}`, `v_rms` for every reachable amplitude |

Slab layout: full arrays are indexed `[j, m2, m3]` (vertex `(j/N, m2/N, m3/N)`, **m3 fastest**), plane arrays
`[m2, m3]`; this is the LOCKED layout `idx = m3 + N*(m2 + N*j)` of `InletSlabGrid.cuh`, so the C++ loader reads
them without transposition. The loader forms `q = 1/exp(lnk)` and the inlet periodic parts
`u0_1 = psi0_1 - m2/N`, `u0_2 = psi0_2 - m3/N` exactly as `candidate_i.Ctx` does.

Stage inputs: for an amplitude `a != eps` the files are `candidate_i.light_case(field, a, N,
nphi=_nphi_light(field, a))` (what `solve_case` uses for intermediate stages); for `a == eps` they are copies
of the main files (what `solve_case` uses at the target), so the loader has one code path.

### Reachable continuation amplitudes (the rule)

`candidate_i.solve_case`: `todo = amplitude_path(eps)` = ladder `(0.25, 0.5, 1.0)` entries `< eps` then `eps`;
`e_conv = 0`. A stage at `todo[0]` that is accepted sets `e_conv` and pops it; a failed stage, while fewer than
`bisect = 4` bisections were used **in total**, inserts `0.5 (e_conv + failed)` at the front; with the budget
exhausted the continuation gives up and makes one final attempt at `eps`. The data-dependent outcomes are
replaced by an exhaustive recursion over (todo, e_conv, bisections used); the union of all attempted amplitudes
is exported. All are dyadic (exact in float64 and in `%g`). Result:

- `eps = 0.25`: 16 amplitudes, `k/64`, k = 1..16;
- `eps = 0.5`: 32 amplitudes, `k/64`, k = 1..32;
- `eps = 1`: 48 amplitudes, `k/64` for k = 1..32 and `0.5 + k/32` for k = 1..16.

`--selftest` checks every PATH line of `raw/sweep2/summary.md` (151 lines, 47 of them `i1o4`, plus `i1` and
`i1o4_fd2`): every amplitude lies in the set, and replaying the rule with the PATH's accept/fail outcomes
reproduces the PATH string exactly. A driver that needs an amplitude outside the exported set fails with the
distinct status `missing_stage_input` (`MissingStageInput` in `ProtoCase.cuh`).

### Saved solutions `--solutions FILE.npz` -> `<out>/<name>/`

`u1.npy`, `u2.npy` (N+1, N, N): periodic parts `psi1 = x2 + u1`, `psi2 = x3 + u2` on planes 0..N (plane 0 is
the inlet data u0), as saved by `candidate_i.save_solution`; `solution.json`: field, eps, N, variant, order,
cand, status, its, r_F, r_out, t, path, hist, metrics_json (the CASE metrics of that solution), opts_json,
versions.

### SF-19 cross-check inputs `--crosscheck field:eps:N` -> `<out>/crosscheck_<field>_<eps:%g>_<N>/`

Triply periodic fields only (SF-19 is periodic); `_ch` fields are refused.

| file | shape | layout / content |
|---|---|---|
| `Y_cells.npy` | (N, N, N) | `ln k` at the cell centres `((i+1/2)h, (j+1/2)h, (k+1/2)h)`, array indexed **`[k, j, i]`**: the flat C-order index is `i + N*(j + N*k)` = `Grid3D::idx`, i.e. the **PROJECT cell layout, x fastest** (opposite of the slab arrays) |
| `v1_face_ref.npy` | (N, N) | `[m2, m3]`: spectral point value `v1(0, (m2+1/2)h, (m3+1/2)h)` (inlet-face centres) |
| `v1_faceavg_ref.npy` | (N, N) | `[m2, m3]`: Gauss-Legendre 3x3 average of `v1` over the face cell `[m2 h, (m2+1) h] x [m3 h, (m3+1) h]` of `x1 = 0` (comparable to an SF-19 face flux) |
| `vperp_vertex_ref_{2,3}.npy` | (N, N) | `[m2, m3]`: `(v2, v3)(0, m2 h, m3 h)` at the inlet vertices (slab layout, m3 fastest) |
| `crosscheck.json` | - | nphi, a, pcg, Q0, mean of `v1_faceavg_ref` (1 to the reference tolerance), layouts, versions |

## Timings (local WSL, 16 cores shared, 2026-10-06)

- `--selftest`: 27 s (export of `gauss:0.25:16` with 16 stage amplitudes from the committed cache: 24 s).
- `gauss:0.25:16 --out ../exports`: 17-25 s (main case 1.3 s from cache; ~1.5 s per `light_case` stage at
  `nphi = 32`). 24^3/32^3 exports need the oracle (minutes per case) and run as the remote N6 job.
- `--crosscheck gauss:0.25:16`: 2 s.
