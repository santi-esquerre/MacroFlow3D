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
