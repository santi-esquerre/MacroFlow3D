# SF-29 inlet labels -- scripts and raw outputs

Artifacts for `docs/experiments/2026-10-02-sf29-inlet-labels.md` (increment SF-29, CPU prototype of
equation (14) with `x1` non-periodic and labels fixed on the inlet face).

Diagnostic numpy/scipy code. Not production code, not tests, not built or run by CMake/ctest; no project
binary. CPU, double precision. Runs on Python 3.13 / numpy 2.5.0 / scipy 1.18.0 (local) and on
Python 3.11 / numpy 1.26.4 / scipy 1.11.4 (V100 host versions; `--selftest` checked locally in a
Python 3.11.16 venv with exactly numpy 1.26.4 / scipy 1.11.4).

This node (N1) provides the label-independent reference: the Darcy flow, the inlet labels, and the oracle
labels at every vertex of a candidate grid, plus the shared conventions used by the candidate scripts.

## Scripts (`scripts/`, run from that directory)

| script | role |
|---|---|
| `oracle.py` | entry point: `--selftest`, oracle per case, `--convergence`, `--nphi`, `--returnmap`, `--midplane` |
| `reference.py` | Darcy reference: `darcy_spectral_box` (closure-probe solver generalized to a box), `SepTrigField` (exact trigonometric `grad phi`), `LogConductivity` (analytic `ln k`, complex-step gradient), `DarcyReference` (periodic fields and the constant-head mirror case) |
| `inlet.py` | `InletLabels`: normalized triangular inlet labels (deviation D-1) as spectral representations evaluable at arbitrary face points |
| `tracing.py` | DOP853 streamline tracing parametrized by `x1` (backward to the inlet, round trip, closure-note return map) |
| `cases.py` | `load_case(field, eps, N)`, `NPHI` table, face fluxes, cache |
| `metrics.py` | shared metrics (`fd_metrics`), the one-line `CASE` print format (`case_line`, `print_case`, `parse_case_line`), observed orders |

### Reuse of the 2026-10-02 closure probes

`reference.py` imports `docs/experiments/artifacts/2026-10-02-closure-probes/scripts/closure_probe.py` by path
(read-only; that artifact is not modified) for `FIELDS`, `wavenumbers`, `darcy_spectral` and `TrigField`.
`darcy_spectral_box` is a copy of `darcy_spectral` generalized to a box `[0,L1]x[0,L2]x[0,L3]` with an
`(N1, N2, N3)` grid; on the unit cube it skips the `1/L` scaling and is bit-identical to the original
(`--selftest` compares `u`, residual, iterations and flux with `np.array_equal`). `SepTrigField` evaluates the
same trigonometric interpolant as `TrigField` (Nyquist planes dropped, but no small-coefficient cut) in separable
form, which makes per-plane tracing affordable at `N_phi = 48`; `--selftest` checks it against `TrigField`
(4e-16).

## Darcy reference

- Periodic fields (`control2d`, `lester2021`, `lester_brk`, `two_mode`, `generic3d`, `gauss`): triply periodic cell
  flow of `k = exp(eps f)`, three cell problems, mean gradient `Keff^-1 e1`, mean flux exactly `e1`
  (as `closure_probe.run`). `phi = -a.x + u`.
- Constant-head field `gauss_ch` (deviation D-4; generally `<field>_ch`): `k_b(x) = exp(eps f(g(x1), x2, x3))`,
  `g(x1) = (1 - cos(pi x1))/2`, periodic cell `[0, 2] x [0, 1]^2`, grid `2 N_phi x N_phi x N_phi`, one cell
  problem with mean gradient along `x1`, scaled so that the mean flux (through every `x1`-plane) is 1. By the
  symmetry `x1 -> -x1` (and `x1 -> 2 - x1`) the fluctuation potential is odd, hence `phi` is constant and
  `v_perp = 0` on `x1 = 0` and `x1 = 1`; the slab `[0, 1]` carries the constant-head Darcy flow. Verified by
  `--selftest` and per case (`CHECK ... constant-head faces`).
- `uniform` (`k = 1`) is a positive control (affine labels).
- `v1 > 0` is checked per case on the reference grid (`max d1 phi < 0`); a violating case aborts with an
  `ABORT case ...` message.
- Reference-quality diagnostics printed per case: PCG residual, `Q0 - 1` (inlet-face flux minus 1), and
  `cont_res` = RMS of `lap u + grad ln k . grad phi` at 4096 random slab points over
  `RMS|grad ln k| RMS|grad phi|` (the continuum mass-balance error of the collocation solve).

### Reference resolution `N_phi(field, eps)`

Chosen independently of the candidate grid `N` by `oracle.py --nphi field:eps`: ladder `N_phi` in
16, 24, 32, 48, 64, each compared with `1.5 N_phi`; the first `N_phi` whose inlet labels and outlet oracle labels
(at the 16 x 16 outlet vertices `x1 = 1`) differ from those at `1.5 N_phi` by `< 1e-8` relative is recorded:
`rel = max_i RMS(psi_i(N_phi) - psi_i(1.5 N_phi)) / max_i RMS(psi_i(1.5 N_phi) - affine_i)`.
For `_ch` fields `N_phi` is the transverse resolution (the `x1` grid of the length-2 cell is `2 N_phi`).
The table is recorded in `scripts/cases.py::NPHI`; raw rows in `raw/oracle_nphi_table.txt`.

| field | eps | N_phi | inlet rel diff (N_phi vs 1.5 N_phi) | outlet rel diff | cont_res at N_phi | Q0-1 at N_phi | previous rung (N_phi: inlet / outlet) | t_ref N_phi / 1.5 N_phi [s] |
|---|---|---|---|---|---|---|---|---|
| `control2d` | 0.25 | **24** | 2.24e-09 | 2.24e-09 | 5.6e-08 | +4.4e-13 | 16: 1.14e-05 / 1.14e-05 | 0.3 / 1.0 |
| `control2d` | 0.5 | **32** | 1.86e-11 | 1.86e-11 | 1.7e-09 | +3.9e-14 | 24: 8.35e-08 / 8.35e-08 | 3.1 / 11.6 |
| `control2d` | 1 | **32** | 3.62e-09 | 3.63e-09 | 2.0e-07 | +7.2e-11 | 24: 2.92e-06 / 2.92e-06 | 5.7 / 20.3 |
| `lester2021` | 0.25 | **48** | 2.13e-12 | 2.51e-12 | 8.9e-11 | -4.4e-16 | 32: 1.34e-08 / 1.34e-08 | 5.0 / 9.9 |
| `lester2021` | 0.5 | **48** | 2.42e-11 | 2.43e-11 | 2.9e-09 | +4.4e-16 | 32: 2.12e-07 / 2.12e-07 | 6.2 / 14.2 |
| `lester2021` | 1 | **48** | 1.48e-09 | 1.48e-09 | 9.3e-08 | +2.2e-16 | 32: 3.22e-06 / 3.22e-06 | 12.6 / 29.2 |
| `lester_brk` | 0.25 | **48** | 2.09e-12 | 2.97e-11 | 9.3e-11 | +0.0e+00 | 32: 4.95e-08 / 2.96e-07 | 4.3 / 9.3 |
| `lester_brk` | 0.5 | **48** | 1.34e-10 | 2.93e-09 | 3.2e-09 | +4.4e-16 | 32: 7.91e-07 / 8.85e-06 | 6.0 / 30.5 |
| `lester_brk` | 1 | **64** | 3.25e-12 | 9.92e-11 | 9.6e-11 | +0.0e+00 | 48: 8.52e-09 / 1.58e-07 | 24.8 / 56.8 |
| `two_mode` | 0.25 | **16** | 3.04e-10 | 2.87e-10 | 2.1e-09 | +0.0e+00 | - | 0.2 / 0.7 |
| `two_mode` | 0.5 | **24** | 8.61e-14 | 1.03e-13 | 1.7e-12 | +0.0e+00 | 16: 4.10e-08 / 3.75e-08 | 1.2 / 4.1 |
| `two_mode` | 1 | **24** | 1.85e-10 | 2.07e-10 | 3.1e-09 | -4.4e-16 | 16: 5.17e-06 / 5.66e-06 | 2.8 / 7.8 |
| `generic3d` | 0.25 | **24** | 3.48e-10 | 3.73e-10 | 1.0e-08 | +3.1e-15 | 16: 2.52e-06 / 2.68e-06 | 0.9 / 3.2 |
| `generic3d` | 0.5 | **32** | 5.94e-12 | 6.25e-12 | 3.4e-10 | +1.3e-15 | 24: 1.63e-08 / 1.56e-08 | 4.6 / 16.1 |
| `generic3d` | 1 | **32** | 4.84e-09 | 4.86e-09 | 9.2e-08 | +1.4e-12 | 24: 2.09e-06 / 1.88e-06 | 20.8 / 83.1 |
| `gauss` | 0.25 | **32** | 8.43e-11 | 8.84e-11 | 8.5e-10 | -3.6e-12 | 24: 3.97e-08 / 4.79e-08 | 2.2 / 7.1 |
| `gauss` | 0.5 | **32** | 2.24e-09 | 4.08e-09 | 3.5e-08 | -2.1e-10 | 24: 7.09e-07 / 1.10e-06 | 4.0 / 14.2 |
| `gauss` | 1 | **48** | 1.87e-12 | 3.79e-11 | 1.4e-10 | -6.7e-14 | 32: 5.59e-08 / 9.03e-07 | 57.3 / 142.4 |
| `gauss_ch` | 0.25 | **48** | 2.19e-10 | 2.08e-10 | 2.9e-09 | +8.1e-12 | 32: 4.36e-07 / 4.24e-07 | 4.2 / 10.9 |
| `gauss_ch` | 0.5 | **48** | 2.69e-09 | 2.36e-09 | 7.7e-08 | -2.9e-11 | 32: 5.39e-06 / 4.70e-06 | 8.2 / 21.4 |
| `gauss_ch` | 1 | **64** | 1.20e-10 | 8.03e-11 | 6.9e-09 | +6.6e-12 | 48: 8.96e-08 / 5.86e-08 | 66.3 / 183.3 |

`N_phi > 48` is needed for `lester_brk` at `eps = 1` and `gauss_ch` at `eps = 1` (both 64; at 48 their outlet
labels differ from 72 by 1.6e-7 and 5.9e-8). `gauss` at `eps = 1` passes at exactly 48 (outlet 3.8e-11; at 32:
9.0e-7). Timings are local WSL (16 cores, shared machine), one process, numpy multithreaded.

## Inlet labels (deviation D-1)

On `x1 = 0`, with `v1 = v1(0, x2, x3)` sampled on an `nf x nf` face grid (`nf = 2 N_phi`) and replaced by its 2-D
trigonometric interpolant `V` (Nyquist lines dropped):

```text
Q(x3)   = int_0^1 V(s, x3) ds                       -> coefficients V(0, m3)
psi2^0  = int_0^{x3} Q = Q0 x3 + periodic            (Q0 = mean face flux = 1 - O(reference error))
psi1^0  = int_0^{x2} V(s, x3) ds / Q(x3) = x2 + num(x2, x3)/Q(x3)
```

`num` and the periodic part of `psi2^0` are exact Fourier sums, so both labels are evaluated at arbitrary
(unwrapped) face points. The face-Jacobian identity `d2 psi1^0 d3 psi2^0 - d3 psi1^0 d2 psi2^0 = V` holds at
roundoff for the representation (analytic derivatives), and against the true `v1` to the spectral accuracy of the
face sampling (independent check: FFT differentiation of sampled labels on a `2 nf` grid). The `psi2^0` jump is
`Q0`, not exactly 1: `|Q0 - 1|` is a reference-resolution diagnostic (|Q0 - 1| <= 2.1e-10 at the tabulated `N_phi`, table above), and all
FD metrics use the affine parts `x2`, `x3` exactly.

## Oracle labels

Vertex grid of the slab: `x1 = j/N` (`j = 0..N`), `x2, x3 = m/N` (`m = 0..N-1`), arrays `(N+1, N, N)`. For each
plane `j >= 1` all `N^2` vertices are traced backward to `x1 = 0` along `dx_perp/dx1 = grad_perp phi / d1 phi`
(vectorized per plane, DOP853 rtol 1e-12, atol 1e-14), the inlet labels are evaluated at the feet, and the feet are
re-integrated forward to report a per-plane round trip (`max` norm) and `nfev`. Plane `j = 0` is the inlet data.

## Shared conventions for the candidate nodes

- CLI case spec: `field:eps:N` (`cases.parse_spec`).
- `cases.load_case(field, eps, N, nphi=None, use_cache=True)` returns a dict:
  - `k`, `lnk`: `(N+1, N, N)`, analytic at vertices; `grad_lnk`: 3 arrays `(N+1, N, N)` (complex step, exact);
  - `vD`: 3 arrays `(N+1, N, N)`, reference Darcy velocity at vertices;
  - `vD_faces`: `{'f1': (N+1, N, N), 'f2': (N, N, N), 'f3': (N, N, N)}`, face-averaged normal Darcy flux on the
    cell faces of the vertex grid (Whitney/MAC 2-form DoFs): `f1[j, m2, m3]` on the `x1`-face in plane
    `x1 = j/N` spanning `[m2, m2+1]/N x [m3, m3+1]/N`; `f2[j, m2, m3]` on the `x2`-face in plane `x2 = m2/N`
    spanning `[j, j+1]/N x [m3, m3+1]/N`; `f3[j, m2, m3]` on the `x3`-face in plane `x3 = m3/N` spanning
    `[j, j+1]/N x [m2, m2+1]/N`. Gauss-Legendre 3 x 3 per face (error `O(h^6)`); the cell flux balance
    `cases.face_divergence` vanishes to that accuracy plus the reference error;
  - `psi0`: 2 arrays `(N, N)`, inlet labels at the inlet-face vertices;
  - `psi_or`: 2 arrays `(N+1, N, N)`, oracle labels at every vertex (`psi_or[i][0] == psi0[i]`);
  - `vperp_in`: 2 arrays `(N, N)`, `(v2, v3)` of the Darcy velocity at the inlet-face vertices (0 for `_ch`);
  - `meta`: field, eps, N, nphi, nf, `L`, mean gradient `a`, PCG residual/iterations, `max_d1phi`, `Q0`,
    per-plane `roundtrip`/`nfev`/`nfev_rt`, timings, cache path;
  - `ref`, `inlet`: the live `DarcyReference` / `InletLabels` (e.g. `ref.velocity(x1, y, z)`,
    `ref.lk.k/lnk/grad_lnk(X1, X2, X3)` at arbitrary points, `inlet.labels(y, z)`).
- Label unknowns: `psi1 = x2 + u1`, `psi2 = x3 + u2`, `u_i` periodic in `(x2, x3)`.
- Metrics: `metrics.fd_metrics(psi1, psi2, vD, psi_or)` (second-order centered FD in `x2, x3`; in `x1` centered on
  interior planes and second-order one-sided on the inlet and outlet planes) returns `e_v`, `e_psi`, `e_i1`,
  `e_i2`, `e_div`, `min_c`, percentiles `p0.1/p1/p5/p50` of `|c|`, and the same percentiles of `|v_D|`
  (`vD_p`). Candidates with their own `c` (e.g. face fluxes) compute `e_v` against `vD_faces` and pass it in the
  metrics dict.
- One-line parseable format (`metrics.case_line` / `print_case`; `parse_case_line` inverts it):

  ```text
  CASE field=<f> eps=<e> N=<N> cand=<name> | r_F=<..> its=<..> | e_v=<..> e_psi=<..> e_i=(<..>,<..>) e_div=<..> min_c=<..> p0.1=<..> p1=<..> p5=<..> p50=<..> | t=<s>
  ```

  Missing values print as `nan` (the oracle prints `r_F=nan its=nan`, `e_psi=0`).
- No regularization of `|c|^2` anywhere.

## Cache

`raw/cache/<field>_eps<eps>_N<N>_nphi<nphi>_v<version>.npz` holds `vD`, `psi_or`, face fluxes, inlet `v_perp`,
per-plane round trips and `nfev`; `k`, `lnk`, `grad_lnk` are recomputed. Files at `N = 16` (0.28 MB) are
committed; files at `N >= 24` (2.2 MB at 32^3, 7.2 MB at 48^3) are gitignored (`raw/cache/.gitignore`) and regenerated
deterministically. A cache hit still rebuilds the reference (one spectral solve). Delete the file or pass
`--no-cache` to recompute.

## Commands (N1)

```bash
cd docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/scripts
python3 oracle.py --selftest                                                    > ../raw/oracle_selftest.txt
python3 oracle.py control2d:0.5:16 lester2021:1.0:16 gauss:0.25:16 gauss_ch:0.25:16 > ../raw/oracle_cases16.txt
python3 oracle.py --convergence gauss:0.25 gauss_ch:0.25                        > ../raw/oracle_convergence.txt
python3 oracle.py --nphi gauss:1.0 gauss_ch:0.25                                > ../raw/oracle_nphi_validation.txt
python3 oracle.py --nphi <all 7 fields x eps 0.25, 0.5, 1>                      > ../raw/oracle_nphi_table.txt
python3 oracle.py --convergence gauss:1.0                                       > ../raw/oracle_convergence_gauss1.txt
python3 oracle.py --returnmap control2d:0.5 lester2021:1.0 lester_brk:1.0 gauss:0.25 gauss:0.5 gauss:1.0 > ../raw/oracle_returnmap.txt
python3 oracle.py --midplane gauss:0.25 gauss:0.5 gauss:1.0 gauss_ch:0.25 gauss_ch:1.0 lester_brk:1.0 generic3d:1.0 \
        --grids 16,24,32,48,64,96,128                                           > ../raw/oracle_midplane.txt
/path/to/py311/bin/python oracle.py --selftest                                  > ../raw/oracle_selftest_py311.txt
```

CLI summary:

| invocation | output |
|---|---|
| `oracle.py --selftest` | `SELFTEST` rows (bit-identity with the closure probe, trig evaluation, complex-step `grad ln k`, `k = 1` affine labels, face-Jacobian identity, label jumps, `Q0`, tracer round trip, constant-head faces, face-flux balance, CASE-line round trip); exit 1 on any failure |
| `oracle.py field:eps:N ...` | per case: `REF`, `ORACLE` (round trip, `nfev`, timings), `CHECK` (plane fluxes, face-flux balance, constant-head faces or closure-note return map, outlet-vs-inlet labels, `|c|` and `|v_D|` percentiles), `CASE ... cand=oracle` |
| `oracle.py --convergence field:eps ...` | the above for `N` in `--grids` (default 16,32,48) and `ORDER` rows (observed orders of `e_v`, `e_i1`, `e_i2`, `e_div`; `min|c|`) |
| `oracle.py --nphi field:eps ...` | `NPHI` ladder rows and the chosen `NPHI_TABLE` row |
| `oracle.py --returnmap field:eps ...` | `RETURNMAP` row: closure-note return map (16 points, seed 3) at `N_phi`, next to the value recorded in the 2026-10-02 probes |
| `oracle.py --midplane field:eps ... --grids ...` | `MIDPLANE` rows: oracle labels on the planes `0.5 - h, 0.5, 0.5 + h` only, centered FD `e_v_mid`, `e_i_mid`, `max|grad_h psi|`, `min|c|`; `MIDPLANE_ORDER` |
| options | `--ref-nphi K` (override `N_phi`), `--grids 16,32,48`, `--no-cache` |

## Raw outputs (`raw/`)

| file | content |
|---|---|
| `oracle_selftest.txt` | `--selftest` (local Python 3.13 stack) |
| `oracle_selftest_py311.txt` | `--selftest` under Python 3.11.16 / numpy 1.26.4 / scipy 1.11.4 |
| `oracle_cases16.txt` | oracle at 16^3 for `control2d:0.5`, `lester2021:1.0`, `gauss:0.25`, `gauss_ch:0.25` |
| `oracle_convergence.txt` | oracle self-consistency over 16/32/48 |
| `oracle_nphi_validation.txt` | `--nphi gauss:1.0 gauss_ch:0.25` |
| `oracle_nphi_table.txt` | the full `N_phi` table (run before the `phase_table` recurrence optimization of `SepTrigField`; identical to ~2e-16) |
| `oracle_convergence_gauss1.txt` | oracle self-consistency `gauss:1.0` over 16/32/48 |
| `oracle_returnmap.txt` | closure-note return maps at `N_phi` vs the recorded `raw/closure.txt` values |
| `oracle_midplane.txt` | 3-plane mid-slab FD self-consistency over N = 16..128 (asymptotic-regime probe) |
| `cache/` | oracle caches (16^3 committed) |

## Timings and cost model (local WSL, 16 cores shared, numpy multithreaded, one process)

Oracle (backward tracing of all planes plus the forward round trip), `t_oracle`, and face fluxes `t_faces`:

| case | `N_phi` | 16^3 | 32^3 | 48^3 | `nfev` per full slab (`nfev_max`) | `t_faces` 48^3 |
|---|---|---|---|---|---|---|
| `gauss:0.25` | 32 | 6.7 s | 40.8 s | 95.9 s | 458 | 24.5 s |
| `gauss_ch:0.25` | 48 | 11.6 s | 70.0 s | 186.2 s | 578 | 27.7 s |
| `gauss:1.0` | 48 | 25.1 s | 84.9 s | 234.6 s | 770 | 28.0 s |
| `control2d:0.5` | 32 | 5.9 s | - | - | 578 | - |
| `lester2021:1.0` | 48 | 15.5 s | - | - | 710 | - |

Reference solve: 1-25 s at `N_phi <= 48` for most cases; 57 s for `gauss:1.0` (`N_phi = 48`, 279 PCG
iterations x 3 cell problems), 66 s for `gauss_ch:1.0` (`N_phi = 64`, grid 128 x 64 x 64).

Cost model: `t_oracle(N) ~ N * nfev_max * t_eval(N^2, N_phi)` (plane `j` costs about `(j/N) nfev_max` forward
plus the same backward; summed over planes, `~ N nfev_max` evaluations), with `t_eval` the cost of one
vectorized `grad phi` evaluation of `P = N^2` points: measured 4.3 ms (`N_phi = 32`) and 6.1-6.7 ms
(`N_phi = 48`) at `P = 2304`; the 2-D sum scales as `P N_phi^2`. Worst cases at 48^3 (`N_phi = 64`,
`nfev_max ~ 900`, `t_eval ~ 11 ms`): `48 x 900 x 11 ms ~ 8 min` plus ~2 min reference and face fluxes.
Full matrix (7 fields x 3 eps x N 16/32/48): about 2 h in one local process; it is planned as the detached
V100-host CPU job of the sweep node (process pool; set `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` per worker).

## Findings of N1 relevant to the criteria (reported, not acted on)

- `gauss:1.0`: the oracle labels themselves, differentiated with the second-order FD of `metrics.fd_metrics`,
  are pre-asymptotic on 16/32/48: `e_v = 0.235 / 0.125 / 0.075` (observed orders 0.91, 1.25), `e_i1` orders
  1.30, 1.54, `e_div` orders 0.31, 0.79, and `min|c|` = 0.057 / 0.006 / 0.020 against `min|v_D|` = 0.046
  (`raw/oracle_convergence_gauss1.txt`). The 3-plane mid-slab probe (`--midplane`, no boundary stencils,
  `raw/oracle_midplane.txt`) shows the second-order regime is reached only at `N ~ 96-128`
  (orders 0.62, 1.21, 1.16, 1.32, 1.63, 1.76 over 16-24-32-48-64-96-128) while `max|grad_h psi|` grows from
  3.6 to 10.3. Mid-slab `e_v_mid` orders over 16-24-32-48-64-96-128:

  | case | orders | `e_v_mid` at 16 / 48 / 128 |
  |---|---|---|
  | `gauss:0.25` | 1.93 1.97 1.98 1.99 2.00 2.00 | 2.1e-2 / 2.5e-3 / 3.5e-4 |
  | `gauss:0.5` | 1.72 1.82 1.90 1.95 1.97 1.99 | 5.5e-2 / 7.4e-3 / 1.1e-3 |
  | `gauss:1.0` | 0.62 1.21 1.16 1.32 1.63 1.76 | 0.38 / 0.13 / 2.8e-2 |
  | `gauss_ch:0.25` | 1.91 1.95 1.98 1.99 1.99 2.00 | 2.2e-2 / 2.6e-3 / 3.6e-4 |
  | `gauss_ch:1.0` | 1.27 0.23 1.12 1.22 1.41 1.62 | 0.53 / 0.19 / 4.7e-2 |
  | `lester_brk:1.0` | 1.84 1.87 1.93 1.96 1.98 1.99 | 7.7e-2 / 9.8e-3 / 1.4e-3 |
  | `generic3d:1.0` | 1.19 1.35 1.51 1.70 1.82 1.90 | 0.12 / 2.7e-2 / 4.6e-3 |

  Criterion (1) at amplitude 1 measures a candidate's FD `e_v` with this same kind of measure, so on 16/32/48
  the exact Darcy labels do not reach order 1.8 either for `gauss`, `gauss_ch` and `generic3d` at `eps = 1`
  (and `gauss:0.5` is marginal on the first pair); see the RISKS of the N1 report.
- `gauss:0.25` and `gauss_ch:0.25`: order ~2 already on 16/32/48 (`raw/oracle_convergence.txt`).
