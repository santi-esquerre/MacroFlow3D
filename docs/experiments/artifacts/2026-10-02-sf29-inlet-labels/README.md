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
| `candidate_i.py` (N2) | candidate (i): non-divergence same-index equation (14) on the slab, inlet Dirichlet labels, analytic `grad ln k`, second-order centered stencils, no `|c|^2` regularization; variants `i0` (spec-literal control: equation rows also on the outlet plane with one-sided `d1`/`d11`/`d1j`, no boundary condition) and `i1` (deviation D-2: outlet rows `c2 = v2_in`, `c3 = v3_in`, i.e. `c x e1 = vperp_in x e1` with the inlet-face tangential Darcy velocity; Neumann `d1 psi_i = 0` for `_ch`). Damped Newton, colored central-FD sparse Jacobian, `splu` (N <= 24) / right-preconditioned GMRES (CGS2, restart 300) with the per-mode inverse of the `k = 1` linearization (N >= 32), amplitude continuation 0.25 -> 0.5 -> 1 with bisection fallback, Levenberg-Marquardt fallback for `i0` (N <= 16). Commands: `python3 candidate_i.py field:eps:N:i0|i1 ...` (CASE line `cand=i0/i1` + `cand=oracle_fd` ceiling, `EXTRA` outlet-row residual and inlet oblique defect, `HISTORY`); `--jactest field:eps:N:var` (Jacobian action vs FD, 3 steps); `--consistency field:eps ...` (residual at the oracle labels on `--grids 16,32,48`, orders); `--spectrum M field:eps:M:var ...` (dense FD Jacobian SVD at the converged state, or at the final iterate and the oracle labels if not converged -> `raw/spectrum_cand_i_<var>_<field>_<eps>_<M>[_state].txt`); `--k1check N` (`k = 1` control: exact nulls of `i0`/`i1`, preconditioner exactness -> `raw/spectrum_cand_i_<var>_uniform_k1_<N>.txt`); options `--maxit --tol --lin-tol --prec lin0|lap --direct-max --restart --no-continuation --bisect --lm --init zero|inlet|oracle (oracle = diagnostic only) --out`. Raw console: `raw/cand_i_smoke_*.txt` |
| `candidate_ii.py` | N3 + C-ii, candidate (ii): dissipation energy of the label pair; default (C-ii) `--energy q1` = `1/2 sum_cells q_cell int_cell |grad psi1^h x grad psi2^h|^2` of the trilinear (Q1) label interpolants (pointwise in-cell product, exactly divergence-free, normal-continuous, face averages = the Whitney fluxes; 3x3x3 Gauss, exact); `--energy whitney` = the N3 edge-averaged energy `1/2 h^3 sum_f omega_f c_f^2` of the mimetic face fluxes (`c_f = curl_h(avg(psi1) G_h psi2)`, control with the hourglass kernel); inlet Dirichlet labels; outlet free (`_ch`) or outlet flux constraint `c1[N] = f1_in` plus the two D-3 rows "mean transverse flux = 0" (periodic fields, Lagrange multipliers); Newton (colored-FD Hessian) with LM globalization; metrics, `oracle_mim` ceiling, TPFA reference, `--gradtest`, `--consistency`, `--spectrum`. Section "Candidate (ii) (N3, corrected by C-ii)" below |

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

## Candidate (ii) (N3, corrected by C-ii): `candidate_ii.py`

Formulation, quadrature, constraints and solver are documented in the module docstring. Since C-ii the default
energy is the Q1 form; the N3 form is the control `--energy whitney`. Semantics fixed here:

- Energy (`--energy q1`, default, `cand=ii`): `E_h = 1/2 sum_cells q_cell int_cell |c(x)|^2 dx`,
  `c(x) = grad psi1^h x grad psi2^h` pointwise inside each cell, `psi_i^h` the trilinear (Q1) interpolants of the
  8 vertex labels (affine parts exact). `c` is exactly divergence-free in each cell and normal-continuous across
  faces, and its face averages are exactly the Whitney face fluxes `c_f` (so `vD_faces` comparisons, metrics and the
  outlet flux constraint are those of N3). It is not the collocated product at the vertices, which is unsound
  and not used. Quadrature: tensor 3x3x3 Gauss-Legendre per cell. `|c|^2` has degree <= 4 per direction
  (`c1 ~ (2,1,1)`, `c2 ~ (1,2,1)`, `c3 ~ (1,1,2)`) and the rule is exact to degree 5, so the integral is exact.
  `q_cell = 1/k` at the cell center (`--q q1`: trilinear `q` from the vertices, `cand=ii_qq1`, also exact).
  `--energy whitney` (`cand=ii_whitney`) is the N3 edge-averaged form `1/2 h^3 sum_f omega_f c_f^2`. It is kept as
  the control that has the `2 N^2` hourglass kernel at `k = 1`.
- Constraints (periodic fields; deviation D-3): `C = c1[N] - t = 0` on the `N^2` outlet `x1`-faces,
  `t = f1_in - mean(f1_in) + 1`, `f1_in = load_case()['vD_faces']['f1'][0]`. `sum c1[N] = N^2` identically, so the
  mean of `f1_in` (`1 - O(1e-10)`, printed as `f1_in mean-1`) is replaced by 1 and the redundant row is absorbed by a
  bordered KKT system (`sum lam = 0`, scalar `mu`). In addition (C-ii, completing D-3), there are two scalar rows for
  zero mean transverse flux:
  `Cm = (h^2 sum_{j,m3} c2[j, 0, m3], h^2 sum_{j,m2} c3[j, m2, 0]) = 0` on the plane `m2 = 0` / `m3 = 0` (multipliers
  `eta` = head jumps across the periodic boundary). The plane fluxes are plane-independent only up to the row sums of
  `c1[N] - c1[0]` (cell balance identity, verified to 1e-16 in the `MEANFLUX` lines). `c1[0]` is the mimetic flux
  of the inlet labels and `t` comes from the reference faces, so the spread is `O(h^2)` (5e-5 at 16^3 for
  `gauss:0.25`). `c_rel = max(max|C| / max|t|, max|Cm|)`. `--no-meanflux` drops the two rows (`cand=..._nomf`, the N3
  constraint set). `--free-outlet` drops all outlet constraints (`cand=ii_free`, control). `_ch` fields are always
  free at the outlet (natural condition `c x e1 = 0`).
- `r_F := g_rel = |grad_x L| / |grad E_h(x0)|`, `L = E_h + h^2 lam . C + eta . Cm`, `x0` = unknowns 0 (inlet labels
  in place, full amplitude). If `|grad E_h(x0)| = 0` (the `uniform` field), `g_rel` is the absolute norm (printed).
- `e_v`: RMS over all faces of the three families of `c_f - vD_faces`, over `RMS(vD_faces)`; `e_div` from
  `div_h c_f` (roundoff); `min_c`, percentiles: cell-averaged face fluxes; `e_psi`, `e_i`: `metrics.fd_metrics`.
  Next to every candidate line: `cand=oracle_mim` = the same face flux of the exact oracle labels (D-5 ceiling).
- `KELVIN` line: `E(oracle) - E_min` for both forms. It also gives the TPFA Darcy flux `c_K` of the same inlet
  (and outlet + D-3) data (`kelvin_tpfa`, with the two head jumps when the D-3 rows are active). The identity
  `E_h(u) = E_K + 1/2 |c(u) - c_K|_W^2` is exact only for the Whitney form (checked in `--gradtest` to 8e-16). The
  Q1 minimizer is a different, Q1-based Darcy discretization, so for it the TPFA flux is only a reference.
- `MEANFLUX` lines (periodic fields): plane fluxes of the candidate and of the oracle labels, their spread over
  planes, the balance-identity defect and the reference face fluxes.

### C-ii commands (local, bounded; outputs in `raw/`)

```bash
cd docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/scripts
python3 candidate_ii.py --gradtest gauss:0.25:12                         > ../raw/cand_ii_q1_gradtest.txt
<py311>/bin/python candidate_ii.py --gradtest gauss:0.25:12              > ../raw/cand_ii_q1_gradtest_py311.txt
python3 candidate_ii.py --consistency gauss:0.25 gauss_ch:0.25           > ../raw/cand_ii_q1_consistency.txt
python3 candidate_ii.py control2d:0.5:16 gauss_ch:0.25:16 gauss:0.25:16  > ../raw/cand_ii_q1_16.txt
python3 candidate_ii.py --free-outlet gauss:0.25:16                      > ../raw/cand_ii_q1_free_outlet.txt
python3 candidate_ii.py --spectrum 12 gauss:0.25:12 gauss_ch:0.25:12     > ../raw/cand_ii_q1_spectrum12.txt
python3 candidate_ii.py --spectrum 12 uniform:0:12                       > ../raw/cand_ii_q1_spectrum12_uniform.txt
python3 candidate_ii.py --energy whitney --spectrum 12 uniform:0:12      > ../raw/cand_ii_q1_ctl_whitney_spectrum12.txt
python3 candidate_ii.py --solver gmres gauss_ch:0.25:16                  > ../raw/cand_ii_q1_gmres16.txt
timeout 900 python3 candidate_ii.py gauss_ch:0.25:32                     > ../raw/cand_ii_q1_32_timing.txt
timeout 900 python3 candidate_ii.py --solver gmres --start oracle gauss_ch:0.25:32 > ../raw/cand_ii_q1_32_from_oracle_gmres.txt
<py311>/bin/python candidate_ii.py --solver gmres --maxit 3 --no-continuation gauss:0.25:12 > ../raw/cand_ii_q1_gmres_py311.txt
```

`<py311>`: a Python 3.11.16 venv with exactly numpy 1.26.4 / scipy 1.11.4 (the V100-host versions).

### Findings of C-ii (reported, not acted on)

- Structure (`cand_ii_q1_gradtest*.txt`, both Python stacks, `ALL PASS`): (a) pointwise `div c` at 6 random
  points in every cell (central differences, which are exact for the degree-2 dependence of `c_a` on `x_a`)
  2.1e-15 / 2.4e-15 relative to `max|c|/h`; (b) face averages of `c . n` (3x3 Gauss) vs Whitney `c_f`
  2.7e-16 / 3.0e-16, normal continuity 4.3e-16 / 4.0e-16; (c) exact gradient vs central FD of `E_h` <= 8.1e-8 over
  steps 1e-5..1e-7 (free and constrained); D-3 row Jacobian vs FD <= 1.3e-10; colored 27-point FD Hessian vs
  directional FD 1.2e-8..1.7e-8.
- Kernel at `k = 1`, `u = 0`, 12^3 (`--gradtest` (d) and the `uniform` spectra): the Q1 Hessian has 0 relative
  eigenvalues below 1e-10 (smallest 1.32e-3, i.e. the `h^2` elliptic tail; no negative ones). Curvature along the
  two checkerboard directions `U1 = (-1)^m3 f(j, m2)`, `U2 = (-1)^m2 g(j, m3)` is 7.0e-2 / 7.3e-2 (0.21 / 0.22 of
  `lambda_max`). The Whitney control has exactly 288 = `2 N^2` null modes and zero curvature on them, with a
  gap ratio of 1.8e11 in the reduced Hessian / KKT spectra. With the outlet constraints, the constraint Jacobian at
  the uniform state has rank 144 of 146: the redundant `xi = 0` row and the `(N/2, N/2)` checkerboard outlet row.
  Both are properties of the Whitney outlet flux, so they appear under both forms.
- Spectra at the solved states, 12^3 (`spectrum_cand_ii_q1_*`): `gauss:0.25` reduced Hessian smallest relative
  singular value 1.83e-3, counts `<1e-2`: 44, `<1e-3`: 0, largest gap below 1e-2 ratio 1.34 (no cluster); KKT
  identical tail (min 1.83e-3, 0 below 1e-3); 0 negative eigenvalues. `gauss_ch:0.25` Hessian min 1.36e-3,
  `<1e-2`: 56, `<1e-3`: 0, gap 1.32, 0 negative. N3 had 319 / 334 values below 1e-3 and 32 / 24 negative
  eigenvalues. The tail matches the normal `h^2` elliptic tail of UNDERSTAND §8 (no gap-separated group).
- Solves at 16^3 (`cand_ii_q1_16.txt`), no plateau, final stage quadratic for `gauss`, `gauss_ch`:

  | case | its | `r_F` | `e_v` | `oracle_mim` `e_v` | TPFA `e_v` | `e_psi` | `E(oracle) - E_min` | t |
  |---|---|---|---|---|---|---|---|---|
  | `control2d:0.5` | 24 | 9.1e-14 | 5.23e-3 | 9.8e-8 | 6.70e-3 | 9.4e-3 | 1.4e-5 | 107 s |
  | `gauss_ch:0.25` | 42 | 1.6e-15 | 1.11e-2 | 6.19e-3 | 5.21e-3 | 0.155 | 1.3e-4 | 160 s |
  | `gauss:0.25` | 31 | 1.7e-15 | 9.90e-3 | 5.78e-3 | 4.49e-3 (with D-3 jumps) | 0.162 | 1.1e-4 | 139 s |

  `control2d` reaches 9.1e-14 with a linear tail (factor ~0.2-0.5 per iteration below 1e-9). `gauss:0.25`: D-3 rows
  at roundoff, `eta = (-2.3e-4, +4.5e-3)` (the mean transverse head gradient `(a2, a3)` of the reference, as N3
  identified). Hourglass content 4e-5 / 3e-6 (oracle 3e-5 / 4e-6): no drift.
  Ratios to the `oracle_mim` ceiling: `e_v` 1.7-1.9 (`gauss`, `gauss_ch`); `control2d` is not comparable (ceiling at
  the reference level, 1e-7). From the oracle start (`--start oracle`, 16^3 `gauss_ch`) Newton converges in 5
  iterations to the same `E_min` (16 digits), so the minimizer is unique locally and `e_psi` is a discretization
  property, not drift. Its error grows smoothly from the inlet (1e-3 absolute at `j = 1`, 7.5e-3 at the outlet,
  ~`2 h^2`). The relative `e_psi` is large because the oracle periodic parts are small (RMS 0.037 / 0.032).
- 32^3 `gauss_ch:0.25` (diagnostic, `--solver gmres --start oracle`, `cand_ii_q1_32_from_oracle_gmres.txt`; the
  prescribed inlet-start timing run did not complete, see cost below): `r_F` 1.5e-15 in 5 iterations,
  `E(oracle) - E_min = 1.3e-5`, hourglass content 5e-7 / 3e-8. The minimizer is assumed to be the one the
  continuation would reach. This was verified at 16^3 (same `E_min` from both starts) and NOT at 32^3. Over 16 -> 32:

  | quantity | 16^3 | 32^3 | observed order |
  |---|---|---|---|
  | `e_v` (all faces) | 1.107e-2 | 4.110e-3 | 1.43 |
  | `e_v` x1-faces / x2-faces / x3-faces | 8.05e-3 / 1.39e-2 / 1.06e-2 | 2.06e-3 / 5.65e-3 / 3.86e-3 | 1.97 / 1.30 / 1.46 |
  | `e_psi` | 0.155 | 0.0613 | 1.34 |
  | `e_i` (1, 2) | 9.9e-3, 8.3e-3 | 3.6e-3, 2.6e-3 | 1.45, 1.67 |
  | `oracle_mim` `e_v` (ceiling) | 6.19e-3 | 1.60e-3 | 1.95 |
  | TPFA `e_v` | 5.21e-3 | 1.31e-3 | 1.99 |
  | `e_v / oracle_mim` | 1.79 | 2.56 | - |

  On this pair the Q1 minimizer's transverse face fluxes and labels converge at ~1.3-1.5, below the ceiling's ~2.
  A third grid (48^3) is needed to tell a pre-asymptotic regime from a reduced order; the N4 sweep (V100 host) has
  to decide it. Not interpreted further here.
- Consistency at the oracle labels (`cand_ii_q1_consistency.txt`, 16/32/48):

  | quantity | `gauss:0.25` (constrained + D-3) | orders | `gauss_ch:0.25` (free) | orders |
  |---|---|---|---|---|
  | `g_rel` (least-squares multipliers) | 3.39e-2 / 4.37e-3 / 1.13e-3 | 2.96, 3.33 | 3.19e-2 / 3.77e-3 / 9.60e-4 | 3.08, 3.37 |
  | interior planes | 3.38e-2 / 4.36e-3 / 1.13e-3 | 2.96, 3.33 | 3.16e-2 / 3.75e-3 / 9.57e-4 | 3.07, 3.37 |
  | outlet plane | 1.35e-3 / 1.95e-4 / 5.94e-5 | 2.79, 2.93 | 4.51e-3 / 3.59e-4 / 7.37e-5 | 3.65, 3.90 |
  | outlet constraint (rms rel) | 5.17e-3 / 1.32e-3 / 5.89e-4 | 1.97, 1.99 | - | - |
  | D-3 rows max | 9.17e-6 / 2.26e-6 / 1.00e-6 | 2.02, 2.01 | - | - |
  | TPFA `e_v` (D-3 jumps for `gauss`) | 4.49e-3 / 1.14e-3 / 5.09e-4 | 1.98, 1.99 | 5.21e-3 / 1.31e-3 / 5.82e-4 | 1.99, 2.00 |
  | `oracle_mim` `e_v` | 5.78e-3 / 1.50e-3 / 6.74e-4 | 1.94, 1.98 | 6.19e-3 / 1.60e-3 / 7.18e-4 | 1.95, 1.98 |

  The D-3 completion removes the N3 floor of the TPFA flux of `gauss:0.25` (N3 without the rows: 6.3e-3 / 4.7e-3 /
  4.6e-3, orders 0.42, 0.04). The outlet-plane stationarity orders, 0.67 / 0.89 for the N3 Whitney form, are 2.79 / 2.93 for the Q1 form.
- `--free-outlet gauss:0.25:16` (`cand_ii_q1_free_outlet.txt`): `e_v = 2.96e-2` (N3: 2.9e-2), outlet `|c_perp|`
  1.26e-2 vs `|v_D perp|` 0.118. This is a different (constant-head outlet) Darcy flow, as in N3.
- Solver and cost (local WSL, 16 cores shared): at 16^3, one Newton iteration = 27-point colored FD Hessian
  2.0-2.5 s (192 gradient evaluations) + `splu` 1.3-1.7 s; the 16^3 solves take 107-181 s (24-45 iterations over
  the three stages). `--solver gmres` (Fourier-mode preconditioner, no shift needed on hourglass modes now) on
  `gauss_ch:0.25:16` reaches the same `E_min` with `r_F` 1.6e-15 in 41 iterations, 251 s. GMRES takes 57-250
  iterations near the solutions, but hits its 2400 cap (linres up to 2.5e-2) in the strongly nonlinear phase of
  the `s = 1` stage; LM and backtracking absorb this. 32^3 (`cand_ii_q1_32_timing.txt`, prescribed `timeout 900`
  run): FD Hessian 26-29 s + `splu` 173-195 s per iteration (KKT `n` = 65 536, RSS ~0.3 GB between steps).
  Killed by the timeout after 3 iterations of the `s = 0.25` stage, so it did not complete. `splu` column
  orderings make no material difference (16^3: COLAMD 1.8 s / MMD_AT_PLUS_A 1.1 s, same fill).
  GMRES at 32^3 (from-oracle run) needs 1355-1822 iterations per Newton step near the solution (91-110 s), against
  57-250 at 16^3: the iteration count grows ~7-10x per refinement, so the constant-coefficient mode preconditioner
  degrades with N. Estimates at ~40 Newton iterations per inlet-start solve: 32^3 direct ~2.4 h, GMRES ~1.4 h
  (Hessian 26 s + GMRES ~100 s per iteration, more in the nonlinear phase). 48^3: Hessian ~90 s per iteration, and GMRES
  would need ~10^4 iterations per Newton step at that growth rate. That exceeds the 2400-iteration cap of
  `_gmres` (restart 60 x 40), so the 48^3 linear solves would not converge as configured; direct is infeasible.
  64^3 is worse still. A full inlet-start sweep at 48^3/64^3 is therefore not practical with this solver. The cheaper routes are
  continuation plus a better preconditioner (e.g. the variable-coefficient Hessian at a coarse level, or ILU) and an
  exact Gauss-point Hessian in place of the 192 gradient evaluations (not implemented, reported for N4).

### N3 record (Whitney form, before C-ii)

The commands below are the N3 runs. Since C-ii they reproduce N3 only with `--energy whitney --no-meanflux`
(the CASE name is then `ii_whitney_nomf` instead of `ii`, and spectrum files are written as
`spectrum_cand_ii_q1_ctl_whitney_nomf_*`); the N3 raw files are kept unchanged.

Commands (local, bounded smokes; outputs in `raw/`):

```bash
cd docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/scripts
python3 candidate_ii.py --gradtest gauss:0.25:12                         > ../raw/cand_ii_smoke_gradtest.txt
<py311>/bin/python candidate_ii.py --gradtest gauss:0.25:12              > ../raw/cand_ii_smoke_gradtest_py311.txt
python3 candidate_ii.py --consistency gauss:0.25 gauss_ch:0.25           > ../raw/cand_ii_smoke_consistency.txt
python3 candidate_ii.py control2d:0.5:16 gauss_ch:0.25:16 gauss:0.25:16  > ../raw/cand_ii_smoke_16.txt
python3 candidate_ii.py --free-outlet gauss:0.25:16                      > ../raw/cand_ii_smoke_free_outlet.txt
python3 candidate_ii.py --spectrum 12 gauss:0.25:12 gauss_ch:0.25:12     > ../raw/cand_ii_smoke_spectrum12.txt
python3 candidate_ii.py --hessian gn gauss_ch:0.25:16 gauss:0.25:16      > ../raw/cand_ii_smoke_gn16.txt
timeout 600 python3 candidate_ii.py gauss:0.25:32                        > ../raw/cand_ii_smoke_32_timing.txt
<py311>/bin/python candidate_ii.py --solver gmres --maxit 3 --no-continuation gauss:0.25:12 > ../raw/cand_ii_smoke_gmres_py311.txt
```

Findings of N3 (reported, not acted on):

- Hourglass deficiency of the prescribed mimetic flux: at the uniform state the Hessian of `E_h` has exactly
  `2 N^2` null modes (`U1 = (-1)^m3 f(j, m2)`, `U2 = (-1)^m2 g(j, m3)`; 288 at 12^3), and the linearized outlet-flux
  row of the `(N/2, N/2)` checkerboard vanishes. At the solved states the dense spectra (12^3) have no exact null
  cluster and no gap, but 319 (`gauss`, reduced Hessian) / 334 (`gauss_ch`) relative singular values below 1e-3
  (~`2 N^2` + the ~30 of a well-posed elliptic tail), 21/27 below 1e-6 (min 5e-8 / 5e-9) and 32/24 negative
  eigenvalues.
- Newton does not reach `r_F <= 1e-12` at 16^3: plateau (linear crawl) at `r_F` = 1.9e-10 (`control2d:0.5`),
  1.6e-3 (`gauss_ch:0.25`), 8.2e-6 (`gauss:0.25`); Gauss-Newton (`--hessian gn`) also plateaus. The flux still
  converges to the discrete Kelvin flux (`E_min - E_K` = -8e-15 / +6.6e-7 / +3.6e-8; `|c - c_K|` = 1e-7 / 1e-3 /
  3e-4) while the labels drift along the hourglass modes (hourglass RMS 2e-3..7e-3 vs oracle 3e-5;
  `e_psi` = 0.41 / 0.29 / 0.16).
- `e_v` of candidate (ii) is therefore that of `c_K`: `gauss_ch:0.25` 5.2e-3 / 1.3e-3 / 5.8e-4 over 16/32/48
  (order 1.99, 2.00, below the `oracle_mim` ceiling); `control2d:0.5` 3.0e-2 at 16^3 (oracle_mim 1e-7).
- Periodic fields (D-3): the outlet flux constraint does not fix the mean transverse flux. `c_K` (constrained) has
  `e_v` 6.3e-3 / 4.7e-3 / 4.6e-3 over 16/32/48 (orders 0.42, 0.04: floor) with a mean transverse flux ~ (2e-4,
  4.5e-3) ~ the transverse mean head gradient `(a2, a3)` of the reference: the energy minimizer has zero mean
  transverse head gradient, the reference has zero mean transverse flux. Imposing zero mean transverse flux
  (scratch check, pressure jumps across the periodic boundaries) gives `e_v` 4.2e-3 -> 1.1e-3 (16 -> 32, order 1.96).
- `--free-outlet` control on `gauss:0.25:16`: `e_v = 2.9e-2` (vs 6.3e-3 constrained), outlet `|c_perp|` 1.2e-2 vs
  `|v_D perp|` 0.118: a different (constant-head outlet) Darcy flow, as expected.
- Consistency at the oracle labels (16/32/48): `gauss_ch:0.25` `g_rel` orders 3.07, 3.37; `gauss:0.25` interior
  2.95, 3.33, outlet plane 0.67, 0.89, constraint residual 1.97, 1.99.
- Cost: 16^3 Newton iteration 2-10 s (FD Hessian 0.3-1.4 s + splu 1-10 s); 16^3 runs 7-15 min (plateau stop).
  32^3 (GMRES + Fourier-mode preconditioner, which is exact for the constant-coefficient operator but
  shifted on the hourglass modes): GMRES does not converge (2400 its, linres 2e-2..3e-1), ~230 s per Newton
  iteration; the run was stopped at 600 s. Extrapolated per Newton iteration: ~800 s at 48^3, ~1900 s at 64^3.

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
| `cand_ii_smoke_*.txt` | N3 candidate (ii) console outputs (commands in "Candidate (ii) (N3)") |
| `cand_ii_q1_*.txt` | C-ii candidate (ii) console outputs, Q1 energy + D-3 rows (commands in "C-ii commands"); `cand_ii_q1_ctl_whitney_spectrum12.txt` is the Whitney control at `k = 1` |
| `spectrum_cand_ii_q1_<field>_<eps>_<N>[_kkt|_hL].txt` | C-ii: sorted relative singular values at the solved state (Q1 energy), same layout as the N3 files; `spectrum_cand_ii_q1_ctl_whitney_uniform_0_12*` = Whitney control at `k = 1` (288 null modes) |
| `spectrum_cand_ii_<field>_<eps>_<N>.txt` | sorted relative singular values at the solved state: Hessian (`_ch`) or reduced Hessian `Z^T H_L Z` (constrained); `_kkt`: KKT matrix with orthonormalized constraint rows; `_hL`: unreduced Lagrangian Hessian |
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
