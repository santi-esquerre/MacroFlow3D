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
| `candidate_i.py` (N2) | candidate (i): non-divergence same-index equation (14) on the slab, inlet Dirichlet labels, analytic `grad ln k`, second-order centered stencils, no `|c|^2` regularization; variants `i0` (spec-literal control: equation rows also on the outlet plane with one-sided `d1`/`d11`/`d1j`, no boundary condition) and `i1` (deviation D-2: outlet rows `c2 = v2_in`, `c3 = v3_in`, i.e. `c x e1 = vperp_in x e1` with the inlet-face tangential Darcy velocity; Neumann `d1 psi_i = 0` for `_ch`). Damped Newton, colored central-FD sparse Jacobian, `splu` (N <= 24) / right-preconditioned GMRES (CGS2, restart 300) with the per-mode inverse of the `k = 1` linearization (N >= 32), amplitude continuation 0.25 -> 0.5 -> 1 with bisection fallback, Levenberg-Marquardt fallback for `i0` (N <= 16). Commands: `python3 candidate_i.py field:eps:N:i0|i1 ...` (CASE line `cand=i0/i1` + `cand=oracle_fd` ceiling, `EXTRA` outlet-row residual and inlet oblique defect, `HISTORY`); `--jactest field:eps:N:var` (Jacobian action vs FD, 3 steps); `--consistency field:eps ...` (residual at the oracle labels on `--grids 16,32,48`, orders); `--spectrum M field:eps:M:var ...` (dense FD Jacobian SVD at the converged state, or at the final iterate and the oracle labels if not converged -> `raw/spectrum_cand_i_<var>_<field>_<eps>_<M>[_state].txt`); `--k1check N` (`k = 1` control: exact nulls of `i0`/`i1`, preconditioner exactness -> `raw/spectrum_cand_i_<var>_uniform_k1_<N>.txt`); options `--maxit --tol --lin-tol --prec lin0|lap --direct-max --restart --no-continuation --bisect --lm --init zero|inlet|oracle (oracle = diagnostic only) --out`. Raw console: `raw/cand_i_smoke_*.txt`. C-i4 additions (section "Candidate (i): 4th-order variant `i1o4`, saved solutions and linear solves (C-i4)"): `--order 4` (`cand=i1o4`), `--save DIR`, `--remetric FILE...`, `--reuse-lu K`, `--selfcheck`; default `--direct-max` is now 32 |
| `candidate_ii.py` | N3 + C-ii, candidate (ii): dissipation energy of the label pair; default (C-ii) `--energy q1` = `1/2 sum_cells q_cell int_cell |grad psi1^h x grad psi2^h|^2` of the trilinear (Q1) label interpolants (pointwise in-cell product, exactly divergence-free, normal-continuous, face averages = the Whitney fluxes; 3x3x3 Gauss, exact); `--energy whitney` = the N3 edge-averaged energy `1/2 h^3 sum_f omega_f c_f^2` of the mimetic face fluxes (`c_f = curl_h(avg(psi1) G_h psi2)`, control with the hourglass kernel); inlet Dirichlet labels; outlet free (`_ch`) or outlet flux constraint `c1[N] = f1_in` plus the two D-3 rows "mean transverse flux = 0" (periodic fields, Lagrange multipliers); Newton (colored-FD Hessian) with LM globalization; metrics, `oracle_mim` ceiling, TPFA reference, `--gradtest`, `--consistency`, `--spectrum`. Section "Candidate (ii) (N3, corrected by C-ii)" below |
| `run_all.py` (N4) | resumable sweep driver: the N4 matrix (oracle ladder with `oracle_fd`/`oracle_mim` ceiling lines, `i1` on 16/24/32/48 (+64 for `gauss`, `gauss_ch`, `control2d`), `i0` at 16^3, (ii)-Q1 on 16/24/32 (4 h cap), `ii_from_oracle` at 48^3 for `gauss:0.25`/`gauss_ch:0.25`, consistency ladders, dense spectra 12^3/16^3) as one subprocess per cell in a pool (`--workers`), one log per cell `raw/sweep/<cell>.txt`, `raw/sweep/manifest.json` (status `done`/`failed`/`timeout`/`unsupported`, elapsed, exit, host, log tail on failure), spectra in `raw/sweep/spectra/`; non-oracle cells start only after the oracle cache of every grid they use exists; `--plan` (matrix, cost estimates, host check), `--summarize` (rebuilds `raw/sweep/summary.md`: per-(field, eps) tables, orders, ceiling ratios, consistency orders, spectrum statistics, D-5 classification per criterion), filters `--only --grids --spectra --kinds`, `--retry`. N4c: `--matrix corrective` (section "Corrective sweep (`raw/sweep2/`)": `i1o4`/`i1` direct solves on 16/20/24/28 (+32), memory-budget scheduler, `raw/sweep2/`) |
| `sweep_digest.py` (N4) | parses `raw/sweep/manifest.json`, the cell logs and the job log only (no computation) and writes `raw/sweep/timeouts.md`: `python3 sweep_digest.py --out ../raw/sweep [--slow 3600]` |

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
- Metrics: `metrics.fd_metrics(psi1, psi2, vD, psi_or, order=2)` (second-order centered FD in `x2, x3`; in `x1`
  centered on interior planes and second-order one-sided on the inlet and outlet planes) returns `e_v`, `e_psi`,
  `e_i1`, `e_i2`, `e_div`, `min_c`, percentiles `p0.1/p1/p5/p50` of `|c|`, and the same percentiles of `|v_D|`
  (`vD_p`). Candidates with their own `c` (e.g. face fluxes) compute `e_v` against `vD_faces` and pass it in the
  metrics dict.
  - C-i4, `order=4`: the reconstruction `c = grad_h psi1 x grad_h psi2`, `e_i` and `div_h` use centered 4th-order
    FD in `x2, x3` and in `x1` centered 4th order on planes `2..N-2` and 5-point 4th-order skewed / one-sided
    stencils on planes `1, N-1` / `0, N` (Fornberg weights, `metrics.fd_weights`). The ceiling of a 4th-order
    candidate is `fd_metrics(psi_or, ..., order=4)`, printed as `cand=oracle_fd4`. On the oracle labels of
    `gauss:0.25` (16/24/32) the 4th-order `e_v` is 3.29e-3 / 7.70e-4 / 2.62e-4 (orders 3.58, 3.75) against the
    2nd-order 1.67e-2 / 7.62e-3 / 4.33e-3 (orders 1.94, 1.97) (`oracle.py --selftest`).
  - C-i4 `e_psi` normalization (corrects a metric artifact present in every earlier output): with
    `den_i = RMS(psi_i^or - affine_i)` and `den_ref = max(den_1, den_2)`, label `i` is normalized by `den_i` if
    `den_i > 1e-6 den_ref`, else by `den_ref`. The previous threshold was `1e-12 den_ref`; `control2d` has
    `den_2 / den_1 = 1.1e-11` (`psi2 = Q0 x3`, periodic part `(Q0 - 1) x3`, roundoff), which passed it, so the
    old `e_psi` of `control2d` was a roundoff difference divided by roundoff (e.g. 0.467 at `eps = 0.25`). The
    dict now also has `e_psi1`, `e_psi2` (`e_psi = max`) and the absolute `a_psi1`, `a_psi2` (RMS of the label
    differences); `e_psi` values of `control2d` in outputs before C-i4 (N2 smoke, N4 sweep) are not comparable.
- One-line parseable format (`metrics.case_line` / `print_case`; `parse_case_line` inverts it):

  ```text
  CASE field=<f> eps=<e> N=<N> cand=<name> | r_F=<..> its=<..> | e_v=<..> e_psi=<..> e_i=(<..>,<..>) e_div=<..> min_c=<..> p0.1=<..> p1=<..> p5=<..> p50=<..> | t=<s>
  ```

  Missing values print as `nan` (the oracle prints `r_F=nan its=nan`, `e_psi=0`). Since C-i4 the line ends with
  ` | e_psi1=<..> e_psi2=<..>` when the per-label values exist; lines without the suffix still parse.
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
| `oracle.py --selftest` | `SELFTEST` rows (bit-identity with the closure probe, trig evaluation, complex-step `grad ln k`, `k = 1` affine labels, face-Jacobian identity, label jumps, `Q0`, tracer round trip, constant-head faces, face-flux balance, CASE-line round trip; C-i4: CASE line with the `e_psi1`/`e_psi2` suffix and an old line, per-label `e_psi` of a perturbed `control2d:0.25` oracle, observed order of the 4th-order `e_v` of the `gauss:0.25` oracle on 16/24/32 -- uses the 24^3/32^3 oracle caches, about 2 min if absent); exit 1 on any failure |
| `oracle.py field:eps:N ...` | per case: `REF`, `ORACLE` (round trip, `nfev`, timings), `CHECK` (plane fluxes, face-flux balance, constant-head faces or closure-note return map, outlet-vs-inlet labels, `|c|` and `|v_D|` percentiles), `CASE ... cand=oracle` |
| `oracle.py --convergence field:eps ...` | the above for `N` in `--grids` (default 16,32,48) and `ORDER` rows (observed orders of `e_v`, `e_i1`, `e_i2`, `e_div`; `min|c|`) |
| `oracle.py --nphi field:eps ...` | `NPHI` ladder rows and the chosen `NPHI_TABLE` row |
| `oracle.py --returnmap field:eps ...` | `RETURNMAP` row: closure-note return map (16 points, seed 3) at `N_phi`, next to the value recorded in the 2026-10-02 probes |
| `oracle.py --midplane field:eps ... --grids ...` | `MIDPLANE` rows: oracle labels on the planes `0.5 - h, 0.5, 0.5 + h` only, centered FD `e_v_mid`, `e_i_mid`, `max|grad_h psi|`, `min|c|`; `MIDPLANE_ORDER` |
| options | `--ref-nphi K` (override `N_phi`), `--grids 16,32,48`, `--no-cache` |

## Candidate (i): 4th-order variant `i1o4`, saved solutions and linear solves (C-i4)

Corrective node C-i4 (owner directive 2026-10-05, deviation D-7). Same continuous problem and unknowns as `i1`
(non-divergence same-index equation (14), analytic `grad ln k`, inlet Dirichlet labels, outlet rows
`c2 = v2_in`, `c3 = v3_in` with row scale `2q/h`, no regularization of `|c|^2`), discretized at 4th order
(`candidate_i.py --order 4`, `cand=i1o4`; a spec suffix `:i1o4` is equivalent):

- `x2, x3` (periodic): `d_j = (-u_{+2} + 8u_{+1} - 8u_{-1} + u_{-2})/(12h)`,
  `d_jj = (-u_{+2} + 16u_{+1} - 30u_0 + 16u_{-1} - u_{-2})/(12h^2)`, `d23 = d3(d2 u)`.
- `x1` (plane 0 = inlet data, unknown planes 1..N; class `Stencils4`): centered (same weights) on planes `2..N-2`;
  planes 1 and N-1 skewed 4th order, `d1` on 5 planes (`0..4` / `N-4..N`), `d11` on 6 planes (`0..5` / `N-5..N`);
  plane N one-sided `d1 = (25u_N - 48u_{N-1} + 36u_{N-2} - 16u_{N-3} + 3u_{N-4})/(12h)` (outlet rows). Mixed
  `d1j` = the `x1` stencil applied to the 4th-order `d_j`. Weights from the Fornberg routine `metrics.fd_weights`,
  checked on polynomials by `--selfcheck --order 4`.
- Residual norms as `i1`: `r_F` over the equation rows (planes 1..N-1), `r_out` over the outlet rows.
- Colored central-FD Jacobian (`Pattern4`): an equation row at plane `jr` depends on both fields on the union of
  the `d1`/`d11` stencil planes, with the 5x5 in-plane box on plane `jr` and the radius-2 in-plane cross on the
  other planes; outlet rows depend on the cross at plane N and the vertex on planes N-4..N-1. Colors
  `(field, (j-1) mod 6, m2 mod p, m3 mod p)`, `p` = smallest divisor >= 5 of N (432 colors at 12^3 and 24^3, 768
  at 16^3 and 32^3); the per-color row-collision check of `Pattern` is kept.
- Per-mode `k = 1` preconditioner (`ModePrec`, only used above `--direct-max`): the same construction with the
  4th-order symbols and `x1` rows (`--k1check 12 --order 4`: `|P^-1 J x - x|/|x| = 1.4e-14`).
- Metrics: `cand=i1o4` uses `fd_metrics(..., order=4)` with the ceiling `cand=oracle_fd4`; the same labels with the
  2nd-order reconstruction are printed as `cand=i1o4_fd2` with the ceiling `cand=oracle_fd`.
- A 4th-order variant of `i0` is not implemented (`--order 4` with `i0` is refused).

Linear solves (both orders):

- default `--direct-max 32` (was 24): sparse direct `splu` (SuperLU, scipy default COLAMD ordering) up to 32^3.
  Every factorization prints a `LINEAR splu` line (`t_fact`, `t_solve`, `nnz(L+U)`, fill, `mem(L+U)` estimated as
  12 bytes per factor nonzero, process peak RSS `maxrss`, step residual). Above `--direct-max`: GMRES + per-mode
  preconditioner, unchanged.
- `--reuse-lu K` (default 0 = off): the last LU is the right preconditioner of GMRES (`restart` = `maxiter` =
  `--restart`) for up to K subsequent Newton steps of the same stage; a frozen-LU solve that misses `--lin-tol`
  triggers a refactorization (`LINEAR frozen-LU GMRES ... -> missed lin-tol, refactor`). Every accepted Newton
  step is solved to `--lin-tol` (default 1e-13) either way, so the iterates do not depend on K beyond roundoff.
- `NEWTON` lines now end with `|dx|2=` and `t_fact=`; the earlier fields are unchanged, so `sweep_digest.py` still
  parses them.

Saved solutions: `--save DIR` writes `DIR/<field>_<eps>_<N>_<cand>.npz` after every solve (converged or not):
`u1`, `u2` on planes 0..N, status, its, `r_F`, `r_out`, continuation path, residual history, the metrics dicts of
the CASE lines (JSON), options, versions. `--remetric FILE...` reloads the case (oracle cache) and reprints the
CASE lines with the current `metrics.py`, without re-solving.

### C-i4 commands (local, bounded; outputs in `raw/`)

```bash
cd docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/scripts
export OMP_NUM_THREADS=8
python3 oracle.py --selftest                                       > ../raw/cand_i4_oracle_selftest.txt
python3 candidate_ii.py --gradtest gauss:0.25:12                   > ../raw/cand_i4_candii_gradtest.txt
python3 candidate_i.py --selfcheck --order 4                       > ../raw/cand_i4_selfcheck.txt
python3 candidate_i.py --jactest gauss:0.25:12:i1 --order 4        > ../raw/cand_i4_jactest.txt
python3 candidate_i.py --consistency gauss:0.25 gauss_ch:0.25 control2d:0.5 --order 4 --grids 16,24,32 \
                                                                   > ../raw/cand_i4_consistency.txt
python3 candidate_i.py control2d:0.5:16:i1 control2d:0.5:24:i1 gauss:0.25:16:i1 gauss:0.25:24:i1 \
        gauss_ch:0.25:16:i1 gauss_ch:0.25:24:i1 --order 4 --save ../raw/solutions \
                                                                   > ../raw/cand_i4_solve_control2d_16_24.txt
        # stopped after the two control2d cases (note at the end of the file); the rest split as follows
python3 candidate_i.py gauss:0.25:16:i1 gauss_ch:0.25:16:i1 --order 4 --save ../raw/solutions \
                                                                   > ../raw/cand_i4_solve_gauss_16.txt
python3 candidate_i.py gauss:0.25:24:i1 --order 4 --reuse-lu 4 --save ../raw/solutions \
                                                                   > ../raw/cand_i4_solve_gauss_0.25_24.txt
python3 candidate_i.py gauss_ch:0.25:24:i1 --order 4 --reuse-lu 4 --save ../raw/solutions \
                                                   > ../raw/cand_i4_solve_gauss_ch_0.25_24_INTERRUPTED.txt
        # interrupted (session stop); not evidence
python3 candidate_i.py control2d:0.25:16:i1 control2d:1.0:16:i1 --save ../raw/solutions \
                                                                   > ../raw/cand_i4_metricfix_control2d_16.txt
python3 candidate_i.py gauss:0.5:24:i1 --save ../raw/solutions     > ../raw/cand_i4_gauss0.5_24_direct_INTERRUPTED.txt
        # interrupted (session stop) in the first continuation stage (eps = 0.25); not evidence
# planned 32^3 timing runs (gauss:0.25:32 i1o4, gauss:0.5:32 i1) were NOT run locally; see "Measured cost"
OMP_NUM_THREADS=4 python3 candidate_i.py --spectrum 12 gauss:0.25:12:i1 gauss_ch:0.25:12:i1 --order 4 \
                                                                   > ../raw/cand_i4_spectrum12.txt
OMP_NUM_THREADS=4 python3 candidate_i.py --k1check 12 --order 4    > ../raw/cand_i4_k1check.txt
python3 candidate_i.py --remetric ../raw/solutions/<file>.npz ...  # metrics only, no solve
# *_py311.txt: the same selfcheck / jactest / oracle selftest under Python 3.11.16, numpy 1.26.4, scipy 1.11.4
```

### C-i4 results (local WSL, 16 cores shared with other work: load average 4-16 during the runs, so timings are
indicative)

Stencils and Jacobian (`cand_i4_selfcheck*.txt`, `cand_i4_jactest*.txt`, `cand_i4_k1check.txt`):

- every 4th-order `x1` stencil is exact on polynomials up to degree 4 (`d1`, 5 points: centered, skewed planes
  1/N-1, one-sided planes 0/N) or 5 (`d11` skewed/one-sided, 6 points), errors <= 4.6e-13; `derivs4` on
  `sin(2 pi x1) cos(2 pi x2) sin(2 pi x3)`: observed orders 3.93-4.96 on 32 -> 64 for `d1, d2, d11, d33, d12, d23`.
  The skewed weights (times 12) are `(-3, -10, 18, -6, 1)` (`d1`, plane 1) and `(10, -15, -4, 14, -6, 1)`
  (`d11`, plane 1); plane N `d1`: `(3, -16, 36, -48, 25)`.
- colored Jacobian vs directional FD at `gauss:0.25:12` (oracle + 1e-2 noise): 2.6e-7 / 2.6e-9 / 2.8e-11 for steps
  1e-3 / 1e-4 / 1e-5 (full, outlet-plane and inlet-plane directions); 432 colors, nnz 393 984. Identical on
  Python 3.11 / numpy 1.26.4 / scipy 1.11.4.
- `k = 1` control at 12^3: no exact nulls; smallest relative singular value 8.31e-4 (24-fold), 28 values < 1e-3,
  92 < 1e-2 (the `i1` control: 1.81e-3, 0 < 1e-3, 36 < 1e-2); the per-mode `lin0` inverse is exact
  (`|P^-1 J x - x|/|x| = 1.4e-14`); colored = dense Jacobian bitwise.

Consistency at the oracle labels (`cand_i4_consistency.txt`; `r_F` normalized as for `i1`, rows split into the
centered planes 2..N-2 and the skewed planes 1 and N-1):

| case | rows | 16 | 24 | 32 | orders 16->24, 24->32 |
|---|---|---|---|---|---|
| `gauss:0.25` | all equation rows | 4.20e-2 | 1.17e-2 | 4.28e-3 | 3.16, 3.49 |
| | centered | 4.36e-2 | 1.21e-2 | 4.39e-3 | 3.17, 3.51 |
| | plane 1 / plane N-1 | 2.89e-2 / 3.04e-2 | 5.68e-3 / 6.62e-3 | 1.76e-3 / 2.13e-3 | 4.01, 4.07 / 3.76, 3.93 |
| | outlet rows `r_out` | 3.05e-3 | 6.81e-4 | 2.18e-4 | 3.70, 3.96 |
| `gauss_ch:0.25` | all equation rows | 4.43e-2 | 1.24e-2 | 4.59e-3 | 3.13, 3.46 |
| | centered | 4.64e-2 | 1.29e-2 | 4.72e-3 | 3.16, 3.49 |
| | plane 1 / plane N-1 | 2.62e-2 / 2.72e-2 | 5.70e-3 / 6.15e-3 | 1.86e-3 / 2.08e-3 | 3.76, 3.89 / 3.67, 3.77 |
| | outlet rows `r_out` | 2.41e-3 | 3.00e-4 | 7.07e-5 | 5.14, 5.03 |
| `control2d:0.5` | all equation rows | 2.21e-2 | 4.47e-3 | 1.29e-3 | 3.94, 4.32 |
| | centered | 9.58e-3 | 2.14e-3 | 7.08e-4 | 3.69, 3.85 |
| | plane 1 / plane N-1 | 4.76e-2 / 6.21e-2 | 1.01e-2 / 1.62e-2 | 3.30e-3 / 5.13e-3 | 3.82, 3.88 / 3.31, 4.00 |
| | outlet rows `r_out` | 9.49e-3 | 2.43e-3 | 8.79e-4 | 3.36, 3.54 |

The truncation error is 4th order in the limit; on 16-32 the `gauss` interior rows are pre-asymptotic (3.2 -> 3.5,
rising), like the 4th-order reconstruction of the oracle labels themselves (`e_v` orders 3.58, 3.75).

Dense spectra at the converged state, 12^3 (`cand_i4_spectrum12.txt`, `spectrum_cand_i_i1o4_*_12.txt`; outlet row
scale `2q/h` as `i1`):

| case | smallest relative singular values | < 1e-2 | < 1e-3 | < 1e-4 | largest consecutive gap among < 1e-2 |
|---|---|---|---|---|---|
| `gauss:0.25` | 4.36e-4 4.42e-4 4.46e-4 4.68e-4 ... | 128 | 32 | 0 | 1.66 (after 48: 2.37e-3 -> 3.93e-3) |
| `gauss_ch:0.25` | 4.30e-4 4.45e-4 4.69e-4 4.72e-4 ... | 128 | 32 | 0 | 1.62 (after 48: 2.51e-3 -> 4.06e-3) |
| `k = 1` control | 8.31e-4 (x24) 8.84e-4 (x4) 1.48e-3 ... | 92 | 28 | 0 | - |

No gap-separated cluster: the small values form a continuum starting at about half the `k = 1` floor of `i1o4`.
The 4th-order operator has a larger `smax` (3.09e3 at `gauss:0.25`, 2.00e3 at `k = 1`; `i1`: 1.92e3 and 1.21e3),
so its relative floor is lower than that of `i1` (1.03e-3 at `gauss:0.25`).

Solves of `i1o4` (`cand_i4_solve_*.txt`, saved in `raw/solutions/`): every case converged to
`r_F <= 1e-13` (`r_out` <= 4e-16), no plateau; Newton converges quadratically once inside the basin. From `u = 0`
the first stage at `eps = 0.25` fails for `gauss:0.25:24` (5 steps, `r_F` 3.57 -> 0.40, line-search factors
1/64-1/4, `|dx|max` up to 1.96) and `gauss_ch:0.25:16` (stagnation at `r_F = 0.44`) with exact linear solves
(`splu`, step residuals <= 1.6e-12): a nonlinear (globalization) failure, not a linear one. The bisection to
`eps = 0.125` and the warm start back to 0.25 then converge (paths printed in the `PATH` lines; the same
`0.25(fail)->0.125->0.25` path the 2nd-order `i1` takes at `gauss:0.25:32` in the sweep).

### C-i4 completed local evidence (`raw/`)

- `cand_i4_selfcheck.txt`, `cand_i4_selfcheck_py311.txt`: `--selfcheck --order 4` (polynomials; derivs4 on
  16/32/64), no solve.
- `cand_i4_jactest.txt`, `cand_i4_jactest_py311.txt`: colored Jacobian vs FD, 12^3.
- `cand_i4_k1check.txt`, `spectrum_cand_i_i1o4_uniform_k1_12.txt`: `k = 1` control, 12^3.
- `cand_i4_spectrum12.txt`, `spectrum_cand_i_i1o4_gauss_0.25_12.txt`, `spectrum_cand_i_i1o4_gauss_ch_0.25_12.txt`:
  dense spectra at the converged state, 12^3.
- `cand_i4_consistency.txt`: residual at the oracle labels, 16^3/24^3/32^3 (no solve).
- `cand_i4_oracle_selftest.txt`, `cand_i4_oracle_selftest_py311.txt`: `oracle.py --selftest` (oracle labels on
  16^3/24^3/32^3).
- `cand_i4_candii_gradtest.txt`: candidate (ii) gradient test, 12^3.
- `cand_i4_solve_gauss_16.txt`: `i1o4` solves `gauss:0.25` and `gauss_ch:0.25`, 16^3.
- `cand_i4_solve_gauss_0.25_24.txt`: `i1o4` solve `gauss:0.25`, 24^3 (`--reuse-lu 4`; 1411 s).
- `cand_i4_solve_control2d_16_24.txt`: `i1o4` solve `control2d:0.5`, 16^3 (counted); the file also holds a
  24^3 CASE line, but the run was cut afterwards and only the 16^3 part is counted (header of the file).
- `cand_i4_metricfix_control2d_16.txt`: 2nd-order `i1` re-solves `control2d:0.25` and `control2d:1`, 16^3, with
  the corrected per-label `e_psi`.
- `solutions/*_16_*.npz`: the saved 16^3 solutions of the solves above (`--remetric` input).
- Not evidence: `cand_i4_solve_gauss_ch_0.25_24_INTERRUPTED.txt`, `cand_i4_gauss0.5_24_direct_INTERRUPTED.txt`
  (session stop; first line of each file).

### Measured cost (from the logs above)

| solve | N | n | nnz(J) | nnz(L+U) | fill | t_fact | maxrss |
|---|---|---|---|---|---|---|---|
| order 4 (`i1o4`) `splu` | 16^3 | 8192 | 9.5e5 | 1.75e7 | 18 | ~5 s | ~0.95 GB |
| order 4 (`i1o4`) `splu` | 24^3 | 27648 | 3.26e6 | 1.27e8 | 39 | 70-170 s | 5.2-5.8 GB |
| order 2 (`i1`) `splu` | 24^3 | 27648 | 1.49e6 | 5.7e7 | 38 | 37 s | 1.56 GB |

- `gauss:0.25:24:i1o4` took 1411 s in total (path `0.25(fail)->0.125->0.25`, 7 final Newton iterations).
- The frozen-LU GMRES (`--reuse-lu`) missed the linear tolerance in 2 of 3 uses at 24^3 (each miss triggers a
  refactorization), so the corrective sweep uses pure direct solves (`--reuse-lu 0`).

No 32^3 or larger case was run locally. All ladder evidence (16/20/24/28 and 32 where planned) comes from the
remote corrective sweep (`raw/sweep2/`).

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

## N4 sweep (`raw/sweep/`)

Remote detached job (no GPU work; CPU-only Python process pool):

| item | value |
|---|---|
| job | `sf29-run-all` (`scripts/remote --increment SF-29 ...`; mirror `~/MacroFlow3D-SF-29`, per-increment state root `~/.macroflow3d-remote/macroflow3d-SF-29/`) |
| command | `cd docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/scripts && OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 python3 run_all.py --workers 22 --out ../raw/sweep` |
| remote log | `~/.macroflow3d-remote/macroflow3d-SF-29/logs/sf29-run-all.log` (copied verbatim to `raw/sweep/sf29-run-all.joblog.txt`) |
| start / end (UTC) | 2026-10-03T06:19:33Z / 2026-10-03T20:59:57Z |
| wall time | 52 822 s (driver `run_all: finished in 52822 s`) |
| exit code | 0 (`scripts/remote status`: `succeeded`) |
| GPU lock | GPU 0 lock held for the whole job by this CPU-only job (`GPU=0 (CUDA_VISIBLE_DEVICES=0, request=auto)` in the job log); no CUDA code ran |
| host stack | `localhost.localdomain`, python 3.11.7, numpy 1.26.4, scipy 1.11.4; 22 worker processes x 3 BLAS threads |
| cells | 338 in the matrix: 305 `done`, 33 `timeout` (cell wall caps: 6 h default, 8 h `i1` at 64^3, 4 h `ii`/`iiorc`), 0 `failed` |

Retrieval (2026-10-05, after the job had finished; no `scripts/remote sync` in between, so the mirror still held the
outputs):

```bash
rsync -av --exclude 'cache/' v100:~/MacroFlow3D-SF-29/docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/raw/sweep/ docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/raw/sweep/
rsync -a v100:/home/sesquerre/.macroflow3d-remote/macroflow3d-SF-29/logs/sf29-run-all.log docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/raw/sweep/sf29-run-all.joblog.txt
```

505 files pulled (= the remote file count of `raw/sweep/`; 503 cell/spectrum `.txt`, `manifest.json`,
`summary.md`), 16.6 MB, largest file < 0.5 MB; the oracle caches (`raw/cache/*.npz`) are not pulled or committed.

| file | content |
|---|---|
| `raw/sweep/<cell>.txt` | console log of one matrix cell (first line `RUN_ALL cell=... cmd=...`; timed-out cells end with `RUN_ALL TIMEOUT ...`) |
| `raw/sweep/spectra/` | dense spectra written by the `spec_*` cells |
| `raw/sweep/manifest.json` | per-cell status, elapsed, cap, exit, command, start/end; run record (workers, threads, host, wall) |
| `raw/sweep/summary.md` | `run_all.py --summarize` output (tables, orders, ceiling ratios, D-5 classification per criterion). Regenerated locally from the pulled logs with `python3 run_all.py --summarize --out ../raw/sweep`: byte-identical to the remote file |
| `raw/sweep/timeouts.md` | `sweep_digest.py --out ../raw/sweep`: the 33 timed-out cells and the 52 `done` cells with elapsed > 3600 s (continuation path, last Newton iterate, GMRES / direct-solve iteration and time statistics, bisections), the elapsed table of `i1` on `gauss`/`gauss_ch`, and the concurrency actually used |
| `raw/sweep/sf29-run-all.joblog.txt` | the remote job log (cell start/done/timeout events with elapsed) |

## Corrective sweep (`raw/sweep2/`)

Node N4c (driver only; the job is run by the orchestrator as a detached remote job). `run_all.py --matrix
corrective` adds a second matrix next to the N4 one (`--matrix n4`, still the default and unchanged). It replaces
the GMRES + `k = 1` preconditioner path that degraded with amplitude in `raw/sweep/` (33 timeouts) by pure direct
`splu` solves (`--direct-max 32 --reuse-lu 0`; frozen-LU GMRES missed `--lin-tol` in 2 of 3 uses at 24^3, see
"Measured cost"), and adds the 4th-order variant `i1o4` and the intermediate grids 20, 28.

Matrix (fields `gauss`, `gauss_ch`, `control2d`, `generic3d` x `eps` 0.25 / 0.5 / 1; `G = 16, 20, 24, 28`):

| type | command (run from `scripts/`) | selection | cells |
|---|---|---|---|
| `orc` | `run_all.py --oracle-cell field:eps --fd4 --grids <every N used by the cells of that (field, eps)>` (oracle caches; CASE lines `oracle_fd`, `oracle_fd4`, `oracle_mim`); every other cell of the (field, eps) depends on it | all 12 (field, eps); grids `12` (spec4), `G`, `32` (gauss/gauss_ch 0.25 `i1o4`/`ii`, and the `i1` ladders of gauss, gauss_ch, control2d) | 12 |
| `i1o4` | `candidate_i.py field:eps:N:i1 --order 4 --direct-max 32 --reuse-lu 0 --save ../raw/sweep2/solutions` | 12 (field, eps) x `G`, + N = 32 for `gauss:0.25`, `gauss_ch:0.25` | 50 |
| `i1` | `candidate_i.py field:eps:N:i1 --direct-max 32 --reuse-lu 0 --save ../raw/sweep2/solutions` | gauss, gauss_ch, control2d x 3 eps x `G` + 32; generic3d x 3 eps x N = 20, 28 (16, 24 are read from `raw/sweep/` by the summary, marked `sweep`) | 51 |
| `ii` | `candidate_ii.py field:eps:N` (Q1) | gauss, gauss_ch, control2d x 3 eps at N = 20; control2d x 3 eps at 16, 24 (corrected per-label `e_psi`); N = 32 for `gauss:0.25`, `gauss_ch:0.25` (lowest priority) | 17 |
| `spec4` | `candidate_i.py --spectrum M --order 4 field:eps:M:i1 --out ../raw/sweep2/spectra` | 12 (field, eps) x M = 12, 16 | 24 |
| `cons4` | `candidate_i.py --consistency --order 4 --grids 16,24,28 field:eps` | 12 (field, eps) | 12 |

166 cells; `--plan` checks every flag against the scripts' argument parsers (no `unsupported` type) and prints
the cells, the counts, 3052 GB-h of memory at the wall caps (upper bound) and the launch order.

Scheduler (replaces the fixed-size pool for this matrix):

- estimated peak memory `mem_gb` / wall cap `cap_s` per cell: `i1o4` N16 1.5/3600, N20 3/7200, N24 7/10800, N28
  14/21600, N32 28/36000; `i1` N16 1/3600, N20 1.5/3600, N24 2.5/7200, N28 5/14400, N32 9/21600; `ii` N16 1/7200,
  N20 2/14400, N24 3/14400, N32 4/36000; `orc` 2/7200; `spec4` 12^3 1/1800, 16^3 4/3600; `cons4` 2/7200.
- `sum(mem_gb of running cells) <= --mem-budget` (default 100) and `running <= --workers` (default 26); every cell
  gets `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = --threads` (default 3).
- priority `orc` > `i1o4` > `i1` > `spec4`, `cons4` > `ii` > `ii` at N = 32; larger `mem_gb` first within a class;
  a cell starts only when its `orc` cell is `done` (or has printed `ORACLE_READY` for every N it uses); the first
  ready cell that does not fit reserves its memory, so later (smaller) cells start only if they fit beside it (no
  starvation of the large cells). A cell above the budget runs alone.
- a cell past its cap is killed (process group) and recorded `timeout`; a non-zero exit is `failed` with the log
  tail; a failed / timed-out `orc` fails its dependents explicitly; SIGTERM / SIGINT of the driver kills the running
  cells and records them `interrupted`.
- `manifest.json` per cell: status, exit, elapsed, `start_utc` / `end_utc`, `mem_gb`, `cap_s`, `peak_rss_gb`
  (`ru_maxrss` from `os.wait4` of the cell process), command, log. Resumable: `done` cells are skipped;
  `--retry` (default `failed,running,pending,interrupted`) selects what is rerun.
- `<out>/solutions/.gitignore` is written on the first run (only the 16^3 npz are committed).
- launcher tests without computation: `--dry-run` (each cell runs `true`, or `sleep S` with `--dry-sleep S`) and
  `--dry-faults` (a fake cell that sleeps past a 2 s cap and one that exits 1).

`--summarize` (also run at the end of the job) writes `raw/sweep2/summary.md`: per (field, eps) and candidate
(`i1o4`, `i1o4_fd2`, `i1`, `ii`) a table over N (status, solver status, `r_F`, its, `e_v`, ceiling, `e_v`/ceiling,
`e_psi`, `e_psi1`, `e_psi2`, `min|c|`), the observed orders over consecutive completed grids of `e_v`, `e_psi` and
the ceiling and the least-squares slope of log(e) vs log(h), the continuation PATH, the `spec4` statistics, the
`cons4` orders, and the D-5 classification per criterion on the three finest completed grids (rules printed in
the file). Ceilings: `oracle_fd4` for `i1o4`, `oracle_fd` for `i1` and `i1o4_fd2`, `oracle_mim` for `ii`.

Job command (orchestrator, detached on the host):

```bash
cd docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/scripts && python3 run_all.py --matrix corrective --workers 26 --mem-budget 100 --threads 3 --out ../raw/sweep2
```

The oracle caches in `raw/cache/` (`cases.CACHE_DIR`, keyed by field, eps, N, `N_phi`) are reused when present
(e.g. those left on the host mirror by `raw/sweep/`); the `orc` cells build the missing ones (20^3, 28^3 are new).

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
| `cand_i4_oracle_selftest.txt`, `cand_i4_oracle_selftest_py311.txt` | C-i4 `oracle.py --selftest` (local stack / Python 3.11.16, numpy 1.26.4, scipy 1.11.4) |
| `cand_i4_*.txt` | C-i4 candidate (i) consoles (commands in "C-i4 commands") |
| `spectrum_cand_i_i1o4_<field>_<eps>_12.txt`, `spectrum_cand_i_i1o4_uniform_k1_12.txt` | C-i4 dense spectra of `i1o4` at 12^3 (converged state) and the `k = 1` control |
| `solutions/` | C-i4 saved solutions `<field>_<eps>_<N>_<cand>.npz` (`candidate_i.py --save`); 16^3 files committed, larger ones gitignored (`solutions/.gitignore`) |

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
