# SF-35: label transport by backward streamline tracing as the production constructor of the Darcy labels

- Date: 2026-10-07 (spec) / campaign 2026-10-15 .. 2026-10-16 (V100 jobs listed under Commands); note written 2026-10-16
- Status: `partial — claim (a) established; claim (b) established at sigma_Y = 0.5 and 1.0, NOT established at sigma_Y = 1.5; pre-registered thresholds not all met (B at three points, C at eps 1.5); owner decision pending`
- Increment: [`SF-35-label-transport-constructor.md`](../plans/active/lester-eq14/increments/SF-35-label-transport-constructor.md)
- Theory: [`docs/theory/lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) §3A (equation (14), same-index
  pairing, inlet-label target), §4 (zero-transverse claim: recorded, not assumed)
- Decision record: [`2026-10-07-label-transport-constructor.md`](../decisions/2026-10-07-label-transport-constructor.md)
  (items 1-5; item 3 = independence rule, item 5 = validity envelope)
- Predecessor note: [`2026-10-06-sf33-gpu-inlet-labels.md`](2026-10-06-sf33-gpu-inlet-labels.md)
- Artifacts: [`artifacts/2026-10-07-sf35-label-transport/`](artifacts/2026-10-07-sf35-label-transport/README.md)
  (`scripts/`, `logs/`, `raw/`, `analysis/`)

Every number below is read from a committed file named next to it; none is retyped without a source. Abbreviations
(all under `A = artifacts/2026-10-07-sf35-label-transport/`): `CT` = `A/analysis/controls.md`, `XC` =
`A/analysis/xcons.md`, `RG` = `A/analysis/regress.md`, `M05`/`M10`/`M15` = `A/analysis/matrix_eps05.md`,
`matrix_eps10.md`, `matrix_eps15.md`, `INFO` = `A/analysis/info_sigma_eps.md`, `GPU` = `A/analysis/gpu.md`, `COST` =
`A/analysis/cost_256.md`, `LD` = `A/analysis/label_diagnostics.md` (post-hoc, not pre-registered), `BL` =
`A/logs/jobs/sf35-build.log`. Numbers are rounded to 3-4 significant digits as in the digests.

## Question

The increment's exact Goal: `Establecer el trazado regresivo de líneas de corriente desde la cara de entrada como
constructor de producción de las etiquetas de Darcy en el slab, verificado sin trazar (controles con etiquetas
exactas, métricas por diferencias finitas, residuo de la ecuación (14) como diagnóstico y concordancia con la
solución elíptica donde existe) y con órdenes observados en 32^3-256^3 para sigma_Y en {0.5, 1, 1.5}.`

Two claims (spec, "Scientific or engineering intent"):

- (a) the traced labels pass checks that do not trace: exact-label positive controls, grid finite-difference
  metrics `e_v`, `e_i`, `e_div`, the equation-(14) residual `r_F` as an independently derived necessary condition,
  and agreement with the elliptic SF-33 solution where it exists;
- (b) on the production field (SF-18 Gaussian, `ell = 1/4`, seed 3001, `sigma2 = 1` normalized, `eps` in
  {0.5, 1, 1.5}) the FD metrics of the traced labels converge under refinement 32/64/128/256, with the error growth
  along `x1` and the backflow statistics measured.

No claim about transport, the return map (SF-34) or `alpha_T` (theory note §4; `alpha_T` is not presupposed).

## Hypothesis

**Pre-registered thresholds A-H.** Quoted by reference, verbatim and unchanged: the section "Acceptance thresholds" of
the increment specification at commit `f8bec23`, registered in the activation bitácora row of 2026-10-15T17:05Z before
any run (summary: A positive controls; B integration certificate; C convergence per amplitude with `accept` iff
order `>= 1.8` on 128 -> 256 for each of `e_v`, `e_i1`, `e_i2`, `e_div`, `r_F`, `reject` iff a quantity changes
`< 10 %` between 128 and 256 above 1e-12; D cross-construction reproducing SF-33 within 1e-6 relative; E recorded;
F GPU tracer non-blocking; G regression; H cost at 256^3). Ladder settings fixed before the runs: primary
`(h/8, 1e-8)`; ladder `(h/16, 1e-8)`, `(h/16, 1e-10)`, `(h/32, 1e-10)`; `max_roundtrip = 1e-8`; `min_step 1e-13`,
`max_arclength 100`. Blocking rules (spec): A or D not met = defect; `eps = 0.5` and `1.0` must be `accept` in C;
`eps = 1.5` must be `accept` unless the stage reports `inlet_backflow` (then recorded, owner decision); B not met at
a point = recorded, no setting changed, exclusion only by owner decision.

**Orchestrator reading rules** (activation row, quoted in `A/README.md`; deviations for the reviewer):
1. criterion D's 1e-6 relative is evaluated against the full-precision SF-33 values in the SF-33 artifact JSON
   (`metrics.i1o4.e_psi*`), not the printed 4 digits;
2. `--construct trace` exit codes 0 ok / 17 `inlet_backflow` / 18 `darcy_failed` / 19 `trace_fail`, and
   `roundtrip_exceeded` gets the new code 21 (20 is `not_run`);
3. per-plane metrics `e_v_j = RMS_j|c - vD| / v_rms` (global `v_rms`), `e_i_j`, `min_c_j`, `vD1_min_j`,
   `vD1_nonpos_j` with the global stencils; GROWTH uses `e_i_j = sqrt((e_i1_j^2 + e_i2_j^2)/2)`;
4. G's byte-compare trio vs base references from `f8bec23` (orchestrator job `sf35-ctest-full-bytecmp`);
5. G's 32^3 regression masks only timing fields;
6. the constructor entry point is a template on the direction-field functor with a `ProductionStage` overload;
7. A2's order `>= 3.5` is evaluated on both pairs 16->32 and 32->64; a 16->32 miss with a 32->64 pass is reported as
   pre-asymptotic, disclosed, not tuned.

**Derived expectations** (recorded at UNDERSTAND, before the runs): `e_v` of the traced labels carries an `O(h^2)` input
inconsistency (2nd-order SF-19 inlet flux, non-solenoidal spline flow) = the SF-33 `oracle_fd4` ceiling, so order
`~2` is the correct expectation (not 4); `e_i`, `e_div` are pure FD error (orders 2.5-4); `r_F` needs Hessians of
labels that are `C^2` with `O(h)` third-derivative jumps at the spline knots, so order `~2`. Correction recorded at
UNDERSTAND: for the A2 control, `Q0 = 1` and `G1 = 1/I_0(0.5)`. `eps = 1.5` was not predicted (backflow possible;
`ell = 1/4` at `sigma^2 = 2.25` unmeasured).

## Build / environment

- Source: `55d95a6329b6bc65cc25f936b419e103a7ae33bb`; campaign scripts `c11a45570d3b2907831301a517e3e060c2c77a80`
  (`A/COMMIT`; the `run.env` files show a second `COMMIT=` line that is cosmetic: the file has two lines).
- Host: V100 (`Tesla V100-PCIE-32GB`, `A/logs/matrix_eps10/run.env`), preset `v100-release` (CUDA), per-increment mirror
  `~/MacroFlow3D-SF-35`, host tracer with `THREADS=32` (labels are bitwise independent of the thread count). Build and
  `ctest -R inlet_slab`: `BL` (9/9 passed).
- Corrective cycles of the campaign scripts (not of the constructor): C1 `875c166` (bash 4.2 empty arrays under
  `set -u` killed the first matrix job at N = 128), C2 `c11a455` (bash 4.2 kept backslashes in a pattern replacement,
  giving a NaN cost row). The eps 0.5 label exports were lost on the mirror by an `rsync --delete` sync; the JSON and
  logs of that matrix are retained.

## Config(s)

Production field: SF-18 Gaussian log-conductivity, `sigma2 = 1` (variance-normalized), `ell = 0.25`, seed 3001,
`k = exp(eps Y)`, so `sigma_Y = eps` and `sigma^2 = eps^2` in {0.25, 1, 2.25}. Matrix: `eps` in {0.5, 1.0, 1.5} x
`N` in {32, 64, 128, 256}; primary integration `(h/8, 1e-8)`; `--ladder` at 32 and 64; `--save-labels` at every point.
Cross-construction `--solve` at (eps, N) = (0.25, 32), (0.25, 64), (0.5, 32). Controls at 16/32/64. Regression:
`--production --n 32 --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder`. Details: `A/README.md`,
`A/scripts/README.md`.

## Commands

Detached V100 jobs (`scripts/remote --increment SF-35 run <job> -- "<command>"`; exact command lines and start/end
times in `A/logs/jobs/<job>.log`, copied from the remote logs). All use
`bash docs/experiments/artifacts/2026-10-07-sf35-label-transport/scripts/run_campaign.sh build/v100-release <part>`
with `THREADS=32`:

| job | part | log |
|---|---|---|
| `sf35-build` | `cmake --preset v100-release`, build, `ctest -R inlet_slab` | `A/logs/jobs/sf35-build.log` |
| `sf35-controls` | `controls` | `A/logs/jobs/sf35-controls.log` |
| `sf35-xcons` | `xcons` | `A/logs/jobs/sf35-xcons.log` |
| `sf35-regress-prod32` | `regress` | `A/logs/jobs/sf35-regress-prod32.log` |
| `sf35-matrix-eps05` (first, killed at N = 128 by the C1 bash defect) and `sf35b-matrix-eps05` | `matrix 05` | `A/logs/jobs/sf35-matrix-eps05.log`, `sf35b-matrix-eps05.log` |
| `sf35b-matrix-eps10` | `matrix 10` | `A/logs/jobs/sf35b-matrix-eps10.log` |
| `sf35b-matrix-eps15` | `matrix 15` | `A/logs/jobs/sf35b-matrix-eps15.log` |
| `sf35b-info` | `info` | `A/logs/jobs/sf35b-info.log` |
| `sf35b-gpu-equiv` | `gpu` | `A/logs/jobs/sf35b-gpu-equiv.log` |
| `sf35c-labeldiag` (post-hoc, not pre-registered) | `python3 .../scripts/label_diagnostics.py ~/sf35_exports_keep --out .../analysis/label_diagnostics.md`; exit 0 | `A/logs/jobs/sf35c-labeldiag.log` |

The full `ctest` suite and the SF-27 trio byte-compare belong to the orchestrator's final job
`sf35-ctest-full-bytecmp`: see the FINAL_AUDIT bitácora row of the increment file.

## Outputs inspected

### A. Positive controls (`CT`, `BL`)

| item | observed | verdict (CT) |
|---|---|---|
| A1 `k = 1`, N = 64 | `max|psi1 - x2|`, `max|psi2 - x3|` = 5.551e-17 / 5.551e-17 (<= 1e-13) | pass |
| A2 straightness `k = exp(0.5 cos 2 pi x2)`, N = 16/32/64 | `dpsi(j) - dpsi(0)` = 0 (<= 1e-10) | pass |
| A2 orders (>= 3.5) `e_v` | 16->32: 3.92; 32->64: 3.98 | pass |
| A2 orders (>= 3.5) `r_F` | 16->32: 3.93; 32->64: 3.98 | pass |
| A3 `control2d`, N = 32 | `max|psi2 - x3|` = 5.529e-14 (<= 1e-10) | pass |
| A4 synthetic failure field | via the ctest entry `inlet_slab_controls` (and `inlet_slab_trace16`): both Passed, 10.34 s / 10.54 s (<= 60 s) | pass |

Disclosure: the A4 assertion on `landings_discarded` is a lower bound in the test, not an exact count.

### B. Integration certificate (`M05`, `M10`, `M15`, section "Verdict B")

| eps | N=32 | N=64 | N=128 | N=256 | B verdict (digest) |
|---|---|---|---|---|---|
| 0.5 | max round trip 2.740e-09, ladder 1.824e-09, pass | 3.403e-10, 1.579e-10, pass | 3.123e-11, pass | 3.770e-12, pass | pass |
| 1.0 | `roundtrip_exceeded` (exit 21), 2.282e-08, ladder 1.655e-08, fail | 2.402e-09, 1.475e-09, pass | 3.227e-10, pass | 4.337e-11, pass | fail |
| 1.5 | `roundtrip_exceeded` (exit 21), 2.527e-07, ladder 8.289e-08, fail | `roundtrip_exceeded` (exit 21), 3.699e-08, ladder 1.121e-08, fail | 4.117e-09, pass | 4.862e-10, pass | fail |

`non_ok = 0` at every point (status histograms in the digests: all streamlines `ok`).

### C. Convergence per amplitude (`M05`, `M10`, `M15`, section "Verdict C" and "Observed orders")

Observed orders on 128 -> 256 and verdicts as the digests print them:

| eps | `e_v` | `e_i1` | `e_i2` | `e_div` | `r_F` | amplitude verdict |
|---|---|---|---|---|---|---|
| 0.5 | 2.00 accept | 3.06 accept | 2.65 accept | 3.80 accept | 1.85 accept | **accept** |
| 1.0 | 2.71 accept | 3.21 accept | 3.04 accept | 2.35 accept | 2.33 accept | **accept** |
| 1.5 | -0.01 reject (0.407 -> 0.411; rel change 0.009) | 1.58 unresolved | 0.56 unresolved | 0.12 reject (14.0 -> 12.9; rel change 0.079) | 0.29 unresolved (1.829e+03 -> 1.496e+03) | **reject** |

At eps 1.5 `e_v`, `e_div` and `r_F` do not decrease on every refinement (digest column "decreases on every
refinement": False). The stage reports no `inlet_backflow` at any point (E below), so the spec's clause "`eps = 1.5`
must be `accept` unless the stage reports `inlet_backflow`" is not satisfied by an exemption.

Quantities at the end points (`M05`, `M10`, `M15`, "Quantities per grid"): `e_v` N=256 = 9.211e-05 (eps 0.5),
4.845e-04 (1.0), 4.106e-01 (1.5); `r_F` N=256 = 1.481e-03, 1.506e-01, 1.496e+03; `min_c` N=256 = 2.284e-01,
4.152e-02, 2.394e-04. At eps 0.25 the SF-33 `oracle_fd4` ceiling of `e_v` is 2.247e-3 / 5.684e-4 / 1.427e-4 at
32/64/128 (spec, Expected numerical effect); at eps 0.5 `e_v` is 5.984e-03 / 1.473e-03 / 3.682e-04 / 9.211e-05 at
32..256 (`M05`), order 2.00-2.02, as derived.

Per plane at eps 1.5 (`M15`, "Per-plane growth"; E, no threshold): at `x1 = 1`, `e_v_j` = 1.521 / 1.439 / 1.260 /
0.857 at N = 32/64/128/256 while `e_i_j` = 1.154e-01 / 5.023e-02 / 3.270e-02 / 1.750e-02; at `x1 = 0.5`, `e_v_j` =
0.394 / 0.217 / 0.181 / 0.134 and `e_i_j` = 9.893e-02 / 3.160e-02 / 5.244e-03 / 1.217e-03. The streamline-invariance
error `e_i` keeps decreasing with N; the metric `|c - vD|` does not converge downstream.

### D. Cross-construction (`XC`)

| (eps, N) | trace / solve | `r_F` (solve) | `e_psi / e_psi1 / e_psi2` | rel diff vs SF-33 full precision | verdict |
|---|---|---|---|---|---|
| (0.25, 32) | ok / converged | 1.097e-14 | 1.250e-02 / 1.250e-02 / 1.140e-02 | 0 / 0 / 0 | pass |
| (0.25, 64) | ok / converged | 5.102e-14 | 1.555e-03 / 1.483e-03 / 1.555e-03 | 0 / 0 / 0 | pass |
| (0.5, 32) | ok / converged | 4.844e-14 | 3.837e-02 / 3.837e-02 / 2.756e-02 | 1.221e-13 / 1.221e-13 / 1.307e-13 | pass |

`a_psi1`, `a_psi2` observed order 32 -> 64 at eps 0.25: 3.07 / 2.87 (>= 2.5): pass. **Verdict D: pass** (`XC`).

### E. Recorded, no threshold

- Backflow (`M05`, `M10`, `M15`, rows `seed_backflow`, `backflow_enc`, `landings_discarded`, `vD1_nonpos`): 0 at every
  matrix point (3 amplitudes x 4 grids); smallest `vD1_min` at N = 256: 0.2279 (eps 0.5), 0.04112 (1.0), 0.006358
  (1.5); inlet minimum `inlet_vmin` at N = 256: 0.2420, 0.04789, 0.008356. No inlet backflow at any point.
- `GROWTH` ratios (`M05`, `M10`, `M15`, rows `e_v_N/e_v_1`, `e_i_N/e_i_1`): `e_v_N/e_v_1` at N = 256 = 2.203 (0.5),
  7.824 (1.0), 4.873e+03 (1.5); `e_i_N/e_i_1` at N = 256 = 26.17 (0.5), 623.9 (1.0), 2.349e+04 (1.5). Full per-plane
  lists: `A/raw/matrix_eps<L>/planes_eps*.csv`.
- `sigma2 = 2.25, eps = 1` vs `sigma2 = 1, eps = 1.5` label difference at 32^3 (`INFO`): `max|dpsi1|` = 2.931e-14,
  `max|dpsi2|` = 1.654e-14 (expected <= 1e-12: met).

### F. GPU tracer (`GPU`, `BL`)

Host vs device at eps 0.5: `max|dpsi1|`, `max|dpsi2|` = 0 at N = 128 and 256, no NaN mismatches, status host/gpu
ok/ok, per-plane `nfev` and status histograms identical. Trace-phase wall times: 176.56 s host vs 20.63 s gpu at 128
(speedup 8.56), 2996.68 s vs 499.55 s at 256 (speedup 6.00). The fast test `inlet_slab_trace_gpu` Passed (33.22 s,
`BL`). **Verdict F (digest): pass.** The `--fmad=true` INFO difference and the 16/32 cases are in the ctest entry, not
in a digest of this artifact.

### G. Regression (`RG`, `BL`)

`--production` 32^3 eps 0.25: 102/102 compared CASE/ORACLE/ORACLE_SUMMARY/EXTRA lines identical to the SF-33 log after
masking timing fields (**identical**); the 9 `inlet_slab` ctest entries (SF-33 and SF-35) Passed (`BL`: 9/9). The `git diff` on the
frozen paths, the SF-27 trio byte-compare and the full V100 `ctest`: see the FINAL_AUDIT bitácora row of the
increment file (orchestrator job).

### H. Cost at 256^3 (`COST`)

| eps | driver total (s) | wall (s) | host max RSS (GiB) | peak device (GiB, device-wide poll) | trace phase (s) |
|---|---|---|---|---|---|
| 0.5 | 3002.1 | 3005.9 | 3.59 | 3.40 | 2988.2 |
| 1.0 | 3122.8 | 3126.5 | 3.59 | 3.40 | 3106.3 |
| 1.5 | 3441.8 | 3445.4 | 3.59 | 3.40 | 3425.7 |

### Post-hoc diagnostics (NOT pre-registered; they change no verdict and are not used to overturn any)

Script `A/scripts/label_diagnostics.py`, job `sf35c-labeldiag` (exit 0), output `LD`, on the label exports of eps 1.0 and
1.5 (the eps 0.5 exports were lost). They were motivated by the eps 1.5 C reject.

1. Wrap check (`LD` section 1): `max|U(next) - U| * N` at eps 1.5 is 14.1 / 20.7 / 22.8 (U1 along x1 / x2 / x3) and 26.4
   / 41.1 / 37.1 (U2) at N = 256, against a periodic-wrap defect that would show ~1 * N = 256; at eps 1.0 at most
   6.21. These are maxima of the discrete derivative of `U`, not wrap jumps; they grow with N at eps 1.5 (e.g. U2 x2:
   9.8 -> 17.5 -> 27.4 -> 41.1), i.e. steeper gradients appear as the grid resolves them.
2. Pointwise self-convergence (`LD` section 2): the labels themselves converge at observed order ~2 on common
   vertices. eps 1.5 psi1 RMS 3.367e-03 -> 8.568e-04 -> 2.156e-04 (orders 1.97, 1.99); psi2 RMS 2.289e-03 -> 5.992e-04
   -> 1.503e-04 (1.93, 1.99); max orders 1.85/1.98 (psi1), 0.64/1.80 (psi2). eps 1.0 RMS orders 1.99-2.00.
3. Concentration of the `c = grad psi1 x grad psi2` non-convergence (`LD` section 3), pair 128 vs 256, normalized by
   RMS|c(256)|: eps 1.5 RMS 0.2712, median 1.079e-03, p99.9 3.278, max 33.21 at (x1, x2, x3) = (0.8047, 0.9375, 0.3594),
   half of the total squared error from 361 of 2,080,768 vertices (0.0173 %); eps 1.0 RMS 6.387e-03, median 6.955e-04,
   half from 4189 vertices (0.2013 %).

Interpretation (hypothesis, not established): at eps 1.5 the RMS-based FD metrics of the C criterion are dominated by
about 0.017 % of the vertices, where the labels carry steep, localized gradients that are not resolved at N <= 256,
while the labels converge pointwise at `O(h^2)`. This reading is consistent with `LD` but was not tested by a finer grid
or by an independent metric; it is a hypothesis about a resolution limit of the FD verification.

## Result

Classification per `docs/AGENTS.md` ("confirmed in runs" = established by the runs cited, within the caveats).

**Claim (a) — ESTABLISHED (confirmed in runs).** A (exact-label controls, orders 3.92-3.98 against the threshold 3.5)
and D (reproduction of the SF-33 elliptic values to <= 1.3e-13 relative; `a_psi` orders 3.07 / 2.87) pass; the FD metrics
and `r_F` are non-tracing checks of the traced labels, and the independence rule (decision item 3, (i)-(iii)) is met.
G's production lines are identical (102/102). What is not independently checked by (i)-(iii) remains as in the decision
record: the spline flow and the D-1 inlet data, to be checked by the return map (SF-34).

**Claim (b) — ESTABLISHED at sigma_Y = 0.5 and 1.0; NOT ESTABLISHED at sigma_Y = 1.5.**
- sigma_Y = 0.5: C `accept` at 32-256 (orders 2.00 / 3.06 / 2.65 / 3.80 / 1.85 on 128 -> 256), B pass.
- sigma_Y = 1.0: C `accept` at 32-256 (orders 2.71 / 3.21 / 3.04 / 2.35 / 2.33); B fails at N = 32 only (below).
- sigma_Y = 1.5: C **reject** (`e_v` 0.407 -> 0.411, `e_div` 14.0 -> 12.9 reject; `e_i1`, `e_i2`, `r_F` unresolved), and
  the stage reports **no inlet backflow** (all backflow counters 0, `vD1_nonpos` 0 at every point). The spec's blocking
  clause for eps 1.5 ("must be `accept` unless the stage reports `inlet_backflow`") is therefore **not met**. Per
  plane, the invariance error `e_i` stays small and keeps decreasing with N, but `|c - vD|` does not converge
  downstream (`e_v_j` at `x1 = 1`: 1.521 -> 0.857 over N = 32 -> 256). Post-hoc (not pre-registered): the labels
  converge pointwise at `O(h^2)`; the RMS FD metrics are dominated by ~0.017 % of the vertices with localized steep
  gradients unresolved at N <= 256 — a hypothesis about a resolution limit of the FD verification, not established.

**B — NOT MET at three matrix points, recorded, no setting changed:** (1.0, 32) `roundtrip_exceeded` 2.282e-08 (ladder
`|dpsi|` 1.655e-08), (1.5, 32) 2.527e-07 (ladder 8.289e-08), (1.5, 64) 3.699e-08 (ladder 1.121e-08). Exclusion only by
owner decision. For scale: the ladder `|dpsi|` at these points is <= 8.3e-08 while the FD errors of C at the same
points are >= 1e-2 (e.g. `e_v` 6.789e-02 at (1.0, 32), 6.276e-01 at (1.5, 32), 4.327e-01 at (1.5, 64)), so the
integration error is not what limits these points; the pre-registered criterion is nevertheless not met. The
points N = 128, 256 pass B at every amplitude; (0.5, all N) and (1.0, N >= 64) pass.

**E, F, H:** recorded as above (no inlet backflow; the two parametrizations of the same amplitude agree to 2.9e-14; the
GPU tracer reproduces the host labels exactly at 128 and 256 with trace speedups 8.56 / 6.00; cost at 256^3 3006 / 3127 /
3445 s, 3.59 GiB host, 3.40 GiB device).

**Increment outcome.** The pre-registered thresholds are not all met: B at the three points above and the blocking
clause of C at sigma_Y = 1.5. Following the spec's failure policy and the SF-33 precedent, the outcome is **owner
decision required (state blocked)**. SF-34 remains blocked until SF-35 is `done` on the default branch.

**Options for the owner (stated neutrally; no recommendation implied by the order; none presupposed).**

- **O1.** Accept the constructor with the validity envelope sigma_Y <= 1 (sigma^2 <= 1 at `ell = 1/4`; SF-18 field, seed
  3001; 32^3-256^3) and the coarse-B exclusions ((1.0, 32); the B points at 1.5 moot under the envelope), recording
  sigma_Y = 1.5 as not established. Cost: smallest; SF-34 re-scoped to sigma^2 <= 1 (the decision record's
  downstream range reached sigma^2 = 2.25); the exclusion of (1.0, 32) is an owner waiver of a pre-registered
  criterion. Risk: sigma_Y = 1.5 stays an open question on a single field/seed/`ell`.
- **O2.** A follow-up increment that resolves sigma_Y = 1.5, for example 256 -> 512^3 with the GPU tracer, under
  newly pre-registered criteria. Estimate from `GPU`, `COST`, `M05`: the trace phase scales ~N^4 (host 176.6 -> 2996.7 s
  from 128 to 256, a factor 17.0; GPU 20.63 -> 499.55 s, a factor 24.2), so one 512^3 trace on the GPU is about
  500 s x (17 - 24) ~ 8.5e3 - 1.2e4 s (2.4 - 3.4 h) per amplitude by extrapolation from two points (not
  measured; the host tracer would be ~14 h by the same factor). Memory: host RSS 3.59 GiB and device 3.40 GiB at 256
  (device-wide poll, includes anything else on the GPU); the host RSS grew by a factor 3.78 from 128 to 256 and the
  device peak by a factor 3.69, so 512^3 may need 8x the field-sized part: at the full 8x factor the device peak would be
  ~27 GiB against 32 GB on the V100 (`Tesla V100-PCIE-32GB`): feasibility is **not established** and tight; it must be
  measured (e.g. at 384^3) before committing; host memory of the V100 machine was not checked here. The label
  exports at 512^3 are 1.08 GB per field. Risk: the post-hoc hypothesis (unresolved steep gradients) may need
  more than a factor 2 in `h`; the 512^3 metric may still not converge.
- **O3.** A new pre-registered verification metric robust to localized under-resolution (for example percentile-based
  `|c - vD|`, or RMS excluding a pre-declared fraction of vertices), applied to the existing exports (`A/exports` is
  gitignored; eps 1.0/1.5 exports exist on the V100 host, the eps 0.5 ones were lost and must be regenerated). It
  would be a new criterion, not a reinterpretation of C. Cost: small (analysis only). Risk: it can only support a
  weaker claim than the pre-registered one, and the choice of percentile must be fixed before looking. Disclosure: the
  percentile distribution of the 128 vs 256 self-convergence of `c` at eps 1.0/1.5 has already been inspected
  post-hoc (`A/analysis/label_diagnostics.md`), so a percentile or exclusion fraction chosen now on these exports is
  not blind; a blind application needs a fresh field/seed or the `|c - vD|` distribution, which was not inspected.
- **O4.** Combinations (for example O1 now with O2 or O3 as a later increment for sigma_Y = 1.5).

## Caveats

- One field realization (SF-18 seed 3001), one correlation length `ell = 1/4` (`L/ell = 4`); no ensemble. The
  post-hoc concentration is a single-realization observation.
- The inlet `v1` comes from the 2nd-order SF-19 flux; `e_v` is limited to order 2 by it (derived expectation; observed
  2.00-2.02 at eps 0.5).
- Peak device memory is a device-wide poll, not a per-process allocation count; the values are
  identical across amplitudes at 128 and 256, which is consistent with the same allocation pattern.
- The eps 0.5 label exports were lost by an `rsync --delete` sync; `LD` and any export-based re-analysis cover eps 1.0
  and 1.5 only.
- `run.env` files contain a cosmetic second `COMMIT=` line (the `COMMIT` file has two lines).
- The A4 test asserts `landings_discarded` as a lower bound, not an exact count.
- The post-hoc diagnostics (`LD`) were written after the verdicts; they are exploratory and change no verdict.
- The wrap-check magnitudes at eps 1.5 grow with N; they are discrete derivative maxima, not jumps, but the check does
  not by itself rule out a localized defect smaller than ~1 * N.
- Not measured: other `ell`, other realizations, the long domain, transport, the return map (SF-34), `alpha_T`.
- The full `ctest`, the frozen-path `git diff` and the SF-27 byte-compare are orchestrator jobs: see the FINAL_AUDIT
  bitácora row of the increment file.

## Next step

Owner decision among O1-O4. SF-34 remains blocked until SF-35 is `done` on the default branch.
