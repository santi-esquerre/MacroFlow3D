# SF-29: equation (14) with `x1` non-periodic and inlet labels -- CPU prototype

- Date: 2026-10-02
- Status: complete (2026-10-06; human review of the SF-29 PR pending)
- Increment: `docs/plans/active/lester-eq14/increments/SF-29-eq14-inlet-labels-cpu-prototype.md`
- Theory: `docs/theory/lester-2023-key-claims.md` (equation (14), two-streamfunction representation);
  context `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`,
  `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`
- Artifacts: `docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/` (scripts, raw outputs, README)

The Question and Hypothesis sections were written BEFORE any candidate run (DAG node N1) and are kept as
written; results did not move them. The deviations D-5..D-7 were added during execution and are dated below. The
sections from "Build / environment" on were filled by DAG node N5 (2026-10-06) from the committed raw files.

## Question

Goal (exact): `Determinar con un prototipo CPU si la ecuación (14) generalizada a x1 no periódica, con etiquetas
fijadas en la cara de entrada, reproduce las etiquetas del flujo de Darcy real con e_v convergente bajo
refinamiento y sin floor.`

Does equation (14) (same-index pairing), posed on the slab `0 <= x1 <= 1` periodic in `x2, x3`, with the labels
fixed on the inlet face `x1 = 0` in flow coordinates, determine the labels of the actual Darcy flow (oracle:
backward streamline tracing of an independent spectral Darcy potential), with `e_v` converging under refinement,
no residual floor and no near-null Jacobian cluster -- for the periodic-cell flows of `FIELDS` and for one
constant-head-faces flow?

## Hypothesis

### Acceptance criteria (spec, verbatim)

- Criteria fixed before running, at amplitudes 0.25, 0.5, 1: (1) `e_v(h)` decreases with observed order >= 1.8 over three grids.
- (2) Labels agree with the oracle to the same order.
- (3) Relative nonlinear residual <= 1e-10 with no floor.
- (4) Dense Jacobian at 12^3-16^3 shows no near-null cluster: no gap-separated group below 1e-3 relative beyond the gauge modes explicitly fixed by the inlet data (report the relative singular-value spectrum).
- (5) Criteria (1)-(4) also hold in the constant-head case.

### Predictions P1-P6 (UNDERSTAND record section 2, verbatim; fixed before any run)

- **P1.** (14) + inlet Dirichlet + nothing at the outlet (the (14) rows at the outlet plane with one-sided stencils,
  the spec's literal candidate (i)) has a null space of dimension 2 per transverse mode (shear + potential):
  the dense Jacobian at 12^3/16^3 shows a gap-separated cluster of ~`2 (N_perp^2 - 1)` relative singular values at
  roundoff/truncation level (one-sided second-order stencils are exact on `x1`-linear functions). Criterion (4)
  fails; Newton converges to a solution contaminated by shear/potential modes, so `e_psi` and `e_v` do not
  converge at order 2 (criteria (1)-(2) fail or are erratic). Expected for cases (a) and (b).
- **P2.** Case (b): the Darcy labels satisfy `d1 psi_i = 0` on both faces (`v` normal to a constant-head face and
  `v . grad psi = 0`, face map nonsingular). (14) + inlet Dirichlet + outlet Neumann `d1 u_i = 0` has, at `k = 1`,
  `beta = 0` and the potential combination fixed (unique). The Darcy labels satisfy it, so for small amplitude the
  unique solution is the Darcy labels; expected to hold at amplitudes 0.25-1 (tested by (4)). This outlet condition
  is also the natural boundary condition of the dissipation energy at a free outlet (see 2.2), so candidate (i-b)
  and candidate (ii-free) discretize the same continuum problem.
- **P3.** Case (a): periodicity of the flow (`c(1, .) = c(0, .)`) removes the potential modes but **not** the
  shear (its `c`-perturbation is `x1`-independent). The shear is excluded only by the helicity equation, which is
  not in (14), or by prescribing the tangential outlet flow. The Darcy labels satisfy
  `c x e1 |_{x1=1} = v_D x e1 |_{x1=0}` (flow periodic; the inlet face velocity is already input data for the
  inlet labels). With the face map nonsingular this is a linear 2x2 solve for `(d1 psi1, d1 psi2)` at each outlet
  point in terms of the in-plane gradients and `v_perp(0, .)`; at `k = 1` it kills `beta` and fixes the potential
  combination (unique). For case (b) `v_perp = 0` and it reduces to P2's Neumann condition. This is the unified
  outlet condition the orchestrator adds to candidate (i) ("candidate (i-1)"); the spec-literal variant is run as
  the control ("candidate (i-0)"). Prediction: (i-1) satisfies (1)-(5); (i-0) fails (4).
- **P4.** The inlet Neumann/oblique condition is also satisfied by the Darcy labels but is not imposed (it would
  over-determine the discrete system); it is reported as a diagnostic.

P5 and P6 (UNDERSTAND record section 2.2, candidate (ii) bullets, verbatim):

- Case (b): minimize `E` over label pairs with inlet Dirichlet data, outlet free. The Darcy flow is the minimizer
  at fixed total flux (constant-head faces <-> free faces in the dual principle); the inlet data fix the inlet
  flux distribution consistently with it. Unique minimizer in `c` (strict convexity in `c`), unique labels for
  given `c` and inlet data (transport). Prediction P5: (ii-free) satisfies (1)-(5) in case (b).
- Case (a): the periodic Darcy flow is the Neumann-problem solution with normal flux `v1(0, .)` on **both** faces.
  Minimize `E` over label pairs with inlet Dirichlet data and the **outlet flux constraint**
  `c1(1, x2, x3) = v1(0, x2, x3)` (a constraint on the outlet-plane labels only: the in-plane face curl). Kelvin's
  principle then gives the periodic Darcy flow; labels unique. Prediction P6: (ii-constrained) satisfies (1)-(5)
  in case (a); criterion (4) is evaluated on the reduced Hessian (tangent space of the constraint) or on the KKT
  matrix with the multiplier block reported separately.

Refinement of P1 (UNDERSTAND record section 8, verbatim; orchestrator instrument, before any candidate run):

> Scratchpad instrument `lin_kernel_check.py` (linearized (14) at `k = 1` on the slab, vertex grid, centered interior
> stencils, inlet Dirichlet; `i0` = one-sided (14) rows at the outlet, `i1` = one-sided Neumann rows), dense SVD:
>
> | N | variant | exact nulls (< 1e-10) | < 1e-3 | < 1e-2 | min rel sv |
> |---|---|---|---|---|---|
> | 8 | i0 | 16 = 2N | 20 | 44 | 3e-18 |
> | 8 | i1 | 0 | 0 | 44 | 1.1e-3 |
> | 10 | i0 | 20 = 2N | 32 | 60 | 6e-19 |
> | 10 | i1 | 0 | 20 | 72 | 6.3e-4 |
>
> Correction to P1's count: the **exactly** null discrete modes of `i0` are the `2N` shear modes with `xi2 = 0` or
> `xi3 = 0` (for those the `d23` coupling vanishes and `d11 (x1) = 0` is exact for the one-sided stencil). The
> remaining continuum shear/potential modes (`xi2 xi3 != 0`) are only approximately null discretely because the
> symbols of the centered `d22`/`d33` and `d23` stencils do not cancel on `x1`-linear modes (O(h^2) defect), so they
> appear as a dense low cluster between the exact nulls and the elliptic tail, not as a second exact cluster. The
> discriminating signature of P1 is therefore: `2N` relative singular values at roundoff (1e-16..1e-18) separated by
> >= 10 decades from the rest. For `i1` no exact nulls; the smallest relative singular value scales ~`h^2`
> (6.3e-4 at N = 10 -> expected ~2.5e-4 at 16^3), i.e. the normal elliptic tail crosses the spec's 1e-3 threshold at
> 12^3-16^3 **without** a gap; criterion (4) is read on the gap-separated cluster, as its text says. The nonlinear
> Jacobians at amplitude 0.25-1 are expected to inherit this structure; the candidates' reports must show the sorted
> spectrum so the gap is visible. (This refinement is recorded in the bitácora at the next metadata commit.)
>
> Extension of the table (same instrument), 2026-10-03T00:35Z:
>
> | N | variant | exact nulls (< 1e-10) | < 1e-3 | < 1e-2 | min rel sv |
> |---|---|---|---|---|---|
> | 12 | i0 | 24 = 2N | 52 | 84 | 2.6e-18 |
> | 12 | i1 | 0 | 28 | 140 | 4.0e-4 |
> | 16 | i0 | 32 = 2N | 104 | 212 | 1.3e-18 |
> | 16 | i1 | 0 | 52 | 556 | 2.0e-4 |
>
> `i1` minimum scales as `h^2` (4.0e-4 -> 2.0e-4 for 12 -> 16; ratio 2.0 vs (16/12)^2 = 1.8), no gap. At 16^3 the
> normal tail already has ~50 values below the spec's 1e-3 threshold; the criterion is therefore applied as "no
> gap-separated group" and the audit requires the sorted spectrum. Expected `i0` signature at 16^3: 32 values at
> 1e-16..1e-18, next values >= ~1e-5 (approximate shear/potential modes), then the tail.

### Deviations from the literal specification (recorded for the human reviewer)

From the SF-29 bitácora UNDERSTAND row (verbatim):

> Deviations recorded for the reviewer: D-1 the triangular inlet construction is normalized (`psi2 = int_0^x3 Q`, `psi1 = int_0^x2 v1 / Q(x3)`) so both labels are affine + periodic in (x2, x3) (the spec's literal `psi2 = x3` gives an `x3`-dependent jump of `psi1`); D-2 outlet condition for (i) as above; D-3 outlet flux constraint for (ii) in the periodic case; D-4 constant-head case realized by the mirror trick `k(g(x1), x2, x3)`, `g = (1 - cos pi x1)/2` on a length-2 periodic cell (odd potential -> constant head, `v_perp = 0` on the faces), reusing the closure-probe spectral solver.

- D-1 (inlet labels; implemented in N1, `scripts/inlet.py`). UNDERSTAND record section 1 (verbatim):

  > Inlet data in flow coordinates (spec: "triangular construction"). **Orchestrator refinement D-1** (recorded for
  > the reviewer): the spec's literal `psi1 = cumulative face flux in x2 at fixed x3, psi2 = x3` makes `psi1` jump by
  > `Q(x3) = int_0^1 v1(0, s, x3) ds` across one `x2` period, and `Q` varies with `x3` for a generic field, so `psi1`
  > would not be affine + periodic in `x2` and no periodic unknown could represent it. The normalized triangular
  > construction keeps the spec's intent and the face-Jacobian identity:
  >
  > ```text
  > Q(x3)   = int_0^1 v1(0, s, x3) ds                      (> 0)
  > psi2^0  = int_0^{x3} Q(t) dt                            (= x3 + periodic; total flux = 1 -> jump 1)
  > psi1^0  = int_0^{x2} v1(0, s, x3) ds / Q(x3)            (= x2 + periodic; jump 1)
  > d2 psi1^0 d3 psi2^0 - d3 psi1^0 d2 psi2^0 = (v1/Q) Q - 0 = v1   (on the face)
  > ```

- D-2: outlet condition of candidate (i-1), `grad psi1 x grad psi2 x e1 |_{x1=1} = v_D x e1 |_{x1=0}` (P3);
  the spec-literal outlet is run as the control (i-0).
- D-3: outlet flux constraint `c1(1, x2, x3) = v1(0, x2, x3)` for candidate (ii) in the periodic case (P6).
- D-4: the constant-head case is realized by the mirror trick `k_b(x) = exp(eps f(g(x1), x2, x3))`,
  `g(x1) = (1 - cos(pi x1))/2`, on the length-2 periodic cell `[0, 2] x [0, 1]^2` (grid `2 N_phi x N_phi x N_phi`,
  mean gradient along `x1`): the fluctuation potential is odd about `x1 = 0` and `x1 = 1`, so the head is constant
  and `v_perp = 0` on both faces of the slab (implemented in N1, `scripts/reference.py`; field name `gauss_ch`).

Deviations added during execution (SF-29 bitácora rows 2026-10-03T03:05Z and 2026-10-05T14:30Z; recorded for the
reviewer, criteria (1)-(5) unchanged):

- D-5 (orchestrator, pre-registered 2026-10-03T03:05Z, before any candidate run): reading rule against the
  oracle-FD ceiling. The sweep reports `e_v^or(h)`, the same discrete reconstruction applied to the exact oracle
  labels, next to every candidate; (1)/(2) are evaluated as written; a cell that misses 1.8 is classed
  `ceiling-limited` (not a pass) iff `e_v <= 1.5 e_v^or` on every grid and the orders differ by <= 0.15;
  `e_v > 1.5 e_v^or` or a floor is FAIL. The reviewer adjudicates the reading. Reason: N1 found that the exact
  labels reconstructed with second-order FD are pre-asymptotic at `eps = 1` on 16/32/48 (see "Outputs inspected",
  N1). The operational form used by the corrective sweep is printed at the top of `raw/sweep2/summary.md`.
- D-6 (orchestrator, 2026-10-03T03:05Z, extended 2026-10-03T06:40Z): grid extension beyond the spec's 48^3
  (`N = 64` for `gauss`, `gauss_ch`, `control2d` with (i-1)), evidence only.
- D-7 (owner directive, 2026-10-05, explicit choice among three options presented by the orchestrator): a
  corrective cycle plus a 4th-order variant of candidate (i-1) (`i1o4`: 4th-order stencils, 4th-order outlet
  row, 4th-order reconstruction for `e_v` and its ceiling), direct linear solves, and the reduced corrective
  sweep on 16/20/24/28(/32). This is a scope extension beyond the spec's two candidates; the ladders stay inside
  the spec's 16^3-48^3 range.

Process notes recorded for the reviewer: the corrective nodes C-ii (candidate (ii) in Q1 form plus two
mean-transverse-flux rows, completing D-3) and C-i4 (per-label `e_psi` metric fix, saved solutions, `i1o4`) were
commissioned by the orchestrator after the audits of N3 and N4; the outcome framing ("Option A", below) is an
owner decision of 2026-10-06.

## Build / environment

No project binary; numpy/scipy only, CPU, double precision. Artifact directory
`docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/` (README sections "Scripts", "N4 sweep", "Corrective
sweep").

- Local WSL (oracle development, N1 tables, single-case smokes, 12^3 dense spectra, C-i4 checks up to 24^3):
  Python 3.13 / numpy 2.5.0 / scipy 1.18.0, 16 cores shared with other work (timings indicative).
  `--selftest`, `--gradtest`, `--selfcheck` and `--jactest` were also run under Python 3.11.16 / numpy 1.26.4 /
  scipy 1.11.4 (`raw/*_py311.txt`).
- V100 host, CPU only (both sweeps): `localhost.localdomain`, Python 3.11.7 / numpy 1.26.4 / scipy 1.11.4.
  Mirror `~/MacroFlow3D-SF-29` (`scripts/remote --increment SF-29`), per-increment state root
  `~/.macroflow3d-remote/macroflow3d-SF-29/`. Both jobs held the GPU 0 lock for their whole duration
  (`GPU=0 (CUDA_VISIBLE_DEVICES=0, request=auto)` in the job logs); no CUDA code ran.

| job | start / end (UTC) | wall | parallelism | cells | source |
|---|---|---|---|---|---|
| `sf29-run-all` (N4, first sweep) | 2026-10-03T06:19Z / 2026-10-03T20:59Z | 52 822 s, exit 0 | 22 workers x 3 BLAS threads | 338: 305 done, 33 timeout, 0 failed | `raw/sweep/summary.md`, `raw/sweep/timeouts.md`, `raw/sweep/sf29-run-all.joblog.txt` |
| `sf29-corrective` (N4c, corrective sweep) | 2026-10-05T12:11Z / 2026-10-06T04:13Z | 57 705 s, exit 0 | up to 26 cells x 3 threads, memory budget 100 GB (max reserved 100.0 GB) | 166: 163 done, 3 timeout, 0 failed | `raw/sweep2/summary.md`, `raw/sweep2/timeouts.md`, `raw/sweep2/sf29-corrective.joblog.txt` |

## Config(s)

- Fields: `control2d`, `lester2021`, `lester_brk`, `two_mode`, `generic3d`, `gauss` (closure-probe `FIELDS`,
  periodic cell, mean flux exactly `e1`) and `gauss_ch` (constant-head faces, D-4). `gauss` is the band-limited
  Gaussian-covariance field of the closure probes (`|m_i| <= 3`, `ell = 1/4`, unit variance, seed 7;
  `docs/experiments/artifacts/2026-10-02-closure-probes/scripts/closure_probe.py`), so `eps = sigma_Y` and
  `L/ell = 4`.
- Amplitudes `eps` in {0.25, 0.5, 1}; `k = exp(eps f)`.
- Reference resolution `N_phi(field, eps)` from `oracle.py --nphi` (21-row table in the artifact README,
  `raw/oracle_nphi_table.txt`), independent of `N`.
- Oracle: backward DOP853 tracing (rtol 1e-12, atol 1e-14) of every vertex to the inlet; inlet labels D-1.
- First sweep (`raw/sweep/`, N4 matrix): all 7 fields x 3 amplitudes; (i-1) `i1` (2nd order) on 16/24/32/48
  (+64 for `gauss`, `gauss_ch`, `control2d`, D-6), linear solves `splu` up to 16^3 and GMRES with the per-mode
  `k = 1` preconditioner above; (i-0) `i0` at 16^3 only; (ii) Q1 energy + D-3 rows on 16/24/32 (4 h cap); dense
  spectra 12^3/16^3; consistency ladders.
- Corrective sweep (`raw/sweep2/`, N4c matrix, D-7): `gauss`, `gauss_ch`, `control2d`, `generic3d` x 3 amplitudes;
  `i1o4` (4th order) on 16/20/24/28 (+32 for `gauss:0.25`, `gauss_ch:0.25`); `i1` (2nd order) on 16/20/24/28/32
  for `gauss`, `gauss_ch`, `control2d` and 20/28 for `generic3d` (16/24 from the first sweep); both with direct
  `splu` solves (`--direct-max 32 --reuse-lu 0`); (ii) Q1 at 20 (and 16/24 for `control2d`, 32 for
  `gauss:0.25`, `gauss_ch:0.25`); `spec4` dense spectra of `i1o4` at 12^3/16^3; `cons4` residual of `i1o4` at the
  oracle labels on 16/24/28.

## Commands

N1 oracle commands: artifact README, section "Commands (N1)" (`oracle.py --selftest`, `--convergence`, `--nphi`,
`--returnmap`, `--midplane --grids 16,24,32,48,64,96,128`). Candidate CLIs (run from `scripts/`):

```bash
python3 candidate_i.py field:eps:N:i0|i1 [--order 4] [--direct-max 32 --reuse-lu 0] [--save DIR]
python3 candidate_i.py --consistency field:eps --order 4 --grids 16,24,28
python3 candidate_i.py --spectrum M field:eps:M:i1 --order 4 --out ../raw/sweep2/spectra
python3 candidate_i.py --k1check N [--order 4]
python3 candidate_ii.py field:eps:N            # Q1 energy (default) + D-3 rows
python3 candidate_ii.py --energy whitney ...   # N3 control with the hourglass kernel
```

The two remote jobs (detached, `scripts/remote --increment SF-29 run <job> -- "..."`):

```bash
# sf29-run-all
cd docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/scripts && OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 python3 run_all.py --workers 22 --out ../raw/sweep
# sf29-corrective
cd docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/scripts && python3 run_all.py --matrix corrective --workers 26 --mem-budget 100 --threads 3 --out ../raw/sweep2
```

Summaries and digests (no computation): `python3 run_all.py [--matrix corrective] --summarize --out ...` (both
regenerated locally, byte-identical to the remote files) and `python3 sweep_digest.py --out ...`.
Retrieval commands: artifact README, sections "N4 sweep" and "Corrective job record and retrieval (N4c)".

## Outputs inspected

All numbers in this section are transcribed from the named file; orders are those printed by the summaries.

### N1: oracle self-consistency and resolution

- Oracle checks (`raw/oracle_selftest.txt`): face Jacobian identity 1.2e-15, round trip <= 4.7e-13,
  `max|v_perp|` on the constant-head faces ~2e-16; return maps equal to the 2026-10-02 closure note to 4 digits
  (`raw/oracle_returnmap.txt`).
- Self-consistency of the exact labels under the 2nd-order FD reconstruction on 16/32/48
  (`raw/oracle_convergence.txt`, `raw/oracle_convergence_gauss1.txt`): `gauss:0.25` `e_v` 1.673e-2 / 4.328e-3 /
  1.936e-3 (orders 1.95, 1.98); `gauss:1` `e_v` 0.235 / 0.125 / 7.54e-2 (orders 0.91, 1.25), `e_div` orders
  0.31, 0.79, `min|c|` 0.0572 / 0.0062 / 0.0205.
- Mid-slab probe (3 planes, centered FD, no boundary stencils; `raw/oracle_midplane.txt`), `e_v_mid` orders over
  16-24-32-48-64-96-128:

  | case | orders |
  |---|---|
  | `gauss:0.25` | 1.93 1.97 1.98 1.99 2.00 2.00 |
  | `gauss:0.5` | 1.72 1.82 1.90 1.95 1.97 1.99 |
  | `gauss:1.0` | 0.62 1.21 1.16 1.32 1.63 1.76 |
  | `gauss_ch:0.25` | 1.91 1.95 1.98 1.99 1.99 2.00 |
  | `gauss_ch:1.0` | 1.27 0.23 1.12 1.22 1.41 1.62 |
  | `lester_brk:1.0` | 1.84 1.87 1.93 1.96 1.98 1.99 |
  | `generic3d:1.0` | 1.19 1.35 1.51 1.70 1.82 1.90 |

  At `eps = 1` the second-order regime of the exact labels is reached only on the last pair, 96 -> 128 (1.76),
  for `gauss`, i.e. at `ell/h = 24-32` (`ell = 1/4`).

### k = 1 controls (`raw/cand_i_smoke_k1check.txt`, `raw/cand_i4_k1check.txt`)

| N | variant | exact nulls (< 1e-10) | next / smallest relative sv | < 1e-3 |
|---|---|---|---|---|
| 12 | `i0` | 24 = 2N | 6.23e-06 | 52 |
| 12 | `i1` | 0 | 1.81e-03 | 0 |
| 16 | `i0` | 32 = 2N | 5.75e-07 | 104 |
| 16 | `i1` | 0 | 1.05e-03 | 0 |
| 12 | `i1o4` | 0 | 8.31e-04 | 28 |

### First sweep (`raw/sweep/summary.md`, `raw/sweep/timeouts.md`)

- `i0` (spec-literal outlet, 16^3): criterion (3) FAIL in 17 of 21 (field, eps) cells (plateaus at `r_F`
  2.8e-4 .. 0.89); the four converged cells are `control2d` at all three amplitudes and `lester2021:0.25`. Example
  `gauss:0.25`: path `0.25(fail)->0.125(fail)->0.0625(fail)->0.25(final)->LM`, `r_F = 1.140e-2`, `e_psi = 1.043`.
- `i1` (2nd order, GMRES + `k = 1` preconditioner above 16^3): criterion (4) PASS in 21/21. 33 timeouts in the
  sweep: `i1` at 48^3 for `gauss`:{0.5, 1}, `gauss_ch`:{0.25, 0.5, 1}, `generic3d`:{0.5, 1}, `lester2021:1`,
  `lester_brk:1`, `two_mode:1` and every 64^3 `gauss`/`gauss_ch` cell; (ii) at 32^3 in 17 cells. GMRES statistics
  of the slow cells: median 900 iterations per Newton step, caps (6000) and restart-stagnation stops;
  `i1-gauss-0.5-N32` ended after 5 bisections at `r_F = 8.614e-3` with 17 GMRES steps at the cap and
  `sum t_lin` 13 363 s of 13 983 s (`raw/sweep/timeouts.md`).
- `i1`, `gauss:0.25`, 16/24/32/48: `e_v` 1.736e-2 / 1.038e-2 / 7.157e-3 / 4.053e-3 (orders 1.27, 1.29, 1.40),
  `e_v`/ceiling 1.04 -> 2.09, `e_psi` 0.2165 -> 0.06834 (orders 0.81, 1.05, 1.28).
- `i1`, `control2d:0.5`, 16..64: `e_v` orders 2.01 2.00 2.00 2.00 at 1.14-1.17x the ceiling; `e_psi` orders
  1.09 1.47 1.67 1.79.
- Metric artifact (superseded): `e_psi` of `control2d` at `eps` 0.25 and 1.0 reads ~0.46 flat (0.4673 -> 0.4579
  over 16..64 at `eps = 0.25`) because `psi2`'s true periodic part is zero and the stored one is `(Q0 - 1) x3`
  (roundoff divided by roundoff). Corrected by the per-label normalization of C-i4 (artifact README, "Shared
  conventions"); the corrected values in `raw/sweep2/summary.md` give `e_psi2` between 2.3e-13 and 2.4e-10 for `control2d`.

### Corrective sweep, candidate (i-1) 4th order `i1o4` (`raw/sweep2/summary.md`)

Ceiling `oracle_fd4` (4th-order reconstruction of the exact labels). Timeout rows are the cells killed at the
21 600 s cap.

#### `i1o4`, gauss, eps = 0.25

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | converged | 8.631e-14 | 10 | 5.506e-03 | 3.291e-03 | 1.67 | 7.727e-02 | 7.727e-02 | 5.859e-02 |
| 20 | done | converged | 4.307e-15 | 13 | 2.984e-03 | 1.500e-03 | 1.99 | 4.584e-02 | 4.584e-02 | 3.349e-02 |
| 24 | done | converged | 6.363e-15 | 7 | 1.808e-03 | 7.699e-04 | 2.35 | 2.886e-02 | 2.886e-02 | 2.106e-02 |
| 28 | done | converged | 8.525e-15 | 8 | 1.175e-03 | 4.340e-04 | 2.71 | 1.902e-02 | 1.902e-02 | 1.414e-02 |
| 32 | done | converged | 1.108e-14 | 8 | 8.014e-04 | 2.620e-04 | 3.06 | 1.303e-02 | 1.303e-02 | 9.952e-03 |

Orders over completed N = 16/20/24/28/32: e_v 2.75 2.75 2.80 2.87; e_psi 2.34 2.54 2.70 2.83; ceiling 3.52 3.66 3.72 3.78. LSQ slope log(e) vs log(h): e_v 2.78; e_psi 2.57; ceiling 3.65.

- PATH N=16: `0.25`
- PATH N=20: `0.25`
- PATH N=24: `0.25(fail)->0.125->0.25`
- PATH N=28: `0.25(fail)->0.125->0.25`
- PATH N=32: `0.25(fail)->0.125->0.25`

#### `i1o4`, gauss, eps = 0.5

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | converged | 5.431e-15 | 7 | 1.908e-02 | 2.149e-02 | 0.89 | 1.403e-01 | 1.403e-01 | 9.954e-02 |
| 20 | done | converged | 1.032e-14 | 7 | 1.191e-02 | 1.157e-02 | 1.03 | 9.365e-02 | 9.365e-02 | 6.330e-02 |
| 24 | done | converged | 1.224e-14 | 8 | 8.474e-03 | 7.278e-03 | 1.16 | 6.831e-02 | 6.831e-02 | 4.405e-02 |
| 28 | done | converged | 1.696e-14 | 8 | 6.382e-03 | 4.541e-03 | 1.41 | 5.159e-02 | 5.159e-02 | 3.295e-02 |

Orders over completed N = 16/20/24/28: e_v 2.11 1.87 1.84; e_psi 1.81 1.73 1.82; ceiling 2.77 2.54 3.06. LSQ slope log(e) vs log(h): e_v 1.95; e_psi 1.78; ceiling 2.75.

- PATH N=16: `0.25->0.5(fail)->0.375->0.5`
- PATH N=20: `0.25->0.5(fail)->0.375->0.5`
- PATH N=24: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5`
- PATH N=28: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5`

#### `i1o4`, gauss, eps = 1

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | stagnation | 5.800e-01 | 5 | 8.473e-02 | 2.714e-01 | 0.31 | 3.787e-01 | 3.787e-01 | 2.829e-01 |
| 20 | done | stagnation | 7.621e-01 | 5 | 7.842e-02 | 1.964e-01 | 0.40 | 3.607e-01 | 3.607e-01 | 2.766e-01 |
| 24 | done | stagnation | 8.441e-01 | 5 | 7.343e-02 | 1.755e-01 | 0.42 | 3.381e-01 | 3.381e-01 | 2.446e-01 |
| 28 | timeout | - | - | - | - | 1.442e-01 | - | - | - | - |

Orders over completed N = 16/20/24: e_v 0.35 0.36; e_psi 0.22 0.35; ceiling 1.45 0.62. LSQ slope log(e) vs log(h): e_v 0.35; e_psi 0.28; ceiling 1.09.

- PATH N=16: `0.25->0.5(fail)->0.375->0.5->1(fail)->0.75->1(fail)->0.875(fail)->0.8125->0.875(fail)->1(final)`
- PATH N=20: `0.25->0.5(fail)->0.375->0.5->1(fail)->0.75(fail)->0.625->0.75->1(fail)->0.875(fail)->1(final)`
- PATH N=24: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5->1(fail)->0.75(fail)->0.625->0.75->1(fail)`

#### `i1o4`, gauss_ch, eps = 0.25

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | converged | 3.407e-15 | 7 | 5.287e-03 | 3.447e-03 | 1.53 | 6.780e-02 | 6.780e-02 | 4.139e-02 |
| 20 | done | converged | 5.462e-15 | 7 | 2.945e-03 | 1.566e-03 | 1.88 | 4.004e-02 | 4.004e-02 | 2.422e-02 |
| 24 | done | converged | 8.914e-15 | 7 | 1.783e-03 | 8.060e-04 | 2.21 | 2.483e-02 | 2.483e-02 | 1.503e-02 |
| 28 | done | converged | 1.056e-14 | 8 | 1.141e-03 | 4.559e-04 | 2.50 | 1.608e-02 | 1.608e-02 | 9.847e-03 |
| 32 | done | converged | 1.396e-14 | 8 | 7.657e-04 | 2.759e-04 | 2.78 | 1.086e-02 | 1.086e-02 | 6.777e-03 |

Orders over completed N = 16/20/24/28/32: e_v 2.62 2.75 2.90 2.99; e_psi 2.36 2.62 2.82 2.94; ceiling 3.54 3.64 3.70 3.76. LSQ slope log(e) vs log(h): e_v 2.79; e_psi 2.64; ceiling 3.64.

- PATH N=16: `0.25(fail)->0.125->0.25`
- PATH N=20: `0.25(fail)->0.125->0.25`
- PATH N=24: `0.25(fail)->0.125->0.25`
- PATH N=28: `0.25(fail)->0.125->0.25`
- PATH N=32: `0.25(fail)->0.125->0.25`

#### `i1o4`, gauss_ch, eps = 0.5

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | converged | 7.014e-15 | 7 | 1.785e-02 | 1.941e-02 | 0.92 | 1.232e-01 | 1.232e-01 | 6.081e-02 |
| 20 | done | converged | 1.124e-14 | 7 | 1.204e-02 | 1.204e-02 | 1.00 | 8.657e-02 | 8.657e-02 | 4.414e-02 |
| 24 | done | converged | 1.613e-14 | 8 | 8.419e-03 | 6.766e-03 | 1.24 | 6.221e-02 | 6.221e-02 | 3.259e-02 |
| 28 | done | converged | 2.212e-14 | 8 | 6.142e-03 | 4.559e-03 | 1.35 | 4.586e-02 | 4.586e-02 | 2.430e-02 |

Orders over completed N = 16/20/24/28: e_v 1.76 1.96 2.05; e_psi 1.58 1.81 1.98; ceiling 2.14 3.16 2.56. LSQ slope log(e) vs log(h): e_v 1.91; e_psi 1.76; ceiling 2.63.

- PATH N=16: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5`
- PATH N=20: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5`
- PATH N=24: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5`
- PATH N=28: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5`

#### `i1o4`, gauss_ch, eps = 1

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | stagnation | 1.409e+00 | 4 | 8.993e-02 | 2.995e-01 | 0.30 | 4.105e-01 | 4.105e-01 | 2.205e-01 |
| 20 | done | linesearch-fail | 2.750e+00 | 3 | 9.766e-02 | 2.269e-01 | 0.43 | 4.096e-01 | 4.096e-01 | 2.247e-01 |
| 24 | done | stagnation | 2.289e+00 | 5 | 1.258e-01 | 1.974e-01 | 0.64 | 4.356e-01 | 4.356e-01 | 2.824e-01 |
| 28 | timeout | - | - | - | - | 1.632e-01 | - | - | - | - |

Orders over completed N = 16/20/24: e_v -0.37 -1.39; e_psi 0.01 -0.34; ceiling 1.24 0.76. LSQ slope log(e) vs log(h): e_v -0.81; e_psi -0.14; ceiling 1.04.

- PATH N=16: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5->1(fail)->0.75(fail)->0.625->0.75->1(fail)`
- PATH N=20: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5->1(fail)->0.75(fail)->0.625->0.75->1(fail)`
- PATH N=24: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5->1(fail)->0.75(fail)->0.625->0.75(fail)->1(final)`

#### `i1o4`, generic3d, eps = 0.25

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | converged | 2.688e-15 | 7 | 3.423e-03 | 1.744e-03 | 1.96 | 4.634e-02 | 2.794e-02 | 4.634e-02 |
| 20 | done | converged | 4.231e-15 | 7 | 1.505e-03 | 6.936e-04 | 2.17 | 2.177e-02 | 1.592e-02 | 2.177e-02 |
| 24 | done | converged | 6.022e-15 | 7 | 7.858e-04 | 3.218e-04 | 2.44 | 1.200e-02 | 9.408e-03 | 1.200e-02 |
| 28 | done | converged | 8.134e-15 | 7 | 4.571e-04 | 1.672e-04 | 2.73 | 7.225e-03 | 5.839e-03 | 7.225e-03 |

Orders over completed N = 16/20/24/28: e_v 3.68 3.56 3.51; e_psi 3.39 3.27 3.29; ceiling 4.13 4.21 4.25. LSQ slope log(e) vs log(h): e_v 3.60; e_psi 3.32; ceiling 4.19.

- PATH N=16: `0.25`
- PATH N=20: `0.25`
- PATH N=24: `0.25`
- PATH N=28: `0.25`

#### `i1o4`, generic3d, eps = 0.5

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | converged | 5.521e-15 | 7 | 1.460e-02 | 7.435e-03 | 1.96 | 1.017e-01 | 7.798e-02 | 1.017e-01 |
| 20 | done | converged | 2.265e-14 | 7 | 7.516e-03 | 3.632e-03 | 2.07 | 5.176e-02 | 4.836e-02 | 5.176e-02 |
| 24 | done | converged | 1.243e-14 | 9 | 4.769e-03 | 1.977e-03 | 2.41 | 3.384e-02 | 3.311e-02 | 3.384e-02 |
| 28 | done | converged | 1.698e-14 | 11 | 3.322e-03 | 1.164e-03 | 2.85 | 2.403e-02 | 2.361e-02 | 2.403e-02 |

Orders over completed N = 16/20/24/28: e_v 2.98 2.50 2.35; e_psi 3.03 2.33 2.22; ceiling 3.21 3.34 3.44. LSQ slope log(e) vs log(h): e_v 2.64; e_psi 2.57; ceiling 3.31.

- PATH N=16: `0.25->0.5`
- PATH N=20: `0.25->0.5`
- PATH N=24: `0.25->0.5`
- PATH N=28: `0.25->0.5`

#### `i1o4`, generic3d, eps = 1

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | converged | 1.580e-14 | 9 | 1.034e-01 | 1.634e-01 | 0.63 | 3.468e-01 | 3.468e-01 | 2.900e-01 |
| 20 | done | converged | 1.969e-14 | 14 | 5.885e-02 | 1.350e-01 | 0.44 | 2.342e-01 | 2.342e-01 | 1.685e-01 |
| 24 | done | converged | 2.835e-14 | 9 | 4.400e-02 | 1.136e-01 | 0.39 | 1.782e-01 | 1.782e-01 | 1.234e-01 |
| 28 | timeout | - | - | - | - | 9.665e-02 | - | - | - | - |

Orders over completed N = 16/20/24: e_v 2.53 1.59; e_psi 1.76 1.50; ceiling 0.86 0.95. LSQ slope log(e) vs log(h): e_v 2.12; e_psi 1.65; ceiling 0.90.

- PATH N=16: `0.25->0.5->1(fail)->0.75->1(fail)->0.875->1(fail)->0.9375->1`
- PATH N=20: `0.25->0.5->1(fail)->0.75->1(fail)->0.875(fail)->0.8125->0.875->1`
- PATH N=24: `0.25->0.5->1(fail)->0.75(fail)->0.625->0.75->1(fail)->0.875->1(fail)->0.9375->1`

#### `i1o4`, control2d, eps = 0.25

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | converged | 1.532e-15 | 2 | 3.067e-03 | 1.715e-03 | 1.79 | 3.220e-02 | 3.220e-02 | 5.175e-12 |
| 20 | done | converged | 2.359e-15 | 2 | 1.129e-03 | 6.740e-04 | 1.68 | 5.592e-03 | 5.592e-03 | 5.227e-12 |
| 24 | done | converged | 3.393e-15 | 2 | 5.247e-04 | 3.153e-04 | 1.66 | 2.667e-03 | 2.667e-03 | 5.261e-12 |
| 28 | done | converged | 4.711e-15 | 2 | 2.772e-04 | 1.661e-04 | 1.67 | 1.883e-03 | 1.883e-03 | 5.286e-12 |

Orders over completed N = 16/20/24/28: e_v 4.48 4.20 4.14; e_psi 7.85 4.06 2.26; ceiling 4.19 4.17 4.16. LSQ slope log(e) vs log(h): e_v 4.29; e_psi 5.08; ceiling 4.17.

- PATH N=16: `0.25`
- PATH N=20: `0.25`
- PATH N=24: `0.25`
- PATH N=28: `0.25`

#### `i1o4`, control2d, eps = 0.5

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | converged | 2.980e-15 | 2 | 7.687e-03 | 4.310e-03 | 1.78 | 3.183e-02 | 3.183e-02 | 2.334e-13 |
| 20 | done | converged | 4.506e-15 | 2 | 3.204e-03 | 1.727e-03 | 1.86 | 1.420e-02 | 1.420e-02 | 2.332e-13 |
| 24 | done | converged | 6.582e-15 | 2 | 1.618e-03 | 8.256e-04 | 1.96 | 1.065e-02 | 1.065e-02 | 2.347e-13 |
| 28 | done | converged | 8.980e-15 | 2 | 8.742e-04 | 4.447e-04 | 1.97 | 6.428e-03 | 6.428e-03 | 2.358e-13 |

Orders over completed N = 16/20/24/28: e_v 3.92 3.75 3.99; e_psi 3.62 1.58 3.28; ceiling 4.10 4.05 4.01. LSQ slope log(e) vs log(h): e_v 3.87; e_psi 2.75; ceiling 4.06.

- PATH N=16: `0.25->0.5`
- PATH N=20: `0.25->0.5`
- PATH N=24: `0.25->0.5`
- PATH N=28: `0.25->0.5`

#### `i1o4`, control2d, eps = 1

| N | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | done | converged | 5.430e-15 | 2 | 3.869e-02 | 1.267e-02 | 3.05 | 2.314e-01 | 2.314e-01 | 2.327e-10 |
| 20 | done | converged | 8.364e-15 | 2 | 2.533e-02 | 5.064e-03 | 5.00 | 1.776e-01 | 1.776e-01 | 2.347e-10 |
| 24 | done | converged | 1.117e-14 | 2 | 1.145e-02 | 2.413e-03 | 4.75 | 8.070e-02 | 8.070e-02 | 2.359e-10 |
| 28 | done | converged | 1.551e-14 | 2 | 5.289e-03 | 1.322e-03 | 4.00 | 3.670e-02 | 3.670e-02 | 2.368e-10 |

Orders over completed N = 16/20/24/28: e_v 1.90 4.35 5.01; e_psi 1.19 4.33 5.11; ceiling 4.11 4.07 3.90. LSQ slope log(e) vs log(h): e_v 3.57; e_psi 3.31; ceiling 4.04.

- PATH N=16: `0.25->0.5->1`
- PATH N=20: `0.25->0.5->1`
- PATH N=24: `0.25->0.5->1`
- PATH N=28: `0.25->0.5->1`

### Corrective sweep, candidate (i-1) 2nd order `i1`, direct solves (`raw/sweep2/summary.md`)

Ceiling `oracle_fd`. Every `gauss`/`gauss_ch` cell at `eps <= 0.5` converged to `r_F <= 5.2e-14` with direct
solves, including `gauss:0.5:32` (`r_F = 4.630e-14`), which had stagnated under GMRES in the first sweep.

#### `i1`, gauss, eps = 0.25

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | converged | 1.490e-15 | 7 | 1.736e-02 | 1.673e-02 | 1.04 | 2.165e-01 | 2.165e-01 | 1.954e-01 |
| 20 | converged | 5.862e-14 | 7 | 1.306e-02 | 1.088e-02 | 1.20 | 1.830e-01 | 1.830e-01 | 1.568e-01 |
| 24 | converged | 3.287e-15 | 8 | 1.038e-02 | 7.624e-03 | 1.36 | 1.556e-01 | 1.556e-01 | 1.287e-01 |
| 28 | converged | 4.516e-15 | 6 | 8.528e-03 | 5.632e-03 | 1.51 | 1.332e-01 | 1.332e-01 | 1.076e-01 |
| 32 | converged | 5.844e-15 | 6 | 7.157e-03 | 4.328e-03 | 1.65 | 1.149e-01 | 1.149e-01 | 9.141e-02 |

Orders over completed N = 16/20/24/28/32: e_v 1.28 1.26 1.27 1.31; e_psi 0.75 0.89 1.01 1.11; ceiling 1.93 1.95 1.96 1.97. LSQ slope log(e) vs log(h): e_v 1.28; e_psi 0.91; ceiling 1.95.

#### `i1`, gauss, eps = 0.5

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | converged | 3.046e-15 | 10 | 4.004e-02 | 4.136e-02 | 0.97 | 2.802e-01 | 2.802e-01 | 2.196e-01 |
| 20 | converged | 4.624e-15 | 12 | 3.120e-02 | 2.773e-02 | 1.13 | 2.412e-01 | 2.412e-01 | 1.794e-01 |
| 24 | converged | 5.168e-14 | 9 | 2.567e-02 | 2.004e-02 | 1.28 | 2.101e-01 | 2.101e-01 | 1.507e-01 |
| 28 | converged | 8.851e-15 | 14 | 2.181e-02 | 1.505e-02 | 1.45 | 1.847e-01 | 1.847e-01 | 1.295e-01 |
| 32 | converged | 4.630e-14 | 6 | 1.893e-02 | 1.172e-02 | 1.62 | 1.638e-01 | 1.638e-01 | 1.132e-01 |

Orders over completed N = 16/20/24/28/32: e_v 1.12 1.07 1.06 1.06; e_psi 0.67 0.76 0.84 0.90; ceiling 1.79 1.78 1.86 1.87. LSQ slope log(e) vs log(h): e_v 1.08; e_psi 0.77; ceiling 1.82.

#### `i1`, gauss, eps = 1

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | converged | 7.461e-15 | 10 | 1.028e-01 | 2.350e-01 | 0.44 | 4.433e-01 | 4.433e-01 | 3.047e-01 |
| 20 | converged | 9.539e-15 | 15 | 8.564e-02 | 1.878e-01 | 0.46 | 4.121e-01 | 4.121e-01 | 2.590e-01 |
| 24 | converged | 1.471e-14 | 7 | 7.540e-02 | 1.725e-01 | 0.44 | 3.878e-01 | 3.878e-01 | 2.283e-01 |
| 28 | converged | 1.965e-14 | 7 | 6.890e-02 | 1.475e-01 | 0.47 | 3.678e-01 | 3.678e-01 | 2.058e-01 |
| 32 | converged | 2.562e-14 | 14 | 6.427e-02 | 1.254e-01 | 0.51 | 3.515e-01 | 3.515e-01 | 1.899e-01 |

Orders over completed N = 16/20/24/28/32: e_v 0.82 0.70 0.58 0.52; e_psi 0.33 0.33 0.34 0.34; ceiling 1.00 0.47 1.02 1.22. LSQ slope log(e) vs log(h): e_v 0.68; e_psi 0.34; ceiling 0.86.

#### `i1`, gauss_ch, eps = 0.25

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | converged | 1.918e-15 | 9 | 1.852e-02 | 1.657e-02 | 1.12 | 2.052e-01 | 2.052e-01 | 1.638e-01 |
| 20 | converged | 4.042e-14 | 12 | 1.415e-02 | 1.080e-02 | 1.31 | 1.737e-01 | 1.737e-01 | 1.304e-01 |
| 24 | converged | 4.212e-15 | 6 | 1.133e-02 | 7.571e-03 | 1.50 | 1.476e-01 | 1.476e-01 | 1.064e-01 |
| 28 | converged | 5.633e-15 | 6 | 9.336e-03 | 5.597e-03 | 1.67 | 1.262e-01 | 1.262e-01 | 8.858e-02 |
| 32 | converged | 7.342e-15 | 6 | 7.842e-03 | 4.303e-03 | 1.82 | 1.086e-01 | 1.086e-01 | 7.489e-02 |

Orders over completed N = 16/20/24/28/32: e_v 1.21 1.22 1.26 1.31; e_psi 0.75 0.89 1.02 1.12; ceiling 1.92 1.95 1.96 1.97. LSQ slope log(e) vs log(h): e_v 1.24; e_psi 0.92; ceiling 1.95.

#### `i1`, gauss_ch, eps = 0.5

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | converged | 3.901e-15 | 11 | 4.119e-02 | 4.123e-02 | 1.00 | 2.617e-01 | 2.617e-01 | 1.700e-01 |
| 20 | converged | 9.564e-15 | 8 | 3.248e-02 | 2.811e-02 | 1.16 | 2.262e-01 | 2.262e-01 | 1.396e-01 |
| 24 | converged | 8.294e-15 | 8 | 2.690e-02 | 2.011e-02 | 1.34 | 1.974e-01 | 1.974e-01 | 1.178e-01 |
| 28 | converged | 1.147e-14 | 6 | 2.291e-02 | 1.516e-02 | 1.51 | 1.735e-01 | 1.735e-01 | 1.017e-01 |
| 32 | converged | 1.479e-14 | 6 | 1.987e-02 | 1.179e-02 | 1.69 | 1.536e-01 | 1.536e-01 | 8.920e-02 |

Orders over completed N = 16/20/24/28/32: e_v 1.06 1.03 1.04 1.07; e_psi 0.65 0.75 0.84 0.91; ceiling 1.72 1.84 1.83 1.88. LSQ slope log(e) vs log(h): e_v 1.05; e_psi 0.77; ceiling 1.81.

#### `i1`, gauss_ch, eps = 1

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | converged | 9.237e-15 | 6 | 9.683e-02 | 2.605e-01 | 0.37 | 4.333e-01 | 4.333e-01 | 2.099e-01 |
| 20 | converged | 1.393e-14 | 13 | 8.259e-02 | 2.151e-01 | 0.38 | 4.142e-01 | 4.142e-01 | 1.899e-01 |
| 24 | converged | 1.928e-14 | 8 | 7.381e-02 | 1.883e-01 | 0.39 | 3.953e-01 | 3.953e-01 | 1.732e-01 |
| 28 | stagnation | 1.457e+00 | 5 | 8.543e-02 | 1.631e-01 | 0.52 | 4.018e-01 | 4.018e-01 | 1.750e-01 |
| 32 | stagnation | 9.463e-01 | 7 | 7.662e-02 | 1.456e-01 | 0.53 | 3.877e-01 | 3.877e-01 | 1.834e-01 |

Orders over completed N = 16/20/24/28/32: e_v 0.71 0.62 -0.95 0.82; e_psi 0.20 0.26 -0.11 0.27; ceiling 0.86 0.73 0.93 0.85. LSQ slope log(e) vs log(h): e_v 0.27; e_psi 0.15; ceiling 0.83.

### Candidate (ii), Q1 energy + D-3 rows (`raw/sweep2/summary.md`; the 16/24 points of `gauss`/`gauss_ch` are in `raw/sweep/summary.md`)

Ceiling `oracle_mim` (mimetic face flux of the exact labels). For `control2d` the ceiling is at the reference
level (1e-7 .. 1e-9), so `e_v`/ceiling is not meaningful there.

#### `ii`, gauss, eps = 0.25

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | converged | 1.817e-15 | 42 | 7.368e-03 | 3.768e-03 | 1.96 | 1.246e-01 | 1.246e-01 | 9.779e-02 |
| 32 | converged | 1.698e-15 | 48 | 3.742e-03 | 1.503e-03 | 2.49 | 6.492e-02 | 6.492e-02 | 5.135e-02 |

Orders over completed N = 20/32: e_v 1.44; e_psi 1.39; ceiling 1.96. LSQ slope log(e) vs log(h): e_v 1.44; e_psi 1.39; ceiling 1.96.

#### `ii`, gauss, eps = 0.5

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | converged | 3.110e-14 | 41 | 1.910e-02 | 1.501e-02 | 1.27 | 1.727e-01 | 1.727e-01 | 1.252e-01 |

Orders: fewer than two completed grids (20).

#### `ii`, gauss, eps = 1

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | converged | 4.900e-16 | 42 | 5.940e-02 | 1.644e-01 | 0.36 | 3.528e-01 | 3.528e-01 | 2.126e-01 |

Orders: fewer than two completed grids (20).

#### `ii`, gauss_ch, eps = 0.25

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | converged | 4.501e-14 | 54 | 8.217e-03 | 4.021e-03 | 2.04 | 1.191e-01 | 1.191e-01 | 8.099e-02 |
| 32 | converged | 1.436e-15 | 65 | 4.110e-03 | 1.603e-03 | 2.56 | 6.131e-02 | 6.131e-02 | 4.132e-02 |

Orders over completed N = 20/32: e_v 1.47; e_psi 1.41; ceiling 1.96. LSQ slope log(e) vs log(h): e_v 1.47; e_psi 1.41; ceiling 1.96.

#### `ii`, gauss_ch, eps = 0.5

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | converged | 8.353e-16 | 71 | 2.083e-02 | 1.549e-02 | 1.34 | 1.665e-01 | 1.665e-01 | 1.023e-01 |

Orders: fewer than two completed grids (20).

#### `ii`, gauss_ch, eps = 1

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 20 | stagnation(plateau: | 1.113e-02 | 79 | 6.534e-02 | 1.598e-01 | 0.41 | 3.998e-01 | 3.998e-01 | 1.947e-01 |

Orders: fewer than two completed grids (20).

#### `ii`, control2d, eps = 0.25

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | linesearch_failed | 8.496e-13 | 19 | 1.794e-03 | 2.235e-08 | 80268.46 | 7.365e-03 | 7.365e-03 | 4.531e-10 |
| 20 | linesearch_failed | 1.494e-12 | 12 | 1.149e-03 | 6.345e-09 | 181087.47 | 4.745e-03 | 4.745e-03 | 2.852e-11 |
| 24 | converged | 8.188e-15 | 15 | 7.994e-04 | 3.118e-09 | 256382.30 | 3.313e-03 | 3.313e-03 | 7.895e-12 |

Orders over completed N = 16/20/24: e_v 2.00 1.99; e_psi 1.97 1.97; ceiling 5.64 3.90. LSQ slope log(e) vs log(h): e_v 1.99; e_psi 1.97; ceiling 4.89.

#### `ii`, control2d, eps = 0.5

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | converged | 5.906e-14 | 17 | 5.226e-03 | 9.806e-08 | 53293.90 | 9.421e-03 | 9.421e-03 | 6.788e-10 |
| 20 | converged | 3.668e-14 | 16 | 3.336e-03 | 2.611e-08 | 127767.14 | 6.037e-03 | 6.037e-03 | 4.787e-10 |
| 24 | converged | 5.715e-14 | 11 | 2.315e-03 | 8.829e-09 | 262204.10 | 4.203e-03 | 4.203e-03 | 3.486e-13 |

Orders over completed N = 16/20/24: e_v 2.01 2.00; e_psi 1.99 1.99; ceiling 5.93 5.95. LSQ slope log(e) vs log(h): e_v 2.01; e_psi 1.99; ceiling 5.94.

#### `ii`, control2d, eps = 1

| N | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 | e_psi2 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | converged | 3.707e-14 | 24 | 1.606e-02 | 5.363e-07 | 29945.93 | 1.460e-02 | 1.460e-02 | 9.475e-10 |
| 20 | linesearch_failed | 4.312e-13 | 32 | 1.026e-02 | 1.464e-07 | 70081.97 | 9.323e-03 | 9.323e-03 | 6.981e-09 |
| 24 | converged | 1.216e-14 | 17 | 7.120e-03 | 5.668e-08 | 125617.50 | 6.476e-03 | 6.476e-03 | 3.203e-09 |

Orders over completed N = 16/20/24: e_v 2.01 2.00; e_psi 2.01 2.00; ceiling 5.82 5.20. LSQ slope log(e) vs log(h): e_v 2.01; e_psi 2.01; ceiling 5.55.

Cost of (ii) at 32^3: 13 146 s (`gauss:0.25`) and 17 967 s (`gauss_ch:0.25`) per solve (`raw/sweep2/summary.md`).
Whitney control (N3 form, `raw/cand_ii_q1_ctl_whitney_spectrum12.txt`, artifact README "Findings of C-ii"):
exactly `2 N^2 = 288` null Hessian modes at `k = 1`, 12^3; the Q1 form has none (smallest relative eigenvalue
1.32e-3).

### `cons4`: 4th-order residual `r_F` of `i1o4` at the exact oracle labels, N = 16 / 24 / 28 (`raw/sweep2/summary.md`)

| field eps | r_F at 16 / 24 / 28 | orders |
|---|---|---|
| gauss 0.25 | 4.203e-02 1.167e-02 6.872e-03 | orders 3.16 3.44 |
| gauss 0.5 | 3.334e-01 1.376e-01 8.563e-02 | orders 2.18 3.08 |
| gauss 1 | 1.032e+01 8.169e+00 1.366e+01 | orders 0.58 -3.33 |
| gauss_ch 0.25 | 4.425e-02 1.241e-02 7.350e-03 | orders 3.13 3.40 |
| gauss_ch 0.5 | 3.908e-01 1.608e-01 1.103e-01 | orders 2.19 2.45 |
| gauss_ch 1 | 1.421e+01 1.742e+01 2.017e+01 | orders -0.50 -0.95 |
| control2d 0.25 | 6.983e-03 1.358e-03 7.092e-04 | orders 4.04 4.21 |
| control2d 0.5 | 2.208e-02 4.474e-03 2.309e-03 | orders 3.94 4.29 |
| control2d 1 | 7.856e-02 1.395e-02 6.868e-03 | orders 4.26 4.59 |
| generic3d 0.25 | 8.533e-03 1.743e-03 9.424e-04 | orders 3.92 3.99 |
| generic3d 0.5 | 5.759e-02 1.840e-02 1.142e-02 | orders 2.81 3.09 |
| generic3d 1 | 2.042e+00 9.559e-01 8.530e-01 | orders 1.87 0.74 |

The residual of the exact labels falls at orders 3.13-4.21 at `eps = 0.25` (all four fields), 2.18-3.09 for
`gauss`, `gauss_ch`, `generic3d` at `eps = 0.5`, and 3.94-4.59 for `control2d` at `eps` 0.5 and 1. At `eps = 1`
it grows with `N` for `gauss` (orders 0.58, -3.33) and `gauss_ch` (-0.50, -0.95) and falls slowly for
`generic3d` (1.87, 0.74). The growth is carried by the interior rows (e.g. `gauss_ch:1` centered rows 12.85 /
15.26 / 17.89, plane-(N-1) rows 29.65 / 45.71 / 54.70), while the plane-1 rows fall at order ~4.2.

### `spec4`: dense spectra of `i1o4` (sorted relative singular values; `raw/sweep2/summary.md`, files in `raw/sweep2/spectra/`)

| field | eps | M | state | n | smallest 3 | < 1e-3 | < 1e-6 | < 1e-10 | largest gap below 1e-2 | first gap >= 100 below 1e-2 |
|---|---|---|---|---|---|---|---|---|---|---|
| gauss | 0.25 | 12 | converged | 3456 | 4.36e-04 4.42e-04 4.46e-04 | 32 | 0 | 0 | 1.66 @ 48: 2.37e-03 -> 3.93e-03 | none |
| gauss | 0.25 | 16 | converged | 8192 | 2.54e-04 2.58e-04 2.64e-04 | 55 | 0 | 0 | 1.32 @ 55: 9.35e-04 -> 1.24e-03 | none |
| gauss | 0.5 | 12 | converged | 3456 | 1.63e-04 1.76e-04 1.86e-04 | 36 | 0 | 0 | 1.17 @ 31: 6.96e-04 -> 8.14e-04 | none |
| gauss | 0.5 | 16 | converged | 8192 | 1.01e-04 1.07e-04 1.13e-04 | 70 | 0 | 0 | 1.09 @ 27: 2.42e-04 -> 2.64e-04 | none |
| gauss | 1 | 12 | final_iterate | 3456 | 2.88e-06 6.33e-06 1.20e-05 | 133 | 0 | 0 | 2.2 @ 1: 2.88e-06 -> 6.33e-06 | none |
| gauss | 1 | 12 | oracle | 3456 | 3.04e-08 3.09e-07 5.23e-07 | 1252 | 5 | 0 | 10.2 @ 1: 3.04e-08 -> 3.09e-07 | none |
| gauss | 1 | 16 | final_iterate | 8192 | 4.68e-07 1.01e-06 4.10e-06 | 446 | 1 | 0 | 4.07 @ 2: 1.01e-06 -> 4.10e-06 | none |
| gauss | 1 | 16 | oracle | 8192 | 9.13e-08 1.13e-07 1.58e-07 | 2716 | 8 | 0 | 1.79 @ 3: 1.58e-07 -> 2.83e-07 | none |
| gauss_ch | 0.25 | 12 | converged | 3456 | 4.30e-04 4.45e-04 4.69e-04 | 32 | 0 | 0 | 1.62 @ 48: 2.51e-03 -> 4.06e-03 | none |
| gauss_ch | 0.25 | 16 | converged | 8192 | 2.32e-04 2.46e-04 2.52e-04 | 55 | 0 | 0 | 1.4 @ 55: 8.04e-04 -> 1.12e-03 | none |
| gauss_ch | 0.5 | 12 | converged | 3456 | 1.44e-04 1.79e-04 1.92e-04 | 36 | 0 | 0 | 1.24 @ 1: 1.44e-04 -> 1.79e-04 | none |
| gauss_ch | 0.5 | 16 | converged | 8192 | 8.07e-05 8.97e-05 9.31e-05 | 81 | 0 | 0 | 1.13 @ 28: 2.09e-04 -> 2.37e-04 | none |
| gauss_ch | 1 | 12 | final_iterate | 3456 | 3.57e-06 3.82e-06 6.55e-06 | 209 | 0 | 0 | 1.71 @ 2: 3.82e-06 -> 6.55e-06 | none |
| gauss_ch | 1 | 12 | oracle | 3456 | 1.54e-08 1.17e-07 3.87e-07 | 1617 | 6 | 0 | 7.61 @ 1: 1.54e-08 -> 1.17e-07 | none |
| gauss_ch | 1 | 16 | final_iterate | 8192 | 1.20e-07 7.20e-07 1.27e-06 | 711 | 2 | 0 | 5.98 @ 1: 1.20e-07 -> 7.20e-07 | none |
| gauss_ch | 1 | 16 | oracle | 8192 | 6.13e-09 2.02e-08 5.06e-08 | 5134 | 23 | 0 | 3.3 @ 1: 6.13e-09 -> 2.02e-08 | none |
| control2d | 0.25 | 12 | converged | 3456 | 4.93e-04 5.28e-04 5.28e-04 | 31 | 0 | 0 | 1.48 @ 47: 2.67e-03 -> 3.95e-03 | none |
| control2d | 0.25 | 16 | converged | 8192 | 3.09e-04 3.24e-04 3.26e-04 | 51 | 0 | 0 | 1.36 @ 51: 9.70e-04 -> 1.32e-03 | none |
| control2d | 0.5 | 12 | converged | 3456 | 2.15e-04 2.31e-04 2.48e-04 | 30 | 0 | 0 | 1.44 @ 20: 3.48e-04 -> 5.01e-04 | none |
| control2d | 0.5 | 16 | converged | 8192 | 1.60e-04 1.62e-04 1.62e-04 | 53 | 0 | 0 | 1.27 @ 33: 2.66e-04 -> 3.39e-04 | none |
| control2d | 1 | 12 | converged | 3456 | 1.07e-05 1.07e-05 5.19e-05 | 63 | 0 | 0 | 4.87 @ 2: 1.07e-05 -> 5.19e-05 | none |
| control2d | 1 | 16 | converged | 8192 | 2.15e-05 2.15e-05 2.22e-05 | 134 | 0 | 0 | 1.39 @ 25: 4.62e-05 -> 6.40e-05 | none |
| generic3d | 0.25 | 12 | converged | 3456 | 4.52e-04 4.68e-04 4.81e-04 | 21 | 0 | 0 | 1.57 @ 48: 2.82e-03 -> 4.44e-03 | none |
| generic3d | 0.25 | 16 | converged | 8192 | 2.68e-04 2.69e-04 2.77e-04 | 46 | 0 | 0 | 1.28 @ 26: 4.19e-04 -> 5.36e-04 | none |
| generic3d | 0.5 | 12 | converged | 3456 | 9.89e-05 1.32e-04 1.42e-04 | 32 | 0 | 0 | 1.34 @ 1: 9.89e-05 -> 1.32e-04 | none |
| generic3d | 0.5 | 16 | converged | 8192 | 7.56e-05 8.07e-05 8.93e-05 | 72 | 0 | 0 | 1.16 @ 18: 1.48e-04 -> 1.72e-04 | none |
| generic3d | 1 | 12 | final_iterate | 3456 | 5.73e-08 3.92e-06 6.10e-06 | 180 | 1 | 0 | 68.4 @ 1: 5.73e-08 -> 3.92e-06 | none |
| generic3d | 1 | 12 | oracle | 3456 | 7.10e-08 2.01e-07 6.78e-07 | 1293 | 6 | 0 | 3.38 @ 2: 2.01e-07 -> 6.78e-07 | none |
| generic3d | 1 | 16 | converged | 8192 | 1.27e-06 1.35e-06 2.14e-06 | 960 | 0 | 0 | 1.59 @ 2: 1.35e-06 -> 2.14e-06 | none |

No spectrum at a converged state has a gap-separated group (largest consecutive ratio below 1e-2 at a converged
state: 4.87, `control2d:1` at 12^3; no value below 1e-6). At `eps = 1` the spectra of `gauss`, `gauss_ch` are at
the final (unconverged) iterate or at the oracle labels: at the exact labels the Jacobian has 5-23 relative
singular values below 1e-6 (smallest 6.13e-9 .. 9.13e-8), a continuum without a gap >= 100.

### Cost of direct solves (`raw/sweep2/timeouts.md`, median over cells, max in parentheses)

| kind | N | elapsed [s] | peak RSS [GB] |
|---|---|---|---|
| `i1` | 16 | 52 (867) | 0.48 |
| `i1` | 24 | 726 (3 868) | 2.99 |
| `i1` | 28 | 1 451 (6 668) | 6.59 |
| `i1` | 32 | 5 139 (16 764) | 13.25 |
| `i1o4` | 16 | 225 (1 225) | 0.91 |
| `i1o4` | 24 | 2 199 (9 847) | 5.86 |
| `i1o4` | 28 | 9 382 (21 603; 3 timeouts) | 16.47 |
| `i1o4` | 32 | 16 663 (17 307) | 25.96 |
| `ii` | 32 | 15 599 (18 010) | 4.76 |

### D-5 classification of the corrective sweep (`raw/sweep2/summary.md`, verbatim, 45 rows)

Reading rules as printed at the top of that file: three finest completed grids; (1) PASS iff order >= 1.8 on
both pairs, else `ceiling-limited` / FAIL per D-5; (2) PASS iff order >= 1.8 on both pairs; (3) final
`r_F <= 1e-10` and status `converged` on every completed grid; (4) no consecutive ratio >= 100 below 1e-2 in the
converged-state `spec4` spectra (`i1o4` and `i1o4_fd2` only). `i1o4_fd2` = the `i1o4` solutions measured with
the 2nd-order reconstruction against the 2nd-order ceiling.

| field | eps | candidate | (1) e_v | (2) e_psi | (3) r_F | (4) spectrum |
|---|---|---|---|---|---|---|
| gauss | 0.25 | i1o4 | **PASS** (N=24/28/32 orders 2.80 2.87) | **PASS** (N=24/28/32 orders 2.70 2.83) | **PASS** (r_F 8.6e-14 4.3e-15 6.4e-15 8.5e-15 1.1e-14; status converged converged converged converged converged) | **PASS** (M=12,16) |
| gauss | 0.25 | i1o4_fd2 | **PASS** (N=24/28/32 orders 1.96 1.96) | **PASS** (N=24/28/32 orders 2.70 2.83) | **PASS** (r_F 8.6e-14 4.3e-15 6.4e-15 8.5e-15 1.1e-14; status converged converged converged converged converged) | **PASS** (M=12,16) |
| gauss | 0.25 | i1 | **FAIL** (N=24/28/32 orders 1.27 1.31 vs ceiling 1.96 1.97, max ratio 1.65) | **FAIL** (N=24/28/32 orders 1.01 1.11) | **PASS** (r_F 1.5e-15 5.9e-14 3.3e-15 4.5e-15 5.8e-15; status converged converged converged converged converged) | **n/a** (no spectrum cell in this matrix) |
| gauss | 0.25 | ii | **INCOMPLETE** (2 completed grid(s)) | **INCOMPLETE** (2 completed grid(s)) | **PASS** (r_F 1.8e-15 1.7e-15; status converged converged) | **n/a** (no spectrum cell in this matrix) |
| gauss | 0.5 | i1o4 | **PASS** (N=20/24/28 orders 1.87 1.84) | **FAIL** (N=20/24/28 orders 1.73 1.82) | **PASS** (r_F 5.4e-15 1.0e-14 1.2e-14 1.7e-14; status converged converged converged converged) | **PASS** (M=12,16) |
| gauss | 0.5 | i1o4_fd2 | **ceiling-limited** (N=20/24/28 orders 1.79 1.78 vs ceiling 1.78 1.86, max ratio 0.90) | **FAIL** (N=20/24/28 orders 1.73 1.82) | **PASS** (r_F 5.4e-15 1.0e-14 1.2e-14 1.7e-14; status converged converged converged converged) | **PASS** (M=12,16) |
| gauss | 0.5 | i1 | **FAIL** (N=24/28/32 orders 1.06 1.06 vs ceiling 1.86 1.87, max ratio 1.62) | **FAIL** (N=24/28/32 orders 0.84 0.90) | **PASS** (r_F 3.0e-15 4.6e-15 5.2e-14 8.9e-15 4.6e-14; status converged converged converged converged converged) | **n/a** (no spectrum cell in this matrix) |
| gauss | 0.5 | ii | **INCOMPLETE** (1 completed grid(s)) | **INCOMPLETE** (1 completed grid(s)) | **PASS** (r_F 3.1e-14; status converged) | **n/a** (no spectrum cell in this matrix) |
| gauss | 1 | i1o4 | **FAIL** (N=16/20/24 orders 0.35 0.36 vs ceiling 1.45 0.62, max ratio 0.42) | **FAIL** (N=16/20/24 orders 0.22 0.35) | **FAIL** (r_F 5.8e-01 7.6e-01 8.4e-01; status stagnation stagnation stagnation) | **INCOMPLETE** (no converged-state spec4 spectrum) |
| gauss | 1 | i1o4_fd2 | **FAIL** (N=16/20/24 orders 0.66 0.70 vs ceiling 1.00 0.47, max ratio 0.49) | **FAIL** (N=16/20/24 orders 0.22 0.35) | **FAIL** (r_F 5.8e-01 7.6e-01 8.4e-01; status stagnation stagnation stagnation) | **INCOMPLETE** (no converged-state spec4 spectrum) |
| gauss | 1 | i1 | **FAIL** (N=24/28/32 orders 0.58 0.52 vs ceiling 1.02 1.22, max ratio 0.51) | **FAIL** (N=24/28/32 orders 0.34 0.34) | **PASS** (r_F 7.5e-15 9.5e-15 1.5e-14 2.0e-14 2.6e-14; status converged converged converged converged converged) | **n/a** (no spectrum cell in this matrix) |
| gauss | 1 | ii | **INCOMPLETE** (1 completed grid(s)) | **INCOMPLETE** (1 completed grid(s)) | **PASS** (r_F 4.9e-16; status converged) | **n/a** (no spectrum cell in this matrix) |
| gauss_ch | 0.25 | i1o4 | **PASS** (N=24/28/32 orders 2.90 2.99) | **PASS** (N=24/28/32 orders 2.82 2.94) | **PASS** (r_F 3.4e-15 5.5e-15 8.9e-15 1.1e-14 1.4e-14; status converged converged converged converged converged) | **PASS** (M=12,16) |
| gauss_ch | 0.25 | i1o4_fd2 | **PASS** (N=24/28/32 orders 1.95 1.96) | **PASS** (N=24/28/32 orders 2.82 2.94) | **PASS** (r_F 3.4e-15 5.5e-15 8.9e-15 1.1e-14 1.4e-14; status converged converged converged converged converged) | **PASS** (M=12,16) |
| gauss_ch | 0.25 | i1 | **FAIL** (N=24/28/32 orders 1.26 1.31 vs ceiling 1.96 1.97, max ratio 1.82) | **FAIL** (N=24/28/32 orders 1.02 1.12) | **PASS** (r_F 1.9e-15 4.0e-14 4.2e-15 5.6e-15 7.3e-15; status converged converged converged converged converged) | **n/a** (no spectrum cell in this matrix) |
| gauss_ch | 0.25 | ii | **INCOMPLETE** (2 completed grid(s)) | **INCOMPLETE** (2 completed grid(s)) | **PASS** (r_F 4.5e-14 1.4e-15; status converged converged) | **n/a** (no spectrum cell in this matrix) |
| gauss_ch | 0.5 | i1o4 | **PASS** (N=20/24/28 orders 1.96 2.05) | **PASS** (N=20/24/28 orders 1.81 1.98) | **PASS** (r_F 7.0e-15 1.1e-14 1.6e-14 2.2e-14; status converged converged converged converged) | **PASS** (M=12,16) |
| gauss_ch | 0.5 | i1o4_fd2 | **ceiling-limited** (N=20/24/28 orders 1.76 1.79 vs ceiling 1.84 1.83, max ratio 0.91) | **PASS** (N=20/24/28 orders 1.81 1.98) | **PASS** (r_F 7.0e-15 1.1e-14 1.6e-14 2.2e-14; status converged converged converged converged) | **PASS** (M=12,16) |
| gauss_ch | 0.5 | i1 | **FAIL** (N=24/28/32 orders 1.04 1.07 vs ceiling 1.83 1.88, max ratio 1.69) | **FAIL** (N=24/28/32 orders 0.84 0.91) | **PASS** (r_F 3.9e-15 9.6e-15 8.3e-15 1.1e-14 1.5e-14; status converged converged converged converged converged) | **n/a** (no spectrum cell in this matrix) |
| gauss_ch | 0.5 | ii | **INCOMPLETE** (1 completed grid(s)) | **INCOMPLETE** (1 completed grid(s)) | **PASS** (r_F 8.4e-16; status converged) | **n/a** (no spectrum cell in this matrix) |
| gauss_ch | 1 | i1o4 | **FAIL** (N=16/20/24 orders -0.37 -1.39 vs ceiling 1.24 0.76, max ratio 0.64) | **FAIL** (N=16/20/24 orders 0.01 -0.34) | **FAIL** (r_F 1.4e+00 2.8e+00 2.3e+00; status stagnation linesearch-fail stagnation) | **INCOMPLETE** (no converged-state spec4 spectrum) |
| gauss_ch | 1 | i1o4_fd2 | **FAIL** (N=16/20/24 orders 0.29 -0.91 vs ceiling 0.86 0.73, max ratio 0.65) | **FAIL** (N=16/20/24 orders 0.01 -0.34) | **FAIL** (r_F 1.4e+00 2.8e+00 2.3e+00; status stagnation linesearch-fail stagnation) | **INCOMPLETE** (no converged-state spec4 spectrum) |
| gauss_ch | 1 | i1 | **FAIL** (N=24/28/32 orders -0.95 0.82 vs ceiling 0.93 0.85, max ratio 0.53) | **FAIL** (N=24/28/32 orders -0.11 0.27) | **FAIL** (r_F 9.2e-15 1.4e-14 1.9e-14 1.5e+00 9.5e-01; status converged converged converged stagnation stagnation) | **n/a** (no spectrum cell in this matrix) |
| gauss_ch | 1 | ii | **INCOMPLETE** (1 completed grid(s)) | **INCOMPLETE** (1 completed grid(s)) | **FAIL** (r_F 1.1e-02; status stagnation(plateau:) | **n/a** (no spectrum cell in this matrix) |
| control2d | 0.25 | i1o4 | **PASS** (N=20/24/28 orders 4.20 4.14) | **PASS** (N=20/24/28 orders 4.06 2.26) | **PASS** (r_F 1.5e-15 2.4e-15 3.4e-15 4.7e-15; status converged converged converged converged) | **PASS** (M=12,16) |
| control2d | 0.25 | i1o4_fd2 | **PASS** (N=20/24/28 orders 2.06 2.04) | **PASS** (N=20/24/28 orders 4.06 2.26) | **PASS** (r_F 1.5e-15 2.4e-15 3.4e-15 4.7e-15; status converged converged converged converged) | **PASS** (M=12,16) |
| control2d | 0.25 | i1 | **PASS** (N=24/28/32 orders 2.08 2.07) | **FAIL** (N=24/28/32 orders 1.63 1.65) | **PASS** (r_F 8.7e-16 1.4e-15 2.0e-15 2.6e-15 3.4e-15; status converged converged converged converged converged) | **n/a** (no spectrum cell in this matrix) |
| control2d | 0.25 | ii | **PASS** (N=16/20/24 orders 2.00 1.99) | **PASS** (N=16/20/24 orders 1.97 1.97) | **FAIL** (r_F 8.5e-13 1.5e-12 8.2e-15; status linesearch_failed linesearch_failed converged) | **n/a** (no spectrum cell in this matrix) |
| control2d | 0.5 | i1o4 | **PASS** (N=20/24/28 orders 3.75 3.99) | **FAIL** (N=20/24/28 orders 1.58 3.28) | **PASS** (r_F 3.0e-15 4.5e-15 6.6e-15 9.0e-15; status converged converged converged converged) | **PASS** (M=12,16) |
| control2d | 0.5 | i1o4_fd2 | **PASS** (N=20/24/28 orders 2.07 2.07) | **FAIL** (N=20/24/28 orders 1.58 3.28) | **PASS** (r_F 3.0e-15 4.5e-15 6.6e-15 9.0e-15; status converged converged converged converged) | **PASS** (M=12,16) |
| control2d | 0.5 | i1 | **PASS** (N=24/28/32 orders 2.00 2.00) | **FAIL** (N=24/28/32 orders 1.41 1.53) | **PASS** (r_F 1.6e-15 2.6e-15 3.7e-15 5.0e-15 6.4e-15; status converged converged converged converged converged) | **n/a** (no spectrum cell in this matrix) |
| control2d | 0.5 | ii | **PASS** (N=16/20/24 orders 2.01 2.00) | **PASS** (N=16/20/24 orders 1.99 1.99) | **PASS** (r_F 5.9e-14 3.7e-14 5.7e-14; status converged converged converged) | **n/a** (no spectrum cell in this matrix) |
| control2d | 1 | i1o4 | **PASS** (N=20/24/28 orders 4.35 5.01) | **PASS** (N=20/24/28 orders 4.33 5.11) | **PASS** (r_F 5.4e-15 8.4e-15 1.1e-14 1.6e-14; status converged converged converged converged) | **PASS** (M=12,16) |
| control2d | 1 | i1o4_fd2 | **PASS** (N=20/24/28 orders 3.03 2.81) | **PASS** (N=20/24/28 orders 4.33 5.11) | **PASS** (r_F 5.4e-15 8.4e-15 1.1e-14 1.6e-14; status converged converged converged converged) | **PASS** (M=12,16) |
| control2d | 1 | i1 | **PASS** (N=24/28/32 orders 1.96 1.96) | **PASS** (N=24/28/32 orders 1.91 1.91) | **PASS** (r_F 3.2e-15 5.1e-15 6.8e-15 8.8e-15 1.1e-14; status converged converged converged converged converged) | **n/a** (no spectrum cell in this matrix) |
| control2d | 1 | ii | **PASS** (N=16/20/24 orders 2.01 2.00) | **PASS** (N=16/20/24 orders 2.01 2.00) | **FAIL** (r_F 3.7e-14 4.3e-13 1.2e-14; status converged linesearch_failed converged) | **n/a** (no spectrum cell in this matrix) |
| generic3d | 0.25 | i1o4 | **PASS** (N=20/24/28 orders 3.56 3.51) | **PASS** (N=20/24/28 orders 3.27 3.29) | **PASS** (r_F 2.7e-15 4.2e-15 6.0e-15 8.1e-15; status converged converged converged converged) | **PASS** (M=12,16) |
| generic3d | 0.25 | i1o4_fd2 | **PASS** (N=20/24/28 orders 2.08 2.05) | **PASS** (N=20/24/28 orders 3.27 3.29) | **PASS** (r_F 2.7e-15 4.2e-15 6.0e-15 8.1e-15; status converged converged converged converged) | **PASS** (M=12,16) |
| generic3d | 0.25 | i1 | **FAIL** (N=20/24/28 orders 1.60 1.59 vs ceiling 2.04 2.04, max ratio 1.64) | **FAIL** (N=20/24/28 orders 1.29 1.31) | **PASS** (r_F 1.5e-15 2.3e-15 3.6e-15 2.1e-14; status converged converged converged converged) | **n/a** (no spectrum cell in this matrix) |
| generic3d | 0.5 | i1o4 | **PASS** (N=20/24/28 orders 2.50 2.35) | **PASS** (N=20/24/28 orders 2.33 2.22) | **PASS** (r_F 5.5e-15 2.3e-14 1.2e-14 1.7e-14; status converged converged converged converged) | **PASS** (M=12,16) |
| generic3d | 0.5 | i1o4_fd2 | **PASS** (N=20/24/28 orders 2.04 2.01) | **PASS** (N=20/24/28 orders 2.33 2.22) | **PASS** (r_F 5.5e-15 2.3e-14 1.2e-14 1.7e-14; status converged converged converged converged) | **PASS** (M=12,16) |
| generic3d | 0.5 | i1 | **FAIL** (N=20/24/28 orders 1.25 1.19 vs ceiling 1.98 1.98, max ratio 2.03) | **FAIL** (N=20/24/28 orders 1.15 1.13) | **PASS** (r_F 3.1e-15 4.6e-15 3.8e-14 9.0e-15; status converged converged converged converged) | **n/a** (no spectrum cell in this matrix) |
| generic3d | 1 | i1o4 | **FAIL** (N=16/20/24 orders 2.53 1.59 vs ceiling 0.86 0.95, max ratio 0.63) | **FAIL** (N=16/20/24 orders 1.76 1.50) | **PASS** (r_F 1.6e-14 2.0e-14 2.8e-14; status converged converged converged) | **PASS** (M=16) |
| generic3d | 1 | i1o4_fd2 | **FAIL** (N=16/20/24 orders 2.04 1.47 vs ceiling 0.75 0.82, max ratio 0.70) | **FAIL** (N=16/20/24 orders 1.76 1.50) | **PASS** (r_F 1.6e-14 2.0e-14 2.8e-14; status converged converged converged) | **PASS** (M=16) |
| generic3d | 1 | i1 | **FAIL** (N=20/24/28 orders -2.60 4.56 vs ceiling 0.82 0.88, max ratio 1.36) | **FAIL** (N=20/24/28 orders -2.18 3.78) | **FAIL** (r_F 7.0e-15 1.0e-14 3.5e-01 2.0e-14; status converged converged stagnation converged) | **n/a** (no spectrum cell in this matrix) |

## Result

Each item names its source. Classification per `docs/AGENTS.md`: all items below are "confirmed in runs" for
the cases listed (CPU prototype, one Gaussian realization, `L/ell = 4`, grids <= 32^3 for the corrective sweep),
except where marked.

**R1. The spec-literal outlet (`i0`, no outlet condition) has no solution.** At `k = 1` the discrete Jacobian has
exactly `2N` null modes (24 at 12^3, 32 at 16^3; `raw/cand_i_smoke_k1check.txt`). At finite amplitude
`grad ln k` lifts them to ~1e-5 (no roundoff cluster: e.g. `gauss:0.25` 12^3 final iterate smallest 1.40e-5,
`raw/sweep/summary.md`), but Newton does not converge: criterion (3) FAIL in 17 of 21 (field, eps) cells at
16^3, plateaus at `r_F` 2.8e-4 .. 0.89 after bisection and Levenberg-Marquardt (`raw/sweep/summary.md`, D-5
counts). It converges only on `control2d` and `lester2021:0.25`. P1 is confirmed in substance (refined count of
the §8 instrument: `2N` exact nulls, not `2(N_perp^2 - 1)`).

**R2. With the outlet condition D-2 the problem is well-posed and consistent.** At every converged state of the
corrective sweep there is no gap-separated group in the dense spectrum (`spec4` table: `eps` 0.25 and 0.5 for all
four fields, `eps = 1` for `control2d` and `generic3d`); the smallest relative singular value scales like `h^2`
(`gauss:0.25`: 4.36e-4 at 12^3 -> 2.54e-4 at 16^3, ratio 1.72 vs `(16/12)^2 = 1.78`) and no value lies below 1e-6.
Newton reaches `r_F` 1e-15 .. 5e-14 with no plateau (all `converged` rows). The 4th-order residual at the exact
labels falls at orders 3.13-4.21 at `eps = 0.25` (`cons4` table), i.e. the discrete equations are consistent
with the Darcy labels and the order approaches 4.

**R3. Candidate (i-1) with 4th-order stencils reproduces the Darcy labels with convergent `e_v` and `e_psi` and
no floor at `eps = 0.25` (`sigma_Y = 0.25`) for all four fields, including the constant-head case.** Observed
orders on the completed pairs (`raw/sweep2/summary.md`, `i1o4` tables): `e_v` 2.75-2.87 (`gauss`, 16..32),
2.62-2.99 (`gauss_ch`, 16..32), 3.51-3.68 (`generic3d`, 16..28), 4.14-4.48 (`control2d`, 16..28); `e_psi`
2.34-2.83, 2.36-2.94, 3.27-3.39 and 7.85/4.06/2.26 respectively; `r_F` <= 8.6e-14. D-5 table: criteria (1)-(4)
PASS for `gauss`, `gauss_ch`, `generic3d`, `control2d` at `eps = 0.25`, hence criterion (5) PASS at 0.25.
At `eps = 0.5`: `gauss_ch` PASS (1)-(4) (orders `e_v` 1.96, 2.05; `e_psi` 1.81, 1.98 on 20/24/28), so criterion
(5) PASS at 0.5; `generic3d` PASS (1)-(4); `gauss` (1) PASS (1.87, 1.84), (2) FAIL (1.73, 1.82), (3), (4) PASS;
`control2d` (1), (3), (4) PASS, (2) FAIL (non-monotone `e_psi` orders 3.62, 1.58, 3.28; `e_psi` 3.18e-2 ->
6.43e-3, LSQ slope 2.75). At `eps = 1`: `control2d` PASS (1)-(4); `generic3d` converges (`r_F` <= 2.8e-14, (3)
and (4) PASS) but (1) FAIL (2.53, 1.59) and (2) FAIL (1.76, 1.50) on 16/20/24; `gauss`, `gauss_ch` FAIL (1)-(3)
(R5). The ratio `e_v`/ceiling grows with `N` in the PASS cells (`gauss:0.25` 1.67 -> 3.06) because the
4th-order reconstruction of the exact labels converges at 3.5-3.8 while the solution error converges at ~2.8;
the operational D-5 rule of the corrective sweep applies the 1.5 ratio only to cells that miss 1.8, so these
cells are PASS as classified; the reviewer adjudicates this reading (D-5).

**R4. The same formulation with 2nd-order stencils (`i1`) converges with direct solves but is pre-asymptotic on
16-32.** Direct `splu` solves converge to `r_F` <= 5.2e-14 at `eps <= 0.5` for every field and grid of the
corrective sweep (`i1` tables), so the first sweep's timeouts and the `gauss:0.5:32` stagnation were linear-solver
(GMRES + `k = 1` preconditioner) failures (`raw/sweep/timeouts.md`). Orders on 16..32: `e_v` 1.26-1.31 (`gauss`
0.25), 1.21-1.31 (`gauss_ch` 0.25), 1.06-1.12 (`gauss` 0.5), 1.03-1.07 (`gauss_ch` 0.5); `e_psi` 0.75 -> 1.11
and 0.75 -> 1.12 at 0.25, rising. `e_v`/ceiling 1.12 -> 1.82 (`gauss_ch:0.25`). The label error is larger than the 4th-order one at equal `N`:
at 32^3, `eps = 0.25`, `e_psi` 0.1149 vs 0.01303 (`gauss`, 8.8x) and 0.1086 vs 0.01086 (`gauss_ch`, 10x); at
28^3, `eps = 0.5`, 0.1847 vs 0.05159 (3.6x) and 0.1735 vs 0.04586 (3.8x).
The 2-D control reaches order 1.79 in `e_psi` only on 48 -> 64 (`raw/sweep/summary.md`, `control2d:0.5`) with
`e_v` second order on every pair. D-5 table: `i1` FAIL (1)/(2) for the 3D fields at 0.25 and 0.5.

**R5. `eps = 1` (`sigma_Y = 1`) is inconclusive by resolution, not refuted.** (a) The 4th-order residual of the
EXACT labels grows with `N` on 16/24/28 (`gauss` 10.32 / 8.169 / 13.66; `gauss_ch` 14.21 / 17.42 / 20.17;
`cons4`). (b) The FD reconstruction of the exact labels converges at order ~1 on these grids (2nd-order ceiling
orders 1.00, 0.47, 1.02, 1.22 for `gauss` on 16..32, `raw/sweep2/summary.md`; N1: 0.91, 1.25 on 16/32/48) and
the mid-slab probe reaches 1.76 only on 96 -> 128 (`raw/oracle_midplane.txt`). (c) The Jacobian at the exact
labels is near-singular (smallest relative singular values 3.04e-8 / 9.13e-8 for `gauss`, 1.54e-8 / 6.13e-9 for
`gauss_ch` at 12^3 / 16^3; `spec4`). (d) Newton stagnates for `i1o4` (`r_F` 0.58-0.84 `gauss`, 1.4-2.8
`gauss_ch`) after repeated bisection (last accepted amplitudes 0.625-0.8125, PATH lines); the 28^3 `i1o4` cells timed out at the 21 600 s cap. The
2nd-order `i1` converges in residual on `gauss` up to 32^3 (`r_F` <= 2.6e-14) with `e_psi` 0.443 -> 0.352
(orders 0.33-0.34): a converged discrete solution far from the under-resolved continuum labels. `ell/h` is 4-7 on
16-28 at `ell = 1/4`; the oracle probes indicate `ell/h >= 24-32` (N1 mid-slab) for the exact labels to be in
the asymptotic regime of a 2nd-order reconstruction. Estimate, not a measurement.

**R6. Candidate (ii) does not qualify.** In the specified Whitney/edge-averaged form the energy has an exact
`2 N^2` hourglass kernel (288 at 12^3) and the D-3 constraint leaves the mean transverse flux free (N3; artifact
README "N3 record"). In the corrected Q1 form (C-ii) both defects are removed (0 null modes, D-3 completed), but
on the 3D Gaussian fields it converges at ~1.4 (`gauss:0.25` `e_v` 1.44, `e_psi` 1.39; `gauss_ch:0.25` 1.47,
1.41 on 20 -> 32, `e_v`/ceiling 1.96 -> 2.56), its label error at 32^3 is 5x that of `i1o4` (`gauss:0.25`
`e_psi` 0.06492 vs 0.01303), and one 32^3 solve takes 13 146-17 967 s. It is second order on `control2d`
(PASS (1), (2)) but with line-search failures at 16/20 ((3) FAIL at 0.25 and 1). Not competitive.

**R7. Solver facts for the next phase.** (a) Newton with exact linear solves converges in 6-14 iterations in
the final stage at `eps <= 0.5` on the 3D fields (2 on `control2d`). (b) The per-mode `k = 1` Fourier
preconditioner is not sufficient beyond `eps = 0.25` at `N >= 32`: median 900 GMRES iterations per Newton step,
caps and restart stagnation, `gauss:0.5:32` stagnated at `r_F = 8.614e-3` (`raw/sweep/timeouts.md`).
(c) The zero start is outside the Newton basin at `eps = 0.25` for `N >= 24` (`gauss` `i1o4`; `gauss_ch` `i1o4`
already at 16), and the bisection `0.25(fail)->0.125->0.25` converges (PATH lines); `eps = 0.5` needs a 0.375
stage on `gauss`/`gauss_ch`. Amplitude continuation with bisection is required. (d) Direct LU cost and memory
grow steeply (cost table: `i1o4` 32^3 median 16 663 s, 25.96 GB peak RSS), so the direct route does not extend
beyond 32^3 on CPU.

**R8. Predictions versus outcomes.**

| prediction (fixed before any candidate run) | outcome | source |
|---|---|---|
| P1: spec-literal outlet has a null cluster, fails (4), Newton contaminated | Confirmed in substance: `2N` exact nulls at `k = 1` (§8 refined count); at finite amplitude no roundoff cluster but no convergence ((3) FAIL 17/21) | R1 |
| P2: constant-head case well-posed with outlet Neumann; solution = Darcy labels | Confirmed at `eps` 0.25 and 0.5 (`gauss_ch` `i1o4` PASS (1)-(4)); at `eps = 1` not shown (R5) | R3, R5 |
| P3: periodic case closed by the outlet condition D-2; (i-1) satisfies (1)-(5), (i-0) fails (4) | (i-1): confirmed at 0.25 (all fields), at 0.5 for `gauss_ch`, `generic3d` (`gauss`, `control2d` miss (2) on one pair), not at 1 (resolution). (i-0): fails, through (3) rather than through a roundoff cluster at finite amplitude | R1, R3, R5 |
| P4: inlet oblique condition satisfied by the Darcy labels, not imposed, reported as diagnostic | Not imposed; reported per case as `EXTRA` in the cell logs (not analysed further in this note) | cell logs |
| P5: (ii-free) satisfies (1)-(5) in the constant-head case | Refuted in the specified Whitney form (hourglass kernel); in the Q1 form order ~1.4 on 20 -> 32 for `gauss_ch` | R6 |
| P6: (ii-constrained) satisfies (1)-(5) in the periodic case | Refuted in the specified form (kernel + D-3 floor); Q1 + completed D-3: order ~1.4 on `gauss`, second order on `control2d` | R6 |

Answer to the Goal, bounded: yes at `sigma_Y = 0.25` and, for the constant-head and asymmetric analytic fields,
at `sigma_Y = 0.5`, with the formulation "(14) same-index, non-divergence, slab, inlet labels D-1, outlet
condition D-2, 4th-order stencils"; not established at `sigma_Y = 1` on grids up to 28^3 (inconclusive by
resolution). Criteria as written at amplitudes 0.25, 0.5, 1: met at 0.25, met at 0.5 for `gauss_ch` and
`generic3d` and missed by one `e_psi` pair for `gauss` (1.73) and `control2d` (1.58), not met at 1. No criterion
was relaxed. The owner's decision on this outcome (2026-10-06, "Option A": formulation decided with bounded
validity, `sigma_Y = 1` an open item) is recorded in
`docs/decisions/2026-10-06-eq14-inlet-label-formulation.md`.

## Caveats

- One Gaussian realization (seed 7, band-limited `|m_i| <= 3`, `L/ell = 4`); the analytic fields are single
  fixed fields. No robustness over realizations.
- Grids: corrective ladders 16/20/24/28 (+32 for two cells); 48^3 exists only for 2nd-order `i1` from the first
  sweep (`gauss:0.25` and the analytic fields that finished). Orders at these resolutions are pre-asymptotic for
  `i1o4` too (rising at `eps = 0.25`).
- `eps = 1` is unresolved: three `i1o4` 28^3 cells timed out at the 21 600 s cap; the resolution requirement
  `ell/h >= 24-32` is an estimate from the oracle probes (N1 mid-slab, `cons4`), not a measurement of a candidate.
- `control2d` `e_psi` at small values: non-monotone order sequences (`eps = 0.25`: 7.85, 4.06, 2.26 with `e_psi`
  1.883e-3 at 28^3; `eps = 0.5`: 3.62, 1.58, 3.28), the latter classified FAIL by the D-5 rule.
- D-5 is a reading rule of the orchestrator; its operational form changed between the sweeps (printed at the top
  of each summary). D-6 and D-7 extend the spec's scope (grid extension; the 4th-order variant by owner
  directive). The reviewer adjudicates.
- The orchestrator's own probes (linearized `k = 1` kernel instrument of §8, the `control2d` 16..64 probe of the
  N2 audit) are recorded in the bitácora and the audit records, not in `raw/`; the 64^3 `control2d` value used in
  R4 is the sweep's own (`raw/sweep/summary.md`).
- The `N_phi` table and the 128^3 mid-slab probe ran locally as single-process jobs (26 + 31 min) at the
  orchestrator's request (bitácora 2026-10-03T03:05Z).
- The flows here have `v1 > 0` everywhere (checked per case). SF-30 found backflow (`v1 <= 0`) at
  `sigma^2 = 4` on the production stack (PR #46, `docs/experiments/2026-10-05-sf30-streamline-closure-gate.md`):
  inlet labels and an `x1`-parametrized oracle are not defined there as built here.
- Nothing in this note bears on `alpha_T` (theory note `docs/theory/lester-2023-key-claims.md` §4).

## Next step

Decision record `docs/decisions/2026-10-06-eq14-inlet-label-formulation.md` (formulation locked with bounded
validity) and the next-phase increments SF-33 (GPU implementation of the inlet-label formulation) and SF-34
(acceptance in the periodic medium against the SF-30 closure measurement).
