# SF-32 — Face-flux reference trackers and the paper's scalings — UNDERSTAND record

- Orchestrator: Claude Fable 5.1, session started 2026-10-06T14:40Z (UTC).
- Increment: SF-32 (`docs/plans/active/lester-eq14/increments/SF-32-reference-trackers-and-scalings.md`).
- Exact Goal: `Reproducir las figuras 3 y 4 de Lester 2023, mostrando la dispersión transversal espuria de los
  trackers convencionales frente al pseudo-simpléctico en un campo de invariantes exactos.`
- Persistent-goal runtime feature: not available in this session; the Goal is carried by this record and `dag.json`.
- Default branch at activation: `master = origin/master = 37bfb25e1d5ae3eae2cbe680d6345513a27c2df6` (PR #49, SF-29
  closure repair). Working tree clean apart from the untracked `docs/references/presentation.pdf` (not touched).
- Checker on `master`: `OK (35 increments, ready=SF-32 SF-33, nonterminal=none)`. SF-31 (the only dependency) is
  `done` on the default branch (PR #47 merged as `b943f8d`). No other increment is nonterminal, so activating SF-32
  gives nonterminal = {SF-32} = 1 (max 2). SF-33 stays READY for a possible concurrent session; if one starts, remote
  V100 work is already isolated per increment (`scripts/remote --increment SF-32`, mirror `~/MacroFlow3D-SF-32`).
- Increment base: `37bfb25e1d5ae3eae2cbe680d6345513a27c2df6`.
- Delivery branch: `science/lester-sf32-reference-trackers-scalings`, orchestrator-owned worktree
  `.agents/worktrees/lester-sf32-reference-trackers-scalings` (Par2_Core submodule populated). The control checkout
  stays on `master`.
- Review class: `Human review: required` (spec) — code that consumes `psi1`/`psi2` as invariants, a new tracker, a
  scientific comparison with the paper's figures (autonomy policy: mandatory human review). Publish as
  `awaiting_review`; `done` only after explicit owner approval, by a metadata-only commit on the same PR.
- Remote: `scripts/remote --increment SF-32` (fresh mirror). The orchestrator owns every remote call (detached jobs,
  one per GPU, `REMOTE_GPU_WAIT` when both GPUs are busy); workers never use `scripts/remote`. V100 toolchain is
  CUDA 11.4 / nvcc 11.4 (SF-30 finding C3: the vendored nlohmann header inside a `.cu` gives an internal compiler
  error there; `const Functor f;` without braces warns). Local is CUDA 13.4 on an RTX 3050 (4 GiB, sm_86).
- Scientific-rigor skill: INVOKED (user instruction). Categories: B (two new numerical methods: Stokes face fluxes
  from the labels and a Pollock-type semi-analytical cell tracker; a scaling study of integrator errors) and
  C (GPU implementation contracts). Scientific interpretation is bounded (section 8): the increment measures
  NUMERICAL transverse spreading of trackers on a surrogate flow with exact invariants; it says nothing about
  Darcy flow or about `alpha_T` (owner decision O3).
- Housekeeping noted, not done yet: the SF-31 mirror `~/MacroFlow3D-SF-31` is still on V100 after the merge of
  PR #47 (bitácora: remove after merge). It does not block SF-32; remove it at the end of this session.

## 1. What the increment delivers and what it must not

Deliverables (spec "In scope"):

1. `StokesFaceVelocity`: cell-face fluxes of `v = grad psi1 x grad psi2` computed from LINE integrals of
   `psi1 grad psi2` around each face (Lester 2023 eqs. 32-33), exactly divergence-free per cell.
2. A Pollock-type (cellwise-linear, RT0) semi-analytical tracker driven by those face fluxes, in the same module
   `src/physics/particles/streamline_tracker/`.
3. Streamfunctions of the Lester (2021) field from the frozen periodic stack (route recorded; fallbacks: reduce
   the amplitude, then an analytic pair; the solver is never touched).
4. Diagnostics `delta psi_i`, `delta x2`, `delta x3` after one period (eqs. 34-35) for the three trackers
   (Pollock, SF-31 RK reference, SF-31 pseudo-symplectic), grid and tolerance ladders run as detached V100 jobs,
   fitted exponents, an experiment note `docs/experiments/2026-10-06-sf32-spurious-spreading.md` with the analogues
   of figures 3b, 3c, 4a, 4b.

Out of scope (spec): runner wiring, ensembles, macrodispersion interpretation. Also: any change under
`src/physics/streamfunctions/` (frozen), `pspta/`, Par2, SF-28, SF-19, SF-18; any change to the SF-31 integrator
cores or engines beyond what an audit finding would force (none expected; the new instrument TEMPLATES on the
existing cores instead of modifying them); `x1`-non-periodic labels (SF-33/34).

What the result may be used for: the quantitative behaviour of the three trackers on the surrogate (spurious
spreading versus `Delta` and `tol`; absence of spreading beyond `tol_psi` for the pseudo-symplectic tracker). What
it does not establish: anything about the Darcy flow of a Gaussian field, about label quality, or about
`alpha_T`.

## 2. Source check: what the paper does in section 5 (read from `docs/references/Lester-2023-WRR.pdf`, pp. 10-15)

- §5.1 (p. 10): triply periodic unit cube, Table 1 parameters, `psi_i = mean + fluctuation`, `bar v = 1`;
  streamfunctions from the finite-difference solution of eq. (14) to `1e-16`; **periodic cubic splines** of the
  grid values give continuous `psi1, psi2`; "for the purpose of testing particle tracking methods we treat these
  interpolated streamfunctions as being exact". The resulting velocity is exactly divergence-free (eq. 13) but
  not exactly helicity-free (Fig. 2b).
- §5.2 (pp. 11-13), Pollock: velocity "arises from the cell-centered finite volume discretization", face
  velocities computed from the streamfunctions on the 256^3 grid via Stokes:
  `v_i^{pqr} = Delta0^-2 \iint_S (grad psi1 x grad psi2).e_i dA = Delta0^-2 \oint_{dS} psi1 grad psi2 . n dl`
  (eq. 32); hence `sum_faces (v+ - v-) = 0` exactly (eq. 33); coarser grids `Delta/Delta0 = 1, 2, ..., 8` by
  "appropriately averaging over the cell face velocities". Pollock = linear interpolation of the face velocities
  per cell, exit position/time analytic, cell to cell. Errors after traversing `Omega` (eq. 34):
  `delta_{x2,i} = x_{2,1,i} - x_{2,0,i}`, `delta_{x3,i} = x_{3,1,i} - x_{3,0,i}` ("as streamlines are periodic
  within Omega" — the exact streamline returns to its start, so the error is the one-period displacement);
  streamfunction errors `e_{psi_i} = psi_i(x_1) - psi_i(x_0)` (eq. 35), "strongly correlated with the spatial
  errors, not shown". Fig. 3b: PDFs of `delta_x2` (dashed), `delta_x3` (solid) for `Delta/Delta0 = 1, 2, 4, 8`;
  Fig. 3c: variances `sigma^2_{x2}` (blue), `sigma^2_{x3}` (red) vs `Delta/Delta0`, "grow roughly as
  `(Delta/Delta0)^2`" (solid line); `sigma^2_{x3}` visibly larger than `sigma^2_{x2}`. Eq. (36):
  `D^m_{ii} = sigma^2_{xi} / (2 <tau_Omega>)`.
- §5.3 (p. 13), RK: "a 4th order Runge-Kutta algorithm with an adaptive step size to achieve a prescribed
  tolerance tol", velocity from the interpolated streamfunctions ("exact"), `tol = 1e-4 .. 1e-8` (five values,
  Fig. 4). Fig. 4a: PDFs; Fig. 4b: variances "grow roughly as `sqrt(tol)`". Pair, error norm and controller are
  NOT given.
- §5.4 (pp. 13-15): the paper's pseudo-symplectic method (inverse functions, 1-D quadrature) is "not
  implemented ... in the purely advective case as the results are trivial (`delta = 0`)". The SF-31 tracker is a
  projection method (SF-31 record D-1); in SF-32 it plays the role of the exact-invariant tracker.

Consequences for SF-32: (a) "after one period" = first return of the unwrapped `x1` to `x1_0 + 1`, the same
observable as the SF-30 return map, on a flow whose streamlines close by construction (affine + periodic
labels, SF-30 §1 argument), so the reference point is the start point itself; (b) the Pollock face fluxes must be
exact surface integrals of the spline velocity (then the ONLY Pollock error is the RT0 interpolation); (c) the
RK exponent of the paper is tied to an unspecified RK4 pair on an unspecified periodic cubic spline, while SF-32
measures a DP5(4) pair on the SF-28 C^2 B-spline — comparability of the exponent is limited and is recorded as
such BEFORE any run (section 6, prediction P2).

## 3. Numerical contract (fixed before any implementation)

### 3.1 Labels and the three velocity objects

Labels as in SF-31: `psi_i(x) = gbar_i . x_u + s_i(x)`, `gbar_1 = (0, 1, 0)`, `gbar_2 = (0, 0, 1)` (`vbar = 1`:
the frozen stack with `qbar = e1` achieves mean flux 1, `AffineGauge::benchmark(1)`), `s_i` the SF-28 periodic
tricubic B-spline (C^2, knots at the cell centres `(j + 1/2) h`) of the cell-centred fluctuation `u_i` on an
`N^3` grid of the unit cube. Three consumers of the SAME pair:

| tracker | velocity object | exact invariant? |
|---|---|---|
| pseudo-symplectic (SF-31) | the labels themselves (projection onto `{psi1 = psi1_0, psi2 = psi2_0}`) | yes, to `tol_psi` |
| RK reference (SF-31 DP5(4)) | `c(x) = grad psi1 x grad psi2` of the splines, pointwise | no (label drift = the measured error) |
| Pollock (new) | face fluxes of `c` on a grid of spacing `Delta = m h` (`m = 1, 2, 4, 8`), RT0 interpolation | no (RT0 error = the measured error) |

"Exact invariants" means: the velocity of the surrogate flow is DEFINED as `grad psi1 x grad psi2` of the
splined pair; its streamlines are exactly the label curves. Whether the pair solves eq. (14) to a small residual
is irrelevant to this measurement (the paper makes the same move, §5.1); it matters only for the route record
(3.5).

### 3.2 Stokes face fluxes (StokesFaceVelocity)

Periodic MAC layout on the Pollock grid `G_Delta` (`n = N/m` cells per axis, spacing `Delta`, cell faces at
`i Delta`): `u[i,j,k]` = flux density through the x-face at `x = i Delta` of cell `(i, j, k)` (the face shared
with cell `i-1`, index mod `n`), `v[i,j,k]` the y-face at `y = j Delta`, `w[i,j,k]` the z-face at `z = k Delta`;
each face value is the AVERAGE of `c . n` over the face (`Delta^-2` times the flux), so a uniform pair gives
`u = 1`. Construction, per face:

```text
u[i,j,k] = Delta^-2 * oint_{dS} psi1 grad psi2 . dl      (eq. 32; right-handed about +e_i)
         = Delta^-2 * (E_y(bottom) + E_z(right) - E_y(top) - E_z(left))
```

with ONE edge integral `E = int_edge psi1 (grad psi2 . t) dl` per grid edge (three edge arrays, each `n^3`),
reused by the two faces that share the edge with opposite signs. Hence the discrete divergence
`u[i+1]-u[i] + v[j+1]-v[j] + w[k+1]-w[k]` is an algebraic sum in which every edge appears twice with opposite
sign: zero up to roundoff (eq. 33), whatever the quadrature.

Periodic decomposition (CORRECTED 2026-10-06T15:10Z after the orchestrator prototype, before any worker was
launched): `psi1 = gbar1 . x + s1` is NOT periodic, so a periodic edge array of `oint psi1 grad psi2 . dl` is
wrong at the faces that touch `x2 = L` (the prototype gave `u - 1 = -(n - 1)` on the last row: the edge at
`x2 = L` was represented by the edge at `x2 = 0`, which differs by `L * Delta(psi2)` along the edge, not by a
closed-loop zero). Exact fix: expand `(gbar1 + grad s1) x (gbar2 + grad s2)` and apply Stokes term by term with
PERIODIC integrands only:

```text
flux / area = (gbar1 x gbar2) . n                                   (mean flow; constant)
            + area^-1 * oint [ s1 (grad s2 + gbar2) - s2 gbar1 ] . dl   (periodic integrand)
```

(`curl(s1 (grad s2 + gbar2)) = grad s1 x (grad s2 + gbar2)`, `curl(s2 gbar1) = grad s2 x gbar1`, so the loop
term equals `iint [grad s1 x (grad s2 + gbar2) + gbar1 x grad s2] . n`.) The per-edge integrand along axis `d` is
`s1 (d_d s2 + gbar2[d]) - s2 gbar1[d]`, a periodic function of position: one periodic edge array per direction is
exact and consistent, the divergence cancellation is purely algebraic, and the constant term cancels trivially.
Quadrature: on an edge the integrand is piecewise polynomial of degree <= 6 (cubic x cubic) with breaks at the
spline knots `(j + 1/2) h` inside the edge; composite Gauss-Legendre with 4 nodes per knot interval (exact for
degree <= 7) makes every edge integral EXACT up to roundoff for the spline velocity.

Consequences fixed now (tests): (S1) `max |div| <= 1e-13 * max|u|` on every grid; (S2) the face value equals a
direct 2-D composite Gauss (4x4 per knot cell) quadrature of `c . n` over the face to `1e-12` relative; (S3) the
coarse face value at `Delta = 2h` equals the area-weighted mean of its four fine face values to `1e-13`
(additivity of exact integrals; this is also the paper's "averaging"); (S4) uniform pair: `u = 1`, `v = w = 0`
to `1e-15`; (S5) pair A (`psi1 = x2 + a sin 2 pi x1`, `psi2 = x3`): `u = 1`,
`v[i] = -a (sin 2 pi x1_{i+1} - sin 2 pi x1_i) / Delta` up to the spline interpolation error of `sin` (measured
on the same grid and reported), `w = 0`; (S6) host mirror (same inline edge integral called on host coefficients)
equals the device result to `1e-13` relative; (S7) no allocation in the evaluation kernels beyond the output
buffers (the prefilter is SF-28's).

### 3.3 Pollock-type tracker (RT0 cellwise-linear, semi-analytical; Pollock 1988)

Inside cell `(i, j, k)` of `G_Delta`, with `x_-` its lower corner and face values `u_-, u_+` (etc.):

```text
v_x(x) = u_- + A_x (x - x_-),   A_x = (u_+ - u_-) / Delta          (same for y, z; each component depends on its own coordinate only)
```

Per axis, from the particle position `x_p` with `v_p = v_x(x_p)` (CORRECTED 2026-10-06T15:25Z after the
orchestrator prototype, before N1 was launched — the textbook forms `ln(v_f / v_p) / A` and
`(v_p e^{A t} - u_-) / A` are catastrophically ill-conditioned when the two face values are equal up to roundoff
(`A ~ 1e-14`): the prototype produced a garbage exit time on pair A, where `A_y` is a rounding residual. The
well-conditioned forms below are mathematically identical, need NO threshold on `A`, and tend smoothly to the
linear case):

- `v_p == 0` -> no exit on this axis (`t_x = +inf`);
- candidate face by the sign of `v_p`: `d = x_+ - x_p` if `v_p > 0`, `d = x_- - x_p` if `v_p < 0`
  (`d` signed, same sign as `v_p`); for the linear interpolant `v_f = v_p + A d` EXACTLY, so define
  `z = A d / v_p` (`>= -1` iff the velocity does not vanish before the face);
- `not (1 + z > 0)` -> the velocity vanishes before the face: no exit on this axis;
- `A == 0` (exact equality only): `t_x = d / v_p`; else `t_x = log1p(z) / A`;
- position after `t` on any axis: `x(t) = x_p + v_p t` if `A == 0`, else `x(t) = x_p + v_p expm1(A t) / A`;
  the final partial time to reach `x1 = target` inside a cell: `t = log1p(A (target - x_p) / v_p) / A`.

- exit time `t_e = min(t_x, t_y, t_z)`; if `t_e = +inf` -> status `kStatusPollockStagnation` (new code, 15),
  particle frozen (no fallback, AGENTS.md).
- position after `t` (`0 < t <= t_e`) by the `expm1` form above; the exit coordinate is SET to the face
  coordinate exactly (not computed from the exponential), the other two are evaluated by the formula; `t += t_e`; the cell index advances by +-1 on the exit axis (mod
  `n`, with the integer wrap counters of the SF-31 bookkeeping when the index wraps); a particle exactly on a
  face belongs to the cell it is moving into (ties `t_x == t_y` advance both indices).
- `advance_to_x1(target)`: cell steps while the unwrapped `x1` at the next exit stays `<= target`... precisely:
  if the current cell's x-exit at `x_+` has unwrapped coordinate `> target` (or the exit is on another axis)
  the step is taken; when the x-exit coordinate equals the target (seeds on a face: `x1_0 = 0`, target
  `x1_0 + 1` is a face coordinate) the particle lands on it exactly; in general (target inside a cell) the final
  partial time is `t = ln((A_x (target - x_-) + u_-) / v_p) / A_x` (or linear), closed form. Returns the
  unwrapped position, the clock and the number of cells crossed.
- `advance_to_time(t_target)`: cell exits while `t + t_e <= t_target`, then the partial step `x(t_target - t)`.
- Guards: `max_cells_per_call` -> `kStatusSubstepLimit` (12); non-finite -> `kStatusNonFinite` (14).
- Double precision; one thread per particle; no atomics or reductions: bitwise deterministic. Host/device
  `__host__ __device__` core on a POD `PeriodicFaceFluxView`, GPU kernels in the module.

Known-answer checks fixed now: (P1) uniform pair: one period in `tau = 1` exactly (dyadic `Delta`), `x2`, `x3`
bitwise unchanged; (P2) pair A: the face flux `v` is the exact cell average of `v2 = -2 pi a cos 2 pi x1` and
`u = 1` is exact, so Pollock's `x2` at every x-face crossing equals the exact `x2_0 - a (sin 2 pi x1 - sin 2 pi
x1_0)` up to the spline error of `sin` (an exactness check of the exit arithmetic, not an order check);
(P3) pair B (`s2 = b sin 2 pi x2`, `c3 = A B` depends on `x1` and `x2`): Pollock error against the closed-form
streamline (`analytic_pairs.hpp exact_position`) decreases under `Delta -> Delta/2` over `n = 16, 32, 64, 128`;
observed order reported, gate = monotone decrease and two-level order `>= 0.8` on `delta_x3` (the paper's RT0
argument gives first order in the velocity, so at least first order in the displacement). RESTATED 2026-10-06T16:40Z
after the N1 report (orchestrator check defect, disclosed): the original target `x1 = x1_0 + 1/2` is an EXACT return
point of pair B by the odd symmetry of `sin` (`s1(1/2) = s1(0)`), so every tracker is exact there and no order can be
read; the targets are now `x1 = 0.25` (an x-face: `delta_x3` about second order, `delta_x2` at the spline floor,
fourth order) and `x1 = 0.30` (interior to a cell: exercises the closed-form partial step). N1 measured (RMS, n = 16 ..
128): target 0.25 `delta_x3` 7.50e-4, 1.47e-4, 4.68e-5, 9.38e-6 (orders 2.35, 1.66, 2.32); target 0.30 `delta_x2`
1.20e-3 .. 2.75e-5 and `delta_x3` 1.00e-3 .. 1.70e-5 (orders 1.4-2.6).
(P4) host core == GPU engine on the same inputs to `1e-14` (ulp-level `log1p`/`expm1` library differences between
host and device; discrete outcomes — statuses, cell counts, cells, wraps — identical; measured <= 4.4e-16); (P5) determinism across two runs (memcmp);
(P6) stagnation path: a pair with a cell where the flux reverses leaves status 15, no NaN, nothing committed past
the last exit.

### 3.4 Return map ("one period") for the three trackers

Seeds: `N_p = 8192` points on the face `x1 = 0`, `(x2, x3)` from the SF-31 stateless 53-bit hash
(`inject_box` with `x0 = x1 = 0`, seed `20261006`), identical for every tracker and every ladder level; subsets
of the first 1024 are also reported (sampling sensitivity). Per seed and tracker: first crossing of the unwrapped
`x1 = 1` (landing to `|x1_u - 1| <= 1e-12`), outputs `delta_x2 = x2_u - x2_0`, `delta_x3 = x3_u - x3_0`
(unwrapped, so a wrap in `x2`/`x3` is not an error), `delta_psi_i = psi_i(x_end) - psi_i(x_0)` (labels of the
splines at the landing point; for the pseudo-symplectic tracker this is its own residual), `tau` (clock at the
crossing), status, step/cell/panel counts. Landing:

- Pollock: lands on the x-face `x1 = 1` exactly (3.3).
- pseudo-symplectic: full panels of `ds` until the panel end has `x1_u >= 1`; from the saved pre-crossing state,
  bisection on the panel length `sigma in (0, ds]` (each trial = `advance_panel` from the saved state) until
  `|x1_u - 1| <= 1e-12` or 60 bisections (the crossing is transversal: `c1 >= min c1 > 0`); the landing state is
  a projected state, so the label residual stays `<= tol_psi`. `ds = h/2` of the label grid (`ds` is not the
  measured variable; a `tol_psi` ladder `{1e-8, 1e-10, 1e-12}` shows the drift floor).
- RK: `rk_advance_to_time` in chunks `dt = dt_max` until `x1_u >= 1` (D-2, CORRECTED 2026-10-06T16:05Z before any
  experiment run: `dt_max = 0.25` ABSOLUTE instead of `h / 2`; with the `h/2` cap the DP5(4) controller is inactive
  over the whole tolerance ladder — measured on G 32^3: identical results at `tol = 1e-4, 1e-6, 1e-8` — and the ladder
  would be flat for a trivial reason; the paper's RK has no cap; the chunk must equal `dt_max` because a shorter chunk
  re-caps the step at every chunk end); from the saved pre-crossing state, bisection
  on the chunk's target time (each trial re-integrates the chunk from the saved state with the clipped final step)
  until `|x1_u - 1| <= 1e-12` or 60 bisections. The re-integration changes the step sequence only in the clipped
  last step, which is within the controller's accepted error; recorded as a limit of the instrument (not Henon's
  device: that would need a new core; SF-31 cores are reused unmodified).

Statistics per run: mean and population variance of `delta_x2`, `delta_x3`, `delta_psi_1`, `delta_psi_2`,
`tau`; RMS of the label drift; max-norm; counts by status; histograms of `delta_x2`, `delta_x3` with 101 bins on
a symmetric range set per tracker as `+- 5 sigma` of its finest/tightest level... fixed instead as the paper's
axes: Pollock `[-0.75, 0.75]`, RK `[-0.04, 0.04]`, pseudo-symplectic `[-1e-9, 1e-9]`; eq. (36) numbers
`D_ii = sigma^2_{xi} / (2 <tau>)` (uniform seeds; also flux-weighted with weights `c1(x_0)`), reported as
protocol numbers of the surrogate, not as coefficients (O3).

### 3.5 Label routes (decision D-1 of this record)

1. PRIMARY: the frozen periodic stack on the Lester (2021) field `Y = eps f_lester2021` with `eps = 0.25`
   (`closure_fields.hpp`; SF-30 positive control), driven exactly as `apps/closure_gate/ev_ladder_main.cu` does
   (public API only, `AffineGauge::benchmark(1)`, SF-19 Darcy velocity as `darcy_velocity`, `epsilon = 1e-6`,
   Anderson on, Newton off, `pcg_rtol = 1e-10`), at `N = 256` (the paper's grid; `ell_eff/h`: shortest
   wavelength `1/4` -> 64 cells) and at `N = 128` (SF-30 ran it: 176 iterations, stagnated at `r_F = 1.09e-6`,
   `e_v = 2.72e-4`, 146 s on V100). The FINAL ACCEPTED STATE of the solve is the label pair whatever its exit
   reason (`converged`, `stagnated`, `budget_exhausted`): the surrogate flow is defined by the pair (3.1), and
   `r_F`, `e_v`, `e_psi`, `min |c|` are recorded next to it. The spec's "if it does not converge, reduce the
   amplitude" is read as: if the stack leaves a state that is unusable as a label pair (`min |c| <= 0.5` on the
   grid, a non-finite field, or `r_F` not below `1e-4` — the SF-30 numbers predict `1e-6`), reduce `eps` to
   `0.125`; if still unusable, the analytic pair is the only field. Deviation from the letter of the spec
   recorded for the reviewer: a stack state at `r_F ~ 1e-6` is accepted as the pair although `1e-8` was not met.
2. CONTROL (always run): the analytic SF-31 pair G scaled to amplitude `e = 0.05` (SF-31 used `0.03`;
   `min |c|` must stay `>= 0.5` on the grid, checked by the instrument and printed), sampled at the cell centres
   and splined at `N = 128` exactly like the stack pair. Solver-free; everything downstream is identical.
3. Label fields are saved once (raw `double` arrays `u1`, `u2`, layout `i + nx (j + ny k)`, plus a JSON side
   file with grid, route, `r_F`, `e_v`, `min |c|`) and the ladders load them; the solver is never re-run inside a
   ladder.

### 3.6 Ladders

- Pollock: `Delta/Delta0 = 1, 2, 4, 8` on each label grid (`n = 256, 128, 64, 32` for the 256^3 pair; `128,
  64, 32, 16` for the 128^3 pairs); exponents fitted on the four points by least squares in log-log of the
  VARIANCE `sigma^2_{xi}` versus `Delta` (spec: `p = 2 +- 0.3` over `>= 3` grids); also the two-level estimates.
- RK: `tol = 1e-4, 1e-5, 1e-6, 1e-7, 1e-8` (the paper's five) plus `1e-9, 1e-10` (reported; the spec band is
  evaluated on the paper's five and on all seven), `dt_max = 0.25` absolute (D-2); variance of `delta_x2`, `delta_x3` versus
  `tol` (spec: `p = 0.5 +- 0.15` over `>= 4` tolerances).
- pseudo-symplectic: `tol_psi = 1e-8, 1e-10, 1e-12`, `ds = h/2`; spec: `max |delta_psi_i| <= 10 tol_psi`.
- Every exponent outside its band is RECORDED, not tuned (spec); the increment then stays active for the human
  review. Fits use all seeds with status 0; the number of non-zero statuses is reported and must be `<= 1 %` for
  a level to enter a fit (otherwise the level is reported and excluded, with the reason).

## 4. Verification design (contract tests in `ctest`, fast tier; all `16^3`-`32^3` locally)

- `streamline_tracker_face_flux` (new executable): S1-S7 of 3.2 on pairs U, A, B, G at `N = 16, 32`
  (`Delta = h, 2h`).
- `streamline_tracker_pollock` (new executable): P1-P6 of 3.3 on pairs U, A, B at `n = 16 .. 128` (P3 ladder;
  the `128` level runs in well under a second on the GPU).
- `spurious_spreading_controls16` (new executable, instrument): on pair U at `16^3` all three trackers return to
  the start exactly (`delta = 0` to `1e-13`, `tau = 1` to `1e-13`); on pair A the pseudo-symplectic and RK
  return maps have `|delta_x2| <= 1e-6` (spline error level, measured) and the Pollock one `<= 1e-6`; on pair B
  Pollock's `delta_x3` variance decreases from `n = 16` to `32` (RESTATED 2026-10-06T17:30Z after the N2b report:
  pair B is EXACT for Pollock over a full period — separable velocity, `x3` a fixed function of the `x2` path, which
  returns exactly — so the control uses pair G, labels `N = 16` vs `32`, gate `var(N=32) <= var(N=16)/2`; the
  pair-B divergence checks stay); the output files are byte-reproducible across
  two runs.
- The experiment itself is not a ctest entry (process rule 4): detached V100 jobs, outputs under
  `docs/experiments/artifacts/2026-10-06-sf32-spurious-spreading/` (summaries, histograms, figures, fitted
  exponents, logs; per-seed CSVs only for the 256^3 primary field, gzip-compressed if above 5 MB).

## 5. Spec thresholds -> pre-registered readings

| # | Spec threshold | Reading fixed now | Node |
|---|---|---|---|
| T1 | Pollock variance `~ Delta^p`, `p = 2 +- 0.3` over `>= 3` grids | least-squares log-log slope of `sigma^2_{x2}` and of `sigma^2_{x3}` over `Delta/Delta0 = 1, 2, 4, 8`, each in `[1.7, 2.3]`, on the primary 256^3 pair; the 128^3 pair and the analytic control are reported (robustness), two-level estimates reported | N2b/N4, experiment |
| T2 | RK `p = 0.5 +- 0.15` over `>= 4` tolerances | slope of `sigma^2_{x2}`, `sigma^2_{x3}` versus `tol` over `1e-4 .. 1e-8` (paper's five), each in `[0.35, 0.65]`; the seven-level slope and monotonicity reported | N2a/N4, experiment |
| T3 | pseudo-symplectic drift `<= 10 x` the Newton tolerance | `max_p max_i |delta_psi_i| <= 10 tol_psi` at every `tol_psi` of the ladder, all seeds active, on every field; `delta_x2`, `delta_x3` RMS reported (expected `<= (tol_psi + 1e-12) / min|grad psi|`) | experiment |
| T4 | an exponent outside the band is recorded, not tuned | every fitted slope with its band verdict goes to the note and the bitácora; no ladder point, seed set, norm or controller is changed after the first result | orchestrator |

Additional contract checks fixed now: S1-S7, P1-P6 (sections 3.2-3.3); the instrument controls of section 4;
determinism and byte-reproducibility of outputs; `ctest -N` count `18 + 3` (SF-31 left 20: 18 + 2; SF-32 adds
3 -> 23).

## 6. Predictions fixed before any run (to be confronted, not tuned)

- P1 (Pollock): the variance slope is expected near 2 (paper). Mechanism check derived here: RT0 interpolation
  omits the transverse variation of each component inside a cell, a velocity error `O(Delta)` with zero cell
  mean; coherent accumulation over `1/Delta` cells gives `std ~ Delta` (variance slope 2), random-walk
  accumulation gives `std ~ Delta^1.5` (slope 3). The paper's Fig. 3c sits at 2 with `sigma^2_{x3} >
  sigma^2_{x2}`. Both outcomes are admissible readings; slope 3 would be outside the band and recorded.
  **P1' (sharpened 2026-10-06T15:25Z by the orchestrator prototype `audits/tools/prototype_pollock.py`, before
  any worker result; the spec band is NOT changed).** On the analytic pair G (`e = 0.05`, exact face fluxes,
  `n = 8, 16, 32, 64`, 1024 seeds) the Pollock variance slopes are 3.09 (`x2`) and 2.94 (`x3`), two-level
  2.61-3.24: the random-walk law, OUTSIDE the band `[1.7, 2.3]`. I therefore predict an out-of-band Pollock
  exponent near 3 on the splined fields in the asymptotic range, and read the paper's 2 as a pre-asymptotic
  value (its coarse grids resolve `ell = 1/16` with only 2-8 cells; our coarsest `Delta/Delta0 = 8` levels are
  also under-resolved for the `1/4` wavelength of `lester2021`, so the four-point fit may bend towards 2 at the
  coarse end — the two-level estimates will show it). The band verdict is reported as specified; nothing is
  tuned. Also predicted from the prototype: Pollock is EXACT on pair A once the well-conditioned forms are used
  (`4.4e-16` over 64 seeds, `tau = 1` exactly on the uniform pair).
- P2 (RK, DP5(4) on a C^2 B-spline): the velocity is C^1, the solution C^2, so a step that crosses a knot has
  local error `O(h^3)` instead of `O(h^6)`; the controller sets `h ~ tol^{1/5}` from the smooth estimate; with a
  resolution-fixed number of knot crossings per period the global error scales like `tol^{3/5}`:
  `std ~ tol^0.6`, variance slope `~ 1.2` (SF-31 measured std slopes 0.55-0.82 with plateaus on 32^3-64^3
  splines). I therefore PREDICT an RK variance slope in `[0.9, 1.7]`, OUTSIDE the paper's band `[0.35, 0.65]`,
  and possibly non-monotone at the tight end. If observed, it is recorded as a finding with this mechanism as the
  candidate explanation; the paper's `sqrt(tol)` belongs to an unspecified RK4 pair/controller/spline and is not
  reproducible as stated. Not tuning the controller or the pair to hit the band is part of the contract.
- P3 (pseudo-symplectic): `max |delta_psi_i| <= tol_psi + 1e-14` for all seeds (SF-31 T1), `delta_x` RMS at the
  `tol_psi / |grad psi|` level (`<= 1e-9` at `tol_psi = 1e-10`): the paper's qualitative figure-3/4 message
  (no spurious spreading) reproduced at the instrument level.
- P4 (face fluxes): divergence `<= 1e-13` relative; coarse = mean of fine to `1e-13`; exactness S2 to `1e-12`.
- P5 (travel time): `<tau>` agrees across the three trackers to `O(Delta^2)` (Pollock) and `O(tol)` (RK);
  flux-weighted `<tau> = 1` to the spline/quadrature level (mean flux 1 through the face).
- D-3 (added 2026-10-06 after the N1 audit; for the human reviewer): Pollock rounding guards — the computed relative
  position is projected on the closed cell `[0, D]` (an `expm1` result can overshoot by one ulp and flip the sign of
  the next distance; no tolerance, nothing added to a denominator), `t*` is capped at `t_e` when `x` is the exit axis,
  a zero face velocity at init picks the lower cell, and in time mode a stagnating cell is an exponential approach
  (status active), not a failure. Verified by the orchestrator's independent numpy replay to 1e-16.
- P6 (route): the stack at 256^3 leaves a usable state (`min |c| ~ 0.78` as at 32^3-128^3, `r_F <= 1e-5`).
  Not predicted: its iteration count, exit reason and wall time at 256^3.

## 7. Code inspected (control checkout at `37bfb25`)

- `src/physics/particles/streamline_tracker/StreamlineTrackerCommon.cuh`: `LabelSample`, label-evaluator concept,
  `SplineLabelPair` (`(xi, w)`), `cross3`, `norm3`, `wrap_position`, status codes 10-14, `inject_box`
  (53-bit hash), `compute_unwrapped`. Status codes 15+ are free for the Pollock tracker.
- `PseudoSymplecticTracker.cuh`: `__host__ __device__` cores `evaluate_label_state`, `project_to_label_curve`,
  `advance_panel(labels, PanelState&, psi1_0, psi2_0, sigma, prm, cnt)`, `advance_to_time`; `PanelState` is a
  POD (copyable: the bisection of 3.4 restarts from a saved copy). Degeneracy threshold D-11 (`16 eps`).
- `ReferenceRkTracker.cuh`: `LabelVelocity<E>`, `dp54_trial_step`, `rk_advance_to_time(vel, RkState&, L,
  t_target, prm, cnt)` (exact landing on `t_target`, FSAL inside a call, `k1` re-evaluated per call); `RkState`
  POD (copyable). Header section 6 documents the C^1 caveat.
- `src/numerics/interpolation/PeriodicTricubicBSpline.cuh`: knots at cell centres, `evaluate_point` host/device,
  `prefilter_periodic_tricubic_bspline` (GPU) and `_host` mirror, `make_host_view`; piecewise cubic between
  consecutive cell centres (the quadrature breakpoints of 3.2).
- `tests/streamline_tracker/analytic_pairs.hpp`: pairs U, A, H, B, G, D; `exact_position` for U, A, H, B;
  `sample_fluctuations`. Pair G at `e = 0.03`; SF-32 uses it with `e = 0.05` through a parameterized copy in the
  instrument (apps must not include test headers).
- `apps/closure_gate/ev_ladder_main.cu`: the complete recipe to drive the frozen stack through its public API
  (`PeriodicGaussianField`/analytic `Y`, `solve_affine_periodic_flow` with `qbar = e1`, `StreamfunctionProblemView`
  with triply periodic `BCSpec`, `fields.prepare(grid)`, `fields.fluctuations()` = `{u1, u2}` device spans,
  `solve_streamfunctions`, `StreamfunctionSolveReport`), the nlohmann-free `JVal` JSON writer (nvcc 11.4-safe,
  reusable by copy), `auto_mg_levels`.
- `apps/closure_gate/closure_fields.hpp`: `AnalyticField::lester2021` (`sin 2 pi x1 cos 2 pi x2 sin 2 pi x3 +
  0.4 sin 2 pi x1 sin 8 pi x3`), `fill_analytic_log_conductivity`, stateless `seed_point`.
- `src/physics/streamfunctions/affine_gauge.cuh`: `AffineGauge::benchmark(vbar)`,
  `PeriodicStreamfunctionFluctuations {u1, u2}`.
- `CMakeLists.txt`: explicit source lists; `STREAMLINE_TRACKER_SOURCES` (three `.cu`); one executable +
  `add_test` per fast contract test; `closure_gate` and `streamfunction_ev_ladder` as documented-experiment
  instruments; 20 ctest entries on the base.
- `docs/experiments/2026-10-05-sf30-streamline-closure-gate.md` §8, §10: eq. (36) reported as protocol numbers;
  frozen-stack `e_v` on `lester2021` at 32/64/128 (orders 1.94, 1.98), `min |c| = 0.775-0.788`.

Documentation/code discrepancies found: none that bears on SF-32. Informational: `ARCHITECTURE.md` §4.3 still
lists `src/physics/particles/streamline_tracker/` as "planned (SF-31/32)" although SF-31 is merged; not touched
here (same decision as SF-31; a docs follow-up).

## 8. Scientific interpretation boundary (Gate 4 statement to be made)

The flow tracked is the paper's surrogate (closed streamlines by construction) on the Lester (2021) field, which
is the one published case where the periodic solution of eq. (14) is also the Darcy flow (SF-30: closes to
`3e-11`). Every transverse displacement measured here is NUMERICAL by construction (the exact answer is zero);
the increment classifies it as numerical and quantifies its scaling with the discretization parameters. Nothing
is inferred about physical transverse macrodispersion, `alpha_T` is neither presupposed nor measured, and the
eq. (36) numbers are properties of the protocol applied to tracker errors (O3).

## 9. Risks

- The 256^3 stack solve is the largest run of this increment (memory ~4 GiB, time unknown; SF-30 never ran it);
  the 128^3 pair and the analytic control are independent of it, so a failure degrades the primary field to
  128^3 (recorded) rather than blocking.
- The RK bisection landing re-integrates a chunk; its effect is bounded by the controller tolerance but adds
  cost (60 trials x a chunk). Acceptable at `N_p = 8192`.
- Pollock corner/edge cases (ties, zero face velocities, a seed exactly on an edge of `G_Delta`): the hash seeds
  are generic doubles, ties have probability zero, but the code must be correct, not lucky (P6, status 15).
- nvcc 11.4 on V100: early V100 builds of accepted nodes; no nlohmann in `.cu`.
- GitHub over SSH port 22 was unreachable from this machine in the SF-30/31 sessions; publication may need the
  HTTPS transport with the `gh` credential (no repository configuration change).
- Band verdicts: P2 predicts an out-of-band RK exponent. The spec's policy covers it (recorded, human review).

## 10. Experiment findings that change the reading (recorded 2026-10-06T19:10Z, after the first V100 analysis)

- F-SYM (not predicted). On BOTH stack pairs (`lester2021`, `eps = 0.25`, 128^3 and 256^3) the Pollock return map is
  exact to roundoff at every `Delta/Delta0` (variances `1e-31 .. 1e-33`, max `|delta| <= 3.9e-15`, 8192/8192 ok), so
  the fitted "slope" (-0.6 .. -1.8, non-monotone) is roundoff scaling, not a tracker property. Cause: the Lester (2021)
  field is mirror-symmetric under `x1 -> 1/2 - x1` (the symmetry that forces its Darcy streamlines to close, finding R4
  of 2026-10-02); the labels inherit it (`u_i(Mx) = +-u_i(x)` to be checked numerically), the Stokes face fluxes on a
  mirror-symmetric Pollock grid (any even `n`) inherit it (`u` even, `v`, `w` odd), the RT0 interpolant maps cells to
  cells, so the Pollock trajectory from `x1 = 0` to `1/4` is mirrored back to `1/2` and again to `1`: exact return.
  The RK (DP5(4), not a symmetric integrator) does not cancel exactly but its errors are ~50x smaller than on the
  analytic pair G at the same tolerance (partial cancellation): the `lester2021` RK exponents are contaminated too.
  Consequence: the spec's primary field is DEGENERATE for the Pollock question (as pair B was for the controls); the
  measurable exponents of this experiment come from the analytic pair G (no symmetry): Pollock 3.030 / 3.040
  (`R^2` 0.99998 / 0.9995; two-level 2.83-3.15) — prediction P1' CONFIRMED, outside the spec band; RK 1.255 / 1.248
  over seven tolerances (paper's five: 1.90 / 1.85) with a plateau at the tight end (two-level 0.12, 0.03, 0.49) —
  prediction P2 CONFIRMED in kind (out of band, C^1 plateau), slope at the upper edge of the predicted range.
- POST-HOC ADDITION (disclosed; made BEFORE its run, 2026-10-06T19:12Z): a fourth field, the frozen stack on
  `lester_brk` (Lester (2021) with the phase `0.9` in the second term, which breaks the mirror symmetry; SF-30's
  negative closure control) at `eps = 0.25`, 128^3 and 256^3, same ladders. Reason: to have a STACK-produced pair
  without the symmetry, so that the exponents read on the analytic control are confronted with a solver-produced
  surrogate. Prediction (fixed now): Pollock variance slope `2.7-3.3` (random-walk law), RK slope `1.0-2.0` with a
  plateau at the tight end, PS `max|delta_psi| <= tol_psi`. The spec bands are unchanged; the `lester2021` results are
  kept and reported as the degenerate case they are.
- F-SYM numeric confirmation (2026-10-06T19:15Z): on the stack labels `max|u_i(Mx) - u_i(x)|` = 3.1e-16 / 2.9e-16
  (128^3) and 4.6e-15 / 5.4e-15 (256^3) with `max|u_i|` ~ 1.4e-2 / 1.7e-2: both fluctuations are EVEN under
  `x1 -> 1/2 - x1` (hence `c1` even, `c2`, `c3` odd). Pair G: `u1` even to 4.9e-17 but `u2` has no parity
  (`cos 2 pi (x1 - x3)` term), so G has no mirror symmetry and its Pollock errors are genuine. Pollock m1 on the 256^3
  stack pair: rms `delta_x2` = 5.6e-16.
