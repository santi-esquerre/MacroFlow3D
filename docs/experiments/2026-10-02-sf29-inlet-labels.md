# SF-29: equation (14) with `x1` non-periodic and inlet labels -- CPU prototype

- Date: 2026-10-02
- Status: planned
- Increment: `docs/plans/active/lester-eq14/increments/SF-29-eq14-inlet-labels-cpu-prototype.md`
- Theory: `docs/theory/lester-2023-key-claims.md` (equation (14), two-streamfunction representation);
  context `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`,
  `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`
- Artifacts: `docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/` (scripts, raw outputs, README)

This note is a skeleton written BEFORE any candidate run (DAG node N1). Criteria and predictions below are
fixed; results may not move them. Sections marked `pending` are filled by later nodes.

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

## Build / environment

No project binary; numpy/scipy only, CPU, double precision.

- Local WSL: Python 3.13 / numpy 2.5.0 / scipy 1.18.0 (oracle development, single-case smokes, 12^3 dense spectra).
- V100 host (CPU only, detached job): Python 3.11.7 / numpy 1.26.4 / scipy 1.11.4 (full sweep).
- The scripts avoid numpy-2-only API and Python >= 3.12 syntax (N1 ran `oracle.py --selftest` under
  Python 3.11 / numpy 1.26.4 / scipy 1.11.4 locally).

## Config(s)

- Fields: `control2d`, `lester2021`, `lester_brk`, `two_mode`, `generic3d`, `gauss` (closure-probe `FIELDS`,
  periodic cell, mean flux exactly `e1`) and `gauss_ch` (constant-head faces, D-4): 7 fields.
- Amplitudes `eps` in {0.25, 0.5, 1}; `k = exp(eps f)`.
- Candidate grids `N` in {16, 32, 48} (vertex grid `(N+1) x N x N`, `h = 1/N`).
- Candidates: (i-0) spec-literal outlet (control), (i-1) outlet condition D-2, (ii) Whitney/mimetic dissipation
  energy (free outlet in the constant-head case, outlet flux constraint D-3 in the periodic case).
- Dense Jacobian relative singular-value spectra at 12^3 and 16^3 at the converged state.
- Reference resolution `N_phi(field, eps)` from `oracle.py --nphi` (table in the artifact README), independent
  of `N`.
- Oracle: backward DOP853 tracing (rtol 1e-12, atol 1e-14) of every vertex to the inlet; inlet labels D-1.

## Commands

pending (filled by the sweep node; the N1 oracle commands are listed in the artifact README).

## Outputs inspected

pending

## Result

pending

## Caveats

pending

## Next step

pending
