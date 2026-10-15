# Label transport by backward streamline tracing is the production constructor of the Darcy labels; equation (14) on the slab is a diagnostic

- Status: accepted (owner, 2026-10-07; human review on the SF-35 PR)
- Date: 2026-10-07
- Deciders: owner (orchestrator session 2026-10-07)
- Evidence: `docs/experiments/2026-10-06-sf33-gpu-inlet-labels.md` (Result; `analysis/preconditioner_gate_b.md`,
  `analysis/ladder_orders_025_c4.md`, `analysis/oracle32_vs_sf29.md`), `.claude/orchestration/SF-33-gpu-inlet-label-streamfunctions/analysis/options-analysis.md` (runtime record, not versioned)
- Builds on: `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md` (O1 = B: inlet-anchored Darcy labels, non-periodic in `x1`),
  `docs/decisions/2026-10-06-eq14-inlet-label-formulation.md` (D-1 inlet labels, slab, stencils, metrics)
- Supersedes: the "Formulation" and "Characteristics" rows of the 2026-10-02 delegated-decisions table; items 1, 3, 5 and 6 of the
  2026-10-06 Decision and its Consequences bullet naming tracing as the oracle

## Context

SF-33 implemented the 2026-10-06 formulation on GPU and established that the GPU solves the SF-29 discrete problem
exactly (14/14 cases of the 9a matrix within 3.84e-11 of the prototype) and that the production oracle — backward
streamline tracing from every vertex to the inlet plane on the SF-28 spline of the SF-19 flow, D-1 labels read at the
feet — is sound (round trips <= 1.2e-9 on every grid, 32^3 oracle vs the SF-29 spectral oracle 0.6 % RMS at order 2.00).
The Newton–Krylov solve of equation (14), however, is not a viable production constructor: the preconditioner gate
fails at 32^3 `eps = 0.5` (GMRES stall), and at `eps = 0.25` the 128^3 stage reaches a genuine linear-solver plateau
for every amplitude `>= 0.14` (flat restart curves with P-A, Eisenstat–Walker forcing, pseudo-transient continuation
and a Galerkin coarse correction; the host coarse LU alone is 79 % of 2 h 59 min). Diagnosis (recorded in the SF-33
note): the linearization's `x1`-independent block has identically zero principal-symbol determinant, so that family of
modes is fixed only by lower-order terms and boundary rows (singular values ~`h^2`), and any preconditioner built from
plane-averaged coefficients responds ~`h^-2` to the transverse coefficient mismatch. Lester et al. (2021 §3, 2023 §5.1)
never used Newton–Krylov on (14): they used explicit first-order pseudo-time stepping.

The downstream needs are SF-34 (`sigma^2` in {0.25, 1, 2.25}, `ell` 1/8–1/16, 64–256^3) and the long domain
(2048 x 256 x 256, `lambda/h = 10`, 2.7e8 unknowns): no Newton–Krylov variant reaches either (Krylov memory alone
exceeds one V100 on the long domain), and (14) differentiates `ln k` at a resolution below the one SF-29 estimated as
necessary. The labels of the Darcy flow are *defined* by transport (`v . grad psi_i = 0` with inlet data); the
constructor that follows that definition — tracing — already exists, is measured, costs 3 min at 128^3 on the host,
needs no `grad ln k`, has no amplitude limit other than the existence of the flow, and is the standard construction of
Euler potentials in other fields.

## Decision

1. **Constructor.** The production constructor of the inlet-anchored Darcy labels on the `x1`-slab is label transport
   along backward-traced streamlines: every vertex `(j/N, m2/N, m3/N)`, `j = 1..N`, is traced backward along
   `g = G + grad s_h` (SF-28 spline of the SF-19 potential, `k = exp(s_Y)`) with the SF-30 DP5(4) arclength integrator
   and Hénon landing to the plane `x1 = 0`, and `(psi1, psi2)` are the D-1 inlet labels at the foot. Production
   settings: `h_max = h/8`, `tol = 1e-8`, forward round trip per plane `<= 1e-8` required. Nothing is clamped or
   regularized; interior backflow and stagnation are followed and counted; inlet backflow (`v1 <= 0` on the inlet
   face) stops the case (D-1 undefined). Realized by SF-35 (`src/physics/streamfunctions/inlet_slab/`, host tracer;
   GPU tracer accepted only on host/GPU equivalence).
2. **Equation (14) as a diagnostic.** The same-index equation (14) in non-divergence form with the 4th-order stencils
   and the D-2 outlet rows (2026-10-06 items 1, 3, 4) is retained as the diagnostic operator: `r_F`, `r_out` evaluated
   at the constructed labels are necessary conditions, never acceptance quantities by themselves. The slab
   Newton–Krylov solver (SF-33) is kept as an instrument for cross-construction where it converges (`<= 64^3`,
   `eps <= 0.5` on the production field).
3. **Independence rule, rewritten.** A constructor is accepted only on checks that do not share its construction. For
   the tracing constructor these are: (i) grid finite-difference metrics (`e_v`, `e_i`, `e_div`, `|c|` percentiles) and
   the equation-(14) residual, which differentiate on the grid and never integrate along curves; (ii) positive controls
   with exactly known labels (`k = 1`; `k = k(x2)` with straight streamlines; `control2d`); (iii) agreement with the
   elliptic construction where it exists (two independent constructions of the same object); (iv) the flow-level SF-30
   return map (SF-34). What is **not** independently checked by (i)–(iii): the spline flow itself and the D-1 inlet data,
   which every check shares with the constructor; the flow-level check is (iv). The forward round trip certifies only
   the self-consistency of the integration; the `h_max`/`tol` ladder and the straight-line control bound its
   systematic error.
4. **One constructor for both domains.** Tracing from the inlet face serves the periodic cell (SF-34) and the long
   domain (Dirichlet in `x`); (14) is the diagnostic on both. The "x-marching" row of 2026-10-02 stays discarded:
   plane marching in `x1` is undefined where `v1 <= 0`; tracing follows characteristics.
5. **Validity envelope.** The amplitudes and resolutions at which SF-35 classifies the constructor `accept` are the
   envelope; it is filled at SF-35 closure, not presupposed. `alpha_T` is not presupposed anywhere (O3 unchanged).

## Consequences

- SF-33 is closed `done` with claim (a) and the oracle established and claim (b) recorded as not established; its
  module is the basis of SF-35 (oracle, inlet labels, production setup, metrics, residual, tests).
- SF-35 (new) establishes the constructor; SF-34 is re-specified to accept the SF-35 labels against the SF-30 return
  map; the long-domain phase follows SF-34 as before.
- The dashboard's "Locked decisions of the inlet-label formulation" keep items 2 (D-1), 4 (stencils, as diagnostic
  operators) and 7 (metrics; `e_psi` now against the elliptic cross-construction where it exists); items 1, 3, 5, 6 are
  superseded by this record.
- `ARCHITECTURE.md` §4.3/§5 and `AGENTS.md` describe the constructor as tracing and (14) as the diagnostic.
- `docs/validation/acceptance-gates.md` Gate 3A gains the mapping for a tracing constructor (status histogram,
  per-plane round trip, error growth along `x1`, backflow statistics; "Newton failures" not applicable).
- Not changed: the frozen periodic-fluctuation stack (SF-02..SF-26), SF-28, SF-30, SF-31/32, the Gaussian-only
  covariance rule, the periodic-cell study and the move-to-long-domain requirement (`e_v(h)` convergent and return map
  concordant up to `sigma^2 = 2.25`).

## Classification (docs/AGENTS.md)

| item | class |
|---|---|
| GPU solves the SF-29 discrete problem; linear-solver plateau at 128^3 and at 32^3 `eps = 0.5` | confirmed in runs (SF-33 note, O-1/O-2/O-4) |
| Production oracle round trips `<= 1.2e-9`; 32^3 oracle vs SF-29 oracle 0.6 %, order 2.00 | confirmed in runs (SF-33 note, O-6) |
| Zero principal-symbol determinant of the `x1`-independent block (mechanism of the plateau) | confirmed by derivation (SF-29 record §2.1); consistent with the runs, not a theorem proved here |
| Items 1-5 of the Decision | accepted scope |
| Host tracer as production constructor; GPU tracer; per-plane growth and backflow accounting | proposed architecture (SF-35) |
| Convergence of the traced labels' FD metrics at `sigma_Y` in {0.5, 1, 1.5} on 32-256^3; behaviour at `ell` 1/8-1/16, other realizations; the long domain | open question |
