# Equation (14) with `x1` non-periodic and inlet labels: formulation for the GPU phase

- Status: accepted (owner, 2026-10-06, Option A after the orchestrator's briefing; human review of the PR pending)
- Date: 2026-10-06
- Deciders: owner (outcome framing, Option A; scope extension D-7 of 2026-10-05), Claude Code orchestrator of
  SF-29 (formulation analysis, deviations D-1..D-6, audits)
- Evidence: `docs/experiments/2026-10-02-sf29-inlet-labels.md` (this record cites its items R1-R8) and its raw
  files under `docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/raw/` (`sweep/summary.md`,
  `sweep/timeouts.md`, `sweep2/summary.md`, `sweep2/timeouts.md`, `oracle_*.txt`, `cand_i*_*.txt`,
  `cand_ii_*`); increment `docs/plans/active/lester-eq14/increments/SF-29-eq14-inlet-labels-cpu-prototype.md`
  (bitácora).
- Builds on: `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md` (O1 = B: the tracker
  preserves the labels of the actual Darcy flow, non-periodic in `x1`, anchored at the inlet face).

## Context

SF-29 Goal: `Determinar con un prototipo CPU si la ecuación (14) generalizada a x1 no periódica, con etiquetas
fijadas en la cara de entrada, reproduce las etiquetas del flujo de Darcy real con e_v convergente bajo
refinamiento y sin floor.`

The periodic-fluctuation stack (SF-02..SF-26) is frozen: on generic smooth periodic `k` the Darcy streamlines do
not close on the torus, so no nondegenerate affine + periodic pair represents the Darcy flow and the periodic
solution of (14) is the closed-streamline field nearest to it (2026-10-02 record and
`docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`). SF-30 (PR #46, merged on `master` at
`9b10f1a`) repeated the closure measurement on the production stack (SF-18 field, SF-19 Darcy, SF-28 spline):
the Darcy streamlines do not close in any of its six cases, including `(sigma^2 = 4, ell = 1/16, 256^3)`, and
backflow (`v1 <= 0`) exists at `sigma^2 = 4`
(`docs/experiments/2026-10-05-sf30-streamline-closure-gate.md`). The labels of the Darcy flow must therefore be
anchored at an inlet face and non-periodic in `x1`. SF-29 tested, on CPU, which formulation determines them.

## Options considered

| option | one-line evidence | source |
|---|---|---|
| (i-0) spec-literal: (14) rows also on the outlet plane, no outlet condition | `2N` exact Jacobian nulls at `k = 1`; Newton plateaus in 17 of 21 cells (`r_F` 2.8e-4 .. 0.89): no solution | note R1 |
| (i-1) outlet condition D-2, 2nd-order stencils | well-posed and convergent (`r_F` <= 5.2e-14 with direct solves) but pre-asymptotic on 16-32 (`gauss`, `gauss_ch`: `e_v` orders 1.03-1.31, `e_psi` <= 1.12; `generic3d`: `e_v` 1.19-1.63) | note R4 |
| (i-1) outlet condition D-2, 4th-order stencils | criteria (1)-(5) PASS at `eps = 0.25` for all four fields (`e_v` orders 2.62-4.48); at 0.5 PASS for `gauss_ch`, `generic3d`; `eps = 1` inconclusive by resolution | note R2, R3, R5 |
| (ii) dissipation energy, Whitney/edge-averaged face fluxes | exact `2N^2` hourglass kernel (288 at 12^3) and the D-3 mean-transverse-flux gap: labels drift, `e_psi` 0.16-0.41 non-convergent | note R6 |
| (ii) dissipation energy, Q1 in-cell form + completed D-3 | kernel removed, but orders ~1.4 on the 3D Gaussian fields (20 -> 32) and 13 146-17 967 s per 32^3 solve | note R6 |

## Decision

The formulation of the next phase is decided with bounded validity (owner, Option A). Locked items:

1. **Equation and unknowns.** Same-index equation (14) in non-divergence form
   `lap psi_i - grad(ln k) . grad psi_i = S_i` (`i = 1, 2`; pairing per
   `docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md`), with analytic/exact `grad ln k` where
   available, on the slab `0 <= x1 <= 1`, periodic in `x2`, `x3`; unknowns `psi1 = x2 + u1`, `psi2 = x3 + u2` with
   `u_i` periodic in `(x2, x3)`. No regularization of `|grad psi1 x grad psi2|^2`.
2. **Inlet labels.** Dirichlet data on `x1 = 0` in flow coordinates by the normalized triangular construction
   (D-1): `Q(x3) = int_0^1 v1(0, s, x3) ds`, `psi2 = int_0^x3 Q`, `psi1 = int_0^x2 v1(0, s, x3) ds / Q(x3)`, built
   from the face velocity of the independent Darcy solve; requires `v1 > 0` on the inlet face.
3. **Outlet condition.** On `x1 = 1`: `(grad psi1 x grad psi2) x e1 = v_perp,in x e1` (D-2: the tangential
   Darcy velocity of the inlet face, by flow periodicity), which reduces to Neumann `d1 psi_i = 0` for
   constant-head faces. No condition is imposed on `d1 psi_i` at the inlet; the inlet oblique defect is a
   diagnostic only (P4).
4. **Discretization.** 4th-order stencils: centered in the interior and in `x2`, `x3`; Fornberg-skewed 4th order
   on the planes adjacent to the faces (planes 1 and `N-1`); one-sided 4th order for the outlet rows
   (recommended). 2nd-order stencils are admissible only with the resolution stated in item 6.
5. **Nonlinear method.** Newton with exact or robustly preconditioned linear solves; amplitude continuation
   `0.25 -> 0.5 -> 1` with bisection on failure, each stage started from the previous amplitude. The per-mode
   `k = 1` Fourier preconditioner is not sufficient beyond `eps = 0.25` at `N >= 32` (note R7); any
   preconditioner of the GPU phase must be validated against direct solves at 16^3-32^3.
6. **Validity and resolution.** Validated at `sigma_Y <= 0.5` for `L/ell = 4` on 16^3-32^3 (`sigma_Y = 0.25`:
   all four fields; `sigma_Y = 0.5`: `gauss_ch`, `generic3d`; `gauss` and `control2d` miss criterion (2) by one
   pair). At `sigma_Y = 1` the prototype is inconclusive by resolution; the required resolution is
   `ell/h >= 24-32` (N1 mid-slab probe, `cons4`), to be settled in the GPU phase. Consequence for the paper's
   parameters (`sigma_Y = 2`, `ell = 1/16`, 256^3, i.e. `ell/h = 16`): an open question, not an extrapolation.
7. **Acceptance metrics unchanged.** `e_v` (reconstruction of matching order) and `e_psi` (per-label) against an
   oracle independent of the solver; `r_F`; dense spectrum at small `N`; consistency (residual) at the oracle
   labels. The ceiling comparison D-5 is a reading aid, not a criterion.

## Consequences

- The GPU phase (SF-33) builds in `src/physics/streamfunctions/`: the slab grid (inlet Dirichlet plane, unknown
  planes 1..N, outlet oblique rows), the 4th-order stencils of item 4, the residual, a Jacobian-vector product,
  Newton-Krylov with a preconditioner validated against direct solves, continuation with bisection, and a
  production-stack oracle (streamline tracing of the SF-19 potential on the SF-28 spline from the inlet plane).
  It reuses SF-18 (generator), SF-19 (Darcy solve; its face velocity gives the inlet labels) and
  `Diagnostics.cuh`. The frozen periodic stack stays bitwise unchanged.
- SF-34 measures acceptance in the periodic medium against the SF-30 closure measurement; the move to the long
  domain requires it (2026-10-02 record).
- Retired: candidate (ii) in both forms; the spec-literal (i-0) outlet; the `k = 1` Fourier preconditioner as the
  only preconditioner.
- Kept as instruments: the CPU prototype (scripts and saved 16^3 solutions) as a reference for the GPU
  implementation at 16-32.
- Backflow (SF-30, `sigma^2 = 4`) is outside this formulation as decided (item 2 requires `v1 > 0` on the inlet
  face); `(4, 1/16)` stays non-blocking.

## Classification (docs/AGENTS.md)

| item | class |
|---|---|
| (i-0) has no solution; (i-1) well-posed with no gap-separated cluster (R1, R2) | confirmed in runs (CPU prototype); the `k = 1` kernel count also confirmed by derivation (UNDERSTAND §8 instrument) |
| (i-1) 4th order reproduces the Darcy labels at `sigma_Y = 0.25` (all fields) and at 0.5 (`gauss_ch`, `generic3d`) (R3) | confirmed in runs (one realization, `L/ell = 4`, <= 32^3) |
| D-2 outlet condition is satisfied by the Darcy labels; reduces to Neumann for constant head | confirmed by derivation (UNDERSTAND record §2, P2/P3) and in runs (`cons4`) |
| Items 1-7 of the Decision as the formulation of SF-33 | accepted scope |
| GPU realization (Newton-Krylov, preconditioner, production oracle) | proposed architecture |
| Behaviour at `sigma_Y = 1` and above; paper parameters | open question |
| Code of the prototype | confirmed in code (artifact `scripts/`, CPU only; nothing under `src/`) |

## Open questions

- `sigma_Y = 1` at `L/ell = 4`: does (i-1) converge at `ell/h >= 16-32` (64^3-128^3)? SF-33 records the measured
  orders without presupposing them.
- The paper's parameters (`sigma_Y = 2`, `ell = 1/16`, 256^3): resolution adequacy and backflow (SF-30 found
  `v1 <= 0` at `sigma^2 = 4`); inlet labels are undefined where `v1 <= 0` on the inlet face.
- One Gaussian realization only; robustness over realizations is untested.
- The long domain (Dirichlet in `x`, 2048 x 256 x 256): the same formulation is intended to serve it (2026-10-02
  delegated decision), not tested here.
