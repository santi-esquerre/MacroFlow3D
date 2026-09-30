# Equation (14) source pairing: root cause of the SF-21/SF-25 wall, and the paper-faithful pseudo-time reproduction plan

- Status: accepted (owner directive, 2026-09-30)
- Date: 2026-09-30
- Deciders: owner (direction), Claude Code orchestrator (analysis and plan)
- Supersedes the *interpretation* (not the measurements) of
  `docs/experiments/2026-08-15-sf25-terminal-solver-campaign-report.md` §8;
  corrects the equation statement in
  `docs/decisions/2026-07-13-lester-eq14-streamfunction-solver.md` item 4.

## Context

The SF-21..SF-25 campaign established, with prespecified and bitwise-
reproducible V100 runs, that every solver in the accepted stack (adaptive
Picard, Anderson, globalized Newton-Krylov, shifted Newton/LM, implicit Psi-tc,
and a fixed-cap explicit pseudo-time flow) stops at a residual shelf
`r_F ~ 1e-3..4e-4` for `sigma_Y^2 = 1` beyond a critical field amplitude
`a* in (0.500, 0.5125]*Y_unit` (32^3, `ell/h = 8`), that h-refinement moves the
reachability boundary the WRONG way (~30x at 64^3), that the Jacobian is
indefinite at the shelf (`lambda_min^gen = -2.06e-3`), and that every state in
the contested region is physically indistinguishable (`e_v ~ 2.5 %`,
invariance defect ~2 %; "F-SAT"). The campaign report's leading hypothesis
was a discretization-induced spurious-branch pathology (H1) with a
formulation gap vs the reference paper (H2, verification item V1) left open.

On 2026-09-30 the owner asked for an independent validation of that
diagnosis, a re-read of Lester et al. (2023), and a paper-faithful
pseudo-time solver. The re-derivation below identifies a single root cause
that the campaign did not test.

## Finding — the implemented coupled system pairs the sources crosswise

### Derivation (elementary vector calculus, no discretization involved)

With `v = grad(psi1) x grad(psi2)` and isotropic Darcy `v = -k grad(phi)`:

```text
curl(v) = grad(ln k) x v                                  (Darcy, k scalar)
curl(grad psi1 x grad psi2)
    = grad(psi1) lap(psi2) - grad(psi2) lap(psi1) - B,    B = H(psi2) grad(psi1) - H(psi1) grad(psi2)
grad(ln k) x (grad psi1 x grad psi2)
    = grad(psi1) (grad ln k . grad psi2) - grad(psi2) (grad ln k . grad psi1)
```

Define `L_i = lap(psi_i) - grad(ln k) . grad(psi_i)`. Subtracting:

```text
grad(psi1) L_2 - grad(psi2) L_1 = B                       (vector equation)
```

Cross with `grad(psi2)` from the right: `L_2 c = B x grad(psi2)`, with
`c = grad(psi1) x grad(psi2)`. Cross with `grad(psi1)`: `L_1 c = B x grad(psi1)`.
Hence

```text
L_i = ((B x grad psi_i) . c) / |c|^2  =  S_i        (SAME index, i = 1, 2)
```

where `S_i` is exactly the quantity defined below the paper's equation (14).
The paper prints `lap psi1 - grad ln k . grad psi1 = S2` and
`lap psi2 - grad ln k . grad psi2 = S1` (indices crossed relative to its own
definitions of `S_i` and `B`). No sign convention for `B` and no relabeling
`psi1 <-> psi2` can turn the crossed form into the derived one.

### Exact counterexample (discriminates the two forms without any k)

Take `k = k(x1)` periodic (any), `psi2 = x3`, `psi1 = x2 + Phi(x3)` with `Phi`
periodic. Then `v = grad psi1 x grad psi2 = e1` exactly (a Darcy flow of
`k(x1)` with `phi = -int dx1/k`), the pair is stagnation-free, and

```text
B   = -Phi''(x3) e3
S_1 = Phi''(x3),   S_2 = 0
L_1 = Phi''(x3),   L_2 = 0      (also k div((1/k) grad psi_i) in divergence form)
```

The same-index system holds identically; the crossed system requires
`Phi'' = 0`. Note this pair is the gauge recombination `psi1 -> psi1 + Phi(psi2)`
of the trivial pair `(x2, x3)`: a correct equation (14) must be satisfied by
every valid pair of the same `v`, including recombinations.

### Numerical confirmation (spectral derivatives, 32^3, double)

Script run in the session scratchpad (recorded verbatim in the SF-25
bitácora row of 2026-09-30):

| check | same index | crossed (paper as printed = code) |
|---|---:|---:|
| `max|L_1 - S_.|` on the exact pair | `2.2e-14` | `11.8` |
| `max|L_2 - S_.|` on the exact pair | `0.0` | `11.8` |
| identity `curl c = g1 lap2 - g2 lap1 - B`, generic random pair | err `3.0e-12` | — |

(`max|Phi''| = 11.8` for the chosen `Phi`, i.e. the crossed residual is O(1),
the full size of the source.)

### Where the code stands

- `src/physics/streamfunctions/ResidualEvaluator.cu:45`
  (`combine_residual_rhs_kernel`): "Pairing: G1 uses s2, G2 uses s1" — the
  crossed form. `S1`, `S2` themselves are computed exactly as defined
  (`NonlinearSources.cuh`, CPU reference `tests/streamfunctions/reference_operators.cpp`).
- The Jacobian-vector product (SF-22) differentiates the same residual, so it
  inherits the pairing (no second copy to fix there).
- Every existing exact control has `S == 0` (homogeneous `k`, zero
  fluctuations); the SF-09/SF-10 tests compare GPU vs a CPU reference of the
  same printed formula. No test could have detected the pairing.
- Documents stating the crossed form (corrected in this PR): theory note §3A,
  solver overview, dashboard "Locked mathematical and discrete decisions",
  `ARCHITECTURE.md` §4.3, the 2026-07-13 decision (item 4, annotated).

## Why this single defect explains the whole SF-21/SF-25 phenomenology

With `psi_i = affine_i + fluct_i`, `B = grad(d_2 fluct_2 - d_3 fluct_1) + O(fluct^2)`,
and on the first-order Darcy solution `d_2 fluct_2 = d_3 fluct_1`
(both equal `d_2 d_3 phi~ / d_1`), so `S_i = O(fluct^2)`: **the two systems
coincide to first order and differ at second order in the field amplitude.**

| measured fact (campaign report §5) | explanation under the root cause |
|---|---|
| F1: `sigma^2 = 0.25` converges to `1e-6` | second-order mismatch too small to obstruct solvability |
| F2/F6: hard critical amplitude `a*`, eta-reachability cliff | the crossed (non-integrable) system loses solvability at finite amplitude; the shelf is its least-squares distance to solvability (`J^T F ~ 0`, `F != 0`) |
| F3/E4: `J` indefinite at the shelf | signature of a least-squares stationary point of an unsolvable system, not of a solution |
| F7: 64^3 / `ell/h = 16` frontier ~30x WORSE | finer grids approximate the *wrong continuum system* better; coarse truncation was partially masking the inconsistency |
| F10 (F-SAT): `e_v ~ 2.5 %` identical at `r_F = 1e-3` and `1e-6`, "truncation-dominated" | the floor is model error of the crossed system, not truncation; algebraic convergence cannot buy physics |
| F8/F9: explicit flow repelled, cross-family floor `~4e-4` | the flow of the crossed residual has no equilibrium near the shelf either |
| E7 (epsilon-robust), E8 (init-robust) | consistent: the obstruction is in the equation, not in regularization or basin |
| paper reaches `1e-16` at 256^3, `sigma^2 = 4` | consistent with the paper's code solving the derived (same-index) system; the printed equation is a typographical index swap |

Nothing in the campaign's measurements is contradicted; the *methods* were
never shown incapable — they were applied to a system with no solution near
the Darcy streamfunctions. This also means a pseudo-time solver "as in the
paper" on the current residual would fail identically (P2-A' already did).

## Decision

1. **Correct the pairing** (`G_1 <- rhs_1 - eta q S_1`, `G_2 <- rhs_2 - eta q S_2`)
   as its own human-review increment **SF-26**, with the exact-pair test above
   as a Gate 2 contract test (16/32/64 second-order convergence; the crossed
   pairing kept as a mutant that must fail), a gauge-recombination invariance
   test on a converged state, and the UNCHANGED sigma^2 = 1 gates (32^3 smoke,
   64^3 suite) re-run with the accepted implicit stack. Prespecified
   falsifiable prediction: the wall disappears; `e_v` and invariance defects
   drop and converge O(h^2).
2. **Keep the implicit stack** (Picard/Anderson/Newton-Krylov) as the accepted
   solver and cross-check.
3. **Implement the paper-faithful explicit pseudo-time solver** (SF-27) on the
   corrected residual, first to evaluate correctness against the implicit stack,
   later performance: flow `d fluct/d tau = -k .* F` (the paper's scaling: the
   linear part `k div((1/k) grad)` has unit diffusion coefficient, so the explicit
   stability limit is `~h^2/12` independent of `sigma^2`; same zero set as `F`),
   harmonic (`eta = 0`, PCG+MG) initialization = the paper's Krylov initial
   estimate, embedded-pair RK2 error control PLUS a hard Gershgorin stability
   cap (the P2-A servo confound must be impossible by construction), `epsilon`
   default 0 with diagnostic-only degeneracy counters.
4. **Reproduce the paper's reference case** (SF-28): 256^3, `ell = 1/16`,
   `sigma^2 = 4`, nested iteration 64 -> 128 -> 256 with the field generated on
   the finest grid and restricted downward, full Gate 3A metrics and `e_v(h)`.
5. **Pseudo-symplectic tracker on GPU** (SF-29): periodic tricubic B-splines,
   one thread per particle, arclength predictor + 2x2 Newton projection on
   `(psi1, psi2)` (least-norm update `delta = J^T (J J^T)^-1 (-r)`, which does
   not require `v1 > 0` unlike the paper's eq. 38 parametrization), RK45 and
   Stokes-face/Pollock-like references on the same field.
6. **Transverse-macrodispersion validation** (SF-30): Gate 4/5 — `D_T = 0` by
   construction vs the spurious `sqrt(tol)` / `Delta^2` growth of the references.
7. The previous SF-26..SF-30 (heterogeneity completion, grid continuation, GPU
   optimization, V100 benchmark, mixed precision) are superseded: the
   heterogeneity gates are re-imposed verbatim inside the new SF-26; grid
   continuation is absorbed by SF-28; optimization/benchmark/mixed precision are
   deferred until after SF-30 (their specifications remain in git history at
   `9a71963`).
8. SF-25 is closed `done` with its scoped terminal method recorded as falsified
   AND superseded by this root cause; its instrument (`terminal_solver_gpu_cases.cu`)
   and the `streamfunction_terminal_dgate` ctest entry are re-evaluated in SF-26
   (the frozen E2 state they assert on will not exist once the equation is fixed).

## Consequences

- Positive: a two-line source correction with a decisive contract test;
  the entire SF-02..SF-24 infrastructure (operators, MG, Picard, Anderson,
  Newton-Krylov, diagnostics, continuation) is expected to work unchanged
  on the corrected system; the `sigma^2 >= 1` gates become feasible.
- The campaign's diagnostic instrument and hygiene fixes remain valid code;
  their scientific readouts are reinterpreted, not discarded.
- Risk: if SF-26's prediction fails (the wall persists after the fix), the
  campaign's H1/H2 hypotheses regain priority and SF-27's pseudo-time solver
  becomes the paper-parity instrument to discriminate them. No gate value is
  changed by this decision.
- Documentation must state the derived same-index form and flag the paper's
  printed form as an index typo relative to its own definitions
  (`docs/theory/lester-2023-key-claims.md` §3A).

## Classification of claims (docs/AGENTS.md convention)

- Confirmed in code: the crossed pairing at `ResidualEvaluator.cu:45`; absence
  of any exact control with `S != 0`.
- Confirmed by derivation and by an exact counterexample + spectral numerical
  check: the same-index system is the correct one.
- Accepted scope: SF-26..SF-30 as re-sequenced in the dashboard.
- Proposed architecture: `PseudoTimeSolver`, `streamline_tracker` module.
- Open question (falsifiable by SF-26): that the pairing fully explains the
  wall (prediction: `sigma^2 = 1` 32^3 smoke reaches `lambda = 1` with the
  unchanged stack).

## References

- Lester, Dentz, Singh, Bandopadhyay (2023), WRR 59, e2022WR033059 — eqs.
  (13)-(14), §5.1 (FD, periodic, homogeneous Krylov init, explicit variable-
  step pseudo-time to 1e-16, 256^3, `ell = 1/16`, `sigma^2 = 4`), §5.4 (eqs.
  37-38, pseudo-symplectic tracking), eq. (32) (Stokes face velocities).
- Zijl (1986), J. Hydrol. 85 — original dual-streamfunction system.
- `docs/experiments/2026-08-15-sf25-terminal-solver-campaign-report.md` — the
  measurements this decision reinterprets.
