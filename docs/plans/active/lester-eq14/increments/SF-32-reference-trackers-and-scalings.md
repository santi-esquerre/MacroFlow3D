# SF-32 — Face-flux reference trackers and the paper's scalings

- State: `active`
- Goal: `Reproducir las figuras 3 y 4 de Lester 2023, mostrando la dispersión transversal espuria de los trackers convencionales frente al pseudo-simpléctico en un campo de invariantes exactos.`
- Depends on: `SF-31`
- Unlocks: `none`
- Branch: `science/lester-sf32-reference-trackers-scalings`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 4`
- Human review: `required`
- Owner: `Claude Fable 5.1 orchestrator session (2026-10-06)`
- Started: `2026-10-06T14:59Z on master=37bfb25`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reproduce Lester 2023 figures 3-4 (spurious transverse spreading of conventional trackers vs the pseudo-symplectic one) on a field whose invariants are exact.

Context: `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md` and `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`.

## Preconditions

- `SF-31` is `done` on the default branch.

## In scope

- `StokesFaceVelocity`: face fluxes from line integrals of `psi1 grad psi2` around faces (Lester 2023 eqs. 32-33), exactly divergence-free per cell.
- A Pollock-type cellwise-linear tracker in the same module.
- Streamfunctions of the Lester (2021) field obtained with the frozen stack; if it does not converge, reduce the amplitude, then fall back to an analytic pair, never touching the solver.
- Diagnostics: `delta psi_i`, `delta x2, delta x3` after one period vs the pseudo-symplectic trajectory (eqs. 34-35).

## Out of scope

- Runner wiring, ensembles, macrodispersion interpretation.

## Files and symbols

- `src/physics/particles/streamline_tracker/*` (extended)
- New tests under `tests/` and an experiment note `docs/experiments/<date>-sf32-spurious-spreading.md`

## Implementation specification

1. Implement `StokesFaceVelocity` and the Pollock-type tracker.
2. Obtain the invariants (frozen stack, reduced amplitude, or analytic pair) and record which route was used.
3. Run grid and tolerance ladders as detached V100 jobs and fit the exponents.
4. Record exponents outside the bands without tuning.

## Expected numerical effect

New capability only; no existing result changes.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R tracker
scripts/remote sync
scripts/remote run sf32-scalings -- "<scaling ladders command>"
scripts/remote wait sf32-scalings
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- Pollock variance proportional to `Delta^p` with `p = 2 +- 0.3` over >= 3 grids.
- RK with `p = 0.5 +- 0.15` over >= 4 tolerances.
- Pseudo-symplectic drift <= 10x the Newton tolerance.
- An exponent outside the band is recorded, not tuned.

## Regression surface

- None existing; consumes SF-31 and the frozen stack.

## Failure and rollback policy

- Out-of-band exponents are reported as findings; the increment stays active until the human review decides.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Implementation matches the scope and contains no unrelated changes.
- [ ] Targeted validation passes and its evidence is recorded.
- [ ] Required regression tests pass.
- [ ] Scientific or engineering findings are appended to the bitácora.
- [ ] Required human review is recorded.
- [ ] PR and commit identifiers are recorded.
- [ ] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
<!-- completion-checklist:end -->

## Advancement rule

No increment is unlocked directly; its outcome feeds the later phases described in the dashboard.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-02T00:00Z | not started | Specification created by the 2026-10-02 re-sequencing (decision record `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`). | Replaces the cancelled SF-27..SF-30 specifications (git history at `4670fb5`). | Activate only when the checker reports it READY. |
| 2026-10-06T14:59Z | activation on `master=37bfb25` (PR #49 merged; checker OK ready=SF-32 SF-33, nonterminal=none); delivery branch `science/lester-sf32-reference-trackers-scalings` | UNDERSTAND: SF-31 `done` on the default branch; no other increment nonterminal (SF-33 stays READY for a concurrent session; remote isolated by `--increment SF-32`). Human-review increment (new tracker consuming `psi1`/`psi2`, scientific comparison with the paper). Scientific-rigor skill invoked (categories B and C; interpretation bounded: numerical spreading on the paper's surrogate, nothing about Darcy or `alpha_T`, O3). Source check (Lester 2023 §5.1-5.4, eqs. 13, 31-38, figs. 3-4): face velocities by the Stokes line integral `Delta^-2 oint psi1 grad psi2 . dl` (eq. 32), exactly divergence-free (eq. 33); coarse grids by averaging face velocities; errors after traversing `Omega` are the one-period displacements (eq. 34) because the surrogate's streamlines close, and the label errors (eq. 35); Pollock variances `~ (Delta/Delta0)^2`, RK variances `~ sqrt(tol)` with an unspecified 4th-order adaptive pair; the paper does not run its own pseudo-symplectic method in the advective case (`delta = 0` trivially). Numerical contract fixed before any code (orchestration record `understanding.md` §3): one edge integral per grid edge with composite 4-point Gauss-Legendre on the spline knot intervals (exact for the degree-6 integrand, so the fluxes are exact surface integrals of the spline velocity and the only Pollock error is the RT0 interpolation); Pollock semi-analytical exits with the exit coordinate set exactly, stagnation status 15, no fallback; return map = first return of unwrapped `x1` to `x1_0 + 1` for the three trackers (Pollock lands on the face; pseudo-symplectic and RK land by bisection on the last panel / chunk from a saved state, SF-31 cores reused unmodified), `N_p = 8192` hash seeds on `x1 = 0` shared by all runs. Label routes (D-1): primary = frozen stack on `lester2021`, `eps = 0.25`, at 256^3 (paper grid) and 128^3, final accepted state used as the pair whatever the exit reason (`r_F`, `e_v`, `min abs(c)` recorded; a stack state at `r_F ~ 1e-6` is accepted as exact invariants by definition of the surrogate — deviation from the letter of the spec, for the reviewer); control = analytic SF-31 pair G at amplitude 0.05 splined at 128^3. Ladders: Pollock `Delta/Delta0 = 1, 2, 4, 8`; RK `tol = 1e-4 .. 1e-8` (paper) plus `1e-9, 1e-10`; pseudo-symplectic `tol_psi = 1e-8, 1e-10, 1e-12`. | Pre-registered readings (T1-T4) and predictions (P1-P6) in the record. P2 predicts an RK variance exponent in `[0.9, 1.7]`, OUTSIDE the paper's band `[0.35, 0.65]`: on the C^2 B-spline the velocity is C^1, a knot crossing costs `O(h^3)` and `h ~ tol^(1/5)` gives `std ~ tol^0.6` (SF-31 measured 0.55-0.82); recorded as a finding if observed, nothing tuned (spec policy). P1: Pollock variance slope near 2 (coherent accumulation of the zero-mean `O(Delta)` RT0 error) with slope 3 (random-walk) as the admissible alternative. DAG: N0 Stokes face fluxes, N2a return-map instrument (pseudo-symplectic + RK + label routes) and N4 analysis script in parallel -> N1 Pollock tracker and N3a face-flux tests -> N2b Pollock in the instrument and N3b Pollock tests -> one integrator -> orchestrator-owned detached V100 jobs on `--increment SF-32` -> N5 experiment note -> FINAL_AUDIT. | Commit activation; start the base build on the SF-32 mirror; launch N0, N2a, N4. |
