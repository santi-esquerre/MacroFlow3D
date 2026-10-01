# SF-26 — heterogeneity gates on the corrected (same-index) equation (14): outcome and the eta = 1 residual floor

- Date: 2026-10-01
- Status: complete (increment evidence; scientific interpretation for human review)
- Theory: `docs/theory/lester-2023-key-claims.md` §3A; decision
  `docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md`

## Question

With the coupled residual corrected to the derived same-index pairing
(`A u_i = div_h(q gbar_i) - eta q S_i`), do the UNCHANGED sigma_Y^2 >= 1
heterogeneity gates (SF-21/SF-25 fixtures, Newton enabled per SF-25 E1) pass
with the unchanged implicit stack, and what is the Gate 3A picture?

## Hypothesis (prespecified in the SF-26 spec before any run)

The SF-21/SF-25 wall was the least-squares shelf of the crossed (unsolvable)
system; prediction: the 32^3 sigma^2 = 1 smoke reaches lambda = 1 with every
accepted stage `r_F <= 1e-6`, the 64^3 quartet reaches lambda = 1, `e_v`
drops from ~2.5 % and decreases ~4x from 32^3 to 64^3. Recorded caveat
(bitácora 2026-09-30T22:40Z, before the runs): the corrected Jacobian carries a
near-null gauge cluster at eta = 1, so the Newton/Krylov phase may
budget-exhaust there.

## Build / environment

- Integrated head `58898bf` (delivery branch `science/lester-sf26-source-pairing-correction`;
  source base `b86602b` + harness `abd873c`).
- Remote `v100` (2x Tesla V100-PCIE-32GB), preset `v100-release`; mirror
  byte-identical to the local tree before the base-reference run. All heavy
  runs were detached `scripts/remote run` jobs, one at a time.
- Independent CPU probes (numpy/scipy, same stencils as the code): scripts and
  raw outputs under `.claude/orchestration/SF-26-source-pairing-correction/`
  (runtime evidence; summarized here).

## Config(s)

Unchanged fixtures, no tuning: `heterogeneity_smoke_sigma025` /
`heterogeneity_smoke_sigma1` (32^3, dx = 1, seed 12345, ell = 8, epsilon 1e-2
degenerate leg, Anderson depth 5/start 5/limit 1e12, Newton enabled);
`anderson_stall_fixture_{a,b}`; `newton_difficult_case`; the SF-25 instrument
(`terminal_dgate_diagnostic`: Newton-disabled freeze; `terminal_resolution_probe`).

## Commands

```bash
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote run sf26-ctest-full -- "ctest --test-dir build/v100-release --output-on-failure"   # 66700 s
scripts/remote run sf26-bytecompare -- "<three default configs into ~/sf26_new_refs; diff vs ~/sf26_base_refs>"
```
Base references were generated on the untouched mirror before the sync (job
`sf26-base-refs`). The spec's `sf26-smoke32` pipeline run was not repeated
(the ctest heavy case runs the identical fixture and prints the per-stage
Gate 3A metrics); the 64^3 quartet was not run (see Result).

## Outputs inspected

`~/sf26_ctest_full.log` (failed-test output), `build/v100-release/Testing/Temporary/LastTest.log`
(passed-test output), `~/sf26_new_refs` vs `~/sf26_base_refs`.

### Full suite: 17/21 passed, 4 failed

Passed: all cheap-tier entries including the new `streamfunction_exact_pair`
(pair A at roundoff, general pair orders 1.989/1.997 (r_F) and 1.965/1.991
(Linf) on 32->64->128, crossed mutant r_F 4.5-10 non-decreasing, analytic
recombination ratio 1.70 with order 1.988/1.997), `streamfunction_gmres` (C02
contracts: GPU restarted residual 2.905e-3 vs host textbook restarted 2.931e-3,
host full-recurrence 8.8e-11; preconditioner efficacy 0.35 / 0.10 at fixed
budget), `picard_adaptive` (re-baselined 131), `jvp`, `newton`, the terminal
instrument (always-pass evidence), `periodic_gaussian`, `affine_periodic_flow`.

Byte-compare: `config_pspta_small`, `config_streamfunctions_homogeneous`,
`config_streamfunctions_continuation` identical to the base references
(manifests identical modulo timestamps).

Failed (all exactly at eta = 1):

| entry | outcome |
|---|---|
| `heterogeneity_smoke_sigma025` | `lambda_floor_exhausted` at lambda = 0.0125 (38/66 accepted, 60 rescue stages, 18996 s) |
| `heterogeneity_smoke_sigma1` | `lambda_floor_exhausted` at lambda = 0 (37/65 accepted, 60 rescue stages, 19056 s) |
| `anderson_stall_fixture_a` (sigma^2 = 1, lambda* = 0.1125) | control Picard stagnated 455 its r_F 2.51e-4; Anderson stagnated 175 its r_F 2.48e-4 (crossed system, Aug-14: Anderson converged in 64 its) |
| `anderson_stall_fixture_b` (sigma^2 = 0.25, lambda* = 0.386) | control 7.51e-4; Anderson 7.41e-4 (crossed: converged in 88 its) |
| `newton_difficult_case` | Newton 50 accepted steps, r_F 1.77e-4 -> 1.73e-4, 5351 Jv, not converged |
| `gauge_recombination_heavy` | precondition (sigma^2 = 0.25 continuation reaches lambda = 1) not met |

Per-stage anatomy of the smokes (both variances identical in kind): every
eta < 1 rescue stage converges trivially (r_F 1e-10..8e-8, Newton 2 steps,
41-72 Jv); every eta = 1 stage: `r_F = 1.50e-6` (1.5x the tolerance),
`accepted = false`, `picard_iterations = 0`, `newton_act = 1`,
`newton_acc = 50`, `newton_jv ~ 5500` (every inner GMRES(10)/100
budget-exhausted). Physics at those states: `e_v = 1.24e-4`, invariance
3.9e-5 / 5.0e-5, `e_div = 3.4e-6`, `|c|` p0.1 % = 1.018, zero degeneracy.

Newton-disabled control (`terminal_dgate_diagnostic` E2 freeze, Picard/Anderson
only): the sigma^2 = 1 continuation also dies at lambda = 0; the direct
lambda = 0.5125 stage stagnates at r_F 5.4e-3 (crossed plateau was 1.12e-3).
`terminal_resolution_probe` (64^3 direct solves at amplitude 0.5125): r_F
1.22e-3 (ell/h = 16), 4.69e-3, 6.88e-4 — 7-13x below the crossed-system
floors, not converged.

### Independent numpy probes (same stencils: harmonic-mean faces, centered gradients/Hessians, mean-zero projection)

1. Smooth analytic field (general exact pair, 12^3, eta = 1, from zero):
   same-index — damped Picard 1196 its to 1e-10, Anderson 89 its, full
   (min-norm) Newton quadratic to 6e-16, explicit pseudo-time to 1e-10 at
   tau ~ 0.9; crossed — Picard 159 its (isolated solution), pseudo-time
   DIVERGES (NaN at tau 0.2) — the SF-25 "repelled flow" was the crossed
   system's instability.
2. Random Gaussian fields (ell/h = 8; 24^3 and 32^3; sigma^2 = 1,
   lambda = 0.1125): same-index Picard/Anderson/pseudo-time all stall at a
   residual floor ~1e-4 (reproducing `anderson_stall_a`); crossed Anderson
   converges (95 / 188 its); crossed flow diverges.
3. Epsilon sweep (24^3, same field): floor 1.0e-4 (eps 1e-2), 7.7e-5 (3e-3),
   7.3e-5 (1e-3), 8.2e-5 (0) — independent of the regularization.
4. Dense full Newton from the Anderson floor (24^3, 2N = 27648, exact linear
   solve, residual 6e-10): smallest relative singular values 7e-6..6e-7
   (68 modes < 1e-3, 18 < 1e-4); full Newton step |d| ~ 0.37 accepted only at
   alpha = 1/32; r_F 7.41e-5 -> 7.03e-5 in 4 steps. No reachable exact zero.
5. Resolution per correlation length (eps = 0, Anderson m = 8, 2000 its,
   two seeds): ell/h = 8 (32^3): 1.0e-4 / 8.3e-5; ell/h = 16 (32^3):
   4.7e-5 / 6.2e-5; ell/h = 16 (48^3): 3.5e-5. The floor decreases with
   resolution, but only ~x2 per doubling of ell/h.
6. Explicit pseudo-time on the rough field at the linear-stability dt:
   decays to 1.2e-4 at tau 2.5, then destabilizes (state-dependent nonlinear
   stiffness) — the SF-27 spec's Gershgorin cap on the LINEAR part is not a
   sufficient stability guarantee.

## Result

- The pairing correction is correct and demonstrated (exact pairs at roundoff,
  general pair second order, crossed mutant O(1), recombination invariance),
  and it removes the crossed system's pathologies (unstable flow, unsolvable
  shelf).
- The prespecified gate prediction is FALSIFIED: the unchanged sigma^2 >= 1
  gates (and the sigma^2 = 0.25 smoke that passed on the crossed system) fail,
  and the failure has a new, fully characterized signature: on random Gaussian
  fields the corrected DISCRETE system has a residual floor at eta = 1
  (~1e-4 at lambda = 0.11, sigma^2 = 1, ell/h = 8) that is independent of the
  solver (Picard, Anderson, restarted/full-recurrence Newton, explicit flow)
  and of epsilon, and decreases only weakly with ell/h. Its origin is the
  exact gauge invariance of the correct equation: the discrete Jacobian has a
  large near-null cluster, the discrete equations are effectively inconsistent
  at the level of that cluster, and the algebraic tolerance 1e-6 is unreachable
  on these grids. Smooth analytic fields have exact discrete zeros.
- Gate 3A physics at the floor states is clean and better than on the crossed
  system (e.g. e_v 1.2e-4 at lambda = 0.0125; 64^3 direct-solve floors 7-13x
  lower), consistent with "the floor is algebraic, the physics converges".
- No gate value, fixture, seed, or tolerance was changed.

## Caveats

- The 64^3 quartet was not run: with the fixed eps = 1e-2 leg and the 1e-6
  stage tolerance it would reproduce the lambda-floor failure (12 h per
  variance). Deferred to the owner's decision on stopping criteria.
- numpy fields use a spectral Gaussian generator (not SF-18) and
  v_rms = 1; statistics, not realizations, match the fixtures.
- Anderson floors are noisy (non-monotone); the dense-Newton probe is the
  authoritative "no reachable zero" evidence at 24^3.

## Next step

Owner decision (human review of this PR): (a) truncation-aware acceptance for
the implicit stack (stop at the measured floor; judge by Gate 3A physics and
h-convergence), (b) SF-27 as planned but with error-controlled stepping that
also bounds the nonlinear stiffness, and realistic step counts, (c) a
gauge-fixing/bordered formulation that removes the manifold (SF-25 H4's
research direction, now justified), or (d) higher ell/h (the paper's 16) with
the floor tracked under refinement. See the proposed decision record
`docs/decisions/2026-10-01-eta1-residual-floor-gauge-degeneracy.md`.
