# SF-26 — heterogeneity gates on the corrected (same-index) equation (14): outcome and the eta = 1 residual floor

- Date: 2026-10-01
- Status: complete (SF-26 closed by owner directive 2026-10-02; open decisions in the decision record)
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
6. Same continuum field (band-limited at 24^3, exact spectral upsampling) at
   24^3 (ell/h = 8) and 48^3 (ell/h = 16), eps = 1e-2, Anderson m = 8, 2000 its:
   floor 7.62e-5 -> 1.31e-5, i.e. a factor 5.8 for h -> h/2 (~h^2.5): the floor
   is a discretization (truncation-structure) effect that vanishes under
   refinement at roughly second order.
7. Explicit pseudo-time on the rough field at the linear-stability dt:
   decays to 1.2e-4 at tau 2.5, then destabilizes (state-dependent nonlinear
   stiffness) — the SF-27 spec's Gershgorin cap on the LINEAR part is not a
   sufficient stability guarantee.

### Post-publication probes (2026-10-01 / 10-02; CPU jobs on the V100 host, outside the mirror)

8. FD (the code's scheme) vs a pseudo-spectral residual on the same field, same
   Anderson (m = 8, w = 0.5, Laplacian preconditioner), eps = 0
   (`probe_spectral.py`, job `sf26-numpy-spectral`). Minimum `r_F` reached:

   | grid, ell/h | lambda*sigma | FD (code) | pseudo-spectral |
   |---|---|---|---|
   | 16^3, 4 | 0.11 | 2.8e-4 | 8.0e-7 |
   | 16^3, 4 | 0.5 | 1.3e-2 | 3.7e-4 |
   | 32^3, 8 | 0.11 | 1.1e-4 | 4.6e-11 |
   | 32^3, 8 | 0.5 | 3.6e-3 | 2.5e-5 |
   | 32^3, 8 | 1.0 | 1.5e-2 (not converging) | diverges from a zero start |

   The FD dense-Newton leg (full rank, min relative singular value 1e-6) again
   makes no progress (step norm 1.5-30, accepted only at alpha <= 1/64).
9. Order / resolution / amplitude ladder with lambda-continuation
   (`probe_order.py`, job `sf26-numpy-order`): one continuum field
   (sigma^2 = 1, ell = L/4, seed 7) at 32^3 / 64^3 / 96^3; residual schemes
   `fd2` (the code's: divergence form, harmonic-mean faces), `fd2c` (2nd order,
   non-divergence form using `grad(lambda Y)`; same sources, same order), `fd4`
   (4th-order non-divergence), `sp` (pseudo-spectral); lambda = 0.1 .. 1.0 with
   warm starts; Anderson m = 8, w = 0.5, Laplacian preconditioner, at most 800
   iterations per stage with a stagnation exit. Minimum `r_F` reached:

   | lambda*sigma | fd2 32^3 | fd2 64^3 | fd2c 32^3 | fd2c 64^3 | fd4 32^3 | fd4 64^3 | fd4 96^3 | sp 32^3 | sp 64^3 | sp 96^3 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | 0.1 | 8.4e-5 | 2.6e-5 | 3.6e-5 | 1.0e-5 | 2.8e-6 | 6.6e-6 | 1.4e-6 | 2.1e-11 | 9.7e-13 | 4.3e-12 |
   | 0.3 | 8.9e-4 | 3.6e-4 | 3.1e-4 | 8.3e-5 | 2.1e-5 | 1.6e-5 | 3.4e-6 | 6.3e-7 | 6.5e-8 | 1.8e-6 |
   | 0.5 | 3.3e-3 | 7.1e-4 | 5.7e-4 | 1.5e-4 | 9.2e-5 | 2.2e-5 | 5.9e-6 | 3.2e-5 | 2.9e-4 | 2.5e-4 |
   | 0.8 | 1.0e-2 | 1.9e-3 | 1.4e-5 | 1.6e-4 | 2.1e-4 | 3.9e-5 | 1.4e-5 | 5.0e-3 | 8.9e-3 | 1.9e-2 |
   | 1.0 | 1.9e-2 | 3.5e-3 | 1.0e-4 | 2.3e-4 | 2.2e-4 | 6.8e-5 | 3.4e-5 | 3.5e-2 | 8.4e-2 | 1.4e-1 |

   `min |grad psi1 x grad psi2|` falls from 0.79 (lambda*sigma = 0.1) to 0.05
   (1.0) identically for every scheme and grid: physical low-speed zones, no
   degeneracy. All ten stage tables are in
   `artifacts/2026-10-01-sf26-probes/raw/order_probe_v100.txt`.

### Audit of the post-publication probes (what the numbers do and do not mean)

- The stagnation exit of `probe_order.py` (no 1 % improvement in 150
  iterations) was too aggressive: 27 of 100 stages stopped at iteration 201
  (the minimum was reached in the first ~50 iterations) and 8 exhausted the
  800-iteration budget while still decreasing. Stopped-at-201 / budget-hit per
  run: fd2 32^3 5/0, fd2 64^3 3/0, fd2c 32^3 2/2, fd2c 64^3 0/0, fd4 32^3 0/1,
  fd4 64^3 1/0, fd4 96^3 0/0, sp 32^3 3/1, sp 64^3 6/2, sp 96^3 7/2. Many
  entries of the table are iteration plateaus or upper bounds, not
  discretization floors.
- Signs that several numbers are iteration-limited: fd4 at lambda*sigma = 0.1
  gives 2.8e-6 / 6.6e-6 / 1.4e-6 at 32^3 / 64^3 / 96^3 (a 4th-order truncation
  floor would fall 16x per doubling); fd2c at 32^3 gives 3.6e-5 at
  lambda*sigma = 0.1 but 2.5e-7 at 0.9 (800 iterations, still decreasing).
- The pseudo-spectral entries at lambda*sigma >= 0.5 are solver failures, not
  floors: they get WORSE under refinement (3.5e-2 / 8.4e-2 / 1.4e-1 at
  lambda*sigma = 1) and stop at iteration 201. Unverified reading: the source
  terms contain second derivatives, so a Laplacian preconditioner does not
  control the highest modes, which finite differences attenuate and the
  spectral derivative does not.
- The dense-Newton leg of `probe_spectral.py` on the pseudo-spectral system is
  NOT valid evidence: that Jacobian is exactly singular (min relative singular
  value 1e-13, 125 modes below 1e-5) and was solved with `rcond = 1e-13`.
- `sigma^2 = 0.25, lambda = 1` and `sigma^2 = 1, lambda = 0.5` are the same
  problem (lambda*sigma = 0.5), not two independent data points.
- One realization (seed 7); no probe computes a physics metric (`e_v`,
  invariance); simple Anderson without the code's safeguards.

## Result

### Conclusions that the evidence supports

- **C1 — The pairing correction is right.** Derivation, exact pairs at roundoff
  (2e-15..9e-15 on 16/32/64), second-order convergence on the non-separable
  exact pair (1.989 / 1.997 in `r_F`, 1.965 / 1.991 in `Linf` on 32->64->128),
  crossed mutant O(1) and non-decreasing, recombination ratio 1.70 with order
  1.988 / 1.997. Default pipelines are byte-identical to the base.
- **C2 — The prespecified gate prediction is falsified.** With the unchanged
  stack and fixtures, both 32^3 smokes exhaust the lambda floor (sigma^2 = 0.25
  at lambda = 0.0125, sigma^2 = 1 at lambda = 0); the sigma^2 = 0.25 smoke had
  passed on the crossed system. Every eta < 1 stage converges; every eta = 1
  stage stops at `r_F = 1.50e-6`. Newton (SF-25 E1 wiring) seizes those stages
  and exhausts its GMRES budget, but Newton is not the root cause: the
  Newton-disabled continuation fails as well. No gate value, fixture, seed, or
  tolerance was changed.
- **C3 — The code's discrete system has a residual floor at eta = 1 on random
  Gaussian fields.** It is the same for Picard, Anderson, restarted
  Newton-GMRES, a dense Newton with an exact linear solve, and the explicit
  flow; it does not depend on epsilon (1.0e-4 at 1e-2, 8.2e-5 at 0); it falls
  under refinement (x5.8 for the same field 24^3 -> 48^3; x2.5-5 from 32^3 to
  64^3 along the ladder) and grows with amplitude (about x220 from
  lambda*sigma = 0.1 to 1.0 at 32^3). At sigma^2 = 1, lambda = 1: 1.9e-2 at
  32^3 and 3.5e-3 at 64^3 (numpy probe).
- **C4 — The corrected equation is solvable when discretized accurately.** The
  pseudo-spectral residual reaches 2e-11 / 1e-12 / 4e-12 at lambda*sigma = 0.1
  (32^3 / 64^3 / 96^3) and 6.5e-8 at 0.3 (64^3) with a plain Anderson
  iteration; with the code's scheme a smooth analytic field converges to 6e-16
  with a full Newton.
- **C5 — Most of the code scheme's floor comes from how the LINEAR operator is
  discretized.** `fd2c` keeps the sources and the order of the code's scheme
  and only replaces the divergence form with harmonic-mean faces of `K` by the
  non-divergence form with `grad(lambda Y)`; its floor is 2.6x lower at
  lambda*sigma = 0.1 and 15x lower at lambda*sigma = 1 (64^3). The measured
  difference is established; why (harmonic means of a lognormal `K` versus the
  smooth `Y`) is an interpretation.
- **C6 — The corrected Jacobian has a near-null cluster at eta = 1** (relative
  singular values 1e-4..1e-6, dozens of modes, sharpening with refinement; the
  crossed system had none), and restarted GMRES(10) stagnates on it while a
  full-recurrence GMRES converges (in-repo test `gmres_dense_lu_oracle`).
- **C7 — The crossed system's explicit flow is unstable** (NaN within
  tau ~ 0.05-0.2 in every probe), which accounts for the "repelled flow" of the
  SF-25 campaign; on the corrected system the flow converges on a smooth field.
- **C8 — No degeneracy.** `|grad psi1 x grad psi2|` stays >= 0.05 up to
  lambda*sigma = 1 in the probes, and the V100 runs report zero unexplained
  degenerate cells.

### Not established

- **N1** Whether any discretization reaches `r_F <= 1e-6` at sigma^2 = 1,
  lambda = 1. The fd4 entries (6.8e-5 at 64^3, 3.4e-5 at 96^3) are upper bounds
  that mix floor and iteration stagnation; the pseudo-spectral iteration failed
  at high amplitude.
- **N2** The physics at the floor. `e_v`, invariance and `e_div` were measured
  only by the V100 smokes at tiny amplitude (lambda <= 0.025: `e_v` 1.2e-4,
  invariance 4e-5 / 5e-5, `e_div` 3.4e-6). Nothing was measured at sigma^2 >= 1
  full amplitude. An earlier statement in this note and in the SF-26 bitácora
  ("Gate 3A physics at the floor is clean and better than on the crossed
  system") is restricted to that tiny-amplitude regime; the "7-13x lower"
  figure refers to `r_F` of the 64^3 direct solves, not to a physics metric.
- **N3** The mechanism. "The gauge degeneracy makes the discrete equations
  inconsistent at the level of the near-null cluster" is consistent with every
  measurement but is not proven; C5 shows that plain truncation of the linear
  operator carries most of the effect.
- **N4** Whether a divergence-form variant with log-space (geometric-mean) face
  coefficients recovers the gain of `fd2c` while keeping `A` symmetric and
  multigrid-compatible. Not tested.
- **N5** Whether an error-controlled pseudo-time integrator converges on rough
  fields. A fixed-step Euler flow decayed to 1.2e-4 at tau = 2.5 and then
  destabilized (lambda*sigma = 0.11, 24^3).
- **N6** How the paper reaches a "finite difference residual 1e-16" at 256^3,
  sigma^2 = 4. Its spatial scheme and residual definition are not specified.
- **N7** Robustness across realizations: the order and spectral probes use one
  seed, and the numpy generator is not SF-18.

## Caveats

- The 64^3 quartet was not run: with the fixed eps = 1e-2 leg and the 1e-6
  stage tolerance it would reproduce the lambda-floor failure (12 h per
  variance).
- The spec's `sf26-smoke32` pipeline run was not repeated (the ctest heavy case
  runs the identical fixture).
- The V100 suite stays at 17/21 on this branch; the four red entries are the
  recorded outcome, not regressions to fix silently. `anderson_stall` and
  `newton_difficult` were NOT re-baselined: their fixtures assume a stall that
  Anderson/Newton then cure, and on the corrected system both arms stall, so
  there is no meaningful new baseline until the open decisions are taken.
- numpy probes: spectral Gaussian generator (not SF-18), `v_rms = 1`, no Darcy
  solve, simple Anderson; see "Audit" above for the instrument limitations.
- The dense-Newton probe recorded 4 of its 6 planned steps.
- `gmres_probe.py` was not preserved (see the artifacts README).

## Artifacts

`docs/experiments/artifacts/2026-10-01-sf26-probes/`: every probe script, every
raw output, and a grep extract of the V100 full-suite logs.

## Next step

None inside SF-26: the increment is closed by owner directive (2026-10-02) with
the gates recorded unmet. The decisions that remain open are listed as D1-D7
in `docs/decisions/2026-10-01-eta1-residual-floor-gauge-degeneracy.md`.
