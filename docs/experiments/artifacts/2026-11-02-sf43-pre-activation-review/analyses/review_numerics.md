I tried to refute every numerics and verification claim in the draft. Two claims fail outright and should be fixed before the owner consultation (B1, B2). Four more need correction (M1 to M4), and there are several minor fixes. Every fix is text, pre-registration rules or a pilot of at most about 0.5 GPU-h. No file was written; all computations ran locally on CPU (`python3 -B`), and the three review scripts reproduce their quoted numbers.

Paths used below:
- `R` = `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/2026-11-02-sf43-pre-activation-review.md`
- `D` = `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/decisions/2026-11-02-sf43-pre-activation-decisions.md`
- `N42` = `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/2026-10-30-sf42-confirmatory-tooling.md`
- `G4` = `/home/sesquerre/Projects/MacroFlow3D/.claude/orchestration/SF-43/phase1/agents/gate4_power.md`

## Findings

**B1 — BLOCKING. The first-order benchmark does not apply to SF-43's boxes.**
- **Statements attacked:**
  - `R`:81-90: `alpha_T^FO = 7.99e-5 sigma^2 lambda` and "a positive sign alone does not discriminate"; the 3.57 and 6.92 decay ratios.
  - `R`:235, the "finite distance" row.
  - `R`:37 and M1 (`R`:422): "a sign null that linear theory already rejects".
  - `D` H3 (54-63) and H17; the classification "first-order values: confirmed" (`D`:219).
- **Evidence:**
  - The SF-18 field is periodic on the box in all three directions: `PeriodicGaussianField.cuh`:28 (`L_d = N_d*h_d`) and `DirichletLongBoxStage.cu`:402,436. The production box is 104 `lambda` long; the distance arms are 208 `lambda`.
  - I summed the first-order displacement variance over the generator's exact discrete modes. The code matches the continuous formula at small `d` (`X_22(1)` = 0.0546 vs 0.0547). The window slope converges to 7.99e-5 as the width grows: 1.1e-6, 1.6e-5 and 5.2e-5 at `W` = 24, 48, 96.
  - Results, per-realization centred, in units of `sigma^2 lambda`:

    | box | window [25, 94] | halves [25, 59] / [60, 94] | [94, 194] |
    |---|---|---|---|
    | `L1` = 104, `W` = 24 | -3.3e-5 | +3.5e-6 / -1.19e-4 | – |
    | `L1` = 104, `W` = 12 | -3.9e-6 | – | – |
    | `L1` = 208, `W` = 24 | +1.0e-6 | – | -6.8e-6 |

  - Mechanism, from the computation:
    - The finite width removes the small transverse wavenumbers, which kills the `1/d^2` tail.
    - The `x`-periodicity makes `X_22(d)` symmetric about `L1/2`. In the production box it plateaus at 0.2945 by `d` = 40 and falls back to 0.281 at `d` = 94.
  - So in SF-43's geometry linear theory predicts a negative window value, and G5.4's null at 0 already separates linear theory from a positive nonlinear `alpha_T`.
  - Smaller points:
    - Per-realization centring makes the 104-`lambda` value larger (-3.34e-5 vs -3.50e-5), not smaller as the draft says.
    - On SF-43's integer planes the unbounded halves ratio is 3.64, not 3.57 (the script uses half-`lambda` planes split at 59.5).
- **Correction:**
  - H3 option (a) stays valid as a stricter, conservative null. Its stated reason must go, and so must the 3.57, 6.92 and "75/155/205 times" benchmarks.
  - Report box-specific first-order values per arm.
  - Add "generator periodic in `x` with period = box length (104 vs 208 `lambda`)" to the domain-assumptions table in `R` 3.b.
  - Pilot data show box length does not move the means (see Survived).

**B2 — BLOCKING. In the displacement basis the order branch is the operative rule at 2.25, so the asymptotic-range question is not moot.**
- **Statements attacked:**
  - `R`:242-245: "order branch in 0 % ... at 2.25 and 4"; "production finest ... mostly moot"; "the operative bound ... does not use the asymptotic-range assumption".
  - M2 (`R`:424).
  - `R`:175: "the pre-registration closes [the four gaps]".
  - `R`:179: the text says production is finest only at `sigma^2 = 4`, but H6 makes it finest at 2.25 too.
- **Evidence:**
  - `G4`:63: displacement-basis level-difference SDs were "assumed equal to the position-basis values". The noise-floor result (`G4`:8, C5) is position basis.
  - J2's particle variance is 8.7 times smaller at 2.25 and 4.4 times smaller at 4 in the displacement basis (`N42`:1199-1212). The draft's own caveat (`R`:465-466) says 4 to 9 times.
  - I ran `mcstat.LadderBound` (Student method) with the review's own scenario S1 level errors and "nf" noise model (`G4`:37-41), at tail 7.8e-4.

    | case | position-basis noise | displacement-basis noise |
    |---|---|---|
    | 2.25, ladder 4/8/16 | order branch 0 % (n = 100) and 1 % (n = 150); `E_h` ≈ 0.18 of the mean | order branch 100 %; `p_L` ≈ 1.3 to 1.4; `E_h` ≈ 0.01 of the mean |
    | 4, ladder 8/16/32 | order branch 0 % | order branch 12 % (n = 100) and 48 % (n = 150) |

  - Sensitivity at 2.25 for variance ratios 1 / 2 / 4 / 8.7: order branch 0 / 13 / 79 / 100 %.
  - Under a pre-asymptotic error law (0.1 h^2 + 0.9 h^0.3, apparent order 1.52), H6's `E_h` (cap at 2 plus Giles maximum) misses the true production error in 11.3 % of replicates against a nominal 7.8e-4. Under pure order 2 it never misses.
  - The fourth "gap" stays open: the `lambda/h` 12 check and the four-level residual are "reported, not gating" (`D`:102-103).
- **Correction:**
  - State that at 2.25, and often at 4, the bound rests on the asymptotic range of a triple whose coarsest level is `lambda/h` 4.
  - Make the `lambda/h` 12 consistency check (2.25) and the four-level residual (4) **gating** for the order branch (fall back if inconsistent at a pre-registered level). This costs no GPU time.
  - Optionally, run a paired 4/8/16 pilot at 2.25 on reserved seeds before pre-registration (about 0.4 GPU-h).

**M1 — MAJOR. Nothing shows `lambda/h = 4` is in the asymptotic range for the transport estimand.**
- **Statements attacked:** H6 (`D`:94-96): "SF-42 E shows second-order flux convergence from `lambda/h = 4`"; `R`:246-247 (`E_h` 0.19 to 0.32 for 4/8/16); "better-conditioned coarser one" (`R`:35, :382).
- **Evidence:**
  - SF-42 E itself says "nothing here is evidence on `alpha_T`" (`N42`:693-694).
  - Bib5 (Cliffe et al. 2011): convergence rates depend on the quantity, about 1.75 for `k_eff` against about 0.75 for point flux.
  - SF-39's spline divergence `e_v` is 22.9 % at `lambda/h` 5 for `sigma^2 = 1`.
  - No `alpha_T` data exist below `lambda/h` 8 at 2.25. J1B's 416x96x96 run logged only solver, memory and status counts.
  - The 4/8/16 `E_h` values come from an order-2 extrapolation of SF-39's unresolved 8-to-16 difference (`G4`:37: level errors 1.45e-3, 3.6e-4, 9.1e-5).
  - "Better-conditioned" refers to the fallback multiplier (3 vs 12), not to numerical conditioning.
- **Correction:** Cite SF-42 E as flux evidence only. Label the `lambda/h` 4 rung an untested assumption, and apply the B2 gating.

**M2 — MAJOR. The fallback is not assumption-free.**
- **Statement attacked:** "a data-range bound that does not use the asymptotic-range assumption" (`R`:244).
- **Evidence:**
  - `3 UCB(range)/(r-1)` bounds the production error only if a monotone error falls at least like `h^0.21` with production finest (`4^p >= 4/3`), or `h^0.24` with production in the middle. It also assumes no error component common to all levels.
  - Bib5 calls it "a recognised conservative heuristic, not a derived bound".
  - SF-42 qualified it only under exact power laws of order 1, 1.5 or 2, or an alternating order-1 sequence (`covproc.py`:146-160).
  - At the noise floor it measures resolution, not discretization: `E_h` ≈ `3 t SE(range)`, which scales with `N` and `M` (`G4`:204).
- **Correction:** List the fallback's conditional validity among the stated assumptions, and describe `E_h` at the noise floor as a resolution bound.

**M3 — MAJOR. Solver non-convergence at `sigma^2 = 4` is not addressed (H11, H12, `R` 3.g).**
- **Evidence:**
  - SF-39 `logs/job_logs/sf39-extend.log`:225, 363, 477, 495-496 record four solves that hit 2000 iterations, all at `sigma^2 = 4`. Three were accepted through the 1e-8 fallback (rel 3.0e-10, 2.2e-10, 9.9e-10). One was unusable (rel 3.4e-7). The other 192 solves in that log took 20 to 100 iterations.
  - On 1664x192x192 the rate was 3 of 61 (one unusable). Seed 5023 failed at both `lambda/h` 16 and 32, so failure is a property of the field.
  - Likely cause (a hypothesis): the head-solve multigrid preconditioner is not symmetric. It uses red-black Gauss-Seidel before and after the coarse correction (`src/multigrid/cycle/v_cycle.cu`:43,52,72; `gsrb_3d.cu`:367-372). The projected V-cycle uses black-red post-smoothing (`projected_positive_v_cycle.cu`:60,87). PCG needs a symmetric preconditioner, and more iterations will not cure a residual floor.
  - Par2's solve goes through the same multigrid PCG (`solve_head.cu`:217) and exits 0 on non-convergence.
  - The verdict uses the recursively updated residual (`pcg.cuh`:799-802) plus a global balance. `max_div`, which measures the local true residual, is recorded but not used.
  - SF-43 puts 250 consumer and 250 Par2 realizations on 1664x192x192. Expect about 12 stagnated solves (95 % range roughly 3 to 35) and about 4 unusable ones. That exceeds the specification's 1 % failure threshold, so its worst-case imputation will trigger.
  - Replacing failed fields conditions Gate 5 at 4 on solvability. That is plausibly correlated with the slow zones that drive `alpha_LP`.
- **Correction:**
  - Pre-register expected failure rates per geometry and keep at least 5 % reserve seeds on that grid.
  - Add a Par2 usability rule that mirrors the consumer's (rel ≤ 1e-8 and balance ≤ 1e-8, parsed from `Converged=`), and a pair-exclusion sensitivity for Gate 5.
  - Report `max_div` for every solve, and add the 2000-iteration solves (about 2 to 4 GPU-h) to the cost.

**M4 — MAJOR. Under an order-1 width effect, the interaction component at 2.25 does not cover the transfer to production (H7).**
- **Evidence:**
  - At 2.25 the interaction is measured from `lambda/h` 8 to 10, but the transfer runs from 10 to 16. Under an order-1 width effect the transfer is 1.5 times the measured interaction; under order 2 it is 1.09 times. At 4 it over-covers (0.5 and 0.25 times).
  - SF-42's K = 4 coverage cells had the production resolution inside the 2x2 (`covproc.py`:44-50); H7's 2x2s do not include it.
  - The Dirichlet outlet boundary layer reaches about `W/(2 pi)`: 3.8 `lambda` at width 24, 7.6 at 48. The window ends 5 `lambda` before the outlet, so the 24-vs-48 width bound partly measures a change in outlet influence.
- **Correction:** State these limits. Consider H10 option (c), which the draft listed but did not evaluate: a 2x2 at `lambda/h` 8/12.
  - It reuses the Gate 5 consumer grid 1248x288x288.
  - It admits 65 common `M`, including 100, with the G4 source of 19.2 `lambda`, so no source change is needed (my lattice run).
  - Its interaction over-covers the 12-to-16 transfer.
  - Cost: the (48, 12) cell is 1248x576x576, about 25 GB, so lane A; roughly +1 to 2 GPU-h net.

**Minor findings**
- **m-a. `E_int` size.** "At most 1e-6 relative" (`R`:247, :349) is the mean difference, not the bound. The bound is `2 UCB` at `N` = 30 and tail 7.8e-4: 1.67e-6 `lambda` = 5.7e-5 of the mean at 2.25, and 3e-6 at 4 (SF-39 `s225_A_hmax8/16`, flux rows). It is still negligible.
- **m-b. H6 wording.**
  - It does not say which `p_L` is range-tested, or which levels the fallback range uses at 4. With the cap inside the minimum, the 2.1 upper limit does nothing unless it is applied per triple.
  - All H6 rules are new code: `LadderBound` has `p_max = 4.0` (`budget.py`:217), with no cap, no per-triple minimum and no Giles term.
  - Four-level ladders were never covered by SF-42's coverage run (`coverage.py`:135).
- **m-c. The 3x PCG rule list is incomplete.** The comparator is SF-39's median at the same `sigma^2`, which is 20 at 0.25. Shallow-hierarchy grids barely change iteration count with `sigma^2` (1040x240: 60 / 70 / 90 at 0.25 / 1 / 2.25). 520x120, 1040x480 and 2080x480 were measured only at higher `sigma^2` (70, 90, 90) and likely sit near 3x at 0.25. Extend H11 to them or measure them in the smoke test.
- **m-d. W8 reading.**
  - The pass-to-pass separation compares two integrations of the same spline field, which is neither divergence-free nor helicity-free. Exponential separation is generic for such a field.
  - Spline vs Pollock (both built from the same FV fluxes) and the `d`-dependence do not discriminate it from physics.
  - The discriminating measurement is the separation exponent as a function of `lambda/h`: two `h_max` passes on about 20 seeds per ladder rung, about 0.2 to 0.5 GPU-h.
- **m-e. No end-to-end exact-solution check of the transport estimand exists.** SF-37 and SF-42 D/E checked convergence and fluxes only. The box first-order profile from B1 gives an exact target at small `sigma^2`: about 30 realizations on 1040x240 (about 0.2 GPU-h), if per-plane variances are kept.
- **m-f. Wording and units.**
  - "15.7 GB" is 15703 MiB, about 16.5 GB.
  - H10 calls `M = 128` "the only" admissible value, but 256 is also admissible in [40, 300].
  - The `R`:350 entry "if cut: fallback only" for the `lambda/h` 4 rung at 4 is wrong; 8/16/32 keeps its order branch.
  - H14 does not say that stage 2 keeps lane A's GPU (≥ 30000 MiB free). Variant C needs it.

## Claims I tried to refute that survived
- **First-order formula (unbounded domain).** `(tau/2) int mu^2 (1-mu^2) exp(-tau^2 mu^2)`, `X_22(inf) = sigma^2 lambda^2 / 3`, the `(sqrt(pi)/8)` tail, and 7.992e-5 all match my own wave-number derivation. `lambda` is the generator's correlation length (integral scale 0.886 `lambda`), and the units are right.
- **H3's null is still conservative.** 7.99e-5 is above every box value, so the null holds and coverage is shift-invariant.
- **Lattice.** `{128, 256}` at 0.25 and 1; none at 2.25 with 19.2 `lambda`; 121 values at 19.08 including 100; 65 at 4 with or without the rung (51 in SF-42's range). The 0.6 % source reduction is negligible: centring is about 5 % of the first-order slope and scales roughly as 1/S^2.
- **H6 "larger or equal".** Against the qualified three-level `LadderBound` (`p_max` 4, Student method) the bound dominates path by path in every branch: the cap, the range moving resamples to the larger fallback, the per-triple minimum and the Giles maximum. This holds provided m-b is fixed.
- **8/16/20 at 2.25 multiplies the fallback by 12** (`budget.py`:272, :302).
- **`E_int` is negligible.** Keeping it costs about 3 % on the other components' quantiles.
- **Memory classes and lane B at 17 GB.**
  - The peaks are measured, or estimated consistently, and include the CUDA context.
  - Three `h_max` passes do not raise the peak: the solve phase dominates, about 3.7 GB vs 1.05 GB for tracking on 1664x192x192.
  - Two GPUs give no PCIe or CPU contention of note.
- **H11 for the two named grids.** They converge (recursive residual ≤ 4e-11) and their cost is measured.
- **The tracker records first passage.** Planes are landed in order; recrossings are counted, not recorded.
- **Box length does not move the planning means.** Displacement-basis means in the 104-`lambda` box over SF-39's longer boxes: 1.025 / 1.036 / 1.019 at 0.25 / 1 / 2.25 (n = 14, SE 4 to 5 %).
- **Other checks that held.** Second-order flux and Dirichlet-face convergence (as flux statements); the 3.5 `lambda` displacement SD at `sigma^2 = 4`; zero expected seed-displacement covariance with uniform weights; the inverse-Gaussian variance and the `d/alpha_L` ranges; the width deficit counting twice; fallback = Eça and Hoekstra's `3 Delta_M` at ratio 2.

Not checked: the Dirichlet boundary correction and flux weighting in the first-order box sum (both stated approximations); host RAM; statistics and cost beyond these numerics points. The main risk is B2: if the coarse `lambda/h` 4 level is pre-asymptotic, the bound at 2.25 can under-cover by orders of magnitude while being reported as conservative.

### Critical Files for Implementation
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/decisions/2026-11-02-sf43-pre-activation-decisions.md
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/2026-11-02-sf43-pre-activation-review.md
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/mcstat/budget.py
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/artifacts/2026-11-02-sf43-pre-activation-review/scripts/firstorder_window.py
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/src/multigrid/cycle/v_cycle.cu