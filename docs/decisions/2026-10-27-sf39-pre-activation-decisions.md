# SF-39 pre-activation decisions: executable estimator definitions, a transfer rule that can pass, rule D on measured direction, a conjunctive plane-based co-primary, bounded-value reporting, pathwise drift, power realism, a width check and the method contrast

- Status: accepted (owner delegation of 2026-10-27: the owner asked the orchestrator to decide each finding of the pre-activation
  review "considering our reality"; ratified by the human merge of the PR that adds this record)
- Date: 2026-10-27
- Deciders: orchestrator under owner delegation (session 2026-10-27); owner by merge
- Evidence: `docs/experiments/2026-10-27-sf39-pre-activation-review.md` (findings B1-B4, M1-M8 and minors; five reader reports,
  four adversarial reviews, own calculations; scripts under `artifacts/2026-10-27-sf39-pre-activation-review/scripts/`); two red-team
  reviews of the draft of this record (executability against the code; statistical operating characteristics, simulated), whose
  corrections are incorporated (summary in the experiment note, section 7)
- Builds on: `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`, `docs/decisions/2026-10-23-sf38-activation-decisions.md`
  (D1, D3, D11, D14-D20), `docs/decisions/2026-10-24-sf41-sf39-planning-decisions.md` (E4-E7)
- Supersedes / refines (without editing those records):
  - 2026-10-23 D15 "the narrow-ladder verdict transfers to the production domain iff the relative 1/2 -> 1 differences [...] have
    the same sign and a ratio in [1/2, 2]": replaced by F4;
  - 2026-10-23 D18: the precedence is kept; the branches are rewritten (F5): the signed `alpha_T` replaces `|alpha_TAM|`, a 90 %
    CI of the signed paired difference replaces the point estimate, the branches apply symmetrically to the `h` and `h_max` ladders
    with an added upper-bound label, and the amplitude label requires both transverse estimators to agree;
  - 2026-10-24 E4 ("Gate 4 uses time moments in the frame of each realization's mean Darcy flux"): kept for `alpha_TAM`, but Gate 4
    becomes conjunctive with the plane-based `alpha_TP` (F3);
  - SF-39 "CI by bootstrap over realizations" and 2026-10-17 item 8 ("SE by block bootstrap"): Student `t` intervals are primary,
    the realization BCa bootstrap a sensitivity (F6);
  - SF-39 "Pilot and `N_R`" (extrapolation from the pilot SE): the upper 80 % confidence limit of the pilot sd (F9);
  - 2026-10-23 D1/D9 window `[0.1, 0.9]^2` (0.8 `L_T`): kept for every study run; the exploratory width arms use fixed-size
    windows (F10);
  - 2026-10-23 D11 (Pollock bridge on the production realizations): extended to every narrow-ladder rung and the coarse twin (F11);
    its cost (~7-10 s per production realization, less on the narrow rungs) is added to the compute projection of the checkpoint;
  - SF-39 rule D "zero within resolution" (95 % CI includes 0 with half-width at target): replaced by F6;
  - 2026-10-24 E5 (ladder `N_R` from the paired sd of `alpha_TAM` and `alpha_LAM`): `N` is computed from the transverse estimators
    only (F9);
  - 2026-10-17 item 1 (`sigma^2 = 6.25` exploratory in the study): deferred out of SF-39 (F12);
  - 2026-10-07 (Gaussian covariance only) item 4 ("RWPT re-run with the same YAML"): superseded for SF-39 by D10/D11 (same physical
    domain, twin series, Pollock bridge) and by F2 (velocity normalization); recorded here, not edited there;
  - `docs/runbooks/remote-v100.md` "two DAG nodes of the same increment [...] must still be serialized": exception for E6 (F13).
- Not changed: the Goal of SF-39; the margins and CI targets frozen by D15 (A: `+-max(0.05 |alpha_TAM|, 1.25e-3 lambda)`,
  `+-2.5 %` `alpha_LAM`; B targets; C: `+-10 %` of Par2's `alpha_LAM`); the production geometry (D1); the E4 frame of
  `alpha_TAM`; medium (b); the tracker, stage, media and Par2 code; Rule 1 and the non-presupposition of `alpha_T`.

## Date of knowledge (integrity statement)

D15 fixed its items before any `sigma^2 = 4` plume statistic existed. E4 and E5 (2026-10-24) were fixed after SF-38's A/B
statistics (seeds 4001-4010, Pollock plume, 8 `lambda` width at `sigma^2 = 4`) and its post-hoc readings P1-P6. The items
of this record were fixed after the same SF-38 statistics and after the SF-41 F-ladder plane variances (3 seeds, 2 `lambda`
section, <= 24 `lambda`), **before any SF-39 realization was generated** (seeds 5001+ are independent of every seed used so far).
Each item below states whether it follows from a design property derivable without data (most do) or was prompted by a
post-hoc reading; none tunes a threshold.

## Context

The pre-activation review (evidence above) found the SF-39 specification not executable as written (B1), two decision rules
that mislabel by construction (B2 transfer rule: passes ~20 % of the time when there is no grid effect; B3 rule D branch 1: its
cited premise is inverted by the existing medium (b) data and it fires ~50 % of the time under the null), an incoherent
"zero within resolution" branch (B4), and eight interpretation threats (M1-M8): a time-window estimator truncated at
`tau_b ~ 27-38` at `sigma^2 = 4` with a first-order positive pre-asymptotic bias of 0.8-1.7e-3 `lambda` plus a flow-meander term
the E4 rotation does not remove; "asymptotic" never tested; integration drift measured only to 64 `lambda`; likely insufficient
power at `sigma^2 = 4`; an uncontrolled transverse-width confounder; the method contrast that carries the project's thesis
only "recorded"; the "frozen" claim partly untrue; `t_b` an extreme-value statistic. The reality the decisions must fit:
scripts and configs only (no tracker/stage/media change), two V100s with ~19-91 GPU-h projected (decision 2026-10-24),
the frozen margins, and the human review of the result.

## Decision

Each item: options considered, choice, why. "Derivable" = follows from a design property without data.

### F1. Integrity statement and scope of "frozen"

The SF-39 thresholds header states the date of knowledge above verbatim: D15 items frozen before any `sigma^2 = 4` plume
statistic; E4, E5 and F1-F14 fixed after SF-38's A/B statistics and post-hoc readings and SF-41's F-ladder plane variances,
before any SF-39 realization. (Options: leave the sentence; delete it. Chosen: qualify it — the record must be true.)

### F2. Executable estimator definitions (B1 i-iv, M8)

- **Per-realization breakthrough** `t_b,r` = the minimum over `ok` particles of `planes_tau` at the plane `L1 - 5 lambda` (h/8 pass).
- **Units.** The runner's box unit is the transverse period, so `lambda` in box units (`ell = (lambda/h)/n_T`) differs between the
  production grid and the narrow-ladder and width-arm grids (at `sigma^2 = 4`: 1/12 against 1/4 and 1/24). Every window is therefore
  fixed in normalized time `tau = U t / ell`; absolute box time coincides only between runs with equal `ell` (production, coarse twin,
  `h_max` passes, Pollock bridge). `G1 = 1` in every run.
- **Normalization velocity** `U` of a run (an arm at one geometry) = the mean of `qbar_r` (`flow_stats.json`, FV inlet flux) over all
  of that arm's realizations, pooled across output directories (F12); it normalizes Gate 4, A and B. **Gate 5** uses one estimator in
  both arms: the ensemble centroid slope `U_c = [mean_x(t_b,min) - mean_x(0)] / t_b,min` of the uniform-weight series (Lagrangian twin;
  Par2, whose flux is not output), with the FV `qbar` of the Lagrangian arm as a sensitivity (area-uniform injection carries its
  centroid at the plane-mean velocity in expectation over realizations, not per realization). Par2's in-pipeline `alpha` (normalized by
  the nominal `J`, overstating by ~1.19 / 1.47 / 1.98) is never used.
- **Frozen window** per amplitude: `tau_b* = U_prod t_b,min / ell_prod`, where `t_b,min = min_r t_b,r` over the 20 production-pilot
  realizations; recorded in the pilot record before any further run and used unchanged for production, every ladder rung, the coarse
  twin, every `h_max` pass, the width arms and the Pollock bridge of that amplitude. A run uses `W = [0.75, 1] x tau_b* ell_run / U_run`
  in its own box time. Options: per-realization `t_b,r` (differs between paired rungs: the paired difference would mix the grid
  effect with a window shift, and fast tails are the most grid-sensitive); ensemble minimum per run (an extreme-value statistic that
  falls with `N_R` and differs per rung). Chosen: one frozen normalized value (derivable).
- **Per-realization estimator.** `Var_k,r(t)` at the endpoints of `W` by linear interpolation between the recorded snapshot times
  (runner auto grid `n ell / 2` in box time); `alpha_k,AM,r = [Var_k,r(t_hi) - Var_k,r(t_lo)] / (2 ell U (t_hi - t_lo))` with
  `[t_lo, t_hi] = W`, i.e. the exact time average of B&dD eq. 4 over `W` for piecewise-linear `Var` (dimensionless, in units of
  `lambda`). The ensemble `alpha_AM` is the mean over realizations; with common `W` and `U` it equals B&dD's estimator applied to the
  ensemble series (linearity). Interpolation rather than explicit `--times` endpoints: the endpoints in a run's box time need `U_run`,
  known only after its stage solve; the interpolation error is second order in the `ell/2` spacing and common to every rung.
- **Particle set and exits.** Both endpoints of `W` are computed over the same particles: those `ok` (with a recorded position) at the
  first snapshot at or after `t_hi`. A production realization has `t_b,r < t_b,min` with probability ~1/21 (exchangeability) and can then
  lose particles through the outlet before `t_hi` (5 `lambda` buffer); the inlet and outlet exits before `t_hi` (`exits_time.csv`), the
  non-ok counts of both passes and the excluded fraction are recorded; the sensitivity analysis uses the full population with exited
  particles frozen at the outlet. Realizations are never dropped (dropping them would select against the fastest fields).
- **Gate 5 common window.** Normalized time `tau = U t / lambda` per arm (`lambda` in each arm's own length unit: `ell` box units for
  the Lagrangian twin, `corr_length` for Par2); `tau_c = min(tau_b*(twin), tau_b*(Par2))`, `tau_b*(Par2)` =
  the minimum over the first 20 Par2 production realizations of the last snapshot time with no particle at `x >= Lx - 5 lambda`
  (conservative at the snapshot resolution), times `U_Par2 / lambda`; `W_c = [0.75 tau_c, tau_c]` for both arms, `Var` by linear
  interpolation (Par2 statistics every 10 steps; Lagrangian snapshots). Par2's wrapped snapshots are used only for `t_b`.

### F3. Plane-based transverse estimator as a conjunctive co-primary (M1)

- **Definition.** `sigma_T,r^2(X) = (var x2 + var x3)/2`, flux-weighted, at `X_a` and `X_b` over the same particles: those with a
  recorded first crossing of both planes (the excluded fraction is recorded; slow or non-ok streamlines correlate with transverse
  position); `alpha_TP,r = [sigma_T,r^2(X_b) - sigma_T,r^2(X_a)] / (2 ell (X_b - X_a))` (units of `lambda`)
  with `X_b = L1 - 5 ell` and `X_a = x0 + (L1 - 10 ell)/2` (the second half of the travel distance); the non-crossing count per plane
  is recorded.
- **Planes** (`--planes`, explicit): the union of `x0 + m ell` (`m = 1 .. floor((L1 - 10 ell)/ell)`), `x0 + (L1 - 10 ell)/4`, `X_a` and
  `X_b`, sorted ascending, values within `1e-12 max(1, |x|)` merged keeping the canonical `X_a` and `X_b`, printed `%.17g` (the runner
  requires a strictly increasing list and refuses duplicates with exit 2; at `sigma^2` 2.25 and 4, `X_a` and `X_b` coincide with
  lattice planes); the analysis finds `X_a`, `X_b` by nearest plane within `1e-12`. The list is a superset of the auto planes.
- **Role.** Criteria A, B and D apply to `alpha_TP` with the same margins and targets as `alpha_TAM`. `physical` requires **both**
  estimators `physical` (an intersection-union rule: no multiplicity correction needed); a discordance is
  `unresolved (estimator-discordant)` (F5). Under Fickian transport with flux-weighted injection `E[tau(X)] = X / qbar`, so `alpha_TP`
  and `alpha_TAM` estimate the same dispersivity when `U = qbar`.
- **Why (derivable).** For `v_s = k_s grad Phi_s` the paths are field lines of `grad Phi_s`: the transverse position at a plane is
  pure streamline geometry, independent of `k_s` (the travel-time tail); centring at a plane removes a uniform tilt (residual
  `O(theta^2)`, ~5e-4 relative) and the section-mean meander `y_c(x1)`, which time moments of a longitudinally stretched plume pick up
  and E4's single rotation does not remove ((0.45-1.1)e-3 `lambda` at 12 `lambda`, (4-10)e-3 at 4 `lambda`, post-hoc estimate); it uses
  the full length instead of stopping at `t_b`; at `sigma^2 = 4` its first-order pre-asymptotic bias is 4-8x smaller (2.0e-4 `lambda`)
  than the window estimator's (0.8-1.7e-3). Options rejected: replace `alpha_TAM` (breaks the B&dD/E4 continuity); either-or
  (inflates false `physical`); reuse SF-38's last-octave `S_T` on powers of two (does not use the last ~35 % of the length at
  `sigma^2` 1 and 2.25).

### F4. Production-domain `h` verdict without the ratio rule (B2)

Notation per estimator `Q` (`alpha_TAM`, `alpha_TP`, `alpha_LAM`): narrow ladder `D_c = Q(prod) - Q(1/2)`, `D_f = Q(2x) - Q(prod)`
(paired, narrow width); production domain `D_tw = Q(prod) - Q(coarse twin)` (paired). `m` = the A margin evaluated at
`min(m_narrow, m_prod)` (the margin at the narrow mean and at the production mean, whichever is smaller).

Production `h` verdict:
- `converged` iff the narrow verdict (A: 90 % CI of `D_f` inside `+-m`) is `converged`, **and** the 90 % CI of `D_tw` lies within
  `+-2^1.5 m`, **and** the narrow ladder does not reject a monotone order `>= 1.5` (the 90 % CI of `s (2^1.5 D_f - D_c)`, `s` = the sign of
  the mean `D_c`, does not lie entirely above 0);
- `not converged` iff the narrow verdict is `not converged`;
- `inconclusive` otherwise.

Why: for an error `C h^p`, `Q(h/2) - Q(h) = D_tw / 2^p`, so `|D_tw| <= 2^1.5 m` bounds the production-vs-2x difference by `m` for every
`p >= 1.5`; the order check is a linear contrast (no ratio of noisy differences). The written rule (same sign and ratio in [1/2, 2] of
two noisy relative differences) passes 0.20 of the time with no grid effect and 0.38 with a true 5 % effect at N = 20-like standard
errors (simulation, derivable). Simulated operating characteristics of F4 (equal N, twin/narrow sd ratio k = 1): no grid effect 0.82;
true production-vs-2x difference 1-2 m with the same error law in both domains <= 0.003. **Assumptions stated, not testable within
SF-39** (a 2x production-domain run does not fit): the production-domain error is represented either by the narrow ladder or by
`|D_tw| / 2^1.5` (fails if the narrow domain is unrepresentative and the twin lies outside its asymptotic range — e.g. the twin at
`lambda/h = 5` for `sigma^2 = 1`, below SF-37's tested range; then F4 can declare `converged` falsely, 0.18-0.82 in simulated
adversarial cases). `nr_rule.py` sizes the twin with margin `2^1.5 m`. The Richardson guard stays for reporting only.

### F5. Rule D on measured direction (B3)

Directions refer to the signed paired difference `finer - coarser` of `alpha_T`, whatever the sign of either mean; "finer" is the 2x
narrow rung (`D_f`) or the production rung against the twin (`D_tw`) for `h`, and `h/16` against `h/8` for `h_max`. "Above 0" means the
90 % CI lies entirely above 0. Per transverse estimator, applied symmetrically to the `h` and `h_max` ladders, in this order (D18
precedence kept):
1. `unresolved (grid-limited, lower bound)` if the verdict of that ladder (production `h` verdict of F4, or `h_max`) is not `converged`
   and a refinement difference lies above 0 (`D_f` or `D_tw` for `h`; `h/16 - h/8` for `h_max`) while the other one's point estimate is
   not below 0: the production value is a lower bound, never `physical`;
2. otherwise `numerical` if a verdict is `not converged`, a refinement difference lies below 0 (the other's point estimate not above 0),
   **and** the finer-setting 90 % CI contains 0 or its upper limit is `<= 2.5e-3 lambda` (the growth vanishes under refinement);
3. otherwise `unresolved (grid-limited, upper bound)` if a verdict is not `converged` and a refinement difference lies below 0 (the value
   falls under refinement but does not vanish, or the verdict is only `inconclusive`): the production value is an upper bound;
4. otherwise `physical` if both verdicts are `converged`, with the sub-label of F6;
5. otherwise `unresolved`.

Amplitude classification: the common label when both transverse estimators agree (`physical` with the same sub-label requires both);
otherwise `unresolved (estimator-discordant)`, with the per-estimator labels listed. The citation of the resolution-vs-variance note
§3.7 is removed: its sign (cube: coarse grids under-disperse) is inverted on medium (b) (SF-41 F ladder: coarse above the fine reference
in 5 of 6 pairs; post-hoc), so the direction is taken from SF-39's own ladders.
Why: under the null a point estimate rises half the time; labels that assert a direction need the direction measured (derivable). The
old exclusion "branches 1-2 do not apply when the finer-setting CI contains 0" would have made `numerical` unreachable in exactly the
case the project's thesis predicts (a positive coarse value that refinement removes). Taking the first non-physical label across the two
estimators would combine two 5 % tests (false `grid-limited` 6-10 % under the null, simulated); agreement keeps it below ~3 %.

### F6. `physical` sub-labels and interval type (B4)

- `physical (positive)`: the production 90 % CI lower limit > 0; the value is reported with its CI.
- `physical (negative)`: the production 90 % CI upper limit < 0 (a contracting effective transverse variance); reported with its CI.
- `physical (bounded)`: otherwise, if the production 90 % CI upper limit `U <= 2.5e-3 lambda`; reported as `alpha_T <= U`.
- neither: `unresolved (imprecise)`.

Every value carries "within this model, medium and domain, at the measured distance" (F7). Intervals: Student `t` on the
realization mean (`N - 1` degrees of freedom), the same family as A's TOST and C's Welch; the realization BCa bootstrap
(`B = 5000`, generator seed fixed in the activation row) is reported beside it as a sensitivity. Why: the old branch mixed 95 %
with the 90 % of A, turned "not rejected" into `physical` and implied a 97.5 % bound of 5e-3 (half B&dD's `sigma^2 = 1` value);
percentile bootstrap undercovers at `N = 20` (0.87-0.92, simulated) (derivable).

### F7. No asymptotic claim; decay characterized (M2)

- The intent's "asymptotic macrodispersivities" becomes "macrodispersivities at the travel distances of the long domain, with the
  approach to the asymptote characterized, not assumed". Values are stated "at the measured distance".
- Descriptive, pre-registered: `Delta_half = alpha_TP[x0 + L/4, x0 + L/2] - alpha_TP[X_a, X_b]` (`L = L1 - 10 ell`) with its
  90 % CI. If the CI lies above 0 ("decreasing"), the second-half value is also reported as an upper bound on the asymptote under
  monotone decay; otherwise no statement beyond "at the measured distance". Used only to downgrade wording, never to promote it.
- `alpha_TAM` (window centroid at ~27-38 `lambda` of travel at `sigma^2 = 4`) and `alpha_TP` (47-94 `lambda`) target different travel
  distances; a first-order gap of ~8e-4 `lambda` at `sigma^2 = 4` is expected, so their difference alone is not evidence of an error.
- Full `sigma_T^2(x1)` and `sigma_T^2(t)` series recorded.

Why: no finite domain separates a slow decay from a small plateau (an octave ratio is degenerate between `a + c x^-2` and `x^-1`);
first-order theory gives `alpha_T(tau) = sqrt(pi) sigma^2 lambda / (8 tau^2) + O(tau^-4)` (derived, confirmed independently),
positive at every finite time.

### F8. Pathwise drift and sensitivity from the `h_max` passes (M3; no code change)

Per realization and plane, from `hmax_8` and `hmax_16` (and `hmax_32`) `planes_x23.npy` of the same stage and seeds:
`rho(X) = RMS |Delta x_T| / sigma_T(X)` over particles crossing `X` in both passes. Thresholds derived from the A margin `m` (in units of
`lambda`), with `Delta X = X_b - X_a` and `S = sigma_T^2(X_b) / (ell Delta X)`:
- `pathwise resolved` if the 95th percentile over realizations of `rho(X_b) <= 0.1 m / S` (even a perturbation fully correlated with
  the displacement biases `alpha_TP` by at most `0.1 m`);
- `pathwise-limited` if the median `rho(X_b) >= sqrt(0.2 m / S)` (a random perturbation alone biases `alpha_TP` by at least `0.1 m`);
- `decorrelated` if the median `rho(X_b) >= 0.5` (the `h_max` verdict is then statistical only);
- the growth of the median `rho` with `x1` (exponential and power-law fits) recorded as a kinematic diagnostic of the represented flow,
  not as a test of Lester's `lambda_inf = 0` (the perturbation is numerical).

Recorded, non-gating. Why: the only integration evidence reaches 64 `lambda` (SF-37 RR15 shadow, `MS(dx_T)/sigma_T^2 <= 1.27e-13`); the
`h_max` TOST is weak (the error falls 2^4-2^5 per rung); and Lester et al. (2025, App. D) reject exactly this construction (RK on a
spline `k grad phi`) for the isotropic case without drift evidence. The per-pass arrays already exist (`EnsembleRunner.cuh`
multi-pass layout).

### F9. Power, `N_R` and the pilot checkpoint (M4)

- **Ladder `N`** (E5 refined): from the paired sd of the transverse estimators only (the larger `N` of `alpha_TAM` and `alpha_TP`), using
  the upper one-sided 80 % confidence limit of the pilot sd (chi-square, 19 degrees of freedom; `N` grows by ~1.39); the coarse twin with
  margin `2^1.5 m` (F4); `alpha_LAM` verdicts are evaluated at that `N`. `nr_rule.py` prints `N` only (no mean, no margin).
- **Production `N_R`**: the B rule for `alpha_TAM` and `alpha_TP` (the larger), with the same upper 80 % confidence limit of the pilot
  sd, capped at 500; `alpha_LAM` reports its achieved half-width against its target.
- **Checkpoint report**: per ladder and amplitude only `N`, measured GPU-seconds per realization, projected GPU-hours and projected bytes
  on the V100 host; no means, margins or verdicts. The owner fixes a budget. If the projection exceeds it, the `sigma^2 = 4` ladders go
  first to the floor of 20 (their verdicts are expected inconclusive, below), then every remaining `N` is scaled by one common factor
  with a floor of 20.
- **Anticipated outcome (pre-registered):** with the A/B pairing as proxy, equivalence at `sigma^2 = 4` needs `N ~ 221` for
  `alpha_T` and `~ 957` for `alpha_LAM` at `+-2.5 %`; within the budget the `sigma^2 = 4` verdicts are expected `inconclusive`
  (rule D: `unresolved`). Such an outcome is reported as such, not re-run with tuned settings.

Why: the B rule and E5 with a point sd meet their targets only ~50 % / ~75 % of the time; letting `alpha_LAM` (N ~ 957) drive the cap
would spend the budget on a quantity already caveated pre-asymptotic at `sigma^2 = 4` (D3); a checkpoint that shows means partially
unblinds the margins (they depend on the finer-setting mean). Margins are not changed: they were frozen by the owner (D15), and
loosening them after seeing SF-38's `sigma^2 = 4` standard errors would be a data-informed relaxation.

### F10. Exploratory width check at `sigma^2 = 4` (M5)

At `lambda/h = 16`, unpaired, `N_R = 30` per arm, recorded with a `+-10 %` equivalence reading, non-gating:
(i) 24 `lambda` (1664x384x384, `--window 0.3,0.7,0.3,0.7` = 9.6 `lambda`, seeds 7101-7130) vs 12 `lambda` (the coarse twin
1664x192x192, window 9.6 `lambda`, its first 30 realizations);
(ii) 12 `lambda` (1664x192x192, `--window 0.36666666666666664,0.63333333333333336,0.36666666666666664,0.63333333333333336` = 3.2
`lambda`, seeds 7201-7230) vs 4 `lambda` (the narrow 1/2 rung 1664x64x64, window 3.2 `lambda`, its first 30 realizations).
The RR12 seed-face assertion passes for both windows (minimum distance 0.008 cell widths; derived as the runner computes it). Both estimators reported; the transverse-domain effect statement of D cites them. Cost < 1 GPU-h (24 `lambda` box
2.45e8 cells, ~14.7 GB, ~65-70 s per realization; estimate). Why: the narrow ladder at `sigma^2 = 4` is 4 `lambda` periodic (the
period of the cube, whose transverse growth is mostly a periodic-medium effect) and production is 12 `lambda` against B&dD's
25.6 `lambda`; nothing in the project measures a width effect on plume statistics; a fixed source size isolates the width
(the 0.8 `L_T` window otherwise changes it). `sigma^2` 1 and 2.25 (25.6 / 24 `lambda`) are not checked (meander term ~1e-5-1e-4).

### F11. Method contrast as pre-registered estimation (M6)

- Pollock bridge on production, on every narrow-ladder rung and on the coarse twin (same realizations, seeds, faces).
- Estimated, with paired 90 % CIs, no verdict label: `Delta_P = alpha(Pollock) - alpha(consumer)` per amplitude, rung and transverse
  estimator, on the **uniform-weight (twin) series** of both arms (their flux weights differ, SF-37 RR7), and its change across
  `lambda/h` (difference in differences). Hypothesis stated (Lester et al. 2023 §5.2, source (A); Pollock error variance `~ Delta^2`):
  `Delta_P > 0`, shrinking under refinement while the consumer's ladder converges.
- Par2 (Euler + face-linear interpolation, the B&dD scheme; different generator) `alpha_T`: descriptive, fixed axes, normalized by its
  centroid slope `U_c` (F2).
- Comparisons with B&dD (0.01 / 0.044 at `sigma^2` 1 / 2.25) as one-sided upper bounds of the consumer's fixed-axis `alpha_T`:
  hypothesis-generating (prompted after SF-38's statistics), with B&dD's conditions stated (Gate 5 list).

Why: the project's thesis (transport free of spurious macrodispersion) needs, on the same fields, the consumer converging and
classical schemes carrying an excess that shrinks with refinement; Gate 4 already requires the Pollock comparison
(`acceptance-gates.md`, Lagrangian-consumer item). The direction alone does not discriminate (coarse-grid effects can share it),
hence the difference in differences; its power is low and it is reported as estimation.

### F12. Remaining definitional fixes (B1 v-xi, minors)

- Covector statistics are not produced (the runner has no option; scope is scripts and configs); the divergence diagnostic is the
  Lagrangian `e_v` from `I`, always recorded.
- Aligned moments (E4), `alpha_TP`, `rho`, the twin/Pollock contrast and every reduction run as SF-39 analysis scripts on the V100
  host (detached jobs) over the per-particle arrays, which stay there (as in SF-37); reduced per-realization CSVs are committed.
- Scripts created by SF-39 (listed in the specification): `run_sf39.sh`, `nr_rule.py`, the analysis package, the Par2 configs,
  `preregistration.md`.
- Par2 configs for SF-39: copies of the SF-38 configs with `stochastic.seed: 9001` (K seeds `9001 + r`), `NR` = that amplitude's
  Lagrangian production `N_R` (at least 20; the first 20 realizations are Par2's pilot for `t_b(Par2)`), `output.output_dir` under the
  SF-39 artifact (`raw/par2_s{1,225,4}`) and an updated header; everything else unchanged (including `cg_rtol: 1.0e-6`, recorded); the
  per-realization series `stats/realization_%04d_timeseries.csv` (`t, mean_x, mean_y, mean_z, var_x, var_y, var_z, active`) feed F2; the
  SF-38 pilot runs are not reused. Par2 runs as one process without restart: it is launched in parts (pilot, then the extension with
  `stochastic.seed` advanced by the pilot count) so that a failure loses one part.
- `sigma^2 = 6.25`: not run in SF-39 (needs its own resolution ladder, resolution-vs-variance §5 item 7); deferred.
- `alpha_LAM` in fixed axes everywhere (A, B, C); the aligned-minus-fixed difference is `O(theta^2)`.
- Failure policy: a narrow size whose medium (b) solve is not usable (`rel <= 1e-10`, or the C0 fallback `<= 1e-8` with balance
  `<= 1e-8`) stops that ladder (owner).
- One cap rule for `N_R` (F9); the "Required regression tests pass" checklist item means: `scripts/regress` (set v2) reproduces its
  references with the activation-base V100 build used for every SF-39 run.
- Recorded additionally: non-ok statuses by type and backflow encounters per run; `alpha_T` with weights `w exp(-I)` (divergence
  sensitivity); `alpha_T` in the plume-drift frame beside the `q_r` frame; the split-half (interleaved lattice halves) particle
  component of the variance of each `alpha` (`N_p = 1e4` fixed; a component above 25 % of the between-realization variance is
  reported as a limitation, no re-run); Gate 5 sensitivity by ANCOVA on Par2's realized `sigma^2`; Par2's non-periodic field with
  periodic `y/z` boundary conditions (a seam) as a stated difference.
- Analysis hierarchy: **primary** per amplitude — the Gate 4 classification (F5) and the Gate 5 verdict; **key secondary** — the
  `alpha_LAM` ladder verdicts and B; **estimation** — F7, F8, F11; **exploratory** — F10, any `sigma^n` trend. One table lists every
  decision point and its outcome, inconclusive ones included.
- The `h_max` ladder is taken from the production run (no duplicated stage builds): production realizations `r = 0 .. N_hmax - 1`
  (seeds `S .. S + N_hmax - 1`, `S = 5001`) run with `--hmax-div 8,16,32` in directory A (their `hmax_8` pass is production data); if
  `N_R > N_hmax`, realizations `N_hmax .. N_R - 1` run with `--hmax-div 8` in directory B with `--seed0 S + N_hmax` (the runner's seed is
  `seed0 + r` and the field depends only on the seed); both directories use the same command line except `--out`, `--seed0`, `--nr`
  and `--hmax-div`; if `N_hmax >= N_R`, there is no directory B and production uses the first `N_R` realizations of A. The analysis pools
  both directories by seed (each directory's own `vmean_norm` and `macrodispersion.csv` cover only its realizations and are not used).
  Before production, one realization is run both ways and compared byte for byte: `field.json`, `flow_stats.json`, `solver_b.json`,
  `pollock/*`, and `hmax_8/{*.npy, moments_*.csv, exits_*.csv, track.json, twin/*}` against the single-pass realization root, plus the
  `stats` and `stats_twin` CSVs; `status.json` key by key; `DONE` and `timings.json` excluded (they differ by construction).
- Disk: the checkpoint report (F9) includes the projected bytes on the V100 host (~110-300 MB per realization per pass with the dense
  plane list, `N_p = 1e4`; directory A about twice that).

### F13. Two GPUs within one increment (E6 made consistent with the runbook)

Two SF-39 jobs may run concurrently on distinct GPUs iff they run the same synced build, write disjoint output directories and no
`sync` of the SF-39 mirror happens while either is `running` or `waiting_gpu`. Every SF-39 job sets `REMOTE_GPU_WAIT=7200`
(SF-34 is also READY). The runbook gets this exception.

### F14. Documentation corrections found by the review

`docs/theory/lester-2023-key-claims.md` (assumptions of §4: stationarity/ergodicity, `Pe -> infinity`, bounded `psi~` as an explicit
assumption; helicity-free from isotropy alone, stagnation-free a separate condition; "necessary but not sufficient" vs Lester 2021;
the pseudo-symplectic tracker never run in pure advection; p. 11's `k`/`phi` interpolation = the consumer's construction; Lester
et al. 2025 App. D; §5 scalings; §9 marked as project inference; Rule 1 exemption of the closure oracle for the Lagrangian
consumer, as in Gate 4); `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` (Euler stepping on linear interpolation, the
numbers of Tables 1-2 and of the protocol, `alpha_T ~ sigma^4`, non-monotone `alpha_T(t)`, convergence checked only at
`sigma^2 = 6.25`); `docs/validation/acceptance-gates.md` Gate 4 "Gate 1-3" (Gate 3 retired).

## Consequences

- SF-39 specification: intent wording (F7), In scope, Implementation, Validation commands, thresholds A-E, failure policy, Files
  rewritten accordingly; append-only bitácora row; Goal unchanged. SF-39 stays `pending` until its activation row, which copies
  the thresholds verbatim and fixes the base commit (the merge of this record's PR).
- New experiment note `docs/experiments/2026-10-27-sf39-pre-activation-review.md` with its scripts.
- `docs/runbooks/remote-v100.md` (F13), the two theory notes and Gate 4 (F14); the dashboard's SF-39 line points to this record.
- No code change; no run; no `alpha` claimed.

## Classification (docs/AGENTS.md)

| item | class |
|---|---|
| Transfer-rule pass rates (0.20 null, 0.38 at 5 %); F4 operating characteristics; false-label rates of the draft and final rule D; percentile-bootstrap coverage; first-order `alpha_T(tau)` law and window/plane biases | derived (simulation, closed form) |
| Missing definitions, scripts and runner options; multi-pass per-pass arrays; Par2 `J` normalization | confirmed in code |
| `tau_b` 27-38 at `sigma^2 = 4`; meander magnitudes; power `N`; medium (b) sign inversion | estimate / post-hoc (SF-38, SF-41 data) |
| F1-F14 | accepted scope (owner delegation, ratified by merge) |
| Analysis scripts, configs, `run_sf39.sh` parts | proposed architecture (SF-39 DAG) |
| `alpha_L`, `alpha_T`, their classification, the width and method-contrast estimates | open question (SF-39) |
