# SF-39 remediation: a confirmatory Monte Carlo study (SF-43) and its tooling and verification (SF-42)

- Status: accepted (owner decisions G1-G4, G10, G11 of 2026-10-29 in the orchestrator session); G5-G8 proposed by the orchestrator within
  those choices and ratified by the human merge of the PR that adds this record; G9 records open questions
- Date: 2026-10-29
- Deciders: owner (G1-G4, G10, G11, each chosen among the options listed); orchestrator (drafting, G5-G9)
- Evidence:
  - the post-review evaluation of SF-39: section "Post-review evaluation" of `docs/experiments/2026-10-27-sf39-macrodispersion-study.md`
    (PR #83, squash-merged on the default branch as `104cc254`; four independent reviews and three skeptic recomputations, every number
    re-verified);
  - three adversarial reviews of the draft of this record and of the two specifications (Monte Carlo design validity, executability
    against the code, governance fidelity and coverage), incorporated below.
- Builds on: `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`, `2026-10-23-sf38-activation-decisions.md` (D1-D21),
  `2026-10-24-sf41-sf39-planning-decisions.md` (E4-E12), `2026-10-27-sf39-pre-activation-decisions.md` (F1-F14),
  `2026-10-27-remote-single-gpu.md`, `2026-10-29-sf39-sigma025-addendum.md`.
- Not changed: the SF-39 Goal, pre-registration and computed verdicts; the remote single-GPU rule; every frozen infrastructure path (Par2
  baseline outputs, `lagrangian_ensemble` defaults, `scripts/regress` set v2).

## Date of knowledge

Written after the SF-39 results at `sigma^2` 0.25, 1, 2.25, 4 were known and evaluated. SF-39 is used as a pilot: its variances, costs,
geometry and failure modes size and shape SF-43; its estimates are never SF-43 evidence. The travel window of G4 is fixed by geometry (the
`sigma^2 = 4` memory cap limits the travel to 94 `lambda`) and by the SF-39 observation of an early transient in the first quarter of the
travel range (a pilot observation, disclosed); it is not chosen to favour any outcome.

## Context

SF-39 measured `alpha_L` and `alpha_T` at `sigma^2` in {0.25, 1, 2.25, 4} with the Lagrangian consumer. Its transverse result is plausible
(`alpha_T > 0` at 1, 2.25 and 4, not produced by the tracker's integration, the spline representation or its divergence). The evaluation
confirmed these weaknesses:

| id | weakness | evidence |
|---|---|---|
| W1 | Gate 5: `inconclusive` at 1 (offset resolved, -4.8 % to -12.8 %) and at 0.25 by 1.7 % of the margin (BCa and a ratio-consistent contrast read `disagree`; power to reach `agree` 0.14-0.16); arms unmatched in generator, injection, time step and Par2 solver tolerance; `alpha_L` not validated | note, Post-review 1 |
| W2 | production-domain `lambda/h` evidence is one factor of 2; F4 at 2.25 and 4 holds only for an unmeasured order `p >= 1.08 / 1.34`; the order contrast cannot detect non-monotone sequences | Post-review 2 |
| W3 | transverse width not converged at any amplitude (2.25: -23 % from 24 to 8 `lambda`, no wider arm; 4: 12-to-24 unresolved) | Caveats |
| W4 | no comparator independent of the FV operator; no continuum order for the Dirichlet `solve_head` path | Post-review 4 |
| W5 | window `[0.75, 1] tau_b*` differs by amplitude (cross-amplitude statements confounded); noisy window estimator (`alpha_TAM` per-realization CV 0.92 at 1, from section 1: 0.00207 / 1.677 x sqrt(50) / 0.0095) | Post-review 7 |
| W6 | gaps to B&dD's own fit (eq. 6) over the same window: 8 / 18 / 28 / 54 % at 0.25 / 1 / 2.25 / 4, unexplained | section 4 |
| W7 | label semantics: "physical" overreaches; `physical (bounded)` at 0.25 is not a classification | Conclusion |
| W8 | exponential separation of the `h/8` and `h/16` passes (F8), in tension with the non-chaotic picture | Post-review 5 |
| W9 | seeds reused across amplitudes; `N_R` too small for several claims; no pre-stated rule for interval-method discordance; multiplicity uncontrolled | Post-review 1, 7 |
| W10 | reproducibility stops at `reduced/` (Gate 5 per-realization values, raw-input hashes, recomputation scripts not in the repository); every check by AI agents; a false completeness statement (fallback solves) survived every AI layer | Post-review 8, section 0 |
| W11 | the V100 caps the grid at about 5.1e8 cells (`docs/experiments/2026-10-22-sf40-lean-lagrangian-stage.md`); the 2.25 and 4 production grids are at the cap; Par2 needs about 125-157 B/cell (about 2e8 cells) | memory model |
| W12 | the split-half particle component of `alpha_LAM` at 2.25 is 33.6 % (pre-registered limitation); `N_p` never justified | Caveats |
| W13 | the time-window estimators end at the first particle exit; after it the reducer and Par2 keep only the particles still inside (survivor selection) | SF-39 `reduce.py`, Par2 `stats.hpp` |

## Decision

### G1. Two increments (owner)

Options: one increment; two; two plus an exploratory one for W8. **Chosen: two.** SF-42 builds the tooling and verification SF-39 lacked;
SF-43 is the confirmatory study. SF-42 depends on SF-38, SF-39 and SF-41 (it reads SF-39's evidence from the default branch); SF-43
depends on SF-39 and SF-42. W8 is recorded as an open question (G9).

### G2. SF-39 is corrected and closed as an exploratory pilot (owner)

Options: correct and close as pilot; amend SF-39 a third time; drop PR #83. **Chosen: correct and close as pilot** (PR #83, closed
`done` and merged as `104cc254`). Its verdicts stand as computed; SF-43 supersedes its estimates where they overlap.

### G3. The V100 memory cap is accepted and declared (owner)

Options: accept; an engineering increment first (mixed-precision multigrid, about 2x capacity); drop `sigma^2 = 4` from the confirmatory
claims. **Chosen: accept.** Where the production grid is at the cap, the discretization and width bounds come from sequences that fit, and
their transfer to the production configuration is a stated assumption inside the budget, with an interaction test where testable.

### G4. A common physical travel window on control planes (owner, re-operationalized by owner decision of the same day)

Options first offered: a common window in travel distance; the SF-39 window; both co-primary. **Chosen: common physical window.** The
reviews showed that time-window estimators cannot reach a common window (the first exit occurs at a mean travel of about 23 `lambda` at
`sigma^2 = 4` and at most about 77 `lambda` at 0.25 in 104-`lambda` boxes, and the estimators then select survivors, W13). Options then
offered: everything on control planes; `alpha_T` on planes and `alpha_L` in time; planes primary with time-based secondaries. **Chosen:
everything on control planes.** The primary estimands, in both codes, are computed from plane crossings at planes `x0 + d`, `d` in
`[25, 94] lambda` of travel from the injection plane `x0 = 5 lambda` (planes 30-99 `lambda` from the inlet of a 104-`lambda` box):

- `alpha_TP`: half the per-realization least-squares slope, against `d`, of the flux-weighted transverse crossing variance
  `(var x2 + var x3) / 2` (fixed axes, as the SF-39 reducer);
- `alpha_TAP`: the same for the transverse variance orthogonal to the realization's mean crossing displacement (drift-aligned at planes);
- `alpha_LP`: the travel-time longitudinal dispersivity, `(U_r^2 / 2)` times the per-realization least-squares slope of the crossing-time
  variance against `d`, with `U_r` the realization's mean crossing speed over the window (a travel-time estimand, not the spatial `alpha_L`
  of SF-39; it coincides with it asymptotically).

Moments are centred per realization. For `alpha_TP` and `alpha_TAP` (a common plane design, no per-realization factor) the ensemble mean of
per-realization slopes equals the slope of the ensemble-mean moment; for `alpha_LP` (factor `U_r^2`) it does not, and the estimand is the
mean of the per-realization values. Within-realization OLS standard errors are never used (serial correlation 0.90-0.95). The injection
source has a fixed transverse size in `lambda` per amplitude, the same in every arm of that amplitude (`0.8` times the production width,
centred), so that width arms change the box, not the source; a width arm narrower than the source is not run. Par2 gets a plane-crossing
output (SF-42). Time-window estimators (SF-39 style, ending at each realization's first exit) and the B&dD-style window are secondary.

### G5. Confirmatory design rules (proposed; ratified by merge)

1. **Independence and pairing.** New seed ranges, disjoint from every earlier campaign and between amplitudes. Common random numbers only
   where the same seed gives the same field: arms on the same box (the `lambda/h` sequences, the `h_max` passes), the matched Gate 5 arms (both
   codes on the same SF-18 field and grid) and the within-seed factorial factors (injection, time step, solver tolerance). Width arms and the
   generator factor are unpaired (the field's modes depend on the box), and are sized as such.
2. **Sample size fixed in advance.** Absolute precision and power targets, computed with SF-39 / SF-42 standard deviations inflated to their
   80 % upper confidence bound and means deflated to their 80 % lower bound; arms that feed the error budget are allocated jointly with the
   production arms for a stated Gate 4 power. One blinded variance re-estimation is allowed (variance only, upward only, at most 2x, at about
   25 % of `N`; the script outputs standard deviations, never means). No other re-sizing.
3. **Intervals.** The procedure per estimand is fixed in the pre-registration from the SF-42 coverage simulation (no data-dependent switch):
   one-sided miss rate at most 1.25 times the nominal tail, at each tail level actually used (the Holm-level tails of the primary family and the
   `B_k` tails, listed in the pre-registration), judged at the 95 % Monte Carlo upper limit with at least 175 / tail replicates (about 5e4 at
   3.6e-3, 2e5 at 9e-4; host compute budgeted), also under a stress family at the upper bootstrap bounds of skewness and kurtosis, and for the
   whole robust-sign pipeline. BCa is reported,
   never a verdict.
4. **Error budget and the robust test.** For a transverse estimator `e` and level `a`, the robust lower limit is
   `RL_e(a) = L_e(a/2) - sum_k B_k(a / (2K))`, with `L_e` the one-sided sampling lower limit and `B_k` the one-sided upper confidence
   bound of `|bias_k|` for each of the `K` components: discretization `E_h`, width `E_W`, integration `E_int`, and the width x `lambda/h`
   interaction where measured. Each `B_k` must be non-increasing in its level; when a `B_k` is built from two estimated inputs (an order and
   a difference), each input is taken at half of the component's level. The `B_k` come from a bootstrap over seeds that resamples jointly
   every arm sharing seeds. By the union bound, `P(RL_e(a) > alpha_e) <= a` under any dependence, conditional on the stated transfer
   assumptions. The p-value is `p_e = inf{a : RL_e(a) > 0}`; the amplitude p-value is `max(p_TP, p_TAP)` (intersection-union). The Pollock
   contrast is reported, not part of the budget.
   - `E_h` bounds the bias of the **production** level `h_p`. Every grid sequence contains `h_p`, with ratio-2 steps where they fit. If the
     lower confidence limit `p_L` of the observed order lies in a pre-registered range (bootstrap resamples with no real order count as
     outside): when a finer level `h_p / r` exists, `E_h = 1.25 UCB|alpha(h_p/r) - alpha(h_p)| r^p_L / (r^p_L - 1)`; when `h_p` is the
     finest level (the cap), `E_h = 1.25 UCB|alpha(h_p) - alpha(r h_p)| / (r^p_L - 1)`, with the asymptotic range a stated assumption.
     Otherwise `E_h = 3 UCB(max_ij |alpha_i - alpha_j|) / (r_min - 1)`, which also covers non-monotone and oscillating sequences.
   - `E_int`: `2 UCB|alpha(h/16) - alpha(h/8)|` from the paired `h_max` passes (order 1 assumed, conservative; SF-39 measured about 3).
   - `E_W`: the upper confidence bound of `|alpha(W*) - alpha(W_prod)|`, `W*` the widest measured width; the residual beyond `W*` is a stated
     assumption.
5. **Gate 5, matched and paired** (`agree` if the 90 % CI of the log ratio lies inside `+-log(1.10)`, `disagree` if entirely outside,
   `inconclusive` otherwise; `disagree` is descriptive, outside the Holm family, and triggers the attribution requirement). Both codes on
   the same SF-18 field and grid for each seed, window injection, uniform weights (the consumer's twin series), Par2 `cg_rtol` 1e-10, a time
   step fixed a priori, the plane-based `alpha_LP` in both. Equivalence of `log(mu_Lagrangian / mu_Par2)` within `+-log(1.10)` (two
   one-sided tests), standard error from the paired linearization `e_r = L_r / mean(L) - P_r / mean(P)`, `t` on `n - 1` degrees of freedom.
   Sized for power 0.8 at a ratio of 1 at the Holm-adjusted worst level `0.05 / m`, with the 80 % upper bounds of the standard deviations
   and the 80 % lower bound of the pairing correlation (0 unless calibrated on a disjoint seed range); power at a ratio of 0.95 reported. At
   2.25 and 4 the matched arms run on grids Par2 can hold.
6. **Primary family and multiplicity.** The robust sign at `sigma^2` 1, 2.25, 4 (three p-values) and Gate 5 at the four amplitudes (four
   TOST p-values), Holm at a family-wise 5 %. Everything else (`robustly negative`, Gate 5 `disagree`, the factorial, the secondaries) is
   descriptive with intervals.
7. **Gate 4 classes** (`docs/validation/acceptance-gates.md`: physical, numerical or unresolved), for SF-43, in this order of precedence:
   - `physical (restricted)`: Holm rejects the robust-sign null (only at `sigma^2` 1, 2.25 and 4, the primary family); meaning "not an
     artefact of discretization, width or integration within this model, medium, width range and travel window, under the stated transfer
     assumptions";
   - `numerical`: the production sampling interval excludes 0 but the bias-corrected estimate (Richardson- and width-extrapolated) has a
     one-sided upper confidence limit below a pre-registered fraction of the production mean;
   - `unresolved (bounded)`: report `alpha_T <= U + sum B_k` (the expected class at 0.25, which is outside the primary family);
   - `unresolved`: anything else, including estimator discordance.

### G6. Verification additions (proposed; SF-42)

- A manufactured-solution order for the Dirichlet solve path through a new driver executable that reuses the production operator, RHS
  builder, preconditioner, PCG and velocity routine with a manufactured source (no production-code change; `solve_head` takes no source).
  Gated in the L2 norm on interior faces and in the max norm on faces at least two cells from the Dirichlet faces (order at least 1.8); the
  Dirichlet faces are first order by construction and, with the first interior layer, are reported, with a manufactured `K` whose `x`
  derivative is nonzero at the Dirichlet faces.
- A pseudo-spectral cross-check of the FV velocity (SF-19 operator class, same interior scheme as `solve_head`) at `sigma^2` 0.25-4, with
  the SF-18 field imported (the generator's modes, or a sample at the reference resolution, not the FV sample), a preconditioner that
  depends on `|grad ln K|` rather than on the contrast, failure on unconverged residuals, and a self-convergence requirement on the
  reference. What transfers: the interior scheme error; what does not: the Dirichlet boundary treatment, the spline representation, long-box
  end effects (covered by the MMS and the ladders).

### G7. Gate 5 attribution factorial (proposed; SF-42 tooling, SF-43 runs)

Factors at `sigma^2` 0.25 and 1: generator G (Par2 random-spectral against SF-18; unpaired, whole-plot), injection I (full plane against
window; within seed), Par2 time step D (`dt` against `dt/2`; within seed), Par2 solver tolerance R (1e-6 against 1e-10; within seed); a
`2^(4-1)` resolution-IV fraction, analysed as a split-plot (within-seed contrasts by one-sample `t` on per-seed contrasts; G and its
interactions by Welch between generators, with per-generator means of the per-seed contrasts and a two-stratum variance; effects on the log
of cell means, bootstrap over seeds; with the defining relation `I = GIDR` the two-factor interactions are aliased in pairs, for example
`GI = DR`, and are reported as such). Free decomposition: Par2's face-centred
linear interpolation is the Pollock bridge's velocity, so consumer-twin against Pollock-twin isolates the representation, and Pollock-twin
against Par2 isolates Euler stepping plus random-versus-lattice injection.

### G8. Reproducibility, blinding and independence of checks (proposed)

- Per-realization reduced tables (full transverse covariances, displacement moments, plane-crossing moments, solver verdicts) for every arm,
  both Gate 5 codes included, are committed.
- Host-side reduction every few dozen realizations, then pruning of per-particle arrays after recording their sha256, except a
  pre-registered retained subset (at least 5 realizations per arm and the human-review sample); every CSV and JSON is kept.
- A recomputation script independent of the analysis package, starting from raw per-realization moment files, is committed.
- A scripted reconciliation of solver verdicts against the job logs (the SF-39 completeness error).
- The pre-registration is a commit pushed to a draft PR of the SF-43 branch before the first run (GitHub timestamps the push; no
  force-push on that branch); its sha256 is recorded in a new bitácora row.
- Blinded phase: from the first production run to the analysis freeze, only costs, solver verdicts and the blinded variance re-estimation
  may be read; `scripts/remote exec` commands are logged on the host during it.
- The human review includes an independent reduction of at least three realizations from the retained arrays.

### G9. Open questions recorded, not scheduled

- G9a (W8): whether the exponential pass-to-pass separation belongs to the continuum Darcy flow, to the discretization or to a
  finite-distance transient (finite-time Lyapunov exponents on `v_s`, the RT0 flow and the pseudo-spectral flow; SF-34 streamsurface
  confinement).
- G9b (W6): the gap to B&dD, if it persists after SF-43's secondary eq. 6 fit on the long-domain arms and Par2.

### G10. Budget (owner)

Options: up to about 150 GPU-h; a cap at about 80 GPU-h accepting a width-dominated `unresolved (bounded)` at `sigma^2 = 1`; width sequences
only at 2.25 and 4. **Chosen: up to about 150 GPU-h** (one GPU, about 6-7 days), with host-side pruning (about 1 TB of raw per-particle
output would not fit the free disk).

### G11. Order of the PRs (owner)

**Chosen: PR #83 first, then this governance PR**, rebased on the default branch after #83 merges, so that it cites merged evidence.

## Model restrictions (scoped out, declared)

Gaussian covariance; scalar isotropic `K`; medium (b) (Dirichlet in `x`, periodic `y`/`z`) and the SF-18 periodic normalized generator;
purely advective transport; finite travel (no asymptotic claim, Lester et al.'s `t -> infinity` claim out of scope). The periodic transverse
boundaries, the Dirichlet `x` boundaries and the generator class are not varied as factors.

## Weakness to remedy

| id | remedy |
|---|---|
| W1 | G5.5 matched paired Gate 5 at four amplitudes; G7 factorial at 0.25 and 1 |
| W2 | G5.4 `E_h` from ratio-2 grid sequences with observed order or the conservative fallback |
| W3 | G5.4 `E_W` from width sequences (unpaired, sized) and interaction tests |
| W4 | G6 MMS order and pseudo-spectral cross-check |
| W5 | G4 common window on planes; secondaries descriptive |
| W6 | SF-43 secondary eq. 6 fit (long-domain arms and Par2); G9b |
| W7 | G5.7 classes |
| W8 | G9a (not scheduled) |
| W9 | G5.1-G5.3, G5.6 |
| W10 | G8 |
| W11 | G3 and G5.5 (Par2 grids) |
| W12 | SF-42 particle-count pilot; SF-43 pre-registered limit on the particle share |
| W13 | G4 (planes); time-window estimators secondary |
| `exp(-I)` meaning | reported as a weight-sensitivity check only |
| Par2 `alpha_T` | reported in the matched configuration (window injection; displacement-based) |

## Consequences

- New specifications SF-42 and SF-43 (`pending`); the checker's increment count rises from 42 to 44. SF-42 is READY once SF-39 is `done`;
  SF-43 once SF-39 and SF-42 are.
- SF-43's exact numbers (seeds, grids, widths, `N`, procedures, budget) are fixed in its pre-registration at activation, from SF-39 and
  SF-42 data only; the planning numbers of the specifications are not commitments.
- Planning cost: about 100-150 GPU-h for SF-43 (production at the targets of G5.2, ratio-2 ladders, unpaired width sequences, the matched
  Gate 5 and the factorial, the long-domain arms, pilots), within G10.

## Classification (docs/AGENTS.md)

| item | class |
|---|---|
| W1-W13 | confirmed in code / data (recomputed) |
| G1-G4, G10, G11 | accepted scope (owner) |
| G5-G8 | proposed, ratified by merge |
| memory cap; first-exit travel distances | confirmed (measured model; SF-39 `t_b`) |
| planning costs and `N` | estimate (SF-39 data) |
| G9a, G9b | open questions |
