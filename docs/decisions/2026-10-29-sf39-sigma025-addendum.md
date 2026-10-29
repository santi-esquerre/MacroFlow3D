# SF-39 addendum: sigma^2 = 0.25 added to the macrodispersion study

- Status: accepted (owner instructions of 2026-10-29 in the SF-39 orchestrator session; ratified by the human merge of PR #83)
- Date: 2026-10-29
- Deciders: owner; recorded and executed by the orchestrator
- Amends, for SF-39 only and without editing those records:
  - the amplitude set fixed by `2026-10-17-lagrangian-consumer-sigma4.md` ("`sigma_Y^2` in {1, 2.25, 4}");
  - the statement "Not changed: the Goal of SF-39" of `2026-10-27-sf39-pre-activation-decisions.md`;
  - the rule of `docs/AGENTS.md` to preserve an increment's exact Goal, and the rule of `AGENTS.md` (Lester protocol item 8) to keep
    a human-review increment `awaiting_review` until approval. The Goal is edited and the increment returns to `active` while the
    addendum is executed;
  - the post-pilot budget stop of E6 (2026-10-24) and F9 (2026-10-27), for `s025` only (item 4 below);
  - it extends D1, D10, D14, F11 and F12 to `s025` as stated in the addendum.

  Every other item of those records, of D1-D21 (2026-10-23), E4-E12 (2026-10-24), F1-F14 (2026-10-27) and of the remote single-GPU
  decision (2026-10-27) stands. D12's Par2 `dt/2` sensitivity (SF-38 E) exists only at 1, 2.25 and 4. At 0.25 it is reported as
  not available; the `sigma^2 = 1` result is cited as context.

## Context

- SF-39 is `awaiting_review` (PR #83; source-bearing audited head `40f5996c`). Its results at `sigma^2` 1, 2.25 and 4 are known
  (`docs/experiments/2026-10-27-sf39-macrodispersion-study.md`).
- The owner wants `sigma^2 = 0.25` in the same study, so that it covers the low-variance end of the B&dD series
  (`docs/theory/beaudoin-de-dreuzy-2013-key-claims.md`: Table 1 `alpha_LAF` 0.24 at `t_LF` 1.35; Table 2 `alpha_TA < 1e-3`).
- Confirmed in the repository:
  - No increment ran the Lagrangian consumer at 0.25: SF-37, SF-38 and SF-41 ran 1, 2.25 and 4 only.
  - There is no Par2 configuration for 0.25.
  - The SF-39 checkpoint writes `checkpoint/tau_b_star.json` once, so a new amplitude cannot be frozen in it.
  - The increment checker requires a unique, non-empty Goal; it does not forbid editing it.
  - No earlier record amends the Goal of an increment that is already `active` or `awaiting_review`. This is the first.
- Three independent reviews of the draft addendum (governance fidelity, executability against the scripts, statistics and physics)
  found that, under the unchanged rules, `alpha_T` at 0.25 is expected to be of order 1e-3 `lambda`. For an `alpha_T` that small:
  - the absolute floors decide the ladder verdicts;
  - `nr_rule` gives `N_R` of about 20, with a 90 % half-width comparable to `alpha_T`;
  - the rule-D label is then largely driven by sampling.

  The owner was shown this estimate (an extrapolation from 1, 2.25 and 4, not a measurement) together with the options of a fixed
  `N_R` of 200 or 500. The owner chose to keep `nr_rule` unchanged.

## Decision

1. **Goal (accepted scope).** The SF-39 Goal reads `sigma^2 en {0.25, 1, 2.25, 4}` instead of `{1, 2.25, 4}`. Nothing else in the
   Goal changes.
2. **Settings of the new amplitude `s025` (accepted scope).** They are fixed in
   `docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/preregistration-addendum-s025.md` before any `s025` check or run:
   - The geometry is the `sigma^2 = 1` geometry in every respect: production 2048x256x256 at `lambda/h` 10; narrow rungs 5 / 10 / 20
     on the 12.8 `lambda` section; coarse twin at 5; plane lists P1 and P4.
   - `--eps 0.5`, with the seed rules of the other amplitudes.
   - This geometry is conservative for the absolute discretization error, because the resolution need decreases with `sigma^2`
     (`docs/experiments/2026-10-23-resolution-vs-variance.md`: a scaling argument, not a bound). The relative error of a small
     `alpha_T` is not bounded by that argument.
   - Par2 uses the SF-38 `s1` configuration with `stochastic.sigma2: 0.25`.
3. **Rules (accepted scope).** Unchanged for `s025`:
   - thresholds A-E, rule D and the amplitude label;
   - F4, F6, F7, F8 and F11;
   - every margin, CI level, the bootstrap seed, and `nr_rule` with its cap (owner choice above).

   F10 is not run at 0.25; it belongs to `sigma^2 = 4`. The addendum pre-states how the verdicts are read at an amplitude where the
   absolute floors govern. These are interpretation notes and anticipated outcomes; no threshold is changed.
4. **Checkpoint and budget (accepted scope).**
   - The `s025` pilot is checkpointed blind into `checkpoint/s025/`, which contains its own write-once `tau_b_star_s025.json`. The
     existing checkpoint files are not touched.
   - The owner budget of 2026-10-28 ("uso libre del servidor; hasta varios días de cómputo; una gpu a la vez") applies. If the `s025`
     projection (GPU-h and bytes) is within it and within the host's free disk, then f = 1 and the printed `N` are final, with no
     stop. Otherwise the section 6 rule of the preregistration applies, and anything beyond it goes to the owner.
   - The `N` and the new `budget.json` sha256 are committed in a bitácora row, and synced, before any `s025` extension run.
5. **Existing amplitudes (accepted scope).** No setting, threshold, run or result of `s1`, `s225` or `s4` changes. The amended
   analysis must reproduce their outputs of evidence commit `4dbf31f2` identically, as defined in the addendum (A6). Any difference
   is a defect and stops the work.
6. **Procedure (accepted scope).**
   - This record, the addendum, the specification amendment and a bitácora row are committed before any code, check or run.
   - The script changes follow worker -> orchestrator audit -> integration. They are integrated, and their tree recorded, before the
     `s025` checks; any later script change is a disclosed corrective.
   - SF-39 is `active` while executing and returns to `awaiting_review` on the same PR #83 with a new source-bearing head. No agent
     merges.

## Consequences

- **Post-hoc addition, disclosed.** The amplitude was added after the results at 1, 2.25 and 4 were known.
  - Mitigation: every `s025` setting and rule is fixed before its first run and is identical to those of the other amplitudes, and
    nothing about them changes.
  - `s025` reuses the `s1` seeds on the `s1` geometry. Each `s025` log-conductivity field is therefore exactly 0.5 times the `s1`
    field of the same seed (runner: `ln K = eps Y`), so `s025` and `s1` estimates are correlated. Any cross-amplitude statement is
    descriptive and paired by seed.
- **No prior verification at 0.25.** The convergence evidence at 0.25 is SF-39's own: the narrow `lambda/h` ladder, the coarse twin
  and the `h_max` ladder. There is no SF-37-type verification at that amplitude, and this is a stated caveat.
- **Limited informativeness of `alpha_T` at 0.25 (anticipated, owner-accepted).**
  - `converged` there means a grid effect below the absolute floors (1.25e-3 `lambda` narrow, 3.54e-3 twin), not agreement within 5 %.
  - `numerical` is practically unreachable, so a `physical` label does not exclude a mostly numerical `alpha_T`.
  - With `N_R` about 20, `physical (positive)`, `physical (bounded)` and `unresolved (estimator-discordant)` are each plausible from
    sampling alone.
  - The note reports the refinement differences as fractions of `alpha_T`, with their intervals, beside the label.
  - `alpha_L` (relative margins) and Gate 5 remain informative.
- **Gate 5.** At 0.25 Gate 5 is not sized by its own rule (`n = N_R`). An unexplained `disagree` keeps all of SF-39 active (RR6).
- **Cost.** About 2-3 extra GPU-h on one shared GPU, projected from the `s1` per-realization costs.
- **Review.** The review covers a larger PR, and the human approval must name the new source-bearing head.
