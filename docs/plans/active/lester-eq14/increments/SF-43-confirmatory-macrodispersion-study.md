# SF-43 — Confirmatory Monte Carlo study of macrodispersion with quantified error budgets

- State: `pending`
- Goal: `Medir alpha_L y alpha_T a sigma^2 en {0.25, 1, 2.25, 4} con un Monte Carlo confirmatorio de semillas independientes y estimadores por planos de control sobre una ventana común de recorrido físico, con cada estimación acompañada de un presupuesto de error cuantificado (muestreo, discretización con orden observado, ancho transversal e integración; la representación como contraste reportado), el contraste de alpha_L con Par2 pareado sobre el mismo campo y atribuido por un factorial, y el signo de alpha_T decidido por una prueba robusta a esos sesgos con control de multiplicidad.`
- Depends on: `SF-39, SF-42`
- Unlocks: `none`
- Branch: `science/lester-sf43-confirmatory-macrodispersion`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 4 + Gate 5 as specified by decision 2026-10-29-sf39-remediation-sf42-sf43 (G5.4-G5.7: robust sign with error budget and Holm; matched paired ratio equivalence with power and factorial attribution)`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Decision record `docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md`. SF-39 is the exploratory pilot: its transverse result is
plausible but rests on one production-domain refinement with an assumed order, unconverged widths, amplitude-dependent time windows that
select surviving particles, and a Gate 5 that could neither resolve nor explain an offset (W1-W13). SF-43 re-measures the macrodispersivities
with independent seeds and a design fixed before any run, so that every reported number carries a quantified error budget and the few
primary claims are confirmatory at a stated family-wise error rate. Model: purely advective transport in smooth, locally isotropic Gaussian
log-conductivity fields (`C(r) = sigma^2 exp(-(r/lambda)^2)`, B&dD eq. 1), medium (b), periodic transverse boundaries; `alpha_T` is not
presupposed (`docs/theory/lester-2023-key-claims.md`, `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md`).

## Preconditions

- SF-39 and SF-42 `done` on the default branch; SF-42's acceptance D (MMS) and E (spectral) met, or owner decisions recorded on budgeting
  them.
- A pre-activation review (docs only, no data, no GPU) merged on the default branch, as for SF-39 (decision 2026-10-27); its merge commit is
  the increment base.
- The pre-registration (artifact `preregistration.md`) committed and pushed to a draft PR of the SF-43 branch before the first run (no
  force-push), its sha256 in a new bitácora row; it fixes every number left open below from SF-39 and SF-42 data only.
- The owner budget of decision G10 (up to about 150 GPU-h, one GPU).

## In scope

1. **Estimands (G4), on control planes in both codes.** For each `sigma^2` in {0.25, 1, 2.25, 4}: `alpha_TP`, `alpha_TAP` (transverse) and
   `alpha_LP` (travel-time longitudinal), per-realization least-squares slopes over planes `x0 + d`, `d` in `[25, 94] lambda` from the
   injection plane, in boxes of 104 `lambda` with 5-`lambda` buffers (definitions in decision G4; transverse crossing variance `(var x2 +
   var x3) / 2`; for `alpha_LP` the estimand is the mean of per-realization values); moments centred per realization; the stationarity
   reading (the two halves `[25, 59.5]` and `[59.5, 94]`) reported as an interval, no claim without a pre-registered equivalence margin.
   Secondary (descriptive): SF-39-style time-window estimators ending at each realization's first exit; the B&dD-style window; the
   displacement-based transverse variants (if SF-42's estimator-noise pilot makes one of them primary, the pre-registration says so); the
   `exp(-I)` weight-sensitivity check (same paths, not a comparator); Par2's transverse dispersivity in the matched configuration (window
   injection, displacement-based).
2. **Production configurations within the V100 cap (G3)** (planning; the pre-registration fixes them). The source is a centred square of
   fixed size `S` in `lambda` per amplitude, the same in every arm of that amplitude (decision G4), flux weights, lattice `M` admissible by
   the SF-42 seed-lattice rule:

   | `sigma^2` | width (`lambda`) | `lambda/h` | grid | cells | source `S` |
   |---|---|---|---|---|---|
   | 0.25, 1 | 24 | 10 | 1040x240x240 | 6.0e7 | 19.2 `lambda` |
   | 2.25 | 24 | 16 | 1664x384x384 | 2.45e8 | 19.2 `lambda` |
   | 4 | 12 | 32 | 3328x384x384 | 4.9e8 (cap) | 9.6 `lambda` |

3. **Discretization budget `E_h` (G5.4).** Grid sequences on the production box that contain the production level, ratio-2 steps where they
   fit, paired by seed: planning `sigma^2` 0.25 and 1 at `lambda/h` 5 / 10 / 20 (production in the middle; the finest 2080x480x480, 4.8e8
   cells); 2.25 at 8 / 16 plus 4 or 20 (chosen in the pre-registration from the SF-42 grid pilot); 4 at 8 / 16 / 32 (production finest, at the
   cap; the asymptotic range is a stated assumption). Sized for a target `E_h` relative to the planning `alpha_T`.
4. **Width budget `E_W` (G5.4).** Unpaired width arms with the amplitude's fixed source (no arm narrower than the source): widths 24 and 48
   `lambda` at `lambda/h` 10 for `sigma^2` 0.25 and 1 (the production `lambda/h`; 48 at 10 is 2.4e8 cells); at 2.25, widths 24 and 48 at
   `lambda/h` 8 and 10 (a 2x2 interaction test; the transfer to 16 is a stated assumption); at 4, widths 12 and 24 at `lambda/h` 8 and 16
   (2x2; the transfer to 32 is a stated assumption). `E_W` bounds `|alpha(2 W_prod) - alpha(W_prod)|`. Sized jointly with the production
   arms for the Gate 4 power target.
5. **Integration and representation.** `h_max` passes 8 / 16 / 32 on a subset, with the binding step control recorded; `E_int` as decision
   G5.4; the Pollock bridge on every realization (reported, not in the budget).
6. **Sampling (G5.2, G5.3).** `N` fixed in the pre-registration from absolute targets, with SF-39 / SF-42 standard deviations at their 80 %
   upper bound and means at their 80 % lower bound; one blinded variance re-estimation allowed. Planning (from SF-39 plane data on
   `[25, 94]`): 90 % half-width of `alpha_TP` at most about 6e-4 / 2.8e-3 / 6.6e-3 `lambda` at `sigma^2` 1 / 2.25 / 4 (about 10 % of the
   planning means) and 2.5e-4 at 0.25, giving `N` of the order of 1000-1650 / 90-110 / 30 / 1100. The interval procedure per estimand is the
   one the SF-42 coverage simulation qualifies (G5.3); BCa is reported only.
7. **Gate 5 (G5.5).** Matched and paired: both codes on the same SF-18 field and grid per seed, window injection, uniform weights (consumer
   twin series), Par2 `cg_rtol` 1e-10, Par2 time step fixed a priori in the pre-registration, `alpha_LP` on the common planes in both. Grids
   Par2 can hold: 1040x240x240 at 0.25 and 1 (`lambda/h` 10), 1248x288x288 at 2.25 (12), 1664x192x192 at 4 (16, width 12), each with the
   amplitude's source. `agree`, `disagree` and `inconclusive` as decision G5.5. Paired log-ratio TOST within `+-log(1.10)`, sized at the
   Holm worst level `0.05 / 7` (planning, unpaired worst case: about 170 / 60 / 30 / 100 per arm; fewer with the pilot's pairing
   correlation). The SF-39-style unmatched comparison is reported for continuity.
8. **Gate 5 attribution (G7).** At `sigma^2` 0.25 and 1, Par2 cells of a `2^(4-1)` resolution-IV fraction of generator, injection, time step
   and solver tolerance, analysed as a split-plot; plus the free decomposition consumer-twin / Pollock-twin / Par2.
9. **Distance and B&dD (W6).** At `sigma^2` 1 and 2.25, boxes of about 208 `lambda` at the production `lambda/h` and width 24 (planning 60-100
   realizations): `alpha_LP`, `alpha_TP` over `[25, 94]` and `[94, 194]`; a secondary fit of B&dD eq. 6 to the time-window `alpha_L(t)` of
   these arms and of Par2; the remaining gap recorded (decision G9b).
10. **Particles.** `M` per configuration from SF-42's particle pilot; the split-half particle share of each primary estimator's variance at
    most a pre-registered fraction, else `M` is raised before the first run.
11. **Reproducibility (G8).** Per-realization reduced tables committed; host-side reduction and pruning with sha256 and a retained subset;
    the independent recomputation; the solver-verdict reconciliation; `exec` logging during the blinded phase; the human review's
    independent reduction of at least three realizations.

## Out of scope

- Asymptotic `alpha_T` and Lester et al.'s `t -> infinity` claim; finite Péclet; other covariance models; anisotropic `K`; `sigma^2 > 4`.
- The model restrictions of the decision record (periodic transverse boundaries, Dirichlet `x` boundaries, generator class) as factors.
- The open questions G9a and G9b (only recorded); memory engineering (G3).
- Changing SF-39's verdicts.

## Files and symbols

- Runners: `lagrangian_ensemble` (medium (b); window and lattice by the SF-42 seed-lattice rule), `macroflow3d_pipeline` (Par2 with SF-42's
  SF-18 field, window, tolerance and plane-crossing options).
- Tooling: SF-42's artifact scripts (reader and reduction, pruning, `mcstat/`, recomputation, reconciliation); a new campaign script modelled
  on `docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/scripts/run_sf39.sh` (parts, bookkeeping, restart, md5 gate, host
  reduction and pruning, no `git -C`).
- New artifact root `docs/experiments/artifacts/<date>-sf43-confirmatory-macrodispersion/` and note
  `docs/experiments/<date>-sf43-confirmatory-macrodispersion.md`.

## Implementation specification

1. **Pre-registration before any run** fixes: seeds (disjoint ranges per amplitude and arm family, reserved replacements), every arm (grid,
   width, `lambda/h`, `--ell`, `--eps`, window, `M`, planes, `--hmax-div`), the travel window, `N` per arm with the sizing computation and the
   joint allocation, the interval procedure per estimand with its coverage evidence, the budget rules of G5.4 (order range, safety factors,
   bootstrap scheme), the Gate 4 classes of G5.7 (with the `numerical` fraction), the Gate 5 rule of G5.5, the primary family and Holm, the
   factorial design and analysis, every reading rule, the retained-array subset and the planned cost.
2. **Blinded phase (G8).** From the first production run to the analysis freeze only costs, solver verdicts and the blinded variance
   re-estimation may be read; `exec` commands are logged. No outcome-dependent decision.
3. **Gate 4.** For each transverse estimator and amplitude, the robust lower limit and p-value of G5.4; the amplitude p-value
   `max(p_TP, p_TAP)`; the class of G5.7 in its order of precedence (`physical (restricted)` only when Holm rejects, at `sigma^2` 1, 2.25, 4;
   at 0.25 `numerical`, `unresolved (bounded)` or `unresolved`); always qualified "within this model, medium, width range and travel window,
   under the stated transfer assumptions".
4. **Gate 5.** The paired TOST of G5.5 at every amplitude, its p-value in the Holm family; the factorial decomposition at 0.25 and 1.
5. **Failure rules.** Failure criteria fixed in advance and applied blind to `alpha`; failure rate per arm reported; an unusable solve is
   replaced by the next reserved seed (disclosed); above 1 % failures in an arm, a worst-case imputation at the 5th / 95th percentiles of
   `alpha` is reported; a sensitivity without fallback-accepted solves; a halted arm stops and goes to the owner.

## Expected numerical effect

No result is predicted. The SF-39 pilot suggests `alpha_TP` of order 0.006 / 0.028 / 0.066 `lambda` on `[25, 94] lambda` at `sigma^2` 1 /
2.25 / 4 and near 0 at 0.25; these size the study and are not hypotheses.

## Validation commands

```bash
# local: tooling tests and dry runs of the campaign script
python3 -W error -m unittest discover -s <SF-42 scripts> -t <SF-42 scripts>
bash <A>/scripts/run_sf43.sh build/v100-release <part> --dry-run
# remote, detached, one job at a time (REMOTE_GPU_WAIT=7200; REMOTE_GPU_MIN_FREE_MIB=30000 for CUDA parts)
scripts/remote --increment SF-43 run sf43-<part> -- "bash <A>/scripts/run_sf43.sh build/v100-release <part>"
scripts/remote --increment SF-43 wait sf43-<part>
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- **A. Design fidelity.** Every arm, `N`, window, procedure and rule as pre-registered (pre-registration commit pushed before the first
  run); every deviation recorded with its reason; no threshold or rule changed after data.
- **B. Budgets.** Every primary and secondary estimand at every amplitude reported with its sampling interval, `E_h` (observed order and its
  interval, or the fallback), `E_W` (and the interaction tests), `E_int`, the Pollock contrast and the stated transfer assumptions.
- **C. Gate 4.** The robust-sign p-values, the Holm decisions and the G5.7 class per amplitude, with the achieved Gate 4 power reported.
- **D. Gate 5.** The paired ratio-equivalence decision per amplitude with achieved power; the factorial decomposition at 0.25 and 1. An
  unexplained `disagree` keeps SF-43 active until attributed or decided by the owner.
- **E. Precision.** Achieved half-widths against the targets (reported, not extended after the fact).
- **F. Independent checks.** The recomputation from raw moments reproduces every reduced value it covers to 1e-10 relative; the solver
  reconciliation matches the job logs; the human review records its independent reduction of at least three realizations.

## Regression surface

- No runner change is expected; any change is a disclosed corrective with byte-identity checks of existing paths.
- Claims: SF-39's estimates are superseded where SF-43 overlaps; the dashboard and the theory notes cite SF-43 for confirmatory values.

## Failure and rollback policy

- An unmet SF-42 precondition without an owner decision blocks activation.
- A projected cost above the owner budget stops before the extension and goes to the owner, `N` unchanged until decided.
- Gate 5 `disagree` without attribution, or a coverage requirement no procedure meets, keeps SF-43 active.

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

Acceptance of SF-43 provides the project's confirmatory finite-distance macrodispersivities for this model and enables the specification of
the finite-Péclet phase and, if the owner schedules them, of the open questions G9a and G9b.

## Bitácora

Append entries; do not rewrite prior observations. Store large raw outputs as artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-29T21:00Z | `pending`; governance PR (docs only) | Specification created on the governance PR of decision `2026-10-29-sf39-remediation-sf42-sf43.md`, after the SF-39 post-review evaluation and three adversarial reviews of the draft. Planning numbers (geometry within the cap, `N`, cost) from SF-39 data; not commitments. | Owner decisions G1-G4, G10, G11 of 2026-10-29. | SF-39 and SF-42 `done`; pre-activation review; pre-registration. |
