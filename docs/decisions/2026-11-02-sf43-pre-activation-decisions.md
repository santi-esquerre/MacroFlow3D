# SF-43 pre-activation decisions: displacement-basis estimands, a null at the first-order value, a qualified Gate 5, sharpened budgets, a two-GPU first stage and a design that fits the deadline

- Status: **proposed** (orchestrator recommendations of the pre-activation review; owner decisions pending the single consultation of
  2026-11-02). Items marked *owner* are decided by the owner among the options listed; items marked *pre-registration latitude* fill
  what decision 2026-10-29 G5.2-G5.4 explicitly leaves to the pre-registration; all are ratified by the human merge of the PR that adds
  this record. The outcome of the consultation is recorded in the section "Owner decisions of the consultation" before merge.
- Date: 2026-11-02
- Deciders: owner (H2-H5, H9, H10, H14, H20); orchestrator (drafting; H1, H6-H8, H11-H13, H15-H19 within the latitude above)
- Evidence: the pre-activation review [`docs/experiments/2026-11-02-sf43-pre-activation-review.md`](../experiments/2026-11-02-sf43-pre-activation-review.md)
  (literature in its verified bibliography; own calculations in its `scripts/`; seven independent analyses of the committed SF-39 and
  SF-42 data; adversarial reviews).
- Builds on: `2026-10-29-sf39-remediation-sf42-sf43.md` (G1-G11), `2026-10-29-sf39-sigma025-addendum.md`,
  `2026-10-27-remote-single-gpu.md`, `2026-10-27-sf39-pre-activation-decisions.md` (format), SF-42's findings (a)-(j) and the owner
  decisions of 2026-11-01 recorded in the SF-42 bitácora (CPU-only slot-hours not counted toward G10).
- Not changed: the SF-43 Goal; G1-G3, G4 except where H2 uses the latitude of specification item 1, G5.1, G5.5's margin and level,
  G5.7's classes (H4 extends their amplitude range), G6-G9, G11; SF-39's and SF-42's verdicts; every frozen production path (no runner
  change is planned).

## Date of knowledge (integrity statement)

Written after SF-39's results and SF-42's pilots were committed (`104cc254`, `ef87a1bc`), before any SF-43 seed existed. SF-39 and
SF-42 data are used as pilots only (variances, pairing correlations, costs, memory, failure modes, per-basis noise). Three items were
prompted by readings of those data and are disclosed as data-informed: H2 (from the SF-42 standard-deviation ratios, not from any
mean), H3 (from theory; it costs no power at the pilot magnitudes, which were known) and H4's admission of `sigma^2 = 0.25` (made
possible by H2). Each is fixed before SF-43 data exist and none relaxes a decision rule: H2 keeps the estimand (section H2), H3 makes
the null harder to reject, H4 adds a hypothesis to the Holm family.

## Context

The review found that the design of the specification (position-basis estimands sized to 10 % precision, the admissible lattice of
SF-42 Finding (a)) costs 147-185 GPU-h with the measured overheads and does not fit before the owner's deadline (analysed results by
2026-11-09 00:00 local), and that four of its elements are ill-posed or unqualified as they stand: the position-basis transverse
estimators waste most of their variance on a term with zero mean; the sign null is already rejected by linear theory; no Gate 5
procedure qualified at the planned `n` (SF-42 Finding (j)); and `E_h` has four gaps against the verification standards.

## Decision

### H1. Integrity and scope (orchestrator)

As above. The planning numbers of the review are not commitments; the pre-registration fixes every number from SF-39/SF-42 data and
this record.

### H2. Displacement basis primary for the transverse estimands (owner)

`alpha_TP` and `alpha_TAP` are computed from the transverse **displacement** `x23(d) - x23(seed)` at each control plane (the
`disp` basis of `sf42red`), not from the crossing position. For uniform weights the two bases have the same expectation exactly (the
covariance of the seed position with the displacement has zero mean by transverse stationarity); for flux weights to `O(1/n_eff)`.
The `pos - disp` contrast is reported per amplitude. Noise: the across-realization SD is smaller by 37 / 9.8 / 3.7 / 1.14 at `sigma^2`
0.25 / 1 / 2.25 / 4 (SF-42 section G, R3), so the 10 % precision targets of G5.2 are met at `N` of about 7-30 instead of 1000-1650 /
90-110 / 30 / 1100. The position basis is reported as a secondary from the same runs.
Options: (a) displacement primary **[recommended]**; (b) position primary, as planned (infeasible before the deadline at the targets);
(c) both co-primary through an intersection-union (needs the position-basis `N`).

### H3. Robust-sign null at the first-order value (owner)

For each transverse estimator and amplitude the null is `alpha_e <= alpha_T^FO(sigma^2)`, with `alpha_T^FO = 7.992e-5 sigma^2 lambda`,
the first-order, unbounded-domain, one-particle value of the same estimand on the pre-registered plane list (`firstorder_window.py`;
an upper value of the first-order effective estimand, so the null is conservative). Rejection requires `RL_e(a) > alpha_T^FO`.
Reason: linear theory predicts `alpha_T > 0` at every finite distance (2.0e-5 / 8.0e-5 / 1.8e-4 / 3.2e-4 `lambda` over `[25, 94]`), so
the sign null of G5.4 cannot separate linear theory from a nonlinear transverse dispersion; the shifted null can. The pipeline's
coverage is shift-invariant, so SF-42's qualification carries over unchanged. Reported descriptively: `RL / alpha_T^FO`, the halves
against the first-order ratio 3.57, and (distance arms) the window ratio against 6.92.
Options: (a) null at `alpha_T^FO` **[recommended]**; (b) null at 0 (G5.4 as decided); (c) null at `10 alpha_T^FO`.

### H4. Primary family (owner)

Robust sign (H3) at `sigma^2` 0.25, 1, 2.25 and 4, and Gate 5 at the four amplitudes: `m = 8`, Holm at a family-wise 5 %. The family is
fixed: a hypothesis that cannot be tested (arm below its floor, halted arm, no qualified procedure) enters with `p = 1`. G5.7 applies
unchanged at every amplitude of the family (`physical (restricted)` becomes reachable at 0.25, which G5.7 had excluded because the
position basis could not resolve it).
Options: (a) `m = 8` with 0.25 **[recommended]**; (b) `m = 7` as G5.6 (0.25 descriptive); (c) Gate 5 out of the family (`m = 4` or 3).

### H5. Gate 5 qualified by the Fieller form (owner)

The paired ratio-of-means TOST of G5.5 (margin `+-log 1.10`, level as G5.5) is computed in **Fieller form**: the two one-sided
hypotheses `mu_L / mu_P >= 1.10` and `<= 1/1.10` are tested as one-sample tests on `D_r = L_r - 1.10 P_r` and `L_r - P_r / 1.10`
(Hauschke et al. 1999; Fieller 1954; Berger & Hsu 1996). The estimand is unchanged and the delta-method linearization disappears; at the
boundary `D_r` has exactly the distribution of SF-42's `sampling` cells, so their coverage evidence applies. **Procedure**, fixed by a
deterministic rule applied to a restricted coverage run before the pre-registration (H15): the first of (i) Hall's (1992)
skewness-corrected t, (ii) bootstrap-t at the Holm level, (iii) bootstrap-t at half the Holm level (already qualified at `n` in {100,
200, 500}) that qualifies at the pre-registered `n` under `normal`, `stress` and `stress-`. **`n`** = 150 / 150 / 150 / 250 pairs at
`sigma^2` 0.25 / 1 / 2.25 / 4 (coverage floors; at 4 the power target with the conservative pairing inputs; the consumer CV there
rests on one seed of eight). Gate 5 arms are not re-sized by the blinded re-estimation (equivalence tests can inflate under it:
Friede & Kieser 2003). Pre-registered expectation: at `sigma^2 = 4` the plane-based `alpha_LP` is dominated by the slow tail
(`d / alpha_L` = 2-7 over the window) and a `disagree` is plausible; it is descriptive, triggers the attribution of G5.5 (free
decomposition consumer twin / Pollock twin / Par2 at every amplitude; G7 at 0.25 and 1) and, per the specification, keeps SF-43 active
until attributed or decided by the owner.
Options: (a) as above **[recommended]**; (b) as above with the coverage run inside the production job (rule pre-registered, resolved
after launch); (c) Gate 5 descriptive (outside the family; H4 (c)).

### H6. `E_h` rules (pre-registration latitude of G5.4)

- Ladders, paired by seed with the production arm: `lambda/h` 5 / 10 / 20 at `sigma^2` 0.25 and 1 (production in the middle);
  4 / 8 / 16 at 2.25 (production finest; the third level of the specification's alternatives "4 or 20" chosen as 4: SF-42 E shows
  second-order flux convergence from `lambda/h = 4` at every amplitude, the rung costs 5 s per realization against 140 s for 20, and
  production-finest is the standard GCI configuration); 8 / 16 / 32 at 4 (production finest) plus a coarse rung at `lambda/h = 4`.
- Order: lower confidence limit `p_L` per triple with constant ratio 2; **`p_use = min(p_L over the triples available, 2)`** (cap at the
  formal order: Eça & Hoekstra 2014; Roy 2005); range `[0.5, 2.1]` for `p_L`; resamples without a real order (non-monotone or
  oscillating differences) count as outside the range (already so in `mcstat.LadderBound`).
- Finest-level branch (`sigma^2` 2.25, 4): `E_h = 1.25 max(UCB|a_P - a_{P-1}|, 2^(-p_use) UCB|a_{P-1} - a_{P-2}|) / (2^p_use - 1)`
  (Giles 2015: the maximum over the last extrapolated differences). Finer-level branch (0.25, 1) and fallback as G5.4.
- Diagnostics reported, not gating: the `lambda/h = 12` level at 2.25 (the Gate 5 consumer arm, same seeds) against the fitted error
  law; the residual of a four-level fit at 4.
- All three changes make `E_h` larger or equal, so SF-42's `LadderBound` qualification holds. Component method: Student (the only
  method that qualified the unpaired and interaction contrasts; it also qualified the ladders).

### H7. `E_W` (pre-registration latitude)

As the specification: widths 24 and 48 `lambda` at `lambda/h` 10 (0.25, 1; the production arm is the 24 cell); 2 x 2 of widths 24 / 48 at
`lambda/h` 8 / 10 at 2.25 and 12 / 24 at `lambda/h` 8 / 16 at 4, with the interaction component; `E_W` taken at the finest measured
`lambda/h` of each amplitude; the transfer to the production `lambda/h` is a stated assumption. Cells on the production box share the
production seeds; wider boxes have their own seed ranges (unpaired, as G5.1).

### H8. `E_int` (pre-registration latitude)

`h_max` passes 8 / 16 / 32 on the first 30 production seeds of each amplitude; `E_int = 2 UCB|alpha(h/16) - alpha(h/8)|` (G5.4).

### H9. Sample sizes and design variant (owner)

Variant A **[recommended]**: production 150 at every amplitude; `E_h` rungs 100; `E_W` cells 150; `h_max` 30; Gate 5 as H5; G7 60 per
cell at 0.25 and 1; distance arms 100 at 1 and 2.25. About 54 GPU-h planning (74 with every pessimistic factor); 37-49 h of wall.
Variant B (faster, less margin in the budgets): production 100, `E_h` 80, `E_W` 100, `h_max` 20, Gate 5 150/150/150/205, G7 30 per
cell, distance 60: about 38 GPU-h, 27-35 h. Variant C (more precision in the budgets and a decay test at `sigma^2 = 4`): production 300
(200 at 4), `E_h` 150, `E_W` 300, `h_max` 40, Gate 5 300/300/200/300, distance 150 plus a 208-`lambda` arm at `sigma^2 = 4` on
`lambda/h = 16` (a scope extension of item 9): about 88 GPU-h, 60-86 h. One blinded variance re-estimation (G5.2) on the production
and `E_W` arms at 25 % of their `N`.

### H10. Sources and lattice (owner; SF-42 Finding (a))

`sigma^2` 0.25 and 1: source 19.2 `lambda` (window 0.1-0.9), `M = 128` (the only common admissible value; 1.37 x SF-39's seed density).
`sigma^2` 2.25: **source 19.08 `lambda`** (window 0.1025-0.8975 at width 24, 0.30125-0.69875 at width 48; 0.795 x the production width)
with `M = 100`; at 19.2 `lambda` no `M` is admissible on every arm. `sigma^2` 4: 9.6 `lambda`, `M = 100`.
Options: (a) 19.08 `lambda` at 2.25 **[recommended]**; (b) 19.2 `lambda` with a per-arm `M` (breaks the common source discretization
across the arms of the amplitude); (c) an `E_W` `lambda/h` without a factor 5.

### H11. SF-42's 3x PCG rule (pre-registration latitude; SF-42 Finding (f))

Accepted explicitly for 1040x240x240 at 0.25 (ratio 3.06) and 2080x240x240 at 1 (3.33): the solves converge (relative residual <=
4e-11), the cost is measured and in the plan, and a grid with more factors of two would change `lambda/h` or the width and invalidate
the grid, memory, cost and lattice pilots.

### H12. Par2 (pre-registration latitude; SF-42 Findings (d), (h))

Horizon factor 3.6, `dt = 0.0244140625 dx` (SF-39's rule scaled to the grid), `cg_rtol` 1e-10, `cg_max_iter` 2000; per-realization
crossing completeness at the last plane reported, a realization below 0.999 flagged and a sensitivity without it reported.

### H13. Schedule, priorities, cutoff and guards (pre-registration latitude)

One job. Priority 1: every arm that feeds the primary family; 2: diagnostics and blinded-re-estimation extensions; 3: descriptive
arms (distance, then G7). Within a priority, units are interleaved by progress fraction and the seed blocks of a common-random-number
group run adjacently. Each arm has a pre-registered confirmatory `N`; the analysis uses the longest complete prefix of its seed order;
an arm below its floor is descriptive and its hypothesis enters Holm with `p = 1`. Clock cutoff **2026-11-08 16:00 local (19:00Z)**,
one hour of grace for running units, then the frozen analysis. Budget guard 150 GPU-h (G10); disk guard; OOM retries not counted as
failures; unusable solves replaced group-wide by reserve seeds; failure rules of the specification.

### H14. Two-GPU first stage (owner decision of 2026-11-02, recorded)

For SF-43 only: stage 1 of the single job holds two GPUs, chosen by `scripts/remote` (`REMOTE_GPU_COUNT=2`, first GPU
`REMOTE_GPU_MIN_FREE_MIB=30000`, second `REMOTE_GPU_MIN_FREE_MIB_2=17000` because its lane runs only grids of at most 16 GB,
`REMOTE_GPU_COUNT_MIN=1`: after the `REMOTE_GPU_WAIT=7200` wait it starts with one GPU), at most 30 h of wall from the stage-1 start
(the engine never starts a unit that would end later) and less when its queue empties; then one GPU. The job narrows the granted list
per lane; no command pins a GPU or writes `CUDA_VISIBLE_DEVICES`. GPU-h of stage 1 = 2 x its wall. This supersedes items 1 and 3 of
`2026-10-27-remote-single-gpu.md` and the "one GPU" of G10 for this scope only; it expires at the end of SF-43's stage 1 and in any case
on 2026-11-09.

### H15. Coverage (pre-registration latitude; SF-42 Finding (j))

A restricted coverage run qualifies the Gate 5 procedures of H5: the Fieller-form procedures at `n` in {100, 150, 200, 250, 300, 400,
500} under `normal`, `stress`, `stress-` (R = 49 000), plus a reported sensitivity at the J3B upper bounds (skewness 2.73, kurtosis
7.59). About 45 core-hours (1.5-2 h on 32 workers at `nice 19`), run in parallel with the smoke test before the pre-registration;
CPU slot-hours reported separately. Not re-simulated, with the reason recorded: the robust-sign pipeline under H3 (shift-invariant),
the `E_h` changes of H6 (conservative), the displacement-basis estimators (their pilot skewness is within the `stress` envelope).

### H16. `numerical` class fraction (pre-registration latitude; G5.7)

Fixed in the pre-registration from the review's section 3.d (proposed value and reachability there).

### H17. Interpretation statements (orchestrator)

Pre-registered readings: `alpha_LP` at `sigma^2` 4 (and partly 2.25) is a finite-distance travel-time quantity, not B&dD's spatial
`alpha_L`; comparisons with B&dD are limited to `alpha_TP` against their `alpha_TAM` (order of magnitude) and `alpha_L` at `sigma^2 <= 1`;
first-order benchmarks as H3; W8 read as in the review (section 2.2); no asymptotic claim.

### H18. Figures off the host (orchestrator)

The host stack has no plotting library; the in-job analysis writes every number and table, and the figures are produced locally from
the committed reduced tables by a committed script.

### H19. Documentation findings (orchestrator)

The discrepancies the full reading found in `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` and `lester-2023-key-claims.md`
(review, section 2 and bibliography section 7) are corrected in SF-43's PR, not here.

### H20. Order of work (owner authorization of 2026-11-02, recorded)

The campaign tooling (engine, runner executor, analysis package, recomputation, reconciliation, the two-GPU helper option) is prepared
by workers in isolated worktrees from `master`, without GPU, remote host or data, before activation; after activation it is integrated on
the increment base and audited like any node. Recorded in the bitácora as an authorized process deviation.

## Owner decisions of the consultation

(to be recorded before merge)

## Consequences

- SF-43's pre-registration fixes the numbers of the chosen variant; the increment base is the merge commit of the PR that adds this
  record and the review.
- The specification's planning numbers (item 6 `N`, item 3 "4 or 20", item 7 unpaired `n`) are superseded by H2, H5, H6, H9 where they
  differ; its Goal and acceptance thresholds are unchanged.
- `scripts/remote` gains the two-GPU request (default unchanged); the runbook documents the exception.

## Classification (docs/AGENTS.md)

| item | class |
|---|---|
| H2, H3, H4, H5, H9, H10, H14, H20 | accepted scope (owner) once decided |
| H6-H8, H11-H13, H15, H16 | proposed, ratified by merge (pre-registration latitude) |
| SD ratios, costs, memory, pilot skewness | confirmed (committed SF-39/SF-42 data, recomputed) |
| first-order values | confirmed (derivation and quadrature, two independent computations) |
| planning `N`, GPU-h and wall times | estimate |
