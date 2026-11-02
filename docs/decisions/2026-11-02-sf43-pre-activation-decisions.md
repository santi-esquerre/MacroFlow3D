# SF-43 pre-activation decisions: displacement-basis estimands, a geometry-robust null, a qualified Gate 5, gated budgets, a two-GPU first stage and a design that fits the deadline

- Status: **accepted** — owner decisions taken in the single consultation of 2026-11-02 (about 15:12Z), every recommended option
  chosen; ratified by the human merge of the PR that adds this record. Items marked *owner* were decided by the owner among the options
  listed; items marked *latitude* fill what decision 2026-10-29 G5.2-G5.4 leaves to the pre-registration and were accepted as written in
  the same consultation. The section "Owner decisions of the consultation" records the outcome.
- Date: 2026-11-02
- Deciders: owner (H2-H5, H9, H14, H15's qualification route, H20); orchestrator (drafting; the latitude items)
- Evidence: [`docs/experiments/2026-11-02-sf43-pre-activation-review.md`](../experiments/2026-11-02-sf43-pre-activation-review.md) (verified
  bibliography; scripts `firstorder_window.py`, `firstorder_box.py`, `design_cost.py`, `lattice_check.py`, `gate4_power_sim.py`,
  `gate5_options.py`; eleven analyses of the committed SF-39/SF-42 data; five adversarial reviews: statistics, numerics, physics and
  citations, executability, deadline and cost).
- Builds on: `2026-10-29-sf39-remediation-sf42-sf43.md` (G1-G11), `2026-10-29-sf39-sigma025-addendum.md`, `2026-10-27-remote-single-gpu.md`,
  `2026-10-27-sf39-pre-activation-decisions.md` (format), SF-42's findings (a)-(j) and the owner decisions of 2026-11-01 (CPU-only
  slot-hours not counted toward G10).
- Not changed: the SF-43 Goal and acceptance thresholds; G1-G3; G4 except through the latitude of specification item 1 (H2); G5.1;
  G5.5's margin and level; G5.7's classes (H4 extends their amplitude range, H16 fixes their levels); G6-G9; G11; SF-39's and SF-42's
  verdicts; every production path (no runner change).

## Date of knowledge (integrity statement)

Written after SF-39's results and SF-42's pilots were committed (`104cc254`, `ef87a1bc`), before any SF-43 seed existed. SF-39 and SF-42
data are pilots only. Data-informed items, disclosed: H2 rests on SF-42's standard-deviation ratios (not on a mean); H3 on theory; H4's
admission of `sigma^2 = 0.25` on H2 and on the displacement-basis pilot mean at 0.25 (5.1e-4 `lambda`, about 21 standard errors from
zero), which is the one item that enlarges what can be claimed and is the owner's decision. Each item is fixed before SF-43 data exist.

## Context

The specified design costs 147-185 GPU-h with the measured overheads (cost audit) and would not finish before the owner's deadline
(analysed results by 2026-11-09 00:00 local); its position-basis transverse estimators spend most of their variance on a term of zero
mean; no Gate 5 procedure was qualified (SF-42 Finding (j)); `E_h` has gaps against the verification standards; and G5.7's levels are
undefined. The adversarial reviews also showed that the SF-18 field is periodic along the mean flow (period = box length), so the
first-order benchmarks of the box differ from those of an unbounded medium, and that SF-42's coverage evidence does not reach every
component of the robust sign at the sizes and Holm levels proposed.

## Decision

### H1. Integrity and scope (orchestrator)

As above. The review's planning numbers are not commitments; the pre-registration fixes every number from SF-39/SF-42 data and this
record.

### H2. Displacement basis primary (owner)

The confirmatory transverse estimands are `alpha_TP` and `alpha_TAP` of the transverse **displacement** `x23(d) - x23(seed)` at each
control plane (`sf42red` basis `disp`), flux-weighted (specification item 1 lets the pre-registration make a displacement variant
primary). The position basis is reported as a secondary from the same runs, with the `pos - disp` contrast; the uniform-weight
displacement estimand, whose expectation equals the position one exactly by transverse stationarity, is reported as a control. Noise:
across-realization SD smaller by 37 / 9.8 / 3.7 / 1.14 at `sigma^2` 0.25 / 1 / 2.25 / 4 (SF-42 R3).
Options: (a) displacement primary **[recommended]**; (b) position primary (infeasible before the deadline at the targets); (c) both
co-primary (needs the position-basis `N`).

### H3. Robust-sign null at the unbounded-medium first-order value (owner)

Null `alpha_e <= c(sigma^2)` with `c = 7.992e-5 sigma^2 lambda`, the first-order value of the plane estimand in an **unbounded** medium
(`firstorder_window.py`); rejection requires `RL_e(a) > c` (an explicit threshold, not a shift of the data). In SF-43's boxes the first
-order value is not positive: the SF-18 field is periodic along `x` with the box length, so the first-order transverse displacement
variance is symmetric about half the box and the window value is -3.5e-5 (width 24, length 104), -4.1e-6 (width 12) `sigma^2 lambda`
(`firstorder_box.py`; one-particle, Dirichlet faces not modelled). The null at `c` is therefore conservative in the box and makes the
claim independent of the box: "exceeds the linear-theory value even of an unbounded medium". Pipeline coverage is unchanged by an
explicit threshold.
Options: (a) `c` = unbounded first-order value **[recommended]**; (b) `c = 0` (G5.4; in the box linear theory is already <= 0, so this
also separates linear theory); (c) `c` = 10 x the unbounded value.

### H4. Primary family (owner)

Robust sign at `sigma^2` 0.25, 1, 2.25, 4 and Gate 5 at the four amplitudes: `m = 8`, Holm at a family-wise 5 % (at most 6.25 % given
G5.3's 1.25 tolerance). The family is fixed; an untestable hypothesis (below its floor, halted arm, no qualified procedure) enters with
`p = 1`. Decisions are taken at the Holm levels (`RL(level) > c`; Gate 5 tests at the level), with a relative tolerance of 1e-9 for
ties; lattice p-values are reported. Admitting 0.25 adds Holm tails smaller than those SF-42 simulated; H15 covers them.
Options: (a) `m = 8` with 0.25 **[recommended]**; (b) `m = 7` as G5.6 (0.25 descriptive; all tails already simulated); (c) Gate 5 out
of the family (`m = 4`).

### H5. Gate 5 in Fieller form (owner)

The paired ratio-of-means TOST of G5.5 is computed as two one-sided one-sample tests on `D_hi = L_r - 1.10 P_r` and `D_lo = L_r - P_r /
1.10` (Fieller 1954; Hauschke et al. 1999; Berger & Hsu 1996), each at the Holm level; the estimand is unchanged. The size depends only
on the standardized law of `D_r` at the margin, which is the law SF-42's `sampling` cells simulate. Procedure = the first of (i) Hall's
(1992) skewness-corrected t, (ii) bootstrap-t at the level (B = 33 599), (iii) bootstrap-t at half the level, that qualifies at the
analysed `n` in H15's run under `normal`, `stress`, `stress-` and, at `sigma^2 = 4`, under the J3B-bound family (skewness 2.73, excess
kurtosis 7.59, both signs); none -> descriptive, `p = 1`. `n` = 200 / 200 / 200 / 250 pairs at `sigma^2` 0.25 / 1 / 2.25 / 4 (Hall
qualifies from 200 in the harness check; at 4 the power target with conservative pairing inputs; the consumer CV there rests on one of
eight fields); floors equal to these `n` (analysed at the largest coverage grid point <= the realized `n`). Ratio confidence interval by
inverting the tests; `agree` / `disagree` / `inconclusive` per G5.5; achieved power = Fieller-form power at the realized `n` with the
observed SD of `D_r` at the margins. Par2 usability mirrors the consumer's: a `Converged=NO` solve is usable only if its relative
residual and balance are <= 1e-8 (else the seed is replaced group-wide); a sensitivity excluding every pair with a fallback-accepted
solve in either code is reported. Gate 5 arms are not re-sized by the blinded re-estimation. Pre-registered expectation: at
`sigma^2 = 4` the plane-based `alpha_LP` is a slow-tail, non-Fickian quantity and a `disagree` is plausible; it is descriptive, triggers
the attribution of G5.5 (consumer twin / Pollock twin / Par2 decomposition at every amplitude; G7 at 0.25 and 1) and, per the
specification, keeps SF-43 active until attributed or decided by the owner.
Options: (a) as above, coverage qualified before the pre-registration **[recommended]**; (b) as above with the coverage run inside
the production job (rule pre-registered, resolved after launch); (c) Gate 5 descriptive (H4 (c)).

### H6. `E_h` rules (latitude)

- Ladders, paired by seed with production: `lambda/h` 5 / 10 / 20 at 0.25 and 1 (production in the middle); 4 / 8 / 16 at 2.25
  (production finest; the 4 level is an untested assumption for the transport estimand: SF-42 E shows second-order flux convergence from
  4, which is flux evidence only); 8 / 16 / 32 plus a 4 rung at 4 (production finest).
- Order: each constant-ratio triple's lower limit `p_L` is range-tested in `[0.5, 2.1]`; `p_use = min(p_L over the triples, 2)`;
  non-monotone or oscillating resamples have no real order.
- **Gate on the asymptotic range (order branch only):** at 2.25 the `lambda/h = 12` level (the Gate 5 consumer arm, production seeds)
  must agree with the 4 / 8 / 16 error law (the bootstrap 90 % interval of `a(12) - a_fit(12)` contains 0, `a_fit = a_inf + C h^p`
  fitted on the triple); at 4 the orders of 4 / 8 / 16 and 8 / 16 / 32 must agree (90 % interval of their difference contains 0). If a
  gate fails, `E_h = max(order branch, fallback)`.
- Finest branch: `E_h = 1.25 max(UCB|a_P - a_{P-1}|, 2^(-p_use) UCB|a_{P-1} - a_{P-2}|) / (2^p_use - 1)` (Giles 2015). Finer branch
  and fallback as G5.4; the fallback uses the ratio-2 levels.
- Stated assumptions: the fallback bounds the production error only for a monotone error decaying at least like `h^0.21` (production
  finest) or `h^0.24` (middle) with no component common to all levels; at the particle-noise floor it is a resolution bound.
- Every change except the gate makes `E_h` larger or equal pointwise; with the gate, `max(...)` keeps it so.

### H7. `E_W` (latitude)

Widths 24 and 48 `lambda` at `lambda/h` 10 at 0.25 and 1 (the production arm is the 24 cell); at 2.25 a 2 x 2 of widths 24 / 48 at
`lambda/h` **8 / 12** (the (24, 8) cell is the 8 rung, the (24, 12) cell is the Gate 5 consumer arm, (48, 12) is 1248x576x576) so that
the interaction covers the transfer 12 -> 16 and the 19.2-`lambda` source is admissible; at 4 a 2 x 2 of 12 / 24 at `lambda/h` 8 / 16.
`E_W` at the finest measured `lambda/h`; interaction component; stated: the transfer to production `lambda/h`, the residual beyond the
widest width, and the outlet boundary layer (about `W / 2 pi`) inside the 5-`lambda` buffer.

### H8. `E_int` (latitude)

The first 30 production seeds of each amplitude run with `--hmax-div 8,16,32` (the h/8 pass is the production realization);
`E_int = 2 UCB|alpha(h/16) - alpha(h/8)|` (G5.4). Its paired-contrast bound is not qualified by SF-42 under `stress` (no paired
procedure is); it is covered by the pipeline route of H15.

### H9. Design variant and sample sizes (owner)

Variant A **[recommended]**: production 200 at every amplitude (first 30 with three passes); ladders 100 at 0.25 and 1, 150 at 2.25 and
4 (shared cells); width cells 150; Gate 5 as H5; G7 all 8 cells x 60 at 0.25 and 1; distance arms 100 at 1 and 2.25 **plus a scope
extension**: a 208-`lambda` arm at `sigma^2 = 4` on 3328x192x192 (`lambda/h` 16, width 12), 100 realizations (the decay test at the
highest variance); **plus a positive control**: 30 realizations at `sigma^2 = 0.04` on 1040x240x240 whose plane variance profile is
compared with the box first-order prediction (`firstorder_box.py`), an end-to-end check of the estimand pipeline. About 60 GPU-h
planning (79 with every pessimistic factor), a two-GPU stage of 12-16 h, 47-63 h of wall (60-79 h on one GPU). Variant B (faster):
production 200 (150 at 4), ladders 60, width 100, G7 30 per cell, distance 60, no extension: 42 GPU-h, 33-44 h. Variant C (more budget
precision): production 300 (250 at 4), ladders 150, width 300, Gate 5 300, distance 150 incl. the extension: 93 GPU-h, 70-93 h, needs two
GPUs and a launch by about Wednesday 15:00. Blinded re-estimation (G5.2) on the production and width arms only:
`n_target = min(2 n_conf, max(n_conf, N_req))`, `N_req` the precision-target `N` at the blinded SD's 80 % upper bound (inert unless the
SD is several times the planning value).

### H10. Sources and lattice (latitude; SF-42 Finding (a))

Sources as decision G4 (0.8 x the production width): 19.2 `lambda` at 0.25, 1 and 2.25, 9.6 at 4. Common admissible `M`: {128, 256} at
0.25 and 1 (128 chosen, 1.37 x SF-39's seed density); 65 values at 2.25 with the H7 width cells (100 chosen); 65 at 4 (100 chosen).
No source change is needed (the earlier 19.08-`lambda` proposal is withdrawn).

### H11. SF-42's 3x PCG rule (latitude; Finding (f))

Accepted explicitly for 1040x240x240 at 0.25 (3.06) and 2080x240x240 at 1 (3.33), and for the shallow-hierarchy grids not measured at
0.25 (520x120x120, 1040x480x480, 2080x480x480; measured in the smoke test): the solves converge and the cost is measured.

### H12. Par2 (latitude; Findings (d), (h))

Horizon 3.6, `dt = 0.0244140625 dx`, `cg_rtol` 1e-10, `cg_max_iter` 2000; per-realization crossing completeness reported (< 0.999
flagged, with a sensitivity); usability as H5; regular snapshots on the Gate 5 Par2 arms (the time-window secondary and the eq. 6 fit of
Par2, specification items 1 and 9).

### H13. Schedule, priorities, floors, cutoff and guards (latitude)

- Priorities: 1 = every arm that feeds the primary family (including the 4 rung at `sigma^2 = 4`); 2 = positive control, diagnostics,
  re-estimation extensions; 3 = descriptive (distance arms, then G7). Order = priority, then progress fraction on `n_conf`; the seed
  blocks of a common-random-number group run adjacently; near-empty-GPU units first only within a priority.
- Lanes: lane A (first GPU) takes the near-empty units it can hold; lane B (second GPU) takes the units its free memory allows (live
  check) and releases its GPU when none remains or at the 30-h cap.
- Floors (below floor the hypothesis is untestable, `p = 1`, estimates descriptive): Gate 4 at an amplitude needs production >= 100,
  every ladder level >= 30 complete paired seeds, every width cell >= 30, `h_max` >= 20; Gate 5 the pairs of H5. Analysis set per arm =
  the longest complete prefix of its pre-registered seed order (replacements take the failed seed's position, group-wide).
- Clock cutoff 2026-11-08 16:00 local (19:00Z), one hour of grace, then the frozen analysis (done by about 20:00 local). Budget guard
  150 GPU-h (G10); disk guard; OOM retries not failures; reserve seeds >= 5 % per group (10 % on the `sigma^2 = 4` grids, where SF-39 saw
  about 5 % stagnated and 1.6 % unusable solves); expected failure rates pre-registered; `max_div` reported for every solve.
- GPU-availability rule (outcome-blind): if no GPU has 30000 MiB free by Thursday 2026-11-05 12:00 local, the job is launched with
  `REMOTE_GPU_MIN_FREE_MIB=17000`; near-empty units then run only when memory allows and are lost by priority otherwise.

### H14. Two-GPU first stage (owner decision of 2026-11-02, recorded)

For SF-43 only: the stage 1 of the single job, and a minutes-long two-lane test inside the smoke test, hold two GPUs chosen by
`scripts/remote` (`REMOTE_GPU_COUNT=2`; first GPU `REMOTE_GPU_MIN_FREE_MIB=30000`; second `REMOTE_GPU_MIN_FREE_MIB_2=17000`, justified by
the measured half-class peaks of at most 15703 MiB plus context; `REMOTE_GPU_COUNT_MIN=1`: after the `REMOTE_GPU_WAIT=7200` wait the job
starts with one GPU); at most 30 h from the stage-1 start, and less: lane B leaves as soon as no unit it can hold remains. The job
narrows the granted list to one id per lane; no command pins a GPU. Stage-1 GPU-h = 2 x its wall. Supersedes items 1 and 3 of
`2026-10-27-remote-single-gpu.md` and the "one GPU" of G10 for this scope only; expires at the end of SF-43's stage 1 and in any case on
2026-11-09. `AGENTS.md` and the runbook record the exception with the helper change.

### H15. Coverage and the qualification route (owner for the route; latitude for the run)

Route **[recommended]**: the robust sign is qualified **at the pipeline level** (G5.3 lists "the whole robust-sign pipeline"; SF-42's
189 pipeline cells had 0 misses in 24 500 at every Holm level), because no component procedure qualifies the paired contrast of
`E_int` under `stress` and the Student sampling limit never qualifies; the bootstrap-t sampling limit is used (qualified at `n` = 200).
Alternative: `E_int` as a stated assumption (SF-39: at most 6e-5 of the mean) with new K = 2 / 3 pipeline cells. Restricted run, CPU, in
the background of the smoke-test job (`nice 19`, at most 32 workers): (i) the H5 Gate 5 procedures at `n` in {100, 150, 200, 250, 300,
400, 500}, `normal` / `stress` / `stress-`, R >= 56 000, about 350 expected misses at the smallest tail, plus the J3B-bound family at
250 (gating at `sigma^2 = 4`); (ii) the bootstrap-t sampling limit at `n` = 200 with the `m = 8` tails; (iii) pipeline cells at the
variant's arm sizes, K = 3 / 4, at the `m = 8` Holm levels, R >= 28 000. About 50-120 host core-hours, 2-4 h on 32 workers; CPU
slot-hours reported separately. Not re-simulated, with the reason recorded: the explicit threshold of H3 (shift-invariant), the `E_h`
changes of H6 (pointwise larger or equal).

### H16. G5.7 classes with levels (latitude)

Per transverse estimator and amplitude, in order of precedence: `physical (restricted)` = Holm rejects H3's null; `numerical` = the
production two-sided 90 % bootstrap-t interval lies above `c` AND the one-sided 95 % upper limit of the bias-corrected estimate is below
0.5 x the production mean, where bias-corrected = production mean + signed Richardson correction (finer branch `(a(h_p/r) - a(h_p)) r^p
/ (r^p - 1)`, finest branch `(a(h_p) - a(r h_p)) / (r^p - 1)`, `p = p_use` if in range else 1) + signed width correction `a(W*) -
a(W_prod)`, upper limit by a seed bootstrap resampling jointly every arm sharing seeds; `unresolved (bounded)` = otherwise, reporting
`alpha_T <= U(0.05) + sum_k B_k(0.05 / K)` (a one-sided 90 % union bound); `unresolved` = anything else, including different classes for
`alpha_TP` and `alpha_TAP`. `numerical` and the bounds are descriptive (outside the Holm family). Only an `E_h` or `E_int` artifact
would support Lester et al.'s attribution of `alpha_T` to numerics; a width artifact would not.

### H17. Interpretation statements (orchestrator)

Pre-registered readings: no asymptotic claim; the estimand is the first-passage distribution at each plane (recrossings counted and
reported; an `alpha_LP` sensitivity at `sigma^2 = 4` without backflow-encounter particles); `alpha_LP` at 4, and partly 2.25, is a
non-Fickian, pre-asymptotic travel-time quantity, compared with B&dD only through their eq. 6 at `t = d/U`; `alpha_TP` against B&dD's
`alpha_TAM` as an order of magnitude; first-order references are the box values per arm (`firstorder_box.py`) beside the unbounded ones;
the halves and the distance window separate "decay toward zero" (first order, Lester 2021's illustrative eq. 5.43) from "plateau" (B&dD,
or numerical drift), not physics from numerics; W8 as in the review.

### H18. Figures (orchestrator)

Produced by the in-job analysis if the host stack provides matplotlib (SF-32's analysis used it in host jobs; checked in the smoke
test), else locally from the committed reduced tables by a committed script.

### H19. Documentation findings (orchestrator)

The discrepancies the full reading found in `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` and `lester-2023-key-claims.md`
(review, bibliography section 7) are corrected in SF-43's PR.

### H20. Order of work and Phase-2 corrections (owner authorization of 2026-11-02, recorded)

The tooling is prepared in isolated worktrees from `master` before activation (no GPU, host or data) and integrated on the increment
base after activation, with the audits and corrective nodes the executability review requires (executor constructor and blinded-SD
contract; reconciliation of Par2-generator logs; retry of failed reductions before the analysis; priority-first ordering; lane-B release;
the retained minimum per arm; Gate 5 pairing from segment metadata; the H6 gate and H5 procedures in code). Recorded in the bitácora as an
authorized process deviation.

### H21. Reduced tables and retention (latitude; G8)

Committed: per-realization estimator tables of every arm and both codes, per-plane moment tables compressed per arm, solver-verdict and
failure ledgers. On the host, for every realization of the primary arms, the plane arrays (seeds, weights, plane positions and times,
about 80 GB) are kept so that the independent recomputation covers the displacement-basis primary estimands for every realization; the
time-pass and Pollock arrays are pruned after reduction except for the retained subset (at least 5 realizations per arm plus the
human-review sample).

## Owner decisions of the consultation

Single consultation of 2026-11-02, about 15:12Z (12:12 local), four questions; the owner chose the recommended option in each, before
any SF-43 seed existed:

| question | owner choice | items fixed |
|---|---|---|
| Design and sizes | variant A complete | H9 as written, including the scope extension (208-`lambda` arm at `sigma^2 = 4`, 3328x192x192, `lambda/h` 16, width 12, 100 realizations; specification item 9 gains it) and the positive control (`sigma^2 = 0.04`, 1040x240x240, 30 realizations) |
| Estimands, null and family | displacement + `c = alpha_FO` + `m = 8` | H2 (a), H3 (a) with `c = 7.992e-5 sigma^2 lambda`, H4 (a) with `sigma^2 = 0.25` in the family |
| Gate 5 and qualification route | Fieller form, coverage before the pre-registration | H5 (a) (coverage run in the smoke-test job, resolved before the pre-registration is committed); H15 route: pipeline-level qualification with the bootstrap-t sampling limit |
| Operating rules | accept everything as written | the latitude items H6-H8, H10-H13, H16, H21 as written, including lane B at `REMOTE_GPU_MIN_FREE_MIB_2=17000` (half-class units only), the GPU-availability rule (launch with 17000 MiB if no GPU has 30000 MiB free by 2026-11-05 12:00 local), the clock cutoff 2026-11-08 16:00 local with one hour of grace, the 150 GPU-h guard, figures in the job and H19's corrections in SF-43's PR |

H14 (two-GPU first stage) and H20 (order of work) are owner decisions of 2026-11-02 taken before the consultation, recorded above
unchanged.

## Consequences

- SF-43's pre-registration fixes the numbers of the chosen variant; the increment base is the merge commit of the PR that adds this
  record and the review.
- The specification's planning numbers (item 6 `N`, item 3 "4 or 20", item 4's 2.25 cells, item 7 unpaired `n`) are superseded where H2,
  H5, H6, H7, H9 differ; its Goal and acceptance thresholds are unchanged; item 9 gains the `sigma^2 = 4` distance arm (owner, H9).
- `scripts/remote` gains the two-GPU request (default unchanged); `AGENTS.md` and the runbook record the exception.

## Classification (docs/AGENTS.md)

| item | class |
|---|---|
| H2, H3, H4, H5, H9, H14, H15 route, H20 | accepted scope (owner, 2026-11-02) |
| H6-H8, H10-H13, H15 run, H16, H21 | accepted as written in the consultation (latitude); ratified by merge |
| SD ratios, costs, memory, pilot moments, lattice sets | confirmed (committed SF-39/SF-42 data, recomputed by scripts) |
| unbounded first-order values | confirmed (two independent derivations) |
| box first-order values | estimate (first order, Dirichlet faces not modelled) |
| planning `N`, GPU-h and wall times | estimate |
