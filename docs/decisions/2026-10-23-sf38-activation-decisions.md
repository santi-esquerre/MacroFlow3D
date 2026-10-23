# SF-38 activation decisions: production geometry per amplitude, an equivalence test for the medium A/B, the Gate 5 protocol, and frozen SF-39 rules

- Status: accepted (owner, 2026-10-23, answers to the SF-38 orchestrator before activation; ratified by the human merge of the PR that adds this record)
- Date: 2026-10-23
- Deciders: owner (SF-38 orchestrator session, 2026-10-22/23)
- Evidence: `docs/experiments/2026-10-22-sf40-lean-lagrangian-stage.md` (measured 3328x384x384 at 28885 MiB; feasibility table),
  `docs/experiments/2026-10-22-sf37-posthoc-review-and-mecom-feasibility.md` (I.5: Pollock seed weights, resolution vs domain),
  `docs/experiments/2026-10-22-lagrangian-memory-audit.md` (§7 CSV header bug, Par2 bytes; §10 medium (b)),
  `docs/experiments/2026-10-23-resolution-vs-variance.md` (§5: error grows with `sigma^2` and travel distance; coarse grids
  under-disperse `sigma_T^2`), Beaudoin & de Dreuzy (2013) §2-3 (local copy `docs/references/Dreuzy-2013-WRR.pdf`, p. 2490-2492),
  and the orchestrator's code reconnaissance and adversarial review of the options (runtime records
  `.claude/orchestration/SF-38-long-domain-infrastructure/analysis/`, not versioned; every figure used below is restated here with
  its class)
- Builds on: `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`, `docs/decisions/2026-10-22-lean-lagrangian-stage.md`
- Supersedes / refines (without editing those records):
  - 2026-10-22 item 6, "SF-38's Goal is not changed": the Goal of SF-38 is rewritten (D2) to name the production geometries,
    the Gate 5 protocol and the equivalence test;
  - 2026-10-17 item 7, "(a) is used if its interior statistics agree with (b) within the sampling band": "agree" is now the
    pre-registered equivalence test of D5-D7 (an inconclusive result selects (b), following that item's "otherwise (b)");
  - 2026-10-17 item 8, "difference within `k` standard errors; SE by block bootstrap" for ladder comparisons: the SF-38 A/B
    and the SF-39 ladder and Gate 5 comparisons are the paired (Gate 5: unpaired) equivalence tests of D5-D6 and D15;
  - 2026-10-22 item 4 (study `lambda/h` 8/16/32): applied per amplitude as in D1 (`sigma^2 = 1` at `lambda/h` 10).

## Context

SF-38 became READY on `master` `ab78d7c` (SF-36 and SF-40 `done`). Its specification still named one production grid,
2048x256x256 at `lambda/h = 10`, and a medium-selection rule written before SF-37 and SF-40. Since then:

- SF-37 fixed the production `lambda/h` 8/16/32 at `sigma^2` 1/2.25/4 (periodic cube, 64 `lambda` path); at `lambda/h = 10`
  the worst-seed `sigma_T^2` error at `sigma^2 = 4` is ~16 % already at 64 `lambda` (audit §3; derived).
- SF-40 measured the lean stage at 3328x384x384 = 4.91e8 cells: peak 28885 MiB at `sigma^2` 2.25 (`lambda/h` 16) and 4
  (`lambda/h` 32), seed 4001 (measured); capacity ~5.1e8 cells (derived).
- The resolution-vs-variance note shows the plume-statistic error grows with travel distance (`x1^0.66-0.80`, median) and
  that coarse grids under-disperse `sigma_T^2` (17-19 of 21 cells): a resolution deficit biases `alpha_T` low, toward the
  Lester claim (post-hoc; Rule 1 makes it a threat to the project's purpose).
- Medium (b) with the unmodified `solve_head` costs 92.6 B/cell (116.6 with its faces alive) and does not fit 3328x384x384
  (confirmed in code, derived bytes).
- The written A/B rule ("(a) iff `|mean_r D_r| <= 2 SE` for every E1 and E2 statistic at both amplitudes", `N_R = 5`) runs
  ~36-44 tests at 4 degrees of freedom (`P(|t_4| > 2) = 0.116` per test): when the media are identical it selects (a) only
  0.4-44 % of the time depending on the correlation of the statistics (derived, Monte Carlo), i.e. it is biased toward (b)
  and SF-41; and with paired, very precise statistics (the plane mean of `v1`, which is the plane flux and only reflects the
  head-drop / `K_eff` convention) it declares scientifically irrelevant differences significant.
- Par2 (after SF-40) fits ~2.47e8 cells with velocity diagnostics off (derived), injects only uniformly over the whole
  transverse plane at `inject_x`, has a closed outlet (particles pile up at `Lx` and stay "active"), uses a different,
  non-periodic random-spectral generator with unnormalized `sigma^2`, and a fixed `dt` (confirmed in code).
- Pollock comparator weights: a seed lying on a transverse cell face takes the column chosen by the sign of the transverse
  velocity (`+-h/2`), an `O(h)` systematic weight bias (post-hoc P1); with `M = 100` the window `[0, 1)` puts 784 of 1e4 seeds
  on faces, `[0.1, 0.9]` none (closest `h/125` for `n_T` 256, 288, 384; derived exactly).
- `read_timeseries_csv` skips one of the two header lines `CsvTimeSeriesWriter` writes; only `run_mode: analysis_only` is
  affected (an extra zero row at `t = 0` and a wrong first-interval derivative); no versioned result used that path; fixing
  the reader changes no output byte (confirmed in code).

## Decision

Owner answers, recorded in the order they were asked (the orchestrator's recommendation is noted where the owner chose
otherwise).

1. **D1 Production geometry per amplitude** (seeds and fields by SF-18 on the box, `h = 1/n_T`, transverse period 1):
   `sigma^2 = 1`: 2048x256x256 at `lambda/h = 10` (`ell = 10/256`; 204.8 x 25.6 x 25.6 `lambda`, the B&dD domain);
   `sigma^2 = 2.25`: 3328x384x384 at `lambda/h = 16` (`ell = 16/384`; 208 x 24 x 24 `lambda`);
   `sigma^2 = 4`: 3328x384x384 at `lambda/h = 32` (`ell = 32/384`; 104 x 12 x 12 `lambda`).
   Known limitations, recorded: at `sigma^2 = 4` the domain is short for `alpha_L` and narrow (12 `lambda`, transversally
   periodic; injection window 9.6 `lambda`); the worst-cell error extrapolated to the full length is ~4-5 % at `sigma^2 = 1`,
   ~8-10 % at 2.25, ~4 % at 4 (post-hoc estimate) — verified by the ladders of D14.
2. **D2 Goal rewritten** in SF-38 (this record supersedes 2026-10-22 item 6).
3. **D3 `alpha_L` at `sigma^2 = 4`:** `alpha_LAM` is computed as at the other amplitudes (B&dD eq. 5 window
   `[0.75 t_b, t_b]`) and reported with the explicit caveat that the regime is pre-asymptotic (B&dD: ~20.5 advection times to
   the longitudinal asymptote at `sigma^2 = 4`; SF-37 D8 shadow asked `lambda/h` 64 for that estimator). (Recommendation was
   to classify only `alpha_T` at `sigma^2 = 4`.)
4. **D4 A/B grids:** `sigma^2 = 1` on its production box 2048x256x256; `sigma^2 = 4` on 3328x256x256 (`lambda/h = 32`,
   104 x 8 x 8 `lambda`: production length and resolution, two thirds of the production width). Medium (b) releases its
   `FlowWorkspace` before the face repack (otherwise ~140.6 B/cell, ~29.7 GiB with the context, within ~1 % of the 30000 MiB ceiling) and passes `mg_levels = 7`
   explicitly. Recorded assumption: the slowest transverse mode decays over `L_T/(2 pi)`, 1.3 `lambda` at 8 `lambda` against
   1.9 `lambda` at 12 `lambda` (both well inside 5 `lambda`), so the A/B may slightly understate (b)'s boundary layer.
5. **D5 A/B rule: equivalence test.** `N_R = 10` fixed (seeds 4001-4010), one analysis, no interim look. Five primary
   statistics per amplitude (definitions in the SF-38 specification): the last-octave slope of `sigma_T^2`, the last-octave
   slope of `var tau * qbar^2`, the interior `var(v1)/qbar^2`, and the tilts `theta_2`, `theta_3` (interior mean transverse
   velocity over `qbar`). Two one-sided tests by intersection-union (90 % confidence interval inside the margin), which is
   level 0.05 for the equivalence claim without a multiplicity correction. Every other E1/E2 statistic is recorded, not
   decisive.
6. **D6 Margins** (half of the SF-39 confidence-interval targets): ratio statistics on the log scale, `+-ln(1 + r)`:
   `r = 0.05` for the `sigma_T^2` slope (SF-39 target 10 % for `alpha_T`), `r = 0.025` for the `var tau * qbar^2` slope
   (target 5 % for `alpha_L`), `r = 0.05` for `var(v1)/qbar^2`; an absolute floor for the `sigma_T^2` slope of `2.5e-3 lambda`
   (`= 2 x 1.25e-3 lambda`: half the SF-39 `alpha_T` floor of `2.5e-3 lambda`, expressed as the slope
   `d sigma_T^2 / d x1 = 2 alpha_T`);
   tilts `|Delta theta| <= 0.005` (absolute), applied to each component `theta_2`, `theta_3` separately (D19: the vector may
   then reach 0.0071, so `theta^2 alpha_L` is up to ~1/4 of the `alpha_T` floor at `sigma^2 = 4`, using the B&dD `alpha_L`
   order only as a magnitude).
7. **D7 Inconclusive -> medium (b)** (2026-10-17 item 7, "otherwise (b)": a medium whose equivalence was not shown is
   not used; stop rule R4 then applies).
8. **D8 Medium (b) solve:** a solver pilot (seed 4001, `sigma^2 = 4`, 3328x256x256) before the A/B; `solve_head` with
   `mg_levels = 7`, PCG `rtol = 1e-10` on its own residual `||r|| / ||r_0||`; independent checks: total-flux balance
   `|Q(x=0) - Q(x=L1)| / |Q(x=0)| <= 1e-8` and the maximum cell divergence of the faces (recorded). Fallback: if the PCG
   stalls above 1e-10 but reaches `<= 1e-8` with the flux-balance check met, that tolerance is used and recorded; otherwise
   stop, owner.
9. **D9 Pollock weights without code:** injection window `[0.1, 0.9]^2` (0.8 `L_T`, centred, as B&dD) with `M = 100`; the
   runner asserts that no seed lies on a transverse cell face; Pollock keeps its own RT0 weight (SF-37 RR7).
10. **D10 Par2 grids: same physical domain** as the Lagrangian study: `sigma^2 = 1` 2048x256x256 at `lambda/h = 10`
    (`mg_levels: 6`, the identical grid); `sigma^2` 2.25 and 4: 2496x288x288 at `lambda/h` 12 and 24 (208 x 24 x 24 and
    104 x 12 x 12 `lambda`), `mg_levels: 5` (six levels leave a 78x9x9 coarsest grid whose red-black colouring races across
    the periodic seam); velocity diagnostics off.
11. **D11 Gate 5 "same configuration": twin series plus a Pollock bridge.** Par2 is not changed (uniform injection over the
    full plane at `x = 5 lambda`). The Lagrangian runner also writes a twin series with uniform weights over the same
    windowed particles (no tracker change), and a Pollock bridge (the Pollock comparator on the SF-19 faces of the same
    realization and seeds, flux-weighted and twin series). SF-39 compares over the same travel distance (`L1 - 10 lambda`,
    the Par2 one) with the same `t_b` definition, in one pre-registered common time window, truncated at `t_b`, normalized by
    the measured mean velocity, with Par2's realized `sigma^2` recorded.
12. **D12 Par2 time step:** `dt` keeps the `dt/Delta` of the base config (`dt = 12.5`, `Delta = 5`, ~35 steps per cell at
    the `sigma^2 = 1` mean velocity); a paired `dt/2` check on the first 3 realizations of each config (recorded, no
    threshold); `n_steps` covers the nominal arrival of the plume centre at `L1 - 5 lambda` with a 1.2 factor; the analysis is
    truncated at `t_b` detected from snapshots.
13. **D13 CSV reader fix in SF-38** (a DAG node, as 2026-10-22 item 7 allows): `read_timeseries_csv` skips every line that
    starts with `#` and then one column-header line; the test that expected the artifact row is updated.
14. **D14 SF-39 ladders at the full production length, two arms:** (i) a paired narrow-section ladder at the production
    length on the same continuum, rungs {1/2, 1, 2} x production `lambda/h` (`sigma^2 = 1`: 5/10/20 at 12.8 `lambda`;
    2.25: 8/16/32 at 8 `lambda`; 4: 16/32/64 at 4 `lambda`), with a convergence check of the narrow sizes before the
    ladder and a representativeness check; (ii) a coarse twin at 1/2 x production `lambda/h` on the production domain
    (`N_R = 20`). Richardson extrapolation (order 2) only if the observed order of the ensemble mean is in [1.5, 2.5] and the
    sequence is monotone; otherwise `|Delta Q|` is reported as a bound; never applied as a correction to `alpha`.
    **D14b:** at `sigma^2 = 1` the twin is at `lambda/h = 5` (1024x128x128; `lambda/h = 8` is not an integer grid on
    204.8 `lambda`), under the same guard.
15. **D15 SF-39 rules frozen now** (before SF-38 produces any plume statistic or `alpha(t)` series at `sigma^2 = 4`):
    SF-39 criteria A and C become equivalence tests; rule D gets the branch "`alpha_T` grows under refinement"; the tolerance
    ladder (vacuous at the `h/8` cap, SF-37 RR15) becomes an `h_max` ladder {`h/8`, `h/16`, `h/32`}; D3, D11 and D14 are
    written into SF-39. Values (owner, D15a-c): SF-39 A margins are half of the CI targets (`alpha_TAM`:
    `+-max(0.05 |alpha_TAM|, 1.25e-3 lambda)`; `alpha_LAM`: `+-2.5 %`), 90 % paired CI, `N_R = 20`; Gate 5 "agree" if the
    90 % CI of the difference of `alpha_LAM` (Lagrangian twin vs Par2, common window) lies within `+-10 %` of `alpha_LAM`;
    the narrow-ladder verdict transfers to the production domain iff the relative 1/2 -> 1 differences of the ensemble
    means in the narrow ladder and in the production twin have the same sign and a ratio in `[1/2, 2]`.
16. **D16 GPU:** one V100, every job serialized on the SF-38 mirror, no budget ceiling; consumption reported at the end
    (GPU 1 stays free). (Recommendation was a 6 GPU-hour ceiling.)
17. **D17 Par2 realized `sigma^2`:** nothing in the Par2 path writes `K` to disk (`save_K` is parsed but unused); a new SF-38
    tool regenerates each Par2 field with the read-only `generate_K_field` at the same seed and parameters, reports mean and
    variance of `ln K`, and checks `K` min/max/mean against the pipeline's `K stats` line. The Par2 path is not changed.
18. **D18 Precedence of SF-39 rule D:** `unresolved (grid-limited)` first (any ladder with `|alpha_TAM|` increasing under
    refinement and not `converged`), then `numerical`, then `physical`, then `unresolved`; increasing / decreasing refer to
    the point estimate of the ensemble mean.
19. **D19 Tilt margin per component** (see item 6). (Recommendation was a vector bound, `+-0.0035` per component.)
20. **D20 Gate 5 reference:** the `+-10 %` margin is relative to `alpha_LAM` of Par2, the baseline.


## Consequences

- SF-38 specification: Goal, intent, In scope, Files, Implementation 3, Expected numerical effect, thresholds B, C0 (new),
  C, D, E, F and the failure policy are rewritten accordingly; append-only bitácora row. SF-38 stays `pending` until its
  activation row, which copies the thresholds verbatim and fixes the base commit before any run.
- SF-39 specification: In scope (geometry, injection, ladders, `h_max` ladder, Gate 5 protocol), estimator note for
  `sigma^2 = 4`, thresholds A, C and D rewritten; append-only bitácora row.
- The solver overview's one-line description of the long domain is updated.
- Not changed: SF-18/SF-19/SF-28/SF-32 numerics, the tracker, the Pollock tracker and comparator, the Par2 adapter and
  `EnsembleRunner` (Par2 path), every existing config, SF-35/36/37/40 outputs and records, Rule 1 and the
  non-presupposition of `alpha_T`, the Gaussian-only covariance rule. No `alpha_L` / `alpha_T` value is claimed here.

## Classification (docs/AGENTS.md)

| item | class |
|---|---|
| 3328x384x384 within 30000 MiB at `lambda/h` 16 and 32 (28885 MiB) | confirmed in runs (SF-40) |
| Medium (b) bytes (92.6 / 116.6 / ~140.6 B/cell), Par2 bytes (~124.8 B/cell, ceiling ~2.47e8 cells), MG level counts | confirmed in code; byte totals derived |
| Par2 injection (uniform, full plane), closed outlet, generator, fixed `dt`; Pollock seed-ownership rule; CSV header bug | confirmed in code |
| Seed-to-face distances of the window `[0.1, 0.9]` with `M = 100` | derived (exact arithmetic) |
| Bias and power of the written A/B rule; TOST power | derived (t distribution, Monte Carlo) |
| Full-length error extrapolations; transverse boundary-layer scale `L_T/(2 pi)` | estimate (post-hoc; scaling) |
| Items 1-20 of the Decision | accepted scope |
| Runner, twin series, Pollock bridge, (b) solve wiring | proposed architecture (SF-38 DAG) |
| Medium choice, boundary-layer thickness of (b), Par2 `dt` sensitivity, `alpha_L`, `alpha_T` | open question (SF-38 C/E, SF-39) |
