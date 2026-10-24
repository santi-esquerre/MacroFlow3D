# SF-41 specified (medium (b) for the Lagrangian consumer: lean Dirichlet stage, non-periodic-in-x1 spline, x-exit handling) and SF-39 amended for medium (b), tilted mean flow, ladder sample size and compute

- Status: accepted (owner, 2026-10-24, answers to the orchestrator's planning questions; ratified by the human merge of the PR that adds this record)
- Date: 2026-10-24
- Deciders: owner (orchestrator session 2026-10-23/24, after the review of the SF-38 PR #62)
- Evidence:
  - SF-38 experiment note `docs/experiments/2026-10-23-sf38-long-domain-infrastructure.md` (PR #62; on the default branch after its merge): criterion C selected medium (b) by the
    pre-registered rule (`C_VERDICT selected=b equivalent=5/10 non_equivalent=0 inconclusive=5`); its section "Post-hoc readings" P1-P6 (not pre-registered, exploratory;
    `artifacts/2026-10-23-sf38-long-domain-infrastructure/analysis/posthoc_ab.md`, written by `scripts/posthoc_ab.py`) and the disclosure "Spec-vs-code discrepancy";
  - the orchestrator's code reconnaissance of 2026-10-24 (runtime records `.claude/orchestration/SF-38-long-domain-infrastructure/`, not versioned; every figure used below is
    restated here with its class and file:line where it is load-bearing);
  - `docs/experiments/2026-10-22-lagrangian-memory-audit.md` §8 (bitwise memory levels) and §10 (medium (b));
  - `docs/experiments/2026-10-22-sf40-lean-lagrangian-stage.md` (level S+, 60.655 B/cell, context 487 MiB);
  - `docs/experiments/2026-10-23-resolution-vs-variance.md` §5 (per-realization error tail, production-length ladders);
  - Beaudoin & de Dreuzy (2013) §2 [6]-[7] and eq. 4-5 (local copy `docs/references/Dreuzy-2013-WRR.pdf`, p. 2490-2492): fixed heads on the faces orthogonal to `L`, periodic
    transverse faces, injection at least `5 lambda` downstream of the inlet "to avoid border effects on the velocity statistics", moments in fixed axes `k`, centred per realization,
    `N_R = 500`.
- Builds on: `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md` (item 7, stop rules R3, R4), `docs/decisions/2026-10-22-lean-lagrangian-stage.md`,
  `docs/decisions/2026-10-23-sf38-activation-decisions.md` (D1, D11, D14-D16, D19)
- Supersedes / refines (without editing those records):
  - 2026-10-23 D15a "`N_R = 20`" for the SF-39 ladders: 20 becomes the **minimum** per rung; the ladder `N_R` is set by the internal-pilot rule of E5;
  - 2026-10-23 D16 (one GPU, serialized) for SF-39 only: two GPUs (E6); SF-41 keeps one GPU;
  - 2026-10-23 D19 (tilt margin of the A/B): not changed; E4 adds how SF-39 measures transverse moments under a per-realization tilt.

## Context

- **Medium (b) is the study's medium.** SF-38 selected it by the pre-registered rule (R4). It is also the configuration of the classical baseline (B&dD: fixed heads, periodic
  transverse faces; Par2 likewise), so Gate 5 compares like with like on boundary conditions (theory note `beaudoin-de-dreuzy-2013-key-claims.md`, "Compare like with like").
- **What (b) needs that does not exist** (code reconnaissance, *confirmed in code*):
  - the SF-28 spline (`src/numerics/interpolation/PeriodicTricubicBSpline.{cuh,cu}`) assumes periodicity in `x1` in the coefficient solve (FFT symbol / cyclic tridiagonal) and in
    the per-axis stencil (`xr = x - L floor(x/L)`, index wrap, `.cuh:171-176, 203-211, 286-290`); there is no non-periodic, clamped or not-a-knot spline anywhere in the repo;
  - `Y` (SF-18) is periodic on the long box, so the `ln k` spline may stay periodic; only the potential needs a non-periodic representation; with `Phi = -H = G1 x + f`, the
    fluctuation `f = -H - G1 x + G1 L1` vanishes on both `x` faces;
  - the forward tracker has no `x` boundary: it works in unwrapped coordinates and the periodic spline silently re-enters a particle that crosses `x = 0` or `x = L1`
    (`LagrangianTracker.cu:471-757`); the plane pass flags a backflow encounter but continues;
  - the medium (b) Pollock path wraps in `x` (`PollockTracker.cuh:65-77`, repack drops `U[nx]`) instead of counting an `x = 0` exit as non-ok as the SF-38 specification required
    (MINOR, unquantified, decision-invariant: SF-38 note, disclosure "Spec-vs-code discrepancy");
  - `DirichletMedium` frees the head after the velocity phase (`DirichletMedium.cu:294-295`).
- **Memory.** Medium (b) as built peaks at 92.571 B/cell exact in its solve phase (94.93 B/cell PID at 3328x256x256; 44426 MiB extrapolated to 3328x384x384: does not fit; SF-38 P5).
  The budget at 3328x384x384 is (30000 - 487 - 11.45) MiB / 4.907e8 cells = 63.04 B/cell (*derived*, SF-40 context and residual). The SF-40 placement tricks that transfer to
  `solve_head` (K uploaded into the MG level-0 coefficient; the right-hand side assembled into the dead level-0 residual; `z`, `Ap` and level-0 `b` sharing one buffer through a
  non-projected external-vector PCG) give an estimated **60.57 B/cell at the solve peak** (≈ 28846 MiB with context; *estimate from a code buffer count, not measured*, bitwise by
  construction: the same arithmetic in the same order); the head in pinned mapped host memory saves 8 B/cell more (bitwise; cost ≈ 2 × 3.9 GB of PCIe traffic per PCG iteration,
  ≈ 33-47 s per solve at 50-60 iterations, *estimate*). The SF-39 narrow-ladder rung `6656x256x256` (4.36e8 cells) is within the same budget.
- **Tilt.** Medium (a) fixes the mean flux (`qbar = e1`, mean gradient free); medium (b) fixes the mean gradient along `x` (mean transverse flux free) (*confirmed in code*, SF-38 P4).
  In (b) each realization's mean flow is tilted: at `sigma^2 = 4` on the 8 `lambda` A/B width, sd of `theta_2`, `theta_3` = 0.0166, 0.0155, rms `|theta|` = 0.0227 (3.2 times the D19 bound),
  correlated -0.92 to -0.997 with the (a) gradient tilt of the same field (*derived, post-hoc*). In fixed axes the transverse variance of a tilted plume picks up ≈ `theta^2 sigma_L^2`
  (geometry). B&dD eq. 4 uses fixed axes.
- **Pairing efficiency.** At `sigma^2 = 4` the paired A/B differences of the Lagrangian statistics kept 0.55-0.57 of the unpaired sd (`V1`: 0.010; `sigma^2 = 1`: 0.02-0.27) (*derived,
  post-hoc*): a small perturbation of the flow decorrelates trajectories at `sigma^2 = 4` (consistent with SF-30 §9). The SF-39 ladders are paired comparisons of the same kind; with the
  frozen `N_R = 20` an `inconclusive` verdict (rule D: `unresolved`) is likely at `sigma^2 = 4` (*inference*, not a power calculation on alpha).
- **Boundary layer of (b)** (*derived, post-hoc*): paired excess of `var(v1)` ~4 `lambda` at 1 % of the interior value, e-folding ~1 `lambda` at both widths; consistent with B&dD's `5 lambda`
  injection buffer.

## Decision

Owner answers, recorded in the order asked (each was the orchestrator's recommendation).

1. **E1 Lean medium (b) stage inside SF-41, with a pre-authorized fallback.** SF-41 builds a medium (b) stage with the SF-40 placement tricks (bitwise: the head is identical bit for bit to
   SF-38's `DirichletMedium` on the same input). If the measured PID peak at a production grid exceeds 30000 MiB, the head (PCG `x`) moves to pinned mapped host memory (bitwise, slower).
   If it still exceeds, stop rule R3 applies (domain reduction by owner decision).
2. **E2 SF-41 verification depth: representation + cross-representation.** Exact controls (`k = 1`; non-periodic layered `k(x1)` against the exact one-dimensional flux; transverse shift),
   spline unit contracts (interpolation, polynomial reproduction, order ladder, CPU/GPU, Hessian consistency, no allocation), a cross-representation test (the same periodic samples
   represented by the periodic SF-28 spline and by the new spline coincide in the interior up to the end-effect decay from the faces), and a Lagrangian `e_v` grid ladder on medium (b) at
   `sigma^2` 1 and 4. SF-37's production `lambda/h` transfers through the cross-representation test; SF-39's production-length ladders re-verify it. No new SF-37-type plume ladder in SF-41.
3. **E3 x-exit handling in SF-41 for both the spline tracker and the Pollock comparator** on non-periodic media: explicit exit statuses through the inlet and the outlet; existing periodic
   paths byte-identical. The SF-38 discrepancy is quantified (recorded) by re-running the SF-38 A/B with the exit-aware Pollock.
4. **E4 SF-39 transverse moments: drift-aligned primary, fixed axes reported.** Gate 4 (classification of `alpha_T`) uses time moments in the frame of each realization's mean Darcy flux
   direction, taken from the flow solve (interior mean face flux), not from the plume; fixed-axis moments (B&dD eq. 4) are reported beside them and used for the B&dD context and for Gate 5
   (`alpha_L`, like Par2); the per-realization tilt and the fixed-minus-aligned difference are recorded.
5. **E5 SF-39 ladder `N_R` by an internal pilot rule.** Each ladder starts at 20 realizations per rung; a pre-registered script computes, from the sd of the paired differences only (the mean
   difference is not printed), the `N_R` giving 80 % power to declare equivalence when the true difference is 0, capped by the compute agreed at the pilot checkpoint (E6); the final analysis uses
   all realizations including the first 20.
6. **E6 SF-39 compute: two GPUs, checkpoint after the pilot.** Independent jobs on GPU 0 and GPU 1 (`scripts/remote --increment SF-39 run <job> --gpu 0|1`, host-wide GPU locks) from one synced
   build, distinct output directories, no `sync` while any SF-39 job runs; after the pilot the projected GPU-hours are reported to the owner, who decides before production.
7. **E7 SF-39 `normalize_variance` on** (as in the SF-37/SF-38 verification); SF-18 raw variance and applied scale recorded per realization; Par2's realized `sigma^2` recorded for Gate 5.

Orchestrator design items written into the specifications (not owner alternatives; reviewed on the PR):

- SF-41 runner integration: `lagrangian_ensemble --medium b`, injection plane `X0 = 5 lambda`, plane/time lists bounded by `L1 - 5 lambda`, the Par2 travel plane, multi-pass `h_max` on one
  stage build (for SF-39's `h_max` ladder), and a per-realization `flow_stats.json` (fraction of `x`-faces with `u <= 0`, percentiles of the cell-centre speed, interior mean-flux direction)
  for the flow-reversal / stagnation context of Lester's hypotheses and for E4.
- SF-39: medium (b) solve check on the narrow sizes (relative residual `<= 1e-10`, flux balance `<= 1e-8`, the C0 fallback); per-realization paired-difference distribution of each ladder
  recorded (tail, resolution-vs-variance note §5.3); B&dD comparison conditions stated (fixed axes, `5 lambda` buffer, transverse width 25.6 `lambda` vs 12 `lambda` at `sigma^2 = 4`).

## Compute estimate for SF-39 (rough, for the checkpoint of E6)

*Estimate*, from SF-38 measured times (runner on (a): 52 s per realization at 2048x256x256, ~195-207 s at 3328x384x384 of which tracking ~21 s and bridge ~7-10 s; Par2 33 / 57 / 62 s per
realization) and an assumed medium (b) stage time of ~110 s at 4.9e8 cells (one solve of 50-60 PCG iterations instead of three correctors of 30; not measured):

| item | `N_R` 20 (ladders) / ~100 (production) | worst case (`N_R` 500 production; ladders 100) |
|---|---|---|
| production, three amplitudes | ~10 GPU-h | ~48 GPU-h |
| narrow `lambda/h` ladder | ~2.3 | ~11 |
| coarse twin | ~0.3 | ~1.6 |
| `h_max` ladder (multi-pass, one stage) | ~1.9 | ~9.5 |
| Par2 at production `N_R` | ~4.2 | ~21 |
| total | ~19 GPU-h | ~91 GPU-h |

With two GPUs the wall time is about half. The ~100 production realizations are an order-of-magnitude reading of B&dD (their 2 % / 4 % variations at `N_R = 500` scaled to the 5 % / 10 %
targets), not a prediction; SF-39's pilot fixes the number.

## Consequences

- New increment `SF-41` (`docs/plans/active/lester-eq14/increments/SF-41-medium-b-lagrangian-stage.md`), `pending`, depends on SF-38, unlocks SF-39; human review required.
- SF-39 specification amended (Depends on adds SF-41; medium (b); E4-E7; design items above); append-only bitácora row.
- Dashboard: master-checklist entry for SF-41; the DAG prose; the solver overview's increment list. `scripts/hooks/check-lester-increments.sh` expects 42 increment files.
- Not changed: SF-38's verdicts and evidence; SF-18/SF-19/SF-28 numerics (the periodic spline module and every periodic consumer stay byte-identical); the frozen periodic stack; Par2;
  Rule 1 and the non-presupposition of `alpha_T`. No `alpha_L` / `alpha_T` value is claimed here.

## Classification (docs/AGENTS.md)

| item | class |
|---|---|
| Medium (b) selected by SF-38's rule; boundary conditions of B&dD and Par2 | confirmed in runs (SF-38) / confirmed in source (B&dD p. 2490; Par2 code) |
| SF-28 periodicity points; no non-periodic spline in the repo; `Y` periodic on the box; tracker without `x` boundary; Pollock (b) wrap; head freed in `DirichletMedium` | confirmed in code |
| Medium (b) 92.571 B/cell solve phase; 94.93 B/cell PID; does not fit 3328x384x384 | confirmed in runs (SF-38) + derived |
| Lean (b) stage 60.57 B/cell; mapped-host head saves 8 B/cell; their bitwise character | estimate (code buffer count, not measured) |
| Tilt statistics, pairing efficiency, boundary-layer reading | derived, post-hoc (SF-38 P1-P4), exploratory |
| SF-39 compute table | estimate |
| Items E1-E7 and the design items | accepted scope |
| Lean (b) stage, non-periodic spline, exit handling, runner integration | proposed architecture (SF-41 DAG) |
| Peak memory of the lean (b) stage, end-effect decay, `e_v` order on (b), `alpha_L`, `alpha_T` | open question (SF-41, SF-39) |
