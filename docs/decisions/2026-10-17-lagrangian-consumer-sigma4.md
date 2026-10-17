# The production consumer at sigma_Y^2 >= 4 is Lagrangian (labels carried by forward-traced particles); the gridded labels become a diagnostic anchored by the O1 envelope and a blind robust metric

- Status: accepted (owner, 2026-10-17; human review on the SF-35 closure PR #53)
- Date: 2026-10-17
- Deciders: owner (orchestrator session 2026-10-17)
- Evidence: `docs/experiments/2026-10-07-sf35-label-transport.md` (SF-35 campaign and envelope),
  `docs/experiments/2026-10-17-sf35-sigma4-feasibility.md` (probes J1/J2, controls P1/P2),
  `docs/experiments/2026-10-05-sf30-streamline-closure-gate.md` §9 (trajectory resolution and ensemble agreement at
  `sigma^2 = 4`), `docs/experiments/2026-10-06-sf32-spurious-spreading.md` (DP5 transverse error on surrogates);
  runtime analyses `.claude/orchestration/SF-35-label-transport-constructor/analysis/{sigma4-feasibility.md,
  O6-evaluation.md}` (not versioned; the probe erratum is reproduced in the experiment note)
- Builds on: `docs/decisions/2026-10-07-label-transport-constructor.md` (tracing constructor, independence rule),
  `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md` (O1 = B inlet-anchored labels; O3 `alpha_T`
  not presupposed)
- Supersedes: the "planned consumer" line of `ARCHITECTURE.md` §4.3 (`streamline_tracker/` SF-31/32 as the
  consumer) and the mission's "correct pseudo-symplectic tracker" as a requirement; dashboard "Later phases" items
  2-4 of 2026-10-07

## Context

The owner requires the macrodispersion study to reach at least `sigma_Y^2 = 4` (one V100-32GB). SF-35 established
the tracing constructor of the gridded labels only up to `sigma_Y = 1` (`ell = 1/4`, seed 3001, 32^3-256^3); at
`sigma_Y = 1.5` criterion C rejects without inlet backflow. The probes of 2026-10-17 at `sigma_Y^2 = 4` showed
that 512^3 fits (25925 MiB peak, in the flow construction, after releasing dead temporaries) but that refinement
256 -> 512 does not extend the range of convergent label derivatives beyond `x1 = 0.75 ell`, and that
re-anchoring does not remove the obstruction: it is localized in the medium. The reading (hypothesis) is a thin
layer of extreme label gradient formed in a slow region and carried downstream, below what any uniform grid within
the memory ceiling resolves. In the planned long domain (2048x256x256, `ell/h = 10`, 1.34e8 cells) gridded labels
would be 12.8x coarser than 512^3 and unverifiable. SF-30 had already measured that at `sigma^2 = 4` individual
trajectories are not resolved beyond 4-8 periods while ensemble statistics agree across tolerances (spread of `R`
<= 3.5 %, of the transverse variances <= 12.8 %).

The quantity the science needs is the statistics of the plume (spatial moments, `alpha_L`, `alpha_T`), not pointwise
label derivatives everywhere. Lester et al. (2023) §5 name two numerical sources of spurious transverse spreading:
(A) a velocity representation that breaks the helicity-free structure, (B) integration that drifts off the
invariant surfaces (theory note §5).

## Decision

1. **Target.** The macrodispersion study reports `alpha_L` and `alpha_T` at `sigma_Y^2` in {1, 2.25, 4};
   `sigma_Y^2 = 6.25` is exploratory. `alpha_T` is not presupposed (Rule 1; Gates 4 and 5): every transverse growth
   is classified physical / numerical / unresolved with refinement evidence.
2. **Production consumer: Lagrangian labels ("O6").** Particles are traced forward with the tolerance-controlled
   DP5(4) arclength integrator of SF-30/SF-35 (Hénon landing on planes and on snapshot times) on the spline flow
   `v_s = k_s grad Phi_s` (`Phi_s = G.x + s_h`, SF-28 spline of the SF-19 potential fluctuation;
   `k_s = exp(s_Y)`, SF-28 spline of `eps Y`). Each particle carries its own labels — its seed and, where they
   exist, its D-1 inlet labels; at injection a local covector pair with `grad psi1 x grad psi2 = v_s(x0)` is enough
   (no D-1 data needed, so inlet backflow does not stop the case). Covectors are transported by
   `d(grad psi)/ds = -(grad u)^T grad psi`, `u = g/|g|`, and the accumulated divergence `int div v_s dt` is
   integrated along the path. Properties, separated by status:
   - **derived, exact for the traced field:** `curl(k grad Phi) = grad k x grad Phi`, hence
     `v_s . curl v_s = 0` pointwise (source (A) absent by the form of the representation); the streamlines of
     `v_s` are those of `grad Phi_s`; a particle's labels are constant by definition;
     `c = grad psi1 x grad psi2 = v_s exp(-int div v_s dt)` along each path (from `c || v_s`, `div c = 0`,
     `c(0) = v_s(x0)`), so the Lagrangian `e_v = 1 - exp(-int div v_s dt)` measures the representation's
     accumulated divergence without grid derivatives;
   - **not exact:** `div v_s = O(h^2)` (measured, criterion V4 of SF-37); the integration error (source (B)) is
     controlled by `tol` and measured by a paired tolerance ladder — there is **no structural guarantee**; the
     burden of proof is on the method (Rule 3);
   - **not claimed:** structure preservation "by construction" of the trajectories; any value of `alpha_T`.
3. **Gridded labels become a diagnostic.** The SF-35 tracing constructor stays the constructor of gridded labels;
   their verified envelope (O1: `sigma_Y <= 1`, `ell = 1/4`, seed 3001, 32^3-256^3) is the **anchor** for
   cross-validation of the Lagrangian consumer (SF-37, V3a) and is accepted against the SF-30 return map in the
   re-scoped SF-34. **O3**: a robust metric of the gridded labels (RMS excluding a fixed fraction of vertices ranked
   by self-convergence defect) is pre-registered in the SF-37 specification before any run and applied blind on
   fresh seeds; it is a diagnostic, non-blocking for the consumer.
4. **Verification of the consumer (independence rule for a consumer).** The Lagrangian consumer shares the
   integrator and the spline flow with the SF-30 closure oracle, which is therefore **not** independent of it. It is
   accepted only on: (i) controls with exact answers (`k = 1`; `k = k(x1)` layers; the Lester (2021) closed-
   streamline field; host/GPU equivalence; covectors against finite differences of the flow map; the identity of
   item 2); (ii) the gridded SF-35 labels inside their envelope (`|psi_grid(x_p) - psi_p|`); (iii) an independent
   representation comparator, Pollock (RT0) on the SF-19 finite-volume face fluxes — exactly solenoidal, `C^0`, not
   structure-preserving — whose plume statistics must approach those of the consumer under refinement and whose
   excess is the §5A leakage; (iv) paired tolerance and grid ladders on the plume statistics with
   confidence-interval criteria once trajectories decorrelate; (v) the Lagrangian `e_v`; and, for `alpha_L`, the
   Par2 RWPT baseline with the same configuration (Gate 5).
5. **Pseudo-symplectic tracker.** Cross-validation of the Lagrangian consumer against the SF-31 pseudo-symplectic
   tracker on gridded labels is deferred; the pseudo-symplectic tracker is future work, not a requirement of the
   study. It becomes feasible again only with verified gridded labels in the domain of the study (for instance after
   a non-periodic-in-`x1` spline, item 7).
6. **SF-34 re-scoped to the O1 arm.** Accept the SF-35 gridded labels against SF-30 at `sigma^2` in {0.25, 1},
   `ell` in {1/8, 1/16}; `sigma^2` 2.25 and 4 recorded, non-blocking. SF-34 is off the critical path of the
   consumer.
7. **Long domain.** Both media are evaluated on the same `K` with a pre-registered criterion (SF-38): (a) SF-19
   affine-periodic on the long box (the periodic spline exists; no boundary layers) and (b) `solve_head` with
   Dirichlet in `x`, periodic in `y`/`z` (the B&dD and Par2 configuration; needs a spline non-periodic in `x1`). (a)
   is used if its interior statistics agree with (b) within the sampling band; otherwise (b), which requires a
   non-periodic spline increment (provisional SF-40, not yet specified) before SF-39.
8. **Statistics.** The independent unit is the realization. Ladder comparisons are paired (same field, same
   seeds). Where trajectories decorrelate between ladder settings the criterion is statistical (difference within
   `k` standard errors; SE by block bootstrap). The number of realizations of the study is fixed by a pilot for a
   target confidence-interval half-width (B&dD: `N_R = 500`, `N_p = 1e4`, `lambda/Delta = 10`, variations <= 2 %
   longitudinal and <= 4 % transverse; Beaudoin & de Dreuzy 2013 §2 [7] and Figure 1, local copy
   `docs/references/Dreuzy-2013-WRR.pdf` p. 2491).
9. **Memory.** The release of dead `N^3` temporaries in `build_production_stage` (P2, byte-neutral at 64/128 in the
   probes) is adopted in SF-36 with its byte-identity control; the measured ceiling is ~202 B/cell at the stage peak
   (~1.6e8 cells on one V100-32GB for any option that solves the flow).

### Increment DAG

```
SF-35 (done) --+-- SF-34 (O1 arm: gridded labels sigma^2 <= 1 vs SF-30)           [parallel, off the critical path]
               +-- SF-36 (Lagrangian tracker core on the cube stage)
                      +-- SF-37 (verification on the cube at sigma^2 <= 4 + blind O3)
                      +-- SF-38 (long domain: infrastructure + medium A/B + ensemble runner + Par2 baseline)
                               +-- SF-39 (macrodispersion study, sigma^2 in {1, 2.25, 4}) <- also SF-37
```

At most two nonterminal increments: first SF-34 and SF-36, then SF-37 and SF-38. Remote work is serialized across
sessions or uses per-increment mirrors (`scripts/remote --increment SF-NN`).

### Stop rules

- R1: SF-37 V1 (tolerance) fails at `sigma^2 = 4` -> the consumer is not defensible at that amplitude; stop, owner.
- R2: SF-37 V5 shows exponential stretching -> only statistical criteria at long range (anticipated); if those do
  not converge either, stop, owner.
- R3: SF-19 does not converge on the long grid or the device peak exceeds 30000 MiB -> reduce the domain (e.g.
  `L_T = 12.8 lambda`) by owner decision.
- R4: SF-38 selects medium (b) -> a non-periodic spline increment (SF-40) precedes SF-39.
- R5: all probe evidence comes from one field -> SF-37 uses fresh seeds 3101-3103.

## Consequences

- SF-35 closed `done` with the O1 envelope; the three B-not-met points excluded by owner decision.
- New increments SF-36..SF-39 (human review required for each); SF-34 re-specified in place; the dashboard's
  "Later phases" rewritten; `check-lester-increments.sh` expects 40 increment files.
- `AGENTS.md` mission, `ARCHITECTURE.md` §4.3/§5, theory note §6, the solver overview's status, and Gate 4 of
  `docs/validation/acceptance-gates.md` (mapping for a Lagrangian-label consumer) describe the consumer as
  Lagrangian and the pseudo-symplectic tracker as future work.
- Not changed: the frozen periodic stack (SF-02..SF-26), SF-28, SF-30, SF-31, SF-32 (kept as verified
  infrastructure and as references), the SF-35 constructor and its outputs (byte-identity is a criterion of
  SF-36), the Gaussian-only covariance rule, Rule 1 and the non-presupposition of `alpha_T`.

## Classification (docs/AGENTS.md)

| item | class |
|---|---|
| 512^3 fits at `sigma_Y^2 = 4` with P2 (peak 25925 MiB in the stage); P2 byte-neutral at 64/128 | confirmed in runs (2026-10-17 note) |
| Refinement 256 -> 512 does not extend convergent `e_v_j` beyond `0.75 ell`; re-anchoring does not remove it | confirmed in runs (one field) |
| Thin unresolved label layer generated in a slow region and carried downstream | open question (hypothesis consistent with the runs) |
| SF-30: trajectories unresolved after 4-8 periods at `sigma^2 = 4`; ensemble statistics agree across tolerances | confirmed in runs (SF-30 note §9) |
| `v_s . curl v_s = 0` pointwise; `c = v_s exp(-int div v_s dt)` along paths; covector transport law | confirmed by derivation (this record, item 2) |
| Items 1-9 of the Decision; the increment DAG; the stop rules | accepted scope |
| Forward tracker module, covector/divergence integration, ensemble runner, Pollock repack, non-cubic stage | proposed architecture (SF-36, SF-37, SF-38) |
| `tol`-convergence of the plume statistics at `sigma^2 = 4`; `h`-convergence and the production `lambda/h`; stretching law; long-domain medium choice; `alpha_L`, `alpha_T` | open question (SF-37, SF-38, SF-39) |
