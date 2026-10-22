# A lean, parametric Lagrangian stage (SF-40, bitwise-only, memory level S+) precedes the long domain; the provisional non-periodic spline becomes SF-41

- Status: accepted (owner, 2026-10-22, planning session; ratified by the human merge of the PR that adds this record)
- Date: 2026-10-22
- Deciders: owner (planning session 2026-10-22)
- Evidence: `docs/experiments/2026-10-22-lagrangian-memory-audit.md` (audit; §3, §4, §8, §9, §10 cited below),
  `docs/experiments/2026-10-19-sf37-lagrangian-cube-verification.md` (SF-37 `lambda/h` ladder and measured peaks),
  `docs/experiments/2026-10-22-sf37-posthoc-review-and-mecom-feasibility.md` (§I.5 item 2 and §II.3 raised the
  resolution-versus-domain tension),
  `docs/experiments/2026-10-17-sf35-sigma4-feasibility.md` (512^3 fits at `sigma_Y^2 = 4`, 25925 MiB)
- Builds on: `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`
- Supersedes / refines (without editing that record, which is read with this record's numbering and memory figures):
  - its "provisional SF-40" numbering: item 7 and stop rule R4 now read **SF-41**;
  - its item 9 memory ceiling (~202 B/cell, ~1.6e8 cells): refined by the audit to an exact-byte stage peak of
    ~213 B/cell at the second prefilter and a measured peak distribution 25921 ... 27981 MiB at 1.34e8 cells
    (audit §4, §5);
  - dashboard "Later phases" item 4 and SF-38's dependency (now SF-36 and SF-40).

## Context

SF-37 fixed the production `lambda/h` of the Lagrangian consumer: 8 / 16 / 32 at `sigma_Y^2` = 1 / 2.25 / 4
(`docs/experiments/2026-10-19-sf37-lagrangian-cube-verification.md`). With the current stage (~197 B/cell in the SF-19
solve phase, ~213 B/cell at the stage peak by exact bytes; audit §4) one V100-32GB holds ~1.45e8 to ~1.57e8 cells
(audit §8-§9). That does not reach the validated `lambda/h` for `sigma_Y^2 >= 2.25` on a long domain: SF-38's
specified 2048x256x256 production box (1.34e8 cells) is `lambda/h` 10, and the legacy `lambda/h ~ 11` was shown to carry
estimated errors at `sigma_Y^2 = 4` (audit §3; post-hoc review §I.5 item 2, §II.3).

The audit found that the peak is orchestration and duplication, not the tracker: the tracker costs ~16 B/cell, while
the flow construction keeps three correctors, a copy of `K` in the level-0 MG coefficient, separate flux/divergence arrays and
host-reconstructable fields alive on the device (audit §4, §8). Techniques that remove them without changing a bit of
the result exist as designs (levels O1, O2, S, S+; audit §8); none is implemented or verified yet.

## Decision

1. **Bitwise-only policy.** Every technique of the memory and time work must reproduce the current outputs byte for
   byte. A technique that breaks bytes is discarded; "almost identical" is not accepted. (Non-bitwise options are
   item 7.)
2. **New increment SF-40** (`docs/plans/active/lester-eq14/increments/SF-40-lean-lagrangian-stage.md`): a lean, parametric stage on `Grid3D` (non-cubic) at memory
   level **S+**, comprising
   - O1: orchestration only (2 of 3 correctors offloaded to host; flux/divergence on dead PCG/MG buffers; `rhs_b` in the
     PCG `z`; `rhs_a` in a level-0 MG buffer; `lnK` on host; `u,v,w` not alive during the PCG);
   - O2: K loaded directly into the level-0 MG coefficient, coarse levels built by the public
     `coarsen_coefficient_geometric` loop (no copy of K);
   - S: a new variant of `projected_pcg_solve` with external vectors where `z`, `Ap` and the MG level-0 `b` share one
     buffer;
   - S+: `b_hat` in pinned, mapped host memory;
   - bitwise tracker speedups (fused two-spline evaluation, single-launch plane+time passes), opt-in;
   - removal of the Par2 baseline's dead buffers `FlowWorkspace::{residual, aux1, aux2}`.

   Additive and opt-in: the frozen periodic stack and the default paths of SF-35/36/37 are untouched.
3. **Numbering.** SF-40 is the lean stage. The provisional non-periodic-in-`x1` spline of the 2026-10-17 decision (item
   7, stop rule R4) becomes **SF-41**; it exists only if SF-38 selects medium (b) (Dirichlet in `x`, periodic in
   `y`/`z`).
4. **Resolution.** The legacy resolution (`lambda/h ~ 11`, MECOM) is discarded. Studies use the `lambda/h` validated by
   SF-37 (8 / 16 / 32 at `sigma_Y^2` = 1 / 2.25 / 4); audit §3.
5. **Memory target.** Level S+ with the box **3328x384x384** (490,733,568 = 4.9e8 cells) within **30000 MiB measured**
   on one V100-32GB. The model gives 28847 MiB including a 500 MiB CUDA context (60.57 B/cell; audit §8-§9). In
   `lambda` units the box is 208 x 24 x 24 at `lambda/h` 16 (`sigma_Y^2` 2.25) and 104 x 12 x 12 at `lambda/h` 32
   (`sigma_Y^2` 4). The 30000 MiB target is acceptance; the model figure is a prediction to be measured. The margin to 30000 MiB is 0.5-1.1 GiB, because the unmodeled device
   overhead grows with size (485 / 622 / 748 MiB measured; ~1129 MiB extrapolated at 4.9e8 cells; audit §4, §9); the
   measured peak of SF-40 criterion D decides.
6. **DAG.** SF-40 depends on SF-37 and unlocks SF-38; SF-38 depends on SF-36 and SF-40. SF-38's Goal is not changed;
   its production geometry and `lambda/h` are fixed at its activation from SF-37 and SF-40's measured capacity.
7. **Recorded, not in SF-40's scope.**
   - the `read_timeseries_csv` header bug (audit §7): fix in SF-38 (whose criterion D depends on it) or in a separate
     bugfix PR;
   - wiring the tracker into `EnsembleRunner`: SF-38;
   - medium (b) savings (audit §10);
   - non-bitwise options below ~60 B/cell (mixed precision, single corrector with imposed gradient, host flow solve,
     two GPUs): each needs a new owner decision;
   - the exact B&dD transverse domain (25.6 `lambda`) does not fit even at S+, and at `lambda/h` 32 the B&dD domain
     (204.8 x 25.6 x 25.6 `lambda`, 4.40e9 cells; 5.50e8 cells at `lambda/h` 16) is infeasible on one GPU (audit §9).
     The domain of SF-38/SF-39 remains an owner decision (stop rule R3 of 2026-10-17).

### Increment DAG

```
SF-37 (done) --- SF-40 (lean parametric Lagrangian stage, S+, bitwise) --- SF-38 (long domain; also SF-36 done)
                                                                             +-- SF-39 (study; also SF-37)
SF-34 off the critical path;  SF-41 only if SF-38 selects medium (b), before SF-39
```

At most two nonterminal increments. Remote work stays serialized across sessions or uses per-increment mirrors
(`scripts/remote --increment SF-NN`).

## Consequences

- New spec SF-40; the dashboard master-checklist entry, "Later phases" and re-sequencing are updated; SF-38 gets
  `Depends on: SF-36, SF-40` with an append-only bitácora row; `scripts/hooks/check-lester-increments.sh` expects 41
  increment files.
- SF-38 is no longer READY until SF-40 is `done` on the default branch; SF-40 is READY once this PR is merged.
- Capacity statements of the 2026-10-17 record (item 9: ~202 B/cell, ~1.6e8 cells) are superseded by audit §4, §8-§9
  until SF-40 measures the new levels.
- Not changed: the frozen periodic stack (SF-02..SF-26), SF-18/SF-19/SF-28 numerics, the outputs and records of
  SF-35/36/37, the 2026-10-17 record (read with the numbering note above), Rule 1 and the non-presupposition of
  `alpha_T`, the Gaussian-only covariance rule. No `alpha_L`/`alpha_T` value is claimed here.

## Classification (docs/AGENTS.md)

| item | class |
|---|---|
| Current stage peak (~197 B/cell solve phase, ~213 B/cell at the second prefilter) | confirmed in code (audit §4, §8) |
| Measured peak distribution 25921 ... 27981 MiB at 1.34e8 cells (sampler misses a ~2 GiB transient) | confirmed in runs (SF-35 probes; SF-37 `sf37-v2.log`, `sf37-science.log`; audit §4-§5) |
| Level byte counts (92.57 / 84.57 / 68.57 / 60.57 B/cell), capacity 5.11e8 cells at S+, 3328x384x384 at 28847 MiB; the margin depends on the unmodeled overhead (open) | derived (audit §8-§9, model script); to be measured in SF-40 |
| S+ time cost (~+8 s per realization at 4.9e8 cells) | estimate (audit §8) |
| Bitwise feasibility of each technique (O1, O2, S, S+, tracker speedups, Par2 buffer removal) | open question (SF-40 A/B) |
| MG convergence on the 52x6x6 coarsest level of 3328x384x384 (7 levels, 50 coarse sweeps) | open question (SF-40 E) |
| Items 1-7 of the Decision; the increment DAG | accepted scope |
| Lean stage API and modules | proposed architecture (SF-40 spec) |
| B&dD domain, non-bitwise options, medium (b) savings, domain of SF-38/SF-39, `alpha_L`, `alpha_T` | open question (owner / SF-38 / SF-39) |
