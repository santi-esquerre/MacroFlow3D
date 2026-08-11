# SF-25 — Heterogeneity completion

- State: `pending`
- Goal: `Completar la continuación de heterogeneidad hasta lambda uno con convergencia terminal Newton para todas las varianzas objetivo.`
- Depends on: `SF-24`
- Unlocks: `SF-26`
- Branch: `science/lester-sf25-heterogeneity-completion`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Re-impose, UNCHANGED, the heterogeneity acceptance set that SF-21 closed
partially (owner option (a), 2026-08-11): the damped Picard/Anderson
fixed-point map proved non-contractive exactly at `eta=1` for
`sigma_Y^2 >= 1` (r_F plateau ~1e-3 at `lambda~0.5`, 32^3), which is the
regime the plan reserves for Newton terminal convergence. With the SF-22..24
Newton-Krylov stack accepted, drive every stage to tolerance and reach the
full lognormal target conductivity for all benchmark variances.

## Preconditions

- SF-21 (partial) provides the audited lambda/eta-rescue continuation
  machinery, the Anderson-wired stage solver, the pipeline surface, and the
  recorded partial evidence (32^3: sigma^2=0.25 reached lambda=1;
  sigma^2=1 floor-exhausted at lambda=0.5 with clean physics).
- SF-24 provides globalized Newton-Krylov with reproducible Picard fallback.

## In scope

- Wire Newton terminal convergence into the heterogeneity stage solver
  (activation from the accepted Picard/Anderson state per the SF-24
  activation-threshold policy; rescue/rollback semantics preserved).
- The MOVED, UNCHANGED SF-21 gates:
  - fixed-seed `32^3` smoke reaches `lambda=1` for `sigma_Y^2=1`
    (seed 12345, ell=8, epsilon fixed 1e-2, every accepted stage
    `r_F <= 1e-6`);
  - the `64^3` suite reaches `lambda=1` for `sigma_Y^2={0.25,1,2.25,4}`
    with `ell/h=8` (seed 12345), run as a recorded experiment through the
    pipeline binary; epsilon legs toward `1e-6` recorded where budget
    allows.

## Out of scope

- Grid transfer, `256^3`, exponential covariance, `sigma_Y^2=6.25`
  (prohibited until the `sigma_Y^2=4` suite is accepted).

## Files and symbols

- Extend the SF-21 stage solver path (`ContinuationController`,
  `StreamfunctionSolver`) with the Newton terminal phase; reuse the SF-21
  benchmark configs; extend stage records with the Newton/Anderson counters
  needed for per-stage attribution.

## Implementation specification

1. Preserve the SF-21 lambda axis, eta-rescue ordering, epsilon staging,
   coefficient-reuse lifecycle, and rollback semantics exactly.
2. Newton activates per stage only from the plan-locked threshold
   (`r_F < 1e-2` initially) and falls back to Picard/Anderson per SF-24.
3. No gate, seed, budget, tolerance, or fixture value may change from the
   SF-21 prespecification; any conflict is a recorded failure.

## Expected numerical effect

Stages that plateau at `eta=1` under the fixed-point map converge
quadratically once inside the Newton basin, making `lambda=1` reachable for
all benchmark variances with controlled cost.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/v100-release --output-on-failure -R streamfunction_heterogeneity
./build/v100-release/macroflow3d_pipeline apps/config_streamfunctions_gaussian_smoke32.yaml
./build/v100-release/macroflow3d_pipeline apps/config_streamfunctions_gaussian_64_var4.yaml
```

(Validation executes on the remote V100 per the standing venue policy.)

## Acceptance thresholds

- `32^3` smoke reaches `lambda=1` for `sigma_Y^2=1` (the SF-21 sigma^2=0.25
  pass stands as recorded evidence and must not regress).
- The `64^3` suite reaches `lambda=1` for `0.25, 1, 2.25, 4` with `ell/h=8`.
- Every accepted stage reports `r_F`, velocity, invariance, divergence,
  degeneracy, eta, epsilon, MG rebuild count, and solver-phase attribution.

## Regression surface

- SF-21 accepted behavior (sigma^2=0.25 smoke, default paths byte-stable),
  Newton/Picard handoff correctness, continuation rollback, total runtime.

## Failure and rollback policy

- Minimum lambda-step failure remains a recorded physical/numerical failure;
  do not skip lambda intervals.
- `sigma_Y^2=6.25` is prohibited in this increment.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Newton terminal convergence is wired into the heterogeneity stage solver.
- [ ] The moved 32^3 sigma^2=1 gate passes unchanged.
- [ ] The 64^3 suite meets the moved acceptance set unchanged.
- [ ] Full Gate 3A metrics and experiment notes are recorded.
- [ ] Gate 4 interpretation and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-25 complete and selects SF-26.
<!-- completion-checklist:end -->

## Advancement rule

SF-26 may prolong accepted lambda-one solutions to finer versions of the same
periodic realization.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-11T23:30Z | created by the owner-directed option (a) restructuring | This increment inherits, UNCHANGED, the sigma^2>=1 heterogeneity gates that SF-21 closed partially: 32^3 sigma^2=1 (floor-exhausted at lambda=0.5 under Picard/Anderson, non-contractive eta=1 plateau r_F~1e-3, zero degeneracy) and the full 64^3 suite. See `docs/decisions/2026-08-11-newton-before-heterogeneity-completion.md` and the SF-21 bitácora/audit records for the complete evidence. | Gates moved verbatim; no tuning. | Activate only when named by `NEXT` after SF-24 is done. |
