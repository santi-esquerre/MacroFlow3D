# Legacy PSPTA AGENTS.md

## Scope

This file applies to `src/physics/particles/pspta/`.

This directory contains legacy PSPTA research infrastructure. It is not the active place to design new invariant construction.

The active invariant-construction direction is the Lester equation (14) streamfunction solver:

- `docs/plans/active/lester-eq14-streamfunction-solver-plan.md`
- `docs/decisions/2026-07-13-lester-eq14-streamfunction-solver.md`

## Status

Treat the existing PSPTA code as a frozen compatibility and migration surface.

It contains:

- legacy x-marching invariant construction;
- transport-near-nullspace operator/eigensolver infrastructure;
- refinement skeletons;
- a pseudo-symplectic transport engine;
- diagnostics and failure accounting.

Do not extend the old invariant-construction architecture. Do not add new Strategy A/C features. Do not make PSPTA the owner of the Lester equation (14) nonlinear solver.

## Allowed work in this directory

Allowed:

- auditing existing PSPTA behavior;
- preserving compatibility while the equation (14) path is brought up;
- extracting reusable containers, diagnostics, or transport-consumer interfaces;
- removing or archiving dead PSPTA pieces after a replacement exists and tests prove the removal is safe;
- fixing build breaks caused by the reformulation branch.

Not allowed without explicit user direction:

- new transport-near-nullspace invariant recovery;
- new eigensolver-based invariant construction;
- new Strategy A/C refinement work;
- silent changes to projection semantics;
- treating PSPTA smoke success as evidence that equation (14) invariants are correct.

## Required pre-reading

For any task in this directory, read:

- `docs/plans/active/lester-eq14-streamfunction-solver-plan.md`;
- `docs/theory/lester-2023-key-claims.md`;
- `docs/validation/acceptance-gates.md`.

Read `docs/plans/archive/pspta-execution-plan.md` only for historical context when auditing or retiring legacy PSPTA code.

## Migration rules

- Keep invariant construction separate from transport consumption.
- Prefer a new equation (14) construction module over burying nonlinear solver logic in `PsptaEngine`.
- If a PSPTA type is reused, document whether it is a confirmed reusable component or a temporary compatibility adapter.
- If a file is retained only for history, mark that in comments or docs rather than leaving it looking active.
- Do not delete code until the replacement path and validation are available, unless the user explicitly asks for removal of a known-dead artifact.

## Validation

For legacy PSPTA compatibility changes, run the existing compatibility checks:

```bash
ctest --test-dir <build-dir> --output-on-failure -R operator_tests
./<build-dir>/macroflow3d_pipeline apps/config_pspta_small.yaml
```

For equation (14) construction changes, Gate 3A in `docs/validation/acceptance-gates.md` applies. PSPTA-specific smoke tests are not sufficient for accepting the new solver.

## Done criteria

A task touching this directory is done only when:

1. the change is clearly classified as compatibility, migration, audit, or removal;
2. no legacy PSPTA path is accidentally re-promoted as active architecture;
3. relevant compatibility or Gate 3A validation is run or explicitly deferred;
4. remaining uncertainty is recorded in the final report or docs.
