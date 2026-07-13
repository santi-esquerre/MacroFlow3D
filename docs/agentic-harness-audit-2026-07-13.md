# Agentic harness audit - 2026-07-13

## Purpose

This audit records the repository structures that guide agent behavior and the current authority hierarchy after the Lester equation (14) direction update.

Use this file as orientation, not as a replacement for the files it references.

## Files and mechanisms found

### Automatically relevant to Codex-style agents

- `AGENTS.md`
  - Scope: whole repository unless overridden by a closer `AGENTS.md`.
  - Authority: highest repository-local instruction source for Codex behavior.
  - Role: mission, worktree policy, repo map, canonical workflows, scientific constraints.

- `docs/AGENTS.md`
  - Scope: `docs/`.
  - Role: documentation rules, required doc directories, decision/experiment/plan conventions.

- `src/numerics/AGENTS.md`
  - Scope: `src/numerics/`.
  - Role: numerical layer boundaries, operator/solver validation expectations.

- `src/physics/particles/pspta/AGENTS.md`
  - Scope: `src/physics/particles/pspta/`.
  - Role: legacy PSPTA audit, migration, compatibility, and removal rules.

### Claude-compatible harness files

- `CLAUDE.md`
  - Includes `@AGENTS.md` and `@ARCHITECTURE.md`.
  - States that `AGENTS.md` and `docs/` are authoritative, not `CLAUDE.md`.
  - Contains a routing table and legacy skill/agent references.

- `docs/CLAUDE.md`, `src/numerics/CLAUDE.md`, `src/physics/particles/pspta/CLAUDE.md`
  - Each includes `@AGENTS.md`.
  - No independent authority beyond the corresponding `AGENTS.md`.

### Local skills

- `skills/macroflow-build/SKILL.md`
  - Local configure/build/test workflow.

- `skills/macroflow-evals/SKILL.md`
  - Validation tier classifier and commands.

- `skills/macroflow-physics-review/SKILL.md`
  - Scientific review workflow.

- `skills/macroflow-pr-review/SKILL.md`
  - Worktree/branch/PR workflow.

- `skills/macroflow-remote-v100/SKILL.md`
  - Remote V100 sync/build/run interface.

These skills are persistent agent-facing context. They do not override `AGENTS.md`, but stale skill text can mislead future agent sessions.

### Plans, decisions, theory, validation

- `docs/plans/active/lester-eq14-streamfunction-solver-plan.md`
  - New authoritative plan for invariant construction through Lester equation (14).

- `docs/plans/archive/pspta-execution-plan.md`
  - Historical / legacy plan for transport-near-nullspace invariant recovery and PSPTA transport integration.
  - Read only when auditing or retiring old PSPTA transport, PETSc/SLEPc, and Strategy A/C code.
  - Superseded for new invariant construction and no longer active.

- `docs/theory/lester-2023-key-claims.md`
  - Scientific basis for helicity-free smooth scalar Darcy flow, two invariants, and equation (14).

- `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md`
  - Historical macrodispersion baseline and Monte Carlo discipline.

- `docs/validation/acceptance-gates.md`
  - Minimum validation gates for scientific and numerical behavior.

- `docs/validation/eval-tiers.md`
  - Tiered commands and evaluation expectations.

- `docs/decisions/`
  - Lightweight decision records.
  - New decision: `2026-07-13-lester-eq14-streamfunction-solver.md`.

- `docs/experiments/`
  - Experiment record templates and required fields.

### Scripts and hooks used by agents

- `scripts/create-worktree.sh`
  - Canonical worktree creator under `~/src/MacroFlow3D/.agents/worktrees/`.

- `scripts/create-pr.sh`
  - Pushes current branch and creates a PR through `gh`.

- `scripts/hooks/check-required-docs.sh`
  - Warns/blocks deletion of required docs in hook contexts.

- `scripts/hooks/scientific-core-guard.sh`
  - Non-blocking reminder for scientific-core changes.

- `scripts/hooks/validate-cmake-presets.sh`
  - Validates `CMakePresets.json`.

- `scripts/remote`
  - Canonical remote interface for sync, exec, long jobs, status, tail, wait, and cancel.
  - Present in the worktree along with `scripts/remote.env`.

### Hidden / workspace structures

- `.agents/worktrees/` exists in the main checkout and contains prior agent worktrees.
- No repository-local `.codex`, `.claude`, `.github`, or MCP resource files were found in the active worktree inventory.

## Authority hierarchy

When instructions conflict:

1. User/developer/system instructions for the current session.
2. Closest applicable `AGENTS.md`.
3. `docs/plans/active/lester-eq14-streamfunction-solver-plan.md` for new invariant construction.
4. `docs/validation/acceptance-gates.md` for required evidence.
5. `ARCHITECTURE.md` for system layering.
6. Skills and `CLAUDE.md` as routing/context helpers.
7. Historical plans, reports, and old comments.

For legacy PSPTA work, read `src/physics/particles/pspta/AGENTS.md`. Use the archived PSPTA plan only as history.

## Obsolete or contradictory information found

- `docs/plans/archive/pspta-execution-plan.md` explicitly said the project was "not going to solve the coupled nonlinear streamfunction PDEs directly." That conflicts with the new direction and is now marked as historical/superseded for invariant construction.
- Root `AGENTS.md`, `CLAUDE.md`, skills, architecture, and validation docs previously described PSPTA / transport-near-nullspace as the current strategic route. These were updated to point to the equation (14) plan for new invariant construction.
- Existing smoke configs use exponential covariance (`covariance_type: 0`). This is not a smooth Gaussian validation case and must not be used as evidence for the initial invariant-existence benchmark without explicit regularization analysis.
- `RefinementAC.cuh` describes a future Strategy C algorithm, while `RefinementAC.cu` returns `not_implemented`. The implementation state is skeleton only.
- Earlier docs consistently refer to `scripts/remote`; this was verified present during the audit.

## Confirmed facts vs plans

Confirmed facts:

- Code has cell-centered scalar fields, CompactMAC and padded velocity fields, variable-coefficient flow operator, PCG, MG, stochastic Gaussian/exponential generation, PSPTA engine, transport operator tests, and optional PETSc/SLEPc eigensolver code.
- Code does not yet contain a Lester equation (14) nonlinear solver.
- `RefinementAC` is not implemented.

Plans / hypotheses:

- Reusing PCG/MG for `A psi = -div(q grad psi)` is a priority hypothesis.
- Matrix-free Newton-Krylov is a future direction after Picard and residual validation.
- Anderson acceleration is an extension, not the first solver.

Open questions:

- Exact gauge and boundary formulation for throughflow versus triply periodic benchmarks.
- Compatibility of current MG coefficient coarsening with `q=1/k`.
- Whether existing differential operators are consistent enough for nonlinear residuals.
- Whether invariant construction should store `psi` as double before PSPTA casts to float.

## How agents should record progress

- Durable decisions: `docs/decisions/`.
- Experiment/run evidence: `docs/experiments/`.
- Active implementation steps: `docs/plans/active/`.
- Workflow changes: matching runbook in `docs/runbooks/`.
- Scientific changes: final report or PR should use the template in `docs/validation/acceptance-gates.md` and cite relevant theory.
