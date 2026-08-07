# Autonomy policy

## Purpose

Define what Claude Code agents may do autonomously and where human judgment is
required.

This policy distinguishes **agent autonomy during implementation/audit** from
**merge authority**. Agents may work with `bypassPermissions`, create commits,
run validation, and the orchestrator may publish a pull request, but **no agent
merges a pull request**. Merge is always a human boundary.

---

## Universal merge boundary

Regardless of change type:

- workers never push or open PRs;
- the integrator never pushes or opens PRs;
- only the main `orchestrator` may push the final audited branch and create or
  update the PR;
- no agent runs `gh pr merge`, merges through the GitHub UI, or pushes directly
  to the default branch;
- a human performs the final merge.

`bypassPermissions` removes Claude Code permission prompts. It does **not**
change this project policy.

---

## High-autonomy changes

These areas may be implemented, audited, and published as **ready-to-merge PRs**
after the required automated checks without a mandatory scientific human review:

- `docs/` — documentation, runbooks, plans, decisions, experiments;
- `AGENTS.md` and `CLAUDE.md` files — agent instructions/context;
- `.claude/` — Claude Code configuration and project agents;
- `skills/` — skill definitions;
- `CMakePresets.json` — build presets;
- `.pre-commit-config.yaml` — pre-commit hooks;
- `scripts/` — automation, hooks, helpers;
- `.github/` — PR templates and workflow configs;
- `.codex/` — Codex compatibility configuration;
- `.clang-format` — formatting rules;
- `.gitignore` and `.worktreeinclude` — repository/worktree support files.

### Conditions

- The applicable Tier A checks must pass.
- No silent runtime behavior change may be introduced through configuration.
- Documented workflows must remain executable.
- Required harness documents must not be deleted.
- If a high-autonomy change is bundled with scientific-core behavior, the
  stricter human-review policy applies to the entire PR.

High-autonomy means **no mandatory scientific review gate before the PR is
ready to merge**. It does not authorize an agent to merge the PR.

---

## Mandatory human review

These areas require explicit human approval before the increment can be marked
`done` in its delivery branch and before merge.

### Lester streamfunction construction

- `src/physics/streamfunctions/`
- any code that constructs, reconstructs, validates, or consumes `psi1` / `psi2`
  as scientific invariants;
- any change to the locked Lester mathematical/discrete contracts.

### Solver / operators

- `src/numerics/solvers/`
- `src/numerics/operators/`
- `src/numerics/blas/`

### Multigrid

- `src/multigrid/`

### Interpolation

- any interpolation-affecting code in `src/physics/` or `src/numerics/`.

### Legacy PSPTA tracking / migration

- `src/physics/particles/pspta/`
- all legacy invariants, transport, projection, Newton, and gauge code.

### PETSc/SLEPc integration

- `src/physics/particles/pspta/invariants/SLEPcBackend.*`
- `src/runtime/PetscSlepcInit.*`
- PETSc/SLEPc linking in `CMakeLists.txt`.

### Macrodispersion evaluation

- `src/runtime/analysis/`
- `src/runtime/ensemble/`
- any code that computes or reports `alpha_L` / `alpha_T`.

### Flow solve and velocity

- `src/physics/flow/`
- `src/physics/stochastic/` when the statistical field definition affects the
  scientific case.

### Application configs affecting physics

- `apps/config_pipeline_pspta.yaml`
- `apps/config_pipeline_par2.yaml`
- `apps/config_pspta_small.yaml`
- any new config whose values alter a scientific model or acceptance case.

### Why

Scientific code requires human judgment because:

- automated tests can pass while numerical behavior is silently wrong;
- positive transverse macrodispersion in the target regime is not automatically
  physical;
- invariant quality cannot be fully assessed by pass/fail tests alone;
- convergence of an algebraic solver does not prove physical correctness;
- a numerically plausible change can violate a locked discretization or
  kinematic contract.

---

## Human approval semantics

For a PR that requires human review, one of the following counts as explicit
approval for the project workflow:

1. an approving GitHub review; or
2. an explicit instruction from the repository owner/maintainer to finalize the
   reviewed PR for merge.

The approval must refer to the exact source-bearing PR head that was reviewed.
After approval, the orchestrator may add a **closure-only metadata commit** to
the same PR branch. That commit may update only:

- increment state/checklist/bitacora;
- dashboard `NEXT`, active goal, and master checklist;
- PR/documentation metadata needed to record closure.

If source, tests, scientific configuration, or numerical behavior changes after
approval, the prior approval is stale: return to audit and human review.

---

## Operational rules

1. **Before starting:** classify the increment as high-autonomy or
   human-review. If uncertain, use human-review.
2. **During implementation:** workers and integrator stay in isolated
   worktrees. Fable audits every candidate result.
3. **Before publication:** the final integrated source commit must pass Fable's
   final audit and all required validation.
4. **High-autonomy increment:** finalize versioned closure metadata, publish the
   PR, and leave merge to a human.
5. **Human-review increment:** publish the audited source PR as
   `awaiting_review`; do not advance `NEXT` yet.
6. **After explicit human approval:** resume the same PR, make only the
   closure-only metadata update, set the increment `done`, advance `NEXT`, run
   the harness checker, and push that metadata commit.
7. **Merge:** human only.
8. **After merge:** the default branch is authoritative. The next increment may
   start only if its predecessor is `done` there and the dashboard points to it.
9. **Mixed scope:** if scientific and high-autonomy changes are inseparable
   within one increment, apply human-review policy to the whole PR.

---

## Exceptional closure repair

If a human merges a reviewed implementation PR while the versioned increment is
still `awaiting_review`, the code may be accepted while the harness state is
stale. Do not start the next increment by inference.

Instead create a **closure-repair PR** containing only durable state repair:

- verify the implementation PR is actually merged into the default branch;
- record its canonical default-branch commit;
- record the human approval/merge fact without inventing a GitHub review event;
- mark the increment `done` and complete its checklist;
- update the dashboard master checklist, `Last completed increment`, active
  goal, and `NEXT`;
- append an explanatory bitacora entry;
- run `bash scripts/hooks/check-lester-increments.sh` plus the documentation
  checks required by the repair;
- do not modify scientific source code.

SF-08 / PR #18 is the historical case that motivated this repair rule.

---

## Related

- `docs/validation/acceptance-gates.md`
- `docs/validation/eval-tiers.md`
- `docs/runbooks/pr-workflow.md`
- `docs/runbooks/lester-increment-workflow.md`
- `docs/runbooks/claude-increment-orchestration.md`
- `skills/macroflow-pr-review/SKILL.md`
- `src/physics/particles/pspta/AGENTS.md`
- `src/numerics/AGENTS.md`
