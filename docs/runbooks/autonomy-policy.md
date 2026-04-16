# Autonomy policy

What agents can do independently vs. what requires human review.

---

## High autonomy

Agents may create, modify, and merge changes in these areas after passing automated checks (Tier A):

- `docs/` — documentation, runbooks, plans, decisions, experiments
- `AGENTS.md` (any level) — agent instructions
- `skills/` — skill definitions
- `CMakePresets.json` — build presets
- `.pre-commit-config.yaml` — pre-commit hooks
- `scripts/` — automation, hooks, helpers
- `.github/` — PR templates, workflow configs
- `.codex/` — Codex configuration
- `.clang-format` — formatting rules
- `.gitignore` — ignore patterns

### Conditions

- Build must still succeed (Tier A).
- No silent behavior changes in configs that affect runtime.
- Documented workflows must remain executable.
- No deletion of required docs (enforced by pre-commit hook).

---

## Mandatory human review

These areas require human approval before merge, regardless of automated check results:

### Solver / operators

- `src/numerics/solvers/`
- `src/numerics/operators/`
- `src/numerics/blas/`

### Multigrid

- `src/multigrid/`

### Interpolation

- Any interpolation-affecting code in `src/physics/` or `src/numerics/`

### PSPTA tracking

- `src/physics/particles/pspta/`
- All files in invariants, transport, projection, Newton, gauge

### PETSc/SLEPc integration

- `src/physics/particles/pspta/invariants/SLEPcBackend.*`
- `src/runtime/PetscSlepcInit.*`
- PETSc/SLEPc linking in `CMakeLists.txt`

### Macrodispersion evaluation

- `src/runtime/analysis/`
- `src/runtime/ensemble/`
- Any code that computes or reports `α_L`, `α_T`

### Flow solve and velocity

- `src/physics/flow/`
- `src/physics/stochastic/`

### Application configs affecting physics

- `apps/config_pipeline_pspta.yaml`
- `apps/config_pipeline_par2.yaml`
- `apps/config_pspta_small.yaml`

### Why

Scientific code requires human judgment because:

- automated tests can pass while numerical behavior is silently wrong
- positive transverse macrodispersion in the target regime is not automatically physical
- invariant quality cannot be fully assessed by pass/fail tests alone
- eigensolver convergence does not prove downstream correctness

---

## Operational rules

1. **Before starting:** identify whether the change is high-autonomy or human-review.
2. **If high-autonomy:** run Tier A, create PR, merge after checks pass.
3. **If human-review:** run appropriate tier (B or C), create PR, request review, do NOT merge.
4. **If mixed:** split into separate branches. Merge high-autonomy parts first.
5. **If uncertain:** treat as human-review.

---

## Related

- `docs/validation/acceptance-gates.md`
- `docs/validation/eval-tiers.md`
- `docs/runbooks/pr-workflow.md`
- `skills/macroflow-pr-review/SKILL.md`
- `src/physics/particles/pspta/AGENTS.md`
- `src/numerics/AGENTS.md`
