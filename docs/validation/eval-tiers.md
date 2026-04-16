# Eval tiers

Practical evaluation structure for MacroFlow3D changes.

---

## Tier A — Build + unit + smoke

**Applies to:** every change, no exceptions.

**Where:** local WSL.

### Commands

```bash
cmake --preset wsl-debug
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
```

### Pass criteria

- Configure succeeds
- Build succeeds with no new warnings in changed files
- All registered tests pass
- Smoke run completes without crash or assertion failure

### Artifacts

- Test stdout/stderr
- Smoke run stdout/stderr

### Maps to

- Gate 0 (repo/tooling hygiene)
- Gate 1 (build/smoke)

---

## Tier B — Operator / invariant integrity

**Applies to:** changes in `src/numerics/`, `src/multigrid/`, operator algebra, eigensolver backend, invariant construction.

**Where:** local WSL for operator tests. Remote V100 for PETSc/SLEPc.

### Commands (local)

```bash
ctest --test-dir build/wsl-debug --output-on-failure -R operator_tests
./build/wsl-debug/run_operator_tests
```

### Commands (remote, if PETSc/SLEPc involved)

```bash
scripts/rsync_to_v100.sh
ssh v100 'cd ~/MacroFlow3D && ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc'
ssh v100 'cd ~/MacroFlow3D && ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver'
```

### Pass criteria

- Operator tests pass
- Residual norms within expected tolerances
- No new unexplained residual growth
- If eigensolver touched: convergence succeeded, residuals small

### Artifacts

- Operator test output
- Residual norms
- Eigensolver convergence log (if applicable)

### Maps to

- Gate 2 (algebra/operator integrity)

---

## Tier C — Physics / ensemble

**Applies to:** PSPTA transport, macrodispersion output, ensemble statistics, or any change affecting the central scientific claim.

**Operational plan:** For PSPTA work, verify alignment with `docs/plans/active/pspta-execution-plan.md` and the current execution phase.

**Where:** local WSL for smoke. Remote V100 for production runs.

### Commands (local smoke)

```bash
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
./build/wsl-debug/macroflow3d_pipeline apps/config_pipeline_par2.yaml
```

### Commands (remote production)

```bash
scripts/rsync_to_v100.sh
scripts/remote_build_and_test.sh
scripts/remote_run_pipeline.sh apps/config_pipeline_pspta.yaml
scripts/remote_run_pipeline.sh apps/config_pipeline_par2.yaml
```

### Pass criteria

- All Tier A and Tier B criteria met
- PSPTA diagnostics inspected:
  - `v·∇ψ1`, `v·∇ψ2` residuals
  - independence / degeneracy signal
  - Newton failure counts and distribution
  - particle status summary (active / exited / failed)
- Before/after comparison if behavior changed
- Transverse macrodispersion not claimed as physical without control
- Run reproducible from config + commit

### Artifacts

- Full pipeline output
- Config file used
- Commit hash
- Build directory
- Diagnostic summaries
- Before/after metric comparison

### Maps to

- Gate 3 (PSPTA local integrity)
- Gate 4 (helicity-free regime)
- Gate 5 (ensemble/macrodispersion)

### Current automation status

Tier C is **not fully automated**. The commands exist and run, but:

- metric extraction is partly manual
- before/after comparison requires prior baseline
- scientific interpretation requires human judgment

This is intentional. Automation of comparison is a future goal, but premature automation risks hiding scientifically significant changes.

---

## Decision tree

```
Is this docs / scripts / AGENTS only?
  → Tier A

Does it touch src/numerics/, src/multigrid/, or operator code?
  → Tier A + Tier B

Does it touch src/physics/ or PSPTA?
  → Tier A + Tier B + Tier C

Does it change macrodispersion output or ensemble stats?
  → Tier A + Tier B + Tier C (mandatory before/after)
```

---

## Related

- `docs/validation/acceptance-gates.md` — gate definitions
- `docs/validation/validation-loop.md` — the fixed loop
- `docs/validation/local-remote-split.md` — where each tier runs
- `skills/macroflow-evals/SKILL.md` — agent-facing skill
