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

**Applies to:** changes in `src/numerics/`, `src/multigrid/`, operator algebra, eigensolver backend, invariant construction, or Lester equation (14) linear operators.

**Where:** local WSL for operator tests. Remote V100 for PETSc/SLEPc.

### Commands (local)

```bash
ctest --test-dir build/wsl-debug --output-on-failure -R operator_tests
./build/wsl-debug/run_operator_tests
```

### Commands (remote, if PETSc/SLEPc involved)

```bash
scripts/remote sync
scripts/remote exec -- "ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc"
scripts/remote exec -- "ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver"
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

**Applies to:** Lester equation (14) invariant construction, legacy PSPTA compatibility/migration, macrodispersion output, ensemble statistics, or any change affecting the central scientific claim.

**Operational plan:** For new invariant construction, verify alignment with `docs/plans/active/lester-eq14-streamfunction-solver-plan.md`. Legacy PSPTA work is compatibility or migration only; use `docs/plans/archive/pspta-execution-plan.md` as historical context.

**Where:** local WSL for smoke. Remote V100 for production runs.

### Commands (local smoke)

```bash
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
./build/wsl-debug/macroflow3d_pipeline apps/config_pipeline_par2.yaml
```

### Commands (remote production)

```bash
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j && ctest --test-dir build/v100-release --output-on-failure"
scripts/remote run pspta-prod -- "./build/v100-release/macroflow3d_pipeline apps/config_pipeline_pspta.yaml"
scripts/remote wait pspta-prod
scripts/remote run par2-prod -- "./build/v100-release/macroflow3d_pipeline apps/config_pipeline_par2.yaml"
scripts/remote wait par2-prod
```

### Pass criteria

- All Tier A and Tier B criteria met
- Legacy PSPTA diagnostics inspected when that path is touched:
  - `v·∇ψ1`, `v·∇ψ2` residuals
  - independence / degeneracy signal
  - Newton failure counts and distribution
  - particle status summary (active / exited / failed)
- For Lester equation (14) solver work, Gate 3A metrics inspected:
  - coupled residual `r_F`
  - velocity reconstruction error `e_v`
  - Darcy invariance errors `e_i`
  - reconstructed-flow divergence `e_div`
  - denominator minimum and percentiles
  - gauge and regularization settings
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

- Gate 3A (Lester equation (14) solver integrity), or Gate 3 only for legacy PSPTA compatibility
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

Does it touch src/physics/ or legacy PSPTA?
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
