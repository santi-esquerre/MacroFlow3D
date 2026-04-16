# macroflow-evals

Eval tiers and execution commands for MacroFlow3D validation.

## When to use

Use this skill to determine which validation tier applies to a change and execute the corresponding commands.

## Eval tiers

### Tier A — Build + unit + smoke

**Applies to:** every change, no exceptions.

```bash
# Configure
cmake --preset wsl-debug

# Build
cmake --build build/wsl-debug -j

# Unit tests
ctest --test-dir build/wsl-debug --output-on-failure

# Smoke
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
```

**Pass criteria:**
- Configure and build succeed
- All tests pass
- Smoke run completes without crash or assertion failure

**Artifacts to preserve:** test output, smoke stdout/stderr.

---

### Tier B — Operator / invariant integrity

**Applies to:** changes in `src/numerics/`, `src/multigrid/`, `src/physics/particles/pspta/invariants/`, operator algebra, eigensolver backend.

**Scientific reference:** `docs/theory/lester-2023-key-claims.md` — operator integrity directly supports invariant quality. Residual growth or spectral drift can create spurious streamsurface leakage (Lester 2023 §5).

```bash
# Operator tests
ctest --test-dir build/wsl-debug --output-on-failure -R operator_tests

# Direct operator test runner
./build/wsl-debug/run_operator_tests
```

If PETSc/SLEPc is involved (remote V100):
```bash
# SLEPc smoke
ssh v100 'cd ~/MacroFlow3D && ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc'

# SLEPc eigensolver validation
ssh v100 'cd ~/MacroFlow3D && ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver'
```

**Pass criteria:**
- Operator tests pass
- Residual norms within expected tolerances
- No new unexplained residual growth
- If eigensolver: convergence succeeded, residuals small

**Artifacts to preserve:** operator test output, residual norms, eigensolver convergence log.

---

### Tier C — Physics / ensemble

**Applies to:** changes affecting PSPTA transport, macrodispersion output, ensemble statistics, or the central scientific claim.

**Scientific references:**
- `docs/theory/lester-2023-key-claims.md` — the regime where purely advective transverse macrodispersion should be zero
- `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` — classical 3D baseline for domain design, Monte Carlo discipline, and historical `α_L`/`α_T` expectations

When interpreting Tier C results, apply the Lester practical checklist (§ "Practical checklist for developers") before accepting transverse spreading as physical.

**Local (smoke-level):**
```bash
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
./build/wsl-debug/macroflow3d_pipeline apps/config_pipeline_par2.yaml
```

**Remote V100 (production-level):**
```bash
# Sync
scripts/rsync_to_v100.sh

# Build
scripts/remote_build_and_test.sh

# PSPTA production config
scripts/remote_run_pipeline.sh apps/config_pipeline_pspta.yaml

# Par2 baseline
scripts/remote_run_pipeline.sh apps/config_pipeline_par2.yaml
```

**Pass criteria:**
- All Tier A and B criteria met
- Change aligns with the current phase of `docs/plans/active/pspta-execution-plan.md`
- PSPTA diagnostics inspected:
  - `v·∇ψ` residuals
  - independence / degeneracy
  - Newton failure counts
  - particle status summary
- Before/after comparison if behavior changed
- Transverse macrodispersion not claimed as physical without control
- Run reproducible from reported config/commit

**Artifacts to preserve:** full pipeline output, config used, commit hash, build dir, diagnostic summaries.

**Honest status:** Tier C is not fully automated. The commands exist and run, but metric comparison and scientific interpretation require human review. The eval tier documents what to run and what to inspect — automation of comparison is a future goal.

---

## Decision tree: which tier?

```
Is this a docs/scripts/AGENTS change only?
  → Tier A is sufficient

Does it touch src/numerics/, src/multigrid/, or operator code?
  → Tier A + Tier B

Does it touch src/physics/ or PSPTA?
  → Tier A + Tier B + Tier C

Does it change macrodispersion output or ensemble stats?
  → Tier A + Tier B + Tier C (with mandatory before/after comparison)
```

## Mapping to acceptance gates

| Tier | Gates covered |
|------|--------------|
| A | Gate 0, Gate 1 |
| B | Gate 2 |
| C | Gate 3, Gate 4, Gate 5 |

## Related

- `docs/validation/eval-tiers.md` — full documentation
- `docs/validation/acceptance-gates.md` — gate definitions
- `docs/validation/validation-loop.md` — the fixed validation loop
- `docs/theory/lester-2023-key-claims.md` — kinematic constraints
- `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` — classical baseline
- `skills/macroflow-build/SKILL.md` — local build commands
- `skills/macroflow-remote-v100/SKILL.md` — remote commands
