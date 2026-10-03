# Eval tiers

Practical evaluation structure for MacroFlow3D changes.

---

## Tier A — Build + unit + smoke

**Applies to:** every change, no exceptions.

**Where:** local WSL for configure/build/fast-subset iteration; remote V100
(detached job) for the authoritative full-suite pass.

**What `ctest` holds:** since SF-27, `ctest` holds fast contract tests only.
Multi-case solver sweeps and science smokes are not registered as `ctest`
entries. They are documented experiments in the heavy tier (see
"Heavy tier — documented experiments (not ctest)" under Tier C), per
`docs/validation/acceptance-gates.md` "Validation tiers in ctest".

### Commands (local — fast dev loop)

```bash
cmake --preset wsl-debug
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R <fast-targeted-case>
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
```

### Commands (remote — authoritative full suite, detached)

The full suite is still long-duration computation: the fast contract tests
together take about 45 min on V100 (2 709.82 s measured in SF-27) and far
longer under local CPU emulation.
It must not be run locally as acceptance evidence. Run it on V100 as a
detached job instead:

```bash
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote run ctest-full -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote wait ctest-full
```

### Pass criteria

- Configure succeeds
- Build succeeds with no new warnings in changed files
- All registered tests pass (authoritative evidence: the remote detached job)
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

If `operator_tests` grows into a long-running multi-case sweep, treat it like
Tier C's remote detached runs instead of running it locally as acceptance
evidence.

### Commands (remote, if PETSc/SLEPc involved — detached, long-duration)

```bash
scripts/remote sync
scripts/remote run ctest-petsc-smoke -- "ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc"
scripts/remote wait ctest-petsc-smoke
scripts/remote run ctest-slepc-eigensolver -- "ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver"
scripts/remote wait ctest-slepc-eigensolver
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

### Commands (remote production — detached, long-duration)

```bash
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote run ctest-full -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote wait ctest-full
scripts/remote run pspta-prod -- "./build/v100-release/macroflow3d_pipeline apps/config_pipeline_pspta.yaml"
scripts/remote wait pspta-prod
scripts/remote run par2-prod -- "./build/v100-release/macroflow3d_pipeline apps/config_pipeline_par2.yaml"
scripts/remote wait par2-prod
```

Do not run two of these jobs concurrently against the same remote mirror; wait
for each before starting the next (see `docs/runbooks/remote-v100.md`
Section 0, Concurrency).

### Heavy tier — documented experiments (not ctest)

The heavy streamfunction cases (Anderson stall fixtures, heterogeneity
smokes, Newton difficult case, terminal D-gate and resolution recorders,
gauge-recombination heavy case) are not `ctest` entries. They are compiled
into `streamfunction_operator_tests` in `heavy_cases()`, outside `--list` and
the no-argument run, and are reachable only with an explicit `--case <name>`.
The index, with purpose, command, wall time, and last recorded outcome for
each case, is `docs/experiments/2026-10-02-heavy-streamfunction-cases-index.md`.

Run each case as a detached V100 job on an increment-scoped mirror:

```bash
scripts/remote --increment SF-NN sync
scripts/remote --increment SF-NN exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote --increment SF-NN run <job> -- "./build/v100-release/streamfunction_operator_tests --case <name>"
scripts/remote --increment SF-NN wait <job>
```

Some cases run for several hours. Record each result in an experiment note or
the increment bitácora; a red outcome is a recorded science result, not a
`ctest` failure.

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
