# AGENTS.md

## Mission

MacroFlow3D is scientific software for 3D macrodispersion in heterogeneous porous media. The current strategic goal is to make the PSPTA / invariant-based transport path scientifically trustworthy for the helicity-free, smooth, locally isotropic Darcy regime.

Optimize in this order:

1. scientific correctness
2. reproducibility
3. maintainability
4. performance
5. development speed

Do not trade correctness for convenience.

---

## How to work in this repository

- Use **Plan mode first** for any non-trivial task.
- Read the closest `AGENTS.md` files before editing.
- Work in a **Git worktree**, not in the main checkout.
- Create worktrees under `~/src/MacroFlow3D/.agents/worktrees/`, not under `/tmp` or ad hoc locations.
- Keep changes **single-purpose**:
  - one physical hypothesis,
  - or one refactor,
  - or one tooling/documentation task.
- Prefer small, reviewable diffs.
- Do not mix solver changes, transport changes, and tooling changes in the same branch unless explicitly asked.

---

## Repo map

Top-level areas:

- `apps/`
  - entry points and runnable configs
  - main binary: `macroflow3d_pipeline`
  - key configs:
    - `apps/config_pipeline_par2.yaml`
    - `apps/config_pipeline_pspta.yaml`
    - `apps/config_pspta_small.yaml`
- `src/core/`
  - scalar types, grids, spans, low-level containers
- `src/runtime/`
  - CUDA context, pipeline runner, ensemble runner, analysis runner, I/O scheduling
- `src/io/`
  - config loading/validation, output layout, writers, manifest/effective config
- `src/numerics/`
  - BLAS-like ops, operators, solvers, preconditioners
- `src/multigrid/`
  - transfer, smoothers, V-cycle
- `src/physics/flow/`
  - head solve, velocity reconstruction, diagnostics
- `src/physics/stochastic/`
  - stochastic conductivity generation
- `src/physics/particles/par2_adapter/`
  - RWPT baseline transport path
- `src/physics/particles/pspta/`
  - PSPTA / invariant-based transport path
- `src/external/`
  - vendorized dependencies (`yaml-cpp`, `Par2_Core`, optional PETSc/SLEPc trees)
- `docs/`
  - runbooks, validation rules, decisions, plans, experiments

Read next when relevant:

- `ARCHITECTURE.md`
- `docs/validation/acceptance-gates.md`
- `docs/runbooks/local-wsl.md`
- `docs/runbooks/remote-v100.md`
- `docs/runbooks/petsc-slepc.md`

Active execution plans (read before starting work in the relevant area):

- `docs/plans/active/pspta-execution-plan.md` — **authoritative operational plan** for PSPTA, invariant recovery, eigensolver integration, refinement, and helicity-free validation. Required reading for any work touching PSPTA, invariants, eigensolver, refinement, or transverse macrodispersion assessment.

Scientific theory references (read before PSPTA, invariant, or macrodispersion work):

- `docs/theory/lester-2023-key-claims.md` — kinematic constraints, helicity-free regime, zero transverse macrodispersion in smooth isotropic Darcy
- `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` — classical 3D macrodispersion baseline, Monte Carlo discipline, historical α_T expectations

More specific local rules live in:

- `src/physics/particles/pspta/AGENTS.md`
- `src/numerics/AGENTS.md`
- `docs/AGENTS.md`

---

## Canonical workflows

### 1. Local WSL development

Use WSL for reading, editing, and light validation.

Typical local cycle:

```bash
cmake -S . -B build/wsl-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DMACROFLOW3D_ENABLE_DIAGNOSTICS=ON \
  -DMACROFLOW3D_ENABLE_PROFILING=OFF \
  -DMACROFLOW3D_ENABLE_NVTX=OFF \
  -DMACROFLOW3D_ENABLE_PETSC=OFF

cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
```

### 2. Remote V100 validation

Use the remote server for heavy builds, profiling, PETSc/SLEPc, and production-like runs.

Canonical flow:

```bash
scripts/rsync_to_v100.sh
scripts/remote_build_and_test.sh
scripts/remote_run_pipeline.sh apps/config_pspta_small.yaml
```

Do not assume local performance conclusions carry over to V100.

---

## Build, test, and run commands

### Configure without PETSc/SLEPc

```bash
cmake -S . -B build/wsl-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DMACROFLOW3D_ENABLE_PETSC=OFF
```

### Configure with PETSc/SLEPc

```bash
cmake -S . -B build/v100-petsc -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DMACROFLOW3D_ENABLE_PETSC=ON \
  -DPETSC_DIR=$HOME/MacroFlow3D/src/external/petsc \
  -DPETSC_ARCH=arch-cuda \
  -DSLEPC_DIR=$HOME/MacroFlow3D/src/external/slepc
```

### Build

```bash
cmake --build build/wsl-debug -j
```

### Test

```bash
ctest --test-dir build/wsl-debug --output-on-failure
```

### Targeted tests

```bash
ctest --test-dir build/wsl-debug --output-on-failure -R operator_tests
ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc
ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver
```

### Run the pipeline

```bash
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
./build/wsl-debug/macroflow3d_pipeline apps/config_pipeline_par2.yaml
```

### Useful direct executables

```bash
./build/wsl-debug/run_operator_tests
./build/v100-petsc/smoke_test_petsc
./build/v100-petsc/validate_slepc_eigensolver
```

---

## Engineering conventions

### Scientific changes

Any change touching:

- flow solve,
- velocity reconstruction,
- interpolation,
- transport stepping,
- invariant construction,
- macrodispersion analysis,

must include:

1. the scientific intent,
2. the numerical effect expected,
3. the validation path,
4. the likely regression surface.

For PSPTA, invariant, eigensolver, or refinement work: also read `docs/plans/active/pspta-execution-plan.md` and confirm the task aligns with the current execution phase before starting.

### Performance rules

- No allocations in hot loops.
- No hidden host-device synchronizations in hot paths.
- Reuse workspaces and buffers.
- Prefer explicit staging areas for diagnostics and I/O.

### Documentation rules

- Put durable project knowledge in the repo, not only in prompts.
- Update docs when behavior or accepted workflow changes.
- Keep the root `AGENTS.md` short; push details into `docs/` or local `AGENTS.md` files.

### Commit / PR hygiene

- One purpose per branch.
- Commit messages should say **what changed** and **why**.
- PR descriptions should include:
  - scope,
  - commands run,
  - outputs checked,
  - files intentionally left untouched.

---

## Hard constraints / do-not rules

- Do **not** treat positive transverse macrodispersion in the smooth, locally isotropic, purely advective regime as automatically physical.
- Do **not** merge “it compiles” changes in the scientific core without validation evidence.
- Do **not** rewrite major subsystems when a local change is enough.
- Do **not** introduce silent behavior changes in configs.
- Do **not** add fallback paths or compatibility layers unless explicitly requested.
- Do **not** use the remote server as an editing environment; local WSL is the source of truth and remote is for synchronized build/run/measure.
- Do **not** assume PSPTA is production-trusted by default; it is an actively validated path.

---

## Definition of done

A task is done only when all of the following are true:

1. the requested change is implemented,
2. the relevant build succeeds,
3. the relevant tests and/or runs were executed,
4. no obvious regression was introduced,
5. outputs/logs were checked at the right level,
6. docs were updated if expectations changed,
7. the final summary states exactly:
   - what changed,
   - what was run,
   - what passed,
   - any remaining risks or open questions.

If a task touches scientific behavior, “done” also requires alignment with `docs/validation/acceptance-gates.md`.
