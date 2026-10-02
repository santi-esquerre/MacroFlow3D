# AGENTS.md

## Mission

MacroFlow3D is scientific software for 3D macrodispersion in heterogeneous porous media. The current strategic goal is to measure macrodispersion in smooth Gaussian lognormal fields with a correct pseudo-symplectic tracker. The Lester et al. equation (14) streamfunction solver is being generalized to `x1` non-periodic with inlet-anchored labels (Darcy labels); the periodic-fluctuation stack (SF-02..SF-26) is frozen verified infrastructure, because its periodic solution on generic Gaussian fields is not the Darcy flow (`docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`, `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`).

The previous PSPTA transport-near-nullspace/eigensolver route is legacy compatibility and migration surface. It is no longer the authoritative invariant-construction strategy and should not be extended unless the task explicitly targets audit, migration, or removal.

Optimize in this order:

1. scientific correctness
2. reproducibility
3. maintainability
4. performance
5. development speed

Do not trade correctness for convenience.

---

## How to work in this repository

- Plan before editing any non-trivial task. In an orchestrated Claude Code increment,
  the persisted intra-increment DAG is the authoritative implementation plan; do not
  create a competing plan that changes the increment scope.
- Read the closest `AGENTS.md` files before editing.
- **Implementation, corrective, and integration work must happen in isolated Git
  worktrees.** In Claude Code, use the project subagents with `isolation: worktree`;
  native worktrees live under `.claude/worktrees/` by default.
- The main orchestrator may remain in the control checkout for reading, auditing,
  comparing commits, updating orchestration metadata, and publishing the final PR,
  but it must not implement substantial source changes there.
- Keep changes **single-purpose**:
  - one physical hypothesis,
  - or one refactor,
  - or one tooling/documentation task.
- Prefer small, reviewable diffs.
- Do not mix solver changes, transport changes, and tooling changes in the same branch unless explicitly asked.

### Lester increment protocol

For any task in the Lester equation (14) streamfunction path:

1. Read `docs/plans/active/lester-eq14-streamfunction-solver-plan.md`.
2. Work only on an increment the checker reports READY (all dependencies `done`
   on the default branch); at most two increments may be nonterminal at a time,
   one orchestrator session each.
3. Read that increment specification and
   `docs/runbooks/lester-increment-workflow.md` completely.
4. Use the exact documented Goal as the persistent runtime goal when the
   agent environment supports goals.
5. Maintain the increment checklist and append-only bitácora during work.
6. Treat **increment ordering** and **intra-increment scheduling** separately:
   increments follow a dependency DAG (at most two nonterminal), and each active
   increment may be decomposed into a DAG with independent nodes run in parallel.
7. The deliverable of the autonomous implementation run is an **audited GitHub
   pull request**. Workers and the integrator do not publish or merge it; the
   orchestrator publishes it only after final audit.
8. For increments requiring human review, keep the delivery branch
   `awaiting_review` until explicit human approval. After approval, the
   orchestrator may add only a closure metadata commit on the same PR branch to
   set `done`, complete the checklist, and check its master-checklist entry.
9. No agent merges a PR. Do not start a dependent increment until the closure state
   is merged and visible on the repository default branch.

Run `bash scripts/hooks/check-lester-increments.sh` before committing any
change to the Lester plan or increment state.

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
  - legacy PSPTA transport/invariant infrastructure to audit, migrate, or retire
- `src/external/`
  - external dependency area
  - tracked vendored source: `yaml-cpp`, `nlohmann`
  - required git submodule: `Par2_Core`
  - optional local/remote-managed trees: PETSc/SLEPc
- `docs/`
  - runbooks, validation rules, decisions, plans, experiments

Read next when relevant:

- `ARCHITECTURE.md`
- `docs/validation/acceptance-gates.md`
- `docs/runbooks/local-wsl.md`
- `docs/runbooks/remote-v100.md`
- `docs/runbooks/petsc-slepc.md`

Active execution plans (read before starting work in the relevant area):

- `docs/plans/active/lester-eq14-streamfunction-solver-plan.md` — **authoritative operational plan** for the Lester et al. equation (14) streamfunction solver, including mathematical scope, discretization work, nonlinear strategy, continuation, validation, and CPU/GPU staging. Required reading for any work touching invariant construction, `psi1`/`psi2` differential operators, streamfunction residuals, or flow reconstruction from invariants.
- `docs/plans/archive/pspta-execution-plan.md` — archived historical plan for the old PSPTA transport-near-nullspace and eigensolver path. Read only when auditing or retiring legacy PSPTA code.

Scientific theory references (read before PSPTA, invariant, or macrodispersion work):

- `docs/theory/lester-2023-key-claims.md` — kinematic constraints, helicity-free regime, Lester equation (14), two-streamfunction representation, the paper's zero-transverse claim and its project-verified limits (closure oracle; `alpha_T` not presupposed)
- `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` — classical 3D macrodispersion baseline, Monte Carlo discipline, historical α_T expectations

More specific local rules live in:

- `src/physics/particles/pspta/AGENTS.md`
- `src/numerics/AGENTS.md`
- `docs/AGENTS.md`

---

## Canonical workflows

### 1. Local WSL development

Use WSL for reading, editing, and **light** validation only: configure, build,
and a small, fast, targeted subset of tests used to iterate on a change.

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
ctest --test-dir build/wsl-debug --output-on-failure -R <fast-targeted-case>
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
```

**The full local `ctest` suite is long-duration computation and must not be
run as increment acceptance/validation evidence.** Several registered `ctest`
entries are multi-case solver/continuation sweeps; running the full suite on
CPU-bound local GPU emulation routinely takes several minutes or more. See
rule 2: the authoritative full-suite/heavy run always goes to V100 as a
detached job, never to a local blocking call.

### 2. Remote V100 validation — long-duration computation is always a detached job

Use the remote server for heavy builds, profiling, PETSc/SLEPc, production-like
runs, and **any computation whose duration is not short and bounded**. This
explicitly includes:

- a full or near-full `ctest` run (as opposed to one fast `-R`-filtered case);
- any test binary that internally sweeps many cases (operator tests,
  streamfunction Picard/Anderson/continuation ladders, etc.);
- production-like pipeline runs, ensemble runs, benchmarks, and profiling runs.

**Rule:** long-duration computation must run as a `scripts/remote run <job>`
detached background job, polled with `scripts/remote wait <job>`. It must
never run inside a subagent's local isolated worktree, and it must never run
as a long blocking `scripts/remote exec` call that ties up the calling agent
turn for the whole duration. Reserve `scripts/remote exec` for short, bounded
steps (configure, compile, a single fast smoke command) that finish in
roughly under a minute.

Canonical flow:

```bash
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote run ctest-full -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote wait ctest-full
scripts/remote run pspta-small -- "./build/v100-release/macroflow3d_pipeline apps/config_pspta_small.yaml"
scripts/remote wait pspta-small
```

Do not assume local performance conclusions carry over to V100.
Do not handwrite ad hoc `ssh`/`tmux`/`rsync` command strings for normal remote work; use `scripts/remote`.

**Concurrency:** the remote mirror (`REMOTE_REPO_DIR` in `scripts/remote.env`,
default `~/MacroFlow3D`) is one shared execution surface, not one per
branch/worktree/agent. Two agents must not `scripts/remote sync` or run heavy
jobs against it at the same time; a second sync overwrites the tree a prior
job is still executing against. DAG nodes that both need remote execution must
be serialized even if their local write scopes would otherwise allow parallel
execution — see `docs/runbooks/remote-v100.md`.

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

For legacy PSPTA transport, eigensolver, or refinement code: treat the code as frozen compatibility surface unless the task is explicitly about audit, migration, or removal. New invariant construction belongs to the Lester equation (14) path.

For any new invariant-construction work, read `docs/plans/active/lester-eq14-streamfunction-solver-plan.md` first and confirm whether the change is in the operator-test, Picard, continuation, or Newton-Krylov phase. Do not jump directly to production grids.

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
- In orchestrated increments:
  - each worker commits only its assigned DAG node;
  - the integrator combines only orchestrator-approved commits;
  - workers and the integrator never push or open PRs;
  - only the orchestrator may push the final audited increment branch and create
    the PR;
  - the orchestrator never auto-merges the PR.
- PR descriptions should include:
  - increment Goal and scope,
  - DAG / delegated tasks,
  - commands run,
  - outputs checked,
  - acceptance evidence,
  - integration notes,
  - remaining risks,
  - files intentionally left untouched.

---

## Hard constraints / do-not rules

- Do **not** treat positive transverse macrodispersion in the smooth, locally isotropic, purely advective regime as automatically physical.
- Do **not** presuppose the value of `alpha_T` in any acceptance criterion; the paper's zero-transverse claim is recorded, not assumed.
- Do **not** start a solver for a new target object without an independent existence or positive-control check.
- Do **not** register multi-case solver sweeps or science smokes as `ctest` entries; they are documented experiments.
- Do **not** merge “it compiles” changes in the scientific core without validation evidence.
- Do **not** treat the existing multigrid preconditioner as automatically valid for `A psi = -div(q grad psi)` with `q=1/k`; reuse is a priority hypothesis that must be verified against the actual operator sign, coefficient placement, boundary conditions, gauge, and residual.
- Do **not** hide small `|grad psi1 x grad psi2|` denominators with arbitrary epsilons. Any regularization must be explicit, configurable, logged, and studied as it tends to zero.
- Do **not** treat exponential-covariance log-conductivity fields as equivalent to smooth Gaussian-covariance fields for invariant existence validation.
- Do **not** assume global streamfunction invariants exist for locally anisotropic tensor conductivity; that case is outside the initial scope.
- Do **not** rewrite major subsystems when a local change is enough.
- Do **not** introduce silent behavior changes in configs.
- Do **not** add fallback paths or compatibility layers unless explicitly requested.
- Do **not** use the remote server as an editing environment; local WSL is the source of truth and remote is for synchronized build/run/measure.
- Do **not** extend the old PSPTA invariant-construction architecture. It is legacy until replaced or intentionally retired.
- Do **not** run long-duration computation — a full/near-full `ctest` suite, a
  multi-case solver sweep, a production/ensemble run, a benchmark, or a
  profiling run — locally, or via a long blocking `scripts/remote exec` call.
  Launch it as a detached `scripts/remote run <job>` background job on V100 and
  collect evidence with `scripts/remote wait <job>` / `scripts/remote tail
  <job>`.

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

### Increment-level definition of done

For an orchestrated Lester increment, a successful worker or a green build is
not enough. The autonomous run is complete only when:

1. the orchestrator has read the required scientific, numerical, architectural,
   validation, and increment documents;
2. a complete intra-increment DAG has been constructed;
3. every required DAG node has been implemented in an isolated worktree;
4. the orchestrator has independently audited all worker results;
5. all blocking/major audit findings have been resolved through corrective DAGs;
6. one isolated integration agent has integrated only approved commits;
7. the orchestrator has independently audited the integrated commit against the
   original increment Goal and acceptance criteria;
8. the increment checklist, bitácora, and required durable docs are updated;
9. `check-lester-increments.sh` and all required validation gates pass;
10. the orchestrator has pushed the final audited branch and opened a GitHub PR.

The orchestrator must stop at the PR. The next increment remains disabled until
that PR is merged and the updated execution state is visible on the default
branch.
