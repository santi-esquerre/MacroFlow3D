# macroflow-remote-v100

Remote build / test / run on the V100 server.

## When to use

Use this skill when a task requires:
- release builds
- PETSc/SLEPc builds
- profiling or benchmarking
- ensemble or heavy pipeline runs
- scientific validation that depends on V100 hardware

## Model

- **Edit locally** in WSL worktree.
- **Sync, exec, and manage long jobs** through `scripts/remote`.
- **Never edit** on the server.
- **Pull back** logs/results as needed.
- **Do not handwrite** ad hoc `ssh` / `tmux` / `rsync` command strings unless you are debugging `scripts/remote` itself.
- The remote tree is an execution mirror, not a Git checkout. Do not rely on remote `git` commands.
- `src/external/Par2_Core` must be populated in the local worktree before sync. If the submodule is missing, initialize it locally instead of expecting the remote mirror to recover it.
- Remote PETSc/SLEPc trees under `src/external/petsc` and `src/external/slepc` are remote-managed and intentionally preserved across syncs.

## Canonical interface

Everything goes through one entry point:

```bash
scripts/remote sync
scripts/remote exec -- "<shell-command>"
scripts/remote run <job> -- "<shell-command>"
scripts/remote status <job>
scripts/remote tail <job>
scripts/remote wait <job>
scripts/remote cancel <job>
```

## Sync

```bash
scripts/remote sync
```

Verify:
```bash
scripts/remote exec -- "pwd && ls src/external"
```

Override host/path:
```bash
REMOTE_HOST=myhost REMOTE_REPO_DIR=~/other/path scripts/remote sync
```

## Remote build and test (one-shot)

```bash
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j && ctest --test-dir build/v100-release --output-on-failure"
```

With PETSc:
```bash
scripts/remote exec -- "cmake --preset v100-petsc && cmake --build build/v100-petsc -j && ctest --test-dir build/v100-petsc --output-on-failure"
```

## Remote tests

```bash
scripts/remote exec -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote exec -- "ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc"
scripts/remote exec -- "ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver"
```

## Remote pipeline runs

For PSPTA-related runs, verify alignment with the current execution phase in `docs/plans/active/pspta-execution-plan.md`.

```bash
# PSPTA small smoke
scripts/remote run pspta-small -- "./build/v100-release/macroflow3d_pipeline apps/config_pspta_small.yaml"
scripts/remote wait pspta-small

# PSPTA production-like
scripts/remote run pspta-prod -- "./build/v100-release/macroflow3d_pipeline apps/config_pipeline_pspta.yaml"
scripts/remote tail pspta-prod
scripts/remote wait pspta-prod

# Par2 baseline
scripts/remote run par2-prod -- "./build/v100-release/macroflow3d_pipeline apps/config_pipeline_par2.yaml"
scripts/remote wait par2-prod
```

## Benchmarks

```bash
scripts/remote run eig-bench -- "./build/v100-petsc/benchmark_eigensolver"
scripts/remote wait eig-bench
```

## Result retrieval

```bash
rsync -az v100:~/MacroFlow3D/output_* ./results/
```

Record for every meaningful run:
- commit hash
- build directory
- config used
- exact command
- relevant output path

## What NOT to do

- Do not edit files on the server.
- Do not treat V100 as the source of truth for code.
- Do not assume local WSL perf matches V100.
- Do not run without first syncing (`scripts/remote sync`).
- Do not bypass `scripts/remote` with raw `ssh`, `tmux`, or `rsync` for normal work.

## Related

- `docs/runbooks/remote-v100.md`
- `docs/runbooks/petsc-slepc.md`
- `scripts/remote`
