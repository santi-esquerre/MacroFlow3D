---
name: macroflow-remote-v100
description: Remote build / test / run on the V100 server. Use when a task requires release builds, PETSc/SLEPc, profiling, benchmarking, ensemble runs, or scientific validation on V100.
allowed-tools: Bash(ssh *) Bash(rsync *) Bash(scripts/*)
---

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
- **Sync** with `rsync` to `v100`.
- **Build and run** remotely via SSH.
- **Never edit** on the server.
- **Pull back** logs/results as needed.

## Sync

```bash
scripts/rsync_to_v100.sh
```

Verify:
```bash
ssh v100 'cd ~/MacroFlow3D && git log --oneline -1'
```

Override host/path:
```bash
REMOTE_HOST=myhost REMOTE_DIR=~/other/path scripts/rsync_to_v100.sh
```

## Remote build and test (one-shot)

```bash
scripts/remote_build_and_test.sh
```

With PETSc:
```bash
BUILD_DIR=build/v100-petsc ENABLE_PETSC=ON scripts/remote_build_and_test.sh
```

## Remote configure/build (manual)

### Release without PETSc
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  cmake -S . -B build/v100-release -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DMACROFLOW3D_ENABLE_PROFILING=ON \
    -DMACROFLOW3D_ENABLE_NVTX=ON \
    -DMACROFLOW3D_ENABLE_PETSC=OFF &&
  cmake --build build/v100-release -j
'
```

### Release with PETSc/SLEPc
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  cmake -S . -B build/v100-petsc -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DMACROFLOW3D_ENABLE_PETSC=ON \
    -DPETSC_DIR=$HOME/MacroFlow3D/src/external/petsc \
    -DPETSC_ARCH=arch-cuda \
    -DSLEPC_DIR=$HOME/MacroFlow3D/src/external/slepc &&
  cmake --build build/v100-petsc -j
'
```

### Profiling build
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  cmake -S . -B build/v100-prof -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DMACROFLOW3D_ENABLE_PROFILING=ON \
    -DMACROFLOW3D_ENABLE_NVTX=ON \
    -DMACROFLOW3D_ENABLE_PETSC=OFF &&
  cmake --build build/v100-prof -j
'
```

## Remote tests

```bash
# All tests (release)
ssh v100 'cd ~/MacroFlow3D && ctest --test-dir build/v100-release --output-on-failure'

# PETSc smoke
ssh v100 'cd ~/MacroFlow3D && ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc'

# SLEPc eigensolver validation
ssh v100 'cd ~/MacroFlow3D && ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver'
```

## Remote pipeline runs

For PSPTA-related runs, verify alignment with the current execution phase in `docs/plans/active/pspta-execution-plan.md`.

```bash
# PSPTA small smoke
scripts/remote_run_pipeline.sh apps/config_pspta_small.yaml

# PSPTA production-like
scripts/remote_run_pipeline.sh apps/config_pipeline_pspta.yaml

# Par2 baseline
scripts/remote_run_pipeline.sh apps/config_pipeline_par2.yaml
```

## Benchmarks

```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  ./build/v100-petsc/benchmark_eigensolver
'
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
- Do not run without first syncing (`rsync_to_v100.sh`).

## Related

- `docs/runbooks/remote-v100.md`
- `docs/runbooks/petsc-slepc.md`
- `scripts/rsync_to_v100.sh`
- `scripts/remote_build_and_test.sh`
- `scripts/remote_run_pipeline.sh`
