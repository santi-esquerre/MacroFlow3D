# Remote V100 runbook

This runbook defines the canonical remote workflow.

The server is used for:
- release builds,
- PETSc/SLEPc builds,
- heavy tests,
- profiling,
- production-like runs,
- larger scientific validation.

The server is **not** the primary edit surface.

---

## 1. Model

Source of truth:
- local WSL worktree

Execution surface:
- remote host `v100`

Transport mechanism:
- `rsync` over SSH

Canonical pattern:
1. edit locally,
2. validate lightly locally,
3. `rsync` to `v100`,
4. build and run remotely,
5. pull back logs/results if needed.

---

## 2. Remote repo layout

Recommended remote location:
```bash
~/MacroFlow3D
```

Recommended build directories:
```bash
~/MacroFlow3D/build/v100-release
~/MacroFlow3D/build/v100-petsc
~/MacroFlow3D/build/v100-prof
```

---

## 3. Sync

### Canonical sync
```bash
scripts/rsync_to_v100.sh
```

This sync should exclude:
- `.git`
- local build dirs
- editor caches
- temporary output
- local-only Codex files if desired

### Verify remote tree
```bash
ssh v100 'cd ~/MacroFlow3D && pwd && ls'
```

---

## 4. Remote configure/build

### 4.1 Release build without PETSc
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  cmake -S . -B build/v100-release -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DMACROFLOW3D_ENABLE_DIAGNOSTICS=OFF \
    -DMACROFLOW3D_ENABLE_PROFILING=ON \
    -DMACROFLOW3D_ENABLE_NVTX=ON \
    -DMACROFLOW3D_ENABLE_PETSC=OFF &&
  cmake --build build/v100-release -j
'
```

### 4.2 Release build with PETSc/SLEPc
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  cmake -S . -B build/v100-petsc -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DMACROFLOW3D_ENABLE_DIAGNOSTICS=OFF \
    -DMACROFLOW3D_ENABLE_PROFILING=ON \
    -DMACROFLOW3D_ENABLE_NVTX=ON \
    -DMACROFLOW3D_ENABLE_PETSC=ON \
    -DPETSC_DIR=$HOME/MacroFlow3D/src/external/petsc \
    -DPETSC_ARCH=arch-cuda \
    -DSLEPC_DIR=$HOME/MacroFlow3D/src/external/slepc &&
  cmake --build build/v100-petsc -j
'
```

---

## 5. Remote tests

### Release test pass
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  ctest --test-dir build/v100-release --output-on-failure
'
```

### PETSc/SLEPc targeted tests
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc
'
```

```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver
'
```

---

## 6. Remote runs

### Small PSPTA smoke
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  ./build/v100-release/macroflow3d_pipeline apps/config_pspta_small.yaml
'
```

### PSPTA production-like config
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  ./build/v100-release/macroflow3d_pipeline apps/config_pipeline_pspta.yaml
'
```

### Baseline Par2 config
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  ./build/v100-release/macroflow3d_pipeline apps/config_pipeline_par2.yaml
'
```

---

## 7. Profiling mode

When profiling:
- use a build with profiling/NVTX enabled,
- keep the config fixed,
- record exact commit and command line,
- do not change multiple variables at once.

Example profiling build:
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  cmake -S . -B build/v100-prof -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DMACROFLOW3D_ENABLE_DIAGNOSTICS=OFF \
    -DMACROFLOW3D_ENABLE_PROFILING=ON \
    -DMACROFLOW3D_ENABLE_NVTX=ON \
    -DMACROFLOW3D_ENABLE_PETSC=OFF &&
  cmake --build build/v100-prof -j
'
```

---

## 8. Result handling

For any meaningful remote run, record:
- commit hash
- build directory
- binary used
- config file used
- exact command
- relevant output path

If a run changes scientific conclusions, preserve the output directory and summarize it in:
- `docs/experiments/`
- or `docs/plans/`
- or the PR description

---

## 9. Failure triage

### Configure failure
Check:
- CUDA version / compiler
- `CMAKE_CUDA_ARCHITECTURES`
- PETSc/SLEPc paths
- missing `ninja`

### Build failure
Check:
- compiler output
- architecture mismatch
- stale build dir

### Test failure
Check:
- regression versus local
- environment mismatch
- accidental path/config drift

### Scientific output mismatch
Do not guess.
Compare:
- local config
- remote config
- build flags
- commit hash
- output manifests

---

## 10. Anti-patterns

Avoid:
- editing files directly on `v100`
- treating the remote tree as the canonical repo state
- running production-like experiments from unvalidated local changes
- overwriting remote outputs without preserving metadata
