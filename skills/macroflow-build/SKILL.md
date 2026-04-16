# macroflow-build

Local configure / build / test flow for MacroFlow3D.

## When to use

Use this skill for any task that requires building the project locally in WSL.

## Prerequisites

- WSL environment with CUDA toolkit, CMake ≥ 3.18, Ninja
- Working directory is a git worktree or the repo root
- GPU architecture known (default: `86` for local WSL, `70` for V100)

## Presets available

| Preset | Use case |
|--------|----------|
| `wsl-debug` | Default local development, diagnostics ON |
| `wsl-release` | Local release build, diagnostics OFF |
| `wsl-petsc-debug` | Local PETSc/SLEPc debug (requires built externals) |

## Standard local flow

### 1. Configure

```bash
cmake --preset wsl-debug
```

Or manually:
```bash
cmake -S . -B build/wsl-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DMACROFLOW3D_ENABLE_DIAGNOSTICS=ON \
  -DMACROFLOW3D_ENABLE_PETSC=OFF
```

### 2. Build

```bash
cmake --build build/wsl-debug -j
```

### 3. Test

```bash
ctest --test-dir build/wsl-debug --output-on-failure
```

### 4. Smoke

```bash
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
```

## sccache

If `sccache` is installed, CMake auto-detects it via `find_program(SCCACHE_PROGRAM sccache)`.
No action needed. To verify:

```bash
cmake --preset wsl-debug 2>&1 | grep -i sccache
```

To install: see `docs/runbooks/sccache.md`.

## Targeted checks

```bash
# Operator tests only
ctest --test-dir build/wsl-debug --output-on-failure -R operator_tests

# Direct executable
./build/wsl-debug/run_operator_tests
```

## Release-ish local build

For local release checks before syncing to V100:

```bash
cmake --preset wsl-release
cmake --build build/wsl-release -j
ctest --test-dir build/wsl-release --output-on-failure
```

## What NOT to do locally

- Do not use local perf numbers as V100 conclusions.
- Do not run ensemble / heavy configs locally.
- Do not run PETSc/SLEPc builds unless explicitly needed and externals are built.

## Validation loop reference

See `docs/validation/validation-loop.md` for the full: configure → build → test → smoke → evals → PR loop.

## Related

- `docs/runbooks/local-wsl.md`
- `docs/runbooks/sccache.md`
- `docs/validation/eval-tiers.md`
- `CMakePresets.json`
