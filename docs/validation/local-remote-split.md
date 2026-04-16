# Local vs remote validation split

Where each validation activity runs.

---

## Local WSL

Local WSL is for fast iteration and light validation.

| Activity | Command |
|----------|---------|
| Lint / pre-commit | `pre-commit run --all-files` |
| Configure (debug) | `cmake --preset wsl-debug` |
| Configure (release) | `cmake --preset wsl-release` |
| Build | `cmake --build build/wsl-debug -j` |
| Unit tests | `ctest --test-dir build/wsl-debug --output-on-failure` |
| Operator tests | `ctest --test-dir build/wsl-debug -R operator_tests` |
| Smoke (PSPTA) | `./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml` |
| Smoke (Par2) | `./build/wsl-debug/macroflow3d_pipeline apps/config_pipeline_par2.yaml` |
| Documentation | edit and review locally |
| Script iteration | test locally before sync |

**Do not use local WSL for:**

- Performance conclusions
- Production-like runs
- PETSc/SLEPc builds (unless externals are built locally)
- Ensemble runs
- Benchmarks

---

## Remote V100

Remote V100 is for heavy validation and production-like runs.

| Activity | Command |
|----------|---------|
| Sync | `scripts/rsync_to_v100.sh` |
| Release build | `scripts/remote_build_and_test.sh` |
| PETSc build | `BUILD_DIR=build/v100-petsc ENABLE_PETSC=ON scripts/remote_build_and_test.sh` |
| PETSc smoke | `ssh v100 'cd ~/MacroFlow3D && ctest --test-dir build/v100-petsc -R smoke_test_petsc'` |
| SLEPc validation | `ssh v100 'cd ~/MacroFlow3D && ctest --test-dir build/v100-petsc -R validate_slepc_eigensolver'` |
| Profiling build | Use preset `v100-prof` |
| PSPTA production | `scripts/remote_run_pipeline.sh apps/config_pipeline_pspta.yaml` |
| Par2 production | `scripts/remote_run_pipeline.sh apps/config_pipeline_par2.yaml` |
| Benchmarks | `ssh v100 'cd ~/MacroFlow3D && ./build/v100-petsc/benchmark_eigensolver'` |
| Ensemble runs | remote only |

**Do not use the remote server for:**

- Editing code
- Running local-only hooks or linting
- Documentation authoring
- Git worktree management

---

## Mapping to eval tiers

| Tier | Local WSL | Remote V100 |
|------|-----------|-------------|
| A (build + unit + smoke) | **primary** | secondary (release build) |
| B (operator / invariants) | operator tests | PETSc/SLEPc tests |
| C (physics / ensemble) | smoke only | **primary** (production runs) |

---

## Flow

```
Local WSL:    lint → configure → build → test → smoke → [Tier B local]
                ↓
Remote V100:  rsync → build → test → [Tier B remote] → [Tier C production]
                ↓
Local WSL:    review results → create PR
```

---

## Related

- `docs/runbooks/local-wsl.md`
- `docs/runbooks/remote-v100.md`
- `docs/validation/eval-tiers.md`
- `docs/validation/validation-loop.md`
