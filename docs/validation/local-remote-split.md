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
| Sync | `scripts/remote sync` |
| Release build | `scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j && ctest --test-dir build/v100-release --output-on-failure"` |
| PETSc build | `scripts/remote exec -- "cmake --preset v100-petsc && cmake --build build/v100-petsc -j && ctest --test-dir build/v100-petsc --output-on-failure"` |
| PETSc smoke | `scripts/remote exec -- "ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc"` |
| SLEPc validation | `scripts/remote exec -- "ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver"` |
| Profiling build | Use preset `v100-prof` |
| PSPTA production | `scripts/remote run pspta-prod -- "./build/v100-release/macroflow3d_pipeline apps/config_pipeline_pspta.yaml"` |
| Par2 production | `scripts/remote run par2-prod -- "./build/v100-release/macroflow3d_pipeline apps/config_pipeline_par2.yaml"` |
| Benchmarks | `scripts/remote run eig-bench -- "./build/v100-petsc/benchmark_eigensolver"` |
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
Remote V100:  scripts/remote sync → scripts/remote exec -- ... → [Tier B remote] → scripts/remote run/wait
                ↓
Local WSL:    review results → create PR
```

---

## Related

- `docs/runbooks/local-wsl.md`
- `docs/runbooks/remote-v100.md`
- `docs/validation/eval-tiers.md`
- `docs/validation/validation-loop.md`
