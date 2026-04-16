# PETSc / SLEPc runbook

This runbook covers the optional eigensolver backend used by the PSPTA / invariant stack.

Use this path only when the task actually touches:
- eigensolver backend work,
- invariant construction via the transport-operator route,
- PETSc/SLEPc integration,
- grid-ladder or convergence studies that require this stack.

---

## 1. What the current repo expects

The top-level CMake logic expects:

- `MACROFLOW3D_ENABLE_PETSC=ON`
- `PETSC_DIR` (defaults to `src/external/petsc`)
- `PETSC_ARCH` (defaults to `arch-cuda`)
- `SLEPC_DIR` (defaults to `src/external/slepc`)

It also checks for:
- `${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.a`
- `${SLEPC_DIR}/${PETSC_ARCH}/lib/libslepc.a`

If those are missing, configure should fail early.

---

## 2. Canonical remote configure

```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  cmake -S . -B build/v100-petsc -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DMACROFLOW3D_ENABLE_PETSC=ON \
    -DPETSC_DIR=$HOME/MacroFlow3D/src/external/petsc \
    -DPETSC_ARCH=arch-cuda \
    -DSLEPC_DIR=$HOME/MacroFlow3D/src/external/slepc
'
```

Then:
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  cmake --build build/v100-petsc -j
'
```

---

## 3. First-line sanity checks

### Smoke test
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  ./build/v100-petsc/smoke_test_petsc
'
```

What it should prove:
- PETSc/SLEPc initialize and finalize,
- CUDA-aware vector creation works,
- linking and headers are correct.

### CTest version
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc
'
```

---

## 4. Eigensolver validation

### Direct run
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  ./build/v100-petsc/validate_slepc_eigensolver
'
```

### CTest form
```bash
ssh v100 '
  cd ~/MacroFlow3D &&
  ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver
'
```

Expected focus:
- convergence success,
- small residual norms,
- plausible near-invariant modes,
- no obvious backend misconfiguration.

---

## 5. When to use this stack

Use PETSc/SLEPc when:
- testing the invariant-eigenproblem path,
- validating the transport-operator near-nullspace approach,
- comparing backend behavior,
- running eigensolver-focused benchmarks.

Do not require PETSc/SLEPc for unrelated work in:
- baseline Par2 transport,
- general docs,
- simple runbook edits,
- non-eigensolver numerical cleanup.

---

## 6. Required reporting for PETSc/SLEPc changes

Any change touching this stack should report:

- configure command
- build command
- smoke test result
- validation result
- residual norms
- iteration counts if available
- any changed paths / arch assumptions

If performance is discussed, also report:
- GPU model
- CUDA version
- build type
- relevant runtime options

---

## 7. Common failure modes

### Missing static libs
Symptoms:
- configure fails before generation
- CMake says `libpetsc.a` or `libslepc.a` not found

Check:
- `PETSC_DIR`
- `PETSC_ARCH`
- `SLEPC_DIR`
- whether the external build actually completed

### Wrong CUDA architecture
Symptoms:
- runtime issues or unsupported architecture complaints

For V100, use:
```bash
-DCMAKE_CUDA_ARCHITECTURES=70
```

### MPI / linkage issues
Symptoms:
- link failures or PETSc init errors

Check:
- MPI install
- PETSc build configuration
- remote environment modules or library paths

### Scientific misread
Symptoms:
- backend converges but invariants still behave badly downstream

Remember:
- eigensolver convergence alone is not enough;
- downstream invariant quality and transport behavior still need validation gates.

---

## 8. Interaction with PSPTA work

For this project, PETSc/SLEPc is part of the **invariant construction and validation path**, not the whole answer.

A successful PETSc/SLEPc run does not automatically prove:
- physically correct transport,
- stable invariant independence,
- acceptable macrodispersion behavior.

Use:
- `validate_slepc_eigensolver`
- invariant diagnostics
- PSPTA smoke / local integrity checks
- macrodispersion gates

together.

---

## 9. Anti-patterns

Avoid:
- enabling PETSc/SLEPc in every build by default
- debugging general transport issues through the eigensolver first
- treating residual convergence as sufficient scientific evidence
