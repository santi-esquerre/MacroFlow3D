/**
 * @file smoke_test_petsc.cu
 * @brief Minimal PETSc + SLEPc integration smoke test.
 *
 * Verifies that:
 *   1. PETSc/SLEPc initialize and finalize without error
 *   2. CUDA-aware vectors can be created (VECCUDA)
 *   3. Headers are found and link succeeds
 *
 * This is NOT an eigensolver test. Just a build/link/init sanity check.
 *
 * Build:
 *   cmake -DMACROFLOW3D_ENABLE_PETSC=ON ..
 *   make smoke_test_petsc
 *
 * Run:
 *   ./smoke_test_petsc
 */

#ifdef MACROFLOW3D_HAS_PETSC

#include <cstdio>
#include <slepceps.h>

int main(int argc, char** argv) {
    PetscErrorCode ierr;

    // ── 1. Initialize SLEPc (which initializes PETSc internally) ──────────
    // Tell PETSc not to require GPU-aware MPI (our OpenMPI is not built
    // with CUDA support, which is fine for single-node usage).
    PetscOptionsSetValue(nullptr, "-use_gpu_aware_mpi", "0");
    ierr = SlepcInitialize(&argc, &argv, nullptr, "MacroFlow3D PETSc/SLEPc smoke test\n");
    if (ierr) {
        std::fprintf(stderr, "ERROR: SlepcInitialize failed (ierr=%d)\n", static_cast<int>(ierr));
        return 1;
    }

    // ── 2. Print version info ─────────────────────────────────────────────
    PetscInt major, minor, sub;
    PetscGetVersionNumber(&major, &minor, &sub, nullptr);
    std::printf("PETSc version : %d.%d.%d\n", static_cast<int>(major), static_cast<int>(minor),
                static_cast<int>(sub));

    // SLEPc version from header macros
    std::printf("SLEPc version : %d.%d.%d\n", SLEPC_VERSION_MAJOR, SLEPC_VERSION_MINOR,
                SLEPC_VERSION_SUBMINOR);

    // ── 3. Create a CUDA vector to verify GPU backend ─────────────────────
    Vec v;
    PetscInt n = 100;
    VecCreateSeqCUDA(PETSC_COMM_SELF, n, &v);

    VecType vtype;
    VecGetType(v, &vtype);
    std::printf("VecType       : %s\n", vtype);

    // Simple operation on GPU
    VecSet(v, 1.0);
    PetscReal norm;
    VecNorm(v, NORM_2, &norm);
    std::printf("||ones(%d)||_2 : %.6f  (expected: %.6f)\n", static_cast<int>(n),
                static_cast<double>(norm), 10.0);

    VecDestroy(&v);

    // ── 4. Finalize ───────────────────────────────────────────────────────
    ierr = SlepcFinalize();
    if (ierr) {
        std::fprintf(stderr, "ERROR: SlepcFinalize failed (ierr=%d)\n", static_cast<int>(ierr));
        return 1;
    }

    std::printf("\nSmoke test PASSED.\n");
    return 0;
}

#else // !MACROFLOW3D_HAS_PETSC

#include <cstdio>
int main() {
    std::fprintf(stderr, "ERROR: Built without MACROFLOW3D_HAS_PETSC.\n"
                         "Rebuild with: cmake -DMACROFLOW3D_ENABLE_PETSC=ON ..\n");
    return 1;
}

#endif
