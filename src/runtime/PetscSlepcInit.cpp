/**
 * @file PetscSlepcInit.cpp
 * @brief Singleton PETSc/SLEPc initialization.
 */

#ifdef MACROFLOW3D_HAS_PETSC

#include "PetscSlepcInit.hpp"
#include <cstdio>
#include <cstdlib>
#include <slepceps.h>

namespace macroflow3d {
namespace runtime {

bool PetscSlepcInit::initialized_ = false;
bool PetscSlepcInit::finalized_ = false;
bool PetscSlepcInit::atexit_registered_ = false;

void PetscSlepcInit::ensure() {
    if (initialized_)
        return;
    if (finalized_) {
        std::fprintf(
            stderr,
            "[PetscSlepcInit] ensure() called after explicit finalization in the same process\n");
        std::abort();
    }

    // Tell PETSc not to require GPU-aware MPI.  Our OpenMPI is not built
    // with CUDA support, which is fine for single-node / single-GPU usage.
    PetscOptionsSetValue(nullptr, "-use_gpu_aware_mpi", "0");

    // SlepcInitialize calls PetscInitialize internally if needed.
    static const char help[] = "MacroFlow3D PSPTA eigensolver\n";
    PetscErrorCode ierr = SlepcInitialize(nullptr, nullptr, nullptr, help);
    if (ierr) {
        std::fprintf(stderr, "[PetscSlepcInit] SlepcInitialize failed (ierr=%d)\n",
                     static_cast<int>(ierr));
        std::abort();
    }

    if (!atexit_registered_) {
        // Register guarded finalization at exit (reverse order: SLEPc finalize calls PETSc).
        std::atexit([]() { PetscSlepcInit::finalize(); });
        atexit_registered_ = true;
    }

    initialized_ = true;

    // Print confirmation
    PetscInt petsc_major, petsc_minor, petsc_sub;
    PetscGetVersionNumber(&petsc_major, &petsc_minor, &petsc_sub, nullptr);
    std::printf("[PetscSlepcInit] PETSc %d.%d.%d + SLEPc initialized\n",
                static_cast<int>(petsc_major), static_cast<int>(petsc_minor),
                static_cast<int>(petsc_sub));
}

bool PetscSlepcInit::initialized() {
    return initialized_;
}

void PetscSlepcInit::finalize() {
    if (!initialized_ || finalized_)
        return;

    PetscErrorCode ierr = SlepcFinalize();
    if (ierr) {
        std::fprintf(stderr, "[PetscSlepcInit] SlepcFinalize failed (ierr=%d)\n",
                     static_cast<int>(ierr));
        std::abort();
    }

    finalized_ = true;
    initialized_ = false;
}

bool PetscSlepcInit::finalized() {
    return finalized_;
}

} // namespace runtime
} // namespace macroflow3d

#endif // MACROFLOW3D_HAS_PETSC
