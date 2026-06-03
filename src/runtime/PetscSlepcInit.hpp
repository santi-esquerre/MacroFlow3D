/**
 * @file PetscSlepcInit.hpp
 * @brief Singleton guard for PETSc/SLEPc initialization and finalization.
 *
 * Call PetscSlepcInit::ensure() before any PETSc/SLEPc call.
 * Finalization defaults to atexit(), but long-running PETSc/CUDA apps can
 * call finalize() explicitly to tear down CUPM resources before process-exit
 * handlers run.
 */

#pragma once

#ifdef MACROFLOW3D_HAS_PETSC

namespace macroflow3d {
namespace runtime {

class PetscSlepcInit {
  public:
    /// Initialize PETSc+SLEPc if not already done. Thread-safe (first call wins).
    static void ensure();

    /// Finalize PETSc+SLEPc once. Safe to call multiple times.
    static void finalize();

    /// True after ensure() has been called successfully.
    static bool initialized();

    /// True after finalize() has completed.
    static bool finalized();

  private:
    static bool initialized_;
    static bool finalized_;
    static bool atexit_registered_;
};

} // namespace runtime
} // namespace macroflow3d

#endif // MACROFLOW3D_HAS_PETSC
