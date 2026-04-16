/**
 * @file PetscSlepcInit.hpp
 * @brief Singleton guard for PETSc/SLEPc initialization and finalization.
 *
 * Call PetscSlepcInit::ensure() before any PETSc/SLEPc call.
 * Finalization happens automatically via atexit().
 */

#pragma once

#ifdef MACROFLOW3D_HAS_PETSC

namespace macroflow3d {
namespace runtime {

class PetscSlepcInit {
  public:
    /// Initialize PETSc+SLEPc if not already done. Thread-safe (first call wins).
    static void ensure();

    /// True after ensure() has been called successfully.
    static bool initialized();

  private:
    static bool initialized_;
};

} // namespace runtime
} // namespace macroflow3d

#endif // MACROFLOW3D_HAS_PETSC
