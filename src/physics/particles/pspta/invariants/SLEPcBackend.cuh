/**
 * @file SLEPcBackend.cuh
 * @brief SLEPc-based eigensolvers for Strategy A invariant computation.
 *
 * Two backends are provided:
 *
 * ## SLEPcBackend (Validation)
 *
 * Assembles the full operator as MATAIJCUSPARSE via MatComputeOperator,
 * then uses Krylov-Schur + STSINVERT + direct LU.  Exact and robust
 * but O(n²) in memory — suitable for validation grids (≤ 32³).
 *
 *   Factory name: "slepc_validation"
 *
 * ## SLEPcProductionBackend (Production)
 *
 * Uses the authoritative CombinedOperatorA CUDA kernels for operator action,
 * with explicit assembly choices controlled inside the backend:
 *   - control/small grids may use exact explicit assembly via MatComputeOperator
 *   - larger grids may use a probed sparse surrogate when exact assembly is
 *     too costly
 *
 * The solved operator must remain faithful to A = D†WD + μL; production-scale
 * approximations are therefore subject to separate fidelity checks.
 *
 *   Factory name: "slepc"  (default)
 *
 * Both solve the standard eigenproblem:
 *
 *   A ψ = λ ψ       where A = D†WD + μL
 *
 * with constant-mode deflation via EPSSetDeflationSpace.
 *
 * @ingroup physics_particles_pspta
 */

#pragma once

#ifdef MACROFLOW3D_HAS_PETSC

#include "EigensolverBackend.cuh"
#include <slepceps.h>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Shell context passed through PETSc MatShell
// ============================================================================

struct ShellContext {
    CombinedOperatorA* A = nullptr;
    cudaStream_t stream = nullptr;
    size_t n = 0; ///< problem size (nx*ny*nz)
};

// ============================================================================
// SLEPcBackend — Validation (explicit matrix assembly)
// ============================================================================

/**
 * @brief VALIDATION eigensolver: assembles full A as MATAIJCUSPARSE,
 *        uses Krylov-Schur + SINVERT + LU.
 *
 * Robust and exact but O(n²) memory.  Use for:
 *   - Correctness verification on small grids
 *   - Reference eigenvalue comparison
 *   - Debugging operator discretization issues
 */
class SLEPcBackend : public EigensolverBackend {
  public:
    SLEPcBackend();
    ~SLEPcBackend() override;

    SLEPcBackend(const SLEPcBackend&) = delete;
    SLEPcBackend& operator=(const SLEPcBackend&) = delete;

    EigensolverResult solve(CombinedOperatorA& A, const EigensolverConfig& config, CudaContext& ctx,
                            std::vector<DeviceBuffer<real>>& eigenvectors) override;

    std::string name() const override { return "slepc_validation"; }

    // Shared helpers (used by production backend too)
    static Mat create_shell_operator(CombinedOperatorA& A, CudaContext& ctx, ShellContext& sctx);
    static Mat assemble_laplacian_preconditioner(const LaplacianOperator3D* L, double mu,
                                                 PetscInt n, int nx, int ny, int nz, double dx,
                                                 double dy, double dz);
    static Mat assemble_explicit_operator(CombinedOperatorA& A, CudaContext& ctx,
                                          const char* mat_type = MATAIJCUSPARSE);
    static Mat assemble_probed_operator(CombinedOperatorA& A, CudaContext& ctx,
                                        double diag_shift = 0.0,
                                        const char* mat_type = MATAIJCUSPARSE);
    static void print_gpu_evidence(EPS eps, Mat A_mat, Mat A_pre, Vec sample);
};

// ============================================================================
// SLEPcProductionBackend — Production (matrix-free MATSHELL)
// ============================================================================

/**
 * @brief PRODUCTION eigensolver: matrix-free MATSHELL + LOBPCG.
 *
 * The operator A = D†WD + μL is applied via CUDA kernels (never assembled).
 * LOBPCG iterates with STPRECOND and an assembled μL preconditioner
 * (MATAIJCUSPARSE + PCILU) for acceleration.
 *
 * O(n) memory, scalable to production grids (64³+).
 */
class SLEPcProductionBackend : public EigensolverBackend {
  public:
    SLEPcProductionBackend();
    ~SLEPcProductionBackend() override;

    SLEPcProductionBackend(const SLEPcProductionBackend&) = delete;
    SLEPcProductionBackend& operator=(const SLEPcProductionBackend&) = delete;

    EigensolverResult solve(CombinedOperatorA& A, const EigensolverConfig& config, CudaContext& ctx,
                            std::vector<DeviceBuffer<real>>& eigenvectors) override;

    std::string name() const override { return "slepc"; }
};

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d

#endif // MACROFLOW3D_HAS_PETSC
