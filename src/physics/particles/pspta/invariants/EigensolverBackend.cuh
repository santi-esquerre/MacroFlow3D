/**
 * @file EigensolverBackend.cuh
 * @brief Interface for eigensolvers used in Strategy A invariant computation.
 *
 * This is a SKELETON interface for future implementation. The actual solver
 * (LOBPCG, interface to SLEPc, etc.) will be implemented in a later phase.
 *
 * @par Strategy A Overview
 * Find the smallest eigenvalues/eigenvectors of:
 *
 *   A * psi = lambda * M * psi
 *
 * where A = D†WD + μL is the combined operator and M is a mass matrix
 * (typically identity for cell-centered discretization with uniform mesh).
 *
 * The eigenvectors with smallest eigenvalues represent approximate
 * Lagrangian invariants (functions nearly constant along streamlines).
 *
 * @ingroup physics_particles_pspta
 */

#pragma once

#include "../../../../core/DeviceBuffer.cuh"
#include "../../../../core/DeviceSpan.cuh"
#include "../../../../core/Scalar.hpp"
#include "TransportOperator3D.cuh"
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Eigensolver configuration
// ============================================================================

/**
 * @brief Configuration for the invariant eigensolver.
 */
struct EigensolverConfig {
    int n_eigenvectors = 6;         ///< Number of smallest eigenpairs to compute (K)
    double tolerance = 1e-6;        ///< Convergence tolerance for eigenvalues
    int max_iterations = 200;       ///< Maximum outer iterations
    int block_size = 0;             ///< Block size for LOBPCG (0 = auto)
    bool verbose = false;           ///< Print convergence info
    std::string backend = "lobpcg"; ///< "lobpcg", "slepc", "slepc_validation"
};

/**
 * @brief Result from solving the eigenproblem.
 */
struct EigensolverResult {
    int n_converged = 0;                ///< Number of converged eigenpairs
    int iterations = 0;                 ///< Iterations performed
    std::vector<double> eigenvalues;    ///< Converged eigenvalues (ascending)
    std::vector<double> residual_norms; ///< Residual ||A*v - lambda*M*v||
    double elapsed_ms = 0.0;            ///< Solver time in milliseconds
    bool success = false;               ///< True if requested eigenpairs found
    std::string message;                ///< Status message
};

// ============================================================================
// EigensolverBackend interface
// ============================================================================

/**
 * @brief Abstract interface for eigensolvers.
 *
 * Implementations:
 * - LOBPCGBackend: Custom LOBPCG implementation (primary)
 * - SLEPcBackend: Interface to SLEPc (optional, requires PETSc)
 *
 * @note This is a SKELETON for future implementation.
 */
class EigensolverBackend {
  public:
    virtual ~EigensolverBackend() = default;

    /**
     * @brief Solve the generalized eigenproblem A*v = lambda*M*v.
     *
     * @param A        Combined operator A = D†WD + μL
     * @param config   Solver configuration
     * @param ctx      CUDA context
     * @param eigenvectors [out] Device buffers for eigenvectors (n_eigenvectors
     * arrays)
     * @return EigensolverResult with eigenvalues and convergence info
     *
     * @note eigenvectors should be pre-allocated with size n_eigenvectors,
     *       each with capacity >= A.size()
     */
    virtual EigensolverResult solve(CombinedOperatorA& A, const EigensolverConfig& config,
                                    CudaContext& ctx,
                                    std::vector<DeviceBuffer<real>>& eigenvectors) = 0;

    /**
     * @brief Get the name of this backend.
     */
    virtual std::string name() const = 0;
};

// ============================================================================
// Factory function
// ============================================================================

/**
 * @brief Create an eigensolver backend.
 *
 * @param backend_name Backend identifier:
 *   - "slepc"            : Matrix-free LOBPCG production backend
 * (MACROFLOW3D_HAS_PETSC)
 *   - "slepc_validation" : Explicit assembly + SINVERT validation backend
 *   - "lobpcg"           : Custom LOBPCG placeholder (not yet implemented)
 * @return Unique pointer to backend, or nullptr if not available.
 */
std::unique_ptr<EigensolverBackend> create_eigensolver_backend(const std::string& backend_name);

// ============================================================================
// Placeholder LOBPCG backend (skeleton)
// ============================================================================

/**
 * @brief LOBPCG eigensolver backend (SKELETON).
 *
 * TODO: Implement in Phase 4:
 * - Matrix-free LOBPCG with soft-locking
 * - MG preconditioning for A
 * - Convergence monitoring
 * - Block operations for multiple eigenvectors
 */
class LOBPCGBackend : public EigensolverBackend {
  public:
    LOBPCGBackend() = default;

    EigensolverResult solve(CombinedOperatorA& A, const EigensolverConfig& config, CudaContext& ctx,
                            std::vector<DeviceBuffer<real>>& eigenvectors) override;

    std::string name() const override { return "lobpcg"; }
};

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
