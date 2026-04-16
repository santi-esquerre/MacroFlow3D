/**
 * @file TransportOperator3D.cuh
 * @brief Matrix-free transport operator D = v·grad and its algebraic transpose
 * D†.
 *
 * @par Mathematical Background
 * For a scalar field ψ defined at cell centers:
 *
 *   (Dψ)_c = v_c · (∇ψ)_c
 *
 * where v_c is the cell-centered velocity (averaged from CompactMAC faces)
 * and (∇ψ)_c uses finite differences.
 *
 * @par Discretization
 * We use central differences in y and z (periodic) and a symmetric treatment
 * in x (central interior, one-sided at boundaries).
 *
 * The key design requirement is that D† is the ALGEBRAIC transpose of D:
 *
 *   <Dx, y> = <x, D†y>   for all x, y
 *
 * This is achieved by explicitly constructing D†ᵢⱼ = Dⱼᵢ.
 *
 * @par Implementation
 * We decompose D into directional components:
 *
 *   Dψ = vx * (∂ψ/∂x) + vy * (∂ψ/∂y) + vz * (∂ψ/∂z)
 *
 * For central differences, the x-derivative stencil at cell (i,j,k) is:
 *
 *   (∂ψ/∂x)_ijk = (ψ_{i+1,j,k} - ψ_{i-1,j,k}) / (2*dx)
 *
 * The contribution to D[ijk, i+1,j,k] is +vx_ijk / (2*dx)
 * The contribution to D[ijk, i-1,j,k] is -vx_ijk / (2*dx)
 *
 * The algebraic transpose D† accumulates these contributions:
 *
 *   (D†φ)_{i+1,j,k} += vx_ijk * φ_ijk / (2*dx)
 *   (D†φ)_{i-1,j,k} -= vx_ijk * φ_ijk / (2*dx)
 *
 * @par Boundary Treatment in x
 * At i=0 and i=nx-1, we use one-sided differences to maintain symmetry of BCs.
 * For Dirichlet (open-X) flows, this imposes a consistent stencil.
 *
 * @ingroup physics_particles_pspta
 */

#pragma once

#include "../../../../core/DeviceBuffer.cuh"
#include "../../../../core/DeviceSpan.cuh"
#include "../../../../core/Grid3D.hpp"
#include "../../../../core/Scalar.hpp"
#include "../../../../numerics/blas/blas.cuh"
#include "../../../../runtime/CudaContext.cuh"
#include "../../../common/fields.cuh"
#include <cuda_runtime.h>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Configuration
// ============================================================================

/**
 * @brief X-boundary treatment for the transport operator.
 */
enum class TransportXBoundary : uint8_t {
    OneSided = 0, ///< One-sided FD at x=0,nx-1 (default, for open-X flows)
    Periodic = 1, ///< Periodic in x (for doubly-periodic test cases)
    Zero = 2      ///< Zero contribution at boundaries (absorbing)
};

/**
 * @brief Configuration for TransportOperator3D.
 */
struct TransportOperatorConfig {
    TransportXBoundary x_bc = TransportXBoundary::OneSided;
};

// ============================================================================
// TransportOperator3D
// ============================================================================

/**
 * @brief Matrix-free transport operator D = v·∇ and its transpose D†.
 *
 * All operations work with real (double) precision vectors.
 */
class TransportOperator3D {
  public:
    /**
     * @brief Construct operator bound to velocity field and grid.
     *
     * @param vel   CompactMAC velocity field (must outlive the operator)
     * @param grid  Grid metadata
     * @param cfg   Boundary configuration
     */
    TransportOperator3D(const VelocityField* vel, const Grid3D& grid,
                        const TransportOperatorConfig& cfg = {});

    ~TransportOperator3D() = default;

    // Non-copyable, movable
    TransportOperator3D(TransportOperator3D&&) = default;
    TransportOperator3D& operator=(TransportOperator3D&&) = default;
    TransportOperator3D(const TransportOperator3D&) = delete;
    TransportOperator3D& operator=(const TransportOperator3D&) = delete;

    // ── Grid info ──────────────────────────────────────────────────────────

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    size_t size() const { return static_cast<size_t>(nx_) * ny_ * nz_; }
    const Grid3D& grid() const { return grid_; }

    // ── Core operators ─────────────────────────────────────────────────────

    /**
     * @brief Apply D: out = D * in = v · ∇(in)
     *
     * @param in   Input vector (device, size = nx*ny*nz)
     * @param out  Output vector (device, size = nx*ny*nz)
     * @param stream CUDA stream
     */
    void apply_D(DeviceSpan<const real> in, DeviceSpan<real> out, cudaStream_t stream) const;

    /**
     * @brief Apply D†: out = D† * in (algebraic transpose of D)
     *
     * Satisfies: <D*x, y> = <x, D†*y> for all x, y.
     *
     * @param in   Input vector (device, size = nx*ny*nz)
     * @param out  Output vector (device, size = nx*ny*nz)
     * @param stream CUDA stream
     */
    void apply_DT(DeviceSpan<const real> in, DeviceSpan<real> out, cudaStream_t stream) const;

    // ── Composite operators ────────────────────────────────────────────────

    /**
     * @brief Apply D†D: out = D† * D * in
     *
     * This is the symmetric positive semi-definite operator that appears in
     * the energy minimization for invariants.
     *
     * @param in   Input vector
     * @param out  Output vector
     * @param work Work vector (size = nx*ny*nz, for intermediate D*in)
     * @param stream CUDA stream
     */
    void apply_DTD(DeviceSpan<const real> in, DeviceSpan<real> out, DeviceSpan<real> work,
                   cudaStream_t stream) const;

    /**
     * @brief Apply D†WD: out = D† * W * D * in
     *
     * W is a diagonal weight matrix (e.g., W = I or W = 1/|v|²).
     *
     * @param in     Input vector
     * @param out    Output vector
     * @param work   Work vector for D*in
     * @param weight Diagonal weights (size = nx*ny*nz, or nullptr for W=I)
     * @param stream CUDA stream
     */
    void apply_DTWD(DeviceSpan<const real> in, DeviceSpan<real> out, DeviceSpan<real> work,
                    DeviceSpan<const real> weight, cudaStream_t stream) const;

    // ── Validation helpers ─────────────────────────────────────────────────

    /**
     * @brief Verify that D(constant) ≈ 0.
     *
     * @param ctx CUDA context for reductions
     * @return max |D(1)|, should be near machine epsilon
     */
    double test_constant_in_kernel(CudaContext& ctx) const;

    /**
     * @brief Verify algebraic adjoint property: <Dx, y> ≈ <x, D†y>
     *
     * @param ctx CUDA context
     * @param seed Random seed for test vectors
     * @return relative error |<Dx,y> - <x,D†y>| / max(|<Dx,y>|, |<x,D†y>|)
     */
    double test_adjoint(CudaContext& ctx, unsigned seed = 12345) const;

  private:
    // Grid
    int nx_, ny_, nz_;
    double dx_, dy_, dz_;
    Grid3D grid_;

    // Velocity (non-owning)
    const VelocityField* vel_;

    // Configuration
    TransportOperatorConfig cfg_;

    // Cached cell-centered velocity (allocated lazily)
    mutable DeviceBuffer<real> d_vx_cc_;
    mutable DeviceBuffer<real> d_vy_cc_;
    mutable DeviceBuffer<real> d_vz_cc_;
    mutable bool vel_cached_ = false;

    void ensure_velocity_cached(cudaStream_t stream) const;

    // Work buffers for test routines
    mutable DeviceBuffer<real> d_work1_;
    mutable DeviceBuffer<real> d_work2_;
    mutable DeviceBuffer<real> d_work3_;
    mutable DeviceBuffer<real> d_work4_;
};

// ============================================================================
// Laplacian operator L (for regularization)
// ============================================================================

/**
 * @brief Matrix-free Laplacian operator L = -∇².
 *
 * Used as regularizer in A = D†WD + μL.
 *
 * ## Discretization
 *
 * Standard 7-point anisotropic stencil using separate dx, dy, dz:
 *
 *   L[ψ]_c = -(ψ_{i+1} - 2ψ_c + ψ_{i-1})/dx²
 *           -(ψ_{j+1} - 2ψ_c + ψ_{j-1})/dy²
 *           -(ψ_{k+1} - 2ψ_c + ψ_{k-1})/dz²
 *
 * The negative sign ensures L is positive semi-definite.
 *
 * ## Boundary Conditions
 *
 * - **Y, Z**: Periodic (modular index wrap)
 * - **X**: Configurable (Neumann default, Periodic option)
 *
 * Neumann at x boundaries sets ghost = interior value, implying ∂ψ/∂x = 0.
 * This is conservative and keeps L independent of inlet gauge choices.
 *
 * ## Self-Adjointness
 *
 * L is symmetric: <Lx, y> = <x, Ly> for all x, y.
 * This is verified by test_symmetry().
 *
 * ## Expected Behavior
 *
 * - L(constant) = 0  (constant functions are in the null-space)
 * - L(linear in one direction) = 0 for that direction's contribution
 * - L is bounded: ||Lψ|| ≤ C ||ψ|| where C depends on grid spacing
 */
class LaplacianOperator3D {
  public:
    /**
     * @brief X-boundary treatment for Laplacian.
     */
    enum class XBoundary : uint8_t {
        Neumann = 0, ///< Zero-flux at x=0,nx-1
        Periodic = 1 ///< Periodic in x
    };

    LaplacianOperator3D(const Grid3D& grid, XBoundary x_bc = XBoundary::Neumann);

    /**
     * @brief Apply L: out = L * in = -∇²(in)
     *
     * Note: L is defined as -∇² so that it is positive semi-definite.
     */
    void apply_L(DeviceSpan<const real> in, DeviceSpan<real> out, cudaStream_t stream) const;

    /**
     * @brief Test symmetry: <Lx, y> ≈ <x, Ly>
     */
    double test_symmetry(CudaContext& ctx, unsigned seed = 12345) const;

    /**
     * @brief Test that L(constant) ≈ 0.
     *
     * @return RMS of L(1), should be near machine epsilon.
     */
    double test_constant(CudaContext& ctx) const;

    /**
     * @brief Test L on linear field: ψ(x,y,z) = ax + by + cz
     *
     * For a purely linear field, ∂²ψ/∂x² = ∂²ψ/∂y² = ∂²ψ/∂z² = 0,
     * so L(linear) should be 0 everywhere except near boundaries
     * where the Neumann BC introduces small artifacts at edges.
     *
     * @param a Coefficient for x term
     * @param b Coefficient for y term
     * @param c Coefficient for z term
     * @return RMS of L(linear), should be small (O(dx²) near boundaries).
     */
    double test_linear(CudaContext& ctx, double a, double b, double c) const;

    size_t size() const { return static_cast<size_t>(nx_) * ny_ * nz_; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    double dz() const { return dz_; }

  private:
    int nx_, ny_, nz_;
    double dx_, dy_, dz_;
    double inv_dx2_, inv_dy2_, inv_dz2_;
    XBoundary x_bc_;

    mutable DeviceBuffer<real> d_work1_;
    mutable DeviceBuffer<real> d_work2_;
};

// ============================================================================
// Combined operator A = D†WD + μL
// ============================================================================

/**
 * @brief Matrix-free combined operator A = D†WD + μL.
 *
 * This is the symmetric positive definite operator for the invariant
 * eigenproblem. Finding eigenvectors with small eigenvalues gives
 * approximate Lagrangian invariants.
 */
class CombinedOperatorA {
  public:
    /**
     * @brief Construct A from transport operator and Laplacian.
     *
     * @param D    Transport operator (must outlive A)
     * @param L    Laplacian operator (must outlive A)
     * @param mu   Regularization parameter (> 0)
     */
    CombinedOperatorA(const TransportOperator3D* D, const LaplacianOperator3D* L, double mu);

    /**
     * @brief Apply A: out = A * in = (D†WD + μL) * in
     *
     * Uses W = I (identity weight matrix).
     *
     * @param in  Input vector
     * @param out Output vector
     * @param stream CUDA stream
     */
    void apply_A(DeviceSpan<const real> in, DeviceSpan<real> out, cudaStream_t stream);

    /**
     * @brief Apply A with custom weight matrix W.
     *
     * @param in     Input vector
     * @param out    Output vector
     * @param weight Diagonal weights for W (or nullptr for W=I)
     * @param stream CUDA stream
     */
    void apply_A_weighted(DeviceSpan<const real> in, DeviceSpan<real> out,
                          DeviceSpan<const real> weight, cudaStream_t stream);

    /**
     * @brief Test symmetry of A: <Ax, y> ≈ <x, Ay>
     */
    double test_symmetry(CudaContext& ctx, unsigned seed = 12345);

    double mu() const { return mu_; }
    void set_mu(double mu) { mu_ = mu; }

    size_t size() const { return D_->size(); }

    /// Public accessors for sub-operators (needed by SLEPc backend for
    /// grid metadata and preconditioner assembly).
    const TransportOperator3D* transport_operator() const { return D_; }
    const LaplacianOperator3D* laplacian_operator() const { return L_; }

  private:
    const TransportOperator3D* D_;
    const LaplacianOperator3D* L_;
    double mu_;

    // Work buffers
    DeviceBuffer<real> d_work_D_;   // for D*in
    DeviceBuffer<real> d_work_DTD_; // for D†D*in
    DeviceBuffer<real> d_work_L_;   // for L*in

    void ensure_work_buffers();
};

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
