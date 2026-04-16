/**
 * @file TransportOperator3D.cu
 * @brief Implementation of matrix-free transport operators D, D†, L, and A.
 *
 * @par Discretization Details
 *
 * The transport operator D = v·∇ is discretized as follows:
 *
 * For a cell (i,j,k) with index c = i + nx*(j + ny*k):
 *
 *   (Dψ)_c = vx_c * (∂ψ/∂x)_c + vy_c * (∂ψ/∂y)_c + vz_c * (∂ψ/∂z)_c
 *
 * where:
 *   - vx_c, vy_c, vz_c are cell-centered velocities (average of adjacent faces)
 *   - Derivatives use central differences (periodic in y,z; one-sided at x
 * bounds)
 *
 * @par Adjoint Construction
 *
 * The algebraic transpose D† is constructed such that:
 *
 *   <Dψ, φ> = Σ_c (Dψ)_c * φ_c
 *           = Σ_c [vx_c * (ψ_{c+1} - ψ_{c-1})/(2dx)] * φ_c + ... (y,z terms)
 *           = Σ_c ψ_c * [vx_{c-1} * φ_{c-1} - vx_{c+1} * φ_{c+1}]/(2dx) + ...
 *           = <ψ, D†φ>
 *
 * Therefore:
 *   (D†φ)_c = -[vx_{c+1} * φ_{c+1} - vx_{c-1} * φ_{c-1}]/(2dx) - ... (y,z
 * terms)
 *
 * Note the sign: D† ≠ -D in general (unless v is constant).
 */

#include "../../../../runtime/cuda_check.cuh"
#include "TransportOperator3D.cuh"
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Device helpers
// ============================================================================

/// Cell-centered index: c = i + nx*(j + ny*k)
__device__ __forceinline__ int idx_cc(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + ny * k);
}

/// Periodic modulo
__device__ __forceinline__ int pmod(int n, int N) {
    return ((n % N) + N) % N;
}

// ============================================================================
// Kernel: Cache cell-centered velocity from CompactMAC faces
// ============================================================================

__global__ void kernel_cache_velocity_cc(const real* __restrict__ U, // (nx+1)*ny*nz
                                         const real* __restrict__ V, // nx*(ny+1)*nz
                                         const real* __restrict__ W, // nx*ny*(nz+1)
                                         real* __restrict__ vx_cc, real* __restrict__ vy_cc,
                                         real* __restrict__ vz_cc, int nx, int ny, int nz) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (c >= total)
        return;

    const int i = c % nx;
    const int j = (c / nx) % ny;
    const int k = c / (nx * ny);

    // U-face indices: U(i,j,k) at i + (nx+1)*j + (nx+1)*ny*k
    const int idx_U_ijk = i + (nx + 1) * j + (nx + 1) * ny * k;
    const int idx_U_ip1 = (i + 1) + (nx + 1) * j + (nx + 1) * ny * k;

    // V-face indices: V(i,j,k) at i + nx*j + nx*(ny+1)*k
    const int idx_V_ijk = i + nx * j + nx * (ny + 1) * k;
    const int idx_V_jp1 = i + nx * (j + 1) + nx * (ny + 1) * k;

    // W-face indices: W(i,j,k) at i + nx*j + nx*ny*k
    const int idx_W_ijk = i + nx * j + nx * ny * k;
    const int idx_W_kp1 = i + nx * j + nx * ny * (k + 1);

    vx_cc[c] = 0.5 * (U[idx_U_ijk] + U[idx_U_ip1]);
    vy_cc[c] = 0.5 * (V[idx_V_ijk] + V[idx_V_jp1]);
    vz_cc[c] = 0.5 * (W[idx_W_ijk] + W[idx_W_kp1]);
}

// ============================================================================
// Kernel: Apply D = v·∇
// ============================================================================

/**
 * @brief Apply transport operator D.
 *
 * For x-boundary treatment OneSided:
 *   i=0:    (∂ψ/∂x)_0 = (ψ_1 - ψ_0) / dx           (forward difference)
 *   i=nx-1: (∂ψ/∂x)_{nx-1} = (ψ_{nx-1} - ψ_{nx-2}) / dx  (backward difference)
 *   else:   (∂ψ/∂x)_i = (ψ_{i+1} - ψ_{i-1}) / (2*dx)     (central)
 *
 * y,z: always periodic central
 */
__global__ void kernel_apply_D(const real* __restrict__ psi, real* __restrict__ out,
                               const real* __restrict__ vx_cc, const real* __restrict__ vy_cc,
                               const real* __restrict__ vz_cc, int nx, int ny, int nz,
                               double inv_2dx, double inv_2dy, double inv_2dz,
                               double inv_dx, // for one-sided
                               int x_bc)      // 0=OneSided, 1=Periodic, 2=Zero
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (c >= total)
        return;

    const int i = c % nx;
    const int j = (c / nx) % ny;
    const int k = c / (nx * ny);

    const double vx = vx_cc[c];
    const double vy = vy_cc[c];
    const double vz = vz_cc[c];

    // ── X derivative ───────────────────────────────────────────────────────
    double dpsi_dx = 0.0;
    if (x_bc == 1) { // Periodic
        const int im = pmod(i - 1, nx);
        const int ip = pmod(i + 1, nx);
        dpsi_dx = (psi[idx_cc(ip, j, k, nx, ny)] - psi[idx_cc(im, j, k, nx, ny)]) * inv_2dx;
    } else if (x_bc == 2) { // Zero at boundary
        if (i > 0 && i < nx - 1) {
            dpsi_dx =
                (psi[idx_cc(i + 1, j, k, nx, ny)] - psi[idx_cc(i - 1, j, k, nx, ny)]) * inv_2dx;
        }
        // else dpsi_dx = 0
    } else { // OneSided (default)
        if (i == 0) {
            dpsi_dx = (psi[idx_cc(1, j, k, nx, ny)] - psi[idx_cc(0, j, k, nx, ny)]) * inv_dx;
        } else if (i == nx - 1) {
            dpsi_dx =
                (psi[idx_cc(nx - 1, j, k, nx, ny)] - psi[idx_cc(nx - 2, j, k, nx, ny)]) * inv_dx;
        } else {
            dpsi_dx =
                (psi[idx_cc(i + 1, j, k, nx, ny)] - psi[idx_cc(i - 1, j, k, nx, ny)]) * inv_2dx;
        }
    }

    // ── Y derivative (always periodic central) ─────────────────────────────
    const int jm = pmod(j - 1, ny);
    const int jp = pmod(j + 1, ny);
    const double dpsi_dy =
        (psi[idx_cc(i, jp, k, nx, ny)] - psi[idx_cc(i, jm, k, nx, ny)]) * inv_2dy;

    // ── Z derivative (always periodic central) ─────────────────────────────
    const int km = pmod(k - 1, nz);
    const int kp = pmod(k + 1, nz);
    const double dpsi_dz =
        (psi[idx_cc(i, j, kp, nx, ny)] - psi[idx_cc(i, j, km, nx, ny)]) * inv_2dz;

    out[c] = vx * dpsi_dx + vy * dpsi_dy + vz * dpsi_dz;
}

// ============================================================================
// Kernel: Apply D† (algebraic transpose)
// ============================================================================

/**
 * @brief Apply algebraic transpose D†.
 *
 * The matrix element D[c, c'] means: contribution of ψ_{c'} to (Dψ)_c.
 * The transpose D†[c', c] = D[c, c'].
 *
 * For central differences in interior:
 *   D[c, c+1] = vx_c / (2*dx)   (from x-derivative)
 *   D[c, c-1] = -vx_c / (2*dx)
 *
 * So D†[c+1, c] = vx_c / (2*dx), which is the contribution to (D†φ)_{c+1}
 *    from φ_c.
 *
 * Equivalently, (D†φ)_c receives contributions from neighbors' D coefficients:
 *   (D†φ)_c = Σ_{c'} D[c', c] * φ_{c'}
 *           = D[c-1, c] * φ_{c-1} + D[c+1, c] * φ_{c+1} + ... (y,z)
 *           = vx_{c-1} / (2*dx) * φ_{c-1} + (-vx_{c+1}) / (2*dx) * φ_{c+1} +
 * ... = (vx_{c-1} * φ_{c-1} - vx_{c+1} * φ_{c+1}) / (2*dx) + ...
 */
__global__ void kernel_apply_DT(const real* __restrict__ phi, real* __restrict__ out,
                                const real* __restrict__ vx_cc, const real* __restrict__ vy_cc,
                                const real* __restrict__ vz_cc, int nx, int ny, int nz,
                                double inv_2dx, double inv_2dy, double inv_2dz, double inv_dx,
                                int x_bc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (c >= total)
        return;

    const int i = c % nx;
    const int j = (c / nx) % ny;
    const int k = c / (nx * ny);

    double result = 0.0;

    // ── X contribution ─────────────────────────────────────────────────────
    // D†[c-1, c] = -vx_c / (2dx) for central difference at c
    // D†[c+1, c] = +vx_c / (2dx)
    // So (D†φ)_c = sum over c' of D[c', c] * φ_{c'}
    //            = D[c-1, c] * φ_{c-1} + D[c+1, c] * φ_{c+1}

    if (x_bc == 1) { // Periodic
        const int im = pmod(i - 1, nx);
        const int ip = pmod(i + 1, nx);
        // Contribution from cell (im) where D[im, c] = vx_im * coef_to_c
        // For central at im: ψ_c is ψ_{im+1}, so D[im, c] = vx_im / (2dx)
        // For central at ip: ψ_c is ψ_{ip-1}, so D[ip, c] = -vx_ip / (2dx)
        result += (vx_cc[idx_cc(im, j, k, nx, ny)] * phi[idx_cc(im, j, k, nx, ny)] -
                   vx_cc[idx_cc(ip, j, k, nx, ny)] * phi[idx_cc(ip, j, k, nx, ny)]) *
                  inv_2dx;
    } else if (x_bc == 2) { // Zero at boundary
        if (i > 0 && i < nx - 1) {
            const int im = i - 1;
            const int ip = i + 1;
            result += (vx_cc[idx_cc(im, j, k, nx, ny)] * phi[idx_cc(im, j, k, nx, ny)] -
                       vx_cc[idx_cc(ip, j, k, nx, ny)] * phi[idx_cc(ip, j, k, nx, ny)]) *
                      inv_2dx;
        }
    } else { // OneSided
        // More complex: need to match the forward structure of D
        // At i=0: D uses forward diff, so D[0, 1] = vx_0 / dx, D[0, 0] = -vx_0 / dx
        // At i=nx-1: D uses backward diff, so D[nx-1, nx-1] = vx_{nx-1}/dx, D[nx-1,
        // nx-2] = -vx_{nx-1}/dx

        if (i == 0) {
            // D[0,0] = -vx_0/dx (from forward at 0)
            // D[1,0] = -vx_1/(2dx) (from central at 1, ψ_0 is ψ_{1-1})
            // So D†[0] = D[0,0]*φ_0 + D[1,0]*φ_1
            //          = -vx_0/dx * φ_0 + (-vx_1/(2dx)) * φ_1
            result += -vx_cc[c] * phi[c] * inv_dx;
            if (nx > 1) {
                result += -vx_cc[idx_cc(1, j, k, nx, ny)] * phi[idx_cc(1, j, k, nx, ny)] * inv_2dx;
            }
        } else if (i == nx - 1) {
            // D[nx-1, nx-1] = +vx_{nx-1}/dx (from backward at nx-1)
            // D[nx-2, nx-1] = +vx_{nx-2}/(2dx) (from central at nx-2)
            result += vx_cc[c] * phi[c] * inv_dx;
            if (nx > 1) {
                result += vx_cc[idx_cc(nx - 2, j, k, nx, ny)] * phi[idx_cc(nx - 2, j, k, nx, ny)] *
                          inv_2dx;
            }
        } else if (i == 1) {
            // From D at i=0 (forward): D[0, 1] = vx_0 / dx
            // From D at i=1 (central): D[1, 2] = vx_1/(2dx), D[1, 0] = -vx_1/(2dx)
            // From D at i=2 (central): D[2, 1] = -vx_2/(2dx)
            result += vx_cc[idx_cc(0, j, k, nx, ny)] * phi[idx_cc(0, j, k, nx, ny)] *
                      inv_dx; // from D[0,1]
            result += -vx_cc[idx_cc(2, j, k, nx, ny)] * phi[idx_cc(2, j, k, nx, ny)] *
                      inv_2dx; // from central at 2
        } else if (i == nx - 2) {
            // From D at i=nx-2 (central): touches nx-3 and nx-1
            // From D at i=nx-1 (backward): D[nx-1, nx-2] = -vx_{nx-1}/dx
            result +=
                vx_cc[idx_cc(nx - 3, j, k, nx, ny)] * phi[idx_cc(nx - 3, j, k, nx, ny)] * inv_2dx;
            result +=
                -vx_cc[idx_cc(nx - 1, j, k, nx, ny)] * phi[idx_cc(nx - 1, j, k, nx, ny)] * inv_dx;
        } else {
            // Interior: standard central from both neighbors
            const int im = i - 1;
            const int ip = i + 1;
            result += (vx_cc[idx_cc(im, j, k, nx, ny)] * phi[idx_cc(im, j, k, nx, ny)] -
                       vx_cc[idx_cc(ip, j, k, nx, ny)] * phi[idx_cc(ip, j, k, nx, ny)]) *
                      inv_2dx;
        }
    }

    // ── Y contribution (always periodic central) ───────────────────────────
    const int jm = pmod(j - 1, ny);
    const int jp = pmod(j + 1, ny);
    result += (vy_cc[idx_cc(i, jm, k, nx, ny)] * phi[idx_cc(i, jm, k, nx, ny)] -
               vy_cc[idx_cc(i, jp, k, nx, ny)] * phi[idx_cc(i, jp, k, nx, ny)]) *
              inv_2dy;

    // ── Z contribution (always periodic central) ───────────────────────────
    const int km = pmod(k - 1, nz);
    const int kp = pmod(k + 1, nz);
    result += (vz_cc[idx_cc(i, j, km, nx, ny)] * phi[idx_cc(i, j, km, nx, ny)] -
               vz_cc[idx_cc(i, j, kp, nx, ny)] * phi[idx_cc(i, j, kp, nx, ny)]) *
              inv_2dz;

    out[c] = result;
}

// ============================================================================
// Kernel: Apply weighted diagonal W
// ============================================================================

__global__ void kernel_apply_diagonal_weight(const real* __restrict__ in, real* __restrict__ out,
                                             const real* __restrict__ weight, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n)
        return;
    out[c] = weight[c] * in[c];
}

// ============================================================================
// Kernel: Apply Laplacian L = -∇²
// ============================================================================

/**
 * @brief Anisotropic 7-point Laplacian stencil: L = -∇²
 *
 * ## Discretization (ANISOTROPIC GRID)
 *
 * For cell (i,j,k) with spacing (dx, dy, dz), the second derivatives are:
 *
 *   ∂²ψ/∂x² = (ψ_{i+1,j,k} - 2·ψ_{i,j,k} + ψ_{i-1,j,k}) / dx²
 *   ∂²ψ/∂y² = (ψ_{i,j+1,k} - 2·ψ_{i,j,k} + ψ_{i,j-1,k}) / dy²
 *   ∂²ψ/∂z² = (ψ_{i,j,k+1} - 2·ψ_{i,j,k} + ψ_{i,j,k-1}) / dz²
 *
 *   L[ψ] = -(∂²ψ/∂x² + ∂²ψ/∂y² + ∂²ψ/∂z²)
 *
 * The NEGATIVE sign ensures L is positive semi-definite (eigenvalues ≥ 0).
 *
 * ## Boundary Conditions
 *
 * **Y and Z (Periodic):**
 *   j±1 and k±1 use modular arithmetic (periodic wrap).
 *   This matches the physical periodicity of the flow domain in y,z.
 *
 * **X (Neumann Homogeneous):**
 *   At i=0:   ghost value ψ_{-1,j,k} = ψ_{0,j,k}   ⟹  ∂ψ/∂x = 0
 *   At i=nx-1: ghost value ψ_{nx,j,k} = ψ_{nx-1,j,k}  ⟹  ∂ψ/∂x = 0
 *
 * ## Justification for Neumann in X
 *
 * In the combined operator A = D†WD + μL the Laplacian L acts as a
 * regularizer ensuring A is SPD even when D†WD has a non-trivial nullspace.
 *
 * The choice of Neumann (zero-flux) at x boundaries is CONSERVATIVE:
 *   - It does NOT impose a specific value on ψ at the boundary
 *   - It allows ψ to take any value consistent with minimal gradient energy
 *   - It preserves symmetry of L (L is self-adjoint under this BC)
 *   - It matches the physical intuition: no "smoothness penalty" flux escapes
 *
 * Alternative choices:
 *   - Periodic in x: Only valid for doubly/triply-periodic test cases
 *   - Dirichlet: Would fix ψ at boundaries, conflicting with inlet gauge
 *
 * The inlet gauge (ψ1=y, ψ2=z at x=0) is applied AFTER the eigensolver,
 * not through the boundary conditions of L. This separation keeps L
 * independent of the specific gauge choice.
 *
 * ## Memory: No explicit stencil storage
 *
 * The operator is matrix-free. Stencil coefficients are computed on-the-fly:
 *   coeff_center = 2/dx² + 2/dy² + 2/dz²
 *   coeff_x = 1/dx²
 *   coeff_y = 1/dy²
 *   coeff_z = 1/dz²
 */
__global__ void kernel_apply_L(const real* __restrict__ psi, real* __restrict__ out, int nx, int ny,
                               int nz, double inv_dx2, double inv_dy2, double inv_dz2,
                               int x_bc) // 0=Neumann, 1=Periodic
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (c >= total)
        return;

    const int i = c % nx;
    const int j = (c / nx) % ny;
    const int k = c / (nx * ny);

    const double psi_c = psi[c];

    // ── X Laplacian ────────────────────────────────────────────────────────
    double Lx;
    if (x_bc == 1) { // Periodic
        const int im = pmod(i - 1, nx);
        const int ip = pmod(i + 1, nx);
        Lx =
            (psi[idx_cc(ip, j, k, nx, ny)] - 2.0 * psi_c + psi[idx_cc(im, j, k, nx, ny)]) * inv_dx2;
    } else { // Neumann
        double psi_im, psi_ip;
        if (i == 0) {
            psi_im = psi_c; // ghost = interior (Neumann)
            psi_ip = psi[idx_cc(1, j, k, nx, ny)];
        } else if (i == nx - 1) {
            psi_im = psi[idx_cc(nx - 2, j, k, nx, ny)];
            psi_ip = psi_c; // ghost = interior (Neumann)
        } else {
            psi_im = psi[idx_cc(i - 1, j, k, nx, ny)];
            psi_ip = psi[idx_cc(i + 1, j, k, nx, ny)];
        }
        Lx = (psi_ip - 2.0 * psi_c + psi_im) * inv_dx2;
    }

    // ── Y Laplacian (periodic) ─────────────────────────────────────────────
    const int jm = pmod(j - 1, ny);
    const int jp = pmod(j + 1, ny);
    const double Ly =
        (psi[idx_cc(i, jp, k, nx, ny)] - 2.0 * psi_c + psi[idx_cc(i, jm, k, nx, ny)]) * inv_dy2;

    // ── Z Laplacian (periodic) ─────────────────────────────────────────────
    const int km = pmod(k - 1, nz);
    const int kp = pmod(k + 1, nz);
    const double Lz =
        (psi[idx_cc(i, j, kp, nx, ny)] - 2.0 * psi_c + psi[idx_cc(i, j, km, nx, ny)]) * inv_dz2;

    // L = -∇² (note the negative sign for positive semi-definiteness)
    out[c] = -(Lx + Ly + Lz);
}

// ============================================================================
// Kernel: Fill with random values (for testing)
// ============================================================================

__global__ void kernel_fill_random(real* data, int n, unsigned seed) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n)
        return;

    // Simple LCG for reproducibility
    unsigned state = seed + c * 1103515245u;
    state = state * 1103515245u + 12345u;
    double u = static_cast<double>(state) / 4294967296.0;
    data[c] = 2.0 * u - 1.0; // [-1, 1]
}

// ============================================================================
// TransportOperator3D implementation
// ============================================================================

TransportOperator3D::TransportOperator3D(const VelocityField* vel, const Grid3D& grid,
                                         const TransportOperatorConfig& cfg)
    : nx_(grid.nx), ny_(grid.ny), nz_(grid.nz), dx_(static_cast<double>(grid.dx)),
      dy_(static_cast<double>(grid.dy)), dz_(static_cast<double>(grid.dz)), grid_(grid), vel_(vel),
      cfg_(cfg) {}

void TransportOperator3D::ensure_velocity_cached(cudaStream_t stream) const {
    if (vel_cached_)
        return;

    const size_t n = size();
    d_vx_cc_.resize(n);
    d_vy_cc_.resize(n);
    d_vz_cc_.resize(n);

    const int block = 256;
    const int grid_k = (static_cast<int>(n) + block - 1) / block;

    kernel_cache_velocity_cc<<<grid_k, block, 0, stream>>>(
        vel_->U.data(), vel_->V.data(), vel_->W.data(), d_vx_cc_.data(), d_vy_cc_.data(),
        d_vz_cc_.data(), nx_, ny_, nz_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    vel_cached_ = true;
}

void TransportOperator3D::apply_D(DeviceSpan<const real> in, DeviceSpan<real> out,
                                  cudaStream_t stream) const {
    ensure_velocity_cached(stream);

    const int block = 256;
    const int grid_k = (static_cast<int>(size()) + block - 1) / block;

    const double inv_2dx = 0.5 / dx_;
    const double inv_2dy = 0.5 / dy_;
    const double inv_2dz = 0.5 / dz_;
    const double inv_dx = 1.0 / dx_;

    kernel_apply_D<<<grid_k, block, 0, stream>>>(
        in.data(), out.data(), d_vx_cc_.data(), d_vy_cc_.data(), d_vz_cc_.data(), nx_, ny_, nz_,
        inv_2dx, inv_2dy, inv_2dz, inv_dx, static_cast<int>(cfg_.x_bc));
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void TransportOperator3D::apply_DT(DeviceSpan<const real> in, DeviceSpan<real> out,
                                   cudaStream_t stream) const {
    ensure_velocity_cached(stream);

    const int block = 256;
    const int grid_k = (static_cast<int>(size()) + block - 1) / block;

    const double inv_2dx = 0.5 / dx_;
    const double inv_2dy = 0.5 / dy_;
    const double inv_2dz = 0.5 / dz_;
    const double inv_dx = 1.0 / dx_;

    kernel_apply_DT<<<grid_k, block, 0, stream>>>(
        in.data(), out.data(), d_vx_cc_.data(), d_vy_cc_.data(), d_vz_cc_.data(), nx_, ny_, nz_,
        inv_2dx, inv_2dy, inv_2dz, inv_dx, static_cast<int>(cfg_.x_bc));
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void TransportOperator3D::apply_DTD(DeviceSpan<const real> in, DeviceSpan<real> out,
                                    DeviceSpan<real> work, cudaStream_t stream) const {
    apply_D(in, work, stream);
    apply_DT(work, out, stream);
}

void TransportOperator3D::apply_DTWD(DeviceSpan<const real> in, DeviceSpan<real> out,
                                     DeviceSpan<real> work, DeviceSpan<const real> weight,
                                     cudaStream_t stream) const {
    apply_D(in, work, stream);

    if (weight.data() != nullptr && weight.size() >= size()) {
        const int block = 256;
        const int grid_k = (static_cast<int>(size()) + block - 1) / block;
        kernel_apply_diagonal_weight<<<grid_k, block, 0, stream>>>(
            work.data(), work.data(), weight.data(), static_cast<int>(size()));
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }

    apply_DT(work, out, stream);
}

double TransportOperator3D::test_constant_in_kernel(CudaContext& ctx) const {
    const size_t n = size();
    if (d_work1_.size() < n)
        d_work1_.resize(n);
    if (d_work2_.size() < n)
        d_work2_.resize(n);

    // Fill with constant 1.0
    blas::fill(ctx, d_work1_.span(), 1.0);

    // Apply D
    apply_D(d_work1_.span(), d_work2_.span(), ctx.cuda_stream());

    // Compute max |D(1)|
    blas::ReductionWorkspace red;
    double norm = blas::nrm2_host(ctx, d_work2_.span(), red);
    return norm / std::sqrt(static_cast<double>(n));
}

double TransportOperator3D::test_adjoint(CudaContext& ctx, unsigned seed) const {
    const size_t n = size();
    if (d_work1_.size() < n)
        d_work1_.resize(n);
    if (d_work2_.size() < n)
        d_work2_.resize(n);
    if (d_work3_.size() < n)
        d_work3_.resize(n);
    if (d_work4_.size() < n)
        d_work4_.resize(n);

    const int block = 256;
    const int grid_k = (static_cast<int>(n) + block - 1) / block;

    // Generate random x and y
    kernel_fill_random<<<grid_k, block, 0, ctx.cuda_stream()>>>(d_work1_.data(),
                                                                static_cast<int>(n), seed);
    kernel_fill_random<<<grid_k, block, 0, ctx.cuda_stream()>>>(
        d_work2_.data(), static_cast<int>(n), seed + 1000000);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // Compute Dx and D†y
    apply_D(d_work1_.span(), d_work3_.span(), ctx.cuda_stream());  // work3 = D*x
    apply_DT(d_work2_.span(), d_work4_.span(), ctx.cuda_stream()); // work4 = D†*y

    // Compute <Dx, y> and <x, D†y>
    blas::ReductionWorkspace red;
    double dot_Dx_y = blas::dot_host(ctx, d_work3_.span(), d_work2_.span(), red);
    double dot_x_DTy = blas::dot_host(ctx, d_work1_.span(), d_work4_.span(), red);

    double diff = std::fabs(dot_Dx_y - dot_x_DTy);
    double scale = std::max(std::fabs(dot_Dx_y), std::fabs(dot_x_DTy));
    if (scale < 1e-14)
        scale = 1.0;

    return diff / scale;
}

// ============================================================================
// LaplacianOperator3D implementation
// ============================================================================

LaplacianOperator3D::LaplacianOperator3D(const Grid3D& grid, XBoundary x_bc)
    : nx_(grid.nx), ny_(grid.ny), nz_(grid.nz), dx_(static_cast<double>(grid.dx)),
      dy_(static_cast<double>(grid.dy)), dz_(static_cast<double>(grid.dz)),
      inv_dx2_(1.0 / (dx_ * dx_)), inv_dy2_(1.0 / (dy_ * dy_)), inv_dz2_(1.0 / (dz_ * dz_)),
      x_bc_(x_bc) {}

void LaplacianOperator3D::apply_L(DeviceSpan<const real> in, DeviceSpan<real> out,
                                  cudaStream_t stream) const {
    const int block = 256;
    const int grid_k = (static_cast<int>(size()) + block - 1) / block;

    kernel_apply_L<<<grid_k, block, 0, stream>>>(in.data(), out.data(), nx_, ny_, nz_, inv_dx2_,
                                                 inv_dy2_, inv_dz2_, static_cast<int>(x_bc_));
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

double LaplacianOperator3D::test_symmetry(CudaContext& ctx, unsigned seed) const {
    const size_t n = size();
    if (d_work1_.size() < n)
        d_work1_.resize(n);
    if (d_work2_.size() < n)
        d_work2_.resize(n);

    DeviceBuffer<real> Lx, Ly;
    Lx.resize(n);
    Ly.resize(n);

    const int block = 256;
    const int grid_k = (static_cast<int>(n) + block - 1) / block;

    // Generate random x and y
    kernel_fill_random<<<grid_k, block, 0, ctx.cuda_stream()>>>(d_work1_.data(),
                                                                static_cast<int>(n), seed);
    kernel_fill_random<<<grid_k, block, 0, ctx.cuda_stream()>>>(
        d_work2_.data(), static_cast<int>(n), seed + 2000000);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // Compute Lx and Ly
    apply_L(d_work1_.span(), Lx.span(), ctx.cuda_stream());
    apply_L(d_work2_.span(), Ly.span(), ctx.cuda_stream());

    // Compute <Lx, y> and <x, Ly>
    blas::ReductionWorkspace red;
    double dot_Lx_y = blas::dot_host(ctx, Lx.span(), d_work2_.span(), red);
    double dot_x_Ly = blas::dot_host(ctx, d_work1_.span(), Ly.span(), red);

    double diff = std::fabs(dot_Lx_y - dot_x_Ly);
    double scale = std::max(std::fabs(dot_Lx_y), std::fabs(dot_x_Ly));
    if (scale < 1e-14)
        scale = 1.0;

    return diff / scale;
}

double LaplacianOperator3D::test_constant(CudaContext& ctx) const {
    const size_t n = size();
    if (d_work1_.size() < n)
        d_work1_.resize(n);
    if (d_work2_.size() < n)
        d_work2_.resize(n);

    // Fill with constant 1.0
    blas::fill(ctx, d_work1_.span(), 1.0);

    // Apply L
    apply_L(d_work1_.span(), d_work2_.span(), ctx.cuda_stream());

    // Compute RMS of L(1)
    blas::ReductionWorkspace red;
    double norm = blas::nrm2_host(ctx, d_work2_.span(), red);
    return norm / std::sqrt(static_cast<double>(n));
}

// Kernel to zero x-boundary cells (for interior-only norm with Neumann BC)
__global__ void kernel_zero_x_boundary(real* data, int nx, int ny, int nz) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (c >= total)
        return;
    const int i = c % nx;
    if (i == 0 || i == nx - 1)
        data[c] = 0.0;
}

// Kernel to fill with linear field: psi = a*x + b*y + c*z
__global__ void kernel_fill_linear(real* data, int nx, int ny, int nz, double dx, double dy,
                                   double dz, double a, double b, double c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (idx >= total)
        return;

    const int i = idx % nx;
    const int j = (idx / nx) % ny;
    const int k = idx / (nx * ny);

    // Cell-center coordinates
    const double x = (i + 0.5) * dx;
    const double y = (j + 0.5) * dy;
    const double z = (k + 0.5) * dz;

    data[idx] = a * x + b * y + c * z;
}

double LaplacianOperator3D::test_linear(CudaContext& ctx, double a, double b, double c) const {
    const size_t n = size();
    if (d_work1_.size() < n)
        d_work1_.resize(n);
    if (d_work2_.size() < n)
        d_work2_.resize(n);

    const int block = 256;
    const int grid_k = (static_cast<int>(n) + block - 1) / block;

    // Fill with linear field
    kernel_fill_linear<<<grid_k, block, 0, ctx.cuda_stream()>>>(d_work1_.data(), nx_, ny_, nz_, dx_,
                                                                dy_, dz_, a, b, c);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // Apply L
    apply_L(d_work1_.span(), d_work2_.span(), ctx.cuda_stream());

    // Zero x-boundary cells for Neumann BC: the Neumann ghost assumption
    // creates O(1/dx) artifacts at i=0,nx-1 for any linear field.
    // The interior stencil gives exactly zero for linear functions.
    if (x_bc_ == XBoundary::Neumann) {
        kernel_zero_x_boundary<<<grid_k, block, 0, ctx.cuda_stream()>>>(d_work2_.data(), nx_, ny_,
                                                                        nz_);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }

    // Compute RMS of L(linear) over interior cells only
    blas::ReductionWorkspace red;
    double norm = blas::nrm2_host(ctx, d_work2_.span(), red);
    return norm / std::sqrt(static_cast<double>(n));
}

// ============================================================================
// CombinedOperatorA implementation
// ============================================================================

CombinedOperatorA::CombinedOperatorA(const TransportOperator3D* D, const LaplacianOperator3D* L,
                                     double mu)
    : D_(D), L_(L), mu_(mu) {}

void CombinedOperatorA::ensure_work_buffers() {
    const size_t n = D_->size();
    if (d_work_D_.size() < n)
        d_work_D_.resize(n);
    if (d_work_DTD_.size() < n)
        d_work_DTD_.resize(n);
    if (d_work_L_.size() < n)
        d_work_L_.resize(n);
}

// Kernel for combining DTD and L parts (forward declaration location)
__global__ void kernel_combine_A(real* __restrict__ out, const real* __restrict__ dtd,
                                 const real* __restrict__ l_part, double mu, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n)
        return;
    out[c] = dtd[c] + mu * l_part[c];
}

void CombinedOperatorA::apply_A(DeviceSpan<const real> in, DeviceSpan<real> out,
                                cudaStream_t stream) {
    ensure_work_buffers();

    // D†D part
    D_->apply_DTD(in, d_work_DTD_.span(), d_work_D_.span(), stream);

    // L part
    L_->apply_L(in, d_work_L_.span(), stream);

    // Combine: out = D†D*in + mu*L*in
    const size_t n = D_->size();
    const int block = 256;
    const int grid_k = (static_cast<int>(n) + block - 1) / block;

    kernel_combine_A<<<grid_k, block, 0, stream>>>(out.data(), d_work_DTD_.data(), d_work_L_.data(),
                                                   mu_, static_cast<int>(n));
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

// (kernel_combine_A already defined above)

void CombinedOperatorA::apply_A_weighted(DeviceSpan<const real> in, DeviceSpan<real> out,
                                         DeviceSpan<const real> weight, cudaStream_t stream) {
    ensure_work_buffers();

    // D†WD part
    D_->apply_DTWD(in, d_work_DTD_.span(), d_work_D_.span(), weight, stream);

    // L part
    L_->apply_L(in, d_work_L_.span(), stream);

    // Combine
    const size_t n = D_->size();
    const int block = 256;
    const int grid_k = (static_cast<int>(n) + block - 1) / block;

    kernel_combine_A<<<grid_k, block, 0, stream>>>(out.data(), d_work_DTD_.data(), d_work_L_.data(),
                                                   mu_, static_cast<int>(n));
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

double CombinedOperatorA::test_symmetry(CudaContext& ctx, unsigned seed) {
    ensure_work_buffers();

    const size_t n = D_->size();
    DeviceBuffer<real> x, y, Ax, Ay;
    x.resize(n);
    y.resize(n);
    Ax.resize(n);
    Ay.resize(n);

    const int block = 256;
    const int grid_k = (static_cast<int>(n) + block - 1) / block;

    // Generate random vectors
    kernel_fill_random<<<grid_k, block, 0, ctx.cuda_stream()>>>(x.data(), static_cast<int>(n),
                                                                seed);
    kernel_fill_random<<<grid_k, block, 0, ctx.cuda_stream()>>>(y.data(), static_cast<int>(n),
                                                                seed + 3000000);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // Compute Ax and Ay
    apply_A_weighted(x.span(), Ax.span(), DeviceSpan<const real>(), ctx.cuda_stream());
    apply_A_weighted(y.span(), Ay.span(), DeviceSpan<const real>(), ctx.cuda_stream());

    // Compute <Ax, y> and <x, Ay>
    blas::ReductionWorkspace red;
    double dot_Ax_y = blas::dot_host(ctx, Ax.span(), y.span(), red);
    double dot_x_Ay = blas::dot_host(ctx, x.span(), Ay.span(), red);

    double diff = std::fabs(dot_Ax_y - dot_x_Ay);
    double scale = std::max(std::fabs(dot_Ax_y), std::fabs(dot_x_Ay));
    if (scale < 1e-14)
        scale = 1.0;

    return diff / scale;
}

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
