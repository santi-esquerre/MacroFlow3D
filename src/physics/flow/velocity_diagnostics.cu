/**
 * @file velocity_diagnostics.cu
 * @brief GPU kernels for velocity-field diagnostics (divergence, vorticity, helicity)
 * @ingroup physics_flow
 *
 * Implements the numerical stencils described in velocity_diagnostics.cuh.
 * Single 3D kernel, one thread per cell center.
 */

#include "../../runtime/cuda_check.cuh"
#include "../stochastic/stochastic.cuh" // compute_field_stats
#include "padded_layout.cuh"
#include "velocity_diagnostics.cuh"

#include <cmath>
#include <cstdio>

namespace macroflow3d {
namespace physics {

// ============================================================================
// Kernel: compute divergence, |omega|, helicity at cell centers
// ============================================================================

/**
 * @brief One thread per cell center (i,j,k).
 *
 * Divergence (all cells):
 *   div = (U(i+1,j,k) - U(i,j,k)) / dx
 *       + (V(i,j+1,k) - V(i,j,k)) / dy
 *       + (W(i,j,k+1) - W(i,j,k)) / dz
 *
 * Curl/helicity (interior cells only, i in [1,nx-2] etc.):
 *   uc = 0.5*(U(i,j,k) + U(i+1,j,k))          — face-averaged to center
 *   vc = 0.5*(V(i,j,k) + V(i,j+1,k))
 *   wc = 0.5*(W(i,j,k) + W(i,j,k+1))
 *
 *   Derivatives via centered differences of cell-centered velocities
 *   (using neighbours' face-averages):
 *     duc/dz = (uc(i,j,k+1) - uc(i,j,k-1)) / (2*dz)   etc.
 *
 *   omega_x = dwc/dy - dvc/dz
 *   omega_y = duc/dz - dwc/dx
 *   omega_z = dvc/dx - duc/dy
 *   |omega| = sqrt(omega_x^2 + omega_y^2 + omega_z^2)
 *   helicity = uc*omega_x + vc*omega_y + wc*omega_z
 */
__global__ void kernel_velocity_diagnostics(const real* __restrict__ U, const real* __restrict__ V,
                                            const real* __restrict__ W, real* __restrict__ div_out,
                                            real* __restrict__ omag_out, real* __restrict__ hel_out,
                                            int nx, int ny, int nz, real inv_dx, real inv_dy,
                                            real inv_dz,
                                            bool compute_curl) // false if any dim < 3
{
    // Use size_t throughout to avoid overflow on large grids.
    const size_t total = static_cast<size_t>(nx) * ny * nz;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    // Decompose linear index → (i,j,k)
    const int i = static_cast<int>(idx % nx);
    const int j = static_cast<int>((idx / nx) % ny);
    const int k = static_cast<int>(idx / (static_cast<size_t>(nx) * ny));

    // ── Divergence (all cells) ─────────────────────────────────────
    const real Uip1 = U[padded_idx(i + 1, j, k, nx, ny)];
    const real Ui = U[padded_idx(i, j, k, nx, ny)];
    const real Vjp1 = V[padded_idx(i, j + 1, k, nx, ny)];
    const real Vj = V[padded_idx(i, j, k, nx, ny)];
    const real Wkp1 = W[padded_idx(i, j, k + 1, nx, ny)];
    const real Wk = W[padded_idx(i, j, k, nx, ny)];

    const real div_val = (Uip1 - Ui) * inv_dx + (Vjp1 - Vj) * inv_dy + (Wkp1 - Wk) * inv_dz;
    div_out[idx] = div_val;

    // ── Curl & helicity (interior only) ────────────────────────────
    if (!compute_curl || i < 1 || i > nx - 2 || j < 1 || j > ny - 2 || k < 1 || k > nz - 2) {
        omag_out[idx] = real(0);
        hel_out[idx] = real(0);
        return;
    }

    // Face-averaged velocities at cell center (i,j,k)
    const real uc = real(0.5) * (Ui + Uip1);
    const real vc = real(0.5) * (Vj + Vjp1);
    const real wc = real(0.5) * (Wk + Wkp1);

    // Face-averaged velocity at neighbor cells
    // uc at (i, j, k±1)
    const real uc_kp1 =
        real(0.5) * (U[padded_idx(i, j, k + 1, nx, ny)] + U[padded_idx(i + 1, j, k + 1, nx, ny)]);
    const real uc_km1 =
        real(0.5) * (U[padded_idx(i, j, k - 1, nx, ny)] + U[padded_idx(i + 1, j, k - 1, nx, ny)]);
    // uc at (i, j±1, k)
    const real uc_jp1 =
        real(0.5) * (U[padded_idx(i, j + 1, k, nx, ny)] + U[padded_idx(i + 1, j + 1, k, nx, ny)]);
    const real uc_jm1 =
        real(0.5) * (U[padded_idx(i, j - 1, k, nx, ny)] + U[padded_idx(i + 1, j - 1, k, nx, ny)]);

    // vc at (i±1, j, k)
    const real vc_ip1 =
        real(0.5) * (V[padded_idx(i + 1, j, k, nx, ny)] + V[padded_idx(i + 1, j + 1, k, nx, ny)]);
    const real vc_im1 =
        real(0.5) * (V[padded_idx(i - 1, j, k, nx, ny)] + V[padded_idx(i - 1, j + 1, k, nx, ny)]);
    // vc at (i, j, k±1)
    const real vc_kp1 =
        real(0.5) * (V[padded_idx(i, j, k + 1, nx, ny)] + V[padded_idx(i, j + 1, k + 1, nx, ny)]);
    const real vc_km1 =
        real(0.5) * (V[padded_idx(i, j, k - 1, nx, ny)] + V[padded_idx(i, j + 1, k - 1, nx, ny)]);

    // wc at (i±1, j, k)
    const real wc_ip1 =
        real(0.5) * (W[padded_idx(i + 1, j, k, nx, ny)] + W[padded_idx(i + 1, j, k + 1, nx, ny)]);
    const real wc_im1 =
        real(0.5) * (W[padded_idx(i - 1, j, k, nx, ny)] + W[padded_idx(i - 1, j, k + 1, nx, ny)]);
    // wc at (i, j±1, k)
    const real wc_jp1 =
        real(0.5) * (W[padded_idx(i, j + 1, k, nx, ny)] + W[padded_idx(i, j + 1, k + 1, nx, ny)]);
    const real wc_jm1 =
        real(0.5) * (W[padded_idx(i, j - 1, k, nx, ny)] + W[padded_idx(i, j - 1, k + 1, nx, ny)]);

    // Centered derivatives (1/(2h))
    const real inv_2dx = real(0.5) * inv_dx;
    const real inv_2dy = real(0.5) * inv_dy;
    const real inv_2dz = real(0.5) * inv_dz;

    const real dwc_dy = (wc_jp1 - wc_jm1) * inv_2dy;
    const real dvc_dz = (vc_kp1 - vc_km1) * inv_2dz;
    const real duc_dz = (uc_kp1 - uc_km1) * inv_2dz;
    const real dwc_dx = (wc_ip1 - wc_im1) * inv_2dx;
    const real dvc_dx = (vc_ip1 - vc_im1) * inv_2dx;
    const real duc_dy = (uc_jp1 - uc_jm1) * inv_2dy;

    const real omega_x = dwc_dy - dvc_dz;
    const real omega_y = duc_dz - dwc_dx;
    const real omega_z = dvc_dx - duc_dy;

    omag_out[idx] = sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z);
    hel_out[idx] = uc * omega_x + vc * omega_y + wc * omega_z;
}

// ============================================================================
// Host API
// ============================================================================

void compute_velocity_diagnostics(const PaddedVelocityField& vel, VelocityDiagnostics& diag,
                                  const Grid3D& grid, const CudaContext& ctx) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;
    // size_t to avoid overflow on large grids (e.g. 2048x256x256 = 134 M cells)
    const size_t total = static_cast<size_t>(nx) * ny * nz;

    if (total == 0)
        return;

    const bool can_curl = (nx >= 3 && ny >= 3 && nz >= 3);
    if (!can_curl) {
        std::printf("       [diag] WARNING: grid too small for curl/helicity "
                    "(%dx%dx%d, need >=3 in each dim). Only divergence computed.\n",
                    nx, ny, nz);
    }

    const real inv_dx = real(1) / grid.dx;
    const real inv_dy = real(1) / grid.dy;
    const real inv_dz = real(1) / grid.dz;

    // Grid-stride launch using size_t-safe block count.
    const int block = 256;
    const size_t n_blocks = (total + block - 1) / block;

    kernel_velocity_diagnostics<<<(unsigned)n_blocks, block, 0, ctx.cuda_stream()>>>(
        vel.U_ptr(), vel.V_ptr(), vel.W_ptr(), diag.divergence.device_ptr(),
        diag.vorticity_mag.device_ptr(), diag.helicity.device_ptr(), nx, ny, nz, inv_dx, inv_dy,
        inv_dz, can_curl);

    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void print_velocity_diagnostics(const VelocityDiagnostics& diag, int realization_id,
                                const CudaContext& ctx) {
    real min_v, max_v, mean_v;

    // Divergence
    compute_field_stats(diag.divergence.span(), min_v, max_v, mean_v, ctx);
    std::printf("       [diag] r=%d  div:  min=%.4e  max=%.4e  mean=%.4e\n", realization_id, min_v,
                max_v, mean_v);

    // Vorticity magnitude  (stats include boundary cells set to 0 — see note below)
    compute_field_stats(diag.vorticity_mag.span(), min_v, max_v, mean_v, ctx);
    std::printf("       [diag] r=%d  |w|:  min=%.4e  max=%.4e  mean=%.4e  (incl. border=0)\n",
                realization_id, min_v, max_v, mean_v);

    // Helicity  (stats include boundary cells set to 0 — see note below)
    compute_field_stats(diag.helicity.span(), min_v, max_v, mean_v, ctx);
    std::printf("       [diag] r=%d  h:    min=%.4e  max=%.4e  mean=%.4e  (incl. border=0)\n",
                realization_id, min_v, max_v, mean_v);
}

} // namespace physics
} // namespace macroflow3d
