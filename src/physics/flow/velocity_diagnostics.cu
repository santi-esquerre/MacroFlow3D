/**
 * @file velocity_diagnostics.cu
 * @brief GPU kernels for velocity-field diagnostics (divergence, vorticity, helicity)
 * @ingroup physics_flow
 *
 * Implements the numerical stencils described in velocity_diagnostics.cuh.
 * Single 3D kernel, one thread per cell center.
 */

#include "../../external/Par2_Core/src/internal/fields/facefield_accessor.cuh"
#include "../../external/Par2_Core/src/internal/fields/potential_flow_accessor.cuh"
#include "../../runtime/cuda_check.cuh"
#include "../stochastic/stochastic.cuh" // compute_field_stats
#include "padded_layout.cuh"
#include "velocity_diagnostics.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

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

// ============================================================================
// Sampled backend diagnostics for FACE vs KH comparison
// ============================================================================

namespace {

par2::ScalarBoundaryType to_par2_scalar_bc(BCType t) {
    switch (t) {
    case BCType::Dirichlet:
        return par2::ScalarBoundaryType::Dirichlet;
    case BCType::Neumann:
        return par2::ScalarBoundaryType::Neumann;
    case BCType::Periodic:
        return par2::ScalarBoundaryType::Periodic;
    default:
        return par2::ScalarBoundaryType::Extrapolate;
    }
}

par2::PotentialBoundaryConfig<real> make_potential_bc(const BCSpec& bc) {
    par2::PotentialBoundaryConfig<real> out;
    out.x.lo.type = to_par2_scalar_bc(bc.xmin.type);
    out.x.lo.value = bc.xmin.value;
    out.x.hi.type = to_par2_scalar_bc(bc.xmax.type);
    out.x.hi.value = bc.xmax.value;
    out.y.lo.type = to_par2_scalar_bc(bc.ymin.type);
    out.y.lo.value = bc.ymin.value;
    out.y.hi.type = to_par2_scalar_bc(bc.ymax.type);
    out.y.hi.value = bc.ymax.value;
    out.z.lo.type = to_par2_scalar_bc(bc.zmin.type);
    out.z.lo.value = bc.zmin.value;
    out.z.hi.type = to_par2_scalar_bc(bc.zmax.type);
    out.z.hi.value = bc.zmax.value;
    return out;
}

par2::GridDesc<real> make_diag_grid(const Grid3D& grid) {
    return par2::make_grid<real>(grid.nx, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz);
}

par2::PotentialFlowView<real> make_potential_view(const KField& K, const HeadField& head,
                                                  const BCSpec& bc) {
    par2::PotentialFlowView<real> out;
    out.K = K.device_ptr();
    out.head = head.device_ptr();
    out.size = K.size();
    out.head_bc = make_potential_bc(bc);
    return out;
}

struct SamplingPlan {
    size_t count = 0;
    size_t stride = 1;
};

SamplingPlan make_sampling_plan(size_t total, size_t max_samples) {
    SamplingPlan p;
    if (total == 0 || max_samples == 0)
        return p;
    const size_t target = std::min(total, max_samples);
    p.stride = (total + target - 1) / target;
    p.count = (total + p.stride - 1) / p.stride;
    return p;
}

real percentile_sorted(const std::vector<real>& sorted, double p) {
    if (sorted.empty())
        return real(0);
    const double clamped = std::max(0.0, std::min(1.0, p));
    const size_t idx =
        std::min(sorted.size() - 1,
                 static_cast<size_t>(std::ceil(clamped * static_cast<double>(sorted.size())) - 1));
    return sorted[idx];
}

__device__ __forceinline__ int wrap_or_clamp_index(int i, int n, bool periodic) {
    if (n <= 1)
        return 0;
    if (periodic) {
        int r = i % n;
        return (r < 0) ? r + n : r;
    }
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

__device__ __forceinline__ void neighbor_indices(int i, int n, bool periodic, int& im, int& ip,
                                                 real& denom, real spacing) {
    if (n <= 1) {
        im = ip = 0;
        denom = real(1);
        return;
    }
    if (periodic || (i > 0 && i < n - 1)) {
        im = wrap_or_clamp_index(i - 1, n, periodic);
        ip = wrap_or_clamp_index(i + 1, n, periodic);
        denom = real(2) * spacing;
        return;
    }
    if (i == 0) {
        im = 0;
        ip = 1;
        denom = spacing;
    } else {
        im = n - 2;
        ip = n - 1;
        denom = spacing;
    }
}

__device__ __forceinline__ void sample_backend_at_cell(int mode, const real* __restrict__ U,
                                                       const real* __restrict__ V,
                                                       const real* __restrict__ W,
                                                       const par2::PotentialFlowView<real>& pf,
                                                       const par2::GridDesc<real>& g, int i, int j,
                                                       int k, real& qx, real& qy, real& qz) {
    const real x = g.px + (real(i) + real(0.5)) * g.dx;
    const real y = g.py + (real(j) + real(0.5)) * g.dy;
    const real z = g.pz + (real(k) + real(0.5)) * g.dz;

    if (mode == static_cast<int>(VelocityEvalDiagnosticMode::KhPotentialReconstruction)) {
        par2::internal::sample_velocity_kh_potential(pf, g, x, y, z, qx, qy, qz);
    } else {
        par2::internal::sample_velocity_facefield_2d_aware(U, V, W, g, i, j, k, true, x, y, z, qx,
                                                           qy, qz);
    }
}

__device__ __forceinline__ bool finite3(real x, real y, real z) {
    return isfinite(x) && isfinite(y) && isfinite(z);
}

__global__ void
kernel_velocity_eval_samples(int mode, const real* __restrict__ U, const real* __restrict__ V,
                             const real* __restrict__ W, par2::PotentialFlowView<real> pf,
                             par2::GridDesc<real> g, bool periodic_x, bool periodic_y,
                             bool periodic_z, size_t total_cells, size_t sample_stride,
                             VelocityDiagnosticSample* __restrict__ out, size_t n_samples) {
    const size_t sid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sid >= n_samples)
        return;

    size_t cell = sid * sample_stride;
    if (cell >= total_cells)
        cell = total_cells - 1;

    const int i = static_cast<int>(cell % g.nx);
    const int j = static_cast<int>((cell / g.nx) % g.ny);
    const int k = static_cast<int>(cell / (static_cast<size_t>(g.nx) * g.ny));

    int im, ip, jm, jp, km, kp;
    real denom_x, denom_y, denom_z;
    neighbor_indices(i, g.nx, periodic_x, im, ip, denom_x, g.dx);
    neighbor_indices(j, g.ny, periodic_y, jm, jp, denom_y, g.dy);
    neighbor_indices(k, g.nz, periodic_z, km, kp, denom_z, g.dz);

    real qx, qy, qz;
    real qxm_x, qxm_y, qxm_z, qxp_x, qxp_y, qxp_z;
    real qym_x, qym_y, qym_z, qyp_x, qyp_y, qyp_z;
    real qzm_x, qzm_y, qzm_z, qzp_x, qzp_y, qzp_z;

    sample_backend_at_cell(mode, U, V, W, pf, g, i, j, k, qx, qy, qz);
    sample_backend_at_cell(mode, U, V, W, pf, g, im, j, k, qxm_x, qxm_y, qxm_z);
    sample_backend_at_cell(mode, U, V, W, pf, g, ip, j, k, qxp_x, qxp_y, qxp_z);
    sample_backend_at_cell(mode, U, V, W, pf, g, i, jm, k, qym_x, qym_y, qym_z);
    sample_backend_at_cell(mode, U, V, W, pf, g, i, jp, k, qyp_x, qyp_y, qyp_z);
    sample_backend_at_cell(mode, U, V, W, pf, g, i, j, km, qzm_x, qzm_y, qzm_z);
    sample_backend_at_cell(mode, U, V, W, pf, g, i, j, kp, qzp_x, qzp_y, qzp_z);

    const real dqx_dx = (qxp_x - qxm_x) / denom_x;
    const real dqy_dx = (qxp_y - qxm_y) / denom_x;
    const real dqz_dx = (qxp_z - qxm_z) / denom_x;

    const real dqx_dy = (qyp_x - qym_x) / denom_y;
    const real dqy_dy = (qyp_y - qym_y) / denom_y;
    const real dqz_dy = (qyp_z - qym_z) / denom_y;

    const real dqx_dz = (qzp_x - qzm_x) / denom_z;
    const real dqy_dz = (qzp_y - qzm_y) / denom_z;
    const real dqz_dz = (qzp_z - qzm_z) / denom_z;

    const real div = dqx_dx + dqy_dy + dqz_dz;
    const real curl_x = dqz_dy - dqy_dz;
    const real curl_y = dqx_dz - dqz_dx;
    const real curl_z = dqy_dx - dqx_dy;
    const real curl_mag = sqrt(curl_x * curl_x + curl_y * curl_y + curl_z * curl_z);
    const real speed = sqrt(qx * qx + qy * qy + qz * qz);
    const real helicity = qx * curl_x + qy * curl_y + qz * curl_z;
    const real eps = real(1.0e-30);

    VelocityDiagnosticSample s;
    s.speed = speed;
    s.div_abs = fabs(div);
    s.curl_mag = curl_mag;
    s.helicity_abs = fabs(helicity);
    s.helicity_norm = fabs(helicity) / (speed * curl_mag + eps);
    s.finite = finite3(qx, qy, qz) && isfinite(div) && finite3(curl_x, curl_y, curl_z) ? 1 : 0;
    out[sid] = s;
}

__global__ void
kernel_velocity_comparison_samples(const real* __restrict__ U, const real* __restrict__ V,
                                   const real* __restrict__ W, par2::PotentialFlowView<real> pf,
                                   par2::GridDesc<real> g, size_t total_cells, size_t sample_stride,
                                   VelocityComparisonSample* __restrict__ out, size_t n_samples) {
    const size_t sid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sid >= n_samples)
        return;

    size_t cell = sid * sample_stride;
    if (cell >= total_cells)
        cell = total_cells - 1;

    const int i = static_cast<int>(cell % g.nx);
    const int j = static_cast<int>((cell / g.nx) % g.ny);
    const int k = static_cast<int>(cell / (static_cast<size_t>(g.nx) * g.ny));

    real fx, fy, fz, kx, ky, kz;
    sample_backend_at_cell(static_cast<int>(VelocityEvalDiagnosticMode::FaceTrilinear), U, V, W, pf,
                           g, i, j, k, fx, fy, fz);
    sample_backend_at_cell(static_cast<int>(VelocityEvalDiagnosticMode::KhPotentialReconstruction),
                           U, V, W, pf, g, i, j, k, kx, ky, kz);

    const real dx = kx - fx;
    const real dy = ky - fy;
    const real dz = kz - fz;

    VelocityComparisonSample s;
    s.diff_mag = sqrt(dx * dx + dy * dy + dz * dz);
    s.face_sq = fx * fx + fy * fy + fz * fz;
    s.kh_sq = kx * kx + ky * ky + kz * kz;
    s.dot = fx * kx + fy * ky + fz * kz;
    s.finite = finite3(fx, fy, fz) && finite3(kx, ky, kz) ? 1 : 0;
    out[sid] = s;
}

} // namespace

VelocityEvalDiagnosticsSummary
compute_velocity_eval_diagnostics(VelocityEvalDiagnosticMode mode,
                                  const PaddedVelocityField& face_velocity, const KField& K,
                                  const HeadField& head, const Grid3D& grid, const BCSpec& bc,
                                  VelocityEvalDiagnosticsWorkspace& workspace,
                                  const CudaContext& ctx, int realization_id, size_t max_samples) {
    const size_t total = grid.num_cells();
    const SamplingPlan plan = make_sampling_plan(total, max_samples);

    VelocityEvalDiagnosticsSummary summary;
    summary.realization_id = realization_id;
    summary.backend = (mode == VelocityEvalDiagnosticMode::KhPotentialReconstruction)
                          ? "KH_POTENTIAL_RECONSTRUCTION"
                          : "FACE_TRILINEAR";
    summary.n_samples = plan.count;
    summary.sample_stride = plan.stride;

    if (plan.count == 0)
        return summary;

    workspace.samples.resize(plan.count);
    workspace.host_samples.resize(plan.count);

    const par2::GridDesc<real> pg = make_diag_grid(grid);
    const par2::PotentialFlowView<real> pf = make_potential_view(K, head, bc);
    const bool periodic_x = bc.xmin.type == BCType::Periodic && bc.xmax.type == BCType::Periodic;
    const bool periodic_y = bc.ymin.type == BCType::Periodic && bc.ymax.type == BCType::Periodic;
    const bool periodic_z = bc.zmin.type == BCType::Periodic && bc.zmax.type == BCType::Periodic;

    const int block = 256;
    const int n_blocks = static_cast<int>((plan.count + block - 1) / block);
    kernel_velocity_eval_samples<<<n_blocks, block, 0, ctx.cuda_stream()>>>(
        static_cast<int>(mode), face_velocity.U_ptr(), face_velocity.V_ptr(), face_velocity.W_ptr(),
        pf, pg, periodic_x, periodic_y, periodic_z, total, plan.stride, workspace.samples.data(),
        plan.count);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(workspace.host_samples.data(), workspace.samples.data(),
                                           plan.count * sizeof(VelocityDiagnosticSample),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream()));

    std::vector<real> helicity_norms;
    helicity_norms.reserve(plan.count);
    double speed_sum = 0;
    double div_sum = 0;
    double curl_sum = 0;
    double helicity_sum = 0;
    double helicity_norm_sum = 0;

    for (const auto& s : workspace.host_samples) {
        if (!s.finite) {
            ++summary.invalid_count;
            continue;
        }
        ++summary.finite_count;
        speed_sum += s.speed;
        div_sum += s.div_abs;
        curl_sum += s.curl_mag;
        helicity_sum += s.helicity_abs;
        helicity_norm_sum += s.helicity_norm;
        summary.speed_max = std::max(summary.speed_max, s.speed);
        summary.div_abs_max = std::max(summary.div_abs_max, s.div_abs);
        summary.curl_mag_max = std::max(summary.curl_mag_max, s.curl_mag);
        summary.helicity_abs_max = std::max(summary.helicity_abs_max, s.helicity_abs);
        summary.helicity_norm_max = std::max(summary.helicity_norm_max, s.helicity_norm);
        helicity_norms.push_back(s.helicity_norm);
    }

    if (summary.finite_count > 0) {
        const real inv = real(1) / real(summary.finite_count);
        summary.speed_mean = real(speed_sum) * inv;
        summary.div_abs_mean = real(div_sum) * inv;
        summary.curl_mag_mean = real(curl_sum) * inv;
        summary.helicity_abs_mean = real(helicity_sum) * inv;
        summary.helicity_norm_mean = real(helicity_norm_sum) * inv;
        std::sort(helicity_norms.begin(), helicity_norms.end());
        summary.helicity_norm_p95 = percentile_sorted(helicity_norms, 0.95);
    }

    return summary;
}

VelocityBackendComparisonSummary compute_velocity_backend_comparison(
    const PaddedVelocityField& face_velocity, const KField& K, const HeadField& head,
    const Grid3D& grid, const BCSpec& bc, VelocityEvalDiagnosticsWorkspace& workspace,
    const CudaContext& ctx, int realization_id, size_t max_samples) {
    const size_t total = grid.num_cells();
    const SamplingPlan plan = make_sampling_plan(total, max_samples);

    VelocityBackendComparisonSummary summary;
    summary.realization_id = realization_id;
    summary.n_samples = plan.count;
    summary.sample_stride = plan.stride;
    if (plan.count == 0)
        return summary;

    workspace.comparison_samples.resize(plan.count);
    workspace.host_comparison_samples.resize(plan.count);

    const par2::GridDesc<real> pg = make_diag_grid(grid);
    const par2::PotentialFlowView<real> pf = make_potential_view(K, head, bc);

    const int block = 256;
    const int n_blocks = static_cast<int>((plan.count + block - 1) / block);
    kernel_velocity_comparison_samples<<<n_blocks, block, 0, ctx.cuda_stream()>>>(
        face_velocity.U_ptr(), face_velocity.V_ptr(), face_velocity.W_ptr(), pf, pg, total,
        plan.stride, workspace.comparison_samples.data(), plan.count);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(
        workspace.host_comparison_samples.data(), workspace.comparison_samples.data(),
        plan.count * sizeof(VelocityComparisonSample), cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream()));

    std::vector<real> diffs;
    diffs.reserve(plan.count);
    double diff_sum = 0;
    double diff_sq_sum = 0;
    double face_sq_sum = 0;
    double kh_sq_sum = 0;
    double dot_sum = 0;

    for (const auto& s : workspace.host_comparison_samples) {
        if (!s.finite) {
            ++summary.invalid_count;
            continue;
        }
        ++summary.finite_count;
        diff_sum += s.diff_mag;
        diff_sq_sum += static_cast<double>(s.diff_mag) * static_cast<double>(s.diff_mag);
        face_sq_sum += s.face_sq;
        kh_sq_sum += s.kh_sq;
        dot_sum += s.dot;
        summary.diff_max = std::max(summary.diff_max, s.diff_mag);
        diffs.push_back(s.diff_mag);
    }

    if (summary.finite_count > 0) {
        const double inv = 1.0 / static_cast<double>(summary.finite_count);
        const double mean = diff_sum * inv;
        const double variance = std::max(0.0, diff_sq_sum * inv - mean * mean);
        summary.diff_mean = static_cast<real>(mean);
        summary.diff_std = static_cast<real>(std::sqrt(variance));
        summary.rel_l2_diff =
            static_cast<real>(std::sqrt(diff_sq_sum / std::max(face_sq_sum, 1.0e-300)));
        summary.vector_correlation =
            static_cast<real>(dot_sum / std::sqrt(std::max(face_sq_sum * kh_sq_sum, 1.0e-300)));
        std::sort(diffs.begin(), diffs.end());
        summary.diff_p50 = percentile_sorted(diffs, 0.50);
        summary.diff_p95 = percentile_sorted(diffs, 0.95);
    }

    return summary;
}

} // namespace physics
} // namespace macroflow3d
