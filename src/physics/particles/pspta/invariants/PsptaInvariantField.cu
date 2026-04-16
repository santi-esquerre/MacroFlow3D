/**
 * @file PsptaInvariantField.cu
 * @brief Implementation of PsptaInvariantField container.
 */

#include "../../../../runtime/cuda_check.cuh"
#include "EigensolverBackend.cuh"
#include "PsptaInvariantField.cuh"
#include <cmath>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Kernel: compute cached derivatives (optional)
// ============================================================================

/**
 * @brief Compute d(psi)/dy and d(psi)/dz using central differences.
 *
 * Periodic lifting in y for psi1 (period Ly) and z for psi2 (period Lz).
 */
__global__ void
kernel_compute_psi_derivatives(const float* __restrict__ psi, float* __restrict__ dpsi_dy,
                               float* __restrict__ dpsi_dz, int nx, int ny, int nz, double dy,
                               double dz,
                               double L_self) // period of psi itself (Ly for psi1, Lz for psi2)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (c >= total)
        return;

    const int i = c % nx;
    const int j = (c / nx) % ny;
    const int k = c / (nx * ny);

    const double psi_c = static_cast<double>(psi[c]);

    // d/dy: periodic central with lifting
    const int jm = (j - 1 + ny) % ny;
    const int jp = (j + 1) % ny;
    const int idx_jm = i + nx * (jm + ny * k);
    const int idx_jp = i + nx * (jp + ny * k);

    double psi_jm = static_cast<double>(psi[idx_jm]);
    double psi_jp = static_cast<double>(psi[idx_jp]);
    psi_jm += L_self * round((psi_c - psi_jm) / L_self);
    psi_jp += L_self * round((psi_c - psi_jp) / L_self);
    dpsi_dy[c] = static_cast<float>((psi_jp - psi_jm) / (2.0 * dy));

    // d/dz: periodic central with lifting
    const int km = (k - 1 + nz) % nz;
    const int kp = (k + 1) % nz;
    const int idx_km = i + nx * (j + ny * km);
    const int idx_kp = i + nx * (j + ny * kp);

    double psi_km = static_cast<double>(psi[idx_km]);
    double psi_kp = static_cast<double>(psi[idx_kp]);
    psi_km += L_self * round((psi_c - psi_km) / L_self);
    psi_kp += L_self * round((psi_c - psi_kp) / L_self);
    dpsi_dz[c] = static_cast<float>((psi_kp - psi_km) / (2.0 * dz));
}

/**
 * @brief Compute d(psi)/dx using one-sided differences at boundaries.
 *
 * Uses forward difference at i=0, backward at i=nx-1, central elsewhere.
 * No periodic lifting in x (open boundary).
 */
__global__ void kernel_compute_psi_dx(const float* __restrict__ psi, float* __restrict__ dpsi_dx,
                                      int nx, int ny, int nz, double dx) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (c >= total)
        return;

    const int i = c % nx;
    const int j = (c / nx) % ny;
    const int k = c / (nx * ny);

    auto idx = [nx, ny](int ii, int jj, int kk) { return ii + nx * (jj + ny * kk); };

    double result;
    if (i == 0) {
        // Forward difference
        result =
            (static_cast<double>(psi[idx(1, j, k)]) - static_cast<double>(psi[idx(0, j, k)])) / dx;
    } else if (i == nx - 1) {
        // Backward difference
        result = (static_cast<double>(psi[idx(nx - 1, j, k)]) -
                  static_cast<double>(psi[idx(nx - 2, j, k)])) /
                 dx;
    } else {
        // Central difference
        result = (static_cast<double>(psi[idx(i + 1, j, k)]) -
                  static_cast<double>(psi[idx(i - 1, j, k)])) /
                 (2.0 * dx);
    }

    dpsi_dx[c] = static_cast<float>(result);
}

// ============================================================================
// Kernel: compute invariance quality reduction
// ============================================================================

/// CAS-based double atomicAdd
__device__ inline void atomic_add_d(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long assumed, old = *addr_ull;
    do {
        assumed = old;
        old =
            atomicCAS(addr_ull, assumed, __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
}

/// CAS-based double atomicMax (for positive values)
__device__ inline void atomic_max_d(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long assumed, old = *addr_ull;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        if (val <= cur)
            break;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

/**
 * @brief Compute v*grad(psi) residuals and reduce to sumsq, max.
 *
 * Output buffer layout (8 doubles):
 *   [0] sumsq_r1  [1] sumsq_r2  [2] max_r1  [3] max_r2
 *   [4] sumsq_cross  [5] max_cross  [6] sumsq_vmag  [7] count
 */
__global__ void kernel_invariance_quality_reduce(
    const float* __restrict__ psi1, const float* __restrict__ psi2, const real* __restrict__ U,
    const real* __restrict__ V, const real* __restrict__ W, int nx, int ny, int nz, double dx,
    double dy, double dz, double Ly, double Lz, double* __restrict__ out) {
    extern __shared__ double smem[];
    const int bs = blockDim.x;
    double* s_ssq1 = smem;
    double* s_ssq2 = smem + bs;
    double* s_max1 = smem + 2 * bs;
    double* s_max2 = smem + 3 * bs;

    const int tid = threadIdx.x;
    s_ssq1[tid] = 0.0;
    s_ssq2[tid] = 0.0;
    s_max1[tid] = 0.0;
    s_max2[tid] = 0.0;
    __syncthreads();

    const int total = nx * ny * nz;
    const int c = blockIdx.x * bs + tid;

    if (c < total) {
        const int i = c % nx;
        const int j = (c / nx) % ny;
        const int k = c / (nx * ny);

        // Cell-center velocity from CompactMAC
        auto idx_U = [nx, ny](int ii, int jj, int kk) {
            return ii + (nx + 1) * jj + (nx + 1) * ny * kk;
        };
        auto idx_V = [nx, ny](int ii, int jj, int kk) { return ii + nx * jj + nx * (ny + 1) * kk; };
        auto idx_W = [nx, ny](int ii, int jj, int kk) { return ii + nx * jj + nx * ny * kk; };

        const double vx = 0.5 * (static_cast<double>(U[idx_U(i, j, k)]) +
                                 static_cast<double>(U[idx_U(i + 1, j, k)]));
        const double vy = 0.5 * (static_cast<double>(V[idx_V(i, j, k)]) +
                                 static_cast<double>(V[idx_V(i, j + 1, k)]));
        const double vz = 0.5 * (static_cast<double>(W[idx_W(i, j, k)]) +
                                 static_cast<double>(W[idx_W(i, j, k + 1)]));

        // Gradient of psi1
        const double p1_c = static_cast<double>(psi1[c]);

        // d/dx (one-sided at boundaries)
        double dp1_dx;
        if (i == 0)
            dp1_dx = (static_cast<double>(psi1[c + 1]) - p1_c) / dx;
        else if (i == nx - 1)
            dp1_dx = (p1_c - static_cast<double>(psi1[c - 1])) / dx;
        else
            dp1_dx =
                (static_cast<double>(psi1[c + 1]) - static_cast<double>(psi1[c - 1])) / (2.0 * dx);

        // d/dy (periodic with lifting for psi1, period Ly)
        const int jm = (j - 1 + ny) % ny;
        const int jp = (j + 1) % ny;
        double p1_jm = static_cast<double>(psi1[i + nx * (jm + ny * k)]);
        double p1_jp = static_cast<double>(psi1[i + nx * (jp + ny * k)]);
        p1_jm += Ly * round((p1_c - p1_jm) / Ly);
        p1_jp += Ly * round((p1_c - p1_jp) / Ly);
        const double dp1_dy = (p1_jp - p1_jm) / (2.0 * dy);

        // d/dz (periodic, no lifting for psi1)
        const int km = (k - 1 + nz) % nz;
        const int kp = (k + 1) % nz;
        const double dp1_dz = (static_cast<double>(psi1[i + nx * (j + ny * kp)]) -
                               static_cast<double>(psi1[i + nx * (j + ny * km)])) /
                              (2.0 * dz);

        // Gradient of psi2
        const double p2_c = static_cast<double>(psi2[c]);

        double dp2_dx;
        if (i == 0)
            dp2_dx = (static_cast<double>(psi2[c + 1]) - p2_c) / dx;
        else if (i == nx - 1)
            dp2_dx = (p2_c - static_cast<double>(psi2[c - 1])) / dx;
        else
            dp2_dx =
                (static_cast<double>(psi2[c + 1]) - static_cast<double>(psi2[c - 1])) / (2.0 * dx);

        // d/dy (periodic, no lifting for psi2)
        const double dp2_dy = (static_cast<double>(psi2[i + nx * (jp + ny * k)]) -
                               static_cast<double>(psi2[i + nx * (jm + ny * k)])) /
                              (2.0 * dy);

        // d/dz (periodic with lifting for psi2, period Lz)
        double p2_km = static_cast<double>(psi2[i + nx * (j + ny * km)]);
        double p2_kp = static_cast<double>(psi2[i + nx * (j + ny * kp)]);
        p2_km += Lz * round((p2_c - p2_km) / Lz);
        p2_kp += Lz * round((p2_c - p2_kp) / Lz);
        const double dp2_dz = (p2_kp - p2_km) / (2.0 * dz);

        // Residuals
        const double r1 = vx * dp1_dx + vy * dp1_dy + vz * dp1_dz;
        const double r2 = vx * dp2_dx + vy * dp2_dy + vz * dp2_dz;

        s_ssq1[tid] = r1 * r1;
        s_ssq2[tid] = r2 * r2;
        s_max1[tid] = fabs(r1);
        s_max2[tid] = fabs(r2);
    }
    __syncthreads();

    // Block reduction
    for (int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_ssq1[tid] += s_ssq1[tid + stride];
            s_ssq2[tid] += s_ssq2[tid + stride];
            if (s_max1[tid + stride] > s_max1[tid])
                s_max1[tid] = s_max1[tid + stride];
            if (s_max2[tid + stride] > s_max2[tid])
                s_max2[tid] = s_max2[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomic_add_d(&out[0], s_ssq1[0]);
        atomic_add_d(&out[1], s_ssq2[0]);
        atomic_max_d(&out[2], s_max1[0]);
        atomic_max_d(&out[3], s_max2[0]);
        int block_count = min(bs, total - (int)(blockIdx.x * bs));
        atomic_add_d(&out[7], static_cast<double>(block_count));
    }
}

// ============================================================================
// PsptaInvariantField implementation
// ============================================================================

void PsptaInvariantField::resize(const Grid3D& grid) {
    nx_ = grid.nx;
    ny_ = grid.ny;
    nz_ = grid.nz;
    dx_ = static_cast<double>(grid.dx);
    dy_ = static_cast<double>(grid.dy);
    dz_ = static_cast<double>(grid.dz);

    const size_t n = num_cells();
    d_psi1_.resize(n);
    d_psi2_.resize(n);

    // Invalidate cached derivatives
    has_cached_derivs_ = false;
    quality_.valid = false;
}

void PsptaInvariantField::clear() {
    nx_ = ny_ = nz_ = 0;
    dx_ = dy_ = dz_ = 0.0;
    d_psi1_.reset();
    d_psi2_.reset();
    d_dpsi1_dx_.reset();
    d_dpsi1_dy_.reset();
    d_dpsi1_dz_.reset();
    d_dpsi2_dx_.reset();
    d_dpsi2_dy_.reset();
    d_dpsi2_dz_.reset();
    has_cached_derivs_ = false;
    quality_ = InvariantQualityReport{};
    construction_info_ = InvariantConstructionInfo{};
}

void PsptaInvariantField::ensure_cached_derivatives(const VelocityField& vel, cudaStream_t stream) {
    if (has_cached_derivs_)
        return;
    if (!is_valid())
        return;

    const size_t n = num_cells();
    d_dpsi1_dx_.resize(n);
    d_dpsi1_dy_.resize(n);
    d_dpsi1_dz_.resize(n);
    d_dpsi2_dx_.resize(n);
    d_dpsi2_dy_.resize(n);
    d_dpsi2_dz_.resize(n);

    const int block = 256;
    const int grid_k = (static_cast<int>(n) + block - 1) / block;

    // psi1 x-derivative
    kernel_compute_psi_dx<<<grid_k, block, 0, stream>>>(d_psi1_.data(), d_dpsi1_dx_.data(), nx_,
                                                        ny_, nz_, dx_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // psi2 x-derivative
    kernel_compute_psi_dx<<<grid_k, block, 0, stream>>>(d_psi2_.data(), d_dpsi2_dx_.data(), nx_,
                                                        ny_, nz_, dx_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // psi1 y,z derivatives (self-period = Ly)
    kernel_compute_psi_derivatives<<<grid_k, block, 0, stream>>>(
        d_psi1_.data(), d_dpsi1_dy_.data(), d_dpsi1_dz_.data(), nx_, ny_, nz_, dy_, dz_, Ly());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // psi2 y,z derivatives (self-period = Lz)
    kernel_compute_psi_derivatives<<<grid_k, block, 0, stream>>>(
        d_psi2_.data(), d_dpsi2_dy_.data(), d_dpsi2_dz_.data(), nx_, ny_, nz_, dy_, dz_, Lz());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));
    has_cached_derivs_ = true;
}

InvariantQualityReport PsptaInvariantField::compute_quality(const VelocityField& vel,
                                                            cudaStream_t stream) const {
    InvariantQualityReport report;
    if (!is_valid())
        return report;

    const size_t n = num_cells();

    // Ensure scratch buffer (8 doubles)
    if (d_quality_scratch_.size() < 8) {
        d_quality_scratch_.resize(8);
    }
    MACROFLOW3D_CUDA_CHECK(
        cudaMemsetAsync(d_quality_scratch_.data(), 0, 8 * sizeof(double), stream));

    const int block = 256;
    const int grid_k = (static_cast<int>(n) + block - 1) / block;
    const size_t smem = 4 * block * sizeof(double);

    kernel_invariance_quality_reduce<<<grid_k, block, smem, stream>>>(
        d_psi1_.data(), d_psi2_.data(), vel.U.data(), vel.V.data(), vel.W.data(), nx_, ny_, nz_,
        dx_, dy_, dz_, Ly(), Lz(), d_quality_scratch_.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    double host_buf[8] = {};
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(host_buf, d_quality_scratch_.data(), 8 * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));

    const double count = host_buf[7];
    report.invariance.n_cells = static_cast<long long>(count);
    if (count > 0.0) {
        report.invariance.rms_r1 = std::sqrt(host_buf[0] / count);
        report.invariance.rms_r2 = std::sqrt(host_buf[1] / count);
    }
    report.invariance.max_r1 = host_buf[2];
    report.invariance.max_r2 = host_buf[3];

    // TODO: Implement cross_product and independence quality metrics
    // For now, leave them at defaults

    report.valid = true;
    quality_ = report;
    return report;
}

// ============================================================================
// Kernel: cast float64 → float32 (element-wise)
// ============================================================================

__global__ void kernel_cast_f64_to_f32(const double* __restrict__ src, float* __restrict__ dst,
                                       int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n)
        dst[c] = static_cast<float>(src[c]);
}

// ============================================================================
// Strategy A eigenvector ingestion
// ============================================================================

void PsptaInvariantField::ingest_eigenvectors(const DeviceBuffer<real>& ev1,
                                              const DeviceBuffer<real>& ev2,
                                              const EigensolverResult& result, double mu,
                                              const std::string& backend_name, CudaContext& ctx,
                                              cudaStream_t stream) {
    const int n = static_cast<int>(num_cells());

    // ── Modal quality diagnostics (f64 precision, before cast) ─────────
    {
        blas::ReductionWorkspace red;

        DeviceSpan<const real> s1(ev1.data(), static_cast<size_t>(n));
        DeviceSpan<const real> s2(ev2.data(), static_cast<size_t>(n));

        const double nrm1 = blas::nrm2_host(ctx, s1, red);
        const double nrm2 = blas::nrm2_host(ctx, s2, red);
        const double dot12 = blas::dot_host(ctx, s1, s2, red);

        modal_quality_ = ModalQualityReport{};
        modal_quality_.n_modes = 2;
        modal_quality_.eigenvalues = result.eigenvalues;
        modal_quality_.residual_norms = result.residual_norms;
        modal_quality_.l2_norms = {nrm1, nrm2};

        const double denom = nrm1 * nrm2;
        modal_quality_.orthogonality = (denom > 1e-30) ? std::fabs(dot12) / denom : 0.0;

        // Gauge-readiness: modes are normalized, orthogonal, and have small
        // residuals
        const bool norms_ok = (nrm1 > 0.1 && nrm2 > 0.1);
        const bool ortho_ok = (modal_quality_.orthogonality < 0.1);
        const bool resid_ok = result.residual_norms.size() >= 2 &&
                              result.residual_norms[0] < 1e-3 && result.residual_norms[1] < 1e-3;
        modal_quality_.gauge_ready = norms_ok && ortho_ok && resid_ok;

        std::printf("  [ModalQuality] ||ψ₁||=%.6f  ||ψ₂||=%.6f  "
                    "|<ψ₁,ψ₂>|/(||ψ₁||·||ψ₂||)=%.2e  gauge_ready=%s\n",
                    nrm1, nrm2, modal_quality_.orthogonality,
                    modal_quality_.gauge_ready ? "YES" : "NO");
    }

    // ── Cast f64 → f32 ────────────────────────────────────────────────
    const int block = 256;
    const int grid_k = (n + block - 1) / block;

    kernel_cast_f64_to_f32<<<grid_k, block, 0, stream>>>(ev1.data(), d_psi1_.data(), n);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    kernel_cast_f64_to_f32<<<grid_k, block, 0, stream>>>(ev2.data(), d_psi2_.data(), n);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Populate construction metadata
    construction_info_ = InvariantConstructionInfo{};
    construction_info_.method = InvariantConstructionMethod::StrategyA;
    construction_info_.mu = mu;
    construction_info_.eigensolver_backend = backend_name;
    construction_info_.eigenvalues = result.eigenvalues;
    construction_info_.n_eigenvectors_computed = result.n_converged;
    construction_info_.eigensolver_iterations = result.iterations;
    if (!result.residual_norms.empty())
        construction_info_.eigensolver_residual = result.residual_norms.back();
    construction_info_.construction_time_ms = result.elapsed_ms;

    // Invalidate cached derivatives — they must be recomputed
    has_cached_derivs_ = false;
    quality_.valid = false;
}

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
