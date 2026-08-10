/**
 * @file PeriodicGaussianField.cu
 * @brief Truly torus-periodic Gaussian-covariance log-conductivity generator - implementation
 *
 * See PeriodicGaussianField.cuh for the full mathematical specification
 * (covariance/spectrum identity, DFT convention, zero mode, Nyquist zeroing,
 * canonical Hermitian rule, seed-to-mode hash, half-cell twist, and
 * normalization semantics).
 */

#include "../../runtime/cuda_check.cuh"
#include "PeriodicGaussianField.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace physics {

namespace {

// ============================================================================
// cuFFT error checking (local to this file; mirrors MACROFLOW3D_CUDA_CHECK /
// MACROFLOW3D_CUBLAS_CHECK in runtime/cuda_check.cuh)
// ============================================================================

inline void cufft_check_impl(cufftResult status, const char* file, int line) {
    if (status != CUFFT_SUCCESS) {
        std::string msg = std::string("cuFFT error at ") + file + ":" + std::to_string(line) +
                          " - code " + std::to_string(static_cast<int>(status));
        throw std::runtime_error(msg);
    }
}

#define MF3D_CUFFT_CHECK(expr) ::macroflow3d::physics::cufft_check_impl((expr), __FILE__, __LINE__)

static constexpr double PI_D = 3.141592653589793238462643383279502884;

// ============================================================================
// Stateless counter-based hash -> Box-Muller pair (see .cuh section 5)
// ============================================================================

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

__device__ __forceinline__ uint64_t hash_mode(unsigned long long seed, int m1, int m2, int m3,
                                              uint32_t salt) {
    uint64_t h = seed ^ 0x9E3779B97F4A7C15ULL;
    h = splitmix64(h ^ (static_cast<uint64_t>(static_cast<int64_t>(m1)) * 0xD1B54A32D192ED03ULL));
    h = splitmix64(h ^ (static_cast<uint64_t>(static_cast<int64_t>(m2)) * 0xA24BAED4963EE407ULL));
    h = splitmix64(h ^ (static_cast<uint64_t>(static_cast<int64_t>(m3)) * 0x9FB21C651E98DF25ULL));
    h = splitmix64(h ^ static_cast<uint64_t>(salt));
    return h;
}

// Maps a 64-bit hash to a uniform double strictly inside (0,1).
__device__ __forceinline__ real uint64_to_open_unit_double(uint64_t x) {
    const uint64_t mantissa = (x >> 11) + 1ULL;          // in [1, 2^53]
    const real denom = static_cast<real>(1ULL << 53) + 1.0; // > 2^53, so ratio < 1
    return static_cast<real>(mantissa) / denom;
}

__device__ __forceinline__ void box_muller_from_hash(unsigned long long seed, int m1, int m2,
                                                      int m3, real& re_std, real& im_std) {
    const uint64_t h1 = hash_mode(seed, m1, m2, m3, 0u);
    const uint64_t h2 = hash_mode(seed, m1, m2, m3, 1u);
    const real u1 = uint64_to_open_unit_double(h1);
    const real u2 = uint64_to_open_unit_double(h2);
    const real r = sqrt(static_cast<real>(-2.0) * log(u1));
    const real theta = static_cast<real>(2.0 * PI_D) * u2;
    re_std = r * cos(theta);
    im_std = r * sin(theta);
}

// ============================================================================
// Kernel: fill half-spectrum from the stateless hash (spectrum + twist)
// ============================================================================

__global__ void kernel_fill_spectrum(cufftDoubleComplex* __restrict__ spec, const int nx,
                                     const int ny, const int nz, const int hx, const real Lx,
                                     const real Ly, const real Lz, const real dx, const real dy,
                                     const real dz, const real sigma2, const real l,
                                     const unsigned long long seed,
                                     unsigned long long* __restrict__ active_count) {
    const int gx = threadIdx.x + blockIdx.x * blockDim.x;
    const int gy = threadIdx.y + blockIdx.y * blockDim.y;
    const int gz = threadIdx.z + blockIdx.z * blockDim.z;
    if (gx >= hx || gy >= ny || gz >= nz)
        return;

    const size_t idx = static_cast<size_t>(gx) +
                       static_cast<size_t>(hx) *
                           (static_cast<size_t>(gy) + static_cast<size_t>(ny) * static_cast<size_t>(gz));

    const int mx = gx; // stored non-negative x-frequency, mx in [0, nx/2]
    const int my_s = (gy < ny / 2) ? gy : gy - ny;
    const int mz_s = (gz < nz / 2) ? gz : gz - nz;

    const bool nyquist = (mx == nx / 2) || (gy == ny / 2) || (gz == nz / 2);
    const bool zero_mode = (mx == 0 && my_s == 0 && mz_s == 0);

    if (nyquist || zero_mode) {
        spec[idx] = make_cuDoubleComplex(0.0, 0.0);
        return;
    }

    // Canonical Hermitian representative for the self-referential mx==0 plane.
    int cmy = my_s;
    int cmz = mz_s;
    bool conj_flag = false;
    if (mx == 0) {
        const bool canonical = (my_s > 0) || (my_s == 0 && mz_s > 0);
        if (!canonical) {
            cmy = -my_s;
            cmz = -mz_s;
            conj_flag = true;
        }
    }

    real re_std = 0.0, im_std = 0.0;
    box_muller_from_hash(seed, mx, cmy, cmz, re_std, im_std);

    const real kx = static_cast<real>(2.0 * PI_D) * static_cast<real>(mx) / Lx;
    const real ky = static_cast<real>(2.0 * PI_D) * static_cast<real>(my_s) / Ly;
    const real kz = static_cast<real>(2.0 * PI_D) * static_cast<real>(mz_s) / Lz;
    const real k2 = kx * kx + ky * ky + kz * kz;

    // Periodicized Gaussian spectrum, E|Yhat_m|^2 (see .cuh section 2).
    const real S = sigma2 * pow(static_cast<real>(PI_D), static_cast<real>(1.5)) * l * l * l *
                   exp(-k2 * l * l * static_cast<real>(0.25)) / (Lx * Ly * Lz);
    const real amp = sqrt(S * static_cast<real>(0.5));

    real re = re_std * amp;
    real im = im_std * amp;
    if (conj_flag) {
        im = -im;
    }

    // Exact half-cell phase twist for cell-centered sampling (see .cuh section 7).
    const real phase = static_cast<real>(0.5) * (kx * dx + ky * dy + kz * dz);
    const real cw = cos(phase);
    const real sw = sin(phase);
    const real re2 = re * cw - im * sw;
    const real im2 = re * sw + im * cw;

    spec[idx] = make_cuDoubleComplex(re2, im2);
    atomicAdd(active_count, 1ULL);
}

// ============================================================================
// Kernels: deterministic two-pass mean/variance reduction
// ============================================================================

__global__ void kernel_block_sum(const real* __restrict__ data, const size_t n,
                                 real* __restrict__ block_sums) {
    extern __shared__ char smem_raw[];
    real* s_sum = reinterpret_cast<real*>(smem_raw);

    const size_t tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    real local = (idx < n) ? data[idx] : static_cast<real>(0.0);
    idx += static_cast<size_t>(blockDim.x) * gridDim.x;
    while (idx < n) {
        local += data[idx];
        idx += static_cast<size_t>(blockDim.x) * gridDim.x;
    }

    s_sum[tid] = local;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = s_sum[0];
    }
}

__global__ void kernel_block_sumsq_dev(const real* __restrict__ data, const size_t n,
                                       const real mean, real* __restrict__ block_sums) {
    extern __shared__ char smem_raw[];
    real* s_sum = reinterpret_cast<real*>(smem_raw);

    const size_t tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    real local = static_cast<real>(0.0);
    if (idx < n) {
        const real d = data[idx] - mean;
        local = d * d;
    }
    idx += static_cast<size_t>(blockDim.x) * gridDim.x;
    while (idx < n) {
        const real d = data[idx] - mean;
        local += d * d;
        idx += static_cast<size_t>(blockDim.x) * gridDim.x;
    }

    s_sum[tid] = local;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = s_sum[0];
    }
}

__global__ void kernel_scale_inplace(real* __restrict__ y, const size_t n, const real scale) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    y[idx] *= scale;
}

} // namespace

// ============================================================================
// Host API implementations
// ============================================================================

void PeriodicGaussianFieldConfig::validate(const Grid3D& grid) const {
    if (!std::isfinite(sigma2) || sigma2 < 0.0) {
        throw std::invalid_argument(
            "PeriodicGaussianFieldConfig: sigma2 must be finite and >= 0");
    }
    if (!std::isfinite(corr_length) || corr_length <= 0.0) {
        throw std::invalid_argument(
            "PeriodicGaussianFieldConfig: corr_length must be finite and > 0");
    }
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        throw std::invalid_argument(
            "PeriodicGaussianFieldConfig: grid extents (nx,ny,nz) must be positive");
    }
    if ((grid.nx % 2) != 0 || (grid.ny % 2) != 0 || (grid.nz % 2) != 0) {
        throw std::invalid_argument(
            "PeriodicGaussianFieldConfig: grid extents (nx,ny,nz) must be even "
            "(required by the c2r half-spectrum layout and Nyquist-plane zeroing)");
    }
    if (!std::isfinite(grid.dx) || grid.dx <= 0.0 || !std::isfinite(grid.dy) || grid.dy <= 0.0 ||
        !std::isfinite(grid.dz) || grid.dz <= 0.0) {
        throw std::invalid_argument(
            "PeriodicGaussianFieldConfig: grid spacings (dx,dy,dz) must be finite and > 0");
    }
}

PeriodicGaussianFieldReport
generate_periodic_gaussian_field(const CudaContext& ctx, const Grid3D& grid,
                                 const PeriodicGaussianFieldConfig& cfg, DeviceSpan<real> y_out,
                                 PeriodicGaussianFieldWorkspace& workspace) {
    cfg.validate(grid);

    const size_t n = grid.num_cells();
    if (y_out.size() != n) {
        throw std::invalid_argument(
            "generate_periodic_gaussian_field: y_out size must equal grid.num_cells()");
    }

    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int hx = nx / 2 + 1;
    const size_t spectrum_count = static_cast<size_t>(hx) * static_cast<size_t>(ny) *
                                  static_cast<size_t>(nz);

    workspace.spectrum.resize(spectrum_count);
    workspace.mode_counter.resize(1);

    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(workspace.mode_counter.data(), 0,
                                           sizeof(unsigned long long), ctx.cuda_stream()));

    const real Lx = grid.Lx(), Ly = grid.Ly(), Lz = grid.Lz();

    // --- Stage 1: fill half-spectrum from the stateless hash ---
    {
        dim3 block(8, 8, 8);
        dim3 grid_dim(static_cast<unsigned int>((hx + block.x - 1) / block.x),
                      static_cast<unsigned int>((ny + block.y - 1) / block.y),
                      static_cast<unsigned int>((nz + block.z - 1) / block.z));
        kernel_fill_spectrum<<<grid_dim, block, 0, ctx.cuda_stream()>>>(
            workspace.spectrum.data(), nx, ny, nz, hx, Lx, Ly, Lz, grid.dx, grid.dy, grid.dz,
            cfg.sigma2, cfg.corr_length, cfg.seed, workspace.mode_counter.data());
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }

    unsigned long long active_mode_count_dev = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(&active_mode_count_dev, workspace.mode_counter.data(),
                                           sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                                           ctx.cuda_stream()));
    ctx.synchronize();

    // --- Stage 2: cuFFT inverse transform (Z2D), plan created per call (v1) ---
    size_t work_area_bytes = 0;
    {
        cufftHandle plan;
        // cuFFT row-major convention (last argument fastest) matches this
        // project's x-fastest field layout when called as (nz, ny, nx).
        MF3D_CUFFT_CHECK(cufftPlan3d(&plan, nz, ny, nx, CUFFT_Z2D));
        MF3D_CUFFT_CHECK(cufftSetStream(plan, ctx.cuda_stream()));
        MF3D_CUFFT_CHECK(cufftGetSize(plan, &work_area_bytes));

        MF3D_CUFFT_CHECK(cufftExecZ2D(plan, workspace.spectrum.data(), y_out.data()));

        ctx.synchronize();
        MF3D_CUFFT_CHECK(cufftDestroy(plan));
    }

    // --- Stage 3: deterministic two-pass mean/variance reduction ---
    const int block1d = 256;
    const int n_blocks = static_cast<int>(std::min<size_t>(256, (n + block1d - 1) / block1d));
    workspace.reduce_blocks.resize(static_cast<size_t>(n_blocks));

    std::vector<real> h_blocks(static_cast<size_t>(n_blocks));

    kernel_block_sum<<<n_blocks, block1d, static_cast<size_t>(block1d) * sizeof(real),
                       ctx.cuda_stream()>>>(y_out.data(), n, workspace.reduce_blocks.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_blocks.data(), workspace.reduce_blocks.data(),
                                           static_cast<size_t>(n_blocks) * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();

    real sum = 0.0;
    for (int i = 0; i < n_blocks; ++i) {
        sum += h_blocks[static_cast<size_t>(i)];
    }
    const real raw_mean = sum / static_cast<real>(n);

    kernel_block_sumsq_dev<<<n_blocks, block1d, static_cast<size_t>(block1d) * sizeof(real),
                             ctx.cuda_stream()>>>(y_out.data(), n, raw_mean,
                                                  workspace.reduce_blocks.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_blocks.data(), workspace.reduce_blocks.data(),
                                           static_cast<size_t>(n_blocks) * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();

    real sumsq = 0.0;
    for (int i = 0; i < n_blocks; ++i) {
        sumsq += h_blocks[static_cast<size_t>(i)];
    }
    const real raw_variance = sumsq / static_cast<real>(n);

    // --- Stage 4: optional uniform variance normalization ---
    real applied_scale = 1.0;
    real final_variance = raw_variance;
    if (cfg.normalize_variance) {
        if (!(raw_variance > 0.0)) {
            throw std::runtime_error(
                "generate_periodic_gaussian_field: raw_variance <= 0, cannot normalize "
                "(sigma2 == 0 or degenerate spectrum)");
        }
        applied_scale = std::sqrt(cfg.sigma2 / raw_variance);
        const int grid1d = static_cast<int>((n + static_cast<size_t>(block1d) - 1) /
                                            static_cast<size_t>(block1d));
        kernel_scale_inplace<<<grid1d, block1d, 0, ctx.cuda_stream()>>>(y_out.data(), n,
                                                                        applied_scale);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        final_variance = raw_variance * applied_scale * applied_scale;
        ctx.synchronize();
    }

    // --- Report: exact-byte accounting ---
    PeriodicGaussianFieldReport report;
    report.raw_mean = raw_mean;
    report.raw_variance = raw_variance;
    report.applied_scale = applied_scale;
    report.final_variance = final_variance;
    report.active_mode_count = static_cast<size_t>(active_mode_count_dev);

    report.spectrum_bytes = workspace.spectrum.capacity() * sizeof(cufftDoubleComplex);
    report.cufft_work_area_bytes = work_area_bytes;
    report.reduction_scratch_bytes = workspace.reduce_blocks.capacity() * sizeof(real) +
                                     workspace.mode_counter.capacity() *
                                         sizeof(unsigned long long);
    report.total_device_bytes =
        report.spectrum_bytes + report.cufft_work_area_bytes + report.reduction_scratch_bytes;

    return report;
}

} // namespace physics
} // namespace macroflow3d
