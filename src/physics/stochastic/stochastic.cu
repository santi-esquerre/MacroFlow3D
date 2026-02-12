/**
 * @file stochastic.cu
 * @brief Stochastic K field generation - Implementation
 * 
 * Direct port of legacy/random_field_generation.cu
 * Randomized Spectral Method (no FFT)
 */

#include "stochastic.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cmath>
#include <curand_kernel.h>
#include <vector>

namespace macroflow3d {
namespace physics {

// ============================================================================
// Constants (matching legacy)
// ============================================================================

// Use constexpr for compile-time constant
static constexpr double PI_D = 3.141592653589793238462643383279502884;

// ============================================================================
// Kernel: Initialize RNG states
// ============================================================================

/**
 * @brief Setup curand states with deterministic seeding
 * 
 * Legacy: curand_init(ix, ix, 0, &state[ix])
 * Modified for reproducibility: curand_init(base_seed + ix, ix, 0, &state[ix])
 */
__global__ void kernel_init_rng(curandState* __restrict__ states,
                                const uint64_t base_seed,
                                const int n_modes) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= n_modes) return;
    
    // Use base_seed + ix as seed, ix as sequence, 0 as offset
    // This ensures reproducibility: same seed → same sequence
    curand_init(base_seed + ix, ix, 0, &states[ix]);
}

// ============================================================================
// Kernel: Generate Fourier mode coefficients (exponential covariance)
// ============================================================================

/**
 * @brief Generate wavenumbers for EXPONENTIAL covariance
 * 
 * Legacy: random_kernel_3D()
 * Uses Cauchy-like distribution via rejection sampling
 */
__global__ void kernel_random_modes_exp(curandState* __restrict__ states,
                                        real* __restrict__ V1,
                                        real* __restrict__ V2,
                                        real* __restrict__ V3,
                                        real* __restrict__ a,
                                        real* __restrict__ b,
                                        const real lambda,
                                        const int n_modes) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= n_modes) return;
    
    curandState localState = states[ix];
    
    // Spherical angles for direction
    double fi = 2.0 * PI_D * curand_uniform_double(&localState);
    double theta = acos(1.0 - 2.0 * curand_uniform_double(&localState));
    
    // Wavenumber magnitude k from modified Cauchy distribution (rejection sampling)
    double k, d;
    int flag = 1;
    while (flag == 1) {
        k = tan(PI_D * 0.5 * curand_uniform_double(&localState));
        d = (k * k) / (1.0 + k * k);
        if (curand_uniform_double(&localState) < d) flag = 0;
    }
    
    // Wavenumber vector components (legacy: divide by lambda)
    V1[ix] = static_cast<real>(k * sin(fi) * sin(theta) / lambda);
    V2[ix] = static_cast<real>(k * cos(fi) * sin(theta) / lambda);
    V3[ix] = static_cast<real>(k * cos(theta) / lambda);
    
    // Fourier coefficients a, b ~ N(0,1) via Box-Muller
    double u1 = curand_uniform_double(&localState);
    double u2 = curand_uniform_double(&localState);
    a[ix] = static_cast<real>(sqrt(-2.0 * log(u1)) * cos(2.0 * PI_D * u2));
    
    u1 = curand_uniform_double(&localState);
    u2 = curand_uniform_double(&localState);
    b[ix] = static_cast<real>(sqrt(-2.0 * log(u1)) * cos(2.0 * PI_D * u2));
    
    states[ix] = localState;
}

// ============================================================================
// Kernel: Generate Fourier mode coefficients (Gaussian covariance)
// ============================================================================

/**
 * @brief Generate wavenumbers for GAUSSIAN covariance
 * 
 * Legacy: random_kernel_3D_gauss()
 * Uses rejection sampling with k² exp(-0.5 k²) envelope
 */
__global__ void kernel_random_modes_gauss(curandState* __restrict__ states,
                                          real* __restrict__ V1,
                                          real* __restrict__ V2,
                                          real* __restrict__ V3,
                                          real* __restrict__ a,
                                          real* __restrict__ b,
                                          const real lambda,
                                          const int n_modes,
                                          const int k_max) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= n_modes) return;
    
    curandState localState = states[ix];
    
    // Spherical angles for direction
    double fi = 2.0 * PI_D * curand_uniform_double(&localState);
    double theta = acos(1.0 - 2.0 * curand_uniform_double(&localState));
    
    // Wavenumber magnitude k via rejection sampling
    double k, d;
    int flag = 1;
    while (flag == 1) {
        k = k_max * curand_uniform_double(&localState);
        d = k * k * exp(-0.5 * k * k);
        if (curand_uniform_double(&localState) * 2.0 * exp(-1.0) < d) flag = 0;
    }
    
    // Scale k (legacy formula)
    k = k / (2.0 * lambda / sqrt(PI_D)) * sqrt(2.0);
    
    // Wavenumber vector components (no additional lambda division here)
    V1[ix] = static_cast<real>(k * sin(fi) * sin(theta));
    V2[ix] = static_cast<real>(k * cos(fi) * sin(theta));
    V3[ix] = static_cast<real>(k * cos(theta));
    
    // Fourier coefficients a, b ~ N(0,1) via Box-Muller
    double u1 = curand_uniform_double(&localState);
    double u2 = curand_uniform_double(&localState);
    a[ix] = static_cast<real>(sqrt(-2.0 * log(u1)) * cos(2.0 * PI_D * u2));
    
    u1 = curand_uniform_double(&localState);
    u2 = curand_uniform_double(&localState);
    b[ix] = static_cast<real>(sqrt(-2.0 * log(u1)) * cos(2.0 * PI_D * u2));
    
    states[ix] = localState;
}

// ============================================================================
// Kernel: Evaluate Gaussian field at all grid points
// ============================================================================

/**
 * @brief Compute logK at each cell via spectral sum
 * 
 * Legacy: conductivity_kernel_3D_logK()
 * logK[idx] = (sigma_f / sqrt(n_modes)) * Σᵢ (aᵢ sin(k·x) + bᵢ cos(k·x))
 * 
 * Cell-centered coordinates: x = h * (ix + 0.5, iy + 0.5, iz + 0.5)
 */
__global__ void kernel_eval_logK(const real* __restrict__ V1,
                                 const real* __restrict__ V2,
                                 const real* __restrict__ V3,
                                 const real* __restrict__ a,
                                 const real* __restrict__ b,
                                 const int n_modes,
                                 real* __restrict__ logK,
                                 const real h,
                                 const int nx, const int ny, const int nz,
                                 const real sigma_f) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (ix >= nx || iy >= ny || iz >= nz) return;
    
    const int idx = ix + iy * nx + iz * nx * ny;
    
    // Cell-centered position
    const real x = h * (static_cast<real>(ix) + static_cast<real>(0.5));
    const real y = h * (static_cast<real>(iy) + static_cast<real>(0.5));
    const real z = h * (static_cast<real>(iz) + static_cast<real>(0.5));
    
    // Sum over all modes
    real sum = static_cast<real>(0.0);
    for (int i = 0; i < n_modes; ++i) {
        const real phase = V1[i] * x + V2[i] * y + V3[i] * z;
        sum += a[i] * sin(phase) + b[i] * cos(phase);
    }
    
    // Scale by sigma_f / sqrt(n_modes) per legacy
    logK[idx] = (sigma_f / sqrt(static_cast<real>(n_modes))) * sum;
}

// ============================================================================
// Kernel: Transform logK → K = exp(logK)
// ============================================================================

__global__ void kernel_exp(real* __restrict__ K,
                           const real* __restrict__ logK,
                           const size_t n) {
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    K[idx] = exp(logK[idx]);
}

// ============================================================================
// Kernel: Stats reduction helpers (min/max/sum)
// ============================================================================

__global__ void kernel_minmax_sum(const real* __restrict__ data,
                                  const size_t n,
                                  real* __restrict__ block_mins,
                                  real* __restrict__ block_maxs,
                                  real* __restrict__ block_sums) {
    extern __shared__ char smem[];
    real* s_min = reinterpret_cast<real*>(smem);
    real* s_max = s_min + blockDim.x;
    real* s_sum = s_max + blockDim.x;
    
    const size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize
    real local_min = (idx < n) ? data[idx] : real(1e30);
    real local_max = (idx < n) ? data[idx] : real(-1e30);
    real local_sum = (idx < n) ? data[idx] : real(0);
    
    // Grid-stride loop
    idx += blockDim.x * gridDim.x;
    while (idx < n) {
        real val = data[idx];
        local_min = fmin(local_min, val);
        local_max = fmax(local_max, val);
        local_sum += val;
        idx += blockDim.x * gridDim.x;
    }
    
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    s_sum[tid] = local_sum;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fmin(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_mins[blockIdx.x] = s_min[0];
        block_maxs[blockIdx.x] = s_max[0];
        block_sums[blockIdx.x] = s_sum[0];
    }
}

// ============================================================================
// Host API implementations
// ============================================================================

void init_stochastic_rng(StochasticWorkspace& workspace,
                         uint64_t seed,
                         const CudaContext& ctx) {
    if (workspace.n_modes <= 0) {
        throw std::runtime_error("StochasticWorkspace not allocated");
    }
    
    const int block = 256;
    const int grid = (workspace.n_modes + block - 1) / block;
    
    kernel_init_rng<<<grid, block, 0, ctx.cuda_stream()>>>(
        workspace.rng_states.data(),
        seed,
        workspace.n_modes);
    
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void generate_gaussian_field(StochasticWorkspace& workspace,
                             const Grid3D& grid,
                             const StochasticConfig& cfg,
                             const CudaContext& ctx) {
    if (!workspace.is_allocated()) {
        throw std::runtime_error("StochasticWorkspace not allocated");
    }
    
    const int n_modes = cfg.n_modes;
    const real lambda = cfg.corr_length;
    const real sigma_f = sqrt(cfg.sigma2);  // Standard deviation of log-K
    
    // === Stage 1: Generate random Fourier modes ===
    {
        const int block = 256;
        const int modes_grid = (n_modes + block - 1) / block;
        
        if (cfg.covariance_type == 0) {
            // Exponential covariance
            kernel_random_modes_exp<<<modes_grid, block, 0, ctx.cuda_stream()>>>(
                workspace.rng_states.data(),
                workspace.k1.data(),
                workspace.k2.data(),
                workspace.k3.data(),
                workspace.coef_a.data(),
                workspace.coef_b.data(),
                lambda,
                n_modes);
        } else {
            // Gaussian covariance (k_max = 100 per legacy)
            const int k_max = 100;
            kernel_random_modes_gauss<<<modes_grid, block, 0, ctx.cuda_stream()>>>(
                workspace.rng_states.data(),
                workspace.k1.data(),
                workspace.k2.data(),
                workspace.k3.data(),
                workspace.coef_a.data(),
                workspace.coef_b.data(),
                lambda,
                n_modes,
                k_max);
        }
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    
    // === Stage 2: Evaluate spectral sum at each grid cell ===
    {
        // Use 8x8x8 block for 3D kernel (matches legacy pattern)
        dim3 block(8, 8, 8);
        dim3 grid_dim((grid.nx + block.x - 1) / block.x,
                      (grid.ny + block.y - 1) / block.y,
                      (grid.nz + block.z - 1) / block.z);
        
        kernel_eval_logK<<<grid_dim, block, 0, ctx.cuda_stream()>>>(
            workspace.k1.data(),
            workspace.k2.data(),
            workspace.k3.data(),
            workspace.coef_a.data(),
            workspace.coef_b.data(),
            n_modes,
            workspace.logK.data(),
            grid.dx,
            grid.nx, grid.ny, grid.nz,
            sigma_f);
        
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
}

void generate_K_lognormal(DeviceSpan<real> K,
                          DeviceSpan<const real> logK,
                          const Grid3D& grid,
                          const StochasticConfig& cfg,
                          const CudaContext& ctx) {
    const size_t n = grid.num_cells();
    if (K.size() < n || logK.size() < n) {
        throw std::runtime_error("K or logK buffer too small");
    }
    
    const int block = 256;
    const int grid_1d = (n + block - 1) / block;
    
    // K = exp(logK) — legacy convention (no mean shift)
    kernel_exp<<<grid_1d, block, 0, ctx.cuda_stream()>>>(
        K.data(),
        const_cast<real*>(logK.data()),  // DeviceSpan<const real> workaround
        n);
    
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void generate_K_field(DeviceSpan<real> K,
                      StochasticWorkspace& workspace,
                      const Grid3D& grid,
                      const StochasticConfig& cfg,
                      const CudaContext& ctx) {
    // 1. Initialize RNG (if not already done, do it with cfg.seed)
    init_stochastic_rng(workspace, cfg.seed, ctx);
    
    // 2. Generate Gaussian field
    generate_gaussian_field(workspace, grid, cfg, ctx);
    
    // 3. Transform to lognormal
    generate_K_lognormal(K, DeviceSpan<const real>(workspace.logK.data(), workspace.n_cells), 
                         grid, cfg, ctx);
}

void compute_field_stats(DeviceSpan<const real> data,
                         real& min_val, real& max_val, real& mean_val,
                         const CudaContext& ctx) {
    const size_t n = data.size();
    if (n == 0) {
        min_val = max_val = mean_val = 0;
        return;
    }
    
    // Use moderate number of blocks
    const int block = 256;
    const int n_blocks = std::min(256, static_cast<int>((n + block - 1) / block));
    
    // Allocate temporary device buffers for block results
    DeviceBuffer<real> d_mins(n_blocks);
    DeviceBuffer<real> d_maxs(n_blocks);
    DeviceBuffer<real> d_sums(n_blocks);
    
    const size_t smem_size = 3 * block * sizeof(real);
    
    kernel_minmax_sum<<<n_blocks, block, smem_size, ctx.cuda_stream()>>>(
        const_cast<real*>(data.data()),
        n,
        d_mins.data(),
        d_maxs.data(),
        d_sums.data());
    
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    
    // Copy block results to host and finalize
    std::vector<real> h_mins(n_blocks), h_maxs(n_blocks), h_sums(n_blocks);
    
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_mins.data(), d_mins.data(), n_blocks * sizeof(real),
                               cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_maxs.data(), d_maxs.data(), n_blocks * sizeof(real),
                               cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_sums.data(), d_sums.data(), n_blocks * sizeof(real),
                               cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    
    ctx.synchronize();
    
    min_val = h_mins[0];
    max_val = h_maxs[0];
    real sum = h_sums[0];
    for (int i = 1; i < n_blocks; ++i) {
        min_val = std::min(min_val, h_mins[i]);
        max_val = std::max(max_val, h_maxs[i]);
        sum += h_sums[i];
    }
    mean_val = sum / static_cast<real>(n);
}

} // namespace physics
} // namespace macroflow3d
