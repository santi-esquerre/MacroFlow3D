#pragma once

/**
 * @file stochastic.cuh
 * @brief Stochastic K field generation (lognormal) - API
 * 
 * Port of legacy/random_field_generation.cu using Randomized Spectral Method.
 * Does NOT use FFT - direct sum of Fourier modes.
 * 
 * Reference:
 *   Räss, Kolyukhin, Minakov (2019), Comp. & Geosci. 131, 158-169
 *   DOI: 10.1016/j.cageo.2019.06.007
 * 
 * Conventions (matching legacy exactly):
 *   - sigma_f = sqrt(sigma2) is the std dev of log-K
 *   - logK = (sigma_f / sqrt(n_modes)) * Σᵢ (aᵢ sin(k·x) + bᵢ cos(k·x))
 *   - K = exp(logK)  (no mean correction in legacy)
 *   - Cell-centered coordinates: x = h * (ix + 0.5, iy + 0.5, iz + 0.5)
 */

#include "../common/physics_config.hpp"
#include "../common/workspaces.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../runtime/CudaContext.cuh"

namespace rwpt {
namespace physics {

/**
 * @brief Initialize RNG states for stochastic field generation
 * 
 * Must be called once before generate_gaussian_field.
 * Uses deterministic seeding: state[i] = curand_init(base_seed + i, i, 0)
 * 
 * @param workspace Pre-allocated stochastic workspace
 * @param seed      Base seed for RNG
 * @param ctx       CUDA context (stream)
 */
void init_stochastic_rng(StochasticWorkspace& workspace,
                         uint64_t seed,
                         const CudaContext& ctx);

/**
 * @brief Generate correlated Gaussian random field G (logK)
 * 
 * Two-stage process:
 *   1. Generate random Fourier mode coefficients (k1,k2,k3,a,b)
 *   2. Evaluate sum at each grid cell
 * 
 * Output is stored in workspace.logK
 * 
 * @param workspace  Contains RNG states and mode coefficient buffers
 * @param grid       Grid specification (nx, ny, nz, dx)
 * @param cfg        Stochastic config (sigma2, corr_length, n_modes, covariance_type)
 * @param ctx        CUDA context (stream)
 */
void generate_gaussian_field(StochasticWorkspace& workspace,
                             const Grid3D& grid,
                             const StochasticConfig& cfg,
                             const CudaContext& ctx);

/**
 * @brief Transform Gaussian field to lognormal K
 * 
 * K = exp(logK) following legacy convention (no mean shift).
 * 
 * @param K          Output: lognormal K field (device memory, size = num_cells)
 * @param logK       Input: Gaussian field from workspace.logK
 * @param grid       Grid specification
 * @param cfg        Config (currently unused but reserved for normalization options)
 * @param ctx        CUDA context
 */
void generate_K_lognormal(DeviceSpan<real> K,
                          DeviceSpan<const real> logK,
                          const Grid3D& grid,
                          const StochasticConfig& cfg,
                          const CudaContext& ctx);

/**
 * @brief Convenience: generate lognormal K in one call
 * 
 * Combines init_stochastic_rng + generate_gaussian_field + generate_K_lognormal.
 * Use when you don't need access to intermediate logK.
 * 
 * @param K          Output: lognormal K field
 * @param workspace  Stochastic workspace
 * @param grid       Grid
 * @param cfg        Config
 * @param ctx        CUDA context
 */
void generate_K_field(DeviceSpan<real> K,
                      StochasticWorkspace& workspace,
                      const Grid3D& grid,
                      const StochasticConfig& cfg,
                      const CudaContext& ctx);

/**
 * @brief Compute basic statistics of a device array (for diagnostics)
 * 
 * @param data       Device data
 * @param n          Number of elements
 * @param min_val    Output: minimum
 * @param max_val    Output: maximum
 * @param mean_val   Output: arithmetic mean
 * @param ctx        CUDA context
 */
void compute_field_stats(DeviceSpan<const real> data,
                         real& min_val, real& max_val, real& mean_val,
                         const CudaContext& ctx);

} // namespace physics
} // namespace rwpt
