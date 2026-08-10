#include "AffinePeriodicFlowSolver.cuh"

#include "../../multigrid/coefficient_hierarchy.cuh"
#include "../../numerics/blas/axpy.cuh"
#include "../../numerics/blas/nrm2.cuh"
#include "../../numerics/constraints/MeanZeroProjector.cuh"
#include "../../numerics/operators/face_coefficient.cuh"
#include "../../numerics/operators/lester_positive_diffusion_operator.cuh"
#include "../../runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace macroflow3d {
namespace physics {
namespace {

constexpr int kBlock1D = 256;
constexpr int kMaxReduceBlocks = 256;
constexpr real kPi = real{3.14159265358979323846};

std::size_t compact_mac_u_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx + 1) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz);
}
std::size_t compact_mac_v_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny + 1) *
           static_cast<std::size_t>(grid.nz);
}
std::size_t compact_mac_w_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz + 1);
}

bool grids_equal(const Grid3D& a, const Grid3D& b) noexcept {
    return a.nx == b.nx && a.ny == b.ny && a.nz == b.nz && a.dx == b.dx && a.dy == b.dy &&
           a.dz == b.dz;
}

bool mg_config_equal(const multigrid::MGConfig& a, const multigrid::MGConfig& b) noexcept {
    return a.num_levels == b.num_levels && a.pre_smooth == b.pre_smooth &&
           a.post_smooth == b.post_smooth && a.coarse_solve_iters == b.coarse_solve_iters &&
           a.check_convergence_every == b.check_convergence_every && a.omega == b.omega &&
           a.verbose == b.verbose;
}

// Replicates multigrid::MGHierarchy(finest, num_levels)'s coarsening/break
// rule exactly (see src/multigrid/mg_types.hpp and the equivalent private
// helper in StreamfunctionWorkspace.cu), without allocating any device
// buffer, so validation cannot drift from the actual hierarchy prepare()
// constructs. This module's own copy is intentional: the streamfunctions
// version is a private, non-exported helper in another translation unit.
std::vector<Grid3D> simulate_mg_hierarchy_levels(const Grid3D& finest, int num_levels) {
    std::vector<Grid3D> levels;
    if (num_levels <= 0) {
        return levels;
    }
    levels.reserve(static_cast<std::size_t>(num_levels));
    Grid3D current = finest;
    for (int l = 0; l < num_levels; ++l) {
        levels.push_back(current);
        if (l < num_levels - 1) {
            current.nx = current.nx / 2;
            current.ny = current.ny / 2;
            current.nz = current.nz / 2;
            current.dx = current.dx * real{2};
            current.dy = current.dy * real{2};
            current.dz = current.dz * real{2};
            if (current.nx < 2 || current.ny < 2 || current.nz < 2) {
                break;
            }
        }
    }
    return levels;
}

bool grid_supports_mg_levels(const Grid3D& finest, const multigrid::MGConfig& mg_config) {
    if (mg_config.num_levels <= 0) {
        return false;
    }
    const auto levels = simulate_mg_hierarchy_levels(finest, mg_config.num_levels);
    if (static_cast<int>(levels.size()) != mg_config.num_levels) {
        return false;
    }
    for (const auto& level : levels) {
        if (level.nx < 2 || level.ny < 2 || level.nz < 2) {
            return false;
        }
        if (level.nx % 2 != 0 || level.ny % 2 != 0 || level.nz % 2 != 0) {
            return false;
        }
    }
    return true;
}

// --- Host validation (distinct messages; see the file header) ---

void require_valid_grid_geometry(const Grid3D& grid) {
    if (grid.nx < 2 || grid.ny < 2 || grid.nz < 2) {
        throw std::invalid_argument(
            "AffinePeriodicFlow requires every grid dimension to be at least 2");
    }
    if (!std::isfinite(grid.dx) || !std::isfinite(grid.dy) || !std::isfinite(grid.dz) ||
        grid.dx <= real{0} || grid.dy <= real{0} || grid.dz <= real{0}) {
        throw std::invalid_argument(
            "AffinePeriodicFlow requires finite, strictly positive grid spacing");
    }
    if (grid.dx != grid.dy || grid.dx != grid.dz) {
        // Inherited from the reused SF-06 affine-RHS assembler (its own
        // require_valid_grid) and from the reused MG hierarchy/operator
        // stack, both of which assume isotropic spacing throughout.
        throw std::invalid_argument(
            "AffinePeriodicFlow requires isotropic grid spacing (dx == dy == dz), inherited "
            "from the reused SF-06 affine-RHS assembler and the reused MG hierarchy/operator "
            "stack");
    }
}

void require_valid_mg_config(const multigrid::MGConfig& mg, const Grid3D& grid) {
    if (mg.pre_smooth < 1 || mg.post_smooth < 1 || mg.pre_smooth != mg.post_smooth ||
        mg.coarse_solve_iters <= 0) {
        throw std::invalid_argument(
            "AffinePeriodicFlow requires equal positive MG pre/post sweeps and a positive "
            "coarse solve iteration count");
    }
    if (!grid_supports_mg_levels(grid, mg)) {
        throw std::invalid_argument(
            "AffinePeriodicFlow grid does not support config.mg.num_levels under the exact "
            "multigrid::MGHierarchy coarsening/break rule and "
            "multigrid::validate_projected_positive_hierarchy's per-level even-extent "
            "requirements");
    }
}

void require_valid_linear_config(const solvers::ProjectedPCGConfig& linear) {
    if (linear.max_iter < 0 || linear.check_every <= 0 || linear.rtol < real{0} ||
        !std::isfinite(linear.rtol)) {
        throw std::invalid_argument(
            "AffinePeriodicFlow requires a non-negative linear.max_iter, a strictly positive "
            "linear.check_every, and a finite non-negative linear.rtol");
    }
}

void require_finite_qbar(const real qbar[3]) {
    if (!std::isfinite(qbar[0]) || !std::isfinite(qbar[1]) || !std::isfinite(qbar[2])) {
        throw std::invalid_argument("AffinePeriodicFlow requires finite config.qbar components");
    }
}

void require_valid_problem(const Grid3D& grid, DeviceSpan<const real> K,
                           const AffinePeriodicFlowConfig& config,
                           const AffinePeriodicVelocityView& velocity) {
    require_valid_grid_geometry(grid);
    require_valid_mg_config(config.mg, grid);
    require_valid_linear_config(config.linear);
    require_finite_qbar(config.qbar);

    const std::size_t n = grid.num_cells();
    if (K.size() != n || K.data() == nullptr) {
        throw std::invalid_argument(
            "AffinePeriodicFlow requires a conductivity span with exactly grid.num_cells() "
            "elements");
    }
    if (velocity.u.size() != compact_mac_u_size(grid) ||
        velocity.v.size() != compact_mac_v_size(grid) ||
        velocity.w.size() != compact_mac_w_size(grid) || velocity.u.data() == nullptr ||
        velocity.v.data() == nullptr || velocity.w.data() == nullptr) {
        throw std::invalid_argument(
            "AffinePeriodicFlow requires an output velocity view with exact CompactMAC face "
            "sizes ((nx+1)*ny*nz, nx*(ny+1)*nz, nx*ny*(nz+1))");
    }
}

// --- Affine-aware CompactMAC face-flux kernels (see the file header for the
// exact formula and face convention; identical wrapped-cell-pair convention
// to streamfunctions::Diagnostics.cuh and physics::VelocityField). ---

__global__ void affine_flux_u_kernel(const real* __restrict__ K, const real* __restrict__ w,
                                     int nx, int ny, int nz, real h, real Gx,
                                     real* __restrict__ U) {
    const std::size_t total = static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz);
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < total; index += stride) {
        const int i = static_cast<int>(index % static_cast<std::size_t>(nx + 1));
        const std::size_t rem = index / static_cast<std::size_t>(nx + 1);
        const int j = static_cast<int>(rem % static_cast<std::size_t>(ny));
        const int k = static_cast<int>(rem / static_cast<std::size_t>(ny));
        const int a_i = (i == 0) ? nx - 1 : i - 1;
        const int b_i = (i == nx) ? 0 : i;
        const std::size_t a = static_cast<std::size_t>(a_i) +
                              static_cast<std::size_t>(nx) *
                                  (static_cast<std::size_t>(j) +
                                   static_cast<std::size_t>(ny) * static_cast<std::size_t>(k));
        const std::size_t b = static_cast<std::size_t>(b_i) +
                              static_cast<std::size_t>(nx) *
                                  (static_cast<std::size_t>(j) +
                                   static_cast<std::size_t>(ny) * static_cast<std::size_t>(k));
        const real Kf = operators::harmonic_mean_positive_cell_coefficient(K[a], K[b]);
        U[index] = Kf * (Gx + (w[b] - w[a]) / h);
    }
}

__global__ void affine_flux_v_kernel(const real* __restrict__ K, const real* __restrict__ w,
                                     int nx, int ny, int nz, real h, real Gy,
                                     real* __restrict__ V) {
    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny + 1) *
                              static_cast<std::size_t>(nz);
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < total; index += stride) {
        const int i = static_cast<int>(index % static_cast<std::size_t>(nx));
        const std::size_t rem = index / static_cast<std::size_t>(nx);
        const int j = static_cast<int>(rem % static_cast<std::size_t>(ny + 1));
        const int k = static_cast<int>(rem / static_cast<std::size_t>(ny + 1));
        const int a_j = (j == 0) ? ny - 1 : j - 1;
        const int b_j = (j == ny) ? 0 : j;
        const std::size_t a = static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(nx) *
                                  (static_cast<std::size_t>(a_j) +
                                   static_cast<std::size_t>(ny) * static_cast<std::size_t>(k));
        const std::size_t b = static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(nx) *
                                  (static_cast<std::size_t>(b_j) +
                                   static_cast<std::size_t>(ny) * static_cast<std::size_t>(k));
        const real Kf = operators::harmonic_mean_positive_cell_coefficient(K[a], K[b]);
        V[index] = Kf * (Gy + (w[b] - w[a]) / h);
    }
}

__global__ void affine_flux_w_kernel(const real* __restrict__ K, const real* __restrict__ w,
                                     int nx, int ny, int nz, real h, real Gz,
                                     real* __restrict__ W) {
    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz + 1);
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < total; index += stride) {
        const int i = static_cast<int>(index % static_cast<std::size_t>(nx));
        const std::size_t rem = index / static_cast<std::size_t>(nx);
        const int j = static_cast<int>(rem % static_cast<std::size_t>(ny));
        const int k = static_cast<int>(rem / static_cast<std::size_t>(ny));
        const int a_k = (k == 0) ? nz - 1 : k - 1;
        const int b_k = (k == nz) ? 0 : k;
        const std::size_t a = static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(nx) *
                                  (static_cast<std::size_t>(j) +
                                   static_cast<std::size_t>(ny) * static_cast<std::size_t>(a_k));
        const std::size_t b = static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(nx) *
                                  (static_cast<std::size_t>(j) +
                                   static_cast<std::size_t>(ny) * static_cast<std::size_t>(b_k));
        const real Kf = operators::harmonic_mean_positive_cell_coefficient(K[a], K[b]);
        W[index] = Kf * (Gz + (w[b] - w[a]) / h);
    }
}

void launch_affine_face_flux(CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> K,
                             DeviceSpan<const real> w, real Gx, real Gy, real Gz,
                             DeviceSpan<real> U, DeviceSpan<real> V, DeviceSpan<real> W) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const real h = grid.dx;

    const std::size_t nu = compact_mac_u_size(grid);
    const std::size_t nv = compact_mac_v_size(grid);
    const std::size_t nw = compact_mac_w_size(grid);

    auto blocks_for = [](std::size_t total) {
        const std::size_t requested = (total + kBlock1D - 1) / kBlock1D;
        return static_cast<int>(requested < 65535 ? requested : 65535);
    };

    affine_flux_u_kernel<<<blocks_for(nu), kBlock1D, 0, ctx.cuda_stream()>>>(
        K.data(), w.data(), nx, ny, nz, h, Gx, U.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    affine_flux_v_kernel<<<blocks_for(nv), kBlock1D, 0, ctx.cuda_stream()>>>(
        K.data(), w.data(), nx, ny, nz, h, Gy, V.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    affine_flux_w_kernel<<<blocks_for(nw), kBlock1D, 0, ctx.cuda_stream()>>>(
        K.data(), w.data(), nx, ny, nz, h, Gz, W.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

// --- Deterministic two-pass block reductions (same pattern as
// PeriodicGaussianField.cu's mean/variance reduction). ---

__global__ void block_sum_unique_u_kernel(const real* __restrict__ U, int nx, int ny, int nz,
                                          real* __restrict__ block_sums) {
    extern __shared__ real s_sum[];
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    const std::size_t tid = threadIdx.x;
    std::size_t c = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    real local = real{0};
    while (c < n) {
        const int i = static_cast<int>(c % static_cast<std::size_t>(nx));
        const std::size_t rem = c / static_cast<std::size_t>(nx);
        const int j = static_cast<int>(rem % static_cast<std::size_t>(ny));
        const int k = static_cast<int>(rem / static_cast<std::size_t>(ny));
        const std::size_t face = static_cast<std::size_t>(i) +
                                 static_cast<std::size_t>(j) * static_cast<std::size_t>(nx + 1) +
                                 static_cast<std::size_t>(k) * static_cast<std::size_t>(nx + 1) *
                                     static_cast<std::size_t>(ny);
        local += U[face];
        c += static_cast<std::size_t>(blockDim.x) * gridDim.x;
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

__global__ void block_sum_unique_v_kernel(const real* __restrict__ V, int nx, int ny, int nz,
                                          real* __restrict__ block_sums) {
    extern __shared__ real s_sum[];
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    const std::size_t tid = threadIdx.x;
    std::size_t c = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    real local = real{0};
    while (c < n) {
        const int i = static_cast<int>(c % static_cast<std::size_t>(nx));
        const std::size_t rem = c / static_cast<std::size_t>(nx);
        const int j = static_cast<int>(rem % static_cast<std::size_t>(ny));
        const int k = static_cast<int>(rem / static_cast<std::size_t>(ny));
        const std::size_t face = static_cast<std::size_t>(i) +
                                 static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
                                 static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                                     static_cast<std::size_t>(ny + 1);
        local += V[face];
        c += static_cast<std::size_t>(blockDim.x) * gridDim.x;
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

__global__ void block_sum_unique_w_kernel(const real* __restrict__ W, int nx, int ny, int nz,
                                          real* __restrict__ block_sums) {
    extern __shared__ real s_sum[];
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    const std::size_t tid = threadIdx.x;
    std::size_t c = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    real local = real{0};
    while (c < n) {
        const int i = static_cast<int>(c % static_cast<std::size_t>(nx));
        const std::size_t rem = c / static_cast<std::size_t>(nx);
        const int j = static_cast<int>(rem % static_cast<std::size_t>(ny));
        const int k = static_cast<int>(rem / static_cast<std::size_t>(ny));
        const std::size_t face = static_cast<std::size_t>(i) +
                                 static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
                                 static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                                     static_cast<std::size_t>(ny);
        local += W[face];
        c += static_cast<std::size_t>(blockDim.x) * gridDim.x;
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

__global__ void compute_divergence_kernel(const real* __restrict__ U, const real* __restrict__ V,
                                          const real* __restrict__ W, int nx, int ny, int nz,
                                          real dx, real dy, real dz, real* __restrict__ div) {
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t c = start; c < n; c += stride) {
        const int i = static_cast<int>(c % static_cast<std::size_t>(nx));
        const std::size_t rem = c / static_cast<std::size_t>(nx);
        const int j = static_cast<int>(rem % static_cast<std::size_t>(ny));
        const int k = static_cast<int>(rem / static_cast<std::size_t>(ny));

        const std::size_t u0 = static_cast<std::size_t>(i) +
                               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx + 1) +
                               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx + 1) *
                                   static_cast<std::size_t>(ny);
        const std::size_t u1 = static_cast<std::size_t>(i + 1) +
                               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx + 1) +
                               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx + 1) *
                                   static_cast<std::size_t>(ny);
        const std::size_t v0 = static_cast<std::size_t>(i) +
                               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
                               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny + 1);
        const std::size_t v1 = static_cast<std::size_t>(i) +
                               static_cast<std::size_t>(j + 1) * static_cast<std::size_t>(nx) +
                               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny + 1);
        const std::size_t w0 = static_cast<std::size_t>(i) +
                               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
                               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny);
        const std::size_t w1 = static_cast<std::size_t>(i) +
                               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
                               static_cast<std::size_t>(k + 1) * static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny);

        div[c] = (U[u1] - U[u0]) / dx + (V[v1] - V[v0]) / dy + (W[w1] - W[w0]) / dz;
    }
}

__global__ void block_max_abs_kernel(const real* __restrict__ x, std::size_t n,
                                     real* __restrict__ block_max) {
    extern __shared__ real s_max[];
    const std::size_t tid = threadIdx.x;
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    real local = real{0};
    while (idx < n) {
        const real v = fabs(x[idx]);
        local = v > local ? v : local;
        idx += static_cast<std::size_t>(blockDim.x) * gridDim.x;
    }
    s_max[tid] = local;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = s_max[tid + s] > s_max[tid] ? s_max[tid + s] : s_max[tid];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_max[blockIdx.x] = s_max[0];
    }
}

int reduce_block_count(std::size_t n) {
    return static_cast<int>(
        std::min<std::size_t>(static_cast<std::size_t>(kMaxReduceBlocks),
                              (n + static_cast<std::size_t>(kBlock1D) - 1) /
                                  static_cast<std::size_t>(kBlock1D)));
}

real mean_unique_u(CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> U,
                   DeviceBuffer<real>& reduce_blocks) {
    const std::size_t n = grid.num_cells();
    const int n_blocks = reduce_block_count(n);
    std::vector<real> h_blocks(static_cast<std::size_t>(n_blocks));
    block_sum_unique_u_kernel<<<n_blocks, kBlock1D,
                               static_cast<std::size_t>(kBlock1D) * sizeof(real),
                               ctx.cuda_stream()>>>(U.data(), grid.nx, grid.ny, grid.nz,
                                                    reduce_blocks.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_blocks.data(), reduce_blocks.data(),
                                           static_cast<std::size_t>(n_blocks) * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    real sum = real{0};
    for (int b = 0; b < n_blocks; ++b) {
        sum += h_blocks[static_cast<std::size_t>(b)];
    }
    return sum / static_cast<real>(n);
}

real mean_unique_v(CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> V,
                   DeviceBuffer<real>& reduce_blocks) {
    const std::size_t n = grid.num_cells();
    const int n_blocks = reduce_block_count(n);
    std::vector<real> h_blocks(static_cast<std::size_t>(n_blocks));
    block_sum_unique_v_kernel<<<n_blocks, kBlock1D,
                               static_cast<std::size_t>(kBlock1D) * sizeof(real),
                               ctx.cuda_stream()>>>(V.data(), grid.nx, grid.ny, grid.nz,
                                                    reduce_blocks.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_blocks.data(), reduce_blocks.data(),
                                           static_cast<std::size_t>(n_blocks) * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    real sum = real{0};
    for (int b = 0; b < n_blocks; ++b) {
        sum += h_blocks[static_cast<std::size_t>(b)];
    }
    return sum / static_cast<real>(n);
}

real mean_unique_w(CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> W,
                   DeviceBuffer<real>& reduce_blocks) {
    const std::size_t n = grid.num_cells();
    const int n_blocks = reduce_block_count(n);
    std::vector<real> h_blocks(static_cast<std::size_t>(n_blocks));
    block_sum_unique_w_kernel<<<n_blocks, kBlock1D,
                               static_cast<std::size_t>(kBlock1D) * sizeof(real),
                               ctx.cuda_stream()>>>(W.data(), grid.nx, grid.ny, grid.nz,
                                                    reduce_blocks.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_blocks.data(), reduce_blocks.data(),
                                           static_cast<std::size_t>(n_blocks) * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    real sum = real{0};
    for (int b = 0; b < n_blocks; ++b) {
        sum += h_blocks[static_cast<std::size_t>(b)];
    }
    return sum / static_cast<real>(n);
}

real max_abs_reduction(CudaContext& ctx, DeviceSpan<const real> x, DeviceBuffer<real>& reduce_blocks) {
    const std::size_t n = x.size();
    const int n_blocks = reduce_block_count(n);
    std::vector<real> h_blocks(static_cast<std::size_t>(n_blocks));
    block_max_abs_kernel<<<n_blocks, kBlock1D, static_cast<std::size_t>(kBlock1D) * sizeof(real),
                          ctx.cuda_stream()>>>(x.data(), n, reduce_blocks.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_blocks.data(), reduce_blocks.data(),
                                           static_cast<std::size_t>(n_blocks) * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();
    real m = real{0};
    for (int b = 0; b < n_blocks; ++b) {
        m = std::max(m, h_blocks[static_cast<std::size_t>(b)]);
    }
    return m;
}

// --- Host 3x3 dense linear algebra (double precision) ---

real determinant3(const real m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// Solves A x = b via Cramer's rule. Throws std::runtime_error if A is
// singular (relative to its own entry scale) or produces a non-finite
// solution.
void solve_3x3_cramer(const real A[3][3], const real b[3], real x[3]) {
    real max_abs = real{0};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            max_abs = std::max(max_abs, std::abs(A[r][c]));
        }
    }
    const real det = determinant3(A);
    const real threshold =
        real{1e3} * std::numeric_limits<real>::epsilon() * max_abs * max_abs * max_abs;
    if (!std::isfinite(det) || max_abs == real{0} || std::abs(det) <= threshold) {
        throw std::runtime_error(
            "AffinePeriodicFlow: effective conductivity tensor K_eff is singular; cannot solve "
            "for the mean pressure gradient producing the target mean flux");
    }
    for (int col = 0; col < 3; ++col) {
        real Ac[3][3];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                Ac[r][c] = A[r][c];
            }
        }
        for (int r = 0; r < 3; ++r) {
            Ac[r][col] = b[r];
        }
        x[col] = determinant3(Ac) / det;
    }
    if (!std::isfinite(x[0]) || !std::isfinite(x[1]) || !std::isfinite(x[2])) {
        throw std::runtime_error(
            "AffinePeriodicFlow: mean-flux solve produced a non-finite pressure gradient");
    }
}

// Closed-form eigenvalues of a real symmetric 3x3 matrix (ascending order).
// Standard trigonometric method (Smith 1961); degrades gracefully to the
// diagonal entries for an already-diagonal matrix.
void symmetric_3x3_eigenvalues(const real S[3][3], real eig[3]) {
    const real p1 = S[0][1] * S[0][1] + S[0][2] * S[0][2] + S[1][2] * S[1][2];
    if (p1 == real{0}) {
        eig[0] = S[0][0];
        eig[1] = S[1][1];
        eig[2] = S[2][2];
        std::sort(eig, eig + 3);
        return;
    }
    const real q = (S[0][0] + S[1][1] + S[2][2]) / real{3};
    const real p2 = (S[0][0] - q) * (S[0][0] - q) + (S[1][1] - q) * (S[1][1] - q) +
                    (S[2][2] - q) * (S[2][2] - q) + real{2} * p1;
    const real p = std::sqrt(p2 / real{6});
    real B[3][3];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            B[r][c] = (S[r][c] - (r == c ? q : real{0})) / p;
        }
    }
    real r_det = determinant3(B) / real{2};
    r_det = std::max(real{-1}, std::min(real{1}, r_det));
    const real phi = std::acos(r_det) / real{3};
    const real eig0 = q + real{2} * p * std::cos(phi);
    const real eig2 = q + real{2} * p * std::cos(phi + real{2} * kPi / real{3});
    const real eig1 = real{3} * q - eig0 - eig2;
    eig[0] = eig0;
    eig[1] = eig1;
    eig[2] = eig2;
    std::sort(eig, eig + 3);
}

} // namespace

void AffinePeriodicFlowWorkspace::ensure_prepared() const {
    if (!prepared_) {
        throw std::logic_error("AffinePeriodicFlowWorkspace requires a preceding prepare() call");
    }
}

void AffinePeriodicFlowWorkspace::prepare(const Grid3D& grid,
                                          const AffinePeriodicFlowConfig& config) {
    require_valid_grid_geometry(grid);
    require_valid_mg_config(config.mg, grid);

    const std::size_t n = grid.num_cells();
    for (auto& w : correctors_) {
        w.resize(n);
    }
    h_tilde_.resize(n);
    rhs_a_.resize(n);
    rhs_b_.resize(n);

    flux_u_.resize(compact_mac_u_size(grid));
    flux_v_.resize(compact_mac_v_size(grid));
    flux_w_.resize(compact_mac_w_size(grid));

    divergence_.resize(n);
    reduce_blocks_.resize(static_cast<std::size_t>(kMaxReduceBlocks));

    affine_rhs_workspace_.prepare(n);
    pcg_workspace_.prepare(n);

    const bool mg_already_current = mg_hierarchy_ != nullptr && mg_preconditioner_.has_value() &&
                                    grids_equal(prepared_grid_, grid) &&
                                    mg_config_equal(prepared_mg_config_, config.mg);
    if (!mg_already_current) {
        mg_preconditioner_.reset();
        mg_hierarchy_ = std::make_unique<multigrid::MGHierarchy>(grid, config.mg.num_levels);
        mg_preconditioner_.emplace(*mg_hierarchy_, config.mg);
    }

    prepared_grid_ = grid;
    prepared_mg_config_ = config.mg;
    prepared_ = true;
}

bool AffinePeriodicFlowWorkspace::prepared_for(const Grid3D& grid,
                                               const AffinePeriodicFlowConfig& config) const noexcept {
    if (!prepared_) {
        return false;
    }
    const std::size_t n = grid.num_cells();
    for (const auto& w : correctors_) {
        if (w.size() != n) {
            return false;
        }
    }
    return h_tilde_.size() == n && rhs_a_.size() == n && rhs_b_.size() == n &&
           flux_u_.size() == compact_mac_u_size(grid) &&
           flux_v_.size() == compact_mac_v_size(grid) &&
           flux_w_.size() == compact_mac_w_size(grid) && divergence_.size() == n &&
           reduce_blocks_.size() == static_cast<std::size_t>(kMaxReduceBlocks) &&
           affine_rhs_workspace_.prepared_for(n) && pcg_workspace_.prepared_for(n) &&
           mg_hierarchy_ != nullptr && mg_preconditioner_.has_value() &&
           grids_equal(prepared_grid_, grid) && mg_config_equal(prepared_mg_config_, config.mg);
}

std::size_t AffinePeriodicFlowWorkspace::allocated_device_bytes() const noexcept {
    const auto report = memory_report();
    return report.total_bytes;
}

AffinePeriodicFlowMemoryReport AffinePeriodicFlowWorkspace::memory_report() const {
    AffinePeriodicFlowMemoryReport report{};
    for (const auto& w : correctors_) {
        report.correctors_bytes += w.capacity() * sizeof(real);
    }
    report.correctors_bytes += h_tilde_.capacity() * sizeof(real);
    report.rhs_scratch_bytes = (rhs_a_.capacity() + rhs_b_.capacity()) * sizeof(real);
    report.flux_scratch_bytes =
        (flux_u_.capacity() + flux_v_.capacity() + flux_w_.capacity() + divergence_.capacity()) *
        sizeof(real);
    report.reduction_scratch_bytes = reduce_blocks_.capacity() * sizeof(real);
    report.affine_rhs_workspace_bytes = affine_rhs_workspace_.allocated_device_bytes();

    report.pcg_workspace_bytes =
        (pcg_workspace_.pcg.r.capacity() + pcg_workspace_.pcg.z.capacity() +
         pcg_workspace_.pcg.p.capacity() + pcg_workspace_.pcg.Ap.capacity()) *
            sizeof(real) +
        pcg_workspace_.pcg.red.d_scalar.capacity() * sizeof(real) +
        pcg_workspace_.projected_rhs.capacity() * sizeof(real) +
        pcg_workspace_.mean_zero.allocated_device_bytes();

    if (mg_hierarchy_ != nullptr) {
        for (const auto& level : mg_hierarchy_->levels) {
            report.mg_hierarchy_bytes += (level.x.capacity() + level.b.capacity() +
                                          level.r.capacity() + level.coefficient.capacity()) *
                                         sizeof(real);
        }
    }
    if (mg_preconditioner_.has_value()) {
        report.mg_preconditioner_bytes = mg_preconditioner_->allocated_device_bytes();
    }

    report.total_bytes = report.correctors_bytes + report.rhs_scratch_bytes +
                         report.flux_scratch_bytes + report.reduction_scratch_bytes +
                         report.affine_rhs_workspace_bytes + report.pcg_workspace_bytes +
                         report.mg_hierarchy_bytes + report.mg_preconditioner_bytes;
    return report;
}

AffinePeriodicFlowReport solve_affine_periodic_flow(CudaContext& context, const Grid3D& grid,
                                                    DeviceSpan<const real> K,
                                                    const AffinePeriodicFlowConfig& config,
                                                    AffinePeriodicVelocityView velocity,
                                                    AffinePeriodicFlowWorkspace& workspace) {
    require_valid_problem(grid, K, config, velocity);

    workspace.prepare(grid, config);

    AffinePeriodicFlowReport report{};

    // K_eff coefficient hierarchy fill: the "explicit sign/coefficient
    // adapter" -- the Lester streamfunction path fills q = 1/K here, this
    // module fills K itself.
    multigrid::populate_coefficient_hierarchy(context, *workspace.mg_hierarchy_, K);

    operators::LesterPositiveDiffusionOperator A(grid, K);
    constraints::MeanZeroProjector projector;

    const streamfunctions::AffineGradient unit_directions[3] = {
        streamfunctions::AffineGradient{real{1}, real{0}, real{0}},
        streamfunctions::AffineGradient{real{0}, real{1}, real{0}},
        streamfunctions::AffineGradient{real{0}, real{0}, real{1}},
    };

    const std::size_t n = grid.num_cells();

    for (int d = 0; d < 3; ++d) {
        // RHS_d = div_h(K e_d), via the SF-06 assembler with a degenerate
        // (e_d, e_d) gauge pair; rhs_b_ is discarded (see the file header).
        streamfunctions::AffineGauge gauge;
        gauge.psi1_gradient = unit_directions[d];
        gauge.psi2_gradient = unit_directions[d];
        (void)streamfunctions::assemble_affine_periodic_rhs(
            context, grid, K, gauge, workspace.rhs_a_.span(), workspace.rhs_b_.span(),
            workspace.affine_rhs_workspace_);

        MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(workspace.correctors_[static_cast<std::size_t>(d)].data(),
                                               0, n * sizeof(real), context.cuda_stream()));
        report.corrector_results[d] = solvers::projected_pcg_solve(
            context, A, *workspace.mg_preconditioner_,
            DeviceSpan<const real>(workspace.rhs_a_.span()),
            workspace.correctors_[static_cast<std::size_t>(d)].span(), config.linear, projector,
            workspace.pcg_workspace_);

        // K_eff column d: affine-aware CompactMAC flux for this corrector
        // alone (G = e_d, w = w_d), reduced over unique faces.
        launch_affine_face_flux(context, grid, K,
                                DeviceSpan<const real>(workspace.correctors_[static_cast<std::size_t>(d)].span()),
                                unit_directions[d].x, unit_directions[d].y, unit_directions[d].z,
                                workspace.flux_u_.span(), workspace.flux_v_.span(),
                                workspace.flux_w_.span());
        report.K_eff[0][d] = mean_unique_u(context, grid, DeviceSpan<const real>(workspace.flux_u_.span()),
                                          workspace.reduce_blocks_);
        report.K_eff[1][d] = mean_unique_v(context, grid, DeviceSpan<const real>(workspace.flux_v_.span()),
                                          workspace.reduce_blocks_);
        report.K_eff[2][d] = mean_unique_w(context, grid, DeviceSpan<const real>(workspace.flux_w_.span()),
                                          workspace.reduce_blocks_);
    }

    // Symmetry defect and positivity evidence, computed BEFORE the
    // mean-flux solve so a degenerate tensor is reported even if the solve
    // itself throws.
    real max_abs_entry = real{0};
    real max_asym = real{0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            max_abs_entry = std::max(max_abs_entry, std::abs(report.K_eff[i][j]));
            max_asym = std::max(max_asym, std::abs(report.K_eff[i][j] - report.K_eff[j][i]));
        }
    }
    report.symmetry_defect_rel = max_abs_entry > real{0} ? max_asym / max_abs_entry : real{0};

    real Ksym[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Ksym[i][j] = real{0.5} * (report.K_eff[i][j] + report.K_eff[j][i]);
        }
    }
    symmetric_3x3_eigenvalues(Ksym, report.eigenvalues_symmetric_part);

    if (!std::isfinite(report.eigenvalues_symmetric_part[0]) ||
        !std::isfinite(report.eigenvalues_symmetric_part[1]) ||
        !std::isfinite(report.eigenvalues_symmetric_part[2]) ||
        report.eigenvalues_symmetric_part[0] <= real{0}) {
        throw std::runtime_error(
            "AffinePeriodicFlow: effective conductivity tensor's symmetric part is not positive "
            "definite; refusing to solve for a mean pressure gradient");
    }

    // Full (unsymmetrized) mean-flux solve: NEVER discard transverse
    // coupling (spec rollback rule).
    solve_3x3_cramer(report.K_eff, config.qbar, report.G);

    // Recombine h_tilde = sum_d G_d w_d.
    MACROFLOW3D_CUDA_CHECK(
        cudaMemsetAsync(workspace.h_tilde_.data(), 0, n * sizeof(real), context.cuda_stream()));
    for (int d = 0; d < 3; ++d) {
        blas::axpy(context, report.G[d],
                  DeviceSpan<const real>(workspace.correctors_[static_cast<std::size_t>(d)].span()),
                  workspace.h_tilde_.span());
    }

    // Final affine-aware CompactMAC velocity, written into the caller's
    // output view.
    launch_affine_face_flux(context, grid, K, DeviceSpan<const real>(workspace.h_tilde_.span()),
                            report.G[0], report.G[1], report.G[2], velocity.u, velocity.v,
                            velocity.w);

    report.achieved_mean_flux[0] =
        mean_unique_u(context, grid, DeviceSpan<const real>(velocity.u), workspace.reduce_blocks_);
    report.achieved_mean_flux[1] =
        mean_unique_v(context, grid, DeviceSpan<const real>(velocity.v), workspace.reduce_blocks_);
    report.achieved_mean_flux[2] =
        mean_unique_w(context, grid, DeviceSpan<const real>(velocity.w), workspace.reduce_blocks_);

    // Mass-conservation diagnostics.
    {
        const std::size_t blocks =
            (n + static_cast<std::size_t>(kBlock1D) - 1) / static_cast<std::size_t>(kBlock1D);
        const int grid1d = static_cast<int>(std::min<std::size_t>(blocks, 65535));
        compute_divergence_kernel<<<grid1d, kBlock1D, 0, context.cuda_stream()>>>(
            velocity.u.data(), velocity.v.data(), velocity.w.data(), grid.nx, grid.ny, grid.nz,
            grid.dx, grid.dy, grid.dz, workspace.divergence_.data());
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    report.div_max_abs = max_abs_reduction(
        context, DeviceSpan<const real>(workspace.divergence_.span()), workspace.reduce_blocks_);
    report.div_rms =
        blas::nrm2_host(context, DeviceSpan<const real>(workspace.divergence_.span()),
                       workspace.pcg_workspace_.pcg.red) /
        std::sqrt(static_cast<real>(n));

    report.memory = workspace.memory_report();

    return report;
}

} // namespace physics
} // namespace macroflow3d
