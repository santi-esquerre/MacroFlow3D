/**
 * @file rhs_head.cu
 * @brief Implementation of RHS builder for head equation
 *
 * Legacy correspondence: RHS_head_3D.cu
 *
 * For Dirichlet BCs, adds contribution: RHS -= 2 * K_cell * H_bc / dx²
 */

#include "../../core/BCSpecDevice.cuh"
#include "../../runtime/cuda_check.cuh"
#include "rhs_head.cuh"

namespace macroflow3d {
namespace physics {

// ============================================================================
// Kernel: Fill with zeros
// ============================================================================

__global__ void kernel_fill_zero(real* __restrict__ data, const size_t n) {
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;
    data[idx] = real(0.0);
}

// ============================================================================
// Kernels: Dirichlet BC contributions to RHS
// Legacy: RHS -= 2.0 * Hb * KC / dx²  (where Hb is Dirichlet value)
// ============================================================================

// Face x=0 (west)
__global__ void kernel_rhs_dirichlet_xmin(real* __restrict__ rhs, const real* __restrict__ K,
                                          int nx, int ny, int nz, real inv_dx2, real h_bc) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= ny || k >= nz)
        return;

    int i = 0;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// Face x=nx-1 (east)
__global__ void kernel_rhs_dirichlet_xmax(real* __restrict__ rhs, const real* __restrict__ K,
                                          int nx, int ny, int nz, real inv_dx2, real h_bc) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (j >= ny || k >= nz)
        return;

    int i = nx - 1;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// Face y=0 (south)
__global__ void kernel_rhs_dirichlet_ymin(real* __restrict__ rhs, const real* __restrict__ K,
                                          int nx, int ny, int nz, real inv_dx2, real h_bc) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || k >= nz)
        return;

    int j = 0;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// Face y=ny-1 (north)
__global__ void kernel_rhs_dirichlet_ymax(real* __restrict__ rhs, const real* __restrict__ K,
                                          int nx, int ny, int nz, real inv_dx2, real h_bc) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || k >= nz)
        return;

    int j = ny - 1;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// Face z=0 (bottom)
__global__ void kernel_rhs_dirichlet_zmin(real* __restrict__ rhs, const real* __restrict__ K,
                                          int nx, int ny, int nz, real inv_dx2, real h_bc) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || j >= ny)
        return;

    int k = 0;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// Face z=nz-1 (top)
__global__ void kernel_rhs_dirichlet_zmax(real* __restrict__ rhs, const real* __restrict__ K,
                                          int nx, int ny, int nz, real inv_dx2, real h_bc) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= nx || j >= ny)
        return;

    int k = nz - 1;
    int idx = i + nx * (j + ny * k);
    real KC = K[idx];
    rhs[idx] -= real(2.0) * h_bc * KC * inv_dx2;
}

// ============================================================================
// Host API
// ============================================================================

void build_rhs_head(DeviceSpan<real> rhs, DeviceSpan<const real> K, const Grid3D& grid,
                    const BCSpec& bc, const CudaContext& ctx) {
    const size_t n = grid.num_cells();
    if (rhs.size() < n || K.size() < n) {
        throw std::runtime_error("RHS or K buffer too small");
    }

    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const real inv_dx2 = real(1.0) / (grid.dx * grid.dx);

    // 1. Zero out RHS (no volumetric sources)
    {
        const int block = 256;
        const int grid_1d = (n + block - 1) / block;
        kernel_fill_zero<<<grid_1d, block, 0, ctx.cuda_stream()>>>(rhs.data(), n);
    }

    // 2. Add Dirichlet contributions
    dim3 block_yz(16, 16);
    dim3 block_xz(16, 16);
    dim3 block_xy(16, 16);

    // x-faces (west/east)
    dim3 grid_yz((ny + block_yz.x - 1) / block_yz.x, (nz + block_yz.y - 1) / block_yz.y);

    if (bc.xmin.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_xmin<<<grid_yz, block_yz, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.xmin.value);
    }
    if (bc.xmax.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_xmax<<<grid_yz, block_yz, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.xmax.value);
    }

    // y-faces (south/north)
    dim3 grid_xz((nx + block_xz.x - 1) / block_xz.x, (nz + block_xz.y - 1) / block_xz.y);

    if (bc.ymin.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_ymin<<<grid_xz, block_xz, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.ymin.value);
    }
    if (bc.ymax.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_ymax<<<grid_xz, block_xz, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.ymax.value);
    }

    // z-faces (bottom/top)
    dim3 grid_xy((nx + block_xy.x - 1) / block_xy.x, (ny + block_xy.y - 1) / block_xy.y);

    if (bc.zmin.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_zmin<<<grid_xy, block_xy, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.zmin.value);
    }
    if (bc.zmax.type == BCType::Dirichlet) {
        kernel_rhs_dirichlet_zmax<<<grid_xy, block_xy, 0, ctx.cuda_stream()>>>(
            rhs.data(), K.data(), nx, ny, nz, inv_dx2, bc.zmax.value);
    }

    // =======================================================================
    // Pin cell for singular systems (legacy: pin1stCell)
    // =======================================================================
    // When NO boundary is Dirichlet, the system is singular (constant mode
    // in null space). Pinning one cell breaks this degeneracy.
    //
    // Legacy implementation: "diagonal doubling" approach
    //   - In operator/smoother/residual: aC *= 2 for cell [0,0,0]
    //   - RHS is NOT modified
    //
    // The diagonal doubling happens in the operator and smoother kernels,
    // NOT here. This comment is just for documentation.
    // =======================================================================

    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace physics
} // namespace macroflow3d
