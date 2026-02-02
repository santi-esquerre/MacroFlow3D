#include "coarsen_K.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>

namespace rwpt {
namespace physics {

/**
 * Legacy: CompactHomogenizationKtensor
 * Geometric mean over 2x2x2 block of fine cells.
 * K_coarse[IX,IY,IZ] = exp(mean(log(K_fine[8 cells])))
 */
__global__ void coarsen_K_kernel(
    real* __restrict__ K_coarse,
    const real* __restrict__ K_fine,
    int NX, int NY, int NZ  // Coarse dimensions
) {
    int IX = threadIdx.x + blockIdx.x * blockDim.x;
    int IY = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (IX >= NX || IY >= NY) return;
    
    // Fine grid dimensions (2x coarse)
    int Nx = 2 * NX;
    int Ny = 2 * NY;
    
    for (int IZ = 0; IZ < NZ; ++IZ) {
        // Fine cell indices (bottom-left-back corner of 2x2x2 block)
        int ix = 2 * IX;
        int iy = 2 * IY;
        int iz = 2 * IZ;
        
        // Fine indexing
        int stride = Nx * Ny;
        int idx000 = ix + iy * Nx + iz * stride;
        int idx100 = idx000 + 1;
        int idx010 = idx000 + Nx;
        int idx110 = idx000 + 1 + Nx;
        int idx001 = idx000 + stride;
        int idx101 = idx001 + 1;
        int idx011 = idx001 + Nx;
        int idx111 = idx001 + 1 + Nx;
        
        // Geometric mean: exp(mean(log(K)))
        real log_sum = log(K_fine[idx000]) + log(K_fine[idx100]) +
                       log(K_fine[idx010]) + log(K_fine[idx110]) +
                       log(K_fine[idx001]) + log(K_fine[idx101]) +
                       log(K_fine[idx011]) + log(K_fine[idx111]);
        
        int coarse_idx = IX + IY * NX + IZ * NX * NY;
        K_coarse[coarse_idx] = exp(log_sum / 8.0);
    }
}

void coarsen_K(
    CudaContext& ctx,
    const Grid3D& coarse_grid,
    const Grid3D& fine_grid,
    DeviceSpan<const real> K_fine,
    DeviceSpan<real> K_coarse
) {
    int NX = coarse_grid.nx;
    int NY = coarse_grid.ny;
    int NZ = coarse_grid.nz;
    
    // Validate dimensions
    assert(fine_grid.nx == 2 * NX && "Fine grid must be 2x coarse in x");
    assert(fine_grid.ny == 2 * NY && "Fine grid must be 2x coarse in y");
    assert(fine_grid.nz == 2 * NZ && "Fine grid must be 2x coarse in z");
    assert(K_fine.size() == fine_grid.num_cells() && "Fine K size mismatch");
    assert(K_coarse.size() == coarse_grid.num_cells() && "Coarse K size mismatch");
    
    dim3 block(16, 16);
    int grid_x = (NX + block.x - 1) / block.x;
    int grid_y = (NY + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    
    coarsen_K_kernel<<<grid, block, 0, ctx.cuda_stream()>>>(
        K_coarse.data(), K_fine.data(), NX, NY, NZ
    );
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace physics
} // namespace rwpt
