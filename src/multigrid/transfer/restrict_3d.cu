#include "restrict_3d.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace macroflow3d {
namespace multigrid {

// Legacy: restriction_linear3D
// Full-weighting: averages 8 fine cells into 1 coarse cell
// Grid-stride for robustness
__global__ void restrict_kernel(
    real* __restrict__ phiCoarse,
    const real* __restrict__ phiFine,
    int NX, int NY, int NZ  // Coarse dimensions
) {
    int IX = threadIdx.x + blockIdx.x * blockDim.x;
    int IY = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Grid-stride in y
    for (; IY < NY; IY += blockDim.y * gridDim.y) {
        // Grid-stride in x
        for (int IX_local = IX; IX_local < NX; IX_local += blockDim.x * gridDim.x) {
            int STRIDE = NX * NY;
            
            // Corresponding fine grid indices (2x resolution)
            int ix = 2 * IX_local;
            int iy = 2 * IY;
            int Nx = 2 * NX;
            int Ny = 2 * NY;
            int stride = Nx * Ny;
            
            for (int IZ = 0; IZ < NZ; ++IZ) {
                int iz = 2 * IZ;
                int IN_IDX = IX_local + IY * NX + IZ * STRIDE;
                int in_idx = ix + iy * Nx + iz * stride;
                
                real result = 0.0;
                // Bottom 4 cells (z-plane)
                result += phiFine[in_idx];
                result += phiFine[in_idx + 1];
                result += phiFine[in_idx + Nx];
                result += phiFine[in_idx + 1 + Nx];
                
                // Top 4 cells (z+1 plane)
                result += phiFine[in_idx + stride];
                result += phiFine[in_idx + 1 + stride];
                result += phiFine[in_idx + Nx + stride];
                result += phiFine[in_idx + 1 + Nx + stride];
                
                phiCoarse[IN_IDX] = result / 8.0;
            }
        }
    }
}

void restrict_3d(
    CudaContext& ctx,
    const Grid3D& fine_grid,
    const Grid3D& coarse_grid,
    DeviceSpan<const real> r_fine,
    DeviceSpan<real> b_coarse
) {
    int NX = coarse_grid.nx;
    int NY = coarse_grid.ny;
    int NZ = coarse_grid.nz;
    
    // Validate dimensions
    assert(fine_grid.nx == 2 * NX && "Fine grid must be 2x coarse in x");
    assert(fine_grid.ny == 2 * NY && "Fine grid must be 2x coarse in y");
    assert(fine_grid.nz == 2 * NZ && "Fine grid must be 2x coarse in z");
    assert(r_fine.size() == fine_grid.num_cells() && "Fine buffer size mismatch");
    assert(b_coarse.size() == coarse_grid.num_cells() && "Coarse buffer size mismatch");
    
    dim3 block(16, 16);
    int grid_x = (NX + block.x - 1) / block.x;
    int grid_y = (NY + block.y - 1) / block.y;
    // Clamp grid dimensions
    grid_x = (grid_x < 65535) ? grid_x : 65535;
    grid_y = (grid_y < 65535) ? grid_y : 65535;
    dim3 grid(grid_x, grid_y);
    
    restrict_kernel<<<grid, block, 0, ctx.cuda_stream()>>>(
        b_coarse.data(), r_fine.data(), NX, NY, NZ
    );
    
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace multigrid
} // namespace macroflow3d
