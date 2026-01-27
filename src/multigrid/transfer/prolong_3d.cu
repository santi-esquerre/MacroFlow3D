#include "prolong_3d.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace rwpt {
namespace multigrid {

// Legacy: prolongation_interior_linear3D
// Simple injection: fine cell (i,j,k) gets value from coarse parent (i/2, j/2, k/2)
// This is NOT octant indexing (that was causing OOB reads)
// Grid-stride loop for robustness
__global__ void prolong_kernel(
    real* __restrict__ phiFine,
    const real* __restrict__ phiCoarse,
    int Nx, int Ny, int Nz  // Fine dimensions
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Grid-stride in y for large grids
    for (; iy < Ny; iy += blockDim.y * gridDim.y) {
        // Grid-stride in x
        for (int ix_local = ix; ix_local < Nx; ix_local += blockDim.x * gridDim.x) {
            // Coarse indices (integer division)
            int IX = ix_local / 2;
            int IY = iy / 2;
            int NX = Nx / 2;
            int NY = Ny / 2;
            
            for (int iz = 0; iz < Nz; ++iz) {
                int IZ = iz / 2;
                
                int in_idx = ix_local + iy * Nx + iz * Nx * Ny;
                int IN_IDX = IX + IY * NX + IZ * NX * NY;
                
                // Simple injection: each fine cell gets value from coarse parent
                phiFine[in_idx] += phiCoarse[IN_IDX];
            }
        }
    }
}

void prolong_3d_add(
    CudaContext& ctx,
    const Grid3D& coarse_grid,
    const Grid3D& fine_grid,
    DeviceSpan<const real> x_coarse,
    DeviceSpan<real> x_fine
) {
    int Nx = fine_grid.nx;
    int Ny = fine_grid.ny;
    int Nz = fine_grid.nz;
    
    // Validate dimensions
    assert(coarse_grid.nx == Nx / 2 && "Fine grid must be 2x coarse in x");
    assert(coarse_grid.ny == Ny / 2 && "Fine grid must be 2x coarse in y");
    assert(coarse_grid.nz == Nz / 2 && "Fine grid must be 2x coarse in z");
    assert(x_coarse.size() == coarse_grid.num_cells() && "Coarse buffer size mismatch");
    assert(x_fine.size() == fine_grid.num_cells() && "Fine buffer size mismatch");
    
    dim3 block(16, 16);
    int grid_x = (Nx + block.x - 1) / block.x;
    int grid_y = (Ny + block.y - 1) / block.y;
    // Clamp grid dimensions
    grid_x = (grid_x < 65535) ? grid_x : 65535;
    grid_y = (grid_y < 65535) ? grid_y : 65535;
    dim3 grid(grid_x, grid_y);
    
    prolong_kernel<<<grid, block, 0, ctx.cuda_stream()>>>(
        x_fine.data(), x_coarse.data(), Nx, Ny, Nz
    );
    
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace multigrid
} // namespace rwpt
