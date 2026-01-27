#include "gsrb_3d.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace rwpt {
namespace multigrid {

// Legacy: GSRB_int from GSRB_Smooth_up_residual_3D.cu
// Red-Black Gauss-Seidel iteration (interior only, boundaries fixed at 0 for Dirichlet)
// For cell (ix,iy,iz), update x if (ix+iy+iz)%2 matches color
// x[i] = -( b[i] - sum_j(K_ij * x[j]) ) / K_ii
// Grid-stride for robustness
__global__ void gsrb_interior_kernel(
    real* __restrict__ x,
    const real* __restrict__ b,
    const real* __restrict__ K,
    real dxdx,
    int Nx, int Ny, int Nz,
    bool is_red  // true for red, false for black
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Grid-stride in y
    for (; iy < Ny - 2; iy += blockDim.y * gridDim.y) {
        // Grid-stride in x
        for (int ix_local = ix; ix_local < Nx - 2; ix_local += blockDim.x * gridDim.x) {
            int stride = Nx * Ny;
            
            for (int iz = 1; iz < Nz - 1; ++iz) {
                // Red-black coloring: based on (ix+1, iy+1, iz) since we offset by 1
                // Legacy uses absolute indices, so we match that
                int ix_abs = ix_local + 1;
                int iy_abs = iy + 1;
                int color_sum = ix_abs + iy_abs + iz;
                
                // Only update cells of correct color
                if ((is_red && (color_sum % 2 != 0)) || (!is_red && (color_sum % 2 == 0))) {
                    continue;
                }
                
                int idx = ix_abs + iy_abs * Nx + iz * stride;
                
                real KC = K[idx];
                
                // Compute harmonic mean conductivities for 6 faces
                real K_xp = 2.0 / (1.0 / KC + 1.0 / K[idx + 1]);
                real K_xm = 2.0 / (1.0 / KC + 1.0 / K[idx - 1]);
                real K_yp = 2.0 / (1.0 / KC + 1.0 / K[idx + Nx]);
                real K_ym = 2.0 / (1.0 / KC + 1.0 / K[idx - Nx]);
                real K_zp = 2.0 / (1.0 / KC + 1.0 / K[idx + stride]);
                real K_zm = 2.0 / (1.0 / KC + 1.0 / K[idx - stride]);
                
                // Sum of neighbor contributions
                real result = x[idx + 1] * K_xp;
                result += x[idx - 1] * K_xm;
                result += x[idx + Nx] * K_yp;
                result += x[idx - Nx] * K_ym;
                result += x[idx + stride] * K_zp;
                result += x[idx - stride] * K_zm;
                
                // Diagonal coefficient
                real aC = K_xp + K_xm + K_yp + K_ym + K_zp + K_zm;
                
                // Update: x = -(b - result/dxdx) / (aC/dxdx)
                x[idx] = -(b[idx] - result / dxdx) / (aC / dxdx);
            }
        }
    }
}

void gsrb_smooth_3d(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    int num_iters,
    const BCSpec& bc
) {
    int Nx = grid.nx;
    int Ny = grid.ny;
    int Nz = grid.nz;
    size_t n = grid.num_cells();
    
    // Validate BC (only Dirichlet 0 supported in Etapa 4)
    assert(bc.xmin.type == BCType::Dirichlet && bc.xmin.value == 0.0 && "Only Dirichlet 0 supported");
    assert(bc.xmax.type == BCType::Dirichlet && bc.xmax.value == 0.0 && "Only Dirichlet 0 supported");
    assert(bc.ymin.type == BCType::Dirichlet && bc.ymin.value == 0.0 && "Only Dirichlet 0 supported");
    assert(bc.ymax.type == BCType::Dirichlet && bc.ymax.value == 0.0 && "Only Dirichlet 0 supported");
    assert(bc.zmin.type == BCType::Dirichlet && bc.zmin.value == 0.0 && "Only Dirichlet 0 supported");
    assert(bc.zmax.type == BCType::Dirichlet && bc.zmax.value == 0.0 && "Only Dirichlet 0 supported");
    
    // Validate sizes
    assert(x.size() == n && "x size mismatch");
    assert(b.size() == n && "b size mismatch");
    assert(K.size() == n && "K size mismatch");
    
    real dxdx = 1.0 / (grid.dx * grid.dx);  // Assuming isotropic
    
    dim3 block(16, 16);
    int grid_x = (Nx + block.x - 1) / block.x;
    int grid_y = (Ny + block.y - 1) / block.y;
    grid_x = (grid_x < 65535) ? grid_x : 65535;
    grid_y = (grid_y < 65535) ? grid_y : 65535;
    dim3 grid_dim(grid_x, grid_y);
    
    for (int iter = 0; iter < num_iters; ++iter) {
        // Red sweep
        gsrb_interior_kernel<<<grid_dim, block, 0, ctx.cuda_stream()>>>(
            x.data(), b.data(), K.data(), dxdx, Nx, Ny, Nz, true
        );
        RWPT_CUDA_CHECK(cudaGetLastError());
        
        // Black sweep
        gsrb_interior_kernel<<<grid_dim, block, 0, ctx.cuda_stream()>>>(
            x.data(), b.data(), K.data(), dxdx, Nx, Ny, Nz, false
        );
        RWPT_CUDA_CHECK(cudaGetLastError());
    }
    
    // Note: boundaries remain at 0 (Dirichlet BC), not updated by GSRB
}

} // namespace multigrid
} // namespace rwpt
