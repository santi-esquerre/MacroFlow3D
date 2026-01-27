#include "residual_3d.cuh"
#include "../../runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace rwpt {
namespace multigrid {

// Legacy: update_int from up_residual_3D.cu
// r = b - A*x where A is variable-coefficient Laplacian
// Uses harmonic mean for face conductivities: K_face = 2/(1/K_L + 1/K_R)
// Grid-stride for robustness
__global__ void residual_interior_kernel(
    real* __restrict__ r,
    const real* __restrict__ x,
    const real* __restrict__ b,
    const real* __restrict__ K,
    real dxdx,  // 1/(dx*dx), assuming isotropic
    int Nx, int Ny, int Nz
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Grid-stride in y
    for (; iy < Ny - 2; iy += blockDim.y * gridDim.y) {
        // Grid-stride in x
        for (int ix_local = ix; ix_local < Nx - 2; ix_local += blockDim.x * gridDim.x) {
            int stride = Nx * Ny;
            int in_idx = (ix_local + 1) + (iy + 1) * Nx;
            
            // Pre-load for iz=1
            real x_current = x[in_idx];
            real K_current = K[in_idx];
            int out_idx = in_idx;
            in_idx += stride;
            real x_top = x[in_idx];
            real K_top = K[in_idx];
            in_idx += stride;
            
            for (int iz = 1; iz < Nz - 1; ++iz) {
                real x_bottom = x_current;
                x_current = x_top;
                x_top = x[in_idx];
                
                real K_bottom = K_current;
                K_current = K_top;
                K_top = K[in_idx];
                
                in_idx += stride;
                out_idx += stride;
                
                // Compute -A*x using harmonic mean
                real Ax = 0.0;
                
                // +x face
                Ax -= 2.0 * (x_current - x[out_idx + 1]) / (1.0 / K_current + 1.0 / K[out_idx + 1]);
                // +y face
                Ax -= 2.0 * (x_current - x[out_idx + Nx]) / (1.0 / K_current + 1.0 / K[out_idx + Nx]);
                // -x face
                Ax -= 2.0 * (x_current - x[out_idx - 1]) / (1.0 / K_current + 1.0 / K[out_idx - 1]);
                // -y face
                Ax -= 2.0 * (x_current - x[out_idx - Nx]) / (1.0 / K_current + 1.0 / K[out_idx - Nx]);
                // +z face
                Ax -= 2.0 * (x_current - x_top) / (1.0 / K_current + 1.0 / K_top);
                // -z face
                Ax -= 2.0 * (x_current - x_bottom) / (1.0 / K_current + 1.0 / K_bottom);
                
                r[out_idx] = b[out_idx] - Ax / dxdx;
            }
        }
    }
}

// Apply Dirichlet BC: boundary nodes have r = b (since x should be 0 at boundary)
__global__ void residual_boundary_kernel(
    real* __restrict__ r,
    const real* __restrict__ b,
    int Nx, int Ny, int Nz
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = Nx * Ny * Nz;
    
    for (; idx < total; idx += blockDim.x * gridDim.x) {
        int iz = idx / (Nx * Ny);
        int rem = idx % (Nx * Ny);
        int iy = rem / Nx;
        int ix = rem % Nx;
        
        // Boundary check
        if (ix == 0 || ix == Nx - 1 ||
            iy == 0 || iy == Ny - 1 ||
            iz == 0 || iz == Nz - 1) {
            // Dirichlet 0: x_boundary should be 0, so r = b - 0 = b
            r[idx] = b[idx];
        }
    }
}

void compute_residual_3d(
    CudaContext& ctx,
    const Grid3D& grid,
    DeviceSpan<const real> x,
    DeviceSpan<const real> b,
    DeviceSpan<const real> K,
    DeviceSpan<real> r,
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
    assert(r.size() == n && "r size mismatch");
    
    real dxdx = 1.0 / (grid.dx * grid.dx);  // Assuming isotropic
    
    dim3 block(16, 16);
    int grid_x = (Nx + block.x - 1) / block.x;
    int grid_y = (Ny + block.y - 1) / block.y;
    grid_x = (grid_x < 65535) ? grid_x : 65535;
    grid_y = (grid_y < 65535) ? grid_y : 65535;
    dim3 grid_dim(grid_x, grid_y);
    
    // Interior residual
    residual_interior_kernel<<<grid_dim, block, 0, ctx.cuda_stream()>>>(
        r.data(), x.data(), b.data(), K.data(), dxdx, Nx, Ny, Nz
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
    
    // Boundary residual (Dirichlet 0)
    int block1d = 256;
    int grid1d = (n + block1d - 1) / block1d;
    grid1d = (grid1d < 65535) ? grid1d : 65535;
    residual_boundary_kernel<<<grid1d, block1d, 0, ctx.cuda_stream()>>>(
        r.data(), b.data(), Nx, Ny, Nz
    );
    RWPT_CUDA_CHECK(cudaGetLastError());
}

} // namespace multigrid
} // namespace rwpt
