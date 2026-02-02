/**
 * @file varcoeff_laplacian.cu
 * @brief Variable-coefficient Laplacian operator implementation
 * 
 * This computes y = A*x where A is the NEGATIVE discrete variable-coefficient Laplacian:
 *   (A*x)_C = -sum_faces( K_face * (x_C - x_neighbor) ) / dx²
 * 
 * K_face is the harmonic mean: K_face = 2 / (1/K_C + 1/K_neighbor)
 * 
 * This matches the legacy stencil_head operator which also produces a NEGATIVE operator.
 * The negative sign ensures CG solves the same system as MG.
 * 
 * For Dirichlet BCs: The BC value contribution goes to RHS (in build_rhs_head),
 * the operator contributes only the diagonal term: -2*KC*xC
 */

#include "varcoeff_laplacian.cuh"
#include "../../runtime/cuda_check.cuh"
#include "../../core/BCSpecDevice.cuh"
#include <cassert>

namespace rwpt {
namespace operators {

VarCoeffLaplacian::VarCoeffLaplacian(
    const Grid3D& grid,
    DeviceSpan<const real> K,
    const BCSpec& bc,
    PinSpec pin
) : grid_(grid), K_(K), bc_(bc), pin_(pin) {
    assert(K.size() == grid.num_cells() && "K field size must match grid");
    bc_.validate();
    // Legacy pin always uses cell [0,0,0] (index 0)
}

// ============================================================================
// Interior kernel: cells not on any boundary
// Produces y = -∇·(K∇x) discretized as -sum_faces(K_face*(xC-xN))/dx²
// Sign convention: NEGATIVE Laplacian, consistent with MG smoother
// ============================================================================
__global__ void varcoeff_apply_interior_kernel(
    const real* __restrict__ x,
    const real* __restrict__ K,
    real* __restrict__ y,
    int Nx, int Ny, int Nz,
    real inv_dx2
) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix >= Nx - 2 || iy >= Ny - 2) return;
    
    int stride = Nx * Ny;
    
    for (int iz = 1; iz < Nz - 1; ++iz) {
        int idx = (ix + 1) + (iy + 1) * Nx + iz * stride;
        
        real xC = x[idx];
        real KC = K[idx];
        
        // Harmonic mean for each face
        real Kxm = 2.0 / (1.0/KC + 1.0/K[idx - 1]);
        real Kxp = 2.0 / (1.0/KC + 1.0/K[idx + 1]);
        real Kym = 2.0 / (1.0/KC + 1.0/K[idx - Nx]);
        real Kyp = 2.0 / (1.0/KC + 1.0/K[idx + Nx]);
        real Kzm = 2.0 / (1.0/KC + 1.0/K[idx - stride]);
        real Kzp = 2.0 / (1.0/KC + 1.0/K[idx + stride]);
        
        // Compute sum(K_face*(xC-xN)) 
        // NEGATIVE Laplacian: y = -sum(K_face*(xC-xN))/dx²
        // This matches legacy stencil_head which produces -2*sum((xC-xN)/(1/KC+1/KN))/dx²
        real Ax = Kxm * (xC - x[idx - 1]) +
                  Kxp * (xC - x[idx + 1]) +
                  Kym * (xC - x[idx - Nx]) +
                  Kyp * (xC - x[idx + Nx]) +
                  Kzm * (xC - x[idx - stride]) +
                  Kzp * (xC - x[idx + stride]);
        
        // NEGATIVE: matches legacy operator sign
        y[idx] = -Ax * inv_dx2;
    }
}

// ============================================================================
// Boundary kernel: handles all boundary cells with BC logic
// This is simpler than separate face/edge/vertex kernels since apply()
// doesn't need red-black ordering
// ============================================================================
__global__ void varcoeff_apply_boundary_kernel(
    const real* __restrict__ x,
    const real* __restrict__ K,
    real* __restrict__ y,
    int Nx, int Ny, int Nz,
    real inv_dx2,
    BCSpecDevice bc,
    bool pin1stCell
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_cells = Nx * Ny * Nz;
    
    for (int idx = tid; idx < total_cells; idx += blockDim.x * gridDim.x) {
        // Convert to 3D index
        int i = idx % Nx;
        int j = (idx / Nx) % Ny;
        int k = idx / (Nx * Ny);
        
        // Skip interior cells (handled by interior kernel)
        bool is_boundary = (i == 0 || i == Nx-1 || j == 0 || j == Ny-1 || k == 0 || k == Nz-1);
        if (!is_boundary) continue;
        
        real xC = x[idx];
        real KC = K[idx];
        real Ax = 0.0;
        real aC = 0.0;  // Diagonal coefficient accumulator for pin
        
        // X-minus neighbor
        if (i > 0) {
            int n_idx = idx - 1;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            // x = 0 boundary
            auto bc_type = static_cast<BCType>(bc.type[0]);
            if (bc_type == BCType::Periodic) {
                int n_idx = (Nx - 1) + j * Nx + k * Nx * Ny;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
                // Note: periodic neighbors do NOT add to aC for pin
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                // result -= 2*KC*xC (not 2*KC*(xC - bc_val))
                Ax += 2.0 * KC * xC;
            }
            // Neumann: no contribution
        }
        
        // X-plus neighbor
        if (i < Nx - 1) {
            int n_idx = idx + 1;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            auto bc_type = static_cast<BCType>(bc.type[1]);
            if (bc_type == BCType::Periodic) {
                int n_idx = 0 + j * Nx + k * Nx * Ny;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                Ax += 2.0 * KC * xC;
            }
        }
        
        // Y-minus neighbor
        if (j > 0) {
            int n_idx = idx - Nx;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            auto bc_type = static_cast<BCType>(bc.type[2]);
            if (bc_type == BCType::Periodic) {
                int n_idx = i + (Ny - 1) * Nx + k * Nx * Ny;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                Ax += 2.0 * KC * xC;
            }
        }
        
        // Y-plus neighbor
        if (j < Ny - 1) {
            int n_idx = idx + Nx;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            auto bc_type = static_cast<BCType>(bc.type[3]);
            if (bc_type == BCType::Periodic) {
                int n_idx = i + 0 * Nx + k * Nx * Ny;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                Ax += 2.0 * KC * xC;
            }
        }
        
        // Z-minus neighbor
        int stride = Nx * Ny;
        if (k > 0) {
            int n_idx = idx - stride;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            auto bc_type = static_cast<BCType>(bc.type[4]);
            if (bc_type == BCType::Periodic) {
                int n_idx = i + j * Nx + (Nz - 1) * stride;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                Ax += 2.0 * KC * xC;
            }
        }
        
        // Z-plus neighbor
        if (k < Nz - 1) {
            int n_idx = idx + stride;
            real Kn = K[n_idx];
            real Kh = 2.0 / (1.0/KC + 1.0/Kn);
            Ax += Kh * (xC - x[n_idx]);
            aC += Kh;
        } else {
            auto bc_type = static_cast<BCType>(bc.type[5]);
            if (bc_type == BCType::Periodic) {
                int n_idx = i + j * Nx + 0 * stride;
                real Kn = K[n_idx];
                real Kh = 2.0 / (1.0/KC + 1.0/Kn);
                Ax += Kh * (xC - x[n_idx]);
            } else if (bc_type == BCType::Dirichlet) {
                // Legacy convention: BC value is in RHS, operator only contributes diagonal
                Ax += 2.0 * KC * xC;
            }
        }
        
        // Legacy pin1stCell: double the diagonal for cell [0,0,0]
        // This is equivalent to adding aC*xC to the result
        // Legacy: if(pin1stCell) H_output[0] = -2.0*(result+aC*HC)/dx²
        if (pin1stCell && idx == 0) {
            Ax += aC * xC;
        }
        
        // NEGATIVE: matches legacy operator sign
        y[idx] = -Ax * inv_dx2;
    }
}

void VarCoeffLaplacian::apply(
    CudaContext& ctx,
    DeviceSpan<const real> x,
    DeviceSpan<real> y
) const {
    int Nx = grid_.nx;
    int Ny = grid_.ny;
    int Nz = grid_.nz;
    size_t n = grid_.num_cells();
    
    assert(x.size() == n && "x size mismatch");
    assert(y.size() == n && "y size mismatch");
    
    real inv_dx2 = 1.0 / (grid_.dx * grid_.dx);
    BCSpecDevice bc_dev = to_device(bc_);
    
    // 1. Interior cells
    {
        dim3 block(16, 16);
        int gx = (Nx - 2 + block.x - 1) / block.x;
        int gy = (Ny - 2 + block.y - 1) / block.y;
        dim3 grid(gx, gy);
        
        varcoeff_apply_interior_kernel<<<grid, block, 0, ctx.cuda_stream()>>>(
            x.data(), K_.data(), y.data(), Nx, Ny, Nz, inv_dx2
        );
        RWPT_CUDA_CHECK(cudaGetLastError());
    }
    
    // 2. Boundary cells (includes pin logic via diagonal doubling for cell [0,0,0])
    {
        int block = 256;
        int grid = (n + block - 1) / block;
        if (grid > 65535) grid = 65535;
        
        varcoeff_apply_boundary_kernel<<<grid, block, 0, ctx.cuda_stream()>>>(
            x.data(), K_.data(), y.data(), Nx, Ny, Nz, inv_dx2, bc_dev, pin_.enabled
        );
        RWPT_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace operators
} // namespace rwpt
