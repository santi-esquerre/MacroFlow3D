/**
 * @file residual_3d.cu
 * @brief Compute residual r = b - A*x for variable-coefficient 3D Laplacian
 *
 * Legacy correspondence: up_residual_3D.cu
 *
 * This implementation replicates the mathematical semantics of the legacy residual
 * computation with modern C++ structure (templates for BC handling).
 */

#include "../../core/BCSpecDevice.cuh"
#include "../../runtime/cuda_check.cuh"
#include "../common/mg_conventions.cuh"
#include "bc_kernel_tags.cuh"
#include "bc_stencil_helpers.cuh"
#include "residual_3d.cuh"
#include <cassert>
#include <cuda_runtime.h>

namespace macroflow3d {
namespace multigrid {

using namespace bc_helpers;
using namespace bc_tags;

// Interior kernel: branch-free, no BC checks
// Legacy: update_int from up_residual_3D.cu
// r = b - A*x where A is variable-coefficient Laplacian
// Uses harmonic mean for face conductivities
__global__ void residual_interior_kernel(Grid3D grid, DeviceSpan<const real> x,
                                         DeviceSpan<const real> b, DeviceSpan<const real> K,
                                         DeviceSpan<real> r) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    const real inv_dx2 = 1.0 / (grid.dx * grid.dx); // 1/dx² for operator scaling

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // Grid-stride in y
    for (; iy < Ny - 2; iy += blockDim.y * gridDim.y) {
        for (int ix_local = ix; ix_local < Nx - 2; ix_local += blockDim.x * gridDim.x) {
            int stride = Nx * Ny;
            int in_idx = (ix_local + 1) + (iy + 1) * Nx;

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

                // Compute -A*x using harmonic mean (branch-free)
                real Ax = 0.0;
                Ax -= 2.0 * (x_current - x[out_idx + 1]) / (1.0 / K_current + 1.0 / K[out_idx + 1]);
                Ax -=
                    2.0 * (x_current - x[out_idx + Nx]) / (1.0 / K_current + 1.0 / K[out_idx + Nx]);
                Ax -= 2.0 * (x_current - x[out_idx - 1]) / (1.0 / K_current + 1.0 / K[out_idx - 1]);
                Ax -=
                    2.0 * (x_current - x[out_idx - Nx]) / (1.0 / K_current + 1.0 / K[out_idx - Nx]);
                Ax -= 2.0 * (x_current - x_top) / (1.0 / K_current + 1.0 / K_top);
                Ax -= 2.0 * (x_current - x_bottom) / (1.0 / K_current + 1.0 / K_bottom);

                r[out_idx] = b[out_idx] - Ax * inv_dx2;
            }
        }
    }
}

// Face kernel template: handle one face with BC logic
template <Face F>
__global__ void residual_face_kernel(Grid3D grid, BCSpecDevice bc, DeviceSpan<const real> x,
                                     DeviceSpan<const real> b, DeviceSpan<const real> K,
                                     DeviceSpan<real> r) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;

    // Determine face index and loops
    int i, j, k;
    if (on_xmin<F>()) {
        i = 0;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (j >= Ny || k >= Nz)
            return;
    } else if (on_xmax<F>()) {
        i = Nx - 1;
        j = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (j >= Ny || k >= Nz)
            return;
    } else if (on_ymin<F>()) {
        j = 0;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || k >= Nz)
            return;
    } else if (on_ymax<F>()) {
        j = Ny - 1;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        k = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || k >= Nz)
            return;
    } else if (on_zmin<F>()) {
        k = 0;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || j >= Ny)
            return;
    } else { // zmax
        k = Nz - 1;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= Nx || j >= Ny)
            return;
    }

    // Face kernels must NOT touch edges/vertices
    if (on_zmin<F>() || on_zmax<F>()) {
        if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1)
            return;
    }
    if (on_xmin<F>() || on_xmax<F>()) {
        if (j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1)
            return;
    }
    if (on_ymin<F>() || on_ymax<F>()) {
        if (i == 0 || i == Nx - 1 || k == 0 || k == Nz - 1)
            return;
    }

    const int idx = i + Nx * (j + Ny * k);

    // Legacy convention: always compute residual, even for Dirichlet nodes
    // The smoother handles Dirichlet BCs via stencil coefficients (aC += 2*KC)
    {
        real val_xm, coef_xm, rhs_xm;
        real val_xp, coef_xp, rhs_xp;
        real val_ym, coef_ym, rhs_ym;
        real val_yp, coef_yp, rhs_yp;
        real val_zm, coef_zm, rhs_zm;
        real val_zp, coef_zp, rhs_zp;

        neighbor_xminus(i, j, k, grid, bc, x, K, val_xm, coef_xm, rhs_xm);
        neighbor_xplus(i, j, k, grid, bc, x, K, val_xp, coef_xp, rhs_xp);
        neighbor_yminus(i, j, k, grid, bc, x, K, val_ym, coef_ym, rhs_ym);
        neighbor_yplus(i, j, k, grid, bc, x, K, val_yp, coef_yp, rhs_yp);
        neighbor_zminus(i, j, k, grid, bc, x, K, val_zm, coef_zm, rhs_zm);
        neighbor_zplus(i, j, k, grid, bc, x, K, val_zp, coef_zp, rhs_zp);

        const real x_center = x[idx];
        // Use -Ax to match interior kernel sign convention:
        // Interior: Ax -= K*(x_C - x_N), then r = b - Ax/dx² which equals b + (pos sum)/dx²
        const real neg_Ax = -(coef_xm * (x_center - val_xm) + coef_xp * (x_center - val_xp) +
                              coef_ym * (x_center - val_ym) + coef_yp * (x_center - val_yp) +
                              coef_zm * (x_center - val_zm) + coef_zp * (x_center - val_zp));
        const real b_adjusted = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);

        const real dx2 = grid.dx * grid.dx; // dx² for operator scaling
        r[idx] = b_adjusted - neg_Ax / dx2;
    }
}

// Edge kernel template
template <Edge E>
__global__ void residual_edge_kernel(Grid3D grid, BCSpecDevice bc, DeviceSpan<const real> x,
                                     DeviceSpan<const real> b, DeviceSpan<const real> K,
                                     DeviceSpan<real> r) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;

    int i, j, k;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Determine edge coordinates
    if (E == XMIN_YMIN) {
        i = 0;
        j = 0;
        k = tid;
        if (k >= Nz)
            return;
    } else if (E == XMIN_YMAX) {
        i = 0;
        j = Ny - 1;
        k = tid;
        if (k >= Nz)
            return;
    } else if (E == XMAX_YMIN) {
        i = Nx - 1;
        j = 0;
        k = tid;
        if (k >= Nz)
            return;
    } else if (E == XMAX_YMAX) {
        i = Nx - 1;
        j = Ny - 1;
        k = tid;
        if (k >= Nz)
            return;
    } else if (E == XMIN_ZMIN) {
        i = 0;
        k = 0;
        j = tid;
        if (j >= Ny)
            return;
    } else if (E == XMIN_ZMAX) {
        i = 0;
        k = Nz - 1;
        j = tid;
        if (j >= Ny)
            return;
    } else if (E == XMAX_ZMIN) {
        i = Nx - 1;
        k = 0;
        j = tid;
        if (j >= Ny)
            return;
    } else if (E == XMAX_ZMAX) {
        i = Nx - 1;
        k = Nz - 1;
        j = tid;
        if (j >= Ny)
            return;
    } else if (E == YMIN_ZMIN) {
        j = 0;
        k = 0;
        i = tid;
        if (i >= Nx)
            return;
    } else if (E == YMIN_ZMAX) {
        j = 0;
        k = Nz - 1;
        i = tid;
        if (i >= Nx)
            return;
    } else if (E == YMAX_ZMIN) {
        j = Ny - 1;
        k = 0;
        i = tid;
        if (i >= Nx)
            return;
    } else { // YMAX_ZMAX
        j = Ny - 1;
        k = Nz - 1;
        i = tid;
        if (i >= Nx)
            return;
    }

    const int idx = i + Nx * (j + Ny * k);

    // Legacy: always compute residual, even with Dirichlet (no skip)
    {
        real val_xm, coef_xm, rhs_xm;
        real val_xp, coef_xp, rhs_xp;
        real val_ym, coef_ym, rhs_ym;
        real val_yp, coef_yp, rhs_yp;
        real val_zm, coef_zm, rhs_zm;
        real val_zp, coef_zp, rhs_zp;

        neighbor_xminus(i, j, k, grid, bc, x, K, val_xm, coef_xm, rhs_xm);
        neighbor_xplus(i, j, k, grid, bc, x, K, val_xp, coef_xp, rhs_xp);
        neighbor_yminus(i, j, k, grid, bc, x, K, val_ym, coef_ym, rhs_ym);
        neighbor_yplus(i, j, k, grid, bc, x, K, val_yp, coef_yp, rhs_yp);
        neighbor_zminus(i, j, k, grid, bc, x, K, val_zm, coef_zm, rhs_zm);
        neighbor_zplus(i, j, k, grid, bc, x, K, val_zp, coef_zp, rhs_zp);

        const real x_center = x[idx];
        // Use -Ax to match interior kernel sign convention
        const real neg_Ax = -(coef_xm * (x_center - val_xm) + coef_xp * (x_center - val_xp) +
                              coef_ym * (x_center - val_ym) + coef_yp * (x_center - val_yp) +
                              coef_zm * (x_center - val_zm) + coef_zp * (x_center - val_zp));
        const real b_adjusted = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);

        const real dx2 = grid.dx * grid.dx; // dx² for operator scaling
        r[idx] = b_adjusted - neg_Ax / dx2;
    }
}

// Vertex kernel template
// pin1stCell: if true, doubles the diagonal for vertex XMIN_YMIN_ZMIN (cell [0,0,0])
template <Vertex V>
__global__ void residual_vertex_kernel(Grid3D grid, BCSpecDevice bc, DeviceSpan<const real> x,
                                       DeviceSpan<const real> b, DeviceSpan<const real> K,
                                       DeviceSpan<real> r, bool pin1stCell) {
    if (threadIdx.x + blockIdx.x * blockDim.x != 0)
        return;

    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;

    int i, j, k;
    if (V == XMIN_YMIN_ZMIN) {
        i = 0;
        j = 0;
        k = 0;
    } else if (V == XMIN_YMIN_ZMAX) {
        i = 0;
        j = 0;
        k = Nz - 1;
    } else if (V == XMIN_YMAX_ZMIN) {
        i = 0;
        j = Ny - 1;
        k = 0;
    } else if (V == XMIN_YMAX_ZMAX) {
        i = 0;
        j = Ny - 1;
        k = Nz - 1;
    } else if (V == XMAX_YMIN_ZMIN) {
        i = Nx - 1;
        j = 0;
        k = 0;
    } else if (V == XMAX_YMIN_ZMAX) {
        i = Nx - 1;
        j = 0;
        k = Nz - 1;
    } else if (V == XMAX_YMAX_ZMIN) {
        i = Nx - 1;
        j = Ny - 1;
        k = 0;
    } else {
        i = Nx - 1;
        j = Ny - 1;
        k = Nz - 1;
    } // XMAX_YMAX_ZMAX

    const int idx = i + Nx * (j + Ny * k);

    // Legacy: always compute residual, no skip for Dirichlet
    {
        real val_xm, coef_xm, rhs_xm;
        real val_xp, coef_xp, rhs_xp;
        real val_ym, coef_ym, rhs_ym;
        real val_yp, coef_yp, rhs_yp;
        real val_zm, coef_zm, rhs_zm;
        real val_zp, coef_zp, rhs_zp;

        neighbor_xminus(i, j, k, grid, bc, x, K, val_xm, coef_xm, rhs_xm);
        neighbor_xplus(i, j, k, grid, bc, x, K, val_xp, coef_xp, rhs_xp);
        neighbor_yminus(i, j, k, grid, bc, x, K, val_ym, coef_ym, rhs_ym);
        neighbor_yplus(i, j, k, grid, bc, x, K, val_yp, coef_yp, rhs_yp);
        neighbor_zminus(i, j, k, grid, bc, x, K, val_zm, coef_zm, rhs_zm);
        neighbor_zplus(i, j, k, grid, bc, x, K, val_zp, coef_zp, rhs_zp);

        const real x_center = x[idx];

        // Compute sum(K_face * (xC - xN)) = result
        // Note: coef_* are K_face values, val_* are neighbor values
        real result = coef_xm * (x_center - val_xm) + coef_xp * (x_center - val_xp) +
                      coef_ym * (x_center - val_ym) + coef_yp * (x_center - val_yp) +
                      coef_zm * (x_center - val_zm) + coef_zp * (x_center - val_zp);

        // Legacy pin1stCell: double diagonal contribution for cell [0,0,0]
        // Legacy: r = rhs - (result - HC*aC)/dx²  (when pin1stCell is true)
        // This is equivalent to adding aC*xC to result, i.e., doubling diagonal
        if (pin1stCell && V == XMIN_YMIN_ZMIN) {
            real aC = coef_xm + coef_xp + coef_ym + coef_yp + coef_zm + coef_zp;
            result += aC * x_center;
        }

        const real b_adjusted = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);

        const real dx2 = grid.dx * grid.dx; // dx² for operator scaling
        // r = b - A*x, where A*x = -result/dx² (negative Laplacian)
        r[idx] = b_adjusted - (-result / dx2);
    }
}

void compute_residual_3d(CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> x,
                         DeviceSpan<const real> b, DeviceSpan<const real> K, DeviceSpan<real> r,
                         const BCSpec& bc, PinSpec pin) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    const size_t n = grid.num_cells();

    bc.validate();

    assert(x.size() == n && "x size mismatch");
    assert(b.size() == n && "b size mismatch");
    assert(K.size() == n && "K size mismatch");
    assert(r.size() == n && "r size mismatch");

    // Convert BC to device-friendly format
    BCSpecDevice bc_dev = to_device(bc);

    // 1. Interior: branch-free
    dim3 block(16, 16);
    int grid_x = (Nx + block.x - 1) / block.x;
    int grid_y = (Ny + block.y - 1) / block.y;
    grid_x = (grid_x < 65535) ? grid_x : 65535;
    grid_y = (grid_y < 65535) ? grid_y : 65535;
    dim3 grid_dim(grid_x, grid_y);

    residual_interior_kernel<<<grid_dim, block, 0, ctx.cuda_stream()>>>(grid, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // 2. Faces (6 kernels)
    dim3 face_block(16, 16);

    dim3 face_grid_yz((Ny + 15) / 16, (Nz + 15) / 16);
    residual_face_kernel<XMIN>
        <<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_face_kernel<XMAX>
        <<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    dim3 face_grid_xz((Nx + 15) / 16, (Nz + 15) / 16);
    residual_face_kernel<YMIN>
        <<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_face_kernel<YMAX>
        <<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    dim3 face_grid_xy((Nx + 15) / 16, (Ny + 15) / 16);
    residual_face_kernel<ZMIN>
        <<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_face_kernel<ZMAX>
        <<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // 3. Edges (12 kernels)
    int edge_block = 256;

    int edge_grid_z = (Nz + edge_block - 1) / edge_block;
    residual_edge_kernel<XMIN_YMIN>
        <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMIN_YMAX>
        <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMAX_YMIN>
        <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMAX_YMAX>
        <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    int edge_grid_y = (Ny + edge_block - 1) / edge_block;
    residual_edge_kernel<XMIN_ZMIN>
        <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMIN_ZMAX>
        <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMAX_ZMIN>
        <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<XMAX_ZMAX>
        <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    int edge_grid_x = (Nx + edge_block - 1) / edge_block;
    residual_edge_kernel<YMIN_ZMIN>
        <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<YMIN_ZMAX>
        <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<YMAX_ZMIN>
        <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_edge_kernel<YMAX_ZMAX>
        <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // 4. Vertices (8 kernels, pin1stCell only affects XMIN_YMIN_ZMIN)
    residual_vertex_kernel<XMIN_YMIN_ZMIN>
        <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, pin.enabled);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMIN_YMIN_ZMAX>
        <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMIN_YMAX_ZMIN>
        <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMIN_YMAX_ZMAX>
        <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMAX_YMIN_ZMIN>
        <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMAX_YMIN_ZMAX>
        <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMAX_YMAX_ZMIN>
        <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    residual_vertex_kernel<XMAX_YMAX_ZMAX>
        <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, r, false);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace multigrid
} // namespace macroflow3d
