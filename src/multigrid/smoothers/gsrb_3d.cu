/**
 * @file gsrb_3d.cu
 * @brief Gauss-Seidel Red-Black smoother for variable-coefficient 3D Laplacian
 *
 * Legacy correspondence: GSRB_Smooth_up_residual_3D_bien.cu
 *
 * This implementation replicates the mathematical semantics of the legacy "bien" variant
 * while using modern C++ (templates, constexpr) instead of C preprocessor macros.
 *
 * See ../LEGACY_VARIANTS.md for details on legacy variant history.
 */

#include "../../core/BCSpecDevice.cuh"
#include "../../runtime/cuda_check.cuh"
#include "../common/mg_conventions.cuh"
#include "bc_kernel_tags.cuh"
#include "bc_stencil_helpers.cuh"
#include "gsrb_3d.cuh"
#include <cassert>
#include <cuda_runtime.h>

namespace macroflow3d {
namespace multigrid {

using namespace bc_helpers;
using namespace bc_tags;

// Interior kernel: branch-free red-black GSRB
__global__ void gsrb_interior_kernel(Grid3D grid, DeviceSpan<real> x, DeviceSpan<const real> b,
                                     DeviceSpan<const real> K, bool is_red) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    const real dx2 = grid.dx * grid.dx; // dx² for RHS scaling

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    for (; iy < Ny - 2; iy += blockDim.y * gridDim.y) {
        for (int ix_local = ix; ix_local < Nx - 2; ix_local += blockDim.x * gridDim.x) {
            int stride = Nx * Ny;

            for (int iz = 1; iz < Nz - 1; ++iz) {
                int ix_abs = ix_local + 1;
                int iy_abs = iy + 1;
                int color_sum = ix_abs + iy_abs + iz;

                if ((is_red && (color_sum % 2 != 0)) || (!is_red && (color_sum % 2 == 0))) {
                    continue;
                }

                int idx = ix_abs + iy_abs * Nx + iz * stride;

                real KC = K[idx];
                real K_xp = 2.0 / (1.0 / KC + 1.0 / K[idx + 1]);
                real K_xm = 2.0 / (1.0 / KC + 1.0 / K[idx - 1]);
                real K_yp = 2.0 / (1.0 / KC + 1.0 / K[idx + Nx]);
                real K_ym = 2.0 / (1.0 / KC + 1.0 / K[idx - Nx]);
                real K_zp = 2.0 / (1.0 / KC + 1.0 / K[idx + stride]);
                real K_zm = 2.0 / (1.0 / KC + 1.0 / K[idx - stride]);

                real result = x[idx + 1] * K_xp;
                result += x[idx - 1] * K_xm;
                result += x[idx + Nx] * K_yp;
                result += x[idx - Nx] * K_ym;
                result += x[idx + stride] * K_zp;
                result += x[idx - stride] * K_zm;

                real aC = K_xp + K_xm + K_yp + K_ym + K_zp + K_zm;
                // Legacy: h = -(rhs - result/dx²) / (aC/dx²) = (result - rhs*dx²) / aC
                x[idx] = (result - b[idx] * dx2) / aC;
            }
        }
    }
}

// Face kernel template
template <Face F>
__global__ void gsrb_face_kernel(Grid3D grid, BCSpecDevice bc, DeviceSpan<real> x,
                                 DeviceSpan<const real> b, DeviceSpan<const real> K, bool is_red) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;

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
    } else {
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

    // Color check
    int color_sum = i + j + k;
    if ((is_red && (color_sum % 2 != 0)) || (!is_red && (color_sum % 2 == 0))) {
        return;
    }

    const int idx = i + Nx * (j + Ny * k);

    // Compute stencil and update (legacy: no skip for Dirichlet)
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

    const real dx2 = grid.dx * grid.dx; // dx² for RHS scaling
    real result = coef_xm * val_xm + coef_xp * val_xp + coef_ym * val_ym + coef_yp * val_yp +
                  coef_zm * val_zm + coef_zp * val_zp;

    real aC = coef_xm + coef_xp + coef_ym + coef_yp + coef_zm + coef_zp;
    real b_adj = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);

    // Legacy: h = -(rhs - result/dx²) / (aC/dx²) = (result - rhs*dx²) / aC
    x[idx] = (result - b_adj * dx2) / aC;
}

// Edge kernel template
template <Edge E>
__global__ void gsrb_edge_kernel(Grid3D grid, BCSpecDevice bc, DeviceSpan<real> x,
                                 DeviceSpan<const real> b, DeviceSpan<const real> K, bool is_red) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;

    int i, j, k;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

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
    } else {
        j = Ny - 1;
        k = Nz - 1;
        i = tid;
        if (i >= Nx)
            return;
    }

    int color_sum = i + j + k;
    if ((is_red && (color_sum % 2 != 0)) || (!is_red && (color_sum % 2 == 0))) {
        return;
    }

    const int idx = i + Nx * (j + Ny * k);

    // Legacy: always update, even with Dirichlet (no skip)
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

    const real dx2 = grid.dx * grid.dx; // dx² for RHS scaling
    real result = coef_xm * val_xm + coef_xp * val_xp + coef_ym * val_ym + coef_yp * val_yp +
                  coef_zm * val_zm + coef_zp * val_zp;

    real aC = coef_xm + coef_xp + coef_ym + coef_yp + coef_zm + coef_zp;
    real b_adj = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);

    // Legacy: h = -(rhs - result/dx²) / (aC/dx²) = (result - rhs*dx²) / aC
    x[idx] = (result - b_adj * dx2) / aC;
}

// Vertex kernel template
// pin1stCell: if true, doubles the diagonal for vertex XMIN_YMIN_ZMIN (cell [0,0,0])
template <Vertex V>
__global__ void gsrb_vertex_kernel(Grid3D grid, BCSpecDevice bc, DeviceSpan<real> x,
                                   DeviceSpan<const real> b, DeviceSpan<const real> K, bool is_red,
                                   bool pin1stCell) {
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
    }

    int color_sum = i + j + k;
    if ((is_red && (color_sum % 2 != 0)) || (!is_red && (color_sum % 2 == 0))) {
        return;
    }

    const int idx = i + Nx * (j + Ny * k);

    // Legacy: always update, even with Dirichlet (no skip)
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

    const real dx2 = grid.dx * grid.dx; // dx² for RHS scaling
    real result = coef_xm * val_xm + coef_xp * val_xp + coef_ym * val_ym + coef_yp * val_yp +
                  coef_zm * val_zm + coef_zp * val_zp;

    real aC = coef_xm + coef_xp + coef_ym + coef_yp + coef_zm + coef_zp;
    real b_adj = b[idx] + (rhs_xm + rhs_xp + rhs_ym + rhs_yp + rhs_zm + rhs_zp);

    // Legacy pin1stCell: double diagonal for cell [0,0,0] to break singular null space
    if (pin1stCell && V == XMIN_YMIN_ZMIN) {
        aC *= 2.0;
    }

    // Legacy: h = -(rhs - result/dx²) / (aC/dx²) = (result - rhs*dx²) / aC
    x[idx] = (result - b_adj * dx2) / aC;
}

void gsrb_smooth_3d(CudaContext& ctx, const Grid3D& grid, DeviceSpan<real> x,
                    DeviceSpan<const real> b, DeviceSpan<const real> K, int num_iters,
                    const BCSpec& bc, PinSpec pin) {
    const int Nx = grid.nx, Ny = grid.ny, Nz = grid.nz;
    const size_t n = grid.num_cells();

    bc.validate();

    assert(x.size() == n && "x size mismatch");
    assert(b.size() == n && "b size mismatch");
    assert(K.size() == n && "K size mismatch");

    BCSpecDevice bc_dev = to_device(bc);

    dim3 block(16, 16);
    int grid_x = (Nx + block.x - 1) / block.x;
    int grid_y = (Ny + block.y - 1) / block.y;
    grid_x = (grid_x < 65535) ? grid_x : 65535;
    grid_y = (grid_y < 65535) ? grid_y : 65535;
    dim3 grid_dim(grid_x, grid_y);

    dim3 face_block(16, 16);
    int edge_block = 256;

    for (int iter = 0; iter < num_iters; ++iter) {
        // RED pass
        gsrb_interior_kernel<<<grid_dim, block, 0, ctx.cuda_stream()>>>(grid, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        // Faces RED
        dim3 face_grid_yz((Ny + 15) / 16, (Nz + 15) / 16);
        gsrb_face_kernel<XMIN>
            <<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<XMAX>
            <<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        dim3 face_grid_xz((Nx + 15) / 16, (Nz + 15) / 16);
        gsrb_face_kernel<YMIN>
            <<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<YMAX>
            <<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        dim3 face_grid_xy((Nx + 15) / 16, (Ny + 15) / 16);
        gsrb_face_kernel<ZMIN>
            <<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<ZMAX>
            <<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        // Edges RED
        int edge_grid_z = (Nz + edge_block - 1) / edge_block;
        gsrb_edge_kernel<XMIN_YMIN>
            <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMIN_YMAX>
            <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_YMIN>
            <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_YMAX>
            <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        int edge_grid_y = (Ny + edge_block - 1) / edge_block;
        gsrb_edge_kernel<XMIN_ZMIN>
            <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMIN_ZMAX>
            <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_ZMIN>
            <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_ZMAX>
            <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        int edge_grid_x = (Nx + edge_block - 1) / edge_block;
        gsrb_edge_kernel<YMIN_ZMIN>
            <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMIN_ZMAX>
            <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMAX_ZMIN>
            <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMAX_ZMAX>
            <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        // Vertices RED (pin1stCell only affects XMIN_YMIN_ZMIN)
        gsrb_vertex_kernel<XMIN_YMIN_ZMIN>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, pin.enabled);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMIN_ZMAX>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMAX_ZMIN>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMAX_ZMAX>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMIN_ZMIN>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMIN_ZMAX>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMAX_ZMIN>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMAX_ZMAX>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, true, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        // BLACK pass (same structure, is_red=false)
        gsrb_interior_kernel<<<grid_dim, block, 0, ctx.cuda_stream()>>>(grid, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        gsrb_face_kernel<XMIN>
            <<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<XMAX>
            <<<face_grid_yz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<YMIN>
            <<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<YMAX>
            <<<face_grid_xz, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<ZMIN>
            <<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_face_kernel<ZMAX>
            <<<face_grid_xy, face_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        gsrb_edge_kernel<XMIN_YMIN>
            <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMIN_YMAX>
            <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_YMIN>
            <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_YMAX>
            <<<edge_grid_z, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMIN_ZMIN>
            <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMIN_ZMAX>
            <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_ZMIN>
            <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<XMAX_ZMAX>
            <<<edge_grid_y, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMIN_ZMIN>
            <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMIN_ZMAX>
            <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMAX_ZMIN>
            <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_edge_kernel<YMAX_ZMAX>
            <<<edge_grid_x, edge_block, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        gsrb_vertex_kernel<XMIN_YMIN_ZMIN>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, pin.enabled);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMIN_ZMAX>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMAX_ZMIN>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMIN_YMAX_ZMAX>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMIN_ZMIN>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMIN_ZMAX>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMAX_ZMIN>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        gsrb_vertex_kernel<XMAX_YMAX_ZMAX>
            <<<1, 1, 0, ctx.cuda_stream()>>>(grid, bc_dev, x, b, K, false, false);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace multigrid
} // namespace macroflow3d
