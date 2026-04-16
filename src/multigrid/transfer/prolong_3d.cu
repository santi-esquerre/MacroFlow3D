#include "../../runtime/cuda_check.cuh"
#include "prolong_3d.cuh"
#include <cassert>
#include <cuda_runtime.h>

namespace macroflow3d {
namespace multigrid {

/**
 * @file prolong_3d.cu
 * @brief Piecewise-constant prolongation for cell-centered multigrid
 *
 * Legacy correspondence: transf_operator_3D.cu (prolongation_* kernels)
 *
 * The prolongation operator P transfers corrections from coarse to fine:
 *   x_fine += P * x_coarse
 *
 * For cell-centered MG with factor-2 coarsening, each coarse cell maps to
 * 8 fine cells. The octant-based mapping determines which coarse cell
 * contributes to each fine cell based on parity (ix%2, iy%2, iz%2).
 *
 * Legacy uses separate kernels for interior/faces/edges/vertices to handle
 * boundaries correctly. We replicate this structure for legacy-compatibility.
 */

// ============================================================================
// Interior kernel: cells not touching any boundary
// Iterates over fine cells (1..Nx-2, 1..Ny-2, 1..Nz-2) - legacy convention
// ============================================================================
__global__ void prolong_interior_kernel(real* __restrict__ phiFine,
                                        const real* __restrict__ phiCoarse, int Nx, int Ny,
                                        int Nz // Fine dimensions
) {
    // Legacy: ix iterates 0..Nx-3 then uses (ix+1), (iy+1), (iz+1)
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= Nx - 2 || iy >= Ny - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int stride = Nx * Ny;
    int STRIDE = NX * NY;

    int IX = ix / 2;
    int IY = iy / 2;
    int fx = ix % 2;
    int fy = iy % 2;

    for (int iz = 0; iz < Nz - 2; ++iz) {
        int IZ = iz / 2;
        int fz = iz % 2;

        int in_idx = (ix + 1) + (iy + 1) * Nx + (iz + 1) * stride;
        int IN_IDX = IX + IY * NX + IZ * STRIDE;
        int offset = fx + fy * NX + fz * STRIDE;

        phiFine[in_idx] += phiCoarse[IN_IDX + offset];
    }
}

// ============================================================================
// Face kernels: cells on exactly one boundary face
// ============================================================================

// z = 0 face (bottom)
__global__ void prolong_face_zmin_kernel(real* __restrict__ phiFine,
                                         const real* __restrict__ phiCoarse, int Nx, int Ny,
                                         int Nz) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= Nx - 2 || iy >= Ny - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int iz = 0;
    int IZ = 0;

    int IX = ix / 2;
    int IY = iy / 2;
    int fx = ix % 2;
    int fy = iy % 2;
    // No fz offset for z=0 face

    int in_idx = (ix + 1) + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fx + fy * NX; // No z-offset

    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// z = Nz-1 face (top)
__global__ void prolong_face_zmax_kernel(real* __restrict__ phiFine,
                                         const real* __restrict__ phiCoarse, int Nx, int Ny,
                                         int Nz) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= Nx - 2 || iy >= Ny - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int NZ = Nz / 2;
    int iz = Nz - 1;
    int IZ = iz / 2; // = NZ - 1

    int IX = ix / 2;
    int IY = iy / 2;
    int fx = ix % 2;
    int fy = iy % 2;
    // No fz offset for z=Nz-1 face (would go OOB)

    int in_idx = (ix + 1) + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fx + fy * NX; // No z-offset

    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// y = 0 face (south)
__global__ void prolong_face_ymin_kernel(real* __restrict__ phiFine,
                                         const real* __restrict__ phiCoarse, int Nx, int Ny,
                                         int Nz) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iz = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= Nx - 2 || iz >= Nz - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = 0;
    int IY = 0;

    int IX = ix / 2;
    int IZ = iz / 2;
    int fx = ix % 2;
    int fz = iz % 2;
    // No fy offset for y=0 face

    int in_idx = (ix + 1) + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fx + fz * NX * NY; // No y-offset

    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// y = Ny-1 face (north)
__global__ void prolong_face_ymax_kernel(real* __restrict__ phiFine,
                                         const real* __restrict__ phiCoarse, int Nx, int Ny,
                                         int Nz) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iz = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= Nx - 2 || iz >= Nz - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = Ny - 1;
    int IY = iy / 2;

    int IX = ix / 2;
    int IZ = iz / 2;
    int fx = ix % 2;
    int fz = iz % 2;
    // No fy offset for y=Ny-1 face

    int in_idx = (ix + 1) + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fx + fz * NX * NY; // No y-offset

    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// x = 0 face (west)
__global__ void prolong_face_xmin_kernel(real* __restrict__ phiFine,
                                         const real* __restrict__ phiCoarse, int Nx, int Ny,
                                         int Nz) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    int iz = threadIdx.y + blockIdx.y * blockDim.y;

    if (iy >= Ny - 2 || iz >= Nz - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = 0;
    int IX = 0;

    int IY = iy / 2;
    int IZ = iz / 2;
    int fy = iy % 2;
    int fz = iz % 2;
    // No fx offset for x=0 face

    int in_idx = ix + (iy + 1) * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fy * NX + fz * NX * NY; // No x-offset

    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// x = Nx-1 face (east)
__global__ void prolong_face_xmax_kernel(real* __restrict__ phiFine,
                                         const real* __restrict__ phiCoarse, int Nx, int Ny,
                                         int Nz) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    int iz = threadIdx.y + blockIdx.y * blockDim.y;

    if (iy >= Ny - 2 || iz >= Nz - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = Nx - 1;
    int IX = ix / 2;

    int IY = iy / 2;
    int IZ = iz / 2;
    int fy = iy % 2;
    int fz = iz % 2;
    // No fx offset for x=Nx-1 face

    int in_idx = ix + (iy + 1) * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;
    int offset = fy * NX + fz * NX * NY; // No x-offset

    phiFine[in_idx] += phiCoarse[IN_IDX + offset];
}

// ============================================================================
// Edge kernels: cells on exactly two boundary faces
// ============================================================================

// Edge along X (y=0, z=0)
__global__ void prolong_edge_x_ymin_zmin_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= Nx - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = 0, iz = 0;
    int IY = 0, IZ = 0;
    int IX = ix / 2;
    int fx = ix % 2;

    int in_idx = (ix + 1) + iy * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fx];
}

// Edge along X (y=0, z=Nz-1)
__global__ void prolong_edge_x_ymin_zmax_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= Nx - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = 0, iz = Nz - 1;
    int IY = 0, IZ = iz / 2;
    int IX = ix / 2;
    int fx = ix % 2;

    int in_idx = (ix + 1) + iy * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fx];
}

// Edge along X (y=Ny-1, z=0)
__global__ void prolong_edge_x_ymax_zmin_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= Nx - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = Ny - 1, iz = 0;
    int IY = iy / 2, IZ = 0;
    int IX = ix / 2;
    int fx = ix % 2;

    int in_idx = (ix + 1) + iy * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fx];
}

// Edge along X (y=Ny-1, z=Nz-1)
__global__ void prolong_edge_x_ymax_zmax_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix >= Nx - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int iy = Ny - 1, iz = Nz - 1;
    int IY = iy / 2, IZ = iz / 2;
    int IX = ix / 2;
    int fx = ix % 2;

    int in_idx = (ix + 1) + iy * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fx];
}

// Edge along Y (x=0, z=0)
__global__ void prolong_edge_y_xmin_zmin_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    if (iy >= Ny - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = 0, iz = 0;
    int IX = 0, IZ = 0;
    int IY = iy / 2;
    int fy = iy % 2;

    int in_idx = ix + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fy * NX];
}

// Edge along Y (x=0, z=Nz-1)
__global__ void prolong_edge_y_xmin_zmax_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    if (iy >= Ny - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = 0, iz = Nz - 1;
    int IX = 0, IZ = iz / 2;
    int IY = iy / 2;
    int fy = iy % 2;

    int in_idx = ix + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fy * NX];
}

// Edge along Y (x=Nx-1, z=0)
__global__ void prolong_edge_y_xmax_zmin_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    if (iy >= Ny - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = Nx - 1, iz = 0;
    int IX = ix / 2, IZ = 0;
    int IY = iy / 2;
    int fy = iy % 2;

    int in_idx = ix + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fy * NX];
}

// Edge along Y (x=Nx-1, z=Nz-1)
__global__ void prolong_edge_y_xmax_zmax_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int iy = threadIdx.x + blockIdx.x * blockDim.x;
    if (iy >= Ny - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = Nx - 1, iz = Nz - 1;
    int IX = ix / 2, IZ = iz / 2;
    int IY = iy / 2;
    int fy = iy % 2;

    int in_idx = ix + (iy + 1) * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fy * NX];
}

// Edge along Z (x=0, y=0)
__global__ void prolong_edge_z_xmin_ymin_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    if (iz >= Nz - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = 0, iy = 0;
    int IX = 0, IY = 0;
    int IZ = iz / 2;
    int fz = iz % 2;

    int in_idx = ix + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fz * NX * NY];
}

// Edge along Z (x=0, y=Ny-1)
__global__ void prolong_edge_z_xmin_ymax_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    if (iz >= Nz - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = 0, iy = Ny - 1;
    int IX = 0, IY = iy / 2;
    int IZ = iz / 2;
    int fz = iz % 2;

    int in_idx = ix + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fz * NX * NY];
}

// Edge along Z (x=Nx-1, y=0)
__global__ void prolong_edge_z_xmax_ymin_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    if (iz >= Nz - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = Nx - 1, iy = 0;
    int IX = ix / 2, IY = 0;
    int IZ = iz / 2;
    int fz = iz % 2;

    int in_idx = ix + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fz * NX * NY];
}

// Edge along Z (x=Nx-1, y=Ny-1)
__global__ void prolong_edge_z_xmax_ymax_kernel(real* __restrict__ phiFine,
                                                const real* __restrict__ phiCoarse, int Nx, int Ny,
                                                int Nz) {
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    if (iz >= Nz - 2)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int ix = Nx - 1, iy = Ny - 1;
    int IX = ix / 2, IY = iy / 2;
    int IZ = iz / 2;
    int fz = iz % 2;

    int in_idx = ix + iy * Nx + (iz + 1) * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    phiFine[in_idx] += phiCoarse[IN_IDX + fz * NX * NY];
}

// ============================================================================
// Vertex kernels: 8 corner cells
// ============================================================================

__global__ void prolong_vertex_kernel(real* __restrict__ phiFine,
                                      const real* __restrict__ phiCoarse, int Nx, int Ny, int Nz,
                                      int ix, int iy, int iz // Vertex position in fine grid
) {
    if (threadIdx.x != 0)
        return;

    int NX = Nx / 2;
    int NY = Ny / 2;
    int IX = ix / 2;
    int IY = iy / 2;
    int IZ = iz / 2;

    int in_idx = ix + iy * Nx + iz * Nx * Ny;
    int IN_IDX = IX + IY * NX + IZ * NX * NY;

    // No offset for vertices (all directions are at boundary)
    phiFine[in_idx] += phiCoarse[IN_IDX];
}

// ============================================================================
// Main function: orchestrate all kernels
// ============================================================================

void prolong_3d_add(CudaContext& ctx, const Grid3D& coarse_grid, const Grid3D& fine_grid,
                    DeviceSpan<const real> x_coarse, DeviceSpan<real> x_fine) {
    int Nx = fine_grid.nx;
    int Ny = fine_grid.ny;
    int Nz = fine_grid.nz;

    // Validate dimensions
    assert(coarse_grid.nx == Nx / 2 && "Fine grid must be 2x coarse in x");
    assert(coarse_grid.ny == Ny / 2 && "Fine grid must be 2x coarse in y");
    assert(coarse_grid.nz == Nz / 2 && "Fine grid must be 2x coarse in z");
    assert(x_coarse.size() == coarse_grid.num_cells() && "Coarse buffer size mismatch");
    assert(x_fine.size() == fine_grid.num_cells() && "Fine buffer size mismatch");

    real* phiFine = x_fine.data();
    const real* phiCoarse = x_coarse.data();
    cudaStream_t stream = ctx.cuda_stream();

    // Block sizes
    dim3 block2d(16, 16);
    int block1d = 256;

    // 1. Interior
    {
        int gx = (Nx - 2 + block2d.x - 1) / block2d.x;
        int gy = (Ny - 2 + block2d.y - 1) / block2d.y;
        dim3 grid(gx, gy);
        prolong_interior_kernel<<<grid, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
    }

    // 2. Faces (6 kernels)
    {
        // z-faces (iterate x, y interior)
        int gx = (Nx - 2 + block2d.x - 1) / block2d.x;
        int gy = (Ny - 2 + block2d.y - 1) / block2d.y;
        dim3 grid_xy(gx, gy);
        prolong_face_zmin_kernel<<<grid_xy, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_face_zmax_kernel<<<grid_xy, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);

        // y-faces (iterate x, z interior)
        gx = (Nx - 2 + block2d.x - 1) / block2d.x;
        int gz = (Nz - 2 + block2d.y - 1) / block2d.y;
        dim3 grid_xz(gx, gz);
        prolong_face_ymin_kernel<<<grid_xz, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_face_ymax_kernel<<<grid_xz, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);

        // x-faces (iterate y, z interior)
        gy = (Ny - 2 + block2d.y - 1) / block2d.y;
        gz = (Nz - 2 + block2d.x - 1) / block2d.x;
        dim3 grid_yz(gy, gz);
        prolong_face_xmin_kernel<<<grid_yz, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
        prolong_face_xmax_kernel<<<grid_yz, block2d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz);
    }

    // 3. Edges (12 kernels)
    {
        // X-edges
        int grid_x = (Nx - 2 + block1d - 1) / block1d;
        prolong_edge_x_ymin_zmin_kernel<<<grid_x, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);
        prolong_edge_x_ymin_zmax_kernel<<<grid_x, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);
        prolong_edge_x_ymax_zmin_kernel<<<grid_x, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);
        prolong_edge_x_ymax_zmax_kernel<<<grid_x, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);

        // Y-edges
        int grid_y = (Ny - 2 + block1d - 1) / block1d;
        prolong_edge_y_xmin_zmin_kernel<<<grid_y, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);
        prolong_edge_y_xmin_zmax_kernel<<<grid_y, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);
        prolong_edge_y_xmax_zmin_kernel<<<grid_y, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);
        prolong_edge_y_xmax_zmax_kernel<<<grid_y, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);

        // Z-edges
        int grid_z = (Nz - 2 + block1d - 1) / block1d;
        prolong_edge_z_xmin_ymin_kernel<<<grid_z, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);
        prolong_edge_z_xmin_ymax_kernel<<<grid_z, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);
        prolong_edge_z_xmax_ymin_kernel<<<grid_z, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);
        prolong_edge_z_xmax_ymax_kernel<<<grid_z, block1d, 0, stream>>>(phiFine, phiCoarse, Nx, Ny,
                                                                        Nz);
    }

    // 4. Vertices (8 kernels)
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, 0, 0, 0);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, 0, 0, Nz - 1);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, 0, Ny - 1, 0);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, 0, Ny - 1, Nz - 1);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, Nx - 1, 0, 0);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, Nx - 1, 0, Nz - 1);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, Nx - 1, Ny - 1, 0);
    prolong_vertex_kernel<<<1, 1, 0, stream>>>(phiFine, phiCoarse, Nx, Ny, Nz, Nx - 1, Ny - 1,
                                               Nz - 1);

    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace multigrid
} // namespace macroflow3d
