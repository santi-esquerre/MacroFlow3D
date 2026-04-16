#include "../../runtime/cuda_check.cuh"
#include "poisson3d_operator.cuh"
#include <cassert>

namespace macroflow3d {
namespace operators {

__global__ void poisson3d_kernel(const real* x, real* y, int nx, int ny, int nz, real dx, real dy,
                                 real dz, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Coefficients for 7-point stencil: -Δ (SPD)
    // Diagonal positive, off-diagonal negative
    real cx = 1.0 / (dx * dx);
    real cy = 1.0 / (dy * dy);
    real cz = 1.0 / (dz * dz);
    real cc = 2.0 * (cx + cy + cz); // Positive diagonal

    for (size_t linear_idx = idx; linear_idx < n; linear_idx += stride) {
        // Convert linear index to (i, j, k)
        int i = linear_idx % nx;
        int j = (linear_idx / nx) % ny;
        int k = linear_idx / (nx * ny);

        real center = x[linear_idx];
        real result = cc * center;

        // X-direction neighbors (negative for -Δ)
        if (i > 0) {
            result -= cx * x[linear_idx - 1];
        }
        if (i < nx - 1) {
            result -= cx * x[linear_idx + 1];
        }

        // Y-direction neighbors (negative for -Δ)
        if (j > 0) {
            result -= cy * x[linear_idx - nx];
        }
        if (j < ny - 1) {
            result -= cy * x[linear_idx + nx];
        }

        // Z-direction neighbors (negative for -Δ)
        if (k > 0) {
            result -= cz * x[linear_idx - nx * ny];
        }
        if (k < nz - 1) {
            result -= cz * x[linear_idx + nx * ny];
        }

        y[linear_idx] = result;
    }
}

void Poisson3DOperator::apply(CudaContext& ctx, DeviceSpan<const real> x,
                              DeviceSpan<real> y) const {

    size_t n = grid.num_cells();
    assert(x.size() >= n && y.size() >= n && "Operator apply: size mismatch");
    if (x.size() < n || y.size() < n)
        return;

    const int block_size = 256;
    const int max_blocks = 65535;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);

    poisson3d_kernel<<<grid_size, block_size, 0, ctx.cuda_stream()>>>(
        x.data(), y.data(), grid.nx, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz, n);

    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace operators
} // namespace macroflow3d
