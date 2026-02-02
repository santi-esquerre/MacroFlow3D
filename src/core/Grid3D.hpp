#pragma once

#include "Scalar.hpp"
#include <cuda_runtime.h>
#include <cstddef>

namespace rwpt {

/**
 * @brief 3D structured grid specification
 * 
 * IMPORTANT: Current implementation assumes ISOTROPIC grids (dx = dy = dz)
 * 
 * This assumption is used throughout the multigrid stack:
 * - Smoother (gsrb_3d.cu): uses grid.dx for all directions
 * - Residual (residual_3d.cu): uses grid.dx for operator scaling
 * - Transfer operators: assume 2:1 coarsening in all directions
 * - Variable-coefficient Laplacian: uses grid.dx for inv_dx2
 * 
 * Legacy code (CCMG_V_cycle.cu) also assumes dx=dy=dz via:
 *   double h = Ly/(double)Ny;  // Uses Ly/Ny for all directions
 *   double dxdx = h*h;
 * 
 * To support anisotropic grids, the following would need modification:
 * - All stencil kernels to use direction-specific spacing
 * - Coarsening ratio per direction
 * - BC helper functions
 * 
 * For now, users should ensure dx == dy == dz when constructing grids.
 */
struct Grid3D {
    int nx, ny, nz;
    real dx, dy, dz;

    // Default constructor
    Grid3D() : nx(0), ny(0), nz(0), dx(0.0), dy(0.0), dz(0.0) {}

    // Constructor with dimensions and spacing
    Grid3D(int nx_, int ny_, int nz_, real dx_, real dy_, real dz_)
        : nx(nx_), ny(ny_), nz(nz_), dx(dx_), dy(dy_), dz(dz_) {}

    // Total number of cells
    size_t num_cells() const {
        return static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    }

    // Linear index: row-major order i + nx*(j + ny*k)
    __host__ __device__
    size_t idx(int i, int j, int k) const {
        return i + nx * (j + ny * k);
    }
};

} // namespace rwpt
