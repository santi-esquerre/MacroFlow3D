#pragma once

/**
 * @file bc_stencil_helpers.cuh
 * @brief Device helpers for computing stencil coefficients with boundary conditions.
 *
 * These helpers compute neighbor contributions for variable-coefficient Laplacian
 * following the conventions documented in ../common/mg_conventions.cuh.
 *
 * Key convention: stencil_coeff is returned WITHOUT dx² scaling.
 * The calling kernel must apply dx² scaling for residual or GSRB update.
 */

#include "../../core/BCSpecDevice.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"

namespace macroflow3d {
namespace multigrid {
namespace bc_helpers {

// Helper to wrap periodic index
__device__ inline int wrap_periodic(int idx, int n) {
    if (idx < 0)
        return n - 1;
    if (idx >= n)
        return 0;
    return idx;
}

// Compute neighbor contribution in X-minus direction
// Returns: ghost_value (for residual), stencil_coeff, rhs_adjustment (for GSRB)
__device__ inline void neighbor_xminus(int i, int j, int k, const Grid3D& grid,
                                       const BCSpecDevice& bc, const DeviceSpan<const real>& x,
                                       const DeviceSpan<const real>& K, real& ghost_value,
                                       real& stencil_coeff, real& rhs_adjust) {
    const int nx = grid.nx, ny = grid.ny;
    const int idx = i + nx * (j + ny * k);

    // Center conductivity
    const real Kc = K[idx];

    if (i == 0) {
        // At xmin boundary
        const auto bc_type = static_cast<BCType>(bc.type[0]); // xmin=0

        if (bc_type == BCType::Periodic) {
            // Wrap to xmax
            const int neighbor_idx = (nx - 1) + nx * (j + ny * k);
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn); // harmonic mean
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh; // Return Kh, not Kh/dx^2
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            // Homogeneous Neumann: no contribution
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Dirichlet BC: ghost contributes ONLY to diagonal, not to result
            // The BC value contribution is in the RHS (built by build_rhs_head)
            // Legacy: aC += 2*KC, result does NOT include bc_val
            ghost_value = 0.0;        // Don't add bc_val*coeff to result
            stencil_coeff = 2.0 * Kc; // Add to diagonal
            rhs_adjust = 0.0;         // Already in RHS
        }
    } else {
        // Interior case
        const int neighbor_idx = (i - 1) + nx * (j + ny * k);
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh; // Return Kh, not Kh/dx^2
        rhs_adjust = 0.0;
    }
}

__device__ inline void neighbor_xplus(int i, int j, int k, const Grid3D& grid,
                                      const BCSpecDevice& bc, const DeviceSpan<const real>& x,
                                      const DeviceSpan<const real>& K, real& ghost_value,
                                      real& stencil_coeff, real& rhs_adjust) {
    const int nx = grid.nx, ny = grid.ny;
    const int idx = i + nx * (j + ny * k);
    const real Kc = K[idx];

    if (i == nx - 1) {
        const auto bc_type = static_cast<BCType>(bc.type[1]); // xmax=1

        if (bc_type == BCType::Periodic) {
            const int neighbor_idx = 0 + nx * (j + ny * k);
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn);
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh;
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Legacy: BC value in RHS, not in result. Only diagonal affected.
            ghost_value = 0.0;
            stencil_coeff = 2.0 * Kc;
            rhs_adjust = 0.0;
        }
    } else {
        const int neighbor_idx = (i + 1) + nx * (j + ny * k);
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh;
        rhs_adjust = 0.0;
    }
}

__device__ inline void neighbor_yminus(int i, int j, int k, const Grid3D& grid,
                                       const BCSpecDevice& bc, const DeviceSpan<const real>& x,
                                       const DeviceSpan<const real>& K, real& ghost_value,
                                       real& stencil_coeff, real& rhs_adjust) {
    const int nx = grid.nx, ny = grid.ny;
    const int idx = i + nx * (j + ny * k);
    const real Kc = K[idx];

    if (j == 0) {
        const auto bc_type = static_cast<BCType>(bc.type[2]); // ymin=2

        if (bc_type == BCType::Periodic) {
            const int neighbor_idx = i + nx * ((ny - 1) + ny * k);
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn);
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh;
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Legacy: BC value in RHS, not in result. Only diagonal affected.
            ghost_value = 0.0;
            stencil_coeff = 2.0 * Kc;
            rhs_adjust = 0.0;
        }
    } else {
        const int neighbor_idx = i + nx * ((j - 1) + ny * k);
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh;
        rhs_adjust = 0.0;
    }
}

__device__ inline void neighbor_yplus(int i, int j, int k, const Grid3D& grid,
                                      const BCSpecDevice& bc, const DeviceSpan<const real>& x,
                                      const DeviceSpan<const real>& K, real& ghost_value,
                                      real& stencil_coeff, real& rhs_adjust) {
    const int nx = grid.nx, ny = grid.ny;
    const int idx = i + nx * (j + ny * k);
    const real Kc = K[idx];

    if (j == ny - 1) {
        const auto bc_type = static_cast<BCType>(bc.type[3]); // ymax=3

        if (bc_type == BCType::Periodic) {
            const int neighbor_idx = i + nx * (0 + ny * k);
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn);
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh;
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Legacy: BC value in RHS, not in result. Only diagonal affected.
            ghost_value = 0.0;
            stencil_coeff = 2.0 * Kc;
            rhs_adjust = 0.0;
        }
    } else {
        const int neighbor_idx = i + nx * ((j + 1) + ny * k);
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh;
        rhs_adjust = 0.0;
    }
}

__device__ inline void neighbor_zminus(int i, int j, int k, const Grid3D& grid,
                                       const BCSpecDevice& bc, const DeviceSpan<const real>& x,
                                       const DeviceSpan<const real>& K, real& ghost_value,
                                       real& stencil_coeff, real& rhs_adjust) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int idx = i + nx * (j + ny * k);
    const real Kc = K[idx];

    if (k == 0) {
        const auto bc_type = static_cast<BCType>(bc.type[4]); // zmin=4

        if (bc_type == BCType::Periodic) {
            const int neighbor_idx = i + nx * (j + ny * (nz - 1));
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn);
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh;
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Legacy: BC value in RHS, not in result. Only diagonal affected.
            ghost_value = 0.0;
            stencil_coeff = 2.0 * Kc;
            rhs_adjust = 0.0;
        }
    } else {
        const int neighbor_idx = i + nx * (j + ny * (k - 1));
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh;
        rhs_adjust = 0.0;
    }
}

__device__ inline void neighbor_zplus(int i, int j, int k, const Grid3D& grid,
                                      const BCSpecDevice& bc, const DeviceSpan<const real>& x,
                                      const DeviceSpan<const real>& K, real& ghost_value,
                                      real& stencil_coeff, real& rhs_adjust) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int idx = i + nx * (j + ny * k);
    const real Kc = K[idx];

    if (k == nz - 1) {
        const auto bc_type = static_cast<BCType>(bc.type[5]); // zmax=5

        if (bc_type == BCType::Periodic) {
            const int neighbor_idx = i + nx * (j + ny * 0);
            const real Kn = K[neighbor_idx];
            const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn);
            ghost_value = x[neighbor_idx];
            stencil_coeff = Kh;
            rhs_adjust = 0.0;
        } else if (bc_type == BCType::Neumann) {
            ghost_value = x[idx];
            stencil_coeff = 0.0;
            rhs_adjust = 0.0;
        } else { // Dirichlet
            // Legacy: BC value in RHS, not in result. Only diagonal affected.
            ghost_value = 0.0;
            stencil_coeff = 2.0 * Kc;
            rhs_adjust = 0.0;
        }
    } else {
        const int neighbor_idx = i + nx * (j + ny * (k + 1));
        const real Kn = K[neighbor_idx];
        const real Kh = 2.0 / (1.0 / Kc + 1.0 / Kn);
        ghost_value = x[neighbor_idx];
        stencil_coeff = Kh;
        rhs_adjust = 0.0;
    }
}

} // namespace bc_helpers
} // namespace multigrid
} // namespace macroflow3d
