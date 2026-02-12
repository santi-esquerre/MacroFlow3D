#pragma once

/**
 * @file mg_conventions.cuh
 * @brief Mathematical conventions for multigrid operator and residual computation.
 * 
 * ============================================================================
 * CONTRACT: This file defines the SINGLE SOURCE OF TRUTH for operator semantics
 * ============================================================================
 * 
 * All modules (VarCoeffLaplacian, residual_3d, gsrb_3d, v_cycle) MUST follow
 * these conventions. Any deviation will cause solver divergence.
 */

#include "../../core/Scalar.hpp"
#include "../../core/Grid3D.hpp"

/**
 * @file mg_conventions.cuh
 * @brief Mathematical conventions for multigrid operator and residual computation.
 * 
 * This file documents the EXACT mathematical conventions inherited from the legacy
 * codebase (GSRB_Smooth_up_residual_3D_bien.cu, up_residual_3D.cu, CCMG_V_cycle.cu).
 * DO NOT change these conventions without verifying against legacy behavior.
 * 
 * ============================================================================
 * CRITICAL ASSUMPTION: ISOTROPIC GRIDS (dx = dy = dz)
 * ============================================================================
 * 
 * All kernels in this multigrid implementation assume dx = dy = dz.
 * The spacing is accessed via grid.dx only (grid.dy, grid.dz are ignored).
 * 
 * This matches the legacy convention where:
 *   double h = Ly / (double)Ny;  // Single spacing for all directions
 *   double dxdx = h * h;
 * 
 * Users MUST ensure isotropic grids. Anisotropic grids will produce
 * incorrect results silently.
 * 
 * ============================================================================
 * OPERATOR DEFINITION (Variable-Coefficient Laplacian)
 * ============================================================================
 * 
 * Physical equation: -∇·(K∇h) = f
 * 
 * Discrete operator A in cell-centered finite differences:
 *   (A*h)_C = -sum_6faces( K_face * (h_C - h_neighbor) ) / dx²
 * 
 * where K_face is the HARMONIC MEAN of conductivity:
 *   K_face = 2 / (1/K_C + 1/K_neighbor)
 * 
 * Sign convention: A is POSITIVE DEFINITE (SPD), representing -∇·(K∇).
 * 
 * ============================================================================
 * SCALING WITH dx²
 * ============================================================================
 * 
 * Let dx² = grid.dx * grid.dx (the actual grid spacing squared).
 * 
 * The operator Ax is computed WITHOUT dx² scaling in the stencil, i.e.:
 *   Ax = sum_6faces( K_face * (x_C - x_neighbor) )
 * 
 * Then:
 *   - Residual: r = b - Ax / dx²
 *   - GSRB update: x = (result - rhs * dx²) / aC
 * 
 * where:
 *   - result = sum_neighbors(K_face * x_neighbor)
 *   - aC = sum_6faces(K_face) = diagonal coefficient
 *   - rhs = right-hand side (from problem or previous level)
 * 
 * This convention comes from discretizing: -∇·(K∇h) / dx² = rhs / dx²
 * 
 * ============================================================================
 * BOUNDARY CONDITIONS
 * ============================================================================
 * 
 * Dirichlet: h = h_bc (fixed value)
 *   - In residual: r = 0 at boundary nodes (MG convention)
 *   - In GSRB: no update (skip boundary nodes)
 *   - In stencil: contributes aC += 2*K_C for the Dirichlet direction
 * 
 * Neumann: ∂h/∂n = flux_bc (typically flux_bc = 0)
 *   - Homogeneous Neumann (flux=0): stencil_coeff = 0 for that direction
 *   - Ghost value = interior value (no contribution)
 * 
 * Periodic: h(xmin) = h(xmax), etc.
 *   - Use neighbor from opposite boundary
 *   - Harmonic mean computed normally
 * 
 * ============================================================================
 * LEGACY CONSISTENCY CHECK
 * ============================================================================
 * 
 * Legacy GSRB interior update (GSRB_int kernel):
 *   result = sum(h_neighbor * K_face)
 *   aC = sum(K_face)
 *   h = -(rhs - result/dxdx) / (aC/dxdx)
 *     = (result - rhs*dxdx) / aC       [where dxdx = h*h in legacy]
 * 
 * Legacy residual interior (update_int kernel):
 *   result = -sum( 2*(h_C - h_neighbor) / (1/K_C + 1/K_neighbor) )
 *   r = rhs - result/dxdx
 * 
 * Current implementation (Task 1 compliant):
 *   dx2 = grid.dx * grid.dx
 *   Residual: r = b - Ax / dx2
 *   GSRB: x = (result - b * dx2) / aC
 * 
 * ============================================================================
 * USAGE IN CODE
 * ============================================================================
 * 
 * Always use these variable names for clarity:
 *   - dx2: the actual value dx² (grid.dx * grid.dx)
 *   - inv_dx2: 1/dx² (only in performance-critical inner loops)
 * 
 * Never use ambiguous names like "dxdx" which could mean either dx² or 1/dx².
 * 
 * ============================================================================
 */

namespace macroflow3d {
namespace multigrid {

// Inline helper: compute harmonic mean of two conductivities
__device__ __host__ inline real harmonic_mean_K(real K1, real K2) {
    return 2.0 / (1.0 / K1 + 1.0 / K2);
}

// Inline helper: compute dx² from grid
__device__ __host__ inline real compute_dx2(const Grid3D& grid) {
    return grid.dx * grid.dx;
}

// Inline helper: compute 1/dx² from grid (for optimized kernels)
__device__ __host__ inline real compute_inv_dx2(const Grid3D& grid) {
    return 1.0 / (grid.dx * grid.dx);
}

} // namespace multigrid
} // namespace macroflow3d
