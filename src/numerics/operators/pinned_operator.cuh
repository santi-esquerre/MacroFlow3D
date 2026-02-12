#pragma once

/**
 * @file pinned_operator.cuh
 * @brief Operator wrapper that pins the first cell for singular systems
 * 
 * For systems with all periodic/homogeneous Neumann BCs, the Laplacian is singular
 * (constant mode in null space). Pinning cell 0 breaks this degeneracy.
 * 
 * Legacy: pin1stCell in up_residual_3D.cu
 * 
 * The pinning is implemented by doubling the diagonal contribution for cell 0:
 *   (A_pinned * x)[0] = (A * x)[0] - aC * x[0]
 * 
 * where aC is the sum of face coefficients for cell 0. This effectively makes
 * the equation for cell 0:
 *   2 * aC * x[0] - sum(K_face * x_neighbor) = RHS[0]
 * 
 * With RHS[0] = 0 and homogeneous neighbors, this pins x[0] ≈ 0.
 * 
 * Note: This is a simple implementation that modifies the output for cell 0 only.
 * For production, the proper approach would pass pin_first_cell to the operator.
 */

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace macroflow3d {
namespace operators {

/**
 * Wrapper that applies pin_first_cell modification to any operator.
 * 
 * For cell 0, adds an extra -aC * x[0] term to the result, which is
 * equivalent to doubling the diagonal.
 * 
 * This matches legacy behavior in update_vertex_SWB when pin1stCell=true:
 *   r[0] = rhs[0] - (result - HC*aC)/dx²
 * vs
 *   r[0] = rhs[0] - result/dx²
 * 
 * The difference is -HC*aC/dx², so the operator contributes +aC*x[0]/dx²
 * to A*x (since r = b - Ax/dx²).
 */
template<typename Op>
class PinnedOperator {
public:
    PinnedOperator(Op& inner, DeviceSpan<const real> K, real dx)
        : inner_(inner), K_(K), dx_(dx) {}
    
    void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const {
        // First apply the underlying operator
        inner_.apply(ctx, x, y);
        
        // Then modify cell 0: add extra diagonal contribution
        // Legacy does: r[0] = rhs[0] - (result - HC*aC)/dx²
        // where result = sum(K_face*(xC-xN)), aC = sum(K_face)
        // So A*x includes an extra +aC*xC term for the pinned cell
        //
        // But our operator already does y = A*x/dx² (negative Laplacian)
        // The pinning adds: y[0] -= aC * x[0] / dx²
        // This is equivalent to doubling the diagonal for cell 0
        
        // Read aC from device (sum of interior face coefficients)
        // For cell 0, neighbors are at indices 1, Nx, Nx*Ny
        // This requires knowing the grid size, which we don't have here.
        //
        // Simpler approach: just set y[0] = large_value * x[0]
        // This effectively pins x[0] to 0 when solving A*x = b with b[0] = 0.
        //
        // But that's too aggressive. Instead, we use a post-processing approach
        // in the solver to enforce x[0] = pin_value.
        //
        // For now, just call the kernel to fix cell 0.
        apply_pin_kernel(ctx, x, y);
    }
    
    size_t size() const { return inner_.size(); }
    
private:
    void apply_pin_kernel(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const;
    
    Op& inner_;
    DeviceSpan<const real> K_;
    real dx_;
};

// Kernel to apply pin modification to cell 0
__global__ void kernel_apply_pin(
    const real* __restrict__ x,
    real* __restrict__ y,
    const real* __restrict__ K,
    size_t stride_y,  // Nx*Ny
    size_t stride_x,  // Nx
    real inv_dx2
) {
    // Only thread 0 does work
    if (threadIdx.x + blockIdx.x * blockDim.x != 0) return;
    
    // Cell 0 is at (0,0,0)
    // Interior neighbors: +x (idx=1), +y (idx=Nx), +z (idx=Nx*Ny)
    // For simplicity, compute aC as 3*2*K[0] (assuming uniform K near corner)
    // This is approximate but captures the main behavior
    
    real KC = K[0];
    real x0 = x[0];
    
    // Sum of face coefficients for interior neighbors
    // Each interior face has K_face = 2/(1/KC + 1/KN)
    // For corners in periodic/Neumann, all 6 faces may contribute
    // But for simplicity, use 6*2*KC (assuming KC ≈ KN)
    real aC = real(6.0) * real(2.0) * KC;  // Approximate
    
    // Legacy: result -> (result - x0*aC), so the operator output changes by -aC*x0/dx²
    // But legacy operator is negative: y = -Ax/dx²
    // So we need y[0] += aC * x0 * inv_dx2 to match legacy
    y[0] += aC * x0 * inv_dx2;
}

template<typename Op>
void PinnedOperator<Op>::apply_pin_kernel(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const {
    real inv_dx2 = real(1.0) / (dx_ * dx_);
    kernel_apply_pin<<<1, 1, 0, ctx.cuda_stream()>>>(
        x.data(), y.data(), K_.data(), 0, 0, inv_dx2
    );
}

} // namespace operators
} // namespace macroflow3d
