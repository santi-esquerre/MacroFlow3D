#pragma once

/**
 * @file negated_operator.cuh
 * @brief Wrapper to negate an operator for CG/PCG
 * 
 * Problem: VarCoeffLaplacian produces A = -∇·(K∇·), which is NEGATIVE definite.
 *          CG/PCG require a POSITIVE definite operator.
 * 
 * Solution: Wrap A as -A (negate output), making it SPD.
 *           The solver sees: (-A)*x = -b, which solves A*x = b.
 * 
 * Usage:
 *   VarCoeffLaplacian A_neg(...);  // Negative definite
 *   NegatedOperator<VarCoeffLaplacian> A_pos(A_neg);  // Positive definite
 *   pcg_solve(ctx, A_pos, b_negated, x, ...);
 * 
 * Important: The RHS must also be negated: if A*x = b, then (-A)*x = -b.
 */

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../blas/blas.cuh"

namespace macroflow3d {
namespace operators {

/**
 * @brief Negate an operator: NegatedOperator.apply(x) = -Operator.apply(x)
 * 
 * Template parameter Op must have:
 *   void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const;
 */
template<typename Op>
class NegatedOperator {
public:
    explicit NegatedOperator(const Op& op) : op_(op) {}
    
    /**
     * @brief Apply negated operator: y = -A*x
     */
    void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const {
        // First, apply the original operator: y = A*x
        op_.apply(ctx, x, y);
        
        // Then negate: y = -y
        blas::scal(ctx, y, real(-1.0));
    }
    
private:
    const Op& op_;  // Reference to wrapped operator
};

/**
 * @brief Helper function to create a negated operator
 */
template<typename Op>
NegatedOperator<Op> negate_operator(const Op& op) {
    return NegatedOperator<Op>(op);
}

} // namespace operators
} // namespace macroflow3d
