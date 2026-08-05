#pragma once

/**
 * @file negated_operator.cuh
 * @brief Wrapper to negate an operator for CG/PCG
 *
 * Problem: VarCoeffLaplacian produces L = div_h(K grad_h), which is negative
 *          semidefinite under periodic boundary conditions.
 *          CG/PCG require a positive-definite operator after their chosen
 *          boundary or gauge handling.
 *
 * Solution: Wrap L as -L (negate output), making it positive semidefinite.
 *           The solver sees: (-L)*x = -b, which solves L*x = b.
 *
 * Usage:
 *   VarCoeffLaplacian L_neg(...);  // Negative semidefinite
 *   NegatedOperator<VarCoeffLaplacian> A_pos(L_neg);  // Positive semidefinite
 *   pcg_solve(ctx, A_pos, b_negated, x, ...);
 *
 * Important: The RHS must also be negated: if L*x = b, then (-L)*x = -b.
 */

#include "../../core/DeviceSpan.cuh"
#include "../../runtime/CudaContext.cuh"
#include "../blas/blas.cuh"

namespace macroflow3d {
namespace operators {

/**
 * @brief Negate an operator: NegatedOperator.apply(x) = -Operator.apply(x)
 *
 * Template parameter Op must have:
 *   void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const;
 */
template <typename Op> class NegatedOperator {
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
    const Op& op_; // Reference to wrapped operator
};

/**
 * @brief Helper function to create a negated operator
 */
template <typename Op> NegatedOperator<Op> negate_operator(const Op& op) {
    return NegatedOperator<Op>(op);
}

} // namespace operators
} // namespace macroflow3d
