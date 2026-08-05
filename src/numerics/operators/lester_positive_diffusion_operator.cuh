#pragma once

/**
 * @file lester_positive_diffusion_operator.cuh
 * @brief Positive periodic diffusion operator for Lester streamfunctions.
 * @ingroup numerics_operators
 *
 * Defines the Lester linear operator on a cell-centered, triply periodic grid:
 *
 *   A(q) u = -div_h(q grad_h u).
 *
 * The cell-centered coefficient q is non-owning and must remain valid while
 * this operator is used.  apply() delegates to the legacy matrix-free
 * div_h(q grad_h) kernel and negates its output on the supplied CUDA stream;
 * it performs no allocation or host-device synchronization.
 */

#include "negated_operator.cuh"
#include "varcoeff_laplacian.cuh"

namespace macroflow3d {
namespace operators {

/**
 * For positive q, this is the positive-semidefinite Lester operator
 * A(q) = -div_h(q grad_h) with periodic boundaries in all three directions
 * and no pinned row.
 */
class LesterPositiveDiffusionOperator {
  public:
    LesterPositiveDiffusionOperator(const Grid3D& grid, DeviceSpan<const real> q)
        : div_q_grad_(grid, q, triply_periodic_bc()) {}

    void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const {
        negate_operator(div_q_grad_).apply(ctx, x, y);
    }

    const Grid3D& grid() const { return div_q_grad_.grid(); }
    size_t size() const { return div_q_grad_.size(); }

  private:
    static BCSpec triply_periodic_bc() {
        BCSpec bc;
        bc.xmin = BCFace(BCType::Periodic, real(0.0));
        bc.xmax = BCFace(BCType::Periodic, real(0.0));
        bc.ymin = BCFace(BCType::Periodic, real(0.0));
        bc.ymax = BCFace(BCType::Periodic, real(0.0));
        bc.zmin = BCFace(BCType::Periodic, real(0.0));
        bc.zmax = BCFace(BCType::Periodic, real(0.0));
        return bc;
    }

    VarCoeffLaplacian div_q_grad_;
};

} // namespace operators
} // namespace macroflow3d
