#include "ShiftedJacobianOperator.cuh"

#include "../../numerics/blas/axpy.cuh"
#include "../../numerics/operators/lester_positive_diffusion_operator.cuh"

#include <cmath>
#include <stdexcept>

namespace macroflow3d {
namespace streamfunctions {

ShiftedJacobianOperator::ShiftedJacobianOperator(JvpWorkspace& jvp, const Grid3D& grid,
                                                 DeviceSpan<const real> q, real mu)
    : jvp_(&jvp), grid_(grid), q_(q), mu_(real{0}) {
    set_mu(mu);
}

void ShiftedJacobianOperator::set_mu(real mu) {
    if (!std::isfinite(mu) || mu < real{0}) {
        throw std::invalid_argument("ShiftedJacobianOperator::set_mu requires a finite mu >= 0");
    }
    mu_ = mu;
}

void ShiftedJacobianOperator::prepare(std::size_t n) {
    if (n == 0) {
        throw std::invalid_argument("ShiftedJacobianOperator::prepare requires n > 0");
    }
    ap_scratch_.resize(2 * n);
    n_ = n;
}

bool ShiftedJacobianOperator::prepared_for(std::size_t n) const noexcept {
    return n_ == n && ap_scratch_.size() == 2 * n;
}

void ShiftedJacobianOperator::ensure_prepared() const {
    if (n_ == 0) {
        throw std::logic_error("ShiftedJacobianOperator requires a preceding prepare() call");
    }
}

JvpApplyReport ShiftedJacobianOperator::apply(CudaContext& ctx, const Grid3D& grid,
                                              ConstCoupledVectorView direction,
                                              const JvpDeltaConfig& delta_config,
                                              CoupledVectorView jv_out) {
    ensure_prepared();

    // Step 1: J p, delegated entirely to the accepted matrix-free JvpWorkspace
    // (D2/D3 contracts unchanged; grid-identity fail-fast delegated here, see
    // the file header).
    JvpApplyReport report = jvp_->apply(ctx, grid, direction, delta_config, jv_out);
    if (report.status != JvpApplyStatus::ok) {
        return report;
    }

    // Step 2: jv_out_i += mu_ * (A p_i), skipped entirely when mu_ == 0
    // (bitwise passthrough of the plain J p above).
    if (mu_ != real{0}) {
        const operators::LesterPositiveDiffusionOperator A(grid_, q_);
        DeviceSpan<real> ap1(ap_scratch_.data(), n_);
        DeviceSpan<real> ap2(ap_scratch_.data() + n_, n_);
        A.apply(ctx, direction.c1, ap1);
        A.apply(ctx, direction.c2, ap2);
        blas::axpy(ctx, mu_, DeviceSpan<const real>(ap1), jv_out.c1);
        blas::axpy(ctx, mu_, DeviceSpan<const real>(ap2), jv_out.c2);
    }

    return report;
}

std::size_t ShiftedJacobianOperator::allocated_device_bytes() const noexcept {
    return ap_scratch_.capacity() * sizeof(real);
}

std::size_t ShiftedJacobianOperator::estimate_device_bytes(std::size_t n) {
    return 2 * n * sizeof(real);
}

} // namespace streamfunctions
} // namespace macroflow3d
