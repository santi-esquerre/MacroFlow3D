#include "CoupledGmres.cuh"

#include <cmath>
#include <stdexcept>

namespace macroflow3d {
namespace streamfunctions {

void validate_coupled_gmres_config(const CoupledGmresConfig& config) {
    if (config.restart < 1 || config.restart > kMaxCoupledGmresRestart) {
        throw std::invalid_argument(
            "CoupledGmresConfig.restart must satisfy 1 <= restart <= kMaxCoupledGmresRestart");
    }
    if (config.max_iterations < 1) {
        throw std::invalid_argument("CoupledGmresConfig.max_iterations must be >= 1");
    }
    if (!std::isfinite(config.rel_tol) || !(config.rel_tol > real{0})) {
        throw std::invalid_argument("CoupledGmresConfig.rel_tol must be finite and positive");
    }
}

void CoupledGmres::prepare(std::size_t n, int m) {
    if (n == 0) {
        throw std::invalid_argument("CoupledGmres::prepare requires n > 0");
    }
    if (m < 1 || m > kMaxCoupledGmresRestart) {
        throw std::invalid_argument("CoupledGmres::prepare requires 1 <= m <= kMaxCoupledGmresRestart");
    }

    const std::size_t basis_elements = static_cast<std::size_t>(m + 1) * 2 * n;
    basis_.resize(basis_elements);
    w_.resize(2 * n);
    z_.resize(2 * n);
    residual_scratch_.resize(2 * n);
    correction_total_.resize(2 * n);
    rhs_.resize(2 * n);

    projection_workspace_.prepare(n);
    reduction_workspace_.ensure_scalar();

    n_ = n;
    m_ = m;
}

bool CoupledGmres::prepared_for(std::size_t n, int m) const noexcept {
    if (n_ != n || m_ != m) {
        return false;
    }
    const std::size_t basis_elements = static_cast<std::size_t>(m + 1) * 2 * n;
    return basis_.size() == basis_elements && w_.size() == 2 * n && z_.size() == 2 * n &&
           residual_scratch_.size() == 2 * n && correction_total_.size() == 2 * n &&
           rhs_.size() == 2 * n && projection_workspace_.prepared_for(n);
}

void CoupledGmres::ensure_prepared() const {
    if (n_ == 0) {
        throw std::logic_error("CoupledGmres requires a preceding prepare() call");
    }
}

DeviceSpan<real> CoupledGmres::basis_vector(int index) noexcept {
    return DeviceSpan<real>(basis_.data() + static_cast<std::size_t>(index) * 2 * n_, 2 * n_);
}

DeviceSpan<const real> CoupledGmres::basis_vector(int index) const noexcept {
    return DeviceSpan<const real>(basis_.data() + static_cast<std::size_t>(index) * 2 * n_, 2 * n_);
}

void CoupledGmres::project_pair(CudaContext& ctx, DeviceSpan<real> c1, DeviceSpan<real> c2) const {
    const constraints::MeanZeroProjector projector;
    projector.project(ctx, c1, projection_workspace_);
    projector.project(ctx, c2, projection_workspace_);
}

real CoupledGmres::dot2n(CudaContext& ctx, DeviceSpan<const real> a, DeviceSpan<const real> b) const {
    return blas::dot_host(ctx, a, b, reduction_workspace_);
}

real CoupledGmres::norm2n(CudaContext& ctx, DeviceSpan<const real> a) const {
    return blas::nrm2_host(ctx, a, reduction_workspace_);
}

void CoupledGmres::project_rhs(CudaContext& ctx, ConstCoupledVectorView b) {
    blas::copy(ctx, b.c1, DeviceSpan<real>(rhs_.data(), n_));
    blas::copy(ctx, b.c2, DeviceSpan<real>(rhs_.data() + n_, n_));
    project_pair(ctx, DeviceSpan<real>(rhs_.data(), n_), DeviceSpan<real>(rhs_.data() + n_, n_));
}

real CoupledGmres::compute_true_residual(CudaContext& ctx, const Grid3D& grid, JvpWorkspace& jvp,
                                         const JvpDeltaConfig& delta_config,
                                         DeviceSpan<const real> x_total_flat,
                                         DeviceSpan<real> r_out_flat, bool& ok) {
    // r_out <- J(x_total) (one JvpWorkspace::apply call; NOT counted in the
    // caller's inner-iteration budget -- residual recomputation is a
    // separate, explicitly checkpointed activity, see the file header).
    ConstCoupledVectorView x_view = split_coupled_vector(x_total_flat, n_);
    CoupledVectorView jx_view = split_coupled_vector(r_out_flat, n_);
    JvpApplyReport jr = jvp.apply(ctx, grid, x_view, delta_config, jx_view);
    if (jr.status != JvpApplyStatus::ok) {
        ok = false;
        return real{0};
    }
    ok = true;

    // r_out <- P(rhs_) - J(x_total).
    blas::scal(ctx, r_out_flat, real{-1});
    blas::axpy(ctx, real{1}, DeviceSpan<const real>(rhs_.span()), r_out_flat);
    project_pair(ctx, DeviceSpan<real>(r_out_flat.data(), n_),
                DeviceSpan<real>(r_out_flat.data() + n_, n_));
    return norm2n(ctx, DeviceSpan<const real>(r_out_flat));
}

std::size_t CoupledGmres::allocated_device_bytes() const noexcept {
    std::size_t bytes = 0;
    const DeviceBuffer<real>* const real_fields[] = {
        &basis_, &w_, &z_, &residual_scratch_, &correction_total_, &rhs_,
    };
    for (const auto* field : real_fields) {
        bytes += field->capacity() * sizeof(real);
    }
    bytes += projection_workspace_.allocated_device_bytes();
    bytes += blas::reduction_workspace_allocated_bytes(reduction_workspace_);
    return bytes;
}

std::size_t CoupledGmres::estimate_device_bytes(std::size_t n, int m) {
    // Mirrors prepare(n, m) exactly: the (m+1)-vector basis (2*(m+1)*n reals,
    // the dominant term -- see the file header for the exact 256^3/m=10
    // arithmetic), five 2*n-real work vectors (w_, z_, residual_scratch_,
    // correction_total_, rhs_), the n-sized MeanZeroWorkspace, and
    // reduction_workspace_'s default-constructed one-scalar footprint.
    std::size_t bytes = static_cast<std::size_t>(m + 1) * 2 * n * sizeof(real);
    bytes += 5 * 2 * n * sizeof(real);
    bytes += constraints::MeanZeroWorkspace::estimate_device_bytes(n);
    bytes += sizeof(real);
    return bytes;
}

} // namespace streamfunctions
} // namespace macroflow3d
