#include "JacobianVectorProduct.cuh"

#include "../../numerics/blas/axpy.cuh"
#include "../../numerics/blas/copy.cuh"
#include "../../numerics/blas/nrm2.cuh"
#include "../../numerics/blas/scal.cuh"
#include "../../runtime/cuda_check.cuh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace streamfunctions {

namespace {

void require_exact_component_size(DeviceSpan<const real> c1, DeviceSpan<const real> c2,
                                  std::size_t n, const char* what) {
    if (c1.size() != n || c2.size() != n || c1.data() == nullptr || c2.data() == nullptr) {
        throw std::invalid_argument(
            std::string("JacobianVectorProduct requires ") + what +
            " to be non-null coupled-vector components of exact grid size");
    }
}

} // namespace

CoupledVectorView split_coupled_vector(DeviceSpan<real> combined, std::size_t n) {
    if (n == 0 || combined.size() != 2 * n) {
        throw std::invalid_argument(
            "split_coupled_vector requires a non-empty combined span of exactly 2*n elements");
    }
    return CoupledVectorView{DeviceSpan<real>(combined.data(), n),
                             DeviceSpan<real>(combined.data() + n, n)};
}

ConstCoupledVectorView split_coupled_vector(DeviceSpan<const real> combined, std::size_t n) {
    if (n == 0 || combined.size() != 2 * n) {
        throw std::invalid_argument(
            "split_coupled_vector requires a non-empty combined span of exactly 2*n elements");
    }
    return ConstCoupledVectorView(DeviceSpan<const real>(combined.data(), n),
                                  DeviceSpan<const real>(combined.data() + n, n));
}

void validate_jvp_delta_config(const JvpDeltaConfig& config) {
    if (!std::isfinite(config.delta_min) || !std::isfinite(config.delta_max) ||
        !(config.delta_min > real{0}) || !(config.delta_max > config.delta_min)) {
        throw std::invalid_argument(
            "JvpDeltaConfig requires finite values with 0 < delta_min < delta_max");
    }
}

void JvpWorkspace::ensure_prepared() const {
    if (n_ == 0) {
        throw std::logic_error("JvpWorkspace requires a preceding prepare() call");
    }
}

void JvpWorkspace::ensure_base_prepared() const {
    ensure_prepared();
    if (!base_prepared_) {
        throw std::logic_error("JvpWorkspace requires a preceding prepare_jvp_base() call");
    }
}

void JvpWorkspace::prepare(std::size_t n) {
    if (n == 0) {
        throw std::invalid_argument("JvpWorkspace::prepare requires n > 0");
    }

    base_f1_.resize(n);
    base_f2_.resize(n);
    proj_p1_.resize(n);
    proj_p2_.resize(n);
    pert_u1_.resize(n);
    pert_u2_.resize(n);
    pert_f1_.resize(n);
    pert_f2_.resize(n);
    norm_scratch_.resize(2);

    residual_workspace_.prepare(n);
    projection_workspace_.prepare(n);
    reduction_workspace_.ensure_scalar();

    n_ = n;
    base_prepared_ = false;
    q_ = DeviceSpan<const real>{};
    base_fluctuations_ = PeriodicStreamfunctionFluctuations{};
    weighted_base_term_ = real{};
}

bool JvpWorkspace::prepared_for(std::size_t n) const noexcept {
    return n_ == n && base_f1_.size() == n && base_f2_.size() == n && proj_p1_.size() == n &&
           proj_p2_.size() == n && pert_u1_.size() == n && pert_u2_.size() == n &&
           pert_f1_.size() == n && pert_f2_.size() == n && norm_scratch_.size() == 2 &&
           residual_workspace_.prepared_for(n) && projection_workspace_.prepared_for(n);
}

real JvpWorkspace::compute_weighted_norm(CudaContext& ctx, DeviceSpan<const real> a,
                                         DeviceSpan<const real> b, real g1, real g2) {
    // Two fixed-shape cuBLAS L2-norm reductions into distinct slots of one
    // small device buffer, no intermediate host synchronization, then ONE
    // memcpy + one synchronize (see the file header, D1).
    blas::nrm2_device(ctx, a, DeviceSpan<real>(norm_scratch_.data() + 0, 1), reduction_workspace_);
    blas::nrm2_device(ctx, b, DeviceSpan<real>(norm_scratch_.data() + 1, 1), reduction_workspace_);

    real host_buf[2] = {};
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(host_buf, norm_scratch_.data(), 2 * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();

    const real sqrt_n = std::sqrt(static_cast<real>(n_));
    const real rms_a = host_buf[0] / sqrt_n;
    const real rms_b = host_buf[1] / sqrt_n;
    const real term_a = rms_a / g1;
    const real term_b = rms_b / g2;
    return std::sqrt(term_a * term_a + term_b * term_b);
}

void JvpWorkspace::prepare_jvp_base(CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> q,
                                    CoupledVectorView base, const AffineGauge& gauge, real eta,
                                    const NonlinearSourceConfig& source_config,
                                    const ResidualHistogramConfig& histogram_config) {
    ensure_prepared();
    require_exact_component_size(DeviceSpan<const real>(base.c1), DeviceSpan<const real>(base.c2),
                                 n_, "base");

    // Freeze the identity apply() will reuse (D4). base_fluctuations_ is a
    // non-owning copy of the caller's spans, not a device copy: the caller
    // contract is that base.c1/base.c2 remain valid and unchanged for the
    // lifetime of every subsequent apply() call against this cached base.
    q_ = q;
    base_fluctuations_ = PeriodicStreamfunctionFluctuations{base.c1, base.c2};
    gauge_ = gauge;
    eta_ = eta;
    source_config_ = source_config;
    histogram_config_ = histogram_config;

    // F(Psi), evaluated exactly once and cached on device. No synchronize()
    // is required here: Jv is formed entirely on-device from base_f1_/
    // base_f2_ inside apply(); nothing here needs a host-visible value of
    // F(Psi) itself.
    enqueue_streamfunction_residual(ctx, grid, q_, base_fluctuations_, gauge_, eta_, source_config_,
                                    histogram_config_, base_f1_.span(), base_f2_.span(),
                                    residual_workspace_);

    // Weighted base-norm term 1 + ||Psi||_w (D1/D2), one documented sync.
    weighted_base_term_ =
        real{1} + compute_weighted_norm(ctx, DeviceSpan<const real>(base.c1),
                                        DeviceSpan<const real>(base.c2), source_config_.v_rms,
                                        real{1});

    base_prepared_ = true;
    ++base_evaluations_;
}

JvpApplyReport JvpWorkspace::apply(CudaContext& ctx, const Grid3D& grid,
                                   ConstCoupledVectorView direction,
                                   const JvpDeltaConfig& delta_config, CoupledVectorView jv_out) {
    ensure_base_prepared();
    validate_jvp_delta_config(delta_config);
    require_exact_component_size(direction.c1, direction.c2, n_, "direction");
    require_exact_component_size(DeviceSpan<const real>(jv_out.c1), DeviceSpan<const real>(jv_out.c2),
                                 n_, "jv_out");

    const constraints::MeanZeroProjector projector;

    // D3: project a workspace COPY of the direction; the caller's direction
    // view is never mutated.
    blas::copy(ctx, direction.c1, proj_p1_.span());
    blas::copy(ctx, direction.c2, proj_p2_.span());
    projector.project(ctx, proj_p1_.span(), projection_workspace_);
    projector.project(ctx, proj_p2_.span(), projection_workspace_);

    // D1: weighted norm of the projected direction, one documented sync.
    const real weighted_p_norm =
        compute_weighted_norm(ctx, DeviceSpan<const real>(proj_p1_.span()),
                              DeviceSpan<const real>(proj_p2_.span()), source_config_.v_rms,
                              real{1});
    if (!std::isfinite(weighted_p_norm) || weighted_p_norm == real{0}) {
        throw std::invalid_argument(
            "JvpWorkspace::apply requires a finite, nonzero weighted direction norm");
    }

    // D2: forward-difference delta, clamped to [delta_min, delta_max].
    real delta = std::sqrt(std::numeric_limits<real>::epsilon()) * weighted_base_term_ /
                weighted_p_norm;

    JvpApplyReport report;
    report.weighted_p_norm = weighted_p_norm;

    if (delta < delta_config.delta_min) {
        delta = delta_config.delta_min;
        report.delta_min_clamped = true;
        ++delta_min_clamp_total_;
    } else if (delta > delta_config.delta_max) {
        delta = delta_config.delta_max;
        report.delta_max_clamped = true;
        ++delta_max_clamp_total_;
    }
    report.delta = delta;

    // Perturbed state Psi + delta*p, then re-projected (idempotent defense,
    // D3).
    blas::copy(ctx, DeviceSpan<const real>(base_fluctuations_.u1), pert_u1_.span());
    blas::axpy(ctx, delta, DeviceSpan<const real>(proj_p1_.span()), pert_u1_.span());
    blas::copy(ctx, DeviceSpan<const real>(base_fluctuations_.u2), pert_u2_.span());
    blas::axpy(ctx, delta, DeviceSpan<const real>(proj_p2_.span()), pert_u2_.span());
    projector.project(ctx, pert_u1_.span(), projection_workspace_);
    projector.project(ctx, pert_u2_.span(), projection_workspace_);

    // Exactly one perturbed residual evaluation per apply() call (D4).
    const PeriodicStreamfunctionFluctuations perturbed_fluctuations{pert_u1_.span(),
                                                                     pert_u2_.span()};
    enqueue_streamfunction_residual(ctx, grid, q_, perturbed_fluctuations, gauge_, eta_,
                                    source_config_, histogram_config_, pert_f1_.span(),
                                    pert_f2_.span(), residual_workspace_);
    const StreamfunctionResidualReport perturbed_report = synchronize_streamfunction_residual_report(
        ctx, grid, eta_, source_config_, histogram_config_, residual_workspace_);
    ++apply_evaluations_;

    report.perturbed_rms_f1 = perturbed_report.rms_f1;
    report.perturbed_rms_f2 = perturbed_report.rms_f2;
    report.perturbed_linf_f1 = perturbed_report.linf_f1;
    report.perturbed_linf_f2 = perturbed_report.linf_f2;

    if (!std::isfinite(perturbed_report.rms_f1) || !std::isfinite(perturbed_report.rms_f2) ||
        !std::isfinite(perturbed_report.linf_f1) || !std::isfinite(perturbed_report.linf_f2)) {
        report.status = JvpApplyStatus::nonfinite_perturbed_residual;
        return report;
    }

    // Jv = (F(Psi + delta*p) - F(Psi)) / delta, into the caller's jv_out.
    // NOT re-projected on output (see the file header, D3).
    const real inv_delta = real{1} / delta;

    blas::copy(ctx, DeviceSpan<const real>(pert_f1_.span()), jv_out.c1);
    blas::axpy(ctx, real{-1}, DeviceSpan<const real>(base_f1_.span()), jv_out.c1);
    blas::scal(ctx, jv_out.c1, inv_delta);

    blas::copy(ctx, DeviceSpan<const real>(pert_f2_.span()), jv_out.c2);
    blas::axpy(ctx, real{-1}, DeviceSpan<const real>(base_f2_.span()), jv_out.c2);
    blas::scal(ctx, jv_out.c2, inv_delta);

    report.status = JvpApplyStatus::ok;
    return report;
}

std::size_t JvpWorkspace::allocated_device_bytes() const noexcept {
    std::size_t bytes = 0;
    const DeviceBuffer<real>* const real_fields[] = {
        &base_f1_, &base_f2_, &proj_p1_, &proj_p2_, &pert_u1_,
        &pert_u2_, &pert_f1_, &pert_f2_, &norm_scratch_,
    };
    for (const auto* field : real_fields) {
        bytes += field->capacity() * sizeof(real);
    }
    bytes += residual_workspace_.allocated_device_bytes();
    bytes += projection_workspace_.allocated_device_bytes();
    bytes += blas::reduction_workspace_allocated_bytes(reduction_workspace_);
    return bytes;
}

std::size_t JvpWorkspace::estimate_device_bytes(std::size_t n) {
    // Mirrors prepare(n) exactly: 8 fields of exactly n real elements
    // (base_f1/f2, proj_p1/p2, pert_u1/u2, pert_f1/f2) plus the fixed
    // 2-element norm_scratch_, plus the residual workspace, the mean-zero
    // projection workspace, and reduction_workspace_'s default-constructed
    // one-scalar footprint (only ever used through blas::nrm2_device here,
    // exactly like ResidualEvaluator.cu's own reduction_workspace_).
    constexpr std::size_t kNumRealFieldsOfSizeN = 8;
    std::size_t bytes = kNumRealFieldsOfSizeN * n * sizeof(real);
    bytes += 2 * sizeof(real); // norm_scratch_
    bytes += StreamfunctionResidualWorkspace::estimate_device_bytes(n);
    bytes += constraints::MeanZeroWorkspace::estimate_device_bytes(n);
    bytes += sizeof(real); // reduction_workspace_: default-constructed d_scalar only.
    return bytes;
}

} // namespace streamfunctions
} // namespace macroflow3d
