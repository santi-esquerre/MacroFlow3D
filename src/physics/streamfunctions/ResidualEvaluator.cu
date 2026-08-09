#include "ResidualEvaluator.cuh"

#include "../../numerics/blas/nrm2.cuh"
#include "../../numerics/operators/lester_positive_diffusion_operator.cuh"
#include "../../runtime/cuda_check.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace macroflow3d {
namespace streamfunctions {
namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 65535;
// Reduction-scalar layout within StreamfunctionResidualWorkspace::scalars_.
constexpr std::size_t kScalarRmsF1 = 0;
constexpr std::size_t kScalarRmsF2 = 1;
constexpr std::size_t kScalarLinfF1 = 2;
constexpr std::size_t kScalarLinfF2 = 3;
constexpr std::size_t kScalarQRms = 4;
constexpr std::size_t kNumScalars = 5;
constexpr std::size_t kNumHistogramSlots =
    static_cast<std::size_t>(kResidualHistogramBins) + 2; // bins + underflow + overflow

// CAS-based double atomicMax, valid for non-negative values (used only on
// |F1|, |F2|, which are already non-negative by construction).
__device__ __forceinline__ void atomic_max_double(double* addr, double val) {
    unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long assumed;
    unsigned long long old = *addr_ull;
    do {
        assumed = old;
        const double current = __longlong_as_double(static_cast<long long>(assumed));
        if (val <= current) {
            break;
        }
        old = atomicCAS(addr_ull, assumed, static_cast<unsigned long long>(__double_as_longlong(val)));
    } while (assumed != old);
}

// G_i[j] = rhs_i[j] - eta * q[j] * s_pair[j], evaluated in place into rhs1, rhs2.
// Pairing: G1 uses s2, G2 uses s1.
__global__ void combine_residual_rhs_kernel(real* rhs1, real* rhs2, const real* q, const real* s1,
                                             const real* s2, real eta, std::size_t n) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < n; index += stride) {
        const real qc = q[index];
        const real g1 = rhs1[index] - eta * qc * s2[index];
        const real g2 = rhs2[index] - eta * qc * s1[index];
        rhs1[index] = g1;
        rhs2[index] = g2;
    }
}

// F_i[j] = A u_i[j] - G_i[j] (G already mean-zero projected), plus Linf
// (max-abs) reduction folded once per thread after the grid-stride loop.
__global__ void finalize_residual_kernel(const real* a_u1, const real* a_u2, const real* g1,
                                          const real* g2, real* f1, real* f2, real* scalars,
                                          std::size_t n) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    real local_max_f1 = real{0};
    real local_max_f2 = real{0};

    for (std::size_t index = start; index < n; index += stride) {
        const real value_f1 = a_u1[index] - g1[index];
        const real value_f2 = a_u2[index] - g2[index];
        f1[index] = value_f1;
        f2[index] = value_f2;

        const real abs_f1 = fabs(value_f1);
        const real abs_f2 = fabs(value_f2);
        if (abs_f1 > local_max_f1) {
            local_max_f1 = abs_f1;
        }
        if (abs_f2 > local_max_f2) {
            local_max_f2 = abs_f2;
        }
    }

    if (local_max_f1 > real{0}) {
        atomic_max_double(&scalars[kScalarLinfF1], local_max_f1);
    }
    if (local_max_f2 > real{0}) {
        atomic_max_double(&scalars[kScalarLinfF2], local_max_f2);
    }
}

// Log-spaced |c| histogram, with c = grad(psi1) x grad(psi2) recomputed with
// the exact SF-09 cross-product expression order.
__global__ void residual_c_histogram_kernel(const real* psi1_x, const real* psi1_y,
                                             const real* psi1_z, const real* psi2_x,
                                             const real* psi2_y, const real* psi2_z,
                                             unsigned long long* histogram, std::size_t n,
                                             real log_min, real inv_bin_width, real c_min,
                                             real c_max) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    unsigned long long local_underflow = 0;
    unsigned long long local_overflow = 0;

    for (std::size_t index = start; index < n; index += stride) {
        const real g1x = psi1_x[index];
        const real g1y = psi1_y[index];
        const real g1z = psi1_z[index];
        const real g2x = psi2_x[index];
        const real g2y = psi2_y[index];
        const real g2z = psi2_z[index];

        const real cx = g1y * g2z - g1z * g2y;
        const real cy = g1z * g2x - g1x * g2z;
        const real cz = g1x * g2y - g1y * g2x;
        const real v = sqrt(cx * cx + cy * cy + cz * cz);

        if (!isfinite(v)) {
            ++local_overflow;
            continue;
        }
        if (v < c_min) {
            ++local_underflow;
            continue;
        }
        if (v >= c_max) {
            ++local_overflow;
            continue;
        }

        const int bin = static_cast<int>(floor((log10(v) - log_min) * inv_bin_width));
        const int clamped = bin < 0 ? 0 : (bin >= kResidualHistogramBins ? kResidualHistogramBins - 1 : bin);
        atomicAdd(&histogram[clamped], 1ULL);
    }

    if (local_underflow != 0) {
        atomicAdd(&histogram[kResidualHistogramBins], local_underflow);
    }
    if (local_overflow != 0) {
        atomicAdd(&histogram[kResidualHistogramBins + 1], local_overflow);
    }
}

std::size_t require_valid_grid(const Grid3D& grid) {
    if (grid.nx < 2 || grid.ny < 2 || grid.nz < 2) {
        throw std::invalid_argument(
            "Streamfunction residual requires every grid dimension to be at least 2");
    }
    if (!std::isfinite(grid.dx) || !std::isfinite(grid.dy) || !std::isfinite(grid.dz) ||
        grid.dx <= real{0} || grid.dy <= real{0} || grid.dz <= real{0}) {
        throw std::invalid_argument(
            "Streamfunction residual requires finite positive direction-specific spacing");
    }
    if (grid.dx != grid.dy || grid.dx != grid.dz) {
        throw std::invalid_argument(
            "Streamfunction residual currently requires the isotropic-spacing grids of the "
            "accepted SF-02/SF-06 operators (dx == dy == dz)");
    }

    const auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(real);
    const auto nx = static_cast<std::size_t>(grid.nx);
    const auto ny = static_cast<std::size_t>(grid.ny);
    const auto nz = static_cast<std::size_t>(grid.nz);
    if (nx > max_elements / ny || nx * ny > max_elements / nz) {
        throw std::invalid_argument(
            "Streamfunction residual requires a grid whose storage size does not overflow");
    }
    return nx * ny * nz;
}

void require_finite_nonnegative_eta(real eta) {
    if (!std::isfinite(eta) || eta < real{0}) {
        throw std::invalid_argument("Streamfunction residual requires a finite, non-negative eta");
    }
}

void require_valid_histogram_config(const ResidualHistogramConfig& config) {
    if (!std::isfinite(config.c_min_rel) || !std::isfinite(config.c_max_rel) ||
        config.c_min_rel <= real{0} || config.c_max_rel <= config.c_min_rel) {
        throw std::invalid_argument(
            "Streamfunction residual histogram config requires finite 0 < c_min_rel < c_max_rel");
    }
}

bool byte_ranges_overlap(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto a_begin = reinterpret_cast<std::uintptr_t>(a);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b);
    if (a_begin <= b_begin) {
        return b_begin - a_begin < a_bytes;
    }
    return a_begin - b_begin < b_bytes;
}

template <typename T>
bool overlaps(const DeviceSpan<T>& a, const DeviceSpan<T>& b) {
    return byte_ranges_overlap(a.data(), a.size() * sizeof(T), b.data(), b.size() * sizeof(T));
}

void require_exact_nonoverlapping_spans(DeviceSpan<const real> q,
                                        const PeriodicStreamfunctionFluctuations& fluctuations,
                                        DeviceSpan<real> f1, DeviceSpan<real> f2, std::size_t n) {
    if (q.size() != n || q.data() == nullptr) {
        throw std::invalid_argument("Streamfunction residual requires a non-null q span of exact grid size");
    }
    if (fluctuations.u1.size() != n || fluctuations.u2.size() != n ||
        fluctuations.u1.data() == nullptr || fluctuations.u2.data() == nullptr) {
        throw std::invalid_argument(
            "Streamfunction residual requires non-null fluctuation spans of exact grid size");
    }
    if (f1.size() != n || f2.size() != n || f1.data() == nullptr || f2.data() == nullptr) {
        throw std::invalid_argument(
            "Streamfunction residual requires non-null f1, f2 outputs of exact grid size");
    }

    const DeviceSpan<const real> others[] = {q, DeviceSpan<const real>(fluctuations.u1),
                                             DeviceSpan<const real>(fluctuations.u2)};
    const DeviceSpan<const real> outputs[] = {DeviceSpan<const real>(f1), DeviceSpan<const real>(f2)};
    if (overlaps(outputs[0], outputs[1])) {
        throw std::invalid_argument("Streamfunction residual f1 and f2 must not overlap");
    }
    for (const auto& output_span : outputs) {
        for (const auto& other : others) {
            if (overlaps(output_span, other)) {
                throw std::invalid_argument(
                    "Streamfunction residual f1, f2 must not overlap q or the fluctuation inputs");
            }
        }
    }
}

} // namespace

void StreamfunctionResidualWorkspace::prepare(std::size_t n) {
    psi1_x_.resize(n);
    psi1_y_.resize(n);
    psi1_z_.resize(n);
    psi2_x_.resize(n);
    psi2_y_.resize(n);
    psi2_z_.resize(n);

    h2g1_x_.resize(n);
    h2g1_y_.resize(n);
    h2g1_z_.resize(n);
    h1g2_x_.resize(n);
    h1g2_y_.resize(n);
    h1g2_z_.resize(n);
    b_x_.resize(n);
    b_y_.resize(n);
    b_z_.resize(n);

    s1_.resize(n);
    s2_.resize(n);
    source_counters_.resize(static_cast<std::size_t>(2 + kMaxDegeneracyThresholds));

    a_u1_.resize(n);
    a_u2_.resize(n);

    g1_.resize(n);
    g2_.resize(n);

    affine_rhs_workspace_.prepare(n);
    g_projection_workspace_.prepare(n);

    scalars_.resize(kNumScalars);
    histogram_.resize(kNumHistogramSlots);

    n_ = n;
    has_last_results_ = false;
}

std::size_t StreamfunctionResidualWorkspace::allocated_device_bytes() const noexcept {
    std::size_t bytes = 0;
    const DeviceBuffer<real>* const real_fields[] = {
        &psi1_x_, &psi1_y_, &psi1_z_, &psi2_x_, &psi2_y_, &psi2_z_,
        &h2g1_x_, &h2g1_y_, &h2g1_z_, &h1g2_x_, &h1g2_y_, &h1g2_z_,
        &b_x_,    &b_y_,    &b_z_,    &s1_,     &s2_,     &a_u1_,
        &a_u2_,   &g1_,     &g2_,     &scalars_,
    };
    for (const auto* field : real_fields) {
        bytes += field->capacity() * sizeof(real);
    }
    bytes += source_counters_.capacity() * sizeof(unsigned long long);
    bytes += histogram_.capacity() * sizeof(unsigned long long);
    bytes += affine_rhs_workspace_.allocated_device_bytes();
    bytes += g_projection_workspace_.allocated_device_bytes();
    bytes += blas::reduction_workspace_allocated_bytes(reduction_workspace_);
    return bytes;
}

std::size_t StreamfunctionResidualWorkspace::estimate_device_bytes(std::size_t n) {
    // Mirrors prepare(n) exactly: 21 fields of exactly n real elements plus
    // scalars_ (kNumScalars) and histogram_ (kNumHistogramSlots), plus the
    // three sub-workspaces and reduction_workspace_, which in this module is
    // only ever used through blas::nrm2_device (cuBLAS, no CUB temp storage),
    // so its capacity is exactly its default-constructed one-scalar buffer.
    constexpr std::size_t kNumRealFieldsOfSizeN = 21;
    std::size_t bytes = kNumRealFieldsOfSizeN * n * sizeof(real);
    bytes += kNumScalars * sizeof(real);
    bytes += kNumHistogramSlots * sizeof(unsigned long long);
    bytes += static_cast<std::size_t>(2 + kMaxDegeneracyThresholds) * sizeof(unsigned long long);
    bytes += AffinePeriodicRhsWorkspace::estimate_device_bytes(n);
    bytes += constraints::MeanZeroWorkspace::estimate_device_bytes(n);
    bytes += sizeof(real); // reduction_workspace_: default-constructed d_scalar only.
    return bytes;
}

DeviceSpan<const real> StreamfunctionResidualWorkspace::combined_rhs_g1() const {
    if (!has_last_results_) {
        throw std::logic_error(
            "StreamfunctionResidualWorkspace::combined_rhs_g1 requires a preceding "
            "enqueue_streamfunction_residual call");
    }
    return g1_.span();
}

DeviceSpan<const real> StreamfunctionResidualWorkspace::combined_rhs_g2() const {
    if (!has_last_results_) {
        throw std::logic_error(
            "StreamfunctionResidualWorkspace::combined_rhs_g2 requires a preceding "
            "enqueue_streamfunction_residual call");
    }
    return g2_.span();
}

bool StreamfunctionResidualWorkspace::prepared_for(std::size_t n) const noexcept {
    return n_ == n && psi1_x_.size() == n && psi1_y_.size() == n && psi1_z_.size() == n &&
           psi2_x_.size() == n && psi2_y_.size() == n && psi2_z_.size() == n &&
           h2g1_x_.size() == n && h2g1_y_.size() == n && h2g1_z_.size() == n &&
           h1g2_x_.size() == n && h1g2_y_.size() == n && h1g2_z_.size() == n &&
           b_x_.size() == n && b_y_.size() == n && b_z_.size() == n && s1_.size() == n &&
           s2_.size() == n &&
           source_counters_.size() == static_cast<std::size_t>(2 + kMaxDegeneracyThresholds) &&
           a_u1_.size() == n && a_u2_.size() == n && g1_.size() == n && g2_.size() == n &&
           affine_rhs_workspace_.prepared_for(n) && g_projection_workspace_.prepared_for(n) &&
           scalars_.size() == kNumScalars && histogram_.size() == kNumHistogramSlots;
}

void enqueue_streamfunction_residual(
    CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> q,
    const PeriodicStreamfunctionFluctuations& fluctuations, const AffineGauge& gauge, real eta,
    const NonlinearSourceConfig& source_config, const ResidualHistogramConfig& histogram_config,
    DeviceSpan<real> f1, DeviceSpan<real> f2, StreamfunctionResidualWorkspace& workspace) {
    const std::size_t n = require_valid_grid(grid);
    require_finite_nonnegative_eta(eta);
    require_valid_histogram_config(histogram_config);
    require_exact_nonoverlapping_spans(q, fluctuations, f1, f2, n);
    if (!workspace.prepared_for(n)) {
        throw std::logic_error(
            "Streamfunction residual requires a workspace prepared for the exact grid size");
    }

    // 1) Affine-periodic RHS, already mean-zero projected in place.
    workspace.last_rhs_diagnostics_ = assemble_affine_periodic_rhs(
        ctx, grid, q, gauge, workspace.g1_.span(), workspace.g2_.span(),
        workspace.affine_rhs_workspace_);

    // 2) Total streamfunction gradients.
    const TotalStreamfunctionGradientOutput grad_output{
        workspace.psi1_x_.span(), workspace.psi1_y_.span(), workspace.psi1_z_.span(),
        workspace.psi2_x_.span(), workspace.psi2_y_.span(), workspace.psi2_z_.span()};
    enqueue_total_streamfunction_gradients(ctx, grid, fluctuations, gauge, grad_output);

    const TotalStreamfunctionGradientView grad_view{
        workspace.psi1_x_.span(), workspace.psi1_y_.span(), workspace.psi1_z_.span(),
        workspace.psi2_x_.span(), workspace.psi2_y_.span(), workspace.psi2_z_.span()};

    // 3) Hessian-vector B field (six HVP scratch outputs are discarded).
    const HessianVectorBOutput hess_output{
        workspace.h2g1_x_.span(), workspace.h2g1_y_.span(), workspace.h2g1_z_.span(),
        workspace.h1g2_x_.span(), workspace.h1g2_y_.span(), workspace.h1g2_z_.span(),
        workspace.b_x_.span(),    workspace.b_y_.span(),    workspace.b_z_.span()};
    enqueue_streamfunction_hessian_vector_b(ctx, grid, fluctuations, grad_view, hess_output);

    const StreamfunctionBFieldView b_view{workspace.b_x_.span(), workspace.b_y_.span(),
                                          workspace.b_z_.span()};

    // 4) Nonlinear sources S1, S2.
    const std::size_t num_counters =
        static_cast<std::size_t>(2 + source_config.num_degeneracy_thresholds);
    const NonlinearSourceOutput source_output{workspace.s1_.span(), workspace.s2_.span()};
    const NonlinearSourceCounters source_counters{
        DeviceSpan<unsigned long long>(workspace.source_counters_.data(), num_counters)};
    enqueue_streamfunction_nonlinear_sources(ctx, grid, grad_view, b_view, source_config,
                                             source_output, source_counters);

    // 5) A.apply(u1), A.apply(u2).
    const operators::LesterPositiveDiffusionOperator op(grid, q);
    op.apply(ctx, DeviceSpan<const real>(fluctuations.u1), workspace.a_u1_.span());
    op.apply(ctx, DeviceSpan<const real>(fluctuations.u2), workspace.a_u2_.span());

    // 6) G_i = rhs_affine_i - eta*(q.*S_pair), in place into g1_, g2_.
    const std::size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    const int blocks = static_cast<int>(
        requested_blocks < static_cast<std::size_t>(kMaxBlocks) ? requested_blocks : kMaxBlocks);
    combine_residual_rhs_kernel<<<blocks, kBlockSize, 0, ctx.cuda_stream()>>>(
        workspace.g1_.data(), workspace.g2_.data(), q.data(), workspace.s1_.data(),
        workspace.s2_.data(), eta, n);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // 7) Project G1, G2.
    const constraints::MeanZeroProjector projector;
    projector.project(ctx, workspace.g1_.span(), workspace.g_projection_workspace_);
    projector.project(ctx, workspace.g2_.span(), workspace.g_projection_workspace_);

    // 8) F_i = A u_i - G_i, plus Linf reduction.
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(workspace.scalars_.data() + kScalarLinfF1, 0,
                                           2 * sizeof(real), ctx.cuda_stream()));
    finalize_residual_kernel<<<blocks, kBlockSize, 0, ctx.cuda_stream()>>>(
        workspace.a_u1_.data(), workspace.a_u2_.data(), workspace.g1_.data(), workspace.g2_.data(),
        f1.data(), f2.data(), workspace.scalars_.data(), n);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // 9) RMS(F1), RMS(F2), RMS(q) as raw L2 norms.
    blas::nrm2_device(ctx, DeviceSpan<const real>(f1),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarRmsF1, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(f2),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarRmsF2, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, q, DeviceSpan<real>(workspace.scalars_.data() + kScalarQRms, 1),
                      workspace.reduction_workspace_);

    // 10) |c| histogram.
    const real v_rms = source_config.v_rms;
    const real c_min = histogram_config.c_min_rel * v_rms;
    const real c_max = histogram_config.c_max_rel * v_rms;
    const real log_min = std::log10(c_min);
    const real log_max = std::log10(c_max);
    const real inv_bin_width = static_cast<real>(kResidualHistogramBins) / (log_max - log_min);

    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(workspace.histogram_.data(), 0,
                                           kNumHistogramSlots * sizeof(unsigned long long),
                                           ctx.cuda_stream()));
    residual_c_histogram_kernel<<<blocks, kBlockSize, 0, ctx.cuda_stream()>>>(
        workspace.psi1_x_.data(), workspace.psi1_y_.data(), workspace.psi1_z_.data(),
        workspace.psi2_x_.data(), workspace.psi2_y_.data(), workspace.psi2_z_.data(),
        workspace.histogram_.data(), n, log_min, inv_bin_width, c_min, c_max);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    workspace.has_last_results_ = true;
}

StreamfunctionResidualReport synchronize_streamfunction_residual_report(
    CudaContext& ctx, const Grid3D& grid, real eta, const NonlinearSourceConfig& source_config,
    const ResidualHistogramConfig& histogram_config, StreamfunctionResidualWorkspace& workspace) {
    const std::size_t n = require_valid_grid(grid);
    require_finite_nonnegative_eta(eta);
    require_valid_histogram_config(histogram_config);
    if (!workspace.prepared_for(n)) {
        throw std::logic_error(
            "Streamfunction residual report requires a workspace prepared for the exact grid size");
    }
    if (!workspace.has_last_results_) {
        throw std::logic_error(
            "Streamfunction residual report requires a preceding enqueue_streamfunction_residual call");
    }

    real scalars_host[kNumScalars] = {};
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(scalars_host, workspace.scalars_.data(),
                                           kNumScalars * sizeof(real), cudaMemcpyDeviceToHost,
                                           ctx.cuda_stream()));

    unsigned long long histogram_host[kNumHistogramSlots] = {};
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(histogram_host, workspace.histogram_.data(),
                                           kNumHistogramSlots * sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));

    const std::size_t num_counters =
        static_cast<std::size_t>(2 + source_config.num_degeneracy_thresholds);
    unsigned long long source_counters_host[2 + kMaxDegeneracyThresholds] = {};
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(source_counters_host, workspace.source_counters_.data(),
                                           num_counters * sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));

    real raw_means_host[2] = {};
    real projected_means_host[2] = {};
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(raw_means_host,
                                           workspace.last_rhs_diagnostics_.raw_means.data(),
                                           2 * sizeof(real), cudaMemcpyDeviceToHost,
                                           ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(projected_means_host,
                                           workspace.last_rhs_diagnostics_.projected_means.data(),
                                           2 * sizeof(real), cudaMemcpyDeviceToHost,
                                           ctx.cuda_stream()));

    ctx.synchronize();

    StreamfunctionResidualReport report;
    const real sqrt_n = std::sqrt(static_cast<real>(n));
    report.rms_f1 = scalars_host[kScalarRmsF1] / sqrt_n;
    report.rms_f2 = scalars_host[kScalarRmsF2] / sqrt_n;
    report.linf_f1 = scalars_host[kScalarLinfF1];
    report.linf_f2 = scalars_host[kScalarLinfF2];
    report.q_rms = scalars_host[kScalarQRms] / sqrt_n;

    const real l_ref = std::cbrt(grid.Lx() * grid.Ly() * grid.Lz());
    const real v_rms = source_config.v_rms;
    report.r1 = report.rms_f1 * l_ref / (report.q_rms * v_rms);
    report.r2 = report.rms_f2 * l_ref / report.q_rms;
    report.r_F = std::sqrt((report.r1 * report.r1 + report.r2 * report.r2) / real{2});

    report.raw_rhs_mean_psi1 = raw_means_host[0];
    report.raw_rhs_mean_psi2 = raw_means_host[1];
    report.projected_rhs_mean_psi1 = projected_means_host[0];
    report.projected_rhs_mean_psi2 = projected_means_host[1];

    report.nonfinite_s1 = source_counters_host[0];
    report.nonfinite_s2 = source_counters_host[1];
    report.num_degeneracy_thresholds = source_config.num_degeneracy_thresholds;
    for (int t = 0; t < source_config.num_degeneracy_thresholds; ++t) {
        report.degeneracy_counts[t] = source_counters_host[2 + t];
    }

    for (int i = 0; i < kResidualHistogramBins; ++i) {
        report.histogram_counts[i] = histogram_host[i];
    }
    report.histogram_underflow = histogram_host[kResidualHistogramBins];
    report.histogram_overflow = histogram_host[kResidualHistogramBins + 1];
    report.histogram_c_min = histogram_config.c_min_rel * v_rms;
    report.histogram_c_max = histogram_config.c_max_rel * v_rms;

    report.eta = eta;
    report.epsilon = source_config.epsilon;
    report.v_rms = v_rms;
    report.L_ref = l_ref;
    report.n = n;

    return report;
}

real residual_histogram_percentile(const StreamfunctionResidualReport& report, real p) {
    if (!std::isfinite(p) || p <= real{0} || p > real{1}) {
        throw std::invalid_argument("residual_histogram_percentile requires p in (0, 1]");
    }

    double total = static_cast<double>(report.histogram_underflow) +
                   static_cast<double>(report.histogram_overflow);
    for (int i = 0; i < kResidualHistogramBins; ++i) {
        total += static_cast<double>(report.histogram_counts[i]);
    }
    if (total <= 0.0) {
        throw std::invalid_argument("residual_histogram_percentile requires a non-empty histogram");
    }

    double cumulative = static_cast<double>(report.histogram_underflow);
    if (cumulative / total >= static_cast<double>(p)) {
        return report.histogram_c_min;
    }

    const real log_min = std::log10(report.histogram_c_min);
    const real log_max = std::log10(report.histogram_c_max);
    const real bin_width = (log_max - log_min) / static_cast<real>(kResidualHistogramBins);

    for (int i = 0; i < kResidualHistogramBins; ++i) {
        cumulative += static_cast<double>(report.histogram_counts[i]);
        if (cumulative / total >= static_cast<double>(p)) {
            return std::pow(real{10}, log_min + static_cast<real>(i + 1) * bin_width);
        }
    }

    return std::numeric_limits<real>::infinity();
}

} // namespace streamfunctions
} // namespace macroflow3d
