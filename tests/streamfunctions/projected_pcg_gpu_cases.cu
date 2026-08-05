#include "streamfunction_operator_test_cases.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/blas/copy.cuh"
#include "src/numerics/constraints/MeanZeroProjector.cuh"
#include "src/numerics/operators/lester_positive_diffusion_operator.cuh"
#include "src/numerics/solvers/pcg.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace macroflow3d::streamfunctions::test {
namespace {

static_assert(std::is_same_v<real, double>, "SF-04 manufactured controls require double precision");

constexpr std::size_t kNaxis = 17;
constexpr std::size_t kN = kNaxis * kNaxis * kNaxis;
constexpr long double kPi = 3.141592653589793238462643383279502884L;
constexpr int kCheckEvery = 5;
constexpr real kRtol = 1.0e-12;
constexpr double kResidualTolerance = 1.0e-10;
constexpr double kSolutionTolerance = 5.0e-9;
constexpr double kFourierRelativeTolerance = 1.0e-11;
constexpr double kFourierPointwiseTolerance = 1.0e-10;
constexpr double kCpuGpuResidualRelativeTolerance = 1.0e-11;
// Six face terms and harmonic divisions accumulate at O(eps*||b||); 4096 eps
// is a fixed, deliberately conservative CPU-long-double/GPU-double floor.
constexpr double kCpuGpuResidualRoundoffFactor = 4096.0;
constexpr double kGaugeFactor = 100.0;
constexpr double kReportedGaugeComparisonFactor = 200.0;
// CUB's double reductions and the CPU long-double reference differ only by
// reduction order.  This fixed 4096-epsilon envelope covers that roundoff
// while remaining far below the prescribed 1e-12 contract checks.
constexpr double kRawDiagnosticRoundoffFactor = 4096.0;
constexpr double kRawMeanTolerance = 1.0e-12;
constexpr double kInitialGaugeTolerance = 1.0e-12;

[[nodiscard]] std::size_t index(std::size_t ix, std::size_t iy, std::size_t iz) {
    return ix + kNaxis * (iy + kNaxis * iz);
}

[[nodiscard]] std::size_t wrap(std::ptrdiff_t value) {
    const auto n = static_cast<std::ptrdiff_t>(kNaxis);
    const auto r = value % n;
    return static_cast<std::size_t>(r < 0 ? r + n : r);
}

[[nodiscard]] std::size_t neighbor(std::size_t ix, std::size_t iy, std::size_t iz,
                                   int axis, std::ptrdiff_t offset) {
    if (axis == 0) return index(wrap(static_cast<std::ptrdiff_t>(ix) + offset), iy, iz);
    if (axis == 1) return index(ix, wrap(static_cast<std::ptrdiff_t>(iy) + offset), iz);
    return index(ix, iy, wrap(static_cast<std::ptrdiff_t>(iz) + offset));
}

[[nodiscard]] long double mean_ld(const std::vector<real>& values) {
    long double sum = 0.0L;
    for (const real value : values) sum += static_cast<long double>(value);
    return sum / static_cast<long double>(values.size());
}

[[nodiscard]] double rms(const std::vector<real>& values) {
    long double sum = 0.0L;
    for (const real value : values) {
        const long double promoted = static_cast<long double>(value);
        sum += promoted * promoted;
    }
    return std::sqrt(static_cast<double>(sum / static_cast<long double>(values.size())));
}

[[nodiscard]] double l2(const std::vector<real>& values) {
    return std::sqrt(static_cast<double>([&] {
        long double sum = 0.0L;
        for (const real value : values) {
            const long double promoted = static_cast<long double>(value);
            sum += promoted * promoted;
        }
        return sum;
    }()));
}

[[nodiscard]] std::vector<real> project_cpu(const std::vector<real>& values) {
    const long double mean = mean_ld(values);
    std::vector<real> projected(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        projected[i] = static_cast<real>(static_cast<long double>(values[i]) - mean);
    }
    return projected;
}

[[nodiscard]] std::vector<real> positive_diffusion_cpu(const std::vector<real>& q,
                                                        const std::vector<real>& u) {
    if (q.size() != kN || u.size() != kN) throw std::invalid_argument("CPU stencil size mismatch");
    const long double inv_h2 = static_cast<long double>(kNaxis) * kNaxis;
    std::vector<real> output(kN);
    for (std::size_t iz = 0; iz < kNaxis; ++iz) {
        for (std::size_t iy = 0; iy < kNaxis; ++iy) {
            for (std::size_t ix = 0; ix < kNaxis; ++ix) {
                const std::size_t center = index(ix, iy, iz);
                const long double qc = static_cast<long double>(q[center]);
                const long double uc = static_cast<long double>(u[center]);
                long double sum = 0.0L;
                for (int axis = 0; axis != 3; ++axis) {
                    for (const std::ptrdiff_t offset : {-1, 1}) {
                        const std::size_t adjacent = neighbor(ix, iy, iz, axis, offset);
                        const long double qn = static_cast<long double>(q[adjacent]);
                        const long double q_face = 2.0L * qc * qn / (qc + qn);
                        sum += q_face * (uc - static_cast<long double>(u[adjacent]));
                    }
                }
                output[center] = static_cast<real>(sum * inv_h2);
            }
        }
    }
    return output;
}

[[nodiscard]] std::vector<real> manufactured_solution() {
    std::vector<real> values(kN);
    for (std::size_t iz = 0; iz < kNaxis; ++iz) {
        for (std::size_t iy = 0; iy < kNaxis; ++iy) {
            for (std::size_t ix = 0; ix < kNaxis; ++ix) {
                const long double x = (static_cast<long double>(ix) + 0.5L) / kNaxis;
                const long double y = (static_cast<long double>(iy) + 0.5L) / kNaxis;
                const long double z = (static_cast<long double>(iz) + 0.5L) / kNaxis;
                values[index(ix, iy, iz)] = static_cast<real>(
                    0.45L * std::sin(2.0L * kPi * x) - 0.30L * std::cos(4.0L * kPi * y) +
                    0.20L * std::sin(2.0L * kPi * z) +
                    0.10L * std::cos(2.0L * kPi * (x + y - z)));
            }
        }
    }
    return project_cpu(values);
}

[[nodiscard]] std::vector<real> coefficient_field(bool smooth) {
    std::vector<real> q(kN, real(1.0));
    if (!smooth) return q;
    for (std::size_t iz = 0; iz < kNaxis; ++iz) {
        for (std::size_t iy = 0; iy < kNaxis; ++iy) {
            for (std::size_t ix = 0; ix < kNaxis; ++ix) {
                const long double phase = 2.0L * kPi *
                    (static_cast<long double>(ix + iy + iz) + 1.5L) / kNaxis;
                q[index(ix, iy, iz)] = static_cast<real>(1.25L + 0.25L * std::cos(phase));
            }
        }
    }
    return q;
}

[[nodiscard]] double rms_difference(const std::vector<real>& left, const std::vector<real>& right) {
    if (left.size() != right.size()) throw std::invalid_argument("RMS difference size mismatch");
    long double sum = 0.0L;
    for (std::size_t i = 0; i < left.size(); ++i) {
        const long double difference = static_cast<long double>(left[i]) - right[i];
        sum += difference * difference;
    }
    return std::sqrt(static_cast<double>(sum / static_cast<long double>(left.size())));
}

[[nodiscard]] double max_abs_difference(const std::vector<real>& left, const std::vector<real>& right) {
    double maximum = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        maximum = std::max(maximum, std::abs(left[i] - right[i]));
    }
    return maximum;
}

[[nodiscard]] std::vector<real> constant_fourier_rhs() {
    std::vector<real> expected(kN);
    const auto lambda = [](int mx, int my, int mz) {
        const auto term = [](int mode) {
            return std::sin(kPi * static_cast<long double>(mode) / kNaxis);
        };
        return 4.0L * kNaxis * kNaxis *
            (term(mx) * term(mx) + term(my) * term(my) + term(mz) * term(mz));
    };
    const long double lambda_x = lambda(1, 0, 0);
    const long double lambda_y = lambda(0, 2, 0);
    const long double lambda_z = lambda(0, 0, 1);
    const long double lambda_mixed = lambda(1, 1, -1);
    for (std::size_t iz = 0; iz < kNaxis; ++iz) for (std::size_t iy = 0; iy < kNaxis; ++iy) for (std::size_t ix = 0; ix < kNaxis; ++ix) {
        const long double x = (static_cast<long double>(ix) + 0.5L) / kNaxis;
        const long double y = (static_cast<long double>(iy) + 0.5L) / kNaxis;
        const long double z = (static_cast<long double>(iz) + 0.5L) / kNaxis;
        expected[index(ix, iy, iz)] = static_cast<real>(
            0.45L * lambda_x * std::sin(2.0L * kPi * x) -
            0.30L * lambda_y * std::cos(4.0L * kPi * y) +
            0.20L * lambda_z * std::sin(2.0L * kPi * z) +
            0.10L * lambda_mixed * std::cos(2.0L * kPi * (x + y - z)));
    }
    return project_cpu(expected);
}

[[nodiscard]] double compatibility_defect_cpu(const std::vector<real>& values) {
    const double value_rms = rms(values);
    return value_rms > 0.0 ? std::abs(static_cast<double>(mean_ld(values))) / value_rms : 0.0;
}

[[nodiscard]] double raw_diagnostic_limit(double reference) {
    return kRawDiagnosticRoundoffFactor * std::numeric_limits<real>::epsilon() *
           std::max(std::abs(reference), 1.0);
}

[[nodiscard]] double gauge_limit_for(const std::vector<real>& values) {
    return kGaugeFactor * std::numeric_limits<real>::epsilon() * std::max(rms(values), 1.0);
}

[[nodiscard]] double reported_gauge_limit_for(const std::vector<real>& values) {
    return kReportedGaugeComparisonFactor * std::numeric_limits<real>::epsilon() *
           std::max(rms(values), 1.0);
}

[[nodiscard]] std::vector<real> residual_cpu(const std::vector<real>& q,
                                              const std::vector<real>& b,
                                              const std::vector<real>& x) {
    const auto ax = positive_diffusion_cpu(q, x);
    std::vector<real> residual(kN);
    for (std::size_t i = 0; i < kN; ++i) residual[i] = b[i] - ax[i];
    return project_cpu(residual);
}

[[nodiscard]] std::vector<real> raw_residual_cpu(const std::vector<real>& q,
                                                  const std::vector<real>& b,
                                                  const std::vector<real>& x) {
    const auto ax = positive_diffusion_cpu(q, x);
    std::vector<real> residual(kN);
    for (std::size_t i = 0; i < kN; ++i) residual[i] = b[i] - ax[i];
    return residual;
}

class IdentityPreconditioner {
  public:
    void apply(CudaContext& context, DeviceSpan<const real> r, DeviceSpan<real> z) const {
        blas::copy(context, r, z);
    }
};

class ZeroOperator {
  public:
    void apply(CudaContext& context, DeviceSpan<const real>, DeviceSpan<real> output) const {
        MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(output.data(), 0, output.size() * sizeof(real),
                                               context.cuda_stream()));
    }
};

class ZeroPreconditioner {
  public:
    void apply(CudaContext& context, DeviceSpan<const real>, DeviceSpan<real> output) const {
        MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(output.data(), 0, output.size() * sizeof(real),
                                               context.cuda_stream()));
    }
};

class CountingPreconditioner {
  public:
    void apply(CudaContext& context, DeviceSpan<const real> r, DeviceSpan<real> z) const {
        ++calls;
        blas::copy(context, r, z);
    }

    mutable int calls = 0;
};

[[nodiscard]] bool bitwise_equal(const std::vector<real>& left, const std::vector<real>& right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size() * sizeof(real)) == 0;
}

[[nodiscard]] const char* status_name(solvers::ProjectedPCGStatus status) {
    switch (status) {
        case solvers::ProjectedPCGStatus::converged: return "converged";
        case solvers::ProjectedPCGStatus::max_iterations: return "max_iterations";
        case solvers::ProjectedPCGStatus::invalid_configuration: return "invalid_configuration";
        case solvers::ProjectedPCGStatus::size_mismatch: return "size_mismatch";
        case solvers::ProjectedPCGStatus::aliasing: return "aliasing";
        case solvers::ProjectedPCGStatus::breakdown_pAp: return "breakdown_pAp";
        case solvers::ProjectedPCGStatus::breakdown_rz: return "breakdown_rz";
        case solvers::ProjectedPCGStatus::nonfinite_value: return "nonfinite_value";
    }
    return "unknown";
}

[[nodiscard]] CaseResult run_manufactured_case(const char* name, bool smooth) {
    const auto q = coefficient_field(smooth);
    const auto u_star = manufactured_solution();
    const auto b = project_cpu(positive_diffusion_cpu(q, u_star));
    const auto b_original = b;
    const Grid3D grid(static_cast<int>(kNaxis), static_cast<int>(kNaxis), static_cast<int>(kNaxis),
                      1.0 / kNaxis, 1.0 / kNaxis, 1.0 / kNaxis);

    CudaContext context(0);
    DeviceBuffer<real> d_q(kN), d_b(kN), d_x(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_q.data(), q.data(), kN * sizeof(real),
                                           cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b.data(), b.data(), kN * sizeof(real),
                                           cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x.data(), 0, kN * sizeof(real), context.cuda_stream()));

    operators::LesterPositiveDiffusionOperator A(grid, d_q.span());
    IdentityPreconditioner identity;
    constraints::MeanZeroProjector projector;
    solvers::ProjectedPCGWorkspace workspace;
    workspace.prepare(kN);
    solvers::ProjectedPCGConfig config;
    config.max_iter = 2000;
    config.check_every = kCheckEvery;
    config.rtol = kRtol;
    const auto result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b.span()), d_x.span(), config, projector, workspace);

    std::vector<real> x, b_after;
    x.resize(kN);
    b_after.resize(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(x.data(), d_x.data(), kN * sizeof(real),
                                           cudaMemcpyDeviceToHost, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(b_after.data(), d_b.data(), kN * sizeof(real),
                                           cudaMemcpyDeviceToHost, context.cuda_stream()));
    context.synchronize();

    const auto cpu_residual = project_cpu([&] {
        const auto ax = positive_diffusion_cpu(q, x);
        std::vector<real> residual(kN);
        for (std::size_t i = 0; i < kN; ++i) residual[i] = b_original[i] - ax[i];
        return residual;
    }());
    const double independent_residual = l2(cpu_residual);
    const double reported_residual = static_cast<double>(result.final_projected_residual);
    const double residual_scale = std::max({l2(b_original), l2(positive_diffusion_cpu(q, x)), 1.0});
    const double residual_difference = std::abs(independent_residual - reported_residual);
    const double residual_comparison_limit =
        kCpuGpuResidualRelativeTolerance * std::max({independent_residual, reported_residual, 1.0}) +
        kCpuGpuResidualRoundoffFactor * std::numeric_limits<real>::epsilon() * residual_scale;
    const double solution_error = rms_difference(project_cpu([&] {
        std::vector<real> error(kN);
        for (std::size_t i = 0; i < kN; ++i) error[i] = x[i] - u_star[i];
        return error;
    }()), std::vector<real>(kN, real(0.0)));
    const double field_rms = rms(x);
    const double cpu_field_mean = static_cast<double>(mean_ld(x));
    const double reported_field_mean = static_cast<double>(result.final_field_mean);
    const double gauge = std::abs(cpu_field_mean);
    const double gauge_limit = kGaugeFactor * std::numeric_limits<real>::epsilon() * std::max(field_rms, 1.0);
    const double reported_gauge = std::abs(reported_field_mean);
    const double reported_gauge_comparison_limit =
        kReportedGaugeComparisonFactor * std::numeric_limits<real>::epsilon() * std::max(field_rms, 1.0);
    const bool rhs_immutable = std::memcmp(b_original.data(), b_after.data(), kN * sizeof(real)) == 0;

    double fourier_rms_error = 0.0;
    double fourier_pointwise_error = 0.0;
    bool fourier_pass = true;
    if (!smooth) {
        const auto fourier = constant_fourier_rhs();
        fourier_rms_error = rms_difference(b_original, fourier) / std::max(rms(fourier), 1.0);
        fourier_pointwise_error = max_abs_difference(b_original, fourier) / std::max(rms(fourier), 1.0);
        fourier_pass = fourier_rms_error <= kFourierRelativeTolerance &&
                       fourier_pointwise_error <= kFourierPointwiseTolerance;
    }

    const bool pass = result.status == solvers::ProjectedPCGStatus::converged && result.converged &&
        static_cast<double>(result.relative_projected_residual) <= kResidualTolerance &&
        gauge <= gauge_limit && reported_gauge <= gauge_limit &&
        std::abs(reported_field_mean - cpu_field_mean) <= reported_gauge_comparison_limit &&
        solution_error <= kSolutionTolerance && rhs_immutable &&
        residual_difference <= residual_comparison_limit && fourier_pass;
    std::cout << std::setprecision(12) << "projected_pcg_metrics case=" << name
              << " status=" << status_name(result.status)
              << " iterations=" << result.iterations
              << " check_every=" << kCheckEvery
              << " rtol=" << kRtol
              << " relative_residual=" << result.relative_projected_residual
              << " reported_residual=" << reported_residual
              << " cpu_residual=" << independent_residual
              << " residual_difference=" << residual_difference
              << " residual_limit=" << residual_comparison_limit
              << " cpu_field_mean=" << cpu_field_mean
              << " reported_field_mean=" << reported_field_mean
              << " gauge=" << gauge
              << " reported_gauge=" << reported_gauge
              << " gauge_limit=" << gauge_limit
              << " reported_gauge_comparison_limit=" << reported_gauge_comparison_limit
              << " solution_rms_error=" << solution_error
              << " rhs_immutable=" << (rhs_immutable ? "true" : "false");
    if (!smooth) {
        std::cout << " fourier_rms_relative_error=" << fourier_rms_error
                  << " fourier_pointwise_relative_error=" << fourier_pointwise_error;
    }
    std::cout << '\n';

    std::ostringstream threshold;
    threshold << "status=converged; relres<=1e-10; gauge<=100eps*max(RMS(x),1); "
              << "RMS(P_CPU(x-u*))<=5e-9; immutable RHS; CPU residual agreement";
    if (!smooth) threshold << "; Fourier RMS<=1e-11, pointwise/RMS<=1e-10";
    return {pass, name, "gpu-projected-pcg", "17x17x17 cell-centered periodic (N=4913)",
            static_cast<double>(result.relative_projected_residual), solution_error,
            "n/a", "n/a", threshold.str()};
}

[[nodiscard]] CaseResult case_projected_pcg_constant_manufactured() {
    return run_manufactured_case("projected_pcg_constant_manufactured", false);
}

[[nodiscard]] CaseResult case_projected_pcg_smooth_manufactured() {
    return run_manufactured_case("projected_pcg_smooth_manufactured", true);
}

[[nodiscard]] CaseResult case_projected_pcg_incompatible_rhs_contract() {
    const auto q = coefficient_field(false);
    const auto u_star = manufactured_solution();
    const auto b_compatible = project_cpu(positive_diffusion_cpu(q, u_star));
    std::vector<real> b_raw = b_compatible;
    for (real& value : b_raw) value += real(0.375);
    const auto b_raw_original = b_raw;
    const Grid3D grid(static_cast<int>(kNaxis), static_cast<int>(kNaxis), static_cast<int>(kNaxis),
                      1.0 / kNaxis, 1.0 / kNaxis, 1.0 / kNaxis);

    CudaContext context(0);
    DeviceBuffer<real> d_q(kN), d_raw(kN), d_compatible(kN), d_x_raw(kN), d_x_compatible(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_q.data(), q.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_raw.data(), b_raw.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_compatible.data(), b_compatible.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x_raw.data(), 0, kN * sizeof(real), context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x_compatible.data(), 0, kN * sizeof(real), context.cuda_stream()));

    operators::LesterPositiveDiffusionOperator A(grid, d_q.span());
    IdentityPreconditioner identity;
    constraints::MeanZeroProjector projector;
    solvers::ProjectedPCGConfig config;
    config.max_iter = 2000;
    config.check_every = kCheckEvery;
    config.rtol = kRtol;
    solvers::ProjectedPCGWorkspace raw_workspace, compatible_workspace;
    raw_workspace.prepare(kN);
    compatible_workspace.prepare(kN);
    const auto raw_result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_raw.span()), d_x_raw.span(), config, projector, raw_workspace);
    const auto compatible_result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_compatible.span()), d_x_compatible.span(), config, projector, compatible_workspace);

    std::vector<real> x_raw(kN), x_compatible(kN), raw_after(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(x_raw.data(), d_x_raw.data(), kN * sizeof(real), cudaMemcpyDeviceToHost,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(x_compatible.data(), d_x_compatible.data(), kN * sizeof(real), cudaMemcpyDeviceToHost,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(raw_after.data(), d_raw.data(), kN * sizeof(real), cudaMemcpyDeviceToHost,
                                           context.cuda_stream()));
    context.synchronize();

    const double cpu_raw_mean = static_cast<double>(mean_ld(b_raw));
    const double cpu_raw_l2 = l2(b_raw);
    const double cpu_raw_defect = compatibility_defect_cpu(b_raw);
    const double raw_residual = l2(residual_cpu(q, b_raw, x_raw));
    const double raw_residual_difference = std::abs(raw_residual - static_cast<double>(raw_result.final_projected_residual));
    const double raw_residual_limit = kCpuGpuResidualRelativeTolerance *
        std::max({raw_residual, static_cast<double>(raw_result.final_projected_residual), 1.0}) +
        kCpuGpuResidualRoundoffFactor * std::numeric_limits<real>::epsilon() *
        std::max({l2(b_raw), l2(positive_diffusion_cpu(q, x_raw)), 1.0});
    const double raw_solution_error = rms_difference(project_cpu([&] {
        std::vector<real> error(kN);
        for (std::size_t i = 0; i < kN; ++i) error[i] = x_raw[i] - u_star[i];
        return error;
    }()), std::vector<real>(kN, real(0.0)));
    const double compatible_solution_error = rms_difference(project_cpu([&] {
        std::vector<real> error(kN);
        for (std::size_t i = 0; i < kN; ++i) error[i] = x_compatible[i] - u_star[i];
        return error;
    }()), std::vector<real>(kN, real(0.0)));
    std::vector<real> solution_difference(kN);
    for (std::size_t i = 0; i < kN; ++i) solution_difference[i] = x_raw[i] - x_compatible[i];
    const double projected_solution_difference = rms(project_cpu(solution_difference));
    const double raw_gauge = std::abs(static_cast<double>(mean_ld(x_raw)));
    const double compatible_gauge = std::abs(static_cast<double>(mean_ld(x_compatible)));
    const bool raw_rhs_immutable = std::memcmp(b_raw_original.data(), raw_after.data(), kN * sizeof(real)) == 0;
    const bool diagnostics_match =
        std::abs(cpu_raw_mean - 0.375) <= kRawMeanTolerance &&
        std::abs(static_cast<double>(raw_result.raw_rhs_mean) - cpu_raw_mean) <= kRawMeanTolerance &&
        std::abs(static_cast<double>(raw_result.raw_rhs_l2_norm) - cpu_raw_l2) <= raw_diagnostic_limit(cpu_raw_l2) &&
        std::abs(static_cast<double>(raw_result.raw_rhs_compatibility_defect) - cpu_raw_defect) <= raw_diagnostic_limit(cpu_raw_defect);
    const bool pass = raw_result.converged && compatible_result.converged &&
        raw_result.status == solvers::ProjectedPCGStatus::converged &&
        compatible_result.status == solvers::ProjectedPCGStatus::converged &&
        static_cast<double>(raw_result.relative_projected_residual) <= kResidualTolerance &&
        static_cast<double>(compatible_result.relative_projected_residual) <= kResidualTolerance &&
        raw_gauge <= gauge_limit_for(x_raw) && compatible_gauge <= gauge_limit_for(x_compatible) &&
        std::abs(static_cast<double>(raw_result.final_field_mean) - static_cast<double>(mean_ld(x_raw))) <= reported_gauge_limit_for(x_raw) &&
        std::abs(static_cast<double>(compatible_result.final_field_mean) - static_cast<double>(mean_ld(x_compatible))) <= reported_gauge_limit_for(x_compatible) &&
        raw_solution_error <= kSolutionTolerance && compatible_solution_error <= kSolutionTolerance &&
        projected_solution_difference <= kSolutionTolerance && raw_rhs_immutable && diagnostics_match &&
        raw_residual_difference <= raw_residual_limit;
    std::cout << std::setprecision(12) << "projected_pcg_contract case=projected_pcg_incompatible_rhs_contract"
              << " raw_iterations=" << raw_result.iterations << " compatible_iterations=" << compatible_result.iterations
              << " raw_mean=" << raw_result.raw_rhs_mean << " cpu_raw_mean=" << cpu_raw_mean
              << " raw_l2=" << raw_result.raw_rhs_l2_norm << " cpu_raw_l2=" << cpu_raw_l2
              << " raw_defect=" << raw_result.raw_rhs_compatibility_defect << " cpu_raw_defect=" << cpu_raw_defect
              << " reported_residual=" << raw_result.final_projected_residual << " cpu_residual=" << raw_residual
              << " residual_difference=" << raw_residual_difference << " raw_gauge=" << raw_gauge
              << " solution_error=" << raw_solution_error << " compatible_solution_error=" << compatible_solution_error
              << " projected_solution_difference=" << projected_solution_difference
              << " raw_rhs_immutable=" << (raw_rhs_immutable ? "true" : "false") << '\n';
    return {pass, "projected_pcg_incompatible_rhs_contract", "gpu-projected-pcg-contract",
            "17x17x17 q=1 periodic (N=4913)", static_cast<double>(raw_result.relative_projected_residual),
            projected_solution_difference, "n/a", "n/a",
            "incompatible constant=0.375 diagnosed/projected; relres<=1e-10; CPU residual/gauge/solution contracts"};
}

[[nodiscard]] CaseResult case_projected_pcg_initial_gauge_contract() {
    const auto q = coefficient_field(false);
    const auto u_star = manufactured_solution();
    const auto b = project_cpu(positive_diffusion_cpu(q, u_star));
    const auto b_original = b;
    std::vector<real> x_initial(kN);
    for (std::size_t iz = 0; iz < kNaxis; ++iz) for (std::size_t iy = 0; iy < kNaxis; ++iy) for (std::size_t ix = 0; ix < kNaxis; ++ix) {
        const long double x = (static_cast<long double>(ix) + 0.5L) / kNaxis;
        const long double y = (static_cast<long double>(iy) + 0.5L) / kNaxis;
        x_initial[index(ix, iy, iz)] = static_cast<real>(2.75L + 0.17L * std::sin(2.0L * kPi * x) -
                                                          0.11L * std::cos(2.0L * kPi * y));
    }
    const double initial_mean = static_cast<double>(mean_ld(x_initial));
    const Grid3D grid(static_cast<int>(kNaxis), static_cast<int>(kNaxis), static_cast<int>(kNaxis),
                      1.0 / kNaxis, 1.0 / kNaxis, 1.0 / kNaxis);
    CudaContext context(0);
    DeviceBuffer<real> d_q(kN), d_b(kN), d_x(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_q.data(), q.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b.data(), b.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_x.data(), x_initial.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    operators::LesterPositiveDiffusionOperator A(grid, d_q.span());
    IdentityPreconditioner identity;
    constraints::MeanZeroProjector projector;
    solvers::ProjectedPCGWorkspace workspace;
    workspace.prepare(kN);
    solvers::ProjectedPCGConfig config;
    config.max_iter = 2000; config.check_every = kCheckEvery; config.rtol = kRtol;
    const auto result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b.span()), d_x.span(), config, projector, workspace);
    std::vector<real> x(kN), b_after(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(x.data(), d_x.data(), kN * sizeof(real), cudaMemcpyDeviceToHost, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(b_after.data(), d_b.data(), kN * sizeof(real), cudaMemcpyDeviceToHost, context.cuda_stream()));
    context.synchronize();
    const double cpu_final_mean = static_cast<double>(mean_ld(x));
    const double solution_error = rms_difference(project_cpu([&] {
        std::vector<real> error(kN);
        for (std::size_t i = 0; i < kN; ++i) error[i] = x[i] - u_star[i];
        return error;
    }()), std::vector<real>(kN, real(0.0)));
    const bool rhs_immutable = std::memcmp(b_original.data(), b_after.data(), kN * sizeof(real)) == 0;
    const bool pass = result.converged && result.status == solvers::ProjectedPCGStatus::converged &&
        static_cast<double>(result.relative_projected_residual) <= kResidualTolerance &&
        std::abs(initial_mean - 2.75) <= kInitialGaugeTolerance &&
        std::abs(cpu_final_mean) <= gauge_limit_for(x) &&
        std::abs(static_cast<double>(result.final_field_mean)) <= gauge_limit_for(x) &&
        std::abs(static_cast<double>(result.final_field_mean) - cpu_final_mean) <= reported_gauge_limit_for(x) &&
        solution_error <= kSolutionTolerance && rhs_immutable;
    std::cout << std::setprecision(12) << "projected_pcg_contract case=projected_pcg_initial_gauge_contract"
              << " iterations=" << result.iterations << " initial_mean=" << initial_mean
              << " cpu_final_mean=" << cpu_final_mean << " reported_final_mean=" << result.final_field_mean
              << " relative_residual=" << result.relative_projected_residual
              << " solution_error=" << solution_error << " rhs_immutable=" << (rhs_immutable ? "true" : "false") << '\n';
    return {pass, "projected_pcg_initial_gauge_contract", "gpu-projected-pcg-contract",
            "17x17x17 q=1 periodic (N=4913)", static_cast<double>(result.relative_projected_residual),
            solution_error, "n/a", "n/a",
            "initial mean=2.75; projected final gauge; relres<=1e-10; RMS(P_CPU(x-u*))<=5e-9; immutable RHS"};
}

[[nodiscard]] CaseResult case_projected_pcg_legacy_api_unchanged() {
    const auto q = coefficient_field(false);
    const auto u_star = manufactured_solution();
    const auto b = project_cpu(positive_diffusion_cpu(q, u_star));
    const Grid3D grid(static_cast<int>(kNaxis), static_cast<int>(kNaxis), static_cast<int>(kNaxis),
                      1.0 / kNaxis, 1.0 / kNaxis, 1.0 / kNaxis);
    CudaContext context(0);
    DeviceBuffer<real> d_q(kN), d_b_legacy(kN), d_b_projected(kN), d_x_legacy(kN), d_x_projected(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_q.data(), q.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b_legacy.data(), b.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b_projected.data(), b.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x_legacy.data(), 0, kN * sizeof(real), context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x_projected.data(), 0, kN * sizeof(real), context.cuda_stream()));
    operators::LesterPositiveDiffusionOperator A(grid, d_q.span());
    IdentityPreconditioner identity;
    solvers::PCGConfig legacy_config;
    legacy_config.max_iter = 2000; legacy_config.check_every = kCheckEvery; legacy_config.rtol = kRtol;
    solvers::PCGWorkspace legacy_workspace;
    const auto legacy_result = solvers::pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b_legacy.span()), d_x_legacy.span(), legacy_config, legacy_workspace);
    constraints::MeanZeroProjector projector;
    solvers::ProjectedPCGConfig projected_config;
    projected_config.max_iter = 2000; projected_config.check_every = kCheckEvery; projected_config.rtol = kRtol;
    solvers::ProjectedPCGWorkspace projected_workspace;
    projected_workspace.prepare(kN);
    const auto projected_result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b_projected.span()), d_x_projected.span(), projected_config, projector, projected_workspace);
    std::vector<real> x_legacy(kN), x_projected(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(x_legacy.data(), d_x_legacy.data(), kN * sizeof(real), cudaMemcpyDeviceToHost, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(x_projected.data(), d_x_projected.data(), kN * sizeof(real), cudaMemcpyDeviceToHost, context.cuda_stream()));
    context.synchronize();
    std::vector<real> difference(kN);
    for (std::size_t i = 0; i < kN; ++i) difference[i] = x_legacy[i] - x_projected[i];
    const double quotient_difference = rms(project_cpu(difference));
    const double projected_cpu_residual = l2(residual_cpu(q, b, x_projected));
    const double projected_residual_difference = std::abs(projected_cpu_residual - static_cast<double>(projected_result.final_projected_residual));
    const double projected_residual_limit = kCpuGpuResidualRelativeTolerance *
        std::max({projected_cpu_residual, static_cast<double>(projected_result.final_projected_residual), 1.0}) +
        kCpuGpuResidualRoundoffFactor * std::numeric_limits<real>::epsilon() *
        std::max({l2(b), l2(positive_diffusion_cpu(q, x_projected)), 1.0});
    const bool pass = legacy_result.converged && projected_result.converged &&
        projected_result.status == solvers::ProjectedPCGStatus::converged &&
        quotient_difference <= kSolutionTolerance &&
        static_cast<double>(projected_result.relative_projected_residual) <= kResidualTolerance &&
        std::abs(static_cast<double>(mean_ld(x_projected))) <= gauge_limit_for(x_projected) &&
        projected_residual_difference <= projected_residual_limit;
    std::cout << std::setprecision(12) << "projected_pcg_contract case=projected_pcg_legacy_api_unchanged"
              << " legacy_iterations=" << legacy_result.iterations << " legacy_residual=" << legacy_result.final_residual
              << " projected_iterations=" << projected_result.iterations << " projected_residual=" << projected_result.final_projected_residual
              << " projected_cpu_residual=" << projected_cpu_residual << " quotient_difference=" << quotient_difference
              << " projected_gauge=" << mean_ld(x_projected) << '\n';
    return {pass, "projected_pcg_legacy_api_unchanged", "gpu-projected-pcg-contract",
            "17x17x17 q=1 periodic (N=4913)", static_cast<double>(projected_result.relative_projected_residual),
            quotient_difference, "n/a", "n/a",
            "legacy seven-argument pcg_solve remains opt-in unchanged; RMS(P_CPU(xlegacy-xprojected))<=5e-9"};
}

[[nodiscard]] CaseResult case_projected_pcg_error_status_contract() {
    const auto q = coefficient_field(false);
    const auto u_star = manufactured_solution();
    const auto b = project_cpu(positive_diffusion_cpu(q, u_star));
    std::vector<real> b_sentinel(kN), x_sentinel(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        b_sentinel[i] = static_cast<real>(-3.25L + static_cast<long double>(i) / kN);
        x_sentinel[i] = static_cast<real>(7.5L - static_cast<long double>(i) / (2 * kN));
    }
    const Grid3D grid(static_cast<int>(kNaxis), static_cast<int>(kNaxis), static_cast<int>(kNaxis),
                      1.0 / kNaxis, 1.0 / kNaxis, 1.0 / kNaxis);
    const solvers::ProjectedPCGConfig invalid_check_every{10, 0, real(1.0e-12)};
    const solvers::ProjectedPCGConfig normal_config{10, 1, real(1.0e-12)};
    const solvers::ProjectedPCGConfig zero_iteration_config{0, 1, real(0.5)};
    const solvers::ProjectedPCGConfig initial_convergence_config{0, 1, real(1.0)};

    CudaContext context(0);
    DeviceBuffer<real> d_q(kN), d_b(kN), d_x(kN), d_shared(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_q.data(), q.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    operators::LesterPositiveDiffusionOperator A(grid, d_q.span());
    IdentityPreconditioner identity;
    ZeroOperator zero_operator;
    ZeroPreconditioner zero_preconditioner;
    constraints::MeanZeroProjector projector;

    solvers::ProjectedPCGWorkspace invalid_workspace;
    invalid_workspace.prepare(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b.data(), b_sentinel.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_x.data(), x_sentinel.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    const auto invalid_result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b.span()), d_x.span(), invalid_check_every, projector, invalid_workspace);
    std::vector<real> invalid_b_after(kN), invalid_x_after(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(invalid_b_after.data(), d_b.data(), kN * sizeof(real), cudaMemcpyDeviceToHost,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(invalid_x_after.data(), d_x.data(), kN * sizeof(real), cudaMemcpyDeviceToHost,
                                           context.cuda_stream()));
    context.synchronize();
    const bool invalid_sentinels = bitwise_equal(b_sentinel, invalid_b_after) &&
                                   bitwise_equal(x_sentinel, invalid_x_after);

    solvers::ProjectedPCGWorkspace size_workspace;
    size_workspace.prepare(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b.data(), b_sentinel.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_x.data(), x_sentinel.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    const auto size_result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b.data(), kN - 1), d_x.span(), normal_config, projector, size_workspace);
    std::vector<real> size_b_after(kN), size_x_after(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(size_b_after.data(), d_b.data(), kN * sizeof(real), cudaMemcpyDeviceToHost,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(size_x_after.data(), d_x.data(), kN * sizeof(real), cudaMemcpyDeviceToHost,
                                           context.cuda_stream()));
    context.synchronize();
    const bool size_sentinels = bitwise_equal(b_sentinel, size_b_after) && bitwise_equal(x_sentinel, size_x_after);

    solvers::ProjectedPCGWorkspace same_span_workspace;
    same_span_workspace.prepare(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_shared.data(), x_sentinel.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    const auto same_span_result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_shared.span()), d_shared.span(), normal_config, projector, same_span_workspace);
    std::vector<real> same_span_after(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(same_span_after.data(), d_shared.data(), kN * sizeof(real), cudaMemcpyDeviceToHost,
                                           context.cuda_stream()));
    context.synchronize();
    const bool same_span_sentinel = bitwise_equal(x_sentinel, same_span_after);

    solvers::ProjectedPCGWorkspace workspace_alias_workspace;
    workspace_alias_workspace.prepare(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b.data(), b_sentinel.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(workspace_alias_workspace.pcg.r.data(), x_sentinel.data(), kN * sizeof(real),
                                           cudaMemcpyHostToDevice, context.cuda_stream()));
    const auto workspace_alias_result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b.span()), workspace_alias_workspace.pcg.r.span(), normal_config, projector,
        workspace_alias_workspace);
    std::vector<real> workspace_alias_b_after(kN), workspace_alias_x_after(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(workspace_alias_b_after.data(), d_b.data(), kN * sizeof(real), cudaMemcpyDeviceToHost,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(workspace_alias_x_after.data(), workspace_alias_workspace.pcg.r.data(),
                                           kN * sizeof(real), cudaMemcpyDeviceToHost, context.cuda_stream()));
    context.synchronize();
    const bool workspace_alias_sentinels = bitwise_equal(b_sentinel, workspace_alias_b_after) &&
                                         bitwise_equal(x_sentinel, workspace_alias_x_after);

    solvers::ProjectedPCGWorkspace pap_workspace;
    pap_workspace.prepare(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b.data(), b.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x.data(), 0, kN * sizeof(real), context.cuda_stream()));
    const auto pap_result = solvers::projected_pcg_solve(context, zero_operator, identity,
        DeviceSpan<const real>(d_b.span()), d_x.span(), normal_config, projector, pap_workspace);

    solvers::ProjectedPCGWorkspace rz_workspace;
    rz_workspace.prepare(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b.data(), b.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x.data(), 0, kN * sizeof(real), context.cuda_stream()));
    const auto rz_result = solvers::projected_pcg_solve(context, A, zero_preconditioner,
        DeviceSpan<const real>(d_b.span()), d_x.span(), normal_config, projector, rz_workspace);

    std::vector<real> nan_rhs = b;
    nan_rhs[kN / 2] = std::numeric_limits<real>::quiet_NaN();
    solvers::ProjectedPCGWorkspace nan_workspace;
    nan_workspace.prepare(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b.data(), nan_rhs.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x.data(), 0, kN * sizeof(real), context.cuda_stream()));
    const auto nan_result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b.span()), d_x.span(), normal_config, projector, nan_workspace);

    CountingPreconditioner zero_iteration_preconditioner;
    solvers::ProjectedPCGWorkspace zero_iteration_workspace;
    zero_iteration_workspace.prepare(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b.data(), b.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x.data(), 0, kN * sizeof(real), context.cuda_stream()));
    const auto zero_iteration_result = solvers::projected_pcg_solve(context, A, zero_iteration_preconditioner,
        DeviceSpan<const real>(d_b.span()), d_x.span(), zero_iteration_config, projector, zero_iteration_workspace);

    CountingPreconditioner initial_convergence_preconditioner;
    solvers::ProjectedPCGWorkspace initial_convergence_workspace;
    initial_convergence_workspace.prepare(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b.data(), b.data(), kN * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x.data(), 0, kN * sizeof(real), context.cuda_stream()));
    const auto initial_convergence_result = solvers::projected_pcg_solve(context, A, initial_convergence_preconditioner,
        DeviceSpan<const real>(d_b.span()), d_x.span(), initial_convergence_config, projector, initial_convergence_workspace);

    const bool statuses = invalid_result.status == solvers::ProjectedPCGStatus::invalid_configuration &&
        size_result.status == solvers::ProjectedPCGStatus::size_mismatch &&
        same_span_result.status == solvers::ProjectedPCGStatus::aliasing &&
        workspace_alias_result.status == solvers::ProjectedPCGStatus::aliasing &&
        pap_result.status == solvers::ProjectedPCGStatus::breakdown_pAp &&
        rz_result.status == solvers::ProjectedPCGStatus::breakdown_rz &&
        nan_result.status == solvers::ProjectedPCGStatus::nonfinite_value &&
        zero_iteration_result.status == solvers::ProjectedPCGStatus::max_iterations &&
        initial_convergence_result.status == solvers::ProjectedPCGStatus::converged;
    const bool no_unexpected_convergence = !invalid_result.converged && !size_result.converged &&
        !same_span_result.converged && !workspace_alias_result.converged && !pap_result.converged &&
        !rz_result.converged && !nan_result.converged && !zero_iteration_result.converged &&
        initial_convergence_result.converged;
    const bool iteration_and_call_contract = zero_iteration_result.iterations == 0 &&
        zero_iteration_preconditioner.calls == 0 && initial_convergence_result.iterations == 0 &&
        initial_convergence_preconditioner.calls == 0;
    const bool pass = statuses && no_unexpected_convergence && iteration_and_call_contract && invalid_sentinels &&
        size_sentinels && same_span_sentinel && workspace_alias_sentinels;
    std::cout << "projected_pcg_contract case=projected_pcg_error_status_contract"
              << " invalid=" << status_name(invalid_result.status) << " invalid_sentinels=" << invalid_sentinels
              << " size=" << status_name(size_result.status) << " size_sentinels=" << size_sentinels
              << " same_span=" << status_name(same_span_result.status) << " same_span_sentinel=" << same_span_sentinel
              << " workspace_alias=" << status_name(workspace_alias_result.status)
              << " workspace_alias_sentinels=" << workspace_alias_sentinels
              << " pAp=" << status_name(pap_result.status) << " rz=" << status_name(rz_result.status)
              << " nan=" << status_name(nan_result.status)
              << " max0=" << status_name(zero_iteration_result.status) << ":iter=" << zero_iteration_result.iterations
              << ":calls=" << zero_iteration_preconditioner.calls
              << " rtol1=" << status_name(initial_convergence_result.status) << ":iter=" << initial_convergence_result.iterations
              << ":calls=" << initial_convergence_preconditioner.calls << '\n';
    return {pass, "projected_pcg_error_status_contract", "gpu-projected-pcg-contract",
            "17x17x17 q=1 periodic (N=4913)", 0.0, 0.0, "n/a", "n/a",
            "eight status paths; invalid/size/alias sentinels immutable; only rtol=1 initial check converges"};
}

[[nodiscard]] CaseResult case_projected_pcg_mutant_no_rhs_projection() {
    const auto q = coefficient_field(false);
    const auto u_star = manufactured_solution();
    const auto b_compatible = project_cpu(positive_diffusion_cpu(q, u_star));
    std::vector<real> b_raw = b_compatible;
    for (real& value : b_raw) value += real(0.375);
    const Grid3D grid(static_cast<int>(kNaxis), static_cast<int>(kNaxis), static_cast<int>(kNaxis),
                      1.0 / kNaxis, 1.0 / kNaxis, 1.0 / kNaxis);
    const solvers::PCGConfig legacy_config{200, 5, real(1.0e-12), false};
    const solvers::ProjectedPCGConfig projected_config{2000, kCheckEvery, kRtol};

    CudaContext context(0);
    DeviceBuffer<real> d_q(kN), d_b_legacy(kN), d_b_projected(kN), d_x_legacy(kN), d_x_projected(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_q.data(), q.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b_legacy.data(), b_raw.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b_projected.data(), b_raw.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x_legacy.data(), 0, kN * sizeof(real), context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_x_projected.data(), 0, kN * sizeof(real), context.cuda_stream()));
    operators::LesterPositiveDiffusionOperator A(grid, d_q.span());
    IdentityPreconditioner identity;
    solvers::PCGWorkspace legacy_workspace;
    const auto legacy_result = solvers::pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b_legacy.span()), d_x_legacy.span(), legacy_config, legacy_workspace);
    constraints::MeanZeroProjector projector;
    solvers::ProjectedPCGWorkspace projected_workspace;
    projected_workspace.prepare(kN);
    const auto projected_result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b_projected.span()), d_x_projected.span(), projected_config, projector, projected_workspace);
    std::vector<real> x_legacy(kN), x_projected(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(x_legacy.data(), d_x_legacy.data(), kN * sizeof(real), cudaMemcpyDeviceToHost, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(x_projected.data(), d_x_projected.data(), kN * sizeof(real), cudaMemcpyDeviceToHost, context.cuda_stream()));
    context.synchronize();
    const double legacy_raw_relative_residual = l2(raw_residual_cpu(q, b_raw, x_legacy)) / l2(b_raw);
    const double projected_solution_error = rms_difference(project_cpu([&] {
        std::vector<real> error(kN);
        for (std::size_t i = 0; i < kN; ++i) error[i] = x_projected[i] - u_star[i];
        return error;
    }()), std::vector<real>(kN, real(0.0)));
    const double projected_gauge = std::abs(static_cast<double>(mean_ld(x_projected)));
    const bool projected_pass = projected_result.status == solvers::ProjectedPCGStatus::converged &&
        projected_result.converged && static_cast<double>(projected_result.relative_projected_residual) <= kResidualTolerance &&
        projected_gauge <= gauge_limit_for(x_projected) && projected_solution_error <= kSolutionTolerance;
    const bool mutant_rejected = !legacy_result.converged || !std::isfinite(legacy_raw_relative_residual) ||
        legacy_raw_relative_residual > 1.0e-6;
    const bool pass = projected_pass && mutant_rejected;
    std::cout << std::setprecision(12) << "projected_pcg_mutant case=projected_pcg_mutant_no_rhs_projection"
              << " legacy_converged=" << (legacy_result.converged ? "true" : "false")
              << " legacy_iterations=" << legacy_result.iterations
              << " legacy_raw_relative_residual=" << legacy_raw_relative_residual
              << " projected_status=" << status_name(projected_result.status)
              << " projected_relative_residual=" << projected_result.relative_projected_residual
              << " projected_gauge=" << projected_gauge
              << " projected_solution_error=" << projected_solution_error << '\n';
    return {pass, "projected_pcg_mutant_no_rhs_projection", "gpu-projected-pcg-mutant",
            "17x17x17 q=1, b_raw=b_compatible+0.375", static_cast<double>(projected_result.relative_projected_residual),
            legacy_raw_relative_residual, "n/a", "n/a",
            "projected solve converges/gauges; unprojected legacy fails or raw CPU ||b_raw-Ax||/||b_raw||>1e-6"};
}

[[nodiscard]] CaseResult case_projected_pcg_mutant_no_x0_projection() {
    const auto q = coefficient_field(false);
    const auto u_star = manufactured_solution();
    const auto b = project_cpu(positive_diffusion_cpu(q, u_star));
    std::vector<real> x0(kN);
    for (std::size_t iz = 0; iz < kNaxis; ++iz) for (std::size_t iy = 0; iy < kNaxis; ++iy) for (std::size_t ix = 0; ix < kNaxis; ++ix) {
        const long double x = (static_cast<long double>(ix) + 0.5L) / kNaxis;
        const long double y = (static_cast<long double>(iy) + 0.5L) / kNaxis;
        x0[index(ix, iy, iz)] = static_cast<real>(2.75L + 0.17L * std::sin(2.0L * kPi * x) -
                                                   0.11L * std::cos(2.0L * kPi * y));
    }
    const Grid3D grid(static_cast<int>(kNaxis), static_cast<int>(kNaxis), static_cast<int>(kNaxis),
                      1.0 / kNaxis, 1.0 / kNaxis, 1.0 / kNaxis);
    const solvers::PCGConfig legacy_config{2000, kCheckEvery, kRtol, false};
    const solvers::ProjectedPCGConfig projected_config{2000, kCheckEvery, kRtol};

    CudaContext context(0);
    DeviceBuffer<real> d_q(kN), d_b_legacy(kN), d_b_projected(kN), d_x_legacy(kN), d_x_projected(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_q.data(), q.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b_legacy.data(), b.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_b_projected.data(), b.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_x_legacy.data(), x0.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_x_projected.data(), x0.data(), kN * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    operators::LesterPositiveDiffusionOperator A(grid, d_q.span());
    IdentityPreconditioner identity;
    solvers::PCGWorkspace legacy_workspace;
    const auto legacy_result = solvers::pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b_legacy.span()), d_x_legacy.span(), legacy_config, legacy_workspace);
    constraints::MeanZeroProjector projector;
    solvers::ProjectedPCGWorkspace projected_workspace;
    projected_workspace.prepare(kN);
    const auto projected_result = solvers::projected_pcg_solve(context, A, identity,
        DeviceSpan<const real>(d_b_projected.span()), d_x_projected.span(), projected_config, projector, projected_workspace);
    std::vector<real> x_legacy(kN), x_projected(kN);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(x_legacy.data(), d_x_legacy.data(), kN * sizeof(real), cudaMemcpyDeviceToHost, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(x_projected.data(), d_x_projected.data(), kN * sizeof(real), cudaMemcpyDeviceToHost, context.cuda_stream()));
    context.synchronize();
    const auto quotient_error = [&](const std::vector<real>& x) {
        std::vector<real> error(kN);
        for (std::size_t i = 0; i < kN; ++i) error[i] = x[i] - u_star[i];
        return rms(project_cpu(error));
    };
    const double legacy_error = quotient_error(x_legacy);
    const double projected_error = quotient_error(x_projected);
    const double legacy_mean = static_cast<double>(mean_ld(x_legacy));
    const double projected_mean = static_cast<double>(mean_ld(x_projected));
    const bool quotient_solutions = std::isfinite(legacy_error) && std::isfinite(projected_error) &&
        legacy_error <= kSolutionTolerance && projected_error <= kSolutionTolerance;
    const bool projected_pass = projected_result.status == solvers::ProjectedPCGStatus::converged && projected_result.converged &&
        static_cast<double>(projected_result.relative_projected_residual) <= kResidualTolerance &&
        std::abs(projected_mean) <= gauge_limit_for(x_projected) &&
        std::abs(static_cast<double>(projected_result.final_field_mean) - projected_mean) <= reported_gauge_limit_for(x_projected);
    const bool mutant_rejected = legacy_result.converged && std::abs(legacy_mean) > 1.0e-6;
    const bool pass = quotient_solutions && projected_pass && mutant_rejected;
    std::cout << std::setprecision(12) << "projected_pcg_mutant case=projected_pcg_mutant_no_x0_projection"
              << " legacy_converged=" << (legacy_result.converged ? "true" : "false")
              << " legacy_iterations=" << legacy_result.iterations << " legacy_mean=" << legacy_mean
              << " legacy_quotient_error=" << legacy_error << " projected_status=" << status_name(projected_result.status)
              << " projected_mean=" << projected_mean << " projected_quotient_error=" << projected_error
              << " projected_relative_residual=" << projected_result.relative_projected_residual << '\n';
    return {pass, "projected_pcg_mutant_no_x0_projection", "gpu-projected-pcg-mutant",
            "17x17x17 q=1, compatible b, x0=2.75+T05B modes", projected_error, legacy_error, "n/a", "n/a",
            "both quotient errors<=5e-9 when finite; projected solve gauges; legacy constant survives with |mean|>1e-6"};
}

} // namespace

CaseRegistry projected_pcg_case_registry() {
    return {
        {"projected_pcg_constant_manufactured", case_projected_pcg_constant_manufactured},
        {"projected_pcg_smooth_manufactured", case_projected_pcg_smooth_manufactured},
        {"projected_pcg_incompatible_rhs_contract", case_projected_pcg_incompatible_rhs_contract},
        {"projected_pcg_initial_gauge_contract", case_projected_pcg_initial_gauge_contract},
        {"projected_pcg_legacy_api_unchanged", case_projected_pcg_legacy_api_unchanged},
        {"projected_pcg_error_status_contract", case_projected_pcg_error_status_contract},
        {"projected_pcg_mutant_no_rhs_projection", case_projected_pcg_mutant_no_rhs_projection},
        {"projected_pcg_mutant_no_x0_projection", case_projected_pcg_mutant_no_x0_projection},
    };
}

} // namespace macroflow3d::streamfunctions::test
