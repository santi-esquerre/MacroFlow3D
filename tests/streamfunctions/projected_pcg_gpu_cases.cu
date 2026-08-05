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

class IdentityPreconditioner {
  public:
    void apply(CudaContext& context, DeviceSpan<const real> r, DeviceSpan<real> z) const {
        blas::copy(context, r, z);
    }
};

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

} // namespace

CaseRegistry projected_pcg_case_registry() {
    return {
        {"projected_pcg_constant_manufactured", case_projected_pcg_constant_manufactured},
        {"projected_pcg_smooth_manufactured", case_projected_pcg_smooth_manufactured},
    };
}

} // namespace macroflow3d::streamfunctions::test
