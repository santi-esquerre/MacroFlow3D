#include "reference_operators.hpp"
#include "streamfunction_operator_test_cases.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/operators/lester_positive_diffusion_operator.cuh"
#include "src/physics/streamfunctions/ContinuationController.hpp"
#include "src/physics/streamfunctions/DifferentialOperators.cuh"
#include "src/physics/streamfunctions/NonlinearSources.cuh"
#include "src/physics/streamfunctions/ResidualEvaluator.cuh"
#include "src/physics/streamfunctions/StreamfunctionTypes.hpp"
#include "src/physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "src/physics/streamfunctions/affine_gauge.cuh"
#include "src/physics/streamfunctions/affine_periodic_rhs.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace macroflow3d::streamfunctions::test {
namespace {
namespace ref = macroflow3d::streamfunctions::reference;

// Shared SF-10 configuration for every isotropic acceptance fixture below,
// unless a case documents a different value.
constexpr double kEpsilon = 1.0e-2;
constexpr double kVRms = 1.0;
constexpr std::size_t kGridN = 16;

// SF-26 T02: exact-pair / general-pair / crossed-mutant / gauge-recombination
// contract cases use eta=1, v_rms=1, epsilon=0 throughout (a positive epsilon
// would add an O(eps^2) model error that swamps the O(h^2) discretization
// signal being measured).
constexpr double kPairPi = 3.14159265358979323846264338327950288;
constexpr double kExactPairEpsilon = 0.0;
constexpr double kExactPairVRms = 1.0;

constexpr double kOracleTolerance = 5.0e-11;
constexpr double kDirectEtaZeroTolerance = 1.0e-13;
constexpr double kDirectEtaOneTolerance = 1.0e-12;
constexpr double kReductionRelativeTolerance = 1.0e-12;
constexpr double kNormalizationRelativeTolerance = 1.0e-14;
constexpr double kEdgeSeparationGuard = 1.0e-9;
constexpr double kMeanZeroGaugeTolerance = 1.0e-12;

// ---------------------------------------------------------------------------
// Small local helpers.
// ---------------------------------------------------------------------------

[[nodiscard]] Grid3D production_grid(const ref::Grid& grid) {
    return {static_cast<int>(grid.nx), static_cast<int>(grid.ny), static_cast<int>(grid.nz),
            static_cast<real>(grid.spacing.x), static_cast<real>(grid.spacing.y),
            static_cast<real>(grid.spacing.z)};
}

[[nodiscard]] AffineGauge production_gauge(const ref::TotalGradientFixture& fixture) {
    return {{static_cast<real>(fixture.psi1_affine_gradient.x),
             static_cast<real>(fixture.psi1_affine_gradient.y),
             static_cast<real>(fixture.psi1_affine_gradient.z)},
            {static_cast<real>(fixture.psi2_affine_gradient.x),
             static_cast<real>(fixture.psi2_affine_gradient.y),
             static_cast<real>(fixture.psi2_affine_gradient.z)}};
}

[[nodiscard]] std::string grid_description(const ref::Grid& grid) {
    std::ostringstream out;
    out << grid.nx << 'x' << grid.ny << 'x' << grid.nz << " h=(" << grid.spacing.x << ','
        << grid.spacing.y << ',' << grid.spacing.z << ')';
    return out.str();
}

[[nodiscard]] NonlinearSourceConfig production_config(double epsilon, double v_rms) {
    NonlinearSourceConfig config{};
    config.epsilon = static_cast<real>(epsilon);
    config.v_rms = static_cast<real>(v_rms);
    return config;
}

[[nodiscard]] std::vector<double> to_double(const std::vector<real>& values) {
    return std::vector<double>(values.begin(), values.end());
}

[[nodiscard]] double rms_of(const std::vector<double>& values) {
    long double sum = 0.0L;
    for (double value : values) sum += static_cast<long double>(value) * value;
    return std::sqrt(static_cast<double>(sum / values.size()));
}

[[nodiscard]] double linf_diff(const std::vector<double>& actual, const std::vector<double>& expected) {
    double maximum = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    return maximum;
}

[[nodiscard]] double normalized_rms_diff(const std::vector<double>& actual, const std::vector<double>& expected) {
    long double err = 0.0L, exp2 = 0.0L;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const long double delta = static_cast<long double>(actual[i]) - expected[i];
        err += delta * delta;
        exp2 += static_cast<long double>(expected[i]) * expected[i];
    }
    const double scale = std::max(std::sqrt(static_cast<double>(exp2 / actual.size())), 1.0);
    return std::sqrt(static_cast<double>(err / actual.size())) / scale;
}

[[nodiscard]] double relative_error(double actual, double expected) {
    return std::abs(actual - expected) / std::max(std::abs(expected), 1.0);
}

// Isotropic SF-10 fixture required by the full evaluator chain: dx==dy==dz.
// `make_total_gradient_fixture` is deliberately rectangular and must not be
// used for any evaluator call (C01 fail-fast on anisotropic spacing).
[[nodiscard]] ref::TotalGradientFixture make_isotropic_fixture(std::size_t n) {
    const ref::Vec3 lengths{1.0, 1.0, 1.0};
    const double h = 1.0 / static_cast<double>(n);
    const ref::Grid grid{n, n, n, {h, h, h}};
    ref::TotalGradientFixture fixture{grid, lengths, {0.0, 1.3, 0.0}, {0.0, 0.0, 1.0}, {}, {}};
    fixture.psi1_fluctuation.resize(grid.cell_count());
    fixture.psi2_fluctuation.resize(grid.cell_count());
    for (std::size_t iz = 0; iz < n; ++iz) {
        for (std::size_t iy = 0; iy < n; ++iy) {
            for (std::size_t ix = 0; ix < n; ++ix) {
                const auto id = grid.index(ix, iy, iz);
                const auto position = grid.cell_center(ix, iy, iz);
                fixture.psi1_fluctuation[id] =
                    ref::total_gradient_periodic_scalar(ref::GradientFixtureField::psi1, position, lengths);
                fixture.psi2_fluctuation[id] =
                    ref::total_gradient_periodic_scalar(ref::GradientFixtureField::psi2, position, lengths);
            }
        }
    }
    return fixture;
}

[[nodiscard]] ref::VectorField cross_field(const ref::VectorField& g1, const ref::VectorField& g2) {
    ref::VectorField c;
    const std::size_t n = g1.x.size();
    c.x.resize(n); c.y.resize(n); c.z.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const ref::Vec3 cc = ref::cross({g1.x[i], g1.y[i], g1.z[i]}, {g2.x[i], g2.y[i], g2.z[i]});
        c.x[i] = cc.x; c.y[i] = cc.y; c.z[i] = cc.z;
    }
    return c;
}

[[nodiscard]] std::vector<double> magnitudes_of(const ref::VectorField& c) {
    std::vector<double> result(c.x.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = std::sqrt(c.x[i] * c.x[i] + c.y[i] * c.y[i] + c.z[i] * c.z[i]);
    }
    return result;
}

template <typename Callable>
[[nodiscard]] bool rejects_with_invalid_argument(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "coupled_residual_contract name=" << name
                  << " exception=none expected=std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument& error) {
        std::cout << "coupled_residual_contract name=" << name
                  << " exception=std::invalid_argument message=" << error.what() << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cout << "coupled_residual_contract name=" << name
                  << " exception=std::exception message=" << error.what()
                  << " expected=std::invalid_argument\n";
        return false;
    } catch (...) {
        std::cout << "coupled_residual_contract name=" << name
                  << " exception=non-standard expected=std::invalid_argument\n";
        return false;
    }
}

template <typename Callable>
[[nodiscard]] bool rejects_with_logic_error(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "coupled_residual_contract name=" << name
                  << " exception=none expected=std::logic_error\n";
        return false;
    } catch (const std::logic_error& error) {
        std::cout << "coupled_residual_contract name=" << name
                  << " exception=std::logic_error message=" << error.what() << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cout << "coupled_residual_contract name=" << name
                  << " exception=std::exception message=" << error.what()
                  << " expected=std::logic_error\n";
        return false;
    } catch (...) {
        std::cout << "coupled_residual_contract name=" << name
                  << " exception=non-standard expected=std::logic_error\n";
        return false;
    }
}

// ---------------------------------------------------------------------------
// GPU fixture: owns q/u1/u2 device state, the production ResidualEvaluator
// workspace, and separate device buffers for manually re-running the
// individual SF-02/06/07/08/09 modules that ResidualEvaluator composes.
// ---------------------------------------------------------------------------

class CoupledResidualGpuFixture {
  public:
    explicit CoupledResidualGpuFixture(const ref::Grid& grid)
        : grid_(production_grid(grid)), context_(0), n_(grid.cell_count()), q_(n_), u1_(n_), u2_(n_),
          f1_(n_), f2_(n_), rhs1_(n_), rhs2_(n_), p1x_(n_), p1y_(n_), p1z_(n_), p2x_(n_), p2y_(n_),
          p2z_(n_), h2g1x_(n_), h2g1y_(n_), h2g1z_(n_), h1g2x_(n_), h1g2y_(n_), h1g2z_(n_), bx_(n_),
          by_(n_), bz_(n_), s1_(n_), s2_(n_), counters_(2 + kMaxDegeneracyThresholds), a_u1_(n_),
          a_u2_(n_) {
        workspace_.prepare(n_);
        manual_rhs_workspace_.prepare(n_);
    }

    void upload(const std::vector<double>& q, const std::vector<double>& u1, const std::vector<double>& u2) {
        const std::vector<real> hq(q.begin(), q.end());
        const std::vector<real> hu1(u1.begin(), u1.end());
        const std::vector<real> hu2(u2.begin(), u2.end());
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(q_.data(), hq.data(), n_ * sizeof(real), cudaMemcpyHostToDevice,
                                               context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u1_.data(), hu1.data(), n_ * sizeof(real), cudaMemcpyHostToDevice,
                                               context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u2_.data(), hu2.data(), n_ * sizeof(real), cudaMemcpyHostToDevice,
                                               context_.cuda_stream()));
    }

    struct ResidualRunResult {
        std::vector<real> f1, f2;
        StreamfunctionResidualReport report;
    };

    [[nodiscard]] ResidualRunResult run_evaluator(const AffineGauge& gauge, real eta,
                                                  const NonlinearSourceConfig& source_config,
                                                  const ResidualHistogramConfig& histogram_config) {
        enqueue_streamfunction_residual(context_, grid_, q_.span(), {u1_.span(), u2_.span()}, gauge, eta,
                                        source_config, histogram_config, f1_.span(), f2_.span(), workspace_);
        ResidualRunResult result;
        result.f1.resize(n_);
        result.f2.resize(n_);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.f1.data(), f1_.data(), n_ * sizeof(real),
                                               cudaMemcpyDeviceToHost, context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.f2.data(), f2_.data(), n_ * sizeof(real),
                                               cudaMemcpyDeviceToHost, context_.cuda_stream()));
        result.report = synchronize_streamfunction_residual_report(context_, grid_, eta, source_config,
                                                                    histogram_config, workspace_);
        return result;
    }

    struct ManualGradients {
        ref::VectorField g1, g2;
    };

    [[nodiscard]] ManualGradients run_gradients(const AffineGauge& gauge) {
        enqueue_total_streamfunction_gradients(
            context_, grid_, {u1_.span(), u2_.span()}, gauge,
            {p1x_.span(), p1y_.span(), p1z_.span(), p2x_.span(), p2y_.span(), p2z_.span()});
        std::vector<real> hp1x(n_), hp1y(n_), hp1z(n_), hp2x(n_), hp2y(n_), hp2z(n_);
        const std::array<std::pair<real*, const DeviceBuffer<real>*>, 6> transfers{
            {{hp1x.data(), &p1x_}, {hp1y.data(), &p1y_}, {hp1z.data(), &p1z_}, {hp2x.data(), &p2x_},
             {hp2y.data(), &p2y_}, {hp2z.data(), &p2z_}}};
        for (const auto& [host_ptr, buffer] : transfers) {
            MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(host_ptr, buffer->data(), n_ * sizeof(real),
                                                   cudaMemcpyDeviceToHost, context_.cuda_stream()));
        }
        context_.synchronize();
        ManualGradients result;
        result.g1 = {to_double(hp1x), to_double(hp1y), to_double(hp1z)};
        result.g2 = {to_double(hp2x), to_double(hp2y), to_double(hp2z)};
        return result;
    }

    void run_hessian_b() {
        enqueue_streamfunction_hessian_vector_b(
            context_, grid_, {u1_.span(), u2_.span()},
            {p1x_.span(), p1y_.span(), p1z_.span(), p2x_.span(), p2y_.span(), p2z_.span()},
            {h2g1x_.span(), h2g1y_.span(), h2g1z_.span(), h1g2x_.span(), h1g2y_.span(), h1g2z_.span(),
             bx_.span(), by_.span(), bz_.span()});
    }

    struct ManualSources {
        std::vector<real> s1, s2;
        std::vector<unsigned long long> counters;
    };

    [[nodiscard]] ManualSources run_sources(const NonlinearSourceConfig& config) {
        const auto num_counters = static_cast<std::size_t>(2 + config.num_degeneracy_thresholds);
        enqueue_streamfunction_nonlinear_sources(
            context_, grid_,
            {p1x_.span(), p1y_.span(), p1z_.span(), p2x_.span(), p2y_.span(), p2z_.span()},
            {bx_.span(), by_.span(), bz_.span()}, config, {s1_.span(), s2_.span()},
            {DeviceSpan<unsigned long long>(counters_.data(), num_counters)});
        ManualSources result;
        result.s1.resize(n_);
        result.s2.resize(n_);
        result.counters.resize(num_counters);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.s1.data(), s1_.data(), n_ * sizeof(real),
                                               cudaMemcpyDeviceToHost, context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.s2.data(), s2_.data(), n_ * sizeof(real),
                                               cudaMemcpyDeviceToHost, context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.counters.data(), counters_.data(),
                                               num_counters * sizeof(unsigned long long),
                                               cudaMemcpyDeviceToHost, context_.cuda_stream()));
        context_.synchronize();
        return result;
    }

    struct ManualApply {
        std::vector<real> a_u1, a_u2;
    };

    [[nodiscard]] ManualApply run_apply() {
        const operators::LesterPositiveDiffusionOperator op(grid_, q_.span());
        op.apply(context_, DeviceSpan<const real>(u1_.span()), a_u1_.span());
        op.apply(context_, DeviceSpan<const real>(u2_.span()), a_u2_.span());
        ManualApply result;
        result.a_u1.resize(n_);
        result.a_u2.resize(n_);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.a_u1.data(), a_u1_.data(), n_ * sizeof(real),
                                               cudaMemcpyDeviceToHost, context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.a_u2.data(), a_u2_.data(), n_ * sizeof(real),
                                               cudaMemcpyDeviceToHost, context_.cuda_stream()));
        context_.synchronize();
        return result;
    }

    struct ManualRhs {
        std::vector<real> rhs1, rhs2;
        AffineRhsHostDiagnostics diagnostics;
    };

    [[nodiscard]] ManualRhs run_affine_rhs(const AffineGauge& gauge) {
        const auto device_diagnostics = assemble_affine_periodic_rhs(context_, grid_, q_.span(), gauge,
                                                                      rhs1_.span(), rhs2_.span(),
                                                                      manual_rhs_workspace_);
        ManualRhs result;
        result.rhs1.resize(n_);
        result.rhs2.resize(n_);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.rhs1.data(), rhs1_.data(), n_ * sizeof(real),
                                               cudaMemcpyDeviceToHost, context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.rhs2.data(), rhs2_.data(), n_ * sizeof(real),
                                               cudaMemcpyDeviceToHost, context_.cuda_stream()));
        result.diagnostics = synchronize_affine_rhs_diagnostics(context_, device_diagnostics);
        return result;
    }

    [[nodiscard]] std::size_t n() const { return n_; }

  private:
    Grid3D grid_;
    CudaContext context_;
    std::size_t n_;
    DeviceBuffer<real> q_, u1_, u2_, f1_, f2_;
    DeviceBuffer<real> rhs1_, rhs2_;
    DeviceBuffer<real> p1x_, p1y_, p1z_, p2x_, p2y_, p2z_;
    DeviceBuffer<real> h2g1x_, h2g1y_, h2g1z_, h1g2x_, h1g2y_, h1g2z_;
    DeviceBuffer<real> bx_, by_, bz_;
    DeviceBuffer<real> s1_, s2_;
    DeviceBuffer<unsigned long long> counters_;
    DeviceBuffer<real> a_u1_, a_u2_;
    StreamfunctionResidualWorkspace workspace_;
    AffinePeriodicRhsWorkspace manual_rhs_workspace_;
};

// ---------------------------------------------------------------------------
// Case 1: production ResidualEvaluator vs coupled_residual_reference.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_coupled_residual_gpu_oracle() {
    const auto fixture = make_isotropic_fixture(kGridN);
    const auto q = ref::make_positive_q_field(fixture.grid, fixture.lengths);
    CoupledResidualGpuFixture gpu(fixture.grid);
    gpu.upload(q, fixture.psi1_fluctuation, fixture.psi2_fluctuation);
    const auto gauge = production_gauge(fixture);
    const auto source_config = production_config(kEpsilon, kVRms);
    const ResidualHistogramConfig histogram_config{};

    const auto run = gpu.run_evaluator(gauge, real{1}, source_config, histogram_config);
    const auto actual_f1 = to_double(run.f1);
    const auto actual_f2 = to_double(run.f2);

    const ref::NonlinearSourceReferenceConfig ref_config{kEpsilon, kVRms};
    const auto expected = ref::coupled_residual_reference(q, fixture, 1.0, ref_config);

    const double nrms_f1 = normalized_rms_diff(actual_f1, expected.f1);
    const double nrms_f2 = normalized_rms_diff(actual_f2, expected.f2);
    const double scale_f1 = std::max(rms_of(expected.f1), 1.0);
    const double scale_f2 = std::max(rms_of(expected.f2), 1.0);
    const double plinf_f1 = linf_diff(actual_f1, expected.f1) / scale_f1;
    const double plinf_f2 = linf_diff(actual_f2, expected.f2) / scale_f2;

    // report.raw_rhs_mean_psi{1,2} are the SF-06 assemble_affine_periodic_rhs
    // diagnostics, captured before the eta*q.*S_i (SAME-index) combination
    // step (see
    // ResidualEvaluator.cuh's documented "diagnostics ... surfaced unchanged"
    // contract) -- i.e. the mean of the *affine-only* RHS, not of
    // CoupledResidualFields::raw_rhs1_mean (which is the mean of the fully
    // combined, S-inclusive raw RHS). Compare against the matching
    // affine-only CPU oracle instead.
    const double expected_affine_mean1 =
        static_cast<double>(ref::long_double_mean(ref::affine_rhs_discrete(fixture.grid, q, fixture.psi1_affine_gradient)));
    const double expected_affine_mean2 =
        static_cast<double>(ref::long_double_mean(ref::affine_rhs_discrete(fixture.grid, q, fixture.psi2_affine_gradient)));
    const double raw_mean1_diff = std::abs(static_cast<double>(run.report.raw_rhs_mean_psi1) - expected_affine_mean1);
    const double raw_mean2_diff = std::abs(static_cast<double>(run.report.raw_rhs_mean_psi2) - expected_affine_mean2);
    const double raw_mean1_tol = std::max(1.0e-12, 1.0e-12 * std::abs(expected_affine_mean1));
    const double raw_mean2_tol = std::max(1.0e-12, 1.0e-12 * std::abs(expected_affine_mean2));

    std::cout << std::setprecision(16) << "coupled_residual_gpu_oracle nrms_f1=" << nrms_f1
              << " nrms_f2=" << nrms_f2 << " plinf_f1=" << plinf_f1 << " plinf_f2=" << plinf_f2
              << " raw_mean1_diff=" << raw_mean1_diff << " raw_mean1_tol=" << raw_mean1_tol
              << " raw_mean2_diff=" << raw_mean2_diff << " raw_mean2_tol=" << raw_mean2_tol << '\n';

    const bool finite =
        std::isfinite(nrms_f1) && std::isfinite(nrms_f2) && std::isfinite(plinf_f1) && std::isfinite(plinf_f2);
    const bool pass = finite && nrms_f1 <= kOracleTolerance && nrms_f2 <= kOracleTolerance &&
                      plinf_f1 <= kOracleTolerance && plinf_f2 <= kOracleTolerance &&
                      raw_mean1_diff <= raw_mean1_tol && raw_mean2_diff <= raw_mean2_tol;
    return {pass, "coupled_residual_gpu_oracle", "gpu-vs-independent-cpu-oracle", grid_description(fixture.grid),
            std::max(nrms_f1, nrms_f2), std::max(plinf_f1, plinf_f2), "<=5e-11", "n/a",
            "normalized RMS and plain Linf/RMS(expected) <=5e-11 for F1,F2; raw RHS means match within 1e-12"};
}

// ---------------------------------------------------------------------------
// Case 2: manual same-module composition vs the production evaluator.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_coupled_residual_direct_agreement() {
    const auto fixture = make_isotropic_fixture(kGridN);
    const auto q = ref::make_positive_q_field(fixture.grid, fixture.lengths);
    CoupledResidualGpuFixture gpu(fixture.grid);
    gpu.upload(q, fixture.psi1_fluctuation, fixture.psi2_fluctuation);
    const auto gauge = production_gauge(fixture);
    const auto source_config = production_config(kEpsilon, kVRms);
    const ResidualHistogramConfig histogram_config{};

    // eta = 0: F_manual = A u - rhs (rhs already mean-zero projected by
    // assemble_affine_periodic_rhs); no source term contributes.
    const auto eval_eta0 = gpu.run_evaluator(gauge, real{0}, source_config, histogram_config);
    const auto manual_rhs_eta0 = gpu.run_affine_rhs(gauge);
    const auto manual_apply = gpu.run_apply();

    const auto n = gpu.n();
    std::vector<double> manual_f1_eta0(n), manual_f2_eta0(n);
    for (std::size_t i = 0; i < n; ++i) {
        manual_f1_eta0[i] = static_cast<double>(manual_apply.a_u1[i]) - static_cast<double>(manual_rhs_eta0.rhs1[i]);
        manual_f2_eta0[i] = static_cast<double>(manual_apply.a_u2[i]) - static_cast<double>(manual_rhs_eta0.rhs2[i]);
    }
    const auto eval_eta0_f1 = to_double(eval_eta0.f1);
    const auto eval_eta0_f2 = to_double(eval_eta0.f2);
    const double eta0_scale = std::max({rms_of(eval_eta0_f1), rms_of(eval_eta0_f2), 1.0});
    const double eta0_norm_linf =
        std::max(linf_diff(manual_f1_eta0, eval_eta0_f1), linf_diff(manual_f2_eta0, eval_eta0_f2)) / eta0_scale;

    // eta = 1: independently rerun SF-07/08/09 and A.apply, then combine on
    // the host with a long-double mean-zero projection of G_i = rhs_i -
    // eta*q.*S_i (SAME index, SF-26).
    const auto grads = gpu.run_gradients(gauge);
    gpu.run_hessian_b();
    const auto sources = gpu.run_sources(source_config);
    const auto manual_rhs_eta1 = gpu.run_affine_rhs(gauge);
    const auto eval_eta1 = gpu.run_evaluator(gauge, real{1}, source_config, histogram_config);

    std::vector<double> g1_raw(n), g2_raw(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double qc = q[i];
        g1_raw[i] = static_cast<double>(manual_rhs_eta1.rhs1[i]) - qc * static_cast<double>(sources.s1[i]);
        g2_raw[i] = static_cast<double>(manual_rhs_eta1.rhs2[i]) - qc * static_cast<double>(sources.s2[i]);
    }
    const auto g1_proj = ref::mean_zero_projected(g1_raw);
    const auto g2_proj = ref::mean_zero_projected(g2_raw);
    std::vector<double> manual_f1_eta1(n), manual_f2_eta1(n);
    for (std::size_t i = 0; i < n; ++i) {
        manual_f1_eta1[i] = static_cast<double>(manual_apply.a_u1[i]) - g1_proj[i];
        manual_f2_eta1[i] = static_cast<double>(manual_apply.a_u2[i]) - g2_proj[i];
    }
    const auto eval_eta1_f1 = to_double(eval_eta1.f1);
    const auto eval_eta1_f2 = to_double(eval_eta1.f2);
    const double eta1_scale = std::max({rms_of(eval_eta1_f1), rms_of(eval_eta1_f2), 1.0});
    const double eta1_norm_linf =
        std::max(linf_diff(manual_f1_eta1, eval_eta1_f1), linf_diff(manual_f2_eta1, eval_eta1_f2)) / eta1_scale;

    std::cout << std::setprecision(16) << "coupled_residual_direct_agreement eta0_norm_linf=" << eta0_norm_linf
              << " threshold0=" << kDirectEtaZeroTolerance << " eta1_norm_linf=" << eta1_norm_linf
              << " threshold1=" << kDirectEtaOneTolerance << '\n';

    const bool pass = std::isfinite(eta0_norm_linf) && std::isfinite(eta1_norm_linf) &&
                      eta0_norm_linf <= kDirectEtaZeroTolerance && eta1_norm_linf <= kDirectEtaOneTolerance;
    return {pass, "coupled_residual_direct_agreement", "gpu-manual-modules-vs-gpu-evaluator",
            grid_description(fixture.grid), eta0_norm_linf, eta1_norm_linf, "<=1e-13 / <=1e-12", "n/a",
            "direct A*u-b (eta=0) and full G composition (eta=1) agree with the evaluator to reduction roundoff"};
}

// ---------------------------------------------------------------------------
// Case 3: reported reductions and normalization vs CPU reductions.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_coupled_residual_reductions_agreement() {
    const auto fixture = make_isotropic_fixture(kGridN);
    const auto q = ref::make_positive_q_field(fixture.grid, fixture.lengths);
    CoupledResidualGpuFixture gpu(fixture.grid);
    gpu.upload(q, fixture.psi1_fluctuation, fixture.psi2_fluctuation);
    const auto gauge = production_gauge(fixture);
    const auto source_config = production_config(kEpsilon, kVRms);
    const ResidualHistogramConfig histogram_config{};
    const auto run = gpu.run_evaluator(gauge, real{1}, source_config, histogram_config);

    const auto f1 = to_double(run.f1);
    const auto f2 = to_double(run.f2);
    const double cpu_rms_f1 = ref::rms_norm(f1);
    const double cpu_rms_f2 = ref::rms_norm(f2);
    const double cpu_linf_f1 = ref::linf_norm(f1);
    const double cpu_linf_f2 = ref::linf_norm(f2);
    const double cpu_q_rms = ref::rms_norm(q);

    const double e_rms_f1 = relative_error(static_cast<double>(run.report.rms_f1), cpu_rms_f1);
    const double e_rms_f2 = relative_error(static_cast<double>(run.report.rms_f2), cpu_rms_f2);
    const double e_linf_f1 = relative_error(static_cast<double>(run.report.linf_f1), cpu_linf_f1);
    const double e_linf_f2 = relative_error(static_cast<double>(run.report.linf_f2), cpu_linf_f2);
    const double e_q_rms = relative_error(static_cast<double>(run.report.q_rms), cpu_q_rms);

    const auto norm_ref = ref::residual_normalization_reference(
        static_cast<double>(run.report.rms_f1), static_cast<double>(run.report.rms_f2),
        static_cast<double>(run.report.q_rms), kVRms, static_cast<double>(run.report.L_ref));
    const double e_r1 = relative_error(static_cast<double>(run.report.r1), norm_ref.r1);
    const double e_r2 = relative_error(static_cast<double>(run.report.r2), norm_ref.r2);
    const double e_rF = relative_error(static_cast<double>(run.report.r_F), norm_ref.r_f);

    const double l_ref = static_cast<double>(run.report.L_ref);

    std::cout << std::setprecision(16) << "coupled_residual_reductions_agreement rms_f1_gpu=" << run.report.rms_f1
              << " rms_f1_cpu=" << cpu_rms_f1 << " rms_f2_gpu=" << run.report.rms_f2
              << " rms_f2_cpu=" << cpu_rms_f2 << " linf_f1_gpu=" << run.report.linf_f1
              << " linf_f1_cpu=" << cpu_linf_f1 << " linf_f2_gpu=" << run.report.linf_f2
              << " linf_f2_cpu=" << cpu_linf_f2 << " q_rms_gpu=" << run.report.q_rms
              << " q_rms_cpu=" << cpu_q_rms << " r1=" << run.report.r1 << " r1_ref=" << norm_ref.r1
              << " r2=" << run.report.r2 << " r2_ref=" << norm_ref.r2 << " r_F=" << run.report.r_F
              << " r_F_ref=" << norm_ref.r_f << " L_ref=" << l_ref << '\n';

    const bool pass = e_rms_f1 <= kReductionRelativeTolerance && e_rms_f2 <= kReductionRelativeTolerance &&
                      e_linf_f1 <= kReductionRelativeTolerance && e_linf_f2 <= kReductionRelativeTolerance &&
                      e_q_rms <= kReductionRelativeTolerance && e_r1 <= kNormalizationRelativeTolerance &&
                      e_r2 <= kNormalizationRelativeTolerance && e_rF <= kNormalizationRelativeTolerance &&
                      l_ref == 1.0;
    return {pass, "coupled_residual_reductions_agreement", "gpu-report-vs-cpu-reduction",
            grid_description(fixture.grid), std::max({e_rms_f1, e_rms_f2, e_linf_f1, e_linf_f2, e_q_rms}),
            std::max({e_r1, e_r2, e_rF}), "<=1e-12 / <=1e-14", "n/a",
            "rms_f1,rms_f2,linf_f1,linf_f2,q_rms within 1e-12 relative; r1,r2,r_F within 1e-14 relative; "
            "L_ref==1 exactly for the unit cube"};
}

// ---------------------------------------------------------------------------
// Case 4: |c| histogram bin-for-bin agreement.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_coupled_residual_histogram_agreement() {
    const auto fixture = make_isotropic_fixture(kGridN);
    const auto q = ref::make_positive_q_field(fixture.grid, fixture.lengths);
    CoupledResidualGpuFixture gpu(fixture.grid);
    gpu.upload(q, fixture.psi1_fluctuation, fixture.psi2_fluctuation);
    const auto gauge = production_gauge(fixture);
    const auto source_config = production_config(kEpsilon, kVRms);

    const auto grads = gpu.run_gradients(gauge);
    const auto c = cross_field(grads.g1, grads.g2);
    auto magnitudes = magnitudes_of(c);
    std::vector<double> sorted = magnitudes;
    std::sort(sorted.begin(), sorted.end());
    std::cout << std::setprecision(16) << "coupled_residual_histogram_agreement c_min_observed=" << sorted.front()
              << " c_max_observed=" << sorted.back() << '\n';

    // Midpoint between the two straddling sorted samples, so the chosen bound
    // never coincides exactly with an observed |c| value (which would put a
    // histogram bin edge exactly on a data point and collapse
    // min_edge_separation to 0).
    const auto pct_midpoint = [&](double p) {
        const auto count = sorted.size();
        const auto idx = static_cast<std::size_t>(
            std::clamp(p * static_cast<double>(count - 1), 0.0, static_cast<double>(count - 2)));
        return 0.5 * (sorted[idx] + sorted[idx + 1]);
    };
    const double split_min = pct_midpoint(0.15);
    const double split_max = pct_midpoint(0.85);
    std::cout << std::setprecision(16) << "coupled_residual_histogram_agreement split_c_min=" << split_min
              << " split_c_max=" << split_max << '\n';

    const ResidualHistogramConfig wide{}; // defaults: c_min_rel=1e-10, c_max_rel=1e4, v_rms=1
    const ResidualHistogramConfig split{static_cast<real>(split_min), static_cast<real>(split_max)};

    const auto run_wide = gpu.run_evaluator(gauge, real{1}, source_config, wide);
    const auto run_split = gpu.run_evaluator(gauge, real{1}, source_config, split);

    const auto cpu_wide = ref::log_histogram_reference(
        c, static_cast<double>(run_wide.report.histogram_c_min), static_cast<double>(run_wide.report.histogram_c_max));
    const auto cpu_split =
        ref::log_histogram_reference(c, static_cast<double>(run_split.report.histogram_c_min),
                                     static_cast<double>(run_split.report.histogram_c_max));

    const auto exact_match = [](const StreamfunctionResidualReport& report, const ref::LogHistogramReference& cpu) {
        if (report.histogram_underflow != cpu.underflow) return false;
        if (report.histogram_overflow != cpu.overflow) return false;
        for (int i = 0; i < kResidualHistogramBins; ++i) {
            if (report.histogram_counts[i] != cpu.counts[static_cast<std::size_t>(i)]) return false;
        }
        return true;
    };

    const bool wide_match = exact_match(run_wide.report, cpu_wide);
    const bool split_match = exact_match(run_split.report, cpu_split);

    std::cout << std::setprecision(16) << "coupled_residual_histogram_agreement wide underflow_gpu="
              << run_wide.report.histogram_underflow << " underflow_cpu=" << cpu_wide.underflow
              << " overflow_gpu=" << run_wide.report.histogram_overflow << " overflow_cpu=" << cpu_wide.overflow
              << " exact_match=" << (wide_match ? "true" : "false")
              << " min_edge_separation=" << cpu_wide.min_edge_separation << '\n';
    std::cout << std::setprecision(16) << "coupled_residual_histogram_agreement split underflow_gpu="
              << run_split.report.histogram_underflow << " underflow_cpu=" << cpu_split.underflow
              << " overflow_gpu=" << run_split.report.histogram_overflow << " overflow_cpu=" << cpu_split.overflow
              << " exact_match=" << (split_match ? "true" : "false")
              << " min_edge_separation=" << cpu_split.min_edge_separation << '\n';

    const bool pass = run_wide.report.histogram_underflow == 0 && run_wide.report.histogram_overflow == 0 &&
                      wide_match && cpu_wide.min_edge_separation > kEdgeSeparationGuard &&
                      run_split.report.histogram_underflow > 0 && run_split.report.histogram_overflow > 0 &&
                      split_match && cpu_split.min_edge_separation > kEdgeSeparationGuard;
    return {pass, "coupled_residual_histogram_agreement", "gpu-vs-cpu-exact-histogram", grid_description(fixture.grid),
            static_cast<double>(cpu_wide.min_edge_separation), static_cast<double>(cpu_split.min_edge_separation),
            "exact", "n/a",
            "wide range (underflow=overflow=0) and split range (both nonzero) match GPU bin-for-bin; "
            "min_edge_separation>1e-9 for both"};
}

// ---------------------------------------------------------------------------
// Case 5: percentile helper bound vs exact sorted percentile.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_coupled_residual_percentile_bound() {
    const auto fixture = make_isotropic_fixture(kGridN);
    const auto q = ref::make_positive_q_field(fixture.grid, fixture.lengths);
    CoupledResidualGpuFixture gpu(fixture.grid);
    gpu.upload(q, fixture.psi1_fluctuation, fixture.psi2_fluctuation);
    const auto gauge = production_gauge(fixture);
    const auto source_config = production_config(kEpsilon, kVRms);

    const auto grads = gpu.run_gradients(gauge);
    const auto c = cross_field(grads.g1, grads.g2);
    const auto magnitudes = magnitudes_of(c);

    const ResidualHistogramConfig wide{};
    const auto run = gpu.run_evaluator(gauge, real{1}, source_config, wide);

    const double c_min = static_cast<double>(run.report.histogram_c_min);
    const double c_max = static_cast<double>(run.report.histogram_c_max);
    const double bin_factor = std::pow(10.0, (std::log10(c_max) - std::log10(c_min)) / kResidualHistogramBins);
    const double bound = bin_factor * bin_factor;

    bool pass = true;
    double worst_deviation = 0.0;
    for (double p : {0.001, 0.01, 0.05, 0.5, 0.95}) {
        const double gpu_value = static_cast<double>(residual_histogram_percentile(run.report, static_cast<real>(p)));
        const double cpu_value = ref::exact_sorted_percentile(magnitudes, p);
        const double ratio = gpu_value / cpu_value;
        const bool ok = std::isfinite(ratio) && ratio <= bound && ratio >= 1.0 / bound;
        pass = pass && ok;
        const double deviation = std::max(ratio, 1.0 / ratio);
        worst_deviation = std::max(worst_deviation, deviation);
        std::cout << std::setprecision(16) << "coupled_residual_percentile_bound p=" << p << " gpu=" << gpu_value
                  << " cpu_exact=" << cpu_value << " ratio=" << ratio << " bin_factor=" << bin_factor
                  << " bound=" << bound << " pass=" << (ok ? "true" : "false") << '\n';
    }
    return {pass, "coupled_residual_percentile_bound", "gpu-histogram-percentile-vs-exact-sorted",
            grid_description(fixture.grid), bin_factor, bound, "ratio in [1/bound,bound]",
            std::to_string(worst_deviation),
            "GPU log-histogram percentile approximates exact sorted percentile within bin_factor^2"};
}

// ---------------------------------------------------------------------------
// Case 6: exact-zero identity for constant q and zero fluctuations.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_coupled_residual_homogeneous_zero() {
    const double h = 1.0 / static_cast<double>(kGridN);
    const ref::Grid grid{kGridN, kGridN, kGridN, {h, h, h}};
    const std::vector<double> q(grid.cell_count(), 1.0);
    const std::vector<double> zero_u1(grid.cell_count(), 0.0);
    const std::vector<double> zero_u2(grid.cell_count(), 0.0);
    const AffineGauge gauge = AffineGauge::benchmark(real{1.3});
    CoupledResidualGpuFixture gpu(grid);
    gpu.upload(q, zero_u1, zero_u2);
    const auto source_config = production_config(kEpsilon, kVRms);
    const ResidualHistogramConfig histogram_config{};

    bool pass = true;
    for (real eta : {real{0}, real{1}}) {
        const auto run = gpu.run_evaluator(gauge, eta, source_config, histogram_config);
        double max_abs = 0.0;
        for (real value : run.f1) max_abs = std::max(max_abs, std::abs(static_cast<double>(value)));
        for (real value : run.f2) max_abs = std::max(max_abs, std::abs(static_cast<double>(value)));
        const bool this_pass =
            max_abs == 0.0 && run.report.rms_f1 == real{0} && run.report.rms_f2 == real{0} &&
            run.report.linf_f1 == real{0} && run.report.linf_f2 == real{0} && run.report.r_F == real{0} &&
            run.report.raw_rhs_mean_psi1 == real{0} && run.report.raw_rhs_mean_psi2 == real{0} &&
            run.report.projected_rhs_mean_psi1 == real{0} && run.report.projected_rhs_mean_psi2 == real{0} &&
            run.report.nonfinite_s1 == 0 && run.report.nonfinite_s2 == 0;
        std::cout << std::setprecision(16) << "coupled_residual_homogeneous_zero eta=" << eta
                  << " max_abs_f=" << max_abs << " rms_f1=" << run.report.rms_f1 << " rms_f2=" << run.report.rms_f2
                  << " linf_f1=" << run.report.linf_f1 << " linf_f2=" << run.report.linf_f2
                  << " r_F=" << run.report.r_F << " raw_mean1=" << run.report.raw_rhs_mean_psi1
                  << " raw_mean2=" << run.report.raw_rhs_mean_psi2
                  << " proj_mean1=" << run.report.projected_rhs_mean_psi1
                  << " proj_mean2=" << run.report.projected_rhs_mean_psi2
                  << " nonfinite_s1=" << run.report.nonfinite_s1 << " nonfinite_s2=" << run.report.nonfinite_s2
                  << " pass=" << (this_pass ? "true" : "false") << '\n';
        pass = pass && this_pass;
    }
    return {pass, "coupled_residual_homogeneous_zero", "gpu-exact-zero-by-construction", grid_description(grid), 0.0,
            0.0, "exact-zero", "n/a",
            "constant q=1, zero fluctuations: F1,F2 exactly zero at eta=0 and eta=1, all report fields exactly zero"};
}

// ---------------------------------------------------------------------------
// Case 7: mean-zero gauge check on F and the projected-RHS diagnostics.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_coupled_residual_mean_zero_gauge() {
    const auto fixture = make_isotropic_fixture(kGridN);
    const auto q = ref::make_positive_q_field(fixture.grid, fixture.lengths);
    CoupledResidualGpuFixture gpu(fixture.grid);
    gpu.upload(q, fixture.psi1_fluctuation, fixture.psi2_fluctuation);
    const auto gauge = production_gauge(fixture);
    const auto source_config = production_config(kEpsilon, kVRms);
    const ResidualHistogramConfig histogram_config{};
    const auto run = gpu.run_evaluator(gauge, real{1}, source_config, histogram_config);

    const auto f1 = to_double(run.f1);
    const auto f2 = to_double(run.f2);
    const long double mean_f1 = ref::long_double_mean(f1);
    const long double mean_f2 = ref::long_double_mean(f2);
    const double rms_f1 = ref::rms_norm(f1);
    const double rms_f2 = ref::rms_norm(f2);
    const double normalized_mean_f1 = static_cast<double>(std::abs(mean_f1)) / std::max(rms_f1, 1.0);
    const double normalized_mean_f2 = static_cast<double>(std::abs(mean_f2)) / std::max(rms_f2, 1.0);
    const double proj_mean1 = std::abs(static_cast<double>(run.report.projected_rhs_mean_psi1));
    const double proj_mean2 = std::abs(static_cast<double>(run.report.projected_rhs_mean_psi2));

    std::cout << std::setprecision(16) << "coupled_residual_mean_zero_gauge normalized_mean_f1=" << normalized_mean_f1
              << " normalized_mean_f2=" << normalized_mean_f2 << " proj_mean1=" << proj_mean1
              << " proj_mean2=" << proj_mean2 << '\n';

    const bool pass = normalized_mean_f1 <= kMeanZeroGaugeTolerance && normalized_mean_f2 <= kMeanZeroGaugeTolerance &&
                      proj_mean1 <= kMeanZeroGaugeTolerance && proj_mean2 <= kMeanZeroGaugeTolerance;
    return {pass, "coupled_residual_mean_zero_gauge", "gpu-mean-zero-by-construction", grid_description(fixture.grid),
            std::max(normalized_mean_f1, normalized_mean_f2), std::max(proj_mean1, proj_mean2), "<=1e-12", "n/a",
            "|mean(F_i)|/max(RMS(F_i),1) and |projected_rhs_mean_i| both <=1e-12"};
}

// ---------------------------------------------------------------------------
// Case 8: error contract for enqueue_streamfunction_residual and
// synchronize_streamfunction_residual_report.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_coupled_residual_error_contract() {
    const auto fixture = make_isotropic_fixture(kGridN);
    const auto q_host = ref::make_positive_q_field(fixture.grid, fixture.lengths);
    const auto grid = production_grid(fixture.grid);
    const auto n = fixture.grid.cell_count();
    CudaContext context(0);
    DeviceBuffer<real> q(n), u1(n), u2(n), f1(n), f2(n);

    const std::vector<real> host_q(q_host.begin(), q_host.end());
    const std::vector<real> host_u1(fixture.psi1_fluctuation.begin(), fixture.psi1_fluctuation.end());
    const std::vector<real> host_u2(fixture.psi2_fluctuation.begin(), fixture.psi2_fluctuation.end());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(q.data(), host_q.data(), n * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u1.data(), host_u1.data(), n * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u2.data(), host_u2.data(), n * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));

    const auto gauge = production_gauge(fixture);
    const NonlinearSourceConfig source_config = production_config(kEpsilon, kVRms);
    const ResidualHistogramConfig histogram_config{};
    const PeriodicStreamfunctionFluctuations fluctuations{u1.span(), u2.span()};

    StreamfunctionResidualWorkspace workspace;
    workspace.prepare(n);

    const auto invoke = [&](const Grid3D& candidate_grid, DeviceSpan<const real> candidate_q,
                            const PeriodicStreamfunctionFluctuations& candidate_fluctuations,
                            const AffineGauge& candidate_gauge, real candidate_eta,
                            const NonlinearSourceConfig& candidate_source_config,
                            const ResidualHistogramConfig& candidate_histogram_config, DeviceSpan<real> candidate_f1,
                            DeviceSpan<real> candidate_f2, StreamfunctionResidualWorkspace& candidate_workspace) {
        enqueue_streamfunction_residual(context, candidate_grid, candidate_q, candidate_fluctuations,
                                        candidate_gauge, candidate_eta, candidate_source_config,
                                        candidate_histogram_config, candidate_f1, candidate_f2, candidate_workspace);
    };

    bool pass = true;
    std::size_t checks = 0;
    const auto require_invalid = [&](const char* name, const auto& callable) {
        ++checks;
        pass = rejects_with_invalid_argument(name, callable) && pass;
    };
    const auto require_logic = [&](const char* name, const auto& callable) {
        ++checks;
        pass = rejects_with_logic_error(name, callable) && pass;
    };

    // --- grid checks ---
    require_invalid("extent_nx_one", [&] {
        invoke({1, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz}, q.span(), fluctuations, gauge, real{1},
              source_config, histogram_config, f1.span(), f2.span(), workspace);
    });
    require_invalid("anisotropic_dy", [&] {
        auto invalid = grid;
        invalid.dy = grid.dx * real{2};
        invoke(invalid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config, f1.span(),
              f2.span(), workspace);
    });
    require_invalid("anisotropic_dz", [&] {
        auto invalid = grid;
        invalid.dz = grid.dx * real{2};
        invoke(invalid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config, f1.span(),
              f2.span(), workspace);
    });
    for (const auto& [value, name] : std::array<std::pair<real, const char*>, 4>{
             {{real{0}, "dx_zero"},
              {real{-1}, "dx_negative"},
              {std::numeric_limits<real>::quiet_NaN(), "dx_nan"},
              {std::numeric_limits<real>::infinity(), "dx_inf"}}}) {
        auto invalid = grid;
        invalid.dx = value;
        invalid.dy = value;
        invalid.dz = value;
        require_invalid(name, [&, invalid] {
            invoke(invalid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config, f1.span(),
                  f2.span(), workspace);
        });
    }

    // --- eta checks ---
    require_invalid("eta_negative", [&] {
        invoke(grid, q.span(), fluctuations, gauge, real{-1}, source_config, histogram_config, f1.span(), f2.span(),
              workspace);
    });
    require_invalid("eta_nan", [&] {
        invoke(grid, q.span(), fluctuations, gauge, std::numeric_limits<real>::quiet_NaN(), source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });
    require_invalid("eta_inf", [&] {
        invoke(grid, q.span(), fluctuations, gauge, std::numeric_limits<real>::infinity(), source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });

    // --- histogram config checks ---
    for (const auto& [value, name] : std::array<std::pair<real, const char*>, 3>{
             {{real{0}, "hist_c_min_zero"},
              {real{-1}, "hist_c_min_negative"},
              {std::numeric_limits<real>::quiet_NaN(), "hist_c_min_nan"}}}) {
        ResidualHistogramConfig invalid{};
        invalid.c_min_rel = value;
        require_invalid(name, [&, invalid] {
            invoke(grid, q.span(), fluctuations, gauge, real{1}, source_config, invalid, f1.span(), f2.span(),
                  workspace);
        });
    }
    require_invalid("hist_c_max_le_min", [&] {
        ResidualHistogramConfig invalid{};
        invalid.c_max_rel = invalid.c_min_rel;
        invoke(grid, q.span(), fluctuations, gauge, real{1}, source_config, invalid, f1.span(), f2.span(), workspace);
    });

    // --- q / u1 / u2 / f1 / f2 null/short/long checks ---
    require_invalid("q_null", [&] {
        invoke(grid, DeviceSpan<const real>(nullptr, n), fluctuations, gauge, real{1}, source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });
    require_invalid("q_short", [&] {
        invoke(grid, DeviceSpan<const real>(q.data(), n - 1), fluctuations, gauge, real{1}, source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });
    require_invalid("q_long", [&] {
        invoke(grid, DeviceSpan<const real>(q.data(), n + 1), fluctuations, gauge, real{1}, source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });

    require_invalid("u1_null", [&] {
        invoke(grid, q.span(), {DeviceSpan<real>(nullptr, n), u2.span()}, gauge, real{1}, source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });
    require_invalid("u1_short", [&] {
        invoke(grid, q.span(), {DeviceSpan<real>(u1.data(), n - 1), u2.span()}, gauge, real{1}, source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });
    require_invalid("u1_long", [&] {
        invoke(grid, q.span(), {DeviceSpan<real>(u1.data(), n + 1), u2.span()}, gauge, real{1}, source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });

    require_invalid("u2_null", [&] {
        invoke(grid, q.span(), {u1.span(), DeviceSpan<real>(nullptr, n)}, gauge, real{1}, source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });
    require_invalid("u2_short", [&] {
        invoke(grid, q.span(), {u1.span(), DeviceSpan<real>(u2.data(), n - 1)}, gauge, real{1}, source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });
    require_invalid("u2_long", [&] {
        invoke(grid, q.span(), {u1.span(), DeviceSpan<real>(u2.data(), n + 1)}, gauge, real{1}, source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });

    require_invalid("f1_null", [&] {
        invoke(grid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config,
              DeviceSpan<real>(nullptr, n), f2.span(), workspace);
    });
    require_invalid("f1_short", [&] {
        invoke(grid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config,
              DeviceSpan<real>(f1.data(), n - 1), f2.span(), workspace);
    });
    require_invalid("f1_long", [&] {
        invoke(grid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config,
              DeviceSpan<real>(f1.data(), n + 1), f2.span(), workspace);
    });

    require_invalid("f2_null", [&] {
        invoke(grid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config, f1.span(),
              DeviceSpan<real>(nullptr, n), workspace);
    });
    require_invalid("f2_short", [&] {
        invoke(grid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config, f1.span(),
              DeviceSpan<real>(f2.data(), n - 1), workspace);
    });
    require_invalid("f2_long", [&] {
        invoke(grid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config, f1.span(),
              DeviceSpan<real>(f2.data(), n + 1), workspace);
    });

    // --- overlap checks (f1-f2, f1-q, f2-u1) ---
    require_invalid("f1_f2_exact_overlap", [&] {
        invoke(grid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config, f1.span(), f1.span(),
              workspace);
    });
    require_invalid("f1_f2_partial_overlap", [&] {
        invoke(grid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config, f1.span(),
              DeviceSpan<real>(f1.data() + 1, n), workspace);
    });
    require_invalid("f1_q_overlap", [&] {
        invoke(grid, DeviceSpan<const real>(f1.data(), n), fluctuations, gauge, real{1}, source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });
    require_invalid("f2_u1_overlap", [&] {
        invoke(grid, q.span(), {DeviceSpan<real>(f2.data(), n), u2.span()}, gauge, real{1}, source_config,
              histogram_config, f1.span(), f2.span(), workspace);
    });

    // --- logic_error checks ---
    require_logic("unprepared_workspace", [&] {
        StreamfunctionResidualWorkspace wrong_workspace;
        wrong_workspace.prepare(n + 7);
        invoke(grid, q.span(), fluctuations, gauge, real{1}, source_config, histogram_config, f1.span(), f2.span(),
              wrong_workspace);
    });
    require_logic("report_without_enqueue", [&] {
        StreamfunctionResidualWorkspace fresh_workspace;
        fresh_workspace.prepare(n);
        (void)synchronize_streamfunction_residual_report(context, grid, real{1}, source_config, histogram_config,
                                                          fresh_workspace);
    });

    // --- accepted run: u1 and u2 permitted to alias (read-only inputs) ---
    ++checks;
    DeviceBuffer<real> shared_u(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(shared_u.data(), host_u1.data(), n * sizeof(real), cudaMemcpyHostToDevice,
                                           context.cuda_stream()));
    const PeriodicStreamfunctionFluctuations overlapping_fluctuations{shared_u.span(), shared_u.span()};
    invoke(grid, q.span(), overlapping_fluctuations, gauge, real{1}, source_config, histogram_config, f1.span(),
          f2.span(), workspace);
    std::vector<real> accepted_f1(n), accepted_f2(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(accepted_f1.data(), f1.data(), n * sizeof(real), cudaMemcpyDeviceToHost,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(accepted_f2.data(), f2.data(), n * sizeof(real), cudaMemcpyDeviceToHost,
                                           context.cuda_stream()));
    const auto accepted_report = synchronize_streamfunction_residual_report(context, grid, real{1}, source_config,
                                                                             histogram_config, workspace);
    (void)accepted_report;
    bool finite = true;
    double accepted_linf = 0.0;
    for (real value : accepted_f1) {
        finite = finite && std::isfinite(static_cast<double>(value));
        accepted_linf = std::max(accepted_linf, std::abs(static_cast<double>(value)));
    }
    for (real value : accepted_f2) {
        finite = finite && std::isfinite(static_cast<double>(value));
        accepted_linf = std::max(accepted_linf, std::abs(static_cast<double>(value)));
    }
    std::cout << std::setprecision(16)
              << "coupled_residual_contract name=accepted_u1_u2_overlap exception=accepted finite="
              << (finite ? "true" : "false") << " linf=" << accepted_linf << '\n';
    pass = pass && finite;

    return {pass, "coupled_residual_error_contract", "host-validation-and-gpu-acceptance",
            grid_description(fixture.grid), accepted_linf, 0.0, "invalid_argument/logic_error + 1 accepted",
            std::to_string(checks),
            "invalid grid/eta/histogram/span/output-input aliases reject; unprepared/never-enqueued workspace "
            "raises logic_error; overlapping read-only u1==u2 accepted with finite output"};
}

// ---------------------------------------------------------------------------
// Case 9: CPU-side mutants vs coupled_residual_reference.
// ---------------------------------------------------------------------------

[[nodiscard]] double normalized_two_field_rms(const std::vector<double>& actual1, const std::vector<double>& actual2,
                                              const std::vector<double>& expected1,
                                              const std::vector<double>& expected2) {
    long double err = 0.0L, exp2 = 0.0L;
    std::size_t count = 0;
    for (std::size_t i = 0; i < actual1.size(); ++i) {
        const long double delta = static_cast<long double>(actual1[i]) - expected1[i];
        err += delta * delta;
        exp2 += static_cast<long double>(expected1[i]) * expected1[i];
        ++count;
    }
    for (std::size_t i = 0; i < actual2.size(); ++i) {
        const long double delta = static_cast<long double>(actual2[i]) - expected2[i];
        err += delta * delta;
        exp2 += static_cast<long double>(expected2[i]) * expected2[i];
        ++count;
    }
    return std::sqrt(static_cast<double>(err / count)) / std::max(std::sqrt(static_cast<double>(exp2 / count)), 1.0);
}

[[nodiscard]] CaseResult case_coupled_residual_mutation_sensitivity() {
    const auto fixture = make_isotropic_fixture(kGridN);
    const auto q = ref::make_positive_q_field(fixture.grid, fixture.lengths);
    const double eta = 1.0;
    const ref::NonlinearSourceReferenceConfig config{kEpsilon, kVRms};
    const auto correct = ref::coupled_residual_reference(q, fixture, eta, config);

    // Independent recomposition, from the same accepted public oracles used by
    // coupled_residual_reference, so mutants can vary only the combination step.
    const auto a_u1 = ref::divergence_form_diffusion(fixture.grid, q, fixture.psi1_fluctuation);
    const auto a_u2 = ref::divergence_form_diffusion(fixture.grid, q, fixture.psi2_fluctuation);
    const auto affine1 = ref::affine_rhs_discrete(fixture.grid, q, fixture.psi1_affine_gradient);
    const auto affine2 = ref::affine_rhs_discrete(fixture.grid, q, fixture.psi2_affine_gradient);
    const auto& s1 = correct.s1;
    const auto& s2 = correct.s2;
    const std::size_t cells = fixture.grid.cell_count();

    // (a) pairing swap: the production residual is SAME-index (SF-26:
    // G1 uses S1, G2 uses S2; see
    // docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md). This
    // mutant reintroduces the CROSSED pairing (the paper's printed equation
    // (14) form, implemented through SF-25: G1 uses S2, G2 uses S1) and must
    // differ measurably from `correct`.
    std::vector<double> raw1_swap(cells), raw2_swap(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        raw1_swap[i] = affine1[i] - eta * q[i] * s2[i];
        raw2_swap[i] = affine2[i] - eta * q[i] * s1[i];
    }
    const auto proj1_swap = ref::mean_zero_projected(raw1_swap);
    const auto proj2_swap = ref::mean_zero_projected(raw2_swap);
    std::vector<double> f1_swap(cells), f2_swap(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        f1_swap[i] = a_u1[i] - proj1_swap[i];
        f2_swap[i] = a_u2[i] - proj2_swap[i];
    }
    const double pairing_dev = normalized_two_field_rms(f1_swap, f2_swap, correct.f1, correct.f2);

    // (b) RHS sign flip: F = Au + P(G) instead of Au - P(G).
    std::vector<double> f1_sign(cells), f2_sign(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        f1_sign[i] = a_u1[i] + correct.projected_rhs1[i];
        f2_sign[i] = a_u2[i] + correct.projected_rhs2[i];
    }
    const double sign_dev = normalized_two_field_rms(f1_sign, f2_sign, correct.f1, correct.f2);

    // (c) projection omitted: F = Au - raw G (correct SAME-index pairing, no
    // mean-zero projection), to isolate the projection-omission effect from
    // the pairing effect tested in (a).
    std::vector<double> raw1(cells), raw2(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        raw1[i] = affine1[i] - eta * q[i] * s1[i];
        raw2[i] = affine2[i] - eta * q[i] * s2[i];
    }
    std::vector<double> f1_noproj(cells), f2_noproj(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        f1_noproj[i] = a_u1[i] - raw1[i];
        f2_noproj[i] = a_u2[i] - raw2[i];
    }
    const double noproj_dev = normalized_two_field_rms(f1_noproj, f2_noproj, correct.f1, correct.f2);

    std::cout << std::setprecision(16) << "coupled_residual_mutant name=pairing_swap normalized_rms=" << pairing_dev
              << '\n';
    std::cout << std::setprecision(16) << "coupled_residual_mutant name=rhs_sign_flip normalized_rms=" << sign_dev
              << '\n';
    std::cout << std::setprecision(16) << "coupled_residual_mutant name=projection_omitted normalized_rms="
              << noproj_dev << '\n';

    // Explicit thresholds, each documented at least 10x below the measured
    // deviation on this fixture (eta=1, 16^3 isotropic unit cube), never below
    // 1e-6. Measured on this fixture AFTER SF-26 (mutant (a) is now the
    // crossed/pre-SF-26 pairing vs the SAME-index production residual, and
    // mutant (c) omits projection on the SAME-index pairing):
    // pairing_dev ~ 1.0801, sign_dev ~ 1.7220, noproj_dev ~ 1.7497e-1 (all
    // comfortably detectable without increasing eta beyond the shared
    // default of 1).
    constexpr double kPairingThreshold = 9.0e-2;
    constexpr double kSignThreshold = 1.4e-1;
    constexpr double kProjectionThreshold = 1.4e-2;

    const bool finite = std::isfinite(pairing_dev) && std::isfinite(sign_dev) && std::isfinite(noproj_dev);
    const bool pass = finite && pairing_dev > kPairingThreshold && sign_dev > kSignThreshold &&
                      noproj_dev > kProjectionThreshold;
    return {pass, "coupled_residual_mutation_sensitivity", "test-only-mutants-vs-independent-cpu-reference",
            grid_description(fixture.grid), std::min({pairing_dev, sign_dev, noproj_dev}),
            std::max({pairing_dev, sign_dev, noproj_dev}), "each > documented threshold",
            std::to_string(std::min({pairing_dev, sign_dev, noproj_dev})),
            "pairing-swap, RHS-sign-flip, and projection-omitted mutants each exceed their documented "
            "normalized-RMS threshold"};
}

// ---------------------------------------------------------------------------
// SF-26 T02: exact-pair, general-pair, crossed-mutant, and gauge-
// recombination contract cases. These build explicit, closed-form Darcy
// streamfunction pairs (continuum solutions of the SAME-index system; see
// docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md) and run the
// production evaluator on them directly -- independent of the fixed SF-07
// `make_isotropic_fixture`/`make_total_gradient_fixture` analytic forms used
// above.
// ---------------------------------------------------------------------------

// gbar1 = (0, vbar, 0), vbar = 1; gbar2 = (0, 0, 1) (benchmark gauge,
// AffineGauge::benchmark(1)); host-side mirror for the CPU oracle calls.
constexpr ref::Vec3 kGbar1{0.0, 1.0, 0.0};
constexpr ref::Vec3 kGbar2{0.0, 0.0, 1.0};

// Pair A: psi1 = x2 + Phi(x3), psi2 = x3, v = grad(psi1) x grad(psi2) = e1
// exactly. S1 = Phi''(x3), S2 = 0.
[[nodiscard]] double pair_a_phi(double z) {
    return 0.3 * std::sin(2.0 * kPairPi * z) + 0.05 * std::cos(4.0 * kPairPi * z);
}
// Upper bound on max|Phi''| (sum of the two mode amplitudes; the true
// pointwise maximum cannot exceed this since the two terms have different
// frequencies and cannot exceed unit amplitude simultaneously by more than
// their sum): 0.3*(2*pi)^2 + 0.05*(4*pi)^2 ~= 19.74.
constexpr double kPairAPhiSecondDerivativeBound =
    0.3 * (2.0 * kPairPi) * (2.0 * kPairPi) + 0.05 * (4.0 * kPairPi) * (4.0 * kPairPi);

// Pair B: psi1 = x2 + Phi(x3), psi2 = x3 + Theta(x2), v = f(x2,x3)*e1 with
// f = 1 - Phi'(x3)*Theta'(x2) (a genuine curl(v) = grad(ln k) x v Darcy flow;
// the gauge recombination of the trivial pair (x2, x3)). S1, S2 both
// nontrivial.
[[nodiscard]] double pair_b_phi(double z) { return 0.1 * std::sin(2.0 * kPairPi * z); }
[[nodiscard]] double pair_b_phi_prime(double z) { return 0.1 * 2.0 * kPairPi * std::cos(2.0 * kPairPi * z); }
[[nodiscard]] double pair_b_theta(double y) { return 0.1 * std::cos(2.0 * kPairPi * y); }
[[nodiscard]] double pair_b_theta_prime(double y) { return -0.1 * 2.0 * kPairPi * std::sin(2.0 * kPairPi * y); }

// k1(x1) = exp(0.7*sin(2*pi*x1) + 0.3*cos(4*pi*x1)), shared by both pairs'
// x1-dependent conductivity factor.
[[nodiscard]] double kx1_conductivity(double x) {
    return std::exp(0.7 * std::sin(2.0 * kPairPi * x) + 0.3 * std::cos(4.0 * kPairPi * x));
}

struct AnalyticPairFixture {
    ref::Grid grid;
    std::vector<double> q;
    std::vector<double> u1;
    std::vector<double> u2;
};

[[nodiscard]] AnalyticPairFixture make_pair_a_fixture(std::size_t n, bool homogeneous_k) {
    const double h = 1.0 / static_cast<double>(n);
    const ref::Grid grid{n, n, n, {h, h, h}};
    AnalyticPairFixture fixture;
    fixture.grid = grid;
    fixture.q.resize(grid.cell_count());
    fixture.u1.resize(grid.cell_count());
    fixture.u2.assign(grid.cell_count(), 0.0);
    for (std::size_t iz = 0; iz < n; ++iz) {
        for (std::size_t iy = 0; iy < n; ++iy) {
            for (std::size_t ix = 0; ix < n; ++ix) {
                const auto id = grid.index(ix, iy, iz);
                const auto p = grid.cell_center(ix, iy, iz);
                const double k = homogeneous_k ? 1.0 : kx1_conductivity(p.x);
                fixture.q[id] = 1.0 / k;
                fixture.u1[id] = pair_a_phi(p.z);
            }
        }
    }
    return fixture;
}

[[nodiscard]] AnalyticPairFixture make_pair_b_fixture(std::size_t n) {
    const double h = 1.0 / static_cast<double>(n);
    const ref::Grid grid{n, n, n, {h, h, h}};
    AnalyticPairFixture fixture;
    fixture.grid = grid;
    fixture.q.resize(grid.cell_count());
    fixture.u1.resize(grid.cell_count());
    fixture.u2.resize(grid.cell_count());
    for (std::size_t iz = 0; iz < n; ++iz) {
        for (std::size_t iy = 0; iy < n; ++iy) {
            for (std::size_t ix = 0; ix < n; ++ix) {
                const auto id = grid.index(ix, iy, iz);
                const auto p = grid.cell_center(ix, iy, iz);
                const double f = 1.0 - pair_b_phi_prime(p.z) * pair_b_theta_prime(p.y);
                const double k = kx1_conductivity(p.x) * f;
                fixture.q[id] = 1.0 / k;
                fixture.u1[id] = pair_b_phi(p.z);
                fixture.u2[id] = pair_b_theta(p.y);
            }
        }
    }
    return fixture;
}

struct ProductionResidualSummary {
    double r_F{};
    double linf_f1{};
    double linf_f2{};
};

[[nodiscard]] ProductionResidualSummary run_production_residual_summary(
    const AnalyticPairFixture& fixture, double eta, double epsilon = kExactPairEpsilon,
    double v_rms = kExactPairVRms) {
    CoupledResidualGpuFixture gpu(fixture.grid);
    gpu.upload(fixture.q, fixture.u1, fixture.u2);
    const AffineGauge gauge = AffineGauge::benchmark(real{1});
    NonlinearSourceConfig source_config{};
    source_config.epsilon = static_cast<real>(epsilon);
    source_config.v_rms = static_cast<real>(v_rms);
    const ResidualHistogramConfig histogram_config{};
    const auto run = gpu.run_evaluator(gauge, static_cast<real>(eta), source_config, histogram_config);
    return {static_cast<double>(run.report.r_F), static_cast<double>(run.report.linf_f1),
            static_cast<double>(run.report.linf_f2)};
}

// Test-local CROSSED recomposition (the paper-printed, pre-SF-26 pairing:
// G1 uses S2, G2 uses S1), built exclusively from the accepted public CPU
// oracle pieces (never re-derived), exactly like
// case_coupled_residual_mutation_sensitivity above.
[[nodiscard]] double crossed_r_F(const AnalyticPairFixture& fixture, double eta, double epsilon, double v_rms) {
    const ref::NonlinearSourceReferenceConfig config{epsilon, v_rms};
    const auto a_u1 = ref::divergence_form_diffusion(fixture.grid, fixture.q, fixture.u1);
    const auto a_u2 = ref::divergence_form_diffusion(fixture.grid, fixture.q, fixture.u2);
    const auto affine1 = ref::affine_rhs_discrete(fixture.grid, fixture.q, kGbar1);
    const auto affine2 = ref::affine_rhs_discrete(fixture.grid, fixture.q, kGbar2);
    const auto g1 = ref::centered_total_gradient_oracle(fixture.grid, fixture.u1, kGbar1);
    const auto g2 = ref::centered_total_gradient_oracle(fixture.grid, fixture.u2, kGbar2);
    const auto hvb = ref::centered_hessian_vector_b_oracle(fixture.grid, fixture.u1, fixture.u2, g1, g2);
    const auto sources = ref::centered_nonlinear_source_oracle(fixture.grid, g1, g2, hvb.b, config);
    const std::size_t cells = fixture.grid.cell_count();
    std::vector<double> raw1(cells), raw2(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        raw1[i] = affine1[i] - eta * fixture.q[i] * sources.s2[i];
        raw2[i] = affine2[i] - eta * fixture.q[i] * sources.s1[i];
    }
    const auto proj1 = ref::mean_zero_projected(raw1);
    const auto proj2 = ref::mean_zero_projected(raw2);
    std::vector<double> f1(cells), f2(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        f1[i] = a_u1[i] - proj1[i];
        f2[i] = a_u2[i] - proj2[i];
    }
    const double rms_f1 = ref::rms_norm(f1);
    const double rms_f2 = ref::rms_norm(f2);
    const double q_rms = ref::rms_norm(fixture.q);
    const ref::Vec3 lengths{static_cast<double>(fixture.grid.nx) * fixture.grid.spacing.x,
                            static_cast<double>(fixture.grid.ny) * fixture.grid.spacing.y,
                            static_cast<double>(fixture.grid.nz) * fixture.grid.spacing.z};
    const double l_ref = ref::dimensionless_length_reference(lengths);
    return ref::residual_normalization_reference(rms_f1, rms_f2, q_rms, v_rms, l_ref).r_f;
}

[[nodiscard]] double max_abs_value(const std::vector<double>& values) {
    double result = 0.0;
    for (double value : values) result = std::max(result, std::abs(value));
    return result;
}

// Case: exact pair A, homogeneous k=1. DISCRETE FACT (prespecified): the
// discrete residual is ZERO TO ROUNDOFF at every grid (the HVP stencil
// equals the operator's 3-point stencil for a 1D field, z-face harmonic
// means of equal q are exact, and mean(q(x1)*Phi''_h(x3))=0 by separability).
[[nodiscard]] CaseResult case_coupled_residual_exact_pair_k1() {
    bool pass = true;
    double worst_r_F = 0.0;
    std::ostringstream detail;
    for (std::size_t n : {std::size_t{16}, std::size_t{32}, std::size_t{64}}) {
        const auto fixture = make_pair_a_fixture(n, /*homogeneous_k=*/true);
        const auto summary = run_production_residual_summary(fixture, 1.0);
        const double max_q = max_abs_value(fixture.q);
        const double linf_bound = 1.0e-10 * kPairAPhiSecondDerivativeBound * max_q;
        const bool this_pass = std::isfinite(summary.r_F) && summary.r_F <= 1.0e-12 &&
                               std::max(summary.linf_f1, summary.linf_f2) <= linf_bound;
        pass = pass && this_pass;
        worst_r_F = std::max(worst_r_F, summary.r_F);
        std::cout << std::setprecision(16) << "coupled_residual_exact_pair_k1 n=" << n
                  << " r_F=" << summary.r_F << " linf_f1=" << summary.linf_f1
                  << " linf_f2=" << summary.linf_f2 << " linf_bound=" << linf_bound
                  << " pass=" << (this_pass ? "true" : "false") << '\n';
        detail << 'n' << n << ":r_F=" << summary.r_F << ' ';
    }
    return {pass, "coupled_residual_exact_pair_k1", "gpu-exact-pair-A-k-constant", "16/32/64 isotropic unit cube",
            worst_r_F, worst_r_F, "r_F<=1e-12, Linf(F1,F2)<=1e-10*max|Phi''|*max(q)", "n/a (roundoff)",
            "pair A (psi1=x2+Phi(x3), psi2=x3, v=e1 exactly) with k=1: the discrete residual must be "
            "exactly zero to roundoff at every grid (separable exact identity), never merely convergent"};
}

// Case: exact pair A, heterogeneous k=k1(x1). Same discrete-exactness claim
// as k=1: k depends only on x1, u1 depends only on x3, so every harmonic
// mean and the mean-zero projection factorize exactly.
[[nodiscard]] CaseResult case_coupled_residual_exact_pair_kx1() {
    bool pass = true;
    double worst_r_F = 0.0;
    std::ostringstream detail;
    for (std::size_t n : {std::size_t{16}, std::size_t{32}, std::size_t{64}}) {
        const auto fixture = make_pair_a_fixture(n, /*homogeneous_k=*/false);
        const auto summary = run_production_residual_summary(fixture, 1.0);
        const double max_q = max_abs_value(fixture.q);
        const double linf_bound = 1.0e-10 * kPairAPhiSecondDerivativeBound * max_q;
        const bool this_pass = std::isfinite(summary.r_F) && summary.r_F <= 1.0e-12 &&
                               std::max(summary.linf_f1, summary.linf_f2) <= linf_bound;
        pass = pass && this_pass;
        worst_r_F = std::max(worst_r_F, summary.r_F);
        std::cout << std::setprecision(16) << "coupled_residual_exact_pair_kx1 n=" << n
                  << " r_F=" << summary.r_F << " linf_f1=" << summary.linf_f1
                  << " linf_f2=" << summary.linf_f2 << " linf_bound=" << linf_bound
                  << " pass=" << (this_pass ? "true" : "false") << '\n';
        detail << 'n' << n << ":r_F=" << summary.r_F << ' ';
    }
    return {pass, "coupled_residual_exact_pair_kx1", "gpu-exact-pair-A-k-x1-dependent",
            "16/32/64 isotropic unit cube", worst_r_F, worst_r_F,
            "r_F<=1e-12, Linf(F1,F2)<=1e-10*max|Phi''|*max(q)", "n/a (roundoff)",
            "pair A with k=k1(x1)=exp(0.7*sin(2*pi*x1)+0.3*cos(4*pi*x1)): still separable, so the "
            "discrete residual remains exactly zero to roundoff at every grid"};
}

// Case: general pair B, genuine O(h^2) discretization control (harmonic
// means, the projection, and both sources are all nontrivial).
//
// SF-26 C03 amendment (prespecified, orchestrator decision, recorded in the
// bitácora): the ladder is extended to n=16/32/64/128 and the order>=1.9 gate
// is imposed ONLY on the two finest transitions (32->64, 64->128), for both
// r_F and Linf(F1,F2). The threshold itself is NOT relaxed. Rationale: at
// n=16 the cos(4*pi*x1) mode of k1(x1) has only 8 cells per wavelength, so
// the 16->32 transition is pre-asymptotic and its order can legitimately sit
// below 1.9 (observed Linf order ~1.87) without indicating a discretization
// defect. The 16->32 orders are still computed and printed as evidence,
// explicitly labeled non-gating, so the pre-asymptotic behavior remains
// visible rather than silently dropped.
[[nodiscard]] CaseResult case_coupled_residual_exact_pair_general() {
    const std::vector<std::size_t> ns{16, 32, 64, 128};
    std::vector<double> r_f_values(ns.size()), linf_values(ns.size()), hs(ns.size());
    for (std::size_t i = 0; i < ns.size(); ++i) {
        const auto fixture = make_pair_b_fixture(ns[i]);
        const auto summary = run_production_residual_summary(fixture, 1.0);
        r_f_values[i] = summary.r_F;
        linf_values[i] = std::max(summary.linf_f1, summary.linf_f2);
        hs[i] = 1.0 / static_cast<double>(ns[i]);
        std::cout << std::setprecision(16) << "coupled_residual_exact_pair_general n=" << ns[i]
                  << " r_F=" << summary.r_F << " linf_f1=" << summary.linf_f1
                  << " linf_f2=" << summary.linf_f2 << '\n';
    }
    bool pass = true;
    std::ostringstream orders;
    for (std::size_t i = 0; i + 1 < ns.size(); ++i) {
        const auto order_rF = ref::observed_order(r_f_values[i], r_f_values[i + 1], hs[i], hs[i + 1]);
        const auto order_linf = ref::observed_order(linf_values[i], linf_values[i + 1], hs[i], hs[i + 1]);
        const bool gated = i >= 1;  // 32->64 and 64->128 only; 16->32 is pre-asymptotic evidence.
        if (gated) {
            const bool ok_rF = order_rF.valid() && order_rF.value >= 1.9;
            const bool ok_linf = order_linf.valid() && order_linf.value >= 1.9;
            pass = pass && ok_rF && ok_linf;
            std::cout << std::setprecision(16) << "coupled_residual_exact_pair_general order n=" << ns[i]
                      << "->" << ns[i + 1] << " order_r_F=" << (order_rF.valid() ? order_rF.value : -1.0)
                      << " order_linf=" << (order_linf.valid() ? order_linf.value : -1.0) << '\n';
            orders << 'n' << ns[i] << "->" << ns[i + 1] << ":rF=" << (order_rF.valid() ? order_rF.value : -1.0)
                   << ",linf=" << (order_linf.valid() ? order_linf.value : -1.0) << ' ';
        } else {
            std::cout << std::setprecision(16) << "coupled_residual_exact_pair_general order n=" << ns[i]
                      << "->" << ns[i + 1] << " order_r_F=" << (order_rF.valid() ? order_rF.value : -1.0)
                      << " order_linf=" << (order_linf.valid() ? order_linf.value : -1.0)
                      << " pre-asymptotic (evidence, not gated)" << '\n';
            orders << 'n' << ns[i] << "->" << ns[i + 1]
                   << ":rF=" << (order_rF.valid() ? order_rF.value : -1.0)
                   << ",linf=" << (order_linf.valid() ? order_linf.value : -1.0)
                   << ",pre-asymptotic(evidence,not-gated) ";
        }
    }
    return {pass, "coupled_residual_exact_pair_general", "gpu-exact-pair-B-general-oh2",
            "16/32/64/128 isotropic unit cube", r_f_values.front(), r_f_values.back(),
            "observed order >=1.9 for r_F and Linf(F1,F2) on the 32->64 and 64->128 transitions "
            "only; 16->32 is printed as pre-asymptotic evidence and is not gated (SF-26 C03 "
            "amendment: at n=16 the cos(4*pi*x1) mode of k1 has only 8 cells per wavelength)",
            orders.str(),
            "pair B (a genuine curl(v)=grad(ln k) x v Darcy flow, the gauge recombination of the "
            "trivial pair) is a real O(h^2) discretization control: harmonic means, the projection, "
            "and both nontrivial sources all contribute; the ladder was extended to n=128 (SF-26 "
            "C03) so the order>=1.9 gate is measured only where the k1(x1) heterogeneity mode is "
            "already well resolved"};
}

// Case: crossed pairing evaluated on exact SAME-index Darcy pairs is an O(1)
// model-error floor that must NOT shrink under refinement (contrast with the
// genuine O(h^2) discretization errors of the correct pairing above).
[[nodiscard]] CaseResult case_coupled_residual_pairing_mutant() {
    const std::vector<std::size_t> ns{16, 32, 64};
    std::vector<double> r_a(ns.size()), r_b(ns.size());
    for (std::size_t i = 0; i < ns.size(); ++i) {
        const auto fixture_a = make_pair_a_fixture(ns[i], /*homogeneous_k=*/false);
        const auto fixture_b = make_pair_b_fixture(ns[i]);
        r_a[i] = crossed_r_F(fixture_a, 1.0, kExactPairEpsilon, kExactPairVRms);
        r_b[i] = crossed_r_F(fixture_b, 1.0, kExactPairEpsilon, kExactPairVRms);
        std::cout << std::setprecision(16) << "coupled_residual_pairing_mutant n=" << ns[i]
                  << " r_F_crossed_pairA=" << r_a[i] << " r_F_crossed_pairB=" << r_b[i] << '\n';
    }
    bool pass = true;
    for (std::size_t i = 0; i < ns.size(); ++i) {
        pass = pass && std::isfinite(r_a[i]) && std::isfinite(r_b[i]) && r_a[i] > 1.0e-2 && r_b[i] > 1.0e-2;
    }
    const double ratio_a = r_a.front() > 0.0 ? r_a.back() / r_a.front() : 0.0;
    const double ratio_b = r_b.front() > 0.0 ? r_b.back() / r_b.front() : 0.0;
    pass = pass && ratio_a >= 0.5 && ratio_b >= 0.5;
    std::cout << std::setprecision(16) << "coupled_residual_pairing_mutant ratio64_16_pairA=" << ratio_a
              << " ratio64_16_pairB=" << ratio_b << '\n';
    return {pass, "coupled_residual_pairing_mutant", "test-local-crossed-recomposition-vs-oracle-pieces",
            "16/32/64 isotropic unit cube, pair A (k=k1(x1)) and pair B",
            std::min({r_a.front(), r_b.front()}), std::min({r_a.back(), r_b.back()}),
            "r_F_crossed>1e-2 at every grid; ratio(n=64,n=16)>=0.5 (non-decreasing under refinement)",
            "ratioA=" + std::to_string(ratio_a) + " ratioB=" + std::to_string(ratio_b),
            "the CROSSED (paper-printed, pre-SF-26) pairing evaluated on exact SAME-index Darcy "
            "streamfunction pairs is an O(1) model-error floor, not a discretization artifact: it "
            "must stay large and must NOT shrink under h-refinement"};
}

// Case: gauge recombination invariance (cheap tier). psi1 -> psi1 +
// alpha*Phihat(psi2) leaves v unchanged (same streamsurfaces), so the
// SAME-index residual on the recombined pair must remain a comparable,
// still-O(h^2) discretization residual.
//
// SF-26 C03 amendment (prespecified, orchestrator decision, recorded in the
// bitácora): the ladder is extended to n=16/32/64/128. The `ratio<=3`
// same-vs-base-residual gate is kept at every grid (unchanged). The
// order>=1.9 gate on r_F(recombined) is imposed ONLY on the two finest
// transitions (32->64, 64->128); the threshold is NOT relaxed. Rationale:
// at n=16 the cos(4*pi*x1) mode of k1(x1) underlying pair B has only 8
// cells per wavelength, so the 16->32 transition is pre-asymptotic. The
// 16->32 order is still computed and printed as non-gating evidence.
[[nodiscard]] CaseResult case_coupled_residual_gauge_recombination_analytic() {
    constexpr double kAlpha = 0.05;
    const std::vector<std::size_t> ns{16, 32, 64, 128};
    std::vector<double> r_base(ns.size()), r_recombined(ns.size());
    double worst_ratio = 0.0;
    for (std::size_t i = 0; i < ns.size(); ++i) {
        const std::size_t n = ns[i];
        const auto base = make_pair_b_fixture(n);
        const auto base_summary = run_production_residual_summary(base, 1.0);
        r_base[i] = base_summary.r_F;

        AnalyticPairFixture recombined = base;
        const std::size_t cells = base.grid.cell_count();
        std::vector<double> u1_raw(cells);
        for (std::size_t iz = 0; iz < n; ++iz) {
            for (std::size_t iy = 0; iy < n; ++iy) {
                for (std::size_t ix = 0; ix < n; ++ix) {
                    const auto id = base.grid.index(ix, iy, iz);
                    const auto p = base.grid.cell_center(ix, iy, iz);
                    const double psi2 = p.z + pair_b_theta(p.y);
                    u1_raw[id] = base.u1[id] + kAlpha * std::sin(2.0 * kPairPi * psi2);
                }
            }
        }
        recombined.u1 = ref::mean_zero_projected(u1_raw);

        const auto recombined_summary = run_production_residual_summary(recombined, 1.0);
        r_recombined[i] = recombined_summary.r_F;
        const double ratio =
            r_base[i] > 0.0 ? r_recombined[i] / r_base[i] : std::numeric_limits<double>::infinity();
        worst_ratio = std::max(worst_ratio, ratio);
        const double cross_evidence = crossed_r_F(recombined, 1.0, kExactPairEpsilon, kExactPairVRms);
        std::cout << std::setprecision(16) << "coupled_residual_gauge_recombination_analytic n=" << n
                  << " r_F_base=" << r_base[i] << " r_F_recombined=" << r_recombined[i]
                  << " ratio=" << ratio << " r_F_crossed_recombined=" << cross_evidence << '\n';
    }
    bool pass = true;
    for (std::size_t i = 0; i < ns.size(); ++i) {
        pass = pass && std::isfinite(r_recombined[i]) && r_recombined[i] <= 3.0 * r_base[i];
    }
    for (std::size_t i = 0; i + 1 < ns.size(); ++i) {
        const auto order = ref::observed_order(r_recombined[i], r_recombined[i + 1],
                                               1.0 / static_cast<double>(ns[i]),
                                               1.0 / static_cast<double>(ns[i + 1]));
        const bool gated = i >= 1;  // 32->64 and 64->128 only; 16->32 is pre-asymptotic evidence.
        if (gated) {
            const bool ok = order.valid() && order.value >= 1.9;
            pass = pass && ok;
            std::cout << std::setprecision(16) << "coupled_residual_gauge_recombination_analytic order n="
                      << ns[i] << "->" << ns[i + 1] << " order=" << (order.valid() ? order.value : -1.0)
                      << '\n';
        } else {
            std::cout << std::setprecision(16) << "coupled_residual_gauge_recombination_analytic order n="
                      << ns[i] << "->" << ns[i + 1] << " order=" << (order.valid() ? order.value : -1.0)
                      << " pre-asymptotic (evidence, not gated)" << '\n';
        }
    }
    return {pass, "coupled_residual_gauge_recombination_analytic", "gpu-gauge-recombination-invariance",
            "16/32/64/128 isotropic unit cube, pair B recombined psi1->psi1+alpha*sin(2*pi*psi2)",
            r_base.front(), r_recombined.back(),
            "r_F(recombined)<=3*r_F(base) at every grid; order>=1.9 on the 32->64 and 64->128 "
            "transitions only (16->32 is printed as pre-asymptotic evidence and is not gated: "
            "SF-26 C03 amendment, cos(4*pi*x1) mode of k1 has only 8 cells per wavelength at n=16)",
            std::to_string(worst_ratio),
            "recombining psi1 -> psi1 + alpha*Phihat(psi2) (alpha=0.05) preserves v exactly, so the "
            "SAME-index residual on the recombined pair must remain a comparable, still-O(h^2) "
            "discretization residual; the ladder was extended to n=128 (SF-26 C03) so the order>=1.9 "
            "gate is measured only where the k1(x1) heterogeneity mode is already well resolved; the "
            "test-local crossed recomposition on the same recombined state is printed as additional "
            "(non-gating) evidence at every grid"};
}

// ---------------------------------------------------------------------------
// SF-26 T02 HEAVY case: gauge recombination on a genuinely converged
// sigma_Y^2=0.25, 32^3 heterogeneous state (VERBATIM fixture parameters of
// heterogeneity_continuation_gpu_cases.cu::run_heterogeneity_smoke, called
// here through the same production `run_streamfunction_heterogeneity_
// continuation` entry point rather than the (private, unexported)
// `run_heterogeneity_smoke` function itself).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_coupled_residual_gauge_recombination_sigma025() {
    constexpr int n = 32;
    const Grid3D grid(n, n, n, real{1}, real{1}, real{1});
    const std::size_t cells = static_cast<std::size_t>(grid.num_cells());

    CudaContext ctx(0);

    // VERBATIM smoke fixture (sigma2=0.25 leg): seed 12345, corr_length 8,
    // normalize_variance, Anderson depth 5/start 5/limit 1e12, newton
    // enabled, degenerate epsilon leg (target==start), lambda axis defaults,
    // default AffinePeriodicFlowConfig (qbar=(1,0,0)).
    physics::PeriodicGaussianFieldConfig field_config;
    field_config.sigma2 = real{0.25};
    field_config.corr_length = real{8};
    field_config.seed = 12345ULL;
    field_config.normalize_variance = true;

    DeviceBuffer<real> y(cells);
    physics::PeriodicGaussianFieldWorkspace field_workspace;
    const physics::PeriodicGaussianFieldReport field_report =
        physics::generate_periodic_gaussian_field(ctx, grid, field_config, y.span(), field_workspace);
    ctx.synchronize();

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    StreamfunctionSolverConfig base_config{}; // full defaults (adaptive Picard, max_iter=500,
                                              // tolerance=1e-6, linear rtol=1e-10)
    base_config.anderson.enabled = true;
    base_config.anderson.depth = 5;
    base_config.anderson.start_iteration = 5;
    base_config.anderson.condition_limit = real{1e12};
    base_config.newton.enabled = true;

    HeterogeneityContinuationConfig continuation_config{}; // lambda axis defaults
    continuation_config.inner.epsilon_log10.target = continuation_config.inner.epsilon_log10.start;
    const physics::AffinePeriodicFlowConfig flow_config{}; // qbar=(1,0,0) default

    const HeterogeneityContinuationReport report = run_streamfunction_heterogeneity_continuation(
        ctx, grid, DeviceSpan<const real>(y.span()), continuation_config, flow_config, base_config, fields,
        workspace);
    ctx.synchronize();

    std::cout << std::setprecision(16)
              << "coupled_residual_gauge_recombination_sigma025 field_raw_mean=" << field_report.raw_mean
              << " field_final_variance=" << field_report.final_variance
              << " status=" << static_cast<int>(report.status) << " final_lambda=" << report.final_lambda
              << " final_eta=" << report.final_eta << " stage_history_size=" << report.stage_history.size()
              << '\n';
    for (const auto& record : report.stage_history) {
        std::cout << "  stage axis=" << static_cast<int>(record.axis) << " lambda=" << record.lambda_value
                  << " eta=" << record.eta_value << " epsilon=" << record.epsilon_value
                  << " accepted=" << (record.base.accepted ? "true" : "false") << " r_F=" << record.base.r_F
                  << '\n';
    }

    const bool preconditions_ok =
        report.status == HeterogeneityStatus::reached_target && report.final_lambda == real{1};
    if (!preconditions_ok) {
        return {false, "coupled_residual_gauge_recombination_sigma025",
                "gpu-heavy-gauge-recombination-sigma025", "32^3, sigma_Y^2=0.25, corr_length=8, seed=12345",
                0.0, 0.0, "status=reached_target, final_lambda=1", "precondition_failed",
                "the VERBATIM run_heterogeneity_smoke(sigma2=0.25) fixture did not reach the full "
                "lognormal target; the case fails per the PRESPECIFIED contract rather than adapting "
                "the fixture (see the printed stage table above)"};
    }

    std::vector<real> y_host(cells);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpyAsync(y_host.data(), y.data(), cells * sizeof(real), cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();

    std::vector<double> q_host(cells);
    for (std::size_t i = 0; i < cells; ++i) q_host[i] = std::exp(-static_cast<double>(y_host[i]));

    const double v_rms = static_cast<double>(report.final_solve.diagnostics.v_d_rms);

    std::vector<real> u1_host(cells), u2_host(cells);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u1_host.data(), fields.u1_span().data(), cells * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u2_host.data(), fields.u2_span().data(), cells * sizeof(real),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));
    ctx.synchronize();

    std::vector<real> q_real(q_host.begin(), q_host.end());
    DeviceBuffer<real> q_dev(cells);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpyAsync(q_dev.data(), q_real.data(), cells * sizeof(real), cudaMemcpyHostToDevice, ctx.cuda_stream()));

    const AffineGauge gauge = AffineGauge::benchmark(real{1});
    NonlinearSourceConfig source_config{};
    source_config.epsilon = real{1e-2};
    source_config.v_rms = static_cast<real>(v_rms);
    const ResidualHistogramConfig histogram_config{};

    StreamfunctionResidualWorkspace base_residual_workspace;
    base_residual_workspace.prepare(cells);
    DeviceBuffer<real> f1_base(cells), f2_base(cells);
    enqueue_streamfunction_residual(ctx, grid, DeviceSpan<const real>(q_dev.span()),
                                    {fields.u1_span(), fields.u2_span()}, gauge, real{1}, source_config,
                                    histogram_config, f1_base.span(), f2_base.span(), base_residual_workspace);
    const StreamfunctionResidualReport base_report = synchronize_streamfunction_residual_report(
        ctx, grid, real{1}, source_config, histogram_config, base_residual_workspace);
    const double r_F_base = static_cast<double>(base_report.r_F);
    const double l_ref = static_cast<double>(base_report.L_ref);

    // Gauge recombination on the host: u1' = P(u1 + alpha*sin(2*pi*(x3+u2)/L3)),
    // L3 = nz*dz = 32 (dx=dy=dz=1), alpha = 0.02*L3/(2*pi) ~= 0.1019 (so the
    // added slope alpha*Phihat'(s)/L3*2*pi <= 0.02).
    constexpr double kL3 = 32.0;
    const double alpha = 0.02 * kL3 / (2.0 * kPairPi);
    const ref::Grid host_grid{static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                              static_cast<std::size_t>(n), {1.0, 1.0, 1.0}};
    std::vector<double> u1_double(u1_host.begin(), u1_host.end());
    std::vector<double> u2_double(u2_host.begin(), u2_host.end());
    std::vector<double> u1_recombined_raw(cells);
    for (std::size_t iz = 0; iz < host_grid.nz; ++iz) {
        for (std::size_t iy = 0; iy < host_grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < host_grid.nx; ++ix) {
                const auto id = host_grid.index(ix, iy, iz);
                const auto p = host_grid.cell_center(ix, iy, iz);
                const double psi2 = p.z + u2_double[id];
                u1_recombined_raw[id] = u1_double[id] + alpha * std::sin(2.0 * kPairPi * psi2 / kL3);
            }
        }
    }
    const auto u1_recombined = ref::mean_zero_projected(u1_recombined_raw);

    std::vector<real> u1_recombined_real(u1_recombined.begin(), u1_recombined.end());
    DeviceBuffer<real> u1p_dev(cells), u2p_dev(cells);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u1p_dev.data(), u1_recombined_real.data(), cells * sizeof(real),
                                           cudaMemcpyHostToDevice, ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u2p_dev.data(), u2_host.data(), cells * sizeof(real),
                                           cudaMemcpyHostToDevice, ctx.cuda_stream()));
    ctx.synchronize();

    StreamfunctionResidualWorkspace recombined_residual_workspace;
    recombined_residual_workspace.prepare(cells);
    DeviceBuffer<real> f1_same(cells), f2_same(cells);
    enqueue_streamfunction_residual(ctx, grid, DeviceSpan<const real>(q_dev.span()),
                                    {u1p_dev.span(), u2p_dev.span()}, gauge, real{1}, source_config,
                                    histogram_config, f1_same.span(), f2_same.span(),
                                    recombined_residual_workspace);
    const StreamfunctionResidualReport same_report = synchronize_streamfunction_residual_report(
        ctx, grid, real{1}, source_config, histogram_config, recombined_residual_workspace);
    const double r_F_same = static_cast<double>(same_report.r_F);

    // Test-local crossed recomposition on the SAME recombined state, CPU
    // oracle pieces only, pipeline's q = exp(-Y), eta=1, epsilon=1e-2, same
    // v_rms; L_ref taken from the production report (base_report.L_ref) so
    // r_F_same and r_F_cross share the identical normalization.
    const ref::NonlinearSourceReferenceConfig cross_config{1.0e-2, v_rms};
    const auto a_u1 = ref::divergence_form_diffusion(host_grid, q_host, u1_recombined);
    const auto a_u2 = ref::divergence_form_diffusion(host_grid, q_host, u2_double);
    const auto affine1 = ref::affine_rhs_discrete(host_grid, q_host, kGbar1);
    const auto affine2 = ref::affine_rhs_discrete(host_grid, q_host, kGbar2);
    const auto g1 = ref::centered_total_gradient_oracle(host_grid, u1_recombined, kGbar1);
    const auto g2 = ref::centered_total_gradient_oracle(host_grid, u2_double, kGbar2);
    const auto hvb = ref::centered_hessian_vector_b_oracle(host_grid, u1_recombined, u2_double, g1, g2);
    const auto sources = ref::centered_nonlinear_source_oracle(host_grid, g1, g2, hvb.b, cross_config);
    std::vector<double> raw1(cells), raw2(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        raw1[i] = affine1[i] - 1.0 * q_host[i] * sources.s2[i];
        raw2[i] = affine2[i] - 1.0 * q_host[i] * sources.s1[i];
    }
    const auto proj1 = ref::mean_zero_projected(raw1);
    const auto proj2 = ref::mean_zero_projected(raw2);
    std::vector<double> f1_cross(cells), f2_cross(cells);
    for (std::size_t i = 0; i < cells; ++i) {
        f1_cross[i] = a_u1[i] - proj1[i];
        f2_cross[i] = a_u2[i] - proj2[i];
    }
    const double rms_f1_cross = ref::rms_norm(f1_cross);
    const double rms_f2_cross = ref::rms_norm(f2_cross);
    const double q_rms_host = ref::rms_norm(q_host);
    const double r_F_cross =
        ref::residual_normalization_reference(rms_f1_cross, rms_f2_cross, q_rms_host, v_rms, l_ref).r_f;

    std::cout << std::setprecision(16) << "coupled_residual_gauge_recombination_sigma025 v_rms=" << v_rms
              << " alpha=" << alpha << " r_F_base=" << r_F_base << " r_F_same=" << r_F_same
              << " r_F_cross=" << r_F_cross << " l_ref=" << l_ref << '\n';

    const bool pass = std::isfinite(r_F_base) && std::isfinite(r_F_same) && std::isfinite(r_F_cross) &&
                      r_F_base <= 1.0e-6 && r_F_same <= (1.0 / 20.0) * r_F_cross && r_F_same <= 1.0e-1;

    return {pass, "coupled_residual_gauge_recombination_sigma025", "gpu-heavy-gauge-recombination-sigma025",
            "32^3, sigma_Y^2=0.25, corr_length=8, seed=12345, converged full lognormal K=exp(Y)", r_F_base,
            r_F_same, "r_F_base<=1e-6; r_F_same<=1e-1; r_F_same<=(1/20)*r_F_cross", std::to_string(r_F_cross),
            "gauge recombination psi1->psi1+alpha*sin(2*pi*(x3+u2)/L3) (alpha~0.1019) on the converged "
            "heterogeneous state must keep the SAME-index production residual small while the "
            "test-local crossed recomposition on the identical recombined state remains at least 20x "
            "larger, demonstrating the production pairing is the physically consistent one on a real "
            "converged heterogeneous field, not merely on synthetic exact pairs"};
}

} // namespace

CaseRegistry coupled_residual_case_registry() {
    return {{"coupled_residual_gpu_oracle", case_coupled_residual_gpu_oracle},
            {"coupled_residual_direct_agreement", case_coupled_residual_direct_agreement},
            {"coupled_residual_reductions_agreement", case_coupled_residual_reductions_agreement},
            {"coupled_residual_histogram_agreement", case_coupled_residual_histogram_agreement},
            {"coupled_residual_percentile_bound", case_coupled_residual_percentile_bound},
            {"coupled_residual_homogeneous_zero", case_coupled_residual_homogeneous_zero},
            {"coupled_residual_mean_zero_gauge", case_coupled_residual_mean_zero_gauge},
            {"coupled_residual_error_contract", case_coupled_residual_error_contract},
            {"coupled_residual_mutation_sensitivity", case_coupled_residual_mutation_sensitivity},
            {"coupled_residual_exact_pair_k1", case_coupled_residual_exact_pair_k1},
            {"coupled_residual_exact_pair_kx1", case_coupled_residual_exact_pair_kx1},
            {"coupled_residual_exact_pair_general", case_coupled_residual_exact_pair_general},
            {"coupled_residual_pairing_mutant", case_coupled_residual_pairing_mutant},
            {"coupled_residual_gauge_recombination_analytic",
             case_coupled_residual_gauge_recombination_analytic}};
}

CaseRegistry coupled_residual_heavy_case_registry() {
    return {{"coupled_residual_gauge_recombination_sigma025",
             case_coupled_residual_gauge_recombination_sigma025}};
}

} // namespace macroflow3d::streamfunctions::test
