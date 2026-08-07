#include "reference_operators.hpp"
#include "streamfunction_operator_test_cases.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/DifferentialOperators.cuh"
#include "src/physics/streamfunctions/NonlinearSources.cuh"
#include "src/physics/streamfunctions/affine_gauge.cuh"
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
#include <tuple>
#include <vector>

namespace macroflow3d::streamfunctions::test {
namespace {
namespace ref = macroflow3d::streamfunctions::reference;

constexpr double kDiscreteOracleTolerance = 5.0e-11;
constexpr double kOrderThreshold = 1.8;
constexpr double kMutationThreshold = 1.0e-2;
// Explicit, regularization-independent degeneracy mask threshold for the
// smooth-order convergence case. `make_total_gradient_fixture` is built from
// O(1)-coefficient trigonometric fluctuations with wavenumbers up to
// 2*pi/1.0, so |c|^2 = |grad(psi1) x grad(psi2)|^2 ranges from ~1e-5 up to
// ~2900 across the domain (median ~200); a threshold of 10 keeps a large
// majority of cells (~94%, printed per level) while excluding the cells
// whose gradients are locally near-parallel, which is exactly the
// discretization-order-degrading region this convergence case must avoid.
constexpr double kMaskThreshold = 10.0;
// epsilon for the "regularization-dominated" unmasked convergence case.
// (epsilon*v_rms)^2 must dominate the *typical* scale of |c|^2 on this
// fixture (up to ~2900, not O(1)) for the regularized denominator to stay
// well-conditioned everywhere; epsilon=5 gives (epsilon*v_rms)^2=25, enough
// to recover clean second-order global convergence (measured below).
constexpr double kLargeEpsilon = 5.0;
constexpr double kMinKeptFraction = 0.5;

template <typename Callable>
[[nodiscard]] bool rejects_with_invalid_argument(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "nonlinear_sources_contract name=" << name
                  << " exception=none expected=std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument& error) {
        std::cout << "nonlinear_sources_contract name=" << name
                  << " exception=std::invalid_argument message=" << error.what() << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cout << "nonlinear_sources_contract name=" << name
                  << " exception=std::exception message=" << error.what()
                  << " expected=std::invalid_argument\n";
        return false;
    } catch (...) {
        std::cout << "nonlinear_sources_contract name=" << name
                  << " exception=non-standard expected=std::invalid_argument\n";
        return false;
    }
}

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
[[nodiscard]] double rms(const std::vector<double>& values) {
    long double sum = 0.0L;
    for (double value : values) sum += static_cast<long double>(value) * value;
    return std::sqrt(static_cast<double>(sum / values.size()));
}
[[nodiscard]] double rms_difference(const std::vector<real>& actual,
                                    const std::vector<double>& expected) {
    long double sum = 0.0L;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const long double delta = static_cast<long double>(actual[i]) - expected[i];
        sum += delta * delta;
    }
    return std::sqrt(static_cast<double>(sum / actual.size()));
}
[[nodiscard]] double periodic_boundary_linf(const ref::Grid& grid, const std::vector<real>& actual,
                                            const std::vector<double>& expected) {
    double maximum = 0.0;
    for (std::size_t z = 0; z < grid.nz; ++z) for (std::size_t y = 0; y < grid.ny; ++y)
        for (std::size_t x = 0; x < grid.nx; ++x) {
            if (x != 0 && x + 1 != grid.nx && y != 0 && y + 1 != grid.ny &&
                z != 0 && z + 1 != grid.nz) continue;
            const auto i = grid.index(x, y, z);
            maximum = std::max(maximum, std::abs(static_cast<double>(actual[i]) - expected[i]));
        }
    return maximum;
}
[[nodiscard]] double normalized(double value, double scale) { return value / std::max(scale, 1.0); }
[[nodiscard]] double dot(const ref::Vec3& a, const ref::Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] NonlinearSourceConfig production_config(double epsilon, double v_rms,
                                                       const std::vector<double>& thresholds = {}) {
    NonlinearSourceConfig config{};
    config.epsilon = static_cast<real>(epsilon);
    config.v_rms = static_cast<real>(v_rms);
    config.num_degeneracy_thresholds = static_cast<int>(thresholds.size());
    for (std::size_t t = 0; t < thresholds.size(); ++t) {
        config.degeneracy_thresholds[t] = static_cast<real>(thresholds[t]);
    }
    return config;
}

// Owns the full SF-07 -> SF-08 -> SF-09 GPU pipeline on one CUDA stream.  The
// gradients and B buffers stay device-resident so callers can rerun the SF-09
// kernel with different configs, download intermediate fields for CPU
// mirrors, or overwrite a gradient input to reproduce a nonfinite injection.
class NonlinearSourceGpuFixture {
  public:
    explicit NonlinearSourceGpuFixture(const ref::TotalGradientFixture& source)
        : grid_(production_grid(source.grid)), context_(0), n_(source.grid.cell_count()),
          u1_(n_), u2_(n_), p1x_(n_), p1y_(n_), p1z_(n_), p2x_(n_), p2y_(n_), p2z_(n_),
          h2g1x_(n_), h2g1y_(n_), h2g1z_(n_), h1g2x_(n_), h1g2y_(n_), h1g2z_(n_), bx_(n_),
          by_(n_), bz_(n_), s1_(n_), s2_(n_), counters_(kMaxCounters) {}

    void compute_gradients_and_b(const ref::TotalGradientFixture& source) {
        const std::vector<real> h1(source.psi1_fluctuation.begin(), source.psi1_fluctuation.end());
        const std::vector<real> h2(source.psi2_fluctuation.begin(), source.psi2_fluctuation.end());
        // Single stream: H2D -> SF-07 gradients -> SF-08 HVP/B. No synchronize
        // here; callers synchronize only when reading results back.
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u1_.data(), h1.data(), h1.size() * sizeof(real),
                                               cudaMemcpyHostToDevice, context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u2_.data(), h2.data(), h2.size() * sizeof(real),
                                               cudaMemcpyHostToDevice, context_.cuda_stream()));
        enqueue_total_streamfunction_gradients(
            context_, grid_, {u1_.span(), u2_.span()}, production_gauge(source),
            {p1x_.span(), p1y_.span(), p1z_.span(), p2x_.span(), p2y_.span(), p2z_.span()});
        enqueue_streamfunction_hessian_vector_b(
            context_, grid_, {u1_.span(), u2_.span()},
            {p1x_.span(), p1y_.span(), p1z_.span(), p2x_.span(), p2y_.span(), p2z_.span()},
            {h2g1x_.span(), h2g1y_.span(), h2g1z_.span(), h1g2x_.span(), h1g2y_.span(),
             h1g2z_.span(), bx_.span(), by_.span(), bz_.span()});
    }

    [[nodiscard]] TotalStreamfunctionGradientView gradients_view() const {
        return {p1x_.span(), p1y_.span(), p1z_.span(), p2x_.span(), p2y_.span(), p2z_.span()};
    }
    [[nodiscard]] StreamfunctionBFieldView b_view() const { return {bx_.span(), by_.span(), bz_.span()}; }

    struct SourcesResult {
        std::vector<real> s1;
        std::vector<real> s2;
        std::vector<unsigned long long> counters;
    };

    [[nodiscard]] SourcesResult run_sources(const NonlinearSourceConfig& config) {
        const auto num_counters = static_cast<std::size_t>(2 + config.num_degeneracy_thresholds);
        enqueue_streamfunction_nonlinear_sources(
            context_, grid_, gradients_view(), b_view(), config, {s1_.span(), s2_.span()},
            {DeviceSpan<unsigned long long>(counters_.data(), num_counters)});
        SourcesResult result;
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

    struct HostFields {
        ref::VectorField g1, g2, b;
    };

    [[nodiscard]] HostFields download_gradients_and_b() {
        std::vector<real> hp1x(n_), hp1y(n_), hp1z(n_), hp2x(n_), hp2y(n_), hp2z(n_), hbx(n_),
            hby(n_), hbz(n_);
        const std::array<std::pair<real*, const DeviceBuffer<real>*>, 9> transfers{
            {{hp1x.data(), &p1x_}, {hp1y.data(), &p1y_}, {hp1z.data(), &p1z_}, {hp2x.data(), &p2x_},
             {hp2y.data(), &p2y_}, {hp2z.data(), &p2z_}, {hbx.data(), &bx_}, {hby.data(), &by_},
             {hbz.data(), &bz_}}};
        for (const auto& [host_ptr, buffer] : transfers) {
            MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(host_ptr, buffer->data(), n_ * sizeof(real),
                                                   cudaMemcpyDeviceToHost, context_.cuda_stream()));
        }
        context_.synchronize();
        HostFields result;
        result.g1 = {std::vector<double>(hp1x.begin(), hp1x.end()),
                     std::vector<double>(hp1y.begin(), hp1y.end()),
                     std::vector<double>(hp1z.begin(), hp1z.end())};
        result.g2 = {std::vector<double>(hp2x.begin(), hp2x.end()),
                     std::vector<double>(hp2y.begin(), hp2y.end()),
                     std::vector<double>(hp2z.begin(), hp2z.end())};
        result.b = {std::vector<double>(hbx.begin(), hbx.end()), std::vector<double>(hby.begin(), hby.end()),
                    std::vector<double>(hbz.begin(), hbz.end())};
        return result;
    }

    // Overwrites only the g1 total-gradient buffers, leaving g2 and B intact.
    void upload_g1(const ref::VectorField& g1) {
        const std::vector<real> hx(g1.x.begin(), g1.x.end());
        const std::vector<real> hy(g1.y.begin(), g1.y.end());
        const std::vector<real> hz(g1.z.begin(), g1.z.end());
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(p1x_.data(), hx.data(), n_ * sizeof(real),
                                               cudaMemcpyHostToDevice, context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(p1y_.data(), hy.data(), n_ * sizeof(real),
                                               cudaMemcpyHostToDevice, context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(p1z_.data(), hz.data(), n_ * sizeof(real),
                                               cudaMemcpyHostToDevice, context_.cuda_stream()));
        context_.synchronize();
    }

  private:
    static constexpr int kMaxCounters = 2 + kMaxDegeneracyThresholds;
    Grid3D grid_;
    CudaContext context_;
    std::size_t n_;
    DeviceBuffer<real> u1_, u2_, p1x_, p1y_, p1z_, p2x_, p2y_, p2z_;
    DeviceBuffer<real> h2g1x_, h2g1y_, h2g1z_, h1g2x_, h1g2y_, h1g2z_, bx_, by_, bz_;
    DeviceBuffer<real> s1_, s2_;
    DeviceBuffer<unsigned long long> counters_;
};

struct OracleMetrics {
    double s1_rms{};
    double s2_rms{};
    double s1_boundary{};
    double s2_boundary{};
    double worst{};
    bool finite{};
};

[[nodiscard]] OracleMetrics compare_to_oracle(const ref::Grid& grid,
                                              const NonlinearSourceGpuFixture::SourcesResult& actual,
                                              const ref::NonlinearSourceFields& expected) {
    OracleMetrics metrics;
    const double s1_scale = rms(expected.s1);
    const double s2_scale = rms(expected.s2);
    metrics.s1_rms = normalized(rms_difference(actual.s1, expected.s1), s1_scale);
    metrics.s2_rms = normalized(rms_difference(actual.s2, expected.s2), s2_scale);
    metrics.s1_boundary = normalized(periodic_boundary_linf(grid, actual.s1, expected.s1), s1_scale);
    metrics.s2_boundary = normalized(periodic_boundary_linf(grid, actual.s2, expected.s2), s2_scale);
    metrics.finite = std::isfinite(metrics.s1_rms) && std::isfinite(metrics.s2_rms) &&
                     std::isfinite(metrics.s1_boundary) && std::isfinite(metrics.s2_boundary);
    metrics.worst = std::max({metrics.s1_rms, metrics.s2_rms, metrics.s1_boundary, metrics.s2_boundary});
    return metrics;
}

[[nodiscard]] ref::NonlinearSourceFields discrete_source_oracle(
    const ref::TotalGradientFixture& fixture, const ref::NonlinearSourceReferenceConfig& config) {
    const auto g1 = ref::centered_total_gradient_oracle(fixture.grid, fixture.psi1_fluctuation,
                                                        fixture.psi1_affine_gradient);
    const auto g2 = ref::centered_total_gradient_oracle(fixture.grid, fixture.psi2_fluctuation,
                                                        fixture.psi2_affine_gradient);
    const auto hvb = ref::centered_hessian_vector_b_oracle(fixture.grid, fixture.psi1_fluctuation,
                                                            fixture.psi2_fluctuation, g1, g2);
    return ref::centered_nonlinear_source_oracle(fixture.grid, g1, g2, hvb.b, config);
}

[[nodiscard]] CaseResult case_nonlinear_sources_gpu_oracle() {
    const auto fixture = ref::make_total_gradient_fixture(16);
    NonlinearSourceGpuFixture gpu(fixture);
    gpu.compute_gradients_and_b(fixture);
    const ref::NonlinearSourceReferenceConfig ref_config{1.0e-2, 1.0};
    const auto actual = gpu.run_sources(production_config(ref_config.epsilon, ref_config.v_rms));
    const auto expected = discrete_source_oracle(fixture, ref_config);
    const auto metrics = compare_to_oracle(fixture.grid, actual, expected);
    std::cout << std::setprecision(16) << "nonlinear_sources_gpu_oracle s1_rms=" << metrics.s1_rms
              << " s2_rms=" << metrics.s2_rms << " s1_boundary_linf=" << metrics.s1_boundary
              << " s2_boundary_linf=" << metrics.s2_boundary << '\n';
    return {metrics.finite && metrics.worst <= kDiscreteOracleTolerance, "nonlinear_sources_gpu_oracle",
            "gpu-vs-independent-long-double-cpu", grid_description(fixture.grid), metrics.worst, 0.0,
            "n/a", "n/a", "S1,S2 normalized RMS and periodic-boundary Linf <=5e-11"};
}

struct MaskedMetrics {
    double l2{};
    double linf{};
    double kept_fraction{};
};

[[nodiscard]] MaskedMetrics masked_metrics(const std::vector<bool>& keep,
                                           const std::vector<real>& actual,
                                           const std::vector<double>& expected) {
    long double error_sum = 0.0L;
    long double expected_sum = 0.0L;
    double max_abs = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < keep.size(); ++i) {
        if (!keep[i]) continue;
        const double diff = static_cast<double>(actual[i]) - expected[i];
        error_sum += static_cast<long double>(diff) * diff;
        expected_sum += static_cast<long double>(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(diff));
        ++count;
    }
    MaskedMetrics metrics;
    metrics.kept_fraction = static_cast<double>(count) / static_cast<double>(keep.size());
    if (count == 0) {
        metrics.l2 = std::numeric_limits<double>::infinity();
        metrics.linf = std::numeric_limits<double>::infinity();
        return metrics;
    }
    const double scale = std::max(std::sqrt(static_cast<double>(expected_sum / count)), 1.0);
    metrics.l2 = std::sqrt(static_cast<double>(error_sum / count)) / scale;
    metrics.linf = max_abs / scale;
    return metrics;
}

struct LevelMetrics {
    double l2_s1{}, l2_s2{}, linf_s1{}, linf_s2{}, kept_fraction{};
};

[[nodiscard]] LevelMetrics evaluate_level(std::size_t cells_per_axis, double epsilon, double v_rms,
                                          bool apply_mask) {
    const auto fixture = ref::make_total_gradient_fixture(cells_per_axis);
    NonlinearSourceGpuFixture gpu(fixture);
    gpu.compute_gradients_and_b(fixture);
    const auto actual = gpu.run_sources(production_config(epsilon, v_rms));
    const auto expected = ref::analytic_nonlinear_source_reference(fixture, {epsilon, v_rms});
    std::vector<bool> keep(fixture.grid.cell_count(), true);
    if (apply_mask) {
        for (std::size_t i = 0; i < keep.size(); ++i) {
            const double c_sq = expected.c.x[i] * expected.c.x[i] + expected.c.y[i] * expected.c.y[i] +
                                expected.c.z[i] * expected.c.z[i];
            keep[i] = c_sq >= kMaskThreshold;
        }
    }
    const auto m1 = masked_metrics(keep, actual.s1, expected.s1);
    const auto m2 = masked_metrics(keep, actual.s2, expected.s2);
    return {m1.l2, m2.l2, m1.linf, m2.linf, m1.kept_fraction};
}

[[nodiscard]] CaseResult case_nonlinear_sources_smooth_order() {
    const std::array<std::size_t, 3> levels{16, 32, 64};
    const std::array<double, 3> spacings{1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0};
    std::array<LevelMetrics, 3> metrics{};
    for (std::size_t level = 0; level < levels.size(); ++level) {
        metrics[level] = evaluate_level(levels[level], 1.0e-2, 1.0, /*apply_mask=*/true);
        std::cout << std::setprecision(16) << "nonlinear_sources_smooth_order level=" << levels[level]
                  << " kept_fraction=" << metrics[level].kept_fraction
                  << " l2_s1=" << metrics[level].l2_s1 << " l2_s2=" << metrics[level].l2_s2
                  << " linf_s1=" << metrics[level].linf_s1 << " linf_s2=" << metrics[level].linf_s2
                  << '\n';
    }
    const auto order_s1_first = ref::observed_order(metrics[0].l2_s1, metrics[1].l2_s1, spacings[0], spacings[1]);
    const auto order_s1_second = ref::observed_order(metrics[1].l2_s1, metrics[2].l2_s1, spacings[1], spacings[2]);
    const auto order_s2_first = ref::observed_order(metrics[0].l2_s2, metrics[1].l2_s2, spacings[0], spacings[1]);
    const auto order_s2_second = ref::observed_order(metrics[1].l2_s2, metrics[2].l2_s2, spacings[1], spacings[2]);
    const bool linf_decreasing = metrics[1].linf_s1 < metrics[0].linf_s1 && metrics[2].linf_s1 < metrics[1].linf_s1 &&
                                 metrics[1].linf_s2 < metrics[0].linf_s2 && metrics[2].linf_s2 < metrics[1].linf_s2;
    const bool mask_ok = metrics[0].kept_fraction >= kMinKeptFraction &&
                         metrics[1].kept_fraction >= kMinKeptFraction &&
                         metrics[2].kept_fraction >= kMinKeptFraction;
    std::cout << std::setprecision(16)
              << "nonlinear_sources_smooth_order order_s1_16_32=" << (order_s1_first.valid() ? order_s1_first.value : -1.0)
              << " order_s1_32_64=" << (order_s1_second.valid() ? order_s1_second.value : -1.0)
              << " order_s2_16_32=" << (order_s2_first.valid() ? order_s2_first.value : -1.0)
              << " order_s2_32_64=" << (order_s2_second.valid() ? order_s2_second.value : -1.0)
              << " linf_strictly_decreases=" << (linf_decreasing ? "true" : "false")
              << " mask_kept_ok=" << (mask_ok ? "true" : "false") << '\n';
    const bool pass = mask_ok && order_s1_first.valid() && order_s1_second.valid() && order_s2_first.valid() &&
                      order_s2_second.valid() && order_s1_first.value >= kOrderThreshold &&
                      order_s1_second.value >= kOrderThreshold && order_s2_first.value >= kOrderThreshold &&
                      order_s2_second.value >= kOrderThreshold && linf_decreasing;
    const double minimum_order = std::min({order_s1_first.valid() ? order_s1_first.value : -1.0,
                                           order_s1_second.valid() ? order_s1_second.value : -1.0,
                                           order_s2_first.valid() ? order_s2_first.value : -1.0,
                                           order_s2_second.valid() ? order_s2_second.value : -1.0});
    return {pass, "nonlinear_sources_smooth_order", "gpu-continuum-masked-away-from-degeneracy",
            "16^3->32^3->64^3 unequal spacing, |c|^2>=10 mask", metrics[0].l2_s1, metrics[2].l2_s1,
            ">=1.8 twice", std::to_string(minimum_order),
            "masked L2 order>=1.8 both refinements and masked Linf strictly decreasing for S1,S2"};
}

[[nodiscard]] CaseResult case_nonlinear_sources_large_epsilon_unmasked() {
    const std::array<std::size_t, 3> levels{16, 32, 64};
    const std::array<double, 3> spacings{1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0};
    std::array<LevelMetrics, 3> metrics{};
    for (std::size_t level = 0; level < levels.size(); ++level) {
        metrics[level] = evaluate_level(levels[level], kLargeEpsilon, 1.0, /*apply_mask=*/false);
        std::cout << std::setprecision(16) << "nonlinear_sources_large_epsilon_unmasked level="
                  << levels[level] << " l2_s1=" << metrics[level].l2_s1
                  << " l2_s2=" << metrics[level].l2_s2 << " linf_s1=" << metrics[level].linf_s1
                  << " linf_s2=" << metrics[level].linf_s2 << '\n';
    }
    const auto order_s1_first = ref::observed_order(metrics[0].l2_s1, metrics[1].l2_s1, spacings[0], spacings[1]);
    const auto order_s1_second = ref::observed_order(metrics[1].l2_s1, metrics[2].l2_s1, spacings[1], spacings[2]);
    const auto order_s2_first = ref::observed_order(metrics[0].l2_s2, metrics[1].l2_s2, spacings[0], spacings[1]);
    const auto order_s2_second = ref::observed_order(metrics[1].l2_s2, metrics[2].l2_s2, spacings[1], spacings[2]);
    std::cout << std::setprecision(16)
              << "nonlinear_sources_large_epsilon_unmasked order_s1_16_32=" << (order_s1_first.valid() ? order_s1_first.value : -1.0)
              << " order_s1_32_64=" << (order_s1_second.valid() ? order_s1_second.value : -1.0)
              << " order_s2_16_32=" << (order_s2_first.valid() ? order_s2_first.value : -1.0)
              << " order_s2_32_64=" << (order_s2_second.valid() ? order_s2_second.value : -1.0) << '\n';
    const bool pass = order_s1_first.valid() && order_s1_second.valid() && order_s2_first.valid() &&
                      order_s2_second.valid() && order_s1_first.value >= kOrderThreshold &&
                      order_s1_second.value >= kOrderThreshold && order_s2_first.value >= kOrderThreshold &&
                      order_s2_second.value >= kOrderThreshold;
    const double minimum_order = std::min({order_s1_first.valid() ? order_s1_first.value : -1.0,
                                           order_s1_second.valid() ? order_s1_second.value : -1.0,
                                           order_s2_first.valid() ? order_s2_first.value : -1.0,
                                           order_s2_second.valid() ? order_s2_second.value : -1.0});
    return {pass, "nonlinear_sources_large_epsilon_unmasked", "gpu-continuum-regularization-dominated",
            "16^3->32^3->64^3 unequal spacing, epsilon=5, no mask", metrics[0].l2_s1, metrics[2].l2_s1,
            ">=1.8 twice", std::to_string(minimum_order), "unmasked L2 order>=1.8 both refinements for S1,S2"};
}

[[nodiscard]] CaseResult case_nonlinear_sources_pure_affine_zero() {
    const auto fixture = ref::make_pure_affine_total_gradient_fixture(16);
    NonlinearSourceGpuFixture gpu(fixture);
    gpu.compute_gradients_and_b(fixture);
    const auto actual = gpu.run_sources(production_config(1.0e-2, 1.0));
    double max_abs = 0.0;
    bool finite = true;
    for (const real value : actual.s1) {
        finite = finite && std::isfinite(static_cast<double>(value));
        max_abs = std::max(max_abs, std::abs(static_cast<double>(value)));
    }
    for (const real value : actual.s2) {
        finite = finite && std::isfinite(static_cast<double>(value));
        max_abs = std::max(max_abs, std::abs(static_cast<double>(value)));
    }
    const double tolerance = 16.0 * std::numeric_limits<real>::epsilon();
    const bool counters_zero = actual.counters.size() == 2 && actual.counters[0] == 0 && actual.counters[1] == 0;
    std::cout << std::setprecision(16) << "nonlinear_sources_pure_affine_zero max_abs=" << max_abs
              << " threshold=" << tolerance << " nonfinite_s1=" << actual.counters[0]
              << " nonfinite_s2=" << actual.counters[1] << '\n';
    return {finite && max_abs <= tolerance && counters_zero, "nonlinear_sources_pure_affine_zero",
            "gpu-analytic-zero-by-construction", grid_description(fixture.grid), max_abs, 0.0, "roundoff",
            "n/a", "B=0 identically => |S1|,|S2|<=16 epsilon and zero nonfinite counters"};
}

[[nodiscard]] CaseResult case_nonlinear_sources_epsilon_explicitness() {
    const auto fixture = ref::make_total_gradient_fixture(16);
    NonlinearSourceGpuFixture gpu(fixture);
    gpu.compute_gradients_and_b(fixture);
    const auto small = gpu.run_sources(production_config(1.0e-2, 1.0));
    const auto tiny = gpu.run_sources(production_config(1.0e-3, 1.0));
    const auto expected_small = discrete_source_oracle(fixture, {1.0e-2, 1.0});
    const auto expected_tiny = discrete_source_oracle(fixture, {1.0e-3, 1.0});
    const auto small_metrics = compare_to_oracle(fixture.grid, small, expected_small);
    const auto tiny_metrics = compare_to_oracle(fixture.grid, tiny, expected_tiny);
    double max_abs_diff = 0.0;
    for (std::size_t i = 0; i < small.s1.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::abs(static_cast<double>(small.s1[i]) - static_cast<double>(tiny.s1[i])));
    }
    std::cout << std::setprecision(16) << "nonlinear_sources_epsilon_explicitness epsilon=1e-2 worst="
              << small_metrics.worst << " epsilon=1e-3 worst=" << tiny_metrics.worst
              << " max_abs_diff_s1=" << max_abs_diff << '\n';
    const bool pass = small_metrics.finite && tiny_metrics.finite &&
                      small_metrics.worst <= kDiscreteOracleTolerance &&
                      tiny_metrics.worst <= kDiscreteOracleTolerance && max_abs_diff > 0.0;
    return {pass, "nonlinear_sources_epsilon_explicitness", "gpu-vs-independent-long-double-cpu-two-epsilons",
            grid_description(fixture.grid), small_metrics.worst, tiny_metrics.worst, "n/a", "n/a",
            "both epsilons normalized RMS<=5e-11 vs matching oracle epsilon; GPU S1 fields differ"};
}

[[nodiscard]] CaseResult case_nonlinear_sources_count_agreement() {
    const auto fixture = ref::make_near_degenerate_total_gradient_fixture(16, 0.37, 1.0e-4);
    NonlinearSourceGpuFixture gpu(fixture);
    gpu.compute_gradients_and_b(fixture);
    const std::vector<double> thresholds{1.0e-3, 1.0e-1};
    const auto config = production_config(1.0e-2, 1.0, thresholds);
    const auto discrete = gpu.download_gradients_and_b();

    const auto mirror = ref::double_precision_nonlinear_source_mirror(discrete.g1, discrete.g2, discrete.b,
                                                                       1.0e-2, 1.0, thresholds);
    const auto gpu_result = gpu.run_sources(config);
    const bool degeneracy_ok = gpu_result.counters.size() == 4 &&
                               gpu_result.counters[2] == mirror.degenerate_counts[0] &&
                               gpu_result.counters[3] == mirror.degenerate_counts[1];
    const bool separation_ok = mirror.degenerate_separation[0] > 1.0e-10 && mirror.degenerate_separation[1] > 1.0e-10;
    std::cout << std::setprecision(16) << "nonlinear_sources_count_agreement degeneracy gpu[2]="
              << gpu_result.counters[2] << " cpu[0]=" << mirror.degenerate_counts[0]
              << " gpu[3]=" << gpu_result.counters[3] << " cpu[1]=" << mirror.degenerate_counts[1]
              << " separation0=" << mirror.degenerate_separation[0]
              << " separation1=" << mirror.degenerate_separation[1] << '\n';

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<ref::NonfiniteInjection> injections{
        {7, {nan, nan, nan}}, {4095, {inf, inf, inf}}, {2048, {nan, nan, nan}}};
    const auto injected_g1 = ref::inject_nonfinite_values(discrete.g1, injections);
    gpu.upload_g1(injected_g1);
    const auto gpu_result_injected = gpu.run_sources(config);
    const auto mirror_injected = ref::double_precision_nonlinear_source_mirror(injected_g1, discrete.g2,
                                                                                discrete.b, 1.0e-2, 1.0,
                                                                                thresholds);
    const bool nonfinite_ok = gpu_result_injected.counters[0] == mirror_injected.nonfinite_s1_count &&
                              gpu_result_injected.counters[1] == mirror_injected.nonfinite_s2_count &&
                              mirror_injected.nonfinite_s1_count > 0 && mirror_injected.nonfinite_s2_count > 0;
    std::cout << std::setprecision(16) << "nonlinear_sources_count_agreement nonfinite gpu_s1="
              << gpu_result_injected.counters[0] << " cpu_s1=" << mirror_injected.nonfinite_s1_count
              << " gpu_s2=" << gpu_result_injected.counters[1]
              << " cpu_s2=" << mirror_injected.nonfinite_s2_count << '\n';

    const bool pass = degeneracy_ok && separation_ok && nonfinite_ok;
    return {pass, "nonlinear_sources_count_agreement", "gpu-vs-cpu-exact-diagnostic-counts",
            grid_description(fixture.grid), 0.0, 0.0, "exact", "n/a",
            "degeneracy counters match exactly with separation>1e-10; nonfinite counters match exactly and are >0"};
}

[[nodiscard]] CaseResult case_nonlinear_sources_error_contract() {
    const auto fixture = ref::make_total_gradient_fixture(16);
    const auto grid = production_grid(fixture.grid);
    const auto n = fixture.grid.cell_count();
    CudaContext context(0);
    DeviceBuffer<real> u1(n), u2(n);
    DeviceBuffer<real> p1x(n), p1y(n), p1z(n), p2x(n), p2y(n), p2z(n);
    DeviceBuffer<real> h2g1x(n), h2g1y(n), h2g1z(n), h1g2x(n), h1g2y(n), h1g2z(n);
    DeviceBuffer<real> bx(n), by(n), bz(n);
    DeviceBuffer<real> s1(n), s2(n);
    DeviceBuffer<unsigned long long> counters(3);

    const std::vector<real> host_u1(fixture.psi1_fluctuation.begin(), fixture.psi1_fluctuation.end());
    const std::vector<real> host_u2(fixture.psi2_fluctuation.begin(), fixture.psi2_fluctuation.end());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u1.data(), host_u1.data(), n * sizeof(real),
                                           cudaMemcpyHostToDevice, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u2.data(), host_u2.data(), n * sizeof(real),
                                           cudaMemcpyHostToDevice, context.cuda_stream()));
    enqueue_total_streamfunction_gradients(context, grid, {u1.span(), u2.span()}, production_gauge(fixture),
                                           {p1x.span(), p1y.span(), p1z.span(), p2x.span(), p2y.span(), p2z.span()});
    enqueue_streamfunction_hessian_vector_b(
        context, grid, {u1.span(), u2.span()},
        {p1x.span(), p1y.span(), p1z.span(), p2x.span(), p2y.span(), p2z.span()},
        {h2g1x.span(), h2g1y.span(), h2g1z.span(), h1g2x.span(), h1g2y.span(), h1g2z.span(), bx.span(),
         by.span(), bz.span()});

    const TotalStreamfunctionGradientView gradients{p1x.span(), p1y.span(), p1z.span(),
                                                     p2x.span(), p2y.span(), p2z.span()};
    const StreamfunctionBFieldView bview{bx.span(), by.span(), bz.span()};
    NonlinearSourceConfig base_config{};
    base_config.epsilon = real{1.0e-2};
    base_config.v_rms = real{1};
    base_config.num_degeneracy_thresholds = 1;
    base_config.degeneracy_thresholds[0] = real{0.1};
    const NonlinearSourceOutput output{s1.span(), s2.span()};
    const NonlinearSourceCounters countersView{DeviceSpan<unsigned long long>(counters.data(), 3)};

    const auto invoke = [&](const Grid3D& candidate_grid, const TotalStreamfunctionGradientView& candidate_gradients,
                            const StreamfunctionBFieldView& candidate_b, const NonlinearSourceConfig& candidate_config,
                            const NonlinearSourceOutput& candidate_output,
                            const NonlinearSourceCounters& candidate_counters) {
        enqueue_streamfunction_nonlinear_sources(context, candidate_grid, candidate_gradients, candidate_b,
                                                 candidate_config, candidate_output, candidate_counters);
    };

    bool pass = true;
    std::size_t checks = 0;
    const auto require_invalid = [&](const char* name, const auto& callable) {
        ++checks;
        pass = rejects_with_invalid_argument(name, callable) && pass;
    };

    require_invalid("extent_zero", [&] {
        invoke({0, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz}, gradients, bview, base_config, output, countersView);
    });
    require_invalid("extent_negative", [&] {
        invoke({-1, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz}, gradients, bview, base_config, output, countersView);
    });
    for (const auto [axis, value, name] :
         std::array<std::tuple<int, real, const char*>, 12>{{
             {0, real{0}, "dx_zero"}, {0, real{-1}, "dx_negative"},
             {0, std::numeric_limits<real>::quiet_NaN(), "dx_nan"},
             {0, std::numeric_limits<real>::infinity(), "dx_inf"},
             {1, real{0}, "dy_zero"}, {1, real{-1}, "dy_negative"},
             {1, std::numeric_limits<real>::quiet_NaN(), "dy_nan"},
             {1, std::numeric_limits<real>::infinity(), "dy_inf"},
             {2, real{0}, "dz_zero"}, {2, real{-1}, "dz_negative"},
             {2, std::numeric_limits<real>::quiet_NaN(), "dz_nan"},
             {2, std::numeric_limits<real>::infinity(), "dz_inf"}}}) {
        auto invalid = grid;
        if (axis == 0) invalid.dx = value;
        if (axis == 1) invalid.dy = value;
        if (axis == 2) invalid.dz = value;
        require_invalid(name, [&] { invoke(invalid, gradients, bview, base_config, output, countersView); });
    }

    require_invalid("epsilon_negative", [&] {
        auto invalid = base_config; invalid.epsilon = real{-1};
        invoke(grid, gradients, bview, invalid, output, countersView);
    });
    require_invalid("epsilon_nan", [&] {
        auto invalid = base_config; invalid.epsilon = std::numeric_limits<real>::quiet_NaN();
        invoke(grid, gradients, bview, invalid, output, countersView);
    });
    require_invalid("epsilon_inf", [&] {
        auto invalid = base_config; invalid.epsilon = std::numeric_limits<real>::infinity();
        invoke(grid, gradients, bview, invalid, output, countersView);
    });
    require_invalid("v_rms_zero", [&] {
        auto invalid = base_config; invalid.v_rms = real{0};
        invoke(grid, gradients, bview, invalid, output, countersView);
    });
    require_invalid("v_rms_negative", [&] {
        auto invalid = base_config; invalid.v_rms = real{-1};
        invoke(grid, gradients, bview, invalid, output, countersView);
    });
    require_invalid("v_rms_nan", [&] {
        auto invalid = base_config; invalid.v_rms = std::numeric_limits<real>::quiet_NaN();
        invoke(grid, gradients, bview, invalid, output, countersView);
    });
    require_invalid("v_rms_inf", [&] {
        auto invalid = base_config; invalid.v_rms = std::numeric_limits<real>::infinity();
        invoke(grid, gradients, bview, invalid, output, countersView);
    });
    require_invalid("num_thresholds_negative", [&] {
        auto invalid = base_config; invalid.num_degeneracy_thresholds = -1;
        invoke(grid, gradients, bview, invalid, output, countersView);
    });
    require_invalid("num_thresholds_too_large", [&] {
        auto invalid = base_config; invalid.num_degeneracy_thresholds = 5;
        invoke(grid, gradients, bview, invalid, output, countersView);
    });
    require_invalid("threshold_negative", [&] {
        auto invalid = base_config; invalid.degeneracy_thresholds[0] = real{-1};
        invoke(grid, gradients, bview, invalid, output, countersView);
    });
    require_invalid("threshold_nan", [&] {
        auto invalid = base_config; invalid.degeneracy_thresholds[0] = std::numeric_limits<real>::quiet_NaN();
        invoke(grid, gradients, bview, invalid, output, countersView);
    });

    const std::array<const char*, 6> gradient_names{"psi1_x", "psi1_y", "psi1_z", "psi2_x", "psi2_y", "psi2_z"};
    const std::array<DeviceSpan<const real>, 6> gradient_spans{gradients.psi1_x, gradients.psi1_y, gradients.psi1_z,
                                                                gradients.psi2_x, gradients.psi2_y, gradients.psi2_z};
    for (std::size_t component = 0; component < gradient_spans.size(); ++component) {
        const auto reject_gradient = [&](const char* kind, DeviceSpan<const real> replacement) {
            auto invalid = gradients;
            switch (component) {
                case 0: invalid.psi1_x = replacement; break;
                case 1: invalid.psi1_y = replacement; break;
                case 2: invalid.psi1_z = replacement; break;
                case 3: invalid.psi2_x = replacement; break;
                case 4: invalid.psi2_y = replacement; break;
                default: invalid.psi2_z = replacement; break;
            }
            require_invalid((std::string("gradient_") + gradient_names[component] + '_' + kind).c_str(),
                            [&] { invoke(grid, invalid, bview, base_config, output, countersView); });
        };
        reject_gradient("null", DeviceSpan<const real>(nullptr, n));
        reject_gradient("short", DeviceSpan<const real>(gradient_spans[component].data(), n - 1));
        reject_gradient("long", DeviceSpan<const real>(gradient_spans[component].data(), n + 1));
    }

    const std::array<const char*, 3> b_names{"b_x", "b_y", "b_z"};
    const std::array<DeviceSpan<const real>, 3> b_spans{bview.b_x, bview.b_y, bview.b_z};
    for (std::size_t component = 0; component < b_spans.size(); ++component) {
        const auto reject_b = [&](const char* kind, DeviceSpan<const real> replacement) {
            auto invalid = bview;
            switch (component) {
                case 0: invalid.b_x = replacement; break;
                case 1: invalid.b_y = replacement; break;
                default: invalid.b_z = replacement; break;
            }
            require_invalid((std::string("b_") + b_names[component] + '_' + kind).c_str(),
                            [&] { invoke(grid, gradients, invalid, base_config, output, countersView); });
        };
        reject_b("null", DeviceSpan<const real>(nullptr, n));
        reject_b("short", DeviceSpan<const real>(b_spans[component].data(), n - 1));
        reject_b("long", DeviceSpan<const real>(b_spans[component].data(), n + 1));
    }

    const std::array<const char*, 2> output_names{"s1", "s2"};
    const std::array<DeviceSpan<real>, 2> output_spans{output.s1, output.s2};
    for (std::size_t component = 0; component < output_spans.size(); ++component) {
        const auto reject_output = [&](const char* kind, DeviceSpan<real> replacement) {
            auto invalid = output;
            if (component == 0) invalid.s1 = replacement;
            else invalid.s2 = replacement;
            require_invalid((std::string("output_") + output_names[component] + '_' + kind).c_str(),
                            [&] { invoke(grid, gradients, bview, base_config, invalid, countersView); });
        };
        reject_output("null", DeviceSpan<real>(nullptr, n));
        reject_output("short", DeviceSpan<real>(output_spans[component].data(), n - 1));
        reject_output("long", DeviceSpan<real>(output_spans[component].data(), n + 1));
    }

    require_invalid("counters_null", [&] {
        invoke(grid, gradients, bview, base_config, output, {DeviceSpan<unsigned long long>(nullptr, 3)});
    });
    require_invalid("counters_short", [&] {
        invoke(grid, gradients, bview, base_config, output, {DeviceSpan<unsigned long long>(counters.data(), 2)});
    });
    require_invalid("counters_long", [&] {
        invoke(grid, gradients, bview, base_config, output, {DeviceSpan<unsigned long long>(counters.data(), 4)});
    });

    require_invalid("output_s1_input_overlap", [&] {
        auto invalid = output; invalid.s1 = p1x.span();
        invoke(grid, gradients, bview, base_config, invalid, countersView);
    });
    require_invalid("output_s2_s1_exact_overlap", [&] {
        auto invalid = output; invalid.s2 = s1.span();
        invoke(grid, gradients, bview, base_config, invalid, countersView);
    });
    require_invalid("output_s2_s1_partial_overlap", [&] {
        auto invalid = output; invalid.s2 = DeviceSpan<real>(s1.data() + 1, n);
        invoke(grid, gradients, bview, base_config, invalid, countersView);
    });
    require_invalid("counters_s1_overlap", [&] {
        invoke(grid, gradients, bview, base_config, output,
               {DeviceSpan<unsigned long long>(reinterpret_cast<unsigned long long*>(s1.data()), 3)});
    });
    require_invalid("counters_input_overlap", [&] {
        invoke(grid, gradients, bview, base_config, output,
               {DeviceSpan<unsigned long long>(reinterpret_cast<unsigned long long*>(p1x.data()), 3)});
    });

    const TotalStreamfunctionGradientView all_read_only_inputs{u1.span(), u1.span(), u1.span(),
                                                                u1.span(), u1.span(), u1.span()};
    const StreamfunctionBFieldView all_read_only_b{u1.span(), u1.span(), u1.span()};
    invoke(grid, all_read_only_inputs, all_read_only_b, base_config, output, countersView);
    std::vector<real> accepted_s1(n), accepted_s2(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(accepted_s1.data(), s1.data(), n * sizeof(real),
                                           cudaMemcpyDeviceToHost, context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(accepted_s2.data(), s2.data(), n * sizeof(real),
                                           cudaMemcpyDeviceToHost, context.cuda_stream()));
    context.synchronize();
    bool finite = true;
    double accepted_linf = 0.0;
    for (const real value : accepted_s1) {
        finite = finite && std::isfinite(static_cast<double>(value));
        accepted_linf = std::max(accepted_linf, std::abs(static_cast<double>(value)));
    }
    for (const real value : accepted_s2) {
        finite = finite && std::isfinite(static_cast<double>(value));
        accepted_linf = std::max(accepted_linf, std::abs(static_cast<double>(value)));
    }
    ++checks;
    std::cout << std::setprecision(16) << "nonlinear_sources_contract name=input_input_overlap exception=accepted finite="
              << (finite ? "true" : "false") << " linf=" << accepted_linf << '\n';
    pass = pass && finite;

    return {pass, "nonlinear_sources_error_contract", "host-validation-and-gpu-acceptance",
            grid_description(fixture.grid), accepted_linf, 0.0, "invalid_argument + 1 accepted",
            std::to_string(checks),
            "invalid extents/spacings/config/spans/output-counter aliases reject; overlapping read-only inputs accepted with finite output"};
}

enum class SourceMutant { pairing_swap, cross_order_flip, b_sign_flip };

[[nodiscard]] std::array<std::vector<double>, 2> nonlinear_source_mutant(
    const ref::VectorField& g1, const ref::VectorField& g2, const ref::VectorField& b, double epsilon,
    double v_rms, SourceMutant mutant) {
    const std::size_t n = g1.x.size();
    std::array<std::vector<double>, 2> result{std::vector<double>(n), std::vector<double>(n)};
    const double regularization = epsilon * v_rms;
    const double regularization_sq = regularization * regularization;
    for (std::size_t i = 0; i < n; ++i) {
        ref::Vec3 gg1{g1.x[i], g1.y[i], g1.z[i]};
        ref::Vec3 gg2{g2.x[i], g2.y[i], g2.z[i]};
        ref::Vec3 bb{b.x[i], b.y[i], b.z[i]};
        if (mutant == SourceMutant::b_sign_flip) { bb.x = -bb.x; bb.y = -bb.y; bb.z = -bb.z; }
        const ref::Vec3 c = (mutant == SourceMutant::cross_order_flip) ? ref::cross(gg2, gg1) : ref::cross(gg1, gg2);
        const double c_sq = dot(c, c);
        const double d = c_sq + regularization_sq;
        const ref::Vec3 bxg1 = ref::cross(bb, gg1);
        const ref::Vec3 bxg2 = ref::cross(bb, gg2);
        if (mutant == SourceMutant::pairing_swap) {
            result[0][i] = dot(bxg2, c) / d;
            result[1][i] = dot(bxg1, c) / d;
        } else {
            result[0][i] = dot(bxg1, c) / d;
            result[1][i] = dot(bxg2, c) / d;
        }
    }
    return result;
}

[[nodiscard]] double normalized_two_field_rms(const std::vector<double>& actual_s1,
                                              const std::vector<double>& actual_s2,
                                              const std::vector<double>& expected_s1,
                                              const std::vector<double>& expected_s2) {
    long double err = 0.0L, exp2 = 0.0L;
    std::size_t count = 0;
    for (std::size_t i = 0; i < actual_s1.size(); ++i) {
        const double d = actual_s1[i] - expected_s1[i];
        err += static_cast<long double>(d) * d;
        exp2 += static_cast<long double>(expected_s1[i]) * expected_s1[i];
        ++count;
    }
    for (std::size_t i = 0; i < actual_s2.size(); ++i) {
        const double d = actual_s2[i] - expected_s2[i];
        err += static_cast<long double>(d) * d;
        exp2 += static_cast<long double>(expected_s2[i]) * expected_s2[i];
        ++count;
    }
    return std::sqrt(static_cast<double>(err / count)) / std::max(std::sqrt(static_cast<double>(exp2 / count)), 1.0);
}

[[nodiscard]] CaseResult case_nonlinear_sources_mutation_sensitivity() {
    const auto fixture = ref::make_total_gradient_fixture(16);
    const auto g1 = ref::centered_total_gradient_oracle(fixture.grid, fixture.psi1_fluctuation, fixture.psi1_affine_gradient);
    const auto g2 = ref::centered_total_gradient_oracle(fixture.grid, fixture.psi2_fluctuation, fixture.psi2_affine_gradient);
    const auto hvb = ref::centered_hessian_vector_b_oracle(fixture.grid, fixture.psi1_fluctuation, fixture.psi2_fluctuation, g1, g2);
    const auto correct = ref::centered_nonlinear_source_oracle(fixture.grid, g1, g2, hvb.b, {1.0e-2, 1.0});

    const auto pairing = nonlinear_source_mutant(g1, g2, hvb.b, 1.0e-2, 1.0, SourceMutant::pairing_swap);
    const double pairing_rms = normalized_two_field_rms(pairing[0], pairing[1], correct.s1, correct.s2);

    const auto flipped = nonlinear_source_mutant(g1, g2, hvb.b, 1.0e-2, 1.0, SourceMutant::cross_order_flip);
    const double flip_rms = normalized_two_field_rms(flipped[0], flipped[1], correct.s1, correct.s2);

    const auto degenerate_fixture = ref::make_near_degenerate_total_gradient_fixture(16, 0.37, 1.0e-4);
    const auto dg1 = ref::centered_total_gradient_oracle(degenerate_fixture.grid, degenerate_fixture.psi1_fluctuation, degenerate_fixture.psi1_affine_gradient);
    const auto dg2 = ref::centered_total_gradient_oracle(degenerate_fixture.grid, degenerate_fixture.psi2_fluctuation, degenerate_fixture.psi2_affine_gradient);
    const auto dhvb = ref::centered_hessian_vector_b_oracle(degenerate_fixture.grid, degenerate_fixture.psi1_fluctuation, degenerate_fixture.psi2_fluctuation, dg1, dg2);
    const auto degenerate_correct = ref::centered_nonlinear_source_oracle(degenerate_fixture.grid, dg1, dg2, dhvb.b, {1.0e-2, 1.0});
    const auto unregularized = ref::centered_nonlinear_source_oracle(degenerate_fixture.grid, dg1, dg2, dhvb.b, {0.0, 1.0});
    const double unregularized_rms = normalized_two_field_rms(unregularized.s1, unregularized.s2, degenerate_correct.s1, degenerate_correct.s2);

    const auto b_flip = nonlinear_source_mutant(g1, g2, hvb.b, 1.0e-2, 1.0, SourceMutant::b_sign_flip);
    const double b_flip_rms = normalized_two_field_rms(b_flip[0], b_flip[1], correct.s1, correct.s2);

    const bool finite = std::isfinite(pairing_rms) && std::isfinite(flip_rms) && std::isfinite(unregularized_rms) &&
                        std::isfinite(b_flip_rms);
    std::cout << std::setprecision(16) << "nonlinear_sources_mutant name=pairing_swap normalized_rms=" << pairing_rms
              << " threshold=" << kMutationThreshold << '\n';
    std::cout << std::setprecision(16) << "nonlinear_sources_mutant name=cross_order_flip normalized_rms=" << flip_rms
              << " threshold=" << kMutationThreshold << '\n';
    std::cout << std::setprecision(16) << "nonlinear_sources_mutant name=unregularized_denominator normalized_rms="
              << unregularized_rms << " threshold=" << kMutationThreshold << '\n';
    std::cout << std::setprecision(16) << "nonlinear_sources_mutant name=b_sign_flip normalized_rms=" << b_flip_rms
              << " threshold=" << kMutationThreshold << '\n';
    const bool pass = finite && pairing_rms > kMutationThreshold && flip_rms > kMutationThreshold &&
                      unregularized_rms > kMutationThreshold && b_flip_rms > kMutationThreshold;
    return {pass, "nonlinear_sources_mutation_sensitivity", "test-only-mutants-vs-independent-oracle",
            grid_description(fixture.grid),
            std::min({pairing_rms, flip_rms, unregularized_rms, b_flip_rms}),
            std::max({pairing_rms, flip_rms, unregularized_rms, b_flip_rms}), ">0.01 each",
            std::to_string(std::min({pairing_rms, flip_rms, unregularized_rms, b_flip_rms})),
            "pairing-swap, cross-order-flip, unregularized-denominator, and B-sign-flip mutants each exceed normalized-RMS threshold 1e-2"};
}

} // namespace

CaseRegistry nonlinear_sources_case_registry() {
    return {{"nonlinear_sources_gpu_oracle", case_nonlinear_sources_gpu_oracle},
            {"nonlinear_sources_smooth_order", case_nonlinear_sources_smooth_order},
            {"nonlinear_sources_large_epsilon_unmasked", case_nonlinear_sources_large_epsilon_unmasked},
            {"nonlinear_sources_pure_affine_zero", case_nonlinear_sources_pure_affine_zero},
            {"nonlinear_sources_epsilon_explicitness", case_nonlinear_sources_epsilon_explicitness},
            {"nonlinear_sources_count_agreement", case_nonlinear_sources_count_agreement},
            {"nonlinear_sources_error_contract", case_nonlinear_sources_error_contract},
            {"nonlinear_sources_mutation_sensitivity", case_nonlinear_sources_mutation_sensitivity}};
}

} // namespace macroflow3d::streamfunctions::test
