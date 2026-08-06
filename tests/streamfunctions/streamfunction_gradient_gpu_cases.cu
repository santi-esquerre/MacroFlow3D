#include "reference_operators.hpp"
#include "streamfunction_operator_test_cases.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/DifferentialOperators.cuh"
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
#include <vector>

namespace macroflow3d::streamfunctions::test {
namespace {
namespace ref = macroflow3d::streamfunctions::reference;

constexpr double kOracleTolerance = 1.0e-12;
constexpr double kOrderThreshold = 1.8;
constexpr std::array<const char*, 6> kNames{"psi1_x", "psi1_y", "psi1_z", "psi2_x", "psi2_y", "psi2_z"};

[[nodiscard]] Grid3D production_grid(const ref::Grid& grid) {
    return {static_cast<int>(grid.nx), static_cast<int>(grid.ny), static_cast<int>(grid.nz),
            static_cast<real>(grid.spacing.x), static_cast<real>(grid.spacing.y),
            static_cast<real>(grid.spacing.z)};
}
[[nodiscard]] AffineGauge production_gauge(const ref::TotalGradientFixture& fixture) {
    return {{static_cast<real>(fixture.psi1_affine_gradient.x), static_cast<real>(fixture.psi1_affine_gradient.y), static_cast<real>(fixture.psi1_affine_gradient.z)},
            {static_cast<real>(fixture.psi2_affine_gradient.x), static_cast<real>(fixture.psi2_affine_gradient.y), static_cast<real>(fixture.psi2_affine_gradient.z)}};
}
[[nodiscard]] std::string grid_description(const ref::Grid& grid) {
    std::ostringstream out;
    out << grid.nx << 'x' << grid.ny << 'x' << grid.nz << " h=(" << grid.spacing.x << ','
        << grid.spacing.y << ',' << grid.spacing.z << ')';
    return out.str();
}
[[nodiscard]] double rms(const std::vector<real>& values) {
    long double sum = 0.0L;
    for (const real value : values) sum += static_cast<long double>(value) * value;
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
[[nodiscard]] double linf_difference(const std::vector<real>& actual,
                                     const std::vector<double>& expected) {
    double maximum = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        maximum = std::max(maximum, std::abs(static_cast<double>(actual[i]) - expected[i]));
    }
    return maximum;
}
[[nodiscard]] double periodic_boundary_linf(const ref::Grid& grid, const std::vector<real>& actual,
                                            const std::vector<double>& expected) {
    double maximum = 0.0;
    for (std::size_t z = 0; z < grid.nz; ++z) for (std::size_t y = 0; y < grid.ny; ++y) for (std::size_t x = 0; x < grid.nx; ++x) {
        if (x != 0 && x + 1 != grid.nx && y != 0 && y + 1 != grid.ny && z != 0 && z + 1 != grid.nz) continue;
        const auto i = grid.index(x, y, z);
        maximum = std::max(maximum, std::abs(static_cast<double>(actual[i]) - expected[i]));
    }
    return maximum;
}
[[nodiscard]] double normalized(double value, double scale) { return value / std::max(scale, 1.0); }

struct GpuGradients { std::array<std::vector<real>, 6> values; };
class GradientFixture {
  public:
    explicit GradientFixture(const ref::TotalGradientFixture& source)
        : grid_(production_grid(source.grid)), context_(0), u1_(source.grid.cell_count()), u2_(source.grid.cell_count()),
          p1x_(source.grid.cell_count()), p1y_(source.grid.cell_count()), p1z_(source.grid.cell_count()),
          p2x_(source.grid.cell_count()), p2y_(source.grid.cell_count()), p2z_(source.grid.cell_count()) {}

    [[nodiscard]] GpuGradients run(const ref::TotalGradientFixture& source) {
        GpuGradients result;
        for (auto& component : result.values) component.resize(source.grid.cell_count());
        std::vector<real> h1(source.psi1_fluctuation.begin(), source.psi1_fluctuation.end());
        std::vector<real> h2(source.psi2_fluctuation.begin(), source.psi2_fluctuation.end());
        // H2D -> gradient kernel -> D2H all use this context's one stream;
        // this test fixture has exactly one explicit synchronization below.
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u1_.data(), h1.data(), h1.size() * sizeof(real), cudaMemcpyHostToDevice, context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u2_.data(), h2.data(), h2.size() * sizeof(real), cudaMemcpyHostToDevice, context_.cuda_stream()));
        enqueue_total_streamfunction_gradients(context_, grid_, {u1_.span(), u2_.span()}, production_gauge(source),
                                               {p1x_.span(), p1y_.span(), p1z_.span(), p2x_.span(), p2y_.span(), p2z_.span()});
        const std::array<DeviceBuffer<real>*, 6> buffers{&p1x_, &p1y_, &p1z_, &p2x_, &p2y_, &p2z_};
        for (std::size_t c = 0; c < buffers.size(); ++c) {
            MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.values[c].data(), buffers[c]->data(), result.values[c].size() * sizeof(real), cudaMemcpyDeviceToHost, context_.cuda_stream()));
        }
        context_.synchronize();
        return result;
    }
  private:
    Grid3D grid_; CudaContext context_; DeviceBuffer<real> u1_, u2_, p1x_, p1y_, p1z_, p2x_, p2y_, p2z_;
};

[[nodiscard]] std::array<std::vector<double>, 6> oracle(const ref::TotalGradientFixture& fixture) {
    const auto psi1 = ref::centered_total_gradient_oracle(fixture.grid, fixture.psi1_fluctuation, fixture.psi1_affine_gradient);
    const auto psi2 = ref::centered_total_gradient_oracle(fixture.grid, fixture.psi2_fluctuation, fixture.psi2_affine_gradient);
    return {psi1.x, psi1.y, psi1.z, psi2.x, psi2.y, psi2.z};
}
[[nodiscard]] std::array<std::vector<double>, 6> analytic(const ref::TotalGradientFixture& fixture) {
    std::array<std::vector<double>, 6> result;
    for (auto& component : result) component.resize(fixture.grid.cell_count());
    for (std::size_t z = 0; z < fixture.grid.nz; ++z) for (std::size_t y = 0; y < fixture.grid.ny; ++y) for (std::size_t x = 0; x < fixture.grid.nx; ++x) {
        const auto id = fixture.grid.index(x, y, z);
        const auto position = fixture.grid.cell_center(x, y, z);
        const auto one = ref::total_gradient_analytic(ref::GradientFixtureField::psi1, position, fixture.lengths, fixture.psi1_affine_gradient);
        const auto two = ref::total_gradient_analytic(ref::GradientFixtureField::psi2, position, fixture.lengths, fixture.psi2_affine_gradient);
        result[0][id] = one.x; result[1][id] = one.y; result[2][id] = one.z;
        result[3][id] = two.x; result[4][id] = two.y; result[5][id] = two.z;
    }
    return result;
}

[[nodiscard]] CaseResult case_gradient_pure_affine() {
    const auto fixture = ref::make_pure_affine_total_gradient_fixture(16);
    GradientFixture gpu(fixture); const auto actual = gpu.run(fixture); const auto expected = oracle(fixture);
    double worst_rms = 0.0, worst_linf = 0.0;
    for (std::size_t c = 0; c < kNames.size(); ++c) {
        const double component_rms = rms_difference(actual.values[c], expected[c]);
        const double component_linf = linf_difference(actual.values[c], expected[c]);
        worst_rms = std::max(worst_rms, component_rms); worst_linf = std::max(worst_linf, component_linf);
        std::cout << std::setprecision(16) << "gradient_affine component=" << kNames[c] << " rms=" << component_rms << " linf=" << component_linf << '\n';
    }
    const double tolerance = 16.0 * std::numeric_limits<real>::epsilon();
    return {worst_linf <= tolerance, "gradient_pure_affine", "gpu-production", grid_description(fixture.grid), worst_rms, worst_linf, "n/a", "n/a", "all six affine constants to <=16 epsilon"};
}

[[nodiscard]] CaseResult case_gradient_gpu_oracle() {
    const auto fixture = ref::make_total_gradient_fixture(16);
    GradientFixture gpu(fixture); const auto actual = gpu.run(fixture); const auto expected = oracle(fixture);
    double worst_global = 0.0, worst_boundary = 0.0;
    for (std::size_t c = 0; c < kNames.size(); ++c) {
        const double scale = rms(expected[c]);
        const double global = normalized(rms_difference(actual.values[c], expected[c]), scale);
        const double boundary = normalized(periodic_boundary_linf(fixture.grid, actual.values[c], expected[c]), scale);
        worst_global = std::max(worst_global, global); worst_boundary = std::max(worst_boundary, boundary);
        std::cout << std::setprecision(16) << "gradient_oracle component=" << kNames[c] << " normalized_rms=" << global << " boundary_linf=" << boundary << '\n';
    }
    return {worst_global <= kOracleTolerance && worst_boundary <= kOracleTolerance, "gradient_gpu_oracle", "gpu-vs-independent-long-double-cpu", grid_description(fixture.grid), worst_global, worst_boundary, "n/a", "n/a", "all six global normalized RMS and periodic-boundary Linf <=1e-12"};
}

[[nodiscard]] CaseResult case_gradient_smooth_order() {
    const std::array<std::size_t, 3> levels{16, 32, 64};
    std::array<std::array<double, 6>, 3> l2{}, linf{};
    for (std::size_t level = 0; level < levels.size(); ++level) {
        const auto fixture = ref::make_total_gradient_fixture(levels[level]);
        GradientFixture gpu(fixture); const auto actual = gpu.run(fixture); const auto expected = analytic(fixture);
        for (std::size_t c = 0; c < kNames.size(); ++c) {
            l2[level][c] = normalized(rms_difference(actual.values[c], expected[c]), rms(expected[c]));
            linf[level][c] = normalized(linf_difference(actual.values[c], expected[c]), rms(expected[c]));
        }
    }
    bool pass = true; double worst_l2 = 0.0, worst_fine_l2 = 0.0, minimum_order = 1.0e9;
    for (std::size_t c = 0; c < kNames.size(); ++c) {
        const auto first = ref::observed_order(l2[0][c], l2[1][c], 1.0 / 16.0, 1.0 / 32.0);
        const auto second = ref::observed_order(l2[1][c], l2[2][c], 1.0 / 32.0, 1.0 / 64.0);
        const bool decreasing = linf[1][c] < linf[0][c] && linf[2][c] < linf[1][c];
        pass = pass && first.valid() && second.valid() && first.value >= kOrderThreshold && second.value >= kOrderThreshold && decreasing;
        minimum_order = std::min({minimum_order, first.valid() ? first.value : -1.0, second.valid() ? second.value : -1.0});
        worst_l2 = std::max(worst_l2, l2[0][c]); worst_fine_l2 = std::max(worst_fine_l2, l2[2][c]);
        std::cout << std::setprecision(16) << "gradient_convergence component=" << kNames[c]
                  << " l2_16=" << l2[0][c] << " linf_16=" << linf[0][c]
                  << " l2_32=" << l2[1][c] << " linf_32=" << linf[1][c]
                  << " l2_64=" << l2[2][c] << " linf_64=" << linf[2][c]
                  << " order_16_32=" << (first.valid() ? first.value : -1.0)
                  << " order_32_64=" << (second.valid() ? second.value : -1.0)
                  << " linf_strictly_decreases=" << (decreasing ? "true" : "false") << '\n';
    }
    return {pass, "gradient_smooth_order", "gpu-continuum", "16^3->32^3->64^3 unequal spacing", worst_l2, worst_fine_l2, ">=1.8 twice", std::to_string(minimum_order), "each component: both L2 orders>=1.8 and Linf strictly decreases"};
}
}  // namespace

CaseRegistry streamfunction_gradient_case_registry() {
    return {{"gradient_pure_affine", case_gradient_pure_affine}, {"gradient_gpu_oracle", case_gradient_gpu_oracle}, {"gradient_smooth_order", case_gradient_smooth_order}};
}
}  // namespace macroflow3d::streamfunctions::test
