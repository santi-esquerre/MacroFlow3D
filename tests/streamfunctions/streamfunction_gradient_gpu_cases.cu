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
#include <stdexcept>
#include <tuple>
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

template <typename Callable>
[[nodiscard]] bool rejects_with_invalid_argument(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "gradient_contract name=" << name << " exception=none expected=std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument& error) {
        std::cout << "gradient_contract name=" << name << " exception=std::invalid_argument message=" << error.what() << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cout << "gradient_contract name=" << name << " exception=std::exception message=" << error.what()
                  << " expected=std::invalid_argument\n";
        return false;
    } catch (...) {
        std::cout << "gradient_contract name=" << name << " exception=non-standard expected=std::invalid_argument\n";
        return false;
    }
}

[[nodiscard]] TotalStreamfunctionGradientOutput output_spans(
    DeviceBuffer<real>& p1x, DeviceBuffer<real>& p1y, DeviceBuffer<real>& p1z,
    DeviceBuffer<real>& p2x, DeviceBuffer<real>& p2y, DeviceBuffer<real>& p2z) {
    return {p1x.span(), p1y.span(), p1z.span(), p2x.span(), p2y.span(), p2z.span()};
}

[[nodiscard]] std::array<std::vector<double>, 6> oracle(const ref::TotalGradientFixture& fixture);

[[nodiscard]] std::array<std::vector<double>, 6> mutation_omit_affine(
    const ref::TotalGradientFixture& fixture) {
    const auto zero = ref::Vec3{0.0, 0.0, 0.0};
    const auto psi1 = ref::centered_total_gradient_oracle(fixture.grid, fixture.psi1_fluctuation, zero);
    const auto psi2 = ref::centered_total_gradient_oracle(fixture.grid, fixture.psi2_fluctuation, zero);
    return {psi1.x, psi1.y, psi1.z, psi2.x, psi2.y, psi2.z};
}

[[nodiscard]] std::array<std::vector<double>, 6> mutation_dx_for_yz(
    const ref::TotalGradientFixture& fixture) {
    const auto correct = oracle(fixture);
    auto result = correct;
    const auto dx = fixture.grid.spacing.x;
    for (std::size_t z = 0; z < fixture.grid.nz; ++z) for (std::size_t y = 0; y < fixture.grid.ny; ++y) for (std::size_t x = 0; x < fixture.grid.nx; ++x) {
        const auto id = fixture.grid.index(x, y, z);
        const auto ym = fixture.grid.index(x, (y + fixture.grid.ny - 1) % fixture.grid.ny, z);
        const auto yp = fixture.grid.index(x, (y + 1) % fixture.grid.ny, z);
        const auto zm = fixture.grid.index(x, y, (z + fixture.grid.nz - 1) % fixture.grid.nz);
        const auto zp = fixture.grid.index(x, y, (z + 1) % fixture.grid.nz);
        result[1][id] = (fixture.psi1_fluctuation[yp] - fixture.psi1_fluctuation[ym]) / (2.0 * dx) + fixture.psi1_affine_gradient.y;
        result[2][id] = (fixture.psi1_fluctuation[zp] - fixture.psi1_fluctuation[zm]) / (2.0 * dx) + fixture.psi1_affine_gradient.z;
        result[4][id] = (fixture.psi2_fluctuation[yp] - fixture.psi2_fluctuation[ym]) / (2.0 * dx) + fixture.psi2_affine_gradient.y;
        result[5][id] = (fixture.psi2_fluctuation[zp] - fixture.psi2_fluctuation[zm]) / (2.0 * dx) + fixture.psi2_affine_gradient.z;
    }
    return result;
}

[[nodiscard]] std::array<std::vector<double>, 6> mutation_clamp_boundary(
    const ref::TotalGradientFixture& fixture) {
    std::array<std::vector<double>, 6> result;
    for (auto& component : result) component.resize(fixture.grid.cell_count());
    const auto clamp = [](std::ptrdiff_t index, std::size_t extent) {
        return static_cast<std::size_t>(std::max<std::ptrdiff_t>(0, std::min(index, static_cast<std::ptrdiff_t>(extent) - 1)));
    };
    for (std::size_t z = 0; z < fixture.grid.nz; ++z) for (std::size_t y = 0; y < fixture.grid.ny; ++y) for (std::size_t x = 0; x < fixture.grid.nx; ++x) {
        const auto id = fixture.grid.index(x, y, z);
        const auto xm = fixture.grid.index(clamp(static_cast<std::ptrdiff_t>(x) - 1, fixture.grid.nx), y, z);
        const auto xp = fixture.grid.index(clamp(static_cast<std::ptrdiff_t>(x) + 1, fixture.grid.nx), y, z);
        const auto ym = fixture.grid.index(x, clamp(static_cast<std::ptrdiff_t>(y) - 1, fixture.grid.ny), z);
        const auto yp = fixture.grid.index(x, clamp(static_cast<std::ptrdiff_t>(y) + 1, fixture.grid.ny), z);
        const auto zm = fixture.grid.index(x, y, clamp(static_cast<std::ptrdiff_t>(z) - 1, fixture.grid.nz));
        const auto zp = fixture.grid.index(x, y, clamp(static_cast<std::ptrdiff_t>(z) + 1, fixture.grid.nz));
        const auto assign = [&](std::size_t offset, const std::vector<double>& field, const ref::Vec3& affine) {
            result[offset][id] = (field[xp] - field[xm]) / (2.0 * fixture.grid.spacing.x) + affine.x;
            result[offset + 1][id] = (field[yp] - field[ym]) / (2.0 * fixture.grid.spacing.y) + affine.y;
            result[offset + 2][id] = (field[zp] - field[zm]) / (2.0 * fixture.grid.spacing.z) + affine.z;
        };
        assign(0, fixture.psi1_fluctuation, fixture.psi1_affine_gradient);
        assign(3, fixture.psi2_fluctuation, fixture.psi2_affine_gradient);
    }
    return result;
}

[[nodiscard]] double normalized_rms_difference(const std::array<std::vector<double>, 6>& actual,
                                                const std::array<std::vector<double>, 6>& expected) {
    long double error_sum = 0.0L, expected_sum = 0.0L;
    std::size_t count = 0;
    for (std::size_t component = 0; component < actual.size(); ++component) {
        for (std::size_t i = 0; i < actual[component].size(); ++i) {
            const long double delta = actual[component][i] - expected[component][i];
            error_sum += delta * delta;
            expected_sum += static_cast<long double>(expected[component][i]) * expected[component][i];
            ++count;
        }
    }
    return std::sqrt(static_cast<double>(error_sum / count)) /
           std::max(std::sqrt(static_cast<double>(expected_sum / count)), 1.0);
}

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

[[nodiscard]] CaseResult case_gradient_error_contract() {
    const auto fixture = ref::make_total_gradient_fixture(16);
    const auto grid = production_grid(fixture.grid);
    const auto gauge = production_gauge(fixture);
    const auto n = fixture.grid.cell_count();
    CudaContext context(0);
    DeviceBuffer<real> u1(n), u2(n), p1x(n), p1y(n), p1z(n), p2x(n), p2y(n), p2z(n);
    const auto output = output_spans(p1x, p1y, p1z, p2x, p2y, p2z);
    const PeriodicStreamfunctionFluctuations input{u1.span(), u2.span()};
    const auto invoke = [&](const Grid3D& candidate_grid, const PeriodicStreamfunctionFluctuations& candidate_input,
                            const AffineGauge& candidate_gauge, const TotalStreamfunctionGradientOutput& candidate_output) {
        enqueue_total_streamfunction_gradients(context, candidate_grid, candidate_input, candidate_gauge, candidate_output);
    };
    bool pass = true;
    std::size_t checks = 0;
    const auto require_invalid = [&](const char* name, const auto& callable) {
        ++checks;
        pass = rejects_with_invalid_argument(name, callable) && pass;
    };
    require_invalid("extent_zero", [&] { invoke({0, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz}, input, gauge, output); });
    require_invalid("extent_negative", [&] { invoke({-1, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz}, input, gauge, output); });
    for (const auto [axis, value, name] : std::array<std::tuple<int, real, const char*>, 12>{{
             {0, real{0}, "dx_zero"}, {0, real{-1}, "dx_negative"}, {0, std::numeric_limits<real>::quiet_NaN(), "dx_nan"}, {0, std::numeric_limits<real>::infinity(), "dx_inf"},
             {1, real{0}, "dy_zero"}, {1, real{-1}, "dy_negative"}, {1, std::numeric_limits<real>::quiet_NaN(), "dy_nan"}, {1, std::numeric_limits<real>::infinity(), "dy_inf"},
             {2, real{0}, "dz_zero"}, {2, real{-1}, "dz_negative"}, {2, std::numeric_limits<real>::quiet_NaN(), "dz_nan"}, {2, std::numeric_limits<real>::infinity(), "dz_inf"}}}) {
        auto invalid = grid;
        if (axis == 0) invalid.dx = value;
        if (axis == 1) invalid.dy = value;
        if (axis == 2) invalid.dz = value;
        require_invalid(name, [&] { invoke(invalid, input, gauge, output); });
    }
    auto nonfinite_psi1 = gauge;
    nonfinite_psi1.psi1_gradient.x = std::numeric_limits<real>::quiet_NaN();
    require_invalid("psi1_affine_nan", [&] { invoke(grid, input, nonfinite_psi1, output); });
    auto nonfinite_psi2 = gauge;
    nonfinite_psi2.psi2_gradient.z = std::numeric_limits<real>::infinity();
    require_invalid("psi2_affine_inf", [&] { invoke(grid, input, nonfinite_psi2, output); });
    require_invalid("input_null", [&] { invoke(grid, {DeviceSpan<real>(nullptr, n), u2.span()}, gauge, output); });
    require_invalid("input_wrong_size", [&] { invoke(grid, {DeviceSpan<real>(u1.data(), n - 1), u2.span()}, gauge, output); });
    auto null_output = output;
    null_output.psi1_x = DeviceSpan<real>(nullptr, n);
    require_invalid("output_null", [&] { invoke(grid, input, gauge, null_output); });
    auto short_output = output;
    short_output.psi2_z = DeviceSpan<real>(p2z.data(), n - 1);
    require_invalid("output_wrong_size", [&] { invoke(grid, input, gauge, short_output); });
    auto input_output_alias = output;
    input_output_alias.psi1_x = u1.span();
    require_invalid("input_output_alias", [&] { invoke(grid, input, gauge, input_output_alias); });
    auto partial_input_output_alias = output;
    partial_input_output_alias.psi1_x = DeviceSpan<real>(u1.data() + 1, n);
    require_invalid("partial_input_output_alias", [&] { invoke(grid, input, gauge, partial_input_output_alias); });
    auto output_alias = output;
    output_alias.psi1_y = p1x.span();
    require_invalid("output_output_alias", [&] { invoke(grid, input, gauge, output_alias); });
    auto partial_output_alias = output;
    partial_output_alias.psi1_y = DeviceSpan<real>(p1x.data() + 1, n);
    require_invalid("partial_output_output_alias", [&] { invoke(grid, input, gauge, partial_output_alias); });

    std::vector<real> host_u1(fixture.psi1_fluctuation.begin(), fixture.psi1_fluctuation.end());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u1.data(), host_u1.data(), n * sizeof(real), cudaMemcpyHostToDevice, context.cuda_stream()));
    invoke(grid, {u1.span(), u1.span()}, gauge, output);
    std::array<std::vector<real>, 6> shared_actual;
    const std::array<DeviceBuffer<real>*, 6> buffers{&p1x, &p1y, &p1z, &p2x, &p2y, &p2z};
    for (std::size_t component = 0; component < buffers.size(); ++component) {
        shared_actual[component].resize(n);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(shared_actual[component].data(), buffers[component]->data(), n * sizeof(real), cudaMemcpyDeviceToHost, context.cuda_stream()));
    }
    context.synchronize();
    auto shared_fixture = fixture;
    shared_fixture.psi2_fluctuation = shared_fixture.psi1_fluctuation;
    const auto shared_expected = oracle(shared_fixture);
    double shared_overlap_error = 0.0;
    for (std::size_t component = 0; component < shared_actual.size(); ++component) {
        shared_overlap_error = std::max(shared_overlap_error, normalized(rms_difference(shared_actual[component], shared_expected[component]), rms(shared_expected[component])));
    }
    const bool shared_overlap_pass = shared_overlap_error <= kOracleTolerance;
    std::cout << std::setprecision(16) << "gradient_contract name=input_input_overlap exception=accepted normalized_rms="
              << shared_overlap_error << " threshold=" << kOracleTolerance << '\n';
    ++checks;
    pass = pass && shared_overlap_pass;

    GradientFixture anisotropic_gpu(fixture);
    const auto anisotropic_actual = anisotropic_gpu.run(fixture);
    const auto anisotropic_expected = oracle(fixture);
    double anisotropic_error = 0.0;
    for (std::size_t component = 0; component < anisotropic_actual.values.size(); ++component) {
        anisotropic_error = std::max(anisotropic_error, normalized(rms_difference(anisotropic_actual.values[component], anisotropic_expected[component]), rms(anisotropic_expected[component])));
    }
    const bool anisotropic_pass = anisotropic_error <= kOracleTolerance;
    std::cout << std::setprecision(16) << "gradient_contract name=anisotropic_positive_spacing exception=accepted normalized_rms="
              << anisotropic_error << " threshold=" << kOracleTolerance << '\n';
    ++checks;
    pass = pass && anisotropic_pass;
    return {pass, "gradient_error_contract", "host-validation-and-gpu-acceptance", grid_description(fixture.grid),
            std::max(shared_overlap_error, anisotropic_error), 0.0, "24 invalid_argument + 2 accepted", std::to_string(checks),
            "invalid extents/spacings/affines/spans/aliases reject as std::invalid_argument; anisotropic spacing and read-only input overlap are accepted"};
}

[[nodiscard]] CaseResult case_gradient_mutation_sensitivity() {
    const auto fixture = ref::make_total_gradient_fixture(16);
    const auto expected = oracle(fixture);
    constexpr double kAffineMutationThreshold = 1.0e-1;
    constexpr double kSpacingMutationThreshold = 1.0e-1;
    constexpr double kBoundaryMutationThreshold = 1.0e-2;
    const double affine_error = normalized_rms_difference(mutation_omit_affine(fixture), expected);
    const double spacing_error = normalized_rms_difference(mutation_dx_for_yz(fixture), expected);
    const double boundary_error = normalized_rms_difference(mutation_clamp_boundary(fixture), expected);
    std::cout << std::setprecision(16) << "gradient_mutant name=omit_affine normalized_rms=" << affine_error
              << " threshold=" << kAffineMutationThreshold << '\n';
    std::cout << std::setprecision(16) << "gradient_mutant name=dx_for_yz normalized_rms=" << spacing_error
              << " threshold=" << kSpacingMutationThreshold << '\n';
    std::cout << std::setprecision(16) << "gradient_mutant name=clamp_boundary normalized_rms=" << boundary_error
              << " threshold=" << kBoundaryMutationThreshold << '\n';
    const bool pass = affine_error > kAffineMutationThreshold && spacing_error > kSpacingMutationThreshold &&
                      boundary_error > kBoundaryMutationThreshold;
    return {pass, "gradient_mutation_sensitivity", "test-only-mutants-vs-independent-oracle", grid_description(fixture.grid),
            std::min({affine_error, spacing_error, boundary_error}), std::max({affine_error, spacing_error, boundary_error}),
            ">0.1, >0.1, >0.01", std::to_string(std::min({affine_error, spacing_error, boundary_error})),
            "test-only affine, anisotropic-spacing, and periodic-wrap mutants must each exceed their explicit normalized-RMS threshold"};
}
}  // namespace

CaseRegistry streamfunction_gradient_case_registry() {
    return {{"gradient_pure_affine", case_gradient_pure_affine}, {"gradient_gpu_oracle", case_gradient_gpu_oracle}, {"gradient_smooth_order", case_gradient_smooth_order}, {"gradient_error_contract", case_gradient_error_contract}, {"gradient_mutation_sensitivity", case_gradient_mutation_sensitivity}};
}
}  // namespace macroflow3d::streamfunctions::test
