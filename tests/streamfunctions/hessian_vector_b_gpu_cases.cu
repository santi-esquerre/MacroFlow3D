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

constexpr double kDiscreteOracleTolerance = 5.0e-11;
constexpr double kOrderThreshold = 1.8;
constexpr std::array<const char*, 9> kNames{
    "Hpsi2_gradpsi1_x", "Hpsi2_gradpsi1_y", "Hpsi2_gradpsi1_z",
    "Hpsi1_gradpsi2_x", "Hpsi1_gradpsi2_y", "Hpsi1_gradpsi2_z", "B_x", "B_y", "B_z"};

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
[[nodiscard]] double linf_difference(const std::vector<real>& actual,
                                     const std::vector<double>& expected) {
    double maximum = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        maximum = std::max(maximum, std::abs(static_cast<double>(actual[i]) - expected[i]));
    }
    return maximum;
}
[[nodiscard]] double periodic_boundary_linf(const ref::Grid& grid,
                                            const std::vector<real>& actual,
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
[[nodiscard]] double normalized(double value, double scale) {
    return value / std::max(scale, 1.0);
}

struct GpuHessianVectorB { std::array<std::vector<real>, 9> values; };

class HessianVectorBFixture {
  public:
    explicit HessianVectorBFixture(const ref::TotalGradientFixture& source)
        : grid_(production_grid(source.grid)), context_(0), u1_(source.grid.cell_count()),
          u2_(source.grid.cell_count()), p1x_(source.grid.cell_count()),
          p1y_(source.grid.cell_count()), p1z_(source.grid.cell_count()),
          p2x_(source.grid.cell_count()), p2y_(source.grid.cell_count()),
          p2z_(source.grid.cell_count()), h2g1x_(source.grid.cell_count()),
          h2g1y_(source.grid.cell_count()), h2g1z_(source.grid.cell_count()),
          h1g2x_(source.grid.cell_count()), h1g2y_(source.grid.cell_count()),
          h1g2z_(source.grid.cell_count()), bx_(source.grid.cell_count()), by_(source.grid.cell_count()),
          bz_(source.grid.cell_count()) {}

    [[nodiscard]] GpuHessianVectorB run(const ref::TotalGradientFixture& source) {
        GpuHessianVectorB result;
        for (auto& component : result.values) component.resize(source.grid.cell_count());
        const std::vector<real> h1(source.psi1_fluctuation.begin(), source.psi1_fluctuation.end());
        const std::vector<real> h2(source.psi2_fluctuation.begin(), source.psi2_fluctuation.end());
        // One stream establishes exact producer/consumer ordering: H2D -> SF-07
        // gradients -> SF-08 HVP/B -> D2H. Synchronization occurs only below.
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u1_.data(), h1.data(), h1.size() * sizeof(real), cudaMemcpyHostToDevice, context_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u2_.data(), h2.data(), h2.size() * sizeof(real), cudaMemcpyHostToDevice, context_.cuda_stream()));
        enqueue_total_streamfunction_gradients(context_, grid_, {u1_.span(), u2_.span()}, production_gauge(source),
                                               {p1x_.span(), p1y_.span(), p1z_.span(), p2x_.span(), p2y_.span(), p2z_.span()});
        enqueue_streamfunction_hessian_vector_b(
            context_, grid_, {u1_.span(), u2_.span()},
            {p1x_.span(), p1y_.span(), p1z_.span(), p2x_.span(), p2y_.span(), p2z_.span()},
            {h2g1x_.span(), h2g1y_.span(), h2g1z_.span(), h1g2x_.span(), h1g2y_.span(),
             h1g2z_.span(), bx_.span(), by_.span(), bz_.span()});
        const std::array<DeviceBuffer<real>*, 9> buffers{
            &h2g1x_, &h2g1y_, &h2g1z_, &h1g2x_, &h1g2y_, &h1g2z_, &bx_, &by_, &bz_};
        for (std::size_t c = 0; c < buffers.size(); ++c) {
            MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(result.values[c].data(), buffers[c]->data(),
                                                    result.values[c].size() * sizeof(real),
                                                    cudaMemcpyDeviceToHost, context_.cuda_stream()));
        }
        context_.synchronize();
        return result;
    }

  private:
    Grid3D grid_;
    CudaContext context_;
    DeviceBuffer<real> u1_, u2_, p1x_, p1y_, p1z_, p2x_, p2y_, p2z_;
    DeviceBuffer<real> h2g1x_, h2g1y_, h2g1z_, h1g2x_, h1g2y_, h1g2z_, bx_, by_, bz_;
};

[[nodiscard]] std::array<std::vector<double>, 9> fields_to_arrays(
    const ref::HessianVectorBFields& fields) {
    return {fields.hessian_psi2_times_gradient_psi1.x,
            fields.hessian_psi2_times_gradient_psi1.y,
            fields.hessian_psi2_times_gradient_psi1.z,
            fields.hessian_psi1_times_gradient_psi2.x,
            fields.hessian_psi1_times_gradient_psi2.y,
            fields.hessian_psi1_times_gradient_psi2.z, fields.b.x, fields.b.y, fields.b.z};
}
[[nodiscard]] ref::HessianVectorBFields discrete_oracle(const ref::TotalGradientFixture& fixture) {
    const auto g1 = ref::centered_total_gradient_oracle(fixture.grid, fixture.psi1_fluctuation,
                                                        fixture.psi1_affine_gradient);
    const auto g2 = ref::centered_total_gradient_oracle(fixture.grid, fixture.psi2_fluctuation,
                                                        fixture.psi2_affine_gradient);
    return ref::centered_hessian_vector_b_oracle(fixture.grid, fixture.psi1_fluctuation,
                                                  fixture.psi2_fluctuation, g1, g2);
}

[[nodiscard]] CaseResult case_hessian_vector_b_gpu_oracle() {
    const auto fixture = ref::make_total_gradient_fixture(16);
    HessianVectorBFixture gpu(fixture);
    const auto actual = gpu.run(fixture);
    const auto expected = fields_to_arrays(discrete_oracle(fixture));
    bool finite = true;
    double worst_global = 0.0, worst_boundary = 0.0;
    for (std::size_t c = 0; c < kNames.size(); ++c) {
        const double scale = rms(expected[c]);
        const double global = normalized(rms_difference(actual.values[c], expected[c]), scale);
        const double boundary = normalized(periodic_boundary_linf(fixture.grid, actual.values[c], expected[c]), scale);
        finite = finite && std::isfinite(global) && std::isfinite(boundary);
        worst_global = std::max(worst_global, global);
        worst_boundary = std::max(worst_boundary, boundary);
        std::cout << std::setprecision(16) << "hessian_vector_b_oracle component=" << kNames[c]
                  << " normalized_rms=" << global << " boundary_linf=" << boundary << '\n';
    }
    return {finite && worst_global <= kDiscreteOracleTolerance && worst_boundary <= kDiscreteOracleTolerance,
            "hessian_vector_b_gpu_oracle", "gpu-vs-independent-long-double-cpu",
            grid_description(fixture.grid), worst_global, worst_boundary, "n/a", "n/a",
            "all nine normalized RMS and periodic-boundary Linf <=5e-11"};
}

[[nodiscard]] CaseResult case_hessian_vector_b_smooth_order() {
    const std::array<std::size_t, 3> levels{16, 32, 64};
    std::array<std::array<double, 9>, 3> l2{}, linf{};
    for (std::size_t level = 0; level < levels.size(); ++level) {
        const auto fixture = ref::make_total_gradient_fixture(levels[level]);
        HessianVectorBFixture gpu(fixture);
        const auto actual = gpu.run(fixture);
        const auto expected = fields_to_arrays(ref::analytic_hessian_vector_b(fixture));
        for (std::size_t c = 0; c < kNames.size(); ++c) {
            l2[level][c] = normalized(rms_difference(actual.values[c], expected[c]), rms(expected[c]));
            linf[level][c] = normalized(linf_difference(actual.values[c], expected[c]), rms(expected[c]));
        }
    }
    bool pass = true;
    double worst_l2 = 0.0, worst_fine_l2 = 0.0, minimum_order = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < kNames.size(); ++c) {
        const auto first = ref::observed_order(l2[0][c], l2[1][c], 1.0 / 16.0, 1.0 / 32.0);
        const auto second = ref::observed_order(l2[1][c], l2[2][c], 1.0 / 32.0, 1.0 / 64.0);
        const bool decreases = std::isfinite(linf[0][c]) && std::isfinite(linf[1][c]) &&
                               std::isfinite(linf[2][c]) && linf[1][c] < linf[0][c] &&
                               linf[2][c] < linf[1][c];
        const double first_value = first.valid() ? first.value : -std::numeric_limits<double>::infinity();
        const double second_value = second.valid() ? second.value : -std::numeric_limits<double>::infinity();
        pass = pass && first.valid() && second.valid() && first.value >= kOrderThreshold &&
               second.value >= kOrderThreshold && decreases;
        minimum_order = std::min({minimum_order, first_value, second_value});
        worst_l2 = std::max(worst_l2, l2[0][c]);
        worst_fine_l2 = std::max(worst_fine_l2, l2[2][c]);
        std::cout << std::setprecision(16) << "hessian_vector_b_convergence component=" << kNames[c]
                  << " l2_16=" << l2[0][c] << " linf_16=" << linf[0][c]
                  << " l2_32=" << l2[1][c] << " linf_32=" << linf[1][c]
                  << " l2_64=" << l2[2][c] << " linf_64=" << linf[2][c]
                  << " order_16_32=" << first_value << " order_32_64=" << second_value
                  << " linf_strictly_decreases=" << (decreases ? "true" : "false") << '\n';
    }
    return {pass, "hessian_vector_b_smooth_order", "gpu-continuum",
            "16^3->32^3->64^3 unequal spacing", worst_l2, worst_fine_l2,
            ">=1.8 twice", std::to_string(minimum_order),
            "each of 9 components: both L2 orders>=1.8 and Linf strictly decreases"};
}

[[nodiscard]] CaseResult case_hessian_vector_b_zero_controls() {
    const auto pure_affine = ref::make_pure_affine_total_gradient_fixture(16);
    HessianVectorBFixture affine_gpu(pure_affine);
    const auto affine_actual = affine_gpu.run(pure_affine);
    const auto parallel = ref::make_parallel_total_gradient_fixture(16, 2.0);
    HessianVectorBFixture parallel_gpu(parallel);
    const auto parallel_actual = parallel_gpu.run(parallel);
    double affine_linf = 0.0, parallel_b_linf = 0.0, parallel_hvp_scale = 0.0;
    bool finite = true;
    for (std::size_t c = 0; c < kNames.size(); ++c) {
        for (real value : affine_actual.values[c]) {
            finite = finite && std::isfinite(static_cast<double>(value));
            affine_linf = std::max(affine_linf, std::abs(static_cast<double>(value)));
        }
        if (c < 6) for (real value : parallel_actual.values[c]) {
            finite = finite && std::isfinite(static_cast<double>(value));
            parallel_hvp_scale = std::max(parallel_hvp_scale, std::abs(static_cast<double>(value)));
        }
    }
    for (std::size_t c = 6; c < 9; ++c) for (real value : parallel_actual.values[c]) {
        finite = finite && std::isfinite(static_cast<double>(value));
        parallel_b_linf = std::max(parallel_b_linf, std::abs(static_cast<double>(value)));
    }
    const double affine_tolerance = 16.0 * std::numeric_limits<real>::epsilon();
    const double parallel_tolerance = 128.0 * std::numeric_limits<real>::epsilon() *
                                      std::max(parallel_hvp_scale, 1.0);
    std::cout << std::setprecision(16) << "hessian_vector_b_zero pure_affine_linf=" << affine_linf
              << " threshold=" << affine_tolerance << '\n';
    std::cout << std::setprecision(16) << "hessian_vector_b_zero parallel_scale=2 b_linf="
              << parallel_b_linf << " hvp_scale=" << parallel_hvp_scale
              << " threshold=" << parallel_tolerance << '\n';
    return {finite && affine_linf <= affine_tolerance && parallel_b_linf <= parallel_tolerance,
            "hessian_vector_b_zero_controls", "gpu-analytic-zero-controls",
            grid_description(pure_affine.grid), affine_linf, parallel_b_linf, "roundoff", "n/a",
            "pure-affine all HVP/B <=16 epsilon; binary parallel B <=128 epsilon*max(HVP,1)"};
}
}  // namespace

CaseRegistry hessian_vector_b_case_registry() {
    return {{"hessian_vector_b_gpu_oracle", case_hessian_vector_b_gpu_oracle},
            {"hessian_vector_b_smooth_order", case_hessian_vector_b_smooth_order},
            {"hessian_vector_b_zero_controls", case_hessian_vector_b_zero_controls}};
}
}  // namespace macroflow3d::streamfunctions::test
