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

constexpr double kDiscreteOracleTolerance = 5.0e-11;
constexpr double kOrderThreshold = 1.8;
constexpr std::array<const char*, 9> kNames{
    "Hpsi2_gradpsi1_x", "Hpsi2_gradpsi1_y", "Hpsi2_gradpsi1_z",
    "Hpsi1_gradpsi2_x", "Hpsi1_gradpsi2_y", "Hpsi1_gradpsi2_z", "B_x", "B_y", "B_z"};

template <typename Callable>
[[nodiscard]] bool rejects_with_invalid_argument(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "hessian_vector_b_contract name=" << name
                  << " exception=none expected=std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument& error) {
        std::cout << "hessian_vector_b_contract name=" << name
                  << " exception=std::invalid_argument message=" << error.what() << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cout << "hessian_vector_b_contract name=" << name
                  << " exception=std::exception message=" << error.what()
                  << " expected=std::invalid_argument\n";
        return false;
    } catch (...) {
        std::cout << "hessian_vector_b_contract name=" << name
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

[[nodiscard]] HessianVectorBOutput output_spans(
    DeviceBuffer<real>& h2g1x, DeviceBuffer<real>& h2g1y, DeviceBuffer<real>& h2g1z,
    DeviceBuffer<real>& h1g2x, DeviceBuffer<real>& h1g2y, DeviceBuffer<real>& h1g2z,
    DeviceBuffer<real>& bx, DeviceBuffer<real>& by, DeviceBuffer<real>& bz) {
    return {h2g1x.span(), h2g1y.span(), h2g1z.span(), h1g2x.span(), h1g2y.span(),
            h1g2z.span(), bx.span(), by.span(), bz.span()};
}

[[nodiscard]] double normalized_fields_rms(const std::array<std::vector<double>, 9>& actual,
                                           const std::array<std::vector<double>, 9>& expected) {
    long double error_sum = 0.0L;
    long double expected_sum = 0.0L;
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

enum class HessianMutant { mixed_sign, dx_for_yz, clamp_boundary, affine_ramp_periodized };

[[nodiscard]] std::array<std::vector<double>, 9> hessian_vector_b_mutant(
    const ref::TotalGradientFixture& fixture, HessianMutant mutant) {
    const auto g1 = ref::centered_total_gradient_oracle(
        fixture.grid, fixture.psi1_fluctuation, fixture.psi1_affine_gradient);
    const auto g2 = ref::centered_total_gradient_oracle(
        fixture.grid, fixture.psi2_fluctuation, fixture.psi2_affine_gradient);
    std::array<std::vector<double>, 9> result;
    for (auto& component : result) component.resize(fixture.grid.cell_count());
    const auto& grid = fixture.grid;
    const bool clamp = mutant == HessianMutant::clamp_boundary;
    const bool affine_ramp = mutant == HessianMutant::affine_ramp_periodized;
    const auto axis_index = [clamp](std::ptrdiff_t value, std::size_t extent) {
        if (clamp) {
            return static_cast<std::size_t>(std::max<std::ptrdiff_t>(
                0, std::min(value, static_cast<std::ptrdiff_t>(extent) - 1)));
        }
        return ref::wrap_index(value, extent);
    };
    const auto sample = [&](const std::vector<double>& fluctuation, const ref::Vec3& affine,
                            std::ptrdiff_t x, std::ptrdiff_t y, std::ptrdiff_t z) {
        const auto ix = axis_index(x, grid.nx);
        const auto iy = axis_index(y, grid.ny);
        const auto iz = axis_index(z, grid.nz);
        const auto id = grid.index(ix, iy, iz);
        if (!affine_ramp) return fluctuation[id];
        const auto p = grid.cell_center(ix, iy, iz);
        return fluctuation[id] + affine.x * p.x + affine.y * p.y + affine.z * p.z;
    };
    for (std::size_t z = 0; z < grid.nz; ++z) for (std::size_t y = 0; y < grid.ny; ++y)
        for (std::size_t x = 0; x < grid.nx; ++x) {
            const auto id = grid.index(x, y, z);
            const auto hessian = [&](const std::vector<double>& u, const ref::Vec3& affine) {
                const auto s = [&](std::ptrdiff_t ox, std::ptrdiff_t oy, std::ptrdiff_t oz) {
                    return sample(u, affine, static_cast<std::ptrdiff_t>(x) + ox,
                                  static_cast<std::ptrdiff_t>(y) + oy,
                                  static_cast<std::ptrdiff_t>(z) + oz);
                };
                ref::SymmetricHessian h{};
                h.xx = (s(1, 0, 0) - 2.0 * s(0, 0, 0) + s(-1, 0, 0)) /
                       (grid.spacing.x * grid.spacing.x);
                h.yy = (s(0, 1, 0) - 2.0 * s(0, 0, 0) + s(0, -1, 0)) /
                       (grid.spacing.y * grid.spacing.y);
                h.zz = (s(0, 0, 1) - 2.0 * s(0, 0, 0) + s(0, 0, -1)) /
                       (grid.spacing.z * grid.spacing.z);
                h.xy = (s(1, 1, 0) - s(1, -1, 0) - s(-1, 1, 0) + s(-1, -1, 0)) /
                       (4.0 * grid.spacing.x * grid.spacing.y);
                h.xz = (s(1, 0, 1) - s(1, 0, -1) - s(-1, 0, 1) + s(-1, 0, -1)) /
                       (4.0 * grid.spacing.x * grid.spacing.z);
                h.yz = (s(0, 1, 1) - s(0, 1, -1) - s(0, -1, 1) + s(0, -1, -1)) /
                       (4.0 * grid.spacing.y * grid.spacing.z);
                if (mutant == HessianMutant::mixed_sign) h.xy = -h.xy;
                if (mutant == HessianMutant::dx_for_yz) {
                    h.yy = (s(0, 1, 0) - 2.0 * s(0, 0, 0) + s(0, -1, 0)) /
                           (grid.spacing.x * grid.spacing.x);
                    h.zz = (s(0, 0, 1) - 2.0 * s(0, 0, 0) + s(0, 0, -1)) /
                           (grid.spacing.x * grid.spacing.x);
                    h.xy = (s(1, 1, 0) - s(1, -1, 0) - s(-1, 1, 0) + s(-1, -1, 0)) /
                           (4.0 * grid.spacing.x * grid.spacing.x);
                    h.xz = (s(1, 0, 1) - s(1, 0, -1) - s(-1, 0, 1) + s(-1, 0, -1)) /
                           (4.0 * grid.spacing.x * grid.spacing.x);
                    h.yz = (s(0, 1, 1) - s(0, 1, -1) - s(0, -1, 1) + s(0, -1, -1)) /
                           (4.0 * grid.spacing.x * grid.spacing.x);
                }
                return h;
            };
            const auto h1 = hessian(fixture.psi1_fluctuation, fixture.psi1_affine_gradient);
            const auto h2 = hessian(fixture.psi2_fluctuation, fixture.psi2_affine_gradient);
            const ref::Vec3 p1{g1.x[id], g1.y[id], g1.z[id]};
            const ref::Vec3 p2{g2.x[id], g2.y[id], g2.z[id]};
            const auto h2g1 = ref::symmetric_hessian_vector_product(h2, p1);
            const auto h1g2 = ref::symmetric_hessian_vector_product(h1, p2);
            result[0][id] = h2g1.x; result[1][id] = h2g1.y; result[2][id] = h2g1.z;
            result[3][id] = h1g2.x; result[4][id] = h1g2.y; result[5][id] = h1g2.z;
            result[6][id] = h2g1.x - h1g2.x;
            result[7][id] = h2g1.y - h1g2.y;
            result[8][id] = h2g1.z - h1g2.z;
        }
    return result;
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

[[nodiscard]] CaseResult case_hessian_vector_b_error_contract() {
    const auto fixture = ref::make_total_gradient_fixture(16);
    const auto grid = production_grid(fixture.grid);
    const auto n = fixture.grid.cell_count();
    CudaContext context(0);
    DeviceBuffer<real> u1(n), u2(n), p1x(n), p1y(n), p1z(n), p2x(n), p2y(n), p2z(n);
    DeviceBuffer<real> h2g1x(n), h2g1y(n), h2g1z(n), h1g2x(n), h1g2y(n), h1g2z(n);
    DeviceBuffer<real> bx(n), by(n), bz(n);
    const PeriodicStreamfunctionFluctuations fluctuations{u1.span(), u2.span()};
    const TotalStreamfunctionGradientView gradients{
        p1x.span(), p1y.span(), p1z.span(), p2x.span(), p2y.span(), p2z.span()};
    const auto output = output_spans(h2g1x, h2g1y, h2g1z, h1g2x, h1g2y, h1g2z, bx, by, bz);
    const auto invoke = [&](const Grid3D& candidate_grid,
                            const PeriodicStreamfunctionFluctuations& candidate_fluctuations,
                            const TotalStreamfunctionGradientView& candidate_gradients,
                            const HessianVectorBOutput& candidate_output) {
        enqueue_streamfunction_hessian_vector_b(context, candidate_grid, candidate_fluctuations,
                                                candidate_gradients, candidate_output);
    };
    bool pass = true;
    std::size_t checks = 0;
    const auto require_invalid = [&](const char* name, const auto& callable) {
        ++checks;
        pass = rejects_with_invalid_argument(name, callable) && pass;
    };
    require_invalid("extent_zero", [&] {
        invoke({0, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz}, fluctuations, gradients, output);
    });
    require_invalid("extent_negative", [&] {
        invoke({-1, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz}, fluctuations, gradients, output);
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
        require_invalid(name, [&] { invoke(invalid, fluctuations, gradients, output); });
    }

    const std::array<const char*, 2> fluctuation_names{"u1", "u2"};
    const std::array<DeviceSpan<real>, 2> fluctuation_spans{u1.span(), u2.span()};
    for (std::size_t component = 0; component < fluctuation_spans.size(); ++component) {
        const auto reject_fluctuation = [&](const char* kind, DeviceSpan<real> replacement) {
            auto invalid = fluctuations;
            if (component == 0) invalid.u1 = replacement;
            else invalid.u2 = replacement;
            require_invalid((std::string("fluctuation_") + fluctuation_names[component] + '_' + kind).c_str(),
                            [&] { invoke(grid, invalid, gradients, output); });
        };
        reject_fluctuation("null", DeviceSpan<real>(nullptr, n));
        reject_fluctuation("short", DeviceSpan<real>(fluctuation_spans[component].data(), n - 1));
        reject_fluctuation("long", DeviceSpan<real>(fluctuation_spans[component].data(), n + 1));
    }
    const std::array<const char*, 6> gradient_names{"psi1_x", "psi1_y", "psi1_z",
                                                      "psi2_x", "psi2_y", "psi2_z"};
    const std::array<DeviceSpan<const real>, 6> gradient_spans{
        gradients.psi1_x, gradients.psi1_y, gradients.psi1_z, gradients.psi2_x,
        gradients.psi2_y, gradients.psi2_z};
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
                            [&] { invoke(grid, fluctuations, invalid, output); });
        };
        reject_gradient("null", DeviceSpan<const real>(nullptr, n));
        reject_gradient("short", DeviceSpan<const real>(gradient_spans[component].data(), n - 1));
        reject_gradient("long", DeviceSpan<const real>(gradient_spans[component].data(), n + 1));
    }
    const std::array<const char*, 9> output_names{
        "h2g1_x", "h2g1_y", "h2g1_z", "h1g2_x", "h1g2_y", "h1g2_z", "b_x", "b_y", "b_z"};
    const std::array<DeviceSpan<real>, 9> output_components{
        output.hessian_psi2_times_grad_psi1_x, output.hessian_psi2_times_grad_psi1_y,
        output.hessian_psi2_times_grad_psi1_z, output.hessian_psi1_times_grad_psi2_x,
        output.hessian_psi1_times_grad_psi2_y, output.hessian_psi1_times_grad_psi2_z,
        output.b_x, output.b_y, output.b_z};
    for (std::size_t component = 0; component < output_components.size(); ++component) {
        const auto reject_output = [&](const char* kind, DeviceSpan<real> replacement) {
            auto invalid = output;
            switch (component) {
                case 0: invalid.hessian_psi2_times_grad_psi1_x = replacement; break;
                case 1: invalid.hessian_psi2_times_grad_psi1_y = replacement; break;
                case 2: invalid.hessian_psi2_times_grad_psi1_z = replacement; break;
                case 3: invalid.hessian_psi1_times_grad_psi2_x = replacement; break;
                case 4: invalid.hessian_psi1_times_grad_psi2_y = replacement; break;
                case 5: invalid.hessian_psi1_times_grad_psi2_z = replacement; break;
                case 6: invalid.b_x = replacement; break;
                case 7: invalid.b_y = replacement; break;
                default: invalid.b_z = replacement; break;
            }
            require_invalid((std::string("output_") + output_names[component] + '_' + kind).c_str(),
                            [&] { invoke(grid, fluctuations, gradients, invalid); });
        };
        reject_output("null", DeviceSpan<real>(nullptr, n));
        reject_output("short", DeviceSpan<real>(output_components[component].data(), n - 1));
        reject_output("long", DeviceSpan<real>(output_components[component].data(), n + 1));
    }

    auto fluctuation_overlap = output;
    fluctuation_overlap.b_x = u1.span();
    require_invalid("output_fluctuation_overlap", [&] {
        invoke(grid, fluctuations, gradients, fluctuation_overlap);
    });
    auto gradient_overlap = output;
    gradient_overlap.b_y = DeviceSpan<real>(p1y.data(), n);
    require_invalid("output_gradient_overlap", [&] {
        invoke(grid, fluctuations, gradients, gradient_overlap);
    });
    auto exact_output_overlap = output;
    exact_output_overlap.hessian_psi2_times_grad_psi1_y = h2g1x.span();
    require_invalid("output_output_exact_overlap", [&] {
        invoke(grid, fluctuations, gradients, exact_output_overlap);
    });
    auto partial_output_overlap = output;
    partial_output_overlap.hessian_psi2_times_grad_psi1_y = DeviceSpan<real>(h2g1x.data() + 1, n);
    require_invalid("output_output_partial_overlap", [&] {
        invoke(grid, fluctuations, gradients, partial_output_overlap);
    });

    const std::vector<real> host_u1(fixture.psi1_fluctuation.begin(), fixture.psi1_fluctuation.end());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u1.data(), host_u1.data(), n * sizeof(real),
                                           cudaMemcpyHostToDevice, context.cuda_stream()));
    const TotalStreamfunctionGradientView all_read_only_inputs{
        u1.span(), u1.span(), u1.span(), u1.span(), u1.span(), u1.span()};
    invoke(grid, {u1.span(), u1.span()}, all_read_only_inputs, output);
    const std::array<DeviceBuffer<real>*, 9> buffers{
        &h2g1x, &h2g1y, &h2g1z, &h1g2x, &h1g2y, &h1g2z, &bx, &by, &bz};
    std::array<std::vector<real>, 9> accepted;
    for (std::size_t component = 0; component < buffers.size(); ++component) {
        accepted[component].resize(n);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(accepted[component].data(), buffers[component]->data(),
                                                n * sizeof(real), cudaMemcpyDeviceToHost,
                                                context.cuda_stream()));
    }
    context.synchronize();
    bool finite = true;
    double accepted_linf = 0.0;
    for (const auto& component : accepted) for (const real value : component) {
        finite = finite && std::isfinite(static_cast<double>(value));
        accepted_linf = std::max(accepted_linf, std::abs(static_cast<double>(value)));
    }
    ++checks;
    std::cout << std::setprecision(16)
              << "hessian_vector_b_contract name=input_input_overlap exception=accepted finite="
              << (finite ? "true" : "false") << " linf=" << accepted_linf << '\n';
    pass = pass && finite;
    return {pass, "hessian_vector_b_error_contract", "host-validation-and-gpu-acceptance",
            grid_description(fixture.grid), accepted_linf, 0.0,
            "69 invalid_argument + 1 accepted", std::to_string(checks),
            "invalid extents/spacings/spans/output aliases reject; all read-only inputs may overlap and yield finite output"};
}

[[nodiscard]] CaseResult case_hessian_vector_b_mutation_sensitivity() {
    const auto fixture = ref::make_total_gradient_fixture(16);
    const auto expected = fields_to_arrays(discrete_oracle(fixture));
    constexpr double kMixedSignThreshold = 1.0e-2;
    constexpr double kSpacingThreshold = 1.0e-1;
    constexpr double kBoundaryThreshold = 1.0e-2;
    constexpr double kAffineRampThreshold = 1.0e-1;
    const double mixed_sign = normalized_fields_rms(
        hessian_vector_b_mutant(fixture, HessianMutant::mixed_sign), expected);
    const double dx_for_yz = normalized_fields_rms(
        hessian_vector_b_mutant(fixture, HessianMutant::dx_for_yz), expected);
    const double clamp_boundary = normalized_fields_rms(
        hessian_vector_b_mutant(fixture, HessianMutant::clamp_boundary), expected);
    const double affine_ramp = normalized_fields_rms(
        hessian_vector_b_mutant(fixture, HessianMutant::affine_ramp_periodized), expected);
    const bool finite = std::isfinite(mixed_sign) && std::isfinite(dx_for_yz) &&
                        std::isfinite(clamp_boundary) && std::isfinite(affine_ramp);
    std::cout << std::setprecision(16) << "hessian_vector_b_mutant name=mixed_sign normalized_rms="
              << mixed_sign << " threshold=" << kMixedSignThreshold << '\n';
    std::cout << std::setprecision(16) << "hessian_vector_b_mutant name=dx_for_yz normalized_rms="
              << dx_for_yz << " threshold=" << kSpacingThreshold << '\n';
    std::cout << std::setprecision(16) << "hessian_vector_b_mutant name=clamp_boundary normalized_rms="
              << clamp_boundary << " threshold=" << kBoundaryThreshold << '\n';
    std::cout << std::setprecision(16)
              << "hessian_vector_b_mutant name=affine_ramp_periodized normalized_rms="
              << affine_ramp << " threshold=" << kAffineRampThreshold << '\n';
    const bool pass = finite && mixed_sign > kMixedSignThreshold && dx_for_yz > kSpacingThreshold &&
                      clamp_boundary > kBoundaryThreshold && affine_ramp > kAffineRampThreshold;
    return {pass, "hessian_vector_b_mutation_sensitivity",
            "test-only-mutants-vs-independent-oracle", grid_description(fixture.grid),
            std::min({mixed_sign, dx_for_yz, clamp_boundary, affine_ramp}),
            std::max({mixed_sign, dx_for_yz, clamp_boundary, affine_ramp}),
            ">0.01, >0.1, >0.01, >0.1",
            std::to_string(std::min({mixed_sign, dx_for_yz, clamp_boundary, affine_ramp})),
            "mixed-sign, anisotropic-spacing, periodic-wrap, and affine-ramp Hessian mutants each exceed explicit normalized-RMS thresholds"};
}
}  // namespace

CaseRegistry hessian_vector_b_case_registry() {
    return {{"hessian_vector_b_gpu_oracle", case_hessian_vector_b_gpu_oracle},
            {"hessian_vector_b_smooth_order", case_hessian_vector_b_smooth_order},
            {"hessian_vector_b_zero_controls", case_hessian_vector_b_zero_controls},
            {"hessian_vector_b_error_contract", case_hessian_vector_b_error_contract},
            {"hessian_vector_b_mutation_sensitivity", case_hessian_vector_b_mutation_sensitivity}};
}
}  // namespace macroflow3d::streamfunctions::test
