#include "reference_operators.hpp"
#include "streamfunction_operator_test_cases.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/numerics/operators/lester_positive_diffusion_operator.cuh"
#include "src/numerics/operators/varcoeff_laplacian.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace macroflow3d::streamfunctions::test {
namespace {

namespace ref = macroflow3d::streamfunctions::reference;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kComparisonThreshold = 1.0e-12;
constexpr double kNullspaceThreshold = 1.0e-13;
constexpr double kHarmonicThreshold = 1.0e-13;
constexpr double kOrderThreshold = 1.8;
constexpr double kFloor = 1.0e-14;

[[nodiscard]] Grid3D production_grid(const ref::Grid& grid) {
    if (std::abs(grid.spacing.x - grid.spacing.y) > 0.0 ||
        std::abs(grid.spacing.x - grid.spacing.z) > 0.0) {
        throw std::invalid_argument("SF-02 production controls require isotropic grid spacing");
    }
    return Grid3D(static_cast<int>(grid.nx), static_cast<int>(grid.ny),
                  static_cast<int>(grid.nz), grid.spacing.x, grid.spacing.y, grid.spacing.z);
}

[[nodiscard]] BCSpec triply_periodic_bc() {
    BCSpec bc;
    bc.xmin = BCFace(BCType::Periodic, real(0.0));
    bc.xmax = BCFace(BCType::Periodic, real(0.0));
    bc.ymin = BCFace(BCType::Periodic, real(0.0));
    bc.ymax = BCFace(BCType::Periodic, real(0.0));
    bc.zmin = BCFace(BCType::Periodic, real(0.0));
    bc.zmax = BCFace(BCType::Periodic, real(0.0));
    return bc;
}

[[nodiscard]] std::string describe_grid(const ref::Grid& grid) {
    std::ostringstream out;
    out << grid.nx << 'x' << grid.ny << 'x' << grid.nz << " h=" << grid.spacing.x;
    return out.str();
}

[[nodiscard]] std::vector<double> difference(const std::vector<double>& left,
                                             const std::vector<double>& right) {
    if (left.size() != right.size()) throw std::invalid_argument("difference size mismatch");
    std::vector<double> result(left.size());
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = left[i] - right[i];
    return result;
}

[[nodiscard]] double q_max(const std::vector<double>& q) {
    if (q.empty()) throw std::invalid_argument("q must not be empty");
    return *std::max_element(q.begin(), q.end());
}

[[nodiscard]] double operator_scale(const ref::Grid& grid, const std::vector<double>& q,
                                    const std::vector<double>& u) {
    const double inverse_spacing_sum = 1.0 / (grid.spacing.x * grid.spacing.x) +
                                       1.0 / (grid.spacing.y * grid.spacing.y) +
                                       1.0 / (grid.spacing.z * grid.spacing.z);
    return std::max(kFloor, kFloor * q_max(q) * inverse_spacing_sum * ref::rms_norm(u));
}

[[nodiscard]] double normalized_error(const ref::Grid& grid, const std::vector<double>& q,
                                      const std::vector<double>& input,
                                      const std::vector<double>& actual,
                                      const std::vector<double>& expected) {
    return ref::rms_norm(difference(actual, expected)) /
           std::max(ref::rms_norm(expected), operator_scale(grid, q, input));
}

[[nodiscard]] bool is_boundary(const ref::Grid& grid, std::size_t index) {
    const std::size_t ix = index % grid.nx;
    const std::size_t iy = (index / grid.nx) % grid.ny;
    const std::size_t iz = index / (grid.nx * grid.ny);
    return ix == 0 || ix + 1 == grid.nx || iy == 0 || iy + 1 == grid.ny || iz == 0 ||
           iz + 1 == grid.nz;
}

[[nodiscard]] double normalized_boundary_error(const ref::Grid& grid, const std::vector<double>& q,
                                               const std::vector<double>& input,
                                               const std::vector<double>& actual,
                                               const std::vector<double>& expected) {
    if (actual.size() != expected.size() || actual.size() != grid.cell_count()) {
        throw std::invalid_argument("boundary error size mismatch");
    }
    long double error_sum = 0.0L;
    long double expected_sum = 0.0L;
    std::size_t count = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!is_boundary(grid, i)) continue;
        const long double delta = static_cast<long double>(actual[i]) - expected[i];
        error_sum += delta * delta;
        const long double value = expected[i];
        expected_sum += value * value;
        ++count;
    }
    if (count == 0) throw std::logic_error("fixture has no boundary cells");
    const double rms_error = std::sqrt(static_cast<double>(error_sum / count));
    const double rms_expected = std::sqrt(static_cast<double>(expected_sum / count));
    return rms_error / std::max(rms_expected, operator_scale(grid, q, input));
}

[[nodiscard]] long double dot(const std::vector<double>& left, const std::vector<double>& right) {
    if (left.size() != right.size()) throw std::invalid_argument("dot size mismatch");
    long double result = 0.0L;
    for (std::size_t i = 0; i < left.size(); ++i) {
        result += static_cast<long double>(left[i]) * right[i];
    }
    return result;
}

class GpuOperatorFixture {
  public:
    GpuOperatorFixture(const ref::Grid& reference_grid, const std::vector<double>& q)
        : reference_grid_(reference_grid), grid_(production_grid(reference_grid)), context_(0),
          d_q_(q.size()), d_input_(q.size()), d_output_(q.size()),
          legacy_(grid_, d_q_.span(), triply_periodic_bc()), positive_(grid_, d_q_.span()) {
        if (q.size() != reference_grid_.cell_count()) {
            throw std::invalid_argument("q size does not match fixture grid");
        }
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_q_.data(), q.data(), q.size() * sizeof(real),
                                               cudaMemcpyHostToDevice, context_.cuda_stream()));
    }

    [[nodiscard]] std::vector<double> apply_positive(const std::vector<double>& input) {
        return apply(input, true);
    }

    [[nodiscard]] std::vector<double> apply_legacy(const std::vector<double>& input) {
        return apply(input, false);
    }

  private:
    [[nodiscard]] std::vector<double> apply(const std::vector<double>& input, bool positive) {
        if (input.size() != reference_grid_.cell_count()) {
            throw std::invalid_argument("input size does not match fixture grid");
        }
        std::vector<double> output(input.size());
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_input_.data(), input.data(),
                                               input.size() * sizeof(real), cudaMemcpyHostToDevice,
                                               context_.cuda_stream()));
        if (positive) {
            positive_.apply(context_, d_input_.span(), d_output_.span());
        } else {
            legacy_.apply(context_, d_input_.span(), d_output_.span());
        }
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(output.data(), d_output_.data(),
                                               output.size() * sizeof(real), cudaMemcpyDeviceToHost,
                                               context_.cuda_stream()));
        // Tests deliberately synchronize only after enqueueing all work for this application.
        context_.synchronize();
        return output;
    }

    ref::Grid reference_grid_;
    Grid3D grid_;
    CudaContext context_;
    DeviceBuffer<real> d_q_;
    DeviceBuffer<real> d_input_;
    DeviceBuffer<real> d_output_;
    operators::VarCoeffLaplacian legacy_;
    operators::LesterPositiveDiffusionOperator positive_;
};

[[nodiscard]] std::vector<double> constant_q(const ref::Grid& grid) {
    return std::vector<double>(grid.cell_count(), 1.0);
}

[[nodiscard]] std::vector<double> analytic_constant_diffusion(
    const ref::TrigonometricFixture& fixture) {
    std::vector<double> result(fixture.grid.cell_count());
    for (std::size_t iz = 0; iz < fixture.grid.nz; ++iz) {
        for (std::size_t iy = 0; iy < fixture.grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < fixture.grid.nx; ++ix) {
                const auto id = fixture.grid.index(ix, iy, iz);
                result[id] = -ref::trigonometric_laplacian(
                    fixture.grid.cell_center(ix, iy, iz), fixture.lengths);
            }
        }
    }
    return result;
}

[[nodiscard]] std::vector<double> analytic_smooth_diffusion(
    const ref::TrigonometricFixture& fixture) {
    std::vector<double> result(fixture.grid.cell_count());
    for (std::size_t iz = 0; iz < fixture.grid.nz; ++iz) {
        for (std::size_t iy = 0; iy < fixture.grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < fixture.grid.nx; ++ix) {
                const auto position = fixture.grid.cell_center(ix, iy, iz);
                const double phase = 2.0 * kPi * (position.x / fixture.lengths.x +
                                                  position.y / fixture.lengths.y +
                                                  position.z / fixture.lengths.z);
                const double q = 1.25 + 0.25 * std::cos(phase);
                const ref::Vec3 gradient_q{
                    -0.25 * 2.0 * kPi / fixture.lengths.x * std::sin(phase),
                    -0.25 * 2.0 * kPi / fixture.lengths.y * std::sin(phase),
                    -0.25 * 2.0 * kPi / fixture.lengths.z * std::sin(phase)};
                const auto gradient_u = ref::trigonometric_gradient(position, fixture.lengths);
                const double gradient_dot = gradient_q.x * gradient_u.x +
                                            gradient_q.y * gradient_u.y +
                                            gradient_q.z * gradient_u.z;
                result[fixture.grid.index(ix, iy, iz)] =
                    -gradient_dot - q * ref::trigonometric_laplacian(position, fixture.lengths);
            }
        }
    }
    return result;
}

[[nodiscard]] std::vector<double> deterministic_y(const ref::TrigonometricFixture& fixture) {
    std::vector<double> result(fixture.grid.cell_count());
    for (std::size_t iz = 0; iz < fixture.grid.nz; ++iz) {
        for (std::size_t iy = 0; iy < fixture.grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < fixture.grid.nx; ++ix) {
                const auto position = fixture.grid.cell_center(ix, iy, iz);
                const double x = 2.0 * kPi * position.x / fixture.lengths.x;
                const double y = 2.0 * kPi * position.y / fixture.lengths.y;
                const double z = 2.0 * kPi * position.z / fixture.lengths.z;
                result[fixture.grid.index(ix, iy, iz)] =
                    0.3 * std::cos(x) + 0.2 * std::sin(2.0 * y) -
                    0.4 * std::cos(3.0 * z) + 0.1 * std::sin(x + y - z);
            }
        }
    }
    return result;
}

[[nodiscard]] CaseResult case_legacy_sign() {
    const auto fixture = ref::make_cubic_trigonometric_fixture(16);
    const auto cpu_a = ref::divergence_form_diffusion(fixture.grid, fixture.q, fixture.scalar);
    GpuOperatorFixture gpu(fixture.grid, fixture.q);
    const auto legacy = gpu.apply_legacy(fixture.scalar);
    const auto positive = gpu.apply_positive(fixture.scalar);
    auto negative_cpu_a = cpu_a;
    for (double& value : negative_cpu_a) value = -value;
    const double legacy_error =
        normalized_error(fixture.grid, fixture.q, fixture.scalar, legacy, negative_cpu_a);
    const double positive_error =
        normalized_error(fixture.grid, fixture.q, fixture.scalar, positive, cpu_a);
    std::vector<double> cancellation(legacy.size());
    for (std::size_t i = 0; i < cancellation.size(); ++i) cancellation[i] = legacy[i] + positive[i];
    const double cancellation_error = ref::rms_norm(cancellation) /
                                      std::max(ref::rms_norm(cpu_a), operator_scale(fixture.grid, fixture.q, fixture.scalar));
    return {legacy_error <= kComparisonThreshold && positive_error <= kComparisonThreshold &&
                cancellation_error <= kNullspaceThreshold,
            "legacy_sign", "gpu-production", describe_grid(fixture.grid), legacy_error, positive_error,
            "n/a", "n/a", "legacy,positive<=1e-12; legacy+positive<=1e-13"};
}

[[nodiscard]] CaseResult case_gpu_oracle(bool smooth_q) {
    auto fixture = ref::make_cubic_trigonometric_fixture(16);
    if (!smooth_q) fixture.q = constant_q(fixture.grid);
    const auto cpu_a = ref::divergence_form_diffusion(fixture.grid, fixture.q, fixture.scalar);
    GpuOperatorFixture gpu(fixture.grid, fixture.q);
    const auto device_a = gpu.apply_positive(fixture.scalar);
    const double global_error =
        normalized_error(fixture.grid, fixture.q, fixture.scalar, device_a, cpu_a);
    const double boundary_error =
        normalized_boundary_error(fixture.grid, fixture.q, fixture.scalar, device_a, cpu_a);
    return {global_error <= kComparisonThreshold && boundary_error <= kComparisonThreshold,
            smooth_q ? "gpu_oracle_smooth" : "gpu_oracle_constant", "gpu-vs-cpu",
            describe_grid(fixture.grid), global_error, boundary_error, "n/a", "n/a",
            "global,boundary<=1e-12"};
}

[[nodiscard]] CaseResult case_nullspace() {
    const auto fixture = ref::make_cubic_trigonometric_fixture(16);
    const std::vector<double> one(fixture.grid.cell_count(), 1.0);
    GpuOperatorFixture gpu(fixture.grid, fixture.q);
    const auto a_one = gpu.apply_positive(one);
    const double inverse_spacing_sum = 3.0 / (fixture.grid.spacing.x * fixture.grid.spacing.x);
    const double defect = ref::rms_norm(a_one) / (q_max(fixture.q) * inverse_spacing_sum);
    return {defect <= kNullspaceThreshold, "nullspace", "gpu-production", describe_grid(fixture.grid),
            defect, 0.0, "n/a", "n/a", "RMS(A1)/(qmax*sum(h^-2))<=1e-13"};
}

[[nodiscard]] CaseResult case_symmetry() {
    const auto fixture = ref::make_cubic_trigonometric_fixture(16);
    const auto y = deterministic_y(fixture);
    GpuOperatorFixture gpu(fixture.grid, fixture.q);
    const auto ax = gpu.apply_positive(fixture.scalar);
    const auto ay = gpu.apply_positive(y);
    const long double x_ay = dot(fixture.scalar, ay);
    const long double y_ax = dot(y, ax);
    const long double denominator = std::abs(x_ay) + std::abs(y_ax);
    const double defect = static_cast<double>(std::abs(x_ay - y_ax) / denominator);
    const bool non_degenerate = std::isfinite(static_cast<double>(denominator)) &&
                                denominator > std::numeric_limits<long double>::epsilon();
    return {non_degenerate && std::isfinite(defect) && defect < kComparisonThreshold,
            "symmetry", "gpu-production", describe_grid(fixture.grid), defect,
            static_cast<double>(denominator), "n/a", "n/a",
            "|xAy-yAx|/(|xAy|+|yAx|)<1e-12; denominator finite/nonzero"};
}

[[nodiscard]] CaseResult case_energy() {
    const auto fixture = ref::make_cubic_trigonometric_fixture(16);
    GpuOperatorFixture gpu(fixture.grid, fixture.q);
    const auto ax = gpu.apply_positive(fixture.scalar);
    const long double energy = dot(fixture.scalar, ax);
    long double face_energy = 0.0L;
    const auto& grid = fixture.grid;
    const double h2 = grid.spacing.x * grid.spacing.x;
    for (std::size_t iz = 0; iz < grid.nz; ++iz) {
        for (std::size_t iy = 0; iy < grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < grid.nx; ++ix) {
                const auto center = grid.index(ix, iy, iz);
                for (const auto axis : {ref::Axis::x, ref::Axis::y, ref::Axis::z}) {
                    std::size_t nx = ix, ny = iy, nz = iz;
                    if (axis == ref::Axis::x) nx = (ix + 1) % grid.nx;
                    if (axis == ref::Axis::y) ny = (iy + 1) % grid.ny;
                    if (axis == ref::Axis::z) nz = (iz + 1) % grid.nz;
                    const auto neighbor = grid.index(nx, ny, nz);
                    const long double delta =
                        static_cast<long double>(fixture.scalar[center]) - fixture.scalar[neighbor];
                    face_energy += ref::harmonic_mean_q(fixture.q[center], fixture.q[neighbor]) *
                                   delta * delta / h2;
                }
            }
        }
    }
    const double relative_match =
        static_cast<double>(std::abs(energy - face_energy) / std::max(face_energy, 1.0L));
    const bool nonnegative = energy >= -kComparisonThreshold * std::max(face_energy, 1.0L);
    return {nonnegative && relative_match <= kComparisonThreshold, "energy", "gpu-production",
            describe_grid(fixture.grid), static_cast<double>(energy), relative_match, "n/a", "n/a",
            "E>=-1e-12*Eface; |E-Eface|/Eface<=1e-12"};
}

[[nodiscard]] CaseResult case_manufactured(bool smooth_q) {
    auto coarse = ref::make_cubic_trigonometric_fixture(16);
    auto fine = ref::make_cubic_trigonometric_fixture(32);
    if (!smooth_q) {
        coarse.q = constant_q(coarse.grid);
        fine.q = constant_q(fine.grid);
    }
    const auto coarse_exact =
        smooth_q ? analytic_smooth_diffusion(coarse) : analytic_constant_diffusion(coarse);
    const auto fine_exact =
        smooth_q ? analytic_smooth_diffusion(fine) : analytic_constant_diffusion(fine);
    GpuOperatorFixture coarse_gpu(coarse.grid, coarse.q);
    GpuOperatorFixture fine_gpu(fine.grid, fine.q);
    const double coarse_error =
        normalized_error(coarse.grid, coarse.q, coarse.scalar,
                         coarse_gpu.apply_positive(coarse.scalar), coarse_exact);
    const double fine_error =
        normalized_error(fine.grid, fine.q, fine.scalar, fine_gpu.apply_positive(fine.scalar),
                         fine_exact);
    const auto order = ref::observed_order(coarse_error, fine_error, coarse.grid.spacing.x,
                                           fine.grid.spacing.x);
    std::ostringstream observed;
    if (order.valid()) observed << std::fixed << std::setprecision(6) << order.value;
    else observed << "n/a(" << order.message << ')';
    return {order.valid() && order.value >= kOrderThreshold,
            smooth_q ? "manufactured_smooth" : "manufactured_constant", "gpu-continuum",
            describe_grid(coarse.grid) + "->" + describe_grid(fine.grid), coarse_error, fine_error,
            ">=1.8", observed.str(), "L2 order>=1.8"};
}

[[nodiscard]] CaseResult case_harmonic_q_face() {
    const auto fixture = ref::make_cubic_trigonometric_fixture(16);
    const auto& grid = fixture.grid;
    const auto center = grid.index(5, 5, 5);
    const auto plus_x = grid.index(6, 5, 5);
    const auto uniform_q = constant_q(grid);
    auto varied_q = uniform_q;
    varied_q[plus_x] = 4.0;
    std::vector<double> input(grid.cell_count(), 0.0);
    input[center] = 1.0;
    GpuOperatorFixture uniform_gpu(grid, uniform_q);
    GpuOperatorFixture varied_gpu(grid, varied_q);
    const auto uniform_output = uniform_gpu.apply_positive(input);
    const auto varied_output = varied_gpu.apply_positive(input);
    const double h2 = grid.spacing.x * grid.spacing.x;
    // Only the +x face differs between fixtures, so Δ(Au)_C*h²=q_f(1,4)-1.
    const double measured_face_q =
        1.0 + (varied_output[center] - uniform_output[center]) * h2;
    const double scaled_error = std::abs(measured_face_q - 1.6) / 1.6;
    const bool rejects_inverse_harmonic_k = std::abs(measured_face_q - 2.5) > 0.5;
    return {scaled_error <= kHarmonicThreshold && rejects_inverse_harmonic_k, "harmonic_q_face",
            "gpu-production", describe_grid(grid), measured_face_q, scaled_error, "n/a", "n/a",
            "q_f(1,4)=1.6 rel<=1e-13; reject inverse-harmonic-K=2.5"};
}

}  // namespace

CaseRegistry gpu_case_registry() {
    return {
        {"legacy_sign", case_legacy_sign},
        {"gpu_oracle_constant", [] { return case_gpu_oracle(false); }},
        {"gpu_oracle_smooth", [] { return case_gpu_oracle(true); }},
        {"nullspace", case_nullspace},
        {"symmetry", case_symmetry},
        {"energy", case_energy},
        {"manufactured_constant", [] { return case_manufactured(false); }},
        {"manufactured_smooth", [] { return case_manufactured(true); }},
        {"harmonic_q_face", case_harmonic_q_face},
    };
}

}  // namespace macroflow3d::streamfunctions::test
