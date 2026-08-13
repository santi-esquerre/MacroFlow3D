#include "reference_operators.hpp"
#include "streamfunction_operator_test_cases.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ref = macroflow3d::streamfunctions::reference;
namespace test = macroflow3d::streamfunctions::test;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kOrderThreshold = 1.8;
constexpr double kDiffusionOrderThreshold = 1.6;
constexpr double kRelativeFloor = 1.0e-14;

using CaseResult = test::CaseResult;

[[nodiscard]] std::string describe_grid(const ref::Grid& grid) {
    std::ostringstream out;
    out << grid.nx << 'x' << grid.ny << 'x' << grid.nz << " h=(" << grid.spacing.x << ','
        << grid.spacing.y << ',' << grid.spacing.z << ')';
    return out.str();
}

[[nodiscard]] std::vector<double> difference(const std::vector<double>& left,
                                             const std::vector<double>& right) {
    if (left.size() != right.size()) throw std::invalid_argument("difference size mismatch");
    std::vector<double> result(left.size());
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = left[i] - right[i];
    return result;
}

[[nodiscard]] double normalized_rms_error(const std::vector<double>& actual,
                                          const std::vector<double>& expected) {
    return ref::rms_norm(difference(actual, expected)) /
           std::max(ref::rms_norm(expected), kRelativeFloor);
}

[[nodiscard]] std::vector<double> analytic_first(const ref::TrigonometricFixture& fixture,
                                                  ref::Axis axis) {
    std::vector<double> result(fixture.grid.cell_count());
    for (std::size_t iz = 0; iz < fixture.grid.nz; ++iz) {
        for (std::size_t iy = 0; iy < fixture.grid.ny; ++iy) {
            for (std::size_t ix = 0; ix < fixture.grid.nx; ++ix) {
                const auto gradient = ref::trigonometric_gradient(
                    fixture.grid.cell_center(ix, iy, iz), fixture.lengths);
                const auto id = fixture.grid.index(ix, iy, iz);
                result[id] = axis == ref::Axis::x ? gradient.x :
                             axis == ref::Axis::y ? gradient.y : gradient.z;
            }
        }
    }
    return result;
}

[[nodiscard]] double mixed_second(const ref::Vec3& position, const ref::Vec3& lengths,
                                  ref::Axis first, ref::Axis second) {
    const double ax = 2.0 * kPi * position.x / lengths.x;
    const double ay = 2.0 * kPi * position.y / lengths.y;
    const double az = 2.0 * kPi * position.z / lengths.z;
    const double kx = 2.0 * kPi / lengths.x;
    const double ky = 2.0 * kPi / lengths.y;
    const double kz = 2.0 * kPi / lengths.z;
    if ((first == ref::Axis::x && second == ref::Axis::y) ||
        (first == ref::Axis::y && second == ref::Axis::x)) return -0.125 * kx * ky * std::sin(ax + ay - az);
    if ((first == ref::Axis::x && second == ref::Axis::z) ||
        (first == ref::Axis::z && second == ref::Axis::x)) return 0.125 * kx * kz * std::sin(ax + ay - az);
    return 0.125 * ky * kz * std::sin(ax + ay - az);
}

[[nodiscard]] std::vector<double> analytic_second(const ref::TrigonometricFixture& fixture,
                                                   ref::Axis axis) {
    std::vector<double> result(fixture.grid.cell_count());
    const double kx = 2.0 * kPi / fixture.lengths.x;
    const double ky = 2.0 * kPi / fixture.lengths.y;
    const double kz = 2.0 * kPi / fixture.lengths.z;
    for (std::size_t iz = 0; iz < fixture.grid.nz; ++iz) for (std::size_t iy = 0; iy < fixture.grid.ny; ++iy) for (std::size_t ix = 0; ix < fixture.grid.nx; ++ix) {
        const auto p = fixture.grid.cell_center(ix, iy, iz);
        const double ax = kx * p.x, ay = ky * p.y, az = kz * p.z;
        const double shared = -0.125 * (axis == ref::Axis::x ? kx * kx : axis == ref::Axis::y ? ky * ky : kz * kz) * std::sin(ax + ay - az);
        result[fixture.grid.index(ix, iy, iz)] = axis == ref::Axis::x ? -kx * kx * std::sin(ax) + shared :
            axis == ref::Axis::y ? -2.0 * ky * ky * std::cos(2.0 * ay) + shared :
                                   2.25 * kz * kz * std::sin(3.0 * az) + shared;
    }
    return result;
}

[[nodiscard]] std::vector<double> analytic_mixed(const ref::TrigonometricFixture& fixture,
                                                  ref::Axis first, ref::Axis second) {
    std::vector<double> result(fixture.grid.cell_count());
    for (std::size_t iz = 0; iz < fixture.grid.nz; ++iz) for (std::size_t iy = 0; iy < fixture.grid.ny; ++iy) for (std::size_t ix = 0; ix < fixture.grid.nx; ++ix) {
        result[fixture.grid.index(ix, iy, iz)] = mixed_second(fixture.grid.cell_center(ix, iy, iz), fixture.lengths, first, second);
    }
    return result;
}

[[nodiscard]] std::vector<double> analytic_diffusion(const ref::TrigonometricFixture& fixture) {
    std::vector<double> result(fixture.grid.cell_count());
    for (std::size_t iz = 0; iz < fixture.grid.nz; ++iz) for (std::size_t iy = 0; iy < fixture.grid.ny; ++iy) for (std::size_t ix = 0; ix < fixture.grid.nx; ++ix) {
        const auto p = fixture.grid.cell_center(ix, iy, iz);
        const double phase = 2.0 * kPi * (p.x / fixture.lengths.x + p.y / fixture.lengths.y + p.z / fixture.lengths.z);
        const double q = 1.25 + 0.25 * std::cos(phase);
        const ref::Vec3 grad_q{-0.25 * 2.0 * kPi / fixture.lengths.x * std::sin(phase),
                                -0.25 * 2.0 * kPi / fixture.lengths.y * std::sin(phase),
                                -0.25 * 2.0 * kPi / fixture.lengths.z * std::sin(phase)};
        const auto grad_u = ref::trigonometric_gradient(p, fixture.lengths);
        const double grad_dot = grad_q.x * grad_u.x + grad_q.y * grad_u.y + grad_q.z * grad_u.z;
        result[fixture.grid.index(ix, iy, iz)] = -grad_dot - q * ref::trigonometric_laplacian(p, fixture.lengths);
    }
    return result;
}

[[nodiscard]] std::vector<double> sign_mutant_diffusion(const ref::TrigonometricFixture& fixture) {
    auto result = ref::divergence_form_diffusion(fixture.grid, fixture.q, fixture.scalar);
    for (double& value : result) value = -value;
    return result;
}

[[nodiscard]] std::vector<double> broken_x_wrap_first(const ref::TrigonometricFixture& fixture) {
    std::vector<double> result(fixture.grid.cell_count());
    const auto& g = fixture.grid;
    for (std::size_t iz = 0; iz < g.nz; ++iz) for (std::size_t iy = 0; iy < g.ny; ++iy) for (std::size_t ix = 0; ix < g.nx; ++ix) {
        const std::size_t plus = ix + 1U < g.nx ? ix + 1U : g.nx - 1U;
        const std::size_t minus = ix == 0U ? 0U : ix - 1U;
        result[g.index(ix, iy, iz)] = (fixture.scalar[g.index(plus, iy, iz)] - fixture.scalar[g.index(minus, iy, iz)]) / (2.0 * g.spacing.x);
    }
    return result;
}

[[nodiscard]] CaseResult make_rate_case(const std::string& name, const std::string& kind,
                                         const ref::TrigonometricFixture& coarse,
                                         const ref::TrigonometricFixture& fine,
                                         double coarse_error, double fine_error, double threshold) {
    const auto order = ref::observed_order(coarse_error, fine_error, coarse.grid.spacing.x, fine.grid.spacing.x);
    std::ostringstream expected, observed, limit;
    expected << ">=" << threshold;
    if (order.valid()) observed << std::fixed << std::setprecision(3) << order.value;
    else observed << "n/a(" << order.message << ')';
    limit << "order " << expected.str();
    return {order.valid() && order.value >= threshold, name, kind,
            describe_grid(coarse.grid) + " -> " + describe_grid(fine.grid), coarse_error, fine_error,
            expected.str(), observed.str(), limit.str()};
}

void print_result(const CaseResult& result) {
    std::cout << std::setprecision(8) << "case=" << result.name << " kind=" << result.kind
              << " grid=" << result.grid << " coarse_norm=" << result.coarse_norm
              << " fine_norm=" << result.fine_norm << " expected_order=" << result.expected_order
              << " observed_order=" << result.observed_order << " threshold=" << result.threshold
              << " verdict=" << (result.pass ? "PASS" : "FAIL") << '\n';
}

[[nodiscard]] CaseResult case_wrap_general() {
    const std::vector<std::pair<std::ptrdiff_t, std::size_t>> checks{{-37, 5}, {-6, 5}, {-1, 5}, {0, 5}, {4, 5}, {5, 5}, {37, 5}};
    bool pass = true;
    for (const auto [index, extent] : checks) {
        const auto expected = static_cast<std::size_t>(((index % static_cast<std::ptrdiff_t>(extent)) + static_cast<std::ptrdiff_t>(extent)) % static_cast<std::ptrdiff_t>(extent));
        pass = pass && ref::wrap_index(index, extent) == expected;
    }
    return {pass, "wrap_general", "positive", "extent=5", 0.0, 0.0, "n/a", "n/a", "all signed indices wrap"};
}

[[nodiscard]] CaseResult case_norms_rate() {
    const std::vector<double> values{3.0, 4.0};
    const auto rate = ref::observed_order(0.04, 0.01, 0.25, 0.125);
    const bool pass = std::abs(ref::rms_norm(values) - std::sqrt(12.5)) < 1.0e-14 &&
                      std::abs(ref::linf_norm(values) - 4.0) < 1.0e-14 && rate.valid() &&
                      std::abs(rate.value - 2.0) < 1.0e-14;
    return {pass, "norms_rate", "positive", "vector=2", ref::rms_norm(values), ref::linf_norm(values), "2.0", rate.valid() ? "2.000" : "n/a", "exact norms and rate"};
}

[[nodiscard]] CaseResult case_first_derivative() {
    const auto coarse = ref::make_anisotropic_trigonometric_fixture(false);
    const auto fine = ref::make_anisotropic_trigonometric_fixture(true);
    double coarse_error = 0.0, fine_error = 0.0;
    for (const auto axis : {ref::Axis::x, ref::Axis::y, ref::Axis::z}) {
        coarse_error = std::max(coarse_error, normalized_rms_error(ref::centered_first(coarse.grid, coarse.scalar, axis), analytic_first(coarse, axis)));
        fine_error = std::max(fine_error, normalized_rms_error(ref::centered_first(fine.grid, fine.scalar, axis), analytic_first(fine, axis)));
    }
    return make_rate_case("first_derivative", "positive", coarse, fine, coarse_error, fine_error, kOrderThreshold);
}

[[nodiscard]] CaseResult case_cubic_derivatives() {
    const auto coarse = ref::make_cubic_trigonometric_fixture(16);
    const auto fine = ref::make_cubic_trigonometric_fixture(32);
    double coarse_error = 0.0, fine_error = 0.0;
    for (const auto axis : {ref::Axis::x, ref::Axis::y, ref::Axis::z}) {
        coarse_error = std::max(coarse_error, normalized_rms_error(ref::centered_first(coarse.grid, coarse.scalar, axis), analytic_first(coarse, axis)));
        fine_error = std::max(fine_error, normalized_rms_error(ref::centered_first(fine.grid, fine.scalar, axis), analytic_first(fine, axis)));
        coarse_error = std::max(coarse_error, normalized_rms_error(ref::centered_second(coarse.grid, coarse.scalar, axis), analytic_second(coarse, axis)));
        fine_error = std::max(fine_error, normalized_rms_error(ref::centered_second(fine.grid, fine.scalar, axis), analytic_second(fine, axis)));
    }
    for (const auto axes : {std::pair{ref::Axis::x, ref::Axis::y}, std::pair{ref::Axis::x, ref::Axis::z}, std::pair{ref::Axis::y, ref::Axis::z}}) {
        coarse_error = std::max(coarse_error, normalized_rms_error(ref::centered_mixed(coarse.grid, coarse.scalar, axes.first, axes.second), analytic_mixed(coarse, axes.first, axes.second)));
        fine_error = std::max(fine_error, normalized_rms_error(ref::centered_mixed(fine.grid, fine.scalar, axes.first, axes.second), analytic_mixed(fine, axes.first, axes.second)));
    }
    return make_rate_case("cubic_derivatives", "positive", coarse, fine, coarse_error, fine_error, kOrderThreshold);
}

[[nodiscard]] CaseResult case_second_mixed() {
    const auto coarse = ref::make_anisotropic_trigonometric_fixture(false);
    const auto fine = ref::make_anisotropic_trigonometric_fixture(true);
    double coarse_error = 0.0, fine_error = 0.0;
    for (const auto axis : {ref::Axis::x, ref::Axis::y, ref::Axis::z}) {
        coarse_error = std::max(coarse_error, normalized_rms_error(ref::centered_second(coarse.grid, coarse.scalar, axis), analytic_second(coarse, axis)));
        fine_error = std::max(fine_error, normalized_rms_error(ref::centered_second(fine.grid, fine.scalar, axis), analytic_second(fine, axis)));
    }
    for (const auto axes : {std::pair{ref::Axis::x, ref::Axis::y}, std::pair{ref::Axis::x, ref::Axis::z}, std::pair{ref::Axis::y, ref::Axis::z}}) {
        coarse_error = std::max(coarse_error, normalized_rms_error(ref::centered_mixed(coarse.grid, coarse.scalar, axes.first, axes.second), analytic_mixed(coarse, axes.first, axes.second)));
        fine_error = std::max(fine_error, normalized_rms_error(ref::centered_mixed(fine.grid, fine.scalar, axes.first, axes.second), analytic_mixed(fine, axes.first, axes.second)));
    }
    return make_rate_case("second_mixed", "positive", coarse, fine, coarse_error, fine_error, kOrderThreshold);
}

[[nodiscard]] CaseResult case_diffusion() {
    const auto coarse = ref::make_anisotropic_trigonometric_fixture(false);
    const auto fine = ref::make_anisotropic_trigonometric_fixture(true);
    return make_rate_case("diffusion", "positive", coarse, fine,
                          normalized_rms_error(ref::divergence_form_diffusion(coarse.grid, coarse.q, coarse.scalar), analytic_diffusion(coarse)),
                          normalized_rms_error(ref::divergence_form_diffusion(fine.grid, fine.q, fine.scalar), analytic_diffusion(fine)),
                          kDiffusionOrderThreshold);
}

[[nodiscard]] CaseResult case_harmonic_mean() {
    const double value = ref::harmonic_mean_q(1.0, 4.0);
    return {std::abs(value - 1.6) < 1.0e-15, "harmonic_mean", "positive", "q=(1,4)", value, value, "n/a", "n/a", "exactly 1.6"};
}

[[nodiscard]] CaseResult case_cross_product() {
    const auto product = ref::cross({1.0, 2.0, 3.0}, {-2.0, 4.0, 1.0});
    const bool pass = std::abs(product.x + 10.0) < 1.0e-14 && std::abs(product.y + 7.0) < 1.0e-14 && std::abs(product.z - 8.0) < 1.0e-14;
    return {pass, "cross_product", "positive", "vectors=2", std::sqrt(product.x * product.x + product.y * product.y + product.z * product.z), 0.0, "n/a", "n/a", "(-10,-7,8)"};
}

[[nodiscard]] CaseResult case_mutation_sign() {
    const auto coarse = ref::make_cubic_trigonometric_fixture(16);
    const auto exact = analytic_diffusion(coarse);
    const double correct = normalized_rms_error(ref::divergence_form_diffusion(coarse.grid, coarse.q, coarse.scalar), exact);
    const double mutant = normalized_rms_error(sign_mutant_diffusion(coarse), exact);
    return {correct < 0.25 && mutant > 0.5, "mutation_sign", "negative-oracle", describe_grid(coarse.grid), correct, mutant, "n/a", "n/a", "correct<0.25; mutant>0.5"};
}

[[nodiscard]] CaseResult case_mutation_wrap() {
    const auto coarse = ref::make_cubic_trigonometric_fixture(16);
    const auto exact = analytic_first(coarse, ref::Axis::x);
    const double correct = normalized_rms_error(ref::centered_first(coarse.grid, coarse.scalar, ref::Axis::x), exact);
    const double mutant = normalized_rms_error(broken_x_wrap_first(coarse), exact);
    return {correct < 0.10 && mutant > 0.10, "mutation_wrap", "negative-oracle", describe_grid(coarse.grid), correct, mutant, "n/a", "n/a", "correct<0.10; mutant>0.10"};
}

using CaseFunction = test::CaseFunction;

[[nodiscard]] const std::map<std::string, CaseFunction>& cases() {
    static const std::map<std::string, CaseFunction> registry = [] {
        std::map<std::string, CaseFunction> result{
            {"wrap_general", case_wrap_general}, {"norms_rate", case_norms_rate},
            {"first_derivative", case_first_derivative}, {"cubic_derivatives", case_cubic_derivatives}, {"second_mixed", case_second_mixed},
            {"diffusion", case_diffusion}, {"harmonic_mean", case_harmonic_mean},
            {"cross_product", case_cross_product}, {"mutation_sign", case_mutation_sign},
            {"mutation_wrap", case_mutation_wrap}};
        for (const auto& [name, function] : test::gpu_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::mean_zero_projector_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::projected_pcg_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::multigrid_reuse_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::affine_periodic_rhs_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::streamfunction_gradient_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::hessian_vector_b_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::nonlinear_sources_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::coupled_residual_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::physical_diagnostics_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::api_workspace_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::homogeneous_solver_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::picard_fixed_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::picard_adaptive_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::streamfunction_continuation_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::anderson_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::heterogeneity_continuation_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::jvp_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::gmres_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        for (const auto& [name, function] : test::newton_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction test case: " + name);
            }
        }
        return result;
    }();
    return registry;
}

// SF-20 T02 / SF-21: expensive heavy-tier cases (`anderson_stall_case_
// registry()`, see streamfunction_anderson_gpu_cases.cu; `heterogeneity_
// continuation_smoke_case_registry()`, see
// heterogeneity_continuation_gpu_cases.cu). Deliberately kept OUT of
// `cases()` above so `--list` and the no-argument "run everything" default
// stay on the fast tier; reachable only via an explicit `--case
// anderson_stall_fixture_a`/`anderson_stall_fixture_b`/
// `heterogeneity_smoke_sigma025`/`heterogeneity_smoke_sigma1`, resolved as a
// fallback in main() below.
[[nodiscard]] const std::map<std::string, CaseFunction>& heavy_cases() {
    static const std::map<std::string, CaseFunction> registry = [] {
        std::map<std::string, CaseFunction> result;
        for (const auto& [name, function] : test::anderson_stall_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction heavy test case: " + name);
            }
        }
        for (const auto& [name, function] : test::heterogeneity_continuation_smoke_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction heavy test case: " + name);
            }
        }
        for (const auto& [name, function] : test::newton_difficult_case_registry()) {
            if (!result.emplace(name, function).second) {
                throw std::logic_error("duplicate streamfunction heavy test case: " + name);
            }
        }
        return result;
    }();
    return registry;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> selected;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--list") {
            if (argc != 2) { std::cerr << "--list cannot be combined with other arguments\n"; return 2; }
            for (const auto& [name, ignored] : cases()) { (void)ignored; std::cout << name << '\n'; }
            return 0;
        }
        if (argument == "--case") {
            if (++i == argc) { std::cerr << "--case requires a case name\n"; return 2; }
            selected.emplace_back(argv[i]);
            continue;
        }
        std::cerr << "unknown argument: " << argument << '\n';
        return 2;
    }
    if (selected.empty()) for (const auto& [name, ignored] : cases()) { (void)ignored; selected.push_back(name); }
    bool pass = true;
    try {
        for (const auto& name : selected) {
            auto found = cases().find(name);
            if (found == cases().end()) {
                // Explicit --case fallback only (never part of the default
                // "run everything" listing): SF-20 heavy stall fixtures.
                found = heavy_cases().find(name);
                if (found == heavy_cases().end()) {
                    std::cerr << "unknown case: " << name << '\n';
                    return 2;
                }
            }
            const auto result = found->second();
            print_result(result);
            pass = pass && result.pass;
        }
    } catch (const std::exception& error) {
        std::cerr << "test exception: " << error.what() << '\n';
        return 1;
    }
    return pass ? 0 : 1;
}
