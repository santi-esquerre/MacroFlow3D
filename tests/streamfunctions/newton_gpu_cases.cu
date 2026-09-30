#include "streamfunction_operator_test_cases.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/blas/scal.cuh"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/physics/streamfunctions/BlockDiagonalMGPreconditioner.cuh"
#include "src/physics/streamfunctions/CoupledGmres.cuh"
#include "src/physics/streamfunctions/JacobianVectorProduct.cuh"
#include "src/physics/streamfunctions/NewtonKrylovSolver.cuh"
#include "src/physics/streamfunctions/StreamfunctionSolver.cuh"
#include "src/physics/streamfunctions/StreamfunctionTypes.hpp"
#include "src/physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "src/physics/streamfunctions/affine_gauge.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// SF-24 T03: globalized Newton-Krylov (`NewtonKrylovSolver.cuh/.cu`, SF-24
// T01) PRESPECIFIED-gate GPU acceptance cases (G1-G7), bound in the SF-24
// increment bitácora BEFORE any run. Every fixture, constant, and threshold
// below is implemented VERBATIM per the handed-down specification; no value
// is retuned after seeing a result (STOP semantics, mirroring
// gmres_gpu_cases.cu/streamfunction_anderson_gpu_cases.cu). This file never
// touches `src/**`.
//
// Fixture provenance:
//   - the 16^3 (amplitude 0.25) and 32^3 (amplitude 0.5) manufactured trig-K
//     fixtures (G1/G2/G3) are copied VERBATIM (grid, K field, uniform Darcy
//     v=(1,0,0), benchmark gauge) from
//     `streamfunction_anderson_gpu_cases.cu`'s `case_anderson_disabled_
//     equivalence` (n=16, amplitude=0.25) and `case_anderson_non_regression_
//     trig32` (n=32, amplitude=0.5), both SF-20 T02 accepted converging
//     adaptive-Picard fixtures;
//   - the "difficult case" fixture (G4/G5) is copied VERBATIM from
//     `streamfunction_anderson_gpu_cases.cu`'s `run_anderson_stall_fixture`
//     as instantiated for `anderson_stall_fixture_a` (32^3, dx=1, SF-18
//     periodic Gaussian field sigma2=1, corr_length=8, seed=12345,
//     normalize_variance=true, Y scaled by lambda*=0.1125, K=exp(Y_lambda),
//     SF-19 affine flow qbar=(1,0,0), linear rtol=1e-11, max_iter=2000,
//     check_every=10, benchmark(1) gauge, log_conductivity_y representation);
//   - the homogeneous K=1 "never-activated" control (G7) is copied VERBATIM
//     from the same file's `HomogeneousProblemBuffers`/`homogeneous_problem_
//     view` (`case_anderson_non_regression_homogeneous16`'s fixture, where
//     the coupled residual is exactly zero at Picard state 0).

namespace macroflow3d::streamfunctions::test {
namespace {

static_assert(std::is_same_v<real, double>, "SF-24 Newton controls require double precision");

constexpr double kPi = 3.14159265358979323846264338327950288;

// ---------------------------------------------------------------------------
// Grid / BC / download / bitwise-equality helpers (mirrors gmres_gpu_cases.cu
// / streamfunction_anderson_gpu_cases.cu).
// ---------------------------------------------------------------------------

[[nodiscard]] Grid3D isotropic_grid(int n, real domain_length = real{1}) {
    const real h = domain_length / static_cast<real>(n);
    return Grid3D{n, n, n, h, h, h};
}

[[nodiscard]] std::size_t compact_mac_u_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx + 1) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz);
}
[[nodiscard]] std::size_t compact_mac_v_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny + 1) *
           static_cast<std::size_t>(grid.nz);
}
[[nodiscard]] std::size_t compact_mac_w_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz + 1);
}

[[nodiscard]] BCSpec triply_periodic() {
    BCSpec bc;
    bc.xmin = BCFace(BCType::Periodic, real{0});
    bc.xmax = BCFace(BCType::Periodic, real{0});
    bc.ymin = BCFace(BCType::Periodic, real{0});
    bc.ymax = BCFace(BCType::Periodic, real{0});
    bc.zmin = BCFace(BCType::Periodic, real{0});
    bc.zmax = BCFace(BCType::Periodic, real{0});
    return bc;
}

[[nodiscard]] std::vector<real> download(const DeviceSpan<const real>& span) {
    std::vector<real> host(span.size());
    if (!host.empty()) {
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(host.data(), span.data(), host.size() * sizeof(real), cudaMemcpyDeviceToHost));
    }
    return host;
}

[[nodiscard]] bool bitwise_equal(const std::vector<real>& a, const std::vector<real>& b) {
    if (a.size() != b.size()) return false;
    if (a.empty()) return true;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(real)) == 0;
}

[[nodiscard]] double rms_diff(const std::vector<real>& a, const std::vector<real>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    long double sum = 0.0L;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const long double d = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
        sum += d * d;
    }
    return static_cast<double>(std::sqrt(static_cast<double>(sum / static_cast<long double>(a.size()))));
}

// SF-22 D1 weighted coupled norm (g1 = v_rms, g2 = 1): ||(a,b)||_w =
// sqrt((RMS(a)/v_rms)^2 + RMS(b)^2).
[[nodiscard]] double weighted_norm(double rms_a, double rms_b, double v_rms) {
    const double term1 = rms_a / v_rms;
    return std::sqrt(term1 * term1 + rms_b * rms_b);
}

[[nodiscard]] const char* status_label(StreamfunctionSolveStatus status) {
    switch (status) {
        case StreamfunctionSolveStatus::converged: return "converged";
        case StreamfunctionSolveStatus::not_converged: return "not_converged";
        case StreamfunctionSolveStatus::invalid_problem: return "invalid_problem";
        default: return "not_run";
    }
}

[[nodiscard]] const char* exit_reason_label(PicardExitReason reason) {
    switch (reason) {
        case PicardExitReason::converged: return "converged";
        case PicardExitReason::budget_exhausted: return "budget_exhausted";
        case PicardExitReason::linear_block_failure: return "linear_block_failure";
        case PicardExitReason::stagnated: return "stagnated";
        case PicardExitReason::omega_floor_rejected: return "omega_floor_rejected";
        case PicardExitReason::newton_exhausted: return "newton_exhausted";
        case PicardExitReason::newton_budget_exhausted: return "newton_budget_exhausted";
        default: return "none";
    }
}

[[nodiscard]] const char* failure_reason_label(NewtonStepFailureReason reason) {
    switch (reason) {
        case NewtonStepFailureReason::none: return "none";
        case NewtonStepFailureReason::gmres_nonfinite: return "gmres_nonfinite";
        case NewtonStepFailureReason::gmres_no_reduction: return "gmres_no_reduction";
        case NewtonStepFailureReason::line_search_exhausted: return "line_search_exhausted";
        default: return "unknown";
    }
}

[[nodiscard]] const char* trial_outcome_label(PicardTrialOutcome outcome) {
    switch (outcome) {
        case PicardTrialOutcome::accepted: return "accepted";
        case PicardTrialOutcome::rejected_nonfinite: return "rejected_nonfinite";
        case PicardTrialOutcome::rejected_degeneracy: return "rejected_degeneracy";
        case PicardTrialOutcome::rejected_percentile: return "rejected_percentile";
        case PicardTrialOutcome::rejected_armijo: return "rejected_armijo";
        default: return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Manufactured heterogeneous trig-K problem, copied VERBATIM from
// streamfunction_anderson_gpu_cases.cu's ManufacturedProblemBuffers (SF-20
// T02, case_anderson_disabled_equivalence / case_anderson_non_regression_
// trig32): K = exp(a*sin(2pi x)sin(2pi y)sin(2pi z)), uniform Darcy
// v=(1,0,0), benchmark(1) gauge, triply periodic.
// ---------------------------------------------------------------------------

struct ManufacturedProblemBuffers {
    DeviceBuffer<real> conductivity;
    DeviceBuffer<real> darcy_u;
    DeviceBuffer<real> darcy_v;
    DeviceBuffer<real> darcy_w;

    explicit ManufacturedProblemBuffers(const Grid3D& grid, double amplitude)
        : conductivity(grid.num_cells()), darcy_u(compact_mac_u_size(grid)),
          darcy_v(compact_mac_v_size(grid)), darcy_w(compact_mac_w_size(grid)) {
        std::vector<real> k_host(grid.num_cells());
        for (int iz = 0; iz < grid.nz; ++iz) {
            const double z = (iz + 0.5) * static_cast<double>(grid.dz);
            for (int iy = 0; iy < grid.ny; ++iy) {
                const double y = (iy + 0.5) * static_cast<double>(grid.dy);
                for (int ix = 0; ix < grid.nx; ++ix) {
                    const double x = (ix + 0.5) * static_cast<double>(grid.dx);
                    const double log_k = amplitude * std::sin(2.0 * kPi * x) *
                                          std::sin(2.0 * kPi * y) * std::sin(2.0 * kPi * z);
                    k_host[grid.idx(ix, iy, iz)] = static_cast<real>(std::exp(log_k));
                }
            }
        }
        const std::vector<real> ones_u(compact_mac_u_size(grid), real{1});
        const std::vector<real> zeros_v(compact_mac_v_size(grid), real{0});
        const std::vector<real> zeros_w(compact_mac_w_size(grid), real{0});
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(conductivity.data(), k_host.data(),
                                          k_host.size() * sizeof(real), cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_u.data(), ones_u.data(), ones_u.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_v.data(), zeros_v.data(), zeros_v.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_w.data(), zeros_w.data(), zeros_w.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
    }
};

[[nodiscard]] StreamfunctionProblemView manufactured_problem_view(
    const Grid3D& grid, const ManufacturedProblemBuffers& buffers) {
    StreamfunctionProblemView problem;
    problem.grid = grid;
    problem.conductivity = DeviceSpan<const real>(buffers.conductivity.span());
    problem.conductivity_representation = ConductivityRepresentation::conductivity_k;
    problem.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(buffers.darcy_u.span()),
                                                         DeviceSpan<const real>(buffers.darcy_v.span()),
                                                         DeviceSpan<const real>(buffers.darcy_w.span())};
    problem.bc = triply_periodic();
    problem.gauge = AffineGauge::benchmark(real{1});
    return problem;
}

// Homogeneous K=1 control, copied VERBATIM from streamfunction_anderson_gpu_
// cases.cu's HomogeneousProblemBuffers/homogeneous_problem_view: the coupled
// residual is exactly zero at Picard state 0, so this converges in zero
// Picard update steps.
struct HomogeneousProblemBuffers {
    DeviceBuffer<real> conductivity;
    DeviceBuffer<real> darcy_u;
    DeviceBuffer<real> darcy_v;
    DeviceBuffer<real> darcy_w;

    explicit HomogeneousProblemBuffers(const Grid3D& grid)
        : conductivity(grid.num_cells()), darcy_u(compact_mac_u_size(grid)),
          darcy_v(compact_mac_v_size(grid)), darcy_w(compact_mac_w_size(grid)) {
        const std::vector<real> ones_cells(grid.num_cells(), real{1});
        const std::vector<real> ones_u(compact_mac_u_size(grid), real{1});
        const std::vector<real> zeros_v(compact_mac_v_size(grid), real{0});
        const std::vector<real> zeros_w(compact_mac_w_size(grid), real{0});
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(conductivity.data(), ones_cells.data(),
                                          ones_cells.size() * sizeof(real), cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_u.data(), ones_u.data(), ones_u.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_v.data(), zeros_v.data(), zeros_v.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_w.data(), zeros_w.data(), zeros_w.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
    }
};

[[nodiscard]] StreamfunctionProblemView homogeneous_problem_view(const Grid3D& grid,
                                                                  const HomogeneousProblemBuffers& buffers) {
    StreamfunctionProblemView problem;
    problem.grid = grid;
    problem.conductivity = DeviceSpan<const real>(buffers.conductivity.span());
    problem.conductivity_representation = ConductivityRepresentation::conductivity_k;
    problem.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(buffers.darcy_u.span()),
                                                         DeviceSpan<const real>(buffers.darcy_v.span()),
                                                         DeviceSpan<const real>(buffers.darcy_w.span())};
    problem.bc = triply_periodic();
    problem.gauge = AffineGauge::benchmark(real{1});
    return problem;
}

[[nodiscard]] StreamfunctionProblemView nullptr_problem_view(const Grid3D& grid) {
    StreamfunctionProblemView problem;
    problem.grid = grid;
    problem.conductivity = DeviceSpan<const real>(nullptr, grid.num_cells());
    problem.conductivity_representation = ConductivityRepresentation::conductivity_k;
    problem.darcy_velocity = CompactMacVelocityConstView{
        DeviceSpan<const real>(nullptr, compact_mac_u_size(grid)),
        DeviceSpan<const real>(nullptr, compact_mac_v_size(grid)),
        DeviceSpan<const real>(nullptr, compact_mac_w_size(grid))};
    problem.bc = triply_periodic();
    problem.gauge = AffineGauge::benchmark(real{1});
    return problem;
}

template <typename Callable>
[[nodiscard]] bool rejects_with_invalid_argument(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "newton_fail_fast name=" << name << " exception=none expected=std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument& error) {
        std::cout << "newton_fail_fast name=" << name
                  << " exception=std::invalid_argument message=" << error.what() << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cout << "newton_fail_fast name=" << name << " exception=std::exception message="
                  << error.what() << " expected=std::invalid_argument\n";
        return false;
    } catch (...) {
        std::cout << "newton_fail_fast name=" << name
                  << " exception=non-standard expected=std::invalid_argument\n";
        return false;
    }
}

// ===========================================================================
// G1: newton_equivalence
// ===========================================================================

[[nodiscard]] bool run_equivalence_fixture(int n, double amplitude, std::ostream& log) {
    const Grid3D grid = isotropic_grid(n);

    CudaContext ctx_a(0);
    ManufacturedProblemBuffers buffers_a(grid, amplitude);
    const StreamfunctionProblemView problem_a = manufactured_problem_view(grid, buffers_a);
    StreamfunctionFields fields_a;
    StreamfunctionWorkspace workspace_a;
    const StreamfunctionSolverConfig config_a{}; // newton disabled (default).
    const StreamfunctionSolveReport report_a =
        solve_streamfunctions(ctx_a, problem_a, config_a, fields_a, workspace_a);
    ctx_a.synchronize();

    CudaContext ctx_b(0);
    ManufacturedProblemBuffers buffers_b(grid, amplitude);
    const StreamfunctionProblemView problem_b = manufactured_problem_view(grid, buffers_b);
    StreamfunctionFields fields_b;
    StreamfunctionWorkspace workspace_b;
    StreamfunctionSolverConfig config_b{};
    config_b.newton.enabled = true; // all other newton fields default.
    const StreamfunctionSolveReport report_b =
        solve_streamfunctions(ctx_b, problem_b, config_b, fields_b, workspace_b);
    ctx_b.synchronize();

    bool pass = true;
    const auto check = [&](const char* name, bool ok) {
        pass = pass && ok;
        log << "  check n=" << n << " " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    check("a_converged", report_a.status == StreamfunctionSolveStatus::converged);
    check("b_converged", report_b.status == StreamfunctionSolveStatus::converged);
    check("a_r_F_le_1e-6", static_cast<double>(report_a.residual.r_F) <= 1e-6);
    check("b_r_F_le_1e-6", static_cast<double>(report_b.residual.r_F) <= 1e-6);
    check("b_newton_activations_ge_1", report_b.newton_activations >= 1);

    const std::vector<real> u1_a = download(fields_a.u1_span());
    const std::vector<real> u2_a = download(fields_a.u2_span());
    const std::vector<real> u1_b = download(fields_b.u1_span());
    const std::vector<real> u2_b = download(fields_b.u2_span());
    const double v_rms = static_cast<double>(report_a.diagnostics.v_d_rms);
    const double rms_u1 = rms_diff(u1_b, u1_a);
    const double rms_u2 = rms_diff(u2_b, u2_a);
    const double field_norm_w = weighted_norm(rms_u1, rms_u2, v_rms);
    check("weighted_field_diff_le_1e-3", field_norm_w <= 1e-3);

    const double e_v_a = static_cast<double>(report_a.diagnostics.e_v);
    const double e_v_b = static_cast<double>(report_b.diagnostics.e_v);
    constexpr double kTiny = 1e-300;
    const double e_v_tol = 1e-3 * std::max(std::abs(e_v_a), kTiny);
    check("e_v_relative_agreement", std::abs(e_v_b - e_v_a) <= e_v_tol);

    const StreamfunctionResidualReport& res_a = report_a.residual;
    const StreamfunctionResidualReport& res_b = report_b.residual;
    const double p_a = static_cast<double>(residual_histogram_percentile(res_a, real{0.001}));
    const double p_b = static_cast<double>(residual_histogram_percentile(res_b, real{0.001}));
    const double p_tol = 1e-3 * std::max(std::abs(p_a), kTiny);
    check("c_percentile_relative_agreement", std::abs(p_b - p_a) <= p_tol);

    log << std::setprecision(17) << "newton_equivalence n=" << n << " amplitude=" << amplitude
        << " a_status=" << status_label(report_a.status)
        << " a_picard_iterations=" << report_a.picard_iterations << " a_r_F=" << report_a.residual.r_F
        << " b_status=" << status_label(report_b.status)
        << " b_picard_iterations=" << report_b.picard_iterations << " b_r_F=" << report_b.residual.r_F
        << " b_newton_activations=" << report_b.newton_activations
        << " weighted_field_diff=" << field_norm_w << " v_rms=" << v_rms << " e_v_a=" << e_v_a
        << " e_v_b=" << e_v_b << " p_a=" << p_a << " p_b=" << p_b << '\n';

    return pass;
}

[[nodiscard]] CaseResult case_newton_equivalence() {
    std::ostringstream log;
    const bool pass16 = run_equivalence_fixture(16, 0.25, log);
    const bool pass32 = run_equivalence_fixture(32, 0.5, log);
    const bool pass = pass16 && pass32;
    std::cout << log.str();
    std::cout << "case=newton_equivalence verdict=" << (pass ? "PASS" : "FAIL") << '\n';

    return {pass,
            "newton_equivalence",
            "gpu-newton-equivalence",
            "16^3 (a=0.25) and 32^3 (a=0.5) manufactured trig-K fixtures (copied verbatim from "
            "streamfunction_anderson_gpu_cases.cu)",
            pass16 ? 1.0 : 0.0,
            pass32 ? 1.0 : 0.0,
            "G1",
            pass ? "all pass" : "some failed",
            "Newton-enabled and Newton-disabled solves converge to r_F<=1e-6 with weighted field "
            "agreement <=1e-3 and relative e_v/|c|-percentile agreement <=1e-3"};
}

// ===========================================================================
// G2: newton_activation_semantics
// ===========================================================================

[[nodiscard]] CaseResult case_newton_activation_semantics() {
    constexpr int n = 16;
    constexpr double amplitude = 0.25;
    const Grid3D grid = isotropic_grid(n);

    CudaContext ctx(0);
    ManufacturedProblemBuffers buffers(grid, amplitude);
    const StreamfunctionProblemView problem = manufactured_problem_view(grid, buffers);
    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    StreamfunctionSolverConfig config{};
    config.newton.enabled = true;
    const StreamfunctionSolveReport report = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();

    bool pass = true;
    const auto check = [&](const char* name, bool ok) {
        pass = pass && ok;
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    check("converged", report.status == StreamfunctionSolveStatus::converged);
    check("exactly_one_activation", report.newton_activations == 1);
    check("history_nonempty", !report.newton_history.empty());

    if (!report.newton_history.empty()) {
        const int first_context = report.newton_history.front().picard_iteration_context;

        // Every record's head r_F must satisfy the accepted-state r_F at its
        // context <= activation_r_F (converging fixture: no stagnation-
        // redirect can fire, so newton_activations==1 already pins "no
        // forced retry"; this checks the activation TRIGGER condition (a)).
        bool every_record_meets_activation = true;
        for (const auto& record : report.newton_history) {
            const std::size_t ctx_idx = static_cast<std::size_t>(record.picard_iteration_context);
            const bool head_ok = ctx_idx < report.picard_history.size() &&
                                 static_cast<double>(report.picard_history[ctx_idx].r_F) <=
                                     static_cast<double>(config.newton.activation_r_F);
            every_record_meets_activation = every_record_meets_activation && head_ok;
        }
        check("every_record_meets_activation_threshold", every_record_meets_activation);

        // First head k* with picard_history[k*].r_F <= activation_r_F equals
        // first_context.
        int k_star = -1;
        for (std::size_t k = 0; k < report.picard_history.size(); ++k) {
            if (static_cast<double>(report.picard_history[k].r_F) <=
                static_cast<double>(config.newton.activation_r_F)) {
                k_star = static_cast<int>(k);
                break;
            }
        }
        check("first_activation_head_matches", k_star == first_context);

        bool forcing_exact = true;
        for (const auto& record : report.newton_history) {
            const real expected = std::clamp(config.newton.forcing_coefficient * std::sqrt(record.r_F),
                                             config.newton.forcing_min, config.newton.forcing_max);
            forcing_exact = forcing_exact && (expected == record.forcing_rel_tol);
        }
        check("forcing_rel_tol_exact_host_recompute", forcing_exact);

        std::cout << std::setprecision(17) << "newton_activation_semantics n=" << n
                  << " first_context=" << first_context << " k_star=" << k_star
                  << " history_size=" << report.newton_history.size() << '\n';
        for (std::size_t i = 0; i < report.newton_history.size(); ++i) {
            const auto& record = report.newton_history[i];
            std::cout << "  record[" << i << "] context=" << record.picard_iteration_context
                      << " r_F=" << record.r_F << " forcing_rel_tol=" << record.forcing_rel_tol << '\n';
        }
    }

    std::cout << "case=newton_activation_semantics verdict=" << (pass ? "PASS" : "FAIL") << '\n';

    return {pass,
            "newton_activation_semantics",
            "gpu-newton-activation",
            "16^3 (a=0.25) manufactured trig-K fixture, newton enabled",
            static_cast<double>(report.newton_activations),
            static_cast<double>(report.newton_history.size()),
            "G2",
            pass ? "all pass" : "some failed",
            "single activation at the first head with r_F<=activation_r_F; every record's forcing_"
            "rel_tol equals the host-recomputed clamp expression exactly"};
}

// ===========================================================================
// G3: newton_forced_failure_preservation (three variants on the 16^3
// fixture)
//
// SF-24 E13 (recorded amendment, corrective C02): the original mechanism used
// `armijo_c = 0.999999` alone to force every line-search trial to be
// rejected. That is invalid for damped alphas: the Armijo-on-merit test
// accepts iff `phi_trial <= (1 - c*alpha) * phi_k`. At alpha=1, c=0.999999
// demands an r_F reduction factor of ~1e-3 (correctly rejected), but at
// alpha=0.5 the threshold relaxes to `(1 - c/2) ~ 0.5` on phi (~0.71 on
// r_F), which any (1-alpha)-damped Newton step satisfies -- so the ladder
// always accepted at alpha=0.5 and the solves converged instead of failing.
// The corrected mechanism forces deterministic single-trial rejection
// instead: `alpha_min = 1` restricts the line search to exactly one trial at
// alpha=1 (the loop `for (alpha = 1; alpha >= alpha_min; alpha *=
// backtrack_factor)` then runs once), `armijo_c = 1 - 1e-8` requires that
// single trial to achieve `phi_t <= 1e-8 * phi_k` (an r_F factor of 1e-4),
// and a loose fixed GMRES tolerance (`forcing_min = forcing_max = 1e-1`)
// floors the achievable single-step r_F reduction near ~1e-1..1e-2 -- about
// 200x above the 1e-4 rejection threshold, so the trial is rejected
// deterministically and the outer Newton step is exhausted after exactly one
// trial.
// ===========================================================================

[[nodiscard]] CaseResult case_newton_forced_failure_preservation() {
    constexpr int n = 16;
    constexpr double amplitude = 0.25;
    const Grid3D grid = isotropic_grid(n);

    bool pass = true;
    const auto check = [&](const char* label, const char* name, bool ok) {
        pass = pass && ok;
        std::cout << "  check " << label << "." << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    // --- run1: newton disabled, full solve. ---------------------------------
    CudaContext ctx1(0);
    ManufacturedProblemBuffers buffers1(grid, amplitude);
    const StreamfunctionProblemView problem1 = manufactured_problem_view(grid, buffers1);
    StreamfunctionFields fields1;
    StreamfunctionWorkspace workspace1;
    const StreamfunctionSolverConfig config1{};
    const StreamfunctionSolveReport report1 =
        solve_streamfunctions(ctx1, problem1, config1, fields1, workspace1);
    ctx1.synchronize();

    int k_star = -1;
    for (std::size_t k = 0; k < report1.picard_history.size(); ++k) {
        if (static_cast<double>(report1.picard_history[k].r_F) <= 1e-2) {
            k_star = static_cast<int>(k);
            break;
        }
    }
    check("setup", "run1_converged", report1.status == StreamfunctionSolveStatus::converged);
    check("setup", "k_star_found", k_star >= 0);
    std::cout << std::setprecision(17) << "newton_forced_failure_preservation k_star=" << k_star << '\n';
    if (k_star < 0) {
        std::cout << "case=newton_forced_failure_preservation verdict=FAIL (k_star not found)\n";
        return {false,
                "newton_forced_failure_preservation",
                "gpu-newton-failure-preservation",
                "16^3 (a=0.25)",
                0.0,
                0.0,
                "G3",
                "FAIL",
                "k_star (first head with r_F<=1e-2) not found in run1's converging fixture"};
    }

    // --- run2: newton disabled, picard.max_iter = k_star -> budget_exhausted
    //     AT the state Newton would have entered. F0 is the resulting state.
    CudaContext ctx2(0);
    ManufacturedProblemBuffers buffers2(grid, amplitude);
    const StreamfunctionProblemView problem2 = manufactured_problem_view(grid, buffers2);
    StreamfunctionFields fields2;
    StreamfunctionWorkspace workspace2;
    StreamfunctionSolverConfig config2{};
    config2.picard.max_iter = k_star;
    const StreamfunctionSolveReport report2 =
        solve_streamfunctions(ctx2, problem2, config2, fields2, workspace2);
    ctx2.synchronize();
    check("setup", "run2_budget_exhausted_at_k_star",
          report2.status == StreamfunctionSolveStatus::not_converged &&
              report2.exit_reason == PicardExitReason::budget_exhausted &&
              report2.picard_iterations == k_star);
    const std::vector<real> f0_u1 = download(fields2.u1_span());
    const std::vector<real> f0_u2 = download(fields2.u2_span());

    // --- Variant A: line-search forced failure, bitwise preservation. -------
    {
        CudaContext ctx3(0);
        ManufacturedProblemBuffers buffers3(grid, amplitude);
        const StreamfunctionProblemView problem3 = manufactured_problem_view(grid, buffers3);
        StreamfunctionFields fields3;
        StreamfunctionWorkspace workspace3;
        StreamfunctionSolverConfig config3{};
        config3.newton.enabled = true;
        config3.newton.alpha_min = real{1};
        config3.newton.armijo_c = real{1} - real{1e-8};
        config3.newton.forcing_min = real{1e-1};
        config3.newton.forcing_max = real{1e-1};
        config3.newton.rescue_picard_steps = 0;
        const StreamfunctionSolveReport report3 =
            solve_streamfunctions(ctx3, problem3, config3, fields3, workspace3);
        ctx3.synchronize();

        check("A", "exit_reason_newton_exhausted",
              report3.status == StreamfunctionSolveStatus::not_converged &&
                  report3.exit_reason == PicardExitReason::newton_exhausted);
        check("A", "activations_eq_2", report3.newton_activations == 2);
        check("A", "rescue_events_eq_1", report3.newton_rescue_events == 1);
        check("A", "step_failures_eq_2", report3.newton_step_failures == 2);
        check("A", "history_size_eq_2", report3.newton_history.size() == 2);

        bool both_line_search_exhausted = report3.newton_history.size() == 2;
        bool ladder_ok = true;
        for (const auto& record : report3.newton_history) {
            both_line_search_exhausted = both_line_search_exhausted &&
                                         record.step_failed &&
                                         record.failure_reason ==
                                             NewtonStepFailureReason::line_search_exhausted;
            // SF-24 E13: with alpha_min = 1, the line search attempts exactly
            // one trial at alpha=1; the corrected armijo_c forces that single
            // trial to be rejected deterministically.
            if (record.trials.size() != 1) {
                ladder_ok = false;
                continue;
            }
            const auto& trial = record.trials.front();
            ladder_ok = ladder_ok && std::abs(static_cast<double>(trial.alpha - real{1})) < 1e-15 &&
                       trial.outcome == PicardTrialOutcome::rejected_armijo;
        }
        check("A", "both_failures_line_search_exhausted", both_line_search_exhausted);
        check("A", "single_trial_alpha_one_rejected_armijo", ladder_ok);

        const std::vector<real> u1_3 = download(fields3.u1_span());
        const std::vector<real> u2_3 = download(fields3.u2_span());
        check("A", "fields_bitwise_equal_F0",
              bitwise_equal(u1_3, f0_u1) && bitwise_equal(u2_3, f0_u2));

        std::cout << std::setprecision(17) << "  variant A: status=" << status_label(report3.status)
                  << " exit_reason=" << exit_reason_label(report3.exit_reason)
                  << " activations=" << report3.newton_activations
                  << " rescue_events=" << report3.newton_rescue_events
                  << " step_failures=" << report3.newton_step_failures << '\n';
        for (std::size_t i = 0; i < report3.newton_history.size(); ++i) {
            const auto& record = report3.newton_history[i];
            std::cout << "    record[" << i << "] context=" << record.picard_iteration_context
                      << " failure_reason=" << failure_reason_label(record.failure_reason)
                      << " trials=" << record.trials.size() << '\n';
            for (std::size_t t = 0; t < record.trials.size(); ++t) {
                const auto& trial = record.trials[t];
                std::cout << "      trial[" << t << "] alpha=" << trial.alpha
                          << " r_F_trial=" << trial.r_F_trial
                          << " outcome=" << trial_outcome_label(trial.outcome) << '\n';
            }
        }
    }

    // --- Variant B: linear-side failure taxonomy, preservation. -------------
    {
        CudaContext ctx4(0);
        ManufacturedProblemBuffers buffers4(grid, amplitude);
        const StreamfunctionProblemView problem4 = manufactured_problem_view(grid, buffers4);
        StreamfunctionFields fields4;
        StreamfunctionWorkspace workspace4;
        StreamfunctionSolverConfig config4{};
        config4.newton.enabled = true;
        config4.newton.gmres.max_iterations = 1;
        config4.newton.alpha_min = real{1};
        config4.newton.forcing_min = real{1e-1};
        config4.newton.forcing_max = real{1e-1};
        config4.newton.armijo_c = real{1} - real{1e-8};
        config4.newton.rescue_picard_steps = 0;
        const StreamfunctionSolveReport report4 =
            solve_streamfunctions(ctx4, problem4, config4, fields4, workspace4);
        ctx4.synchronize();

        check("B", "exit_reason_newton_exhausted",
              report4.status == StreamfunctionSolveStatus::not_converged &&
                  report4.exit_reason == PicardExitReason::newton_exhausted);

        bool failure_taxonomy_ok = !report4.newton_history.empty();
        for (const auto& record : report4.newton_history) {
            if (!record.step_failed) continue;
            failure_taxonomy_ok = failure_taxonomy_ok &&
                                  (record.failure_reason == NewtonStepFailureReason::gmres_no_reduction ||
                                   record.failure_reason == NewtonStepFailureReason::line_search_exhausted);
        }
        check("B", "failure_reason_in_expected_set", failure_taxonomy_ok);

        const std::vector<real> u1_4 = download(fields4.u1_span());
        const std::vector<real> u2_4 = download(fields4.u2_span());
        check("B", "fields_bitwise_equal_F0",
              bitwise_equal(u1_4, f0_u1) && bitwise_equal(u2_4, f0_u2));

        std::cout << std::setprecision(17) << "  variant B: status=" << status_label(report4.status)
                  << " exit_reason=" << exit_reason_label(report4.exit_reason) << '\n';
        for (std::size_t i = 0; i < report4.newton_history.size(); ++i) {
            const auto& record = report4.newton_history[i];
            std::cout << "    record[" << i << "] context=" << record.picard_iteration_context
                      << " failure_reason=" << failure_reason_label(record.failure_reason)
                      << " gmres_status=" << static_cast<int>(record.gmres_status) << '\n';
        }
    }

    // --- Variant C: rescue counting. -----------------------------------------
    {
        CudaContext ctx5(0);
        ManufacturedProblemBuffers buffers5(grid, amplitude);
        const StreamfunctionProblemView problem5 = manufactured_problem_view(grid, buffers5);
        StreamfunctionFields fields5;
        StreamfunctionWorkspace workspace5;
        StreamfunctionSolverConfig config5{};
        config5.newton.enabled = true;
        config5.newton.alpha_min = real{1};
        config5.newton.armijo_c = real{1} - real{1e-8};
        config5.newton.forcing_min = real{1e-1};
        config5.newton.forcing_max = real{1e-1};
        // rescue_picard_steps left at its default (5).
        const StreamfunctionSolveReport report5 =
            solve_streamfunctions(ctx5, problem5, config5, fields5, workspace5);
        ctx5.synchronize();

        check("C", "activations_eq_2", report5.newton_activations == 2);
        check("C", "rescue_events_eq_1", report5.newton_rescue_events == 1);
        check("C", "exit_reason_newton_exhausted",
              report5.status == StreamfunctionSolveStatus::not_converged &&
                  report5.exit_reason == PicardExitReason::newton_exhausted);

        int first_ctx = -1, retry_ctx = -1;
        if (report5.newton_history.size() >= 2) {
            first_ctx = report5.newton_history.front().picard_iteration_context;
            retry_ctx = report5.newton_history[1].picard_iteration_context;
        }
        const int expected_offset = config5.newton.rescue_picard_steps + 1;
        check("C", "retry_context_offset_matches_rescue_plus_one",
              first_ctx >= 0 && retry_ctx >= 0 && (retry_ctx - first_ctx) == expected_offset);

        int accepted_between = 0;
        for (const auto& trial : report5.trial_history) {
            if (trial.outcome == PicardTrialOutcome::accepted && trial.iteration > first_ctx &&
                trial.iteration < retry_ctx) {
                ++accepted_between;
            }
        }
        check("C", "exactly_5_accepted_trials_between_contexts",
              accepted_between == config5.newton.rescue_picard_steps);

        std::cout << std::setprecision(17) << "  variant C: activations=" << report5.newton_activations
                  << " rescue_events=" << report5.newton_rescue_events << " first_ctx=" << first_ctx
                  << " retry_ctx=" << retry_ctx << " accepted_between=" << accepted_between << '\n';
    }

    std::cout << "case=newton_forced_failure_preservation verdict=" << (pass ? "PASS" : "FAIL") << '\n';

    return {pass,
            "newton_forced_failure_preservation",
            "gpu-newton-failure-preservation",
            "16^3 (a=0.25) manufactured trig-K fixture, three SF-24 E13 single-trial-ladder variants "
            "(alpha_min=1, armijo_c=1-1e-8, forcing_min=forcing_max=1e-1)",
            static_cast<double>(k_star),
            0.0,
            "G3",
            pass ? "all pass" : "some failed",
            "SF-24 E13: alpha_min=1 restricts the line search to a single alpha=1 trial; armijo_c="
            "1-1e-8 demands a 1e-8 merit reduction (1e-4 r_F factor) that the 1e-1 forcing floor "
            "cannot deliver, so every Newton activation is rejected deterministically at that one "
            "trial; the accepted state stays bitwise-preserved across every failed activation; the "
            "rescue window advances the outer Picard context by exactly rescue_picard_steps+1 before "
            "the forced retry"};
}

// ===========================================================================
// SF-25 Phase-1 (S2-a) ON-path evidence: newton_rescue_omega_reset.
//
// Reuses the SAME 16^3 (amplitude 0.25) "Variant A" forced-line-search-
// failure mechanism as `case_newton_forced_failure_preservation` (SF-24
// E13): alpha_min=1, armijo_c=1-1e-8, forcing_min=forcing_max=1e-1,
// rescue_picard_steps=0, which deterministically produces exactly ONE
// rescue-window entry (newton_rescue_events==1, the degenerate 0-step
// rescue forces an immediate retry that fails again -> newton_exhausted).
// `rescue_resets_omega` threads straight through that single rescue-window
// entry point, so this fixture is a clean, deterministic ON/OFF control for
// the counter.
// ===========================================================================

[[nodiscard]] CaseResult case_newton_rescue_omega_reset() {
    constexpr int n = 16;
    constexpr double amplitude = 0.25;
    const Grid3D grid = isotropic_grid(n);

    bool pass = true;
    const auto check = [&](const char* name, bool ok) {
        pass = pass && ok;
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    const auto run = [&](bool rescue_resets_omega) {
        CudaContext ctx(0);
        ManufacturedProblemBuffers buffers(grid, amplitude);
        const StreamfunctionProblemView problem = manufactured_problem_view(grid, buffers);
        StreamfunctionFields fields;
        StreamfunctionWorkspace workspace;
        StreamfunctionSolverConfig config{};
        config.newton.enabled = true;
        config.newton.alpha_min = real{1};
        config.newton.armijo_c = real{1} - real{1e-8};
        config.newton.forcing_min = real{1e-1};
        config.newton.forcing_max = real{1e-1};
        config.newton.rescue_picard_steps = 0;
        config.newton.rescue_resets_omega = rescue_resets_omega;
        const StreamfunctionSolveReport report =
            solve_streamfunctions(ctx, problem, config, fields, workspace);
        ctx.synchronize();
        return report;
    };

    const StreamfunctionSolveReport report_off = run(/*rescue_resets_omega=*/false);
    const StreamfunctionSolveReport report_on = run(/*rescue_resets_omega=*/true);

    check("off_exit_reason_newton_exhausted",
          report_off.status == StreamfunctionSolveStatus::not_converged &&
              report_off.exit_reason == PicardExitReason::newton_exhausted);
    check("off_rescue_events_eq_1", report_off.newton_rescue_events == 1);
    check("off_newton_rescue_omega_resets_eq_0", report_off.newton_rescue_omega_resets == 0);

    check("on_exit_reason_newton_exhausted",
          report_on.status == StreamfunctionSolveStatus::not_converged &&
              report_on.exit_reason == PicardExitReason::newton_exhausted);
    check("on_rescue_events_eq_1", report_on.newton_rescue_events == 1);
    check("on_newton_rescue_omega_resets_eq_1", report_on.newton_rescue_omega_resets == 1);

    // OFF/ON must otherwise be identical (same activations/step_failures/
    // history size): the reset only touches the persistent `omega` used by
    // the (never-reached, 0-step) rescue window's Picard trials, not the
    // Newton phase itself.
    check("activations_equal", report_off.newton_activations == report_on.newton_activations);
    check("step_failures_equal", report_off.newton_step_failures == report_on.newton_step_failures);

    std::cout << std::setprecision(17) << "newton_rescue_omega_reset: off.rescue_omega_resets="
              << report_off.newton_rescue_omega_resets
              << " on.rescue_omega_resets=" << report_on.newton_rescue_omega_resets
              << " off.rescue_events=" << report_off.newton_rescue_events
              << " on.rescue_events=" << report_on.newton_rescue_events << '\n';
    std::cout << "case=newton_rescue_omega_reset verdict=" << (pass ? "PASS" : "FAIL") << '\n';

    return {pass,
            "newton_rescue_omega_reset",
            "gpu-newton-rescue-omega-reset",
            "16^3 (a=0.25) manufactured trig-K fixture, SF-24 E13 single-trial-ladder forced "
            "failure (alpha_min=1, armijo_c=1-1e-8, forcing_min=forcing_max=1e-1, "
            "rescue_picard_steps=0)",
            static_cast<double>(report_off.newton_rescue_omega_resets),
            static_cast<double>(report_on.newton_rescue_omega_resets),
            "off: newton_rescue_omega_resets==0; on: newton_rescue_omega_resets==1",
            pass ? "all pass" : "some failed",
            "SF-25 Phase-1 (S2-a): rescue_resets_omega=false reproduces newton_rescue_omega_resets"
            "==0 bitwise (the counter never increments); rescue_resets_omega=true increments it "
            "exactly once, at the single rescue-window entry this deterministic forced-failure "
            "fixture produces"};
}

// ===========================================================================
// G4/G5: newton_difficult_case (HEAVY tier, separate registry).
//
// The problem fixture below is copied VERBATIM from streamfunction_anderson_
// gpu_cases.cu's run_anderson_stall_fixture, as instantiated for
// anderson_stall_fixture_a (sigma2=1.0, lambda_star=0.1125).
// ===========================================================================

__global__ void newton_difficult_exp_kernel(const real* __restrict__ in, real* __restrict__ out,
                                            std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = exp(in[idx]);
    }
}

void newton_difficult_device_exp(CudaContext& ctx, DeviceSpan<const real> in, DeviceSpan<real> out) {
    const std::size_t n = in.size();
    constexpr int kBlock = 256;
    const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
    newton_difficult_exp_kernel<<<blocks, kBlock, 0, ctx.cuda_stream()>>>(in.data(), out.data(), n);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    ctx.synchronize();
}

struct DifficultCaseProblem {
    Grid3D grid;
    DeviceBuffer<real> y;
    DeviceBuffer<real> darcy_u, darcy_v, darcy_w;
};

[[nodiscard]] DifficultCaseProblem build_difficult_case_problem() {
    constexpr int n = 32;
    const Grid3D grid{n, n, n, real{1}, real{1}, real{1}};
    constexpr double sigma2 = 1.0;
    constexpr double lambda_star = 0.1125;

    CudaContext ctx(0);

    DifficultCaseProblem problem{grid, DeviceBuffer<real>(grid.num_cells()),
                                 DeviceBuffer<real>(compact_mac_u_size(grid)),
                                 DeviceBuffer<real>(compact_mac_v_size(grid)),
                                 DeviceBuffer<real>(compact_mac_w_size(grid))};

    physics::PeriodicGaussianFieldConfig field_cfg;
    field_cfg.sigma2 = static_cast<real>(sigma2);
    field_cfg.corr_length = real{8};
    field_cfg.seed = 12345ULL;
    field_cfg.normalize_variance = true;
    physics::PeriodicGaussianFieldWorkspace field_workspace;
    (void)physics::generate_periodic_gaussian_field(ctx, grid, field_cfg, problem.y.span(), field_workspace);
    ctx.synchronize();

    blas::scal(ctx, problem.y.span(), static_cast<real>(lambda_star));
    ctx.synchronize();

    DeviceBuffer<real> k_lambda(grid.num_cells());
    newton_difficult_device_exp(ctx, DeviceSpan<const real>(problem.y.span()), k_lambda.span());

    physics::AffinePeriodicFlowConfig flow_config;
    flow_config.qbar[0] = real{1};
    flow_config.qbar[1] = real{0};
    flow_config.qbar[2] = real{0};
    flow_config.linear.rtol = real{1e-11};
    flow_config.linear.max_iter = 2000;
    flow_config.linear.check_every = 10;

    physics::AffinePeriodicVelocityView velocity_view{problem.darcy_u.span(), problem.darcy_v.span(),
                                                       problem.darcy_w.span()};
    physics::AffinePeriodicFlowWorkspace flow_workspace;
    (void)physics::solve_affine_periodic_flow(ctx, grid, DeviceSpan<const real>(k_lambda.span()),
                                               flow_config, velocity_view, flow_workspace);
    ctx.synchronize();

    return problem;
}

[[nodiscard]] StreamfunctionProblemView difficult_case_view(const DifficultCaseProblem& problem) {
    StreamfunctionProblemView view;
    view.grid = problem.grid;
    view.conductivity = DeviceSpan<const real>(problem.y.span());
    view.conductivity_representation = ConductivityRepresentation::log_conductivity_y;
    view.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(problem.darcy_u.span()),
                                                       DeviceSpan<const real>(problem.darcy_v.span()),
                                                       DeviceSpan<const real>(problem.darcy_w.span())};
    view.bc = triply_periodic();
    view.gauge = AffineGauge::benchmark(real{1});
    return view;
}

[[nodiscard]] long long total_evals_control(const StreamfunctionSolveReport& report) {
    return static_cast<long long>(report.picard_history.size()) +
           static_cast<long long>(report.trial_history.size());
}

[[nodiscard]] long long total_evals_newton(const StreamfunctionSolveReport& report) {
    long long total = total_evals_control(report);
    total += static_cast<long long>(report.newton_history.size());
    for (const auto& record : report.newton_history) {
        total += static_cast<long long>(record.trials.size());
    }
    total += report.newton_jv_evaluations;
    return total;
}

[[nodiscard]] CaseResult case_newton_difficult_case() {
    const DifficultCaseProblem problem = build_difficult_case_problem();

    // CONTROL: the SF-20 gate config (anderson enabled depth 5, start 5,
    // condition_limit 1e12; everything else default).
    StreamfunctionSolverConfig control_config{};
    control_config.anderson.enabled = true;
    control_config.anderson.depth = 5;
    control_config.anderson.start_iteration = 5;
    control_config.anderson.condition_limit = real{1e12};

    // NEWTON: CONTROL + newton.enabled = true (all newton defaults).
    StreamfunctionSolverConfig newton_config = control_config;
    newton_config.newton.enabled = true;

    CudaContext ctx_control(0);
    const StreamfunctionProblemView problem_control = difficult_case_view(problem);
    StreamfunctionFields fields_control;
    StreamfunctionWorkspace workspace_control;
    const auto ct0 = std::chrono::steady_clock::now();
    const StreamfunctionSolveReport report_control =
        solve_streamfunctions(ctx_control, problem_control, control_config, fields_control, workspace_control);
    ctx_control.synchronize();
    const auto ct1 = std::chrono::steady_clock::now();
    const double wall_control = std::chrono::duration<double>(ct1 - ct0).count();

    CudaContext ctx_newton(0);
    const StreamfunctionProblemView problem_newton = difficult_case_view(problem);
    StreamfunctionFields fields_newton;
    StreamfunctionWorkspace workspace_newton;
    const auto nt0 = std::chrono::steady_clock::now();
    const StreamfunctionSolveReport report_newton =
        solve_streamfunctions(ctx_newton, problem_newton, newton_config, fields_newton, workspace_newton);
    ctx_newton.synchronize();
    const auto nt1 = std::chrono::steady_clock::now();
    const double wall_newton = std::chrono::duration<double>(nt1 - nt0).count();

    bool pass = true;
    const auto check = [&](const char* name, bool ok) {
        pass = pass && ok;
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    check("control_converged", report_control.status == StreamfunctionSolveStatus::converged &&
                                   static_cast<double>(report_control.residual.r_F) <= 1e-6);
    check("newton_converged", report_newton.status == StreamfunctionSolveStatus::converged &&
                                  static_cast<double>(report_newton.residual.r_F) <= 1e-6);

    const long long evals_control = total_evals_control(report_control);
    const long long evals_newton = total_evals_newton(report_newton);
    const bool evals_gate = evals_newton < evals_control;
    const bool wall_gate = wall_newton < wall_control;
    check("evals_or_wall_improvement", evals_gate || wall_gate);

    std::cout << std::setprecision(17) << "newton_difficult_case CONTROL: status="
              << status_label(report_control.status) << " r_F=" << report_control.residual.r_F
              << " picard_iterations=" << report_control.picard_iterations
              << " evals=" << evals_control << " wall_seconds=" << wall_control << '\n';
    std::cout << "newton_difficult_case NEWTON:  status=" << status_label(report_newton.status)
              << " r_F=" << report_newton.residual.r_F
              << " picard_iterations=" << report_newton.picard_iterations
              << " evals=" << evals_newton << " wall_seconds=" << wall_newton
              << " newton_activations=" << report_newton.newton_activations
              << " newton_steps_accepted=" << report_newton.newton_steps_accepted
              << " newton_jv_evaluations=" << report_newton.newton_jv_evaluations << '\n';
    for (std::size_t i = 0; i < report_newton.newton_history.size(); ++i) {
        const auto& record = report_newton.newton_history[i];
        std::cout << "  newton_history[" << i << "] context=" << record.picard_iteration_context
                  << " r_F=" << record.r_F << " forcing_rel_tol=" << record.forcing_rel_tol
                  << " gmres_inner=" << record.gmres_total_inner_iterations
                  << " gmres_status=" << static_cast<int>(record.gmres_status)
                  << " accepted_alpha=" << record.accepted_alpha << '\n';
    }

    // G5: determinism. Rerun NEWTON a second time from fresh fields/workspace.
    CudaContext ctx_newton2(0);
    const StreamfunctionProblemView problem_newton2 = difficult_case_view(problem);
    StreamfunctionFields fields_newton2;
    StreamfunctionWorkspace workspace_newton2;
    const StreamfunctionSolveReport report_newton2 = solve_streamfunctions(
        ctx_newton2, problem_newton2, newton_config, fields_newton2, workspace_newton2);
    ctx_newton2.synchronize();

    const std::vector<real> u1_first = download(fields_newton.u1_span());
    const std::vector<real> u2_first = download(fields_newton.u2_span());
    const std::vector<real> u1_second = download(fields_newton2.u1_span());
    const std::vector<real> u2_second = download(fields_newton2.u2_span());

    long long total_inner_first = 0, total_inner_second = 0;
    for (const auto& record : report_newton.newton_history) {
        total_inner_first += record.gmres_total_inner_iterations;
    }
    for (const auto& record : report_newton2.newton_history) {
        total_inner_second += record.gmres_total_inner_iterations;
    }

    check("determinism_fields_bitwise",
          bitwise_equal(u1_first, u1_second) && bitwise_equal(u2_first, u2_second));
    check("determinism_r_F", report_newton.residual.r_F == report_newton2.residual.r_F);
    check("determinism_picard_iterations",
          report_newton.picard_iterations == report_newton2.picard_iterations);
    check("determinism_newton_activations",
          report_newton.newton_activations == report_newton2.newton_activations);
    check("determinism_newton_steps_accepted",
          report_newton.newton_steps_accepted == report_newton2.newton_steps_accepted);
    check("determinism_newton_jv_evaluations",
          report_newton.newton_jv_evaluations == report_newton2.newton_jv_evaluations);
    check("determinism_total_inner_gmres_iterations", total_inner_first == total_inner_second);

    std::cout << "case=newton_difficult_case verdict=" << (pass ? "PASS" : "FAIL") << '\n';

    std::ostringstream detail;
    detail << "32^3, SF-18 Gaussian field sigma2=1 corr_length=8 seed=12345 lambda*=0.1125, "
              "SF-19 affine flow qbar=(1,0,0), SF-20 anderson gate config vs +newton defaults";

    return {pass,
            "newton_difficult_case",
            "gpu-newton-difficult",
            detail.str(),
            static_cast<double>(evals_control),
            static_cast<double>(evals_newton),
            "G4/G5",
            pass ? "all pass" : "some failed",
            "both CONTROL (SF-20 anderson gate) and NEWTON (+newton defaults) converge to r_F<=1e-6; "
            "NEWTON reduces total evaluations or wall time; a second NEWTON run reproduces bitwise-"
            "identical fields and identical scalar counters (determinism)"};
}

// ===========================================================================
// G6: newton_memory_accounting
// ===========================================================================

[[nodiscard]] CaseResult case_newton_memory_accounting() {
    bool pass = true;
    const auto check = [&](const char* name, bool ok) {
        pass = pass && ok;
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    for (const int n : {16, 32}) {
        const Grid3D grid = isotropic_grid(n);
        const std::size_t cells = grid.num_cells();

        StreamfunctionSolverConfig config{};
        config.newton.enabled = true;

        StreamfunctionFields fields;
        StreamfunctionWorkspace workspace;
        fields.prepare(grid);
        workspace.prepare(grid, config);

        const StreamfunctionMemoryReport estimate = estimate_streamfunction_memory(grid, config);
        const StreamfunctionMemoryReport actual = make_streamfunction_memory_report(fields, workspace, grid);

        const std::size_t workspace_only_estimate = estimate.total_bytes - estimate.fields_bytes;
        check((std::string("n") + std::to_string(n) + "_allocated_matches_estimate_minus_fields").c_str(),
              workspace.allocated_device_bytes() == workspace_only_estimate);

        const std::size_t expected_jvp = JvpWorkspace::estimate_device_bytes(cells);
        const std::size_t expected_gmres = CoupledGmres::estimate_device_bytes(cells, config.newton.gmres.restart);
        const std::size_t expected_delta = 2ULL * cells * sizeof(real);
        const std::size_t expected_preconditioner =
            BlockDiagonalMGPreconditioner::estimate_device_bytes(workspace.hierarchy());

        check((std::string("n") + std::to_string(n) + "_jvp_bytes_exact").c_str(),
              actual.newton_jvp_bytes == expected_jvp);
        check((std::string("n") + std::to_string(n) + "_gmres_bytes_exact").c_str(),
              actual.newton_gmres_bytes == expected_gmres);
        check((std::string("n") + std::to_string(n) + "_delta_bytes_exact").c_str(),
              actual.newton_delta_bytes == expected_delta);
        check((std::string("n") + std::to_string(n) + "_preconditioner_bytes_matches_static").c_str(),
              actual.newton_preconditioner_bytes == expected_preconditioner);
        check((std::string("n") + std::to_string(n) + "_all_four_positive").c_str(),
              actual.newton_jvp_bytes > 0 && actual.newton_gmres_bytes > 0 &&
                  actual.newton_delta_bytes > 0 && actual.newton_preconditioner_bytes > 0);

        std::cout << "newton_memory_accounting n=" << cells << " jvp=" << actual.newton_jvp_bytes
                  << " gmres=" << actual.newton_gmres_bytes << " delta=" << actual.newton_delta_bytes
                  << " preconditioner=" << actual.newton_preconditioner_bytes
                  << " workspace_allocated=" << workspace.allocated_device_bytes()
                  << " workspace_only_estimate=" << workspace_only_estimate << '\n';
    }

    // Disabled config at 16^3: all four newton_*_bytes fields exactly zero,
    // and the enabled-vs-disabled total_bytes difference equals exactly the
    // sum of the four newton fields (the enabled estimate's own values).
    {
        const int n = 16;
        const Grid3D grid = isotropic_grid(n);

        StreamfunctionSolverConfig disabled_config{};
        StreamfunctionFields fields;
        StreamfunctionWorkspace workspace;
        fields.prepare(grid);
        workspace.prepare(grid, disabled_config);
        const StreamfunctionMemoryReport actual_disabled =
            make_streamfunction_memory_report(fields, workspace, grid);

        check("disabled_jvp_bytes_zero", actual_disabled.newton_jvp_bytes == 0);
        check("disabled_gmres_bytes_zero", actual_disabled.newton_gmres_bytes == 0);
        check("disabled_preconditioner_bytes_zero", actual_disabled.newton_preconditioner_bytes == 0);
        check("disabled_delta_bytes_zero", actual_disabled.newton_delta_bytes == 0);

        StreamfunctionSolverConfig enabled_config{};
        enabled_config.newton.enabled = true;
        const StreamfunctionMemoryReport estimate_enabled =
            estimate_streamfunction_memory(grid, enabled_config);
        const StreamfunctionMemoryReport estimate_disabled =
            estimate_streamfunction_memory(grid, disabled_config);

        const std::size_t total_diff = estimate_enabled.total_bytes - estimate_disabled.total_bytes;
        const std::size_t newton_sum = estimate_enabled.newton_jvp_bytes + estimate_enabled.newton_gmres_bytes +
                                       estimate_enabled.newton_preconditioner_bytes +
                                       estimate_enabled.newton_delta_bytes;
        check("enabled_minus_disabled_equals_newton_sum", total_diff == newton_sum);

        std::cout << "newton_memory_accounting disabled_total=" << estimate_disabled.total_bytes
                  << " enabled_total=" << estimate_enabled.total_bytes << " diff=" << total_diff
                  << " newton_sum=" << newton_sum << '\n';
    }

    std::cout << "case=newton_memory_accounting verdict=" << (pass ? "PASS" : "FAIL") << '\n';

    return {pass,
            "newton_memory_accounting",
            "contract",
            "16^3, 32^3 newton-enabled + 16^3 disabled control",
            0.0,
            0.0,
            "G6",
            pass ? "all pass" : "some failed",
            "workspace.allocated_device_bytes() equals estimate.total_bytes-fields_bytes; the four "
            "newton_*_bytes fields match their static estimators exactly and are >0 when enabled, "
            "exactly 0 when disabled, and their sum equals the enabled-vs-disabled total_bytes delta"};
}

// ===========================================================================
// G7: newton_fail_fast
// ===========================================================================

[[nodiscard]] CaseResult case_newton_fail_fast() {
    const Grid3D base_grid = isotropic_grid(16);
    const StreamfunctionProblemView base_problem = nullptr_problem_view(base_grid);

    // Base config: adaptive enabled (default), newton DISABLED so C01 does
    // not preempt the field-specific checks below (the header doc: every
    // newton field is "validated regardless of enabled").
    StreamfunctionSolverConfig base_config{};

    std::size_t checks = 0;
    bool pass = true;
    const auto require_invalid = [&](const char* name, const auto& callable) {
        ++checks;
        const bool ok = rejects_with_invalid_argument(name, callable);
        pass = pass && ok;
    };

    require_invalid("newton_enabled_without_adaptive", [&] {
        auto config = base_config;
        config.newton.enabled = true;
        config.adaptive.enabled = false;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("activation_r_F_nonpositive", [&] {
        auto config = base_config;
        config.newton.activation_r_F = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("stagnation_activation_r_F_below_activation", [&] {
        auto config = base_config;
        config.newton.stagnation_activation_r_F =
            config.newton.activation_r_F - real{1e-4};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("forcing_min_nonpositive", [&] {
        auto config = base_config;
        config.newton.forcing_min = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("forcing_max_below_forcing_min", [&] {
        auto config = base_config;
        config.newton.forcing_min = real{1e-2};
        config.newton.forcing_max = real{1e-3};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("forcing_max_above_one", [&] {
        auto config = base_config;
        config.newton.forcing_max = real{1.5};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("armijo_c_ge_one", [&] {
        auto config = base_config;
        config.newton.armijo_c = real{1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("alpha_min_nonpositive", [&] {
        auto config = base_config;
        config.newton.alpha_min = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("backtrack_factor_ge_one", [&] {
        auto config = base_config;
        config.newton.backtrack_factor = real{1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("max_newton_iterations_lt_1", [&] {
        auto config = base_config;
        config.newton.max_newton_iterations = 0;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("rescue_picard_steps_negative", [&] {
        auto config = base_config;
        config.newton.rescue_picard_steps = -1;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("nested_gmres_restart_16", [&] {
        auto config = base_config;
        config.newton.gmres.restart = 16;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("nested_delta_min_ge_max", [&] {
        auto config = base_config;
        config.newton.delta.delta_min = real{1e-2};
        config.newton.delta.delta_max = real{1e-2};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // Never-activated path: homogeneous K=1 fixture, newton enabled.
    {
        constexpr int n = 16;
        const Grid3D grid = isotropic_grid(n);
        CudaContext ctx(0);
        HomogeneousProblemBuffers buffers(grid);
        const StreamfunctionProblemView problem = homogeneous_problem_view(grid, buffers);
        StreamfunctionFields fields;
        StreamfunctionWorkspace workspace;
        StreamfunctionSolverConfig config{};
        config.newton.enabled = true;
        const StreamfunctionSolveReport report =
            solve_streamfunctions(ctx, problem, config, fields, workspace);
        ctx.synchronize();

        ++checks;
        const bool never_activated_ok = report.status == StreamfunctionSolveStatus::converged &&
                                        report.newton_activations == 0 && report.newton_history.empty() &&
                                        report.newton_jv_evaluations == 0;
        pass = pass && never_activated_ok;
        std::cout << std::setprecision(17)
                  << "newton_fail_fast never_activated_path status=" << status_label(report.status)
                  << " newton_activations=" << report.newton_activations
                  << " newton_history_empty=" << (report.newton_history.empty() ? "true" : "false")
                  << " newton_jv_evaluations=" << report.newton_jv_evaluations
                  << " check=" << (never_activated_ok ? "PASS" : "FAIL") << '\n';
    }

    std::cout << "newton_fail_fast total_checks=" << checks << '\n';
    std::cout << "case=newton_fail_fast verdict=" << (pass ? "PASS" : "FAIL") << '\n';

    return {pass,
            "newton_fail_fast",
            "contract",
            "16^3 nullptr-conductivity view + config.newton mutations, plus a homogeneous K=1 "
            "never-activated control",
            static_cast<double>(checks),
            0.0,
            "G7",
            pass ? "all pass" : "some failed",
            "every C01/E10 validation rejection throws std::invalid_argument; a homogeneous K=1 "
            "fixture with newton enabled converges at head 0 without ever activating Newton"};
}

} // namespace

CaseRegistry newton_case_registry() {
    return {
        {"newton_equivalence", case_newton_equivalence},
        {"newton_activation_semantics", case_newton_activation_semantics},
        {"newton_forced_failure_preservation", case_newton_forced_failure_preservation},
        {"newton_rescue_omega_reset", case_newton_rescue_omega_reset},
        {"newton_memory_accounting", case_newton_memory_accounting},
        {"newton_fail_fast", case_newton_fail_fast},
    };
}

CaseRegistry newton_difficult_case_registry() {
    return {{"newton_difficult_case", case_newton_difficult_case}};
}

} // namespace macroflow3d::streamfunctions::test
