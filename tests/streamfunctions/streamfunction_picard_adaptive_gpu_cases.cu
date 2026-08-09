#include "streamfunction_operator_test_cases.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/NonlinearSources.cuh"
#include "src/physics/streamfunctions/ResidualEvaluator.cuh"
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// SF-15 T02: deterministic adaptive-branch GPU acceptance cases for
// `solve_streamfunctions` (StreamfunctionSolver.cuh/.cu, SF-15 T01
// adaptive-Picard globalization). These cases force each documented
// adaptive-loop branch (accept+growth, floor-rejected backtracking,
// stagnation, degeneracy-guard rejection, disabled-mode SF-14 equivalence,
// and the `config.adaptive` validation contract) with extreme-but-valid
// configuration values and assert the SF-15 header/`.cu` semantics exactly
// (including bitwise omega trajectories and a test-side residual
// bitwise-integrity check). They never touch `src/**`.

namespace macroflow3d::streamfunctions::test {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kManufacturedAmplitude = 0.25;

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

[[nodiscard]] Grid3D isotropic_grid(int n, real domain_length = real{1}) {
    const real h = domain_length / static_cast<real>(n);
    return Grid3D{n, n, n, h, h, h};
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

[[nodiscard]] StreamfunctionSolverConfig valid_config() { return StreamfunctionSolverConfig{}; }

template <typename T>
[[nodiscard]] bool finite_value(T value) {
    return std::isfinite(static_cast<double>(value));
}

// Uploads a manufactured heterogeneous K = exp(a*sin(2*pi*x)*sin(2*pi*y)*
// sin(2*pi*z)) conductivity field (host-computed, then uploaded), and the
// uniform reference Darcy field v=(1,0,0) consistent with the benchmark
// gauge g1=(0,1,0), g2=(0,0,1): v = g1 x g2 = (1,0,0). Mirrors
// streamfunction_picard_fixed_gpu_cases.cu's ManufacturedProblemBuffers.
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
                    const std::size_t index = grid.idx(ix, iy, iz);
                    k_host[index] = static_cast<real>(std::exp(log_k));
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
        default: return "none";
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

template <typename Callable>
[[nodiscard]] bool rejects_with_invalid_argument(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "picard_adaptive_config_error_contract name=" << name
                  << " exception=none expected=std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument& error) {
        std::cout << "picard_adaptive_config_error_contract name=" << name
                  << " exception=std::invalid_argument message=" << error.what() << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cout << "picard_adaptive_config_error_contract name=" << name
                  << " exception=std::exception message=" << error.what()
                  << " expected=std::invalid_argument\n";
        return false;
    } catch (...) {
        std::cout << "picard_adaptive_config_error_contract name=" << name
                  << " exception=non-standard expected=std::invalid_argument\n";
        return false;
    }
}

template <typename Callable>
[[nodiscard]] bool accepts_without_throw(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "picard_adaptive_config_error_contract name=" << name
                  << " exception=none expected=none\n";
        return true;
    } catch (const std::exception& error) {
        std::cout << "picard_adaptive_config_error_contract name=" << name
                  << " exception=" << error.what() << " expected=none\n";
        return false;
    } catch (...) {
        std::cout << "picard_adaptive_config_error_contract name=" << name
                  << " exception=non-standard expected=none\n";
        return false;
    }
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

// ---------------------------------------------------------------------------
// Case 1: picard_adaptive_accept_growth.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_picard_adaptive_accept_growth() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);
    const StreamfunctionSolverConfig config = valid_config(); // adaptive.enabled default true

    CudaContext ctx(0);
    ManufacturedProblemBuffers buffers(grid, kManufacturedAmplitude);
    const StreamfunctionProblemView problem = manufactured_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionSolveReport report = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("status_converged", report.status == StreamfunctionSolveStatus::converged);
    add_check("exit_reason_converged", report.exit_reason == PicardExitReason::converged);

    // RULE-EXACT host-side recomputation of the omega trajectory: start at
    // config.picard.omega; after every config.adaptive.easy_streak-th
    // consecutive zero-backtrack acceptance, multiply the persistent omega
    // by config.adaptive.growth_factor, capped at config.adaptive.omega_max
    // (the exact same operations, in the exact same order, as
    // StreamfunctionSolver.cu's acceptance branch -- real == double, so this
    // reproduces the recorded trial omega bitwise).
    const auto iterations = static_cast<std::size_t>(report.picard_iterations);
    std::vector<real> expected_omega(iterations);
    {
        real omega = config.picard.omega;
        int streak = 0;
        for (std::size_t k = 0; k < iterations; ++k) {
            expected_omega[k] = omega;
            ++streak;
            if (streak == config.adaptive.easy_streak) {
                omega = std::min(omega * config.adaptive.growth_factor, config.adaptive.omega_max);
                streak = 0;
            }
        }
    }

    add_check("trial_history_size_eq_iterations", report.trial_history.size() == iterations);

    bool every_trial_accepted_zero_backtrack = true;
    bool omega_sequence_bitwise_exact = true;
    bool omega_in_range = true;
    for (std::size_t k = 0; k < report.trial_history.size(); ++k) {
        const auto& trial = report.trial_history[k];
        every_trial_accepted_zero_backtrack =
            every_trial_accepted_zero_backtrack && trial.outcome == PicardTrialOutcome::accepted &&
            trial.iteration == static_cast<int>(k);
        if (k < expected_omega.size()) {
            omega_sequence_bitwise_exact = omega_sequence_bitwise_exact && trial.omega == expected_omega[k];
        }
        omega_in_range = omega_in_range && trial.omega >= config.adaptive.omega_min &&
                        trial.omega <= config.adaptive.omega_max;
    }
    add_check("every_trial_accepted_zero_backtrack", every_trial_accepted_zero_backtrack);
    add_check("omega_sequence_bitwise_exact", omega_sequence_bitwise_exact);
    add_check("every_omega_in_range", omega_in_range);

    const bool final_omega_matches_expected =
        !expected_omega.empty() && report.final_omega == expected_omega.back();
    add_check("final_omega_matches_last_expected", final_omega_matches_expected || expected_omega.empty());

    bool r_f_strictly_decreasing = true;
    for (std::size_t k = 1; k < report.picard_history.size(); ++k) {
        r_f_strictly_decreasing = r_f_strictly_decreasing &&
                                  report.picard_history[k].r_F < report.picard_history[k - 1].r_F;
    }
    add_check("r_F_strictly_decreasing", r_f_strictly_decreasing);

    bool history_finite = true;
    for (const auto& entry : report.picard_history) {
        history_finite = history_finite && finite_value(entry.r_F) && finite_value(entry.r1) &&
                         finite_value(entry.r2);
    }
    for (const auto& trial : report.trial_history) {
        history_finite = history_finite && finite_value(trial.omega) && finite_value(trial.r_F_trial);
    }
    add_check("all_scalars_finite", history_finite && finite_value(report.final_omega));

    std::cout << std::setprecision(17) << "picard_adaptive_accept_growth N=" << n
              << " a=" << kManufacturedAmplitude << " status=" << status_label(report.status)
              << " exit_reason=" << exit_reason_label(report.exit_reason)
              << " picard_iterations=" << report.picard_iterations
              << " final_r_F=" << report.residual.r_F << " final_omega=" << report.final_omega
              << " wall_seconds=" << wall_seconds << '\n';
    std::cout << "  omega sequence (observed):";
    for (const auto& trial : report.trial_history) std::cout << ' ' << trial.omega;
    std::cout << '\n';
    std::cout << "  omega sequence (expected):";
    for (real value : expected_omega) std::cout << ' ' << value;
    std::cout << '\n';
    std::cout << "  r_F trajectory:";
    for (const auto& entry : report.picard_history) std::cout << ' ' << entry.r_F;
    std::cout << '\n';
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << n << "^3, K=exp(" << kManufacturedAmplitude
           << "*sin(2pi x)sin(2pi y)sin(2pi z)), benchmark gauge, uniform Darcy v=(1,0,0), "
              "adaptive defaults (enabled)";

    return {pass,
            "picard_adaptive_accept_growth",
            "gpu-picard-adaptive-accept-growth",
            detail.str(),
            static_cast<double>(report.residual.r_F),
            static_cast<double>(report.picard_iterations),
            "converged, every trial accepted with zero backtracks, omega trajectory rule-exact",
            pass ? "all pass" : "some failed",
            "manufactured heterogeneous conductivity, adaptive defaults: the omega-growth branch "
            "engages every 3rd consecutive easy acceptance and the recorded omega trajectory must "
            "match the documented policy bitwise"};
}

// ---------------------------------------------------------------------------
// Case 2: picard_adaptive_reject_to_floor.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_picard_adaptive_reject_to_floor() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);
    StreamfunctionSolverConfig config = valid_config();
    // Extreme-but-valid: armijo_c in [0, 1); 0.999999 makes the Armijo
    // sufficient-decrease factor (1 - armijo_c*omega_try) so close to 1 that
    // Picard's per-step reduction (~0.775 on this manufactured problem)
    // cannot satisfy it at any omega_try, so every trial is rejected.
    config.adaptive.armijo_c = real{0.999999};

    CudaContext ctx(0);
    ManufacturedProblemBuffers buffers(grid, kManufacturedAmplitude);
    const StreamfunctionProblemView problem = manufactured_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionSolveReport report = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("exit_reason_omega_floor_rejected",
              report.exit_reason == PicardExitReason::omega_floor_rejected);
    add_check("status_not_converged", report.status == StreamfunctionSolveStatus::not_converged);
    add_check("picard_iterations_zero", report.picard_iterations == 0);

    const std::vector<real> expected_omega = {real{0.25}, real{0.125}, real{0.0625},
                                              real{0.03125}, real{0.015625}, real{0.01}};
    add_check("trial_history_size_eq_6", report.trial_history.size() == expected_omega.size());

    bool omega_sequence_exact = true;
    bool all_rejected_armijo = true;
    for (std::size_t k = 0; k < report.trial_history.size(); ++k) {
        const auto& trial = report.trial_history[k];
        if (k < expected_omega.size()) {
            omega_sequence_exact = omega_sequence_exact && trial.omega == expected_omega[k];
        }
        all_rejected_armijo =
            all_rejected_armijo && trial.outcome == PicardTrialOutcome::rejected_armijo &&
            trial.iteration == 0;
    }
    add_check("omega_sequence_exact_halving_to_floor", omega_sequence_exact);
    add_check("every_trial_rejected_armijo", all_rejected_armijo);

    // final_omega on this exit path: the persistent `omega` local never
    // receives an accepted trial's value (no trial is ever accepted before
    // the floor exit), so it remains the initial config.picard.omega -- this
    // is the IMPLEMENTED behavior (StreamfunctionSolver.cu: `omega` is only
    // reassigned inside the `outcome == accepted` branch).
    add_check("final_omega_eq_picard_omega", report.final_omega == config.picard.omega);

    // Bitwise integrity: the accepted state (fields.u1_span()/u2_span()) was
    // never touched by any of the 6 rejected trials. Re-evaluate the
    // coupled residual test-side, from the returned fields, using the exact
    // same eta/source/histogram configuration the solver used, and compare
    // the resulting r_F bitwise against the recorded state-0 head residual.
    StreamfunctionResidualWorkspace test_residual_workspace;
    const std::size_t n_cells = grid.num_cells();
    test_residual_workspace.prepare(n_cells);
    DeviceBuffer<real> f1_out(n_cells);
    DeviceBuffer<real> f2_out(n_cells);

    NonlinearSourceConfig source_config{};
    source_config.epsilon = config.epsilon;
    source_config.v_rms = report.diagnostics.v_d_rms;
    source_config.num_degeneracy_thresholds = config.num_degeneracy_thresholds;
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        source_config.degeneracy_thresholds[t] = config.degeneracy_thresholds[t];
    }

    enqueue_streamfunction_residual(ctx, grid, DeviceSpan<const real>(workspace.q()),
                                    fields.fluctuations(), problem.gauge, config.eta, source_config,
                                    config.histogram, f1_out.span(), f2_out.span(),
                                    test_residual_workspace);
    const StreamfunctionResidualReport test_report = synchronize_streamfunction_residual_report(
        ctx, grid, config.eta, source_config, config.histogram, test_residual_workspace);

    const bool bitwise_integrity_ok =
        !report.picard_history.empty() && test_report.r_F == report.picard_history.front().r_F;
    add_check("bitwise_integrity_accepted_state_untouched", bitwise_integrity_ok);

    std::cout << std::setprecision(17) << "picard_adaptive_reject_to_floor N=" << n
              << " a=" << kManufacturedAmplitude << " status=" << status_label(report.status)
              << " exit_reason=" << exit_reason_label(report.exit_reason)
              << " picard_iterations=" << report.picard_iterations
              << " final_omega=" << report.final_omega << " wall_seconds=" << wall_seconds << '\n';
    std::cout << "  trial omega sequence:";
    for (const auto& trial : report.trial_history) {
        std::cout << " (" << trial.omega << ',' << trial_outcome_label(trial.outcome) << ')';
    }
    std::cout << '\n';
    std::cout << "  bitwise integrity: test_side_r_F=" << test_report.r_F
              << " picard_history[0].r_F="
              << (report.picard_history.empty() ? real{-1} : report.picard_history.front().r_F)
              << " equal=" << (bitwise_integrity_ok ? "true" : "false") << '\n';
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << n << "^3, K=exp(" << kManufacturedAmplitude
           << "*sin(2pi x)sin(2pi y)sin(2pi z)), benchmark gauge, uniform Darcy v=(1,0,0), "
              "adaptive.armijo_c=0.999999";

    return {pass,
            "picard_adaptive_reject_to_floor",
            "gpu-picard-adaptive-reject-to-floor",
            detail.str(),
            static_cast<double>(report.residual.r_F),
            static_cast<double>(report.picard_iterations),
            "omega_floor_rejected after 6 rejected-armijo trials halving to the 0.01 floor, "
            "accepted state bitwise-untouched",
            pass ? "all pass" : "some failed",
            "an armijo_c so close to 1 that no relaxation factor can satisfy the sufficient-"
            "decrease test on this problem forces every trial to be rejected down to the omega "
            "floor, exercising the structured omega_floor_rejected exit"};
}

// ---------------------------------------------------------------------------
// Case 3: picard_adaptive_stagnation.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_picard_adaptive_stagnation() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);
    StreamfunctionSolverConfig config = valid_config();
    // Valid: picard.tolerance must be finite > 0; 1e-30 makes convergence-by-
    // tolerance unreachable so the stagnation branch is what actually exits.
    config.picard.tolerance = real{1e-30};
    // Extreme-but-valid: stagnation_min_reduction in (0,1); 0.99 demands a
    // >=99% residual reduction over the window, which this problem's
    // ~0.775^10 (~92%) reduction cannot meet.
    config.adaptive.stagnation_min_reduction = real{0.99};

    CudaContext ctx(0);
    ManufacturedProblemBuffers buffers(grid, kManufacturedAmplitude);
    const StreamfunctionProblemView problem = manufactured_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionSolveReport report = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("exit_reason_stagnated", report.exit_reason == PicardExitReason::stagnated);
    add_check("status_not_converged", report.status == StreamfunctionSolveStatus::not_converged);
    add_check("picard_iterations_eq_window",
              report.picard_iterations == config.adaptive.stagnation_window);

    const bool history_ok = report.picard_history.size() ==
                            static_cast<std::size_t>(config.adaptive.stagnation_window) + 1;
    add_check("history_size_eq_window_plus_one", history_ok);

    const bool decreased_over_window =
        history_ok && report.picard_history.back().r_F < report.picard_history.front().r_F;
    add_check("r_F_decreased_over_window", decreased_over_window);

    bool exit_info_finite = finite_value(report.final_omega) && finite_value(report.residual.r_F) &&
                            finite_value(report.diagnostics.v_d_rms);
    add_check("exit_info_complete_and_finite", exit_info_finite);

    std::cout << std::setprecision(17) << "picard_adaptive_stagnation N=" << n
              << " a=" << kManufacturedAmplitude << " status=" << status_label(report.status)
              << " exit_reason=" << exit_reason_label(report.exit_reason)
              << " picard_iterations=" << report.picard_iterations
              << " final_omega=" << report.final_omega << " wall_seconds=" << wall_seconds << '\n';
    std::cout << "  r_F trajectory:";
    for (const auto& entry : report.picard_history) std::cout << ' ' << entry.r_F;
    std::cout << '\n';
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << n << "^3, K=exp(" << kManufacturedAmplitude
           << "*sin(2pi x)sin(2pi y)sin(2pi z)), benchmark gauge, uniform Darcy v=(1,0,0), "
              "picard.tolerance=1e-30, adaptive.stagnation_min_reduction=0.99";

    return {pass,
            "picard_adaptive_stagnation",
            "gpu-picard-adaptive-stagnation",
            detail.str(),
            static_cast<double>(report.residual.r_F),
            static_cast<double>(report.picard_iterations),
            "stagnated at exactly the stagnation_window'th accepted step, r_F still decreased "
            "over the window",
            pass ? "all pass" : "some failed",
            "an unreachable tolerance plus a 99%-reduction stagnation requirement forces the "
            "stagnation exit on a problem whose residual is still monotonically decreasing, "
            "proving the check never relabels a decrease as an increase"};
}

// ---------------------------------------------------------------------------
// Case 4: picard_adaptive_degeneracy_rejection.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_picard_adaptive_degeneracy_rejection() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);
    StreamfunctionSolverConfig config = valid_config();
    // Extreme-but-valid: PhysicalDiagnosticsConfig requires finite, strictly
    // positive, strictly increasing thresholds; a single threshold of 1e6 is
    // valid and makes every cell count as |c|-degenerate at that threshold.
    // Only the diagnostics-side threshold list drives the adaptive guard
    // (config.diagnostics.num_degeneracy_thresholds); the source-side
    // config.num_degeneracy_thresholds is left at its default (0).
    config.diagnostics.num_degeneracy_thresholds = 1;
    config.diagnostics.degeneracy_thresholds[0] = real{1e6};

    CudaContext ctx(0);
    ManufacturedProblemBuffers buffers(grid, kManufacturedAmplitude);
    const StreamfunctionProblemView problem = manufactured_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionSolveReport report = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("exit_reason_omega_floor_rejected",
              report.exit_reason == PicardExitReason::omega_floor_rejected);
    add_check("status_not_converged", report.status == StreamfunctionSolveStatus::not_converged);
    add_check("picard_iterations_zero", report.picard_iterations == 0);

    bool every_trial_rejected_degeneracy = !report.trial_history.empty();
    for (const auto& trial : report.trial_history) {
        every_trial_rejected_degeneracy =
            every_trial_rejected_degeneracy && trial.outcome == PicardTrialOutcome::rejected_degeneracy;
    }
    add_check("every_trial_rejected_degeneracy", every_trial_rejected_degeneracy);

    const std::size_t n_cells = grid.num_cells();
    const bool unexplained_all_cells =
        report.diagnostics.num_degeneracy_thresholds >= 1 &&
        report.diagnostics.degeneracy_unexplained[0] == static_cast<unsigned long long>(n_cells);
    add_check("final_state_degeneracy_unexplained_eq_n", unexplained_all_cells);

    std::cout << std::setprecision(17) << "picard_adaptive_degeneracy_rejection N=" << n
              << " a=" << kManufacturedAmplitude << " status=" << status_label(report.status)
              << " exit_reason=" << exit_reason_label(report.exit_reason)
              << " picard_iterations=" << report.picard_iterations
              << " final_omega=" << report.final_omega << " wall_seconds=" << wall_seconds << '\n';
    std::cout << "  trial outcomes:";
    for (const auto& trial : report.trial_history) {
        std::cout << " (" << trial.omega << ',' << trial_outcome_label(trial.outcome) << ')';
    }
    std::cout << '\n';
    std::cout << "  degeneracy_unexplained[0]=" << report.diagnostics.degeneracy_unexplained[0]
              << " degeneracy_total[0]=" << report.diagnostics.degeneracy_total[0]
              << " degeneracy_low_speed[0]=" << report.diagnostics.degeneracy_low_speed[0]
              << " n=" << n_cells << '\n';
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << n << "^3, K=exp(" << kManufacturedAmplitude
           << "*sin(2pi x)sin(2pi y)sin(2pi z)), benchmark gauge, uniform Darcy v=(1,0,0), "
              "diagnostics.num_degeneracy_thresholds=1, degeneracy_thresholds[0]=1e6";

    return {pass,
            "picard_adaptive_degeneracy_rejection",
            "gpu-picard-adaptive-degeneracy-rejection",
            detail.str(),
            static_cast<double>(report.residual.r_F),
            static_cast<double>(report.picard_iterations),
            "every trial rejected_degeneracy, floor exit, final unexplained fraction == 1",
            pass ? "all pass" : "some failed",
            "an unreachably-large |c| degeneracy threshold makes every cell count as unexplained "
            "under a uniform, non-low-speed Darcy field, forcing the SF-11-diagnostics-based "
            "rejected_degeneracy guard on every trial"};
}

// ---------------------------------------------------------------------------
// Case 5: picard_adaptive_fixed_mode_equivalence.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_picard_adaptive_fixed_mode_equivalence() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);

    StreamfunctionSolverConfig disabled_config = valid_config();
    disabled_config.adaptive.enabled = false;

    CudaContext ctx_disabled(0);
    ManufacturedProblemBuffers buffers_disabled(grid, kManufacturedAmplitude);
    const StreamfunctionProblemView problem_disabled =
        manufactured_problem_view(grid, buffers_disabled);
    StreamfunctionFields fields_disabled;
    StreamfunctionWorkspace workspace_disabled;

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionSolveReport disabled_report = solve_streamfunctions(
        ctx_disabled, problem_disabled, disabled_config, fields_disabled, workspace_disabled);
    ctx_disabled.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double disabled_wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    // Second run, adaptive defaults enabled, for audit comparison only (no
    // assertions on this run beyond finiteness): documents the expected
    // divergence in iteration count from the fixed-relaxation baseline.
    StreamfunctionSolverConfig enabled_config = valid_config();
    CudaContext ctx_enabled(0);
    ManufacturedProblemBuffers buffers_enabled(grid, kManufacturedAmplitude);
    const StreamfunctionProblemView problem_enabled = manufactured_problem_view(grid, buffers_enabled);
    StreamfunctionFields fields_enabled;
    StreamfunctionWorkspace workspace_enabled;

    const auto t2 = std::chrono::steady_clock::now();
    const StreamfunctionSolveReport enabled_report = solve_streamfunctions(
        ctx_enabled, problem_enabled, enabled_config, fields_enabled, workspace_enabled);
    ctx_enabled.synchronize();
    const auto t3 = std::chrono::steady_clock::now();
    const double enabled_wall_seconds = std::chrono::duration<double>(t3 - t2).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    // SF-14 recorded values from this same binary lineage (picard_fixed_
    // manufactured_16, before SF-15): 40 Picard iterations,
    // r_F ~= 9.9099533174336207e-07.
    add_check("disabled_iterations_eq_40", disabled_report.picard_iterations == 40);
    add_check("disabled_status_converged",
              disabled_report.status == StreamfunctionSolveStatus::converged);
    add_check("disabled_exit_reason_converged",
              disabled_report.exit_reason == PicardExitReason::converged);
    add_check("disabled_r_F_in_range",
              static_cast<double>(disabled_report.residual.r_F) > 0.0 &&
                  static_cast<double>(disabled_report.residual.r_F) <= 1e-6);
    add_check("disabled_final_omega_eq_picard_omega",
              disabled_report.final_omega == disabled_config.picard.omega);
    add_check("disabled_trial_history_empty", disabled_report.trial_history.empty());

    bool every_recorded_block_solve_tight = true;
    for (std::size_t k = 1; k < disabled_report.picard_history.size(); ++k) {
        const auto& entry = disabled_report.picard_history[k];
        every_recorded_block_solve_tight =
            every_recorded_block_solve_tight && entry.psi1_result.converged &&
            entry.psi2_result.converged &&
            static_cast<double>(entry.psi1_result.relative_projected_residual) <= 1e-10 &&
            static_cast<double>(entry.psi2_result.relative_projected_residual) <= 1e-10;
    }
    add_check("every_recorded_block_solve_converged_and_tight", every_recorded_block_solve_tight);

    std::cout << std::setprecision(17) << "picard_adaptive_fixed_mode_equivalence N=" << n
              << " a=" << kManufacturedAmplitude << '\n';
    std::cout << "  disabled(adaptive.enabled=false): status=" << status_label(disabled_report.status)
              << " exit_reason=" << exit_reason_label(disabled_report.exit_reason)
              << " picard_iterations=" << disabled_report.picard_iterations
              << " r_F=" << disabled_report.residual.r_F
              << " final_omega=" << disabled_report.final_omega
              << " wall_seconds=" << disabled_wall_seconds << '\n';
    std::cout << "  enabled(adaptive defaults, comparison only): status="
              << status_label(enabled_report.status)
              << " exit_reason=" << exit_reason_label(enabled_report.exit_reason)
              << " picard_iterations=" << enabled_report.picard_iterations
              << " r_F=" << enabled_report.residual.r_F
              << " final_omega=" << enabled_report.final_omega
              << " wall_seconds=" << enabled_wall_seconds << '\n';
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << n << "^3, K=exp(" << kManufacturedAmplitude
           << "*sin(2pi x)sin(2pi y)sin(2pi z)), benchmark gauge, uniform Darcy v=(1,0,0), "
              "adaptive.enabled=false vs adaptive defaults (comparison)";

    return {pass,
            "picard_adaptive_fixed_mode_equivalence",
            "gpu-picard-adaptive-fixed-mode-equivalence",
            detail.str(),
            static_cast<double>(disabled_report.residual.r_F),
            static_cast<double>(disabled_report.picard_iterations),
            "adaptive.enabled=false reproduces the SF-14 recorded profile exactly (40 iterations, "
            "r_F<=1e-6, converged, empty trial_history)",
            pass ? "all pass" : "some failed",
            "adaptive.enabled=false must reproduce the SF-14 fixed-relaxation path exactly on the "
            "same manufactured problem/binary lineage; the adaptive-default run is printed only "
            "for audit comparison"};
}

// ---------------------------------------------------------------------------
// Case 6: picard_adaptive_config_error_contract.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_picard_adaptive_config_error_contract() {
    const Grid3D base_grid = isotropic_grid(16);
    const StreamfunctionProblemView base_problem = nullptr_problem_view(base_grid);
    const StreamfunctionSolverConfig base_config = valid_config(); // picard.omega == 0.25

    std::size_t checks = 0;
    bool pass = true;
    const auto require_invalid = [&](const char* name, const auto& callable) {
        ++checks;
        pass = rejects_with_invalid_argument(name, callable) && pass;
    };
    const auto require_ok = [&](const char* name, const auto& callable) {
        ++checks;
        pass = accepts_without_throw(name, callable) && pass;
    };

    // omega_min: finite, in (0, picard.omega].
    require_invalid("adaptive_omega_min_zero", [&] {
        auto config = base_config;
        config.adaptive.omega_min = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("adaptive_omega_min_gt_picard_omega", [&] {
        auto config = base_config;
        config.adaptive.omega_min = real{0.5};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // backtrack_factor: finite, in (0, 1).
    require_invalid("adaptive_backtrack_factor_zero", [&] {
        auto config = base_config;
        config.adaptive.backtrack_factor = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("adaptive_backtrack_factor_one", [&] {
        auto config = base_config;
        config.adaptive.backtrack_factor = real{1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // growth_factor: finite, >= 1.
    require_invalid("adaptive_growth_factor_lt_one", [&] {
        auto config = base_config;
        config.adaptive.growth_factor = real{0.5};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // omega_max: finite, in [picard.omega, 1].
    require_invalid("adaptive_omega_max_lt_picard_omega", [&] {
        auto config = base_config;
        config.adaptive.omega_max = real{0.1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("adaptive_omega_max_gt_one", [&] {
        auto config = base_config;
        config.adaptive.omega_max = real{1.5};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // easy_streak: >= 1.
    require_invalid("adaptive_easy_streak_zero", [&] {
        auto config = base_config;
        config.adaptive.easy_streak = 0;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // armijo_c: finite, in [0, 1).
    require_invalid("adaptive_armijo_c_one", [&] {
        auto config = base_config;
        config.adaptive.armijo_c = real{1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("adaptive_armijo_c_negative", [&] {
        auto config = base_config;
        config.adaptive.armijo_c = real{-1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // stagnation_window: >= 1.
    require_invalid("adaptive_stagnation_window_zero", [&] {
        auto config = base_config;
        config.adaptive.stagnation_window = 0;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // stagnation_min_reduction: finite, in (0, 1).
    require_invalid("adaptive_stagnation_min_reduction_zero", [&] {
        auto config = base_config;
        config.adaptive.stagnation_min_reduction = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("adaptive_stagnation_min_reduction_one", [&] {
        auto config = base_config;
        config.adaptive.stagnation_min_reduction = real{1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // max_unexplained_fraction: finite, in (0, 1].
    require_invalid("adaptive_max_unexplained_fraction_zero", [&] {
        auto config = base_config;
        config.adaptive.max_unexplained_fraction = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("adaptive_max_unexplained_fraction_gt_one", [&] {
        auto config = base_config;
        config.adaptive.max_unexplained_fraction = real{1.5};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // unexplained_growth_factor: finite, >= 1.
    require_invalid("adaptive_unexplained_growth_factor_lt_one", [&] {
        auto config = base_config;
        config.adaptive.unexplained_growth_factor = real{0.5};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // unexplained_growth_offset: finite, >= 0.
    require_invalid("adaptive_unexplained_growth_offset_negative", [&] {
        auto config = base_config;
        config.adaptive.unexplained_growth_offset = real{-1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // percentile_collapse_factor: finite, > 1.
    require_invalid("adaptive_percentile_collapse_factor_one", [&] {
        auto config = base_config;
        config.adaptive.percentile_collapse_factor = real{1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // Positives: all defaults, and enabled=false with otherwise-invalid-
    // looking-but-valid boundary values.
    require_ok("adaptive_all_defaults_valid",
              [&] { validate_streamfunction_problem(base_grid, base_problem, base_config); });
    require_ok("adaptive_disabled_boundary_values_valid", [&] {
        auto config = base_config;
        config.adaptive.enabled = false;
        config.adaptive.armijo_c = real{0};
        config.adaptive.omega_max = config.picard.omega;
        config.adaptive.max_unexplained_fraction = real{1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    std::cout << "picard_adaptive_config_error_contract total_checks=" << checks
              << " pass=" << (pass ? "true" : "false") << '\n';

    return {pass,
            "picard_adaptive_config_error_contract",
            "contract",
            "16x16x16 + config.adaptive mutations",
            static_cast<double>(checks),
            0.0,
            "all checks pass",
            pass ? "all pass" : "some failed",
            "validate_streamfunction_problem throws std::invalid_argument for every documented "
            "config.adaptive precondition violation (validated unconditionally, regardless of "
            "adaptive.enabled), and accepts default/boundary-valid configurations"};
}

} // namespace

CaseRegistry picard_adaptive_case_registry() {
    return {{"picard_adaptive_accept_growth", case_picard_adaptive_accept_growth},
            {"picard_adaptive_reject_to_floor", case_picard_adaptive_reject_to_floor},
            {"picard_adaptive_stagnation", case_picard_adaptive_stagnation},
            {"picard_adaptive_degeneracy_rejection", case_picard_adaptive_degeneracy_rejection},
            {"picard_adaptive_fixed_mode_equivalence", case_picard_adaptive_fixed_mode_equivalence},
            {"picard_adaptive_config_error_contract", case_picard_adaptive_config_error_contract}};
}

} // namespace macroflow3d::streamfunctions::test
