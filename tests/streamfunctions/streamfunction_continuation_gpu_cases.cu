#include "streamfunction_operator_test_cases.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/ContinuationController.hpp"
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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// SF-17 T02: `streamfunction_continuation` acceptance cases for
// `ContinuationController.hpp/.cu` (SF-17 T01 adaptive eta/epsilon
// continuation with rollback). Cases 1-6 (host-only, GPU-free) exercise
// `ContinuationStepper` and `validate_streamfunction_continuation_config`
// bitwise/exactly against the documented policy in the file header. Cases
// 7-13 run the production `run_streamfunction_continuation` driver on GPU
// fixtures mirrored exactly from `streamfunction_picard_fixed_gpu_cases.cu`
// / `streamfunction_picard_adaptive_gpu_cases.cu`. Every threshold below is
// prespecified by the increment task and is never tuned after seeing
// results; a failing case is reported honestly, never relaxed. This file
// never touches `src/**`.

namespace macroflow3d::streamfunctions::test {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

// ---------------------------------------------------------------------------
// Shared GPU fixture helpers, mirrored exactly from
// streamfunction_picard_fixed_gpu_cases.cu / streamfunction_picard_adaptive_gpu_cases.cu.
// ---------------------------------------------------------------------------

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

// Homogeneous (K=1) problem buffers, mirrored from
// streamfunction_picard_fixed_gpu_cases.cu's HomogeneousProblemBuffers.
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

// Manufactured heterogeneous K = exp(a*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z))
// conductivity field, mirrored exactly from streamfunction_picard_fixed_gpu_cases.cu
// / streamfunction_picard_adaptive_gpu_cases.cu's ManufacturedProblemBuffers.
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

[[nodiscard]] std::vector<real> download(const DeviceSpan<const real>& span) {
    std::vector<real> host(span.size());
    if (!host.empty()) {
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(host.data(), span.data(), host.size() * sizeof(real), cudaMemcpyDeviceToHost));
    }
    return host;
}

[[nodiscard]] std::vector<real> download_fields_u1(StreamfunctionFields& fields) {
    return download(DeviceSpan<const real>(fields.u1_span()));
}
[[nodiscard]] std::vector<real> download_fields_u2(StreamfunctionFields& fields) {
    return download(DeviceSpan<const real>(fields.u2_span()));
}

void scribble_fields(StreamfunctionFields& fields, real value) {
    const std::vector<real> garbage_u1(fields.u1_span().size(), value);
    const std::vector<real> garbage_u2(fields.u2_span().size(), value);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(fields.u1_span().data(), garbage_u1.data(),
                                      garbage_u1.size() * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(fields.u2_span().data(), garbage_u2.data(),
                                      garbage_u2.size() * sizeof(real), cudaMemcpyHostToDevice));
}

// A fabricated (never-solved) StreamfunctionSolveReport with the given
// status. `make_stage_record` (ContinuationController.cu) unconditionally
// calls `residual_histogram_percentile` on every stage's report, including
// rejected/fabricated ones, which throws on a default-constructed (empty)
// histogram; this helper gives the fabricated report a minimally-valid
// single-underflow-count histogram so that call succeeds, exactly mirroring
// what a real (non-degenerate) `StreamfunctionResidualReport` always
// provides.
[[nodiscard]] StreamfunctionSolveReport fabricated_report(StreamfunctionSolveStatus status) {
    StreamfunctionSolveReport report;
    report.status = status;
    report.exit_reason = PicardExitReason::budget_exhausted;
    report.residual.histogram_underflow = 1;
    report.residual.histogram_c_min = real{1};
    report.residual.histogram_c_max = real{2};
    return report;
}

[[nodiscard]] const char* status_label(ContinuationStatus status) {
    switch (status) {
        case ContinuationStatus::reached_target: return "reached_target";
        case ContinuationStatus::baseline_failed: return "baseline_failed";
        case ContinuationStatus::step_floor_exhausted: return "step_floor_exhausted";
        case ContinuationStatus::invalid_problem: return "invalid_problem";
        default: return "unknown";
    }
}

[[nodiscard]] const char* axis_label(ContinuationAxis axis) {
    return axis == ContinuationAxis::eta ? "eta" : "epsilon";
}

[[nodiscard]] const char* failure_label(ContinuationStageFailure failure) {
    switch (failure) {
        case ContinuationStageFailure::none: return "none";
        case ContinuationStageFailure::solver_not_converged: return "solver_not_converged";
        case ContinuationStageFailure::solver_invalid_problem: return "solver_invalid_problem";
        default: return "unknown";
    }
}

void print_stage_history(const std::vector<ContinuationStageRecord>& history) {
    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto& r = history[i];
        std::cout << std::setprecision(17) << "  stage[" << i << "] axis=" << axis_label(r.axis)
                  << " param_start=" << r.param_start << " param_attempted=" << r.param_attempted
                  << " step_attempted=" << r.step_attempted
                  << " accepted=" << (r.accepted ? "true" : "false")
                  << " failure=" << failure_label(r.failure)
                  << " picard_iterations=" << r.picard_iterations << " r_F=" << r.r_F << '\n';
    }
}

template <typename Callable>
[[nodiscard]] bool rejects_with_invalid_argument(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "continuation_config_error_contract name=" << name
                  << " exception=none expected=std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument& error) {
        std::cout << "continuation_config_error_contract name=" << name
                  << " exception=std::invalid_argument message=" << error.what() << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cout << "continuation_config_error_contract name=" << name
                  << " exception=std::exception message=" << error.what()
                  << " expected=std::invalid_argument\n";
        return false;
    } catch (...) {
        std::cout << "continuation_config_error_contract name=" << name
                  << " exception=non-standard expected=std::invalid_argument\n";
        return false;
    }
}

template <typename Callable>
[[nodiscard]] bool accepts_without_throw(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "continuation_config_error_contract name=" << name
                  << " exception=none expected=none\n";
        return true;
    } catch (const std::exception& error) {
        std::cout << "continuation_config_error_contract name=" << name
                  << " exception=" << error.what() << " expected=none\n";
        return false;
    } catch (...) {
        std::cout << "continuation_config_error_contract name=" << name
                  << " exception=non-standard expected=none\n";
        return false;
    }
}

// ---------------------------------------------------------------------------
// Case 1: continuation_stepper_eta_all_accept (host-only).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_stepper_eta_all_accept() {
    const ContinuationAxisConfig config{real{0},   real{1},    real{0.1},  real{0.0125},
                                        real{0.25}, real{0.5},  real{1.5}, 2};
    ContinuationStepper stepper(config);

    // RULE-EXACT host-side recomputation of the all-accept trajectory: the
    // SAME operations, in the SAME order, real == double, as
    // ContinuationStepper::attempt_param()/on_accept() (mirroring the
    // picard_adaptive test file's established pattern) -- decimal literals
    // like 0.35 do not reproduce 0.2+0.15 bitwise, so the oracle must use
    // the identical arithmetic rather than typed decimal constants.
    std::vector<real> expected_attempts;
    {
        real param = config.start;
        real step = config.initial_step;
        int streak = 0;
        while (param < config.target) {
            const real attempt = std::min(param + step, config.target);
            expected_attempts.push_back(attempt);
            param = attempt;
            ++streak;
            if (streak >= config.easy_streak) {
                step = std::min(step * config.growth_factor, config.max_step);
                streak = 0;
            }
        }
    }

    std::vector<real> observed_attempts;
    while (!stepper.reached_target()) {
        const real attempt = stepper.attempt_param();
        observed_attempts.push_back(attempt);
        stepper.on_accept();
    }

    bool pass = observed_attempts.size() == expected_attempts.size();
    for (std::size_t i = 0; pass && i < expected_attempts.size(); ++i) {
        pass = pass && observed_attempts[i] == expected_attempts[i];
    }
    pass = pass && stepper.current_param() == real{1};

    std::cout << std::setprecision(17) << "continuation_stepper_eta_all_accept:\n";
    std::cout << "  expected:";
    for (real v : expected_attempts) std::cout << ' ' << v;
    std::cout << "\n  observed:";
    for (real v : observed_attempts) std::cout << ' ' << v;
    std::cout << '\n';

    return {pass,
            "continuation_stepper_eta_all_accept",
            "positive",
            "eta axis defaults",
            static_cast<double>(observed_attempts.size()),
            static_cast<double>(expected_attempts.size()),
            "0.1,0.2,0.35,0.5,0.725,0.95,1.0 (exact ==)",
            pass ? "matches exactly" : "mismatch",
            "eta-axis defaults, all-accept sequence must reach target exactly with the documented "
            "growth-after-2-easy-stages policy, clamped exactly at 1.0"};
}

// ---------------------------------------------------------------------------
// Case 2: continuation_stepper_epsilon_all_accept (host-only).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_stepper_epsilon_all_accept() {
    const ContinuationAxisConfig config{real{2}, real{6},   real{1.0}, real{0.125},
                                        real{1.0}, real{0.5}, real{1.5}, 2};
    ContinuationStepper stepper(config);

    const std::vector<real> expected_attempts = {real{3}, real{4}, real{5}, real{6}};
    std::vector<real> observed_attempts;
    while (!stepper.reached_target()) {
        const real attempt = stepper.attempt_param();
        observed_attempts.push_back(attempt);
        stepper.on_accept();
    }

    bool pass = observed_attempts.size() == expected_attempts.size();
    for (std::size_t i = 0; pass && i < expected_attempts.size(); ++i) {
        pass = pass && observed_attempts[i] == expected_attempts[i];
    }
    pass = pass && stepper.current_param() == real{6};

    std::cout << std::setprecision(17) << "continuation_stepper_epsilon_all_accept:\n";
    std::cout << "  expected:";
    for (real v : expected_attempts) std::cout << ' ' << v;
    std::cout << "\n  observed:";
    for (real v : observed_attempts) std::cout << ' ' << v;
    std::cout << '\n';

    return {pass,
            "continuation_stepper_epsilon_all_accept",
            "positive",
            "epsilon_log10 axis defaults",
            static_cast<double>(observed_attempts.size()),
            static_cast<double>(expected_attempts.size()),
            "p=3,4,5,6 exactly (decades, growth capped at max_step)",
            pass ? "matches exactly" : "mismatch",
            "epsilon_log10-axis defaults, all-accept sequence must land exactly on the integer "
            "decade sequence 3,4,5,6, proving physical epsilon decays exactly by decades on the "
            "no-failure path"};
}

// ---------------------------------------------------------------------------
// Case 3: continuation_stepper_reject_halve (host-only).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_stepper_reject_halve() {
    const ContinuationAxisConfig config{real{0},   real{1},    real{0.1},  real{0.0125},
                                        real{0.25}, real{0.5},  real{1.5}, 2};
    ContinuationStepper stepper(config);

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    // Reject #1: step 0.1 -> 0.05.
    add_check("attempt1_eq_0.1", stepper.attempt_param() == real{0.1});
    add_check("reject1_retries", stepper.on_reject());
    add_check("step_after_reject1_eq_0.05", stepper.current_step() == real{0.05});
    add_check("param_unchanged_after_reject1", stepper.current_param() == real{0});

    // Reject #2: step 0.05 -> 0.025.
    add_check("attempt2_eq_0.05", stepper.attempt_param() == real{0.05});
    add_check("reject2_retries", stepper.on_reject());
    add_check("step_after_reject2_eq_0.025", stepper.current_step() == real{0.025});
    add_check("param_unchanged_after_reject2", stepper.current_param() == real{0});

    // Reject #3: step 0.025 -> 0.0125 (clamp).
    add_check("attempt3_eq_0.025", stepper.attempt_param() == real{0.025});
    add_check("reject3_retries", stepper.on_reject());
    add_check("step_after_reject3_eq_0.0125", stepper.current_step() == real{0.0125});
    add_check("param_unchanged_after_reject3", stepper.current_param() == real{0});

    // Accept at the halved step: param advances, step (the succeeded step)
    // is kept unchanged.
    const real attempt4 = stepper.attempt_param();
    add_check("attempt4_eq_0.0125", attempt4 == real{0.0125});
    stepper.on_accept();
    add_check("param_after_accept_eq_0.0125", stepper.current_param() == real{0.0125});
    add_check("step_after_accept_kept_halved_0.0125", stepper.current_step() == real{0.0125});

    for (const auto& [name, ok] : checks) {
        std::cout << "continuation_stepper_reject_halve check " << name << "=" << (ok ? "PASS" : "FAIL")
                  << '\n';
    }

    return {pass,
            "continuation_stepper_reject_halve",
            "positive",
            "eta axis defaults",
            0.0,
            0.0,
            "3 halvings 0.1->0.05->0.025->0.0125(clamp), param unchanged, accept keeps halved step",
            pass ? "all pass" : "some failed",
            "rejection halves the persistent step and retries the SAME parameter interval; the "
            "eventually-accepted attempt keeps the halved (succeeded) step"};
}

// ---------------------------------------------------------------------------
// Case 4: continuation_stepper_floor (host-only).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_stepper_floor() {
    // initial_step == min_step: the very first attempt is already at the
    // floor, so a single reject must report floor exhaustion.
    const ContinuationAxisConfig config{real{0},   real{1},    real{0.1},  real{0.1},
                                        real{0.25}, real{0.5},  real{1.5}, 2};
    ContinuationStepper stepper(config);

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("attempt_eq_0.1", stepper.attempt_param() == real{0.1});
    add_check("reject_at_floor_returns_false", !stepper.on_reject());
    add_check("step_unchanged_at_floor", stepper.current_step() == real{0.1});
    add_check("param_unchanged_at_floor", stepper.current_param() == real{0});

    for (const auto& [name, ok] : checks) {
        std::cout << "continuation_stepper_floor check " << name << "=" << (ok ? "PASS" : "FAIL")
                  << '\n';
    }

    return {pass,
            "continuation_stepper_floor",
            "positive",
            "eta axis, initial_step==min_step==0.1",
            0.0,
            0.0,
            "on_reject() at step<=min_step returns false after exactly one floor attempt",
            pass ? "all pass" : "some failed",
            "a rejection at a step already at the floor is a structured failure: no retry, no "
            "further halving"};
}

// ---------------------------------------------------------------------------
// Case 5: continuation_stepper_growth_streak_reset (host-only).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_stepper_growth_streak_reset() {
    const ContinuationAxisConfig config{real{0},   real{1},    real{0.1},  real{0.0125},
                                        real{0.25}, real{0.5},  real{1.5}, 2};
    ContinuationStepper stepper(config);

    bool pass = true;
    std::vector<std::tuple<std::string, bool, real>> checks;
    const auto add_check = [&](const char* name, bool ok, real observed_step) {
        checks.emplace_back(name, ok, observed_step);
        pass = pass && ok;
    };

    // The step value produced by ONE halving of config.initial_step, via the
    // SAME arithmetic ContinuationStepper::on_reject() uses
    // (backtrack_factor * step, clamped to min_step): exact since
    // multiplying/dividing by the power-of-two 0.5 introduces no additional
    // rounding beyond config.initial_step's own representation.
    const real halved_step = std::max(config.initial_step * config.backtrack_factor, config.min_step);
    // The step value produced by ONE growth of halved_step, via the SAME
    // arithmetic ContinuationStepper::on_accept() uses (growth_factor * step,
    // capped to max_step): 0.05*1.5 does not reproduce a decimal literal
    // 0.075 bitwise, so the oracle must use the identical multiplication.
    const real grown_step = std::min(halved_step * config.growth_factor, config.max_step);

    // Stage 1: easy accept. Streak=1, no growth (step stays initial_step).
    (void)stepper.attempt_param();
    stepper.on_accept();
    add_check("step_after_easy1_unchanged", stepper.current_step() == config.initial_step,
              stepper.current_step());

    // Stage 2: reject once (halve), then accept -> "halved-accept". This
    // accept has 1 halving, so it resets the streak and does NOT grow.
    (void)stepper.attempt_param();
    add_check("reject_retries", stepper.on_reject(), stepper.current_step());
    add_check("step_after_halve", stepper.current_step() == halved_step, stepper.current_step());
    (void)stepper.attempt_param();
    stepper.on_accept();
    add_check("step_after_halved_accept_unchanged", stepper.current_step() == halved_step,
              stepper.current_step());

    // Stage 3: easy accept. Streak=1 (freshly counted after the halved
    // accept's reset), no growth yet.
    (void)stepper.attempt_param();
    stepper.on_accept();
    add_check("step_after_easy3_unchanged", stepper.current_step() == halved_step,
              stepper.current_step());

    // Stage 4: easy accept. This is the 2nd consecutive easy stage
    // following the halved-accept: growth engages now, not earlier.
    (void)stepper.attempt_param();
    stepper.on_accept();
    add_check("step_after_easy4_grew", stepper.current_step() == grown_step, stepper.current_step());

    std::cout << std::setprecision(17) << "continuation_stepper_growth_streak_reset halved_step="
              << halved_step << " grown_step=" << grown_step << '\n';
    for (const auto& [name, ok, observed_step] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL")
                  << " (step_at_check=" << observed_step << ")\n";
    }

    return {pass,
            "continuation_stepper_growth_streak_reset",
            "positive",
            "eta axis defaults",
            0.0,
            0.0,
            "growth engages only after the 2 consecutive easy stages following the halved-accept",
            pass ? "all pass" : "some failed",
            "easy, halved-accept, easy, easy: the halving resets the easy streak, so growth only "
            "fires after the SECOND easy stage that follows the halved-accept, not before"};
}

// ---------------------------------------------------------------------------
// Case 6: continuation_config_validation_contract (host-only).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_config_validation_contract() {
    const StreamfunctionContinuationConfig base_config{}; // dashboard-locked defaults

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

    require_ok("defaults_valid",
              [&] { validate_streamfunction_continuation_config(base_config); });

    require_invalid("eta_start_gt_target", [&] {
        auto config = base_config;
        config.eta.start = real{2};
        config.eta.target = real{1};
        validate_streamfunction_continuation_config(config);
    });
    require_invalid("eta_initial_step_zero", [&] {
        auto config = base_config;
        config.eta.initial_step = real{0};
        validate_streamfunction_continuation_config(config);
    });
    require_invalid("eta_min_step_zero", [&] {
        auto config = base_config;
        config.eta.min_step = real{0};
        validate_streamfunction_continuation_config(config);
    });
    require_invalid("eta_min_step_gt_initial_step", [&] {
        auto config = base_config;
        config.eta.min_step = real{0.5};
        validate_streamfunction_continuation_config(config);
    });
    require_invalid("eta_initial_step_gt_max_step", [&] {
        auto config = base_config;
        config.eta.max_step = real{0.05};
        validate_streamfunction_continuation_config(config);
    });
    require_invalid("eta_backtrack_factor_zero", [&] {
        auto config = base_config;
        config.eta.backtrack_factor = real{0};
        validate_streamfunction_continuation_config(config);
    });
    require_invalid("eta_backtrack_factor_one", [&] {
        auto config = base_config;
        config.eta.backtrack_factor = real{1};
        validate_streamfunction_continuation_config(config);
    });
    require_invalid("eta_growth_factor_lt_one", [&] {
        auto config = base_config;
        config.eta.growth_factor = real{0.5};
        validate_streamfunction_continuation_config(config);
    });
    require_invalid("eta_easy_streak_zero", [&] {
        auto config = base_config;
        config.eta.easy_streak = 0;
        validate_streamfunction_continuation_config(config);
    });
    require_invalid("eta_start_nonfinite", [&] {
        auto config = base_config;
        config.eta.start = std::numeric_limits<real>::quiet_NaN();
        validate_streamfunction_continuation_config(config);
    });

    require_invalid("epsilon_start_gt_target", [&] {
        auto config = base_config;
        config.epsilon_log10.start = real{7};
        config.epsilon_log10.target = real{6};
        validate_streamfunction_continuation_config(config);
    });
    require_invalid("epsilon_initial_step_zero", [&] {
        auto config = base_config;
        config.epsilon_log10.initial_step = real{0};
        validate_streamfunction_continuation_config(config);
    });
    require_invalid("epsilon_min_step_gt_initial_step", [&] {
        auto config = base_config;
        config.epsilon_log10.min_step = real{2};
        validate_streamfunction_continuation_config(config);
    });

    std::cout << "continuation_config_validation_contract total_checks=" << checks
              << " pass=" << (pass ? "true" : "false") << '\n';

    return {pass,
            "continuation_config_validation_contract",
            "contract",
            "StreamfunctionContinuationConfig eta/epsilon_log10 mutations",
            static_cast<double>(checks),
            0.0,
            "every violated precondition throws std::invalid_argument with a distinct message",
            pass ? "all pass" : "some failed",
            "validate_streamfunction_continuation_config rejects every documented per-axis "
            "precondition violation, for both eta and epsilon_log10, with a distinct message, and "
            "accepts the dashboard-locked defaults"};
}

// ---------------------------------------------------------------------------
// Case 7: continuation_warm_start_semantics (GPU, 16^3 homogeneous).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_warm_start_semantics() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);

    CudaContext ctx(0);
    HomogeneousProblemBuffers buffers(grid);
    const StreamfunctionProblemView problem = homogeneous_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    StreamfunctionSolverConfig config = valid_config(); // initial_state = zero_source (default)
    const StreamfunctionSolveReport report1 = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("zero_source_converged", report1.status == StreamfunctionSolveStatus::converged);
    add_check("zero_source_zero_iterations", report1.picard_iterations == 0);
    const std::vector<real> u1_after_zero_source = download_fields_u1(fields);
    const std::vector<real> u2_after_zero_source = download_fields_u2(fields);
    bool zero_source_exact_zero = true;
    for (real value : u1_after_zero_source) zero_source_exact_zero = zero_source_exact_zero && value == real{0};
    for (real value : u2_after_zero_source) zero_source_exact_zero = zero_source_exact_zero && value == real{0};
    add_check("zero_source_fields_exact_zero", zero_source_exact_zero);

    config.initial_state = PicardInitialState::warm_start;
    const StreamfunctionSolveReport report2 = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();

    add_check("warm_start_converged", report2.status == StreamfunctionSolveStatus::converged);
    add_check("warm_start_zero_iterations", report2.picard_iterations == 0);

    // Top-level psi1_result/psi2_result must be default-constructed (no
    // warm-start initialization block solve exists).
    add_check("warm_start_psi1_result_default",
              report2.psi1_result.iterations == 0 && !report2.psi1_result.converged);
    add_check("warm_start_psi2_result_default",
              report2.psi2_result.iterations == 0 && !report2.psi2_result.converged);

    // History entry 0 default PCG records.
    add_check("warm_start_history_nonempty", !report2.picard_history.empty());
    const bool history0_default =
        !report2.picard_history.empty() && report2.picard_history.front().psi1_result.iterations == 0 &&
        !report2.picard_history.front().psi1_result.converged &&
        report2.picard_history.front().psi2_result.iterations == 0 &&
        !report2.picard_history.front().psi2_result.converged;
    add_check("warm_start_history0_default_pcg_records", history0_default);

    std::cout << std::setprecision(17) << "continuation_warm_start_semantics N=" << n
              << " report1_status=" << (report1.status == StreamfunctionSolveStatus::converged ? "converged" : "OTHER")
              << " report1_iterations=" << report1.picard_iterations
              << " report2_status=" << (report2.status == StreamfunctionSolveStatus::converged ? "converged" : "OTHER")
              << " report2_iterations=" << report2.picard_iterations
              << " report2_psi1_iterations=" << report2.psi1_result.iterations
              << " report2_psi1_converged=" << report2.psi1_result.converged << '\n';
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    return {pass,
            "continuation_warm_start_semantics",
            "gpu-continuation-warm-start",
            std::to_string(n) + "^3, K=1, benchmark gauge, uniform Darcy v=(1,0,0)",
            static_cast<double>(report1.picard_iterations),
            static_cast<double>(report2.picard_iterations),
            "warm_start second call converges at 0 iterations, top-level psi1/psi2_result and "
            "history[0] default-constructed",
            pass ? "all pass" : "some failed",
            "zero_source converges with exact-zero fluctuations on homogeneous K=1; re-solving the "
            "SAME fields with initial_state=warm_start must skip the zero-source init entirely and "
            "converge with 0 Picard iterations, leaving the top-level PCG results default"};
}

// ---------------------------------------------------------------------------
// Case 8: continuation_injected_failure_bitwise_restore (GPU, 16^3 homogeneous).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_injected_failure_bitwise_restore() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);

    CudaContext ctx(0);
    HomogeneousProblemBuffers buffers(grid);
    const StreamfunctionProblemView problem = homogeneous_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    StreamfunctionSolverConfig base_config = valid_config();
    StreamfunctionContinuationConfig continuation_config{}; // defaults

    struct InjectionState {
        int call_index = 0;
        std::vector<real> baseline_u1;
        std::vector<real> baseline_u2;
        bool restore_check_evaluated = false;
        bool restore_check_ok = false;
    };
    auto state = std::make_shared<InjectionState>();

    StageSolveFn stage_solver = [state](CudaContext& c, const StreamfunctionProblemView& p,
                                       const StreamfunctionSolverConfig& cfg, StreamfunctionFields& f,
                                       StreamfunctionWorkspace& w) -> StreamfunctionSolveReport {
        const int call = state->call_index++;
        if (call == 0) {
            // Baseline: dispatch to the real solver and snapshot the
            // accepted state (host-side) for later bitwise comparison.
            StreamfunctionSolveReport report = solve_streamfunctions(c, p, cfg, f, w);
            state->baseline_u1 = download_fields_u1(f);
            state->baseline_u2 = download_fields_u2(f);
            return report;
        }
        if (call == 1) {
            // First eta-leg attempt: scribble garbage into fields and
            // fabricate a not_converged report -- never a real solve. The
            // driver's snapshot of the just-accepted baseline state is a
            // pending stream-ordered `cudaMemcpyAsync` on `c.cuda_stream()`
            // (a non-blocking stream, see CudaContext); `scribble_fields`
            // below issues a legacy-default-stream BLOCKING `cudaMemcpy`,
            // which does NOT implicitly wait on that non-blocking stream, so
            // without this synchronize the scribble can race ahead of the
            // pending snapshot copy and corrupt it. Mirrors the call==2
            // rationale below.
            c.synchronize();
            scribble_fields(f, real{12345.0});
            return fabricated_report(StreamfunctionSolveStatus::not_converged);
        }
        if (call == 2) {
            // Second eta-leg attempt (post-halving retry): verify the
            // driver restored fields BITWISE from the accepted snapshot
            // before this call was made. The driver's restore is a
            // stream-ordered cudaMemcpyAsync on `c.cuda_stream()` (a
            // non-blocking stream, see CudaContext); a legacy blocking
            // cudaMemcpy from THIS test does not implicitly wait on that
            // stream, so an explicit synchronize is required before peeking
            // at device memory mid-run (the production driver never needs
            // this itself: every GPU-side consumer of the restored state is
            // stream-ordered after it on the SAME stream).
            c.synchronize();
            const std::vector<real> observed_u1 = download_fields_u1(f);
            const std::vector<real> observed_u2 = download_fields_u2(f);
            state->restore_check_evaluated = true;
            state->restore_check_ok =
                observed_u1 == state->baseline_u1 && observed_u2 == state->baseline_u2;
            return solve_streamfunctions(c, p, cfg, f, w);
        }
        return solve_streamfunctions(c, p, cfg, f, w);
    };

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionContinuationReport report = run_streamfunction_continuation(
        ctx, problem, base_config, continuation_config, fields, workspace, stage_solver);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("restore_check_evaluated", state->restore_check_evaluated);
    add_check("restore_check_bitwise_ok", state->restore_check_ok);

    // (a) next attempted eta equals the halved-step attempt.
    add_check("history_size_ge_3", report.stage_history.size() >= 3);
    const bool next_attempt_halved =
        report.stage_history.size() >= 3 && report.stage_history[1].param_attempted == real{0.1} &&
        report.stage_history[1].step_attempted == real{0.1} &&
        report.stage_history[2].param_attempted == real{0.05} &&
        report.stage_history[2].step_attempted == real{0.05};
    add_check("next_attempt_eta_halved_exactly", next_attempt_halved);

    // (c) run still ends reached_target with final_eta=1, final_epsilon=1e-6.
    add_check("status_reached_target", report.status == ContinuationStatus::reached_target);
    add_check("final_eta_eq_1", report.final_eta == real{1});
    add_check("final_epsilon_eq_1e-6", report.final_epsilon == epsilon_from_log10(real{6}));

    // (d) stage_history contains the rejected record with the documented failure.
    const bool rejected_record_present =
        report.stage_history.size() >= 2 && !report.stage_history[1].accepted &&
        report.stage_history[1].failure == ContinuationStageFailure::solver_not_converged;
    add_check("rejected_record_present", rejected_record_present);

    std::cout << std::setprecision(17) << "continuation_injected_failure_bitwise_restore N=" << n
              << " status=" << status_label(report.status) << " final_eta=" << report.final_eta
              << " final_epsilon=" << report.final_epsilon
              << " stage_history_size=" << report.stage_history.size()
              << " wall_seconds=" << wall_seconds << '\n';
    print_stage_history(report.stage_history);
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    return {pass,
            "continuation_injected_failure_bitwise_restore",
            "gpu-continuation-rollback",
            std::to_string(n) + "^3, K=1, injected not_converged failure on the first eta-leg attempt",
            static_cast<double>(report.stage_history.size()),
            static_cast<double>(report.final_eta),
            "halved retry attempt, bitwise-restored accepted state, reached_target",
            pass ? "all pass" : "some failed",
            "a StageSolveFn that scribbles fields then fabricates a not_converged report on the "
            "first eta-leg attempt: the driver must restore fields bitwise from the accepted "
            "snapshot BEFORE the halved retry, and the run must still reach the target"};
}

// ---------------------------------------------------------------------------
// Case 9: continuation_floor_exhaustion (GPU, 16^3 homogeneous).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_floor_exhaustion() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);

    CudaContext ctx(0);
    HomogeneousProblemBuffers buffers(grid);
    const StreamfunctionProblemView problem = homogeneous_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    StreamfunctionSolverConfig base_config = valid_config();
    StreamfunctionContinuationConfig continuation_config{}; // defaults

    struct InjectionState {
        int call_index = 0;
        std::vector<real> baseline_u1;
        std::vector<real> baseline_u2;
    };
    auto state = std::make_shared<InjectionState>();

    StageSolveFn stage_solver = [state](CudaContext& c, const StreamfunctionProblemView& p,
                                       const StreamfunctionSolverConfig& cfg, StreamfunctionFields& f,
                                       StreamfunctionWorkspace& w) -> StreamfunctionSolveReport {
        const int call = state->call_index++;
        if (call == 0) {
            StreamfunctionSolveReport report = solve_streamfunctions(c, p, cfg, f, w);
            state->baseline_u1 = download_fields_u1(f);
            state->baseline_u2 = download_fields_u2(f);
            return report;
        }
        // Always-fail after the baseline: never touches fields, always
        // reports not_converged.
        return fabricated_report(StreamfunctionSolveStatus::not_converged);
    };

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionContinuationReport report = run_streamfunction_continuation(
        ctx, problem, base_config, continuation_config, fields, workspace, stage_solver);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("status_step_floor_exhausted", report.status == ContinuationStatus::step_floor_exhausted);
    add_check("failed_axis_eta", report.failed_axis == ContinuationAxis::eta);

    const std::vector<real> u1_final = download_fields_u1(fields);
    const std::vector<real> u2_final = download_fields_u2(fields);
    const bool fields_match_baseline = u1_final == state->baseline_u1 && u2_final == state->baseline_u2;
    add_check("fields_bitwise_eq_baseline", fields_match_baseline);

    // stage_history[0] is the baseline; [1..4] are the 4 rejected eta
    // attempts with steps 0.1, 0.05, 0.025, 0.0125 (the last at the floor).
    const std::vector<real> expected_steps = {real{0.1}, real{0.05}, real{0.025}, real{0.0125}};
    const bool history_size_ok = report.stage_history.size() == 1 + expected_steps.size();
    add_check("history_size_eq_5", history_size_ok);

    bool halving_sequence_ok = history_size_ok;
    for (std::size_t i = 0; history_size_ok && i < expected_steps.size(); ++i) {
        const auto& record = report.stage_history[i + 1];
        halving_sequence_ok = halving_sequence_ok && record.step_attempted == expected_steps[i] &&
                              !record.accepted &&
                              record.failure == ContinuationStageFailure::solver_not_converged;
    }
    add_check("halving_sequence_exact_and_all_rejected", halving_sequence_ok);

    std::cout << std::setprecision(17) << "continuation_floor_exhaustion N=" << n
              << " status=" << status_label(report.status) << " failed_axis=" << axis_label(report.failed_axis)
              << " stage_history_size=" << report.stage_history.size() << " wall_seconds=" << wall_seconds
              << '\n';
    print_stage_history(report.stage_history);
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    return {pass,
            "continuation_floor_exhaustion",
            "gpu-continuation-floor-exhaustion",
            std::to_string(n) + "^3, K=1, always-fail stage_solver after the baseline",
            static_cast<double>(report.stage_history.size()),
            0.0,
            "step_floor_exhausted, failed_axis=eta, fields==baseline, halving 0.1,0.05,0.025,0.0125",
            pass ? "all pass" : "some failed",
            "an always-fail StageSolveFn after the baseline forces the eta stepper to halve down to "
            "its floor and exit step_floor_exhausted after exactly one floor attempt, leaving "
            "fields exactly at the baseline accepted state"};
}

// ---------------------------------------------------------------------------
// Case 10: continuation_invalid_problem_propagation (GPU, 16^3 homogeneous).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_invalid_problem_propagation() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);

    CudaContext ctx(0);
    HomogeneousProblemBuffers buffers(grid);
    const StreamfunctionProblemView problem = homogeneous_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    StreamfunctionSolverConfig base_config = valid_config();
    StreamfunctionContinuationConfig continuation_config{}; // defaults

    struct InjectionState {
        int call_index = 0;
        std::vector<real> baseline_u1;
        std::vector<real> baseline_u2;
    };
    auto state = std::make_shared<InjectionState>();

    StageSolveFn stage_solver = [state](CudaContext& c, const StreamfunctionProblemView& p,
                                       const StreamfunctionSolverConfig& cfg, StreamfunctionFields& f,
                                       StreamfunctionWorkspace& w) -> StreamfunctionSolveReport {
        const int call = state->call_index++;
        if (call == 0) {
            StreamfunctionSolveReport report = solve_streamfunctions(c, p, cfg, f, w);
            state->baseline_u1 = download_fields_u1(f);
            state->baseline_u2 = download_fields_u2(f);
            return report;
        }
        // First eta-leg attempt (no halvings ever occur): invalid_problem.
        return fabricated_report(StreamfunctionSolveStatus::invalid_problem);
    };

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionContinuationReport report = run_streamfunction_continuation(
        ctx, problem, base_config, continuation_config, fields, workspace, stage_solver);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("status_invalid_problem", report.status == ContinuationStatus::invalid_problem);
    // No halvings: exactly one eta-leg record (the invalid_problem one),
    // plus the baseline.
    add_check("history_size_eq_2", report.stage_history.size() == 2);
    const bool invalid_record_ok =
        report.stage_history.size() == 2 && !report.stage_history[1].accepted &&
        report.stage_history[1].failure == ContinuationStageFailure::solver_invalid_problem;
    add_check("invalid_problem_record_present", invalid_record_ok);

    const std::vector<real> u1_final = download_fields_u1(fields);
    const std::vector<real> u2_final = download_fields_u2(fields);
    const bool fields_match_baseline = u1_final == state->baseline_u1 && u2_final == state->baseline_u2;
    add_check("fields_bitwise_eq_baseline", fields_match_baseline);

    std::cout << std::setprecision(17) << "continuation_invalid_problem_propagation N=" << n
              << " status=" << status_label(report.status) << " stage_history_size="
              << report.stage_history.size() << " wall_seconds=" << wall_seconds << '\n';
    print_stage_history(report.stage_history);
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    return {pass,
            "continuation_invalid_problem_propagation",
            "gpu-continuation-invalid-problem",
            std::to_string(n) + "^3, K=1, invalid_problem injected on the first eta-leg attempt",
            static_cast<double>(report.stage_history.size()),
            0.0,
            "invalid_problem propagates immediately, no halvings, fields==baseline",
            pass ? "all pass" : "some failed",
            "a StageSolveFn returning invalid_problem mid-eta-leg must propagate status="
            "invalid_problem immediately, with no step halving, leaving fields at the last "
            "accepted (baseline) state"};
}

// ---------------------------------------------------------------------------
// Case 11: continuation_trig_control (GPU, 32^3, headline acceptance case).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_trig_control() {
    constexpr int n = 32;
    constexpr double kAmplitude = 0.5;
    const Grid3D grid = isotropic_grid(n);

    CudaContext ctx(0);
    ManufacturedProblemBuffers buffers(grid, kAmplitude);
    const StreamfunctionProblemView problem = manufactured_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    const StreamfunctionSolverConfig base_config = valid_config(); // defaults: adaptive enabled, tolerance 1e-6
    const StreamfunctionContinuationConfig continuation_config{}; // defaults

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionContinuationReport report =
        run_streamfunction_continuation(ctx, problem, base_config, continuation_config, fields, workspace);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("status_reached_target", report.status == ContinuationStatus::reached_target);
    add_check("final_eta_eq_1", report.final_eta == real{1});
    add_check("final_epsilon_eq_1e-6", report.final_epsilon == epsilon_from_log10(real{6}));

    bool every_accepted_stage_within_tolerance = true;
    std::size_t accepted_count = 0;
    for (const auto& record : report.stage_history) {
        if (record.accepted) {
            ++accepted_count;
            every_accepted_stage_within_tolerance =
                every_accepted_stage_within_tolerance && static_cast<double>(record.r_F) <= 1e-6;
        }
    }
    add_check("every_accepted_stage_r_F_le_1e-6", every_accepted_stage_within_tolerance);
    add_check("stage_history_nonempty", !report.stage_history.empty());

    const std::size_t n_cells = grid.num_cells();
    add_check("snapshot_bytes_exact",
              report.snapshot_bytes == 2 * static_cast<std::size_t>(n_cells) * sizeof(double));

    std::cout << std::setprecision(17) << "continuation_trig_control N=" << n << " a=" << kAmplitude
              << " status=" << status_label(report.status) << " final_eta=" << report.final_eta
              << " final_epsilon=" << report.final_epsilon << " accepted_stages=" << accepted_count
              << " total_stages=" << report.stage_history.size() << " snapshot_bytes="
              << report.snapshot_bytes << " wall_seconds=" << wall_seconds << '\n';
    print_stage_history(report.stage_history);
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << n << "^3, K=exp(" << kAmplitude
           << "*sin(2pi x)sin(2pi y)sin(2pi z)), benchmark gauge, uniform Darcy v=(1,0,0), "
              "adaptive Picard + default eta/epsilon continuation";

    return {pass,
            "continuation_trig_control",
            "gpu-continuation-trig-headline",
            detail.str(),
            static_cast<double>(accepted_count),
            static_cast<double>(report.stage_history.size()),
            "reached_target, final_eta=1, final_epsilon=1e-6, every accepted stage r_F<=1e-6",
            pass ? "all pass" : "some failed",
            "the headline SF-17 acceptance case: a smooth trigonometric log-conductivity field "
            "(a=0.5) that the SF-14/15 fixed-amplitude research case did not gate on must reach the "
            "unregularized (eta=1, epsilon=1e-6) target via adaptive eta/epsilon continuation"};
}

// ---------------------------------------------------------------------------
// Case 12: continuation_homogeneous_control (GPU, 16^3, K=1).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_continuation_homogeneous_control() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);

    CudaContext ctx(0);
    HomogeneousProblemBuffers buffers(grid);
    const StreamfunctionProblemView problem = homogeneous_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    const StreamfunctionSolverConfig base_config = valid_config();
    const StreamfunctionContinuationConfig continuation_config{};

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionContinuationReport report =
        run_streamfunction_continuation(ctx, problem, base_config, continuation_config, fields, workspace);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("status_reached_target", report.status == ContinuationStatus::reached_target);
    add_check("final_eta_eq_1", report.final_eta == real{1});
    add_check("final_epsilon_eq_1e-6", report.final_epsilon == epsilon_from_log10(real{6}));

    bool every_stage_accepted_first_try = !report.stage_history.empty();
    bool every_stage_zero_iterations = true;
    bool every_stage_r_f_exact_zero = true;
    for (const auto& record : report.stage_history) {
        every_stage_accepted_first_try = every_stage_accepted_first_try && record.accepted;
        every_stage_zero_iterations = every_stage_zero_iterations && record.picard_iterations == 0;
        every_stage_r_f_exact_zero = every_stage_r_f_exact_zero && record.r_F == real{0};
    }
    add_check("every_stage_accepted_first_attempt", every_stage_accepted_first_try);
    add_check("every_stage_zero_picard_iterations", every_stage_zero_iterations);
    add_check("every_stage_r_F_exact_zero", every_stage_r_f_exact_zero);

    // Eta attempts must match the exact no-failure sequence (case 1), and
    // epsilon attempts the exact decades (case 2). The eta oracle is
    // recomputed via the SAME arithmetic ContinuationStepper uses (see case
    // 1's rationale): decimal literals like 0.35 do not reproduce 0.2+0.15
    // bitwise.
    std::vector<real> expected_eta_attempts;
    {
        const StreamfunctionContinuationConfig defaults{};
        real param = defaults.eta.start;
        real step = defaults.eta.initial_step;
        int streak = 0;
        while (param < defaults.eta.target) {
            const real attempt = std::min(param + step, defaults.eta.target);
            expected_eta_attempts.push_back(attempt);
            param = attempt;
            ++streak;
            if (streak >= defaults.eta.easy_streak) {
                step = std::min(step * defaults.eta.growth_factor, defaults.eta.max_step);
                streak = 0;
            }
        }
    }
    const std::vector<real> expected_epsilon_attempts = {real{1e-3}, real{1e-4}, real{1e-5}, real{1e-6}};

    std::vector<real> observed_eta_attempts;
    std::vector<real> observed_epsilon_attempts;
    for (const auto& record : report.stage_history) {
        if (record.axis == ContinuationAxis::eta && record.step_attempted != real{0}) {
            observed_eta_attempts.push_back(record.param_attempted);
        } else if (record.axis == ContinuationAxis::epsilon) {
            observed_epsilon_attempts.push_back(record.param_attempted);
        }
    }
    bool eta_sequence_exact = observed_eta_attempts.size() == expected_eta_attempts.size();
    for (std::size_t i = 0; eta_sequence_exact && i < expected_eta_attempts.size(); ++i) {
        eta_sequence_exact = eta_sequence_exact && observed_eta_attempts[i] == expected_eta_attempts[i];
    }
    add_check("eta_attempt_sequence_exact", eta_sequence_exact);

    bool epsilon_sequence_exact = observed_epsilon_attempts.size() == expected_epsilon_attempts.size();
    for (std::size_t i = 0; epsilon_sequence_exact && i < expected_epsilon_attempts.size(); ++i) {
        epsilon_sequence_exact =
            epsilon_sequence_exact && observed_epsilon_attempts[i] == expected_epsilon_attempts[i];
    }
    add_check("epsilon_attempt_sequence_exact_decades", epsilon_sequence_exact);

    std::cout << std::setprecision(17) << "continuation_homogeneous_control N=" << n
              << " status=" << status_label(report.status) << " final_eta=" << report.final_eta
              << " final_epsilon=" << report.final_epsilon
              << " stage_history_size=" << report.stage_history.size() << " wall_seconds=" << wall_seconds
              << '\n';
    print_stage_history(report.stage_history);
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    return {pass,
            "continuation_homogeneous_control",
            "gpu-continuation-homogeneous-control",
            std::to_string(n) + "^3, K=1, benchmark gauge, uniform Darcy v=(1,0,0), full defaults",
            static_cast<double>(report.stage_history.size()),
            static_cast<double>(report.final_eta),
            "reached_target, every stage accepted first try, 0 Picard iterations, r_F==0 exactly, "
            "exact eta/epsilon attempt sequences",
            pass ? "all pass" : "some failed",
            "homogeneous K=1 makes the nonlinear source exactly zero at u1=u2=0 for any (eta, "
            "epsilon), so every continuation stage (warm-started at the exact fixed point) accepts "
            "on the first attempt with r_F==0.0 exactly"};
}

} // namespace

CaseRegistry streamfunction_continuation_case_registry() {
    return {{"continuation_stepper_eta_all_accept", case_continuation_stepper_eta_all_accept},
            {"continuation_stepper_epsilon_all_accept", case_continuation_stepper_epsilon_all_accept},
            {"continuation_stepper_reject_halve", case_continuation_stepper_reject_halve},
            {"continuation_stepper_floor", case_continuation_stepper_floor},
            {"continuation_stepper_growth_streak_reset", case_continuation_stepper_growth_streak_reset},
            {"continuation_config_validation_contract", case_continuation_config_validation_contract},
            {"continuation_warm_start_semantics", case_continuation_warm_start_semantics},
            {"continuation_injected_failure_bitwise_restore",
             case_continuation_injected_failure_bitwise_restore},
            {"continuation_floor_exhaustion", case_continuation_floor_exhaustion},
            {"continuation_invalid_problem_propagation", case_continuation_invalid_problem_propagation},
            {"continuation_trig_control", case_continuation_trig_control},
            {"continuation_homogeneous_control", case_continuation_homogeneous_control}};
}

} // namespace macroflow3d::streamfunctions::test
