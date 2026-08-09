#include "ContinuationController.hpp"

#include "../../core/DeviceBuffer.cuh"
#include "../../runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace streamfunctions {
namespace {

void require_finite(real value, const char* message) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(message);
    }
}

void validate_continuation_axis_config(const ContinuationAxisConfig& c, const char* axis) {
    // Finiteness first: every subsequent comparison assumes finite operands.
    std::string prefix = std::string("StreamfunctionContinuationConfig.") + axis + ": ";
    require_finite(c.start, (prefix + "start must be finite").c_str());
    require_finite(c.target, (prefix + "target must be finite").c_str());
    require_finite(c.initial_step, (prefix + "initial_step must be finite").c_str());
    require_finite(c.min_step, (prefix + "min_step must be finite").c_str());
    require_finite(c.max_step, (prefix + "max_step must be finite").c_str());
    require_finite(c.backtrack_factor, (prefix + "backtrack_factor must be finite").c_str());
    require_finite(c.growth_factor, (prefix + "growth_factor must be finite").c_str());

    if (!(c.start <= c.target)) {
        throw std::invalid_argument(prefix + "start must be <= target");
    }
    if (!(c.initial_step > real{0})) {
        throw std::invalid_argument(prefix + "initial_step must be > 0");
    }
    if (!(c.min_step > real{0})) {
        throw std::invalid_argument(prefix + "min_step must be > 0");
    }
    if (!(c.min_step <= c.initial_step)) {
        throw std::invalid_argument(prefix + "min_step must be <= initial_step");
    }
    if (!(c.initial_step <= c.max_step)) {
        throw std::invalid_argument(prefix + "initial_step must be <= max_step");
    }
    if (!(c.backtrack_factor > real{0} && c.backtrack_factor < real{1})) {
        throw std::invalid_argument(prefix + "backtrack_factor must be in (0, 1)");
    }
    if (!(c.growth_factor >= real{1})) {
        throw std::invalid_argument(prefix + "growth_factor must be >= 1");
    }
    if (!(c.easy_streak >= 1)) {
        throw std::invalid_argument(prefix + "easy_streak must be >= 1");
    }
}

} // namespace

void validate_streamfunction_continuation_config(const StreamfunctionContinuationConfig& config) {
    validate_continuation_axis_config(config.eta, "eta");
    validate_continuation_axis_config(config.epsilon_log10, "epsilon_log10");
}

real epsilon_from_log10(real p) noexcept { return std::pow(real{10}, -p); }

ContinuationStepper::ContinuationStepper(const ContinuationAxisConfig& config)
    : config_(config), param_(config.start), step_(config.initial_step) {}

real ContinuationStepper::attempt_param() const noexcept {
    return std::min(param_ + step_, config_.target);
}

void ContinuationStepper::on_accept() {
    param_ = attempt_param();

    const bool easy = (halvings_in_attempt_ == 0);
    if (easy) {
        ++easy_streak_count_;
        if (easy_streak_count_ >= config_.easy_streak) {
            step_ = std::min(step_ * config_.growth_factor, config_.max_step);
            easy_streak_count_ = 0;
        }
    } else {
        easy_streak_count_ = 0;
    }
    halvings_in_attempt_ = 0;
}

bool ContinuationStepper::on_reject() {
    if (step_ <= config_.min_step) {
        // Exactly one floor attempt: structured failure, no further retry.
        return false;
    }
    step_ = std::max(step_ * config_.backtrack_factor, config_.min_step);
    ++halvings_in_attempt_;
    easy_streak_count_ = 0; // Any halving resets the easy streak.
    return true;
}

namespace {

// Snapshot fields -> the driver-owned accepted-state buffers, D2D,
// stream-ordered.
void snapshot_fields(CudaContext& context, const StreamfunctionFields& fields,
                     DeviceBuffer<real>& accepted_u1, DeviceBuffer<real>& accepted_u2,
                     std::size_t n) {
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(accepted_u1.data(), fields.u1_span().data(),
                                           n * sizeof(real), cudaMemcpyDeviceToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(accepted_u2.data(), fields.u2_span().data(),
                                           n * sizeof(real), cudaMemcpyDeviceToDevice,
                                           context.cuda_stream()));
}

// Restore the accepted-state snapshot -> fields, D2D, stream-ordered.
void restore_fields(CudaContext& context, StreamfunctionFields& fields,
                    const DeviceBuffer<real>& accepted_u1, const DeviceBuffer<real>& accepted_u2,
                    std::size_t n) {
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(fields.u1_span().data(), accepted_u1.data(),
                                           n * sizeof(real), cudaMemcpyDeviceToDevice,
                                           context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(fields.u2_span().data(), accepted_u2.data(),
                                           n * sizeof(real), cudaMemcpyDeviceToDevice,
                                           context.cuda_stream()));
}

ContinuationStageFailure failure_for_status(StreamfunctionSolveStatus status) {
    switch (status) {
    case StreamfunctionSolveStatus::not_converged:
        return ContinuationStageFailure::solver_not_converged;
    case StreamfunctionSolveStatus::invalid_problem:
        return ContinuationStageFailure::solver_invalid_problem;
    default:
        return ContinuationStageFailure::none;
    }
}

ContinuationStageRecord make_stage_record(ContinuationAxis axis, real param_start,
                                          real param_attempted, real step_attempted,
                                          const StreamfunctionSolveReport& report,
                                          const StreamfunctionSolverConfig& stage_config,
                                          std::size_t n) {
    ContinuationStageRecord record;
    record.axis = axis;
    record.param_start = param_start;
    record.param_attempted = param_attempted;
    record.step_attempted = step_attempted;
    record.accepted = (report.status == StreamfunctionSolveStatus::converged);
    record.failure = failure_for_status(report.status);

    record.exit_reason = report.exit_reason;
    record.picard_iterations = report.picard_iterations;
    record.final_omega = report.final_omega;
    record.r_F = report.residual.r_F;
    record.r1 = report.residual.r1;
    record.r2 = report.residual.r2;

    record.unexplained_fraction =
        stage_config.diagnostics.num_degeneracy_thresholds > 0
            ? static_cast<real>(report.diagnostics.degeneracy_unexplained[0]) /
                  static_cast<real>(n)
            : real{0};
    record.c_percentile_p001 = residual_histogram_percentile(report.residual, real{0.001});

    record.psi1_iterations = report.psi1_result.iterations;
    record.psi2_iterations = report.psi2_result.iterations;

    return record;
}

} // namespace

StreamfunctionContinuationReport run_streamfunction_continuation(
    CudaContext& context, const StreamfunctionProblemView& problem,
    const StreamfunctionSolverConfig& base_config,
    const StreamfunctionContinuationConfig& continuation_config, StreamfunctionFields& fields,
    StreamfunctionWorkspace& workspace, StageSolveFn stage_solver) {
    // Host misuse throws std::invalid_argument; not caught here.
    validate_streamfunction_continuation_config(continuation_config);

    if (!stage_solver) {
        stage_solver = [](CudaContext& ctx, const StreamfunctionProblemView& p,
                          const StreamfunctionSolverConfig& cfg, StreamfunctionFields& f,
                          StreamfunctionWorkspace& w) { return solve_streamfunctions(ctx, p, cfg, f, w); };
    }

    const std::size_t n = problem.grid.num_cells();

    // Driver-OWNED accepted-state snapshot; NOT part of StreamfunctionWorkspace
    // (see the file header). Allocated once, up front: no allocation occurs
    // inside the stage loop below.
    DeviceBuffer<real> accepted_u1(n);
    DeviceBuffer<real> accepted_u2(n);

    StreamfunctionContinuationReport report;
    report.snapshot_bytes = (accepted_u1.capacity() + accepted_u2.capacity()) * sizeof(real);

    // --- Baseline stage: zero_source at (eta.start, epsilon(epsilon_log10.start)). ---
    StreamfunctionSolverConfig baseline_config = base_config;
    baseline_config.eta = continuation_config.eta.start;
    baseline_config.epsilon = epsilon_from_log10(continuation_config.epsilon_log10.start);
    baseline_config.initial_state = PicardInitialState::zero_source;

    StreamfunctionSolveReport baseline_report =
        stage_solver(context, problem, baseline_config, fields, workspace);

    report.stage_history.push_back(make_stage_record(
        ContinuationAxis::eta, continuation_config.eta.start, continuation_config.eta.start,
        real{0}, baseline_report, baseline_config, n));

    if (baseline_report.status == StreamfunctionSolveStatus::invalid_problem) {
        report.status = ContinuationStatus::invalid_problem;
        report.failed_axis = ContinuationAxis::eta;
        report.final_solve = baseline_report;
        return report;
    }
    if (baseline_report.status != StreamfunctionSolveStatus::converged) {
        // No accepted state ever existed; fields are left exactly as the
        // failed baseline attempt produced them (SF-13..16 semantics; there
        // is nothing earlier to roll back to).
        report.status = ContinuationStatus::baseline_failed;
        report.failed_axis = ContinuationAxis::eta;
        report.final_solve = baseline_report;
        return report;
    }

    // Baseline accepted: establish the first accepted-state snapshot.
    snapshot_fields(context, fields, accepted_u1, accepted_u2, n);
    real accepted_eta = continuation_config.eta.start;
    real accepted_epsilon_log10 = continuation_config.epsilon_log10.start;
    report.final_solve = baseline_report;

    // --- Eta leg: fixed epsilon, warm-started stages. ---
    ContinuationStepper eta_stepper(continuation_config.eta);
    while (!eta_stepper.reached_target()) {
        const real param_start = eta_stepper.current_param();
        const real attempt = eta_stepper.attempt_param();
        const real step_attempted = eta_stepper.current_step();

        StreamfunctionSolverConfig stage_config = base_config;
        stage_config.eta = attempt;
        stage_config.epsilon = epsilon_from_log10(accepted_epsilon_log10);
        stage_config.initial_state = PicardInitialState::warm_start;

        StreamfunctionSolveReport stage_report =
            stage_solver(context, problem, stage_config, fields, workspace);

        report.stage_history.push_back(make_stage_record(ContinuationAxis::eta, param_start,
                                                          attempt, step_attempted, stage_report,
                                                          stage_config, n));

        if (stage_report.status == StreamfunctionSolveStatus::invalid_problem) {
            restore_fields(context, fields, accepted_u1, accepted_u2, n);
            report.status = ContinuationStatus::invalid_problem;
            report.failed_axis = ContinuationAxis::eta;
            report.final_eta = accepted_eta;
            report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
            return report;
        }
        if (stage_report.status == StreamfunctionSolveStatus::converged) {
            snapshot_fields(context, fields, accepted_u1, accepted_u2, n);
            accepted_eta = attempt;
            eta_stepper.on_accept();
            report.final_solve = stage_report;
        } else {
            restore_fields(context, fields, accepted_u1, accepted_u2, n);
            if (!eta_stepper.on_reject()) {
                report.status = ContinuationStatus::step_floor_exhausted;
                report.failed_axis = ContinuationAxis::eta;
                report.final_eta = accepted_eta;
                report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
                return report;
            }
        }
    }

    // --- Epsilon leg: fixed final eta, warm-started stages. ---
    ContinuationStepper epsilon_stepper(continuation_config.epsilon_log10);
    while (!epsilon_stepper.reached_target()) {
        const real param_start_log10 = epsilon_stepper.current_param();
        const real attempt_log10 = epsilon_stepper.attempt_param();
        const real step_attempted = epsilon_stepper.current_step();

        StreamfunctionSolverConfig stage_config = base_config;
        stage_config.eta = accepted_eta;
        stage_config.epsilon = epsilon_from_log10(attempt_log10);
        stage_config.initial_state = PicardInitialState::warm_start;

        StreamfunctionSolveReport stage_report =
            stage_solver(context, problem, stage_config, fields, workspace);

        report.stage_history.push_back(make_stage_record(
            ContinuationAxis::epsilon, epsilon_from_log10(param_start_log10),
            epsilon_from_log10(attempt_log10), step_attempted, stage_report, stage_config, n));

        if (stage_report.status == StreamfunctionSolveStatus::invalid_problem) {
            restore_fields(context, fields, accepted_u1, accepted_u2, n);
            report.status = ContinuationStatus::invalid_problem;
            report.failed_axis = ContinuationAxis::epsilon;
            report.final_eta = accepted_eta;
            report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
            return report;
        }
        if (stage_report.status == StreamfunctionSolveStatus::converged) {
            snapshot_fields(context, fields, accepted_u1, accepted_u2, n);
            accepted_epsilon_log10 = attempt_log10;
            epsilon_stepper.on_accept();
            report.final_solve = stage_report;
        } else {
            restore_fields(context, fields, accepted_u1, accepted_u2, n);
            if (!epsilon_stepper.on_reject()) {
                report.status = ContinuationStatus::step_floor_exhausted;
                report.failed_axis = ContinuationAxis::epsilon;
                report.final_eta = accepted_eta;
                report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
                return report;
            }
        }
    }

    report.status = ContinuationStatus::reached_target;
    // failed_axis is meaningful only on failure; left at its default on
    // success.
    report.final_eta = accepted_eta;
    report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
    return report;
}

} // namespace streamfunctions
} // namespace macroflow3d
