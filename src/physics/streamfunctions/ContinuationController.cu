#include "ContinuationController.hpp"

#include "../../core/DeviceBuffer.cuh"
#include "../../numerics/blas/scal.cuh"
#include "../../runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
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

// Sink for one ContinuationStageRecord + the StreamfunctionSolveReport it was
// built from (SF-21 needs the full report to harvest Gate-3A metrics per
// stage; SF-17 only keeps the trimmed record).
using StageRecordSink =
    std::function<void(const ContinuationStageRecord&, const StreamfunctionSolveReport&)>;

// Shared by run_streamfunction_continuation (SF-17) and
// run_streamfunction_heterogeneity_continuation (SF-21): runs the
// epsilon_log10 stepper leg at a fixed eta, every stage warm-started, exactly
// the SF-17 epsilon-leg semantics documented in the file header. `base_config`
// supplies every field a stage config copies verbatim except `eta` (set to
// `fixed_eta`), `epsilon` (set to the attempted physical value), and
// `initial_state` (always `warm_start`) -- notably `base_config.
// coefficient_state` is NOT touched here, so SF-17 (which leaves it at the
// default `rebuild`) and SF-21 (which passes a `base_config` with `reuse`
// already set) get their own, distinct, unmodified behavior. `accepted_u1`/
// `accepted_u2` are the caller-owned accepted-state snapshot pair this leg
// snapshots into/restores from; `accepted_epsilon_log10`/`final_solve` are
// updated in place on every accepted stage. Returns `reached_target`,
// `step_floor_exhausted`, or `invalid_problem`; the caller is responsible for
// translating that into its own status/failed-axis representation.
ContinuationStatus run_epsilon_leg(CudaContext& context, const StreamfunctionProblemView& problem,
                                   const StreamfunctionSolverConfig& base_config,
                                   const ContinuationAxisConfig& epsilon_axis, real fixed_eta,
                                   StageSolveFn& stage_solver, StreamfunctionFields& fields,
                                   StreamfunctionWorkspace& workspace,
                                   DeviceBuffer<real>& accepted_u1, DeviceBuffer<real>& accepted_u2,
                                   std::size_t n, real& accepted_epsilon_log10,
                                   StreamfunctionSolveReport& final_solve,
                                   const StageRecordSink& sink) {
    ContinuationStepper epsilon_stepper(epsilon_axis);
    while (!epsilon_stepper.reached_target()) {
        const real param_start_log10 = epsilon_stepper.current_param();
        const real attempt_log10 = epsilon_stepper.attempt_param();
        const real step_attempted = epsilon_stepper.current_step();

        StreamfunctionSolverConfig stage_config = base_config;
        stage_config.eta = fixed_eta;
        stage_config.epsilon = epsilon_from_log10(attempt_log10);
        stage_config.initial_state = PicardInitialState::warm_start;

        StreamfunctionSolveReport stage_report =
            stage_solver(context, problem, stage_config, fields, workspace);

        const ContinuationStageRecord record = make_stage_record(
            ContinuationAxis::epsilon, epsilon_from_log10(param_start_log10),
            epsilon_from_log10(attempt_log10), step_attempted, stage_report, stage_config, n);
        sink(record, stage_report);

        if (stage_report.status == StreamfunctionSolveStatus::invalid_problem) {
            restore_fields(context, fields, accepted_u1, accepted_u2, n);
            return ContinuationStatus::invalid_problem;
        }
        if (stage_report.status == StreamfunctionSolveStatus::converged) {
            snapshot_fields(context, fields, accepted_u1, accepted_u2, n);
            accepted_epsilon_log10 = attempt_log10;
            epsilon_stepper.on_accept();
            final_solve = stage_report;
        } else {
            restore_fields(context, fields, accepted_u1, accepted_u2, n);
            if (!epsilon_stepper.on_reject()) {
                return ContinuationStatus::step_floor_exhausted;
            }
        }
    }
    return ContinuationStatus::reached_target;
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

    // --- Epsilon leg: fixed final eta, warm-started stages. Shared with
    // SF-21 via run_epsilon_leg (see its doc comment); base_config here still
    // leaves coefficient_state at its caller-supplied default (rebuild),
    // exactly reproducing the pre-refactor behavior bitwise. ---
    const ContinuationStatus epsilon_status = run_epsilon_leg(
        context, problem, base_config, continuation_config.epsilon_log10, accepted_eta,
        stage_solver, fields, workspace, accepted_u1, accepted_u2, n, accepted_epsilon_log10,
        report.final_solve,
        [&report](const ContinuationStageRecord& record, const StreamfunctionSolveReport&) {
            report.stage_history.push_back(record);
        });

    if (epsilon_status == ContinuationStatus::invalid_problem) {
        report.status = ContinuationStatus::invalid_problem;
        report.failed_axis = ContinuationAxis::epsilon;
        report.final_eta = accepted_eta;
        report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
        return report;
    }
    if (epsilon_status == ContinuationStatus::step_floor_exhausted) {
        report.status = ContinuationStatus::step_floor_exhausted;
        report.failed_axis = ContinuationAxis::epsilon;
        report.final_eta = accepted_eta;
        report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
        return report;
    }

    report.status = ContinuationStatus::reached_target;
    // failed_axis is meaningful only on failure; left at its default on
    // success.
    report.final_eta = accepted_eta;
    report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
    return report;
}

namespace {

// --- SF-21 heterogeneity continuation helpers. ---

std::size_t heterogeneity_compact_mac_u_size(const Grid3D& g) {
    return static_cast<std::size_t>(g.nx + 1) * static_cast<std::size_t>(g.ny) *
           static_cast<std::size_t>(g.nz);
}
std::size_t heterogeneity_compact_mac_v_size(const Grid3D& g) {
    return static_cast<std::size_t>(g.nx) * static_cast<std::size_t>(g.ny + 1) *
           static_cast<std::size_t>(g.nz);
}
std::size_t heterogeneity_compact_mac_w_size(const Grid3D& g) {
    return static_cast<std::size_t>(g.nx) * static_cast<std::size_t>(g.ny) *
           static_cast<std::size_t>(g.nz + 1);
}

// K_lambda = exp(Y_lambda), elementwise. Y_lambda = lambda*Y is built by the
// caller with a plain D2D copy + blas::scal (no new kernel needed for that
// part); this is the one small elementwise kernel the increment spec calls
// for.
__global__ void heterogeneity_exp_kernel(const real* y_lambda, real* k_lambda, std::size_t n) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < n; index += stride) {
        k_lambda[index] = exp(y_lambda[index]);
    }
}

void enqueue_heterogeneity_conductivity(CudaContext& ctx, DeviceSpan<const real> y_lambda,
                                        DeviceSpan<real> k_lambda) {
    const std::size_t n = k_lambda.size();
    if (n == 0) {
        return;
    }
    constexpr int kHeterogeneityBlockSize = 256;
    constexpr int kHeterogeneityMaxBlocks = 65535;
    const std::size_t requested_blocks = (n + kHeterogeneityBlockSize - 1) / kHeterogeneityBlockSize;
    const int blocks = static_cast<int>(
        requested_blocks < static_cast<std::size_t>(kHeterogeneityMaxBlocks)
            ? requested_blocks
            : kHeterogeneityMaxBlocks);
    heterogeneity_exp_kernel<<<blocks, kHeterogeneityBlockSize, 0, ctx.cuda_stream()>>>(
        y_lambda.data(), k_lambda.data(), n);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

// Wraps a ContinuationStageRecord + the StreamfunctionSolveReport it was
// built from into one HeterogeneityStageRecord, harvesting the Gate-3A
// physical metrics documented on HeterogeneityStageRecord.
HeterogeneityStageRecord make_heterogeneity_stage_record(HeterogeneityAxis axis,
                                                          const ContinuationStageRecord& base,
                                                          real lambda_value, real eta_value,
                                                          real epsilon_value, int mg_rebuild_count,
                                                          const StreamfunctionSolveReport& report) {
    HeterogeneityStageRecord record;
    record.base = base;
    record.axis = axis;
    record.lambda_value = lambda_value;
    record.eta_value = eta_value;
    record.epsilon_value = epsilon_value;
    record.mg_rebuild_count = mg_rebuild_count;

    record.e_v = report.diagnostics.e_v;
    record.invariance_e_psi1 = report.diagnostics.invariance_e_psi1;
    record.invariance_e_psi2 = report.diagnostics.invariance_e_psi2;
    record.e_div = report.diagnostics.e_div;
    record.c_percentile_p001 = base.c_percentile_p001;
    if (report.diagnostics.num_degeneracy_thresholds > 0) {
        record.degeneracy_total0 = report.diagnostics.degeneracy_total[0];
        record.degeneracy_unexplained0 = report.diagnostics.degeneracy_unexplained[0];
    }

    // SF-25 per-stage Anderson/Newton attribution: copied verbatim off this
    // SAME StreamfunctionSolveReport, for every stage kind (baseline, lambda
    // attempt, eta-rescue eta=0/ramp, epsilon) since this is the single
    // record-builder site every call site routes through.
    record.anderson_accepted = report.anderson_accepted;
    record.anderson_rejected = report.anderson_rejected;
    record.anderson_condition_resets = report.anderson_condition_resets;
    record.newton_activations = report.newton_activations;
    record.newton_steps_accepted = report.newton_steps_accepted;
    record.newton_step_failures = report.newton_step_failures;
    record.newton_rescue_events = report.newton_rescue_events;
    record.newton_jv_evaluations = report.newton_jv_evaluations;

    return record;
}

} // namespace

void validate_streamfunction_heterogeneity_continuation_config(
    const HeterogeneityContinuationConfig& config) {
    validate_continuation_axis_config(config.lambda, "lambda");
    validate_streamfunction_continuation_config(config.inner);
}

HeterogeneityContinuationReport run_streamfunction_heterogeneity_continuation(
    CudaContext& context, const Grid3D& grid, DeviceSpan<const real> Y,
    const HeterogeneityContinuationConfig& continuation_config,
    const physics::AffinePeriodicFlowConfig& flow_config, StreamfunctionSolverConfig base_config,
    StreamfunctionFields& fields, StreamfunctionWorkspace& workspace, StageSolveFn stage_solver) {
    // Host misuse throws std::invalid_argument; not caught here.
    validate_streamfunction_heterogeneity_continuation_config(continuation_config);

    const std::size_t n = grid.num_cells();
    if (Y.size() != n) {
        throw std::invalid_argument(
            "run_streamfunction_heterogeneity_continuation requires Y sized grid.num_cells()");
    }

    if (!stage_solver) {
        stage_solver = [](CudaContext& ctx, const StreamfunctionProblemView& p,
                          const StreamfunctionSolverConfig& cfg, StreamfunctionFields& f,
                          StreamfunctionWorkspace& w) { return solve_streamfunctions(ctx, p, cfg, f, w); };
    }

    const std::size_t u_size = heterogeneity_compact_mac_u_size(grid);
    const std::size_t v_size = heterogeneity_compact_mac_v_size(grid);
    const std::size_t w_size = heterogeneity_compact_mac_w_size(grid);

    // Driver-OWNED buffers, allocated ONCE up front: no allocation occurs
    // inside any loop below (see the file header).
    DeviceBuffer<real> y_lambda(n);
    DeviceBuffer<real> k_lambda(n);
    DeviceBuffer<real> flow_u(u_size);
    DeviceBuffer<real> flow_v(v_size);
    DeviceBuffer<real> flow_w(w_size);
    DeviceBuffer<real> accepted_u1(n);
    DeviceBuffer<real> accepted_u2(n);
    DeviceBuffer<real> rescue_u1(n);
    DeviceBuffer<real> rescue_u2(n);
    physics::AffinePeriodicFlowWorkspace flow_workspace;

    HeterogeneityContinuationReport report;
    report.snapshot_bytes = (accepted_u1.capacity() + accepted_u2.capacity() +
                             rescue_u1.capacity() + rescue_u2.capacity()) *
                            sizeof(real);
    report.driver_owned_bytes =
        report.snapshot_bytes + (y_lambda.capacity() + k_lambda.capacity() + flow_u.capacity() +
                                 flow_v.capacity() + flow_w.capacity()) *
                                     sizeof(real);

    int mg_rebuild_count = 0;

    BCSpec bc;
    bc.xmin = BCFace(BCType::Periodic, real{0});
    bc.xmax = BCFace(BCType::Periodic, real{0});
    bc.ymin = BCFace(BCType::Periodic, real{0});
    bc.ymax = BCFace(BCType::Periodic, real{0});
    bc.zmin = BCFace(BCType::Periodic, real{0});
    bc.zmax = BCFace(BCType::Periodic, real{0});
    const AffineGauge gauge = AffineGauge::benchmark(real{1});
    const real eps_start = epsilon_from_log10(continuation_config.inner.epsilon_log10.start);

    // Build Y_lambda = lambda*Y, K_lambda = exp(Y_lambda), and re-run the
    // SF-19 affine-periodic Darcy solve on K_lambda; no allocation (every
    // buffer sized above), one D2D copy + blas::scal + the exp kernel + one
    // flow solve, all stream-ordered on context.cuda_stream().
    auto build_attempt = [&](real lambda_value) -> physics::AffinePeriodicFlowReport {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(y_lambda.data(), Y.data(), n * sizeof(real),
                                               cudaMemcpyDeviceToDevice, context.cuda_stream()));
        blas::scal(context, y_lambda.span(), lambda_value);
        enqueue_heterogeneity_conductivity(context, DeviceSpan<const real>(y_lambda.span()),
                                           k_lambda.span());

        physics::AffinePeriodicVelocityView velocity{flow_u.span(), flow_v.span(), flow_w.span()};
        return physics::solve_affine_periodic_flow(context, grid,
                                                    DeviceSpan<const real>(k_lambda.span()),
                                                    flow_config, velocity, flow_workspace);
    };

    auto make_problem_view = [&]() -> StreamfunctionProblemView {
        StreamfunctionProblemView view;
        view.grid = grid;
        view.conductivity = DeviceSpan<const real>(y_lambda.span());
        view.conductivity_representation = ConductivityRepresentation::log_conductivity_y;
        view.darcy_velocity =
            CompactMacVelocityConstView{DeviceSpan<const real>(flow_u.span()),
                                        DeviceSpan<const real>(flow_v.span()),
                                        DeviceSpan<const real>(flow_w.span())};
        view.bc = bc;
        view.gauge = gauge;
        return view;
    };

    // --- Baseline: lambda = 0, zero_source, rebuild. Exact zero
    // fluctuations by construction (K_lambda = exp(0*Y) = 1 everywhere). ---
    report.final_flow = build_attempt(real{0});
    StreamfunctionProblemView problem_view = make_problem_view();

    StreamfunctionSolverConfig baseline_config = base_config;
    baseline_config.eta = real{1};
    baseline_config.epsilon = eps_start;
    baseline_config.initial_state = PicardInitialState::zero_source;
    baseline_config.coefficient_state = CoefficientState::rebuild;
    ++mg_rebuild_count;

    StreamfunctionSolveReport baseline_report =
        stage_solver(context, problem_view, baseline_config, fields, workspace);

    {
        const ContinuationStageRecord base_record = make_stage_record(
            ContinuationAxis::eta, real{0}, real{0}, real{0}, baseline_report, baseline_config, n);
        report.stage_history.push_back(make_heterogeneity_stage_record(
            HeterogeneityAxis::lambda, base_record, real{0}, real{1}, eps_start, mg_rebuild_count,
            baseline_report));
    }

    if (baseline_report.status == StreamfunctionSolveStatus::invalid_problem) {
        report.status = HeterogeneityStatus::invalid_problem;
        report.failed_axis = HeterogeneityAxis::lambda;
        report.final_solve = baseline_report;
        report.total_mg_rebuilds = mg_rebuild_count;
        return report;
    }
    if (baseline_report.status != StreamfunctionSolveStatus::converged) {
        report.status = HeterogeneityStatus::baseline_failed;
        report.failed_axis = HeterogeneityAxis::lambda;
        report.final_solve = baseline_report;
        report.total_mg_rebuilds = mg_rebuild_count;
        return report;
    }

    snapshot_fields(context, fields, accepted_u1, accepted_u2, n);
    real accepted_lambda = real{0};
    real accepted_eta = real{1};
    real accepted_epsilon_log10 = continuation_config.inner.epsilon_log10.start;
    report.final_solve = baseline_report;

    // --- Lambda leg with eta rescue (decisions 1 and 2 of the SF-21
    // activation bitácora). ---
    ContinuationStepper lambda_stepper(continuation_config.lambda);
    while (!lambda_stepper.reached_target()) {
        const real attempt_lambda = lambda_stepper.attempt_param();
        const real step_attempted = lambda_stepper.current_step();
        const real param_start = lambda_stepper.current_param();

        // FIRST solve at this attempted lambda: rebuild q/hierarchy/RHS. The
        // resulting flow report is only recorded into report.final_flow if
        // this attempt (directly or via rescue) is ultimately ACCEPTED (see
        // below); a rejected attempt must not clobber the last accepted
        // lambda's flow report.
        const physics::AffinePeriodicFlowReport attempt_flow = build_attempt(attempt_lambda);
        problem_view = make_problem_view();

        StreamfunctionSolverConfig stage_config = base_config;
        stage_config.eta = real{1};
        stage_config.epsilon = eps_start;
        stage_config.initial_state = PicardInitialState::warm_start;
        stage_config.coefficient_state = CoefficientState::rebuild;
        ++mg_rebuild_count;

        StreamfunctionSolveReport stage_report =
            stage_solver(context, problem_view, stage_config, fields, workspace);

        {
            const ContinuationStageRecord record =
                make_stage_record(ContinuationAxis::eta, param_start, attempt_lambda,
                                  step_attempted, stage_report, stage_config, n);
            report.stage_history.push_back(make_heterogeneity_stage_record(
                HeterogeneityAxis::lambda, record, attempt_lambda, real{1}, eps_start,
                mg_rebuild_count, stage_report));
        }

        if (stage_report.status == StreamfunctionSolveStatus::invalid_problem) {
            restore_fields(context, fields, accepted_u1, accepted_u2, n);
            report.status = HeterogeneityStatus::invalid_problem;
            report.failed_axis = HeterogeneityAxis::lambda;
            report.final_lambda = accepted_lambda;
            report.final_eta = accepted_eta;
            report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
            report.total_mg_rebuilds = mg_rebuild_count;
            return report;
        }

        if (stage_report.status == StreamfunctionSolveStatus::converged) {
            snapshot_fields(context, fields, accepted_u1, accepted_u2, n);
            accepted_lambda = attempt_lambda;
            lambda_stepper.on_accept();
            report.final_solve = stage_report;
            report.final_flow = attempt_flow;
            continue;
        }

        // --- ETA RESCUE, exact ordering: restore accepted; solve attempted
        // lambda at eta=0 (warm-started, reuse); ramp eta 0->1. ---
        restore_fields(context, fields, accepted_u1, accepted_u2, n);

        StreamfunctionSolverConfig rescue_eta0_config = base_config;
        rescue_eta0_config.eta = real{0};
        rescue_eta0_config.epsilon = eps_start;
        rescue_eta0_config.initial_state = PicardInitialState::warm_start;
        rescue_eta0_config.coefficient_state = CoefficientState::reuse;

        StreamfunctionSolveReport rescue_eta0_report =
            stage_solver(context, problem_view, rescue_eta0_config, fields, workspace);

        {
            const ContinuationStageRecord record =
                make_stage_record(ContinuationAxis::eta, real{0}, real{0}, real{0},
                                  rescue_eta0_report, rescue_eta0_config, n);
            report.stage_history.push_back(make_heterogeneity_stage_record(
                HeterogeneityAxis::eta_rescue, record, attempt_lambda, real{0}, eps_start,
                mg_rebuild_count, rescue_eta0_report));
        }

        if (rescue_eta0_report.status == StreamfunctionSolveStatus::invalid_problem) {
            restore_fields(context, fields, accepted_u1, accepted_u2, n);
            report.status = HeterogeneityStatus::invalid_problem;
            report.failed_axis = HeterogeneityAxis::eta_rescue;
            report.final_lambda = accepted_lambda;
            report.final_eta = accepted_eta;
            report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
            report.total_mg_rebuilds = mg_rebuild_count;
            return report;
        }

        bool rescue_failed = rescue_eta0_report.status != StreamfunctionSolveStatus::converged;
        StreamfunctionSolveReport rescue_final_solve = rescue_eta0_report;

        if (!rescue_failed) {
            // eta=0 solve converged: this is the rescue leg's OWN accepted
            // state (separate from the outer lambda-accepted snapshot).
            snapshot_fields(context, fields, rescue_u1, rescue_u2, n);

            ContinuationStepper eta_rescue_stepper(continuation_config.inner.eta);
            while (!eta_rescue_stepper.reached_target()) {
                const real eta_param_start = eta_rescue_stepper.current_param();
                const real eta_attempt = eta_rescue_stepper.attempt_param();
                const real eta_step = eta_rescue_stepper.current_step();

                StreamfunctionSolverConfig rescue_stage_config = base_config;
                rescue_stage_config.eta = eta_attempt;
                rescue_stage_config.epsilon = eps_start;
                rescue_stage_config.initial_state = PicardInitialState::warm_start;
                rescue_stage_config.coefficient_state = CoefficientState::reuse;

                StreamfunctionSolveReport rescue_stage_report =
                    stage_solver(context, problem_view, rescue_stage_config, fields, workspace);

                {
                    const ContinuationStageRecord record = make_stage_record(
                        ContinuationAxis::eta, eta_param_start, eta_attempt, eta_step,
                        rescue_stage_report, rescue_stage_config, n);
                    report.stage_history.push_back(make_heterogeneity_stage_record(
                        HeterogeneityAxis::eta_rescue, record, attempt_lambda, eta_attempt,
                        eps_start, mg_rebuild_count, rescue_stage_report));
                }

                if (rescue_stage_report.status == StreamfunctionSolveStatus::invalid_problem) {
                    restore_fields(context, fields, accepted_u1, accepted_u2, n);
                    report.status = HeterogeneityStatus::invalid_problem;
                    report.failed_axis = HeterogeneityAxis::eta_rescue;
                    report.final_lambda = accepted_lambda;
                    report.final_eta = accepted_eta;
                    report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
                    report.total_mg_rebuilds = mg_rebuild_count;
                    return report;
                }

                if (rescue_stage_report.status == StreamfunctionSolveStatus::converged) {
                    snapshot_fields(context, fields, rescue_u1, rescue_u2, n);
                    eta_rescue_stepper.on_accept();
                    rescue_final_solve = rescue_stage_report;
                } else {
                    restore_fields(context, fields, rescue_u1, rescue_u2, n);
                    if (!eta_rescue_stepper.on_reject()) {
                        rescue_failed = true;
                        break;
                    }
                }
            }
        }

        if (!rescue_failed) {
            // Rescue SUCCESS (eta reached 1, converged): ACCEPT the lambda
            // attempt into the outer accepted-state snapshot.
            snapshot_fields(context, fields, accepted_u1, accepted_u2, n);
            accepted_lambda = attempt_lambda;
            accepted_eta = real{1};
            lambda_stepper.on_accept();
            report.final_solve = rescue_final_solve;
            report.final_flow = attempt_flow;
            continue;
        }

        // ANY rescue failure: restore the outer accepted state and let the
        // lambda stepper halve/retry the SAME interval, or exit on floor
        // exhaustion. The failed interval is never skipped.
        restore_fields(context, fields, accepted_u1, accepted_u2, n);
        if (!lambda_stepper.on_reject()) {
            report.status = HeterogeneityStatus::lambda_floor_exhausted;
            report.failed_axis = HeterogeneityAxis::lambda;
            report.final_lambda = accepted_lambda;
            report.final_eta = accepted_eta;
            report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
            report.total_mg_rebuilds = mg_rebuild_count;
            return report;
        }
    }

    // --- Epsilon leg: fixed lambda = 1, eta = 1; the conductivity/gauge are
    // fixed at the accepted lambda=1 state, so every stage reuses the
    // coefficients built for it. Shared run_epsilon_leg helper (see the
    // run_streamfunction_continuation refactor above). ---
    StreamfunctionSolverConfig epsilon_base_config = base_config;
    epsilon_base_config.coefficient_state = CoefficientState::reuse;

    const ContinuationStatus epsilon_status = run_epsilon_leg(
        context, problem_view, epsilon_base_config, continuation_config.inner.epsilon_log10,
        accepted_eta, stage_solver, fields, workspace, accepted_u1, accepted_u2, n,
        accepted_epsilon_log10, report.final_solve,
        [&](const ContinuationStageRecord& record, const StreamfunctionSolveReport& stage_report) {
            report.stage_history.push_back(make_heterogeneity_stage_record(
                HeterogeneityAxis::epsilon, record, accepted_lambda, accepted_eta,
                record.param_attempted, mg_rebuild_count, stage_report));
        });

    report.total_mg_rebuilds = mg_rebuild_count;

    if (epsilon_status == ContinuationStatus::invalid_problem) {
        report.status = HeterogeneityStatus::invalid_problem;
        report.failed_axis = HeterogeneityAxis::epsilon;
        report.final_lambda = accepted_lambda;
        report.final_eta = accepted_eta;
        report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
        return report;
    }
    if (epsilon_status == ContinuationStatus::step_floor_exhausted) {
        report.status = HeterogeneityStatus::epsilon_floor_exhausted;
        report.failed_axis = HeterogeneityAxis::epsilon;
        report.final_lambda = accepted_lambda;
        report.final_eta = accepted_eta;
        report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
        return report;
    }

    report.status = HeterogeneityStatus::reached_target;
    report.final_lambda = accepted_lambda;
    report.final_eta = accepted_eta;
    report.final_epsilon = epsilon_from_log10(accepted_epsilon_log10);
    return report;
}

} // namespace streamfunctions
} // namespace macroflow3d
