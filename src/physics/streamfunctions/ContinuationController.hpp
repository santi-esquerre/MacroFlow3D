#pragma once

/**
 * @file ContinuationController.hpp
 * @brief Adaptive eta/epsilon homotopy continuation with rollback (SF-17).
 *
 * SF-13..16 solve the Lester equation (14) coupled nonlinear system at ONE
 * fixed `(eta, epsilon)` pair. SF-17 adds a host-orchestrated homotopy on top
 * of that unchanged solver: advance `eta` from 0 to 1 (the full nonlinear
 * source coupling), then reduce the nonlinear-source denominator
 * regularization `epsilon` by decades toward the unregularized problem,
 * warm-starting every stage from the last ACCEPTED state
 * (`PicardInitialState::warm_start`, `StreamfunctionSolver.cuh`) and rolling
 * back to that accepted state whenever a stage fails to converge.
 *
 * A fixed-epsilon accepted result is NEVER the original (unregularized)
 * system's solution; it is labeled by the `epsilon` it was accepted at. Do
 * not present it otherwise. Failed parameter intervals are never skipped:
 * every rejected attempt either retries at a halved step from the SAME
 * accepted parameter, or exits with a structured failure -- it never jumps
 * past the interval that failed.
 *
 * == Reusable stage stepper (host-only, GPU-free, unit-testable) ==
 *
 * One machine, `ContinuationStepper`, drives both axes:
 *   - `eta` runs it in linear physical space, `[eta.start, eta.target]`.
 *   - `epsilon` runs it in `p = -log10(epsilon)` space,
 *     `[epsilon_log10.start, epsilon_log10.target]`; the physical value is
 *     `epsilon = pow(10, -p)` (`epsilon_from_log10`). With the default
 *     `initial_step = 1.0` (one decade) and no failures, the accepted `p`
 *     sequence is exactly the integers `epsilon_log10.start .. target`, i.e.
 *     physical epsilon decays exactly by decades (`1e-2, 1e-3, ..., 1e-6`)
 *     on the no-failure path -- the property the increment spec requires
 *     ("reduce epsilon by decades").
 *
 * Stepper state: {current param, persistent step, easy-streak count,
 * halvings-in-the-current-attempt}.
 *   - `attempt = min(current_param + step, target)` -- an EXACT clamp, so
 *     the final accepted value equals `target` exactly in double precision.
 *   - ACCEPT: `param <- attempt`; the persistent step is left exactly as the
 *     step that succeeded (it is only changed by growth below, never
 *     silently reset). If the accepted attempt had ZERO halvings ("easy"),
 *     the easy-streak counter increments; once it reaches
 *     `easy_streak` consecutive easy stages, the persistent step grows by
 *     `growth_factor`, capped at `max_step`, and the streak resets to 0. Any
 *     halving (even on an eventually-accepted attempt) resets the streak to
 *     0 immediately, without growing the step.
 *   - REJECT: if the step already attempted was AT the floor
 *     (`step <= min_step`), the stepper reports floor exhaustion after
 *     exactly that one floor attempt -- mirroring the SF-15
 *     `omega_floor_rejected` rule: never silently accepted, never retried
 *     beyond the single floor trial. Otherwise the step halves
 *     (`step *= backtrack_factor`, clamped to `min_step`) and the SAME
 *     parameter interval is retried; a rejected interval is never skipped.
 *
 * == Continuation driver ==
 *
 * `run_streamfunction_continuation` orchestrates:
 *   1. Baseline stage at `(eta.start, epsilon_from_log10(epsilon_log10.start))`
 *      with `PicardInitialState::zero_source` (recorded with
 *      `step_attempted = 0`, `axis = eta`). Converged -> snapshot `fields`
 *      into the driver-OWNED accepted-state buffers (two device buffers of
 *      `n` cells, allocated ONCE up front -- deliberately NOT added to
 *      `StreamfunctionWorkspace`, so the SF-12 closed-form memory report and
 *      its pinned test remain untouched). Not converged ->
 *      `ContinuationStatus::baseline_failed` (no accepted state exists to
 *      restore into `fields`). `invalid_problem` ->
 *      `ContinuationStatus::invalid_problem` immediately (a degenerate
 *      measured Darcy `v_rms` is state-independent; no step reduction can
 *      fix it).
 *   2. Eta leg: `ContinuationStepper` over `eta` at the FIXED starting
 *      epsilon, every stage warm-started (`PicardInitialState::warm_start`)
 *      from `fields` (which always holds the last accepted state).
 *   3. Epsilon leg: only after the eta leg reaches `eta.target` exactly, the
 *      SAME stepper machine over `epsilon_log10` at the FIXED final eta.
 *
 * Every attempt (baseline included) appends exactly one
 * `ContinuationStageRecord` to `stage_history`, accepted or rejected --
 * append-only, never rewritten.
 *
 * Rollback contract: `fields` is NEVER overwritten by a stage's in-progress
 * Picard iteration until that stage's FINAL result is known.
 * `solve_streamfunctions(..., PicardInitialState::warm_start)` does mutate
 * `fields` in place while iterating, so on a `not_converged` result the
 * driver restores `fields` bitwise (device-to-device copy) from the
 * accepted-state snapshot BEFORE deciding the stepper's retry/exit outcome.
 * On EVERY exit path (`reached_target`, `baseline_failed`,
 * `step_floor_exhausted`, `invalid_problem`), the caller's `fields` hold
 * exactly the last ACCEPTED state (or are untouched, for `baseline_failed`,
 * since no accepted state ever existed).
 *
 * No allocation occurs inside the stage loop: the two snapshot buffers are
 * allocated once, up front, sized to `problem.grid.num_cells()`. Device
 * copies are stream-ordered `cudaMemcpyAsync` (device-to-device) on
 * `context.cuda_stream()`; `solve_streamfunctions`'s own residual/diagnostics
 * synchronization already provides every host-visible sync point this driver
 * needs (it only reads already-synchronized host report fields), so no
 * additional `context.synchronize()` is issued here.
 *
 * `StageSolveFn` is a deterministic test-injection seam (production default,
 * an empty `std::function`, dispatches to the real `solve_streamfunctions`):
 * it lets tests inject deterministic stage failures -- including mutating
 * `fields` before reporting failure -- to verify step halving and bitwise
 * state restore. It is never a fallback/compatibility path.
 */

#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"
#include "ResidualEvaluator.cuh"
#include "StreamfunctionSolver.cuh"
#include "StreamfunctionTypes.hpp"
#include "StreamfunctionWorkspace.cuh"

#include <cstddef>
#include <functional>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {

/**
 * One continuation axis's stepper configuration (SF-17). See the file
 * header for the exact stepper semantics. `start`/`target` are in the
 * axis's own space: physical `eta` for the eta axis, `p = -log10(epsilon)`
 * for the epsilon axis (`epsilon_from_log10` converts `p` to physical
 * epsilon). Validated by `validate_streamfunction_continuation_config`.
 */
struct ContinuationAxisConfig {
    real start{};
    real target{};
    real initial_step{};
    real min_step{};
    real max_step{};
    real backtrack_factor{};
    real growth_factor{};
    int easy_streak{};
};

/**
 * Composed, host-only SF-17 continuation configuration. Defaults are the
 * dashboard-locked values from the SF-17 activation bitácora:
 *   - `eta`: linear space, `[0, 1]`, initial step `0.1`, min `0.0125`, max
 *     `0.25`, halve on failure (`backtrack_factor = 0.5`), grow by `1.5`
 *     after 2 consecutive easy stages.
 *   - `epsilon_log10`: `p = -log10(epsilon)` space, `[2, 6]` (physical
 *     `1e-2 .. 1e-6`; `8`, i.e. `1e-8`, is configurable via `target` but not
 *     the default), initial step `1.0` (one decade -- see the file header's
 *     "decades on the no-failure path" note), min `0.125`, max `1.0`, same
 *     halve/grow rule as eta. The epsilon min/max step values are a project
 *     choice recorded in the activation bitácora; the spec fixes only "by
 *     decades" plus the shared adaptive-step/rollback requirement.
 */
struct StreamfunctionContinuationConfig {
    ContinuationAxisConfig eta{real{0},   real{1},     real{0.1},  real{0.0125},
                               real{0.25}, real{0.5},   real{1.5},  2};
    ContinuationAxisConfig epsilon_log10{real{2}, real{6},   real{1.0}, real{0.125},
                                          real{1.0}, real{0.5}, real{1.5}, 2};
};

/**
 * Host validation of `config`, throwing `std::invalid_argument` with a
 * distinct message per violated precondition, for EACH axis
 * (`config.eta`, then `config.epsilon_log10`): every field finite;
 * `start <= target`; `initial_step > 0`;
 * `0 < min_step <= initial_step <= max_step`; `backtrack_factor` in
 * `(0, 1)`; `growth_factor >= 1`; `easy_streak >= 1`.
 */
void validate_streamfunction_continuation_config(const StreamfunctionContinuationConfig& config);

/**
 * Host-only, GPU-free stage stepper for one continuation axis. See the file
 * header for the exact attempt/accept/reject semantics. Not thread-safe;
 * one instance drives one axis's leg of one `run_streamfunction_continuation`
 * call.
 */
class ContinuationStepper {
  public:
    explicit ContinuationStepper(const ContinuationAxisConfig& config);

    // The value the NEXT attempt would use: min(current_param() + current_step(), target).
    [[nodiscard]] real attempt_param() const noexcept;

    // The parameter value of the last ACCEPTED attempt (or the axis's
    // `start`, before any attempt has been accepted).
    [[nodiscard]] real current_param() const noexcept { return param_; }

    // The persistent step (see the file header: unchanged by rejection
    // retries beyond the halving itself; only accept-time growth changes
    // it further).
    [[nodiscard]] real current_step() const noexcept { return step_; }

    [[nodiscard]] bool reached_target() const noexcept { return param_ >= config_.target; }

    // Record ACCEPTANCE of the attempt returned by the most recent
    // attempt_param() call: advances current_param() to that attempt and
    // applies the easy-streak / growth rule.
    void on_accept();

    // Record REJECTION of the attempt returned by the most recent
    // attempt_param() call. Returns true if the caller should retry (the
    // step was halved, current_param() is unchanged); returns false if the
    // attempted step was already at the floor (structured floor-exhaustion
    // failure -- exactly one floor attempt, current_param() is unchanged).
    [[nodiscard]] bool on_reject();

  private:
    ContinuationAxisConfig config_;
    real param_;
    real step_;
    int easy_streak_count_ = 0;
    int halvings_in_attempt_ = 0;
};

// Physical epsilon for a p = -log10(epsilon) axis value (the epsilon_log10
// continuation axis's space).
[[nodiscard]] real epsilon_from_log10(real p) noexcept;

/** Which axis a `ContinuationStageRecord`/failure belongs to. */
enum class ContinuationAxis { eta, epsilon };

/**
 * Why one continuation stage attempt was rejected (`none` for an accepted
 * attempt). `solver_not_converged` covers every `StreamfunctionSolveStatus::
 * not_converged` exit reason (budget exhaustion, linear-block failure,
 * stagnation, omega-floor rejection); `solver_invalid_problem` is the
 * `StreamfunctionSolveStatus::invalid_problem` measured-`v_rms` failure.
 */
enum class ContinuationStageFailure { none, solver_not_converged, solver_invalid_problem };

/**
 * One recorded continuation stage attempt (baseline or leg), append-only:
 * every attempt (accepted or rejected) gets exactly one record, in
 * evaluation order. `param_start`/`param_attempted` are PHYSICAL values
 * (physical `eta`, or physical `epsilon = pow(10, -p)` on the epsilon leg);
 * `step_attempted` is in the axis's OWN space (linear eta, or
 * `-log10(epsilon)` decades) -- `0` for the baseline record. See the file
 * header for `unexplained_fraction`/`c_percentile_p001` semantics.
 */
struct ContinuationStageRecord {
    ContinuationAxis axis{ContinuationAxis::eta};
    real param_start{};
    real param_attempted{};
    real step_attempted{};
    bool accepted{false};
    ContinuationStageFailure failure{ContinuationStageFailure::none};

    PicardExitReason exit_reason{PicardExitReason::none};
    int picard_iterations{0};
    real final_omega{};
    real r_F{};
    real r1{};
    real r2{};

    // config.diagnostics.num_degeneracy_thresholds > 0 (guards active):
    // diagnostics.degeneracy_unexplained[0] / n. Otherwise 0.
    real unexplained_fraction{};

    // residual_histogram_percentile(report.residual, 0.001).
    real c_percentile_p001{};

    int psi1_iterations{0};
    int psi2_iterations{0};
};

/**
 * Overall outcome of one `run_streamfunction_continuation` call.
 *   - `reached_target`: the epsilon leg's stepper reached
 *     `config.epsilon_log10.target` (equivalently: both legs reached their
 *     targets exactly).
 *   - `baseline_failed`: the baseline stage did not converge; no accepted
 *     state ever existed.
 *   - `step_floor_exhausted`: some leg's stepper rejected an attempt already
 *     at its step floor; `failed_axis` names which leg.
 *   - `invalid_problem`: a stage (baseline or leg) returned
 *     `StreamfunctionSolveStatus::invalid_problem`; `failed_axis` names
 *     which leg (or `eta` for the baseline, matching the baseline record's
 *     axis).
 */
enum class ContinuationStatus { reached_target, baseline_failed, step_floor_exhausted,
                                 invalid_problem };

/**
 * Result of one `run_streamfunction_continuation` call. `final_eta`/
 * `final_epsilon` are the PHYSICAL parameters of the last ACCEPTED state
 * (unchanged from their zero-initialized default if `status ==
 * baseline_failed`, since no state was ever accepted). `final_solve` is the
 * full `StreamfunctionSolveReport` of the last accepted stage (the baseline,
 * if no leg stage was ever accepted). `snapshot_bytes` is the exact owned
 * capacity, in bytes, of the driver's two accepted-state device buffers
 * (`2 * n * sizeof(real)` after allocation; independent of whether any stage
 * accepted).
 */
struct StreamfunctionContinuationReport {
    ContinuationStatus status{ContinuationStatus::baseline_failed};
    ContinuationAxis failed_axis{ContinuationAxis::eta};

    real final_eta{};
    real final_epsilon{};

    std::vector<ContinuationStageRecord> stage_history{};
    StreamfunctionSolveReport final_solve{};

    std::size_t snapshot_bytes{0};
};

/**
 * Deterministic test-injection seam for one continuation stage solve. The
 * production default (an empty `std::function`, see
 * `run_streamfunction_continuation`) dispatches to the real
 * `solve_streamfunctions`; a test may substitute a functor that mutates
 * `fields` and returns an arbitrary `StreamfunctionSolveReport` to exercise
 * rollback and step-halving deterministically. Never a runtime fallback
 * path.
 */
using StageSolveFn = std::function<StreamfunctionSolveReport(
    CudaContext&, const StreamfunctionProblemView&, const StreamfunctionSolverConfig&,
    StreamfunctionFields&, StreamfunctionWorkspace&)>;

/**
 * Run the SF-17 eta/epsilon homotopy continuation described in the file
 * header. Throws `std::invalid_argument` if `continuation_config` fails
 * `validate_streamfunction_continuation_config`, or if `stage_solver` (or
 * the real `solve_streamfunctions`, for the default) throws it for host
 * misuse -- neither is caught here.
 */
[[nodiscard]] StreamfunctionContinuationReport run_streamfunction_continuation(
    CudaContext& context, const StreamfunctionProblemView& problem,
    const StreamfunctionSolverConfig& base_config,
    const StreamfunctionContinuationConfig& continuation_config, StreamfunctionFields& fields,
    StreamfunctionWorkspace& workspace, StageSolveFn stage_solver = {});

} // namespace streamfunctions
} // namespace macroflow3d
