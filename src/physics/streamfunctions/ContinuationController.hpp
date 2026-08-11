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
 *
 * == SF-21: heterogeneity continuation ==
 *
 * `run_streamfunction_heterogeneity_continuation` adds an OUTER `lambda` leg
 * on top of the SAME reusable `ContinuationStepper` machine and the SAME
 * warm-started/rollback discipline described above, to reach the target
 * lognormal conductivity `K = exp(Y)` from the homogeneous state `K = 1`
 * (`lambda = 0`) without assuming strongly heterogeneous Picard convergence
 * from a homogeneous initial guess (the increment spec's Goal).
 *
 * Per attempted `lambda` value, the driver builds `Y_lambda = lambda * Y`
 * (one D2D copy of the caller's periodic `Y` plus `blas::scal`) and
 * `K_lambda = exp(Y_lambda)` (one small elementwise kernel), re-runs the
 * SF-19 `physics::solve_affine_periodic_flow` on `K_lambda` with prescribed
 * mean flux `(1, 0, 0)` to obtain the reference Darcy velocity for THIS
 * `lambda`, and solves the streamfunction problem with
 * `conductivity = Y_lambda` (`ConductivityRepresentation::log_conductivity_y`,
 * the existing `q = exp(-Y)` kernel -- no new q path) and
 * `gauge = AffineGauge::benchmark(1.0)` (`vbar = 1` exact by the prescribed
 * mean flux, not measured). `q` and the MG hierarchy are rebuilt
 * (`CoefficientState::rebuild`) exactly once per DISTINCT attempted `lambda`
 * value and REUSED (`CoefficientState::reuse`) across every subsequent
 * warm-started stage at that SAME `lambda` (the eta-rescue ramp below):
 * neither the conductivity nor the gauge changes across those stages, which
 * is exactly `CoefficientState::reuse`'s documented caller contract.
 *
 * Baseline: `lambda = 0` is solved `zero_source` at `(eta = 1,
 * epsilon = epsilon_start)`; `K_lambda = exp(0 * Y) = 1` everywhere
 * (homogeneous), so the fluctuations are exactly zero by construction. Not
 * converged -> `HeterogeneityStatus::baseline_failed`; `invalid_problem` ->
 * immediate structured failure (Gate-1/rollback rule: a degenerate measured
 * `v_rms` is state-independent, so no step reduction can fix it, and the
 * driver restores first wherever an accepted state already exists).
 *
 * LAMBDA ATTEMPT (every subsequent stepper attempt): one warm-started solve
 * at `(eta = 1, epsilon = epsilon_start)` -- `epsilon` is NEVER reduced
 * during the lambda leg (staged-regularization rule: the epsilon leg runs
 * ONLY after `lambda = 1` is accepted at `eta = 1`). Converged -> ACCEPT
 * (snapshot, `lambda_stepper.on_accept()`). Not converged -> ETA RESCUE, in
 * this EXACT order:
 *   1. restore the last ACCEPTED state bitwise (device-to-device, from the
 *      driver's outer accepted-state snapshot);
 *   2. solve the SAME attempted `lambda` at `eta = 0`, warm-started, reusing
 *      the coefficients already built for this `lambda` (eta does not affect
 *      `q`/the hierarchy/the affine RHS, only the nonlinear source coupling);
 *   3. if that `eta = 0` solve converges, snapshot it into a SEPARATE,
 *      driver-owned rescue-leg accepted-state pair and ramp `eta` from 0 to 1
 *      with a FRESH `ContinuationStepper(continuation_config.inner.eta)`
 *      (the SAME locked eta axis numbers as SF-17), every stage
 *      warm-started/reused, with the SAME accept/reject/halving/floor
 *      semantics and bitwise restore (against the RESCUE snapshot, not the
 *      outer one) on each rejected eta-ramp stage;
 *   4. rescue SUCCESS (`eta` reaches 1 converged) ACCEPTS the lambda attempt:
 *      the final rescue state is snapshotted into the OUTER accepted-state
 *      pair, `accepted_lambda <- attempt_lambda`, `accepted_eta <- 1`,
 *      `lambda_stepper.on_accept()`;
 *   5. rescue FAILURE (the `eta = 0` solve itself does not converge, OR the
 *      eta-ramp stepper rejects an attempt already at its step floor)
 *      restores the OUTER accepted state (bitwise) and calls
 *      `lambda_stepper.on_reject()`: halved retry of the SAME lambda interval
 *      (never skipped), or `HeterogeneityStatus::lambda_floor_exhausted` if
 *      the lambda step was already at its floor.
 * `invalid_problem` at ANY stage (lambda attempt, rescue eta=0, or any
 * eta-ramp stage) is an immediate structured failure, restoring the last
 * known-good state first (the outer accepted snapshot).
 *
 * After `lambda = 1` is accepted at `eta = 1`: the SF-17 epsilon leg runs at
 * that fixed `(lambda, eta)`, sharing the SAME `run_epsilon_leg` helper
 * `run_streamfunction_continuation` uses (refactored out of that driver
 * without changing its behavior -- see `ContinuationController.cu`), except
 * every stage uses `CoefficientState::reuse` (the conductivity/gauge are
 * fixed at `lambda = 1` throughout this leg, exactly the `reuse` contract).
 *
 * Every attempt (baseline, lambda, eta-rescue, epsilon) appends exactly one
 * `HeterogeneityStageRecord`, wrapping a `ContinuationStageRecord` (built by
 * the SAME `make_stage_record` helper) with the `lambda`/`eta`/`epsilon`
 * values in effect, the cumulative MG rebuild count, and the Gate-3A
 * physical metrics harvested from that stage's `StreamfunctionSolveReport.
 * diagnostics` (`e_v`, the two Darcy-invariance errors, the reconstructed
 * MAC-divergence error, the |c| 0.1% percentile, and the threshold-0
 * degeneracy total/unexplained counts when `config.diagnostics.
 * num_degeneracy_thresholds > 0`; see `PhysicalDiagnosticsReport` in
 * `Diagnostics.cuh` for every available field).
 *
 * Allocation policy: every driver-owned buffer (`Y_lambda`, `K_lambda`, the
 * SF-19 CompactMAC velocity triple, the outer accepted-state pair, and the
 * separate eta-rescue accepted-state pair) is allocated ONCE, up front,
 * sized to `grid.num_cells()`/the exact CompactMAC face counts; none of the
 * loops above allocate. Every device operation is stream-ordered on
 * `context.cuda_stream()`; this driver adds no additional
 * `context.synchronize()` beyond what `solve_streamfunctions` and
 * `physics::solve_affine_periodic_flow` already issue internally (both
 * already return fully host-synchronized reports).
 *
 * Overflow note: for `sigma_Y^2 <= 4`, `lambda * Y` stays within
 * approximately `+-8` over the full `lambda in [0, 1]` sweep, keeping
 * `K_lambda` within roughly `[3e-4, 3e3]` -- no overflow regime.
 * `sigma_Y^2 = 6.25` is PROHIBITED for this increment (see the increment
 * spec's failure/rollback policy); this driver does not enforce that bound
 * itself (it is a property of the caller-supplied `Y`, not of the driver),
 * so callers must not exceed it.
 *
 * `stage_solver` (the `StageSolveFn` injection seam) is used for EVERY
 * streamfunction stage solve here (baseline, lambda attempts, eta-rescue
 * stages, epsilon stages), letting tests inject deterministic failures at
 * any of those points. The SF-19 affine-periodic Darcy flow solve is always
 * the real `physics::solve_affine_periodic_flow` call -- this increment does
 * not add a flow-solve injection seam; a test that needs to exercise the
 * rescue/rollback machinery does so entirely through `stage_solver`.
 *
 * Gate 4 interpretation: this driver validates invariant CONSTRUCTION under
 * physical heterogeneity in the smooth, locally isotropic, triply periodic
 * regime. It makes no transport or transverse-macrodispersion claim.
 */

#include "../../core/BCSpec.hpp"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../flow/AffinePeriodicFlowSolver.cuh"
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

// --- SF-21: heterogeneity continuation. See the file header for the exact
// nested lambda/eta-rescue/epsilon protocol. ---

/**
 * Composed, host-only SF-21 configuration. `lambda` is the spec-locked outer
 * axis (initial step `0.1`, min `0.0125`, max `0.2`, halve on failure, grow
 * by `1.5` after 2 consecutive easy stages, `[0, 1]`); `inner.eta` is the
 * SAME SF-17 eta axis used for the rescue ramp; `inner.epsilon_log10` is the
 * SF-17 epsilon leg run once at `lambda = eta = 1`.
 */
struct HeterogeneityContinuationConfig {
    ContinuationAxisConfig lambda{real{0},  real{1},   real{0.1}, real{0.0125},
                                  real{0.2}, real{0.5}, real{1.5}, 2};
    StreamfunctionContinuationConfig inner{};
};

/**
 * Host validation: `config.lambda` via the same axis validator as
 * `StreamfunctionContinuationConfig` (distinct `"lambda"` message prefix),
 * then `config.inner` via `validate_streamfunction_continuation_config`.
 */
void validate_streamfunction_heterogeneity_continuation_config(
    const HeterogeneityContinuationConfig& config);

/** Which SF-21 leg a `HeterogeneityStageRecord`/failure belongs to. */
enum class HeterogeneityAxis { lambda, eta_rescue, epsilon };

/**
 * One recorded SF-21 stage attempt (baseline, lambda, eta-rescue, or
 * epsilon), append-only. `base` is built by the SAME `make_stage_record`
 * helper `ContinuationController.cu` uses for SF-17 (its `axis` field is
 * `eta` for baseline/lambda/eta-rescue records and `epsilon` for epsilon-leg
 * records -- lambda has no `ContinuationAxis` value of its own; use this
 * record's own `axis` field, not `base.axis`, to distinguish the lambda leg).
 * `lambda_value`/`eta_value`/`epsilon_value` are the PHYSICAL parameters in
 * effect for this stage; `mg_rebuild_count` is the CUMULATIVE count of
 * `CoefficientState::rebuild` calls up to and including this stage. The
 * trailing fields are the Gate-3A physical metrics harvested from this
 * stage's `StreamfunctionSolveReport.diagnostics` (see
 * `PhysicalDiagnosticsReport` in `Diagnostics.cuh`); `degeneracy_total0`/
 * `degeneracy_unexplained0` are threshold-0 counts, left at 0 when
 * `config.diagnostics.num_degeneracy_thresholds == 0` (guards inactive).
 */
struct HeterogeneityStageRecord {
    ContinuationStageRecord base{};
    HeterogeneityAxis axis{HeterogeneityAxis::lambda};

    real lambda_value{};
    real eta_value{};
    real epsilon_value{};
    int mg_rebuild_count{0};

    real e_v{};
    real invariance_e_psi1{};
    real invariance_e_psi2{};
    real e_div{};
    real c_percentile_p001{};
    unsigned long long degeneracy_total0{0};
    unsigned long long degeneracy_unexplained0{0};
};

/**
 * Overall outcome of one `run_streamfunction_heterogeneity_continuation`
 * call.
 *   - `reached_target`: the epsilon leg reached `config.inner.epsilon_log10.
 *     target` at `lambda = eta = 1` (equivalently: every leg reached its
 *     target exactly).
 *   - `baseline_failed`: the `lambda = 0` baseline stage did not converge; no
 *     accepted state ever existed.
 *   - `lambda_floor_exhausted`: the lambda stepper rejected an attempt (after
 *     its eta-rescue failed) already at the lambda step floor.
 *   - `epsilon_floor_exhausted`: the epsilon leg's stepper rejected an
 *     attempt already at its step floor.
 *   - `invalid_problem`: some stage returned `StreamfunctionSolveStatus::
 *     invalid_problem`; `failed_axis` names which leg.
 */
enum class HeterogeneityStatus {
    reached_target,
    baseline_failed,
    lambda_floor_exhausted,
    epsilon_floor_exhausted,
    invalid_problem,
};

/**
 * Result of one `run_streamfunction_heterogeneity_continuation` call.
 * `final_lambda`/`final_eta`/`final_epsilon` are the PHYSICAL parameters of
 * the last ACCEPTED state (all zero-initialized defaults if `status ==
 * baseline_failed`). `final_flow` is the SF-19 `physics::
 * AffinePeriodicFlowReport` of the ACCEPTED lambda's Darcy flow solve.
 * `total_mg_rebuilds` is the total count of `CoefficientState::rebuild`
 * calls across the whole run. `snapshot_bytes` is the exact owned capacity of
 * the driver's two accepted-state snapshot pairs (the outer lambda-accepted
 * pair and the eta-rescue-leg pair), in bytes; `driver_owned_bytes` also adds
 * `Y_lambda`/`K_lambda` and the SF-19 CompactMAC velocity triple (excludes
 * the internally owned `physics::AffinePeriodicFlowWorkspace`, whose own
 * capacities are reported separately by its own `memory_report()`).
 */
struct HeterogeneityContinuationReport {
    HeterogeneityStatus status{HeterogeneityStatus::baseline_failed};
    HeterogeneityAxis failed_axis{HeterogeneityAxis::lambda};

    real final_lambda{};
    real final_eta{};
    real final_epsilon{};

    std::vector<HeterogeneityStageRecord> stage_history{};
    StreamfunctionSolveReport final_solve{};
    physics::AffinePeriodicFlowReport final_flow{};

    int total_mg_rebuilds{0};
    std::size_t snapshot_bytes{0};
    std::size_t driver_owned_bytes{0};
};

/**
 * Run the SF-21 lambda/eta-rescue/epsilon heterogeneity continuation
 * described in the file header. `Y` is the caller-owned, triply periodic
 * SF-18 log-conductivity field at `lambda = 1` scale (`grid.num_cells()`
 * elements); `flow_config` parameterizes every per-lambda SF-19
 * `physics::solve_affine_periodic_flow` call (default `qbar = (1, 0, 0)`).
 * `base_config` supplies every non-`eta`/`epsilon`/`initial_state`/
 * `coefficient_state` field (`mg`, `linear`, `histogram`, `diagnostics`,
 * `picard`, `adaptive`) for every stage; those four fields are overridden per
 * stage by this driver exactly as documented above. Throws
 * `std::invalid_argument` if `continuation_config` fails
 * `validate_streamfunction_heterogeneity_continuation_config`, if `Y.size()
 * != grid.num_cells()`, or if `stage_solver`/`physics::
 * solve_affine_periodic_flow`/the real `solve_streamfunctions` throw it for
 * host misuse -- none of these are caught here.
 */
[[nodiscard]] HeterogeneityContinuationReport run_streamfunction_heterogeneity_continuation(
    CudaContext& context, const Grid3D& grid, DeviceSpan<const real> Y,
    const HeterogeneityContinuationConfig& continuation_config,
    const physics::AffinePeriodicFlowConfig& flow_config, StreamfunctionSolverConfig base_config,
    StreamfunctionFields& fields, StreamfunctionWorkspace& workspace,
    StageSolveFn stage_solver = {});

} // namespace streamfunctions
} // namespace macroflow3d
