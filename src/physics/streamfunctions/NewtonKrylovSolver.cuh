#pragma once

/**
 * @file NewtonKrylovSolver.cuh
 * @brief SF-24 globalized Newton-Krylov phase for the coupled Lester
 *        equation (14) streamfunction residual, and its reproducible Picard
 *        rescue/retry fallback protocol.
 *
 * This file implements ONE Newton phase ACTIVATION: given the current
 * accepted coupled state, it drives Newton iterations (inexact-Newton GMRES
 * step + Armijo line search) until the phase converges, exhausts its
 * per-activation iteration budget, or suffers a Newton STEP FAILURE. It does
 * NOT own the rescue-Picard-then-retry state machine (E6): that lives in
 * `solve_streamfunctions` (`StreamfunctionSolver.cu`) because a rescue
 * advance reuses the UNCHANGED SF-15/SF-20 outer Picard/Anderson loop body,
 * which this file cannot see without creating a header cycle back into
 * `StreamfunctionSolver.cuh` (see the "module boundary" note below). The
 * caller therefore drives the OUTER activation/rescue/retry loop; this file
 * drives the INNER per-activation Newton iteration loop.
 *
 * == The solved system ==
 *
 * The accepted coupled residual (`ResidualEvaluator.cuh`) is
 *
 *   F1(u1, u2) = A u1 - P( rhs_affine1 - eta * q .* S2(u1, u2) )
 *   F2(u1, u2) = A u2 - P( rhs_affine2 - eta * q .* S1(u1, u2) )
 *
 * treated as one coupled map `F(Psi) = (F1, F2)` of `Psi = (u1, u2)`. One
 * Newton iteration solves the coupled Newton linearization `J(Psi) delta =
 * -F(Psi)` INEXACTLY, via the SF-23 restarted, right-preconditioned
 * `CoupledGmres` against the SF-22 matrix-free `JvpWorkspace` action `J(Psi)
 * p` and the SF-23 `BlockDiagonalMGPreconditioner`, then globalizes the
 * accepted step with an Armijo line search on the merit `phi = 0.5*r_F^2`. No
 * new PDE discretization, no explicit Jacobian matrix, and no new fixed
 * point are introduced anywhere in this file; every linear-algebra/Krylov
 * primitive is reused verbatim from SF-22/SF-23.
 *
 * == Activation (E2, driven by the caller; documented here for the full
 *    picture) ==
 *
 * `solve_streamfunctions` calls `run_newton_phase` once per Newton phase
 * ENTRY: either the accepted-state coupled residual `r_F_k` at an outer
 * Picard-loop HEAD satisfies `r_F_k <= config.newton.activation_r_F`, or the
 * SF-15 stagnation exit condition would otherwise fire AND `r_F_k <=
 * config.newton.stagnation_activation_r_F` AND the accepted state's SF-11
 * degeneracy is clean (`f_prev <= config.adaptive.max_unexplained_fraction`
 * when guards are active, else vacuously clean). On every Newton-phase
 * entry, if `config.anderson.enabled`, the caller clears
 * `workspace.anderson()`'s history first (the SF-21 "one fixed-point-map
 * instance" premise: a Newton acceptance changes the accepted state outside
 * Anderson's own staging, exactly like a continuation-stage boundary).
 *
 * == Per-Newton-iteration HEAD (E3 step 1) ==
 *
 * At the current ACCEPTED state `Psi = (fields.u1_span(), fields.u2_span())`
 * (always mean-zero projected by construction of every acceptance path):
 * evaluate `F(Psi)` via `enqueue_streamfunction_residual` (the IDENTICAL
 * call shape the Picard head uses), writing into `workspace.f1()`/`f2()`.
 * `residual_out` (the caller's `report.residual`, passed by reference so
 * this file never needs to know about `StreamfunctionSolveReport` -- see the
 * module-boundary note) is updated at EVERY Newton head, converged or not.
 * `r_F_k <= config.picard.tolerance` converges the WHOLE solve (the same
 * tolerance the Picard loop uses: Newton reaches the SAME fixed point, not a
 * different one). Reaching `config.newton.max_newton_iterations` HEADs
 * without converging is a structured, TERMINAL budget exhaustion (the caller
 * maps it to `PicardExitReason::newton_budget_exhausted`; it is never
 * retried).
 *
 * == Extra base-evaluation cost note (documented, not accidental) ==
 *
 * `jvp.prepare_jvp_base(...)` (step 2) internally re-evaluates `F(Psi)` via
 * its OWN `enqueue_streamfunction_residual` call (SF-22 D4), i.e. `F(Psi)` is
 * evaluated TWICE per Newton iteration: once by this file's own HEAD (to
 * drive convergence/budget checks and populate `b = -F`), once again inside
 * `prepare_jvp_base` (to seed the SF-22 cached base for every subsequent
 * `apply()` call). This is a DELIBERATE structural-simplicity choice (one
 * uniform "evaluate F at the accepted state" head, reusing `prepare_jvp_base`
 * exactly as SF-22/SF-23 accept it, rather than threading a manually-shared
 * base residual through a modified `JvpWorkspace` API) -- the SAME cost
 * tradeoff the SF-15 Picard head documents for its own redundant per-
 * iteration residual re-evaluation. Eliminating it is out of SF-24 scope.
 *
 * == Forcing (E4) ==
 *
 *   rel_tol_k = clamp(config.newton.forcing_coefficient * sqrt(r_F_k),
 *                     config.newton.forcing_min, config.newton.forcing_max)
 *
 * A LOCAL copy of `config.newton.gmres` has its `rel_tol` OVERWRITTEN by
 * `rel_tol_k` every Newton iteration (`restart`/`max_iterations` are used
 * verbatim from `config.newton.gmres`); `config.newton.gmres.rel_tol` itself
 * is therefore never read by this file.
 *
 * == Linear-accept rule (E5) ==
 *
 * `CoupledGmresStatus::converged` proceeds to the line search. `max_iterations`
 * or `breakdown` proceeds to the line search ONLY if the final true-residual
 * checkpoint shows STRICT reduction below the FIRST checkpoint's true
 * residual (`checkpoints.back().true_residual < checkpoints.front().true_residual`,
 * both unpreconditioned true norms -- see `CoupledGmres.cuh`); otherwise this
 * is a Newton STEP FAILURE (`gmres_no_reduction`). `nonfinite` is
 * unconditionally a Newton STEP FAILURE (`gmres_nonfinite`).
 *
 * == Line search / merit / Armijo test (E7) ==
 *
 * For `alpha = 1`, then `alpha *= config.newton.backtrack_factor` while
 * `alpha >= config.newton.alpha_min`: form the trial `u_trial_i = u_i +
 * alpha*delta_i` in `workspace.u_trial1()`/`u_trial2()` (memcpy from the
 * accepted state, then one `blas::axpy`), mean-zero project both, evaluate
 * the trial residual via `enqueue_streamfunction_residual` into
 * `workspace.rhs1()`/`rhs2()` (idle at this point, exactly the SF-15 Picard-
 * trial convention), and classify through the IDENTICAL guard chain and
 * order SF-15 backtracking trials use (`StreamfunctionSolver.cuh` step 4):
 * `rejected_nonfinite` -> `rejected_degeneracy` (guards active only) ->
 * `rejected_percentile` (guards active only) -> Armijo. `f_prev`/`p_k` (the
 * accepted-state degeneracy fraction and |c| 0.1% percentile the guard chain
 * compares every trial against) are RE-CAPTURED once at THIS Newton HEAD
 * (the same mechanism the Picard head uses, not carried over from a
 * different state), where `p_k = residual_histogram_percentile(residual_k,
 * 0.001)`. The Armijo test itself is on the merit `phi = 0.5*r_F^2` rather
 * than `r_F` directly:
 *
 *   ACCEPT iff 0.5*r_F_trial^2 <= (1 - config.newton.armijo_c*alpha) * 0.5*r_F_k^2
 *
 * Every trial, accepted or rejected, is recorded (`NewtonTrialRecord`). On
 * ACCEPTANCE, `u_trial1()`/`u_trial2()` are copied into
 * `fields.u1_span()`/`u2_span()` -- the ONLY place this file mutates the
 * accepted state -- and the Newton iteration loop continues to its next
 * HEAD; `omega`/easy-streak Picard state is untouched (Newton does not use
 * or update it). If every attempted `alpha` down to (and including) the
 * floor is rejected, the line search exhausts without acceptance: a Newton
 * STEP FAILURE (`line_search_exhausted`).
 *
 * == Newton STEP FAILURE / rescue / retry state machine (E6, split across
 *    this file and the caller) ==
 *
 * A Newton STEP FAILURE (any of `gmres_nonfinite`, `gmres_no_reduction`,
 * `line_search_exhausted`) leaves the accepted state INTACT by construction:
 * neither the GMRES solve nor any line-search trial ever touches
 * `fields.u1_span()`/`u2_span()` except on an ACCEPTED trial. `run_newton_phase`
 * returns `NewtonPhaseStatus::step_failed` immediately (after recording the
 * failing `NewtonIterationRecord`) and performs NO rescue/retry logic itself.
 *
 * The CALLER (`solve_streamfunctions`) then implements the rest of the E6
 * state machine: if this Newton-phase ACTIVATION has not yet used its one
 * rescue-retry, it runs `config.newton.rescue_picard_steps` ACCEPTED advances
 * of the UNCHANGED SF-15/SF-20 Picard/Anderson outer-loop body (reusing that
 * exact code path -- not a forked variant), then calls `run_newton_phase`
 * again ONCE (a "retry", counted by the caller in `report.newton_rescue_events`).
 * A SECOND `step_failed` return for the SAME activation is TERMINAL: the
 * caller maps it to `PicardExitReason::newton_exhausted`, `status =
 * not_converged`, and stops with the last accepted state (unaffected by
 * either Newton phase call). `config.newton.rescue_picard_steps == 0` is a
 * valid, deterministic degenerate case (no intervening Picard advances,
 * immediate retry) used to test bitwise-reproducible failure-protocol state
 * preservation.
 *
 * == Module boundary (why report fields are passed individually) ==
 *
 * `StreamfunctionSolveReport` (the top-level solve report) is declared in
 * `StreamfunctionSolver.cuh`, which itself must include THIS header to get
 * `NewtonIterationRecord` for its own additive `newton_history` field. To
 * avoid a header cycle, this file never includes `StreamfunctionSolver.cuh`
 * and `run_newton_phase` never takes a `StreamfunctionSolveReport&`; instead
 * the caller passes the exact report sub-objects this file needs to update
 * by reference (`residual_out`, `newton_history`, and the accept/failure
 * counters). `PicardTrialOutcome` (reused here for `NewtonTrialRecord::outcome`)
 * lives in the lower, uncontested `StreamfunctionTypes.hpp` for the same
 * reason.
 *
 * == Disabled-path bitwise-preservation claim ==
 *
 * `run_newton_phase` is called ONLY from inside `if (config.newton.enabled)`
 * blocks in `solve_streamfunctions`, and only inside the `config.adaptive.enabled`
 * loop (SF-24 C01: `require_valid_newton_config` in `StreamfunctionWorkspace.cu`
 * rejects `config.newton.enabled && !config.adaptive.enabled` before this file
 * is ever reached). When `config.newton.enabled == false`,
 * `StreamfunctionWorkspace` never constructs its optional Newton sub-
 * workspace (see `StreamfunctionWorkspace.cuh`) and this file's code is never
 * reached, so the disabled adaptive-Picard/Anderson loop executes IDENTICAL
 * statements to the pre-SF-24 SF-15/SF-20 loop (bitwise-preserving, matching
 * every prior additive increment's own disabled-path claim).
 *
 * == Determinism (E11) ==
 *
 * Every primitive this file drives (`enqueue_streamfunction_residual`,
 * `JvpWorkspace`, `CoupledGmres`, `BlockDiagonalMGPreconditioner`,
 * `MeanZeroProjector`, `blas`) is an already-accepted deterministic
 * primitive; this file introduces no new device reduction, no clock, and no
 * data-dependent iteration-order randomness.
 */

#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"
#include "CoupledGmres.cuh"
#include "JacobianVectorProduct.cuh"
#include "ResidualEvaluator.cuh"
#include "StreamfunctionTypes.hpp"
#include "StreamfunctionWorkspace.cuh"

#include <vector>

namespace macroflow3d {
namespace streamfunctions {

/** Structured reason a Newton STEP FAILED (E6/E8); `none` for an accepted step. */
enum class NewtonStepFailureReason {
    none,
    gmres_nonfinite,
    gmres_no_reduction,
    line_search_exhausted,
};

/**
 * One recorded Newton line-search trial (E7/E8): the step length attempted,
 * the trial's coupled residual, and its classification through the SAME
 * guard chain (`PicardTrialOutcome`, `StreamfunctionTypes.hpp`) an SF-15
 * Picard backtracking trial uses.
 */
struct NewtonTrialRecord {
    real alpha{};
    real r_F_trial{};
    PicardTrialOutcome outcome{PicardTrialOutcome::accepted};
};

/**
 * One recorded Newton iteration (E3/E8): the outer Picard iteration `k` at
 * which the enclosing Newton phase activation entered (constant across every
 * record produced by the SAME `run_newton_phase` call), the HEAD residual
 * and forcing tolerance, the GMRES summary, every line-search trial, and the
 * outcome (`accepted_alpha == 0` and `step_failed == true` together mean no
 * trial was accepted).
 */
struct NewtonIterationRecord {
    int picard_iteration_context{};
    real r_F{};
    real forcing_rel_tol{};

    CoupledGmresStatus gmres_status{CoupledGmresStatus::max_iterations};
    int gmres_total_inner_iterations{};
    int gmres_outer_cycles{};
    real gmres_final_true_residual{};

    std::vector<NewtonTrialRecord> trials{};
    real accepted_alpha{};
    bool step_failed{false};
    NewtonStepFailureReason failure_reason{NewtonStepFailureReason::none};
};

/** Terminal disposition of one `run_newton_phase` activation call (E3/E6). */
enum class NewtonPhaseStatus { converged, budget_exhausted, step_failed };

struct NewtonPhaseOutcome {
    NewtonPhaseStatus status{NewtonPhaseStatus::step_failed};
};

/**
 * Runs ONE Newton phase activation: repeated Newton iterations (HEAD ->
 * `prepare_jvp_base` -> inexact GMRES solve -> Armijo line search) against
 * the workspace's optional SF-24 Newton sub-workspace
 * (`workspace.newton_jvp()`/`newton_gmres()`/`newton_preconditioner()`/
 * `newton_delta1()`/`newton_delta2()`; throws `std::logic_error` if the
 * workspace was not prepared with `config.newton.enabled`), until:
 *
 *   - `converged`: the coupled residual reached `config.picard.tolerance` at
 *     a Newton HEAD (the SAME tolerance/fixed point the Picard loop uses);
 *   - `budget_exhausted`: `config.newton.max_newton_iterations` HEADs
 *     elapsed without converging;
 *   - `step_failed`: a GMRES or line-search failure (see the file header,
 *     E5-E7); the accepted state (`fields`) is UNCHANGED from this call's
 *     entry state.
 *
 * `residual_out` is updated (by value-copy) at every Newton HEAD, whatever
 * the eventual outcome; `newton_history` receives exactly one
 * `NewtonIterationRecord` per Newton iteration attempted (including the
 * failing one, if any); `newton_steps_accepted`/`newton_step_failures` are
 * incremented in place. `picard_iteration_context` is the outer Picard-loop
 * `k` at which the caller entered this activation (recorded verbatim into
 * every produced `NewtonIterationRecord`, NOT incremented by this file).
 *
 * See the file header's "module boundary" note for why this takes report
 * sub-objects by reference rather than a `StreamfunctionSolveReport&`.
 */
[[nodiscard]] NewtonPhaseOutcome run_newton_phase(
    CudaContext& context, const StreamfunctionProblemView& problem,
    const StreamfunctionSolverConfig& config, StreamfunctionFields& fields,
    StreamfunctionWorkspace& workspace, const NonlinearSourceConfig& source_config,
    int picard_iteration_context, std::vector<NewtonIterationRecord>& newton_history,
    int& newton_steps_accepted, int& newton_step_failures,
    StreamfunctionResidualReport& residual_out);

} // namespace streamfunctions
} // namespace macroflow3d
