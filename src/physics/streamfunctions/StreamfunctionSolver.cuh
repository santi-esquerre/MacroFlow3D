#pragma once

/**
 * @file StreamfunctionSolver.cuh
 * @brief Public entry point declaration for the Lester equation (14)
 *        streamfunction solver.
 *
 * SF-12 declared this API without a definition. SF-13 (`StreamfunctionSolver.cu`)
 * provided the first `solve_streamfunctions` body: a v1 ZERO-SOURCE
 * (harmonic-coordinate) linear solve. SF-14 extended that body with a
 * fixed-relaxation Picard outer loop over the coupled nonlinear sources.
 * SF-15 adds adaptive globalization (backtracking relaxation, omega growth,
 * stagnation detection, trial-rejection guards) on top of the SAME
 * fixed-point map: no new fixed point is introduced anywhere in this file.
 *
 * SF-17 adds a `config.initial_state` switch (`PicardInitialState` in
 * `StreamfunctionTypes.hpp`). `zero_source` (default) reproduces the SF-13
 * v1 initialization below EXACTLY. `warm_start` instead skips the zero-init
 * and the zero-source block solves entirely: state 0 is the caller-provided
 * `fields.u1_span()`/`u2_span()`, mean-zero projected in place before the
 * Picard loop; `report.psi1_result`/`psi2_result` remain default-constructed
 * until the first Picard update step. This is the mechanism
 * `ContinuationController` (SF-17) uses to carry the last ACCEPTED state
 * across a continuation stage; see `PicardInitialState` for the full
 * contract.
 *
 * v1 zero-source initialization (SF-13, unchanged under
 * `initial_state == zero_source`):
 *   - `u1`, `u2` are zero-initialized on every call (no warm start).
 *   - `q = 1/K` or `q = exp(-Y)` is filled from `problem.conductivity` per
 *     `problem.conductivity_representation`.
 *   - the affine periodic RHS pair is assembled and mean-zero-projected
 *     (`assemble_affine_periodic_rhs`), then each linear block
 *     `A psi_i = rhs_i` is solved ONCE via the accepted projected-PCG/MG
 *     stack, sequentially: psi1 first, then psi2, sharing the workspace's
 *     single hierarchy/PCG workspace. This produces the Picard state k = 0
 *     (the zero-source, harmonic-coordinate estimate).
 *   - the measured Darcy `v_rms` (SF-11 physical diagnostics) is computed
 *     exactly once per call, from `problem.darcy_velocity`, a state-independent
 *     Darcy property; it is never recomputed inside the Picard loop.
 *   - `status == invalid_problem` is returned (after the zero-source linear
 *     solves and SF-11 diagnostics, before any residual evaluation) when
 *     `report.diagnostics.v_d_rms` is non-finite or non-positive: the SF-09
 *     nonlinear-source contract and the `r1` residual normalization both
 *     require a strictly positive `v_rms`.
 *
 * SF-14 fixed-relaxation Picard fixed-point map (unchanged, still the map
 * SF-15 globalizes):
 *   - The nonlinear system solved is `A u1 = P(rhs_affine1 - eta*q*S2)`,
 *     `A u2 = P(rhs_affine2 - eta*q*S1)` (pairing F1<->S2, F2<->S1), i.e. the
 *     full coupled Lester equation (14) system in the periodic fluctuations.
 *   - At the head of every outer iteration `k = 0, 1, ...`, the coupled
 *     residual `F1`, `F2` is evaluated ONCE from the current, immutable
 *     ACCEPTED state (`enqueue_streamfunction_residual`), producing BOTH
 *     block right-hand sides `G1`, `G2` in the same enqueue.
 *   - The two linear blocks are then solved sequentially, at the SAME frozen
 *     accepted state k, against the SAME operator/MG hierarchy/PCG
 *     workspace: `u_hat1` solves `A u_hat1 = G1`, `u_hat2` solves
 *     `A u_hat2 = G2`. No residual re-evaluation occurs between the two
 *     block solves, so both sources are guaranteed to come from one
 *     immutable state. Each block solve starts from a ZERO initial guess
 *     (not a warm start from the current `u_i`): near the Picard fixed point
 *     a warm start makes the initial projected residual O(||F_i||) (tiny),
 *     which makes the PCG relative-residual stopping criterion
 *     (`final/initial <= rtol`) unattainable at double precision;
 *     zero-initializing keeps the initial projected residual at the O(1)
 *     scale set by the affine RHS `G_i`, so `rtol` stays attainable at
 *     every Picard iteration. This "map" step (the two block solves) is
 *     performed EXACTLY ONCE per outer iteration `k`, regardless of how
 *     many backtracking trials SF-15 subsequently evaluates: `u_hat1`,
 *     `u_hat2` (stored in `f1`/`f2`) are frozen for the whole backtracking
 *     search at iteration k.
 *   - The candidate next state is `u_i <- (1-omega)*u_i + omega*u_hat_i`,
 *     mean-zero projected. SF-14 applied this once, unconditionally, with
 *     the fixed `config.picard.omega`; SF-15 (below) instead searches over
 *     `omega` with rejection/backtracking before accepting a candidate.
 *
 * SF-15 adaptive-Picard globalization (`config.adaptive`, see
 * `AdaptivePicardConfig` in `StreamfunctionTypes.hpp` for every default and
 * tunable):
 *
 *   `config.adaptive.enabled == false` reproduces the SF-14 loop EXACTLY
 *   (same code path, same fixed relaxation, no rejection, no stagnation
 *   check): `report.exit_reason` is still populated for that path
 *   (`converged`, `budget_exhausted`, or `linear_block_failure`) and
 *   `report.final_omega == config.picard.omega`, but the numerical result is
 *   bitwise identical to SF-14.
 *
 *   `config.adaptive.enabled == true` (the default) instead:
 *
 *   1. HEAD (once per outer iteration k): evaluate the coupled residual at
 *      the current ACCEPTED state, exactly as SF-14's head step, producing
 *      `r_F_k`, `G1`/`G2`, and (borrowed from `ResidualEvaluator.cuh`) the
 *      |c| 0.1% histogram percentile `p_k =
 *      residual_histogram_percentile(residual_k, 0.001)`. If
 *      `config.diagnostics.num_degeneracy_thresholds > 0` ("guards active"),
 *      SF-11 physical diagnostics are ALSO re-run at this accepted state to
 *      capture `f_prev`, the threshold-0 unexplained-cell fraction
 *      (`degeneracy_unexplained[0] / n`); this is the only place guards-active
 *      adds cost beyond the SF-14 head evaluation. `report.picard_history`
 *      gets one entry per outer iteration, with the same layout convention
 *      SF-14 documented (entry k's `psi1_result`/`psi2_result` are the block
 *      solves that PRODUCED state k).
 *   2. Exit checks, in order: `r_F_k <= config.picard.tolerance` ->
 *      `exit_reason = converged`; `k == config.picard.max_iter` ->
 *      `exit_reason = budget_exhausted`; stagnation (once at least
 *      `config.adaptive.stagnation_window` full accepted-step transitions
 *      have occurred, i.e. `k >= stagnation_window`, and
 *      `r_F_k > (1 - stagnation_min_reduction) * r_F_{k-stagnation_window}`,
 *      both read directly from `report.picard_history`) ->
 *      `exit_reason = stagnated`. A residual INCREASE is never relabeled as
 *      stagnation: this check only fires on an insufficient DECREASE.
 *   3. MAP (once per outer iteration k, only if none of the above exited):
 *      the two block solves `A u_hat_i = G_i`, exactly as SF-14, reusing
 *      `f1`/`f2` as scratch for `u_hat1`/`u_hat2`. A linear-solve failure
 *      exits immediately with `exit_reason = linear_block_failure`, using
 *      the same NaN-sentinel trailing `PicardIterationRecord` SF-14
 *      documented.
 *   4. BACKTRACKING (inner loop over a per-iteration trial relaxation
 *      `omega_try`, starting at the persistent `omega`): each trial computes
 *      `u_trial_i = (1-omega_try)*u_i + omega_try*u_hat_i` into the
 *      workspace's OWN `u_trial1()`/`u_trial2()` scratch pair (SF-15;
 *      `StreamfunctionWorkspace.cuh`) via `cudaMemcpyAsync` + `blas::scal` +
 *      `blas::axpy` + mean-zero projection -- the accepted `u1`/`u2` in
 *      `fields` are NEVER overwritten by a trial, only by an ACCEPTED trial
 *      (see step 5). The trial's residual is then evaluated from
 *      `u_trial1`/`u_trial2` via `enqueue_streamfunction_residual`, reusing
 *      `workspace.rhs1()`/`rhs2()` as the (idle, post-initialization) output
 *      buffers -- this overwrites the residual workspace's borrowed `G1`/`G2`
 *      views, which is safe because the MAP step (3) already consumed them
 *      for this outer iteration. If guards are active, SF-11 diagnostics are
 *      also re-run on the trial state for `f_t`. Every trial, accepted or
 *      rejected, is appended to `report.trial_history` as one
 *      `PicardTrialRecord{iteration=k, omega=omega_try, r_F_trial, outcome}`.
 *      A trial is rejected, checked in this fixed order, for the FIRST
 *      matching reason:
 *        - `rejected_nonfinite`: `r_F_trial` non-finite, or the trial
 *          residual report's `nonfinite_s1`/`nonfinite_s2` counters are
 *          nonzero;
 *        - `rejected_degeneracy` (guards active only): the trial's
 *          unexplained fraction `f_t` exceeds
 *          `config.adaptive.max_unexplained_fraction`, OR exceeds
 *          `config.adaptive.unexplained_growth_factor * f_prev +
 *          config.adaptive.unexplained_growth_offset`;
 *        - `rejected_percentile` (guards active only): the trial's |c| 0.1%
 *          percentile collapses by more than
 *          `config.adaptive.percentile_collapse_factor` relative to `p_k`
 *          (`p_t < p_k / percentile_collapse_factor`) WITHOUT a matching
 *          increase in the unexplained/low-speed population
 *          (`f_t > f_prev`);
 *        - `rejected_armijo`: `r_F_trial > (1 - armijo_c*omega_try) * r_F_k`
 *          (an Armijo-style sufficient-decrease test; `armijo_c == 0`
 *          degenerates to "any non-increase is accepted", never "any
 *          increase").
 *      A residual INCREASE never satisfies the Armijo test and is therefore
 *      never accepted under any guard configuration.
 *   5. On ACCEPTANCE: `u_trial1`/`u_trial2` are copied into
 *      `fields.u1_span()`/`u2_span()` (the only place the accepted state
 *      changes). The persistent `omega` becomes the accepted `omega_try`.
 *      If this trial was accepted with ZERO backtracks (the very first trial
 *      of the iteration), the "easy streak" counter increments; once it
 *      reaches `config.adaptive.easy_streak`, `omega` grows by
 *      `config.adaptive.growth_factor`, capped at `config.adaptive.omega_max`,
 *      and the streak resets to 0. Any backtrack resets the streak to 0
 *      without growing `omega` beyond the accepted `omega_try`. `omega`
 *      persists across outer iterations (it is not reset at the top of each
 *      iteration's backtracking search).
 *   6. On REJECTION: `omega_try` is halved (`config.adaptive.backtrack_factor`,
 *      default 0.5), clamped to `config.adaptive.omega_min` (default 0.01).
 *      If the REJECTED trial's `omega_try` was already AT the floor
 *      (`omega_try <= omega_min`), the loop exits immediately with
 *      `exit_reason = omega_floor_rejected` -- a rejected floor trial is a
 *      structured failure, never silently accepted and never retried beyond
 *      the one floor trial; `fields.u1_span()`/`u2_span()` remain exactly
 *      the last accepted state (state k), untouched.
 *
 *   Cost note: the head residual/diagnostics evaluation (step 1) is
 *   deliberately re-run at the accepted state every outer iteration even
 *   though its `r_F`/`f_prev` values could in principle be cached from the
 *   PREVIOUS iteration's final accepted trial evaluation; this one redundant
 *   evaluation per accepted step is retained for structural simplicity
 *   (a single, uniform "evaluate residual at state X" code path for both the
 *   head and every trial). Eliminating it is deferred to a later increment
 *   (see the plan; not SF-15 scope).
 *
 * `status` semantics (rewritten for SF-15; `exit_reason` on
 * `StreamfunctionSolveReport` carries the precise reason in every case):
 *   - `converged`: the coupled nonlinear residual reached
 *     `r_F <= config.picard.tolerance` at the head of some iteration k
 *     (`exit_reason == PicardExitReason::converged`).
 *   - `not_converged`: any of `exit_reason == budget_exhausted`,
 *     `linear_block_failure`, `stagnated`, or `omega_floor_rejected`. Every
 *     exit, regardless of reason, carries a final `picard_iterations` count,
 *     the final `residual`/`diagnostics` reports, `final_omega`, and the
 *     complete `picard_history`/`trial_history`.
 *   - `invalid_problem`: unchanged from SF-13 (degenerate measured `v_rms`).
 *
 * SF-20 Anderson acceleration (`config.anderson`, see `AndersonConfig` in
 * `StreamfunctionTypes.hpp` and `AndersonAccelerator.cuh`; default
 * `enabled == false`, bitwise identical to SF-15):
 *
 *   `config.anderson.enabled == false` executes IDENTICAL statements to the
 *   SF-15 adaptive loop above (every Anderson step below is wrapped in
 *   `if (config.anderson.enabled)`); `workspace.anderson()` is never
 *   constructed and `report.anderson_*` fields stay default (`0`).
 *
 *   `config.anderson.enabled == true` inserts, once per outer iteration `k`,
 *   AFTER the MAP step (3, unchanged: `u_hat1`/`u_hat2` in `workspace.f1()`/
 *   `f2()`) and BEFORE the SF-15 BACKTRACKING search (4):
 *
 *   1. HISTORY MAINTENANCE (unconditional, every enabled outer iteration):
 *      `workspace.anderson().update_history(context, fields.u1_span(),
 *      fields.u2_span(), workspace.f1(), workspace.f2())` forms the coupled
 *      fixed-point residual `f_k = u_hat_k - x_k` (`x_k` the current
 *      ACCEPTED state, read here BEFORE any trial touches it) and appends a
 *      new `(DeltaX, DeltaF)` ring-buffer column against the PREVIOUS
 *      accepted `(x, f)` pair, if one is staged; the staged pair is then
 *      replaced by `(x_k, f_k)` unconditionally. This runs regardless of `k`
 *      relative to `start_iteration`, so history accumulates from the first
 *      enabled outer iteration.
 *   2. ATTEMPT GATE: acceleration is attempted only when
 *      `k >= config.anderson.start_iteration` AND
 *      `workspace.anderson().num_columns() >= 1`.
 *   3. CANDIDATE FORMATION (attempt gate true): `workspace.anderson()
 *      .form_candidate(context, workspace.f1(), workspace.f2(),
 *      config.anderson.condition_limit, workspace.u_trial1(),
 *      workspace.u_trial2(), condition_estimate)` solves the small
 *      Gram-system least squares and writes the UNPROJECTED candidate
 *      `x_acc = u_hat_k - sum_j gamma_j*(DeltaX_j + DeltaF_j)` (algebraically
 *      `x_k + f_k - (X_k+F_k)*gamma`, see `AndersonAccelerator.cuh`) into the
 *      SAME `u_trial1()`/`u_trial2()` scratch pair the SF-15 backtracking
 *      search below uses (reused safely: stream-ordered, and Anderson always
 *      runs to completion, whether accepted or rejected, before backtracking
 *      would touch those buffers again). A `false` return (condition
 *      estimate `> condition_limit`, or `num_columns() == 0`, unreachable
 *      here given the attempt gate) counts `anderson_condition_resets`,
 *      clears the history (`workspace.anderson().clear()`), and falls
 *      through to the UNCHANGED SF-15 backtracking search over the Picard
 *      candidate (`u_hat1`/`u_hat2`, still held in `f1()`/`f2()`) for this
 *      same `k` -- omega/streak state is untouched by a condition reset.
 *   4. TRIAL EVALUATION (candidate formed): `workspace.u_trial1()`/
 *      `u_trial2()` are mean-zero projected in place (gauge), then evaluated
 *      through the IDENTICAL trial guard chain as an SF-15 backtracking
 *      trial (step 4 above: nonfinite -> degeneracy -> percentile ->
 *      Armijo), with `omega_try` fixed at `1` for the Armijo sufficient-
 *      decrease test (`r_F_trial > (1 - armijo_c) * r_F_k`) and reusing
 *      `workspace.rhs1()`/`rhs2()` as the residual-evaluation output buffers
 *      (idle at this point, exactly as the SF-15 backtracking trial reuses
 *      them). The trial is appended to `report.trial_history` with
 *      `PicardTrialRecord::anderson == true`.
 *        - ACCEPTED: `u_trial1()`/`u_trial2()` are copied into
 *          `fields.u1_span()`/`u2_span()` (the only accepted-state change,
 *          identical mechanism to an SF-15 acceptance); `anderson_accepted`
 *          is counted; the persistent `omega`/easy-streak state is left
 *          UNCHANGED (Anderson acceptance neither grows nor shrinks it); the
 *          SF-15 backtracking search is SKIPPED entirely for this `k`
 *          (`continue` to the next outer iteration's HEAD).
 *        - REJECTED (any guard): the history is cleared
 *          (`workspace.anderson().clear()`), `anderson_rejected` is
 *          counted, and the loop falls through to the UNCHANGED SF-15
 *          backtracking search over the Picard candidate for this same `k`,
 *          starting at the persistent `omega` exactly as if Anderson had
 *          never been attempted.
 *
 *   Anderson therefore NEVER bypasses a trial guard and NEVER introduces a
 *   new fixed point: an accepted Anderson candidate is just another
 *   guard-passing point on the same coupled fixed-point map that the SF-15
 *   backtracking search also samples; a rejected one is invisible to
 *   `fields.u1_span()`/`u2_span()` and only costs one extra evaluated trial
 *   plus a cleared history.
 *
 * SF-21 adds `config.coefficient_state` (`CoefficientState` in
 * `StreamfunctionTypes.hpp`): `rebuild` (default) reproduces every prior
 * increment's q-fill/MG-hierarchy-population/affine-RHS-assembly EXACTLY;
 * `reuse` skips all three, requires `initial_state == warm_start`, and
 * requires the workspace to already hold a valid rebuild for its currently
 * prepared grid. See `CoefficientState` for the full caller contract.
 *
 * SF-21 also clears the Anderson history (`workspace.anderson().clear()`)
 * ONCE at the start of every `solve_streamfunctions` call whenever
 * `config.anderson.enabled`, before the Picard loop begins. Rationale: the
 * accelerator's premise (`AndersonAccelerator.cuh`) is a history of ONE
 * fixed-point map instance -- delta pairs `(DeltaX, DeltaF)` staged against a
 * PREVIOUS accepted `(x, f)` are only meaningful when every pair in the
 * history comes from the SAME map. The SF-21 heterogeneity continuation
 * driver (`ContinuationController`) calls `solve_streamfunctions` repeatedly
 * on the SAME workspace across different problem instances (different
 * lambda-scaled conductivity, different eta, a freshly re-run affine Darcy
 * reference velocity) while carrying the accepted state forward via
 * `initial_state == warm_start`. Without an entry clear, any Anderson history
 * left over from a PRIOR call's final outer iterations would still be staged
 * (a `prev_x`/`prev_f` pair from the OLD map instance); the NEXT call's first
 * `update_history` would then form a `(DeltaX, DeltaF)` column straddling the
 * parameter change -- mixing residuals of two different fixed-point maps into
 * one Gram system. Clearing at solve entry guarantees every history column
 * `form_candidate` ever consumes was staged entirely within the CURRENT call.
 * `clear()` on a fresh or already-cleared accelerator (every SF-20
 * single-solve-call use, and every call before `workspace.anderson()` has
 * been touched) is a no-op, so this is bitwise-preserving for every existing
 * SF-20 caller. No other Anderson semantics (algebra, guard chain, counters,
 * mid-loop rejection-clear) change.
 *
 * SF-24 adds `config.newton` (`NewtonKrylovConfig`, `StreamfunctionTypes.hpp`;
 * default `enabled == false`), a globalized Newton-Krylov phase inserted into
 * the `config.adaptive.enabled == true` loop above (the SF-14 fixed-Picard
 * path, `config.adaptive.enabled == false`, is UNTOUCHED by SF-24). Every
 * Newton statement below is guarded by `if (config.newton.enabled)`, so
 * `enabled == false` executes IDENTICAL statements to the pre-SF-24
 * SF-15/SF-20 adaptive loop (the same standard SF-20 met for its own disabled
 * path). The full per-Newton-iteration mechanics (HEAD, forcing, GMRES
 * linear-accept rule, Armijo line search, and the Newton-step-failure/rescue/
 * retry state machine) are documented in `NewtonKrylovSolver.cuh`; this
 * section documents only the OUTER activation/rescue/retry state machine
 * `solve_streamfunctions` itself drives around `run_newton_phase`.
 *
 *   ACTIVATION (E2): inserted at the outer-loop HEAD, AFTER the existing
 *   `converged`/`budget_exhausted` exit checks and BEFORE the stagnation
 *   check. Entry into a Newton phase happens for exactly one of three
 *   reasons: (a) `r_F_k <= config.newton.activation_r_F`; (b) the SF-15
 *   stagnation condition would otherwise fire AND `r_F_k <=
 *   config.newton.stagnation_activation_r_F` AND the accepted state's SF-11
 *   degeneracy is clean (`f_prev <= config.adaptive.max_unexplained_fraction`
 *   when guards are active, vacuously clean otherwise) -- this REPLACES a
 *   `stagnated` exit with a Newton-phase entry instead; (c) a forced retry
 *   after a completed rescue window (below), which bypasses (a)/(b)
 *   entirely. On EVERY entry, if `config.anderson.enabled`,
 *   `workspace.anderson().clear()` runs first (the SF-21 premise: a Newton
 *   acceptance changes the accepted state outside Anderson's own staging).
 *   `report.newton_activations` counts every entry (original activations and
 *   forced retries alike).
 *
 *   RESCUE WINDOW (E6): while an internal `newton_rescue_remaining` counter is
 *   positive, outer iteration HEADs are forced through the UNCHANGED SF-15/
 *   SF-20 Picard/Anderson loop body (Newton activation is suppressed for
 *   exactly that many iterations, decrementing the counter each time); their
 *   normal `converged`/`budget_exhausted`/`linear_block_failure`/`stagnated`/
 *   `omega_floor_rejected` exits resolve exactly as they always have. When
 *   the counter reaches zero, the NEXT HEAD forces a Newton-phase retry
 *   unconditionally (reason (c) above), counted once in
 *   `report.newton_rescue_events`.
 *
 *   RESOLUTION: `run_newton_phase` returning `converged` ends the WHOLE solve
 *   (`status = converged`, `exit_reason = converged`, the SAME
 *   `config.picard.tolerance` fixed point the Picard loop targets);
 *   `budget_exhausted` ends the whole solve (`status = not_converged`,
 *   `exit_reason = newton_budget_exhausted`); `step_failed` on an activation
 *   that has not yet used its rescue-retry schedules
 *   `config.newton.rescue_picard_steps` rescue-window iterations (`0` valid,
 *   forcing an immediate retry) and continues the outer loop; `step_failed` a
 *   SECOND time for the SAME activation is terminal (`status =
 *   not_converged`, `exit_reason = newton_exhausted`, the last ACCEPTED state
 *   unchanged by either failed attempt).
 *
 * SF-25 Phase-1 (S2) state-machine hygiene: three independently config-gated
 * fixes, each default OFF and exactly bitwise-preserving in that state (see
 * `NewtonKrylovConfig::rescue_resets_omega`,
 * `AdaptivePicardConfig::FloorGuardConfig`, and
 * `AndersonConfig::restart_on_stagnation`/`max_restarts` in
 * `StreamfunctionTypes.hpp` for the full field-level contract).
 *
 *   (a) `config.newton.rescue_resets_omega`: on entry to the rescue window
 *       (the same statement that sets `newton_rescue_remaining =
 *       config.newton.rescue_picard_steps` after a Newton step failure), the
 *       persistent adaptive `omega` is reset to `config.picard.omega`;
 *       `report.newton_rescue_omega_resets` counts each such reset.
 *   (b) `config.adaptive.floor_guard`: at the omega-floor rejection site (the
 *       backtracking trial at `omega_try <= config.adaptive.omega_min`),
 *       intercepts the would-be `omega_floor_rejected` exit when at least
 *       `floor_guard.window` accepted outer iterations have elapsed since
 *       solve start or the last guard reset AND the accepted-iteration
 *       residual history is still descending by `floor_guard.drop_factor`
 *       over that window; resets `omega` to `config.picard.omega`, counts
 *       `report.omega_floor_guard_resets`, and lets the outer loop continue
 *       (state k unchanged) instead of exiting; capped at
 *       `floor_guard.max_resets` interceptions per solve.
 *   (c) `config.anderson.restart_on_stagnation` (meaningful only together
 *       with `config.anderson.enabled`): at the ordinary SF-15 `stagnated`
 *       exit site, intercepts it while `report.anderson_stagnation_restarts
 *       < config.anderson.max_restarts`: clears the Anderson history
 *       (`AndersonAccelerator::clear()`), resets `omega` to
 *       `config.picard.omega`, resets the stagnation-window baseline (so the
 *       identical window cannot immediately re-fire), counts
 *       `report.anderson_stagnation_restarts`, and continues the outer loop
 *       instead of exiting.
 */

#include "../../numerics/solvers/pcg.cuh"
#include "../../runtime/CudaContext.cuh"
#include "Diagnostics.cuh"
#include "NewtonKrylovSolver.cuh"
#include "ResidualEvaluator.cuh"
#include "StreamfunctionTypes.hpp"
#include "StreamfunctionWorkspace.cuh"

#include <vector>

namespace macroflow3d {
namespace streamfunctions {

/**
 * Minimal solve status, extended by SF-14 with truthful Picard-loop
 * semantics; see the file header for the exact meaning of each value.
 */
enum class StreamfunctionSolveStatus {
    not_run,
    converged,
    not_converged,
    invalid_problem,
};

/**
 * Precise reason the Picard loop stopped (SF-15). See the file header for
 * the exact condition triggering each value. `none` is the
 * default-constructed sentinel and is never the value of a returned report
 * whose `status != not_run`.
 */
enum class PicardExitReason {
    none,
    converged,
    budget_exhausted,
    linear_block_failure,
    stagnated,
    omega_floor_rejected,
    // SF-24: terminal Newton-Krylov phase exits (see the file header's SF-24
    // section and NewtonKrylovSolver.cuh). Both map to status =
    // not_converged, exactly like every prior not_converged reason above.
    newton_exhausted,
    newton_budget_exhausted,
};

// `PicardTrialOutcome` (the disposition of one SF-15 backtracking trial, and,
// SF-24, one Newton line-search trial) is defined in `StreamfunctionTypes.hpp`
// (included via StreamfunctionWorkspace.cuh below), not here -- see that
// header for the rationale (it must be visible to both this file and
// `NewtonKrylovSolver.cuh` without either including the other).

/**
 * One recorded SF-15 backtracking trial: which outer iteration it belongs
 * to, the relaxation factor it tried, the trial's coupled residual `r_F`,
 * and its outcome. Every trial evaluated by the adaptive loop is recorded
 * here, accepted or rejected, in evaluation order.
 *
 * `anderson` (SF-20, default `false`) marks a trial as a type-II Anderson
 * mixing candidate (`omega == 1` by construction, see
 * `StreamfunctionSolver.cuh`'s SF-20 section) rather than a plain SF-15
 * backtracking trial; it is evaluated through the IDENTICAL guard chain
 * (`outcome` uses the same `PicardTrialOutcome` values) before ever touching
 * the accepted state.
 */
struct PicardTrialRecord {
    int iteration{};
    real omega{};
    real r_F_trial{};
    PicardTrialOutcome outcome{PicardTrialOutcome::accepted};
    bool anderson{false};
};

/**
 * One recorded Picard iteration (SF-14): the coupled residual reductions
 * evaluated AT state k, together with the linear block results that
 * PRODUCED state k. See the file header for the exact layout convention
 * (entry 0 carries default-constructed `ProjectedPCGResult`s because state 0
 * is the SF-13 zero-source initialization, not a Picard update). The
 * TRAILING record appended when a block solve fails to converge is a
 * documented exception: its `r_F`/`r1`/`r2` are the NaN sentinel
 * (`std::isnan`) meaning "never evaluated" (the state-(k+1) residual that
 * would have produced them was never computed), and that record is
 * identified by its non-converged `psi1_result`/`psi2_result`, not by its
 * residual fields.
 */
struct PicardIterationRecord {
    real r_F{};
    real r1{};
    real r2{};
    solvers::ProjectedPCGResult psi1_result{};
    solvers::ProjectedPCGResult psi2_result{};
};

/**
 * Host-assembled report for one `solve_streamfunctions` call, composing the
 * accepted per-primitive reports rather than duplicating their fields:
 * `residual` (SF-09/10 coupled residual and its dimensionless reductions, at
 * the FINAL Picard state), `diagnostics` (SF-11 physical/Gate-3A
 * diagnostics, re-evaluated at the final state), `psi1_result`/
 * `psi2_result` (the SF-04 projected-PCG result for the linear block solves
 * that produced the final state; for a converged-at-k=0 solve these remain
 * the zero-source initialization results), `picard_history`/
 * `picard_iterations` (SF-14 fixed-relaxation Picard loop record),
 * `exit_reason`/`final_omega`/`trial_history` (SF-15 adaptive-Picard
 * globalization record; see the file header), and `memory` (the SF-12
 * exact-byte memory report for the workspace that produced this solve).
 * `status` is the overall outcome; see `StreamfunctionSolveStatus`.
 *
 * SF-20 Anderson acceleration counters (all `0`/unset when
 * `config.anderson.enabled == false`, matching the bitwise-preserving
 * disabled path): `anderson_accepted` counts outer iterations where an
 * Anderson-mixed candidate was accepted (state advanced without the SF-15
 * Picard backtracking search running that iteration); `anderson_rejected`
 * counts outer iterations where an attempted Anderson candidate failed the
 * trial guard chain (history cleared, SF-15 Picard backtracking ran as the
 * fallback); `anderson_condition_resets` counts outer iterations where the
 * Gram-system condition estimate exceeded `config.anderson.condition_limit`
 * BEFORE a candidate was even evaluated (history cleared, same Picard
 * fallback). `anderson_history_bytes` mirrors
 * `memory.anderson_history_bytes` (the workspace's actual, exact
 * `4 * depth * n * sizeof(real)` history footprint) for convenient top-level
 * reporting; see `StreamfunctionMemoryReport` for the full byte accounting.
 */
struct StreamfunctionSolveReport {
    StreamfunctionSolveStatus status{StreamfunctionSolveStatus::not_run};

    StreamfunctionResidualReport residual{};
    PhysicalDiagnosticsReport diagnostics{};
    solvers::ProjectedPCGResult psi1_result{};
    solvers::ProjectedPCGResult psi2_result{};
    StreamfunctionMemoryReport memory{};

    std::vector<PicardIterationRecord> picard_history{};
    int picard_iterations{0};

    // SF-15 adaptive-Picard globalization.
    PicardExitReason exit_reason{PicardExitReason::none};
    real final_omega{};
    std::vector<PicardTrialRecord> trial_history{};

    // SF-20 Anderson acceleration; see above.
    int anderson_accepted{0};
    int anderson_rejected{0};
    int anderson_condition_resets{0};
    std::size_t anderson_history_bytes{0};

    // SF-24 globalized Newton-Krylov; see the file header's SF-24 section and
    // NewtonKrylovSolver.cuh. All default-zero/empty and untouched whenever
    // config.newton.enabled == false.
    std::vector<NewtonIterationRecord> newton_history{};
    int newton_activations{0};
    int newton_steps_accepted{0};
    int newton_step_failures{0};
    int newton_rescue_events{0};
    long long newton_jv_evaluations{0};

    // SF-25 Phase-1 (S2) state-machine hygiene counters: additive, zero when
    // the corresponding config flag is off (config.newton.rescue_resets_omega,
    // config.adaptive.floor_guard.enabled, config.anderson.restart_on_
    // stagnation respectively). See StreamfunctionTypes.hpp for the exact
    // semantics of each counted event.
    int newton_rescue_omega_resets{0};
    int omega_floor_guard_resets{0};
    int anderson_stagnation_restarts{0};
};

/**
 * Solve the Lester equation (14) streamfunction system for one problem
 * instance, writing the accepted invariants into `fields` and using `workspace`
 * for every scratch/solver buffer (see `StreamfunctionWorkspace.cuh`).
 *
 * SF-14 iterates the full coupled nonlinear system via fixed-relaxation
 * Picard; SF-15 globalizes that SAME fixed-point map with adaptive
 * relaxation, backtracking, and stagnation detection (`config.adaptive`,
 * default enabled). See the file header for the exact solved system,
 * convergence semantics, and the `invalid_problem` condition. Throws
 * `std::invalid_argument` (via `validate_streamfunction_problem`) for
 * host-detectable problem/config misuse.
 */
[[nodiscard]] StreamfunctionSolveReport solve_streamfunctions(
    CudaContext& context, const StreamfunctionProblemView& problem,
    const StreamfunctionSolverConfig& config, StreamfunctionFields& fields,
    StreamfunctionWorkspace& workspace);

} // namespace streamfunctions
} // namespace macroflow3d
