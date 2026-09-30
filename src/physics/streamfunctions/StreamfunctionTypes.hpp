#pragma once

/**
 * @file StreamfunctionTypes.hpp
 * @brief Public problem/config/report types for the Lester equation (14)
 *        streamfunction solver (SF-12).
 *
 * This header defines the SF-12 public surface: the nonowning problem view,
 * the composed solver configuration, host validation, and the exact-byte
 * memory report shared by the estimator and the workspace. It intentionally
 * composes rather than duplicates the accepted SF-02..SF-11 primitive types:
 * `AffineGauge` / `PeriodicStreamfunctionFluctuations` (SF-06),
 * `multigrid::MGConfig` (accepted MG stack), `solvers::ProjectedPCGConfig`
 * (SF-04), `ResidualHistogramConfig` (SF-09/10), `PhysicalDiagnosticsConfig`
 * (SF-11), and `CompactMacVelocityConstView` (SF-11).
 *
 * `StreamfunctionSolveReport` and `solve_streamfunctions` are declared in
 * `StreamfunctionSolver.cuh`; `StreamfunctionFields` and
 * `StreamfunctionWorkspace` are declared in `StreamfunctionWorkspace.cuh`.
 * This header only defines types with no owned device storage.
 */

#include "../../core/BCSpec.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../multigrid/mg_types.hpp"
#include "../../numerics/solvers/pcg.cuh"
#include "CoupledGmres.cuh"
#include "Diagnostics.cuh"
#include "NonlinearSources.cuh"
#include "ResidualEvaluator.cuh"
#include "affine_gauge.cuh"

#include <cstddef>
#include <type_traits>

namespace macroflow3d {
namespace streamfunctions {

// `AffineGauge`, `AffineGradient`, and `PeriodicStreamfunctionFluctuations`
// are the accepted SF-06 types from affine_gauge.cuh, included above. SF-12
// re-exports them by including that header rather than redefining them; they
// are visible under `macroflow3d::streamfunctions` exactly as before.

/**
 * Which physical quantity `StreamfunctionProblemView::conductivity` stores.
 * The Lester equation (14) linear subproblem coefficient is always
 * `q = 1/K`; when the caller supplies `log_conductivity_y` (`Y = ln K`), the
 * consumer (SF-13) is responsible for computing `q = exp(-Y)`. This SF-12
 * increment only records which representation is in effect; it does not
 * convert it.
 */
enum class ConductivityRepresentation { conductivity_k, log_conductivity_y };

/**
 * Strictly nonowning view of one Lester equation (14) streamfunction
 * problem instance. No member allocates device memory; the caller owns and
 * keeps every referenced buffer alive for the lifetime of any call that
 * consumes this view.
 *
 * `grid` must equal, field for field, the grid argument every consuming call
 * (`validate_streamfunction_problem`, `solve_streamfunctions`, workspace
 * `prepare`) receives explicitly; a mismatch is rejected by
 * `validate_streamfunction_problem`.
 *
 * `conductivity` is cell-centered, `grid.num_cells()` elements, and holds
 * either `K` or `Y = ln K` per `conductivity_representation`. `darcy_velocity`
 * is the reference Darcy flow used by SF-11 physical diagnostics and by the
 * `v_rms` normalization threaded through `NonlinearSourceConfig`/
 * `ResidualHistogramConfig`, expressed as the SF-11 CompactMAC layout
 * (`CompactMacVelocityConstView`, see `Diagnostics.cuh`). `bc` must be
 * triply periodic for the v1 benchmark (see `validate_streamfunction_problem`).
 * `gauge` is the SF-06 affine gauge fixing `psi1 = vbar*x2 + u1_tilde`,
 * `psi2 = x3 + u2_tilde`.
 */
struct StreamfunctionProblemView {
    Grid3D grid{};
    DeviceSpan<const real> conductivity{};
    ConductivityRepresentation conductivity_representation{
        ConductivityRepresentation::conductivity_k};
    CompactMacVelocityConstView darcy_velocity{};
    BCSpec bc{};
    AffineGauge gauge{};
};

static_assert(std::is_trivially_copyable<StreamfunctionProblemView>::value,
              "StreamfunctionProblemView must be safe to copy by value (nonowning view)");

/**
 * Fixed-relaxation Picard configuration for the coupled nonlinear iteration
 * (SF-14). Both defaults and the omega value are dashboard-locked for this
 * increment: the outer loop performs plain fixed-point relaxation with no
 * step rejection, no adaptive relaxation, and no continuation. Those are
 * explicitly deferred to SF-15 and later increments.
 *
 *   - `max_iter`: maximum number of Picard update steps (block-solve pairs)
 *     performed after the state-0 (zero-source) evaluation. `max_iter == 0`
 *     performs no update step at all: the reported solution is exactly the
 *     zero-source initialization, and `picard_iterations == 0`.
 *   - `tolerance`: the nonlinear stop rule is `r_F <= tolerance`, checked at
 *     the head of every loop iteration (including k = 0, before any update).
 *   - `omega`: the SAME fixed relaxation factor applied to both `u1` and
 *     `u2` as a pair on every update step: `u_i <- (1-omega)*u_i +
 *     omega*u_hat_i`, where `u_hat_i` solves the frozen-state linear block.
 */
struct FixedPicardConfig {
    int max_iter{500};
    real tolerance{real{1e-6}};
    real omega{real{0.25}};
};

/**
 * Adaptive-Picard globalization configuration (SF-15), composed as
 * `StreamfunctionSolverConfig::adaptive`. Every default here is
 * dashboard-locked for the SF-15 increment. `enabled == false` disables the
 * globalization entirely and `solve_streamfunctions` reproduces the SF-14
 * fixed-relaxation Picard path exactly (bitwise-identical loop body); see
 * `StreamfunctionSolver.cuh` for the exact enabled-path semantics
 * (backtracking, growth, stagnation, and the trial-rejection guards).
 *
 *   - `enabled`: globalization on/off switch; `false` reproduces SF-14
 *     exactly.
 *   - `omega_min`: the smallest relaxation factor a backtracking trial may
 *     use; a rejected trial AT this floor is a structured failure
 *     (`omega_floor_rejected`), not a silently-accepted step.
 *   - `backtrack_factor`: multiplier applied to a rejected trial's `omega`
 *     to produce the next trial's `omega`, clamped to `omega_min`.
 *   - `growth_factor`: multiplier applied to the persistent `omega` after
 *     `easy_streak` consecutive zero-backtrack acceptances, capped at
 *     `omega_max`.
 *   - `omega_max`: the upper cap on the persistent relaxation factor after
 *     growth.
 *   - `easy_streak`: number of consecutive zero-backtrack (immediate)
 *     acceptances required before `omega` grows.
 *   - `armijo_c`: the sufficient-decrease constant in the Armijo-style
 *     acceptance test `r_F_trial <= (1 - armijo_c*omega_trial) * r_F_k`.
 *   - `stagnation_window`: number of accepted steps over which the residual
 *     reduction is measured for the stagnation exit rule.
 *   - `stagnation_min_reduction`: the minimum fractional residual reduction
 *     required over `stagnation_window` accepted steps to avoid a
 *     `stagnated` exit.
 *   - `max_unexplained_fraction`: absolute cap on the SF-11 degenerate-cell
 *     unexplained fraction at a trial state (only enforced when
 *     `config.diagnostics.num_degeneracy_thresholds > 0`).
 *   - `unexplained_growth_factor`, `unexplained_growth_offset`: a trial is
 *     also rejected when its unexplained fraction exceeds
 *     `unexplained_growth_factor * f_previous + unexplained_growth_offset`
 *     relative to the accepted state's unexplained fraction (same
 *     guard-active condition).
 *   - `percentile_collapse_factor`: a trial is rejected when its |c| 0.1%
 *     histogram percentile collapses by more than this decade factor
 *     relative to the accepted state's percentile, without a matching
 *     increase in the Darcy low-speed population (same guard-active
 *     condition; see `StreamfunctionSolver.cuh` for the exact test).
 *   - `floor_guard`: SF-25 Phase-1 state-machine hygiene (S2), config-gated,
 *     default OFF and bitwise-preserving (see `FloorGuardConfig` below).
 */
struct AdaptivePicardConfig {
    bool enabled{true};
    real omega_min{real{0.01}};
    real backtrack_factor{real{0.5}};
    real growth_factor{real{1.2}};
    real omega_max{real{1}};
    int easy_streak{3};
    real armijo_c{real{1e-4}};
    int stagnation_window{10};
    real stagnation_min_reduction{real{0.01}};
    real max_unexplained_fraction{real{0.01}};
    real unexplained_growth_factor{real{2}};
    real unexplained_growth_offset{real{1e-4}};
    real percentile_collapse_factor{real{10}};

    /**
     * SF-25 Phase-1 (S2) omega-floor-collapse guard, config-gated, default
     * `enabled == false` and exactly bitwise-preserving in that state (the
     * intercept below is never reached when `enabled == false`, so the
     * `omega_floor_rejected` exit fires exactly as before -- see
     * `StreamfunctionSolver.cu`'s backtracking step for the insertion
     * point).
     *
     * `enabled == true` intercepts an omega-floor rejection that would
     * otherwise terminate the adaptive stage (`PicardExitReason::
     * omega_floor_rejected`) IFF BOTH: (i) at least `window` ACCEPTED outer
     * iterations exist since solve start or since the last guard reset
     * (`k - floor_guard_last_reset_k >= window`, mirroring the
     * `stagnation_window` index-safety convention above); and (ii) the
     * accepted-iteration residual history is still descending by at least
     * `drop_factor` over that window: `r_F[k] <= drop_factor *
     * r_F[k-window]`, read from `report.picard_history` exactly like the
     * stagnation check. When intercepted: the persistent adaptive `omega` is
     * reset to `config.picard.omega` (the solve-start initial value), the
     * additive `StreamfunctionSolveReport::omega_floor_guard_resets` counter
     * increments, and the outer loop CONTINUES to the next iteration's HEAD
     * (the accepted state is left untouched -- exactly as an ordinary
     * `omega_floor_rejected` exit leaves it, except the solve does not
     * terminate). After `max_resets` interceptions for this solve, any
     * further floor rejection behaves exactly as today
     * (`omega_floor_rejected`). If the guard condition does not hold (either
     * (i) or (ii) fails), the floor rejection also behaves exactly as
     * today.
     *
     *   - `window`: the accepted-iteration lookback window (validated `>=
     *     1`).
     *   - `drop_factor`: the required fractional descent over `window`
     *     accepted iterations (validated finite, in `(0, 1)`).
     *   - `max_resets`: the maximum number of guard interceptions permitted
     *     for one `solve_streamfunctions` call (validated `>= 0`; `0`
     *     disables interception even when `enabled == true`, matching the
     *     `rescue_picard_steps == 0` degenerate-but-valid convention used
     *     elsewhere in this file).
     */
    struct FloorGuardConfig {
        bool enabled{false};
        int window{5};
        real drop_factor{real{0.9}};
        int max_resets{3};
    } floor_guard{};
};

/**
 * Anderson acceleration configuration (SF-20), composed as
 * `StreamfunctionSolverConfig::anderson`. `enabled == false` is the
 * dashboard-locked default and is bitwise-preserving: `solve_streamfunctions`
 * executes IDENTICAL statements to the pre-SF-20 SF-15 adaptive-Picard loop
 * (the Anderson bookkeeping and candidate formation are wrapped in
 * `if (config.anderson.enabled)`; see `StreamfunctionSolver.cuh` for the
 * exact insertion point, coupled-state history semantics, and safeguard
 * chain). `enabled == true` wraps the SAME SF-15 fixed-point map with a
 * type-II Anderson mixing candidate, evaluated through the identical trial
 * guard chain before ever touching the accepted state.
 *
 *   - `enabled`: acceleration on/off switch; `false` reproduces the SF-15
 *     adaptive loop exactly.
 *   - `depth`: number of retained coupled `(DeltaX, DeltaF)` history columns
 *     (validated to `[3, 8]`); the accelerator's device history storage is
 *     exactly `4 * depth * n * sizeof(real)` bytes for a grid of `n` cells
 *     (see `AndersonAccelerator.cuh`).
 *   - `start_iteration`: the first outer Picard iteration `k` (validated
 *     `>= 1`) at which an accelerated candidate may be attempted; history
 *     accumulates from the first enabled outer iteration regardless of this
 *     threshold, so by `k == start_iteration` up to `min(start_iteration,
 *     depth)` columns are typically already available.
 *   - `condition_limit`: the maximum allowed condition-number estimate
 *     (R-diagonal ratio from the pivoted-QR solve of the small Gram system)
 *     before an iteration's acceleration attempt is abandoned in favor of the
 *     unchanged SF-15 Picard fallback (validated finite `> 1`).
 *   - `restart_on_stagnation`, `max_restarts`: SF-25 Phase-1 (S2)
 *     stagnation-restart hygiene, config-gated, default `restart_on_
 *     stagnation == false` and exactly bitwise-preserving in that state.
 *     `restart_on_stagnation == true` (only meaningful together with
 *     `enabled == true`; inert otherwise, since no Anderson history exists
 *     to restart) intercepts the moment the SF-15 `stagnated` exit
 *     (`PicardExitReason::stagnated`) would fire: while the additive
 *     `StreamfunctionSolveReport::anderson_stagnation_restarts` counter is
 *     `< max_restarts`, the Anderson history is cleared
 *     (`AndersonAccelerator::clear()`), the persistent adaptive `omega` is
 *     reset to `config.picard.omega` (the solve-start initial value), the
 *     stagnation-window baseline is reset so the SAME window cannot
 *     immediately re-fire on the next outer iteration, the counter
 *     increments, and the outer loop CONTINUES instead of exiting. Once
 *     `max_restarts` interceptions have occurred, the `stagnated` exit fires
 *     exactly as today. `max_restarts` is validated `>= 0` regardless of
 *     `restart_on_stagnation`/`enabled` (mirroring this file's other
 *     unconditional-validation convention); `0` disables interception even
 *     when `restart_on_stagnation == true`.
 */
struct AndersonConfig {
    bool enabled{false};
    int depth{5};
    int start_iteration{5};
    real condition_limit{real{1e12}};
    bool restart_on_stagnation{false};
    int max_restarts{2};
};

/**
 * The disposition of one SF-15 backtracking trial (or, SF-24, one Newton
 * line-search trial -- see `NewtonKrylovSolver.cuh`). Defined here, rather
 * than in `StreamfunctionSolver.cuh` where `PicardTrialRecord` lives, so that
 * both `StreamfunctionSolver.cuh` (the SF-15 `trial_history`) and
 * `NewtonKrylovSolver.cuh` (the SF-24 `NewtonTrialRecord`) can reuse the SAME
 * type without either header including the other (SF-24 T01: `StreamfunctionSolver.cuh`
 * includes `NewtonKrylovSolver.cuh` for the additive `StreamfunctionSolveReport`
 * fields, so `NewtonKrylovSolver.cuh` cannot include `StreamfunctionSolver.cuh`
 * back). See `StreamfunctionSolver.cuh` (step 4) for the exact, order-sensitive
 * rejection test each non-`accepted` value corresponds to for an SF-15/SF-20
 * trial, and `NewtonKrylovSolver.cuh` for the SF-24 Newton line-search trial's
 * identical guard chain with an Armijo test on `0.5*r_F^2` in place of `r_F`.
 */
enum class PicardTrialOutcome {
    accepted,
    rejected_nonfinite,
    rejected_degeneracy,
    rejected_percentile,
    rejected_armijo,
};

/**
 * SF-24 globalized Newton-Krylov configuration, composed as
 * `StreamfunctionSolverConfig::newton`. `enabled == false` (default) is
 * bitwise-preserving: every Newton statement in `solve_streamfunctions`'
 * adaptive loop is guarded by `if (config.newton.enabled)`, so the disabled
 * path executes IDENTICAL statements to the pre-SF-24 SF-15/SF-20 adaptive
 * loop (the same standard SF-20 Anderson met for its own disabled path). See
 * `NewtonKrylovSolver.cuh` for the full activation, forcing, line-search, and
 * failure-protocol semantics this configuration drives, and
 * `StreamfunctionSolver.cuh` (SF-24 section) for the exact adaptive-loop
 * insertion point.
 *
 *   - `enabled`: Newton phase on/off switch; `false` reproduces the SF-15/
 *     SF-20 adaptive loop exactly. `enabled == true` REQUIRES
 *     `StreamfunctionSolverConfig::adaptive.enabled == true` (SF-24 C01):
 *     the Newton phase is integrated only into the adaptive Picard loop of
 *     `solve_streamfunctions`, and the SF-14 fixed-relaxation path is a
 *     frozen compatibility surface that never activates Newton.
 *     `validate_streamfunction_problem`/`estimate_streamfunction_memory`
 *     reject `newton.enabled && !adaptive.enabled` with
 *     `std::invalid_argument` rather than silently allocating a Newton
 *     sub-workspace that is never used.
 *   - `activation_r_F`: the accepted-state coupled residual threshold at or
 *     below which the Newton phase activates at an outer-loop HEAD (checked
 *     before the stagnation exit rule).
 *   - `stagnation_activation_r_F`: the (looser, `>= activation_r_F`)
 *     threshold under which a stagnation exit is instead redirected into the
 *     Newton phase, provided the accepted state's SF-11 degeneracy is clean
 *     (see `NewtonKrylovSolver.cuh`).
 *   - `forcing_coefficient`, `forcing_min`, `forcing_max`: the inexact-Newton
 *     forcing rule `rel_tol_k = clamp(forcing_coefficient * sqrt(r_F_k),
 *     forcing_min, forcing_max)` that OVERWRITES `gmres.rel_tol` every Newton
 *     iteration (the `gmres` field's own `rel_tol` default is otherwise
 *     unused by the Newton phase).
 *   - `armijo_c`: the sufficient-decrease constant in the Newton line-search
 *     Armijo test on the merit `phi = 0.5*r_F^2`:
 *     `phi_trial <= (1 - armijo_c*alpha) * phi_k`.
 *   - `alpha_min`: the smallest line-search step length attempted (default
 *     `2^-5`); a rejected trial at or below this floor ends the line search
 *     without acceptance (a Newton step failure, not a silently-accepted
 *     step).
 *   - `backtrack_factor`: multiplier applied to `alpha` between line-search
 *     trials (`alpha *= backtrack_factor`, starting at `alpha = 1`).
 *   - `max_newton_iterations`: the Newton-iteration budget for one Newton
 *     phase activation; reaching it without convergence is a structured
 *     `newton_budget_exhausted` exit (terminal for the whole solve, not
 *     retried).
 *   - `rescue_picard_steps`: number of ACCEPTED SF-15/SF-20 Picard/Anderson
 *     advances run (through the UNCHANGED loop body) after a Newton step
 *     failure, before retrying the Newton phase once; `0` is a valid value
 *     that skips straight to the retry (used to test bitwise-deterministic
 *     failure-protocol state preservation).
 *   - `gmres`: the SF-23 `CoupledGmresConfig` used verbatim for `restart`/
 *     `max_iterations`; `rel_tol` is overwritten every Newton iteration by
 *     the forcing rule above (its own default value is therefore never
 *     actually used by the Newton phase).
 *   - `delta`: the SF-22 `JvpDeltaConfig` forward-difference delta clamp,
 *     passed straight through to every `JvpWorkspace`/`CoupledGmres` call.
 *   - `rescue_resets_omega`: SF-25 Phase-1 (S2) rescue-omega-reset hygiene,
 *     config-gated, default `false` and exactly bitwise-preserving in that
 *     state. `true` resets the persistent adaptive `omega` to its
 *     solve-start initial value (`config.picard.omega`) on ENTRY to the
 *     rescue window (the same point `solve_streamfunctions` sets
 *     `newton_rescue_remaining = config.newton.rescue_picard_steps` after a
 *     Newton step failure), before any rescue-window Picard/Anderson step
 *     runs. Without this, the rescue window's Picard/Anderson trials start
 *     from whatever `omega` the SF-15 backtracking search had collapsed to
 *     immediately before Newton activated, which can be at or near
 *     `adaptive.omega_min` and make every rescue trial doomed to reject. The
 *     additive `StreamfunctionSolveReport::newton_rescue_omega_resets`
 *     counter increments once per rescue-window entry where the reset is
 *     applied (regardless of `rescue_picard_steps`, including the
 *     degenerate `0`-step case).
 */
struct NewtonKrylovConfig {
    bool enabled{false};
    real activation_r_F{real{1e-2}};
    real stagnation_activation_r_F{real{1e-1}};
    real forcing_coefficient{real{1}};
    real forcing_min{real{1e-8}};
    real forcing_max{real{1e-1}};
    real armijo_c{real{1e-4}};
    real alpha_min{real{0.03125}}; // 2^-5
    real backtrack_factor{real{0.5}};
    int max_newton_iterations{50};
    int rescue_picard_steps{5};
    CoupledGmresConfig gmres{};
    JvpDeltaConfig delta{};
    bool rescue_resets_omega{false};
};

/**
 * How `solve_streamfunctions` initializes `u1`/`u2` before the Picard loop
 * (SF-17). `zero_source` is the SF-13 default and remains bitwise identical
 * to every prior increment: `u1`/`u2` are zero-initialized and the two
 * zero-source block solves run to produce Picard state 0. `warm_start`
 * (SF-17, used by `ContinuationController` to carry the last ACCEPTED state
 * across a continuation stage) skips both the zero-init and the zero-source
 * block solves entirely; state 0 is instead the caller-provided
 * `fields.u1_span()`/`u2_span()`, mean-zero projected in place (gauge
 * defense) before the Picard loop starts. In both cases entry 0 of
 * `report.picard_history` carries default-constructed
 * `psi1_result`/`psi2_result` per the documented layout convention (see
 * `StreamfunctionSolveReport::picard_history`); that does not change between
 * modes. What differs is the top-level `report.psi1_result`/`psi2_result`:
 * under `zero_source` they hold the initialization's own PCG results until
 * the first successful Picard update overwrites them (unchanged SF-13..16
 * behavior); under `warm_start` no initialization solve exists, so they
 * genuinely remain default-constructed until the first successful Picard
 * update step.
 */
enum class PicardInitialState { zero_source, warm_start };

/**
 * SF-21: whether `solve_streamfunctions` (re)builds the coefficient-dependent
 * state -- `q` (`workspace.q()`), the MG coefficient hierarchy
 * (`multigrid::populate_coefficient_hierarchy`), and the affine periodic RHS
 * pair (`workspace.rhs1()`/`rhs2()`, via `assemble_affine_periodic_rhs`) --
 * or reuses whatever the workspace already holds from a prior call.
 *
 *   - `rebuild` (default): bitwise identical to every prior increment. Fills
 *     `q` from `problem.conductivity`, repopulates the MG hierarchy from `q`,
 *     and reassembles/mean-zero-projects `rhs1()`/`rhs2()` from `q` and
 *     `problem.gauge`, exactly as before.
 *   - `reuse`: skips all three steps. `workspace.q()`, the MG hierarchy, and
 *     `rhs1()`/`rhs2()` retain whatever content the workspace already holds.
 *
 * DOCUMENTED CALLER CONTRACT (undefined behavior otherwise): `reuse` is valid
 * ONLY when the immediately preceding `solve_streamfunctions` call on the
 * SAME `workspace` used the SAME grid, the SAME `problem.conductivity`
 * contents (and representation), and the SAME `problem.gauge` -- i.e. nothing
 * that would change `q`, the MG hierarchy, or the affine RHS. This is the
 * exact `ContinuationController` SF-21 heterogeneity-continuation use case:
 * one MG hierarchy rebuild per lambda value, reused across every warm-started
 * solve call at that lambda (eta-rescue ramps included).
 *
 * `reuse` REQUIRES `config.initial_state == PicardInitialState::warm_start`
 * and throws `std::invalid_argument` otherwise (enforced by
 * `solve_streamfunctions`, not by host-only `validate_streamfunction_problem`,
 * since it is a cross-call/workspace-history precondition, not a per-call
 * value precondition). Rationale: the `zero_source` initialization path
 * solves `A u_i = rhs_i` directly against `workspace.rhs1()`/`rhs2()` as its
 * RHS, so those buffers must hold a freshly assembled affine RHS for the
 * CURRENT `q`/gauge; running `zero_source` with `reuse` would silently solve
 * against a stale RHS from a previous `q`. Under `warm_start`, by contrast,
 * neither the zero-init nor the zero-source block solves run at all (see
 * `PicardInitialState`), and `rhs1()`/`rhs2()` are consumed only as the
 * SF-15 adaptive-Picard trial residual evaluator's OWN scratch OUTPUT buffers
 * (`enqueue_streamfunction_residual` fully overwrites them every trial before
 * anything reads them back) -- their pre-call contents are never read under
 * `warm_start`, so skipping their (re)assembly is safe exactly in that case.
 *
 * A `reuse` call also throws `std::invalid_argument` if the workspace has
 * never completed a `rebuild` call for its currently prepared grid (tracked
 * by `StreamfunctionWorkspace`'s internal `coefficients_valid_` guard, which
 * `prepare()` clears whenever the grid changes): there would be nothing valid
 * to reuse.
 */
enum class CoefficientState { rebuild, reuse };

/**
 * Composed, host-only configuration for one `solve_streamfunctions` call
 * (SF-13 consumes this; SF-12 only defines and validates it).
 *
 * This struct composes, rather than duplicates, every accepted sub-config:
 * `mg` (the accepted multigrid V-cycle stack), `linear` (SF-04 projected
 * PCG), `histogram` (SF-09/10 |c| histogram bin range), `diagnostics`
 * (SF-11 physical diagnostics thresholds), `picard` (SF-14 fixed-relaxation
 * Picard iteration limits). `eta` and `epsilon` are the
 * Lester equation (14) nonlinear-source parameters threaded into
 * `NonlinearSourceConfig::epsilon` and the residual combination weight;
 * defaults follow the plan's documented starting values: `eta = 1`
 * (unweighted nonlinear source coupling) and `epsilon = 1e-2` (the plan's
 * documented starting regularization value, see
 * `docs/plans/active/lester-eq14-streamfunction-solver-plan.md`).
 * `num_degeneracy_thresholds`/`degeneracy_thresholds` mirror
 * `NonlinearSourceConfig`'s fields exactly (same fixed-size array, same
 * semantics: purely diagnostic `|c|` degeneracy multipliers of `v_rms`) so
 * SF-13 can build a per-iteration `NonlinearSourceConfig` from this
 * configuration plus the runtime-measured `v_rms`.
 *
 * `v_rms` is deliberately absent: it is measured solver state (the current
 * Darcy or streamfunction-reconstructed RMS speed), not a caller-chosen
 * configuration value, and must never be cached here.
 */
struct StreamfunctionSolverConfig {
    multigrid::MGConfig mg{};
    solvers::ProjectedPCGConfig linear{};
    ResidualHistogramConfig histogram{};
    PhysicalDiagnosticsConfig diagnostics{};
    FixedPicardConfig picard{};
    AdaptivePicardConfig adaptive{};
    AndersonConfig anderson{};

    // SF-24: globalized Newton-Krylov phase, opt-in, bitwise-preserving when
    // disabled. See NewtonKrylovConfig above.
    NewtonKrylovConfig newton{};

    real eta{real{1}};
    real epsilon{real{1e-2}};
    int num_degeneracy_thresholds{0};
    real degeneracy_thresholds[kMaxDegeneracyThresholds]{};

    // SF-17: how u1/u2 are initialized before the Picard loop. Defaults to
    // `zero_source`, the bitwise-unchanged SF-13..16 behavior; see
    // `PicardInitialState` above.
    PicardInitialState initial_state{PicardInitialState::zero_source};

    // SF-21: whether q/the MG hierarchy/the affine RHS are (re)built or
    // reused from the workspace's prior contents. Defaults to `rebuild`, the
    // bitwise-unchanged SF-13..19 behavior; see `CoefficientState` above for
    // the full caller contract.
    CoefficientState coefficient_state{CoefficientState::rebuild};
};

/**
 * Exact-byte memory report shared by `estimate_streamfunction_memory` (pure
 * host prediction) and `StreamfunctionWorkspace::memory_report` (actual
 * capacities of a prepared workspace). Every `*_bytes` field is an exact sum
 * of owned `DeviceBuffer` capacities in bytes; nothing here is rounded or
 * fudged. `fine_grid_equivalent_fields = total_bytes / (n * sizeof(real))`
 * in double precision, where `n` is the fine grid's cell count, matching the
 * plan's ~24.6-field, ~3.1 GiB-at-256^3 budget language.
 *
 * Category split:
 *   - `fields_bytes`: `StreamfunctionFields` (`u1`, `u2`) only.
 *   - `solve_path_bytes`: everything `StreamfunctionWorkspace` allocates
 *     that participates in assembling and solving the linear subproblem and
 *     evaluating the coupled nonlinear residual (`q`, `rhs1`/`rhs2`,
 *     `f1`/`f2`, the SF-15 `u_trial1`/`u_trial2` backtracking trial pair,
 *     the `v_psi` CompactMAC scratch, the top-level affine-RHS
 *     workspace, the residual workspace, the projected-PCG workspace, the MG
 *     hierarchy, and the MG preconditioner).
 *   - `diagnostics_path_bytes`: the SF-11 physical-diagnostics workspace
 *     only (an optional, separately triggered evaluation).
 *   - `total_bytes = fields_bytes + solve_path_bytes + diagnostics_path_bytes`.
 *
 * The per-subworkspace breakdown fields duplicate parts of `solve_path_bytes`
 * / `diagnostics_path_bytes` by name for direct reporting; they always sum
 * back to their respective category total.
 *
 * SF-20 additive extension: `anderson_history_bytes`, `anderson_staging_bytes`,
 * and `anderson_scratch_bytes` are folded into `solve_path_bytes` alongside
 * the existing breakdown fields, and are exactly zero whenever
 * `config.anderson.enabled == false` -- the SF-12 closed-form memory tests'
 * disabled-config fixtures are therefore untouched (`solve_path_bytes` and
 * `total_bytes` are bitwise unchanged for those fixtures). When enabled,
 * `anderson_history_bytes == 4 * config.anderson.depth * n * sizeof(real)`
 * exactly (the `DeltaX1`/`DeltaX2`/`DeltaF1`/`DeltaF2` coupled history
 * columns); `anderson_staging_bytes == 4 * n * sizeof(real)` exactly (the
 * previous-accepted-state/previous-fixed-point-residual staging pair,
 * `prev_x1`/`prev_x2`/`prev_f1`/`prev_f2`); `anderson_scratch_bytes` is the
 * small dense Gram/rhs device transfer buffer plus the (unused-by-dot_device
 * but API-required) `blas::ReductionWorkspace` -- both independent of `n`.
 * See `AndersonAccelerator.cuh` for the exact accounting.
 *
 * SF-24 additive extension: `newton_jvp_bytes`, `newton_gmres_bytes`,
 * `newton_preconditioner_bytes`, and `newton_delta_bytes` are folded into
 * `solve_path_bytes` alongside the existing breakdown fields, and are exactly
 * zero whenever `config.newton.enabled == false` (same disabled-config
 * bitwise-preservation contract as the SF-20 Anderson fields above).  When
 * enabled: `newton_jvp_bytes == JvpWorkspace::estimate_device_bytes(n)`;
 * `newton_gmres_bytes == CoupledGmres::estimate_device_bytes(n,
 * config.newton.gmres.restart)`; `newton_preconditioner_bytes ==
 * BlockDiagonalMGPreconditioner::estimate_device_bytes(hierarchy)` (the
 * per-level `MeanZeroWorkspace` set plus the two finest-grid-sized scratch
 * buffers); `newton_delta_bytes == 2 * n * sizeof(real)` exactly (the two
 * dedicated Newton-step delta buffers,`newton_delta1()`/`newton_delta2()`,
 * distinct from the preconditioner's own internal scratch). See
 * `StreamfunctionWorkspace.cuh` for the exact optional-sub-workspace
 * lifecycle these bytes are colocated with.
 */
struct StreamfunctionMemoryReport {
    std::size_t fields_bytes{};
    std::size_t solve_path_bytes{};
    std::size_t diagnostics_path_bytes{};
    std::size_t total_bytes{};
    double fine_grid_equivalent_fields{};

    // solve_path_bytes breakdown.
    std::size_t scratch_fields_bytes{}; // q, rhs1, rhs2, f1, f2, u_trial1, u_trial2, v_psi (U/V/W)
    std::size_t residual_workspace_bytes{};
    std::size_t affine_rhs_workspace_bytes{}; // top-level workspace instance only
    std::size_t pcg_workspace_bytes{};
    std::size_t mg_hierarchy_bytes{};
    std::size_t mg_preconditioner_bytes{};

    // diagnostics_path_bytes breakdown (currently a single sub-workspace).
    std::size_t diagnostics_workspace_bytes{};

    // SF-20 Anderson acceleration: additive, zero when disabled (see above).
    std::size_t anderson_history_bytes{};
    std::size_t anderson_staging_bytes{};
    std::size_t anderson_scratch_bytes{};

    // SF-24 Newton-Krylov: additive, zero when disabled (see above).
    std::size_t newton_jvp_bytes{};
    std::size_t newton_gmres_bytes{};
    std::size_t newton_preconditioner_bytes{};
    std::size_t newton_delta_bytes{};
};

/**
 * Host-only validation of one problem/config pair against a grid, throwing
 * `std::invalid_argument` with a distinct message per violated precondition:
 *
 *   - `problem.conductivity.size() != grid.num_cells()`;
 *   - a CompactMAC `darcy_velocity` component with the wrong exact size
 *     (`(nx+1)*ny*nz`, `nx*(ny+1)*nz`, `nx*ny*(nz+1)`);
 *   - any non-`Periodic` face in `problem.bc` (the v1 benchmark is triply
 *     periodic only; see the increment spec);
 *   - non-finite or non-positive `grid.dx`/`dy`/`dz`;
 *   - anisotropic spacing (`dx == dy == dz` is required, inherited from the
 *     SF-02/SF-06/SF-10 evaluator chain; this restriction is NOT weakened by
 *     the SF-11 diagnostics module, which is anisotropy-capable on its own
 *     but is still driven through this solver-level, isotropic-only gate);
 *   - any grid extent `< 2`;
 *   - `problem.grid` not exactly matching the `grid` argument (extents and
 *     spacings all equal; see `StreamfunctionProblemView::grid`);
 *   - a non-finite affine gauge gradient component;
 *   - a grid that cannot support `config.mg.num_levels` under the exact
 *     `multigrid::MGHierarchy` coarsening/break rule together with
 *     `multigrid::validate_projected_positive_hierarchy`'s per-level
 *     even-extent and exact 2x geometric-doubling requirements;
 *   - invalid `config` values: `eta`/`epsilon` must be finite and `>= 0`;
 *     `num_degeneracy_thresholds` must be in `[0, kMaxDegeneracyThresholds]`
 *     and every used threshold must be finite and `>= 0` (mirroring
 *     `NonlinearSourceConfig`); `config.histogram` must satisfy
 *     `ResidualHistogramConfig`'s own `0 < c_min_rel < c_max_rel` rule;
 *     `config.diagnostics` must satisfy `PhysicalDiagnosticsConfig`'s own
 *     rules (finite strictly positive `angle_exclusion_rel`/`low_speed_rel`,
 *     a valid threshold count, and a finite strictly positive strictly
 *     increasing threshold list); `config.linear` must have
 *     `max_iter >= 0`, `check_every > 0`, and finite `rtol >= 0`;
 *     `config.mg` must have `pre_smooth == post_smooth >= 1` and
 *     `coarse_solve_iters > 0` (the exact rule enforced downstream by
 *     `multigrid::validate_projected_positive_hierarchy`); `config.picard`
 *     must have `max_iter >= 0`, a finite `tolerance > 0`, and a finite
 *     `omega` in `(0, 1]`; `config.adaptive` (SF-15) is validated regardless
 *     of `enabled` and requires: finite `omega_min` in
 *     `(0, config.picard.omega]`; finite `backtrack_factor` in `(0, 1)`;
 *     finite `growth_factor >= 1`; finite `omega_max` in
 *     `[config.picard.omega, 1]`; `easy_streak >= 1`; finite `armijo_c` in
 *     `[0, 1)`; `stagnation_window >= 1`; finite `stagnation_min_reduction`
 *     in `(0, 1)`; finite `max_unexplained_fraction` in `(0, 1]`; finite
 *     `unexplained_growth_factor >= 1`; finite `unexplained_growth_offset
 *     >= 0`; finite `percentile_collapse_factor > 1`; nested
 *     `adaptive.floor_guard` (SF-25) is likewise validated regardless of
 *     `adaptive.enabled`/`adaptive.floor_guard.enabled` and requires:
 *     `window >= 1`; finite `drop_factor` in `(0, 1)`; `max_resets >= 0`;
 *     `config.anderson`
 *     (SF-20) is likewise validated regardless of `enabled` and requires:
 *     `depth` in `[3, 8]`; `start_iteration >= 1`; finite
 *     `condition_limit > 1`; `max_restarts >= 0` (SF-25, regardless of
 *     `restart_on_stagnation`); `config.newton` (SF-24) is likewise validated
 *     regardless of `enabled`, and additionally (SF-24 C01) rejects
 *     `newton.enabled && !adaptive.enabled` as its FIRST check (the Newton
 *     phase is integrated only into the adaptive Picard loop; see
 *     `NewtonKrylovConfig::enabled` above), then requires: finite
 *     `activation_r_F > 0`;
 *     finite `stagnation_activation_r_F >= activation_r_F`; finite
 *     `0 < forcing_min <= forcing_max <= 1`; finite `forcing_coefficient > 0`;
 *     finite `armijo_c` in `[0, 1)`; finite `alpha_min` in `(0, 1]`; finite
 *     `backtrack_factor` in `(0, 1)`; `max_newton_iterations >= 1`;
 *     `rescue_picard_steps >= 0`; nested `gmres` valid per
 *     `validate_coupled_gmres_config`; nested `delta` valid per
 *     `validate_jvp_delta_config`.
 *
 * Device-resident values of `K`/`Y` are intentionally NOT reduced or
 * validated here, matching the SF-06 wording that finiteness/positivity of
 * device field contents is a precondition checked by the kernels that
 * consume them, not by host-side validation.
 */
void validate_streamfunction_problem(const Grid3D& grid, const StreamfunctionProblemView& problem,
                                      const StreamfunctionSolverConfig& config);

} // namespace streamfunctions
} // namespace macroflow3d
