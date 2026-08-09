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

    real eta{real{1}};
    real epsilon{real{1e-2}};
    int num_degeneracy_thresholds{0};
    real degeneracy_thresholds[kMaxDegeneracyThresholds]{};

    // SF-17: how u1/u2 are initialized before the Picard loop. Defaults to
    // `zero_source`, the bitwise-unchanged SF-13..16 behavior; see
    // `PicardInitialState` above.
    PicardInitialState initial_state{PicardInitialState::zero_source};
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
 *     >= 0`; finite `percentile_collapse_factor > 1`.
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
