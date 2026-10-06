#pragma once

/**
 * @file SlabSolverTypes.cuh
 * @brief SF-33 N2: configurations, status enums and reports of the inlet-slab nonlinear solver
 *        (SlabGmres, SlabModePreconditioner, SlabNewtonKrylov).
 *
 * Statuses are DISTINCT values; none is ever mapped to another (spec item 10):
 *   linear (SlabLinearStatus):   converged | max_iterations | stagnation | breakdown | nonfinite
 *   nonlinear (SlabSolveStatus): converged | linesearch-fail | stagnation | maxit | linear_failure
 * | nan_inf | continuation_floor Defaults are the SF-29 prototype's (`candidate_i.py`): Newton tol
 * 1e-13, maxit 40, Armijo 1e-4, lambda >= 1/1024, stagnation = merit not below half the merit 4
 * iterations earlier; continuation ladder (0.25, 0.5, 1.0), STAGE_OK = 1e-9, at most 4 bisections.
 * GMRES defaults of the SF-33 N2 task: lin_tol 1e-12 (true relative residual), restart 50, inner
 * cap 6000.
 *
 * Linear forcing (SF-33 N7a): `forcing = ew` (default) is inexact Newton with Eisenstat-Walker
 * choice 2 forcing terms eta_k (the GMRES relative tolerance of Newton step k, see
 * SlabNewtonKrylov.cuh); `forcing = fixed` solves every Newton system to gmres.tol (lin_tol), the
 * N2-N6 behaviour, bitwise.
 *
 * Pseudo-transient continuation (SF-33 N7b): `psitc.enabled` solves every Newton system shifted,
 * (J + mu_k D) p = -E, with switched evolution relaxation (SER) for mu_k; see SlabNewtonKrylov.cuh.
 * The LIBRARY default is `enabled = false` (mu = 0: the N7a behaviour, bitwise); the production
 * policy (Psi-tc on, max_newton 120) is set by the driver (apps/inlet_slab).
 */

#include "../../../core/Scalar.hpp"
#include "InletSlabGrid.cuh"

#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

// ------------------------------------------------------------------------------------------------
// Statuses
// ------------------------------------------------------------------------------------------------

enum class SlabLinearStatus {
    not_run,
    converged,
    max_iterations,
    stagnation,
    breakdown,
    nonfinite
};

inline const char* to_string(SlabLinearStatus s) {
    switch (s) {
    case SlabLinearStatus::not_run:
        return "not_run";
    case SlabLinearStatus::converged:
        return "converged";
    case SlabLinearStatus::max_iterations:
        return "max_iterations";
    case SlabLinearStatus::stagnation:
        return "stagnation";
    case SlabLinearStatus::breakdown:
        return "breakdown";
    case SlabLinearStatus::nonfinite:
        return "nonfinite";
    }
    return "unknown";
}

enum class SlabSolveStatus {
    not_run,
    converged,
    linesearch_fail,
    stagnation,
    maxit,
    linear_failure,
    nan_inf,
    continuation_floor
};

/// Strings of the prototype where it has one (`linesearch-fail`), the SF-33 names otherwise.
inline const char* to_string(SlabSolveStatus s) {
    switch (s) {
    case SlabSolveStatus::not_run:
        return "not_run";
    case SlabSolveStatus::converged:
        return "converged";
    case SlabSolveStatus::linesearch_fail:
        return "linesearch-fail";
    case SlabSolveStatus::stagnation:
        return "stagnation";
    case SlabSolveStatus::maxit:
        return "maxit";
    case SlabSolveStatus::linear_failure:
        return "linear_failure";
    case SlabSolveStatus::nan_inf:
        return "nan_inf";
    case SlabSolveStatus::continuation_floor:
        return "continuation_floor";
    }
    return "unknown";
}

// ------------------------------------------------------------------------------------------------
// Configurations
// ------------------------------------------------------------------------------------------------

/// Linear forcing policy of the Newton iteration (SF-33 N7a).
enum class SlabForcing {
    fixed, ///< every Newton system solved to gmres.tol (lin_tol): exact-Newton surrogate (N2-N6)
    ew     ///< inexact Newton, Eisenstat-Walker choice 2 forcing terms (default)
};

inline const char* to_string(SlabForcing f) {
    switch (f) {
    case SlabForcing::fixed:
        return "fixed";
    case SlabForcing::ew:
        return "ew";
    }
    return "unknown";
}

/// Eisenstat-Walker (1996) choice 2 parameters. The floor eta_min is the GMRES tolerance of the
/// Newton config (gmres.tol = lin_tol, 1e-12): no forcing term is ever tighter than the fixed mode.
struct SlabEwConfig {
    real gamma = 0.9;
    real alpha = 2.0;
    real eta0 = 0.1;    ///< forcing term of the first Newton step of every solve() call
    real eta_max = 0.1; ///< cap (applied before the floor and the oversolving guard)
};

/// Pseudo-transient continuation with switched evolution relaxation (SF-33 N7b; see
/// SlabNewtonKrylov.cuh for the formulas). Disabled: mu = 0 exactly (no shift kernel, no
/// preconditioner shift, unchanged log lines).
struct SlabPsitcConfig {
    bool enabled = false;
    real mu0 = 1.0;      ///< mu of the first Newton step of every solve() call
    real mu_max = 100.0; ///< clamp of every mu (SER value and line-search retries)
    int max_retries = 4; ///< line-search failure: mu *= retry_factor, re-solve, at most this often
    real retry_factor = 4.0;
};

/// SF-33 N7c (probe): Galerkin coarse-space correction on the x1-constant / x1-linear column
/// subspace combined with P-A (SlabCoarseCorrection.cuh). off = P-A alone (the N7b behaviour,
/// bitwise); add = P_A^-1 + V E^-1 V^T; mult = multiplicative (one extra operator application).
enum class SlabCoarseMode { off, add, mult };

inline const char* to_string(SlabCoarseMode m) {
    switch (m) {
    case SlabCoarseMode::off:
        return "off";
    case SlabCoarseMode::add:
        return "add";
    case SlabCoarseMode::mult:
        return "mult";
    }
    return "unknown";
}

struct SlabGmresConfig {
    real tol = 1e-12;          ///< stop when the TRUE relative residual ||b - A x|| / ||b|| <= tol
    int restart = 50;          ///< Krylov basis size m (must be <= the prepared restart)
    int max_iterations = 6000; ///< total inner-iteration cap over all cycles
    /// A cycle ends early when the Givens recurrence estimate |g_{k+1}| / ||b|| drops below
    /// inner_tol_factor * tol (the prototype's 0.1 tol); the stop decision is always the true
    /// residual recomputed at the end of the cycle.
    real inner_tol_factor = 0.1;
    /// Restart stagnation (prototype rule): stop when the true residual at the end of a cycle is
    /// > stagnation_factor x the true residual two cycles earlier.
    real stagnation_factor = 0.9;
    /// MGS re-orthogonalization: one extra pass when ||w_after|| < reorth_threshold ||w_before||.
    real reorth_threshold = 0.70710678118654752440; // 1/sqrt(2)
};

struct SlabNewtonConfig {
    real tol = 1e-13;        ///< converged iff r_F <= tol and r_out <= tol
    int max_iterations = 40; ///< Newton iterations per call (status maxit)
    real armijo = 1e-4;      ///< accept iff m_new < (1 - armijo lambda) m (and finite)
    real lambda_min = 1.0 / 1024.0;
    int stagnation_window = 5;    ///< history entries compared (prototype: last 5 iterates)
    real stagnation_factor = 0.5; ///< stagnation iff merit_last > factor * merit_{last-4}
    SlabGmresConfig gmres;
    std::string prec_name = "P-A";         ///< printed in the LINEAR line as `gmres+<prec_name>`
    SlabForcing forcing = SlabForcing::ew; ///< linear forcing policy (SF-33 N7a)
    SlabEwConfig ew;                       ///< used only when forcing == ew
    SlabPsitcConfig psitc;                 ///< pseudo-transient continuation (SF-33 N7b)
    /// SF-33 N7c (probe): coarse correction on top of P-A; rebuilt with P-A at every factor()
    /// (same base, same mu). Requires SlabNewtonKrylov::prepare_coarse. Default off (bitwise N7b).
    SlabCoarseMode coarse = SlabCoarseMode::off;
};

struct SlabContinuationConfig {
    std::vector<real> ladder = {0.25, 0.5, 1.0}; ///< EPS_LADDER of the prototype
    real stage_ok = 1e-9; ///< a non-converged stage is accepted iff its merit <= stage_ok
    int max_bisections = 4;
    std::string field = "field"; ///< printed in STAGE/STAGE_END/PATH lines and Newton labels
    std::string cand = "i1o4";   ///< candidate name of the prototype's lines
};

// ------------------------------------------------------------------------------------------------
// Reports
// ------------------------------------------------------------------------------------------------

struct SlabGmresReport {
    SlabLinearStatus status = SlabLinearStatus::not_run;
    int iterations = 0; ///< total inner iterations (applications of A M^-1)
    int cycles = 0;     ///< restart cycles (restarts = cycles - 1)
    int reorthogonalizations = 0;
    real b_norm = 0.0;
    real rel_residual = 0.0;      ///< TRUE relative residual at exit ||b - A x|| / ||b||
    real rel_recurrence = 0.0;    ///< Givens estimate |g| / ||b|| at exit (end of the last cycle)
    std::vector<real> cycle_true; ///< true relative residual at the end of every cycle
    std::vector<real> cycle_recurrence; ///< recurrence estimate at the end of every cycle
    double seconds = 0.0;
};

struct SlabPrecFactorReport {
    int singular_modes = 0; ///< modes with an exactly zero or non-finite pivot (never hidden)
    double seconds = 0.0;
};

struct SlabNewtonStepRecord {
    real r_F = 0.0, r_out = 0.0, lambda = 0.0, dx_max = 0.0, dx_l2 = 0.0;
    real eta = 0.0; ///< GMRES relative tolerance used by this step (forcing term)
    real mu = 0.0;  ///< pseudo-time shift of the solve whose direction was used (or the last tried)
    real mu_ser = 0.0;             ///< SER value of this step (before any line-search retry)
    std::vector<real> mu_retries;  ///< mu of every line-search retry (empty: none)
    int linear_iterations_all = 0; ///< GMRES iterations of every solve of this step (retries incl.)
    std::vector<int> linear_its_solves; ///< GMRES iterations per linear solve of this step
    SlabGmresReport linear;             ///< the last linear solve of this step
    double t_lin = 0.0, t_fact = 0.0;
    int prec_singular_modes = 0;
};

struct SlabNewtonReport {
    SlabSolveStatus status = SlabSolveStatus::not_run;
    SlabForcing forcing = SlabForcing::fixed; ///< forcing policy of this solve() call
    bool psitc = false;                       ///< pseudo-transient continuation on (SF-33 N7b)
    int linesearch_retries = 0;               ///< total Psi-tc line-search retries (all steps)
    int its = 0;
    real r_F = 0.0, r_out = 0.0;
    std::vector<real> hist_r_F; ///< entry 0 = start state, then one per accepted step
    std::vector<real> hist_r_out;
    std::vector<SlabNewtonStepRecord> steps; ///< one per Newton step that ran a linear solve
    int linear_iterations_total = 0;
    int linear_iterations_max = 0;
    SlabLinearStatus last_linear_status = SlabLinearStatus::not_run;
    double seconds = 0.0;
    real merit() const { return std::sqrt(r_F * r_F + r_out * r_out); }
};

struct SlabStageRecord {
    real eps = 0.0;
    real from_eps = 0.0;
    bool warm_start = false;
    bool final_attempt = false;
    bool accepted = false;
    SlabNewtonReport newton;
};

struct SlabContinuationReport {
    SlabSolveStatus status = SlabSolveStatus::not_run;
    std::string path; ///< the PATH string, e.g. "0.25(fail)->0.125->0.25->0.5"
    std::vector<SlabStageRecord> stages;
    int bisections = 0;
    bool floor_reached = false;
    real eps_accepted = 0.0;       ///< last accepted amplitude (0 if none)
    SlabNewtonReport final_newton; ///< report of the last Newton call (target or final attempt)
    double seconds = 0.0;
};

// ------------------------------------------------------------------------------------------------
// Callbacks
// ------------------------------------------------------------------------------------------------

/// One log line (no trailing newline). The default logger prints to stdout and flushes.
using SlabLogger = std::function<void(const std::string&)>;

/// Stage inputs for an amplitude of the continuation path. The returned object must stay valid and
/// unchanged until the provider is called again (the solver references it during the stage).
using StageInputProvider = std::function<const SlabStageInputs&(real amplitude)>;

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
