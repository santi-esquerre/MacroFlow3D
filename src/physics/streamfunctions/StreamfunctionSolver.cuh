#pragma once

/**
 * @file StreamfunctionSolver.cuh
 * @brief Public entry point declaration for the Lester equation (14)
 *        streamfunction solver.
 *
 * SF-12 declared this API without a definition. SF-13 (`StreamfunctionSolver.cu`)
 * now provides the first `solve_streamfunctions` body: a v1 ZERO-SOURCE
 * (harmonic-coordinate) linear solve.
 *
 * v1 semantics (SF-13):
 *   - `u1`, `u2` are zero-initialized on every call (no warm start; a
 *     warm-start policy is SF-14 scope).
 *   - `q = 1/K` or `q = exp(-Y)` is filled from `problem.conductivity` per
 *     `problem.conductivity_representation`.
 *   - the affine periodic RHS pair is assembled and mean-zero-projected
 *     (`assemble_affine_periodic_rhs`), then each linear block
 *     `A psi_i = rhs_i` is solved ONCE via the accepted projected-PCG/MG
 *     stack, sequentially: psi1 first, then psi2, sharing the workspace's
 *     single hierarchy/PCG workspace. This solves the ZERO-SOURCE
 *     (eta-independent) linear problem only; it does NOT iterate the coupled
 *     nonlinear system `A psi1 = -q S2`, `A psi2 = -q S1`. Picard iteration
 *     over the nonlinear sources is SF-14 scope.
 *   - `status == converged` reflects only whether both linear block solves
 *     converged (`psi1_result.converged && psi2_result.converged`).
 *     `report.residual.r_F` is the honest coupled nonlinear residual of the
 *     zero-source solution against the full Lester equation (14) system; it
 *     is `~= 0` only when the zero-source solution actually solves the full
 *     system (e.g. the homogeneous K=1 control), and is NOT itself part of
 *     `status`.
 *   - `status == invalid_problem` is returned (after the linear solves and
 *     SF-11 diagnostics, before the residual evaluation) when the measured
 *     Darcy `v_rms` (`report.diagnostics.v_d_rms`) is non-finite or
 *     non-positive: the SF-09 nonlinear-source contract and the `r1`
 *     residual normalization both require a strictly positive `v_rms`.
 *   - the gauge is maintained by the projected PCG (mean-zero rhs/iterates);
 *     `psi*_result.final_field_mean` is the gauge evidence.
 *   - one call synchronizes the host only in the PCG reductions and the two
 *     report syntheses (`synchronize_streamfunction_physical_diagnostics_report`,
 *     `synchronize_streamfunction_residual_report`); no allocation occurs
 *     after `fields.prepare`/`workspace.prepare` within a call.
 */

#include "../../numerics/solvers/pcg.cuh"
#include "../../runtime/CudaContext.cuh"
#include "Diagnostics.cuh"
#include "ResidualEvaluator.cuh"
#include "StreamfunctionTypes.hpp"
#include "StreamfunctionWorkspace.cuh"

namespace macroflow3d {
namespace streamfunctions {

/**
 * Minimal, SF-12-scoped solve status. Kept deliberately small; SF-13/14 are
 * expected to extend it with continuation- and Picard-iteration-specific
 * states as the nonlinear strategy is implemented.
 */
enum class StreamfunctionSolveStatus {
    not_run,
    converged,
    not_converged,
    invalid_problem,
};

/**
 * Host-assembled report for one `solve_streamfunctions` call, composing the
 * accepted per-primitive reports rather than duplicating their fields:
 * `residual` (SF-09/10 coupled residual and its dimensionless reductions),
 * `diagnostics` (SF-11 physical/Gate-3A diagnostics), `psi1_result`/
 * `psi2_result` (the SF-04 projected-PCG result for each sequential linear
 * block solve), and `memory` (the SF-12 exact-byte memory report for the
 * workspace that produced this solve). `status` is the SF-12-scoped overall
 * outcome; see `StreamfunctionSolveStatus`.
 */
struct StreamfunctionSolveReport {
    StreamfunctionSolveStatus status{StreamfunctionSolveStatus::not_run};

    StreamfunctionResidualReport residual{};
    PhysicalDiagnosticsReport diagnostics{};
    solvers::ProjectedPCGResult psi1_result{};
    solvers::ProjectedPCGResult psi2_result{};
    StreamfunctionMemoryReport memory{};
};

/**
 * Solve the Lester equation (14) streamfunction system for one problem
 * instance, writing the accepted invariants into `fields` and using `workspace`
 * for every scratch/solver buffer (see `StreamfunctionWorkspace.cuh`).
 *
 * v1 (SF-13) solves the zero-source linear problem only; see the file header
 * for the exact solved system, convergence semantics, and the
 * `invalid_problem` condition. Throws `std::invalid_argument` (via
 * `validate_streamfunction_problem`) for host-detectable problem/config
 * misuse.
 */
[[nodiscard]] StreamfunctionSolveReport solve_streamfunctions(
    CudaContext& context, const StreamfunctionProblemView& problem,
    const StreamfunctionSolverConfig& config, StreamfunctionFields& fields,
    StreamfunctionWorkspace& workspace);

} // namespace streamfunctions
} // namespace macroflow3d
