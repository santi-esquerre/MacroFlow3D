#pragma once

/**
 * @file StreamfunctionSolver.cuh
 * @brief Public entry point declaration for the Lester equation (14)
 *        streamfunction solver (SF-12: declaration only).
 *
 * IMPORTANT (SF-12 scope lock): this header declares `StreamfunctionSolveReport`
 * and `solve_streamfunctions` but intentionally provides NO definition. The
 * `solve_streamfunctions` body (a `StreamfunctionSolver.cu` translation unit)
 * is SF-13's explicit deliverable: "SF-13 may implement the first complete
 * homogeneous solve through this API" (see
 * `docs/plans/active/lester-eq14/increments/SF-12-public-api-workspace.md`).
 * No other SF-12 file calls this declaration; it exists only to fix the
 * public function signature promised by
 * `docs/plans/active/lester-eq14-streamfunction-solver-plan.md`'s
 * "Architecture and memory constraints" section ahead of SF-13. Do not add a
 * stub or default-constructing definition here or anywhere else before
 * SF-13.
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
 * DECLARATION ONLY in SF-12 (see the file header). No translation unit in
 * this increment defines or calls this function.
 */
[[nodiscard]] StreamfunctionSolveReport solve_streamfunctions(
    CudaContext& context, const StreamfunctionProblemView& problem,
    const StreamfunctionSolverConfig& config, StreamfunctionFields& fields,
    StreamfunctionWorkspace& workspace);

} // namespace streamfunctions
} // namespace macroflow3d
