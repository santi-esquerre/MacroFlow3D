#pragma once

/**
 * @file StreamfunctionSolver.cuh
 * @brief Public entry point declaration for the Lester equation (14)
 *        streamfunction solver.
 *
 * SF-12 declared this API without a definition. SF-13 (`StreamfunctionSolver.cu`)
 * provided the first `solve_streamfunctions` body: a v1 ZERO-SOURCE
 * (harmonic-coordinate) linear solve. SF-14 extends that body with a
 * fixed-relaxation Picard outer loop over the coupled nonlinear sources.
 *
 * v1 zero-source initialization (SF-13, unchanged):
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
 * SF-14 fixed-relaxation Picard outer loop (see `StreamfunctionSolver.cu`):
 *   - The nonlinear system solved is `A u1 = P(rhs_affine1 - eta*q*S2)`,
 *     `A u2 = P(rhs_affine2 - eta*q*S1)` (pairing F1<->S2, F2<->S1), i.e. the
 *     full coupled Lester equation (14) system in the periodic fluctuations.
 *   - At the head of every iteration `k = 0, 1, ...`, the coupled residual
 *     `F1`, `F2` is evaluated ONCE from the current, immutable state
 *     (`enqueue_streamfunction_residual`), producing BOTH block right-hand
 *     sides `G1`, `G2` in the same enqueue. If `r_F <= config.picard.tolerance`,
 *     the loop stops with `status == converged`. Otherwise, if
 *     `k == config.picard.max_iter`, the loop stops with
 *     `status == not_converged` (iteration budget exhausted).
 *   - Otherwise, the two linear blocks are solved sequentially, at the SAME
 *     frozen state k, against the SAME operator/MG hierarchy/PCG workspace:
 *     `u_hat1` solves `A u_hat1 = G1`, `u_hat2` solves `A u_hat2 = G2`. No
 *     residual re-evaluation occurs between the two block solves, so both
 *     sources are guaranteed to come from one immutable state. If either
 *     block solve fails to converge, the loop stops immediately with
 *     `status == not_converged` (the failing pair is still recorded in
 *     `picard_history`).
 *   - Both fields are then updated as a PAIR with the SAME fixed relaxation
 *     factor `config.picard.omega`: `u_i <- (1-omega)*u_i + omega*u_hat_i`,
 *     then mean-zero projected. This is plain fixed-point (Picard) iteration:
 *     no step rejection, no adaptive relaxation, no hidden damping, and no
 *     continuation. Those are explicitly out of scope for SF-14.
 *   - `report.picard_history[k]` holds the residual reductions
 *     `(r_F, r1, r2)` evaluated AT state k, together with the linear block
 *     results that PRODUCED state k (i.e. entry k's `psi1_result`/
 *     `psi2_result` are the solves performed during the transition from
 *     state k-1 to state k). Entry 0 therefore holds default-constructed
 *     `ProjectedPCGResult`s: state 0 is the zero-source initialization
 *     above, whose own `ProjectedPCGResult`s remain visible at the top level
 *     (`report.psi1_result`/`psi2_result`) until the first Picard update
 *     overwrites them. `report.picard_iterations` is the number of
 *     completed update steps (0 when the loop converges or exhausts its
 *     budget at k = 0, e.g. the homogeneous K=1 control, which converges
 *     with `picard_iterations == 0` and `r_F == 0`).
 *   - the gauge is maintained throughout by the projected PCG and by the
 *     explicit post-update mean-zero projection; `psi*_result.final_field_mean`
 *     is the gauge evidence for the block solve that produced each state.
 *   - one call synchronizes the host only in the PCG reductions and the
 *     repeated residual/diagnostics report syntheses; no allocation occurs
 *     anywhere in the Picard loop (no new device memory is introduced by
 *     SF-14; `f1`/`f2` are reused as scratch for `u_hat1`/`u_hat2`).
 *
 * `status` semantics (rewritten for SF-14, replacing the SF-13 wording):
 *   - `converged`: the coupled nonlinear residual reached
 *     `r_F <= config.picard.tolerance` at the head of some iteration k.
 *   - `not_converged`: either the Picard iteration budget
 *     (`config.picard.max_iter`) was exhausted without reaching tolerance,
 *     or a linear block solve failed to converge during some update step.
 *     `picard_history` distinguishes the two: iteration-budget exhaustion
 *     leaves `picard_history.size() == max_iter + 1` with every recorded
 *     block solve converged; a linear failure truncates the history early
 *     at the failing entry.
 *   - `invalid_problem`: unchanged from SF-13 (degenerate measured `v_rms`).
 */

#include "../../numerics/solvers/pcg.cuh"
#include "../../runtime/CudaContext.cuh"
#include "Diagnostics.cuh"
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
 * One recorded Picard iteration (SF-14): the coupled residual reductions
 * evaluated AT state k, together with the linear block results that
 * PRODUCED state k. See the file header for the exact layout convention
 * (entry 0 carries default-constructed `ProjectedPCGResult`s because state 0
 * is the SF-13 zero-source initialization, not a Picard update).
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
 * `picard_iterations` (SF-14 fixed-relaxation Picard loop record), and
 * `memory` (the SF-12 exact-byte memory report for the workspace that
 * produced this solve). `status` is the overall outcome; see
 * `StreamfunctionSolveStatus`.
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
};

/**
 * Solve the Lester equation (14) streamfunction system for one problem
 * instance, writing the accepted invariants into `fields` and using `workspace`
 * for every scratch/solver buffer (see `StreamfunctionWorkspace.cuh`).
 *
 * SF-14 iterates the full coupled nonlinear system via fixed-relaxation
 * Picard; see the file header for the exact solved system, convergence
 * semantics, and the `invalid_problem` condition. Throws
 * `std::invalid_argument` (via `validate_streamfunction_problem`) for
 * host-detectable problem/config misuse.
 */
[[nodiscard]] StreamfunctionSolveReport solve_streamfunctions(
    CudaContext& context, const StreamfunctionProblemView& problem,
    const StreamfunctionSolverConfig& config, StreamfunctionFields& fields,
    StreamfunctionWorkspace& workspace);

} // namespace streamfunctions
} // namespace macroflow3d
