#pragma once

/**
 * @file SlabNewtonKrylov.cuh
 * @brief SF-33 N2: damped Newton-Krylov with backtracking line search and amplitude continuation
 *        with bisection for the inlet-slab equation (14) (the SF-29 prototype's `newton` and
 *        `solve_case`, `candidate_i.py`, with the exact linear solve replaced by GMRES + P-A).
 *
 * Newton (solve)
 * --------------
 * Unknown x = [u1 | u2] (2 N^3, planes 1..N); the residual E(x) and the norms (r_F, r_out) are
 * those of N0 (`evaluate_residual` on the full planes assembled with the stage's u0 on plane 0);
 * merit m = sqrt(r_F^2 + r_out^2). Iteration it = 1..max_iterations: if r_F <= tol and r_out <=
 * tol: converged (checked before every step and after the loop); freeze the base (N1
 * `prepare_base`), factor P-A (SlabModePreconditioner), solve J dx = -E with GMRES (true relative
 * residual <= eta_k, the forcing term below); one `LINEAR` line per solve; if GMRES is not
 * `converged` (i.e. eta_k not reached within the restart / inner caps) (or P-A has
 * singular modes): STOP with `linear_failure` (the step is never taken; the linear status is
 * recorded in the report); backtracking lambda = 1, 1/2, ..., lambda_min (1/1024): accept iff the
 * trial merit is finite and m_new < (1 - 1e-4 lambda) m; none accepted: `linesearch-fail` (state
 * unchanged); accept, log the `NEWTON` line (prototype format), append (r_F, r_out) to the history;
 *   `stagnation` when >= 5 history entries exist and merit_last > 0.5 merit_{last-4} (and not
 *   converged).
 * After the loop: `maxit` unless converged. A non-finite merit of the START state gives `nan_inf`
 * (no step is attempted). GMRES non-finite results are reported as `linear_failure` with the linear
 * status `nonfinite`.
 *
 * Linear forcing (SF-33 N7a; cfg.forcing, every NEWTON line after it=0 prints `eta=`)
 * -----------------------------------------------------------------------------------
 *   fixed: eta_k = gmres.tol (lin_tol) for every step: the N2-N6 behaviour, bitwise (same code
 *          path, the GMRES config is a copy whose tol is that same value).
 *   ew (default): inexact Newton, Eisenstat-Walker (1996) choice 2. The norm is the MERIT
 *          ||F_k|| = m_k = sqrt(r_F^2 + r_out^2) at the start of step k (the line-search merit; k
 *          is 0-based per solve() call, i.e. it restarts at every continuation stage):
 *            eta_0 = ew.eta0;  k >= 1: eta_k = gamma (m_k / m_{k-1})^alpha, and if
 *            gamma eta_{k-1}^alpha > 0.1 then eta_k = max(eta_k, gamma eta_{k-1}^alpha);
 *            then eta_k = min(eta_k, ew.eta_max); eta_k = max(eta_k, gmres.tol) (eta_min =
 *            lin_tol); oversolving guard eta_k = max(eta_k, 0.5 tol / m_k) (tol = Newton tol: the
 *            last solves are not tighter than needed to bring r_F, r_out <= tol).
 *          Defaults gamma 0.9, alpha 2, eta0 = eta_max = 0.1. The GMRES stop is still the TRUE
 *          relative residual ||J dx + E||_2 / ||E||_2 <= eta_k (raw Euclidean norm of the residual
 *          vector; the merit ratio is used only to choose eta_k). Not reaching eta_k is
 *          `linear_failure`, exactly as in the fixed mode: an unconverged step is never taken.
 *          The Armijo line search and the stagnation / maxit / nan_inf rules are unchanged.
 *   The converged discrete state (r_F, r_out <= tol) does not depend on the policy; the iterate
 *   history does.
 *
 * Pseudo-transient continuation (SF-33 N7b; cfg.psitc, library default OFF)
 * -------------------------------------------------------------------------
 * Enabled: Newton step k solves the SHIFTED system (J(x_k) + mu_k D) p = -E(x_k) with
 *   D = diag(q_v / h^2) on the equation rows (planes 1..N-1, both fields), D = 0 on the outlet
 *       rows (plane N: exact linear constraints of the oblique condition, never relaxed);
 *   mu_k = clamp(mu0_eff * m_k / m_0, 0, mu_max)  (switched evolution relaxation, SER), with the
 *       MERIT norm m = sqrt(r_F^2 + r_out^2) (the line-search / Eisenstat-Walker norm) and m_0 the
 *       merit of the START state of this solve() call (i.e. of the stage's start state);
 *   mu0_eff = mu0 (h / h_ref)^2 (SF-33 C4, psitc_effective_mu0; default h_ref = 1/16, so N = 16
 *       is bitwise the N7b schedule and N = 32/64/128 use 1/4, 1/16, 1/64 of mu0; h_ref = 0: no
 *       scaling; the absolute shift mu0_eff D = mu0 q_v / h_ref^2 is grid independent). Rationale: with D = q_v / h^2 the weakly determined family has eigenvalues
 *       ~h^2, and a grid-independent mu kept it damped until mu <~ h^2 (N8' at 128^3: stagnation
 *       rule before the slow phase ended). mu_max / retry_factor are not scaled. mu0_eff is
 *       printed on the STAGE lines of solve_with_continuation (` psitc_mu0_eff=`);
 *   the operator applied matrix-free as J p + mu (D .* p) (N1 JVP + one kernel,
 *       slab_add_pseudo_time_shift; no allocation) and P-A factored for the shifted operator
 *       (SlabModePreconditioner::factor(..., mu): exact for the shifted plane-averaged operator);
 *   Armijo backtracking on the merit unchanged; on line-search failure mu <- min(retry_factor mu,
 *       mu_max) and the step is re-solved (P-A re-factored at the same base), at most max_retries
 *       times (and only while mu actually grows); then `linesearch-fail`. Each retry prints a
 *       `NEWTON ... Psi-tc retry r/R with mu=..` line; every LINEAR / NEWTON line carries `mu=`.
 *   Eisenstat-Walker forcing (if on) applies to the shifted solves with the same eta_k (the merit
 *   history is that of the nonlinear iterates); stage acceptance, r_F, r_out <= tol, stagnation /
 *   maxit rules unchanged. The converged discrete state does not depend on mu (the shift only
 *   changes the step; acceptance is on the unshifted residual; mu_k -> 0 with SER as E -> 0).
 * Disabled: mu = 0 exactly, no shift kernel, no preconditioner shift, log lines unchanged: the N7a
 * iteration bitwise.
 *
 * Coarse-space correction (SF-33 N7c PROBE; cfg.coarse, default off)
 * -------------------------------------------------------------------
 * cfg.coarse = add | mult: after every P-A factor() (same base, same mu) the Galerkin coarse
 * matrix E = V^T (J + mu D) V of SlabCoarseCorrection is rebuilt (K operator applications + host
 * LU; one `COARSE build` log line with K, times, rcond estimate, min/max |U_kk|) and GMRES uses
 * the combined preconditioner (SlabCoarseCorrection::apply); after each linear solve one
 * `COARSE apply` line gives the application count and times. A singular E (zero pivot) is treated
 * like a singular P-A mode (linear solve not run, linear_failure). off: P-A alone, bitwise N7b.
 *
 * Continuation (solve_with_continuation)
 * --------------------------------------
 * Target eps; stages [e in ladder if e < eps - 1e-12] + [eps]; first start x = 0, later stages warm
 * started from the last ACCEPTED state; inputs of a stage from the StageInputProvider (plane-0 data
 * = the new stage's u0; the unknowns keep their values). A stage is accepted iff its Newton status
 * is `converged` or its merit <= stage_ok (1e-9). A failed stage is retried at the midpoint
 * (e_conv + e) / 2 from the last accepted state, at most max_bisections (4) bisections in total;
 * when they are exhausted the continuation reports `continuation_floor`, and (if the failed stage
 * was not the target) one final attempt at the target from the last accepted state is made and
 * logged
 * `(final)` in the PATH (its Newton report is `final_newton`). Lines: STAGE, STAGE_END,
 * CONTINUATION bisection k/K, CONTINUATION gave up, PATH, in the prototype's formats; since SF-33
 * N7a the final attempt prints its own `STAGE_END ... (final attempt) -> accepted|FAILED` line and
 * a `CONTINUATION reporting ...` line names the state actually reported (the final attempt's, or
 * the failed target stage's when the target itself failed).
 *
 * Memory (device): N0 residual workspace + N1 JVP workspace (4 (N+1) N^2 doubles) + P-A (see
 * SlabModePreconditioner.cuh) + GMRES (2 (m+1) N^3 * 8 + 3 vectors) + Newton vectors: U1, U2 full
 * arrays, E, E_trial, x_trial, dx, rhs, x_conv (6 vectors of 2 N^3 doubles).
 *
 * Synchronizations: residual norms (1 per evaluation: start, every line-search trial), |dx|max and
 * |dx|_2 (2 per step), P-A factor (1 per step), GMRES (see SlabGmres.cuh), and one final
 * cudaStreamSynchronize before solve() / solve_with_continuation() return (x is complete and
 * host-visible on return). No allocation after prepare(); no device<->host field transfer in the
 * loops (only scalars).
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/DeviceSpan.cuh"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "InletSlabGrid.cuh"
#include "SlabCoarseCorrection.cuh"
#include "SlabGmres.cuh"
#include "SlabJacobianVectorProduct.cuh"
#include "SlabModePreconditioner.cuh"
#include "SlabResidual.cuh"
#include "SlabSolverTypes.cuh"

#include <cstddef>
#include <string>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

/// Prints the line to stdout and flushes (the prototype's `_log`).
void slab_stdout_logger(const std::string& line);

class SlabNewtonKrylov {
  public:
    /// The only allocating call: every workspace for `grid` with a GMRES basis of `restart` and
    /// host reports reserved for `max_newton` iterations and `max_gmres_iterations` cycles.
    void prepare(CudaContext& ctx, const InletSlabGrid& grid, int restart = 50, int max_newton = 40,
                 int max_gmres_iterations = 6000);
    bool prepared() const { return grid_.n > 0; }
    const InletSlabGrid& grid() const { return grid_; }

    /// SF-33 N7c (probe): allocates the coarse correction (SlabCoarseCorrection, `profiles` 1 or
    /// 2) used when SlabNewtonConfig::coarse != off. Optional; allocating; call after prepare().
    /// assembly / factor: SlabCoarseCorrection productization options (defaults = the probe).
    void prepare_coarse(CudaContext& ctx, int profiles,
                        SlabCoarseAssembly assembly = SlabCoarseAssembly::direct,
                        SlabCoarseFactor factor = SlabCoarseFactor::dense);

    /// Newton from the start vector x (in/out, 2 N^3). See the header comment.
    SlabNewtonReport solve(CudaContext& ctx, const SlabStageInputs& inputs, DeviceSpan<real> x,
                           const std::string& label, const SlabNewtonConfig& cfg,
                           const SlabLogger& log = slab_stdout_logger);

    /// Amplitude continuation to `eps` (x out: the final state). See the header comment.
    SlabContinuationReport solve_with_continuation(CudaContext& ctx, real eps,
                                                   const StageInputProvider& provider,
                                                   DeviceSpan<real> x,
                                                   const SlabContinuationConfig& ccfg,
                                                   const SlabNewtonConfig& ncfg,
                                                   const SlabLogger& log = slab_stdout_logger);

    /// Residual and norms at x (assembled with inputs.u0); one sync. Leaves E in the internal
    /// buffer (exposed for tests).
    SlabResidualNorms residual_norms(CudaContext& ctx, const SlabStageInputs& inputs,
                                     DeviceSpan<const real> x);

    std::size_t allocated_bytes() const;
    std::vector<const void*> buffer_pointers() const;

    SlabGmres& gmres() { return gmres_; }
    SlabModePreconditioner& preconditioner() { return prec_; }
    SlabJvpWorkspace& jvp() { return jws_; }
    SlabCoarseCorrection& coarse() { return coarse_; }

  private:
    DeviceSpan<real> span(DeviceBuffer<real>& b) { return DeviceSpan<real>(b.data(), b.size()); }
    DeviceSpan<const real> cspan(const DeviceBuffer<real>& b) const {
        return DeviceSpan<const real>(b.data(), b.size());
    }
    InletSlabGrid grid_;
    int max_newton_reserved_ = 0;
    SlabResidualWorkspace rws_;
    SlabJvpWorkspace jws_;
    SlabModePreconditioner prec_;
    SlabGmres gmres_;
    SlabCoarseCorrection coarse_; ///< SF-33 N7c probe (prepared only by prepare_coarse)
    DeviceBuffer<real> U1_, U2_, E_, Et_, xt_, dx_, rhs_, xconv_;
};

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
