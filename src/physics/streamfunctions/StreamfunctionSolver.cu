#include "StreamfunctionSolver.cuh"

#include "../../multigrid/coefficient_hierarchy.cuh"
#include "../../numerics/blas/axpy.cuh"
#include "../../numerics/blas/scal.cuh"
#include "../../numerics/constraints/MeanZeroProjector.cuh"
#include "../../numerics/operators/lester_positive_diffusion_operator.cuh"
#include "../../runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace macroflow3d {
namespace streamfunctions {
namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 65535;

// q = 1/K (conductivity_k) or q = exp(-Y) (log_conductivity_y). Finiteness
// and positivity of the device contents of K/Y are a kernel-side
// precondition (SF-06 wording); this kernel applies no floor or clamp.
__global__ void fill_streamfunction_coefficient_kernel(const real* conductivity, real* q,
                                                        std::size_t n, bool is_log_conductivity) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < n; index += stride) {
        const real value = conductivity[index];
        q[index] = is_log_conductivity ? exp(-value) : real{1} / value;
    }
}

void enqueue_fill_streamfunction_coefficient(CudaContext& ctx, DeviceSpan<const real> conductivity,
                                             ConductivityRepresentation representation,
                                             DeviceSpan<real> q) {
    const std::size_t n = q.size();
    if (n == 0) {
        return;
    }
    const std::size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    const int blocks = static_cast<int>(
        requested_blocks < static_cast<std::size_t>(kMaxBlocks) ? requested_blocks : kMaxBlocks);
    const bool is_log_conductivity =
        representation == ConductivityRepresentation::log_conductivity_y;
    fill_streamfunction_coefficient_kernel<<<blocks, kBlockSize, 0, ctx.cuda_stream()>>>(
        conductivity.data(), q.data(), n, is_log_conductivity);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace

StreamfunctionSolveReport solve_streamfunctions(CudaContext& context,
                                                 const StreamfunctionProblemView& problem,
                                                 const StreamfunctionSolverConfig& config,
                                                 StreamfunctionFields& fields,
                                                 StreamfunctionWorkspace& workspace) {
    const Grid3D grid = problem.grid;

    // Host misuse throws std::invalid_argument (SF-12 error contract); this
    // is intentionally not caught here.
    validate_streamfunction_problem(grid, problem, config);

    // Idempotent: allocates only on first use or when grid/config changed.
    fields.prepare(grid);
    workspace.prepare(grid, config);

    StreamfunctionSolveReport report;

    // SF-21: clear any Anderson history staged by a PREVIOUS
    // solve_streamfunctions call on this workspace before this call's Picard
    // loop can stage any new history. The accelerator's premise is a history
    // of ONE fixed-point map instance; continuation drivers reuse the same
    // workspace/accelerator across problem instances (see
    // StreamfunctionSolver.cuh for the full rationale). clear() on a fresh or
    // already-cleared accelerator is a no-op, so single-call SF-20 behavior
    // is unaffected.
    if (config.anderson.enabled) {
        workspace.anderson().clear();
    }

    // SF-20: CoefficientState::reuse skips the q-fill, MG coefficient
    // hierarchy population, and affine-RHS (re)assembly below, reusing
    // whatever the workspace already holds from a prior CoefficientState::
    // rebuild call. See CoefficientState in StreamfunctionTypes.hpp for the
    // exact caller contract this enforces.
    if (config.coefficient_state == CoefficientState::reuse) {
        if (config.initial_state != PicardInitialState::warm_start) {
            throw std::invalid_argument(
                "StreamfunctionSolverConfig::coefficient_state == reuse requires "
                "initial_state == warm_start (zero_source consumes rhs1()/rhs2() for its "
                "initialization solves and must not run against stale buffers)");
        }
        if (!workspace.coefficients_valid()) {
            throw std::invalid_argument(
                "StreamfunctionSolverConfig::coefficient_state == reuse requires a preceding "
                "coefficient_state == rebuild call on this workspace for its currently prepared "
                "grid; the workspace holds no valid q/MG hierarchy/affine RHS to reuse");
        }
    } else {
        // q = 1/K or q = exp(-Y).
        enqueue_fill_streamfunction_coefficient(context, problem.conductivity,
                                                problem.conductivity_representation, workspace.q());

        multigrid::populate_coefficient_hierarchy(context, workspace.hierarchy(),
                                                  DeviceSpan<const real>(workspace.q()));

        // Assemble and mean-zero-project the affine periodic RHS pair. The
        // returned device diagnostics need not be separately synchronized:
        // the PCG result and the residual report both carry RHS-mean
        // evidence.
        (void)assemble_affine_periodic_rhs(context, grid, DeviceSpan<const real>(workspace.q()),
                                           problem.gauge, workspace.rhs1(), workspace.rhs2(),
                                           workspace.affine_rhs_workspace());

        workspace.mark_coefficients_valid();
    }

    operators::LesterPositiveDiffusionOperator A(grid, DeviceSpan<const real>(workspace.q()));
    constraints::MeanZeroProjector projector;

    // SF-17: `zero_source` (default) is bitwise identical to SF-13..16 --
    // u1/u2 are zero-initialized and the two zero-source block solves below
    // produce Picard state 0. `warm_start` instead takes state 0 from the
    // caller-provided fields (mean-zero projected in place for gauge
    // defense) and performs NEITHER the zero-init NOR the zero-source block
    // solves; report.psi1_result/psi2_result then remain default-constructed
    // until the first Picard update step (see PicardInitialState).
    if (config.initial_state == PicardInitialState::zero_source) {
        // Zero-initialize u1, u2 on every call (deterministic exact control).
        MACROFLOW3D_CUDA_CHECK(
            cudaMemsetAsync(fields.u1_span().data(), 0, fields.u1_span().size() * sizeof(real),
                            context.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(
            cudaMemsetAsync(fields.u2_span().data(), 0, fields.u2_span().size() * sizeof(real),
                            context.cuda_stream()));

        // Sequential block solves, psi1 then psi2, sharing the single
        // hierarchy/PCG workspace (see StreamfunctionWorkspace.cuh).
        report.psi1_result = solvers::projected_pcg_solve(
            context, A, workspace.preconditioner(), DeviceSpan<const real>(workspace.rhs1()),
            fields.u1_span(), config.linear, projector, workspace.pcg_workspace());
        report.psi2_result = solvers::projected_pcg_solve(
            context, A, workspace.preconditioner(), DeviceSpan<const real>(workspace.rhs2()),
            fields.u2_span(), config.linear, projector, workspace.pcg_workspace());
    } else {
        // warm_start: state 0 is the caller-provided fields, mean-zero
        // projected in place (gauge defense). No zero-init, no zero-source
        // block solves.
        projector.project(context, fields.u1_span(), workspace.pcg_workspace().mean_zero);
        projector.project(context, fields.u2_span(), workspace.pcg_workspace().mean_zero);
    }

    // SF-11 physical diagnostics; the measured Darcy v_rms feeds the
    // nonlinear-source and residual normalization below.
    enqueue_streamfunction_physical_diagnostics(context, grid, fields.fluctuations(), problem.gauge,
                                                problem.darcy_velocity, config.diagnostics,
                                                workspace.v_psi(), workspace.diagnostics_workspace());
    report.diagnostics = synchronize_streamfunction_physical_diagnostics_report(
        context, grid, config.diagnostics, workspace.diagnostics_workspace());

    const real v_rms = report.diagnostics.v_d_rms;
    if (!std::isfinite(v_rms) || v_rms <= real{0}) {
        // The SF-09 source contract and the r1 normalization both require a
        // strictly positive v_rms; a degenerate measured Darcy field makes
        // the nonlinear residual undefined, not merely inaccurate.
        report.status = StreamfunctionSolveStatus::invalid_problem;
        report.memory = make_streamfunction_memory_report(fields, workspace, grid);
        report.anderson_history_bytes = report.memory.anderson_history_bytes;
        return report;
    }

    NonlinearSourceConfig source_config{};
    source_config.epsilon = config.epsilon;
    source_config.v_rms = v_rms;
    source_config.num_degeneracy_thresholds = config.num_degeneracy_thresholds;
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        source_config.degeneracy_thresholds[t] = config.degeneracy_thresholds[t];
    }

    // SF-14 fixed-relaxation Picard / SF-15 adaptive-Picard outer loop.
    // State 0 is the SF-13 zero-source initialization already produced
    // above; report.psi1_result/psi2_result still hold that initialization's
    // PCG results until the first successful update step overwrites them
    // (see the header doc for the exact picard_history layout convention).
    report.picard_history.reserve(static_cast<std::size_t>(config.picard.max_iter) + 1);

    solvers::ProjectedPCGResult step_psi1_result{};
    solvers::ProjectedPCGResult step_psi2_result{};

    const std::size_t n = grid.num_cells();
    const bool guards_active = config.diagnostics.num_degeneracy_thresholds > 0;

    if (!config.adaptive.enabled) {
        // Bitwise-identical SF-14 fixed-relaxation path.
        for (int k = 0;; ++k) {
            // Evaluate the coupled residual F1, F2 AT the current, immutable
            // state k; this single enqueue also produces BOTH block RHSs
            // G1 = P(rhs_affine1 - eta*q*S2), G2 = P(rhs_affine2 - eta*q*S1)
            // in the residual workspace's private g1_/g2_ buffers, borrowed
            // read-only below via combined_rhs_g1()/g2().
            enqueue_streamfunction_residual(context, grid, DeviceSpan<const real>(workspace.q()),
                                            fields.fluctuations(), problem.gauge, config.eta,
                                            source_config, config.histogram, workspace.f1(),
                                            workspace.f2(), workspace.residual_workspace());
            const StreamfunctionResidualReport residual_k =
                synchronize_streamfunction_residual_report(context, grid, config.eta,
                                                            source_config, config.histogram,
                                                            workspace.residual_workspace());
            report.residual = residual_k;

            PicardIterationRecord record;
            record.r_F = residual_k.r_F;
            record.r1 = residual_k.r1;
            record.r2 = residual_k.r2;
            record.psi1_result = step_psi1_result;
            record.psi2_result = step_psi2_result;
            report.picard_history.push_back(record);

            if (residual_k.r_F <= config.picard.tolerance) {
                report.status = StreamfunctionSolveStatus::converged;
                report.exit_reason = PicardExitReason::converged;
                report.picard_iterations = k;
                break;
            }
            if (k == config.picard.max_iter) {
                report.status = StreamfunctionSolveStatus::not_converged;
                report.exit_reason = PicardExitReason::budget_exhausted;
                report.picard_iterations = k;
                break;
            }

            // Block solves at the frozen state k: b = G views (borrowed from
            // the residual workspace, NOT re-evaluated between the two
            // solves), x = f1/f2 (reused as scratch for u_hat1/u_hat2; fully
            // overwritten here and again by the next iteration's residual
            // enqueue above).
            //
            // Zero (not warm-start) initial guess: near the Picard fixed
            // point the current state u_i is already close to the block
            // solution, so a warm start makes the initial projected residual
            // O(||F_i||) (tiny), and the PCG RELATIVE stopping criterion
            // (final/initial <= rtol) then demands an absolute residual at
            // the double-precision rounding floor -- unattainable, so PCG
            // stagnates at max_iterations even though the outer Picard
            // iteration is healthy. Starting from x=0 keeps the initial
            // projected residual at the O(1) scale fixed by the affine RHS
            // G_i at every Picard iteration, so `rtol=1e-10` stays
            // attainable throughout; the fixed point is unchanged.
            MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(workspace.f1().data(), 0,
                                                   workspace.f1().size() * sizeof(real),
                                                   context.cuda_stream()));
            step_psi1_result = solvers::projected_pcg_solve(
                context, A, workspace.preconditioner(),
                workspace.residual_workspace().combined_rhs_g1(), workspace.f1(), config.linear,
                projector, workspace.pcg_workspace());

            MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(workspace.f2().data(), 0,
                                                   workspace.f2().size() * sizeof(real),
                                                   context.cuda_stream()));
            step_psi2_result = solvers::projected_pcg_solve(
                context, A, workspace.preconditioner(),
                workspace.residual_workspace().combined_rhs_g2(), workspace.f2(), config.linear,
                projector, workspace.pcg_workspace());

            report.psi1_result = step_psi1_result;
            report.psi2_result = step_psi2_result;

            if (!step_psi1_result.converged || !step_psi2_result.converged) {
                report.status = StreamfunctionSolveStatus::not_converged;
                report.exit_reason = PicardExitReason::linear_block_failure;
                report.picard_iterations = k;

                // The state-(k+1) residual was never evaluated (the failed
                // update is not applied to u1/u2 below), so this record's
                // r_F/r1/r2 are set to the documented NaN sentinel ("never
                // evaluated") rather than a misleading default zero; only
                // the failing linear results are meaningful here.
                PicardIterationRecord failed_record;
                failed_record.r_F = std::numeric_limits<real>::quiet_NaN();
                failed_record.r1 = std::numeric_limits<real>::quiet_NaN();
                failed_record.r2 = std::numeric_limits<real>::quiet_NaN();
                failed_record.psi1_result = step_psi1_result;
                failed_record.psi2_result = step_psi2_result;
                report.picard_history.push_back(failed_record);
                break;
            }

            // Paired fixed relaxation: u_i <- (1-omega)*u_i + omega*u_hat_i,
            // then re-project both fields to mean zero (gauge maintenance).
            const real omega = config.picard.omega;
            blas::scal(context, fields.u1_span(), real{1} - omega);
            blas::axpy(context, omega, DeviceSpan<const real>(workspace.f1()), fields.u1_span());
            blas::scal(context, fields.u2_span(), real{1} - omega);
            blas::axpy(context, omega, DeviceSpan<const real>(workspace.f2()), fields.u2_span());
            projector.project(context, fields.u1_span(), workspace.pcg_workspace().mean_zero);
            projector.project(context, fields.u2_span(), workspace.pcg_workspace().mean_zero);
        }
        report.final_omega = config.picard.omega;
    } else {
        // SF-15 adaptive-Picard globalization over the SAME fixed-point map.
        report.trial_history.reserve(
            (static_cast<std::size_t>(config.picard.max_iter) + 1) * 4);

        real omega = config.picard.omega;
        int easy_streak_count = 0;

        for (int k = 0;; ++k) {
            // HEAD: residual at the ACCEPTED state k (identical enqueue to
            // the SF-14 path).
            enqueue_streamfunction_residual(context, grid, DeviceSpan<const real>(workspace.q()),
                                            fields.fluctuations(), problem.gauge, config.eta,
                                            source_config, config.histogram, workspace.f1(),
                                            workspace.f2(), workspace.residual_workspace());
            const StreamfunctionResidualReport residual_k =
                synchronize_streamfunction_residual_report(context, grid, config.eta,
                                                            source_config, config.histogram,
                                                            workspace.residual_workspace());
            report.residual = residual_k;
            const real r_F_k = residual_k.r_F;
            const real p_k = residual_histogram_percentile(residual_k, real{0.001});

            // Guard-active only: unexplained fraction at the accepted state,
            // captured BEFORE any trial evaluation overwrites the shared
            // diagnostics workspace.
            real f_prev = real{0};
            if (guards_active) {
                enqueue_streamfunction_physical_diagnostics(
                    context, grid, fields.fluctuations(), problem.gauge, problem.darcy_velocity,
                    config.diagnostics, workspace.v_psi(), workspace.diagnostics_workspace());
                const PhysicalDiagnosticsReport diag_k =
                    synchronize_streamfunction_physical_diagnostics_report(
                        context, grid, config.diagnostics, workspace.diagnostics_workspace());
                f_prev = static_cast<real>(diag_k.degeneracy_unexplained[0]) /
                         static_cast<real>(n);
            }

            PicardIterationRecord record;
            record.r_F = r_F_k;
            record.r1 = residual_k.r1;
            record.r2 = residual_k.r2;
            record.psi1_result = step_psi1_result;
            record.psi2_result = step_psi2_result;
            report.picard_history.push_back(record);

            if (r_F_k <= config.picard.tolerance) {
                report.status = StreamfunctionSolveStatus::converged;
                report.exit_reason = PicardExitReason::converged;
                report.picard_iterations = k;
                break;
            }
            if (k == config.picard.max_iter) {
                report.status = StreamfunctionSolveStatus::not_converged;
                report.exit_reason = PicardExitReason::budget_exhausted;
                report.picard_iterations = k;
                break;
            }
            if (k >= config.adaptive.stagnation_window) {
                const real r_F_window_start =
                    report.picard_history[static_cast<std::size_t>(
                                               k - config.adaptive.stagnation_window)]
                        .r_F;
                if (r_F_k > (real{1} - config.adaptive.stagnation_min_reduction) *
                                r_F_window_start) {
                    report.status = StreamfunctionSolveStatus::not_converged;
                    report.exit_reason = PicardExitReason::stagnated;
                    report.picard_iterations = k;
                    break;
                }
            }

            // MAP (once per outer iteration k): the two block solves at the
            // frozen state k, exactly as SF-14. See the SF-14 code path
            // above for the zero-initial-guess rationale.
            MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(workspace.f1().data(), 0,
                                                   workspace.f1().size() * sizeof(real),
                                                   context.cuda_stream()));
            step_psi1_result = solvers::projected_pcg_solve(
                context, A, workspace.preconditioner(),
                workspace.residual_workspace().combined_rhs_g1(), workspace.f1(), config.linear,
                projector, workspace.pcg_workspace());

            MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(workspace.f2().data(), 0,
                                                   workspace.f2().size() * sizeof(real),
                                                   context.cuda_stream()));
            step_psi2_result = solvers::projected_pcg_solve(
                context, A, workspace.preconditioner(),
                workspace.residual_workspace().combined_rhs_g2(), workspace.f2(), config.linear,
                projector, workspace.pcg_workspace());

            report.psi1_result = step_psi1_result;
            report.psi2_result = step_psi2_result;

            if (!step_psi1_result.converged || !step_psi2_result.converged) {
                report.status = StreamfunctionSolveStatus::not_converged;
                report.exit_reason = PicardExitReason::linear_block_failure;
                report.picard_iterations = k;

                PicardIterationRecord failed_record;
                failed_record.r_F = std::numeric_limits<real>::quiet_NaN();
                failed_record.r1 = std::numeric_limits<real>::quiet_NaN();
                failed_record.r2 = std::numeric_limits<real>::quiet_NaN();
                failed_record.psi1_result = step_psi1_result;
                failed_record.psi2_result = step_psi2_result;
                report.picard_history.push_back(failed_record);
                break;
            }

            // SF-20: Anderson acceleration, inserted AFTER the MAP step and
            // BEFORE the SF-15 backtracking search. See StreamfunctionSolver.cuh
            // for the exact, order-sensitive semantics implemented below.
            // config.anderson.enabled == false skips this entire block, so
            // the disabled path is bitwise identical to pre-SF-20 SF-15.
            bool anderson_accepted_this_iteration = false;
            if (config.anderson.enabled) {
                AndersonAccelerator& anderson = workspace.anderson();

                // History maintenance (unconditional, every enabled outer
                // iteration): x_k is the CURRENT accepted state
                // (fields.u1_span()/u2_span(), untouched by MAP above);
                // u_hat_k is this iteration's frozen MAP output (f1()/f2()).
                anderson.update_history(context, DeviceSpan<const real>(fields.u1_span()),
                                        DeviceSpan<const real>(fields.u2_span()),
                                        DeviceSpan<const real>(workspace.f1()),
                                        DeviceSpan<const real>(workspace.f2()));

                if (k >= config.anderson.start_iteration && anderson.num_columns() >= 1) {
                    real condition_estimate = std::numeric_limits<real>::infinity();
                    const bool formed = anderson.form_candidate(
                        context, DeviceSpan<const real>(workspace.f1()),
                        DeviceSpan<const real>(workspace.f2()), config.anderson.condition_limit,
                        workspace.u_trial1(), workspace.u_trial2(), condition_estimate);

                    if (!formed) {
                        // Gram system too ill-conditioned (or, unreachable
                        // here given the guard above, no history): do NOT
                        // accelerate this iteration; clear history and fall
                        // through to the unchanged Picard backtracking below.
                        anderson.clear();
                        ++report.anderson_condition_resets;
                    } else {
                        projector.project(context, workspace.u_trial1(),
                                          workspace.pcg_workspace().mean_zero);
                        projector.project(context, workspace.u_trial2(),
                                          workspace.pcg_workspace().mean_zero);

                        const PeriodicStreamfunctionFluctuations anderson_fluctuations{
                            workspace.u_trial1(), workspace.u_trial2()};

                        // Trial residual evaluation reuses rhs1()/rhs2(),
                        // exactly as an SF-15 backtracking trial does (idle
                        // at this point in the outer iteration).
                        enqueue_streamfunction_residual(
                            context, grid, DeviceSpan<const real>(workspace.q()),
                            anderson_fluctuations, problem.gauge, config.eta, source_config,
                            config.histogram, workspace.rhs1(), workspace.rhs2(),
                            workspace.residual_workspace());
                        const StreamfunctionResidualReport residual_a =
                            synchronize_streamfunction_residual_report(
                                context, grid, config.eta, source_config, config.histogram,
                                workspace.residual_workspace());
                        const real r_F_a = residual_a.r_F;
                        const real p_a = residual_histogram_percentile(residual_a, real{0.001});

                        real f_a = real{0};
                        if (guards_active) {
                            enqueue_streamfunction_physical_diagnostics(
                                context, grid, anderson_fluctuations, problem.gauge,
                                problem.darcy_velocity, config.diagnostics, workspace.v_psi(),
                                workspace.diagnostics_workspace());
                            const PhysicalDiagnosticsReport diag_a =
                                synchronize_streamfunction_physical_diagnostics_report(
                                    context, grid, config.diagnostics,
                                    workspace.diagnostics_workspace());
                            f_a = static_cast<real>(diag_a.degeneracy_unexplained[0]) /
                                  static_cast<real>(n);
                        }

                        // IDENTICAL guard order to the SF-15 backtracking
                        // trial classification below, with omega_try fixed
                        // at 1 for the Armijo test (see StreamfunctionSolver.cuh).
                        PicardTrialOutcome outcome;
                        if (!std::isfinite(r_F_a) || residual_a.nonfinite_s1 > 0 ||
                            residual_a.nonfinite_s2 > 0) {
                            outcome = PicardTrialOutcome::rejected_nonfinite;
                        } else if (guards_active &&
                                  (f_a > config.adaptive.max_unexplained_fraction ||
                                   f_a > config.adaptive.unexplained_growth_factor * f_prev +
                                             config.adaptive.unexplained_growth_offset)) {
                            outcome = PicardTrialOutcome::rejected_degeneracy;
                        } else if (guards_active &&
                                  p_a < p_k / config.adaptive.percentile_collapse_factor &&
                                  f_a > f_prev) {
                            outcome = PicardTrialOutcome::rejected_percentile;
                        } else if (r_F_a > (real{1} - config.adaptive.armijo_c) * r_F_k) {
                            outcome = PicardTrialOutcome::rejected_armijo;
                        } else {
                            outcome = PicardTrialOutcome::accepted;
                        }

                        PicardTrialRecord anderson_trial;
                        anderson_trial.iteration = k;
                        anderson_trial.omega = real{1};
                        anderson_trial.r_F_trial = r_F_a;
                        anderson_trial.outcome = outcome;
                        anderson_trial.anderson = true;
                        report.trial_history.push_back(anderson_trial);

                        if (outcome == PicardTrialOutcome::accepted) {
                            // Only accepted-state change: copy the trial
                            // pair into fields, exactly as an SF-15
                            // acceptance. omega/easy-streak state is left
                            // UNCHANGED by an Anderson acceptance.
                            MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(
                                fields.u1_span().data(), workspace.u_trial1().data(),
                                n * sizeof(real), cudaMemcpyDeviceToDevice,
                                context.cuda_stream()));
                            MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(
                                fields.u2_span().data(), workspace.u_trial2().data(),
                                n * sizeof(real), cudaMemcpyDeviceToDevice,
                                context.cuda_stream()));
                            ++report.anderson_accepted;
                            anderson_accepted_this_iteration = true;
                        } else {
                            // REJECTED: never accept a candidate that fails
                            // the trial guard chain. Clear history and fall
                            // through to the unchanged Picard backtracking
                            // search below, starting at the persistent omega
                            // exactly as if Anderson had not been attempted.
                            anderson.clear();
                            ++report.anderson_rejected;
                        }
                    }
                }
            }

            if (anderson_accepted_this_iteration) {
                // Skip the SF-15 backtracking search entirely for this k;
                // proceed to the next outer iteration's HEAD.
                continue;
            }

            // BACKTRACKING: search over omega_try, starting at the
            // persistent omega, without recomputing the MAP (u_hat1/u_hat2,
            // held in f1/f2) between trials.
            real omega_try = omega;
            int backtracks = 0;
            bool accepted_this_iteration = false;
            bool exit_now = false;

            while (!accepted_this_iteration) {
                MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(
                    workspace.u_trial1().data(), fields.u1_span().data(), n * sizeof(real),
                    cudaMemcpyDeviceToDevice, context.cuda_stream()));
                blas::scal(context, workspace.u_trial1(), real{1} - omega_try);
                blas::axpy(context, omega_try, DeviceSpan<const real>(workspace.f1()),
                          workspace.u_trial1());
                projector.project(context, workspace.u_trial1(),
                                  workspace.pcg_workspace().mean_zero);

                MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(
                    workspace.u_trial2().data(), fields.u2_span().data(), n * sizeof(real),
                    cudaMemcpyDeviceToDevice, context.cuda_stream()));
                blas::scal(context, workspace.u_trial2(), real{1} - omega_try);
                blas::axpy(context, omega_try, DeviceSpan<const real>(workspace.f2()),
                          workspace.u_trial2());
                projector.project(context, workspace.u_trial2(),
                                  workspace.pcg_workspace().mean_zero);

                const PeriodicStreamfunctionFluctuations trial_fluctuations{
                    workspace.u_trial1(), workspace.u_trial2()};

                // Trial residual evaluation reuses rhs1()/rhs2() as output
                // buffers: those are idle after the top-level affine-RHS
                // assembly and are not touched again by this outer
                // iteration's MAP step (already consumed above).
                enqueue_streamfunction_residual(
                    context, grid, DeviceSpan<const real>(workspace.q()), trial_fluctuations,
                    problem.gauge, config.eta, source_config, config.histogram, workspace.rhs1(),
                    workspace.rhs2(), workspace.residual_workspace());
                const StreamfunctionResidualReport residual_t =
                    synchronize_streamfunction_residual_report(context, grid, config.eta,
                                                                source_config, config.histogram,
                                                                workspace.residual_workspace());
                const real r_F_t = residual_t.r_F;
                const real p_t = residual_histogram_percentile(residual_t, real{0.001});

                real f_t = real{0};
                if (guards_active) {
                    enqueue_streamfunction_physical_diagnostics(
                        context, grid, trial_fluctuations, problem.gauge, problem.darcy_velocity,
                        config.diagnostics, workspace.v_psi(), workspace.diagnostics_workspace());
                    const PhysicalDiagnosticsReport diag_t =
                        synchronize_streamfunction_physical_diagnostics_report(
                            context, grid, config.diagnostics, workspace.diagnostics_workspace());
                    f_t = static_cast<real>(diag_t.degeneracy_unexplained[0]) /
                          static_cast<real>(n);
                }

                PicardTrialOutcome outcome;
                if (!std::isfinite(r_F_t) || residual_t.nonfinite_s1 > 0 ||
                    residual_t.nonfinite_s2 > 0) {
                    outcome = PicardTrialOutcome::rejected_nonfinite;
                } else if (guards_active &&
                          (f_t > config.adaptive.max_unexplained_fraction ||
                           f_t > config.adaptive.unexplained_growth_factor * f_prev +
                                     config.adaptive.unexplained_growth_offset)) {
                    outcome = PicardTrialOutcome::rejected_degeneracy;
                } else if (guards_active &&
                          p_t < p_k / config.adaptive.percentile_collapse_factor &&
                          f_t > f_prev) {
                    outcome = PicardTrialOutcome::rejected_percentile;
                } else if (r_F_t >
                          (real{1} - config.adaptive.armijo_c * omega_try) * r_F_k) {
                    outcome = PicardTrialOutcome::rejected_armijo;
                } else {
                    outcome = PicardTrialOutcome::accepted;
                }

                PicardTrialRecord trial_record;
                trial_record.iteration = k;
                trial_record.omega = omega_try;
                trial_record.r_F_trial = r_F_t;
                trial_record.outcome = outcome;
                report.trial_history.push_back(trial_record);

                if (outcome == PicardTrialOutcome::accepted) {
                    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(
                        fields.u1_span().data(), workspace.u_trial1().data(), n * sizeof(real),
                        cudaMemcpyDeviceToDevice, context.cuda_stream()));
                    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(
                        fields.u2_span().data(), workspace.u_trial2().data(), n * sizeof(real),
                        cudaMemcpyDeviceToDevice, context.cuda_stream()));

                    // Persistent omega becomes the accepted trial's omega;
                    // growth (if the easy streak completes) applies on top.
                    omega = omega_try;
                    if (backtracks == 0) {
                        ++easy_streak_count;
                        if (easy_streak_count == config.adaptive.easy_streak) {
                            omega = std::min(omega * config.adaptive.growth_factor,
                                            config.adaptive.omega_max);
                            easy_streak_count = 0;
                        }
                    } else {
                        easy_streak_count = 0;
                    }
                    accepted_this_iteration = true;
                } else {
                    ++backtracks;
                    if (omega_try <= config.adaptive.omega_min) {
                        // The rejected trial was already at the floor: a
                        // structured failure, not a silently-accepted step.
                        // fields.u1_span()/u2_span() remain exactly the last
                        // accepted state (state k), untouched.
                        report.status = StreamfunctionSolveStatus::not_converged;
                        report.exit_reason = PicardExitReason::omega_floor_rejected;
                        report.picard_iterations = k;
                        exit_now = true;
                        break;
                    }
                    omega_try = std::max(omega_try * config.adaptive.backtrack_factor,
                                        config.adaptive.omega_min);
                }
            }

            if (exit_now) {
                break;
            }
        }
        report.final_omega = omega;
    }

    // Re-run SF-11 physical diagnostics on the FINAL Picard state so
    // report.diagnostics reflects the accepted invariants, not the
    // v_rms-measurement-only evaluation performed before the loop.
    enqueue_streamfunction_physical_diagnostics(context, grid, fields.fluctuations(), problem.gauge,
                                                problem.darcy_velocity, config.diagnostics,
                                                workspace.v_psi(), workspace.diagnostics_workspace());
    report.diagnostics = synchronize_streamfunction_physical_diagnostics_report(
        context, grid, config.diagnostics, workspace.diagnostics_workspace());

    report.memory = make_streamfunction_memory_report(fields, workspace, grid);
    report.anderson_history_bytes = report.memory.anderson_history_bytes;

    return report;
}

} // namespace streamfunctions
} // namespace macroflow3d
