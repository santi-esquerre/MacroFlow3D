#include "NewtonKrylovSolver.cuh"

#include "../../numerics/blas/axpy.cuh"
#include "../../numerics/blas/scal.cuh"
#include "../../numerics/constraints/MeanZeroProjector.cuh"
#include "../../runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>

namespace macroflow3d {
namespace streamfunctions {

namespace {

// IDENTICAL guard-chain order to the SF-15 backtracking trial classification
// (StreamfunctionSolver.cuh step 4), with the Armijo test on the merit
// phi = 0.5*r_F^2 in place of r_F itself (see the file header, E7).
PicardTrialOutcome classify_newton_trial(const StreamfunctionResidualReport& residual_t, real r_F_t,
                                         real p_t, real f_t, real r_F_k, real p_k, real f_prev,
                                         real alpha, bool guards_active,
                                         const NewtonKrylovConfig& newton_config,
                                         const AdaptivePicardConfig& adaptive_config) {
    if (!std::isfinite(r_F_t) || residual_t.nonfinite_s1 > 0 || residual_t.nonfinite_s2 > 0) {
        return PicardTrialOutcome::rejected_nonfinite;
    }
    if (guards_active &&
        (f_t > adaptive_config.max_unexplained_fraction ||
         f_t > adaptive_config.unexplained_growth_factor * f_prev +
                   adaptive_config.unexplained_growth_offset)) {
        return PicardTrialOutcome::rejected_degeneracy;
    }
    if (guards_active && p_t < p_k / adaptive_config.percentile_collapse_factor && f_t > f_prev) {
        return PicardTrialOutcome::rejected_percentile;
    }
    const real phi_k = real{0.5} * r_F_k * r_F_k;
    const real phi_t = real{0.5} * r_F_t * r_F_t;
    if (phi_t > (real{1} - newton_config.armijo_c * alpha) * phi_k) {
        return PicardTrialOutcome::rejected_armijo;
    }
    return PicardTrialOutcome::accepted;
}

} // namespace

NewtonPhaseOutcome run_newton_phase(CudaContext& context, const StreamfunctionProblemView& problem,
                                     const StreamfunctionSolverConfig& config,
                                     StreamfunctionFields& fields, StreamfunctionWorkspace& workspace,
                                     const NonlinearSourceConfig& source_config,
                                     int picard_iteration_context,
                                     std::vector<NewtonIterationRecord>& newton_history,
                                     int& newton_steps_accepted, int& newton_step_failures,
                                     StreamfunctionResidualReport& residual_out) {
    const Grid3D grid = problem.grid;
    const std::size_t n = grid.num_cells();
    const bool guards_active = config.diagnostics.num_degeneracy_thresholds > 0;
    const constraints::MeanZeroProjector projector;

    JvpWorkspace& jvp = workspace.newton_jvp();
    CoupledGmres& gmres = workspace.newton_gmres();
    BlockDiagonalMGPreconditioner& preconditioner = workspace.newton_preconditioner();

    for (int newton_iter = 0;; ++newton_iter) {
        // --- E3 step 1: HEAD -----------------------------------------------
        enqueue_streamfunction_residual(context, grid, DeviceSpan<const real>(workspace.q()),
                                        fields.fluctuations(), problem.gauge, config.eta,
                                        source_config, config.histogram, workspace.f1(),
                                        workspace.f2(), workspace.residual_workspace());
        const StreamfunctionResidualReport residual_k = synchronize_streamfunction_residual_report(
            context, grid, config.eta, source_config, config.histogram,
            workspace.residual_workspace());
        residual_out = residual_k;
        const real r_F_k = residual_k.r_F;
        const real p_k = residual_histogram_percentile(residual_k, real{0.001});

        // Guard-comparison baseline (E7): recaptured at THIS Newton HEAD, the
        // same mechanism the Picard head uses.
        real f_prev = real{0};
        if (guards_active) {
            enqueue_streamfunction_physical_diagnostics(
                context, grid, fields.fluctuations(), problem.gauge, problem.darcy_velocity,
                config.diagnostics, workspace.v_psi(), workspace.diagnostics_workspace());
            const PhysicalDiagnosticsReport diag_k =
                synchronize_streamfunction_physical_diagnostics_report(
                    context, grid, config.diagnostics, workspace.diagnostics_workspace());
            f_prev = static_cast<real>(diag_k.degeneracy_unexplained[0]) / static_cast<real>(n);
        }

        if (r_F_k <= config.picard.tolerance) {
            return NewtonPhaseOutcome{NewtonPhaseStatus::converged};
        }
        if (newton_iter == config.newton.max_newton_iterations) {
            return NewtonPhaseOutcome{NewtonPhaseStatus::budget_exhausted};
        }

        // --- E3 step 2: freeze the JVP base at the accepted state ----------
        CoupledVectorView base_view{fields.u1_span(), fields.u2_span()};
        jvp.prepare_jvp_base(context, grid, DeviceSpan<const real>(workspace.q()), base_view,
                             problem.gauge, config.eta, source_config, config.histogram);

        // --- E3 step 3: b = -F(Psi) (f1()/f2() hold F from the HEAD above) -
        blas::scal(context, workspace.f1(), real{-1});
        blas::scal(context, workspace.f2(), real{-1});
        const ConstCoupledVectorView b{DeviceSpan<const real>(workspace.f1()),
                                      DeviceSpan<const real>(workspace.f2())};

        // --- E3 step 4: inexact-Newton forcing ------------------------------
        const real rel_tol_k = std::clamp(config.newton.forcing_coefficient * std::sqrt(r_F_k),
                                          config.newton.forcing_min, config.newton.forcing_max);
        CoupledGmresConfig local_gmres_config = config.newton.gmres;
        local_gmres_config.rel_tol = rel_tol_k;

        // --- E3 step 5: J(Psi) M^-1 u = b, delta = M^-1 u -------------------
        const CoupledVectorView delta_view{workspace.newton_delta1(), workspace.newton_delta2()};
        const CoupledGmresReport gmres_report = gmres.solve(
            context, grid, jvp, preconditioner, b, local_gmres_config, config.newton.delta,
            delta_view);

        NewtonIterationRecord record;
        record.picard_iteration_context = picard_iteration_context;
        record.r_F = r_F_k;
        record.forcing_rel_tol = rel_tol_k;
        record.gmres_status = gmres_report.status;
        record.gmres_total_inner_iterations = gmres_report.total_inner_iterations;
        record.gmres_outer_cycles = gmres_report.outer_cycles;
        record.gmres_final_true_residual =
            gmres_report.checkpoints.empty() ? real{0} : gmres_report.checkpoints.back().true_residual;

        // --- E3 step 6: linear-accept rule -----------------------------------
        bool proceed_to_line_search = false;
        NewtonStepFailureReason gmres_failure_reason = NewtonStepFailureReason::none;
        if (gmres_report.status == CoupledGmresStatus::converged) {
            proceed_to_line_search = true;
        } else if (gmres_report.status == CoupledGmresStatus::max_iterations ||
                  gmres_report.status == CoupledGmresStatus::breakdown) {
            if (gmres_report.checkpoints.size() >= 2 &&
                gmres_report.checkpoints.back().true_residual <
                    gmres_report.checkpoints.front().true_residual) {
                proceed_to_line_search = true;
            } else {
                gmres_failure_reason = NewtonStepFailureReason::gmres_no_reduction;
            }
        } else {
            gmres_failure_reason = NewtonStepFailureReason::gmres_nonfinite;
        }

        if (!proceed_to_line_search) {
            record.step_failed = true;
            record.failure_reason = gmres_failure_reason;
            record.accepted_alpha = real{0};
            newton_history.push_back(record);
            ++newton_step_failures;
            return NewtonPhaseOutcome{NewtonPhaseStatus::step_failed};
        }

        // --- E3 step 7: Armijo line search over alpha -------------------------
        bool accepted = false;
        real accepted_alpha = real{0};
        for (real alpha = real{1}; alpha >= config.newton.alpha_min;
             alpha *= config.newton.backtrack_factor) {
            MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(
                workspace.u_trial1().data(), fields.u1_span().data(), n * sizeof(real),
                cudaMemcpyDeviceToDevice, context.cuda_stream()));
            blas::axpy(context, alpha, DeviceSpan<const real>(workspace.newton_delta1()),
                      workspace.u_trial1());
            projector.project(context, workspace.u_trial1(), workspace.pcg_workspace().mean_zero);

            MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(
                workspace.u_trial2().data(), fields.u2_span().data(), n * sizeof(real),
                cudaMemcpyDeviceToDevice, context.cuda_stream()));
            blas::axpy(context, alpha, DeviceSpan<const real>(workspace.newton_delta2()),
                      workspace.u_trial2());
            projector.project(context, workspace.u_trial2(), workspace.pcg_workspace().mean_zero);

            const PeriodicStreamfunctionFluctuations trial_fluctuations{workspace.u_trial1(),
                                                                        workspace.u_trial2()};

            enqueue_streamfunction_residual(
                context, grid, DeviceSpan<const real>(workspace.q()), trial_fluctuations,
                problem.gauge, config.eta, source_config, config.histogram, workspace.rhs1(),
                workspace.rhs2(), workspace.residual_workspace());
            const StreamfunctionResidualReport residual_t = synchronize_streamfunction_residual_report(
                context, grid, config.eta, source_config, config.histogram,
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
                f_t = static_cast<real>(diag_t.degeneracy_unexplained[0]) / static_cast<real>(n);
            }

            const PicardTrialOutcome outcome =
                classify_newton_trial(residual_t, r_F_t, p_t, f_t, r_F_k, p_k, f_prev, alpha,
                                      guards_active, config.newton, config.adaptive);

            NewtonTrialRecord trial_record;
            trial_record.alpha = alpha;
            trial_record.r_F_trial = r_F_t;
            trial_record.outcome = outcome;
            record.trials.push_back(trial_record);

            if (outcome == PicardTrialOutcome::accepted) {
                MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(
                    fields.u1_span().data(), workspace.u_trial1().data(), n * sizeof(real),
                    cudaMemcpyDeviceToDevice, context.cuda_stream()));
                MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(
                    fields.u2_span().data(), workspace.u_trial2().data(), n * sizeof(real),
                    cudaMemcpyDeviceToDevice, context.cuda_stream()));
                accepted = true;
                accepted_alpha = alpha;
                ++newton_steps_accepted;
                break;
            }
        }

        record.accepted_alpha = accepted_alpha;
        record.step_failed = !accepted;
        if (!accepted) {
            record.failure_reason = NewtonStepFailureReason::line_search_exhausted;
            newton_history.push_back(record);
            ++newton_step_failures;
            return NewtonPhaseOutcome{NewtonPhaseStatus::step_failed};
        }
        newton_history.push_back(record);
        // Accepted: continue to the next Newton iteration's HEAD.
    }
}

} // namespace streamfunctions
} // namespace macroflow3d
