#include "StreamfunctionSolver.cuh"

#include "../../multigrid/coefficient_hierarchy.cuh"
#include "../../numerics/blas/axpy.cuh"
#include "../../numerics/blas/scal.cuh"
#include "../../numerics/constraints/MeanZeroProjector.cuh"
#include "../../numerics/operators/lester_positive_diffusion_operator.cuh"
#include "../../runtime/cuda_check.cuh"

#include <cmath>
#include <limits>

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

    // q = 1/K or q = exp(-Y).
    enqueue_fill_streamfunction_coefficient(context, problem.conductivity,
                                            problem.conductivity_representation, workspace.q());

    // Zero-initialize u1, u2 on every call (deterministic exact control; a
    // warm-start policy is SF-14 scope).
    MACROFLOW3D_CUDA_CHECK(
        cudaMemsetAsync(fields.u1_span().data(), 0, fields.u1_span().size() * sizeof(real),
                        context.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(
        cudaMemsetAsync(fields.u2_span().data(), 0, fields.u2_span().size() * sizeof(real),
                        context.cuda_stream()));

    multigrid::populate_coefficient_hierarchy(context, workspace.hierarchy(),
                                              DeviceSpan<const real>(workspace.q()));

    // Assemble and mean-zero-project the affine periodic RHS pair. The
    // returned device diagnostics need not be separately synchronized: the
    // PCG result and the residual report both carry RHS-mean evidence.
    (void)assemble_affine_periodic_rhs(context, grid, DeviceSpan<const real>(workspace.q()),
                                       problem.gauge, workspace.rhs1(), workspace.rhs2(),
                                       workspace.affine_rhs_workspace());

    operators::LesterPositiveDiffusionOperator A(grid, DeviceSpan<const real>(workspace.q()));
    constraints::MeanZeroProjector projector;

    // Sequential block solves, psi1 then psi2, sharing the single
    // hierarchy/PCG workspace (see StreamfunctionWorkspace.cuh).
    report.psi1_result =
        solvers::projected_pcg_solve(context, A, workspace.preconditioner(),
                                     DeviceSpan<const real>(workspace.rhs1()), fields.u1_span(),
                                     config.linear, projector, workspace.pcg_workspace());
    report.psi2_result =
        solvers::projected_pcg_solve(context, A, workspace.preconditioner(),
                                     DeviceSpan<const real>(workspace.rhs2()), fields.u2_span(),
                                     config.linear, projector, workspace.pcg_workspace());

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
        return report;
    }

    NonlinearSourceConfig source_config{};
    source_config.epsilon = config.epsilon;
    source_config.v_rms = v_rms;
    source_config.num_degeneracy_thresholds = config.num_degeneracy_thresholds;
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        source_config.degeneracy_thresholds[t] = config.degeneracy_thresholds[t];
    }

    // SF-14 fixed-relaxation Picard outer loop. State 0 is the SF-13
    // zero-source initialization already produced above; report.psi1_result/
    // psi2_result still hold that initialization's PCG results until the
    // first successful update step overwrites them (see the header doc for
    // the exact picard_history layout convention).
    report.picard_history.reserve(static_cast<std::size_t>(config.picard.max_iter) + 1);

    solvers::ProjectedPCGResult step_psi1_result{};
    solvers::ProjectedPCGResult step_psi2_result{};

    for (int k = 0;; ++k) {
        // Evaluate the coupled residual F1, F2 AT the current, immutable
        // state k; this single enqueue also produces BOTH block RHSs
        // G1 = P(rhs_affine1 - eta*q*S2), G2 = P(rhs_affine2 - eta*q*S1) in
        // the residual workspace's private g1_/g2_ buffers, borrowed
        // read-only below via combined_rhs_g1()/g2().
        enqueue_streamfunction_residual(context, grid, DeviceSpan<const real>(workspace.q()),
                                        fields.fluctuations(), problem.gauge, config.eta,
                                        source_config, config.histogram, workspace.f1(),
                                        workspace.f2(), workspace.residual_workspace());
        const StreamfunctionResidualReport residual_k = synchronize_streamfunction_residual_report(
            context, grid, config.eta, source_config, config.histogram,
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
            report.picard_iterations = k;
            break;
        }
        if (k == config.picard.max_iter) {
            report.status = StreamfunctionSolveStatus::not_converged;
            report.picard_iterations = k;
            break;
        }

        // Block solves at the frozen state k: b = G views (borrowed from the
        // residual workspace, NOT re-evaluated between the two solves), x =
        // f1/f2 (reused as scratch for u_hat1/u_hat2; fully overwritten here
        // and again by the next iteration's residual enqueue above).
        //
        // Zero (not warm-start) initial guess: near the Picard fixed point
        // the current state u_i is already close to the block solution, so a
        // warm start makes the initial projected residual O(||F_i||) (tiny),
        // and the PCG RELATIVE stopping criterion (final/initial <= rtol)
        // then demands an absolute residual at the double-precision rounding
        // floor -- unattainable, so PCG stagnates at max_iterations even
        // though the outer Picard iteration is healthy. Starting from x=0
        // keeps the initial projected residual at the O(1) scale fixed by
        // the affine RHS G_i at every Picard iteration, so `rtol=1e-10`
        // stays attainable throughout; the fixed point is unchanged.
        MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(workspace.f1().data(), 0,
                                               workspace.f1().size() * sizeof(real),
                                               context.cuda_stream()));
        step_psi1_result = solvers::projected_pcg_solve(
            context, A, workspace.preconditioner(), workspace.residual_workspace().combined_rhs_g1(),
            workspace.f1(), config.linear, projector, workspace.pcg_workspace());

        MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(workspace.f2().data(), 0,
                                               workspace.f2().size() * sizeof(real),
                                               context.cuda_stream()));
        step_psi2_result = solvers::projected_pcg_solve(
            context, A, workspace.preconditioner(), workspace.residual_workspace().combined_rhs_g2(),
            workspace.f2(), config.linear, projector, workspace.pcg_workspace());

        report.psi1_result = step_psi1_result;
        report.psi2_result = step_psi2_result;

        if (!step_psi1_result.converged || !step_psi2_result.converged) {
            report.status = StreamfunctionSolveStatus::not_converged;
            report.picard_iterations = k;

            // The state-(k+1) residual was never evaluated (the failed
            // update is not applied to u1/u2 below), so this record's
            // r_F/r1/r2 are set to the documented NaN sentinel ("never
            // evaluated") rather than a misleading default zero; only the
            // failing linear results are meaningful here.
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

    // Re-run SF-11 physical diagnostics on the FINAL Picard state so
    // report.diagnostics reflects the accepted invariants, not the
    // v_rms-measurement-only evaluation performed before the loop.
    enqueue_streamfunction_physical_diagnostics(context, grid, fields.fluctuations(), problem.gauge,
                                                problem.darcy_velocity, config.diagnostics,
                                                workspace.v_psi(), workspace.diagnostics_workspace());
    report.diagnostics = synchronize_streamfunction_physical_diagnostics_report(
        context, grid, config.diagnostics, workspace.diagnostics_workspace());

    report.memory = make_streamfunction_memory_report(fields, workspace, grid);

    return report;
}

} // namespace streamfunctions
} // namespace macroflow3d
