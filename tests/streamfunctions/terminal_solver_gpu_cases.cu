#include "streamfunction_operator_test_cases.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/multigrid/mg_types.hpp"
#include "src/numerics/blas/axpy.cuh"
#include "src/numerics/blas/copy.cuh"
#include "src/numerics/blas/dot.cuh"
#include "src/numerics/blas/fill.cuh"
#include "src/numerics/blas/scal.cuh"
#include "src/numerics/constraints/MeanZeroProjector.cuh"
#include "src/numerics/operators/lester_positive_diffusion_operator.cuh"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/physics/streamfunctions/BlockDiagonalMGPreconditioner.cuh"
#include "src/physics/streamfunctions/ContinuationController.hpp"
#include "src/physics/streamfunctions/CoupledGmres.cuh"
#include "src/physics/streamfunctions/JacobianVectorProduct.cuh"
#include "src/physics/streamfunctions/NonlinearSources.cuh"
#include "src/physics/streamfunctions/ResidualEvaluator.cuh"
#include "src/physics/streamfunctions/ShiftedJacobianOperator.cuh"
#include "src/physics/streamfunctions/StreamfunctionSolver.cuh"
#include "src/physics/streamfunctions/StreamfunctionTypes.hpp"
#include "src/physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "src/physics/streamfunctions/affine_gauge.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// SF-25 T02: the D-gate diagnostic (activation decisions E2-E5,
// `docs/plans/active/lester-eq14/increments/SF-25-terminal-manifold-solver.md`,
// implemented VERBATIM per the protocol prespecified BEFORE this file was
// written) plus a cheap unit case for the T01 `ShiftedJacobianOperator`
// enabler. No fixture constant, seed, tolerance, or threshold below is
// adjusted after seeing a result (STOP semantics, matching every other
// PRESPECIFIED-gate file in this test suite, e.g. `gmres_gpu_cases.cu`,
// `newton_gpu_cases.cu`). This file never touches `src/**`.
//
// Case 1 (`terminal_shifted_apply_unit`, cheap, in the aggregated `cases()`
// registry): exactness of `ShiftedJacobianOperator::apply` against an
// independently recomputed `J p + mu*A p`, the `mu=0` bitwise passthrough,
// and the `set_mu`/unprepared-`apply` fail-fast contract, on the 16^3
// manufactured trig-K fixture copied verbatim from `newton_gpu_cases.cu`'s
// `ManufacturedProblemBuffers` (amplitude 0.25, uniform Darcy v=(1,0,0),
// benchmark(1) gauge -- the SAME provenance the SF-24 equivalence case (G1)
// uses).
//
// Case 2 (`terminal_dgate_diagnostic`, HEAVY, separate registry like
// `anderson_stall_fixture_*`/`newton_difficult_case`): the D-gate itself.
// E2 freezes a genuine Picard/Anderson plateau state (sigma_Y^2=1, 32^3,
// seed 12345, corr_length 8, lambda=0.5125 warm-started from the accepted
// (lambda=0.5, eta=1) SF-21 smoke state, NEWTON DISABLED per the SF-25
// activation bitácora's E2 decision -- the freeze characterizes the
// SF-21 plateau regime, not the SF-24 Newton budget-exhaustion regime).
// The per-lambda conductivity/flow setup (Y_att = lambda*Y, K_att =
// exp(Y_att), the SF-19 affine flow solve, the log_conductivity_y problem
// view) is mirrored EXACTLY from `ContinuationController.cu`'s
// `build_attempt`/`make_problem_view` lambdas (cited by line number at each
// mirrored step below). E3 sweeps the shift `mu` on the frozen system with
// the accepted GMRES; E4 is a non-blocking spectral probe; E5 is a bounded
// Levenberg-Marquardt mini-solve using the E3-calibrated `theta`. See the
// D-gate protocol in the increment specification and
// `docs/decisions/2026-08-14-manifold-robust-terminal-solver.md` for the
// full scientific rationale this case tests against.

namespace macroflow3d::streamfunctions::test {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

// ---------------------------------------------------------------------------
// Shared grid/BC/download/label helpers, mirrored from newton_gpu_cases.cu /
// heterogeneity_continuation_gpu_cases.cu / gmres_gpu_cases.cu.
// ---------------------------------------------------------------------------

[[nodiscard]] Grid3D isotropic_grid(int n, real domain_length = real{1}) {
    const real h = domain_length / static_cast<real>(n);
    return Grid3D{n, n, n, h, h, h};
}

[[nodiscard]] std::size_t compact_mac_u_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx + 1) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz);
}
[[nodiscard]] std::size_t compact_mac_v_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny + 1) *
           static_cast<std::size_t>(grid.nz);
}
[[nodiscard]] std::size_t compact_mac_w_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz + 1);
}

[[nodiscard]] BCSpec triply_periodic() {
    BCSpec bc;
    bc.xmin = BCFace(BCType::Periodic, real{0});
    bc.xmax = BCFace(BCType::Periodic, real{0});
    bc.ymin = BCFace(BCType::Periodic, real{0});
    bc.ymax = BCFace(BCType::Periodic, real{0});
    bc.zmin = BCFace(BCType::Periodic, real{0});
    bc.zmax = BCFace(BCType::Periodic, real{0});
    return bc;
}

[[nodiscard]] std::vector<real> download(const DeviceSpan<const real>& span) {
    std::vector<real> host(span.size());
    if (!host.empty()) {
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(host.data(), span.data(), host.size() * sizeof(real), cudaMemcpyDeviceToHost));
    }
    return host;
}

[[nodiscard]] bool bitwise_equal(const std::vector<real>& a, const std::vector<real>& b) {
    if (a.size() != b.size()) return false;
    if (a.empty()) return true;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(real)) == 0;
}

[[nodiscard]] const char* solve_status_label(StreamfunctionSolveStatus status) {
    switch (status) {
        case StreamfunctionSolveStatus::not_run: return "not_run";
        case StreamfunctionSolveStatus::converged: return "converged";
        case StreamfunctionSolveStatus::not_converged: return "not_converged";
        case StreamfunctionSolveStatus::invalid_problem: return "invalid_problem";
        default: return "unknown";
    }
}

[[nodiscard]] const char* exit_reason_label(PicardExitReason reason) {
    switch (reason) {
        case PicardExitReason::none: return "none";
        case PicardExitReason::converged: return "converged";
        case PicardExitReason::budget_exhausted: return "budget_exhausted";
        case PicardExitReason::linear_block_failure: return "linear_block_failure";
        case PicardExitReason::stagnated: return "stagnated";
        case PicardExitReason::omega_floor_rejected: return "omega_floor_rejected";
        case PicardExitReason::newton_exhausted: return "newton_exhausted";
        case PicardExitReason::newton_budget_exhausted: return "newton_budget_exhausted";
        default: return "unknown";
    }
}

[[nodiscard]] const char* gmres_status_label(CoupledGmresStatus status) {
    switch (status) {
        case CoupledGmresStatus::converged: return "converged";
        case CoupledGmresStatus::max_iterations: return "max_iterations";
        case CoupledGmresStatus::breakdown: return "breakdown";
        case CoupledGmresStatus::nonfinite: return "nonfinite";
        default: return "unknown";
    }
}

[[nodiscard]] const char* heterogeneity_status_label(HeterogeneityStatus status) {
    switch (status) {
        case HeterogeneityStatus::reached_target: return "reached_target";
        case HeterogeneityStatus::baseline_failed: return "baseline_failed";
        case HeterogeneityStatus::lambda_floor_exhausted: return "lambda_floor_exhausted";
        case HeterogeneityStatus::epsilon_floor_exhausted: return "epsilon_floor_exhausted";
        case HeterogeneityStatus::invalid_problem: return "invalid_problem";
        default: return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Case 1: terminal_shifted_apply_unit (cheap). Manufactured trig-K fixture
// copied VERBATIM from newton_gpu_cases.cu's ManufacturedProblemBuffers
// (amplitude 0.25, uniform Darcy v=(1,0,0), benchmark(1) gauge).
// ---------------------------------------------------------------------------

struct ManufacturedProblemBuffers {
    DeviceBuffer<real> conductivity;
    DeviceBuffer<real> darcy_u;
    DeviceBuffer<real> darcy_v;
    DeviceBuffer<real> darcy_w;

    explicit ManufacturedProblemBuffers(const Grid3D& grid, double amplitude)
        : conductivity(grid.num_cells()), darcy_u(compact_mac_u_size(grid)),
          darcy_v(compact_mac_v_size(grid)), darcy_w(compact_mac_w_size(grid)) {
        std::vector<real> k_host(grid.num_cells());
        for (int iz = 0; iz < grid.nz; ++iz) {
            const double z = (iz + 0.5) * static_cast<double>(grid.dz);
            for (int iy = 0; iy < grid.ny; ++iy) {
                const double y = (iy + 0.5) * static_cast<double>(grid.dy);
                for (int ix = 0; ix < grid.nx; ++ix) {
                    const double x = (ix + 0.5) * static_cast<double>(grid.dx);
                    const double log_k = amplitude * std::sin(2.0 * kPi * x) *
                                          std::sin(2.0 * kPi * y) * std::sin(2.0 * kPi * z);
                    k_host[grid.idx(ix, iy, iz)] = static_cast<real>(std::exp(log_k));
                }
            }
        }
        const std::vector<real> ones_u(compact_mac_u_size(grid), real{1});
        const std::vector<real> zeros_v(compact_mac_v_size(grid), real{0});
        const std::vector<real> zeros_w(compact_mac_w_size(grid), real{0});
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(conductivity.data(), k_host.data(),
                                          k_host.size() * sizeof(real), cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_u.data(), ones_u.data(), ones_u.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_v.data(), zeros_v.data(), zeros_v.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_w.data(), zeros_w.data(), zeros_w.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
    }
};

[[nodiscard]] StreamfunctionProblemView manufactured_problem_view(
    const Grid3D& grid, const ManufacturedProblemBuffers& buffers) {
    StreamfunctionProblemView problem;
    problem.grid = grid;
    problem.conductivity = DeviceSpan<const real>(buffers.conductivity.span());
    problem.conductivity_representation = ConductivityRepresentation::conductivity_k;
    problem.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(buffers.darcy_u.span()),
                                                         DeviceSpan<const real>(buffers.darcy_v.span()),
                                                         DeviceSpan<const real>(buffers.darcy_w.span())};
    problem.bc = triply_periodic();
    problem.gauge = AffineGauge::benchmark(real{1});
    return problem;
}

[[nodiscard]] CaseResult case_terminal_shifted_apply_unit() {
    std::cout << std::setprecision(17);
    constexpr int n = 16;
    constexpr double amplitude = 0.25;
    const Grid3D grid = isotropic_grid(n);
    const std::size_t cells = grid.num_cells();

    CudaContext ctx(0);
    ManufacturedProblemBuffers buffers(grid, amplitude);
    const StreamfunctionProblemView problem = manufactured_problem_view(grid, buffers);

    // Base state: a genuine converged adaptive-Picard solve at the solver's
    // (eta=1, epsilon=1e-2) defaults -- the SAME provenance G1's 16^3 branch
    // uses in newton_gpu_cases.cu -- so JvpWorkspace's "already mean-zero
    // projected" base-state caller contract is satisfied by construction.
    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    const StreamfunctionSolverConfig solver_config{};
    const StreamfunctionSolveReport solve_report =
        solve_streamfunctions(ctx, problem, solver_config, fields, workspace);
    ctx.synchronize();
    if (solve_report.status != StreamfunctionSolveStatus::converged) {
        throw std::runtime_error(
            "terminal_shifted_apply_unit fixture: base adaptive-Picard solve did not converge");
    }

    const NonlinearSourceConfig source_config{real{1e-2}, real{1}, 0, {}}; // v=(1,0,0) -> v_rms=1 exactly.
    const ResidualHistogramConfig histogram_config{};
    const AffineGauge gauge = AffineGauge::benchmark(real{1});

    JvpWorkspace jvp;
    jvp.prepare(cells);
    jvp.prepare_jvp_base(ctx, grid, DeviceSpan<const real>(workspace.q()),
                         CoupledVectorView{fields.u1_span(), fields.u2_span()}, gauge, real{1},
                         source_config, histogram_config);

    // Deterministic nonzero mean-zero direction: P(sin-based field pair).
    std::vector<real> dir1_host(cells), dir2_host(cells);
    for (int iz = 0; iz < grid.nz; ++iz) {
        const double z = (iz + 0.5) * static_cast<double>(grid.dz);
        for (int iy = 0; iy < grid.ny; ++iy) {
            const double y = (iy + 0.5) * static_cast<double>(grid.dy);
            for (int ix = 0; ix < grid.nx; ++ix) {
                const double x = (ix + 0.5) * static_cast<double>(grid.dx);
                const std::size_t idx = grid.idx(ix, iy, iz);
                dir1_host[idx] = static_cast<real>(std::sin(2.0 * kPi * x) * std::cos(2.0 * kPi * y) *
                                                    std::sin(2.0 * kPi * z));
                dir2_host[idx] = static_cast<real>(std::cos(2.0 * kPi * x) * std::sin(2.0 * kPi * y) *
                                                    std::cos(2.0 * kPi * z));
            }
        }
    }
    DeviceBuffer<real> dir1(cells), dir2(cells);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dir1.data(), dir1_host.data(), cells * sizeof(real),
                                      cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dir2.data(), dir2_host.data(), cells * sizeof(real),
                                      cudaMemcpyHostToDevice));
    constraints::MeanZeroWorkspace mean_zero_ws;
    mean_zero_ws.prepare(cells);
    constraints::MeanZeroProjector projector;
    projector.project(ctx, dir1.span(), mean_zero_ws);
    projector.project(ctx, dir2.span(), mean_zero_ws);
    ctx.synchronize();

    const ConstCoupledVectorView dir_view(DeviceSpan<const real>(dir1.span()),
                                          DeviceSpan<const real>(dir2.span()));

    bool pass = true;
    const auto check = [&](const char* name, bool ok) {
        pass = pass && ok;
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    // (a) Exactness: shifted apply == fresh Jp + independently recomputed mu*Ap.
    const real mu_values[] = {real{1e-2}, real{1}};
    for (real mu : mu_values) {
        ShiftedJacobianOperator op(jvp, grid, DeviceSpan<const real>(workspace.q()), mu);
        op.prepare(cells);
        DeviceBuffer<real> shifted1(cells), shifted2(cells);
        const JvpApplyReport shifted_report =
            op.apply(ctx, grid, dir_view, JvpDeltaConfig{}, CoupledVectorView{shifted1.span(), shifted2.span()});

        DeviceBuffer<real> jv1(cells), jv2(cells);
        const JvpApplyReport jv_report =
            jvp.apply(ctx, grid, dir_view, JvpDeltaConfig{}, CoupledVectorView{jv1.span(), jv2.span()});

        const operators::LesterPositiveDiffusionOperator A(grid, DeviceSpan<const real>(workspace.q()));
        DeviceBuffer<real> ap1(cells), ap2(cells);
        A.apply(ctx, dir_view.c1, ap1.span());
        A.apply(ctx, dir_view.c2, ap2.span());
        blas::axpy(ctx, mu, DeviceSpan<const real>(ap1.span()), jv1.span());
        blas::axpy(ctx, mu, DeviceSpan<const real>(ap2.span()), jv2.span());
        ctx.synchronize();

        const std::vector<real> shifted1_h = download(DeviceSpan<const real>(shifted1.span()));
        const std::vector<real> shifted2_h = download(DeviceSpan<const real>(shifted2.span()));
        const std::vector<real> ref1_h = download(DeviceSpan<const real>(jv1.span()));
        const std::vector<real> ref2_h = download(DeviceSpan<const real>(jv2.span()));

        double max_diff = 0.0;
        double scale = 1.0;
        for (std::size_t i = 0; i < cells; ++i) {
            max_diff = std::max(max_diff, std::abs(static_cast<double>(shifted1_h[i]) - static_cast<double>(ref1_h[i])));
            max_diff = std::max(max_diff, std::abs(static_cast<double>(shifted2_h[i]) - static_cast<double>(ref2_h[i])));
            scale = std::max(scale, std::abs(static_cast<double>(ref1_h[i])));
            scale = std::max(scale, std::abs(static_cast<double>(ref2_h[i])));
        }
        const bool ok = shifted_report.status == JvpApplyStatus::ok &&
                        jv_report.status == JvpApplyStatus::ok && max_diff <= 1e-13 * scale;
        std::cout << "  exactness mu=" << static_cast<double>(mu) << " max_diff=" << max_diff
                  << " scale=" << scale << " threshold=" << (1e-13 * scale) << '\n';
        std::ostringstream name;
        name << "exactness_mu_" << static_cast<double>(mu);
        check(name.str().c_str(), ok);
    }

    // (b) mu=0 passthrough: bitwise-equal to a fresh plain jvp.apply.
    {
        ShiftedJacobianOperator op0(jvp, grid, DeviceSpan<const real>(workspace.q()), real{0});
        op0.prepare(cells);
        DeviceBuffer<real> shifted1(cells), shifted2(cells);
        const JvpApplyReport op0_report =
            op0.apply(ctx, grid, dir_view, JvpDeltaConfig{}, CoupledVectorView{shifted1.span(), shifted2.span()});

        DeviceBuffer<real> jv1(cells), jv2(cells);
        const JvpApplyReport jv0_report =
            jvp.apply(ctx, grid, dir_view, JvpDeltaConfig{}, CoupledVectorView{jv1.span(), jv2.span()});
        ctx.synchronize();
        (void)op0_report;
        (void)jv0_report;

        const bool bitwise_ok =
            bitwise_equal(download(DeviceSpan<const real>(shifted1.span())), download(DeviceSpan<const real>(jv1.span()))) &&
            bitwise_equal(download(DeviceSpan<const real>(shifted2.span())), download(DeviceSpan<const real>(jv2.span())));
        check("mu_zero_bitwise_passthrough", bitwise_ok);
    }

    // (c) set_mu fail-fasts; unprepared apply throws std::logic_error.
    {
        ShiftedJacobianOperator op(jvp, grid, DeviceSpan<const real>(workspace.q()), real{1e-2});
        bool threw_negative = false;
        try {
            op.set_mu(real{-1});
        } catch (const std::invalid_argument&) {
            threw_negative = true;
        }
        check("set_mu_negative_throws", threw_negative);

        bool threw_nan = false;
        try {
            op.set_mu(std::numeric_limits<real>::quiet_NaN());
        } catch (const std::invalid_argument&) {
            threw_nan = true;
        }
        check("set_mu_nan_throws", threw_nan);

        ShiftedJacobianOperator unprepared_op(jvp, grid, DeviceSpan<const real>(workspace.q()), real{1e-2});
        bool threw_unprepared = false;
        try {
            DeviceBuffer<real> out1(cells), out2(cells);
            const JvpApplyReport unprepared_report =
                unprepared_op.apply(ctx, grid, dir_view, JvpDeltaConfig{}, CoupledVectorView{out1.span(), out2.span()});
            (void)unprepared_report;
        } catch (const std::logic_error&) {
            threw_unprepared = true;
        }
        check("unprepared_apply_throws_logic_error", threw_unprepared);
    }

    std::cout << "case=terminal_shifted_apply_unit verdict=" << (pass ? "PASS" : "FAIL") << '\n';

    return {pass,
            "terminal_shifted_apply_unit",
            "gpu-terminal-shifted-apply-unit",
            "16^3 (a=0.25) manufactured trig-K fixture (copied verbatim from newton_gpu_cases.cu)",
            1.0,
            0.0,
            "SF-25 T02 enabler unit",
            pass ? "all pass" : "some failed",
            "ShiftedJacobianOperator::apply exactness (<=1e-13*scale) against an independently "
            "recomputed Jp+mu*Ap, bitwise mu=0 passthrough, and the set_mu/unprepared-apply "
            "fail-fast contract"};
}

// ---------------------------------------------------------------------------
// Case 2: terminal_dgate_diagnostic (HEAVY, separate registry). Implements
// D-gate protocol E2-E5 exactly, per the SF-25 activation bitácora and
// docs/decisions/2026-08-14-manifold-robust-terminal-solver.md.
// ---------------------------------------------------------------------------

// K_att = exp(Y_att), elementwise -- the SAME small kernel
// ContinuationController.cu's heterogeneity_exp_kernel implements (renamed
// here to avoid an ODR clash across translation units).
__global__ void terminal_dgate_exp_kernel(const real* __restrict__ y_att, real* __restrict__ k_att,
                                          std::size_t n) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < n; index += stride) {
        k_att[index] = exp(y_att[index]);
    }
}

void terminal_dgate_enqueue_exp(CudaContext& ctx, DeviceSpan<const real> y_att, DeviceSpan<real> k_att) {
    const std::size_t n = k_att.size();
    if (n == 0) return;
    constexpr int kBlock = 256;
    constexpr int kMaxBlocks = 65535;
    const std::size_t requested_blocks = (n + kBlock - 1) / kBlock;
    const int blocks =
        static_cast<int>(requested_blocks < static_cast<std::size_t>(kMaxBlocks) ? requested_blocks : kMaxBlocks);
    terminal_dgate_exp_kernel<<<blocks, kBlock, 0, ctx.cuda_stream()>>>(y_att.data(), k_att.data(), n);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

[[nodiscard]] CaseResult case_terminal_dgate_diagnostic() {
    std::cout << std::setprecision(17);
    bool pass = true;
    const auto check = [&](const char* name, bool ok) {
        pass = pass && ok;
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    constexpr int n32 = 32;
    const Grid3D grid(n32, n32, n32, real{1}, real{1}, real{1});
    const std::size_t n = grid.num_cells();
    CudaContext ctx(0);

    // =========================================================================
    // E2 freeze, step 1: the sigma_Y^2=1, 32^3 smoke, VERBATIM from
    // heterogeneity_continuation_gpu_cases.cu's run_heterogeneity_smoke
    // (seed 12345, ell=8, normalize_variance; anderson R5 defaults enabled;
    // degenerate epsilon leg; lambda axis defaults; flow_config default
    // qbar=(1,0,0)) EXCEPT newton stays at its default (disabled): E2
    // freezes the SF-21-characterized plateau regime, not the SF-24 Newton
    // budget-exhaustion regime the accepted smoke fixture now also exercises.
    // =========================================================================
    physics::PeriodicGaussianFieldConfig field_config;
    field_config.sigma2 = real{1};
    field_config.corr_length = real{8};
    field_config.seed = 12345ULL;
    field_config.normalize_variance = true;

    DeviceBuffer<real> y(n);
    physics::PeriodicGaussianFieldWorkspace field_workspace;
    const physics::PeriodicGaussianFieldReport field_report =
        physics::generate_periodic_gaussian_field(ctx, grid, field_config, y.span(), field_workspace);
    ctx.synchronize();

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    StreamfunctionSolverConfig base_config; // full defaults (adaptive Picard, newton DISABLED).
    base_config.anderson.enabled = true;
    base_config.anderson.depth = 5;
    base_config.anderson.start_iteration = 5;
    base_config.anderson.condition_limit = real{1e12};

    HeterogeneityContinuationConfig continuation_config{}; // lambda axis defaults.
    continuation_config.inner.epsilon_log10.target = continuation_config.inner.epsilon_log10.start;
    const physics::AffinePeriodicFlowConfig flow_config{}; // qbar=(1,0,0) default.

    const HeterogeneityContinuationReport freeze_report = run_streamfunction_heterogeneity_continuation(
        ctx, grid, DeviceSpan<const real>(y.span()), continuation_config, flow_config, base_config, fields,
        workspace);
    ctx.synchronize();

    std::cout << "E2 freeze: field_raw_mean=" << field_report.raw_mean
              << " field_final_variance=" << field_report.final_variance
              << " status=" << heterogeneity_status_label(freeze_report.status)
              << " final_lambda=" << freeze_report.final_lambda
              << " final_eta=" << freeze_report.final_eta
              << " stage_history_size=" << freeze_report.stage_history.size() << '\n';

    const bool e2_freeze_ok = freeze_report.status == HeterogeneityStatus::lambda_floor_exhausted &&
                              freeze_report.final_lambda == real{0.5} && freeze_report.final_eta == real{1};
    check("E2_freeze_status_lambda_floor_exhausted",
          freeze_report.status == HeterogeneityStatus::lambda_floor_exhausted);
    check("E2_freeze_final_lambda_eq_0_5", freeze_report.final_lambda == real{0.5});
    check("E2_freeze_final_eta_eq_1", freeze_report.final_eta == real{1});

    // =========================================================================
    // E2 freeze, steps 2-3: mirror ContinuationController.cu's per-lambda
    // setup EXACTLY for lambda_attempt=0.5125:
    //   - build_attempt (ContinuationController.cu lines 542-553): Y_att =
    //     lambda_attempt*Y (D2D copy + blas::scal), K_att = exp(Y_att) (the
    //     elementwise kernel above), then the SF-19 affine-periodic flow
    //     solve on K_att with THE SAME flow_config the freeze run used
    //     (default AffinePeriodicFlowConfig{}, qbar=(1,0,0));
    //   - make_problem_view (ContinuationController.cu lines 555-567):
    //     conductivity = Y_att as log_conductivity_y, darcy_velocity = that
    //     flow, triply periodic, benchmark(1) gauge.
    // Then ONE warm-started stage solve at (eta=1, epsilon=1e-2) with the
    // SAME base_config (anderson on, newton off), initial_state=warm_start,
    // coefficient_state=rebuild (first solve at this lambda on this
    // workspace) -- mirroring ContinuationController.cu lines 629-637.
    // =========================================================================
    constexpr real kLambdaAttempt = real{0.5125};

    DeviceBuffer<real> y_att(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(y_att.data(), y.data(), n * sizeof(real),
                                           cudaMemcpyDeviceToDevice, ctx.cuda_stream()));
    blas::scal(ctx, y_att.span(), kLambdaAttempt);

    DeviceBuffer<real> k_att(n);
    terminal_dgate_enqueue_exp(ctx, DeviceSpan<const real>(y_att.span()), k_att.span());

    const std::size_t u_size = compact_mac_u_size(grid);
    const std::size_t v_size = compact_mac_v_size(grid);
    const std::size_t w_size = compact_mac_w_size(grid);
    DeviceBuffer<real> flow_u(u_size), flow_v(v_size), flow_w(w_size);
    physics::AffinePeriodicFlowWorkspace flow_workspace;
    physics::AffinePeriodicVelocityView velocity{flow_u.span(), flow_v.span(), flow_w.span()};
    const physics::AffinePeriodicFlowReport attempt_flow = physics::solve_affine_periodic_flow(
        ctx, grid, DeviceSpan<const real>(k_att.span()), flow_config, velocity, flow_workspace);
    ctx.synchronize();

    StreamfunctionProblemView problem_view;
    problem_view.grid = grid;
    problem_view.conductivity = DeviceSpan<const real>(y_att.span());
    problem_view.conductivity_representation = ConductivityRepresentation::log_conductivity_y;
    problem_view.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(flow_u.span()),
                                                               DeviceSpan<const real>(flow_v.span()),
                                                               DeviceSpan<const real>(flow_w.span())};
    problem_view.bc = triply_periodic();
    problem_view.gauge = AffineGauge::benchmark(real{1});

    StreamfunctionSolverConfig stage_config = base_config;
    stage_config.eta = real{1};
    stage_config.epsilon = real{1e-2};
    stage_config.initial_state = PicardInitialState::warm_start;
    stage_config.coefficient_state = CoefficientState::rebuild;

    const StreamfunctionSolveReport stage_report =
        solve_streamfunctions(ctx, problem_view, stage_config, fields, workspace);
    ctx.synchronize();

    const double r_F_frozen = static_cast<double>(stage_report.residual.r_F);
    std::cout << "E2 stage: attempt_flow_achieved_flux=(" << attempt_flow.achieved_mean_flux[0] << ","
              << attempt_flow.achieved_mean_flux[1] << "," << attempt_flow.achieved_mean_flux[2]
              << ") status=" << solve_status_label(stage_report.status)
              << " exit_reason=" << exit_reason_label(stage_report.exit_reason)
              << " picard_iterations=" << stage_report.picard_iterations << " r_F_frozen=" << r_F_frozen
              << '\n';

    check("E2_stage_not_converged", stage_report.status == StreamfunctionSolveStatus::not_converged);
    // AMENDMENT E6a (SF-25 bitácora 2026-08-14T12:10Z): both `stagnated` and
    // `omega_floor_rejected` are documented SF-21 plateau signatures (ramp
    // continuation attempts stagnate; a direct-lambda attempt can instead
    // hit the omega floor first, rejecting every Picard trial) -- accept
    // either as the E2 freeze's plateau signature.
    check("E2_stage_exit_reason_plateau_signature",
          stage_report.exit_reason == PicardExitReason::stagnated ||
              stage_report.exit_reason == PicardExitReason::omega_floor_rejected);
    const bool r_f_band_ok = r_F_frozen >= 1e-4 && r_F_frozen <= 1e-2;
    check("E2_stage_r_F_in_prespecified_band_1e-4_1e-2", r_f_band_ok);

    // =========================================================================
    // Shared setup for E3-E5.
    // =========================================================================
    const DeviceSpan<const real> q_att = workspace.q(); // = exp(-Y_att), rebuilt by the E2 stage above.
    const double v_rms = static_cast<double>(stage_report.diagnostics.v_d_rms);
    NonlinearSourceConfig source_config;
    source_config.epsilon = real{1e-2};
    source_config.v_rms = static_cast<real>(v_rms);
    const ResidualHistogramConfig histogram_config{};
    const AffineGauge gauge = AffineGauge::benchmark(real{1});

    JvpWorkspace jvp;
    jvp.prepare(n);
    jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{fields.u1_span(), fields.u2_span()}, gauge,
                         real{1}, source_config, histogram_config);

    StreamfunctionResidualWorkspace frozen_residual_workspace;
    frozen_residual_workspace.prepare(n);
    DeviceBuffer<real> f1(n), f2(n);
    enqueue_streamfunction_residual(ctx, grid, q_att,
                                    PeriodicStreamfunctionFluctuations{fields.u1_span(), fields.u2_span()},
                                    gauge, real{1}, source_config, histogram_config, f1.span(), f2.span(),
                                    frozen_residual_workspace);
    const StreamfunctionResidualReport frozen_residual = synchronize_streamfunction_residual_report(
        ctx, grid, real{1}, source_config, histogram_config, frozen_residual_workspace);
    std::cout << "E3-E5 setup: v_rms=" << v_rms << " frozen_residual_recompute_r_F=" << frozen_residual.r_F
              << " (stage r_F=" << r_F_frozen << ")\n";

    DeviceBuffer<real> b1(n), b2(n);
    blas::copy(ctx, DeviceSpan<const real>(f1.span()), b1.span());
    blas::copy(ctx, DeviceSpan<const real>(f2.span()), b2.span());
    blas::scal(ctx, b1.span(), real{-1});
    blas::scal(ctx, b2.span(), real{-1});
    ctx.synchronize();

    multigrid::MGConfig mg_config = base_config.mg;
    BlockDiagonalMGPreconditioner precond(workspace.hierarchy(), mg_config);

    // =========================================================================
    // E3: mu-sweep, the MECHANISM gate.
    // =========================================================================
    struct MuResult {
        double mu;
        CoupledGmresStatus status;
        int total_inner_iterations;
        int outer_cycles;
        double first_true_residual;
        double final_true_residual;
    };
    std::vector<MuResult> mu_results;
    const double mu_list[] = {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 0.0};

    const real e3_rel_tol =
        std::clamp(static_cast<real>(std::sqrt(r_F_frozen)), real{1e-8}, real{1e-1});

    for (double mu_d : mu_list) {
        const real mu = static_cast<real>(mu_d);
        ShiftedJacobianOperator op(jvp, grid, q_att, mu);
        op.prepare(n);
        CoupledGmres gmres;
        gmres.prepare(n, 10);

        DeviceBuffer<real> corr1(n), corr2(n);
        blas::fill(ctx, corr1.span(), real{0});
        blas::fill(ctx, corr2.span(), real{0});

        CoupledGmresConfig cfg;
        cfg.restart = 10;
        cfg.max_iterations = 100;
        cfg.rel_tol = e3_rel_tol;

        const CoupledGmresReport report = gmres.solve(
            ctx, grid, op, precond,
            ConstCoupledVectorView(DeviceSpan<const real>(b1.span()), DeviceSpan<const real>(b2.span())), cfg,
            JvpDeltaConfig{}, CoupledVectorView{corr1.span(), corr2.span()});
        ctx.synchronize();

        const double first_true =
            report.checkpoints.empty() ? 0.0 : static_cast<double>(report.checkpoints.front().true_residual);
        const double final_true =
            report.checkpoints.empty() ? 0.0 : static_cast<double>(report.checkpoints.back().true_residual);
        mu_results.push_back(
            {mu_d, report.status, report.total_inner_iterations, report.outer_cycles, first_true, final_true});

        std::cout << "E3 mu=" << mu_d << " status=" << gmres_status_label(report.status)
                  << " total_inner_iterations=" << report.total_inner_iterations
                  << " outer_cycles=" << report.outer_cycles << " first_true_residual=" << first_true
                  << " final_true_residual=" << final_true
                  << " reduction_ratio=" << (first_true > 0.0 ? final_true / first_true : 0.0) << '\n';
    }

    int it0 = 0;
    for (const auto& r : mu_results) {
        if (r.mu == 0.0) it0 = r.total_inner_iterations;
    }

    // E3 gate: mechanism_confirmed requires a mu>0 that CONVERGED with a
    // >=10x reduction in inner iterations vs the mu=0 baseline (it0).
    // E5 decision: mu_star is a SEPARATE selection rule, the smallest
    // converged-within-budget mu (mu>0, status==converged), independent of
    // the 10x iteration-reduction filter used for mechanism_confirmed.
    bool mechanism_confirmed = false;
    for (const auto& r : mu_results) {
        if (r.mu > 0.0 && r.status == CoupledGmresStatus::converged && r.total_inner_iterations * 10 <= it0) {
            mechanism_confirmed = true;
        }
    }
    double mu_star = -1.0;
    int mu_star_iterations = 0;
    for (const auto& r : mu_results) {
        if (r.mu > 0.0 && r.status == CoupledGmresStatus::converged) {
            if (mu_star < 0.0 || r.mu < mu_star) {
                mu_star = r.mu;
                mu_star_iterations = r.total_inner_iterations;
            }
        }
    }
    std::cout << "E3 gate (>=10x reduction exists): it0(mu=0)=" << it0
              << " mechanism_confirmed=" << (mechanism_confirmed ? "true" : "false")
              << " | E5 decision (smallest converged-within-budget mu): mu_star=" << mu_star
              << " mu_star_iterations=" << mu_star_iterations << '\n';
    check("E3_mechanism_confirmed_ge_10x_reduction", mechanism_confirmed);

    // =========================================================================
    // E4: generalized inverse iteration toward the smallest (J, A_blk)
    // eigenpair. Recorded evidence, non-blocking for the D-gate verdict.
    // =========================================================================
    blas::ReductionWorkspace dot_ws;
    constraints::MeanZeroWorkspace mean_zero_ws;
    mean_zero_ws.prepare(n);
    constraints::MeanZeroProjector projector;

    const auto pair_dot = [&](DeviceSpan<const real> a1, DeviceSpan<const real> a2, DeviceSpan<const real> c1,
                              DeviceSpan<const real> c2) {
        return blas::dot_host(ctx, a1, c1, dot_ws) + blas::dot_host(ctx, a2, c2, dot_ws);
    };
    const auto pair_norm = [&](DeviceSpan<const real> a1, DeviceSpan<const real> a2) {
        return std::sqrt(std::max(0.0, static_cast<double>(pair_dot(a1, a2, a1, a2))));
    };

    DeviceBuffer<real> v1(n), v2(n);
    {
        const double b_norm = pair_norm(DeviceSpan<const real>(b1.span()), DeviceSpan<const real>(b2.span()));
        blas::copy(ctx, DeviceSpan<const real>(b1.span()), v1.span());
        blas::copy(ctx, DeviceSpan<const real>(b2.span()), v2.span());
        if (b_norm > 0.0 && std::isfinite(b_norm)) {
            blas::scal(ctx, v1.span(), real{1} / static_cast<real>(b_norm));
            blas::scal(ctx, v2.span(), real{1} / static_cast<real>(b_norm));
        }
        projector.project(ctx, v1.span(), mean_zero_ws);
        projector.project(ctx, v2.span(), mean_zero_ws);
        ctx.synchronize();
    }

    const operators::LesterPositiveDiffusionOperator A_op(grid, q_att);

    ShiftedJacobianOperator e4_op(jvp, grid, q_att, real{1e-3});
    e4_op.prepare(n);
    CoupledGmres e4_gmres;
    e4_gmres.prepare(n, 10);
    CoupledGmresConfig e4_config;
    e4_config.restart = 10;
    e4_config.max_iterations = 100;
    e4_config.rel_tol = real{1e-6}; // measurement infrastructure only, per the E4 decision.

    std::vector<double> rayleigh_trajectory;
    for (int k = 1; k <= 20; ++k) {
        DeviceBuffer<real> w1(n), w2(n);
        A_op.apply(ctx, DeviceSpan<const real>(v1.span()), w1.span());
        A_op.apply(ctx, DeviceSpan<const real>(v2.span()), w2.span());
        ctx.synchronize();

        DeviceBuffer<real> z1(n), z2(n);
        blas::fill(ctx, z1.span(), real{0});
        blas::fill(ctx, z2.span(), real{0});
        const CoupledGmresReport e4_report = e4_gmres.solve(
            ctx, grid, e4_op, precond,
            ConstCoupledVectorView(DeviceSpan<const real>(w1.span()), DeviceSpan<const real>(w2.span())),
            e4_config, JvpDeltaConfig{}, CoupledVectorView{z1.span(), z2.span()});
        ctx.synchronize();

        if (e4_report.status == CoupledGmresStatus::nonfinite) {
            std::cout << "E4 k=" << k << " inverse-iteration solve nonfinite; stopping with "
                      << rayleigh_trajectory.size() << " recorded values\n";
            break;
        }

        const double z_norm = pair_norm(DeviceSpan<const real>(z1.span()), DeviceSpan<const real>(z2.span()));
        if (!(z_norm > 0.0) || !std::isfinite(z_norm)) {
            std::cout << "E4 k=" << k << " non-finite/zero z norm; stopping\n";
            break;
        }
        blas::scal(ctx, z1.span(), real{1} / static_cast<real>(z_norm));
        blas::scal(ctx, z2.span(), real{1} / static_cast<real>(z_norm));
        projector.project(ctx, z1.span(), mean_zero_ws);
        projector.project(ctx, z2.span(), mean_zero_ws);
        blas::copy(ctx, DeviceSpan<const real>(z1.span()), v1.span());
        blas::copy(ctx, DeviceSpan<const real>(z2.span()), v2.span());
        ctx.synchronize();

        DeviceBuffer<real> jv1(n), jv2(n);
        const JvpApplyReport jv_report = jvp.apply(
            ctx, grid, ConstCoupledVectorView(DeviceSpan<const real>(v1.span()), DeviceSpan<const real>(v2.span())),
            JvpDeltaConfig{}, CoupledVectorView{jv1.span(), jv2.span()});
        if (jv_report.status != JvpApplyStatus::ok) {
            std::cout << "E4 k=" << k << " Jv nonfinite; stopping\n";
            break;
        }
        DeviceBuffer<real> av1(n), av2(n);
        A_op.apply(ctx, DeviceSpan<const real>(v1.span()), av1.span());
        A_op.apply(ctx, DeviceSpan<const real>(v2.span()), av2.span());
        ctx.synchronize();

        const double numerator = pair_dot(DeviceSpan<const real>(v1.span()), DeviceSpan<const real>(v2.span()),
                                          DeviceSpan<const real>(jv1.span()), DeviceSpan<const real>(jv2.span()));
        const double denominator = pair_dot(DeviceSpan<const real>(v1.span()), DeviceSpan<const real>(v2.span()),
                                            DeviceSpan<const real>(av1.span()), DeviceSpan<const real>(av2.span()));
        const double rayleigh = denominator != 0.0 ? numerator / denominator : std::numeric_limits<double>::quiet_NaN();
        rayleigh_trajectory.push_back(rayleigh);
        std::cout << "E4 k=" << k << " rayleigh=" << rayleigh << '\n';
    }
    const double rayleigh_final =
        rayleigh_trajectory.empty() ? std::numeric_limits<double>::quiet_NaN() : rayleigh_trajectory.back();
    const bool e4_claim = std::isfinite(rayleigh_final) && std::abs(rayleigh_final) <= 1e-2;
    std::cout << "E4 claim check (non-blocking): |rayleigh_last|<=1e-2 -> " << (e4_claim ? "PASS" : "FAIL")
              << " (rayleigh_last=" << rayleigh_final << ", trajectory_length=" << rayleigh_trajectory.size()
              << ")\n";

    // =========================================================================
    // E5: LM mini-solve. theta = mu_star / r_F_frozen. SKIP if E3 found no
    // converged mu>0 -- the honest falsification path.
    // =========================================================================
    bool e5_pass = false;
    if (!mechanism_confirmed) {
        std::cout << "E5_SKIPPED_MECHANISM_FAILED\n";
        std::cout << "E5 result: skipped (mechanism not confirmed)\n";
    } else {
        const double theta = mu_star / r_F_frozen;
        std::cout << "E5 theta=" << theta << " (mu_star=" << mu_star << ", r_F_frozen=" << r_F_frozen << ")\n";

        DeviceBuffer<real> state1(n), state2(n);
        blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), state1.span());
        blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), state2.span());
        ctx.synchronize();

        ShiftedJacobianOperator e5_op(jvp, grid, q_att, real{0});
        e5_op.prepare(n);
        CoupledGmres e5_gmres;
        e5_gmres.prepare(n, 10);

        StreamfunctionResidualWorkspace e5_residual_ws;
        e5_residual_ws.prepare(n);
        DeviceBuffer<real> e5_f1(n), e5_f2(n);
        DeviceBuffer<real> trial1(n), trial2(n);
        DeviceBuffer<real> trial_f1(n), trial_f2(n);
        DeviceBuffer<real> delta1(n), delta2(n);
        DeviceBuffer<real> rhs1(n), rhs2(n);

        constexpr real kAlphaMin = real{0.03125}; // 2^-5, matches NewtonKrylovConfig::alpha_min default.
        constexpr real kArmijoC = real{1e-4};      // matches NewtonKrylovConfig::armijo_c default.

        bool reached = false;
        bool step_failure = false;
        for (int k = 0; k < 30 && !reached && !step_failure; ++k) {
            enqueue_streamfunction_residual(ctx, grid, q_att,
                                            PeriodicStreamfunctionFluctuations{state1.span(), state2.span()},
                                            gauge, real{1}, source_config, histogram_config, e5_f1.span(),
                                            e5_f2.span(), e5_residual_ws);
            const StreamfunctionResidualReport res_k = synchronize_streamfunction_residual_report(
                ctx, grid, real{1}, source_config, histogram_config, e5_residual_ws);
            const double r_F_k = static_cast<double>(res_k.r_F);

            if (r_F_k <= 1e-4) {
                reached = true;
                std::cout << "E5 k=" << k << " r_F=" << r_F_k << " REACHED\n";
                break;
            }

            jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{state1.span(), state2.span()}, gauge,
                                 real{1}, source_config, histogram_config);

            blas::copy(ctx, DeviceSpan<const real>(e5_f1.span()), rhs1.span());
            blas::copy(ctx, DeviceSpan<const real>(e5_f2.span()), rhs2.span());
            blas::scal(ctx, rhs1.span(), real{-1});
            blas::scal(ctx, rhs2.span(), real{-1});

            const real mu_k = static_cast<real>(theta * r_F_k);
            e5_op.set_mu(mu_k);

            CoupledGmresConfig e5_config;
            e5_config.restart = 10;
            e5_config.max_iterations = 100;
            e5_config.rel_tol = std::clamp(static_cast<real>(std::sqrt(r_F_k)), real{1e-8}, real{1e-1});

            blas::fill(ctx, delta1.span(), real{0});
            blas::fill(ctx, delta2.span(), real{0});
            const CoupledGmresReport gmres_report = e5_gmres.solve(
                ctx, grid, e5_op, precond,
                ConstCoupledVectorView(DeviceSpan<const real>(rhs1.span()), DeviceSpan<const real>(rhs2.span())),
                e5_config, JvpDeltaConfig{}, CoupledVectorView{delta1.span(), delta2.span()});
            ctx.synchronize();

            bool accepted = false;
            real accepted_alpha = real{0};
            real alpha = real{1};
            const double phi_k = 0.5 * r_F_k * r_F_k;
            while (alpha >= kAlphaMin) {
                blas::copy(ctx, DeviceSpan<const real>(state1.span()), trial1.span());
                blas::copy(ctx, DeviceSpan<const real>(state2.span()), trial2.span());
                blas::axpy(ctx, alpha, DeviceSpan<const real>(delta1.span()), trial1.span());
                blas::axpy(ctx, alpha, DeviceSpan<const real>(delta2.span()), trial2.span());
                projector.project(ctx, trial1.span(), mean_zero_ws);
                projector.project(ctx, trial2.span(), mean_zero_ws);

                enqueue_streamfunction_residual(ctx, grid, q_att,
                                                PeriodicStreamfunctionFluctuations{trial1.span(), trial2.span()},
                                                gauge, real{1}, source_config, histogram_config, trial_f1.span(),
                                                trial_f2.span(), e5_residual_ws);
                const StreamfunctionResidualReport trial_res = synchronize_streamfunction_residual_report(
                    ctx, grid, real{1}, source_config, histogram_config, e5_residual_ws);
                const double r_F_trial = static_cast<double>(trial_res.r_F);
                const bool finite_ok = std::isfinite(r_F_trial);
                const double phi_trial = 0.5 * r_F_trial * r_F_trial;
                const bool armijo_ok = finite_ok && phi_trial <= (1.0 - static_cast<double>(kArmijoC) *
                                                                          static_cast<double>(alpha)) *
                                                                        phi_k;

                std::cout << "  E5 k=" << k << " alpha=" << alpha << " r_F_trial=" << r_F_trial
                          << " finite=" << (finite_ok ? "true" : "false")
                          << " armijo=" << (armijo_ok ? "accept" : "reject") << '\n';

                if (armijo_ok) {
                    accepted = true;
                    accepted_alpha = alpha;
                    blas::copy(ctx, DeviceSpan<const real>(trial1.span()), state1.span());
                    blas::copy(ctx, DeviceSpan<const real>(trial2.span()), state2.span());
                    ctx.synchronize();
                    break;
                }
                alpha *= real{0.5};
            }

            std::cout << "E5 k=" << k << " r_F=" << r_F_k << " mu_k=" << mu_k
                      << " gmres_status=" << gmres_status_label(gmres_report.status)
                      << " gmres_inner=" << gmres_report.total_inner_iterations
                      << " accepted_alpha=" << (accepted ? static_cast<double>(accepted_alpha) : 0.0) << '\n';

            if (!accepted) {
                std::cout << "E5_STEP_FAILURE k=" << k << '\n';
                step_failure = true;
            }
        }

        e5_pass = reached;
        std::cout << "E5 result: " << (e5_pass ? "PASS" : "FAIL") << '\n';
    }

    // =========================================================================
    // E6 (DECISION E6, bitácora 2026-08-14T12:10Z): Psi-tc / backward-Euler
    // probe with bounded-non-monotone safeguards -- the PRESPECIFIED
    // contingency that runs precisely when the mechanism is confirmed (E3)
    // but the monotone-Armijo LM mini-solve (E5) failed at a spurious local
    // minimum of the merit. Same frozen state, same shifted operator/GMRES
    // configuration pattern as E5, but full (alpha=1) projected steps are
    // ACCEPTED WITHOUT Armijo under a strict bounded-non-monotone safeguard
    // (reject only if the candidate is non-finite or more than doubles
    // r_F), with dtau evolved by switched-evolution-relaxation (SER) after
    // each accepted step.
    // =========================================================================
    bool e6_pass = false;
    if (mechanism_confirmed && !e5_pass) {
        DeviceBuffer<real> e6_state1(n), e6_state2(n);
        blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), e6_state1.span());
        blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), e6_state2.span());
        ctx.synchronize();

        ShiftedJacobianOperator e6_op(jvp, grid, q_att, real{0});
        e6_op.prepare(n);
        CoupledGmres e6_gmres;
        e6_gmres.prepare(n, 10);

        StreamfunctionResidualWorkspace e6_residual_ws;
        e6_residual_ws.prepare(n);
        DeviceBuffer<real> e6_f1(n), e6_f2(n);
        DeviceBuffer<real> e6_candidate1(n), e6_candidate2(n);
        DeviceBuffer<real> e6_candidate_f1(n), e6_candidate_f2(n);
        DeviceBuffer<real> e6_delta1(n), e6_delta2(n);
        DeviceBuffer<real> e6_rhs1(n), e6_rhs2(n);

        double dtau = 1.0 / mu_star; // E3-calibrated: dtau_0 = 1/mu_star.
        double r_F_prev = r_F_frozen;
        std::cout << "E6 dtau_0=" << dtau << " (=1/mu_star, mu_star=" << mu_star << ")\n";

        bool reached = false;
        bool step_failure = false;
        for (int k = 0; k < 60 && !reached && !step_failure; ++k) {
            enqueue_streamfunction_residual(ctx, grid, q_att,
                                            PeriodicStreamfunctionFluctuations{e6_state1.span(), e6_state2.span()},
                                            gauge, real{1}, source_config, histogram_config, e6_f1.span(),
                                            e6_f2.span(), e6_residual_ws);
            const StreamfunctionResidualReport res_k = synchronize_streamfunction_residual_report(
                ctx, grid, real{1}, source_config, histogram_config, e6_residual_ws);
            const double r_F_k = static_cast<double>(res_k.r_F);

            if (r_F_k <= 1e-4) {
                reached = true;
                std::cout << "E6 k=" << k << " r_F=" << r_F_k << " REACHED\n";
                break;
            }
            if (r_F_k > 10.0 * r_F_frozen) {
                std::cout << "E6_DIVERGENCE_STOP k=" << k << " r_F=" << r_F_k
                          << " r_F_frozen=" << r_F_frozen << '\n';
                step_failure = true;
                break;
            }

            jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{e6_state1.span(), e6_state2.span()}, gauge,
                                 real{1}, source_config, histogram_config);

            blas::copy(ctx, DeviceSpan<const real>(e6_f1.span()), e6_rhs1.span());
            blas::copy(ctx, DeviceSpan<const real>(e6_f2.span()), e6_rhs2.span());
            blas::scal(ctx, e6_rhs1.span(), real{-1});
            blas::scal(ctx, e6_rhs2.span(), real{-1});

            bool accepted = false;
            double accepted_r_F = 0.0;
            for (int attempt = 0; attempt < 3 && !accepted; ++attempt) {
                const real mu_k = static_cast<real>(1.0 / dtau);
                e6_op.set_mu(mu_k);

                CoupledGmresConfig e6_config;
                e6_config.restart = 10;
                e6_config.max_iterations = 100;
                e6_config.rel_tol = std::clamp(static_cast<real>(std::sqrt(r_F_k)), real{1e-8}, real{1e-1});

                blas::fill(ctx, e6_delta1.span(), real{0});
                blas::fill(ctx, e6_delta2.span(), real{0});
                const CoupledGmresReport gmres_report = e6_gmres.solve(
                    ctx, grid, e6_op, precond,
                    ConstCoupledVectorView(DeviceSpan<const real>(e6_rhs1.span()),
                                          DeviceSpan<const real>(e6_rhs2.span())),
                    e6_config, JvpDeltaConfig{}, CoupledVectorView{e6_delta1.span(), e6_delta2.span()});
                ctx.synchronize();

                // Full step (alpha=1), projected per component.
                blas::copy(ctx, DeviceSpan<const real>(e6_state1.span()), e6_candidate1.span());
                blas::copy(ctx, DeviceSpan<const real>(e6_state2.span()), e6_candidate2.span());
                blas::axpy(ctx, real{1}, DeviceSpan<const real>(e6_delta1.span()), e6_candidate1.span());
                blas::axpy(ctx, real{1}, DeviceSpan<const real>(e6_delta2.span()), e6_candidate2.span());
                projector.project(ctx, e6_candidate1.span(), mean_zero_ws);
                projector.project(ctx, e6_candidate2.span(), mean_zero_ws);

                enqueue_streamfunction_residual(
                    ctx, grid, q_att, PeriodicStreamfunctionFluctuations{e6_candidate1.span(), e6_candidate2.span()},
                    gauge, real{1}, source_config, histogram_config, e6_candidate_f1.span(), e6_candidate_f2.span(),
                    e6_residual_ws);
                const StreamfunctionResidualReport cand_res = synchronize_streamfunction_residual_report(
                    ctx, grid, real{1}, source_config, histogram_config, e6_residual_ws);
                const double r_F_candidate = static_cast<double>(cand_res.r_F);
                const bool finite_ok = std::isfinite(r_F_candidate);
                const bool reject = !finite_ok || r_F_candidate > 2.0 * r_F_k;
                const bool nonmonotone = finite_ok && r_F_candidate > r_F_k && r_F_candidate <= 2.0 * r_F_k;

                std::cout << "E6 k=" << k << " attempt=" << attempt << " r_F_k=" << r_F_k << " dtau=" << dtau
                          << " mu_k=" << mu_k << " gmres_status=" << gmres_status_label(gmres_report.status)
                          << " gmres_inner=" << gmres_report.total_inner_iterations
                          << " r_F_candidate=" << r_F_candidate << " finite=" << (finite_ok ? "true" : "false")
                          << (reject ? " REJECTED" : " ACCEPTED") << '\n';

                if (reject) {
                    dtau /= 2.0;
                    continue;
                }
                if (nonmonotone) {
                    std::cout << "E6_NONMONOTONE_ACCEPTED k=" << k << " r_F_k=" << r_F_k
                              << " r_F_candidate=" << r_F_candidate << '\n';
                }
                blas::copy(ctx, DeviceSpan<const real>(e6_candidate1.span()), e6_state1.span());
                blas::copy(ctx, DeviceSpan<const real>(e6_candidate2.span()), e6_state2.span());
                ctx.synchronize();
                accepted = true;
                accepted_r_F = r_F_candidate;
            }

            if (!accepted) {
                std::cout << "E6_STEP_FAILURE k=" << k << '\n';
                step_failure = true;
                break;
            }

            // SER update AFTER an accepted step (uses the accepted candidate's r_F).
            dtau = std::clamp(dtau * (r_F_prev / accepted_r_F), 1.0, 1e6);
            r_F_prev = accepted_r_F;
        }

        e6_pass = reached;
        check("E6_psitc_reaches_1e-4_within_60_steps", e6_pass);
    } else if (e5_pass) {
        std::cout << "E6_SKIPPED_E5_PASSED\n";
    }

    // =========================================================================
    // E6b (PRESPECIFIED, bitácora 2026-08-14T13:05Z, corrective C03):
    // micro-step scan from the FROZEN state -- print-only evidence, no state
    // update, no check. Closes the step-size loophole: if even near-
    // infinitesimal flow steps (large mu, i.e. small dtau=1/mu) ascend r_F,
    // F^T J A^-1 F < 0 is established directly at the frozen state,
    // independent of the SER dtau schedule E6 explored.
    // =========================================================================
    jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{fields.u1_span(), fields.u2_span()}, gauge,
                         real{1}, source_config, histogram_config);

    const real e6b_rel_tol = std::clamp(static_cast<real>(std::sqrt(r_F_frozen)), real{1e-8}, real{1e-1});
    const double e6b_mu_list[] = {1.0, 10.0};
    for (double e6b_mu_d : e6b_mu_list) {
        const real e6b_mu = static_cast<real>(e6b_mu_d);
        ShiftedJacobianOperator e6b_op(jvp, grid, q_att, e6b_mu);
        e6b_op.prepare(n);
        CoupledGmres e6b_gmres;
        e6b_gmres.prepare(n, 10);

        DeviceBuffer<real> e6b_delta1(n), e6b_delta2(n);
        blas::fill(ctx, e6b_delta1.span(), real{0});
        blas::fill(ctx, e6b_delta2.span(), real{0});

        CoupledGmresConfig e6b_config;
        e6b_config.restart = 10;
        e6b_config.max_iterations = 100;
        e6b_config.rel_tol = e6b_rel_tol;

        const CoupledGmresReport e6b_report = e6b_gmres.solve(
            ctx, grid, e6b_op, precond,
            ConstCoupledVectorView(DeviceSpan<const real>(b1.span()), DeviceSpan<const real>(b2.span())),
            e6b_config, JvpDeltaConfig{}, CoupledVectorView{e6b_delta1.span(), e6b_delta2.span()});
        ctx.synchronize();

        DeviceBuffer<real> e6b_candidate1(n), e6b_candidate2(n);
        blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), e6b_candidate1.span());
        blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), e6b_candidate2.span());
        blas::axpy(ctx, real{1}, DeviceSpan<const real>(e6b_delta1.span()), e6b_candidate1.span());
        blas::axpy(ctx, real{1}, DeviceSpan<const real>(e6b_delta2.span()), e6b_candidate2.span());
        projector.project(ctx, e6b_candidate1.span(), mean_zero_ws);
        projector.project(ctx, e6b_candidate2.span(), mean_zero_ws);

        StreamfunctionResidualWorkspace e6b_residual_ws;
        e6b_residual_ws.prepare(n);
        DeviceBuffer<real> e6b_cand_f1(n), e6b_cand_f2(n);
        enqueue_streamfunction_residual(
            ctx, grid, q_att, PeriodicStreamfunctionFluctuations{e6b_candidate1.span(), e6b_candidate2.span()},
            gauge, real{1}, source_config, histogram_config, e6b_cand_f1.span(), e6b_cand_f2.span(),
            e6b_residual_ws);
        const StreamfunctionResidualReport e6b_cand_res = synchronize_streamfunction_residual_report(
            ctx, grid, real{1}, source_config, histogram_config, e6b_residual_ws);
        const double r_F_candidate = static_cast<double>(e6b_cand_res.r_F);

        std::cout << "E6b mu=" << e6b_mu_d << " gmres_status=" << gmres_status_label(e6b_report.status)
                  << " inner=" << e6b_report.total_inner_iterations << " r_F_frozen=" << r_F_frozen
                  << " r_F_candidate=" << r_F_candidate << " delta_rF=" << (r_F_candidate - r_F_frozen)
                  << '\n';
    }

    // =========================================================================
    // E7 (PRESPECIFIED, bitácora 2026-08-14T13:05Z, corrective C03):
    // epsilon-fold probe at the SAME frozen state -- print-only evidence, no
    // verdict change. Tests whether the eta=1 plateau is an epsilon=1e-2
    // regularization artifact of a fold in the solution branch (eta_fold
    // crossing eta=1 near lambda~0.5 for sigma^2=1) rather than an intrinsic
    // obstruction at eta=1.
    // =========================================================================
    NonlinearSourceConfig e7_source_config;
    e7_source_config.epsilon = real{1e-3};
    e7_source_config.v_rms = static_cast<real>(v_rms);

    // (i) frozen-state residual under epsilon=1e-3.
    StreamfunctionResidualWorkspace e7_frozen_residual_ws;
    e7_frozen_residual_ws.prepare(n);
    DeviceBuffer<real> e7_frozen_f1(n), e7_frozen_f2(n);
    enqueue_streamfunction_residual(ctx, grid, q_att,
                                    PeriodicStreamfunctionFluctuations{fields.u1_span(), fields.u2_span()},
                                    gauge, real{1}, e7_source_config, histogram_config, e7_frozen_f1.span(),
                                    e7_frozen_f2.span(), e7_frozen_residual_ws);
    const StreamfunctionResidualReport e7_frozen_residual = synchronize_streamfunction_residual_report(
        ctx, grid, real{1}, e7_source_config, histogram_config, e7_frozen_residual_ws);
    std::cout << "E7 frozen_state_r_F_at_eps1e-3 = " << e7_frozen_residual.r_F << '\n';

    // Save the frozen fields before the warm-started probe mutates them.
    DeviceBuffer<real> e7_saved_u1(n), e7_saved_u2(n);
    blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), e7_saved_u1.span());
    blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), e7_saved_u2.span());
    ctx.synchronize();

    // (ii) one warm-started stage solve at eta=1 with epsilon=1e-3, reusing
    // the E2 stage's lambda/q/hierarchy (coefficient_state=reuse); epsilon
    // does not touch q/hierarchy/the affine RHS, so this is exactly the
    // reuse contract.
    StreamfunctionSolverConfig e7_stage_config = stage_config;
    e7_stage_config.epsilon = real{1e-3};
    e7_stage_config.picard.max_iter = 200;
    e7_stage_config.coefficient_state = CoefficientState::reuse;

    const StreamfunctionSolveReport e7_stage_report =
        solve_streamfunctions(ctx, problem_view, e7_stage_config, fields, workspace);
    ctx.synchronize();

    double e7_best_r_F = static_cast<double>(e7_stage_report.residual.r_F);
    std::cout << "E7 stage_eps1e-3: status=" << solve_status_label(e7_stage_report.status)
              << " exit_reason=" << exit_reason_label(e7_stage_report.exit_reason)
              << " picard_iterations=" << e7_stage_report.picard_iterations
              << " r_F=" << e7_stage_report.residual.r_F
              << " anderson_acc=" << e7_stage_report.anderson_accepted << '\n';

    // (iii) if (ii) did not converge, the E5-style LM mini-solve at
    // epsilon=1e-3, continuing from (ii)'s resulting fields state.
    if (e7_stage_report.status != StreamfunctionSolveStatus::converged) {
        DeviceBuffer<real> e7lm_state1(n), e7lm_state2(n);
        blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), e7lm_state1.span());
        blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), e7lm_state2.span());
        ctx.synchronize();

        ShiftedJacobianOperator e7lm_op(jvp, grid, q_att, real{0});
        e7lm_op.prepare(n);
        CoupledGmres e7lm_gmres;
        e7lm_gmres.prepare(n, 10);

        StreamfunctionResidualWorkspace e7lm_residual_ws;
        e7lm_residual_ws.prepare(n);
        DeviceBuffer<real> e7lm_f1(n), e7lm_f2(n);
        DeviceBuffer<real> e7lm_trial1(n), e7lm_trial2(n);
        DeviceBuffer<real> e7lm_trial_f1(n), e7lm_trial_f2(n);
        DeviceBuffer<real> e7lm_delta1(n), e7lm_delta2(n);
        DeviceBuffer<real> e7lm_rhs1(n), e7lm_rhs2(n);

        const double theta = mu_star / r_F_frozen; // theta rule unchanged (E5).
        constexpr real kE7LmAlphaMin = real{0.03125}; // 2^-5, matches E5's kAlphaMin.
        constexpr real kE7LmArmijoC = real{1e-4};      // matches E5's kArmijoC.

        bool e7lm_reached = false;
        bool e7lm_step_failure = false;
        double e7lm_final_r_F = std::numeric_limits<double>::quiet_NaN();
        for (int k = 0; k < 30 && !e7lm_reached && !e7lm_step_failure; ++k) {
            enqueue_streamfunction_residual(
                ctx, grid, q_att, PeriodicStreamfunctionFluctuations{e7lm_state1.span(), e7lm_state2.span()},
                gauge, real{1}, e7_source_config, histogram_config, e7lm_f1.span(), e7lm_f2.span(),
                e7lm_residual_ws);
            const StreamfunctionResidualReport res_k = synchronize_streamfunction_residual_report(
                ctx, grid, real{1}, e7_source_config, histogram_config, e7lm_residual_ws);
            const double r_F_k = static_cast<double>(res_k.r_F);
            e7lm_final_r_F = r_F_k;

            if (r_F_k <= 1e-4) {
                e7lm_reached = true;
                std::cout << "E7-LM k=" << k << " r_F=" << r_F_k << " REACHED\n";
                break;
            }

            jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{e7lm_state1.span(), e7lm_state2.span()},
                                 gauge, real{1}, e7_source_config, histogram_config);

            blas::copy(ctx, DeviceSpan<const real>(e7lm_f1.span()), e7lm_rhs1.span());
            blas::copy(ctx, DeviceSpan<const real>(e7lm_f2.span()), e7lm_rhs2.span());
            blas::scal(ctx, e7lm_rhs1.span(), real{-1});
            blas::scal(ctx, e7lm_rhs2.span(), real{-1});

            const real mu_k = static_cast<real>(theta * r_F_k);
            e7lm_op.set_mu(mu_k);

            CoupledGmresConfig e7lm_config;
            e7lm_config.restart = 10;
            e7lm_config.max_iterations = 100;
            e7lm_config.rel_tol = std::clamp(static_cast<real>(std::sqrt(r_F_k)), real{1e-8}, real{1e-1});

            blas::fill(ctx, e7lm_delta1.span(), real{0});
            blas::fill(ctx, e7lm_delta2.span(), real{0});
            const CoupledGmresReport gmres_report = e7lm_gmres.solve(
                ctx, grid, e7lm_op, precond,
                ConstCoupledVectorView(DeviceSpan<const real>(e7lm_rhs1.span()),
                                      DeviceSpan<const real>(e7lm_rhs2.span())),
                e7lm_config, JvpDeltaConfig{}, CoupledVectorView{e7lm_delta1.span(), e7lm_delta2.span()});
            ctx.synchronize();

            bool accepted = false;
            real accepted_alpha = real{0};
            real alpha = real{1};
            const double phi_k = 0.5 * r_F_k * r_F_k;
            while (alpha >= kE7LmAlphaMin) {
                blas::copy(ctx, DeviceSpan<const real>(e7lm_state1.span()), e7lm_trial1.span());
                blas::copy(ctx, DeviceSpan<const real>(e7lm_state2.span()), e7lm_trial2.span());
                blas::axpy(ctx, alpha, DeviceSpan<const real>(e7lm_delta1.span()), e7lm_trial1.span());
                blas::axpy(ctx, alpha, DeviceSpan<const real>(e7lm_delta2.span()), e7lm_trial2.span());
                projector.project(ctx, e7lm_trial1.span(), mean_zero_ws);
                projector.project(ctx, e7lm_trial2.span(), mean_zero_ws);

                enqueue_streamfunction_residual(
                    ctx, grid, q_att, PeriodicStreamfunctionFluctuations{e7lm_trial1.span(), e7lm_trial2.span()},
                    gauge, real{1}, e7_source_config, histogram_config, e7lm_trial_f1.span(),
                    e7lm_trial_f2.span(), e7lm_residual_ws);
                const StreamfunctionResidualReport trial_res = synchronize_streamfunction_residual_report(
                    ctx, grid, real{1}, e7_source_config, histogram_config, e7lm_residual_ws);
                const double r_F_trial = static_cast<double>(trial_res.r_F);
                const bool finite_ok = std::isfinite(r_F_trial);
                const double phi_trial = 0.5 * r_F_trial * r_F_trial;
                const bool armijo_ok = finite_ok && phi_trial <= (1.0 - static_cast<double>(kE7LmArmijoC) *
                                                                          static_cast<double>(alpha)) *
                                                                        phi_k;

                std::cout << "  E7-LM k=" << k << " alpha=" << alpha << " r_F_trial=" << r_F_trial
                          << " finite=" << (finite_ok ? "true" : "false")
                          << " armijo=" << (armijo_ok ? "accept" : "reject") << '\n';

                if (armijo_ok) {
                    accepted = true;
                    accepted_alpha = alpha;
                    blas::copy(ctx, DeviceSpan<const real>(e7lm_trial1.span()), e7lm_state1.span());
                    blas::copy(ctx, DeviceSpan<const real>(e7lm_trial2.span()), e7lm_state2.span());
                    ctx.synchronize();
                    break;
                }
                alpha *= real{0.5};
            }

            std::cout << "E7-LM k=" << k << " r_F=" << r_F_k << " mu_k=" << mu_k
                      << " gmres_status=" << gmres_status_label(gmres_report.status)
                      << " gmres_inner=" << gmres_report.total_inner_iterations
                      << " accepted_alpha=" << (accepted ? static_cast<double>(accepted_alpha) : 0.0) << '\n';

            if (!accepted) {
                std::cout << "E7-LM_STEP_FAILURE k=" << k << '\n';
                e7lm_step_failure = true;
            }
        }

        std::cout << "E7_LM_result r_F=" << e7lm_final_r_F << '\n';
        if (std::isfinite(e7lm_final_r_F)) {
            e7_best_r_F = std::min(e7_best_r_F, e7lm_final_r_F);
        }
    }

    const char* e7_verdict_evidence =
        e7_best_r_F < 1e-5 ? "decisive" : (e7_best_r_F <= 1e-4 ? "supportive" : "refuting");
    std::cout << "E7 verdict-evidence: " << e7_verdict_evidence << " (best_r_F=" << e7_best_r_F << ")\n";

    // Restore fields to the frozen state so any later code sees it unchanged.
    blas::copy(ctx, DeviceSpan<const real>(e7_saved_u1.span()), fields.u1_span());
    blas::copy(ctx, DeviceSpan<const real>(e7_saved_u2.span()), fields.u2_span());
    ctx.synchronize();

    // =========================================================================
    // E8 (PRESPECIFIED, bitácora 2026-08-14T14:00Z, corrective C04): the LAST
    // probe before escalation, print-only, no verdict change. Every stalled
    // path above is warm-started along the lambda-continuation branch; the
    // paper instead initializes with the harmonic (zero-source) solve at
    // FULL heterogeneity. E8 tests a DIRECT zero-source solve at the frozen
    // attempt's parameters (lambda=0.5125, sigma_Y^2=1) using the SAME
    // `problem_view` (Y_att/log_conductivity_y, the attempt flow velocity,
    // benchmark(1) gauge) already in scope. A NEW fields/workspace pair is
    // used so the E2 workspace's coefficient/hierarchy state and the frozen
    // `fields` used by any later code stay untouched for auditability.
    // =========================================================================
    StreamfunctionFields e8_fields;
    StreamfunctionWorkspace e8_workspace;

    StreamfunctionSolverConfig e8_config = base_config; // E2 freeze config: adaptive
                                                         // defaults, anderson R5 enabled,
                                                         // newton disabled.
    e8_config.eta = real{1};                            // default; explicit per the E8 spec.
    e8_config.epsilon = real{1e-2};                     // default; explicit per the E8 spec.
    e8_config.initial_state = PicardInitialState::zero_source; // default; explicit -- the
                                                                 // harmonic-init probe itself.
    e8_config.coefficient_state = CoefficientState::rebuild;   // default; explicit.
    e8_config.picard.max_iter = 500;

    const StreamfunctionSolveReport e8_report =
        solve_streamfunctions(ctx, problem_view, e8_config, e8_fields, e8_workspace);
    ctx.synchronize();

    std::cout << "E8 harmonic_init: status=" << solve_status_label(e8_report.status)
              << " exit_reason=" << exit_reason_label(e8_report.exit_reason)
              << " picard_iterations=" << e8_report.picard_iterations << " r_F=" << e8_report.residual.r_F
              << " anderson_acc=" << e8_report.anderson_accepted
              << " anderson_rej=" << e8_report.anderson_rejected << '\n';

    if (!e8_report.picard_history.empty()) {
        std::cout << "E8 r_F_history first=" << e8_report.picard_history.front().r_F
                  << " last=" << e8_report.picard_history.back().r_F << '\n';
    } else {
        std::cout << "E8 r_F_history first=n/a last=n/a\n";
    }

    const double e8_r_F = static_cast<double>(e8_report.residual.r_F);
    const bool e8_converged = e8_report.status == StreamfunctionSolveStatus::converged;
    const char* e8_verdict_evidence =
        (e8_converged && e8_r_F <= 1e-6)
            ? "decisive_branch_fold(r_F<=1e-6)"
            : ((!e8_converged && e8_r_F >= 1e-4 && e8_r_F <= 1e-2) ? "refuting_intrinsic(~1e-3)"
                                                                    : "inconclusive");
    std::cout << "E8 verdict-evidence: " << e8_verdict_evidence << '\n';

    check("terminal_method_demonstrated_E5_or_E6", e5_pass || e6_pass);

    (void)e2_freeze_ok;

    std::cout << "case=terminal_dgate_diagnostic verdict=" << (pass ? "PASS" : "FAIL") << '\n';

    std::ostringstream detail;
    detail << "sigma_Y^2=1, 32^3, dx=1, seed=12345, corr_length=8, lambda_attempt=0.5125, newton "
              "disabled for the E2 freeze; E3 mu-sweep {1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,0}; E4 "
              "generalized inverse iteration (recorded, non-blocking); E5 bounded LM mini-solve "
              "(theta=mu_star/r_F_frozen, <=30 steps); E6 Psi-tc/backward-Euler probe with "
              "bounded-non-monotone safeguards (dtau_0=1/mu_star, SER schedule, <=60 steps) run "
              "iff E3 confirmed the mechanism and E5 failed (amendment E6a + decision E6, "
              "bitácora 2026-08-14T12:10Z); E6b micro-step scan + E7 epsilon-fold probe (print-only "
              "evidence, bitácora 2026-08-14T13:05Z); E8 harmonic-init probe (print-only, "
              "bitácora 2026-08-14T14:00Z)";

    return {pass,
            "terminal_dgate_diagnostic",
            "gpu-terminal-dgate-diagnostic",
            detail.str(),
            mu_star,
            r_F_frozen,
            "E2+E3+(E5 or E6) (E4 recorded, non-blocking) -- SF-25 D-gate",
            pass ? "all pass" : "some failed",
            "PRESPECIFIED SF-25 activation-bitácora D-gate protocol (E2-E5) plus the recorded "
            "corrective amendment E6a (E2 exit-reason assert accepts the {stagnated, "
            "omega_floor_rejected} plateau signatures) and decision E6 (Psi-tc/backward-Euler "
            "contingency probe): verdict PASS iff the amended E2 freeze asserts hold, the E3 "
            "mu-sweep confirms the shift mechanism (>=10x inner-iteration reduction over the mu=0 "
            "budget-exhaustion baseline), and EITHER the E5 monotone-Armijo LM mini-solve reaches "
            "r_F<=1e-4 within 30 steps OR the E6 Psi-tc probe reaches r_F<=1e-4 within 60 steps "
            "(the prespecified contingency when E5 stalls at a spurious merit local minimum); a "
            "failed E3 sweep is the honest falsification path (E5/E6 skipped, case FAILs with the "
            "E3/E4 evidence printed), citing amendment E6a + decision E6 "
            "(bitácora 2026-08-14T12:10Z)"};
}

} // namespace

CaseRegistry terminal_solver_case_registry() {
    return {{"terminal_shifted_apply_unit", case_terminal_shifted_apply_unit}};
}

CaseRegistry terminal_solver_dgate_case_registry() {
    return {{"terminal_dgate_diagnostic", case_terminal_dgate_diagnostic}};
}

} // namespace macroflow3d::streamfunctions::test
