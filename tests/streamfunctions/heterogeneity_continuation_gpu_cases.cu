#include "streamfunction_operator_test_cases.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/ContinuationController.hpp"
#include "src/physics/streamfunctions/StreamfunctionSolver.cuh"
#include "src/physics/streamfunctions/StreamfunctionTypes.hpp"
#include "src/physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "src/physics/streamfunctions/affine_gauge.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// SF-21 (first built pre-re-sequencing as SF-20 T02, ported onto the
// Anderson-bearing solver by SF-21 T02r): `streamfunction_heterogeneity_unit`
// / `streamfunction_heterogeneity_smoke` acceptance cases for
// `run_streamfunction_heterogeneity_continuation`
// (`ContinuationController.hpp/.cu`, the SF-21 CoefficientState extension +
// lambda/rescue continuation driver). Cases 1-4 (host+small-GPU, cheap tier)
// exercise `CoefficientState` and the exact lambda/eta-rescue ordering/
// rollback contract via `StageSolveFn` injection, mirrored from
// `streamfunction_continuation_gpu_cases.cu`'s established injection
// pattern. Case 7 (cheap tier) is the SF-21 re-activation addition: a
// deterministic proof that `solve_streamfunctions` clears the Anderson
// history at solve entry (decision R2a). Cases 5-6 (heavy tier -- see
// `heterogeneity_continuation_smoke_case_registry()`/
// `streamfunction_operator_tests.cpp`'s `heavy_cases()`) are the
// PRESPECIFIED (SF-20 activation bitácora, decision 5(b)) 32^3 physical
// Gaussian smokes: fixed seed 12345, ell=8, sigma_Y^2 in {0.25, 1.0}, GATING
// on reached_target/final_lambda==1/every accepted stage r_F<=1e-6, now also
// running with the SF-20-validated Anderson defaults enabled (SF-21
// re-activation decision R5). These fixtures/gates are implemented
// VERBATIM; no tolerance/fixture/seed/budget here may be adjusted to make a
// failing case pass -- a failure is reported honestly. This file never
// touches `src/**`.

namespace macroflow3d::streamfunctions::test {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

// ---------------------------------------------------------------------------
// Shared GPU fixture helpers, mirrored from streamfunction_continuation_gpu_cases.cu.
// ---------------------------------------------------------------------------

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

[[nodiscard]] StreamfunctionSolverConfig valid_config() { return StreamfunctionSolverConfig{}; }

// Homogeneous (K=1) problem buffers, mirrored from
// streamfunction_continuation_gpu_cases.cu's HomogeneousProblemBuffers.
// Used only for the host-only CoefficientState contract case (Case 1).
struct HomogeneousProblemBuffers {
    DeviceBuffer<real> conductivity;
    DeviceBuffer<real> darcy_u;
    DeviceBuffer<real> darcy_v;
    DeviceBuffer<real> darcy_w;

    explicit HomogeneousProblemBuffers(const Grid3D& grid)
        : conductivity(grid.num_cells()), darcy_u(compact_mac_u_size(grid)),
          darcy_v(compact_mac_v_size(grid)), darcy_w(compact_mac_w_size(grid)) {
        const std::vector<real> ones_cells(grid.num_cells(), real{1});
        const std::vector<real> ones_u(compact_mac_u_size(grid), real{1});
        const std::vector<real> zeros_v(compact_mac_v_size(grid), real{0});
        const std::vector<real> zeros_w(compact_mac_w_size(grid), real{0});
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(conductivity.data(), ones_cells.data(),
                                          ones_cells.size() * sizeof(real), cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_u.data(), ones_u.data(), ones_u.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_v.data(), zeros_v.data(), zeros_v.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_w.data(), zeros_w.data(), zeros_w.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
    }
};

[[nodiscard]] StreamfunctionProblemView homogeneous_problem_view(const Grid3D& grid,
                                                                  const HomogeneousProblemBuffers& buffers) {
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

// Manufactured heterogeneous K = exp(a*sin(2pi x)sin(2pi y)sin(2pi z))
// conductivity field plus the uniform reference Darcy field v=(1,0,0),
// mirrored from streamfunction_anderson_gpu_cases.cu's
// ManufacturedProblemBuffers. Used by the SF-21 entry-clear case (7), which
// needs a real (non-injected) fixture that actually accumulates Anderson
// history.
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

// Manufactured Y = amplitude*sin(2pi x)*sin(2pi y)*sin(2pi z) log-conductivity
// field, mirrored from the picard/continuation ManufacturedProblemBuffers's
// K, but uploaded directly as Y (the SF-21 driver builds K_lambda=exp(lambda*Y)
// internally from a raw log-conductivity buffer).
//
// NOTE: on a dx=dy=dz=1 grid (every injection-only case below), cell centers
// sit at half-integer coordinates, so sin(2*pi*(k+0.5)) == 0 exactly for
// every integer k; the sampled field is EXACTLY zero at every cell center and
// K_lambda == exp(lambda*Y) == 1 identically, regardless of `amplitude`. This
// is intentional and benign for those cases: their contracts (rescue
// ordering, halving/rollback, floor exhaustion) are driven entirely by the
// injected StageSolveFn outcomes, not by the field values themselves.
[[nodiscard]] DeviceBuffer<real> manufactured_y_field(const Grid3D& grid, double amplitude) {
    std::vector<real> y_host(grid.num_cells());
    for (int iz = 0; iz < grid.nz; ++iz) {
        const double z = (iz + 0.5) * static_cast<double>(grid.dz);
        for (int iy = 0; iy < grid.ny; ++iy) {
            const double y = (iy + 0.5) * static_cast<double>(grid.dy);
            for (int ix = 0; ix < grid.nx; ++ix) {
                const double x = (ix + 0.5) * static_cast<double>(grid.dx);
                const std::size_t index = grid.idx(ix, iy, iz);
                y_host[index] = static_cast<real>(
                    amplitude * std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y) * std::sin(2.0 * kPi * z));
            }
        }
    }
    DeviceBuffer<real> y_device(grid.num_cells());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(y_device.data(), y_host.data(), y_host.size() * sizeof(real),
                                      cudaMemcpyHostToDevice));
    return y_device;
}

[[nodiscard]] std::vector<real> download(const DeviceSpan<const real>& span) {
    std::vector<real> host(span.size());
    if (!host.empty()) {
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(host.data(), span.data(), host.size() * sizeof(real), cudaMemcpyDeviceToHost));
    }
    return host;
}
[[nodiscard]] std::vector<real> download_fields_u1(StreamfunctionFields& fields) {
    return download(DeviceSpan<const real>(fields.u1_span()));
}
[[nodiscard]] std::vector<real> download_fields_u2(StreamfunctionFields& fields) {
    return download(DeviceSpan<const real>(fields.u2_span()));
}

// A fabricated (never-solved) StreamfunctionSolveReport with the given
// status. Mirrors streamfunction_continuation_gpu_cases.cu's
// fabricated_report: make_stage_record unconditionally calls
// residual_histogram_percentile, which throws on a default-constructed
// (empty) histogram, so this gives a minimally-valid single-underflow-count
// histogram.
[[nodiscard]] StreamfunctionSolveReport fabricated_report(StreamfunctionSolveStatus status) {
    StreamfunctionSolveReport report;
    report.status = status;
    report.exit_reason = PicardExitReason::budget_exhausted;
    report.residual.histogram_underflow = 1;
    report.residual.histogram_c_min = real{1};
    report.residual.histogram_c_max = real{2};
    return report;
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

[[nodiscard]] const char* heterogeneity_axis_label(HeterogeneityAxis axis) {
    switch (axis) {
        case HeterogeneityAxis::lambda: return "lambda";
        case HeterogeneityAxis::eta_rescue: return "eta_rescue";
        case HeterogeneityAxis::epsilon: return "epsilon";
        default: return "unknown";
    }
}

void print_heterogeneity_stage_history(const std::vector<HeterogeneityStageRecord>& history) {
    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto& r = history[i];
        std::cout << std::setprecision(17) << "  stage[" << i << "] axis=" << heterogeneity_axis_label(r.axis)
                  << " lambda=" << r.lambda_value << " eta=" << r.eta_value
                  << " epsilon=" << r.epsilon_value << " mg_rebuilds=" << r.mg_rebuild_count
                  << " accepted=" << (r.base.accepted ? "true" : "false")
                  << " picard_iterations=" << r.base.picard_iterations << " r_F=" << r.base.r_F
                  << " e_v=" << r.e_v << " invariance_e_psi1=" << r.invariance_e_psi1
                  << " invariance_e_psi2=" << r.invariance_e_psi2 << " e_div=" << r.e_div
                  << " c_p001=" << r.c_percentile_p001 << " degeneracy_total0=" << r.degeneracy_total0
                  << " degeneracy_unexplained0=" << r.degeneracy_unexplained0
                  << " anderson_acc=" << r.anderson_accepted << " anderson_rej=" << r.anderson_rejected
                  << " anderson_resets=" << r.anderson_condition_resets
                  << " newton_act=" << r.newton_activations << " newton_acc=" << r.newton_steps_accepted
                  << " newton_fail=" << r.newton_step_failures
                  << " newton_rescue=" << r.newton_rescue_events << " newton_jv=" << r.newton_jv_evaluations
                  << '\n';
    }
}

// ---------------------------------------------------------------------------
// Case 1: heterogeneity_coefficient_state_contract (host+small-GPU, 16^3
// homogeneous / manufactured fixtures).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_heterogeneity_coefficient_state_contract() {
    constexpr int n = 16;
    const Grid3D grid(n, n, n, real{1}, real{1}, real{1});

    CudaContext ctx(0);

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    // (a) reuse + zero_source (default initial_state) on a FRESH workspace
    // must throw std::invalid_argument, and the message must name the
    // initial_state==warm_start requirement.
    {
        HomogeneousProblemBuffers buffers(grid);
        const StreamfunctionProblemView problem = homogeneous_problem_view(grid, buffers);
        StreamfunctionFields fields;
        StreamfunctionWorkspace workspace;
        StreamfunctionSolverConfig config = valid_config();
        config.coefficient_state = CoefficientState::reuse; // initial_state left at default zero_source
        bool threw = false;
        std::string message;
        try {
            (void)solve_streamfunctions(ctx, problem, config, fields, workspace);
        } catch (const std::invalid_argument& error) {
            threw = true;
            message = error.what();
        }
        add_check("reuse_zero_source_throws", threw);
        add_check("reuse_zero_source_message_names_warm_start",
                  message.find("warm_start") != std::string::npos);
        std::cout << "heterogeneity_coefficient_state_contract reuse_zero_source_message=" << message
                  << '\n';
    }

    // (b) reuse + warm_start on a FRESH workspace (never completed a
    // rebuild) must throw std::invalid_argument, message naming the
    // preceding-rebuild requirement.
    {
        HomogeneousProblemBuffers buffers(grid);
        const StreamfunctionProblemView problem = homogeneous_problem_view(grid, buffers);
        StreamfunctionFields fields;
        StreamfunctionWorkspace workspace;
        StreamfunctionSolverConfig config = valid_config();
        config.coefficient_state = CoefficientState::reuse;
        config.initial_state = PicardInitialState::warm_start;
        bool threw = false;
        std::string message;
        try {
            (void)solve_streamfunctions(ctx, problem, config, fields, workspace);
        } catch (const std::invalid_argument& error) {
            threw = true;
            message = error.what();
        }
        add_check("reuse_no_prior_rebuild_throws", threw);
        add_check("reuse_no_prior_rebuild_message_names_rebuild",
                  message.find("rebuild") != std::string::npos);
        std::cout << "heterogeneity_coefficient_state_contract reuse_no_prior_rebuild_message=" << message
                  << '\n';
    }

    // (c) rebuild-then-reuse equivalence: two independent, deterministically
    // identical setups (manufactured heterogeneous field, so the nonlinear
    // source is nontrivial). Both are first solved to an accepted baseline
    // (rebuild + zero_source). Path A then re-solves warm-started with
    // coefficient_state=rebuild (re-fills q/hierarchy/RHS from the SAME
    // conductivity/gauge -- idempotent, since every kernel here is a pure
    // function of its device inputs, which are themselves unchanged). Path B
    // re-solves warm-started with coefficient_state=reuse (skips that
    // refill). Because CoefficientState::reuse's documented contract is
    // "skip work that would have reproduced the SAME q/hierarchy/RHS", the
    // two paths must be bitwise identical: this is the direct verification
    // of that identical-by-construction argument, not merely an assumption.
    {
        constexpr double kAmplitude = 0.5;
        DeviceBuffer<real> y_a = manufactured_y_field(grid, kAmplitude);
        DeviceBuffer<real> y_b = manufactured_y_field(grid, kAmplitude);
        HomogeneousProblemBuffers flow_buffers_a(grid); // Darcy v=(1,0,0), reused as the fixed reference velocity
        HomogeneousProblemBuffers flow_buffers_b(grid);

        StreamfunctionProblemView problem_a;
        problem_a.grid = grid;
        problem_a.conductivity = DeviceSpan<const real>(y_a.span());
        problem_a.conductivity_representation = ConductivityRepresentation::log_conductivity_y;
        problem_a.darcy_velocity =
            CompactMacVelocityConstView{DeviceSpan<const real>(flow_buffers_a.darcy_u.span()),
                                        DeviceSpan<const real>(flow_buffers_a.darcy_v.span()),
                                        DeviceSpan<const real>(flow_buffers_a.darcy_w.span())};
        problem_a.bc = triply_periodic();
        problem_a.gauge = AffineGauge::benchmark(real{1});

        StreamfunctionProblemView problem_b = problem_a;
        problem_b.conductivity = DeviceSpan<const real>(y_b.span());
        problem_b.darcy_velocity =
            CompactMacVelocityConstView{DeviceSpan<const real>(flow_buffers_b.darcy_u.span()),
                                        DeviceSpan<const real>(flow_buffers_b.darcy_v.span()),
                                        DeviceSpan<const real>(flow_buffers_b.darcy_w.span())};

        StreamfunctionFields fields_a, fields_b;
        StreamfunctionWorkspace workspace_a, workspace_b;

        const StreamfunctionSolverConfig baseline_config = valid_config(); // rebuild, zero_source
        const StreamfunctionSolveReport baseline_a =
            solve_streamfunctions(ctx, problem_a, baseline_config, fields_a, workspace_a);
        const StreamfunctionSolveReport baseline_b =
            solve_streamfunctions(ctx, problem_b, baseline_config, fields_b, workspace_b);
        ctx.synchronize();
        add_check("baseline_a_converged", baseline_a.status == StreamfunctionSolveStatus::converged);
        add_check("baseline_b_converged", baseline_b.status == StreamfunctionSolveStatus::converged);
        add_check("baseline_fields_bitwise_equal",
                  download_fields_u1(fields_a) == download_fields_u1(fields_b) &&
                      download_fields_u2(fields_a) == download_fields_u2(fields_b));

        StreamfunctionSolverConfig warm_rebuild = valid_config();
        warm_rebuild.initial_state = PicardInitialState::warm_start;
        warm_rebuild.coefficient_state = CoefficientState::rebuild; // explicit, matches the default
        const StreamfunctionSolveReport report_a =
            solve_streamfunctions(ctx, problem_a, warm_rebuild, fields_a, workspace_a);

        StreamfunctionSolverConfig warm_reuse = valid_config();
        warm_reuse.initial_state = PicardInitialState::warm_start;
        warm_reuse.coefficient_state = CoefficientState::reuse;
        const StreamfunctionSolveReport report_b =
            solve_streamfunctions(ctx, problem_b, warm_reuse, fields_b, workspace_b);
        ctx.synchronize();

        add_check("reuse_equivalence_status_eq",
                  report_a.status == report_b.status);
        add_check("reuse_equivalence_picard_iterations_eq",
                  report_a.picard_iterations == report_b.picard_iterations);
        add_check("reuse_equivalence_exit_reason_eq", report_a.exit_reason == report_b.exit_reason);
        add_check("reuse_equivalence_r_F_bitwise_eq", report_a.residual.r_F == report_b.residual.r_F);
        add_check("reuse_equivalence_r1_bitwise_eq", report_a.residual.r1 == report_b.residual.r1);
        add_check("reuse_equivalence_r2_bitwise_eq", report_a.residual.r2 == report_b.residual.r2);
        add_check("reuse_equivalence_fields_bitwise_eq",
                  download_fields_u1(fields_a) == download_fields_u1(fields_b) &&
                      download_fields_u2(fields_a) == download_fields_u2(fields_b));

        std::cout << std::setprecision(17)
                  << "heterogeneity_coefficient_state_contract reuse_equivalence status_a="
                  << static_cast<int>(report_a.status) << " status_b=" << static_cast<int>(report_b.status)
                  << " iterations_a=" << report_a.picard_iterations
                  << " iterations_b=" << report_b.picard_iterations << " r_F_a=" << report_a.residual.r_F
                  << " r_F_b=" << report_b.residual.r_F << '\n';
    }

    for (const auto& [name, ok] : checks) {
        std::cout << "heterogeneity_coefficient_state_contract check " << name << "="
                  << (ok ? "PASS" : "FAIL") << '\n';
    }

    return {pass,
            "heterogeneity_coefficient_state_contract",
            "gpu-heterogeneity-coefficient-state",
            std::to_string(n) + "^3, K=1 (throw contracts) + manufactured K=exp(0.5*sin*sin*sin) (reuse "
                                "equivalence)",
            0.0,
            0.0,
            "reuse+zero_source throws, reuse+no-prior-rebuild throws, rebuild-then-reuse is bitwise "
            "equivalent to rebuild-then-rebuild",
            pass ? "all pass" : "some failed",
            "CoefficientState::reuse's caller contract (initial_state==warm_start, a preceding "
            "rebuild call) is enforced by std::invalid_argument, and skipping the q/hierarchy/RHS "
            "refill under reuse is behaviorally identical to redoing that (idempotent) refill under "
            "rebuild"};
}

// ---------------------------------------------------------------------------
// Case 2: heterogeneity_rescue_ordering (GPU, 16^3, mild manufactured Y).
//
// NOTE: at dx=1, manufactured_y_field's sin(2pi*(k+0.5)) sampling is EXACTLY
// zero at every cell center (see manufactured_y_field's doc comment above),
// so Y is identically zero here and K_lambda == 1 regardless of `kAmplitude`.
// This is intentional and benign: this case's contract (rescue ordering) is
// driven entirely by the injected StageSolveFn outcomes, not by the field.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_heterogeneity_rescue_ordering() {
    constexpr int n = 16;
    const Grid3D grid(n, n, n, real{1}, real{1}, real{1});
    constexpr double kAmplitude = 1.0; // mild at the small attempted lambdas exercised here

    CudaContext ctx(0);
    DeviceBuffer<real> y = manufactured_y_field(grid, kAmplitude);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    const StreamfunctionSolverConfig base_config = valid_config();
    HeterogeneityContinuationConfig continuation_config{}; // spec-locked defaults
    const physics::AffinePeriodicFlowConfig flow_config{};

    struct InjectionState {
        int lambda_attempt_index = 0;
        bool intercept_all_after_second = false;
    };
    auto state = std::make_shared<InjectionState>();

    StageSolveFn stage_solver = [state](CudaContext& c, const StreamfunctionProblemView& p,
                                       const StreamfunctionSolverConfig& cfg, StreamfunctionFields& f,
                                       StreamfunctionWorkspace& w) -> StreamfunctionSolveReport {
        const bool is_lambda_main_attempt =
            cfg.coefficient_state == CoefficientState::rebuild &&
            cfg.initial_state == PicardInitialState::warm_start;
        if (is_lambda_main_attempt) {
            ++state->lambda_attempt_index;
            if (state->lambda_attempt_index >= 2) {
                state->intercept_all_after_second = true;
            }
            // Never touch fields: the driver must restore from its own
            // accepted snapshot, never read this call's side effects.
            return fabricated_report(StreamfunctionSolveStatus::not_converged);
        }
        if (state->intercept_all_after_second) {
            return fabricated_report(StreamfunctionSolveStatus::not_converged);
        }
        // Baseline (zero_source, rebuild) or the first lambda attempt's
        // rescue (eta=0 + eta ramp, all coefficient_state=reuse): dispatch
        // to the real solver.
        return solve_streamfunctions(c, p, cfg, f, w);
    };

    const auto t0 = std::chrono::steady_clock::now();
    const HeterogeneityContinuationReport report = run_streamfunction_heterogeneity_continuation(
        ctx, grid, DeviceSpan<const real>(y.span()), continuation_config, flow_config, base_config, fields,
        workspace, stage_solver);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    // stage_history[0] = baseline (lambda=0). [1] = the injected first
    // lambda-attempt failure. [2] = eta-rescue at eta=0. [3..] = the eta
    // ramp, exactly the SF-17 default no-failure schedule (0.1, 0.2, 0.35,
    // 0.5, 0.725, 0.95, 1.0), since attempted lambda=0.1 with amplitude 1.0
    // Y is a very mild nonlinear source.
    add_check("history_size_ge_11", report.stage_history.size() >= 11);
    bool ordering_ok = report.stage_history.size() >= 10;
    if (ordering_ok) {
        const auto& baseline = report.stage_history[0];
        ordering_ok = ordering_ok && baseline.axis == HeterogeneityAxis::lambda &&
                     baseline.base.accepted && baseline.lambda_value == real{0};

        const auto& failed_attempt = report.stage_history[1];
        ordering_ok = ordering_ok && failed_attempt.axis == HeterogeneityAxis::lambda &&
                     !failed_attempt.base.accepted && failed_attempt.lambda_value == real{0.1} &&
                     failed_attempt.eta_value == real{1};

        const auto& rescue_eta0 = report.stage_history[2];
        ordering_ok = ordering_ok && rescue_eta0.axis == HeterogeneityAxis::eta_rescue &&
                     rescue_eta0.eta_value == real{0} && rescue_eta0.lambda_value == real{0.1};
    }
    add_check("baseline_then_failed_attempt_then_eta0_rescue_exact", ordering_ok);

    // RULE-EXACT host-side recomputation of the no-failure eta-ramp
    // trajectory (mirrors streamfunction_continuation_gpu_cases.cu's
    // case_continuation_stepper_eta_all_accept oracle): decimal literals
    // like 0.35 do not reproduce 0.2+0.15 bitwise, so the oracle must use
    // the SAME arithmetic the SF-17 ContinuationStepper uses, in the SAME
    // order.
    std::vector<real> expected_eta_ramp;
    {
        const ContinuationAxisConfig& eta_axis = continuation_config.inner.eta;
        real param = eta_axis.start;
        real step = eta_axis.initial_step;
        int streak = 0;
        while (param < eta_axis.target) {
            const real attempt = std::min(param + step, eta_axis.target);
            expected_eta_ramp.push_back(attempt);
            param = attempt;
            ++streak;
            if (streak >= eta_axis.easy_streak) {
                step = std::min(step * eta_axis.growth_factor, eta_axis.max_step);
                streak = 0;
            }
        }
    }
    bool ramp_ok = report.stage_history.size() >= 3 + expected_eta_ramp.size();
    std::size_t ramp_end_index = 0;
    if (ramp_ok) {
        for (std::size_t i = 0; i < expected_eta_ramp.size(); ++i) {
            const auto& record = report.stage_history[3 + i];
            ramp_ok = ramp_ok && record.axis == HeterogeneityAxis::eta_rescue &&
                     record.lambda_value == real{0.1} && record.eta_value == expected_eta_ramp[i] &&
                     record.base.accepted;
        }
        ramp_end_index = 3 + expected_eta_ramp.size() - 1;
    }
    add_check("eta_ramp_exact_sf17_schedule_all_accepted", ramp_ok);

    // The rescue's success ACCEPTS the lambda attempt: the record right
    // after the ramp is the NEXT lambda attempt, at the grown/same step from
    // the accepted lambda=0.1 (easy-streak==1 after this single accept, so
    // no growth yet: same step 0.1, attempt = 0.1+0.1 = 0.2).
    bool next_attempt_ok = ramp_ok && report.stage_history.size() > ramp_end_index + 1;
    if (next_attempt_ok) {
        const auto& next_attempt = report.stage_history[ramp_end_index + 1];
        next_attempt_ok = next_attempt_ok && next_attempt.axis == HeterogeneityAxis::lambda &&
                         next_attempt.lambda_value == real{0.2} &&
                         next_attempt.base.step_attempted == real{0.1};
    }
    add_check("rescue_success_accepts_lambda_next_attempt_same_step", next_attempt_ok);

    // mg_rebuild_count increments once per attempted lambda (baseline=1,
    // first attempt=2) and stays flat across every eta-rescue stage.
    bool mg_count_ok = report.stage_history.size() >= 3 + expected_eta_ramp.size();
    if (mg_count_ok) {
        mg_count_ok = mg_count_ok && report.stage_history[0].mg_rebuild_count == 1 &&
                     report.stage_history[1].mg_rebuild_count == 2 &&
                     report.stage_history[2].mg_rebuild_count == 2;
        for (std::size_t i = 0; i < expected_eta_ramp.size(); ++i) {
            mg_count_ok = mg_count_ok && report.stage_history[3 + i].mg_rebuild_count == 2;
        }
    }
    add_check("mg_rebuild_count_once_per_lambda_flat_across_rescue", mg_count_ok);

    std::cout << std::setprecision(17) << "heterogeneity_rescue_ordering N=" << n
              << " status=" << heterogeneity_status_label(report.status)
              << " stage_history_size=" << report.stage_history.size()
              << " total_mg_rebuilds=" << report.total_mg_rebuilds << " wall_seconds=" << wall_seconds
              << '\n';
    print_heterogeneity_stage_history(report.stage_history);
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    return {pass,
            "heterogeneity_rescue_ordering",
            "gpu-heterogeneity-rescue-ordering",
            std::to_string(n) + "^3, K_lambda=exp(lambda*sin*sin*sin), injected not_converged on the "
                                "first lambda attempt",
            static_cast<double>(report.stage_history.size()),
            static_cast<double>(report.total_mg_rebuilds),
            "baseline, failed attempt, eta=0 rescue, exact SF-17 eta ramp, lambda accepted, next "
            "attempt at same step, mg_rebuild_count flat across rescue",
            pass ? "all pass" : "some failed",
            "a StageSolveFn that fails only the first main lambda attempt (never touching fields) "
            "must trigger the documented eta-rescue ordering exactly: restore, eta=0, eta ramp "
            "0->1, and accept the SAME lambda value on rescue success"};
}

// ---------------------------------------------------------------------------
// Case 3: heterogeneity_rescue_failure_lambda_halving (GPU, 16^3, injection-only).
//
// NOTE: at dx=1, manufactured_y_field's sin(2pi*(k+0.5)) sampling is EXACTLY
// zero at every cell center (see manufactured_y_field's doc comment above),
// so Y is identically zero here and K_lambda == 1 regardless of `kAmplitude`.
// This is intentional and benign: this case's contract (halving/rollback) is
// driven entirely by the injected StageSolveFn outcomes, not by the field.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_heterogeneity_rescue_failure_lambda_halving() {
    constexpr int n = 16;
    const Grid3D grid(n, n, n, real{1}, real{1}, real{1});
    constexpr double kAmplitude = 1.0;

    CudaContext ctx(0);
    DeviceBuffer<real> y = manufactured_y_field(grid, kAmplitude);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    const StreamfunctionSolverConfig base_config = valid_config();
    HeterogeneityContinuationConfig continuation_config{};
    const physics::AffinePeriodicFlowConfig flow_config{};

    struct InjectionState {
        int call_index = 0;
        std::vector<real> accepted_u1;
        std::vector<real> accepted_u2;
        bool restore_check_evaluated = false;
        bool restore_check_ok = false;
    };
    auto state = std::make_shared<InjectionState>();

    // call 0: baseline -> real (converges: lambda=0 exact-zero fluctuations).
    // call 1: first lambda-attempt (0.1) -> fake fail, no touch.
    // call 2: eta=0 rescue for lambda=0.1 -> fake fail too, no touch (forces
    //         rescue failure at the very first rescue stage).
    // call 3: second lambda-attempt (0.05, the halved step) -> BEFORE
    //         faking this one too, verify the driver already restored
    //         fields bitwise to the outer accepted (baseline) snapshot.
    // call >=4: fake fail forever (bounded floor exhaustion at 0.025, 0.0125).
    StageSolveFn stage_solver = [state](CudaContext& c, const StreamfunctionProblemView& p,
                                       const StreamfunctionSolverConfig& cfg, StreamfunctionFields& f,
                                       StreamfunctionWorkspace& w) -> StreamfunctionSolveReport {
        const int call = state->call_index++;
        if (call == 0) {
            StreamfunctionSolveReport report = solve_streamfunctions(c, p, cfg, f, w);
            c.synchronize();
            state->accepted_u1 = download_fields_u1(f);
            state->accepted_u2 = download_fields_u2(f);
            return report;
        }
        if (call == 3) {
            c.synchronize();
            const std::vector<real> observed_u1 = download_fields_u1(f);
            const std::vector<real> observed_u2 = download_fields_u2(f);
            state->restore_check_evaluated = true;
            state->restore_check_ok =
                observed_u1 == state->accepted_u1 && observed_u2 == state->accepted_u2;
        }
        return fabricated_report(StreamfunctionSolveStatus::not_converged);
    };

    const auto t0 = std::chrono::steady_clock::now();
    const HeterogeneityContinuationReport report = run_streamfunction_heterogeneity_continuation(
        ctx, grid, DeviceSpan<const real>(y.span()), continuation_config, flow_config, base_config, fields,
        workspace, stage_solver);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("restore_check_evaluated", state->restore_check_evaluated);
    add_check("restore_check_bitwise_ok", state->restore_check_ok);

    // stage_history: [0]=baseline, [1]=lambda attempt 0.1 (step 0.1, rejected),
    // [2]=eta_rescue eta=0 (rejected), [3]=lambda attempt 0.05 (step 0.05, the
    // halved retry of the SAME interval: accepted(0) + step/2(0.05)).
    add_check("history_size_ge_4", report.stage_history.size() >= 4);
    bool halving_ok = report.stage_history.size() >= 4;
    if (halving_ok) {
        const auto& first_attempt = report.stage_history[1];
        halving_ok = halving_ok && first_attempt.axis == HeterogeneityAxis::lambda &&
                    first_attempt.lambda_value == real{0.1} &&
                    first_attempt.base.step_attempted == real{0.1} && !first_attempt.base.accepted;

        const auto& second_attempt = report.stage_history[3];
        halving_ok = halving_ok && second_attempt.axis == HeterogeneityAxis::lambda &&
                    second_attempt.lambda_value == real{0.05} &&
                    second_attempt.base.step_attempted == real{0.05} && !second_attempt.base.accepted;
    }
    add_check("second_attempt_eq_accepted_plus_halved_step", halving_ok);

    std::cout << std::setprecision(17) << "heterogeneity_rescue_failure_lambda_halving N=" << n
              << " status=" << heterogeneity_status_label(report.status)
              << " stage_history_size=" << report.stage_history.size() << " wall_seconds=" << wall_seconds
              << '\n';
    print_heterogeneity_stage_history(report.stage_history);
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    return {pass,
            "heterogeneity_rescue_failure_lambda_halving",
            "gpu-heterogeneity-rescue-failure-halving",
            std::to_string(n) + "^3, K_lambda=exp(lambda*sin*sin*sin), injected not_converged on the "
                                "first lambda attempt AND its eta=0 rescue",
            static_cast<double>(report.stage_history.size()),
            0.0,
            "outer accepted state restored bitwise before the halved retry, next attempt = accepted "
            "+ step/2",
            pass ? "all pass" : "some failed",
            "when both the main lambda attempt and its eta=0 rescue fail, the driver must restore "
            "the OUTER accepted state (bitwise) and halve the lambda step, retrying the SAME "
            "interval from the last accepted lambda"};
}

// ---------------------------------------------------------------------------
// Case 4: heterogeneity_lambda_floor_exhaustion (GPU, 16^3, injection-only).
//
// NOTE: at dx=1, manufactured_y_field's sin(2pi*(k+0.5)) sampling is EXACTLY
// zero at every cell center (see manufactured_y_field's doc comment above),
// so Y is identically zero here and K_lambda == 1 regardless of `kAmplitude`.
// This is intentional and benign: this case's contract (floor exhaustion) is
// driven entirely by the injected StageSolveFn outcomes, not by the field.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_heterogeneity_lambda_floor_exhaustion() {
    constexpr int n = 16;
    const Grid3D grid(n, n, n, real{1}, real{1}, real{1});
    constexpr double kAmplitude = 1.0;

    CudaContext ctx(0);
    DeviceBuffer<real> y = manufactured_y_field(grid, kAmplitude);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    const StreamfunctionSolverConfig base_config = valid_config();
    HeterogeneityContinuationConfig continuation_config{};
    const physics::AffinePeriodicFlowConfig flow_config{};

    struct InjectionState {
        int call_index = 0;
        std::vector<real> baseline_u1;
        std::vector<real> baseline_u2;
    };
    auto state = std::make_shared<InjectionState>();

    // call 0: baseline -> real. Every subsequent call (every main lambda
    // attempt AND every eta=0 rescue) fakes not_converged, never touching
    // fields: this forces every rescue to fail at its very first stage
    // (never even entering the eta ramp), so the lambda stepper halves
    // straight to the floor.
    StageSolveFn stage_solver = [state](CudaContext& c, const StreamfunctionProblemView& p,
                                       const StreamfunctionSolverConfig& cfg, StreamfunctionFields& f,
                                       StreamfunctionWorkspace& w) -> StreamfunctionSolveReport {
        const int call = state->call_index++;
        if (call == 0) {
            StreamfunctionSolveReport report = solve_streamfunctions(c, p, cfg, f, w);
            c.synchronize();
            state->baseline_u1 = download_fields_u1(f);
            state->baseline_u2 = download_fields_u2(f);
            return report;
        }
        return fabricated_report(StreamfunctionSolveStatus::not_converged);
    };

    const auto t0 = std::chrono::steady_clock::now();
    const HeterogeneityContinuationReport report = run_streamfunction_heterogeneity_continuation(
        ctx, grid, DeviceSpan<const real>(y.span()), continuation_config, flow_config, base_config, fields,
        workspace, stage_solver);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("status_lambda_floor_exhausted",
              report.status == HeterogeneityStatus::lambda_floor_exhausted);
    add_check("failed_axis_lambda", report.failed_axis == HeterogeneityAxis::lambda);
    add_check("final_lambda_eq_0", report.final_lambda == real{0});

    const std::vector<real> u1_final = download_fields_u1(fields);
    const std::vector<real> u2_final = download_fields_u2(fields);
    add_check("fields_bitwise_eq_baseline",
              u1_final == state->baseline_u1 && u2_final == state->baseline_u2);

    // stage_history: [0]=baseline; then 4 (attempt, eta0-rescue) pairs at
    // steps 0.1, 0.05, 0.025, 0.0125 (the last at the floor, one attempt).
    const std::vector<real> expected_steps = {real{0.1}, real{0.05}, real{0.025}, real{0.0125}};
    const bool history_size_ok = report.stage_history.size() == 1 + 2 * expected_steps.size();
    add_check("history_size_eq_9", history_size_ok);

    bool sequence_ok = history_size_ok;
    for (std::size_t i = 0; history_size_ok && i < expected_steps.size(); ++i) {
        const auto& attempt = report.stage_history[1 + 2 * i];
        const auto& rescue_eta0 = report.stage_history[2 + 2 * i];
        sequence_ok = sequence_ok && attempt.axis == HeterogeneityAxis::lambda &&
                     attempt.base.step_attempted == expected_steps[i] && !attempt.base.accepted &&
                     rescue_eta0.axis == HeterogeneityAxis::eta_rescue && rescue_eta0.eta_value == real{0} &&
                     !rescue_eta0.base.accepted;
    }
    add_check("halving_sequence_exact_0.1_0.05_0.025_0.0125_all_rejected", sequence_ok);

    // mg_rebuild_count: 1 for the baseline, +1 per main lambda attempt
    // (never for the eta=0 rescue calls, which use CoefficientState::reuse).
    bool mg_count_ok = history_size_ok && report.stage_history[0].mg_rebuild_count == 1;
    for (std::size_t i = 0; mg_count_ok && i < expected_steps.size(); ++i) {
        const int expected_count = static_cast<int>(i) + 2;
        mg_count_ok = mg_count_ok && report.stage_history[1 + 2 * i].mg_rebuild_count == expected_count &&
                     report.stage_history[2 + 2 * i].mg_rebuild_count == expected_count;
    }
    add_check("mg_rebuild_count_once_per_lambda_attempt", mg_count_ok);
    add_check("total_mg_rebuilds_eq_5", report.total_mg_rebuilds == 5);

    std::cout << std::setprecision(17) << "heterogeneity_lambda_floor_exhaustion N=" << n
              << " status=" << heterogeneity_status_label(report.status)
              << " failed_axis=" << heterogeneity_axis_label(report.failed_axis)
              << " stage_history_size=" << report.stage_history.size()
              << " total_mg_rebuilds=" << report.total_mg_rebuilds << " wall_seconds=" << wall_seconds
              << '\n';
    print_heterogeneity_stage_history(report.stage_history);
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    return {pass,
            "heterogeneity_lambda_floor_exhaustion",
            "gpu-heterogeneity-lambda-floor-exhaustion",
            std::to_string(n) + "^3, K_lambda=exp(lambda*sin*sin*sin), always-fail stage_solver after "
                                "the baseline",
            static_cast<double>(report.stage_history.size()),
            static_cast<double>(report.total_mg_rebuilds),
            "lambda_floor_exhausted, failed_axis=lambda, fields==baseline, halving "
            "0.1,0.05,0.025,0.0125 (one floor attempt), mg_rebuild_count once per lambda attempt",
            pass ? "all pass" : "some failed",
            "an always-fail StageSolveFn after the baseline forces every rescue to fail at its "
            "eta=0 stage, driving the lambda stepper to halve down to its floor and exit "
            "lambda_floor_exhausted after exactly one floor attempt, leaving fields exactly at the "
            "baseline accepted state"};
}

// ---------------------------------------------------------------------------
// Case 7 (SF-21 re-activation addition, decision R2a): heterogeneity_
// anderson_entry_clear -- proves solve_streamfunctions clears the Anderson
// history at solve entry.
//
// Design: a single manufactured 16^3 trig fixture (amplitude 0.5, domain
// length 1 i.e. dx=dy=dz=1/n -- see the grid construction below for why this
// must NOT be dx=1 as in the other cases in this file -- the SAME fixture
// family as streamfunction_anderson_gpu_cases.cu's non-regression controls)
// is solved TWICE, back-to-back, on the SAME workspace/fields/
// config/problem, with Anderson enabled (depth=5, start_iteration=5,
// condition_limit=1e12) and initial_state left at its default zero_source.
// Because zero_source unconditionally zero-initializes fields.u1/u2 and
// re-derives Picard state 0 from scratch every call (SF-13, unchanged), the
// SAME initial state is reproduced by construction on both calls without any
// extra field-reset bookkeeping. If (and only if) the Anderson history is
// cleared at solve entry, the second call is a byte-for-byte replay of the
// first: identical outer iteration count, identical per-trial (omega,
// r_F_trial, outcome, anderson-flag) sequence, and identical final residual.
// Without the entry clear, the first call's LEFTOVER staged (x, f) pair
// would still be resident in workspace.anderson() when the second call's own
// first `update_history` runs, forming a (DeltaX, DeltaF) column that
// straddles two different solve instances and perturbing the second call's
// early Anderson candidates -- so this comparison is a genuine, non-vacuous
// probe of the entry-clear behavior, not merely of solve determinism.
// `workspace.anderson().num_columns() > 0` is asserted after the FIRST
// solve's return specifically to establish that such leftover history really
// exists to be cleared (otherwise the whole test would pass vacuously even
// with no entry clear at all, since there would be nothing stale to leak).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_heterogeneity_anderson_entry_clear() {
    constexpr int n = 16;
    // Domain length 1 (dx=dy=dz=1/n), NOT dx=1 as in the other cases in this
    // file: with dx=1 the cell centers are half-integers, and
    // sin(2pi*(k+0.5)) = sin(pi) = 0 EXACTLY for every integer k, so the
    // manufactured trig log-K field below would degenerate to log_k == 0
    // (K == 1, exactly homogeneous) at every cell center. That makes the
    // affine RHS exactly zero, the zero_source initial state the exact
    // discrete solution, and the first solve converge at iteration 0 with no
    // Picard iteration and no Anderson history -- silently defeating the
    // non-vacuousness check below (T02r-F1). Mirrors
    // streamfunction_anderson_gpu_cases.cu's isotropic_grid(n) so the trig
    // field is nontrivial at cell centers.
    const Grid3D grid(n, n, n, real{1} / n, real{1} / n, real{1} / n);
    constexpr double kAmplitude = 0.5;

    StreamfunctionSolverConfig config = valid_config();
    config.anderson.enabled = true;
    config.anderson.depth = 5;
    config.anderson.start_iteration = 5;
    config.anderson.condition_limit = real{1e12};

    CudaContext ctx(0);
    ManufacturedProblemBuffers buffers(grid, kAmplitude);
    const StreamfunctionProblemView problem = manufactured_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionSolveReport report1 = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();

    const int columns_after_first_solve = workspace.anderson().num_columns();

    const StreamfunctionSolveReport report2 = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();
    const auto t2 = std::chrono::steady_clock::now();

    const double wall1 = std::chrono::duration<double>(t1 - t0).count();
    const double wall2 = std::chrono::duration<double>(t2 - t1).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("first_solve_converged", report1.status == StreamfunctionSolveStatus::converged);
    add_check("second_solve_converged", report2.status == StreamfunctionSolveStatus::converged);

    // Non-vacuousness: the first call must actually have left history
    // behind for the entry clear to matter.
    add_check("leftover_history_after_first_solve", columns_after_first_solve > 0);

    add_check("status_eq", report1.status == report2.status);
    add_check("picard_iterations_eq", report1.picard_iterations == report2.picard_iterations);
    add_check("exit_reason_eq", report1.exit_reason == report2.exit_reason);
    add_check("final_omega_bitwise_eq", report1.final_omega == report2.final_omega);
    add_check("r_F_bitwise_eq", report1.residual.r_F == report2.residual.r_F);
    add_check("r1_bitwise_eq", report1.residual.r1 == report2.residual.r1);
    add_check("r2_bitwise_eq", report1.residual.r2 == report2.residual.r2);
    add_check("anderson_accepted_eq", report1.anderson_accepted == report2.anderson_accepted);
    add_check("anderson_rejected_eq", report1.anderson_rejected == report2.anderson_rejected);
    add_check("anderson_condition_resets_eq",
              report1.anderson_condition_resets == report2.anderson_condition_resets);

    bool picard_history_eq = report1.picard_history.size() == report2.picard_history.size();
    for (std::size_t i = 0; picard_history_eq && i < report1.picard_history.size(); ++i) {
        const auto& a = report1.picard_history[i];
        const auto& b = report2.picard_history[i];
        picard_history_eq = picard_history_eq && a.r_F == b.r_F && a.r1 == b.r1 && a.r2 == b.r2;
    }
    add_check("picard_history_bitwise_eq", picard_history_eq);

    bool trial_history_eq = report1.trial_history.size() == report2.trial_history.size();
    for (std::size_t i = 0; trial_history_eq && i < report1.trial_history.size(); ++i) {
        const auto& a = report1.trial_history[i];
        const auto& b = report2.trial_history[i];
        trial_history_eq = trial_history_eq && a.iteration == b.iteration && a.omega == b.omega &&
                           a.r_F_trial == b.r_F_trial && a.outcome == b.outcome &&
                           a.anderson == b.anderson;
    }
    add_check("trial_history_bitwise_eq", trial_history_eq);

    const std::vector<real> u1_1 = download(fields.u1_span());
    // fields now holds call 2's result; re-download call 1's fields is not
    // possible after overwrite, so field-level agreement is inferred from
    // the exhaustive report-level bitwise checks above instead.
    (void)u1_1;

    std::cout << std::setprecision(17) << "heterogeneity_anderson_entry_clear N=" << n
              << " amplitude=" << kAmplitude
              << " columns_after_first_solve=" << columns_after_first_solve << '\n'
              << "  call1: status=" << static_cast<int>(report1.status)
              << " picard_iterations=" << report1.picard_iterations << " r_F=" << report1.residual.r_F
              << " anderson_accepted=" << report1.anderson_accepted
              << " anderson_rejected=" << report1.anderson_rejected
              << " trial_history_size=" << report1.trial_history.size() << " wall_seconds=" << wall1
              << '\n'
              << "  call2: status=" << static_cast<int>(report2.status)
              << " picard_iterations=" << report2.picard_iterations << " r_F=" << report2.residual.r_F
              << " anderson_accepted=" << report2.anderson_accepted
              << " anderson_rejected=" << report2.anderson_rejected
              << " trial_history_size=" << report2.trial_history.size() << " wall_seconds=" << wall2
              << '\n';
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << n << "^3, K=exp(" << kAmplitude
           << "*sin(2pi x)sin(2pi y)sin(2pi z)), benchmark gauge, uniform Darcy v=(1,0,0), anderson "
              "depth=5 start=5 limit=1e12, two back-to-back solve_streamfunctions calls on the SAME "
              "workspace";

    return {pass,
            "heterogeneity_anderson_entry_clear",
            "gpu-heterogeneity-anderson-entry-clear",
            detail.str(),
            static_cast<double>(report1.picard_iterations),
            static_cast<double>(report2.picard_iterations),
            "leftover history after call 1 > 0; call 2 is a bitwise replay of call 1 (status, "
            "iterations, exit_reason, final_omega, residual, picard_history, trial_history, "
            "anderson counters)",
            pass ? "all pass" : "some failed",
            "solve_streamfunctions must clear workspace.anderson()'s history at solve entry "
            "(re-activation decision R2a): otherwise a second call on the same workspace would "
            "straddle stale history from the first call's final outer iterations into its own early "
            "Anderson candidates, perturbing its trajectory relative to an identical from-scratch "
            "solve"};
}

// ---------------------------------------------------------------------------
// Cases 5-6: heterogeneity_smoke_sigma{025,1} (GPU, 32^3, PRESPECIFIED
// physical Gaussian gates -- SF-20 activation bitácora decision 5(b); SF-21
// re-activation decision R5 enables the SF-20-validated Anderson defaults in
// the base config). Heavy tier: see
// heterogeneity_continuation_smoke_case_registry() below and
// streamfunction_operator_tests.cpp's heavy_cases().
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult run_heterogeneity_smoke(const char* name, double sigma2) {
    constexpr int n = 32;
    const Grid3D grid(n, n, n, real{1}, real{1}, real{1});

    CudaContext ctx(0);

    physics::PeriodicGaussianFieldConfig field_config;
    field_config.sigma2 = static_cast<real>(sigma2);
    field_config.corr_length = real{8};
    field_config.seed = 12345ULL;
    field_config.normalize_variance = true;

    DeviceBuffer<real> y(grid.num_cells());
    physics::PeriodicGaussianFieldWorkspace field_workspace;
    const physics::PeriodicGaussianFieldReport field_report =
        physics::generate_periodic_gaussian_field(ctx, grid, field_config, y.span(), field_workspace);
    ctx.synchronize();

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    StreamfunctionSolverConfig base_config = valid_config(); // full defaults (adaptive Picard,
                                                               // max_iter=500, tolerance=1e-6,
                                                               // linear rtol=1e-10)
    // SF-21 re-activation decision R5: enable the SF-20-validated Anderson
    // defaults in these smokes (everything else in the fixture stays
    // VERBATIM from the parked SF-20 T02 commit).
    base_config.anderson.enabled = true;
    base_config.anderson.depth = 5;
    base_config.anderson.start_iteration = 5;
    base_config.anderson.condition_limit = real{1e12};

    // SF-25 (E1a): enable the SF-24-accepted Newton terminal phase with its
    // validated defaults (activation r_F<=1e-2, stagnation redirect <=1e-1,
    // forcing clamp(sqrt(r_F),1e-8,1e-1), rescue 5, retry once) -- the ONLY
    // sanctioned fixture change of SF-25; everything else stays VERBATIM from
    // the SF-21 prespecification.
    base_config.newton.enabled = true;

    HeterogeneityContinuationConfig continuation_config{}; // lambda axis defaults
    // Degenerate epsilon leg: start==target==2 (epsilon fixed at 1e-2), so
    // this run gates exactly on the lambda leg, per the PRESPECIFIED fixture.
    continuation_config.inner.epsilon_log10.target = continuation_config.inner.epsilon_log10.start;
    const physics::AffinePeriodicFlowConfig flow_config{}; // qbar=(1,0,0) default

    const auto t0 = std::chrono::steady_clock::now();
    const HeterogeneityContinuationReport report = run_streamfunction_heterogeneity_continuation(
        ctx, grid, DeviceSpan<const real>(y.span()), continuation_config, flow_config, base_config, fields,
        workspace);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name_, bool ok) {
        checks.emplace_back(name_, ok);
        pass = pass && ok;
    };

    add_check("status_reached_target", report.status == HeterogeneityStatus::reached_target);
    add_check("final_lambda_eq_1", report.final_lambda == real{1});

    std::size_t accepted_count = 0;
    std::size_t rescue_stage_count = 0;
    bool every_accepted_r_f_ok = true;
    bool attribution_counters_populated = false;
    for (const auto& record : report.stage_history) {
        if (record.axis == HeterogeneityAxis::eta_rescue) {
            ++rescue_stage_count;
        }
        if (record.base.accepted) {
            ++accepted_count;
            every_accepted_r_f_ok =
                every_accepted_r_f_ok && static_cast<double>(record.base.r_F) <= 1e-6;
            if (record.newton_activations >= 1 || record.anderson_accepted >= 1) {
                attribution_counters_populated = true;
            }
        }
    }
    add_check("every_accepted_stage_r_F_le_1e-6", every_accepted_r_f_ok);
    // SF-25 T03: additive sanity check (not a moved gate) -- with newton and
    // anderson both enabled on this physical 32^3 run, solver-phase
    // attribution must be actually populated on at least one accepted stage.
    add_check("attribution_counters_populated", attribution_counters_populated);

    std::cout << std::setprecision(17) << name << " N=" << n << " sigma2=" << sigma2
              << " corr_length=" << field_config.corr_length << " seed=" << field_config.seed
              << " field_raw_mean=" << field_report.raw_mean
              << " field_final_variance=" << field_report.final_variance
              << " status=" << heterogeneity_status_label(report.status)
              << " final_lambda=" << report.final_lambda << " final_eta=" << report.final_eta
              << " final_epsilon=" << report.final_epsilon
              << " total_mg_rebuilds=" << report.total_mg_rebuilds
              << " stage_history_size=" << report.stage_history.size()
              << " accepted_stage_count=" << accepted_count
              << " eta_rescue_stage_count=" << rescue_stage_count << " wall_seconds=" << wall_seconds
              << '\n';
    print_heterogeneity_stage_history(report.stage_history);
    for (const auto& [check_name, ok] : checks) {
        std::cout << "  check " << check_name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << n << "^3, dx=1, sigma_Y^2=" << sigma2
           << ", corr_length=8, seed=12345, epsilon fixed at 1e-2 (degenerate epsilon leg), anderson "
              "depth=5 start=5 limit=1e12 + newton enabled (SF-25 E1a)";

    return {pass,
            name,
            "gpu-heterogeneity-physical-smoke",
            detail.str(),
            static_cast<double>(accepted_count),
            static_cast<double>(report.stage_history.size()),
            "reached_target, final_lambda=1, every accepted stage r_F<=1e-6",
            pass ? "all pass" : "some failed",
            "PRESPECIFIED (SF-20 activation bitácora decision 5(b)) 32^3 physical Gaussian smoke: "
            "the lambda/eta-rescue continuation must reach the full lognormal target conductivity "
            "for this seed/correlation length/variance, gating exclusively on the lambda leg (the "
            "epsilon leg is held degenerate at 1e-2), now run with the SF-20-validated Anderson "
            "defaults enabled (SF-21 re-activation decision R5) + newton enabled (SF-25 E1a)"};
}

[[nodiscard]] CaseResult case_heterogeneity_smoke_sigma025() {
    return run_heterogeneity_smoke("heterogeneity_smoke_sigma025", 0.25);
}

[[nodiscard]] CaseResult case_heterogeneity_smoke_sigma1() {
    return run_heterogeneity_smoke("heterogeneity_smoke_sigma1", 1.0);
}

} // namespace

CaseRegistry heterogeneity_continuation_case_registry() {
    return {{"heterogeneity_coefficient_state_contract", case_heterogeneity_coefficient_state_contract},
            {"heterogeneity_rescue_ordering", case_heterogeneity_rescue_ordering},
            {"heterogeneity_rescue_failure_lambda_halving",
             case_heterogeneity_rescue_failure_lambda_halving},
            {"heterogeneity_lambda_floor_exhaustion", case_heterogeneity_lambda_floor_exhaustion},
            {"heterogeneity_anderson_entry_clear", case_heterogeneity_anderson_entry_clear}};
}

CaseRegistry heterogeneity_continuation_smoke_case_registry() {
    return {{"heterogeneity_smoke_sigma025", case_heterogeneity_smoke_sigma025},
            {"heterogeneity_smoke_sigma1", case_heterogeneity_smoke_sigma1}};
}

} // namespace macroflow3d::streamfunctions::test
