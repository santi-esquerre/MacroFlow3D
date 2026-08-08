#include "streamfunction_operator_test_cases.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/DifferentialOperators.cuh"
#include "src/physics/streamfunctions/NonlinearSources.cuh"
#include "src/physics/streamfunctions/StreamfunctionSolver.cuh"
#include "src/physics/streamfunctions/StreamfunctionTypes.hpp"
#include "src/physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "src/physics/streamfunctions/affine_gauge.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// SF-13 T02: homogeneous (K=1) exact-control GPU acceptance cases for
// `solve_streamfunctions` (StreamfunctionSolver.cuh/.cu, SF-13 T01). On the
// triply periodic unit torus with K=1 and the benchmark gauge, every
// quantity in the chain is exact in continuous arithmetic: the affine RHS
// telescopes to zero, the unique mean-zero solution is u1=u2=0, all
// Hessians of the total streamfunctions vanish so S1=S2=0, and the
// reconstructed velocity equals the exact uniform Darcy field. These cases
// assert hard, non-relaxed thresholds (see the increment task spec's
// rollback policy: any nonzero systematic source or velocity error here
// BLOCKS nonlinear work) and never touch `src/**`.

namespace macroflow3d::streamfunctions::test {
namespace {

// ---------------------------------------------------------------------------
// Small local helpers, following streamfunction_api_workspace_gpu_cases.cu.
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

[[nodiscard]] Grid3D isotropic_grid(int n, real domain_length = real{1}) {
    const real h = domain_length / static_cast<real>(n);
    return Grid3D{n, n, n, h, h, h};
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

// Same default config as the SF-12 tests' valid_config(): every sub-config
// keeps its production default. config.mg.num_levels defaults to 4, which is
// exactly attainable for 16^3 (16->8->4->2) and is <= the maximum attainable
// depth for 32^3 and 64^3 as well, so a single config works unchanged for all
// three grids in case 1 without weakening any documented default.
[[nodiscard]] StreamfunctionSolverConfig valid_config() { return StreamfunctionSolverConfig{}; }

// Uploads K=1 (constant), and the uniform Darcy field v=(1,0,0) consistent
// with the benchmark gauge g1=(0,1,0), g2=(0,0,1): v = g1 x g2 = (1,0,0).
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

[[nodiscard]] std::vector<real> download(const DeviceSpan<const real>& span) {
    std::vector<real> host(span.size());
    if (!host.empty()) {
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(host.data(), span.data(), host.size() * sizeof(real), cudaMemcpyDeviceToHost));
    }
    return host;
}

[[nodiscard]] double host_rms(const std::vector<real>& values) {
    if (values.empty()) return 0.0;
    long double sum = 0.0L;
    for (real value : values) {
        const long double v = static_cast<long double>(value);
        sum += v * v;
    }
    return std::sqrt(static_cast<double>(sum / static_cast<long double>(values.size())));
}

[[nodiscard]] double host_mean(const std::vector<real>& values) {
    if (values.empty()) return 0.0;
    long double sum = 0.0L;
    for (real value : values) sum += static_cast<long double>(value);
    return static_cast<double>(sum / static_cast<long double>(values.size()));
}

// Test-owned device scratch for evaluating S1/S2 from an already-accepted
// streamfunction state, using only the production primitives (SF-07 total
// gradients, SF-08 Hessian-vector B, SF-09 nonlinear sources). Never writes
// into any StreamfunctionWorkspace/StreamfunctionFields buffer.
struct SourceEvalScratch {
    DeviceBuffer<real> p1x, p1y, p1z, p2x, p2y, p2z;
    DeviceBuffer<real> h2g1x, h2g1y, h2g1z, h1g2x, h1g2y, h1g2z, bx, by, bz;
    DeviceBuffer<real> s1, s2;
    DeviceBuffer<unsigned long long> counters;

    explicit SourceEvalScratch(std::size_t n, int num_thresholds)
        : p1x(n), p1y(n), p1z(n), p2x(n), p2y(n), p2z(n), h2g1x(n), h2g1y(n), h2g1z(n), h1g2x(n),
          h1g2y(n), h1g2z(n), bx(n), by(n), bz(n), s1(n), s2(n),
          counters(static_cast<std::size_t>(2 + num_thresholds)) {}
};

struct SourceEvalResult {
    std::vector<real> s1;
    std::vector<real> s2;
};

// Evaluates S1, S2 at the current state of `fields` using the same
// NonlinearSourceConfig construction `solve_streamfunctions` uses
// internally (config.epsilon, measured v_rms, config's degeneracy
// thresholds). Test-side only; never modifies fields/workspace/report.
[[nodiscard]] SourceEvalResult evaluate_sources(CudaContext& ctx, const Grid3D& grid,
                                                 StreamfunctionFields& fields, const AffineGauge& gauge,
                                                 const StreamfunctionSolverConfig& config, real v_rms) {
    const std::size_t n = grid.num_cells();
    SourceEvalScratch scratch(n, config.num_degeneracy_thresholds);

    enqueue_total_streamfunction_gradients(
        ctx, grid, fields.fluctuations(), gauge,
        {scratch.p1x.span(), scratch.p1y.span(), scratch.p1z.span(), scratch.p2x.span(),
         scratch.p2y.span(), scratch.p2z.span()});

    const TotalStreamfunctionGradientView gradients{
        scratch.p1x.span(), scratch.p1y.span(), scratch.p1z.span(),
        scratch.p2x.span(), scratch.p2y.span(), scratch.p2z.span()};

    enqueue_streamfunction_hessian_vector_b(
        ctx, grid, fields.fluctuations(), gradients,
        {scratch.h2g1x.span(), scratch.h2g1y.span(), scratch.h2g1z.span(), scratch.h1g2x.span(),
         scratch.h1g2y.span(), scratch.h1g2z.span(), scratch.bx.span(), scratch.by.span(),
         scratch.bz.span()});

    NonlinearSourceConfig source_config{};
    source_config.epsilon = config.epsilon;
    source_config.v_rms = v_rms;
    source_config.num_degeneracy_thresholds = config.num_degeneracy_thresholds;
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        source_config.degeneracy_thresholds[t] = config.degeneracy_thresholds[t];
    }

    const auto num_counters = static_cast<std::size_t>(2 + config.num_degeneracy_thresholds);
    enqueue_streamfunction_nonlinear_sources(
        ctx, grid, gradients, StreamfunctionBFieldView{scratch.bx.span(), scratch.by.span(), scratch.bz.span()},
        source_config, {scratch.s1.span(), scratch.s2.span()},
        {DeviceSpan<unsigned long long>(scratch.counters.data(), num_counters)});

    ctx.synchronize();

    SourceEvalResult result;
    result.s1 = download(DeviceSpan<const real>(scratch.s1.span()));
    result.s2 = download(DeviceSpan<const real>(scratch.s2.span()));
    return result;
}

// ---------------------------------------------------------------------------
// Case 1 (shared implementation): homogeneous_solver_exact_{16,32,64}.
// ---------------------------------------------------------------------------

constexpr double kExactThreshold = 1.0e-13;

[[nodiscard]] CaseResult run_homogeneous_exact_case(int n) {
    const std::string case_name = "homogeneous_solver_exact_" + std::to_string(n);
    const Grid3D grid = isotropic_grid(n);
    const StreamfunctionSolverConfig config = valid_config();

    CudaContext ctx(0);
    HomogeneousProblemBuffers buffers(grid);
    const StreamfunctionProblemView problem = homogeneous_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    const StreamfunctionSolveReport report = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();

    const std::vector<real> u1_host = download(fields.u1_span());
    const std::vector<real> u2_host = download(fields.u2_span());
    const double rms_u1 = host_rms(u1_host);
    const double rms_u2 = host_rms(u2_host);
    const double mean_u1 = host_mean(u1_host);
    const double mean_u2 = host_mean(u2_host);

    const SourceEvalResult sources =
        evaluate_sources(ctx, grid, fields, problem.gauge, config, report.diagnostics.v_d_rms);
    const double rms_s1 = host_rms(sources.s1);
    const double rms_s2 = host_rms(sources.s2);

    const double eps_machine = static_cast<double>(std::numeric_limits<real>::epsilon());
    const double gauge_tol_u1 = 100.0 * eps_machine * std::max(rms_u1, 1.0);
    const double gauge_tol_u2 = 100.0 * eps_machine * std::max(rms_u2, 1.0);

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("status_converged", report.status == StreamfunctionSolveStatus::converged);
    add_check("psi1_converged", report.psi1_result.converged);
    add_check("psi2_converged", report.psi2_result.converged);
    add_check("rms_u1_le_1e-13", rms_u1 <= kExactThreshold);
    add_check("rms_u2_le_1e-13", rms_u2 <= kExactThreshold);
    add_check("rms_s1_le_1e-13", rms_s1 <= kExactThreshold);
    add_check("rms_s2_le_1e-13", rms_s2 <= kExactThreshold);
    add_check("gauge_mean_u1", std::abs(mean_u1) <= gauge_tol_u1);
    add_check("gauge_mean_u2", std::abs(mean_u2) <= gauge_tol_u2);
    add_check("e_v_le_1e-13", report.diagnostics.e_v <= kExactThreshold);
    add_check("r_F_le_1e-13", report.residual.r_F <= kExactThreshold);
    add_check("e_div_le_1e-13", report.diagnostics.e_div <= kExactThreshold);
    add_check("invariance_e_psi1_le_1e-13", report.diagnostics.invariance_e_psi1 <= kExactThreshold);
    add_check("invariance_e_psi2_le_1e-13", report.diagnostics.invariance_e_psi2 <= kExactThreshold);
    add_check("v_rms_within_1e-13_of_1",
              std::abs(static_cast<double>(report.diagnostics.v_d_rms) - 1.0) <= kExactThreshold);
    add_check("residual_v_rms_equals_measured",
              static_cast<double>(report.residual.v_rms) ==
                  static_cast<double>(report.diagnostics.v_d_rms));

    // --- Gate 3A metric block ---
    std::cout << std::setprecision(17)
              << "homogeneous_solver_exact grid=" << n << "^3\n"
              << "  r_F=" << report.residual.r_F << " r1=" << report.residual.r1
              << " r2=" << report.residual.r2 << '\n'
              << "  rms_f1=" << report.residual.rms_f1 << " rms_f2=" << report.residual.rms_f2 << '\n'
              << "  rms_u1=" << rms_u1 << " rms_u2=" << rms_u2 << '\n'
              << "  rms_S1(test-side)=" << rms_s1 << " rms_S2(test-side)=" << rms_s2 << '\n'
              << "  e_v=" << report.diagnostics.e_v << " e_div=" << report.diagnostics.e_div << '\n'
              << "  invariance_e_psi1=" << report.diagnostics.invariance_e_psi1
              << " invariance_e_psi2=" << report.diagnostics.invariance_e_psi2 << '\n'
              << "  c_min=" << report.diagnostics.c_min << " c_max=" << report.diagnostics.c_max
              << " c_mean=" << report.diagnostics.c_mean << '\n'
              << "  mean_u1(measured)=" << mean_u1 << " mean_u2(measured)=" << mean_u2
              << " psi1_final_field_mean=" << report.psi1_result.final_field_mean
              << " psi2_final_field_mean=" << report.psi2_result.final_field_mean << '\n'
              << "  v_rms(measured)=" << report.diagnostics.v_d_rms
              << " L_ref=" << report.diagnostics.L_ref << " epsilon=" << config.epsilon
              << " eta=" << config.eta << '\n'
              << "  psi1_pcg iterations=" << report.psi1_result.iterations
              << " initial_projected_residual=" << report.psi1_result.initial_projected_residual
              << " final_projected_residual=" << report.psi1_result.final_projected_residual << '\n'
              << "  psi2_pcg iterations=" << report.psi2_result.iterations
              << " initial_projected_residual=" << report.psi2_result.initial_projected_residual
              << " final_projected_residual=" << report.psi2_result.final_projected_residual << '\n'
              << "  note: v_rms==1 and L_ref==1 on this benchmark gauge fixture, so raw RMS "
                 "values coincide with normalized ones\n";

    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << n << "^3, K=1, benchmark gauge, uniform Darcy v=(1,0,0)";

    return {pass,
            case_name,
            "gpu-exact-control",
            detail.str(),
            rms_s1,
            rms_s2,
            "all metrics <= 1e-13 (exact control)",
            pass ? "all pass" : "some failed",
            "homogeneous K=1 solve on the triply periodic unit torus: zero-RHS PCG converges "
            "immediately, u1=u2=0, S1=S2=0, and every SF-11 physical diagnostic error is exact "
            "to floating-point roundoff; see the printed Gate 3A metric block above"};
}

[[nodiscard]] CaseResult case_homogeneous_solver_exact_16() { return run_homogeneous_exact_case(16); }
[[nodiscard]] CaseResult case_homogeneous_solver_exact_32() { return run_homogeneous_exact_case(32); }
[[nodiscard]] CaseResult case_homogeneous_solver_exact_64() { return run_homogeneous_exact_case(64); }

// ---------------------------------------------------------------------------
// Case 2: homogeneous_solver_repeated_reuse.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_homogeneous_solver_repeated_reuse() {
    constexpr int kAxis = 32;
    constexpr int kCalls = 3;
    const Grid3D grid = isotropic_grid(kAxis);
    const StreamfunctionSolverConfig config = valid_config();

    CudaContext ctx(0);
    HomogeneousProblemBuffers buffers(grid);
    const StreamfunctionProblemView problem = homogeneous_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    struct CallMetrics {
        StreamfunctionSolveStatus status{};
        double rms_u1{}, rms_u2{}, r_F{}, e_v{};
        std::size_t bytes{};
        std::vector<const void*> pointers;
        std::size_t free_bytes{};
    };

    const auto observed_pointers = [&] {
        std::vector<const void*> pointers{fields.u1_span().data(),
                                          fields.u2_span().data(),
                                          workspace.q().data(),
                                          workspace.rhs1().data(),
                                          workspace.rhs2().data(),
                                          workspace.f1().data(),
                                          workspace.f2().data(),
                                          workspace.v_psi().u.data(),
                                          workspace.v_psi().v.data(),
                                          workspace.v_psi().w.data()};
        return pointers;
    };
    const auto observed_bytes = [&] { return fields.allocated_device_bytes() + workspace.allocated_device_bytes(); };
    const auto free_bytes = [&] {
        std::size_t free_b = 0, total_b = 0;
        MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        return free_b;
    };

    std::vector<CallMetrics> calls;
    bool pass = true;
    for (int i = 0; i < kCalls; ++i) {
        const StreamfunctionSolveReport report = solve_streamfunctions(ctx, problem, config, fields, workspace);
        ctx.synchronize();

        CallMetrics metrics;
        metrics.status = report.status;
        metrics.rms_u1 = host_rms(download(fields.u1_span()));
        metrics.rms_u2 = host_rms(download(fields.u2_span()));
        metrics.r_F = static_cast<double>(report.residual.r_F);
        metrics.e_v = static_cast<double>(report.diagnostics.e_v);
        metrics.bytes = observed_bytes();
        metrics.pointers = observed_pointers();
        metrics.free_bytes = free_bytes();
        calls.push_back(metrics);

        const bool call_ok = report.status == StreamfunctionSolveStatus::converged &&
                             metrics.rms_u1 <= kExactThreshold && metrics.rms_u2 <= kExactThreshold &&
                             metrics.r_F <= kExactThreshold && metrics.e_v <= kExactThreshold;
        pass = pass && call_ok;

        std::cout << std::setprecision(17) << "homogeneous_solver_repeated_reuse call=" << i
                  << " status=" << (report.status == StreamfunctionSolveStatus::converged ? "converged" : "OTHER")
                  << " rms_u1=" << metrics.rms_u1 << " rms_u2=" << metrics.rms_u2 << " r_F=" << metrics.r_F
                  << " e_v=" << metrics.e_v << " bytes=" << metrics.bytes << " free_bytes=" << metrics.free_bytes
                  << " check=" << (call_ok ? "PASS" : "FAIL") << '\n';
    }

    // Stability across calls 2 and 3 vs the first call: exact byte/pointer/
    // free-memory reuse (SF-12 case-4 method).
    for (int i = 1; i < kCalls; ++i) {
        const bool bytes_ok = calls[i].bytes == calls[0].bytes;
        const bool pointers_ok = calls[i].pointers == calls[0].pointers;
        const bool free_ok = calls[i].free_bytes == calls[0].free_bytes;
        pass = pass && bytes_ok && pointers_ok && free_ok;
        std::cout << "homogeneous_solver_repeated_reuse stability call=" << i
                  << " bytes_ok=" << (bytes_ok ? "true" : "false")
                  << " pointers_ok=" << (pointers_ok ? "true" : "false")
                  << " free_ok=" << (free_ok ? "true" : "false") << '\n';
    }

    std::ostringstream detail;
    detail << kAxis << "^3, " << kCalls << " sequential solve_streamfunctions calls, one fields+workspace pair";

    return {pass,
            "homogeneous_solver_repeated_reuse",
            "gpu-allocation-freedom",
            detail.str(),
            calls.empty() ? 0.0 : calls.front().r_F,
            calls.empty() ? 0.0 : calls.back().r_F,
            "every call satisfies the exact-control thresholds; bytes/pointers/free memory "
            "identical across calls 2 and 3 vs call 1",
            pass ? "stable" : "UNSTABLE",
            "3 repeated solve_streamfunctions calls on one owned fields+workspace pair at 32^3: "
            "every call's metrics individually satisfy the case-1 exact-control thresholds, and "
            "after the first call the observed allocated bytes, data pointers, and "
            "cudaMemGetInfo free bytes are exactly unchanged on calls 2 and 3 (hierarchy reuse, "
            "no reallocation)"};
}

} // namespace

CaseRegistry homogeneous_solver_case_registry() {
    return {{"homogeneous_solver_exact_16", case_homogeneous_solver_exact_16},
            {"homogeneous_solver_exact_32", case_homogeneous_solver_exact_32},
            {"homogeneous_solver_exact_64", case_homogeneous_solver_exact_64},
            {"homogeneous_solver_repeated_reuse", case_homogeneous_solver_repeated_reuse}};
}

} // namespace macroflow3d::streamfunctions::test
