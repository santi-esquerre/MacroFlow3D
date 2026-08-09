#include "streamfunction_operator_test_cases.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/StreamfunctionSolver.cuh"
#include "src/physics/streamfunctions/StreamfunctionTypes.hpp"
#include "src/physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "src/physics/streamfunctions/affine_gauge.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// SF-14 T02: fixed-relaxation Picard manufactured-nonlinear GPU acceptance
// cases for `solve_streamfunctions` (StreamfunctionSolver.cuh/.cu, SF-14 T01
// fixed-relaxation Picard loop). These cases exercise the coupled nonlinear
// system on a manufactured heterogeneous conductivity field and assert the
// documented SF-14 `PicardIterationRecord`/history-layout semantics; they
// never touch `src/**`. Amplitudes and thresholds are prespecified in the
// increment task and are never tuned after seeing results (see the task's
// rollback policy): a gating-case convergence failure is reported honestly.

namespace macroflow3d::streamfunctions::test {
namespace {

// ---------------------------------------------------------------------------
// Shared helpers, following streamfunction_homogeneous_gpu_cases.cu.
// ---------------------------------------------------------------------------

constexpr double kPi = 3.14159265358979323846264338327950288;

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

// All config defaults per the task spec: omega=0.25, picard.tolerance=1e-6,
// picard.max_iter=500, PCG rtol 1e-10 (already the ProjectedPCGConfig
// default), eta=1, epsilon=1e-2.
[[nodiscard]] StreamfunctionSolverConfig valid_config() { return StreamfunctionSolverConfig{}; }

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

template <typename T>
[[nodiscard]] bool finite_value(T value) {
    return std::isfinite(static_cast<double>(value));
}

[[nodiscard]] bool pcg_result_finite(const solvers::ProjectedPCGResult& result) {
    return finite_value(result.raw_rhs_mean) && finite_value(result.raw_rhs_l2_norm) &&
           finite_value(result.raw_rhs_compatibility_defect) &&
           finite_value(result.initial_projected_residual) &&
           finite_value(result.final_projected_residual) &&
           finite_value(result.relative_projected_residual) && finite_value(result.final_field_mean);
}

// Uploads a manufactured heterogeneous K = exp(a*sin(2*pi*x)*sin(2*pi*y)*
// sin(2*pi*z)) conductivity field (host-computed, then uploaded), and the
// uniform reference Darcy field v=(1,0,0) consistent with the benchmark
// gauge g1=(0,1,0), g2=(0,0,1): v = g1 x g2 = (1,0,0).
struct ManufacturedProblemBuffers {
    DeviceBuffer<real> conductivity;
    DeviceBuffer<real> darcy_u;
    DeviceBuffer<real> darcy_v;
    DeviceBuffer<real> darcy_w;

    explicit ManufacturedProblemBuffers(const Grid3D& grid, double amplitude) : conductivity(grid.num_cells()),
        darcy_u(compact_mac_u_size(grid)), darcy_v(compact_mac_v_size(grid)),
        darcy_w(compact_mac_w_size(grid)) {
        std::vector<real> k_host(grid.num_cells());
        for (int iz = 0; iz < grid.nz; ++iz) {
            const double z = (iz + 0.5) * static_cast<double>(grid.dz);
            for (int iy = 0; iy < grid.ny; ++iy) {
                const double y = (iy + 0.5) * static_cast<double>(grid.dy);
                for (int ix = 0; ix < grid.nx; ++ix) {
                    const double x = (ix + 0.5) * static_cast<double>(grid.dx);
                    const double log_k =
                        amplitude * std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y) * std::sin(2.0 * kPi * z);
                    const std::size_t index = grid.idx(ix, iy, iz);
                    k_host[index] = static_cast<real>(std::exp(log_k));
                }
            }
        }
        const std::vector<real> ones_u(compact_mac_u_size(grid), real{1});
        const std::vector<real> zeros_v(compact_mac_v_size(grid), real{0});
        const std::vector<real> zeros_w(compact_mac_w_size(grid), real{0});
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(conductivity.data(), k_host.data(), k_host.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_u.data(), ones_u.data(), ones_u.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_v.data(), zeros_v.data(), zeros_v.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_w.data(), zeros_w.data(), zeros_w.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
    }
};

[[nodiscard]] StreamfunctionProblemView manufactured_problem_view(const Grid3D& grid,
                                                                   const ManufacturedProblemBuffers& buffers) {
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

// Homogeneous (K=1) problem buffers, following
// streamfunction_homogeneous_gpu_cases.cu, used by case 3.
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

template <typename Callable>
[[nodiscard]] bool rejects_with_invalid_argument(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "picard_fixed_contract name=" << name
                  << " exception=none expected=std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument& error) {
        std::cout << "picard_fixed_contract name=" << name
                  << " exception=std::invalid_argument message=" << error.what() << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cout << "picard_fixed_contract name=" << name
                  << " exception=std::exception message=" << error.what()
                  << " expected=std::invalid_argument\n";
        return false;
    } catch (...) {
        std::cout << "picard_fixed_contract name=" << name
                  << " exception=non-standard expected=std::invalid_argument\n";
        return false;
    }
}

template <typename Callable>
[[nodiscard]] bool accepts_without_throw(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "picard_fixed_contract name=" << name << " exception=none expected=none\n";
        return true;
    } catch (const std::exception& error) {
        std::cout << "picard_fixed_contract name=" << name << " exception=" << error.what()
                  << " expected=none\n";
        return false;
    } catch (...) {
        std::cout << "picard_fixed_contract name=" << name << " exception=non-standard expected=none\n";
        return false;
    }
}

// True iff `report.picard_history` has the exact size/converged layout the
// SF-14 header documents for `report.status`:
//   - converged: size == picard_iterations + 1.
//   - not_converged, budget exhaustion: size == max_iter + 1 (== iterations+1
//     since iterations==max_iter here), and every recorded block solve
//     converged.
//   - not_converged, linear-block failure: size == iterations + 2 (the
//     iterations+1 state entries plus one truncated failing entry), and the
//     final entry holds a non-converged block solve.
[[nodiscard]] bool history_layout_consistent(const StreamfunctionSolveReport& report) {
    const auto& history = report.picard_history;
    const auto iterations = static_cast<std::size_t>(report.picard_iterations);
    if (report.status == StreamfunctionSolveStatus::converged) {
        return history.size() == iterations + 1;
    }
    if (report.status == StreamfunctionSolveStatus::not_converged) {
        if (history.empty()) return false;
        if (history.size() == iterations + 1) {
            for (const auto& entry : history) {
                if (!entry.psi1_result.converged || !entry.psi2_result.converged) {
                    // Entry 0 carries default-constructed (not-yet-run)
                    // results; only skip that one entry's converged flag.
                    if (&entry == &history.front()) continue;
                    return false;
                }
            }
            return true;
        }
        if (history.size() == iterations + 2) {
            const auto& last = history.back();
            return !last.psi1_result.converged || !last.psi2_result.converged;
        }
        return false;
    }
    return false;
}

// The trailing entry of a linear-block-failure history (the layout
// documented on `history_layout_consistent` above: history.size() ==
// iterations + 2 with a non-converged last entry) is the ONLY layout where a
// history entry's r_F/r1/r2 are the documented NaN sentinel ("never
// evaluated") rather than a finite value; every other entry, in every other
// layout, must still be strictly finite.
[[nodiscard]] bool history_scalars_finite(const StreamfunctionSolveReport& report) {
    const auto& history = report.picard_history;
    const auto iterations = static_cast<std::size_t>(report.picard_iterations);
    const bool has_trailing_failure_record =
        report.status == StreamfunctionSolveStatus::not_converged &&
        history.size() == iterations + 2 && !history.empty() &&
        (!history.back().psi1_result.converged || !history.back().psi2_result.converged);

    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto& entry = history[i];
        const bool is_trailing_failure_entry = has_trailing_failure_record && i + 1 == history.size();
        if (is_trailing_failure_entry) {
            if (!std::isnan(static_cast<double>(entry.r_F)) ||
                !std::isnan(static_cast<double>(entry.r1)) ||
                !std::isnan(static_cast<double>(entry.r2))) {
                return false;
            }
        } else {
            if (!finite_value(entry.r_F) || !finite_value(entry.r1) || !finite_value(entry.r2)) return false;
        }
        if (!pcg_result_finite(entry.psi1_result) || !pcg_result_finite(entry.psi2_result)) return false;
    }
    return true;
}

[[nodiscard]] bool report_scalars_finite(const StreamfunctionSolveReport& report) {
    if (!finite_value(report.residual.rms_f1) || !finite_value(report.residual.rms_f2) ||
        !finite_value(report.residual.linf_f1) || !finite_value(report.residual.linf_f2) ||
        !finite_value(report.residual.q_rms) || !finite_value(report.residual.r_F) ||
        !finite_value(report.residual.r1) || !finite_value(report.residual.r2)) {
        return false;
    }
    if (!finite_value(report.diagnostics.e_v) || !finite_value(report.diagnostics.e_div) ||
        !finite_value(report.diagnostics.invariance_e_psi1) ||
        !finite_value(report.diagnostics.invariance_e_psi2) || !finite_value(report.diagnostics.c_min) ||
        !finite_value(report.diagnostics.c_max) || !finite_value(report.diagnostics.c_mean) ||
        !finite_value(report.diagnostics.v_d_rms) || !finite_value(report.diagnostics.L_ref)) {
        return false;
    }
    if (!pcg_result_finite(report.psi1_result) || !pcg_result_finite(report.psi2_result)) return false;
    return history_scalars_finite(report);
}

struct ManufacturedRunResult {
    StreamfunctionSolveReport report;
    double rms_u1{};
    double rms_u2{};
    double mean_u1{};
    double mean_u2{};
    double wall_seconds{};
};

[[nodiscard]] ManufacturedRunResult run_manufactured(int n, double amplitude,
                                                       const StreamfunctionSolverConfig& config) {
    const Grid3D grid = isotropic_grid(n);

    CudaContext ctx(0);
    ManufacturedProblemBuffers buffers(grid, amplitude);
    const StreamfunctionProblemView problem = manufactured_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    const auto t0 = std::chrono::steady_clock::now();
    ManufacturedRunResult result;
    result.report = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    result.wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    const std::vector<real> u1_host = download(fields.u1_span());
    const std::vector<real> u2_host = download(fields.u2_span());
    result.rms_u1 = host_rms(u1_host);
    result.rms_u2 = host_rms(u2_host);
    result.mean_u1 = host_mean(u1_host);
    result.mean_u2 = host_mean(u2_host);
    return result;
}

void print_metric_block(const char* label, int n, double amplitude, const ManufacturedRunResult& run,
                         const StreamfunctionSolverConfig& config) {
    const auto& report = run.report;
    std::cout << std::setprecision(17) << label << " N=" << n << " a=" << amplitude
              << " status="
              << (report.status == StreamfunctionSolveStatus::converged        ? "converged"
                  : report.status == StreamfunctionSolveStatus::not_converged  ? "not_converged"
                  : report.status == StreamfunctionSolveStatus::invalid_problem ? "invalid_problem"
                                                                                 : "not_run")
              << " picard_iterations=" << report.picard_iterations
              << " history_size=" << report.picard_history.size() << " wall_seconds=" << run.wall_seconds
              << '\n';

    if (!report.picard_history.empty()) {
        std::cout << "  r_F trajectory: entry0=" << report.picard_history.front().r_F;
        if (report.picard_history.size() > 2) {
            const std::size_t mid = report.picard_history.size() / 2;
            std::cout << " mid[" << mid << "]=" << report.picard_history[mid].r_F;
        }
        std::cout << " final=" << report.picard_history.back().r_F << '\n';

        std::cout << "  full r_F list:";
        for (const auto& entry : report.picard_history) std::cout << ' ' << entry.r_F;
        std::cout << '\n';

        std::cout << "  per-iteration PCG iterations (psi1,psi2):";
        for (const auto& entry : report.picard_history) {
            std::cout << " (" << entry.psi1_result.iterations << ',' << entry.psi2_result.iterations << ')';
        }
        std::cout << '\n';
    }

    std::cout << "  final r1=" << report.residual.r1 << " r2=" << report.residual.r2
              << " q_rms=" << report.residual.q_rms << " v_rms=" << report.diagnostics.v_d_rms
              << " L_ref=" << report.diagnostics.L_ref << '\n'
              << "  gauge mean_u1=" << run.mean_u1 << " mean_u2=" << run.mean_u2 << '\n'
              << "  |c| min=" << report.diagnostics.c_min << " max=" << report.diagnostics.c_max
              << " mean=" << report.diagnostics.c_mean << '\n'
              << "  degeneracy counts (thresholds=" << report.diagnostics.num_degeneracy_thresholds
              << "):";
    for (int t = 0; t < report.diagnostics.num_degeneracy_thresholds; ++t) {
        std::cout << ' ' << report.diagnostics.degeneracy_total[t];
    }
    std::cout << '\n'
              << "  e_v=" << report.diagnostics.e_v << " e_div=" << report.diagnostics.e_div
              << " invariance_e_psi1=" << report.diagnostics.invariance_e_psi1
              << " invariance_e_psi2=" << report.diagnostics.invariance_e_psi2
              << " [vs uniform reference -- DIAGNOSTIC ONLY, not a gate (no true Darcy solve until "
                 "SF-19)]\n"
              << "  epsilon=" << config.epsilon << " eta=" << config.eta
              << " omega=" << config.picard.omega << '\n';
}

// ---------------------------------------------------------------------------
// Case 1 (shared implementation): picard_fixed_manufactured_{16,32} (GATING).
// ---------------------------------------------------------------------------

constexpr double kManufacturedAmplitude = 0.25;

[[nodiscard]] CaseResult run_manufactured_gating_case(int n) {
    const std::string case_name = "picard_fixed_manufactured_" + std::to_string(n);
    const StreamfunctionSolverConfig config = valid_config();

    const ManufacturedRunResult run = run_manufactured(n, kManufacturedAmplitude, config);
    const auto& report = run.report;

    print_metric_block(case_name.c_str(), n, kManufacturedAmplitude, run, config);

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("status_converged", report.status == StreamfunctionSolveStatus::converged);
    add_check("r_F_le_tolerance", static_cast<double>(report.residual.r_F) <= 1e-6);
    add_check("picard_iterations_le_500", report.picard_iterations <= 500);

    bool every_block_solve_ok = true;
    for (std::size_t k = 1; k < report.picard_history.size(); ++k) {
        const auto& entry = report.picard_history[k];
        every_block_solve_ok = every_block_solve_ok && entry.psi1_result.converged &&
                               entry.psi2_result.converged &&
                               static_cast<double>(entry.psi1_result.relative_projected_residual) <= 1e-10 &&
                               static_cast<double>(entry.psi2_result.relative_projected_residual) <= 1e-10;
    }
    add_check("every_recorded_block_solve_converged_and_tight", every_block_solve_ok);

    const double eps_machine = static_cast<double>(std::numeric_limits<real>::epsilon());
    const double gauge_tol_u1 = 100.0 * eps_machine * std::max(run.rms_u1, 1.0);
    const double gauge_tol_u2 = 100.0 * eps_machine * std::max(run.rms_u2, 1.0);
    add_check("gauge_mean_u1", std::abs(run.mean_u1) <= gauge_tol_u1);
    add_check("gauge_mean_u2", std::abs(run.mean_u2) <= gauge_tol_u2);

    add_check("all_report_scalars_finite", report_scalars_finite(report));
    add_check("history_layout_consistent", history_layout_consistent(report));

    const bool history_r_f_finite =
        std::all_of(report.picard_history.begin(), report.picard_history.end(),
                    [](const PicardIterationRecord& entry) { return finite_value(entry.r_F); });
    add_check("history_r_F_all_finite", history_r_f_finite);

    const bool required_nonlinear_iteration =
        !report.picard_history.empty() && static_cast<double>(report.picard_history.front().r_F) > 1e-6;
    add_check("case_not_vacuous_entry0_above_tolerance", required_nonlinear_iteration);

    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << n << "^3, K=exp(" << kManufacturedAmplitude << "*sin(2pi x)sin(2pi y)sin(2pi z)), "
           << "benchmark gauge, uniform Darcy v=(1,0,0)";

    return {pass,
            case_name,
            "gpu-picard-manufactured-gating",
            detail.str(),
            static_cast<double>(report.residual.r_F),
            static_cast<double>(report.picard_iterations),
            "r_F<=1e-6 within <=500 Picard iterations",
            pass ? "all pass" : "some failed",
            "manufactured heterogeneous conductivity gating case: fixed-relaxation Picard on the "
            "coupled Lester equation (14) system must converge within budget on the triply "
            "periodic unit torus; see the printed Gate 3A metric block above"};
}

[[nodiscard]] CaseResult case_picard_fixed_manufactured_16() { return run_manufactured_gating_case(16); }
[[nodiscard]] CaseResult case_picard_fixed_manufactured_32() { return run_manufactured_gating_case(32); }

// ---------------------------------------------------------------------------
// Case 2: picard_fixed_research_amplitude_05 (RESEARCH, no convergence gate).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_picard_fixed_research_amplitude_05() {
    constexpr int kAxis = 32;
    constexpr double kAmplitude = 0.5;
    const StreamfunctionSolverConfig config = valid_config();

    const ManufacturedRunResult run = run_manufactured(kAxis, kAmplitude, config);
    const auto& report = run.report;

    print_metric_block("picard_fixed_research_amplitude_05", kAxis, kAmplitude, run, config);

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("status_not_invalid_problem", report.status != StreamfunctionSolveStatus::invalid_problem);
    add_check("all_report_scalars_finite", report_scalars_finite(report));
    add_check("history_layout_consistent", history_layout_consistent(report));

    const char* outcome_label = report.status == StreamfunctionSolveStatus::converged     ? "CONVERGED"
                                : report.status == StreamfunctionSolveStatus::not_converged ? "NOT_CONVERGED"
                                                                                             : "INVALID_PROBLEM";
    std::cout << "  *** RESEARCH OUTCOME (a=" << kAmplitude << "): " << outcome_label
              << " after picard_iterations=" << report.picard_iterations
              << " final r_F=" << report.residual.r_F << " (recorded as research data either way) ***\n";

    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << kAxis << "^3, K=exp(" << kAmplitude << "*sin(2pi x)sin(2pi y)sin(2pi z)), "
           << "benchmark gauge, uniform Darcy v=(1,0,0)";

    return {pass,
            "picard_fixed_research_amplitude_05",
            "gpu-picard-manufactured-research",
            detail.str(),
            static_cast<double>(report.residual.r_F),
            static_cast<double>(report.picard_iterations),
            "no exception, status!=invalid_problem, all scalars finite, history layout consistent",
            outcome_label,
            "research amplitude (a=0.5) case: the fixed-relaxation Picard convergence outcome is "
            "recorded honestly (never gated/tuned); see the printed Gate 3A metric block and "
            "RESEARCH OUTCOME line above"};
}

// ---------------------------------------------------------------------------
// Case 3: picard_fixed_homogeneous_zero_iterations.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_picard_fixed_homogeneous_zero_iterations() {
    constexpr int kAxis = 32;
    const Grid3D grid = isotropic_grid(kAxis);
    const StreamfunctionSolverConfig config = valid_config();

    CudaContext ctx(0);
    HomogeneousProblemBuffers buffers(grid);
    const StreamfunctionProblemView problem = homogeneous_problem_view(grid, buffers);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    struct CallMetrics {
        StreamfunctionSolveStatus status{};
        int picard_iterations{};
        std::size_t history_size{};
        real history0_r_f{};
        real r_f{};
        std::size_t bytes{};
        std::vector<const void*> pointers;
    };

    const auto observed_pointers = [&] {
        return std::vector<const void*>{fields.u1_span().data(), fields.u2_span().data(),
                                        workspace.q().data(),    workspace.rhs1().data(),
                                        workspace.rhs2().data(), workspace.f1().data(),
                                        workspace.f2().data()};
    };
    const auto observed_bytes = [&] {
        return fields.allocated_device_bytes() + workspace.allocated_device_bytes();
    };

    std::vector<CallMetrics> calls;
    bool pass = true;
    for (int i = 0; i < 2; ++i) {
        const StreamfunctionSolveReport report = solve_streamfunctions(ctx, problem, config, fields, workspace);
        ctx.synchronize();

        CallMetrics metrics;
        metrics.status = report.status;
        metrics.picard_iterations = report.picard_iterations;
        metrics.history_size = report.picard_history.size();
        metrics.history0_r_f = report.picard_history.empty() ? real{-1} : report.picard_history.front().r_F;
        metrics.r_f = report.residual.r_F;
        metrics.bytes = observed_bytes();
        metrics.pointers = observed_pointers();
        calls.push_back(metrics);

        const bool call_ok = report.status == StreamfunctionSolveStatus::converged &&
                             report.picard_iterations == 0 && report.picard_history.size() == 1 &&
                             report.picard_history.front().r_F == real{0} && report.residual.r_F == real{0};
        pass = pass && call_ok;

        std::cout << std::setprecision(17) << "picard_fixed_homogeneous_zero_iterations call=" << i
                  << " status="
                  << (report.status == StreamfunctionSolveStatus::converged ? "converged" : "OTHER")
                  << " picard_iterations=" << report.picard_iterations
                  << " history_size=" << report.picard_history.size() << " history0_r_F="
                  << metrics.history0_r_f << " r_F=" << metrics.r_f << " bytes=" << metrics.bytes
                  << " check=" << (call_ok ? "PASS" : "FAIL") << '\n';
    }

    const bool bytes_ok = calls.size() == 2 && calls[1].bytes == calls[0].bytes;
    const bool pointers_ok = calls.size() == 2 && calls[1].pointers == calls[0].pointers;
    pass = pass && bytes_ok && pointers_ok;
    std::cout << "picard_fixed_homogeneous_zero_iterations stability bytes_ok="
              << (bytes_ok ? "true" : "false") << " pointers_ok=" << (pointers_ok ? "true" : "false")
              << '\n';

    std::ostringstream detail;
    detail << kAxis << "^3, K=1, benchmark gauge, uniform Darcy v=(1,0,0), 2 sequential calls";

    return {pass,
            "picard_fixed_homogeneous_zero_iterations",
            "gpu-picard-homogeneous-control",
            detail.str(),
            calls.empty() ? 0.0 : static_cast<double>(calls.front().r_f),
            calls.empty() ? 0.0 : static_cast<double>(calls.back().r_f),
            "picard_iterations==0, history.size()==1, history[0].r_F==0 exactly, byte/pointer "
            "stability across 2 calls",
            pass ? "all pass" : "some failed",
            "homogeneous K=1 control: the coupled residual is exactly zero at state 0, so the "
            "Picard loop converges with zero update steps on every call, and the workspace is "
            "exactly reused (no reallocation) across repeated calls"};
}

// ---------------------------------------------------------------------------
// Case 4: picard_fixed_config_error_contract.
// ---------------------------------------------------------------------------

[[nodiscard]] StreamfunctionProblemView nullptr_problem_view(const Grid3D& grid) {
    StreamfunctionProblemView problem;
    problem.grid = grid;
    problem.conductivity = DeviceSpan<const real>(nullptr, grid.num_cells());
    problem.conductivity_representation = ConductivityRepresentation::conductivity_k;
    problem.darcy_velocity = CompactMacVelocityConstView{
        DeviceSpan<const real>(nullptr, compact_mac_u_size(grid)),
        DeviceSpan<const real>(nullptr, compact_mac_v_size(grid)),
        DeviceSpan<const real>(nullptr, compact_mac_w_size(grid))};
    problem.bc = triply_periodic();
    problem.gauge = AffineGauge::benchmark(real{1});
    return problem;
}

[[nodiscard]] CaseResult case_picard_fixed_config_error_contract() {
    const Grid3D base_grid = isotropic_grid(16);
    const StreamfunctionProblemView base_problem = nullptr_problem_view(base_grid);
    const StreamfunctionSolverConfig base_config = valid_config();

    std::size_t checks = 0;
    bool pass = true;
    const auto require_invalid = [&](const char* name, const auto& callable) {
        ++checks;
        pass = rejects_with_invalid_argument(name, callable) && pass;
    };
    const auto require_ok = [&](const char* name, const auto& callable) {
        ++checks;
        pass = accepts_without_throw(name, callable) && pass;
    };

    require_invalid("picard_max_iter_negative", [&] {
        auto config = base_config;
        config.picard.max_iter = -1;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("picard_tolerance_zero", [&] {
        auto config = base_config;
        config.picard.tolerance = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("picard_tolerance_negative", [&] {
        auto config = base_config;
        config.picard.tolerance = real{-1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("picard_tolerance_nan", [&] {
        auto config = base_config;
        config.picard.tolerance = std::numeric_limits<real>::quiet_NaN();
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("picard_tolerance_inf", [&] {
        auto config = base_config;
        config.picard.tolerance = std::numeric_limits<real>::infinity();
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("picard_omega_zero", [&] {
        auto config = base_config;
        config.picard.omega = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("picard_omega_negative", [&] {
        auto config = base_config;
        config.picard.omega = real{-0.25};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("picard_omega_gt_one", [&] {
        auto config = base_config;
        config.picard.omega = real{1.5};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("picard_omega_nan", [&] {
        auto config = base_config;
        config.picard.omega = std::numeric_limits<real>::quiet_NaN();
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    require_ok("picard_max_iter_zero_valid", [&] {
        auto config = base_config;
        config.picard.max_iter = 0;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_ok("picard_omega_one_boundary_valid", [&] {
        auto config = base_config;
        config.picard.omega = real{1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    std::cout << "picard_fixed_config_error_contract total_checks=" << checks
              << " pass=" << (pass ? "true" : "false") << '\n';

    return {pass,
            "picard_fixed_config_error_contract",
            "contract",
            "16x16x16 + config.picard mutations",
            static_cast<double>(checks),
            0.0,
            "all checks pass",
            pass ? "all pass" : "some failed",
            "validate_streamfunction_problem throws std::invalid_argument for every documented "
            "config.picard precondition violation (max_iter<0, tolerance<=0/nonfinite, omega "
            "outside (0,1]/nonfinite); max_iter=0 and omega=1 are valid boundary values"};
}

} // namespace

CaseRegistry picard_fixed_case_registry() {
    return {{"picard_fixed_manufactured_16", case_picard_fixed_manufactured_16},
            {"picard_fixed_manufactured_32", case_picard_fixed_manufactured_32},
            {"picard_fixed_research_amplitude_05", case_picard_fixed_research_amplitude_05},
            {"picard_fixed_homogeneous_zero_iterations", case_picard_fixed_homogeneous_zero_iterations},
            {"picard_fixed_config_error_contract", case_picard_fixed_config_error_contract}};
}

} // namespace macroflow3d::streamfunctions::test
