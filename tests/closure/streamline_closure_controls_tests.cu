/**
 * @file streamline_closure_controls_tests.cu
 * @brief SF-30 N2 fast controls of the streamline-closure gate (16^3; one
 *        32^3 control2d solve for the refinement statement).
 *
 * Runs the full `closure_gate::run_closure_gate` pipeline (field -> SF-19
 * Darcy -> SF-28 spline -> first-return map) on the controls. Every gate was
 * pre-registered in the SF-30 UNDERSTAND record (section 4.10) and the N2
 * task specification before this file existed; none may be loosened.
 *
 * Cases: lester2021_closes, lester_brk_does_not_close,
 * control2d_planar_and_second_order, gaussian2d_is_planar,
 * gaussian_smoke_and_determinism, darcy_nonconvergence_is_reported,
 * seeds_file, timing_record.
 */

#include "apps/closure_gate/closure_gate.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::closure_gate;

namespace {

struct TestReport {
    bool overall_pass = true;
    int checks = 0;

    void check(bool cond, const std::string& name, const std::string& detail = "") {
        ++checks;
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(), detail.empty() ? "" : "  ",
                    detail.c_str());
        std::fflush(stdout);
        overall_pass = overall_pass && cond;
    }
};

std::string fmt(const char* f, double a) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a);
    return buf;
}

std::string fmt2(const char* f, double a, double b) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b);
    return buf;
}

std::string make_temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base && *base ? base : "/tmp") + "/sf30_controls_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (!mkdtemp(buf.data())) throw std::runtime_error("mkdtemp failed");
    return std::string(buf.data());
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

ClosureGateConfig analytic(const std::string& field, int n, double eps) {
    ClosureGateConfig c;
    c.field = field;
    c.n = n;
    c.has_eps = true;
    c.eps = eps;
    c.has_n_seeds = true;
    c.n_seeds = 64;
    return c;
}

ClosureGateConfig gaussian(const std::string& field, int n) {
    ClosureGateConfig c;
    c.field = field;
    c.n = n;
    c.has_sigma2 = c.has_ell = c.has_seed = true;
    c.sigma2 = 0.25;
    c.ell = 0.25;
    c.seed = 3001;
    c.has_n_seeds = true;
    c.n_seeds = 64;
    return c;
}

const ToleranceResult& at_tol(const ClosureGateResult& r, double tol) {
    for (const auto& t : r.tolerances)
        if (t.tol == tol) return t;
    throw std::runtime_error("tolerance not found");
}

void print_short(const ClosureGateResult& r) {
    std::printf("  %s %d^3: G=(%.12e, %.3e, %.3e) pcg iters=(%d,%d,%d) mean_flux=(%.3e,%.3e,%.3e) "
                "backflow_fraction=%.3e\n",
                r.config.field.c_str(), r.config.n, r.darcy.G[0], r.darcy.G[1], r.darcy.G[2],
                r.darcy.corrector_results[0].iterations, r.darcy.corrector_results[1].iterations,
                r.darcy.corrector_results[2].iterations, r.darcy.achieved_mean_flux[0], r.darcy.achieved_mean_flux[1],
                r.darcy.achieved_mean_flux[2], r.direction.backflow_volume_fraction);
    for (const auto& t : r.tolerances) {
        const PeriodStatistics& p = t.periods.front();
        std::printf("    tol=%.0e ok=%lld/%lld R=%.4e mean=(%.4e,%.4e) max|d|=%.4e max|d3|=%.4e <tau>_w=%.12f "
                    "nfev/streamline=%.1f\n",
                    t.tol, p.counts.ok, p.counts.seeds, p.unweighted.valid ? p.unweighted.s.R : NAN,
                    p.unweighted.s.mean[0], p.unweighted.s.mean[1], p.unweighted.s.max_abs, p.unweighted.max_abs_d3,
                    p.mean_tau_weighted, t.totals.field_evaluations_per_streamline_period);
    }
}

double R_at(const ClosureGateResult& r, double tol, int n = 1) {
    const PeriodStatistics& p = at_tol(r, tol).periods[static_cast<std::size_t>(n - 1)];
    return p.unweighted.valid ? p.unweighted.s.R : NAN;
}

// ---------------------------------------------------------------------------
// 1 + 2: lester2021 closes, lester_brk does not
// ---------------------------------------------------------------------------

double case_lester2021(TestReport& rep, CudaContext& ctx) {
    ClosureGateConfig c = analytic("lester2021", 16, 1.0);
    c.pcg_rtol = 1e-12;
    c.tols = {1e-8, 1e-10, 1e-12};
    c.working_tol = 1e-12;
    const ClosureGateResult r = run_closure_gate(ctx, c);
    print_short(r);
    rep.check(r.darcy_converged, "lester2021_darcy_converged");
    const double R = R_at(r, 1e-12);
    rep.check(R <= 1e-8, "lester2021_closes_R_at_1e-12", fmt("R=%.3e (gate <= 1e-8)", R));
    const PeriodStatistics& p = at_tol(r, 1e-12).periods.front();
    rep.check(p.counts.ok == 64 && p.counts.seeds == 64, "lester2021_all_64_ok",
              "ok=" + std::to_string(p.counts.ok));
    rep.check(p.counts.ok_with_backflow == 0 && r.direction.backflow_cells == 0, "lester2021_no_backflow",
              "ok_with_backflow=" + std::to_string(p.counts.ok_with_backflow));
    const double fe = std::max({std::abs(r.darcy.achieved_mean_flux[0] - 1.0), std::abs(r.darcy.achieved_mean_flux[1]),
                                std::abs(r.darcy.achieved_mean_flux[2])});
    rep.check(fe <= 1e-9, "lester2021_mean_flux_e1", fmt("max dev=%.3e (gate <= 1e-9)", fe));
    rep.check(std::abs(r.darcy.G[1]) <= 1e-9 && std::abs(r.darcy.G[2]) <= 1e-9, "lester2021_G_transverse_zero",
              fmt2("|G2|=%.3e |G3|=%.3e (gate <= 1e-9)", std::abs(r.darcy.G[1]), std::abs(r.darcy.G[2])));
    return R;
}

void case_lester_brk(TestReport& rep, CudaContext& ctx, double R_sym) {
    ClosureGateConfig c = analytic("lester_brk", 16, 1.0);
    c.pcg_rtol = 1e-12;
    c.tols = {1e-8, 1e-10, 1e-12};
    c.working_tol = 1e-12;
    const ClosureGateResult r = run_closure_gate(ctx, c);
    print_short(r);
    rep.check(r.darcy_converged, "lester_brk_darcy_converged");
    const double R = R_at(r, 1e-12);
    rep.check(R >= 1e-3, "lester_brk_R_ge_1e-3", fmt("R=%.4e (gate >= 1e-3)", R));
    rep.check(R >= 1e3 * R_sym, "lester_brk_R_ge_1e3_R_lester2021",
              fmt2("R=%.4e  1e3*R(lester2021)=%.4e", R, 1e3 * R_sym));
}

// ---------------------------------------------------------------------------
// 3: control2d planar and second order
// ---------------------------------------------------------------------------

void case_control2d(TestReport& rep, CudaContext& ctx) {
    double Rn[2] = {0, 0};
    const int grids[2] = {16, 32};
    for (int g = 0; g < 2; ++g) {
        ClosureGateConfig c = analytic("control2d", grids[g], 1.0);
        c.pcg_rtol = 1e-12;
        c.tols = {1e-10};
        c.working_tol = 1e-10;
        const ClosureGateResult r = run_closure_gate(ctx, c);
        print_short(r);
        const PeriodStatistics& p = at_tol(r, 1e-10).periods.front();
        rep.check(r.darcy_converged && p.counts.ok == 64, "control2d_" + std::to_string(grids[g]) + "_all_ok",
                  "ok=" + std::to_string(p.counts.ok));
        rep.check(p.unweighted.valid && p.unweighted.max_abs_d3 <= 1e-10,
                  "control2d_" + std::to_string(grids[g]) + "_planar_max_abs_d3",
                  fmt("max|d3|=%.3e (gate <= 1e-10)", p.unweighted.max_abs_d3));
        Rn[g] = R_at(r, 1e-10);
    }
    std::printf("  control2d R(16^3)=%.6e R(32^3)=%.6e ratio=%.3f\n", Rn[0], Rn[1], Rn[0] / Rn[1]);
    rep.check(Rn[1] <= Rn[0] / 2.5, "control2d_R_falls_2.5x_16_to_32",
              fmt2("R16=%.4e R32=%.4e", Rn[0], Rn[1]) + fmt(" ratio=%.3f (gate >= 2.5)", Rn[0] / Rn[1]));
}

// ---------------------------------------------------------------------------
// 4: gaussian2d is planar
// ---------------------------------------------------------------------------

void case_gaussian2d(TestReport& rep, CudaContext& ctx) {
    ClosureGateConfig c = gaussian("gaussian2d", 16);
    c.tols = {1e-8};
    c.working_tol = 1e-8;
    const ClosureGateResult r = run_closure_gate(ctx, c);
    print_short(r);
    std::printf("  x3 control: raw_variance=%.6e applied_scale=%.6e Y variance=%.17g\n",
                r.field.x3_control.raw_variance, r.field.x3_control.applied_scale, r.field.variance);
    const PeriodStatistics& p = at_tol(r, 1e-8).periods.front();
    rep.check(r.darcy_converged && p.counts.ok == 64, "gaussian2d_all_ok", "ok=" + std::to_string(p.counts.ok));
    rep.check(p.unweighted.valid && p.unweighted.max_abs_d3 <= 1e-10, "gaussian2d_planar_max_abs_d3",
              fmt("max|d3|=%.3e (gate <= 1e-10)", p.unweighted.max_abs_d3));
    const int N = 16;
    double kdev = 0.0;
    for (int k = 1; k < N; ++k)
        for (int c2 = 0; c2 < N * N; ++c2)
            kdev = std::max(kdev, std::abs(r.Y[static_cast<std::size_t>(c2 + N * N * k)] - r.Y[static_cast<std::size_t>(c2)]));
    rep.check(kdev == 0.0, "gaussian2d_independent_of_k", fmt("max k-dev=%.3e", kdev));
    rep.check(std::abs(r.field.variance - 0.25) <= 1e-12, "gaussian2d_variance_0.25",
              fmt("|var-0.25|=%.3e (gate <= 1e-12)", std::abs(r.field.variance - 0.25)));
}

// ---------------------------------------------------------------------------
// 5: gaussian smoke and determinism
// ---------------------------------------------------------------------------

void case_gaussian_smoke(TestReport& rep, CudaContext& ctx) {
    const std::string base = make_temp_dir();
    std::string sums[3], csvs[3];
    const int threads[3] = {4, 4, 1};
    ClosureGateResult last;
    for (int k = 0; k < 3; ++k) {
        ClosureGateConfig c = gaussian("gaussian", 16);
        c.periods = 2;
        c.threads = threads[k];
        c.out_dir = base + "/run" + std::to_string(k);
        ClosureGateResult r = run_closure_gate(ctx, c);
        sums[k] = read_file(c.out_dir + "/summary.json");
        csvs[k] = read_file(c.out_dir + "/streamlines.csv");
        if (k == 0) {
            print_short(r);
            last = std::move(r);
        }
    }
    const PeriodStatistics& p1 = at_tol(last, 1e-8).periods[0];
    rep.check(last.darcy_converged && p1.counts.ok == 64, "gaussian_smoke_completes_all_ok",
              "ok=" + std::to_string(p1.counts.ok) + " of " + std::to_string(p1.counts.seeds));
    rep.check(!sums[0].empty() && sums[0] == sums[1] && csvs[0] == csvs[1] && !csvs[0].empty(),
              "gaussian_smoke_two_runs_byte_identical",
              "summary bytes=" + std::to_string(sums[0].size()) + " csv bytes=" + std::to_string(csvs[0].size()));
    rep.check(sums[0] == sums[2] && csvs[0] == csvs[2], "gaussian_smoke_threads_1_vs_4_byte_identical");
    const ToleranceResult& t8 = at_tol(last, 1e-8);
    const ToleranceResult& t10 = at_tol(last, 1e-10);
    double l1 = NAN, l2 = NAN;
    for (const auto& e : t8.ladder) (e.n == 1 ? l1 : l2) = e.max_dist;
    std::printf("  ladder 1e-8 vs 1e-10: max dist n=1 %.3e, n=2 %.3e\n", l1, l2);
    rep.check(t8.has_ladder && t8.ladder_reference_tol == 1e-10 && l1 <= 1e-6, "gaussian_smoke_ladder_1e-8_vs_1e-10",
              fmt("max dist n=1 =%.3e (gate <= 1e-6)", l1));
    const PeriodStatistics& p2 = t10.periods.size() >= 2 ? t10.periods[1] : t10.periods[0];
    rep.check(p2.n == 2 && p2.unweighted.valid && p2.count == 64, "gaussian_smoke_period2_recorded",
              fmt2("n=2 R=%.4e max|d|=%.4e", p2.unweighted.s.R, p2.unweighted.s.max_abs));
    const bool csv_has_p2 = csvs[0].find(",y_2,z_2,s_2,tau_2") != std::string::npos;
    const bool json_has_schema = sums[0].find("\"sf30-closure-gate-1\"") != std::string::npos;
    rep.check(csv_has_p2 && json_has_schema, "gaussian_smoke_outputs_schema");
    std::printf("  balance checks (not gates): <tau>_w n=1 %.10f, flux-weighted mean d n=1 (%.3e, %.3e)\n",
                p1.mean_tau_weighted, p1.weighted.s.mean[0], p1.weighted.s.mean[1]);
    std::filesystem::remove_all(base);
}

// ---------------------------------------------------------------------------
// 6: Darcy non-convergence
// ---------------------------------------------------------------------------

void case_nonconvergence(TestReport& rep, CudaContext& ctx) {
    const std::string base = make_temp_dir();
    ClosureGateConfig c = analytic("lester_brk", 16, 1.0);
    c.pcg_max_iter = 1;
    c.out_dir = base;
    const ClosureGateResult r = run_closure_gate(ctx, c);
    std::printf("  pcg iters=(%d,%d,%d) converged=(%d,%d,%d)\n", r.darcy.corrector_results[0].iterations,
                r.darcy.corrector_results[1].iterations, r.darcy.corrector_results[2].iterations,
                r.darcy.corrector_results[0].converged, r.darcy.corrector_results[1].converged,
                r.darcy.corrector_results[2].converged);
    rep.check(!r.darcy_converged, "darcy_nonconvergence_reported");
    rep.check(r.tolerances.empty(), "darcy_nonconvergence_no_streamline_statistics");
    const std::string s = read_file(base + "/summary.json");
    rep.check(s.find("\"darcy_converged\": false") != std::string::npos &&
                  s.find("\"tolerances\": null") != std::string::npos &&
                  !std::filesystem::exists(base + "/streamlines.csv"),
              "darcy_nonconvergence_summary_written");
    std::filesystem::remove_all(base);
}

// ---------------------------------------------------------------------------
// 7: seeds file
// ---------------------------------------------------------------------------

void case_seeds_file(TestReport& rep, CudaContext& ctx) {
    const std::string base = make_temp_dir();
    const double pts[2][2] = {{0.08564916714362436, 0.28420116374879145}, {0.1 + 1.0 / 3.0, 0.7071067811865476}};
    {
        std::ofstream f(base + "/seeds.csv");
        f << "y0,z0\n";
        for (const auto& p : pts) f << detail_gate::fmt17(p[0]) << ',' << detail_gate::fmt17(p[1]) << '\n';
    }
    ClosureGateConfig c = analytic("lester2021", 16, 1.0);
    c.has_n_seeds = false;
    c.seeds_file = base + "/seeds.csv";
    c.tols = {1e-8};
    c.working_tol = 1e-8;
    c.out_dir = base + "/out";
    const ClosureGateResult r = run_closure_gate(ctx, c);
    bool exact = r.seeds.size() == 2;
    for (int i = 0; exact && i < 2; ++i)
        exact = std::memcmp(&r.seeds[i][0], &pts[i][0], 8) == 0 && std::memcmp(&r.seeds[i][1], &pts[i][1], 8) == 0;
    rep.check(exact, "seeds_file_read_back_bitwise", "n=" + std::to_string(r.seeds.size()));
    const PeriodStatistics& p = at_tol(r, 1e-8).periods.front();
    rep.check(p.counts.seeds == 2 && p.counts.ok == 2, "seeds_file_used",
              "seeds=" + std::to_string(p.counts.seeds) + " ok=" + std::to_string(p.counts.ok));
    // The CSV columns y0, z0 also round-trip exactly.
    std::istringstream csv(read_file(c.out_dir + "/streamlines.csv"));
    std::string line;
    std::getline(csv, line);
    bool csv_exact = true;
    for (int i = 0; i < 2; ++i) {
        std::getline(csv, line);
        std::istringstream ls(line);
        std::string id, y, z;
        std::getline(ls, id, ',');
        std::getline(ls, y, ',');
        std::getline(ls, z, ',');
        const double yv = std::strtod(y.c_str(), nullptr), zv = std::strtod(z.c_str(), nullptr);
        csv_exact = csv_exact && std::memcmp(&yv, &pts[i][0], 8) == 0 && std::memcmp(&zv, &pts[i][1], 8) == 0;
    }
    rep.check(csv_exact, "seeds_file_csv_17_digit_round_trip");
    std::filesystem::remove_all(base);
}

} // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    TestReport rep;
    try {
        CudaContext ctx(0);
        std::printf("=== SF-30 N2: 1 lester2021_closes ===\n");
        const double R_sym = case_lester2021(rep, ctx);
        std::printf("=== SF-30 N2: 2 lester_brk_does_not_close ===\n");
        case_lester_brk(rep, ctx, R_sym);
        std::printf("=== SF-30 N2: 3 control2d_planar_and_second_order ===\n");
        case_control2d(rep, ctx);
        std::printf("=== SF-30 N2: 4 gaussian2d_is_planar ===\n");
        case_gaussian2d(rep, ctx);
        std::printf("=== SF-30 N2: 5 gaussian_smoke_and_determinism ===\n");
        case_gaussian_smoke(rep, ctx);
        std::printf("=== SF-30 N2: 6 darcy_nonconvergence_is_reported ===\n");
        case_nonconvergence(rep, ctx);
        std::printf("=== SF-30 N2: 7 seeds_file ===\n");
        case_seeds_file(rep, ctx);
    } catch (const std::exception& e) {
        rep.check(false, "unexpected_exception", e.what());
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("=== SF-30 N2: 8 timing_record ===\n  wall time %.3f s (no gate; must be well under 60 s)\n", secs);
    std::printf("\n=== streamline_closure_controls16: %d checks, %s ===\n", rep.checks,
                rep.overall_pass ? "PASS" : "FAIL");
    return rep.overall_pass ? 0 : 1;
}
