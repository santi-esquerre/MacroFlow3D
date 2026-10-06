/**
 * @file spurious_spreading_controls_tests.cu
 * @brief SF-32 N2b: fast 16^3 controls of the `spurious_spreading`
 *        instrument (ctest `spurious_spreading_controls16`).
 *
 * Drives the instrument's LIBRARY code (apps/spurious_spreading/
 * return_map_run.{hpp,cu}: compute_return_map, seeds_csv_text,
 * summary_json_text, write_run_dir) with in-memory analytic label pairs
 * (sample_analytic_pair, the code path of `analytic-labels`); the executable
 * is never spawned. Style of tests/streamline_tracker/*_tests.cu: printed
 * [PASS]/[FAIL] lines, main returns 0/1. Fast tier.
 *
 * Every pair, grid, seed count, tolerance and threshold below was
 * PRE-REGISTERED by the SF-32 orchestrator (node N2b specification, section
 * 3.2; orchestration record section 4) before this file existed; none may be
 * loosened to make a failing line pass.
 *
 *  - pair_U (N = 16, 256 seeds): all three trackers return every seed with
 *    |delta_x2|, |delta_x3| <= 1e-13, |tau - 1| <= 1e-12, status 0,
 *    land_err <= 1e-12.
 *  - pair_A (a = 0.1, N = 16): pseudo-symplectic (tol_psi 1e-10, ds = h/2)
 *    max |delta_psi_i| <= 1e-10 + 1e-14 and |delta_x2| <= 1e-6; RK (tol 1e-8,
 *    dt_max 0.25) |delta_x2| <= 1e-6; Pollock (m = 1) |delta_x2|, |delta_x3|,
 *    |tau - 1| <= 1e-13; all statuses 0.
 *  - pair_B (a = 0.1, b = 0.08): Pollock var(delta_x3) at n = 32 (N = 32,
 *    m = 1) < at n = 16 (N = 16, m = 1), both printed; divergence_max_rel
 *    <= 1e-13.
 *  - pair_G (e = 0.03, N = 16): the three trackers finish with every seed
 *    status 0; pseudo-symplectic max |delta_psi_i| <= 1e-10 + 1e-14; RK and
 *    Pollock rms(delta_x) printed (measured quantities, no gate); summary
 *    statistics recomputed here from the per-seed arrays equal the
 *    instrument's summary values to 1e-15 relative.
 *  - byte_reproducibility (pair G, each tracker): two runs with no_timing give
 *    identical seeds.csv and summary.json texts, and the run directory written
 *    twice (temporary directory under the working directory, removed
 *    afterwards) has identical file bytes.
 *  - delta_ratio_must_divide_N: compute_return_map throws UsageError for m = 3
 *    on N = 16 (the executable maps it to exit code 2).
 */

#include "apps/spurious_spreading/label_routes.hpp"
#include "apps/spurious_spreading/return_map_run.hpp"

#include "src/runtime/CudaContext.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace spurious_spreading;
using macroflow3d::CudaContext;

namespace {

struct TestReport {
    bool overall_pass = true;
    int checks = 0;
    int failed = 0;

    void check(bool cond, const std::string& name, const std::string& detail = "") {
        ++checks;
        if (!cond)
            ++failed;
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(),
                    detail.empty() ? "" : "  ", detail.c_str());
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

constexpr int kN = 16;
constexpr long long kSeeds = 256;

LoadedLabels make_labels(AnalyticPair pair, double amp, double amp_b, int N) {
    const AnalyticPairParams prm{pair, amp, amp_b};
    LoadedLabels L;
    L.n = N;
    L.h = 1.0 / static_cast<double>(N);
    L.gbar1[0] = 0.0;
    L.gbar1[1] = 1.0;
    L.gbar1[2] = 0.0;
    L.gbar2[0] = 0.0;
    L.gbar2[1] = 0.0;
    L.gbar2[2] = 1.0;
    double min_c_exact = 0.0;
    sample_analytic_pair(prm, N, L.u1, L.u2, min_c_exact);
    L.meta = jobj({{"route", "analytic"},
                   {"pair", analytic_pair_name(pair)},
                   {"amplitude", amp},
                   {"amplitude_b", pair == AnalyticPair::B ? JVal(amp_b) : JVal(nullptr)},
                   {"n", N},
                   {"h", L.h},
                   {"min_abs_c_exact_cells", min_c_exact},
                   {"source", "spurious_spreading_controls_tests (in memory)"}});
    return L;
}

RunOptions opts(TrackerKind t) {
    RunOptions o;
    o.labels = "(in-memory)";
    o.tracker = t;
    o.seeds = kSeeds;
    o.out = "(in-memory)";
    o.no_timing = true;
    // pre-registered levels of the controls
    o.tol_psi = 1e-10;
    o.ds_ratio = 0.5;
    o.tol = 1e-8;
    o.dt_max = 0.25;
    o.delta_ratio = 1;
    return o;
}

struct Extremes {
    long long n = 0, n_ok = 0;
    double dx2 = 0, dx3 = 0, tau1 = 0, land = 0, dpsi = 0;
    double rms_dx2 = 0, rms_dx3 = 0;
};

Extremes extremes(const ReturnMapRun& r) {
    Extremes e;
    e.n = static_cast<long long>(r.status.size());
    double s2 = 0, s3 = 0;
    for (std::size_t p = 0; p < r.status.size(); ++p) {
        if (r.status[p] != 0)
            continue;
        ++e.n_ok;
        e.dx2 = std::fmax(e.dx2, std::fabs(r.delta_x2[p]));
        e.dx3 = std::fmax(e.dx3, std::fabs(r.delta_x3[p]));
        e.tau1 = std::fmax(e.tau1, std::fabs(r.tau[p] - 1.0));
        e.land = std::fmax(e.land, r.land_err[p]);
        e.dpsi =
            std::fmax(e.dpsi, std::fmax(std::fabs(r.delta_psi1[p]), std::fabs(r.delta_psi2[p])));
        s2 += r.delta_x2[p] * r.delta_x2[p];
        s3 += r.delta_x3[p] * r.delta_x3[p];
    }
    if (e.n_ok > 0) {
        e.rms_dx2 = std::sqrt(s2 / static_cast<double>(e.n_ok));
        e.rms_dx3 = std::sqrt(s3 / static_cast<double>(e.n_ok));
    }
    // A NaN anywhere in an ok row must not hide behind fmax.
    for (std::size_t p = 0; p < r.status.size(); ++p) {
        if (r.status[p] == 0 &&
            (!std::isfinite(r.delta_x2[p]) || !std::isfinite(r.delta_x3[p]) ||
             !std::isfinite(r.tau[p]) || !std::isfinite(r.delta_psi1[p]) ||
             !std::isfinite(r.delta_psi2[p]) || !std::isfinite(r.land_err[p]))) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            e.dx2 = e.dx3 = e.tau1 = e.land = e.dpsi = nan;
        }
    }
    return e;
}

ReturnMapRun run(const CudaContext& ctx, const LoadedLabels& L, const RunOptions& o) {
    ReturnMapRun r = compute_return_map(ctx, L, o);
    if (r.exit_code != 0)
        throw std::runtime_error("compute_return_map refused the labels (exit_code " +
                                 std::to_string(r.exit_code) + ")");
    return r;
}

double num(const JVal& obj, const char* key) {
    const JVal* v = obj.find(key);
    if (v == nullptr || !v->is_number())
        return std::numeric_limits<double>::quiet_NaN();
    return v->as_double();
}

const char* tname(TrackerKind t) {
    return tracker_name(t);
}

// ---------------------------------------------------------------------------
// pair U
// ---------------------------------------------------------------------------

void case_pair_U(const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== pair_U (uniform, N = 16, 256 seeds) ===\n");
    const LoadedLabels L = make_labels(AnalyticPair::U, 0.0, 0.0, kN);
    for (TrackerKind t : {TrackerKind::pseudo_symplectic, TrackerKind::rk, TrackerKind::pollock}) {
        const ReturnMapRun r = run(ctx, L, opts(t));
        const Extremes e = extremes(r);
        const std::string tag = std::string("pair_U/") + tname(t);
        std::printf("  %-18s n_ok %lld/%lld  max|dx2| %.3e  max|dx3| %.3e  max|tau-1| %.3e  "
                    "max land_err %.3e\n",
                    tname(t), e.n_ok, e.n, e.dx2, e.dx3, e.tau1, e.land);
        rep.check(e.n_ok == e.n, tag + "/all_status_0",
                  fmt2("n_ok %.0f of %.0f", static_cast<double>(e.n_ok), static_cast<double>(e.n)));
        rep.check(e.dx2 <= 1e-13 && e.dx3 <= 1e-13, tag + "/delta_x_le_1e-13",
                  fmt2("max|dx2| %.3e  max|dx3| %.3e", e.dx2, e.dx3));
        rep.check(e.tau1 <= 1e-12, tag + "/tau_minus_1_le_1e-12", fmt("max|tau-1| %.3e", e.tau1));
        rep.check(e.land <= 1e-12, tag + "/land_err_le_1e-12", fmt("max %.3e", e.land));
    }
}

// ---------------------------------------------------------------------------
// pair A
// ---------------------------------------------------------------------------

void case_pair_A(const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== pair_A (a = 0.1, N = 16, 256 seeds) ===\n");
    const LoadedLabels L = make_labels(AnalyticPair::A, 0.1, 0.0, kN);
    {
        const ReturnMapRun r = run(ctx, L, opts(TrackerKind::pseudo_symplectic));
        const Extremes e = extremes(r);
        std::printf(
            "  pseudo_symplectic  n_ok %lld  max|dpsi| %.3e  max|dx2| %.3e  max|dx3| %.3e\n",
            e.n_ok, e.dpsi, e.dx2, e.dx3);
        rep.check(e.n_ok == e.n, "pair_A/pseudo_symplectic/all_status_0");
        rep.check(e.dpsi <= 1e-10 + 1e-14, "pair_A/pseudo_symplectic/max_delta_psi_le_tol_psi",
                  fmt("max|delta_psi_i| %.3e", e.dpsi));
        rep.check(e.dx2 <= 1e-6, "pair_A/pseudo_symplectic/delta_x2_le_1e-6",
                  fmt("max|dx2| %.3e", e.dx2));
    }
    {
        const ReturnMapRun r = run(ctx, L, opts(TrackerKind::rk));
        const Extremes e = extremes(r);
        std::printf(
            "  rk                 n_ok %lld  max|dx2| %.3e  max|dx3| %.3e  max|dpsi| %.3e\n",
            e.n_ok, e.dx2, e.dx3, e.dpsi);
        rep.check(e.n_ok == e.n, "pair_A/rk/all_status_0");
        rep.check(e.dx2 <= 1e-6, "pair_A/rk/delta_x2_le_1e-6", fmt("max|dx2| %.3e", e.dx2));
    }
    {
        const ReturnMapRun r = run(ctx, L, opts(TrackerKind::pollock));
        const Extremes e = extremes(r);
        std::printf(
            "  pollock (m = 1)    n_ok %lld  max|dx2| %.3e  max|dx3| %.3e  max|tau-1| %.3e  "
            "max land_err %.3e  div %.3e\n",
            e.n_ok, e.dx2, e.dx3, e.tau1, e.land, num(r.summary, "divergence_max_rel"));
        rep.check(e.n_ok == e.n, "pair_A/pollock/all_status_0");
        rep.check(
            e.dx2 <= 1e-13 && e.dx3 <= 1e-13 && e.tau1 <= 1e-13, "pair_A/pollock/exact_le_1e-13",
            fmt2("max|dx2| %.3e  max|dx3| %.3e", e.dx2, e.dx3) + fmt("  max|tau-1| %.3e", e.tau1));
    }
}

// ---------------------------------------------------------------------------
// pair B
// ---------------------------------------------------------------------------

void case_pair_B(const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== pair_B (a = 0.1, b = 0.08; Pollock m = 1 at N = 16 and N = 32) ===\n");
    double var3[2] = {0, 0};
    const int Ns[2] = {16, 32};
    for (int q = 0; q < 2; ++q) {
        const LoadedLabels L = make_labels(AnalyticPair::B, 0.1, 0.08, Ns[q]);
        const ReturnMapRun r = run(ctx, L, opts(TrackerKind::pollock));
        const Extremes e = extremes(r);
        const JVal& st = *r.summary.find("stats");
        var3[q] = num(*st.find("delta_x3"), "var");
        const double div = num(r.summary, "divergence_max_rel");
        std::printf("  n = %d: n_ok %lld  var(dx2) %.6e  var(dx3) %.6e  rms(dx3) %.6e  "
                    "divergence_max_rel %.3e\n",
                    Ns[q], e.n_ok, num(*st.find("delta_x2"), "var"), var3[q], e.rms_dx3, div);
        const std::string tag = "pair_B/n" + std::to_string(Ns[q]);
        rep.check(e.n_ok == e.n, tag + "/all_status_0");
        rep.check(div <= 1e-13, tag + "/divergence_max_rel_le_1e-13", fmt("%.3e", div));
    }
    rep.check(var3[1] < var3[0], "pair_B/pollock_var_delta_x3_decreases_16_to_32",
              fmt2("var16 %.6e  var32 %.6e", var3[0], var3[1]));
}

// ---------------------------------------------------------------------------
// pair G (+ writer consistency)
// ---------------------------------------------------------------------------

bool rel_eq(double a, double b, double tol) {
    if (std::isnan(a) || std::isnan(b))
        return false;
    if (a == b)
        return true;
    return std::fabs(a - b) <= tol * std::fmax(std::fabs(a), std::fabs(b));
}

/// Recompute the summary statistics from the per-seed arrays (status == 0,
/// population variance, sequential sums in seed order) and compare.
bool check_summary_consistency(const ReturnMapRun& r, std::string& detail) {
    const std::size_t n = r.status.size();
    const double tol = 1e-15;
    bool ok_all = true;
    std::ostringstream bad;
    auto cmp = [&](const std::string& name, double mine, double theirs) {
        if (!rel_eq(mine, theirs, tol)) {
            ok_all = false;
            char buf[200];
            std::snprintf(buf, sizeof(buf), " %s(%.17g vs %.17g)", name.c_str(), mine, theirs);
            bad << buf;
        }
    };
    std::vector<char> ok(n);
    long long m = 0;
    std::map<int, long long> sc;
    for (std::size_t p = 0; p < n; ++p) {
        ok[p] = r.status[p] == 0;
        m += ok[p] ? 1 : 0;
        ++sc[r.status[p]];
    }
    const JVal& S = r.summary;
    cmp("n_seeds", static_cast<double>(n), num(S, "n_seeds"));
    cmp("n_ok", static_cast<double>(m), num(S, "n_ok"));
    const JVal& jsc = *S.find("status_counts");
    for (const auto& kv : sc)
        cmp("status_" + std::to_string(kv.first), static_cast<double>(kv.second),
            num(jsc, std::to_string(kv.first).c_str()));
    if (jsc.keys().size() != sc.size()) {
        ok_all = false;
        bad << " status_counts_size";
    }

    auto mean_of = [&](const std::vector<double>& v) {
        double s = 0;
        for (std::size_t p = 0; p < n; ++p)
            if (ok[p])
                s += v[p];
        return s / static_cast<double>(m);
    };
    auto var_of = [&](const std::vector<double>& v) {
        const double mu = mean_of(v);
        double s = 0;
        for (std::size_t p = 0; p < n; ++p)
            if (ok[p])
                s += (v[p] - mu) * (v[p] - mu);
        return s / static_cast<double>(m);
    };
    const JVal& st = *S.find("stats");
    const std::pair<const char*, const std::vector<double>*> comps[4] = {
        {"delta_x2", &r.delta_x2},
        {"delta_x3", &r.delta_x3},
        {"delta_psi1", &r.delta_psi1},
        {"delta_psi2", &r.delta_psi2}};
    for (const auto& c : comps) {
        const std::vector<double>& v = *c.second;
        double sq = 0, mx = 0;
        for (std::size_t p = 0; p < n; ++p)
            if (ok[p]) {
                sq += v[p] * v[p];
                mx = std::fmax(mx, std::fabs(v[p]));
            }
        const JVal& j = *st.find(c.first);
        cmp(std::string(c.first) + ".mean", mean_of(v), num(j, "mean"));
        cmp(std::string(c.first) + ".var", var_of(v), num(j, "var"));
        cmp(std::string(c.first) + ".rms", std::sqrt(sq / static_cast<double>(m)), num(j, "rms"));
        cmp(std::string(c.first) + ".max_abs", mx, num(j, "max_abs"));
    }
    double tmin = std::numeric_limits<double>::infinity(), tmax = -tmin;
    for (std::size_t p = 0; p < n; ++p)
        if (ok[p]) {
            tmin = std::fmin(tmin, r.tau[p]);
            tmax = std::fmax(tmax, r.tau[p]);
        }
    const JVal& jt = *st.find("tau");
    const double tau_mean = mean_of(r.tau);
    cmp("tau.mean", tau_mean, num(jt, "mean"));
    cmp("tau.var", var_of(r.tau), num(jt, "var"));
    cmp("tau.min", tmin, num(jt, "min"));
    cmp("tau.max", tmax, num(jt, "max"));

    auto wmean = [&](const std::vector<double>& v) {
        double s = 0, ws = 0;
        for (std::size_t p = 0; p < n; ++p)
            if (ok[p]) {
                s += r.c1[p] * v[p];
                ws += r.c1[p];
            }
        return s / ws;
    };
    auto wvar = [&](const std::vector<double>& v) {
        const double mu = wmean(v);
        double s = 0, ws = 0;
        for (std::size_t p = 0; p < n; ++p)
            if (ok[p]) {
                s += r.c1[p] * (v[p] - mu) * (v[p] - mu);
                ws += r.c1[p];
            }
        return s / ws;
    };
    const double tau_f = wmean(r.tau);
    cmp("flux_weighted_tau_mean", tau_f, num(S, "flux_weighted_tau_mean"));
    cmp("D22_uniform", var_of(r.delta_x2) / (2.0 * tau_mean), num(S, "D22_uniform"));
    cmp("D33_uniform", var_of(r.delta_x3) / (2.0 * tau_mean), num(S, "D33_uniform"));
    cmp("D22_flux", wvar(r.delta_x2) / (2.0 * tau_f), num(S, "D22_flux"));
    cmp("D33_flux", wvar(r.delta_x3) / (2.0 * tau_f), num(S, "D33_flux"));
    double ml = 0;
    int mi = 0;
    for (std::size_t p = 0; p < n; ++p)
        if (ok[p]) {
            ml = std::fmax(ml, r.land_err[p]);
            mi = std::max(mi, r.land_iters[p]);
        }
    const JVal& jl = *S.find("landing");
    cmp("landing.max_err", ml, num(jl, "max_err"));
    cmp("landing.max_iterations", static_cast<double>(mi), num(jl, "max_iterations"));
    detail = ok_all ? "all fields equal to 1e-15 relative" : bad.str();
    return ok_all;
}

void case_pair_G(const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== pair_G (e = 0.03, N = 16, 256 seeds) ===\n");
    const LoadedLabels L = make_labels(AnalyticPair::G, 0.03, 0.0, kN);
    for (TrackerKind t : {TrackerKind::pseudo_symplectic, TrackerKind::rk, TrackerKind::pollock}) {
        const ReturnMapRun r = run(ctx, L, opts(t));
        const Extremes e = extremes(r);
        const std::string tag = std::string("pair_G/") + tname(t);
        std::printf("  %-18s n_ok %lld/%lld  rms(dx2) %.6e  rms(dx3) %.6e  max|dpsi| %.3e  "
                    "mean tau %.15f\n",
                    tname(t), e.n_ok, e.n, e.rms_dx2, e.rms_dx3, e.dpsi,
                    num(*r.summary.find("stats")->find("tau"), "mean"));
        if (t == TrackerKind::pollock)
            std::printf("  %-18s divergence_max_rel %.3e  max land_err %.3e\n", "",
                        num(r.summary, "divergence_max_rel"), e.land);
        rep.check(e.n_ok == e.n, tag + "/all_status_0");
        if (t == TrackerKind::pseudo_symplectic)
            rep.check(e.dpsi <= 1e-10 + 1e-14, tag + "/max_delta_psi_le_tol_psi",
                      fmt("max|delta_psi_i| %.3e", e.dpsi));
        std::string detail;
        const bool cons = check_summary_consistency(r, detail);
        rep.check(cons, tag + "/summary_equals_recomputed_1e-15", detail);
    }
}

// ---------------------------------------------------------------------------
// byte reproducibility
// ---------------------------------------------------------------------------

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void case_byte_reproducibility(const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== byte_reproducibility (pair G, N = 16, no_timing) ===\n");
    const LoadedLabels L = make_labels(AnalyticPair::G, 0.03, 0.0, kN);
    namespace fs = std::filesystem;
    const fs::path dir = fs::current_path() / "sf32_n2b_controls_tmp";
    fs::remove_all(dir);
    for (TrackerKind t : {TrackerKind::pseudo_symplectic, TrackerKind::rk, TrackerKind::pollock}) {
        const std::string tag = std::string("byte_reproducibility/") + tname(t);
        RunOptions o = opts(t);
        o.out = (dir / tname(t)).string();
        const ReturnMapRun r1 = run(ctx, L, o);
        write_run_dir(o.out, r1);
        const std::string c1 = read_file(o.out + "/seeds.csv");
        const std::string s1 = read_file(o.out + "/summary.json");
        const ReturnMapRun r2 = run(ctx, L, o);
        write_run_dir(o.out, r2);
        const std::string c2 = read_file(o.out + "/seeds.csv");
        const std::string s2 = read_file(o.out + "/summary.json");
        rep.check(seeds_csv_text(r1) == seeds_csv_text(r2) &&
                      summary_json_text(r1) == summary_json_text(r2),
                  tag + "/in_memory_texts_identical");
        rep.check(!c1.empty() && !s1.empty() && c1 == c2 && s1 == s2 && c1 == seeds_csv_text(r1) &&
                      s1 == summary_json_text(r1),
                  tag + "/files_identical",
                  fmt2("seeds.csv %.0f B, summary.json %.0f B", static_cast<double>(c1.size()),
                       static_cast<double>(s1.size())));
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    rep.check(!ec && !fs::exists(dir), "byte_reproducibility/temporary_directory_removed");
}

void case_delta_ratio_must_divide(const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== delta_ratio_must_divide_N ===\n");
    const LoadedLabels L = make_labels(AnalyticPair::U, 0.0, 0.0, kN);
    RunOptions o = opts(TrackerKind::pollock);
    o.delta_ratio = 3;
    bool threw = false;
    try {
        (void)compute_return_map(ctx, L, o);
    } catch (const UsageError& e) {
        threw = true;
        std::printf("  UsageError: %s\n", e.what());
    }
    rep.check(threw, "delta_ratio_must_divide_N/m3_on_N16_is_usage_error");
}

} // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    TestReport rep;
    auto guarded = [&](const char* name, auto&& fn) {
        try {
            fn();
        } catch (const std::exception& e) {
            rep.check(false, std::string(name) + "/unexpected_exception", e.what());
        }
    };
    try {
        CudaContext ctx(0);
        guarded("pair_U", [&] { case_pair_U(ctx, rep); });
        guarded("pair_A", [&] { case_pair_A(ctx, rep); });
        guarded("pair_B", [&] { case_pair_B(ctx, rep); });
        guarded("pair_G", [&] { case_pair_G(ctx, rep); });
        guarded("byte_reproducibility", [&] { case_byte_reproducibility(ctx, rep); });
        guarded("delta_ratio_must_divide_N", [&] { case_delta_ratio_must_divide(ctx, rep); });
    } catch (const std::exception& e) {
        rep.check(false, "setup/unexpected_exception", e.what());
    }
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n%d checks, %d failed, wall %.2f s\n", rep.checks, rep.failed, wall);
    std::printf("%s\n", rep.overall_pass ? "ALL PASSED" : "SOME CHECKS FAILED");
    return rep.overall_pass ? 0 : 1;
}
