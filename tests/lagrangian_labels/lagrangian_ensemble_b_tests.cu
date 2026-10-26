/**
 * @file lagrangian_ensemble_b_tests.cu
 * @brief SF-41 N5: `lagrangian_ensemble --medium b` (contract C-RUNNER-B), fast tier.
 *
 * Box 128x16x16 (h = 1/16, L1 = 8), ell 0.5 (lambda/h = 8, L1 = 16 lambda), eps 1, M 10, seed 4001.
 *
 *  1. unit: auto_planes_b / auto_times_b; compute_flow_stats_b on synthetic faces (uniform and tilted flow with
 *     exact values, a pseudo-random field against an independent sort-based recomputation).
 *  2. run A: nr 2, --hmax-div 8,16, twin and Pollock bridge on: exit 0; file set (per-pass sub-directories,
 *     per-pass series roots, flow_stats.json, solver_b.json, exits files, pollock exits files); planes and times
 *     resolved as specified; flow_stats.json qbar equals the solver record's; flow_stats.json equals an
 *     independent in-process stage build + recomputation (<= 1e-13 relative); exit totals consistent between
 *     track.json, exits csv and status.json; Pollock seeds equal the tracker's.
 *  3. determinism: a second run in a fresh directory is byte-identical except timings.json.
 *  4. crash/restart: --test-abort-after at two phases of the same directory, then the normal command: the tree is
 *     byte-identical to the uninterrupted run; a restart under different options is refused (exit 7).
 *  5. single pass (b) with --fused-splines --single-launch --head-mapped-host, twin and bridge off, one
 *     realization: the layout is the single-pass layout and its flux-weighted series is byte-identical to the first
 *     pass of run A (fused / mapped-host variants are bitwise by design).
 *  6. medium (a) with a multi-pass list: the first pass is byte-identical to the single-pass (a) run.
 *  7. usage errors: --medium c, --head-mapped-host with medium a, duplicate --hmax-div, a (b) plane >= L1.
 *
 * Usage: lagrangian_ensemble_b_tests <lagrangian_ensemble exe>
 */

#include "src/external/nlohmann/json.hpp"
#include "src/physics/particles/lagrangian_labels/DirichletLongBoxStage.cuh"
#include "src/physics/particles/lagrangian_labels/EnsembleRunner.cuh"
#include "src/runtime/CudaContext.cuh"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace ll = macroflow3d::lagrangian_labels;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int g_failed = 0;

void check(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("CHECK %s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ",
                detail.c_str());
    std::fflush(stdout);
    if (!ok)
        ++g_failed;
}

std::string strf(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char* f, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool rel_close(double a, double b, double rtol) {
    return std::fabs(a - b) <= rtol * std::max(std::fabs(a), std::fabs(b)) || std::fabs(a - b) <= 1e-300;
}

std::string g_ens;
fs::path g_base;

int run_cmd(const std::string& args, const fs::path& log) {
    const std::string cmd = g_ens + " " + args + " > " + log.string() + " 2>&1";
    const int st = std::system(cmd.c_str());
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

std::map<std::string, std::string> tree_files(const fs::path& root) {
    std::map<std::string, std::string> m;
    for (const auto& e : fs::recursive_directory_iterator(root))
        if (e.is_regular_file())
            m[fs::relative(e.path(), root).string()] = e.path().string();
    return m;
}

/// Same file set; every file equal except timings.json.
std::string compare_trees(const fs::path& a, const fs::path& b) {
    const auto fa = tree_files(a), fb = tree_files(b);
    for (const auto& kv : fa)
        if (!fb.count(kv.first))
            return "only in first: " + kv.first;
    for (const auto& kv : fb)
        if (!fa.count(kv.first))
            return "only in second: " + kv.first;
    for (const auto& kv : fa) {
        if (fs::path(kv.first).filename() == "timings.json")
            continue;
        if (slurp(kv.second) != slurp(fb.at(kv.first)))
            return "differs: " + kv.first;
    }
    return "";
}

std::vector<std::string> split_args(const std::string& s) {
    std::vector<std::string> v;
    std::istringstream is(s);
    for (std::string t; is >> t;)
        v.push_back(t);
    return v;
}

ll::EnsembleConfig parse_cfg(const std::string& args) {
    const std::vector<std::string> v = split_args("lagrangian_ensemble " + args);
    std::vector<const char*> argv;
    for (const std::string& s : v)
        argv.push_back(s.c_str());
    return ll::parse_ensemble_args(static_cast<int>(argv.size()), argv.data());
}

// ---- unit ------------------------------------------------------------------------------------------

struct Faces {
    int nx, ny, nz;
    std::vector<real> U, V, W;
    Faces(int nx_, int ny_, int nz_)
        : nx(nx_), ny(ny_), nz(nz_), U(std::size_t(nx_ + 1) * ny_ * nz_), V(std::size_t(nx_) * (ny_ + 1) * nz_),
          W(std::size_t(nx_) * ny_ * (nz_ + 1)) {}
    real& u(int i, int j, int k) { return U[std::size_t(i) + std::size_t(nx + 1) * (j + std::size_t(ny) * k)]; }
    real& v(int i, int j, int k) { return V[std::size_t(i) + std::size_t(nx) * j + std::size_t(nx) * (ny + 1) * k]; }
    real& w(int i, int j, int k) { return W[std::size_t(i) + std::size_t(nx) * j + std::size_t(nx) * ny * k]; }
};

/// Independent recomputation: full sort + numpy-style linear interpolation; interior sums with the plane index
/// as the outer loop. Returns false and describes the first mismatch.
bool flow_stats_match(const ll::FlowStatsB& S, Faces& F, double lambda, double qbar, double rtol, std::string* why) {
    const int nx = F.nx, ny = F.ny, nz = F.nz;
    const double h = 1.0 / ny, L1 = nx * h;
    std::vector<double> sp;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double u = 0.5 * (F.u(i, j, k) + F.u(i + 1, j, k));
                const double v = 0.5 * (F.v(i, j, k) + F.v(i, j + 1, k));
                const double w = 0.5 * (F.w(i, j, k) + F.w(i, j, k + 1));
                sp.push_back(std::sqrt(u * u + v * v + w * w) / qbar);
            }
    std::sort(sp.begin(), sp.end());
    const double pc[4] = {0.0, 0.1, 1.0, 50.0};
    for (int q = 0; q < 4; ++q) {
        const double pos = pc[q] / 100.0 * double(sp.size() - 1);
        const std::size_t lo = std::size_t(std::floor(pos));
        const double fr = pos - double(lo);
        const double val = lo + 1 < sp.size() ? sp[lo] + (sp[lo + 1] - sp[lo]) * fr : sp[lo];
        if (!rel_close(S.speed_pct[q], val, rtol)) {
            *why = strf("percentile %g: %.17g vs %.17g", pc[q], S.speed_pct[q], val);
            return false;
        }
    }
    long long nonpos = 0;
    for (real u : F.U)
        nonpos += (u <= 0.0);
    if (nonpos != S.x_faces_nonpositive || S.x_faces != (long long)F.U.size()) {
        *why = "nonpositive count";
        return false;
    }
    double su = 0, sv = 0, sw = 0;
    long long cu = 0, cl = 0;
    for (int i = 0; i <= nx; ++i)
        if (i * h >= 5 * lambda - 1e-9 * h && i * h <= L1 - 5 * lambda + 1e-9 * h) {
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    su += F.u(i, j, k);
            cu += (long long)ny * nz;
        }
    for (int i = 0; i < nx; ++i)
        if ((i + 0.5) * h >= 5 * lambda - 1e-9 * h && (i + 0.5) * h <= L1 - 5 * lambda + 1e-9 * h) {
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j) {
                    sv += F.v(i, j, k);
                    sw += F.w(i, j, k);
                }
            cl += (long long)ny * nz;
        }
    const double mu = su / cu, mv = sv / cl, mw = sw / cl;
    if (!rel_close(S.mean_u, mu, rtol) || !rel_close(S.mean_v, mv, rtol) || !rel_close(S.mean_w, mw, rtol) ||
        !rel_close(S.theta_2, std::atan(mv / mu), rtol) || !rel_close(S.theta_3, std::atan(mw / mu), rtol)) {
        *why = strf("interior means: %.17g %.17g %.17g vs %.17g %.17g %.17g", S.mean_u, S.mean_v, S.mean_w, mu, mv, mw);
        return false;
    }
    return true;
}

void test_unit() {
    const std::vector<double> pl = ll::auto_planes_b(2.5, 8.0, 0.5);
    check("auto_planes_b_L16lambda", pl == std::vector<double>({3.0, 3.5, 4.5, 5.5}), strf("size %zu", pl.size()));
    // production-like: L1 = 8, lambda = 10/256, x0 = 5 lambda: m = 1..128 then the (merged) L1 - 5 lambda plane
    const double lam = 10.0 / 256.0;
    const std::vector<double> pe = ll::auto_planes_b(5.0 * lam, 8.0, lam);
    check("auto_planes_b_E1_geometry", pe.size() == 9 && std::fabs(pe.back() - (8.0 - 5.0 * lam)) < 1e-12 && std::fabs(pe[7] - (5 * lam + 128 * lam)) < 1e-12,
          strf("size %zu", pe.size()));
    const std::vector<double> t1 = ll::auto_times_b(8.0, 0.5, 1.0);
    check("auto_times_b_qbar1", t1.size() == 12 && t1.front() == 0.25 && t1.back() == 3.0, strf("size %zu", t1.size()));
    check("auto_times_b_qbar2", ll::auto_times_b(8.0, 0.5, 2.0).size() == 6);
    check("auto_times_b_at_least_two", ll::auto_times_b(8.0, 0.5, 100.0).size() == 2);

    // flow statistics: uniform / tilted flow (exact values), nx = 32, ny = nz = 8 (h = 1/8, L1 = 4, lambda = 1/8)
    {
        Faces F(32, 8, 8);
        std::fill(F.U.begin(), F.U.end(), 2.0);
        std::fill(F.V.begin(), F.V.end(), 1.0);
        std::fill(F.W.begin(), F.W.end(), -0.5);
        const ll::FlowStatsB S = ll::compute_flow_stats_b(F.U, F.V, F.W, 32, 8, 8, 0.125, 2.0);
        const double sp = std::sqrt(4.0 + 1.0 + 0.25) / 2.0;
        check("flow_stats_uniform_tilt",
              S.x_faces_nonpositive == 0 && S.x_nonpositive_fraction == 0.0 && S.speed_pct[0] == sp &&
                  S.speed_pct[3] == sp && S.mean_u == 2.0 && S.mean_v == 1.0 && S.mean_w == -0.5 &&
                  S.theta_2 == std::atan(0.5) && S.theta_3 == std::atan(-0.25) && S.plane_lo == 5 && S.plane_hi == 27 &&
                  S.layer_lo == 5 && S.layer_hi == 26,
              strf("planes %d..%d layers %d..%d", S.plane_lo, S.plane_hi, S.layer_lo, S.layer_hi));
        // 5 lambda = 0.625 = 5 h: planes 5..27 (27 h = 3.375 = L1 - 5 lambda), layers (i + 1/2) h in [0.625, 3.375]: 5..26
        // non-positive u: half of plane 0 set to 0 and plane nx set to -1
        for (int k = 0; k < 8; ++k)
            for (int j = 0; j < 8; ++j) {
                if (j < 4)
                    F.u(0, j, k) = 0.0;
                F.u(32, j, k) = -1.0;
            }
        const ll::FlowStatsB S2 = ll::compute_flow_stats_b(F.U, F.V, F.W, 32, 8, 8, 0.125, 2.0);
        check("flow_stats_nonpositive_fraction",
              S2.x_faces_nonpositive == 32 + 64 && S2.x_faces == 33 * 64 &&
                  S2.x_nonpositive_fraction == 96.0 / (33.0 * 64.0));
    }
    {   // pseudo-random positive-biased field vs the independent recomputation
        Faces F(48, 8, 8);
        unsigned long long s = 12345;
        auto rnd = [&] {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            return double(s >> 11) / double(1ULL << 53);
        };
        for (real& x : F.U)
            x = 1.7 * rnd() - 0.2; // about 12 % of the x-faces are non-positive
        for (real& x : F.V)
            x = 0.4 * (rnd() - 0.5);
        for (real& x : F.W)
            x = 0.4 * (rnd() - 0.5);
        const double qbar = 1.0 + 0.1 * rnd();
        const ll::FlowStatsB S = ll::compute_flow_stats_b(F.U, F.V, F.W, 48, 8, 8, 0.2, qbar);
        std::string why;
        check("flow_stats_random_vs_sort", flow_stats_match(S, F, 0.2, qbar, 1e-13, &why) && S.x_faces_nonpositive > 0, why);
        const std::string js = ll::flow_stats_b_json(S);
        const json j = json::parse(js);
        check("flow_stats_json_roundtrip",
              j.at("qbar").get<double>() == qbar && j.at("interior").at("theta_2").get<double>() == S.theta_2 &&
                  j.at("speed_over_qbar_percentiles").at("p0.1").get<double>() == S.speed_pct[1] &&
                  j.at("x_faces_nonpositive").get<long long>() == S.x_faces_nonpositive);
    }
    bool threw = false;
    try {
        Faces F(16, 8, 8);
        (void)ll::compute_flow_stats_b(F.U, F.V, F.W, 16, 8, 8, 0.5, 1.0); // L1 = 2 < 10 lambda: empty interior
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check("flow_stats_empty_interior_throws", threw);
}

std::vector<double> csv_col(const fs::path& p, int col) {
    std::vector<double> v;
    std::ifstream f(p);
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string tok;
        for (int c = 0; c <= col && std::getline(ss, tok, ','); ++c)
            if (c == col)
                v.push_back(std::strtod(tok.c_str(), nullptr));
    }
    return v;
}

const std::string kB = "--medium b --grid 128,16,16 --ell 0.5 --eps 1 --seed0 4001 --np-side 10 ";

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <lagrangian_ensemble>\n", argv[0]);
        return 2;
    }
    g_ens = argv[1];
    g_base = fs::temp_directory_path() / strf("lagrangian_ensemble_b_test_%d", static_cast<int>(::getpid()));
    fs::remove_all(g_base);
    fs::create_directories(g_base);

    test_unit();

    // ---- run A: nr 2, two passes, twin + bridge ---------------------------------------------------------
    const std::string argsA = kB + "--nr 2 --hmax-div 8,16 --pollock-bridge on";
    const fs::path A = g_base / "A";
    int rc = run_cmd(argsA + " --out " + A.string(), g_base / "A.log");
    check("runA_exit0", rc == 0, strf("rc=%d", rc));
    if (rc != 0) {
        std::printf("kept %s\nSUMMARY: %d failed\n", g_base.string().c_str(), g_failed);
        return 1;
    }
    const std::string logA = slurp(g_base / "A.log");
    {
        const ll::EnsembleConfig cfg = parse_cfg(argsA + " --out " + A.string());
        check("resolved_defaults_b", cfg.medium_b && cfg.x0 == 2.5 && cfg.times_auto && cfg.times.empty() &&
                                         cfg.planes == std::vector<double>({3.0, 3.5, 4.5, 5.5}) &&
                                         cfg.hmax_list == std::vector<int>({8, 16}));
        check("resolved_line_b", logA.find("RESOLVED grid=128x16x16 L1=8 lambda=0.5 planes(auto)=3,3.5,4.5,5.5 "
                                           "times(auto)=n*lambda/2 up to (L1-10 lambda)/qbar") != std::string::npos &&
                                     logA.find("medium=b x0=2.5 hmax_div=8,16") != std::string::npos);
    }
    for (int r = 0; r < 2; ++r) {
        const std::string rn = strf("r%d", r);
        const fs::path rd = A / strf("realizations/r_%04d", r);
        bool files = fs::exists(rd / "DONE") && fs::exists(rd / "timings.json") && fs::exists(rd / "field.json") &&
                     fs::exists(rd / "status.json") && fs::exists(rd / "flow_stats.json") &&
                     fs::exists(rd / "solver_b.json") && fs::exists(rd / "pollock/pollock_exits_planes.csv") &&
                     fs::exists(rd / "pollock/pollock_exits_time.csv") &&
                     fs::exists(rd / "pollock/pollock_twin_moments_time.csv") && !fs::exists(rd.string() + ".tmp") &&
                     !fs::exists(rd / "track.json");
        for (int d : {8, 16}) {
            const fs::path pd = rd / strf("hmax_%d", d);
            files = files && fs::exists(pd / "track.json") && fs::exists(pd / "exits_planes.csv") &&
                    fs::exists(pd / "exits_time.csv") && fs::exists(pd / "twin/moments_time.csv") &&
                    fs::exists(pd / "status.json") &&
                    fs::exists(A / strf("hmax_%d/stats/realization_%04d_timeseries.csv", d, r)) &&
                    fs::exists(A / strf("hmax_%d/stats_twin/realization_%04d_timeseries.csv", d, r));
        }
        check(rn + "_file_set", files);
        const json fsj = json::parse(slurp(rd / "flow_stats.json"));
        const json sj = json::parse(slurp(rd / "solver_b.json"));
        const double qbar = fsj.at("qbar").get<double>();
        check(rn + "_qbar_equals_solver_record", qbar == sj.at("qbar").get<double>() && qbar > 0.0 &&
                                                     sj.at("usable").get<bool>() && sj.at("converged").get<bool>(),
              strf("qbar %.17g", qbar));
        const auto& pc = fsj.at("speed_over_qbar_percentiles");
        const double p0 = pc.at("p0").get<double>(), p01 = pc.at("p0.1").get<double>(),
                     p1 = pc.at("p1").get<double>(), p50 = pc.at("p50").get<double>();
        const double fr = fsj.at("x_nonpositive_fraction").get<double>();
        check(rn + "_flow_stats_self_consistent",
              0.0 <= p0 && p0 <= p01 && p01 <= p1 && p1 <= p50 && p50 > 0.3 && p50 < 3.0 && fr >= 0.0 && fr <= 1.0 &&
                  fr == double(fsj.at("x_faces_nonpositive").get<long long>()) / double(fsj.at("x_faces").get<long long>()) &&
                  std::fabs(fsj.at("interior").at("theta_2").get<double>()) < 0.5 &&
                  std::fabs(fsj.at("interior").at("theta_3").get<double>()) < 0.5,
              strf("p0 %.3e p50 %.3e frac %.3e", p0, p50, fr));
        check(rn + "_solver_line_printed", logA.find(strf("R=%d B_SOLVER r=%d converged=1", r, r)) != std::string::npos &&
                                               logA.find(strf("R=%d RESOLVED_TIMES qbar=", r)) != std::string::npos &&
                                               logA.find(strf("R=%d FLOWSTATS qbar=", r)) != std::string::npos);
        // planes / times of the tracker as resolved
        const std::vector<double> times = ll::auto_times_b(8.0, 0.5, qbar);
        const json t8 = json::parse(slurp(rd / "hmax_8/track.json"));
        const json t16 = json::parse(slurp(rd / "hmax_16/track.json"));
        check(rn + "_track_json_planes_times",
              t8.at("planes").get<std::vector<double>>() == std::vector<double>({3.0, 3.5, 4.5, 5.5}) &&
                  t8.at("times").get<std::vector<double>>() == times && t16.at("times").get<std::vector<double>>() == times &&
                  t8.at("nonperiodic_x").get<bool>() && !t8.contains("timings_s") &&
                  t8.at("domain_L1").get<double>() == 8.0,
              strf("n_times %zu", times.size()));
        // exits: totals of track.json == last row of exits csv == status.json (per pass)
        for (int d : {8, 16}) {
            const fs::path pd = rd / strf("hmax_%d", d);
            const json tj = json::parse(slurp(pd / "track.json"));
            const json st = json::parse(slurp(pd / "status.json"));
            const std::vector<double> ei = csv_col(pd / "exits_planes.csv", 1), eo = csv_col(pd / "exits_planes.csv", 2);
            const std::vector<double> ti = csv_col(pd / "exits_time.csv", 1), to = csv_col(pd / "exits_time.csv", 2);
            const bool ok = !ei.empty() && ei.back() == tj.at("plane_pass").at("exit_inlet").get<double>() &&
                            eo.back() == tj.at("plane_pass").at("exit_outlet").get<double>() &&
                            ti.back() == tj.at("time_pass").at("exit_inlet").get<double>() &&
                            to.back() == tj.at("time_pass").at("exit_outlet").get<double>() &&
                            st.at("tracker_plane_pass").at("exit_outlet").get<double>() == eo.back() &&
                            st.at("tracker_time_pass").at("exit_outlet").get<double>() == to.back() &&
                            st.at("hmax_div").get<int>() == d && ti.size() == times.size();
            check(rn + strf("_hmax%d_exit_totals_consistent", d), ok);
        }
        const json st = json::parse(slurp(rd / "status.json"));
        const json pj = json::parse(slurp(rd / "pollock/pollock.json"));
        const std::vector<double> pei = csv_col(rd / "pollock/pollock_exits_planes.csv", 1);
        check(rn + "_pollock_exit_files", pj.at("seeds_equal_tracker").get<bool>() && pj.at("nonperiodic_x").get<bool>() &&
                                              !pei.empty() &&
                                              pei.back() == st.at("pollock").at("exit_inlet_plane").get<double>() &&
                                              st.at("hmax_div") == json::array({8, 16}));
    }
    check("analysis_roots_per_pass", fs::exists(A / "hmax_8/analysis/macrodispersion.csv") &&
                                         fs::exists(A / "hmax_16/ensemble/ensemble_timeseries.csv") &&
                                         fs::exists(A / "hmax_8/analysis_config.yaml") && !fs::exists(A / "stats") &&
                                         !fs::exists(A / "analysis_config.yaml"));
    {
        const std::string cfg = slurp(A / "hmax_8/analysis_config.yaml");
        double vm = 0.0;
        for (int r = 0; r < 2; ++r)
            vm += json::parse(slurp(A / strf("realizations/r_%04d/flow_stats.json", r))).at("qbar").get<double>();
        vm /= 2.0;
        check("analysis_vmean_norm_is_mean_qbar", cfg.find("vmean_norm: " + strf("%.17g", vm)) != std::string::npos);
    }

    // independent in-process check of flow_stats.json: same stage, independent recomputation
    {
        CudaContext ctx(0);
        ll::DirichletLongBoxSpec ds;
        ds.nx = 128;
        ds.ny = ds.nz = 16;
        ds.source = ll::LongBoxSourceKind::gaussian;
        ds.sigma2 = 1.0;
        ds.ell = 0.5;
        ds.seed = 4001;
        ds.normalize_variance = true;
        ds.eps = 1.0;
        ds.G1 = 1.0;
        ds.cg_rtol = 1e-10;
        ds.cg_max_iter = 2000;
        ds.keep_host_faces = true;
        ds.build_splines = false;
        ll::DirichletLongBoxStage st = ll::build_dirichlet_long_box_stage(ctx, ds);
        Faces F(128, 16, 16);
        F.U = st.face_u;
        F.V = st.face_v;
        F.W = st.face_w;
        const json fsj = json::parse(slurp(A / "realizations/r_0000/flow_stats.json"));
        const double qbar = st.solver.qbar;
        ll::FlowStatsB S = ll::compute_flow_stats_b(F.U, F.V, F.W, 128, 16, 16, 0.5, qbar);
        std::string why;
        check("flow_stats_json_vs_independent_stage",
              qbar == fsj.at("qbar").get<double>() && flow_stats_match(S, F, 0.5, qbar, 1e-13, &why) &&
                  rel_close(fsj.at("interior").at("mean_u").get<double>(), S.mean_u, 1e-13) &&
                  rel_close(fsj.at("speed_over_qbar_percentiles").at("p50").get<double>(), S.speed_pct[3], 1e-13) &&
                  fsj.at("x_faces_nonpositive").get<long long>() == S.x_faces_nonpositive,
              why);
    }

    // ---- determinism ------------------------------------------------------------------------------------
    const fs::path B = g_base / "B";
    rc = run_cmd(argsA + " --out " + B.string(), g_base / "B.log");
    check("runB_exit0", rc == 0, strf("rc=%d", rc));
    std::string d = compare_trees(A, B);
    check("determinism_A_vs_B_bytes", d.empty(), d);

    // ---- crash and restart ------------------------------------------------------------------------------
    {
        const fs::path C = g_base / "C";
        for (const char* ab : {"0,tmp", "1,partial"}) {
            rc = run_cmd(argsA + " --out " + C.string() + " --test-abort-after " + ab, g_base / (std::string("crash_") + ab + ".log"));
            check(std::string("crash_") + ab + "_exit99", rc == ll::kExitTestAbort, strf("rc=%d", rc));
            if (std::string(ab) == "0,tmp") // the per-pass series temporaries are left behind and cleaned on restart
                check("crash_tmp_in_pass_roots", fs::exists(C / "hmax_8/stats/realization_0000_timeseries.csv.tmp") &&
                                                     fs::exists(C / "hmax_16/stats_twin/realization_0000_timeseries.csv.tmp"));
        }
        // r = 0 completed (DONE) and the process died inside r = 1 (some files written)
        check("crash_leaves_partial_state", fs::exists(C / "realizations/r_0000/DONE") &&
                                                fs::exists(C / "realizations/r_0001.tmp") &&
                                                !fs::exists(C / "realizations/r_0001"));
        rc = run_cmd(argsA + " --out " + C.string(), g_base / "rerun.log");
        check("restart_exit0", rc == 0, strf("rc=%d", rc));
        d = compare_trees(A, C);
        check("restart_bytes_equal_uninterrupted", d.empty(), d);
        rc = run_cmd(kB + "--nr 2 --hmax-div 8,32 --pollock-bridge on --out " + A.string(), g_base / "mismatch.log");
        check("restart_different_options_exit7", rc == ll::kExitConfigMismatch, strf("rc=%d", rc));
    }

    // ---- single pass with the placement options (fused, single launch, mapped-host head) ---------------------
    {
        const fs::path S = g_base / "S";
        rc = run_cmd(kB + "--nr 1 --fused-splines --single-launch --head-mapped-host --twin off --out " + S.string(),
                     g_base / "S.log");
        check("single_pass_exit0", rc == 0, strf("rc=%d", rc));
        const fs::path rd = S / "realizations/r_0000";
        check("single_pass_layout", fs::exists(rd / "track.json") && fs::exists(rd / "exits_planes.csv") &&
                                        fs::exists(rd / "flow_stats.json") && fs::exists(rd / "solver_b.json") &&
                                        !fs::exists(rd / "hmax_8") && !fs::exists(rd / "pollock") &&
                                        fs::exists(S / "stats/realization_0000_timeseries.csv") &&
                                        !fs::exists(S / "stats_twin") && fs::exists(S / "analysis/macrodispersion.csv"));
        check("single_pass_series_equals_run_A_pass8_bytes (fused + single launch + mapped-host head are bitwise)",
              slurp(S / "stats/realization_0000_timeseries.csv") ==
                      slurp(A / "hmax_8/stats/realization_0000_timeseries.csv") &&
                  slurp(rd / "flow_stats.json") == slurp(A / "realizations/r_0000/flow_stats.json") &&
                  slurp(rd / "exits_planes.csv") == slurp(A / "realizations/r_0000/hmax_8/exits_planes.csv") &&
                  slurp(rd / "weights.npy") == slurp(A / "realizations/r_0000/hmax_8/weights.npy"));
        const json sj = json::parse(slurp(rd / "status.json"));
        check("single_pass_status_json_has_exits", sj.at("tracker_plane_pass").contains("exit_inlet") &&
                                                       sj.at("tracker_time_pass").contains("exit_outlet") &&
                                                       !sj.contains("pollock") && !sj.contains("hmax_div"));
        check("single_pass_stdout_status_line",
              slurp(g_base / "S.log").find("STATUS ok tracker_non_ok_plane=") != std::string::npos &&
                  slurp(g_base / "S.log").find("exit_inlet_plane=") != std::string::npos);
    }

    // ---- explicit lists beyond the travel time: exits through x1 = L1 are counted per time -------------------
    {
        const fs::path E = g_base / "E";
        rc = run_cmd(kB + "--nr 1 --planes 3,5,7,7.9 --times 1,3,6,9,12 --pollock-bridge on --twin off --out " + E.string(),
                     g_base / "E.log");
        check("exits_run_exit0", rc == 0, strf("rc=%d", rc));
        const fs::path rd = E / "realizations/r_0000";
        const json tj = json::parse(slurp(rd / "track.json"));
        const json pj = json::parse(slurp(rd / "pollock/pollock.json"));
        const std::vector<double> eo = csv_col(rd / "exits_time.csv", 2), ei = csv_col(rd / "exits_time.csv", 1);
        const std::vector<double> po = csv_col(rd / "pollock/pollock_exits_time.csv", 2);
        bool mono = eo.size() == 5 && po.size() == 5;
        for (std::size_t i = 1; i < eo.size() && mono; ++i)
            mono = eo[i] >= eo[i - 1] && po[i] >= po[i - 1];
        check("exits_outlet_counted_per_time",
              mono && eo.back() > 0 && po.back() > 0 && eo.front() == 0 && ei.back() == 0 &&
                  eo.back() == tj.at("time_pass").at("exit_outlet").get<double>() &&
                  po.back() == pj.at("exit_outlet_time_total").get<double>() &&
                  tj.at("plane_pass").at("exit_outlet").get<double>() == 0.0 &&
                  tj.at("planes").get<std::vector<double>>() == std::vector<double>({3.0, 5.0, 7.0, 7.9}),
              strf("tracker exits %g, pollock exits %g", eo.empty() ? -1.0 : eo.back(), po.empty() ? -1.0 : po.back()));
        const json sj = json::parse(slurp(rd / "status.json"));
        check("exits_status_json_and_line",
              sj.at("tracker_time_pass").at("exit_outlet").get<double>() == eo.back() &&
                  slurp(g_base / "E.log").find(strf("exit_outlet_time=%lld", static_cast<long long>(eo.back()))) != std::string::npos &&
                  slurp(g_base / "E.log").find("times(list)=1,3,6,9,12") != std::string::npos);
    }

    // ---- medium (a): multi-pass first pass equals the single-pass run -----------------------------------------
    {
        const std::string a = "--grid 64,16,16 --ell 0.25 --eps 1 --seed0 4001 --nr 1 --np-side 10 --twin off ";
        const fs::path A1 = g_base / "A1", A2 = g_base / "A2";
        const int r1 = run_cmd(a + "--out " + A1.string(), g_base / "A1.log");
        const int r2 = run_cmd(a + "--hmax-div 8,16 --out " + A2.string(), g_base / "A2.log");
        check("medium_a_runs_exit0", r1 == 0 && r2 == 0, strf("rc=%d,%d", r1, r2));
        check("medium_a_multi_first_pass_equals_single",
              slurp(A1 / "stats/realization_0000_timeseries.csv") ==
                      slurp(A2 / "hmax_8/stats/realization_0000_timeseries.csv") &&
                  slurp(A1 / "realizations/r_0000/track.json") ==
                      slurp(A2 / "realizations/r_0000/hmax_8/track.json") &&
                  !fs::exists(A1 / "realizations/r_0000/flow_stats.json") &&
                  !fs::exists(A2 / "realizations/r_0000/flow_stats.json") &&
                  !fs::exists(A1 / "realizations/r_0000/hmax_8") &&
                  slurp(A1 / "realizations/r_0000/DONE").find("hmax_div=8 ") != std::string::npos &&
                  slurp(A2 / "realizations/r_0000/DONE").find("hmax_div=8,16 ") != std::string::npos);
    }

    // ---- usage -------------------------------------------------------------------------------------------
    {
        const std::string g = "--grid 128,16,16 --ell 0.5 --eps 1 --seed0 4001 --np-side 10 ";
        struct U { const char* tag; std::string args; const char* needle; };
        const U us[] = {
            {"medium_c", g + "--medium c --out " + (g_base / "U1").string(), "--medium"},
            {"mapped_host_needs_b", g + "--head-mapped-host --out " + (g_base / "U2").string(), "--head-mapped-host"},
            {"hmax_duplicate", g + "--medium b --hmax-div 8,8 --out " + (g_base / "U3").string(), "--hmax-div"},
            {"plane_beyond_L1", g + "--medium b --planes 3,8 --out " + (g_base / "U4").string(), "--planes"},
            {"x0_beyond_L1", g + "--medium b --x0 9 --out " + (g_base / "U5").string(), "--x0"}};
        for (const U& u : us) {
            const std::string log = (g_base / (std::string(u.tag) + ".log")).string();
            const int rcu = run_cmd(u.args, log);
            check(std::string("usage_") + u.tag, rcu == ll::kExitUsage && slurp(log).find(u.needle) != std::string::npos,
                  strf("rc=%d", rcu));
        }
    }

    if (g_failed == 0)
        fs::remove_all(g_base);
    else
        std::printf("kept %s\n", g_base.string().c_str());
    std::printf("SUMMARY: %d failed\n", g_failed);
    std::printf(g_failed == 0 ? "ALL_PASS\n" : "FAILED %d\n", g_failed);
    return g_failed == 0 ? 0 : 1;
}
