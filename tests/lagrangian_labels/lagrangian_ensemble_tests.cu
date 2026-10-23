/**
 * @file lagrangian_ensemble_tests.cu
 * @brief SF-38 N3: ensemble runner (contract C-RUNNER, RR3, RR7 mechanics, RR11, RR12), fast tier.
 *
 * Box 64x16x16, ell 0.25, eps 1, M 10 (window [0.1, 0.9]^2: no seed on a transverse face for
 * n_T = 16, verified below), 2 realizations, auto planes/times, twin and Pollock bridge on.
 *
 *  1. auto rules (RR3) at the tiny box and at 2048x256x256; the seed-face predicate (RR12) at
 *     n_T = 16/256/384.
 *  2. in-process run A: stats CSVs read back by read_timeseries_csv reproduce the in-process moments
 *     (flux-weighted and twin) to <= 1e-12 relative; twin weights are exactly 1/N_p; twin moments
 *     equal an independent host recomputation from the saved per-particle arrays; the in-process
 *     compute_macrodispersion agrees with analysis/macrodispersion.csv.
 *  3. determinism: a second run of the same command in a fresh directory gives a byte-identical tree
 *     except timings.json (and analysis_config.yaml up to the directory name).
 *  4. crash/restart: --test-abort-after (hidden) exits 99 at every phase (tmp, rename at r = 0;
 *     partial, tmp, rename at r = 1) of the same directory in turn; rerunning the normal command gives a
 *     tree byte-identical to the uninterrupted run; a restart under different options is refused (exit 7).
 *  5. `cd DIR && macroflow3d_pipeline analysis_config.yaml` (analysis_only, output_dir ".") accepts the
 *     written config on a copy of the directory and reproduces analysis/macrodispersion.csv and
 *     ensemble/ensemble_timeseries.csv; analysis_config.yaml is byte-identical across directories.
 *  5b. particles with non-ok tracker statuses (--max-arclength 1) are recorded in status.json and the
 *      STATUS line; the realization is written with DONE and the exit code is 0.
 *  6. seed-face assertion fires through the executable (exit 3, names the seed); --field-only writes
 *     field.json and no flow outputs; usage errors exit 2 naming the option.
 *
 * Usage: lagrangian_ensemble_tests <lagrangian_ensemble exe> <macroflow3d_pipeline exe>
 */

#include "src/analysis/macrodispersion/MacrodispersionAnalysis.hpp"
#include "src/external/nlohmann/json.hpp"
#include "src/physics/particles/lagrangian_labels/EnsembleRunner.cuh"
#include "src/physics/streamfunctions/inlet_slab/NpyIo.hpp"
#include "src/runtime/CudaContext.cuh"

#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace ll = macroflow3d::lagrangian_labels;
namespace sl = macroflow3d::streamfunctions::inlet_slab;
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
    char buf[1024];
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
    if (std::isnan(a) || std::isnan(b))
        return std::isnan(a) && std::isnan(b);
    return std::fabs(a - b) <= rtol * std::fabs(b) || std::fabs(a - b) <= 1e-300;
}

std::string g_ens, g_pipe;
fs::path g_base;

/// Runs `exe args > log 2>&1`; returns the exit status (-1 on abnormal termination).
int run_cmd(const std::string& exe, const std::string& args, const fs::path& log) {
    const std::string cmd = exe + " " + args + " > " + log.string() + " 2>&1";
    const int st = std::system(cmd.c_str());
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

const std::string kCommon = "--grid 64,16,16 --ell 0.25 --eps 1 --seed0 4001 --nr 2 --np-side 10 "
                            "--pollock-bridge on";

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

std::map<std::string, std::string> tree_files(const fs::path& root) {
    std::map<std::string, std::string> m;
    for (const auto& e : fs::recursive_directory_iterator(root))
        if (e.is_regular_file())
            m[fs::relative(e.path(), root).string()] = e.path().string();
    return m;
}

/// Byte-compares two output trees: same file set; every file equal except timings.json (must exist)
/// (analysis_config.yaml is compared byte for byte: output_dir is ".").
std::string compare_trees(const fs::path& a, const fs::path& b) {
    const auto fa = tree_files(a), fb = tree_files(b);
    for (const auto& kv : fa)
        if (!fb.count(kv.first))
            return "only in first: " + kv.first;
    for (const auto& kv : fb)
        if (!fa.count(kv.first))
            return "only in second: " + kv.first;
    for (const auto& kv : fa) {
        const std::string& rel = kv.first;
        if (fs::path(rel).filename() == "timings.json")
            continue;
        const std::string x = slurp(kv.second), y = slurp(fb.at(rel));
        if (x != y)
            return "differs: " + rel;
    }
    return "";
}

/// Numeric CSV compare (header lines must match, rows to rtol).
bool csv_close(const fs::path& a, const fs::path& b, double rtol, std::string* why) {
    std::ifstream fa(a), fb(b);
    std::string la, lb;
    int row = 0;
    while (true) {
        const bool ga = static_cast<bool>(std::getline(fa, la)), gb = static_cast<bool>(std::getline(fb, lb));
        if (ga != gb) {
            *why = "row count differs";
            return false;
        }
        if (!ga)
            return true;
        ++row;
        if (!la.empty() && (la[0] == '#' || std::isalpha(static_cast<unsigned char>(la[0])))) {
            if (la != lb) {
                *why = "header differs";
                return false;
            }
            continue;
        }
        const char *pa = la.c_str(), *pb = lb.c_str();
        for (int col = 0; *pa && *pb; ++col) {
            char *ea = nullptr, *eb = nullptr;
            const double x = std::strtod(pa, &ea), y = std::strtod(pb, &eb);
            if (!rel_close(x, y, rtol)) {
                *why = strf("row %d col %d: %.17g vs %.17g", row, col, x, y);
                return false;
            }
            pa = *ea == ',' ? ea + 1 : ea;
            pb = *eb == ',' ? eb + 1 : eb;
        }
    }
}

bool series_close(const std::vector<io::TimeSeriesPoint<double>>& a,
                  const std::vector<io::TimeSeriesPoint<double>>& b, double rtol, std::string* why) {
    if (a.size() != b.size()) {
        *why = "size differs";
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        bool ok = rel_close(a[i].time, b[i].time, rtol) && a[i].active == b[i].active;
        for (int k = 0; k < 3; ++k)
            ok = ok && rel_close(a[i].mean[k], b[i].mean[k], rtol) &&
                 rel_close(a[i].var[k], b[i].var[k], rtol);
        if (!ok) {
            *why = strf("row %zu differs", i);
            return false;
        }
    }
    return true;
}

/// Uniform-weight moments of x1,x2,x3 over the particles with a finite record at each time
/// (independent host recomputation from times_x.npy: Np x T x 3, NaN where not reached).
std::vector<io::TimeSeriesPoint<double>> twin_from_npy(const fs::path& npy,
                                                       const std::vector<double>& times) {
    const sl::NpyArray a = sl::read_npy(npy.string());
    const std::size_t Np = a.shape[0], T = a.shape[1];
    std::vector<io::TimeSeriesPoint<double>> s(T);
    for (std::size_t n = 0; n < T; ++n) {
        s[n].time = times[n];
        std::size_t cnt = 0;
        for (int k = 0; k < 3; ++k) {
            double sum = 0.0;
            cnt = 0;
            for (std::size_t p = 0; p < Np; ++p) {
                const double x = a.data[(p * T + n) * 3 + static_cast<std::size_t>(k)];
                if (std::isnan(x))
                    continue;
                sum += x;
                ++cnt;
            }
            const double m = cnt ? sum / static_cast<double>(cnt) : std::nan("");
            double v2 = 0.0;
            for (std::size_t p = 0; p < Np; ++p) {
                const double x = a.data[(p * T + n) * 3 + static_cast<std::size_t>(k)];
                if (!std::isnan(x))
                    v2 += (x - m) * (x - m);
            }
            s[n].mean[k] = m;
            s[n].var[k] = cnt ? v2 / static_cast<double>(cnt) : std::nan("");
        }
        s[n].active = static_cast<int>(cnt);
    }
    return s;
}

void test_unit() {
    // auto rules at the tiny box: L1 = 4, lambda = 1/4 (powers of two: x0 + L1 coincides with m = 16)
    const std::vector<double> pl = ll::auto_planes(0.0, 4.0, 0.25);
    check("auto_planes_tiny", pl == std::vector<double>({0.25, 0.5, 1.0, 1.5, 2.0, 4.0}),
          strf("size %zu", pl.size()));
    const std::vector<double> tm = ll::auto_times(4.0, 0.25);
    check("auto_times_tiny", tm.size() == 32 && tm.front() == 0.125 && tm.back() == 4.0,
          strf("size %zu", tm.size()));
    // B1 geometry: L1 = 8, lambda = 10/256 -> m = 1..128 (8 planes) + L1 - 10 lambda + L1; 410 times
    const std::vector<double> pb = ll::auto_planes(0.0, 2048.0 / 256.0, 10.0 / 256.0);
    check("auto_planes_B1", pb.size() == 10 && pb.back() == 8.0 && pb[8] == 8.0 - 10.0 * (10.0 / 256.0),
          strf("size %zu", pb.size()));
    const std::vector<double> tb = ll::auto_times(2048.0 / 256.0, 10.0 / 256.0);
    check("auto_times_B1", tb.size() == 410, strf("size %zu", tb.size()));
    const std::vector<double> t3 = ll::auto_times(3328.0 / 384.0, 32.0 / 384.0);
    check("auto_times_B3", t3.size() == 208, strf("size %zu", t3.size())); // 2 * 104

    // RR12 arithmetic: n_T = 16, window [0.1,0.9]: 16 x = 1.6 + 1.28 (m + 1/2) (M = 10) never within 0.08 of an integer
    ll::InjectionSpec inj;
    inj.M = 10;
    inj.w2a = inj.w3a = 0.1;
    inj.w2b = inj.w3b = 0.9;
    std::string what;
    check("seedface_none_nT16_M10", !ll::find_seed_on_face(inj, 16, &what), what);
    inj.M = 100;
    check("seedface_none_nT256_M100", !ll::find_seed_on_face(inj, 256, &what), what);
    check("seedface_none_nT384_M100", !ll::find_seed_on_face(inj, 384, &what), what);
    inj.M = 8; // 16 (0.1 + 0.8 (1 + 1/2)/8 ) = 16 * 0.25 = 4 exactly: on a face
    check("seedface_hit_nT16_M8", ll::find_seed_on_face(inj, 16, &what) && what.find("seed p=") == 0, what);
    inj.w2a = inj.w3a = 0.0;
    inj.w2b = inj.w3b = 1.0; // [0,1), M = 8: 16 x = 2m + 1
    check("seedface_hit_window01", ll::find_seed_on_face(inj, 16, &what), what);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <lagrangian_ensemble> <macroflow3d_pipeline>\n", argv[0]);
        return 2;
    }
    g_ens = argv[1];
    g_pipe = argv[2];
    g_base = fs::temp_directory_path() / strf("lagrangian_ensemble_test_%d", static_cast<int>(::getpid()));
    fs::remove_all(g_base);
    fs::create_directories(g_base);

    test_unit();

    // ---- run A (in-process) -------------------------------------------------------------------
    const fs::path A = g_base / "A";
    ll::EnsembleRunSummary sum;
    int rcA = -1;
    {
        CudaContext ctx(0);
        const ll::EnsembleConfig cfg = parse_cfg(kCommon + " --out " + A.string());
        check("resolved_planes_tiny", cfg.planes == ll::auto_planes(0.0, 4.0, 0.25) && cfg.planes_auto);
        rcA = ll::run_ensemble(ctx, cfg, &sum);
    }
    check("runA_exit0", rcA == 0, strf("rc=%d", rcA));
    if (rcA != 0) {
        std::printf("SUMMARY: %d failed\n", g_failed);
        return 1;
    }
    const std::vector<double> times_auto = ll::auto_times(4.0, 0.25);
    for (int r = 0; r < 2; ++r) {
        const std::string rn = strf("r%d", r);
        const fs::path rd = A / strf("realizations/r_%04d", r);
        check(rn + "_done_marker", fs::exists(rd / "DONE") && fs::exists(rd / "timings.json") &&
                                       fs::exists(rd / "field.json") && fs::exists(rd / "twin/moments_time.csv") &&
                                       fs::exists(rd / "pollock/pollock_twin_moments_time.csv") &&
                                       fs::exists(rd / "track.json") && !fs::exists(rd.string() + ".tmp"));
        const json tj = json::parse(slurp(rd / "track.json"));
        check(rn + "_track_json_no_timings", !tj.contains("timings_s"));
        // CSV round trip (D): read_timeseries_csv reproduces the in-process moments, <= 1e-12 relative
        std::string why;
        const auto flux = analysis::read_timeseries_csv<double>((A / strf("stats/realization_%04d_timeseries.csv", r)).string());
        const auto twin = analysis::read_timeseries_csv<double>((A / strf("stats_twin/realization_%04d_timeseries.csv", r)).string());
        check(rn + "_csv_roundtrip_flux", sum.computed[static_cast<std::size_t>(r)] &&
                                              series_close(flux, sum.flux_series[static_cast<std::size_t>(r)], 1e-12, &why) &&
                                              flux.size() == times_auto.size(),
              why);
        check(rn + "_csv_roundtrip_twin", series_close(twin, sum.twin_series[static_cast<std::size_t>(r)], 1e-12, &why), why);
        // twin weights uniform
        const std::vector<double>& w = sum.twin_weights[static_cast<std::size_t>(r)];
        bool uniform = w.size() == 100;
        for (double x : w)
            uniform = uniform && x == 1.0 / 100.0;
        check(rn + "_twin_weights_uniform", uniform);
        // independent recomputation of the twin from the saved per-particle arrays
        check(rn + "_twin_independent_host", series_close(twin_from_npy(rd / "times_x.npy", times_auto), twin, 1e-12, &why), why);
        // the flux-weighted weights are not uniform (a twin that equals the flux series would be vacuous)
        const sl::NpyArray wf = sl::read_npy((rd / "weights.npy").string());
        double wmax = 0.0, wmin = 1.0;
        for (double x : wf.data) {
            wmax = std::max(wmax, x);
            wmin = std::min(wmin, x);
        }
        check(rn + "_flux_weights_nonuniform", wmax > 1.01 * wmin, strf("min %.3e max %.3e", wmin, wmax));
        // Pollock bridge twin (host recomputation)
        const auto ptwin = analysis::read_timeseries_csv<double>((rd / "pollock/pollock_twin_moments_time.csv").string());
        check(rn + "_pollock_twin_independent_host",
              series_close(twin_from_npy(rd / "pollock/pollock_times_x.npy", times_auto), ptwin, 1e-12, &why), why);
    }
    {   // in-process analysis agrees with analysis/macrodispersion.csv (same format as the pipeline)
        const auto rows = analysis::compute_macrodispersion<double>(sum.flux_series, 0.25, 1.0);
        const fs::path tmp = g_base / "inproc_macrodispersion.csv";
        analysis::write_macrodispersion_csv(tmp.string(), rows);
        std::string why;
        check("macrodispersion_inprocess_vs_file", csv_close(tmp, A / "analysis/macrodispersion.csv", 1e-12, &why) && !rows.empty(), why);
    }
    check("analysis_outputs_exist", fs::exists(A / "ensemble/ensemble_timeseries.csv") && fs::exists(A / "analysis_config.yaml"));

    // ---- determinism: second run in a fresh directory -----------------------------------------
    const fs::path B = g_base / "B";
    int rc = run_cmd(g_ens, kCommon + " --out " + B.string(), g_base / "B.log");
    check("runB_exit0", rc == 0, strf("rc=%d", rc));
    std::string d = compare_trees(A, B);
    check("determinism_A_vs_B_bytes", d.empty(), d);

    // ---- crash and restart --------------------------------------------------------------------
    // One directory crashed repeatedly at every phase of both realizations, then restarted normally.
    {
        const fs::path C = g_base / "C";
        struct Step { const char* abort; const char* state; };
        const Step steps[] = {
            {"0,tmp", "stats/realization_0000_timeseries.csv.tmp"},   // stats .tmp written, nothing renamed
            {"0,rename", "realizations/r_0000"},                        // directory renamed, DONE missing
            {"1,partial", "realizations/r_0001.tmp"},                   // some files of r1 written
            {"1,tmp", "stats/realization_0001_timeseries.csv.tmp"},
            {"1,rename", "realizations/r_0001"}};
        for (const Step& st : steps) {
            const std::string tag = std::string("crash_") + st.abort;
            rc = run_cmd(g_ens, kCommon + " --out " + C.string() + " --test-abort-after " + st.abort,
                         g_base / (tag + ".log"));
            check(tag + "_exit99", rc == ll::kExitTestAbort, strf("rc=%d", rc));
            const bool no_done = std::string(st.state).find("realizations/r_") != 0 ||
                                 std::string(st.state).find(".tmp") != std::string::npos ||
                                 !fs::exists(C / st.state / "DONE");
            check(tag + "_leaves_partial_state", fs::exists(C / st.state) && no_done, st.state);
            check(tag + "_no_final_analysis", !fs::exists(C / "analysis/macrodispersion.csv"));
        }
        check("restart_skips_done_realization",
              slurp(g_base / "crash_1,tmp.log").find("R=0 SKIP") != std::string::npos);
        rc = run_cmd(g_ens, kCommon + " --out " + C.string(), g_base / "rerun.log");
        check("restart_exit0", rc == 0, strf("rc=%d", rc));
        d = compare_trees(A, C);
        check("restart_bytes_equal_uninterrupted", d.empty(), d);
        bool no_tmp = true;
        for (const auto& kv : tree_files(C))
            no_tmp = no_tmp && fs::path(kv.first).extension() != ".tmp";
        check("restart_no_tmp_left", no_tmp);
    }
    // restart under different options is refused
    rc = run_cmd(g_ens, "--grid 64,16,16 --ell 0.25 --eps 1.1 --seed0 4001 --nr 2 --np-side 10 --pollock-bridge on --out " + A.string(),
                 g_base / "mismatch.log");
    check("restart_different_options_exit7", rc == ll::kExitConfigMismatch, strf("rc=%d", rc));

    // ---- macroflow3d_pipeline accepts analysis_config.yaml (cd DIR && pipeline analysis_config.yaml) -----
    {
        const fs::path P = g_base / "P";
        fs::copy(A, P, fs::copy_options::recursive);
        check("analysis_config_relative_output_dir",
              slurp(P / "analysis_config.yaml").find("output_dir: \".\"") != std::string::npos &&
                  slurp(P / "analysis_config.yaml").find("cd DIR && macroflow3d_pipeline analysis_config.yaml") != std::string::npos);
        fs::remove(P / "analysis/macrodispersion.csv");
        fs::remove(P / "ensemble/ensemble_timeseries.csv");
        rc = run_cmd("cd " + P.string() + " && " + g_pipe, "analysis_config.yaml", g_base / "pipeline.log");
        check("pipeline_analysis_only_exit0", rc == 0, strf("rc=%d", rc));
        std::string why;
        check("pipeline_macrodispersion_reproduced",
              csv_close(P / "analysis/macrodispersion.csv", A / "analysis/macrodispersion.csv", 1e-12, &why), why);
        check("pipeline_ensemble_series_reproduced",
              csv_close(P / "ensemble/ensemble_timeseries.csv", A / "ensemble/ensemble_timeseries.csv", 1e-12, &why), why);
        check("pipeline_macrodispersion_bytes_equal",
              slurp(P / "analysis/macrodispersion.csv") == slurp(A / "analysis/macrodispersion.csv"));
    }

    // ---- non-ok particles are recorded, never an exit condition -------------------------------------
    {
        const fs::path N = g_base / "N";
        // an arclength cap of 1 box unit stops particles before the far planes / late times
        rc = run_cmd(g_ens, "--grid 64,16,16 --ell 0.25 --eps 1 --seed0 4001 --nr 1 --np-side 10 --pollock-bridge on --max-arclength 1.0 --out " + N.string(),
                     g_base / "nonok.log");
        check("nonok_exit0", rc == 0, strf("rc=%d", rc));
        const fs::path rd = N / "realizations/r_0000";
        check("nonok_realization_written", fs::exists(rd / "DONE") && fs::exists(rd / "status.json") &&
                                               fs::exists(N / "stats/realization_0000_timeseries.csv") &&
                                               fs::exists(N / "analysis/macrodispersion.csv"));
        const json sj = json::parse(slurp(rd / "status.json"));
        const long long ge2p = sj["tracker_plane_pass"]["status_ge2"].get<long long>();
        const long long ge2t = sj["tracker_time_pass"]["status_ge2"].get<long long>();
        long long hist_ge2 = 0;
        for (int k = 2; k < 7; ++k)
            hist_ge2 += sj["tracker_plane_pass"]["status_hist"][static_cast<std::size_t>(k)].get<long long>();
        check("nonok_counts_in_status_json", ge2p > 0 && ge2t > 0 && hist_ge2 == ge2p && sj.contains("pollock") &&
                                                 sj["pollock"]["weighted_not_active"].get<long long>() == 0,
              strf("plane>=2 %lld time>=2 %lld", ge2p, ge2t));
        const std::string log = slurp(g_base / "nonok.log");
        check("nonok_status_line_reports_counts",
              log.find(strf("STATUS ok tracker_non_ok_plane=%lld tracker_non_ok_time=%lld", ge2p, ge2t)) != std::string::npos);
        check("nonok_dropped_particles_reduce_active", [&] {
            const auto s2 = analysis::read_timeseries_csv<double>((N / "stats/realization_0000_timeseries.csv").string());
            return !s2.empty() && s2.back().active < 100;
        }());
        check("nonok_no_failure_exit_in_log", log.find("TRACK_FAILED") == std::string::npos);
    }

    // ---- seed-face assertion, field-only, usage ------------------------------------------------
    {
        rc = run_cmd(g_ens, "--grid 64,16,16 --ell 0.25 --eps 1 --seed0 4001 --np-side 8 --window 0,1,0,1 --out " + (g_base / "SF").string(),
                     g_base / "seedface.log");
        const std::string log = slurp(g_base / "seedface.log");
        check("seedface_exit3_names_seed", rc == ll::kExitSeedFace && log.find("seed p=") != std::string::npos, strf("rc=%d", rc));
        check("seedface_nothing_tracked", !fs::exists(g_base / "SF/realizations/r_0000"));
        rc = run_cmd(g_ens, "--grid 64,16,16 --ell 0.25 --seed0 4001 --nr 2 --field-only --out " + (g_base / "FO").string(),
                     g_base / "fieldonly.log");
        check("field_only_exit0", rc == 0, strf("rc=%d", rc));
        bool ok = true;
        for (int r = 0; r < 2; ++r) {
            const fs::path rd = g_base / strf("FO/realizations/r_%04d", r);
            ok = ok && fs::exists(rd / "field.json") && fs::exists(rd / "DONE") && fs::exists(rd / "timings.json") &&
                 !fs::exists(rd / "track.json") && !fs::exists(rd / "seeds.npy");
            const json fj = json::parse(slurp(rd / "field.json"));
            ok = ok && fj["seed"] == 4001 + r && fj["raw_variance"].get<double>() > 0.0 &&
                 fj["applied_scale"].get<double>() > 0.0;
        }
        check("field_only_writes_field_json_only", ok && !fs::exists(g_base / "FO/stats") && !fs::exists(g_base / "FO/analysis"));
        const std::string flog = slurp(g_base / "fieldonly.log");
        check("field_only_prints_FIELD_line", flog.find("R=0 FIELD gaussian (SF-18)") != std::string::npos &&
                                                  flog.find("raw_variance=") != std::string::npos && flog.find("STAGE ") == std::string::npos);
        rc = run_cmd(g_ens, "--grid 64,16,16 --ell 0.25 --seed0 4001 --eps 1 --field-only --out " + (g_base / "FO2").string(), g_base / "fo2.log");
        check("field_only_contradiction_exit2", rc == ll::kExitUsage && slurp(g_base / "fo2.log").find("--eps") != std::string::npos, strf("rc=%d", rc));
        rc = run_cmd(g_ens, kCommon + " --bogus 3 --out " + (g_base / "U1").string(), g_base / "u1.log");
        check("usage_unknown_option_exit2", rc == ll::kExitUsage && slurp(g_base / "u1.log").find("--bogus") != std::string::npos, strf("rc=%d", rc));
        rc = run_cmd(g_ens, "--grid 64,16,8 --ell 0.25 --eps 1 --seed0 4001 --out " + (g_base / "U2").string(), g_base / "u2.log");
        check("usage_bad_grid_exit2", rc == ll::kExitUsage && slurp(g_base / "u2.log").find("--grid") != std::string::npos, strf("rc=%d", rc));
    }

    if (g_failed == 0)
        fs::remove_all(g_base);
    else
        std::printf("kept %s\n", g_base.string().c_str());
    std::printf("SUMMARY: %d failed\n", g_failed);
    std::printf(g_failed == 0 ? "ALL_PASS\n" : "FAILED %d\n", g_failed);
    return g_failed == 0 ? 0 : 1;
}
