/**
 * @file lagrangian_medium_ab_tests.cu
 * @brief SF-38 N4: medium A/B application `lagrangian_medium_ab` (contract C-AB; RR4, RR5, RR12), fast tier.
 *
 * Box 128x16x16, ell 0.25 (L1 = 8, lambda/h = 4, planes m = 1,2,4,8,16 at X_m = 1.25 + 0.25 m <= 6.75),
 * eps 0.5, M = 10 (window [0.1, 0.9]^2: no seed on a transverse face for n_T = 16), NR = 2, seed0 4001.
 *
 *  1. run A: exit 0; the output tree (per-realization files, ab_manifest.json, ab_primary.csv) exists; stdout
 *     carries the B_SOLVER, MEM_PHASE_B, E1, E2, PRIMARY lines of both media.
 *  2. primary statistics: primary.json equals an independent recomputation from e2_*.csv (S_T, S_tau) and from
 *     the e1_*.csv profiles (qbar, V1, theta_2, theta_3), per medium, to 1e-12 relative; e1.json agrees with
 *     the same recomputation; ab_primary.csv rows (order, seeds, %.17g values) match primary.json.
 *  3. determinism: a second run in a fresh directory is byte-identical except timings.json.
 *  4. crash/restart: --test-abort-after exits 99 at partial/tmp/rename of r = 1 in one directory in turn;
 *     rerunning the normal command gives a tree byte-identical to run A; a restart under other options exits 7.
 *  5. k = 1 control (--eps 0): media (a) and (b) give the same E1 and E2 statistics within solver tolerance
 *     (qbar, V1, theta ~ 0, S_T, S_tau ~ 0).
 *  6. C0 fallback: --cg-rtol-b 1e-12 capped at the iteration count n9 of a 1e-9 run (rel9 in (1e-12, 1e-8]) stops
 *     unconverged at the same iterate: rel <= 1e-8, balance <= 1e-8 -> fallback=1, exit 0, record in
 *     solver_b.json; --cg-max-iter-b 1 -> exit 8, failure_r_0000.json holds the
 *     record and no DONE is written.
 *  7. exits: usage errors -> 2 (naming the option; too few planes), seed-face assertion -> 3.
 *  8. SF-41 N6 (medium (b) Pollock non-periodic in x): on the exit-free run A, e2_b.csv / status.json of the new
 *     path equal byte for byte those of the pre-SF-41 periodic view (--test-b-periodic-view 1), exits_b.csv
 *     reports zeros. A non-zero exit count is covered at lattice level by ctest lagrangian_pollock_nonperiodic;
 *     no field reaching an inlet/outlet exit was found through this application (random Y, eps <= 2 solvable).
 *
 * Usage: lagrangian_medium_ab_tests <lagrangian_medium_ab exe>
 */

#include "src/external/nlohmann/json.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <array>
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

bool rel_close(double a, double b, double rtol, double atol = 1e-300) {
    if (std::isnan(a) || std::isnan(b))
        return std::isnan(a) && std::isnan(b);
    return std::fabs(a - b) <= rtol * std::fabs(b) || std::fabs(a - b) <= atol;
}

std::string g_exe;
fs::path g_base;

int run_cmd(const std::string& args, const fs::path& log) {
    const std::string cmd = g_exe + " " + args + " > " + log.string() + " 2>&1";
    const int st = std::system(cmd.c_str());
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

const std::string kBox = "--grid 128,16,16 --ell 0.25 --seed0 4001 --np-side 10";
const std::string kCommon = kBox + " --eps 0.5 --nr 2";

std::map<std::string, std::string> tree_files(const fs::path& root) {
    std::map<std::string, std::string> m;
    for (const auto& e : fs::recursive_directory_iterator(root))
        if (e.is_regular_file())
            m[fs::relative(e.path(), root).string()] = e.path().string();
    return m;
}

/// Same file set; every file byte-equal except timings.json. "" if equal, else the first difference.
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

std::vector<std::vector<std::string>> read_csv(const fs::path& p, std::vector<std::string>* header = nullptr) {
    std::ifstream f(p);
    std::vector<std::vector<std::string>> rows;
    std::string ln;
    bool first = true;
    while (std::getline(f, ln)) {
        std::vector<std::string> cells;
        std::size_t a = 0;
        while (true) {
            const std::size_t b = ln.find(',', a);
            cells.push_back(ln.substr(a, b == std::string::npos ? b : b - a));
            if (b == std::string::npos)
                break;
            a = b + 1;
        }
        if (first && header)
            *header = cells;
        else if (!first)
            rows.push_back(cells);
        first = false;
    }
    return rows;
}

double num(const std::string& s) { return std::strtod(s.c_str(), nullptr); }

struct Stats {
    double qbar = 0, V1 = 0, th2 = 0, th3 = 0, S_T = 0, S_tau = 0;
};

/// Independent recomputation of one medium's statistics from e1_<m>.csv and e2_<m>.csv (RR5): no shared code
/// with the application.
Stats recompute(const fs::path& rdir, const char* m, int nx, int ny, double lambda, int* n_planes = nullptr) {
    Stats S;
    const double h = 1.0 / ny, L1 = static_cast<double>(nx) / ny;
    const auto e1 = read_csv(rdir / strf("e1_%s.csv", m));
    double su = 0, sv = 0, sw = 0;
    int np = 0, nl = 0;
    for (const auto& row : e1) {
        const int i = static_cast<int>(num(row[1]));
        if (row[0] == "plane") {
            if (i == 0)
                S.qbar = num(row[3]);
            const double x = i * h;
            if (x >= 5 * lambda - 1e-9 * h && x <= L1 - 5 * lambda + 1e-9 * h) {
                su += num(row[4]);
                ++np;
            }
        } else {
            const double x = (i + 0.5) * h;
            if (x >= 5 * lambda - 1e-9 * h && x <= L1 - 5 * lambda + 1e-9 * h) {
                sv += num(row[5]);
                sw += num(row[7]);
                ++nl;
            }
        }
    }
    S.V1 = su / np / (S.qbar * S.qbar);
    S.th2 = sv / nl / S.qbar;
    S.th3 = sw / nl / S.qbar;
    const auto e2 = read_csv(rdir / strf("e2_%s.csv", m));
    const std::size_t P = e2.size();
    if (n_planes)
        *n_planes = static_cast<int>(P);
    const double mstar = std::pow(2.0, static_cast<double>(P - 1)); // planes m = 1, 2, 4, ...
    const double dx = (mstar / 2.0) * lambda;
    auto sigT = [&](const std::vector<std::string>& r) { return (num(r[4]) + num(r[6])) / 2.0; };
    S.S_T = (sigT(e2[P - 1]) - sigT(e2[P - 2])) / dx;
    S.S_tau = S.qbar * S.qbar * (num(e2[P - 1][8]) - num(e2[P - 2][8])) / dx;
    return S;
}

bool stats_match(const json& p, const Stats& S, double rtol, std::string* why) {
    const std::pair<const char*, double> kv[] = {{"qbar", S.qbar}, {"V1", S.V1},     {"theta_2", S.th2},
                                                 {"theta_3", S.th3}, {"S_T", S.S_T}, {"S_tau", S.S_tau}};
    for (const auto& e : kv) {
        const double a = p[e.first].is_null() ? NAN : p[e.first].get<double>();
        if (!rel_close(a, e.second, rtol, 1e-14)) {
            *why = strf("%s: file %.17g vs recomputed %.17g", e.first, a, e.second);
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <lagrangian_medium_ab>\n", argv[0]);
        return 2;
    }
    g_exe = argv[1];
    g_base = fs::temp_directory_path() / strf("lagrangian_medium_ab_test_%d", static_cast<int>(::getpid()));
    fs::remove_all(g_base);
    fs::create_directories(g_base);

    // ---- 1. run A ------------------------------------------------------------------------------
    const fs::path A = g_base / "A";
    const int rcA = run_cmd(kCommon + " --out " + A.string(), g_base / "A.log");
    check("runA_exit0", rcA == 0, strf("rc=%d", rcA));
    if (rcA != 0) {
        std::printf("--- A.log ---\n%s\nSUMMARY: %d failed\n", slurp(g_base / "A.log").c_str(), g_failed);
        return 1;
    }
    const std::string log = slurp(g_base / "A.log");
    for (const char* key : {"R=0 B_SOLVER r=0 converged=1", "R=1 B_SOLVER r=1 converged=1", "fallback=0",
                            "MEM_PHASE_B phase=solve", "MEM_PHASE_B phase=faces", "R=0 E1 medium=a",
                            "R=0 E1 medium=b", "R=1 E2 medium=a planes=5 m_star=16", "R=1 E2 medium=b planes=5",
                            "R=0 PRIMARY medium=a", "R=1 PRIMARY medium=b", "R=0 FIELD", "R=0 SETUP", "R=0 STAGE",
                            "R=0 MEM_GEOM", "R=0 MEM_PHASE phase=S2", "TIMING r=0", "RUN_DONE nr=2"})
        check(std::string("stdout_has_") + key, log.find(key) != std::string::npos);
    check("tree_top", fs::exists(A / "ab_manifest.json") && fs::exists(A / "ab_primary.csv"));
    for (int r = 0; r < 2; ++r) {
        const fs::path rd = A / strf("realizations/r_%04d", r);
        bool all = !fs::exists(rd.string() + ".tmp");
        for (const char* f : {"e1_a.csv", "e1_b.csv", "e1.json", "e2_a.csv", "e2_b.csv", "primary.json",
                              "solver_b.json", "status.json", "field.json", "timings.json", "DONE",
                              "exits_b.csv"})
            all = all && fs::exists(rd / f);
        check(strf("r%d_files", r), all);
    }
    {
        const json man = json::parse(slurp(A / "ab_manifest.json"));
        check("manifest_content", man["grid"] == json::array({128, 16, 16}) && man["seeds"].size() == 2 &&
                                      man["seeds"][1] == 4002 && man["eps"] == 0.5 &&
                                      man["command_line_without_out"].get<std::string>().find("--out") ==
                                          std::string::npos);
    }

    // ---- 2. primary statistics: independent recomputation ---------------------------------------
    const auto abp = read_csv(A / "ab_primary.csv");
    std::vector<std::string> hdr;
    read_csv(A / "ab_primary.csv", &hdr);
    check("ab_primary_header", hdr == std::vector<std::string>({"r", "seed", "medium", "S_T", "S_tau", "V1",
                                                                  "theta_2", "theta_3", "qbar"}));
    check("ab_primary_rows", abp.size() == 4);
    for (int r = 0; r < 2; ++r) {
        const fs::path rd = A / strf("realizations/r_%04d", r);
        const json pj = json::parse(slurp(rd / "primary.json"));
        const json e1j = json::parse(slurp(rd / "e1.json"));
        check(strf("r%d_primary_m_star", r), pj["m_star"] == 16 && pj["m_half"] == 8);
        for (int mi = 0; mi < 2; ++mi) {
            const char* m = mi == 0 ? "a" : "b";
            int np = 0;
            const Stats S = recompute(rd, m, 128, 16, 0.25, &np);
            std::string why;
            check(strf("r%d_%s_planes_ge3", r, m), np >= 3, strf("np=%d", np));
            check(strf("r%d_%s_primary_vs_recomputed", r, m), stats_match(pj[m], S, 1e-12, &why), why);
            json e1x = {{"qbar", e1j[m]["qbar"]}, {"V1", e1j[m]["V1"]}, {"theta_2", e1j[m]["theta_2"]},
                        {"theta_3", e1j[m]["theta_3"]}, {"S_T", S.S_T}, {"S_tau", S.S_tau}};
            check(strf("r%d_%s_e1json_vs_recomputed", r, m), stats_match(e1x, S, 1e-12, &why), why);
            // row of ab_primary.csv
            const auto& row = abp[static_cast<std::size_t>(2 * r + mi)];
            bool ok = row[0] == strf("%d", r) && row[1] == strf("%d", 4001 + r) && row[2] == m;
            const char* keys[] = {"S_T", "S_tau", "V1", "theta_2", "theta_3", "qbar"};
            for (int k = 0; k < 6; ++k)
                ok = ok && rel_close(num(row[3 + static_cast<std::size_t>(k)]), pj[m][keys[k]].get<double>(), 0.0);
            check(strf("r%d_%s_ab_primary_row", r, m), ok);
            check(strf("r%d_%s_qbar_positive_finite", r, m), std::isfinite(S.qbar) && S.qbar > 0);
        }
        const json sb = json::parse(slurp(rd / "solver_b.json"));
        check(strf("r%d_solver_b_record", r), sb["converged"] == true && sb["fallback"] == false &&
                                                 sb["rel"].get<double>() <= 1e-10 &&
                                                 sb["balance"].get<double>() <= 1e-8 && sb["mg_levels"] == 3);
        const json st = json::parse(slurp(rd / "status.json"));
        check(strf("r%d_status_json", r), st["a"]["Np"] == 100 && st["b"]["Np"] == 100 &&
                                              st["a"]["planes"].size() == 5 && st["b"]["planes"].size() == 5);
        const json fj = json::parse(slurp(rd / "field.json"));
        check(strf("r%d_field_json", r), fj.contains("raw_variance") && fj.contains("applied_scale") &&
                                             fj.contains("final_variance") && fj["G"].size() == 3);
    }

    // ---- 3. determinism --------------------------------------------------------------------------
    const fs::path A2 = g_base / "A2";
    const int rcA2 = run_cmd(kCommon + " --out " + A2.string(), g_base / "A2.log");
    check("rerun_exit0", rcA2 == 0, strf("rc=%d", rcA2));
    const std::string d2 = rcA2 == 0 ? compare_trees(A, A2) : "rerun failed";
    check("rerun_byte_identical", d2.empty(), d2);

    // ---- 4. crash / restart ------------------------------------------------------------------------
    const fs::path C = g_base / "C";
    for (const char* ph : {"partial", "tmp", "rename"}) {
        const int rc = run_cmd(kCommon + " --out " + C.string() + " --test-abort-after 1," + ph,
                               g_base / strf("C_%s.log", ph));
        check(std::string("abort_exit99_") + ph, rc == 99, strf("rc=%d", rc));
    }
    check("abort_leaves_r0_done", fs::exists(C / "realizations/r_0000/DONE"));
    const int rcC = run_cmd(kCommon + " --out " + C.string(), g_base / "C_final.log");
    check("restart_exit0", rcC == 0, strf("rc=%d", rcC));
    check("restart_skips_r0", slurp(g_base / "C_final.log").find("R=0 SKIP done") != std::string::npos);
    const std::string dC = rcC == 0 ? compare_trees(A, C) : "restart failed";
    check("restart_byte_identical", dC.empty(), dC);
    {
        const int rc = run_cmd(strf("%s --eps 0.6 --nr 2 --out %s", kBox.c_str(), C.string().c_str()),
                               g_base / "C_mismatch.log");
        check("restart_option_mismatch_exit7", rc == 7, strf("rc=%d", rc));
    }

    // ---- 5. k = 1 control ----------------------------------------------------------------------------
    {
        const fs::path K = g_base / "K1";
        const int rc = run_cmd(kBox + " --eps 0 --nr 1 --out " + K.string(), g_base / "K1.log");
        check("k1_exit0", rc == 0, strf("rc=%d", rc));
        if (rc == 0) {
            const fs::path rd = K / "realizations/r_0000";
            const json pj = json::parse(slurp(rd / "primary.json"));
            const json e1j = json::parse(slurp(rd / "e1.json"));
            const double qa = e1j["a"]["qbar"], qb = e1j["b"]["qbar"];
            check("k1_qbar_equal", std::fabs(qa - qb) <= 1e-8 * std::fabs(qa), strf("a %.15g b %.15g", qa, qb));
            bool ok = true;
            std::string worst;
            for (const char* m : {"a", "b"}) {
                const json& e = e1j[m];
                for (const char* k : {"V1", "theta_2", "theta_3", "var_v2", "var_v3"}) {
                    const double v = e[k].get<double>();
                    if (!(std::fabs(v) <= 1e-8)) {
                        ok = false;
                        worst = strf("%s %s=%.3e", m, k, v);
                    }
                }
                for (const char* k : {"S_T", "S_tau"}) {
                    const double v = pj[m][k].get<double>();
                    if (!(std::fabs(v) <= 1e-8)) {
                        ok = false;
                        worst = strf("%s %s=%.3e", m, k, v);
                    }
                }
            }
            check("k1_statistics_zero_both_media", ok, worst);
            check("k1_a_vs_b_primary",
                  std::fabs(pj["a"]["S_T"].get<double>() - pj["b"]["S_T"].get<double>()) <= 1e-8 &&
                      std::fabs(pj["a"]["S_tau"].get<double>() - pj["b"]["S_tau"].get<double>()) <= 1e-8);
        }
    }

    // ---- 6. C0 fallback and exit 8 ------------------------------------------------------------------
    {
        // A run at 1e-9 stops at n9 iterations with rel9 in (1e-12, 1e-8]; the same box at --cg-rtol-b 1e-12 capped at
        // n9 iterations therefore stops unconverged at the SAME iterate (deterministic solver): the C0 fallback.
        const fs::path D = g_base / "F9";
        const int rc9 = run_cmd(kBox + " --eps 0.5 --nr 1 --cg-rtol-b 1e-9 --out " + D.string(), g_base / "F9.log");
        check("fallback_probe_exit0", rc9 == 0, strf("rc=%d", rc9));
        const json s9 = rc9 == 0 ? json::parse(slurp(D / "realizations/r_0000/solver_b.json")) : json::object();
        const int n9 = rc9 == 0 ? s9["iterations"].get<int>() : -1;
        const double rel9 = rc9 == 0 ? s9["rel"].get<double>() : -1.0;
        check("fallback_probe_window", n9 > 0 && rel9 > 1e-12 && rel9 <= 1e-8, strf("n9=%d rel9=%.3e", n9, rel9));
        if (n9 > 0 && rel9 > 1e-12 && rel9 <= 1e-8) {
            const fs::path F = g_base / "FB";
            const int rc = run_cmd(strf("%s --eps 0.5 --nr 1 --cg-rtol-b 1e-12 --cg-max-iter-b %d --out %s",
                                        kBox.c_str(), n9, F.string().c_str()),
                                   g_base / "FB.log");
            check("fallback_exit0", rc == 0, strf("rc=%d", rc));
            if (rc == 0) {
                const json sb = json::parse(slurp(F / "realizations/r_0000/solver_b.json"));
                check("fallback_record", sb["converged"] == false && sb["fallback"] == true &&
                                             sb["rel"].get<double>() <= 1e-8 && sb["rel"].get<double>() > 1e-12 &&
                                             sb["balance"].get<double>() <= 1e-8,
                      strf("rel=%.3e balance=%.3e", sb["rel"].get<double>(), sb["balance"].get<double>()));
                check("fallback_line", slurp(g_base / "FB.log").find("fallback=1") != std::string::npos);
                check("fallback_realization_done", fs::exists(F / "realizations/r_0000/DONE") &&
                                                       fs::exists(F / "ab_primary.csv"));
            }
        }
        const fs::path X = g_base / "X8";
        const int rc = run_cmd(kBox + " --eps 0.5 --nr 1 --cg-max-iter-b 1 --out " + X.string(), g_base / "X8.log");
        check("exit8_solver_failure", rc == 8, strf("rc=%d", rc));
        check("exit8_record_no_done", fs::exists(X / "failure_r_0000.json") &&
                                          !fs::exists(X / "realizations/r_0000/DONE") &&
                                          !fs::exists(X / "ab_primary.csv"));
        if (fs::exists(X / "failure_r_0000.json")) {
            const json f = json::parse(slurp(X / "failure_r_0000.json"));
            check("exit8_record_content", f["exit"] == 8 && f["solver_b"]["converged"] == false &&
                                              f["solver_b"]["fallback"] == false);
        }
    }

    // ---- 7. exits 2 and 3 ------------------------------------------------------------------------------
    {
        const fs::path U = g_base / "U";
        check("usage_unknown_option", run_cmd(kCommon + " --bogus 1 --out " + U.string(), g_base / "u1.log") == 2);
        check("usage_missing_eps",
              run_cmd("--grid 128,16,16 --ell 0.25 --seed0 1 --nr 1 --out " + U.string(), g_base / "u2.log") == 2 &&
                  slurp(g_base / "u2.log").find("--eps") != std::string::npos);
        check("usage_too_few_planes",
              run_cmd("--grid 32,16,16 --ell 0.25 --eps 0.5 --seed0 1 --nr 1 --out " + U.string(), g_base / "u3.log") == 2);
        check("usage_bad_mg_levels",
              run_cmd(kCommon + " --mg-levels-b 9 --out " + U.string(), g_base / "u4.log") == 2);
        check("usage_no_output_written", !fs::exists(U));
        const int rc = run_cmd("--grid 128,16,16 --ell 0.25 --eps 0.5 --seed0 1 --nr 1 --np-side 5 --out " +
                                   U.string(),
                               g_base / "u5.log");
        const std::string l5 = slurp(g_base / "u5.log");
        check("seed_face_exit3", rc == 3 && l5.find("SEED_FACE_FAILED") != std::string::npos, strf("rc=%d", rc));
        check("seed_face_before_tracking", !fs::exists(U / "realizations/r_0000"));
    }

    // ---- 8. SF-41 N6: medium (b) Pollock non-periodic in x ------------------------------------------
    {
        auto read_exits = [](const fs::path& p, std::vector<std::array<long long, 2>>* rows,
                             std::array<long long, 3>* tot, std::string* head) {
            std::ifstream f(p);
            std::string ln;
            std::getline(f, *head);
            *tot = {-1, -1, -1};
            while (std::getline(f, ln)) {
                if (ln.rfind("# Np=", 0) == 0)
                    (*tot)[0] = std::atoll(ln.c_str() + 5);
                else if (ln.rfind("# exit_inlet_total=", 0) == 0)
                    (*tot)[1] = std::atoll(ln.c_str() + 19);
                else if (ln.rfind("# exit_outlet_total=", 0) == 0)
                    (*tot)[2] = std::atoll(ln.c_str() + 20);
                else {
                    const std::size_t c1 = ln.find(','), c2 = ln.find(',', c1 + 1);
                    rows->push_back({std::atoll(ln.c_str() + c1 + 1), std::atoll(ln.c_str() + c2 + 1)});
                }
            }
        };
        const fs::path rdA = A / "realizations/r_0000";
        std::vector<std::array<long long, 2>> rows;
        std::array<long long, 3> tot{};
        std::string head;
        read_exits(rdA / "exits_b.csv", &rows, &tot, &head);
        bool zeros = rows.size() == 5 && tot == std::array<long long, 3>({100, 0, 0});
        for (const auto& r : rows)
            zeros = zeros && r[0] == 0 && r[1] == 0;
        check("exits_b_header", head == "x1,exit_inlet,exit_outlet", head);
        check("exits_b_zero_without_exits", zeros);

        const fs::path P = g_base / "Pv";
        const int rcP = run_cmd(kBox + " --eps 0.5 --nr 1 --test-b-periodic-view 1 --out " + P.string(),
                                g_base / "Pv.log");
        check("periodic_view_run_exit0", rcP == 0, strf("rc=%d", rcP));
        if (rcP == 0) {
            const fs::path rdP = P / "realizations/r_0000";
            check("e2_b_new_equals_periodic_view", slurp(rdA / "e2_b.csv") == slurp(rdP / "e2_b.csv"));
            check("e2_a_unchanged", slurp(rdA / "e2_a.csv") == slurp(rdP / "e2_a.csv"));
            check("status_json_new_equals_periodic_view", slurp(rdA / "status.json") == slurp(rdP / "status.json"));
            check("primary_json_new_equals_periodic_view", slurp(rdA / "primary.json") == slurp(rdP / "primary.json"));
        }
    }

    fs::remove_all(g_base);
    std::printf(g_failed ? "SUMMARY: %d failed\n" : "ALL_PASS\n", g_failed);
    return g_failed ? 1 : 0;
}
