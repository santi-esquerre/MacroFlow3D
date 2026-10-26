/**
 * @file lagrangian_medium_ab_main.cu
 * @brief SF-38 N4 `lagrangian_medium_ab`: medium A/B application (contract C-AB; reading rules RR4, RR5,
 *        RR12 of docs/experiments/artifacts/2026-10-23-sf38-long-domain-infrastructure/preregistration.md).
 *
 * Per realization r (seed seed0 + r), the SAME unit-amplitude field Y feeds
 *   (a) the affine-periodic medium: LongBoxStage (SF-18 -> SF-19, faces kept) -> E1 (eulerian_face_stats)
 *       -> periodic repack -> Pollock plane pass (E2);
 *   (b) the Dirichlet-in-x medium: build_dirichlet_medium(Y, eps, G1 = G[0] of (a)) -> E1 -> Pollock plane
 *       pass (E2) on the faces NON-periodic in x (SF-41 N6: NonPeriodicXFaceFluxes, outlet plane U[nx] kept,
 *       exit through x = 0 / x = L1 = status 8 / 9), with the C0 solver record and the fallback rule (D8).
 * Primary statistics per medium (Implementation 3 of the specification, RR5):
 *   S_T   = [sigma_T^2(X_ms) - sigma_T^2(X_ms/2)] / ((ms/2) lambda), ms = m* (largest m),  sigma_T^2 = (var_x2 + var_x3)/2
 *   S_tau = qbar^2 [var_t(X_ms) - var_t(X_ms/2)] / ((ms/2) lambda)
 *   V1, theta_2, theta_3 from E1 of the same medium.  Lengths are box units (lambda = ell).
 *
 * Output tree under --out DIR (every file byte-reproducible except timings.json):
 *   ab_manifest.json                     command line (without --out / the test hook), grid, ell, eps, seeds,
 *                                        planes, option signature
 *   realizations/r_%04d/
 *     e1_a.csv, e1_b.csv                 one file per medium, header
 *                                        kind,i,x1,u_mean,u_var,v_mean,v_var,w_mean,w_var; kind=plane rows
 *                                        (i = 0..nx, x1 = i h) fill u_*; kind=layer rows (i = 0..nx-1,
 *                                        x1 = (i+1/2) h) fill v_*, w_*; unused cells are empty; raw (not
 *                                        normalized by qbar) population statistics, %.17g
 *     e1.json                            per medium: qbar, interior plane/layer ranges, V1, theta_2, theta_3,
 *                                        var_v2, var_v3 (normalized as EulerianFaceStats.hpp)
 *     e2_a.csv, e2_b.csv                 x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_t,var_t
 *     primary.json                       per medium the five primary statistics (+ qbar), m*, planes used
 *     solver_b.json                      the C0 record of medium (b) incl. the fallback flag
 *     exits_b.csv                        SF-41 N6, medium (b) only: header x1,exit_inlet,exit_outlet; one row per
 *                                        plane (x1 %.17g; integers: particles that left through x = 0 / x = L1
 *                                        before reaching that plane, cumulative); then comment lines
 *                                        '# Np=N', '# exit_inlet_total=N', '# exit_outlet_total=N'
 *     status.json                        per medium Pollock status histogram, seed_nonpos, per-plane counts
 *     field.json                         SF-18 raw variance / applied scale / final variance and G, SF-19 lines
 *     timings.json                       wall-clock values, build commit, memory lines (non-deterministic)
 *     DONE                               option signature, written last
 *   ab_primary.csv                       after the last realization: r,seed,medium,S_T,S_tau,V1,theta_2,
 *                                        theta_3,qbar (%.17g; r ascending, medium a then b)
 *   failure_r_%04d.json                  only if realization r failed with exit 4 or 8 (the reason)
 * Atomicity / restart as lagrangian_ensemble: a realization is built in realizations/r_%04d.tmp/, renamed,
 * DONE last; restart skips DONE realizations, deletes *.tmp and unfinished r_%04d directories; DONE holds the
 * option signature (a restart under different options is refused, exit 7).
 *
 * Exit codes: 0 all realizations completed; 1 unexpected exception; 2 usage; 3 seed face (RR12); 4 stage (a)
 * failure (SF-19 not converged / faces or G unusable); 7 option mismatch on restart; 8 medium (b) solver
 * failure beyond the fallback; 99 test abort.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/external/nlohmann/json.hpp"
#include "src/physics/particles/lagrangian_labels/DirichletMedium.cuh"
#include "src/physics/particles/lagrangian_labels/EnsembleRunner.cuh" // find_seed_on_face, UsageError
#include "src/physics/particles/lagrangian_labels/EulerianFaceStats.hpp"
#include "src/physics/particles/lagrangian_labels/FaceFluxRepack.cuh"
#include "src/physics/particles/lagrangian_labels/LongBoxStage.cuh"
#include "src/physics/particles/lagrangian_labels/PollockComparator.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/runtime/CudaContext.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace ll = macroflow3d::lagrangian_labels;
namespace sl = macroflow3d::streamfunctions::inlet_slab;
namespace stt = macroflow3d::physics::particles::streamline_tracker;
namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kExitOk = 0, kExitException = 1, kExitUsage = 2, kExitSeedFace = 3, kExitStage = 4,
              kExitMismatch = 7, kExitSolverB = 8, kExitTestAbort = 99;

double since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

void out_line(const std::string& s) {
    std::fputs(s.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

// ================================================================================================
// Configuration
// ================================================================================================

struct AbConfig {
    int nx = 0, ny = 0, nz = 0;
    double sigma2 = 1.0, ell = 0.0, eps = 0.0;
    unsigned long long seed0 = 0;
    int nr = 1;
    int np_side = 100;
    double window[4] = {0.1, 0.9, 0.1, 0.9};
    double x0_lambda = 5.0;
    int mg_levels_b = 0; ///< resolved (auto -> box_auto_mg_levels)
    bool mg_levels_auto = true;
    double cg_rtol_b = 1e-10;
    int cg_max_iter_b = 2000;
    std::string out;
    std::string command_line; ///< without --out and --test-abort-after
    int test_abort_r = -1;
    bool test_b_periodic_view = false; ///< TEST ONLY: medium (b) Pollock on the pre-SF-41 periodic view
    std::string test_abort_phase;
    // derived
    double L1 = 0.0, X0 = 0.0;
    std::vector<long long> plane_m; ///< m = 1, 2, 4, ...
    std::vector<double> planes;     ///< X_m = X0 + m lambda
};

double parse_double(const std::string& opt, const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size() || !std::isfinite(v))
        throw ll::UsageError(opt + ": '" + s + "' is not a finite number");
    return v;
}

long long parse_int(const std::string& opt, const std::string& s) {
    char* end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (s.empty() || end != s.c_str() + s.size())
        throw ll::UsageError(opt + ": '" + s + "' is not an integer");
    return v;
}

std::vector<double> parse_list(const std::string& opt, const std::string& s) {
    std::vector<double> v;
    std::size_t a = 0;
    while (true) {
        const std::size_t b = s.find(',', a);
        v.push_back(parse_double(opt, s.substr(a, b == std::string::npos ? b : b - a)));
        if (b == std::string::npos)
            break;
        a = b + 1;
    }
    return v;
}

std::string ab_usage() {
    return "usage: lagrangian_medium_ab --grid nx,ny,nz --ell L --eps E --seed0 S --nr N --out DIR [options]\n"
           "  --grid nx,ny,nz         box [0, nx/ny] x [0,1]^2, h = 1/ny (ny == nz, even, >= 4)\n"
           "  --ell L                 correlation length in box units (= lambda)\n"
           "  --eps E                 k = exp(E Y)\n"
           "  --sigma2 V              SF-18 variance (default 1)\n"
           "  --seed0 S               realization r uses seed S + r\n"
           "  --nr N                  number of realizations\n"
           "  --np-side M             M x M seed lattice (default 100)\n"
           "  --window a,b,c,d        injection window w2a,w2b,w3a,w3b (default 0.1,0.9,0.1,0.9)\n"
           "  --x0-lambda K           injection plane X0 = K lambda (default 5)\n"
           "  --mg-levels-b auto|K    MG levels of medium (b) (default auto = box_auto_mg_levels)\n"
           "  --cg-rtol-b R           PCG relative tolerance of medium (b) (default 1e-10)\n"
           "  --cg-max-iter-b N       PCG iteration cap of medium (b) (default 2000)\n"
           "  --out DIR               output directory\n"
           "planes: X_m = X0 + m lambda, m = 1,2,4,... while X_m <= L1 - 5 lambda (at least 2 planes)\n"
           "exit codes: 0 ok, 1 exception, 2 usage, 3 seed on a cell face (RR12), 4 stage (a) failure,\n"
           "            7 DONE realization under different options, 8 medium (b) solver failure beyond the\n"
           "            fallback (rel <= 1e-8 and balance <= 1e-8 when not converged at --cg-rtol-b)\n";
}

AbConfig parse_args(int argc, const char* const* argv) {
    static const std::set<std::string> value_opts = {
        "--grid",  "--ell",       "--eps",         "--sigma2",         "--seed0", "--nr",
        "--np-side", "--window",  "--x0-lambda",   "--mg-levels-b",    "--cg-rtol-b",
        "--cg-max-iter-b", "--out", "--test-abort-after",
        "--test-b-periodic-view"};
    AbConfig c;
    std::map<std::string, std::string> val;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (!value_opts.count(a))
            throw ll::UsageError("unknown option '" + a + "'");
        if (val.count(a))
            throw ll::UsageError("option " + a + " given twice");
        if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)
            throw ll::UsageError("option " + a + " needs a value");
        val[a] = argv[++i];
        if (a != "--out" && a != "--test-abort-after" && a != "--test-b-periodic-view")
            c.command_line += (c.command_line.empty() ? "" : " ") + a + " " + val[a];
    }
    for (const char* o : {"--grid", "--ell", "--eps", "--seed0", "--nr", "--out"})
        if (!val.count(o))
            throw ll::UsageError(std::string("missing required option ") + o);
    {
        const std::vector<double> g = parse_list("--grid", val["--grid"]);
        if (g.size() != 3)
            throw ll::UsageError("--grid: expected nx,ny,nz");
        for (double d : g)
            if (d != std::floor(d) || d < 4 || d > 1e6)
                throw ll::UsageError("--grid: extents must be integers >= 4");
        c.nx = static_cast<int>(g[0]);
        c.ny = static_cast<int>(g[1]);
        c.nz = static_cast<int>(g[2]);
        if (c.ny != c.nz)
            throw ll::UsageError("--grid: ny must equal nz (the lean stage takes one transverse N)");
        if (c.nx % 2 != 0 || c.ny % 2 != 0)
            throw ll::UsageError("--grid: extents must be even");
    }
    c.ell = parse_double("--ell", val["--ell"]);
    if (!(c.ell > 0.0))
        throw ll::UsageError("--ell: must be > 0");
    c.eps = parse_double("--eps", val["--eps"]);
    if (val.count("--sigma2")) {
        c.sigma2 = parse_double("--sigma2", val["--sigma2"]);
        if (!(c.sigma2 >= 0.0))
            throw ll::UsageError("--sigma2: must be >= 0");
    }
    {
        const std::string& s = val["--seed0"];
        char* end = nullptr;
        if (s.empty() || s[0] == '-' || s[0] == '+')
            throw ll::UsageError("--seed0: '" + s + "' is not an unsigned integer");
        c.seed0 = std::strtoull(s.c_str(), &end, 10);
        if (end != s.c_str() + s.size())
            throw ll::UsageError("--seed0: '" + s + "' is not an unsigned integer");
    }
    {
        const long long nr = parse_int("--nr", val["--nr"]);
        if (nr < 1 || nr > 100000)
            throw ll::UsageError("--nr: must be in [1, 100000]");
        c.nr = static_cast<int>(nr);
    }
    if (val.count("--np-side")) {
        const long long m = parse_int("--np-side", val["--np-side"]);
        if (m < 1 || m > 4096)
            throw ll::UsageError("--np-side: must be in [1, 4096]");
        c.np_side = static_cast<int>(m);
    }
    if (val.count("--window")) {
        const std::vector<double> w = parse_list("--window", val["--window"]);
        if (w.size() != 4)
            throw ll::UsageError("--window: expected w2a,w2b,w3a,w3b");
        for (int k = 0; k < 4; ++k)
            c.window[k] = w[static_cast<std::size_t>(k)];
    }
    if (!(c.window[0] < c.window[1]) || !(c.window[2] < c.window[3]))
        throw ll::UsageError("--window: need w2a < w2b and w3a < w3b");
    if (val.count("--x0-lambda")) {
        c.x0_lambda = parse_double("--x0-lambda", val["--x0-lambda"]);
        if (!(c.x0_lambda > 0.0))
            throw ll::UsageError("--x0-lambda: must be > 0");
    }
    if (val.count("--cg-rtol-b")) {
        c.cg_rtol_b = parse_double("--cg-rtol-b", val["--cg-rtol-b"]);
        if (!(c.cg_rtol_b > 0.0))
            throw ll::UsageError("--cg-rtol-b: must be > 0");
    }
    if (val.count("--cg-max-iter-b")) {
        const long long n = parse_int("--cg-max-iter-b", val["--cg-max-iter-b"]);
        if (n < 1 || n > 10000000)
            throw ll::UsageError("--cg-max-iter-b: must be in [1, 1e7]");
        c.cg_max_iter_b = static_cast<int>(n);
    }
    {
        const std::string s = val.count("--mg-levels-b") ? val["--mg-levels-b"] : "auto";
        if (s == "auto") {
            c.mg_levels_auto = true;
            c.mg_levels_b = ll::box_auto_mg_levels(c.nx, c.ny, c.nz);
        } else {
            c.mg_levels_auto = false;
            const long long k = parse_int("--mg-levels-b", s);
            if (k < 1 || k > 30)
                throw ll::UsageError("--mg-levels-b: must be 'auto' or an integer in [1, 30]");
            const long long f = 1LL << (k - 1);
            if (c.nx % f != 0 || c.ny % f != 0 || c.nz % f != 0 || c.nx / f < 2 || c.ny / f < 2)
                throw ll::UsageError("--mg-levels-b: nx, ny, nz must be divisible by 2^(K-1) with "
                                     "coarsest extents >= 2");
            c.mg_levels_b = static_cast<int>(k);
        }
    }
    c.out = val["--out"];
    if (c.out.empty())
        throw ll::UsageError("--out: empty");
    if (val.count("--test-b-periodic-view")) {
        if (val["--test-b-periodic-view"] != "1")
            throw ll::UsageError("--test-b-periodic-view: expected 1");
        c.test_b_periodic_view = true;
    }
    if (val.count("--test-abort-after")) {
        const std::string& s = val["--test-abort-after"];
        const std::size_t comma = s.find(',');
        if (comma == std::string::npos)
            throw ll::UsageError("--test-abort-after: expected r,phase");
        c.test_abort_r = static_cast<int>(parse_int("--test-abort-after", s.substr(0, comma)));
        c.test_abort_phase = s.substr(comma + 1);
        if (c.test_abort_phase != "stage" && c.test_abort_phase != "partial" &&
            c.test_abort_phase != "tmp" && c.test_abort_phase != "rename")
            throw ll::UsageError("--test-abort-after: phase must be stage|partial|tmp|rename");
    }

    // derived geometry and planes (RR5): X_m = X0 + m lambda, m = 1, 2, 4, ... while X_m <= L1 - 5 lambda
    c.L1 = static_cast<double>(c.nx) / static_cast<double>(c.ny);
    c.X0 = c.x0_lambda * c.ell;
    const double xmax = c.L1 - 5.0 * c.ell;
    for (long long m = 1;; m *= 2) {
        const double x = c.X0 + static_cast<double>(m) * c.ell;
        if (x > xmax + 1e-9 * c.ell) // closed interval; 1e-9 lambda resolves exact-grid ties
            break;
        c.plane_m.push_back(m);
        c.planes.push_back(x);
    }
    if (c.planes.size() < 2)
        throw ll::UsageError(fmt("the plane rule gives %zu plane(s) (X0 = %g, L1 - 5 lambda = %g): at least "
                                 "2 are needed for the last-octave statistics",
                                 c.planes.size(), c.X0, xmax));
    return c;
}

std::string join_d(const std::vector<double>& v, const char* f) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i)
        s += (i ? "," : "") + fmt(f, v[i]);
    return s;
}

std::string signature(const AbConfig& c) {
    return fmt("grid=%dx%dx%d sigma2=%.17g ell=%.17g eps=%.17g seed0=%llu M=%d window=%.17g,%.17g,%.17g,%.17g "
               "x0_lambda=%.17g mg_levels_b=%d cg_rtol_b=%.17g cg_max_iter_b=%d planes=%s",
               c.nx, c.ny, c.nz, c.sigma2, c.ell, c.eps, c.seed0, c.np_side, c.window[0], c.window[1],
               c.window[2], c.window[3], c.x0_lambda, c.mg_levels_b, c.cg_rtol_b, c.cg_max_iter_b,
               join_d(c.planes, "%.17g").c_str());
}

// ================================================================================================
// File helpers and atomicity (the same scheme as EnsembleRunner.cu, whose helpers are file-local)
// ================================================================================================

void write_json(const std::string& path, const json& j) {
    std::ofstream f(path, std::ios::trunc);
    if (!f)
        throw std::runtime_error("cannot write " + path);
    f << j.dump(2) << "\n";
    if (!f)
        throw std::runtime_error("write failed for " + path);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string rdir_of(const AbConfig& c, int r) { return fmt("%s/realizations/r_%04d", c.out.c_str(), r); }
std::string failure_path(const AbConfig& c, int r) { return fmt("%s/failure_r_%04d.json", c.out.c_str(), r); }

std::string done_text(const AbConfig& c, int r) {
    return "signature: " + signature(c) + "\n" +
           fmt("r=%d\nseed=%llu\n", r, c.seed0 + static_cast<unsigned long long>(r));
}

void test_abort(const AbConfig& c, int r, const char* phase) {
    if (c.test_abort_r == r && c.test_abort_phase == phase) {
        out_line(fmt("R=%d TEST_ABORT phase=%s (simulated crash, exit %d)", r, phase, kExitTestAbort));
        std::_Exit(kExitTestAbort);
    }
}

void commit_realization(const AbConfig& c, int r, const std::string& tmpdir) {
    test_abort(c, r, "tmp");
    const std::string fin = rdir_of(c, r);
    fs::rename(tmpdir, fin);
    test_abort(c, r, "rename");
    const std::string done_tmp = fin + "/DONE.tmp";
    {
        std::ofstream f(done_tmp, std::ios::trunc);
        f << done_text(c, r);
        if (!f)
            throw std::runtime_error("cannot write " + done_tmp);
    }
    fs::rename(done_tmp, fin + "/DONE");
}

std::string build_commit() {
#ifdef MACROFLOW3D_GIT_HASH
    return MACROFLOW3D_GIT_HASH;
#else
    return "unknown";
#endif
}

// ================================================================================================
// Per-medium results and output writers
// ================================================================================================

struct Primary {
    double S_T = 0.0, S_tau = 0.0, V1 = 0.0, theta_2 = 0.0, theta_3 = 0.0, qbar = 0.0;
};

/// RR5 / Implementation 3. pm: Pollock plane moments at the planes of c (last two = m*/2 and m*).
Primary compute_primary(const AbConfig& c, const ll::EulerianFaceStatsResult& e1,
                        const std::vector<ll::PollockPlaneMoments>& pm) {
    const std::size_t P = pm.size();
    const ll::PollockPlaneMoments& lo = pm[P - 2];
    const ll::PollockPlaneMoments& hi = pm[P - 1];
    const double dx = (static_cast<double>(c.plane_m.back()) / 2.0) * c.ell;
    Primary p;
    p.qbar = e1.qbar;
    p.S_T = (0.5 * (hi.var_x2 + hi.var_x3) - 0.5 * (lo.var_x2 + lo.var_x3)) / dx;
    p.S_tau = e1.qbar * e1.qbar * (hi.var_t - lo.var_t) / dx;
    p.V1 = e1.V1;
    p.theta_2 = e1.theta_2;
    p.theta_3 = e1.theta_3;
    return p;
}

json primary_json(const Primary& p) {
    return {{"S_T", p.S_T},       {"S_tau", p.S_tau},     {"V1", p.V1},
            {"theta_2", p.theta_2}, {"theta_3", p.theta_3}, {"qbar", p.qbar}};
}

void write_e1_csv(const std::string& path, const ll::EulerianFaceStatsResult& e, int nx, double h) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f)
        throw std::runtime_error("cannot write " + path);
    std::fprintf(f, "kind,i,x1,u_mean,u_var,v_mean,v_var,w_mean,w_var\n");
    for (int i = 0; i <= nx; ++i)
        std::fprintf(f, "plane,%d,%.17g,%.17g,%.17g,,,,\n", i, i * h, e.u_mean[static_cast<std::size_t>(i)],
                     e.u_var[static_cast<std::size_t>(i)]);
    for (int i = 0; i < nx; ++i)
        std::fprintf(f, "layer,%d,%.17g,,,%.17g,%.17g,%.17g,%.17g\n", i, (i + 0.5) * h,
                     e.v_mean[static_cast<std::size_t>(i)], e.v_var[static_cast<std::size_t>(i)],
                     e.w_mean[static_cast<std::size_t>(i)], e.w_var[static_cast<std::size_t>(i)]);
    std::fclose(f);
}

json e1_json(const ll::EulerianFaceStatsResult& e, double h) {
    return {{"qbar", e.qbar},
            {"plane_lo", e.plane_lo},
            {"plane_hi", e.plane_hi},
            {"layer_lo", e.layer_lo},
            {"layer_hi", e.layer_hi},
            {"interior_x1_planes", {e.plane_lo * h, e.plane_hi * h}},
            {"interior_x1_layers", {(e.layer_lo + 0.5) * h, (e.layer_hi + 0.5) * h}},
            {"V1", e.V1},
            {"theta_2", e.theta_2},
            {"theta_3", e.theta_3},
            {"var_v2", e.var_v2},
            {"var_v3", e.var_v3}};
}

void write_e2_csv(const std::string& path, const std::vector<ll::PollockPlaneMoments>& pm) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f)
        throw std::runtime_error("cannot write " + path);
    std::fprintf(f, "x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_t,var_t\n");
    for (const ll::PollockPlaneMoments& m : pm)
        std::fprintf(f, "%.17g,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", m.x1, m.contributing, m.non_ok,
                     m.mean_x2, m.var_x2, m.mean_x3, m.var_x3, m.mean_t, m.var_t);
    std::fclose(f);
}

void write_exits_csv(const std::string& path, const ll::PollockLatticeResult& Q) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f)
        throw std::runtime_error("cannot write " + path);
    std::fprintf(f, "x1,exit_inlet,exit_outlet\n");
    for (std::size_t m = 0; m < Q.moments_planes.size(); ++m)
        std::fprintf(f, "%.17g,%lld,%lld\n", Q.moments_planes[m].x1, Q.exit_inlet_plane.at(m),
                     Q.exit_outlet_plane.at(m));
    std::fprintf(f, "# Np=%d\n# exit_inlet_total=%lld\n# exit_outlet_total=%lld\n", Q.Np,
                 Q.exit_inlet_plane_total, Q.exit_outlet_plane_total);
    std::fclose(f);
}

json status_json(const ll::PollockLatticeResult& Q) {
    long long nonactive = 0;
    for (std::size_t k = 1; k < Q.status_hist_plane.size(); ++k)
        nonactive += Q.status_hist_plane[k];
    json planes = json::array();
    for (const ll::PollockPlaneMoments& m : Q.moments_planes)
        planes.push_back({{"x1", m.x1}, {"contributing", m.contributing}, {"non_ok", m.non_ok}});
    return {{"Np", Q.Np},
            {"seed_nonpos", Q.seed_nonpos},
            {"status_hist_plane", Q.status_hist_plane},
            {"non_active_particles", nonactive},
            {"planes", planes}};
}

/// Pollock plane pass (empty time list), the SF-32 documented cells-per-call default.
template <class FaceView> // PeriodicFaceFluxView (a) or NonPeriodicXFaceFluxView (b)
ll::PollockLatticeResult pollock_planes(CudaContext& ctx, const FaceView& view, const ll::InjectionSpec& inj,
                                        const std::vector<double>& planes) {
    stt::PollockParams prm;
    prm.max_cells_per_call = 10000000;
    return ll::track_pollock_lattice(ctx, view, inj, planes, {}, prm);
}

std::string hist_string(const std::array<long long, ll::kPollockStatusSlots>& h) {
    std::string s;
    for (std::size_t k = 0; k < h.size(); ++k) {
        if (h[k] == 0 && k != 0)
            continue;
        s += fmt(s.empty() ? "%zu:%lld" : ",%zu:%lld", k, h[k]);
    }
    return s;
}

// ================================================================================================
// One realization
// ================================================================================================

int run_realization(CudaContext& ctx, const AbConfig& c, int r) {
    const auto t_all = Clock::now();
    const unsigned long long seed = c.seed0 + static_cast<unsigned long long>(r);
    const std::string tmpdir = rdir_of(c, r) + ".tmp";
    fs::remove_all(tmpdir);
    fs::create_directories(tmpdir);
    fs::remove(failure_path(c, r));
    auto P = [&](const std::string& s) { out_line(fmt("R=%d ", r) + s); };
    auto fail = [&](const std::string& status, int code, json record) {
        P("STATUS " + status);
        record["exit"] = code;
        record["status"] = status;
        record["r"] = r;
        record["seed"] = seed;
        write_json(failure_path(c, r), record);
        fs::remove_all(tmpdir);
        return code;
    };
    const double h = 1.0 / static_cast<double>(c.ny);
    json TJ;
    TJ["r"] = r;
    TJ["seed"] = seed;
    TJ["commit"] = build_commit();
    const ll::InjectionSpec inj = [&] {
        ll::InjectionSpec s;
        s.X0 = c.X0;
        s.M = c.np_side;
        s.w2a = c.window[0];
        s.w2b = c.window[1];
        s.w3a = c.window[2];
        s.w3b = c.window[3];
        return s;
    }();

    // ================= medium (a) =================================================================
    ll::LongBoxSpec spec;
    spec.nx = c.nx;
    spec.ny = c.ny;
    spec.nz = c.nz;
    spec.source = ll::LongBoxSourceKind::gaussian;
    spec.sigma2 = c.sigma2;
    spec.ell = c.ell;
    spec.seed = seed;
    spec.normalize_variance = true;
    spec.field_name = "gaussian";
    spec.eps = c.eps;
    spec.pcg_rtol = 1e-10;
    spec.bhat_mapped_host = true;
    spec.keep_face_fluxes = true;
    spec.keep_host_field = true;

    const auto t_stage = Clock::now();
    auto stage = std::make_unique<ll::LongBoxStage>(ll::build_long_box_stage(ctx, spec));
    const double stage_s = since(t_stage);
    const sl::ProductionStageReport& rep = stage->report;

    P(fmt("FIELD gaussian (SF-18) N=%d grid=%dx%dx%d sigma2=%g ell=%g seed=%llu normalize_variance=1 "
          "applied_scale=%.12e raw_variance=%.12e final_variance=%.12e active_modes=%zu | stage field "
          "k = exp(eps Y), eps = %g",
          c.nx, c.nx, c.ny, c.nz, c.sigma2, c.ell, seed, rep.sf18.applied_scale, rep.sf18.raw_variance,
          rep.sf18.final_variance, rep.sf18.active_mode_count, c.eps));
    std::vector<std::string> setup_lines; // SETUP block: the stage summary with the two vertex-vD lines made explicit
    {
        const std::string s = rep.summary();
        std::size_t a = 0;
        while (a < s.size()) {
            std::size_t b = s.find('\n', a);
            if (b == std::string::npos)
                b = s.size();
            const std::string ln = s.substr(a, b - a);
            if (ln.rfind("  inlet:", 0) == 0)
                setup_lines.push_back(fmt("  inlet: vmin=%.6e mean(U plane 0)=%.15f (mean-1=%.3e) Q0-1=%.3e "
                                          "v_rms=n/a (lean stage: vertex vD not built)",
                                          rep.inlet_vmin, rep.inlet_mean, rep.inlet_mean - 1.0, rep.Q0 - 1.0));
            else if (ln.rfind("  v1(U-face)", 0) == 0)
                setup_lines.push_back("  v1(U-face) vs spline flow k g1: n/a (lean stage); spline potential GPU "
                                      "vs host prefilter: n/a");
            else
                setup_lines.push_back(ln);
            a = b + 1;
        }
        for (const std::string& ln : setup_lines)
            P("SETUP " + ln);
    }
    P(fmt("SETUP stage eps=%g built in %.3fs (lean stage, Grid3D %dx%dx%d, bhat=mapped, faces kept)", c.eps,
          stage_s, c.nx, c.ny, c.nz));
    P(fmt("STAGE status=%s grid=%dx%dx%d mg_levels=%d G=%.12e,%.12e,%.12e darcy_converged=%d flow_ready=%d "
          "seconds=%.3f",
          sl::to_string(rep.status), c.nx, c.ny, c.nz, rep.mg_levels, stage->G[0], stage->G[1], stage->G[2],
          rep.darcy_converged ? 1 : 0, stage->flow_ready() ? 1 : 0, stage_s));
    const std::vector<std::string> mems_a = ll::mem_lines(*stage, nullptr);
    for (const std::string& l : mems_a)
        P(l);
    const double G1 = stage->G[0];
    if (!stage->flow_ready() || !stage->faces_available() || !(std::isfinite(G1) && G1 > 0.0)) {
        P(fmt("STAGE_FAILED status=%s darcy_converged=%d faces_available=%d G1=%.12e: the SF-19 flow did not "
              "converge, no faces were kept, or G[0] is not a positive head drop for medium (b)",
              sl::to_string(rep.status), rep.darcy_converged ? 1 : 0, stage->faces_available() ? 1 : 0, G1));
        return fail("darcy_failed", kExitStage,
                    {{"stage_status", sl::to_string(rep.status)},
                     {"darcy_converged", rep.darcy_converged},
                     {"faces_available", stage->faces_available()},
                     {"G1", G1}});
    }
    json field_j = {{"kind", "gaussian"},
                    {"grid", {c.nx, c.ny, c.nz}},
                    {"sigma2", c.sigma2},
                    {"ell", c.ell},
                    {"seed", seed},
                    {"normalize_variance", true},
                    {"raw_mean", rep.sf18.raw_mean},
                    {"raw_variance", rep.sf18.raw_variance},
                    {"applied_scale", rep.sf18.applied_scale},
                    {"final_variance", rep.sf18.final_variance},
                    {"active_modes", rep.sf18.active_mode_count},
                    {"G", {stage->G[0], stage->G[1], stage->G[2]}},
                    {"mg_levels_a", rep.mg_levels},
                    {"sf19_lines", setup_lines}};
    test_abort(c, r, "stage");

    const auto t_e1a = Clock::now();
    const ll::EulerianFaceStatsResult e1a = ll::eulerian_face_stats(
        stage->face_flux_u, stage->face_flux_v, stage->face_flux_w, c.nx, c.ny, c.nz, c.L1, c.ell);
    const double e1a_s = since(t_e1a);
    P(fmt("E1 medium=a qbar=%.12e V1=%.12e theta_2=%.12e theta_3=%.12e", e1a.qbar, e1a.V1, e1a.theta_2,
          e1a.theta_3));

    // keep the host Y and the faces; release every device array of the stage before anything else
    std::vector<real> Y = std::move(stage->host_field);
    std::vector<real> fu = std::move(stage->face_flux_u), fv = std::move(stage->face_flux_v),
                      fw = std::move(stage->face_flux_w);
    stage.reset();

    const auto t_e2a = Clock::now();
    ll::PollockLatticeResult Qa;
    double repack_a_s = 0.0;
    {
        const auto t_rep = Clock::now();
        ll::PeriodicFaceFluxes faces =
            ll::PeriodicFaceFluxes::upload_and_repack(ctx, fu, fv, fw, c.nx, c.ny, c.nz, h);
        repack_a_s = since(t_rep);
        std::vector<real>().swap(fu); // host faces of (a) are dead
        std::vector<real>().swap(fv);
        std::vector<real>().swap(fw);
        Qa = pollock_planes(ctx, faces.view(), inj, c.planes);
    } // periodic faces of (a) freed here
    const double e2a_s = since(t_e2a);
    P(fmt("E2 medium=a planes=%zu m_star=%lld", c.planes.size(), c.plane_m.back()));
    for (std::size_t m = 0; m < Qa.moments_planes.size(); ++m) {
        const ll::PollockPlaneMoments& q = Qa.moments_planes[m];
        P(fmt("E2PLANE medium=a m=%lld x1=%.10g contributing=%d non_ok=%d var_T=%.12e var_t=%.12e",
              c.plane_m[m], q.x1, q.contributing, q.non_ok, 0.5 * (q.var_x2 + q.var_x3), q.var_t));
    }
    P(fmt("POLLOCK medium=a Np=%d seed_nonpos=%lld hist_plane=%s", Qa.Np, Qa.seed_nonpos,
          hist_string(Qa.status_hist_plane).c_str()));
    const Primary pa = compute_primary(c, e1a, Qa.moments_planes);
    P(fmt("PRIMARY medium=a S_T=%.12e S_tau=%.12e V1=%.12e theta_2=%.12e theta_3=%.12e", pa.S_T, pa.S_tau,
          pa.V1, pa.theta_2, pa.theta_3));

    // ================= medium (b) =================================================================
    ll::DirichletMediumSpec bs;
    bs.nx = c.nx;
    bs.ny = c.ny;
    bs.nz = c.nz;
    bs.eps = c.eps;
    bs.G1 = G1;
    bs.mg_levels = c.mg_levels_b;
    bs.cg_rtol = c.cg_rtol_b;
    bs.cg_max_iter = c.cg_max_iter_b;
    bs.keep_host_faces = true;
    bs.build_periodic_faces = c.test_b_periodic_view; // production: non-periodic upload below
    const auto t_b = Clock::now();
    auto med = std::make_unique<ll::DirichletMedium>(ll::build_dirichlet_medium(ctx, bs, Y));
    const double b_s = since(t_b);
    std::vector<real>().swap(Y);
    const ll::DirichletMediumReport& B = med->report;

    // C0 / D8 record and the fallback rule
    const bool finite_ok = std::isfinite(B.rel) && std::isfinite(B.balance) && std::isfinite(B.max_div) &&
                           std::isfinite(B.qbar) && std::isfinite(B.Q_in) && std::isfinite(B.Q_out);
    constexpr double kFallbackTol = 1e-8;
    bool fallback = false, usable = false;
    std::string b_verdict;
    if (!finite_ok) {
        b_verdict = "non_finite";
    } else if (B.converged) {
        usable = B.balance <= kFallbackTol;
        b_verdict = usable ? "converged" : "converged_balance_above_1e-8";
    } else if (B.rel <= kFallbackTol && B.balance <= kFallbackTol) {
        usable = fallback = true;
        b_verdict = "fallback";
    } else {
        b_verdict = "not_converged_beyond_fallback";
    }
    P(fmt("B_SOLVER r=%d converged=%d iterations=%d initial=%.6e final=%.6e rel=%.6e qbar=%.12e Q_in=%.12e "
          "Q_out=%.12e balance=%.6e max_div=%.6e fallback=%d",
          r, B.converged ? 1 : 0, B.iterations, B.initial_residual, B.final_residual, B.rel, B.qbar, B.Q_in,
          B.Q_out, B.balance, B.max_div, fallback ? 1 : 0));
    const std::vector<std::string> mems_b = med->mem_lines();
    for (const std::string& l : mems_b)
        P(l);
    const json solver_j = {{"r", r},
                           {"seed", seed},
                           {"converged", B.converged},
                           {"iterations", B.iterations},
                           {"initial_residual", B.initial_residual},
                           {"final_residual", B.final_residual},
                           {"rel", B.rel},
                           {"qbar", B.qbar},
                           {"Q_in", B.Q_in},
                           {"Q_out", B.Q_out},
                           {"balance", B.balance},
                           {"max_div", B.max_div},
                           {"head_drop", B.head_drop},
                           {"G1", G1},
                           {"mg_levels", c.mg_levels_b},
                           {"cg_rtol", c.cg_rtol_b},
                           {"cg_max_iter", c.cg_max_iter_b},
                           {"fallback", fallback},
                           {"fallback_tolerance", kFallbackTol},
                           {"verdict", b_verdict}};
    if (!usable) {
        P(fmt("B_SOLVER_FAILED verdict=%s: not converged at cg_rtol=%g beyond the fallback (rel <= 1e-8 and "
              "balance <= 1e-8), a converged solve with balance > 1e-8, or non-finite values; stop -> owner",
              b_verdict.c_str(), c.cg_rtol_b));
        return fail("solver_b_failed", kExitSolverB, {{"solver_b", solver_j}});
    }

    const auto t_e1b = Clock::now();
    const ll::EulerianFaceStatsResult e1b =
        ll::eulerian_face_stats(med->face_u, med->face_v, med->face_w, c.nx, c.ny, c.nz, c.L1, c.ell);
    const double e1b_s = since(t_e1b);
    P(fmt("E1 medium=b qbar=%.12e V1=%.12e theta_2=%.12e theta_3=%.12e", e1b.qbar, e1b.V1, e1b.theta_2,
          e1b.theta_3));
    const auto t_e2b = Clock::now();
    ll::PollockLatticeResult Qb;
    if (c.test_b_periodic_view) {
        std::vector<real>().swap(med->face_u);
        std::vector<real>().swap(med->face_v);
        std::vector<real>().swap(med->face_w);
        Qb = pollock_planes(ctx, med->periodic_view(), inj, c.planes);
    } else {
        ll::NonPeriodicXFaceFluxes faces = ll::NonPeriodicXFaceFluxes::upload(
            ctx, med->face_u, med->face_v, med->face_w, c.nx, c.ny, c.nz, h);
        std::vector<real>().swap(med->face_u); // host faces of (b) are dead
        std::vector<real>().swap(med->face_v);
        std::vector<real>().swap(med->face_w);
        Qb = pollock_planes(ctx, faces.view(), inj, c.planes);
    } // device faces of (b) freed here
    const double e2b_s = since(t_e2b);
    med.reset();
    P(fmt("E2 medium=b planes=%zu m_star=%lld", c.planes.size(), c.plane_m.back()));
    for (std::size_t m = 0; m < Qb.moments_planes.size(); ++m) {
        const ll::PollockPlaneMoments& q = Qb.moments_planes[m];
        P(fmt("E2PLANE medium=b m=%lld x1=%.10g contributing=%d non_ok=%d var_T=%.12e var_t=%.12e",
              c.plane_m[m], q.x1, q.contributing, q.non_ok, 0.5 * (q.var_x2 + q.var_x3), q.var_t));
    }
    P(fmt("POLLOCK medium=b Np=%d seed_nonpos=%lld hist_plane=%s", Qb.Np, Qb.seed_nonpos,
          hist_string(Qb.status_hist_plane).c_str()));
    const Primary pb = compute_primary(c, e1b, Qb.moments_planes);
    P(fmt("PRIMARY medium=b S_T=%.12e S_tau=%.12e V1=%.12e theta_2=%.12e theta_3=%.12e", pb.S_T, pb.S_tau,
          pb.V1, pb.theta_2, pb.theta_3));

    // ================= outputs ====================================================================
    const auto t_write = Clock::now();
    write_e1_csv(tmpdir + "/e1_a.csv", e1a, c.nx, h);
    write_e1_csv(tmpdir + "/e1_b.csv", e1b, c.nx, h);
    write_json(tmpdir + "/e1.json", {{"a", e1_json(e1a, h)}, {"b", e1_json(e1b, h)}});
    write_e2_csv(tmpdir + "/e2_a.csv", Qa.moments_planes);
    test_abort(c, r, "partial");
    write_e2_csv(tmpdir + "/e2_b.csv", Qb.moments_planes);
    const long long mstar = c.plane_m.back();
    write_json(tmpdir + "/primary.json",
               {{"r", r},
                {"seed", seed},
                {"lambda", c.ell},
                {"m_star", mstar},
                {"m_half", mstar / 2},
                {"x1_half", c.planes[c.planes.size() - 2]},
                {"x1_star", c.planes.back()},
                {"a", primary_json(pa)},
                {"b", primary_json(pb)}});
    write_json(tmpdir + "/solver_b.json", solver_j);
    if (!c.test_b_periodic_view)
        write_exits_csv(tmpdir + "/exits_b.csv", Qb);
    write_json(tmpdir + "/status.json", {{"a", status_json(Qa)}, {"b", status_json(Qb)}});
    write_json(tmpdir + "/field.json", field_j);
    TJ["seconds"] = {{"stage_a", stage_s},   {"e1_a", e1a_s},        {"repack_a", repack_a_s},
                     {"e2_a_total", e2a_s},  {"medium_b", b_s},      {"e1_b", e1b_s},
                     {"e2_b", e2b_s},        {"write", since(t_write)}, {"total", since(t_all)}};
    TJ["pollock_timings_s"] = {{"a_plane_kernel", Qa.seconds_plane_kernel},
                               {"a_download", Qa.seconds_download},
                               {"a_moments", Qa.seconds_moments},
                               {"b_plane_kernel", Qb.seconds_plane_kernel},
                               {"b_download", Qb.seconds_download},
                               {"b_moments", Qb.seconds_moments}};
    TJ["mem_lines_a"] = mems_a;
    TJ["mem_lines_b"] = mems_b;
    write_json(tmpdir + "/timings.json", TJ);

    commit_realization(c, r, tmpdir);
    P(fmt("STATUS ok solver_b=%s fallback=%d non_active_a=%lld non_active_b=%lld", b_verdict.c_str(),
          fallback ? 1 : 0, static_cast<long long>(Qa.Np) - Qa.status_hist_plane[0],
          static_cast<long long>(Qb.Np) - Qb.status_hist_plane[0]));
    out_line(fmt("TIMING r=%d seconds=%.3f", r, since(t_all)));
    return kExitOk;
}

// ================================================================================================
// Aggregate and driver
// ================================================================================================

double json_num(const json& j) { return j.is_null() ? std::numeric_limits<double>::quiet_NaN() : j.get<double>(); }

void write_ab_primary(const AbConfig& c) {
    const std::string path = c.out + "/ab_primary.csv";
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f)
        throw std::runtime_error("cannot write " + tmp);
    std::fprintf(f, "r,seed,medium,S_T,S_tau,V1,theta_2,theta_3,qbar\n");
    for (int r = 0; r < c.nr; ++r) {
        json j;
        {
            std::ifstream in(rdir_of(c, r) + "/primary.json");
            if (!in)
                throw std::runtime_error(fmt("missing primary.json of realization %d", r));
            in >> j;
        }
        for (const char* m : {"a", "b"}) {
            const json& p = j[m];
            std::fprintf(f, "%d,%llu,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", r,
                         c.seed0 + static_cast<unsigned long long>(r), m, json_num(p["S_T"]),
                         json_num(p["S_tau"]), json_num(p["V1"]), json_num(p["theta_2"]),
                         json_num(p["theta_3"]), json_num(p["qbar"]));
        }
    }
    std::fclose(f);
    fs::rename(tmp, path);
}

int run_ab(CudaContext& ctx, const AbConfig& c) {
    {   // RR12: before any tracking
        std::string what;
        if (ll::find_seed_on_face(
                [&] {
                    ll::InjectionSpec s;
                    s.X0 = c.X0;
                    s.M = c.np_side;
                    s.w2a = c.window[0];
                    s.w2b = c.window[1];
                    s.w3a = c.window[2];
                    s.w3b = c.window[3];
                    return s;
                }(),
                c.ny, &what)) {
            std::fprintf(stderr, "ERROR: seed-face assertion (RR12) failed before tracking: %s\n", what.c_str());
            out_line("SEED_FACE_FAILED " + what);
            return kExitSeedFace;
        }
    }
    out_line(fmt("RESOLVED grid=%dx%dx%d L1=%.10g lambda=%.10g X0=%.10g mg_levels_b=%d(%s) cg_rtol_b=%g "
                 "cg_max_iter_b=%d m_star=%lld planes=%s m=%s",
                 c.nx, c.ny, c.nz, c.L1, c.ell, c.X0, c.mg_levels_b, c.mg_levels_auto ? "auto" : "given",
                 c.cg_rtol_b, c.cg_max_iter_b, c.plane_m.back(), join_d(c.planes, "%.10g").c_str(),
                 [&] {
                     std::string s;
                     for (std::size_t i = 0; i < c.plane_m.size(); ++i)
                         s += (i ? "," : "") + std::to_string(c.plane_m[i]);
                     return s;
                 }()
                     .c_str()));

    fs::create_directories(c.out + "/realizations");
    // restart: refuse a DONE realization of other options before touching anything
    for (int r = 0; r < c.nr; ++r) {
        const std::string d = rdir_of(c, r);
        if (fs::exists(d + "/DONE") && slurp(d + "/DONE") != done_text(c, r)) {
            std::fprintf(stderr,
                         "ERROR: %s/DONE was written under different options; refusing to mix (use a fresh --out)\n",
                         d.c_str());
            return kExitMismatch;
        }
    }
    {   // delete stale temporaries and unfinished realization directories
        std::vector<fs::path> stale;
        for (const auto& e : fs::directory_iterator(c.out + "/realizations"))
            if (e.path().extension() == ".tmp")
                stale.push_back(e.path());
        for (const fs::path& p : stale)
            fs::remove_all(p);
        for (int r = 0; r < c.nr; ++r) {
            const std::string d = rdir_of(c, r);
            if (fs::exists(d) && !fs::exists(d + "/DONE"))
                fs::remove_all(d);
        }
        fs::remove(c.out + "/ab_primary.csv.tmp");
    }
    {   // manifest (deterministic: no --out, no wall-clock, no commit)
        std::vector<unsigned long long> seeds;
        for (int r = 0; r < c.nr; ++r)
            seeds.push_back(c.seed0 + static_cast<unsigned long long>(r));
        std::vector<long long> pm(c.plane_m.begin(), c.plane_m.end());
        const std::string tmp = c.out + "/ab_manifest.json.tmp";
        write_json(tmp, {{"tool", "lagrangian_medium_ab"},
                         {"command_line_without_out", "lagrangian_medium_ab " + c.command_line},
                         {"grid", {c.nx, c.ny, c.nz}},
                         {"L1", c.L1},
                         {"lambda", c.ell},
                         {"ell", c.ell},
                         {"eps", c.eps},
                         {"sigma2", c.sigma2},
                         {"seed0", c.seed0},
                         {"nr", c.nr},
                         {"seeds", seeds},
                         {"np_side", c.np_side},
                         {"window", {c.window[0], c.window[1], c.window[2], c.window[3]}},
                         {"X0", c.X0},
                         {"plane_m", pm},
                         {"planes", c.planes},
                         {"mg_levels_b", c.mg_levels_b},
                         {"cg_rtol_b", c.cg_rtol_b},
                         {"cg_max_iter_b", c.cg_max_iter_b},
                         {"signature", signature(c)}});
        fs::rename(tmp, c.out + "/ab_manifest.json");
    }
    for (int r = 0; r < c.nr; ++r) {
        if (fs::exists(rdir_of(c, r) + "/DONE")) {
            out_line(fmt("R=%d SKIP done (DONE present)", r));
            continue;
        }
        const int rc = run_realization(ctx, c, r);
        if (rc != kExitOk)
            return rc;
    }
    write_ab_primary(c);
    out_line(fmt("RUN_DONE nr=%d", c.nr));
    return kExitOk;
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--help") {
            std::fputs(ab_usage().c_str(), stdout);
            return kExitOk;
        }
    AbConfig cfg;
    try {
        cfg = parse_args(argc, argv);
    } catch (const ll::UsageError& e) {
        std::fprintf(stderr, "lagrangian_medium_ab: usage error: %s\n(see --help)\n", e.what());
        return kExitUsage;
    }
    try {
        CudaContext ctx(0);
        return run_ab(ctx, cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lagrangian_medium_ab: ERROR: %s\n", e.what());
        return kExitException;
    }
}
