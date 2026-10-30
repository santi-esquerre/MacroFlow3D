/**
 * @file par2_sf42_options_tests.cu
 * @brief SF-42 N1: contract tests of the opt-in options of the Par2 path of `macroflow3d_pipeline`
 *        (SF-18 medium (b) conductivity, transverse injection window, plane-crossing statistics,
 *        Y/K field dump).
 *
 * Usage: par2_sf42_options_tests <macroflow3d_pipeline exe>      (fast tier, well under 60 s)
 *
 * A. Config validation (in process): every inconsistent combination of the new keys is rejected
 *    with a message naming the key; the YAML parser rejects unknown sub-keys, missing sub-keys and
 *    wrong list shapes.
 * B. Pipeline run, 32x16x16, dx = 1/16 (`%.17g`), sf18 source + window + crossings + dump, NR = 2:
 *    - Y of the dump equals, bit for bit, a direct call of the SF-18 generator with the consumer's
 *      configuration (`DirichletLongBoxStage` phase F: sigma2, corr_length = ell, seed = seed0 + r,
 *      normalize_variance), K equals std::exp(eps * Y) bit for bit (the consumer's phase S1
 *      expression), var Y = sigma2 to 1e-12 relative (biased, mean subtracted);
 *    - every injected particle lies inside the window (injection dump);
 *    - the plane-crossing CSV equals an INDEPENDENT brute-force reference to 1e-12 relative. The
 *      reference never touches the device crossing code: it reads the full-precision (fixed, 20
 *      decimals) unwrapped positions of EVERY step from the pipeline's own snapshot output
 *      (`analysis.snapshots.every = 1`, `include_unwrapped`) plus the injection dump, scans every
 *      particle's piecewise-linear trajectory for the first crossing of each plane, and computes
 *      the moments in long double (two-pass, biased). The case crosses several planes in one step
 *      and wraps transverse coordinates periodically (both asserted from the data).
 * C. A run with zero crossings (nan rows), default K source and full-plane injection.
 *
 * Tolerances (stated before the first run): counts exact; values 1e-12 relative to max(|ref|, floor)
 * where floor is 0 for t, var, mean x2/x3, and the natural scale sqrt(var) (resp. sqrt(var_a var_b)
 * for covariances) for the mean displacements and covariances, which can be near 0.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/io/config/Config.hpp"
#include "src/io/config/ConfigDefaults.hpp"
#include "src/io/config/ConfigValidator.hpp"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/physics/streamfunctions/inlet_slab/NpyIo.hpp"
#include "src/runtime/CudaContext.cuh"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace fs = std::filesystem;
namespace npy = macroflow3d::streamfunctions::inlet_slab;

namespace {

int g_failed = 0;
int g_checks = 0;

void check(const std::string& name, bool ok, const std::string& detail = "") {
    ++g_checks;
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

// ============================================================================================
// A. config validation
// ============================================================================================

io::AppConfig valid_sf18_config() {
    io::AppConfig c = io::make_default_config();
    c.grid.nx = 32;
    c.grid.ny = 16;
    c.grid.nz = 16;
    c.grid.dx = 1.0 / 16.0;
    c.stochastic.corr_length = 0.25;
    c.stochastic.field_source = "sf18_medium_b";
    c.stochastic.sf18_medium_b.present = true;
    c.stochastic.sf18_medium_b.sigma2 = 1.0;
    c.stochastic.sf18_medium_b.eps = 1.0;
    c.stochastic.sf18_medium_b.seed0 = 5001;
    c.transport.inject_x = 0.25;
    return c;
}

/// True iff validate_config rejects `c` with an error mentioning `token`.
bool rejects(const io::AppConfig& c, const std::string& token) {
    const io::ValidationResult r = io::validate_config(c);
    for (const std::string& e : r.errors)
        if (e.find(token) != std::string::npos)
            return true;
    return false;
}

void case_a_validation() {
    {
        const io::AppConfig c = valid_sf18_config();
        const io::ValidationResult r = io::validate_config(c);
        check("A.valid_sf18_config_accepted", r.ok());
    }
    {
        io::AppConfig c = valid_sf18_config();
        c.stochastic.sf18_medium_b.present = false;
        check("A.field_source_without_block", rejects(c, "stochastic.sf18_medium_b"));
    }
    {
        io::AppConfig c = valid_sf18_config();
        c.stochastic.field_source = "";
        check("A.block_without_field_source", rejects(c, "stochastic.sf18_medium_b"));
    }
    {
        io::AppConfig c = valid_sf18_config();
        c.stochastic.field_source = "random_spectral";
        check("A.unknown_field_source", rejects(c, "stochastic.field_source"));
    }
    for (const double e : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
        io::AppConfig c = valid_sf18_config();
        c.stochastic.sf18_medium_b.eps = e;
        check(strf("A.eps_%g_rejected", e), rejects(c, "stochastic.sf18_medium_b.eps"));
    }
    for (const double s : {-0.5, std::numeric_limits<double>::quiet_NaN()}) {
        io::AppConfig c = valid_sf18_config();
        c.stochastic.sf18_medium_b.sigma2 = s;
        check(strf("A.sigma2_%g_rejected", s), rejects(c, "stochastic.sf18_medium_b.sigma2"));
    }
    {
        io::AppConfig c = valid_sf18_config();
        c.stochastic.K_mean = 2.0;
        check("A.K_mean_not_one_rejected", rejects(c, "stochastic.K_mean"));
    }
    {
        io::AppConfig c = valid_sf18_config();
        c.grid.nx = 31;
        check("A.odd_extent_rejected", rejects(c, "grid"));
    }
    // K_mean != 1 stays legal for the default source (no behaviour change)
    {
        io::AppConfig c = io::make_default_config();
        c.stochastic.K_mean = 2.0;
        check("A.default_source_K_mean_unchanged", io::validate_config(c).ok());
    }

    // ---- injection window ----
    {
        io::AppConfig c = valid_sf18_config();
        c.transport.inject_window.enabled = true;
        c.transport.inject_window.w = {0.1, 0.9, 0.2, 0.8};
        check("A.window_valid_accepted", io::validate_config(c).ok());
        const double bad[][4] = {{-0.1, 0.9, 0.2, 0.8}, {0.1, 1.1, 0.2, 0.8}, {0.5, 0.5, 0.2, 0.8},
                                 {0.6, 0.5, 0.2, 0.8},  {0.1, 0.9, -0.2, 0.8}, {0.1, 0.9, 0.2, 1.2},
                                 {0.1, 0.9, 0.8, 0.8},  {0.1, 0.9, 0.9, 0.2}};
        for (int i = 0; i < 8; ++i) {
            io::AppConfig d = c;
            d.transport.inject_window.w = {bad[i][0], bad[i][1], bad[i][2], bad[i][3]};
            check(strf("A.window_bad_%d_rejected", i), rejects(d, "transport.inject_window"));
        }
        io::AppConfig d = c;
        d.transport.inject_window.w[2] = std::numeric_limits<double>::quiet_NaN();
        check("A.window_nan_rejected", rejects(d, "transport.inject_window"));
        // the window is independent of the SF-18 source
        io::AppConfig e = io::make_default_config();
        e.transport.inject_window = c.transport.inject_window;
        check("A.window_with_default_source_accepted", io::validate_config(e).ok());
    }

    // ---- plane crossings ----
    {
        io::AppConfig c = valid_sf18_config();
        c.analysis.plane_crossings.enabled = true;
        c.analysis.plane_crossings.planes = {0.5, 1.0, 1.5};
        check("A.planes_valid_accepted", io::validate_config(c).ok());
        const std::vector<std::vector<double>> bad = {
            {},                                 // empty
            {1.0, 0.5},                         // decreasing
            {0.5, 0.5},                         // not strictly increasing
            {0.25, 1.0},                        // == inject_x
            {0.1, 1.0},                         // < inject_x
            {0.5, 2.0},                         // == Lx (32 / 16)
            {0.5, 2.5},                         // > Lx
            {0.5, std::numeric_limits<double>::quiet_NaN()}};
        for (size_t i = 0; i < bad.size(); ++i) {
            io::AppConfig d = c;
            d.analysis.plane_crossings.planes = bad[i];
            check(strf("A.planes_bad_%zu_rejected", i),
                  rejects(d, "analysis.plane_crossings.planes"));
        }
        io::AppConfig d = c;
        d.flow.bc.xmin = BCFace(BCType::Periodic, 0.0);
        d.flow.bc.xmax = BCFace(BCType::Periodic, 0.0);
        check("A.planes_with_periodic_x_rejected", rejects(d, "analysis.plane_crossings"));
        // disabled block: planes are not validated and nothing is required
        io::AppConfig e = valid_sf18_config();
        e.analysis.plane_crossings.enabled = false;
        e.analysis.plane_crossings.planes = {};
        check("A.planes_disabled_not_validated", io::validate_config(e).ok());
    }

    // ---- field dump ----
    {
        io::AppConfig c = valid_sf18_config();
        c.output.dump_field.enabled = true;
        c.output.dump_field.realizations = {0};
        check("A.dump_valid_accepted", io::validate_config(c).ok());
        io::AppConfig d = c;
        d.stochastic.field_source = "";
        d.stochastic.sf18_medium_b.present = false;
        check("A.dump_without_sf18_rejected", rejects(d, "output.dump_field"));
        d = c;
        d.output.dump_field.realizations = {};
        check("A.dump_empty_realizations_rejected", rejects(d, "output.dump_field.realizations"));
        d = c;
        d.output.dump_field.realizations = {1}; // NR = 1 (macrodispersion disabled)
        check("A.dump_realization_out_of_range_rejected",
              rejects(d, "output.dump_field.realizations"));
        d = c;
        d.output.dump_field.realizations = {-1};
        check("A.dump_negative_realization_rejected", rejects(d, "output.dump_field.realizations"));
        d = c;
        d.analysis.macrodispersion.enabled = true;
        d.analysis.macrodispersion.NR = 3;
        d.output.dump_field.realizations = {0, 2, 2};
        check("A.dump_duplicate_rejected", rejects(d, "output.dump_field.realizations"));
        d.output.dump_field.realizations = {0, 2};
        check("A.dump_two_of_three_accepted", io::validate_config(d).ok());
    }
}

// ---- YAML parser ----

std::string yaml_text(const std::string& stoch_extra, const std::string& transport_extra,
                      const std::string& analysis_extra, const std::string& output_extra) {
    return "run_mode: ensemble\n"
           "grid: {nx: 32, ny: 16, nz: 16, dx: 0.0625}\n"
           "stochastic:\n  sigma2: 1.0\n  corr_length: 0.25\n" + stoch_extra +
           "transport:\n  n_particles: 10\n  inject_x: 0.25\n" + transport_extra +
           "analysis:\n  macrodispersion: {enabled: true, NR: 2}\n" + analysis_extra +
           "output:\n  output_dir: ./unused\n" + output_extra;
}

const std::string kSf18Block =
    "  field_source: sf18_medium_b\n  sf18_medium_b: {sigma2: 1.5, eps: 0.5, seed0: 7001}\n";

/// load_config_yaml error text, "" if it loaded.
std::string load_error(const fs::path& dir, const std::string& name, const std::string& text) {
    const fs::path p = dir / name;
    std::ofstream(p) << text;
    try {
        (void)io::load_config_yaml(p.string());
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

void case_a_yaml(const fs::path& dir) {
    // positive parse
    {
        const fs::path p = dir / "ok.yaml";
        std::ofstream(p) << yaml_text(
            kSf18Block, "  inject_window: [0.1, 0.9, 0.2, 0.8]\n",
            "  plane_crossings: {enabled: true, planes: [0.5, 1.0, 1.5], dump_injection: true}\n",
            "  dump_field: {enabled: true, realizations: [0, 1]}\n");
        const io::AppConfig c = io::load_config_yaml(p.string());
        check("A.yaml_parse_sf18",
              c.stochastic.field_source == "sf18_medium_b" && c.stochastic.sf18_medium_b.present &&
                  c.stochastic.sf18_medium_b.sigma2 == 1.5 && c.stochastic.sf18_medium_b.eps == 0.5 &&
                  c.stochastic.sf18_medium_b.seed0 == 7001);
        check("A.yaml_parse_window",
              c.transport.inject_window.enabled && c.transport.inject_window.w[0] == 0.1 &&
                  c.transport.inject_window.w[3] == 0.8);
        check("A.yaml_parse_planes",
              c.analysis.plane_crossings.enabled && c.analysis.plane_crossings.planes.size() == 3 &&
                  c.analysis.plane_crossings.planes[2] == 1.5 &&
                  c.analysis.plane_crossings.dump_injection);
        check("A.yaml_parse_dump",
              c.output.dump_field.enabled && c.output.dump_field.realizations.size() == 2 &&
                  c.output.dump_field.realizations[1] == 1);
        check("A.yaml_parsed_config_validates", io::validate_config(c).ok());
        // absent keys leave the defaults untouched
        const fs::path q = dir / "plain.yaml";
        std::ofstream(q) << yaml_text("", "", "", "");
        const io::AppConfig d = io::load_config_yaml(q.string());
        check("A.yaml_defaults_off",
              d.stochastic.field_source.empty() && !d.stochastic.sf18_medium_b.present &&
                  !d.transport.inject_window.enabled && !d.analysis.plane_crossings.enabled &&
                  !d.output.dump_field.enabled);
    }
    struct Bad {
        const char* name;
        std::string text;
        const char* token;
    };
    const std::vector<Bad> bad = {
        {"unk_sf18", yaml_text("  field_source: sf18_medium_b\n  sf18_medium_b: {sigma2: 1, eps: 1, "
                               "seed0: 1, ell: 2}\n", "", "", ""),
         "stochastic.sf18_medium_b.ell"},
        {"missing_seed0", yaml_text("  field_source: sf18_medium_b\n  sf18_medium_b: {sigma2: 1, "
                                    "eps: 1}\n", "", "", ""),
         "stochastic.sf18_medium_b.seed0"},
        {"missing_eps", yaml_text("  field_source: sf18_medium_b\n  sf18_medium_b: {sigma2: 1, "
                                  "seed0: 1}\n", "", "", ""),
         "stochastic.sf18_medium_b.eps"},
        {"window_len3", yaml_text("", "  inject_window: [0.1, 0.9, 0.2]\n", "", ""),
         "transport.inject_window"},
        {"window_len5", yaml_text("", "  inject_window: [0.1, 0.9, 0.2, 0.8, 0.5]\n", "", ""),
         "transport.inject_window"},
        {"window_scalar", yaml_text("", "  inject_window: 0.5\n", "", ""),
         "transport.inject_window"},
        {"unk_planes", yaml_text("", "", "  plane_crossings: {enabled: true, planes: [1.0], "
                                          "every: 2}\n", ""),
         "analysis.plane_crossings.every"},
        {"planes_scalar", yaml_text("", "", "  plane_crossings: {enabled: true, planes: 1.0}\n", ""),
         "analysis.plane_crossings.planes"},
        {"unk_dump", yaml_text("", "", "", "  dump_field: {enabled: true, realizations: [0], "
                                           "fmt: npy}\n"),
         "output.dump_field.fmt"},
        {"dump_realizations_scalar", yaml_text("", "", "", "  dump_field: {enabled: true, "
                                                           "realizations: 0}\n"),
         "output.dump_field.realizations"},
        {"unk_stoch_source", yaml_text("  fieldsource: sf18_medium_b\n", "", "", ""),
         "stochastic.fieldsource"},
    };
    for (const Bad& b : bad) {
        const std::string e = load_error(dir, std::string(b.name) + ".yaml", b.text);
        check(std::string("A.yaml_reject_") + b.name, e.find(b.token) != std::string::npos,
              e.empty() ? "loaded without error" : "");
    }
}

// ============================================================================================
// B/C. pipeline runs
// ============================================================================================

constexpr int kNx = 32, kNy = 16, kNz = 16;
constexpr double kH = 1.0 / 16.0;       // = 1.0 / ny: the consumer's cell size
constexpr double kEll = 0.25;
constexpr double kSigma2 = 1.5;
constexpr double kEps = 1.2;
constexpr unsigned long long kSeed0 = 5001;
constexpr int kNp = 400;
constexpr double kDt = 0.05;
constexpr int kSteps = 30;
constexpr double kInjectX = 0.25;
constexpr double kW[4] = {0.85, 1.0, 0.0, 0.2}; // near the periodic faces: forces wrapping
const std::vector<double> kPlanes = {0.4, 0.42, 0.44, 0.46, 0.9, 1.3, 1.7, 1.95};

struct RunSpec {
    fs::path out;
    bool sf18 = true;
    bool window = true;
    bool dump = true;
    bool snapshots = true;
    int nr = 2;
    int steps = kSteps;
    int sample_every = 10;
    std::vector<double> planes = kPlanes;
};

std::string write_config(const fs::path& path, const RunSpec& s) {
    std::ostringstream y;
    y << "run_mode: ensemble\n";
    y << "grid: {nx: " << kNx << ", ny: " << kNy << ", nz: " << kNz << ", dx: " << strf("%.17g", kH)
      << "}\n";
    y << "stochastic:\n  sigma2: 1.0\n  corr_length: " << strf("%.17g", kEll)
      << "\n  n_modes: 1000\n  covariance_type: 1\n  seed: 42\n  K_mean: 1.0\n";
    if (s.sf18)
        y << "  field_source: sf18_medium_b\n  sf18_medium_b: {sigma2: " << strf("%.17g", kSigma2)
          << ", eps: " << strf("%.17g", kEps) << ", seed0: " << kSeed0 << "}\n";
    y << "flow:\n  solver: mg_cg\n  mg_levels: 3\n  mg_pre_smooth: 4\n  mg_post_smooth: 4\n"
         "  mg_coarse_iters: 50\n  mg_max_cycles: 20\n  cg_max_iter: 500\n  cg_rtol: 1.0e-10\n"
         "  cg_check_every: 10\n  rtol: 1.0e-6\n  pin: {mode: off}\n  bc:\n"
         "    west:   { type: dirichlet, value: 2.0 }\n    east:   { type: dirichlet, value: 0.0 }\n"
         "    south:  { type: periodic }\n    north:  { type: periodic }\n"
         "    bottom: { type: periodic }\n    top:    { type: periodic }\n";
    y << "transport:\n  n_particles: " << kNp << "\n  dt: " << strf("%.17g", kDt)
      << "\n  n_steps: " << s.steps << "\n  seed: 777\n  output_every: 1000\n  inject_x: "
      << strf("%.17g", kInjectX) << "\n";
    if (s.window)
        y << "  inject_window: [" << strf("%.17g", kW[0]) << ", " << strf("%.17g", kW[1]) << ", "
          << strf("%.17g", kW[2]) << ", " << strf("%.17g", kW[3]) << "]\n";
    y << "analysis:\n  macrodispersion: {enabled: true, NR: " << s.nr
      << ", lambda: 0.25, vmean_norm: 1.0, sample_every: " << s.sample_every
      << ", var_estimator: biased}\n";
    if (s.snapshots)
        y << "  snapshots: {enabled: true, every: 1, legacy_format: false, include_unwrapped: "
             "true, include_wrap_counts: true, precision: 20}\n";
    y << "  plane_crossings:\n    enabled: true\n    dump_injection: true\n    planes: [";
    for (size_t k = 0; k < s.planes.size(); ++k)
        y << (k ? ", " : "") << strf("%.17g", s.planes[k]);
    y << "]\n";
    y << "output:\n  output_dir: " << s.out.string() << "\n  save_K: false\n";
    if (s.dump)
        y << "  dump_field:\n    enabled: true\n    realizations: [0, 1]\n";
    std::ofstream(path) << y.str();
    return y.str();
}

int run_pipeline(const std::string& exe, const fs::path& cfg, const fs::path& log) {
    const std::string cmd = exe + " " + cfg.string() + " > " + log.string() + " 2>&1";
    const int st = std::system(cmd.c_str());
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> v;
    std::string cur;
    for (char ch : s) {
        if (ch == sep) {
            v.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    v.push_back(cur);
    return v;
}

/// CSV with one header line (after optional '#' comment lines): column name -> values.
struct Csv {
    std::vector<std::string> comments;
    std::vector<std::string> header;
    std::vector<std::vector<double>> rows;
    bool ok = false;
};

Csv read_csv(const fs::path& p) {
    Csv c;
    std::ifstream f(p);
    if (!f)
        return c;
    std::string line;
    bool have_header = false;
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        if (!have_header && line[0] == '#') {
            c.comments.push_back(line);
            continue;
        }
        if (!have_header) {
            c.header = split(line, ',');
            have_header = true;
            continue;
        }
        std::vector<double> row;
        for (const std::string& t : split(line, ','))
            row.push_back(std::strtod(t.c_str(), nullptr));
        c.rows.push_back(row);
    }
    c.ok = have_header;
    return c;
}

int col(const Csv& c, const std::string& name) {
    for (size_t i = 0; i < c.header.size(); ++i)
        if (c.header[i] == name)
            return static_cast<int>(i);
    return -1;
}


/// Direct SF-18 call with the consumer's configuration (DirichletLongBoxStage phase F).
std::vector<double> consumer_Y(CudaContext& ctx, unsigned long long seed) {
    const Grid3D grid(kNx, kNy, kNz, kH, kH, kH);
    physics::PeriodicGaussianFieldConfig cfg;
    cfg.sigma2 = kSigma2;
    cfg.corr_length = kEll;
    cfg.seed = seed;
    cfg.normalize_variance = true;
    DeviceBuffer<real> y(grid.num_cells());
    physics::PeriodicGaussianFieldWorkspace ws;
    physics::generate_periodic_gaussian_field(ctx, grid, cfg, y.span(), ws);
    ctx.synchronize();
    std::vector<double> h(grid.num_cells());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), y.data(), h.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost));
    return h;
}

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

// ---- brute-force reference ----

struct Traj {
    // [step][particle]; step 0 = injection
    std::vector<std::vector<double>> x, y, z;
    std::vector<std::vector<char>> wrapped; // transverse wrap counter != 0 at that step
    long long wrapped_rows = 0;
    bool ok = false;
};

Traj read_trajectories(const fs::path& out, const Csv& inj, int steps) {
    Traj t;
    t.x.assign(steps + 1, std::vector<double>(kNp));
    t.y = t.x;
    t.z = t.x;
    t.wrapped.assign(steps + 1, std::vector<char>(kNp, 0));
    const int ci = col(inj, "x1"), cy = col(inj, "x2"), cz = col(inj, "x3");
    if (ci < 0 || cy < 0 || cz < 0 || static_cast<int>(inj.rows.size()) != kNp)
        return t;
    for (int i = 0; i < kNp; ++i) {
        t.x[0][i] = inj.rows[i][ci];
        t.y[0][i] = inj.rows[i][cy];
        t.z[0][i] = inj.rows[i][cz];
    }
    for (int s = 1; s <= steps; ++s) {
        const Csv c = read_csv(out / "snapshots" / "r_0000" / strf("step_%08d.csv", s));
        const int cx = col(c, "xu"), cyu = col(c, "yu"), czu = col(c, "zu"), wy = col(c, "wrapY"),
                  wz = col(c, "wrapZ");
        if (!c.ok || cx < 0 || cyu < 0 || czu < 0 || wy < 0 || wz < 0 ||
            static_cast<int>(c.rows.size()) != kNp)
            return t;
        for (int i = 0; i < kNp; ++i) {
            t.x[s][i] = c.rows[i][cx];
            t.y[s][i] = c.rows[i][cyu];
            t.z[s][i] = c.rows[i][czu];
            if (c.rows[i][wy] != 0 || c.rows[i][wz] != 0) {
                ++t.wrapped_rows;
                t.wrapped[s][i] = 1;
            }
        }
    }
    t.ok = true;
    return t;
}

struct RefRow {
    long long n_crossed = 0;
    long double v[14]; // mean_t var_t mean_x2 mean_x3 var_x2 var_x3 cov_x2x3 mean_dx2 ...
};

void two_pass(const std::vector<long double>& a, const std::vector<long double>& b, long double& ma,
              long double& mb, long double& va, long double& vb, long double& cab) {
    const long double n = static_cast<long double>(a.size());
    ma = mb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;
    va = vb = cab = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        va += (a[i] - ma) * (a[i] - ma);
        vb += (b[i] - mb) * (b[i] - mb);
        cab += (a[i] - ma) * (b[i] - mb);
    }
    va /= n;
    vb /= n;
    cab /= n;
}

RefRow reference_plane(const Traj& T, int steps, double dt, double X, int* multi_cross_steps,
                       const std::vector<double>& planes, long long* wrapped_crossings) {
    RefRow r;
    std::vector<long double> tt, y2, y3, d2, d3;
    for (int i = 0; i < kNp; ++i) {
        for (int n = 0; n < steps; ++n) {
            if (T.x[n][i] < X && X <= T.x[n + 1][i]) {
                const double f = (X - T.x[n][i]) / (T.x[n + 1][i] - T.x[n][i]);
                const double tc = static_cast<double>(n) * dt + f * dt;
                const double yc = T.y[n][i] + f * (T.y[n + 1][i] - T.y[n][i]);
                const double zc = T.z[n][i] + f * (T.z[n + 1][i] - T.z[n][i]);
                if (wrapped_crossings && T.wrapped[n + 1][i])
                    ++*wrapped_crossings;
                tt.push_back(tc);
                y2.push_back(yc);
                y3.push_back(zc);
                d2.push_back(yc - T.y[0][i]);
                d3.push_back(zc - T.z[0][i]);
                if (multi_cross_steps) {
                    int cnt = 0;
                    for (double Xk : planes)
                        cnt += (T.x[n][i] < Xk && Xk <= T.x[n + 1][i]);
                    if (cnt >= 2)
                        ++*multi_cross_steps;
                }
                break;
            }
        }
    }
    r.n_crossed = static_cast<long long>(tt.size());
    const long double nan = std::numeric_limits<long double>::quiet_NaN();
    if (tt.empty()) {
        for (long double& v : r.v)
            v = nan;
        return r;
    }
    long double dummy1, dummy2, dummy3;
    std::vector<long double> zero(tt.size(), 0.0L);
    two_pass(tt, zero, r.v[0], dummy1, r.v[1], dummy2, dummy3);
    two_pass(y2, y3, r.v[2], r.v[3], r.v[4], r.v[5], r.v[6]);
    two_pass(d2, d3, r.v[7], r.v[8], r.v[9], r.v[10], r.v[11]);
    return r;
}

bool near(double got, long double ref, double floor_scale) {
    if (std::isnan(got) || std::isnan(static_cast<double>(ref)))
        return std::isnan(got) && std::isnan(static_cast<double>(ref));
    const long double tol = 1e-12L * std::max<long double>(std::fabs(ref), floor_scale);
    return std::fabs(static_cast<long double>(got) - ref) <= tol;
}

void case_b(const std::string& exe, const fs::path& base, CudaContext& ctx) {
    RunSpec s;
    s.out = base / "B";
    const fs::path cfgp = base / "B.yaml";
    write_config(cfgp, s);
    const int rc = run_pipeline(exe, cfgp, base / "B.log");
    check("B.pipeline_exit_0", rc == 0, strf("exit %d (log %s)", rc, (base / "B.log").c_str()));
    if (rc != 0)
        return;

    // ---- effective config / manifest record the new keys ----
    {
        std::ifstream f(s.out / "effective_config.yaml");
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string e = ss.str();
        check("B.effective_config_records_options",
              e.find("sf18_medium_b") != std::string::npos &&
                  e.find("inject_window") != std::string::npos &&
                  e.find("plane_crossings") != std::string::npos &&
                  e.find("dump_field") != std::string::npos);
        std::ifstream m(s.out / "manifest.json");
        std::stringstream ms;
        ms << m.rdbuf();
        check("B.manifest_records_options", ms.str().find("sf18_medium_b") != std::string::npos &&
                                                 ms.str().find("plane_crossings") != std::string::npos);
        // exact (%.17g) values of the options in the effective config; the file as a whole is not
        // reloadable by the strict parser (pre-existing: it lists e.g. transport.velocity_layout)
        check("B.effective_config_exact_values",
              e.find(strf("    sigma2: %.17g\n    eps: %.17g\n    seed0: %llu\n", kSigma2, kEps,
                          kSeed0)) != std::string::npos &&
                  e.find(strf("  inject_window: [%.17g, %.17g, %.17g, %.17g]\n", kW[0], kW[1], kW[2],
                              kW[3])) != std::string::npos &&
                  e.find("    planes: [0.40000000000000002, 0.41999999999999998") !=
                      std::string::npos &&
                  e.find("    realizations: [0, 1]\n") != std::string::npos);
    }

    // ---- field identity (bitwise) ----
    const size_t n = static_cast<size_t>(kNx) * kNy * kNz;
    for (int r = 0; r < 2; ++r) {
        const fs::path dir = s.out / "fields" / strf("r_%04d", r);
        npy::NpyArray Yd, Kd;
        try {
            Yd = npy::read_npy_shape((dir / "Y.npy").string(), {(size_t)kNz, (size_t)kNy, (size_t)kNx});
            Kd = npy::read_npy_shape((dir / "K.npy").string(), {(size_t)kNz, (size_t)kNy, (size_t)kNx});
        } catch (const std::exception& ex) {
            check(strf("B.dump_files_r%d", r), false, ex.what());
            continue;
        }
        const std::vector<double> Yref = consumer_Y(ctx, kSeed0 + r);
        check(strf("B.Y_bitwise_equal_consumer_r%d", r), bitwise_equal(Yd.data, Yref));
        std::vector<double> Kref(n);
        for (size_t c = 0; c < n; ++c) {
            const double lnK = kEps * Yref[c];
            Kref[c] = std::exp(lnK);
        }
        check(strf("B.K_bitwise_equal_exp_eps_Y_r%d", r), bitwise_equal(Kd.data, Kref));
        long double mean = 0, var = 0;
        for (double v : Yd.data)
            mean += v;
        mean /= n;
        for (double v : Yd.data)
            var += (v - mean) * (v - mean);
        var /= n;
        check(strf("B.var_Y_equals_sigma2_r%d", r), std::fabs(var - kSigma2) <= 1e-12 * kSigma2,
              strf("var=%.17Lg", var));
    }
    {
        const npy::NpyArray a = npy::read_npy((s.out / "fields/r_0000/Y.npy").string());
        const npy::NpyArray b = npy::read_npy((s.out / "fields/r_0001/Y.npy").string());
        check("B.realizations_use_seed0_plus_r", !bitwise_equal(a.data, b.data));
        check("B.no_dump_for_unlisted_realization", !fs::exists(s.out / "fields" / "r_0002"));
    }

    // ---- injection inside the window ----
    const Csv inj = read_csv(s.out / "crossings" / "realization_0000_injection.csv");
    const double Ly = kNy * kH, Lz = kNz * kH;
    {
        const int cy = col(inj, "x2"), cz = col(inj, "x3"), cx = col(inj, "x1");
        bool all_in = inj.ok && static_cast<int>(inj.rows.size()) == kNp;
        double y_min = 1e300, y_max = -1e300, z_min = 1e300, z_max = -1e300;
        const double tol = 4e-16;
        for (const auto& row : inj.rows) {
            all_in = all_in && row[cx] == kInjectX;
            all_in = all_in && row[cy] >= kW[0] * Ly - tol && row[cy] <= kW[1] * Ly + tol;
            all_in = all_in && row[cz] >= kW[2] * Lz - tol && row[cz] <= kW[3] * Lz + tol;
            y_min = std::min(y_min, row[cy]);
            y_max = std::max(y_max, row[cy]);
            z_min = std::min(z_min, row[cz]);
            z_max = std::max(z_max, row[cz]);
        }
        check("B.injection_inside_window", all_in,
              strf("y in [%.4f, %.4f], z in [%.4f, %.4f]", y_min, y_max, z_min, z_max));
        check("B.injection_fills_window",
              (y_max - y_min) > 0.8 * (kW[1] - kW[0]) * Ly && (z_max - z_min) > 0.8 * (kW[3] - kW[2]) * Lz);
    }

    // ---- crossings against the brute-force reference ----
    const Csv cr = read_csv(s.out / "crossings" / "realization_0000_planes.csv");
    check("B.crossings_header",
          cr.ok && cr.comments.size() == 1 &&
              cr.comments[0] == "# format_version=1 sf42_plane_crossings weights=uniform "
                                "time_origin=injection" &&
              cr.header == split("x1,n_injected,n_crossed,mean_t,var_t,mean_x2,mean_x3,var_x2,var_x3,"
                                 "cov_x2x3,mean_dx2,mean_dx3,var_dx2,var_dx3,cov_dx2dx3", ','));
    check("B.crossings_file_per_realization",
          fs::exists(s.out / "crossings" / "realization_0001_planes.csv") &&
              !fs::exists(s.out / "crossings" / "realization_0002_planes.csv"));
    const Traj T = read_trajectories(s.out, inj, kSteps);
    check("B.reference_trajectories_read", T.ok);
    check("B.periodic_wrapping_exercised", T.ok && T.wrapped_rows > 0,
          strf("%lld wrapped rows", T.wrapped_rows));
    if (!T.ok || !cr.ok || cr.rows.size() != kPlanes.size())
        return;
    int multi = 0;
    long long wrapped_crossings = 0;
    bool all_ok = true;
    long long partial = 0;
    for (size_t k = 0; k < kPlanes.size(); ++k) {
        const RefRow ref = reference_plane(T, kSteps, kDt, kPlanes[k], k == 0 ? &multi : nullptr,
                                           kPlanes, &wrapped_crossings);
        const auto& row = cr.rows[k];
        bool ok = row[col(cr, "x1")] == kPlanes[k] && row[col(cr, "n_injected")] == kNp &&
                  static_cast<long long>(row[col(cr, "n_crossed")]) == ref.n_crossed;
        if (ref.n_crossed > 0 && ref.n_crossed < kNp)
            ++partial;
        const char* names[12] = {"mean_t",  "var_t",    "mean_x2",  "mean_x3",  "var_x2",    "var_x3",
                                 "cov_x2x3", "mean_dx2", "mean_dx3", "var_dx2",  "var_dx3",   "cov_dx2dx3"};
        // reference slots: v[0..1] = mean_t,var_t ; v[2..6] = x2,x3 means/vars/cov ; v[7..11] = dx
        const int slot[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        for (int q = 0; q < 12 && ref.n_crossed > 0; ++q) {
            double floor_scale = 0.0;
            if (q == 7)
                floor_scale = std::sqrt(static_cast<double>(ref.v[9]));
            if (q == 8)
                floor_scale = std::sqrt(static_cast<double>(ref.v[10]));
            if (q == 6)
                floor_scale = std::sqrt(static_cast<double>(ref.v[4] * ref.v[5]));
            if (q == 11)
                floor_scale = std::sqrt(static_cast<double>(ref.v[9] * ref.v[10]));
            const double got = row[col(cr, names[q])];
            const bool good = near(got, ref.v[slot[q]], floor_scale);
            if (!good)
                std::printf("   MISMATCH plane %.3f %s: got %.17g ref %.17Lg\n", kPlanes[k], names[q],
                            got, ref.v[slot[q]]);
            ok = ok && good;
        }
        all_ok = all_ok && ok;
        std::printf("   plane %.3f: n_crossed %lld / %d\n", kPlanes[k], ref.n_crossed, kNp);
    }
    check("B.crossings_equal_bruteforce_1e-12", all_ok);
    check("B.crossings_of_wrapped_particles", wrapped_crossings > 0,
          strf("%lld first crossings with a nonzero wrap counter", wrapped_crossings));
    check("B.several_planes_in_one_step", multi > 0, strf("%d (particle, step) pairs", multi));
    check("B.some_plane_partially_crossed", partial > 0, strf("%lld planes", partial));
}

void case_c(const std::string& exe, const fs::path& base) {
    RunSpec s;
    s.out = base / "C";
    s.sf18 = false;
    s.window = false;
    s.dump = false;
    s.snapshots = false;
    s.nr = 1;
    s.steps = 2;
    s.sample_every = 1; // the macrodispersion post-processing needs at least one sample
    s.planes = {1.9};
    const fs::path cfgp = base / "C.yaml";
    write_config(cfgp, s);
    const int rc = run_pipeline(exe, cfgp, base / "C.log");
    check("C.pipeline_exit_0", rc == 0, strf("exit %d", rc));
    if (rc != 0)
        return;
    const Csv cr = read_csv(s.out / "crossings" / "realization_0000_planes.csv");
    bool ok = cr.ok && cr.rows.size() == 1 && cr.rows[0][col(cr, "n_crossed")] == 0 &&
              cr.rows[0][col(cr, "n_injected")] == kNp;
    for (size_t c = 3; ok && c < cr.rows[0].size(); ++c)
        ok = std::isnan(cr.rows[0][c]);
    check("C.zero_crossings_give_nan_rows", ok);
    std::ifstream f(s.out / "crossings" / "realization_0000_planes.csv");
    std::stringstream ss;
    ss << f.rdbuf();
    check("C.nan_spelled_nan", ss.str().find(",nan,") != std::string::npos);
    check("C.full_plane_default_no_injection_window_in_config",
          [&] {
              std::ifstream e(s.out / "effective_config.yaml");
              std::stringstream es;
              es << e.rdbuf();
              return es.str().find("inject_window") == std::string::npos &&
                     es.str().find("sf18_medium_b") == std::string::npos &&
                     es.str().find("dump_field") == std::string::npos;
          }());
    check("C.no_field_dump_dir", !fs::exists(s.out / "fields"));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <macroflow3d_pipeline exe>\n", argv[0]);
        return 2;
    }
    const std::string exe = argv[1];
    const fs::path base = fs::temp_directory_path() / strf("par2_sf42_options_%d", (int)getpid());
    fs::remove_all(base);
    fs::create_directories(base);

    case_a_validation();
    case_a_yaml(base);
    {
        CudaContext ctx(0);
        case_b(exe, base, ctx);
    }
    case_c(exe, base);

    std::printf("par2_sf42_options: %d checks, %d failed\n", g_checks, g_failed);
    if (g_failed == 0)
        fs::remove_all(base);
    else
        std::printf("artifacts kept in %s\n", base.c_str());
    return g_failed == 0 ? 0 : 1;
}
