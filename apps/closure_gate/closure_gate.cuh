#pragma once

/**
 * @file closure_gate.cuh
 * @brief SF-30 streamline-closure gate pipeline (N2): field -> SF-19 Darcy
 *        solve -> SF-28 spline of the potential -> first-return map of the
 *        face `x1 = 0`, with statistics and the `summary.json` /
 *        `streamlines.csv` / `timing.json` writers.
 *
 * The whole pipeline is `run_closure_gate(config)`, shared by the
 * `closure_gate` executable (`closure_gate_main.cu`, argument parsing only)
 * and the 16^3 controls test (`tests/closure/streamline_closure_controls_tests.cu`).
 *
 * Contract (SF-30 N2 task specification, section 3.1; UNDERSTAND section 4):
 *   1. `Y = ln K` at cell centres of the unit cube `N^3` (`h = 1/N`): SF-18
 *      (`gaussian`), its x3-average rescaled to `sigma2` (`gaussian2d`), or an
 *      analytic N1 field times `eps`; `K = exp(Y)` (no geometric-mean factor).
 *   2. `physics::solve_affine_periodic_flow`, `qbar = e1`. If any corrector PCG
 *      did not converge, the result has `darcy_converged == false` and no
 *      streamline is integrated (exit code 3 of the executable).
 *   3. SF-28 GPU prefilter of `h_tilde` and of `Y`; host views.
 *   4. Direction field `g = G + grad s_h`, `k = exp(s_Y)` (the code's sign:
 *      `v = K (G + grad h_tilde)`), evaluated only through
 *      `interpolation::evaluate_point` (it owns the periodic reduction).
 *   5. Cell-centre diagnostics of `g`.
 *   6.-8. Seeds on `x1 = 0`, N1 integrator per tolerance, statistics.
 *   9. Outputs. `summary.json` is a pure function of the configuration
 *      (thread count and output directory excluded) and of the machine.
 *
 * No regularization, no clamping, no fallback: every non-`ok` streamline is
 * counted and excluded from the displacement statistics by status.
 */

#include "apps/closure_gate/closure_fields.hpp"
#include "apps/closure_gate/closure_statistics.hpp"
#include "apps/closure_gate/streamline_integrator.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace macroflow3d {
namespace closure_gate {

// ============================================================================
// Configuration
// ============================================================================

/// Usage / configuration error (exit code 2 of the executable).
struct ConfigError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

constexpr const char* kSchemaVersion = "sf30-closure-gate-1";
constexpr std::uint64_t kDefaultSeedRng = 20261005ULL;
/// Library default of `solvers::ProjectedPCGConfig::max_iter` (used when no
/// `--pcg-max-iter` is given; recorded explicitly in summary.json).
inline int library_default_pcg_max_iter() { return solvers::ProjectedPCGConfig{}.max_iter; }
inline double library_default_pcg_rtol() { return solvers::ProjectedPCGConfig{}.rtol; }

inline int default_thread_count() {
    const unsigned hc = std::thread::hardware_concurrency();
    return static_cast<int>(std::min(hc == 0 ? 1u : hc, 32u));
}

struct ClosureGateConfig {
    std::string field;                ///< gaussian|gaussian2d|control2d|lester2021|lester_brk|two_mode|generic3d
    int n = 0;                        ///< cubic grid N^3, N even
    std::string out_dir;              ///< empty: no files written (test use)

    // gaussian / gaussian2d (required for them, forbidden otherwise)
    bool has_sigma2 = false, has_ell = false, has_seed = false;
    double sigma2 = 0.0;
    double ell = 0.0;
    std::uint64_t seed = 0;

    // analytic fields (required for them, forbidden otherwise)
    bool has_eps = false;
    double eps = 0.0;

    // seeds
    int n_seeds = 1024;
    bool has_n_seeds = false;
    std::uint64_t seed_rng = kDefaultSeedRng;
    bool has_seed_rng = false;
    std::string seeds_file; ///< non-empty: CSV `y0,z0`; excludes --seeds / --seed-rng

    // integration
    std::vector<double> tols = {1e-6, 1e-8, 1e-10};
    double working_tol = 1e-8;
    int periods = 1;

    // Darcy
    double pcg_rtol = library_default_pcg_rtol();
    int pcg_max_iter = -1; ///< -1: library default
    int mg_levels = 0;     ///< 0: auto

    int threads = default_thread_count();
    int device = 0;
};

inline bool is_gaussian_kind(const std::string& f) { return f == "gaussian" || f == "gaussian2d"; }

inline bool is_known_field(const std::string& f) {
    return is_gaussian_kind(f) || f == "control2d" || f == "lester2021" || f == "lester_brk" ||
           f == "two_mode" || f == "generic3d";
}

/// Deepest MG hierarchy whose levels are all even with coarsest extent >= 4
/// (coarsest exactly 4 for N = 4 * 2^m), halving while the next level is
/// even and >= 4.
inline int auto_mg_levels(int N) {
    int levels = 1;
    int n = N;
    while (n % 2 == 0 && n / 2 >= 4 && (n / 2) % 2 == 0) {
        n /= 2;
        ++levels;
    }
    return levels;
}

inline int coarsest_extent(int N, int levels) {
    int n = N;
    for (int l = 1; l < levels; ++l) n /= 2;
    return n;
}

/// Sorts the ladder loosest -> tightest and validates the whole config.
/// Throws ConfigError on any inconsistency.
inline void normalize_and_validate(ClosureGateConfig& c) {
    if (!is_known_field(c.field)) throw ConfigError("unknown or missing --field '" + c.field + "'");
    if (c.n < 4 || c.n % 2 != 0) throw ConfigError("--n must be an even integer >= 4");
    if (is_gaussian_kind(c.field)) {
        if (!c.has_sigma2 || !c.has_ell || !c.has_seed)
            throw ConfigError("--field " + c.field + " requires --sigma2, --ell and --seed");
        if (c.has_eps) throw ConfigError("--eps is only valid for analytic fields");
        if (!(c.sigma2 > 0.0) || !std::isfinite(c.sigma2)) throw ConfigError("--sigma2 must be finite and > 0");
        if (!(c.ell > 0.0) || !std::isfinite(c.ell)) throw ConfigError("--ell must be finite and > 0");
    } else {
        if (!c.has_eps) throw ConfigError("--field " + c.field + " requires --eps");
        if (c.has_sigma2 || c.has_ell || c.has_seed)
            throw ConfigError("--sigma2/--ell/--seed are only valid for gaussian fields");
        if (!std::isfinite(c.eps)) throw ConfigError("--eps must be finite");
    }
    if (!c.seeds_file.empty() && (c.has_n_seeds || c.has_seed_rng))
        throw ConfigError("--seeds-file excludes --seeds and --seed-rng");
    if (c.seeds_file.empty() && c.n_seeds < 1) throw ConfigError("--seeds must be >= 1");
    if (c.tols.empty()) throw ConfigError("--tols must not be empty");
    for (double t : c.tols)
        if (!(t > 0.0) || !std::isfinite(t)) throw ConfigError("--tols entries must be finite and > 0");
    std::sort(c.tols.begin(), c.tols.end(), [](double a, double b) { return a > b; });
    for (std::size_t i = 1; i < c.tols.size(); ++i)
        if (c.tols[i] == c.tols[i - 1]) throw ConfigError("--tols has duplicate entries");
    if (std::find(c.tols.begin(), c.tols.end(), c.working_tol) == c.tols.end())
        throw ConfigError("--working-tol must be a member of --tols");
    if (c.periods < 1) throw ConfigError("--periods must be >= 1");
    if (!(c.pcg_rtol >= 0.0) || !std::isfinite(c.pcg_rtol)) throw ConfigError("--pcg-rtol must be finite and >= 0");
    if (c.pcg_max_iter != -1 && c.pcg_max_iter < 1) throw ConfigError("--pcg-max-iter must be >= 1");
    if (c.mg_levels < 0) throw ConfigError("--mg-levels must be 'auto' or an integer >= 1");
    if (c.threads < 1) throw ConfigError("--threads must be >= 1");
    if (c.device < 0) throw ConfigError("--device must be >= 0");
}

// ============================================================================
// Result records
// ============================================================================

struct FieldRecord {
    double mean = 0.0, variance = 0.0, min = 0.0, max = 0.0; ///< of Y = ln K (population variance)
    bool has_sf18 = false;
    physics::PeriodicGaussianFieldReport sf18{};
    bool has_x3_control = false;
    X3AveragedControlReport x3_control{};
};

struct SplineRecord {
    interpolation::PeriodicTricubicBSplineReport potential{};
    interpolation::PeriodicTricubicBSplineReport log_conductivity{};
    double potential_gpu_vs_host_rel = 0.0; ///< max|c_gpu - c_host| / max|c_host|
};

struct DirectionDiagnostics {
    double min_g1 = 0.0, min_abs_g = 0.0, max_abs_g = 0.0;
    std::size_t cells = 0, backflow_cells = 0;
    double backflow_volume_fraction = 0.0;
    double v_rms_spline = 0.0;
};

struct StatusCounts {
    long long seeds = 0, ok = 0, ok_with_backflow = 0, domain = 0;
    long long by_status[7] = {0, 0, 0, 0, 0, 0, 0};
    double non_ok_fraction = 0.0;
};

struct OptionalStats {
    bool valid = false;
    DisplacementStatistics s{};
    double max_abs_d2 = 0.0, max_abs_d3 = 0.0;
};

struct PeriodStatistics {
    int n = 0;
    StatusCounts counts{};
    long long count = 0; ///< streamlines entering the statistic (ok, completed >= n)
    OptionalStats unweighted{}, weighted{};
    long long count_no_backflow = 0;
    bool has_R_no_backflow = false;
    double R_no_backflow = 0.0;
    double mean_arclength = 0.0, mean_tau = 0.0, mean_tau_weighted = 0.0;
    // n == 1 only
    double D22 = 0.0, D33 = 0.0, D22_weighted = 0.0, D33_weighted = 0.0;
};

struct IntegratorTotals {
    long long accepted_steps = 0, rejected_steps = 0, landing_steps = 0, landing_rejected = 0,
              landings_discarded = 0, field_evaluations = 0;
    long long integrated_streamlines = 0; ///< all seeds except seed_backflow
    double field_evaluations_per_streamline_period = 0.0;
    double min_g1_hat = 0.0;
    double backflow_arclength = 0.0;
};

struct LadderEntry {
    int n = 0;
    long long both_ok = 0, status_differs = 0;
    bool valid = false;
    double max_dist = 0.0, rms_dist = 0.0;
};

struct ToleranceResult {
    double tol = 0.0;
    std::vector<StreamlineResult> streamlines;
    IntegratorTotals totals{};
    std::vector<PeriodStatistics> periods;
    bool has_ladder = false;
    double ladder_reference_tol = 0.0;
    std::vector<LadderEntry> ladder;
    double seconds = 0.0;
};

struct RoundTripRecord {
    double tol = 0.0;
    long long eligible = 0;        ///< ok and no backflow encounter in period 1
    long long backward_ok = 0;     ///< eligible with a backward status `ok`
    long long backward_status[7] = {0, 0, 0, 0, 0, 0, 0};
    bool valid = false;
    double max_dist = 0.0, rms_dist = 0.0;
    bool period1_rerun_used = false;           ///< n_periods > 1: period-1 flags from a 1-period rerun
    bool period1_rerun_bitwise_identical = true; ///< rerun landing == records[0] (bitwise)
    double seconds = 0.0;
};

struct ClosureGateResult {
    ClosureGateConfig config{};
    int mg_levels_used = 0;
    int pcg_max_iter_used = 0;
    double h = 0.0;

    FieldRecord field{};
    std::vector<double> Y; ///< host cell-centred ln K (kept for tests/diagnostics)

    physics::AffinePeriodicFlowReport darcy{};
    bool darcy_converged = false;

    SplineRecord splines{};
    DirectionDiagnostics direction{};

    std::vector<std::array<double, 2>> seeds;
    std::vector<double> weights; ///< k(seed) * g1(seed)
    std::vector<ToleranceResult> tolerances;
    RoundTripRecord round_trip{};

    // timing (never in summary.json)
    double t_field = 0.0, t_darcy = 0.0, t_splines = 0.0, t_diagnostics = 0.0, t_total = 0.0;
};

// ============================================================================
// Direction-field functor (N1 contract: void(const double x[3], double g[3], double& k) const)
// ============================================================================

struct SplineDirectionField {
    interpolation::PeriodicTricubicBSplineView potential{};
    interpolation::PeriodicTricubicBSplineView log_conductivity{};
    double G[3] = {0.0, 0.0, 0.0};

    void operator()(const double x[3], double g[3], double& k) const {
        real v = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
        interpolation::evaluate_point(potential, x[0], x[1], x[2], v, gx, gy, gz);
        g[0] = G[0] + gx;
        g[1] = G[1] + gy;
        g[2] = G[2] + gz;
        real y = 0.0, yx = 0.0, yy = 0.0, yz = 0.0;
        interpolation::evaluate_point(log_conductivity, x[0], x[1], x[2], y, yx, yy, yz);
        k = std::exp(y);
    }
};

// ============================================================================
// Helpers
// ============================================================================

namespace detail_gate {

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

inline std::vector<double> download(const real* d, std::size_t n) {
    std::vector<double> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

inline void upload(DeviceBuffer<real>& d, const std::vector<double>& h) {
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice));
}

inline const char* pcg_status_name(solvers::ProjectedPCGStatus s) {
    switch (s) {
    case solvers::ProjectedPCGStatus::converged: return "converged";
    case solvers::ProjectedPCGStatus::max_iterations: return "max_iterations";
    case solvers::ProjectedPCGStatus::invalid_configuration: return "invalid_configuration";
    case solvers::ProjectedPCGStatus::size_mismatch: return "size_mismatch";
    case solvers::ProjectedPCGStatus::aliasing: return "aliasing";
    case solvers::ProjectedPCGStatus::breakdown_pAp: return "breakdown_pAp";
    case solvers::ProjectedPCGStatus::breakdown_rz: return "breakdown_rz";
    case solvers::ProjectedPCGStatus::nonfinite_value: return "nonfinite_value";
    }
    return "unknown";
}

/// Reads a seeds CSV with header `y0,z0` (exact) and one `y0,z0` pair per
/// line (strtod, full precision). Blank lines are not allowed.
inline std::vector<std::array<double, 2>> read_seeds_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("closure_gate: cannot open seeds file '" + path + "'");
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("closure_gate: empty seeds file '" + path + "'");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "y0,z0") throw std::runtime_error("closure_gate: seeds file header must be exactly 'y0,z0'");
    std::vector<std::array<double, 2>> out;
    long long lineno = 1;
    while (std::getline(in, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t comma = line.find(',');
        if (comma == std::string::npos)
            throw std::runtime_error("closure_gate: seeds file line " + std::to_string(lineno) + ": expected 'y0,z0'");
        const std::string a = line.substr(0, comma), b = line.substr(comma + 1);
        char* end = nullptr;
        const double y0 = std::strtod(a.c_str(), &end);
        if (a.empty() || *end != '\0')
            throw std::runtime_error("closure_gate: seeds file line " + std::to_string(lineno) + ": bad y0");
        const double z0 = std::strtod(b.c_str(), &end);
        if (b.empty() || *end != '\0')
            throw std::runtime_error("closure_gate: seeds file line " + std::to_string(lineno) + ": bad z0");
        if (!std::isfinite(y0) || !std::isfinite(z0))
            throw std::runtime_error("closure_gate: seeds file line " + std::to_string(lineno) + ": non-finite");
        out.push_back({y0, z0});
    }
    if (out.empty()) throw std::runtime_error("closure_gate: seeds file '" + path + "' has no points");
    return out;
}

inline void field_statistics(const std::vector<double>& Y, FieldRecord& r) {
    double mean = 0.0, mn = std::numeric_limits<double>::infinity(), mx = -mn;
    for (double v : Y) {
        mean += v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    mean /= static_cast<double>(Y.size());
    double var = 0.0;
    for (double v : Y) var += (v - mean) * (v - mean);
    var /= static_cast<double>(Y.size());
    r.mean = mean;
    r.variance = var;
    r.min = mn;
    r.max = mx;
}

/// Cell-centre diagnostics of g = G + grad s_h. Per-plane partials computed
/// in parallel (fixed plane ownership), combined sequentially in plane
/// order: independent of the thread count.
inline DirectionDiagnostics direction_diagnostics(const SplineDirectionField& f, int N, int n_threads) {
    struct Partial {
        double min_g1 = std::numeric_limits<double>::infinity();
        double min_abs = std::numeric_limits<double>::infinity();
        double max_abs = 0.0;
        std::size_t backflow = 0;
        double sum_v2 = 0.0;
    };
    std::vector<Partial> parts(static_cast<std::size_t>(N));
    const double h = 1.0 / N;
    auto plane = [&](int k) {
        Partial p;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const double x[3] = {(i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h};
                double g[3], kk;
                f(x, g, kk);
                const double a = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
                p.min_g1 = std::min(p.min_g1, g[0]);
                p.min_abs = std::min(p.min_abs, a);
                p.max_abs = std::max(p.max_abs, a);
                if (!(g[0] > 0.0)) ++p.backflow;
                p.sum_v2 += kk * kk * a * a;
            }
        parts[static_cast<std::size_t>(k)] = p;
    };
    const int workers = std::max(1, std::min(n_threads, N));
    std::vector<std::thread> pool;
    for (int w = 0; w < workers; ++w)
        pool.emplace_back([&, w] {
            for (int k = w; k < N; k += workers) plane(k);
        });
    for (auto& t : pool) t.join();
    DirectionDiagnostics d;
    d.min_g1 = std::numeric_limits<double>::infinity();
    d.min_abs_g = std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (const Partial& p : parts) {
        d.min_g1 = std::min(d.min_g1, p.min_g1);
        d.min_abs_g = std::min(d.min_abs_g, p.min_abs);
        d.max_abs_g = std::max(d.max_abs_g, p.max_abs);
        d.backflow_cells += p.backflow;
        sum += p.sum_v2;
    }
    d.cells = static_cast<std::size_t>(N) * N * N;
    d.backflow_volume_fraction = static_cast<double>(d.backflow_cells) / static_cast<double>(d.cells);
    d.v_rms_spline = std::sqrt(sum / static_cast<double>(d.cells));
    return d;
}

inline StatusCounts count_statuses(const std::vector<StreamlineResult>& rs) {
    StatusCounts c;
    c.seeds = static_cast<long long>(rs.size());
    for (const auto& r : rs) {
        ++c.by_status[static_cast<int>(r.status)];
        if (r.status == StreamlineStatus::ok) {
            ++c.ok;
            if (r.backflow_encounter) ++c.ok_with_backflow;
        }
    }
    c.domain = c.seeds - c.by_status[static_cast<int>(StreamlineStatus::seed_backflow)];
    c.non_ok_fraction = c.domain > 0 ? static_cast<double>(c.domain - c.ok) / static_cast<double>(c.domain) : 0.0;
    return c;
}

inline OptionalStats optional_stats(const std::vector<std::array<double, 2>>& d, const std::vector<double>& w) {
    OptionalStats o;
    if (d.empty()) return o;
    double W = 0.0;
    for (double x : w) W += x;
    if (!(W > 0.0)) return o;
    o.valid = true;
    o.s = displacement_statistics(d, w);
    for (std::size_t i = 0; i < d.size(); ++i) {
        if (w[i] > 0.0) {
            o.max_abs_d2 = std::max(o.max_abs_d2, std::abs(d[i][0]));
            o.max_abs_d3 = std::max(o.max_abs_d3, std::abs(d[i][1]));
        }
    }
    return o;
}

inline PeriodStatistics period_statistics(const std::vector<StreamlineResult>& rs,
                                          const std::vector<std::array<double, 2>>& seeds,
                                          const std::vector<double>& weights, const StatusCounts& counts, int n) {
    PeriodStatistics p;
    p.n = n;
    p.counts = counts;
    std::vector<std::array<double, 2>> d, d_nb;
    std::vector<double> w, ones, ones_nb;
    double sum_s = 0.0, sum_tau = 0.0, sum_wtau = 0.0, W = 0.0;
    for (std::size_t i = 0; i < rs.size(); ++i) {
        const StreamlineResult& r = rs[i];
        if (r.status != StreamlineStatus::ok || r.periods_completed < n) continue;
        const PlaneRecord& rc = r.records[static_cast<std::size_t>(n - 1)];
        const std::array<double, 2> di = {rc.x2 - seeds[i][0], rc.x3 - seeds[i][1]};
        d.push_back(di);
        w.push_back(weights[i]);
        ones.push_back(1.0);
        sum_s += rc.s;
        sum_tau += rc.tau;
        sum_wtau += weights[i] * rc.tau;
        W += weights[i];
        if (!r.backflow_encounter) {
            d_nb.push_back(di);
            ones_nb.push_back(1.0);
        }
    }
    p.count = static_cast<long long>(d.size());
    p.count_no_backflow = static_cast<long long>(d_nb.size());
    p.unweighted = optional_stats(d, ones);
    p.weighted = optional_stats(d, w);
    const OptionalStats nb = optional_stats(d_nb, ones_nb);
    p.has_R_no_backflow = nb.valid;
    p.R_no_backflow = nb.valid ? nb.s.R : 0.0;
    if (!d.empty()) {
        p.mean_arclength = sum_s / static_cast<double>(d.size());
        p.mean_tau = sum_tau / static_cast<double>(d.size());
        p.mean_tau_weighted = W > 0.0 ? sum_wtau / W : 0.0;
        if (n == 1) {
            p.D22 = p.unweighted.s.var_d2 / (2.0 * p.mean_tau);
            p.D33 = p.unweighted.s.var_d3 / (2.0 * p.mean_tau);
            if (p.weighted.valid) {
                p.D22_weighted = p.weighted.s.var_d2 / (2.0 * p.mean_tau_weighted);
                p.D33_weighted = p.weighted.s.var_d3 / (2.0 * p.mean_tau_weighted);
            }
        }
    }
    return p;
}

inline IntegratorTotals integrator_totals(const std::vector<StreamlineResult>& rs, int n_periods) {
    IntegratorTotals t;
    t.min_g1_hat = std::numeric_limits<double>::infinity();
    for (const auto& r : rs) {
        t.accepted_steps += r.accepted_steps;
        t.rejected_steps += r.rejected_steps;
        t.landing_steps += r.landing_steps;
        t.landing_rejected += r.landing_rejected;
        t.landings_discarded += r.landings_discarded;
        t.field_evaluations += r.field_evaluations;
        t.min_g1_hat = std::min(t.min_g1_hat, r.min_g1_hat);
        t.backflow_arclength += r.backflow_arclength;
        if (r.status != StreamlineStatus::seed_backflow) ++t.integrated_streamlines;
    }
    if (rs.empty()) t.min_g1_hat = 0.0;
    const double denom = static_cast<double>(t.integrated_streamlines) * n_periods;
    t.field_evaluations_per_streamline_period = denom > 0.0 ? static_cast<double>(t.field_evaluations) / denom : 0.0;
    return t;
}

inline LadderEntry ladder_entry(const std::vector<StreamlineResult>& a, const std::vector<StreamlineResult>& b, int n) {
    LadderEntry e;
    e.n = n;
    double sq = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].status != b[i].status) ++e.status_differs;
        if (a[i].status != StreamlineStatus::ok || b[i].status != StreamlineStatus::ok) continue;
        const PlaneRecord& p = a[i].records[static_cast<std::size_t>(n - 1)];
        const PlaneRecord& q = b[i].records[static_cast<std::size_t>(n - 1)];
        const double dd = std::hypot(p.x2 - q.x2, p.x3 - q.x3);
        ++e.both_ok;
        e.max_dist = std::max(e.max_dist, dd);
        sq += dd * dd;
    }
    e.valid = e.both_ok > 0;
    e.rms_dist = e.valid ? std::sqrt(sq / static_cast<double>(e.both_ok)) : 0.0;
    return e;
}

inline std::vector<int> recorded_periods(int n_periods) {
    std::vector<int> p;
    for (int q = 1; q <= n_periods; q *= 2) p.push_back(q);
    if (p.back() != n_periods) p.push_back(n_periods);
    return p;
}

// ---------------------------------------------------------------------------
// Minimal ordered JSON writer (17 significant digits; non-finite -> null)
// ---------------------------------------------------------------------------

class JsonWriter {
  public:
    std::string str() const { return out_ + "\n"; }

    void begin_object(const char* key = nullptr) { open(key, '{'); }
    void end_object() { close('}'); }
    void begin_array(const char* key = nullptr) { open(key, '['); }
    void end_array() { close(']'); }

    void number(const char* key, double v) {
        prefix(key);
        if (!std::isfinite(v)) {
            out_ += "null";
        } else {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", v);
            out_ += buf;
        }
    }
    void integer(const char* key, long long v) {
        prefix(key);
        out_ += std::to_string(v);
    }
    void uinteger(const char* key, unsigned long long v) {
        prefix(key);
        out_ += std::to_string(v);
    }
    void boolean(const char* key, bool v) {
        prefix(key);
        out_ += v ? "true" : "false";
    }
    void null(const char* key) {
        prefix(key);
        out_ += "null";
    }
    void string(const char* key, const std::string& v) {
        prefix(key);
        out_ += '"';
        for (char c : v) {
            if (c == '"' || c == '\\') {
                out_ += '\\';
                out_ += c;
            } else if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                out_ += buf;
            } else {
                out_ += c;
            }
        }
        out_ += '"';
    }
    void number_array(const char* key, const double* v, int n) {
        begin_array(key);
        for (int i = 0; i < n; ++i) number(nullptr, v[i]);
        end_array();
    }

  private:
    std::string out_;
    std::vector<bool> first_;

    void prefix(const char* key) {
        if (!first_.empty()) {
            if (!first_.back()) out_ += ',';
            first_.back() = false;
            out_ += '\n';
            out_.append(2 * first_.size(), ' ');
        }
        if (key) {
            out_ += '"';
            out_ += key;
            out_ += "\": ";
        }
    }
    void open(const char* key, char c) {
        prefix(key);
        out_ += c;
        first_.push_back(true);
    }
    void close(char c) {
        const bool empty = first_.back();
        first_.pop_back();
        if (!empty) {
            out_ += '\n';
            out_.append(2 * first_.size(), ' ');
        }
        out_ += c;
    }
};

inline std::string fmt17(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

inline void write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("closure_gate: cannot write '" + path + "'");
    f << content;
    if (!f) throw std::runtime_error("closure_gate: write failed for '" + path + "'");
}

} // namespace detail_gate

// ============================================================================
// Serialization
// ============================================================================

inline void json_stats(detail_gate::JsonWriter& j, const char* key, const OptionalStats& o) {
    if (!o.valid) {
        j.null(key);
        return;
    }
    j.begin_object(key);
    j.integer("count", static_cast<long long>(o.s.count));
    j.number("weight_sum", o.s.weight_sum);
    j.number_array("mean", o.s.mean, 2);
    j.number("R", o.s.R);
    j.number("rms_abs_d", o.s.rms_abs);
    j.number("max_abs_d", o.s.max_abs);
    j.number("var_d2", o.s.var_d2);
    j.number("var_d3", o.s.var_d3);
    j.number("max_abs_d2", o.max_abs_d2);
    j.number("max_abs_d3", o.max_abs_d3);
    j.end_object();
}

inline void json_pcg(detail_gate::JsonWriter& j, const solvers::ProjectedPCGResult& r) {
    j.begin_object();
    j.boolean("converged", r.converged);
    j.string("status", detail_gate::pcg_status_name(r.status));
    j.integer("iterations", r.iterations);
    j.number("raw_rhs_mean", r.raw_rhs_mean);
    j.number("raw_rhs_l2_norm", r.raw_rhs_l2_norm);
    j.number("raw_rhs_compatibility_defect", r.raw_rhs_compatibility_defect);
    j.number("initial_projected_residual", r.initial_projected_residual);
    j.number("final_projected_residual", r.final_projected_residual);
    j.number("relative_projected_residual", r.relative_projected_residual);
    j.number("final_field_mean", r.final_field_mean);
    j.end_object();
}

inline void json_spline_report(detail_gate::JsonWriter& j, const char* key,
                               const interpolation::PeriodicTricubicBSplineReport& r) {
    j.begin_object(key);
    j.integer("coefficient_bytes", static_cast<long long>(r.coefficient_bytes));
    j.integer("spectrum_bytes", static_cast<long long>(r.spectrum_bytes));
    j.integer("cufft_work_area_bytes", static_cast<long long>(r.cufft_work_area_bytes));
    j.integer("total_device_bytes", static_cast<long long>(r.total_device_bytes));
    j.end_object();
}

inline std::string summary_json(const ClosureGateResult& R) {
    using detail_gate::JsonWriter;
    const ClosureGateConfig& c = R.config;
    JsonWriter j;
    j.begin_object();
    j.string("schema_version", kSchemaVersion);

    // --- configuration after defaults (thread count and output dir excluded)
    j.begin_object("configuration");
    j.string("field", c.field);
    j.integer("n", c.n);
    j.begin_array("grid");
    j.integer(nullptr, c.n);
    j.integer(nullptr, c.n);
    j.integer(nullptr, c.n);
    j.end_array();
    j.number("h", R.h);
    j.number("L", 1.0);
    if (is_gaussian_kind(c.field)) {
        j.number("sigma2", c.sigma2);
        j.number("ell", c.ell);
        j.uinteger("seed", c.seed);
        j.null("eps");
    } else {
        j.null("sigma2");
        j.null("ell");
        j.null("seed");
        j.number("eps", c.eps);
    }
    j.string("seeds_source", c.seeds_file.empty() ? "generated" : "file");
    j.integer("n_seeds", static_cast<long long>(R.seeds.size()));
    if (c.seeds_file.empty()) {
        j.uinteger("seed_rng", c.seed_rng);
        j.null("seeds_file");
    } else {
        j.null("seed_rng");
        j.string("seeds_file", c.seeds_file);
    }
    j.number_array("tols", c.tols.data(), static_cast<int>(c.tols.size()));
    j.number("working_tol", c.working_tol);
    j.integer("periods", c.periods);
    j.number("pcg_rtol", c.pcg_rtol);
    j.integer("pcg_max_iter", R.pcg_max_iter_used);
    j.boolean("pcg_max_iter_is_library_default", c.pcg_max_iter == -1);
    j.integer("pcg_check_every", solvers::ProjectedPCGConfig{}.check_every);
    j.string("mg_levels_requested", c.mg_levels == 0 ? std::string("auto") : std::to_string(c.mg_levels));
    j.integer("mg_levels", R.mg_levels_used);
    j.integer("mg_coarsest_extent", coarsest_extent(c.n, R.mg_levels_used));
    j.number_array("qbar", std::array<double, 3>{1.0, 0.0, 0.0}.data(), 3);
    j.integer("device", c.device);
    j.begin_object("integrator");
    {
        IntegratorOptions o;
        j.number("h_max", R.h);
        j.number("min_step", o.min_step);
        j.number("max_arclength_per_period", o.max_arclength_per_period);
        j.string("method", "DP5(4) arclength, Henon landing on x1 = x1_0 + n");
    }
    j.end_object();
    j.string("direction_field", "g = G + grad s_h, k = exp(s_Y)");
    j.end_object();

    // --- step 1: field
    j.begin_object("field");
    j.string("kind", c.field);
    j.number("Y_mean", R.field.mean);
    j.number("Y_variance", R.field.variance);
    j.number("Y_min", R.field.min);
    j.number("Y_max", R.field.max);
    j.string("K", "exp(Y)");
    if (R.field.has_sf18) {
        j.begin_object("sf18_report");
        j.number("raw_mean", R.field.sf18.raw_mean);
        j.number("raw_variance", R.field.sf18.raw_variance);
        j.number("applied_scale", R.field.sf18.applied_scale);
        j.number("final_variance", R.field.sf18.final_variance);
        j.integer("active_mode_count", static_cast<long long>(R.field.sf18.active_mode_count));
        j.end_object();
    } else {
        j.null("sf18_report");
    }
    if (R.field.has_x3_control) {
        j.begin_object("x3_averaged_control");
        j.number("raw_mean", R.field.x3_control.raw_mean);
        j.number("raw_variance", R.field.x3_control.raw_variance);
        j.number("applied_scale", R.field.x3_control.applied_scale);
        j.end_object();
    } else {
        j.null("x3_averaged_control");
    }
    j.end_object();

    // --- step 2: Darcy
    j.boolean("darcy_converged", R.darcy_converged);
    j.begin_object("darcy");
    j.boolean("converged", R.darcy_converged);
    j.begin_array("K_eff");
    for (int i = 0; i < 3; ++i) j.number_array(nullptr, R.darcy.K_eff[i], 3);
    j.end_array();
    j.number("symmetry_defect_rel", R.darcy.symmetry_defect_rel);
    j.number_array("eigenvalues_symmetric_part", R.darcy.eigenvalues_symmetric_part, 3);
    j.number_array("G", R.darcy.G, 3);
    j.number_array("achieved_mean_flux", R.darcy.achieved_mean_flux, 3);
    j.number("div_max_abs", R.darcy.div_max_abs);
    j.number("div_rms", R.darcy.div_rms);
    j.begin_array("corrector_results");
    for (int d = 0; d < 3; ++d) json_pcg(j, R.darcy.corrector_results[d]);
    j.end_array();
    j.end_object();

    if (!R.darcy_converged) {
        j.null("splines");
        j.null("direction_field_diagnostics");
        j.null("tolerances");
        j.null("round_trip");
        j.null("first_seeds");
        j.end_object();
        return j.str();
    }

    // --- step 3: splines
    j.begin_object("splines");
    json_spline_report(j, "potential_prefilter", R.splines.potential);
    json_spline_report(j, "log_conductivity_prefilter", R.splines.log_conductivity);
    j.number("potential_gpu_vs_host_prefilter_rel", R.splines.potential_gpu_vs_host_rel);
    j.end_object();

    // --- step 5: diagnostics
    j.begin_object("direction_field_diagnostics");
    j.integer("cells", static_cast<long long>(R.direction.cells));
    j.number("min_g1", R.direction.min_g1);
    j.number("min_abs_g", R.direction.min_abs_g);
    j.number("max_abs_g", R.direction.max_abs_g);
    j.integer("backflow_cells", static_cast<long long>(R.direction.backflow_cells));
    j.number("backflow_volume_fraction", R.direction.backflow_volume_fraction);
    j.number("v_rms_spline", R.direction.v_rms_spline);
    j.end_object();

    // --- step 8: statistics
    j.begin_array("tolerances");
    for (const ToleranceResult& t : R.tolerances) {
        j.begin_object();
        j.number("tol", t.tol);
        j.begin_object("integrator_totals");
        j.integer("integrated_streamlines", t.totals.integrated_streamlines);
        j.integer("accepted_steps", t.totals.accepted_steps);
        j.integer("rejected_steps", t.totals.rejected_steps);
        j.integer("landing_steps", t.totals.landing_steps);
        j.integer("landing_rejected", t.totals.landing_rejected);
        j.integer("landings_discarded", t.totals.landings_discarded);
        j.integer("field_evaluations", t.totals.field_evaluations);
        j.number("field_evaluations_per_streamline_period", t.totals.field_evaluations_per_streamline_period);
        j.number("min_g1_hat", t.totals.min_g1_hat);
        j.number("backflow_arclength", t.totals.backflow_arclength);
        j.end_object();
        j.begin_array("periods");
        for (const PeriodStatistics& p : t.periods) {
            j.begin_object();
            j.integer("n", p.n);
            j.begin_object("counts");
            j.integer("seeds", p.counts.seeds);
            for (int s = 0; s < 7; ++s)
                j.integer(to_string(static_cast<StreamlineStatus>(s)), p.counts.by_status[s]);
            j.integer("ok_with_backflow_encounter", p.counts.ok_with_backflow);
            j.integer("domain", p.counts.domain);
            j.number("non_ok_fraction", p.counts.non_ok_fraction);
            j.end_object();
            j.integer("count", p.count);
            json_stats(j, "unweighted", p.unweighted);
            json_stats(j, "flux_weighted", p.weighted);
            j.integer("count_no_backflow", p.count_no_backflow);
            if (p.has_R_no_backflow) j.number("R_no_backflow", p.R_no_backflow);
            else j.null("R_no_backflow");
            if (p.count > 0) {
                j.number("mean_arclength", p.mean_arclength);
                j.number("mean_tau", p.mean_tau);
                j.number("mean_tau_flux_weighted", p.mean_tau_weighted);
            } else {
                j.null("mean_arclength");
                j.null("mean_tau");
                j.null("mean_tau_flux_weighted");
            }
            if (p.n == 1 && p.count > 0) {
                j.number("reinjection_D22", p.D22);
                j.number("reinjection_D33", p.D33);
                j.number("reinjection_D22_flux_weighted", p.D22_weighted);
                j.number("reinjection_D33_flux_weighted", p.D33_weighted);
            }
            j.end_object();
        }
        j.end_array();
        if (t.has_ladder) {
            j.begin_object("ladder_vs_tightest");
            j.number("tightest_tol", t.ladder_reference_tol);
            j.begin_array("periods");
            for (const LadderEntry& e : t.ladder) {
                j.begin_object();
                j.integer("n", e.n);
                j.integer("both_ok", e.both_ok);
                j.integer("status_differs", e.status_differs);
                if (e.valid) {
                    j.number("max_distance", e.max_dist);
                    j.number("rms_distance", e.rms_dist);
                } else {
                    j.null("max_distance");
                    j.null("rms_distance");
                }
                j.end_object();
            }
            j.end_array();
            j.end_object();
        } else {
            j.null("ladder_vs_tightest");
        }
        j.end_object();
    }
    j.end_array();

    // --- round trip
    j.begin_object("round_trip");
    j.number("tol", R.round_trip.tol);
    j.string("eligibility", "status ok and no backflow encounter in period 1");
    j.integer("eligible", R.round_trip.eligible);
    j.integer("backward_ok", R.round_trip.backward_ok);
    j.begin_object("backward_status");
    for (int s = 0; s < 7; ++s)
        j.integer(to_string(static_cast<StreamlineStatus>(s)), R.round_trip.backward_status[s]);
    j.end_object();
    if (R.round_trip.valid) {
        j.number("max_distance", R.round_trip.max_dist);
        j.number("rms_distance", R.round_trip.rms_dist);
    } else {
        j.null("max_distance");
        j.null("rms_distance");
    }
    j.boolean("period1_rerun_used", R.round_trip.period1_rerun_used);
    j.boolean("period1_rerun_bitwise_identical", R.round_trip.period1_rerun_bitwise_identical);
    j.end_object();

    j.begin_array("first_seeds");
    for (std::size_t i = 0; i < std::min<std::size_t>(5, R.seeds.size()); ++i) {
        const double p[3] = {0.0, R.seeds[i][0], R.seeds[i][1]};
        j.number_array(nullptr, p, 3);
    }
    j.end_array();
    j.end_object();
    return j.str();
}

inline std::string streamlines_csv(const ClosureGateResult& R) {
    using detail_gate::fmt17;
    const ToleranceResult* wt = nullptr;
    for (const auto& t : R.tolerances)
        if (t.tol == R.config.working_tol) wt = &t;
    if (!wt) return std::string();
    const std::vector<int> rec = detail_gate::recorded_periods(R.config.periods);
    std::string s = "id,y0,z0,w,status,completed_periods,backflow_encounter,min_g1_hat,backflow_arclength,"
                    "accepted_steps,rejected_steps";
    for (int p : rec) {
        const std::string q = std::to_string(p);
        s += ",y_" + q + ",z_" + q + ",s_" + q + ",tau_" + q;
    }
    s += '\n';
    for (std::size_t i = 0; i < wt->streamlines.size(); ++i) {
        const StreamlineResult& r = wt->streamlines[i];
        s += std::to_string(i) + ',' + fmt17(R.seeds[i][0]) + ',' + fmt17(R.seeds[i][1]) + ',' +
             fmt17(R.weights[i]) + ',' + to_string(r.status) + ',' + std::to_string(r.periods_completed) + ',' +
             (r.backflow_encounter ? "1" : "0") + ',' + fmt17(r.min_g1_hat) + ',' + fmt17(r.backflow_arclength) +
             ',' + std::to_string(r.accepted_steps) + ',' + std::to_string(r.rejected_steps);
        for (int p : rec) {
            if (r.periods_completed >= p) {
                const PlaneRecord& rc = r.records[static_cast<std::size_t>(p - 1)];
                s += ',' + fmt17(rc.x2) + ',' + fmt17(rc.x3) + ',' + fmt17(rc.s) + ',' + fmt17(rc.tau);
            } else {
                s += ",,,,";
            }
        }
        s += '\n';
    }
    return s;
}

inline std::string timing_json(const ClosureGateResult& R) {
    detail_gate::JsonWriter j;
    j.begin_object();
    j.string("schema_version", "sf30-closure-gate-timing-1");
    j.integer("threads", R.config.threads);
    char host[256] = {0};
    if (gethostname(host, sizeof(host) - 1) == 0) j.string("host", host);
    else j.null("host");
    j.number("field_s", R.t_field);
    j.number("darcy_s", R.t_darcy);
    j.number("splines_s", R.t_splines);
    j.number("diagnostics_s", R.t_diagnostics);
    j.begin_array("tolerances");
    for (const auto& t : R.tolerances) {
        j.begin_object();
        j.number("tol", t.tol);
        j.number("seconds", t.seconds);
        j.end_object();
    }
    j.end_array();
    j.number("round_trip_s", R.round_trip.seconds);
    j.number("total_s", R.t_total);
    j.end_object();
    return j.str();
}

inline void write_outputs(const ClosureGateResult& R) {
    const std::string& dir = R.config.out_dir;
    if (dir.empty()) return;
    std::filesystem::create_directories(dir);
    detail_gate::write_file(dir + "/summary.json", summary_json(R));
    if (R.darcy_converged) detail_gate::write_file(dir + "/streamlines.csv", streamlines_csv(R));
    detail_gate::write_file(dir + "/timing.json", timing_json(R));
}

// ============================================================================
// Pipeline
// ============================================================================

/**
 * Runs the whole gate. Validates (and normalizes) `config` first
 * (`ConfigError` on usage errors). Writes the outputs when
 * `config.out_dir` is non-empty. On Darcy non-convergence returns with
 * `darcy_converged == false` and no streamline results (outputs written).
 */
inline ClosureGateResult run_closure_gate(CudaContext& ctx, ClosureGateConfig config) {
    using detail_gate::seconds_since;
    const auto t_all = std::chrono::steady_clock::now();
    normalize_and_validate(config);

    ClosureGateResult R;
    R.config = config;
    const int N = config.n;
    R.h = 1.0 / N;
    const Grid3D grid(N, N, N, 1.0 / N, 1.0 / N, 1.0 / N);
    const std::size_t ncell = grid.num_cells();

    // ---- seeds (read before any GPU work so a bad file fails fast)
    if (config.seeds_file.empty()) {
        for (int i = 0; i < config.n_seeds; ++i)
            R.seeds.push_back(seed_point(config.seed_rng, static_cast<std::uint64_t>(i)));
    } else {
        R.seeds = detail_gate::read_seeds_file(config.seeds_file);
    }

    // ---- 1. field
    auto t0 = std::chrono::steady_clock::now();
    if (is_gaussian_kind(config.field)) {
        DeviceBuffer<real> y_dev(ncell);
        physics::PeriodicGaussianFieldConfig gcfg;
        gcfg.sigma2 = config.sigma2;
        gcfg.corr_length = config.ell;
        gcfg.seed = config.seed;
        gcfg.normalize_variance = true;
        physics::PeriodicGaussianFieldWorkspace gws;
        R.field.sf18 = physics::generate_periodic_gaussian_field(ctx, grid, gcfg, y_dev.span(), gws);
        R.field.has_sf18 = true;
        ctx.synchronize();
        R.Y = detail_gate::download(y_dev.data(), ncell);
        if (config.field == "gaussian2d") {
            R.field.x3_control = make_x3_averaged_control(grid, config.sigma2, R.Y);
            R.field.has_x3_control = true;
        }
    } else {
        fill_analytic_log_conductivity(grid, analytic_field_from_name(config.field), config.eps, R.Y);
    }
    detail_gate::field_statistics(R.Y, R.field);
    std::vector<double> K(ncell);
    for (std::size_t c = 0; c < ncell; ++c) K[c] = std::exp(R.Y[c]);
    DeviceBuffer<real> y_dev(ncell), k_dev(ncell);
    detail_gate::upload(y_dev, R.Y);
    detail_gate::upload(k_dev, K);
    R.t_field = seconds_since(t0);

    // ---- 2. Darcy
    t0 = std::chrono::steady_clock::now();
    physics::AffinePeriodicFlowConfig fcfg;
    fcfg.qbar[0] = 1.0;
    fcfg.qbar[1] = 0.0;
    fcfg.qbar[2] = 0.0;
    fcfg.linear.rtol = config.pcg_rtol;
    if (config.pcg_max_iter != -1) fcfg.linear.max_iter = config.pcg_max_iter;
    R.pcg_max_iter_used = fcfg.linear.max_iter;
    R.mg_levels_used = config.mg_levels == 0 ? auto_mg_levels(N) : config.mg_levels;
    fcfg.mg.num_levels = R.mg_levels_used;
    physics::AffinePeriodicFlowWorkspace fws;
    DeviceBuffer<real> u(static_cast<std::size_t>(N + 1) * N * N), v(static_cast<std::size_t>(N) * (N + 1) * N),
        w(static_cast<std::size_t>(N) * N * (N + 1));
    R.darcy = physics::solve_affine_periodic_flow(ctx, grid, DeviceSpan<const real>(k_dev.span()), fcfg,
                                                  physics::AffinePeriodicVelocityView{u.span(), v.span(), w.span()},
                                                  fws);
    ctx.synchronize();
    R.darcy_converged = R.darcy.corrector_results[0].converged && R.darcy.corrector_results[1].converged &&
                        R.darcy.corrector_results[2].converged;
    R.t_darcy = seconds_since(t0);
    if (!R.darcy_converged) {
        R.t_total = seconds_since(t_all);
        write_outputs(R);
        return R;
    }

    // ---- 3. splines
    t0 = std::chrono::steady_clock::now();
    const DeviceSpan<const real> htilde = fws.potential_fluctuation();
    interpolation::PeriodicTricubicBSplineWorkspace sh_ws, sy_ws;
    R.splines.potential = interpolation::prefilter_periodic_tricubic_bspline(ctx, grid, htilde, sh_ws);
    R.splines.log_conductivity =
        interpolation::prefilter_periodic_tricubic_bspline(ctx, grid, DeviceSpan<const real>(y_dev.span()), sy_ws);
    ctx.synchronize();
    std::vector<double> coeff_h = detail_gate::download(sh_ws.coefficients.data(), ncell);
    std::vector<double> coeff_y = detail_gate::download(sy_ws.coefficients.data(), ncell);
    {
        const std::vector<double> h_host = detail_gate::download(htilde.data(), ncell);
        std::vector<double> c_host(ncell);
        interpolation::prefilter_periodic_tricubic_bspline_host(grid, h_host.data(), c_host.data());
        double dmax = 0.0, cmax = 0.0;
        for (std::size_t c = 0; c < ncell; ++c) {
            dmax = std::max(dmax, std::abs(coeff_h[c] - c_host[c]));
            cmax = std::max(cmax, std::abs(c_host[c]));
        }
        R.splines.potential_gpu_vs_host_rel = cmax > 0.0 ? dmax / cmax : dmax;
    }
    SplineDirectionField field;
    field.potential = interpolation::make_host_view(grid, coeff_h.data());
    field.log_conductivity = interpolation::make_host_view(grid, coeff_y.data());
    for (int d = 0; d < 3; ++d) field.G[d] = R.darcy.G[d];
    R.t_splines = seconds_since(t0);

    // ---- 5. diagnostics
    t0 = std::chrono::steady_clock::now();
    R.direction = detail_gate::direction_diagnostics(field, N, config.threads);
    R.t_diagnostics = seconds_since(t0);

    // ---- 6. seeds, flux weights
    std::vector<std::array<double, 3>> seeds3;
    seeds3.reserve(R.seeds.size());
    R.weights.resize(R.seeds.size());
    for (std::size_t i = 0; i < R.seeds.size(); ++i) {
        seeds3.push_back({0.0, R.seeds[i][0], R.seeds[i][1]});
        const double x[3] = {0.0, R.seeds[i][0], R.seeds[i][1]};
        double g[3], kk;
        field(x, g, kk);
        R.weights[i] = kk * g[0];
    }

    // ---- 7./8. integration and statistics per tolerance
    for (double tol : config.tols) {
        t0 = std::chrono::steady_clock::now();
        ToleranceResult T;
        T.tol = tol;
        IntegratorOptions opt;
        opt.tol = tol;
        opt.h_max = R.h;
        opt.n_periods = config.periods;
        opt.sigma = +1;
        T.streamlines = integrate_streamlines(field, seeds3, opt, config.threads);
        T.totals = detail_gate::integrator_totals(T.streamlines, config.periods);
        const StatusCounts counts = detail_gate::count_statuses(T.streamlines);
        for (int n = 1; n <= config.periods; ++n)
            T.periods.push_back(detail_gate::period_statistics(T.streamlines, R.seeds, R.weights, counts, n));
        T.seconds = seconds_since(t0);
        R.tolerances.push_back(std::move(T));
    }
    // ladder vs tightest (last after loosest->tightest sort)
    const ToleranceResult& tight = R.tolerances.back();
    for (std::size_t t = 0; t + 1 < R.tolerances.size(); ++t) {
        ToleranceResult& T = R.tolerances[t];
        T.has_ladder = true;
        T.ladder_reference_tol = tight.tol;
        T.ladder.push_back(detail_gate::ladder_entry(T.streamlines, tight.streamlines, 1));
        if (config.periods > 1)
            T.ladder.push_back(detail_gate::ladder_entry(T.streamlines, tight.streamlines, config.periods));
    }

    // ---- round trip at the working tolerance
    t0 = std::chrono::steady_clock::now();
    {
        const ToleranceResult* wt = nullptr;
        for (const auto& t : R.tolerances)
            if (t.tol == config.working_tol) wt = &t;
        RoundTripRecord& rt = R.round_trip;
        rt.tol = config.working_tol;
        // Period-1 records and period-1 backflow flags. With n_periods > 1 the
        // streamline flag covers every period, so a 1-period forward rerun
        // (bitwise the same steps up to the first landing) gives the exact
        // period-1 flag.
        std::vector<StreamlineResult> p1;
        const std::vector<StreamlineResult>* fw = &wt->streamlines;
        IntegratorOptions opt;
        opt.tol = config.working_tol;
        opt.h_max = R.h;
        opt.n_periods = 1;
        if (config.periods > 1) {
            rt.period1_rerun_used = true;
            p1 = integrate_streamlines(field, seeds3, opt, config.threads);
            for (std::size_t i = 0; i < p1.size(); ++i) {
                const StreamlineResult& a = p1[i];
                const StreamlineResult& b = wt->streamlines[i];
                if (a.periods_completed >= 1 && b.periods_completed >= 1) {
                    const PlaneRecord& ra = a.records[0];
                    const PlaneRecord& rb = b.records[0];
                    if (ra.x1 != rb.x1 || ra.x2 != rb.x2 || ra.x3 != rb.x3 || ra.s != rb.s || ra.tau != rb.tau)
                        rt.period1_rerun_bitwise_identical = false;
                } else if ((a.periods_completed >= 1) != (b.periods_completed >= 1)) {
                    rt.period1_rerun_bitwise_identical = false;
                }
            }
            fw = &p1;
        }
        std::vector<std::size_t> idx;
        std::vector<std::array<double, 3>> back_seeds;
        for (std::size_t i = 0; i < fw->size(); ++i) {
            const StreamlineResult& r = (*fw)[i];
            if (r.status != StreamlineStatus::ok || r.backflow_encounter) continue;
            idx.push_back(i);
            back_seeds.push_back({r.records[0].x1, r.records[0].x2, r.records[0].x3});
        }
        rt.eligible = static_cast<long long>(idx.size());
        if (!back_seeds.empty()) {
            IntegratorOptions bo = opt;
            bo.sigma = -1;
            const std::vector<StreamlineResult> bw = integrate_streamlines(field, back_seeds, bo, config.threads);
            double sq = 0.0;
            for (std::size_t m = 0; m < bw.size(); ++m) {
                ++rt.backward_status[static_cast<int>(bw[m].status)];
                if (bw[m].status != StreamlineStatus::ok) continue;
                const PlaneRecord& q = bw[m].records[0];
                const std::array<double, 3>& s0 = seeds3[idx[m]];
                const double dd = std::sqrt((q.x1 - s0[0]) * (q.x1 - s0[0]) + (q.x2 - s0[1]) * (q.x2 - s0[1]) +
                                            (q.x3 - s0[2]) * (q.x3 - s0[2]));
                ++rt.backward_ok;
                rt.max_dist = std::max(rt.max_dist, dd);
                sq += dd * dd;
            }
            rt.valid = rt.backward_ok > 0;
            rt.rms_dist = rt.valid ? std::sqrt(sq / static_cast<double>(rt.backward_ok)) : 0.0;
        }
        rt.seconds = seconds_since(t0);
    }

    R.t_total = seconds_since(t_all);
    write_outputs(R);
    return R;
}

// ============================================================================
// stdout digest
// ============================================================================

inline void print_digest(const ClosureGateResult& R, std::FILE* out) {
    const ClosureGateConfig& c = R.config;
    std::fprintf(out, "closure_gate: field=%s grid=%d^3 h=%.6g", c.field.c_str(), c.n, R.h);
    if (is_gaussian_kind(c.field))
        std::fprintf(out, " sigma2=%.6g ell=%.6g seed=%llu", c.sigma2, c.ell, static_cast<unsigned long long>(c.seed));
    else
        std::fprintf(out, " eps=%.6g", c.eps);
    std::fprintf(out, " seeds=%zu threads=%d\n", R.seeds.size(), c.threads);
    std::fprintf(out, "  Y: mean=%.6e var=%.6e min=%.6e max=%.6e\n", R.field.mean, R.field.variance, R.field.min,
                 R.field.max);
    if (R.field.has_x3_control)
        std::fprintf(out, "  x3 control: raw_variance=%.6e applied_scale=%.6e\n", R.field.x3_control.raw_variance,
                     R.field.x3_control.applied_scale);
    std::fprintf(out, "  Darcy: mg_levels=%d (coarsest %d^3) pcg_rtol=%.1e max_iter=%d iterations=(%d,%d,%d) "
                      "rel_res=(%.2e,%.2e,%.2e) converged=%s\n",
                 R.mg_levels_used, coarsest_extent(c.n, R.mg_levels_used), c.pcg_rtol, R.pcg_max_iter_used,
                 R.darcy.corrector_results[0].iterations, R.darcy.corrector_results[1].iterations,
                 R.darcy.corrector_results[2].iterations, R.darcy.corrector_results[0].relative_projected_residual,
                 R.darcy.corrector_results[1].relative_projected_residual,
                 R.darcy.corrector_results[2].relative_projected_residual, R.darcy_converged ? "yes" : "NO");
    std::fprintf(out, "  G=(%.10e, %.10e, %.10e) mean_flux=(%.3e, %.3e, %.3e) div_max=%.2e\n", R.darcy.G[0],
                 R.darcy.G[1], R.darcy.G[2], R.darcy.achieved_mean_flux[0], R.darcy.achieved_mean_flux[1],
                 R.darcy.achieved_mean_flux[2], R.darcy.div_max_abs);
    if (!R.darcy_converged) {
        std::fprintf(out, "  Darcy PCG did not converge: no streamline statistics\n");
        return;
    }
    std::fprintf(out, "  spline gpu-vs-host prefilter rel=%.2e  g: min_g1=%.4e min|g|=%.4e max|g|=%.4e "
                      "backflow_fraction=%.4e v_rms=%.4e\n",
                 R.splines.potential_gpu_vs_host_rel, R.direction.min_g1, R.direction.min_abs_g, R.direction.max_abs_g,
                 R.direction.backflow_volume_fraction, R.direction.v_rms_spline);
    for (const ToleranceResult& t : R.tolerances) {
        const StatusCounts& k = t.periods.front().counts;
        std::fprintf(out, "  tol=%.0e  ok=%lld/%lld (seed_backflow=%lld, non_ok_fraction=%.3e, ok_with_backflow=%lld) "
                          "nfev/streamline-period=%.1f\n",
                     t.tol, k.ok, k.seeds, k.by_status[1], k.non_ok_fraction, k.ok_with_backflow,
                     t.totals.field_evaluations_per_streamline_period);
        for (const PeriodStatistics& p : t.periods) {
            if (p.n != 1 && p.n != c.periods) continue;
            if (!p.unweighted.valid) {
                std::fprintf(out, "    n=%d: no ok streamline\n", p.n);
                continue;
            }
            std::fprintf(out, "    n=%d: R=%.4e mean=(%.4e, %.4e) max|d|=%.4e rms|d|=%.4e | flux-weighted: R=%.4e "
                              "mean=(%.4e, %.4e) <tau>_w=%.10f\n",
                         p.n, p.unweighted.s.R, p.unweighted.s.mean[0], p.unweighted.s.mean[1], p.unweighted.s.max_abs,
                         p.unweighted.s.rms_abs, p.weighted.s.R, p.weighted.s.mean[0], p.weighted.s.mean[1],
                         p.mean_tau_weighted);
        }
        for (const LadderEntry& e : t.ladder)
            std::fprintf(out, "    ladder vs %.0e at n=%d: max=%.3e rms=%.3e (both_ok=%lld, status_differs=%lld)\n",
                         t.ladder_reference_tol, e.n, e.max_dist, e.rms_dist, e.both_ok, e.status_differs);
    }
    std::fprintf(out, "  round trip (tol=%.0e): eligible=%lld backward_ok=%lld max=%.3e rms=%.3e\n", R.round_trip.tol,
                 R.round_trip.eligible, R.round_trip.backward_ok, R.round_trip.max_dist, R.round_trip.rms_dist);
}

} // namespace closure_gate
} // namespace macroflow3d
