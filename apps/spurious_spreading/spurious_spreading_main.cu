/**
 * @file spurious_spreading_main.cu
 * @brief SF-32 N2a: `spurious_spreading` -- the measurement instrument of
 *        Lester et al. (2023) eqs. 34-36 (spurious transverse spreading of a
 *        tracker after one period on a surrogate flow with exact invariants).
 *
 * Documented-experiment instrument (like apps/closure_gate/closure_gate), NOT
 * a ctest entry. Nothing existing calls it. Subcommands:
 *
 *   spurious_spreading solve-labels --field
 * <lester2021|lester_brk|control2d|two_mode|generic3d|homogeneous>
 *       --n <N> --out <prefix> [--eps 0.25] [--max-iter 1000] [--tolerance 1e-8]
 *       [--epsilon 1e-6] [--anderson 1] [--newton 0] [--pcg-rtol 1e-10]
 *       [--mg-levels auto] [--no-timing]
 *     Frozen periodic stack (ev_ladder recipe), saves the final state as the
 *     label pair (label_routes.hpp).
 *
 *   spurious_spreading analytic-labels --pair <U|A|B|G> --n <N> --out <prefix>
 *       [--amplitude a] [--amplitude-b b] [--no-timing]
 *     Closed-form pair sampled at the cell centres (analytic_pair_g.hpp);
 *     default amplitudes: G 0.05, A 0.1, B 0.1 (b = 0.08), U 0.
 *
 *   spurious_spreading return-map --labels <prefix> --tracker <pseudo_symplectic|rk|pollock>
 *       --out <run_dir> [--seeds 8192] [--seed 20261006] [--tol-psi 1e-10]
 *       [--ds-ratio 0.5] [--tol 1e-6] [--dt-max-ratio 0.5] [--delta-ratio 1]
 *       [--max-panels 10000000] [--max-chunks 10000000] [--no-timing]
 *     Seeds: inject_box (SF-31 53-bit hash) on the face x1 = 0, identical for
 *     every tracker and level given --seed. One-period return map
 *     (return_map.cuh), outputs <run_dir>/seeds.csv and <run_dir>/summary.json
 *     (schema below; dag.json output_schema). <run_dir> is created if missing.
 *
 * Exit codes: 0 ok; 2 usage; 3 tracker not available (pollock until N2b) or
 * label pair refused by the usability guard (min_abs_c_grid <= 0.5);
 * 4 Darcy PCG did not converge in solve-labels; 1 exception.
 *
 * seeds.csv: header
 *   seed_index,x2_0,x3_0,status,delta_x2,delta_x3,delta_psi1,delta_psi2,tau,count,land_err
 * doubles with %.17g; for status != 0 the four deltas are written as `nan`
 * (tau, count and land_err are then those of the last accepted state).
 *
 * summary.json (key order fixed): field (copy of the labels metadata object),
 * tracker, level (pseudo_symplectic: {tol_psi, ds}; rk: {tol, dt_max}),
 * n_seeds, n_ok, status_counts ({"code": count}, codes ascending), stats
 * (over status == 0; population variance: delta_x2, delta_x3, delta_psi1,
 * delta_psi2: {mean, var, rms, max_abs}; tau: {mean, var, min, max}),
 * flux_weighted_tau_mean (weights c1(seed) of the spline velocity),
 * D22_uniform = var(delta_x2) / (2 mean tau), D33_uniform, D22_flux, D33_flux
 * (flux-weighted variance about the flux-weighted mean over twice the
 * flux-weighted mean tau; eq. 36, protocol numbers, not coefficients),
 * divergence_max_rel (null except pollock), min_abs_c_grid, landing
 * {max_err, max_iterations} (over status == 0), config (every CLI value),
 * wall_seconds (LAST; 0 with --no-timing, so two runs with the same inputs
 * are byte-identical).
 *
 * Determinism: one thread per particle, no atomics or reductions on the
 * device; host statistics are sequential sums in seed order.
 */

#include "apps/spurious_spreading/json_writer.hpp"
#include "apps/spurious_spreading/label_routes.hpp"
#include "apps/spurious_spreading/return_map.cuh"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Scalar.hpp"
#include "src/physics/particles/streamline_tracker/StreamlineTrackerCommon.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace spurious_spreading;
using macroflow3d::CudaContext;
using macroflow3d::DeviceBuffer;
using macroflow3d::real;
namespace stt = macroflow3d::physics::particles::streamline_tracker;

void print_usage() {
    std::fprintf(
        stderr,
        "usage:\n"
        "  spurious_spreading solve-labels --field <lester2021|lester_brk|control2d|two_mode|"
        "generic3d|homogeneous>\n"
        "      --n <N (power of 2)> --out <prefix> [--eps 0.25] [--max-iter 1000] [--tolerance "
        "1e-8]\n"
        "      [--epsilon 1e-6] [--anderson 1] [--newton 0] [--pcg-rtol 1e-10] [--mg-levels auto]\n"
        "      [--no-timing]\n"
        "  spurious_spreading analytic-labels --pair <U|A|B|G> --n <N (power of 2)> --out "
        "<prefix>\n"
        "      [--amplitude a] [--amplitude-b b] [--no-timing]\n"
        "  spurious_spreading return-map --labels <prefix> --tracker "
        "<pseudo_symplectic|rk|pollock>\n"
        "      --out <run_dir> [--seeds 8192] [--seed 20261006] [--tol-psi 1e-10] [--ds-ratio "
        "0.5]\n"
        "      [--tol 1e-6] [--dt-max-ratio 0.5] [--delta-ratio 1] [--max-panels 10000000]\n"
        "      [--max-chunks 10000000] [--no-timing]\n"
        "exit codes: 0 ok, 2 usage, 3 tracker not available / labels refused (min_abs_c_grid <= "
        "0.5),\n"
        "            4 Darcy PCG not converged (solve-labels), 1 exception\n");
}

// ===========================================================================
// return-map options
// ===========================================================================

enum class TrackerKind { pseudo_symplectic, rk, pollock };

const char* tracker_name(TrackerKind k) {
    switch (k) {
    case TrackerKind::pseudo_symplectic:
        return "pseudo_symplectic";
    case TrackerKind::rk:
        return "rk";
    case TrackerKind::pollock:
        return "pollock";
    }
    return "?";
}

struct RunOptions {
    std::string labels;
    TrackerKind tracker = TrackerKind::pseudo_symplectic;
    double tol_psi = 1e-10;
    double ds_ratio = 0.5;
    double tol = 1e-6;
    double dt_max_ratio = 0.5;
    double delta_ratio = 1.0;
    long long seeds = 8192;
    unsigned long long seed = 20261006ULL;
    long long max_panels = 10000000LL;
    long long max_chunks = 10000000LL;
    bool no_timing = false;
    std::string out;
};

RunOptions parse_run_options(int argc, char** argv) {
    RunOptions o;
    bool have_labels = false, have_tracker = false, have_out = false;
    for (int a = 1; a < argc; ++a) {
        const std::string key = argv[a];
        if (key == "--no-timing") {
            o.no_timing = true;
            continue;
        }
        if (a + 1 >= argc)
            throw UsageError("missing value for " + key);
        const std::string val = argv[++a];
        if (key == "--labels") {
            o.labels = val;
            have_labels = true;
        } else if (key == "--tracker") {
            if (val == "pseudo_symplectic")
                o.tracker = TrackerKind::pseudo_symplectic;
            else if (val == "rk")
                o.tracker = TrackerKind::rk;
            else if (val == "pollock")
                o.tracker = TrackerKind::pollock;
            else
                throw UsageError("unknown --tracker '" + val + "'");
            have_tracker = true;
        } else if (key == "--tol-psi") {
            o.tol_psi = parse_double_arg(key, val);
        } else if (key == "--ds-ratio") {
            o.ds_ratio = parse_double_arg(key, val);
        } else if (key == "--tol") {
            o.tol = parse_double_arg(key, val);
        } else if (key == "--dt-max-ratio") {
            o.dt_max_ratio = parse_double_arg(key, val);
        } else if (key == "--delta-ratio") {
            o.delta_ratio = parse_double_arg(key, val);
        } else if (key == "--seeds") {
            o.seeds = parse_int_arg(key, val);
        } else if (key == "--seed") {
            o.seed = parse_u64_arg(key, val);
        } else if (key == "--max-panels") {
            o.max_panels = parse_int_arg(key, val);
        } else if (key == "--max-chunks") {
            o.max_chunks = parse_int_arg(key, val);
        } else if (key == "--out") {
            o.out = val;
            have_out = true;
        } else {
            throw UsageError("unknown option " + key);
        }
    }
    if (!have_labels || o.labels.empty())
        throw UsageError("return-map: --labels <prefix> is required");
    if (!have_tracker)
        throw UsageError("return-map: --tracker is required");
    if (!have_out || o.out.empty())
        throw UsageError("return-map: --out <run_dir> is required");
    if (o.seeds < 1 || o.seeds > 100000000LL)
        throw UsageError("--seeds must be in [1, 1e8]");
    if (!(o.tol_psi > 0.0))
        throw UsageError("--tol-psi must be > 0");
    if (!(o.ds_ratio > 0.0))
        throw UsageError("--ds-ratio must be > 0");
    if (!(o.tol > 0.0))
        throw UsageError("--tol must be > 0");
    if (!(o.dt_max_ratio > 0.0))
        throw UsageError("--dt-max-ratio must be > 0");
    if (!(o.delta_ratio >= 1.0))
        throw UsageError("--delta-ratio must be >= 1");
    if (o.max_panels < 1 || o.max_chunks < 1)
        throw UsageError("--max-panels and --max-chunks must be >= 1");
    return o;
}

JVal run_config_json(const RunOptions& o) {
    return jobj({{"labels", o.labels},
                 {"tracker", tracker_name(o.tracker)},
                 {"tol_psi", o.tol_psi},
                 {"ds_ratio", o.ds_ratio},
                 {"tol", o.tol},
                 {"dt_max_ratio", o.dt_max_ratio},
                 {"delta_ratio", o.delta_ratio},
                 {"seeds", o.seeds},
                 {"seed", o.seed},
                 {"max_panels", o.max_panels},
                 {"max_chunks", o.max_chunks},
                 {"no_timing", o.no_timing},
                 {"out", o.out},
                 {"max_newton_iter", 8},
                 {"trust_factor", 1.0},
                 {"min_cross_norm", 0.0},
                 {"min_cross_sin2", static_cast<double>(stt::kDefaultMinCrossSin2)},
                 {"rk_min_step", 1e-14},
                 {"rk_max_steps_per_call", 1000000},
                 {"rk_initial_step_ratio_of_dt_max", 0.1},
                 {"landing_tol", static_cast<double>(kLandingTol)},
                 {"landing_max_trials", kMaxLandingTrials},
                 {"target_x1", 1.0}});
}

// ===========================================================================
// Tracker dispatch
// ===========================================================================

/// Thrown by a tracker that is not built into this instrument (exit code 3).
struct TrackerUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Device output buffers of one return map (n entries each).
struct DeviceOutputs {
    DeviceBuffer<uint8_t> status;
    DeviceBuffer<real> x1u, x2u, x3u, tau, dpsi1, dpsi2;
    DeviceBuffer<unsigned long long> count;
    DeviceBuffer<int> land_iters;

    explicit DeviceOutputs(std::size_t n)
        : status(n), x1u(n), x2u(n), x3u(n), tau(n), dpsi1(n), dpsi2(n), count(n), land_iters(n) {}

    ReturnMapOut view() {
        return ReturnMapOut{status.data(), x1u.data(),        x2u.data(),   x3u.data(),  tau.data(),
                            count.data(),  land_iters.data(), dpsi1.data(), dpsi2.data()};
    }
};

/// Inputs every tracker sees.
struct TrackerInputs {
    const CudaContext& ctx;
    const SplineLabels& labels;
    const RunOptions& o;
    const real* d_x2_0;
    const real* d_x3_0;
    int n;
};

/// What a tracker reports besides the per-seed device outputs.
struct TrackerLevel {
    JVal level;
    JVal divergence_max_rel; ///< null unless pollock
};

unsigned grid_for(int n) {
    return static_cast<unsigned>((n + kReturnMapBlock - 1) / kReturnMapBlock);
}

TrackerLevel run_pseudo_symplectic(const TrackerInputs& in, DeviceOutputs& out) {
    const double h = in.labels.h();
    const real ds = static_cast<real>(in.o.ds_ratio * h);
    stt::PseudoSymplecticParams prm{static_cast<real>(in.o.tol_psi), 8, static_cast<real>(1.0),
                                    static_cast<real>(0.0), stt::kDefaultMinCrossSin2};
    ps_return_map_kernel<<<grid_for(in.n), kReturnMapBlock, 0, in.ctx.cuda_stream()>>>(
        in.labels.pair(), prm, ds, in.o.max_panels, static_cast<real>(1.0), in.d_x2_0, in.d_x3_0,
        in.n, out.view());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    in.ctx.synchronize();
    TrackerLevel lv;
    lv.level = jobj({{"tol_psi", in.o.tol_psi}, {"ds", static_cast<double>(ds)}});
    lv.divergence_max_rel = JVal(nullptr);
    return lv;
}

TrackerLevel run_rk(const TrackerInputs& in, DeviceOutputs& out) {
    const double h = in.labels.h();
    const real dt_max = static_cast<real>(in.o.dt_max_ratio * h);
    stt::ReferenceRkParams prm{static_cast<real>(in.o.tol), dt_max, static_cast<real>(1e-14),
                               1000000};
    rk_return_map_kernel<<<grid_for(in.n), kReturnMapBlock, 0, in.ctx.cuda_stream()>>>(
        in.labels.pair(), prm, dt_max, in.o.max_chunks, static_cast<real>(1.0), in.d_x2_0,
        in.d_x3_0, in.n, out.view());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    in.ctx.synchronize();
    TrackerLevel lv;
    lv.level = jobj({{"tol", in.o.tol}, {"dt_max", static_cast<double>(dt_max)}});
    lv.divergence_max_rel = JVal(nullptr);
    return lv;
}

/// Pollock return map: implemented by DAG node N2b (StokesFaceVelocity on the
/// grid Delta = delta_ratio h + PollockTracker). Its level is
/// {delta_ratio, n_cells, delta} and it fills divergence_max_rel.
TrackerLevel run_pollock(const TrackerInputs& in, DeviceOutputs& out) {
    (void)in;
    (void)out;
    throw TrackerUnavailable("pollock tracker not available until N2b");
}

TrackerLevel run_tracker(TrackerKind k, const TrackerInputs& in, DeviceOutputs& out) {
    switch (k) {
    case TrackerKind::pseudo_symplectic:
        return run_pseudo_symplectic(in, out);
    case TrackerKind::rk:
        return run_rk(in, out);
    case TrackerKind::pollock:
        return run_pollock(in, out);
    }
    throw std::logic_error("run_tracker: invalid tracker");
}

// ===========================================================================
// Host statistics
// ===========================================================================

template <class T> std::vector<T> download(const DeviceBuffer<T>& d, std::size_t n) {
    std::vector<T> h(n);
    if (n > 0) {
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(h.data(), d.data(), n * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return h;
}

const double kNaN = std::numeric_limits<double>::quiet_NaN();

/// {mean, var (population), rms, max_abs} over the masked entries (NaN if none).
JVal moments_json(const std::vector<double>& v, const std::vector<uint8_t>& ok) {
    double sum = 0.0, sq = 0.0, mx = 0.0;
    long long m = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (!ok[i])
            continue;
        sum += v[i];
        sq += v[i] * v[i];
        mx = std::fmax(mx, std::fabs(v[i]));
        ++m;
    }
    if (m == 0)
        return jobj({{"mean", kNaN}, {"var", kNaN}, {"rms", kNaN}, {"max_abs", kNaN}});
    const double mean = sum / static_cast<double>(m);
    double var = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (ok[i])
            var += (v[i] - mean) * (v[i] - mean);
    }
    var /= static_cast<double>(m);
    return jobj({{"mean", mean},
                 {"var", var},
                 {"rms", std::sqrt(sq / static_cast<double>(m))},
                 {"max_abs", mx}});
}

double masked_mean(const std::vector<double>& v, const std::vector<uint8_t>& ok) {
    double s = 0.0;
    long long m = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (ok[i]) {
            s += v[i];
            ++m;
        }
    }
    return m ? s / static_cast<double>(m) : kNaN;
}

double masked_var(const std::vector<double>& v, const std::vector<uint8_t>& ok) {
    const double mean = masked_mean(v, ok);
    double s = 0.0;
    long long m = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (ok[i]) {
            s += (v[i] - mean) * (v[i] - mean);
            ++m;
        }
    }
    return m ? s / static_cast<double>(m) : kNaN;
}

double weighted_mean(const std::vector<double>& v, const std::vector<double>& w,
                     const std::vector<uint8_t>& ok) {
    double s = 0.0, ws = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (ok[i]) {
            s += w[i] * v[i];
            ws += w[i];
        }
    }
    return ws != 0.0 ? s / ws : kNaN;
}

double weighted_var(const std::vector<double>& v, const std::vector<double>& w,
                    const std::vector<uint8_t>& ok) {
    const double mean = weighted_mean(v, w, ok);
    double s = 0.0, ws = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (ok[i]) {
            s += w[i] * (v[i] - mean) * (v[i] - mean);
            ws += w[i];
        }
    }
    return ws != 0.0 ? s / ws : kNaN;
}

// ===========================================================================
// Files
// ===========================================================================

void write_text(const std::string& path, const std::string& text) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr)
        throw std::runtime_error("cannot open '" + path + "' for writing");
    const std::size_t w = std::fwrite(text.data(), 1, text.size(), f);
    const int rc = std::fclose(f);
    if (w != text.size() || rc != 0)
        throw std::runtime_error("failed writing '" + path + "'");
}

void append_g17(std::string& s, double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    s += buf;
}

// ===========================================================================
// return-map
// ===========================================================================

int run_return_map(int argc, char** argv) {
    const RunOptions o = parse_run_options(argc, argv);
    const auto t0 = std::chrono::steady_clock::now();

    LoadedLabels L = read_labels(o.labels);
    CudaContext ctx(0);
    SplineLabels labels;
    labels.build(ctx, L.n, L.u1, L.u2, L.gbar1, L.gbar2);
    const double min_c = labels.min_abs_c_grid(ctx);

    std::printf("==== spurious_spreading return-map (SF-32 N2a) ====\n");
    const JVal* route = L.meta.find("route");
    std::printf(
        "labels = %s  route = %s  n = %d  h = %.17g  tracker = %s  seeds = %lld  seed = %llu\n",
        o.labels.c_str(), (route && route->is_string()) ? route->as_string().c_str() : "?", L.n,
        L.h, tracker_name(o.tracker), o.seeds, o.seed);
    std::printf("  min_abs_c_grid (spline, cell centres) = %.17g\n", min_c);
    if (!(min_c > 0.5)) {
        std::fprintf(stderr,
                     "return-map: min_abs_c_grid = %.17g <= 0.5: label pair refused by the "
                     "pre-registered usability guard (understanding.md 3.5)\n",
                     min_c);
        return 3;
    }

    // ---- seeds on the face x1 = 0 (SF-31 inject_box; identical for every tracker) ----
    const int n = static_cast<int>(o.seeds);
    const std::size_t nn = static_cast<std::size_t>(n);
    DeviceBuffer<real> px(nn), py(nn), pz(nn);
    DeviceBuffer<uint8_t> pst(nn);
    DeviceBuffer<int32_t> pwx(nn), pwy(nn), pwz(nn);
    macroflow3d::physics::particles::ParticlesSoA<real> soa;
    soa.x = px.data();
    soa.y = py.data();
    soa.z = pz.data();
    soa.n = n;
    soa.status = pst.data();
    soa.wrapX = pwx.data();
    soa.wrapY = pwy.data();
    soa.wrapZ = pwz.data();
    stt::inject_box(ctx.cuda_stream(), soa, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, n, o.seed);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    DeviceBuffer<real> c1_dev(nn);
    seed_c1_kernel<<<grid_for(n), kReturnMapBlock, 0, ctx.cuda_stream()>>>(
        labels.pair(), py.data(), pz.data(), n, c1_dev.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    ctx.synchronize();
    const std::vector<real> x1_0 = download(px, nn);
    const std::vector<real> x2_0 = download(py, nn);
    const std::vector<real> x3_0 = download(pz, nn);
    const std::vector<real> c1 = download(c1_dev, nn);
    for (std::size_t p = 0; p < nn; ++p) {
        if (x1_0[p] != 0.0)
            throw std::logic_error("return-map: a seed is not on the face x1 = 0");
    }

    // ---- the tracker ----
    DeviceOutputs dout(nn);
    TrackerLevel lv;
    try {
        lv = run_tracker(o.tracker, TrackerInputs{ctx, labels, o, py.data(), pz.data(), n}, dout);
    } catch (const TrackerUnavailable& e) {
        std::fprintf(stderr, "return-map: %s\n", e.what());
        return 3;
    }
    const std::vector<uint8_t> status = download(dout.status, nn);
    const std::vector<real> x1u = download(dout.x1u, nn);
    const std::vector<real> x2u = download(dout.x2u, nn);
    const std::vector<real> x3u = download(dout.x3u, nn);
    const std::vector<real> tau = download(dout.tau, nn);
    const std::vector<real> dpsi1 = download(dout.dpsi1, nn);
    const std::vector<real> dpsi2 = download(dout.dpsi2, nn);
    const std::vector<unsigned long long> count = download(dout.count, nn);
    const std::vector<int> land_iters = download(dout.land_iters, nn);

    // ---- per-seed quantities ----
    std::vector<uint8_t> ok(nn);
    std::vector<double> dx2(nn), dx3(nn), land_err(nn);
    std::map<int, long long> status_counts;
    long long n_ok = 0;
    double max_land_err = 0.0;
    int max_land_iters = 0;
    for (std::size_t p = 0; p < nn; ++p) {
        ok[p] = status[p] == stt::kStatusActive ? 1 : 0;
        ++status_counts[static_cast<int>(status[p])];
        dx2[p] = x2u[p] - x2_0[p];
        dx3[p] = x3u[p] - x3_0[p];
        land_err[p] = std::fabs(x1u[p] - 1.0);
        if (ok[p]) {
            ++n_ok;
            max_land_err = std::fmax(max_land_err, land_err[p]);
            if (land_iters[p] > max_land_iters)
                max_land_iters = land_iters[p];
        }
    }

    // ---- seeds.csv ----
    mkdir_p(o.out);
    std::string csv = "seed_index,x2_0,x3_0,status,delta_x2,delta_x3,delta_psi1,delta_psi2,tau,"
                      "count,land_err\n";
    csv.reserve(csv.size() + nn * 220);
    for (std::size_t p = 0; p < nn; ++p) {
        char head[32];
        std::snprintf(head, sizeof(head), "%zu,", p);
        csv += head;
        append_g17(csv, x2_0[p]);
        csv += ',';
        append_g17(csv, x3_0[p]);
        std::snprintf(head, sizeof(head), ",%d,", static_cast<int>(status[p]));
        csv += head;
        if (ok[p]) {
            append_g17(csv, dx2[p]);
            csv += ',';
            append_g17(csv, dx3[p]);
            csv += ',';
            append_g17(csv, dpsi1[p]);
            csv += ',';
            append_g17(csv, dpsi2[p]);
        } else {
            csv += "nan,nan,nan,nan";
        }
        csv += ',';
        append_g17(csv, tau[p]);
        std::snprintf(head, sizeof(head), ",%llu,", count[p]);
        csv += head;
        append_g17(csv, land_err[p]);
        csv += '\n';
    }
    write_text(o.out + "/seeds.csv", csv);

    // ---- summary.json ----
    const std::vector<double> c1d(c1.begin(), c1.end());
    const std::vector<double> taud(tau.begin(), tau.end());
    const std::vector<double> dp1(dpsi1.begin(), dpsi1.end());
    const std::vector<double> dp2(dpsi2.begin(), dpsi2.end());
    double tau_min = kNaN, tau_max = kNaN;
    for (std::size_t p = 0; p < nn; ++p) {
        if (!ok[p])
            continue;
        if (std::isnan(tau_min) || taud[p] < tau_min)
            tau_min = taud[p];
        if (std::isnan(tau_max) || taud[p] > tau_max)
            tau_max = taud[p];
    }
    const double tau_mean = masked_mean(taud, ok);
    const double tau_f = weighted_mean(taud, c1d, ok);
    const double var2 = masked_var(dx2, ok), var3 = masked_var(dx3, ok);
    const double var2_f = weighted_var(dx2, c1d, ok), var3_f = weighted_var(dx3, c1d, ok);

    JVal sc = JVal::object();
    for (const auto& kv : status_counts)
        sc[std::to_string(kv.first)] = kv.second;

    JVal stats = JVal::object();
    stats["delta_x2"] = moments_json(dx2, ok);
    stats["delta_x3"] = moments_json(dx3, ok);
    stats["delta_psi1"] = moments_json(dp1, ok);
    stats["delta_psi2"] = moments_json(dp2, ok);
    stats["tau"] = jobj(
        {{"mean", tau_mean}, {"var", masked_var(taud, ok)}, {"min", tau_min}, {"max", tau_max}});

    JVal sum;
    sum["field"] = L.meta;
    sum["tracker"] = tracker_name(o.tracker);
    sum["level"] = lv.level;
    sum["n_seeds"] = static_cast<long long>(n);
    sum["n_ok"] = n_ok;
    sum["status_counts"] = sc;
    sum["stats"] = stats;
    sum["flux_weighted_tau_mean"] = tau_f;
    sum["D22_uniform"] = var2 / (2.0 * tau_mean);
    sum["D33_uniform"] = var3 / (2.0 * tau_mean);
    sum["D22_flux"] = var2_f / (2.0 * tau_f);
    sum["D33_flux"] = var3_f / (2.0 * tau_f);
    sum["divergence_max_rel"] = lv.divergence_max_rel;
    sum["min_abs_c_grid"] = min_c;
    sum["landing"] =
        jobj({{"max_err", n_ok ? max_land_err : kNaN}, {"max_iterations", max_land_iters}});
    sum["config"] = run_config_json(o);
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    sum["wall_seconds"] = o.no_timing ? 0.0 : wall;
    write_text(o.out + "/summary.json", sum.dump(2) + "\n");

    // ---- console ----
    std::printf("  level:");
    for (std::size_t i = 0; i < lv.level.keys().size(); ++i) {
        const JVal& v = lv.level.at(i);
        std::printf("  %s = %.17g", lv.level.keys()[i].c_str(),
                    v.is_number() ? v.as_double() : kNaN);
    }
    std::printf("\n");
    std::printf("  n_ok = %lld / %d", n_ok, n);
    for (const auto& kv : status_counts)
        std::printf("  [status %d: %lld]", kv.first, kv.second);
    std::printf("\n");
    const JVal& m2 = *stats.find("delta_x2");
    const JVal& m3 = *stats.find("delta_x3");
    const JVal& mp1 = *stats.find("delta_psi1");
    const JVal& mp2 = *stats.find("delta_psi2");
    auto g = [](const JVal& j, const char* k) {
        const JVal* v = j.find(k);
        return (v && v->is_number()) ? v->as_double() : kNaN;
    };
    std::printf("  delta_x2:   mean %.6e  var %.6e  rms %.6e  max %.6e\n", g(m2, "mean"),
                g(m2, "var"), g(m2, "rms"), g(m2, "max_abs"));
    std::printf("  delta_x3:   mean %.6e  var %.6e  rms %.6e  max %.6e\n", g(m3, "mean"),
                g(m3, "var"), g(m3, "rms"), g(m3, "max_abs"));
    std::printf("  delta_psi1: mean %.6e  rms %.6e  max %.6e\n", g(mp1, "mean"), g(mp1, "rms"),
                g(mp1, "max_abs"));
    std::printf("  delta_psi2: mean %.6e  rms %.6e  max %.6e\n", g(mp2, "mean"), g(mp2, "rms"),
                g(mp2, "max_abs"));
    std::printf("  tau: mean %.17g  min %.17g  max %.17g  flux-weighted mean %.17g\n", tau_mean,
                tau_min, tau_max, tau_f);
    std::printf("  eq.36 protocol numbers: D22_u %.6e  D33_u %.6e  D22_f %.6e  D33_f %.6e\n",
                var2 / (2.0 * tau_mean), var3 / (2.0 * tau_mean), var2_f / (2.0 * tau_f),
                var3_f / (2.0 * tau_f));
    std::printf("  landing: max |x1_u - 1| = %.3e  max trials = %d\n", max_land_err,
                max_land_iters);
    std::printf("  outputs: %s/seeds.csv, %s/summary.json  (wall %.3f s)\n", o.out.c_str(),
                o.out.c_str(), wall);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }
    const std::string cmd = argv[1];
    try {
        if (cmd == "solve-labels")
            return run_solve_labels(argc - 1, argv + 1);
        if (cmd == "analytic-labels")
            return run_analytic_labels(argc - 1, argv + 1);
        if (cmd == "return-map")
            return run_return_map(argc - 1, argv + 1);
        if (cmd == "--help" || cmd == "-h") {
            print_usage();
            return 0;
        }
        std::fprintf(stderr, "error: unknown subcommand '%s'\n", cmd.c_str());
        print_usage();
        return 2;
    } catch (const UsageError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "spurious_spreading: exception: %s\n", e.what());
        return 1;
    }
}
