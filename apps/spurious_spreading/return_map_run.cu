/**
 * @file return_map_run.cu
 * @brief SF-32 N2a/N2b: `return-map` subcommand of `spurious_spreading` as
 *        library code (see return_map_run.hpp). Moved verbatim in behaviour
 *        from spurious_spreading_main.cu (N2a) and extended by N2b with
 *        the Pollock tracker and the absolute RK `--dt-max` (decision D-2).
 */

#include "apps/spurious_spreading/return_map_run.hpp"

#include "apps/spurious_spreading/return_map.cuh"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/particles/streamline_tracker/PollockTracker.cuh"
#include "src/physics/particles/streamline_tracker/StokesFaceVelocity.cuh"
#include "src/physics/particles/streamline_tracker/StreamlineTrackerCommon.cuh"
#include "src/runtime/cuda_check.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>

namespace spurious_spreading {

using macroflow3d::CudaContext;
using macroflow3d::DeviceBuffer;
using macroflow3d::real;
namespace stt = macroflow3d::physics::particles::streamline_tracker;

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

// ===========================================================================
// Options
// ===========================================================================

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
        } else if (key == "--dt-max") {
            o.dt_max = parse_double_arg(key, val);
        } else if (key == "--delta-ratio") {
            o.delta_ratio = parse_int_arg(key, val);
        } else if (key == "--seeds") {
            o.seeds = parse_int_arg(key, val);
        } else if (key == "--seed") {
            o.seed = parse_u64_arg(key, val);
        } else if (key == "--max-panels") {
            o.max_panels = parse_int_arg(key, val);
        } else if (key == "--max-chunks") {
            o.max_chunks = parse_int_arg(key, val);
        } else if (key == "--max-cells") {
            o.max_cells = parse_int_arg(key, val);
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
    if (!(o.dt_max > 0.0) || !std::isfinite(o.dt_max))
        throw UsageError("--dt-max must be finite and > 0");
    if (o.delta_ratio < 1)
        throw UsageError("--delta-ratio must be an integer >= 1");
    if (o.max_panels < 1 || o.max_chunks < 1)
        throw UsageError("--max-panels and --max-chunks must be >= 1");
    if (o.max_cells < 1 || o.max_cells > 2147483647LL)
        throw UsageError("--max-cells must be in [1, 2147483647]");
    return o;
}

JVal run_config_json(const RunOptions& o) {
    return jobj({{"labels", o.labels},
                 {"tracker", tracker_name(o.tracker)},
                 {"tol_psi", o.tol_psi},
                 {"ds_ratio", o.ds_ratio},
                 {"tol", o.tol},
                 {"dt_max", o.dt_max},
                 {"rk_chunk", o.dt_max},
                 {"delta_ratio", o.delta_ratio},
                 {"seeds", o.seeds},
                 {"seed", o.seed},
                 {"max_panels", o.max_panels},
                 {"max_chunks", o.max_chunks},
                 {"max_cells", o.max_cells},
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

namespace {

// ===========================================================================
// Tracker dispatch
// ===========================================================================

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

template <class T> std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> h(n);
    if (n > 0) {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return h;
}

template <class T> std::vector<T> download(const DeviceBuffer<T>& d, std::size_t n) {
    return download(d.data(), n);
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
    // D-2: dt_max is an absolute time and the crossing-detection chunk equals it.
    const real dt_max = static_cast<real>(in.o.dt_max);
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

/// Pollock return map (N2b): StokesFaceVelocity on the grid Delta = m h
/// (n = N / m cells per axis), then the N1 core per particle
/// (return_map.cuh, pollock_return_map_one).
TrackerLevel run_pollock(const TrackerInputs& in, DeviceOutputs& out) {
    const int N = in.labels.n();
    const long long m = in.o.delta_ratio;
    if (m < 1 || m > N || N % m != 0) // also checked before any work in compute_return_map
        throw UsageError("--delta-ratio must divide the label grid N");
    const int nc = static_cast<int>(N / m);
    const real delta = static_cast<real>(static_cast<double>(m) * in.labels.h());
    const macroflow3d::Grid3D grid(nc, nc, nc, delta, delta, delta);

    stt::StokesFaceFluxWorkspace ws;
    stt::compute_stokes_face_fluxes(in.ctx.cuda_stream(), in.labels.pair(), grid, ws);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    in.ctx.synchronize();
    const stt::PeriodicFaceFluxView fv = ws.view();

    // Divergence diagnostic on the downloaded faces (not a hot path).
    const std::size_t cells = grid.num_cells();
    const std::vector<real> hu = download(fv.u, cells);
    const std::vector<real> hv = download(fv.v, cells);
    const std::vector<real> hw = download(fv.w, cells);
    stt::PeriodicFaceFluxView hview = fv;
    hview.u = hu.data();
    hview.v = hv.data();
    hview.w = hw.data();
    const double div_rel = static_cast<double>(stt::max_relative_divergence(hview));

    const stt::PollockParams prm{static_cast<int>(in.o.max_cells)};
    pollock_return_map_kernel<<<grid_for(in.n), kReturnMapBlock, 0, in.ctx.cuda_stream()>>>(
        in.labels.pair(), fv, prm, static_cast<real>(1.0), in.d_x2_0, in.d_x3_0, in.n, out.view());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    in.ctx.synchronize();

    TrackerLevel lv;
    lv.level = jobj({{"delta_ratio", m}, {"n_cells", nc}, {"delta", static_cast<double>(delta)}});
    lv.divergence_max_rel = JVal(div_rel);
    return lv;
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
// Host statistics (sequential sums in seed order)
// ===========================================================================

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

} // namespace

// ===========================================================================
// One run
// ===========================================================================

ReturnMapRun compute_return_map(const CudaContext& ctx, const LoadedLabels& L,
                                const RunOptions& o) {
    const auto t0 = std::chrono::steady_clock::now();
    if (o.tracker == TrackerKind::pollock &&
        (o.delta_ratio < 1 || o.delta_ratio > L.n || L.n % o.delta_ratio != 0)) {
        throw UsageError("--delta-ratio " + std::to_string(o.delta_ratio) +
                         " does not divide the label grid N = " + std::to_string(L.n));
    }
    ReturnMapRun R;
    SplineLabels labels;
    labels.build(ctx, L.n, L.u1, L.u2, L.gbar1, L.gbar2);
    R.min_abs_c_grid = labels.min_abs_c_grid(ctx);
    if (!(R.min_abs_c_grid > 0.5)) {
        R.exit_code = 3;
        return R;
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
    const TrackerLevel lv =
        run_tracker(o.tracker, TrackerInputs{ctx, labels, o, py.data(), pz.data(), n}, dout);
    R.level = lv.level;
    R.divergence_max_rel = lv.divergence_max_rel;
    R.status = download(dout.status, nn);
    const std::vector<real> x1u = download(dout.x1u, nn);
    const std::vector<real> x2u = download(dout.x2u, nn);
    const std::vector<real> x3u = download(dout.x3u, nn);
    const std::vector<real> tau = download(dout.tau, nn);
    const std::vector<real> dpsi1 = download(dout.dpsi1, nn);
    const std::vector<real> dpsi2 = download(dout.dpsi2, nn);
    R.count = download(dout.count, nn);
    R.land_iters = download(dout.land_iters, nn);

    R.x2_0.assign(x2_0.begin(), x2_0.end());
    R.x3_0.assign(x3_0.begin(), x3_0.end());
    R.c1.assign(c1.begin(), c1.end());
    R.x1u.assign(x1u.begin(), x1u.end());
    R.x2u.assign(x2u.begin(), x2u.end());
    R.x3u.assign(x3u.begin(), x3u.end());
    R.tau.assign(tau.begin(), tau.end());
    R.delta_psi1.assign(dpsi1.begin(), dpsi1.end());
    R.delta_psi2.assign(dpsi2.begin(), dpsi2.end());

    // ---- per-seed quantities ----
    std::vector<uint8_t> ok(nn);
    R.delta_x2.resize(nn);
    R.delta_x3.resize(nn);
    R.land_err.resize(nn);
    std::map<int, long long> status_counts;
    long long n_ok = 0;
    double max_land_err = 0.0;
    int max_land_iters = 0;
    for (std::size_t p = 0; p < nn; ++p) {
        ok[p] = R.status[p] == stt::kStatusActive ? 1 : 0;
        ++status_counts[static_cast<int>(R.status[p])];
        R.delta_x2[p] = R.x2u[p] - R.x2_0[p];
        R.delta_x3[p] = R.x3u[p] - R.x3_0[p];
        R.land_err[p] = std::fabs(R.x1u[p] - 1.0);
        if (ok[p]) {
            ++n_ok;
            max_land_err = std::fmax(max_land_err, R.land_err[p]);
            if (R.land_iters[p] > max_land_iters)
                max_land_iters = R.land_iters[p];
        }
    }

    // ---- summary object (key order fixed; dag.json output_schema) ----
    double tau_min = kNaN, tau_max = kNaN;
    for (std::size_t p = 0; p < nn; ++p) {
        if (!ok[p])
            continue;
        if (std::isnan(tau_min) || R.tau[p] < tau_min)
            tau_min = R.tau[p];
        if (std::isnan(tau_max) || R.tau[p] > tau_max)
            tau_max = R.tau[p];
    }
    const double tau_mean = masked_mean(R.tau, ok);
    const double tau_f = weighted_mean(R.tau, R.c1, ok);
    const double var2 = masked_var(R.delta_x2, ok), var3 = masked_var(R.delta_x3, ok);
    const double var2_f = weighted_var(R.delta_x2, R.c1, ok);
    const double var3_f = weighted_var(R.delta_x3, R.c1, ok);

    JVal sc = JVal::object();
    for (const auto& kv : status_counts)
        sc[std::to_string(kv.first)] = kv.second;

    JVal stats = JVal::object();
    stats["delta_x2"] = moments_json(R.delta_x2, ok);
    stats["delta_x3"] = moments_json(R.delta_x3, ok);
    stats["delta_psi1"] = moments_json(R.delta_psi1, ok);
    stats["delta_psi2"] = moments_json(R.delta_psi2, ok);
    stats["tau"] = jobj(
        {{"mean", tau_mean}, {"var", masked_var(R.tau, ok)}, {"min", tau_min}, {"max", tau_max}});

    JVal& sum = R.summary;
    sum["field"] = L.meta;
    sum["tracker"] = tracker_name(o.tracker);
    sum["level"] = R.level;
    sum["n_seeds"] = static_cast<long long>(n);
    sum["n_ok"] = n_ok;
    sum["status_counts"] = sc;
    sum["stats"] = stats;
    sum["flux_weighted_tau_mean"] = tau_f;
    sum["D22_uniform"] = var2 / (2.0 * tau_mean);
    sum["D33_uniform"] = var3 / (2.0 * tau_mean);
    sum["D22_flux"] = var2_f / (2.0 * tau_f);
    sum["D33_flux"] = var3_f / (2.0 * tau_f);
    sum["divergence_max_rel"] = R.divergence_max_rel;
    sum["min_abs_c_grid"] = R.min_abs_c_grid;
    sum["landing"] =
        jobj({{"max_err", n_ok ? max_land_err : kNaN}, {"max_iterations", max_land_iters}});
    sum["config"] = run_config_json(o);
    R.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    sum["wall_seconds"] = o.no_timing ? 0.0 : R.wall_seconds;
    return R;
}

std::string seeds_csv_text(const ReturnMapRun& R) {
    const std::size_t nn = R.status.size();
    std::string csv = "seed_index,x2_0,x3_0,status,delta_x2,delta_x3,delta_psi1,delta_psi2,tau,"
                      "count,land_err\n";
    csv.reserve(csv.size() + nn * 220);
    for (std::size_t p = 0; p < nn; ++p) {
        const bool ok = R.status[p] == stt::kStatusActive;
        char head[32];
        std::snprintf(head, sizeof(head), "%zu,", p);
        csv += head;
        append_g17(csv, R.x2_0[p]);
        csv += ',';
        append_g17(csv, R.x3_0[p]);
        std::snprintf(head, sizeof(head), ",%d,", static_cast<int>(R.status[p]));
        csv += head;
        if (ok) {
            append_g17(csv, R.delta_x2[p]);
            csv += ',';
            append_g17(csv, R.delta_x3[p]);
            csv += ',';
            append_g17(csv, R.delta_psi1[p]);
            csv += ',';
            append_g17(csv, R.delta_psi2[p]);
        } else {
            csv += "nan,nan,nan,nan";
        }
        csv += ',';
        append_g17(csv, R.tau[p]);
        std::snprintf(head, sizeof(head), ",%llu,", R.count[p]);
        csv += head;
        append_g17(csv, R.land_err[p]);
        csv += '\n';
    }
    return csv;
}

std::string summary_json_text(const ReturnMapRun& R) {
    return R.summary.dump(2) + "\n";
}

void write_run_dir(const std::string& out_dir, const ReturnMapRun& R) {
    mkdir_p(out_dir);
    write_text(out_dir + "/seeds.csv", seeds_csv_text(R));
    write_text(out_dir + "/summary.json", summary_json_text(R));
}

// ===========================================================================
// return-map subcommand
// ===========================================================================

int run_return_map(int argc, char** argv) {
    const RunOptions o = parse_run_options(argc, argv);

    LoadedLabels L = read_labels(o.labels);
    CudaContext ctx(0);

    std::printf("==== spurious_spreading return-map (SF-32 N2a/N2b) ====\n");
    const JVal* route = L.meta.find("route");
    std::printf(
        "labels = %s  route = %s  n = %d  h = %.17g  tracker = %s  seeds = %lld  seed = %llu\n",
        o.labels.c_str(), (route && route->is_string()) ? route->as_string().c_str() : "?", L.n,
        L.h, tracker_name(o.tracker), o.seeds, o.seed);

    const ReturnMapRun R = compute_return_map(ctx, L, o);
    std::printf("  min_abs_c_grid (spline, cell centres) = %.17g\n", R.min_abs_c_grid);
    if (R.exit_code == 3) {
        std::fprintf(stderr,
                     "return-map: min_abs_c_grid = %.17g <= 0.5: label pair refused by the "
                     "pre-registered usability guard (understanding.md 3.5)\n",
                     R.min_abs_c_grid);
        return 3;
    }
    write_run_dir(o.out, R);

    // ---- console ----
    const JVal& sum = R.summary;
    const JVal& lvl = R.level;
    std::printf("  level:");
    for (std::size_t i = 0; i < lvl.keys().size(); ++i) {
        const JVal& v = lvl.at(i);
        std::printf("  %s = %.17g", lvl.keys()[i].c_str(), v.is_number() ? v.as_double() : kNaN);
    }
    std::printf("\n");
    if (R.divergence_max_rel.is_number())
        std::printf("  divergence_max_rel (face fluxes) = %.3e\n",
                    R.divergence_max_rel.as_double());
    auto g = [](const JVal& j, const char* k) {
        const JVal* v = j.find(k);
        return (v && v->is_number()) ? v->as_double() : kNaN;
    };
    std::printf("  n_ok = %lld / %lld", sum.find("n_ok")->as_int(), sum.find("n_seeds")->as_int());
    const JVal& sc = *sum.find("status_counts");
    for (std::size_t i = 0; i < sc.keys().size(); ++i)
        std::printf("  [status %s: %lld]", sc.keys()[i].c_str(), sc.at(i).as_int());
    std::printf("\n");
    const JVal& stats = *sum.find("stats");
    const JVal& m2 = *stats.find("delta_x2");
    const JVal& m3 = *stats.find("delta_x3");
    const JVal& mp1 = *stats.find("delta_psi1");
    const JVal& mp2 = *stats.find("delta_psi2");
    const JVal& mt = *stats.find("tau");
    std::printf("  delta_x2:   mean %.6e  var %.6e  rms %.6e  max %.6e\n", g(m2, "mean"),
                g(m2, "var"), g(m2, "rms"), g(m2, "max_abs"));
    std::printf("  delta_x3:   mean %.6e  var %.6e  rms %.6e  max %.6e\n", g(m3, "mean"),
                g(m3, "var"), g(m3, "rms"), g(m3, "max_abs"));
    std::printf("  delta_psi1: mean %.6e  rms %.6e  max %.6e\n", g(mp1, "mean"), g(mp1, "rms"),
                g(mp1, "max_abs"));
    std::printf("  delta_psi2: mean %.6e  rms %.6e  max %.6e\n", g(mp2, "mean"), g(mp2, "rms"),
                g(mp2, "max_abs"));
    std::printf("  tau: mean %.17g  min %.17g  max %.17g  flux-weighted mean %.17g\n",
                g(mt, "mean"), g(mt, "min"), g(mt, "max"), g(sum, "flux_weighted_tau_mean"));
    std::printf("  eq.36 protocol numbers: D22_u %.6e  D33_u %.6e  D22_f %.6e  D33_f %.6e\n",
                g(sum, "D22_uniform"), g(sum, "D33_uniform"), g(sum, "D22_flux"),
                g(sum, "D33_flux"));
    const JVal& land = *sum.find("landing");
    std::printf("  landing: max |x1_u - 1| = %.3e  max trials = %d\n", g(land, "max_err"),
                static_cast<int>(g(land, "max_iterations")));
    std::printf("  outputs: %s/seeds.csv, %s/summary.json  (wall %.3f s)\n", o.out.c_str(),
                o.out.c_str(), R.wall_seconds);
    return 0;
}

} // namespace spurious_spreading
