/**
 * @file lagrangian_gpu_host_tests.cu
 * @brief SF-36 N4: forward Lagrangian-label tracker (LagrangianTracker.cuh), criteria C and I.
 *
 * C. PLANE pass with a single plane X for every lattice seed (M = N) against the host
 *    `inlet_slab::trace_to_plane(stage.direction_field(), seed, X, opt)` (sigma = +1, same tol,
 *    h_max = h/8, min_step, cap) on {k = 1, generic3d eps 0.25, gauss (seed 3001, ell 0.25,
 *    sigma2 1) eps 0.5 and eps 2.0} at N = 16 and 32: max |dx| <= 1e-12 (expected exactly 0),
 *    per-particle status and nfev identical (the count of differing step sequences is reported).
 * I. Two launches bitwise identical (all outputs); positions, tau, I and statuses bitwise identical
 *    with covectors on vs off (both passes); block 64 vs 256 bitwise identical; no device
 *    allocation in track() after prepare (storage pointer stability AND cudaMemGetInfo);
 *    moments_time.csv written and read back by analysis::read_timeseries_csv reproduces the
 *    in-memory moments; compute_macrodispersion runs on it.
 * Usage: lagrangian_gpu_host_tests [--n N] [--full]  (default ctest tier C at 16/32; --n N: C at N;
 *        --full: C at 64^3 on the four fields). Final line `SUMMARY:`, exit 0 iff all checks pass.
 * Also printed (not thresholds): with k = 1, tau at plane X equals X - X0.
 */

#include "src/analysis/macrodispersion/MacrodispersionAnalysis.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/particles/lagrangian_labels/LagrangianTracker.cuh"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/NpyIo.hpp"
#include "src/physics/streamfunctions/inlet_slab/SlabOracle.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;
namespace ll = macroflow3d::lagrangian_labels;
namespace cg = macroflow3d::closure_gate;

namespace {

struct TestReport {
    bool overall_pass = true;
    int checks = 0, failed = 0;
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

std::string fmt(const char* f, double v) {
    char b[160];
    std::snprintf(b, sizeof(b), f, v);
    return b;
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}
bool same_vec(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!same_bits(a[i], b[i]))
            return false;
    return true;
}
bool same_stats(const std::vector<ll::PassStats>& a, const std::vector<ll::PassStats>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(ll::PassStats)) == 0);
}

/// Everything the tracker outputs, bitwise.
bool same_all(const ll::LagrangianTrackResult& a, const ll::LagrangianTrackResult& b) {
    bool ok = same_vec(a.seeds, b.seeds) && same_vec(a.weights, b.weights) &&
              same_vec(a.planes_x23, b.planes_x23) && same_vec(a.planes_tau, b.planes_tau) &&
              same_vec(a.planes_I, b.planes_I) && same_vec(a.planes_cov, b.planes_cov) &&
              same_vec(a.times_x, b.times_x) && same_vec(a.times_I, b.times_I) &&
              same_vec(a.times_cov, b.times_cov) && a.status_plane == b.status_plane &&
              a.status_time == b.status_time && same_stats(a.stats_plane, b.stats_plane) &&
              same_stats(a.stats_time, b.stats_time) &&
              a.moments_time.size() == b.moments_time.size() &&
              a.moments_planes.size() == b.moments_planes.size();
    for (std::size_t n = 0; ok && n < a.moments_time.size(); ++n) {
        const auto &p = a.moments_time[n], &q = b.moments_time[n];
        ok = same_bits(p.time, q.time) && p.active == q.active;
        for (int k = 0; k < 3; ++k)
            ok = ok && same_bits(p.mean[k], q.mean[k]) && same_bits(p.var[k], q.var[k]);
    }
    for (std::size_t m = 0; ok && m < a.moments_planes.size(); ++m) {
        const auto &p = a.moments_planes[m], &q = b.moments_planes[m];
        ok = same_bits(p.mean_x2, q.mean_x2) && same_bits(p.var_x2, q.var_x2) &&
             same_bits(p.mean_x3, q.mean_x3) && same_bits(p.var_x3, q.var_x3) &&
             same_bits(p.mean_tau, q.mean_tau) && same_bits(p.var_tau, q.var_tau) &&
             same_bits(p.rms_ev, q.rms_ev) && same_bits(p.max_ev, q.max_ev) &&
             p.contributing == q.contributing && p.non_ok == q.non_ok;
    }
    return ok;
}

// ------------------------------------------------------------------------------------------------
// Criterion C
// ------------------------------------------------------------------------------------------------
void run_case_C(TestReport& rep, CudaContext& ctx, const std::string& name,
                const sl::SlabFieldSource& src, double eps, int N) {
    const std::string tag = name + " N=" + std::to_string(N) + ": ";
    const auto t0 = std::chrono::steady_clock::now();
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    sl::ProductionStageOptions so;
    so.require_inlet_labels = false; // the forward tracker does not need the D-1 labels
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, eps, so);
    rep.check(st.flow_ready(), tag + "stage flow ready", sl::to_string(st.report.status));
    if (!st.flow_ready())
        return;
    const double tstage =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const double X0 = 0.0, X = 0.5;
    ll::InjectionSpec inj;
    inj.X0 = X0;
    inj.M = N;
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = grid.h / 8.0;
    to.min_step = 1e-13;
    to.planes = {X};
    ll::LagrangianTracker tr;
    tr.prepare(ctx, st, inj, to);
    const ll::LagrangianTrackResult& R = tr.track(ctx);

    cg::IntegratorOptions opt;
    opt.tol = to.tol;
    opt.h_max = to.h_max;
    opt.min_step = to.min_step;
    opt.max_arclength_per_period = R.max_arclength; // the default cap, identical on both sides
    opt.n_periods = 1;
    opt.sigma = +1;
    const sl::SlabSplineDirectionField fld = st.direction_field();

    double maxdx = 0.0, maxdtau = 0.0;
    long long n_status_diff = 0, n_nfev_diff = 0, n_ok = 0, n_seed_bf = 0, n_tau_bits = 0;
    for (int p = 0; p < R.Np; ++p) {
        const std::array<double, 3> seed = {R.seeds[3 * p], R.seeds[3 * p + 1], R.seeds[3 * p + 2]};
        const cg::StreamlineResult h = sl::trace_to_plane(fld, seed, X, opt);
        const ll::PassStats& d = R.stats_plane[static_cast<std::size_t>(p)];
        if (static_cast<int>(h.status) != d.status)
            ++n_status_diff;
        if (h.field_evaluations != d.nfev)
            ++n_nfev_diff;
        if (h.status == cg::StreamlineStatus::seed_backflow)
            ++n_seed_bf;
        if (h.status == cg::StreamlineStatus::ok && d.status == 0) {
            ++n_ok;
            const std::size_t i = static_cast<std::size_t>(p);
            maxdx = std::max(maxdx, std::abs(h.records[0].x2 - R.planes_x23[2 * i]));
            maxdx = std::max(maxdx, std::abs(h.records[0].x3 - R.planes_x23[2 * i + 1]));
            maxdtau = std::max(maxdtau, std::abs(h.records[0].tau - R.planes_tau[i]));
            if (!same_bits(h.records[0].tau, R.planes_tau[i]))
                ++n_tau_bits;
        }
    }
    const double tall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    rep.check(n_status_diff == 0, tag + "per-particle status identical to host trace_to_plane",
              "differing " + std::to_string(n_status_diff) + " of " + std::to_string(R.Np));
    rep.check(maxdx <= 1e-12, tag + "max |dx| (x2, x3 at the plane) <= 1e-12",
              fmt("max|dx| %.3e", maxdx) + " over " + std::to_string(n_ok) + " ok particles");
    rep.check(n_nfev_diff == 0, tag + "per-particle nfev identical (same step sequences)",
              "differing step sequences " + std::to_string(n_nfev_diff));
    std::printf("  %sINFO max|dtau| %.3e (bitwise-different tau %lld), seed_backflow %lld, "
                "dev statuses ok %d  [stage %.1fs, case %.1fs, plane kernel %.3fs]\n",
                tag.c_str(), maxdtau, n_tau_bits, n_seed_bf, R.totals_plane.status_hist[0], tstage,
                tall, R.seconds_plane_kernel);
    if (name.rfind("k = 1", 0) == 0) {
        double mx = 0.0;
        for (int p = 0; p < R.Np; ++p)
            mx = std::max(mx, std::abs(R.planes_tau[static_cast<std::size_t>(p)] - (X - X0)));
        std::printf("  %sSANITY (k = 1) max |tau(X) - (X - X0)| = %.3e\n", tag.c_str(), mx);
    }
}

/// INFO (criterion E is verified at 64^3 later): max |a x b - v exp(-I)| / |v| at every record of the
/// plane and time passes, plus max |I| and max |cov|.
void identity_info(const std::string& tag, const sl::ProductionStage& st,
                   const ll::LagrangianTrackResult& A, const char* what) {
    const sl::SlabSplineDirectionField fld = st.direction_field();
    auto resid = [&](const double* x, const double* cov, double I) {
        double g[3], k;
        fld(x, g, k);
        const double v[3] = {k * g[0], k * g[1], k * g[2]};
        const double a[3] = {cov[0], cov[1], cov[2]}, b[3] = {cov[3], cov[4], cov[5]};
        const double c[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                             a[0] * b[1] - a[1] * b[0]};
        const double e = std::exp(-I);
        const double vn = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        const double d[3] = {c[0] - v[0] * e, c[1] - v[1] * e, c[2] - v[2] * e};
        return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / vn;
    };
    double mp = 0.0, mt = 0.0, amax = 0.0, imax = 0.0;
    const std::size_t P = A.planes.size(), T = A.times.size();
    for (std::size_t p = 0; p < static_cast<std::size_t>(A.Np); ++p) {
        for (std::size_t m = 0; m < P && static_cast<std::size_t>(A.stats_plane[p].n_reached) > m;
             ++m) {
            const double x[3] = {A.planes[m], A.planes_x23[(p * P + m) * 2],
                                 A.planes_x23[(p * P + m) * 2 + 1]};
            mp = std::max(mp, resid(x, &A.planes_cov[(p * P + m) * 6], A.planes_I[p * P + m]));
            imax = std::max(imax, std::abs(A.planes_I[p * P + m]));
            for (int i = 0; i < 6; ++i)
                amax = std::max(amax, std::abs(A.planes_cov[(p * P + m) * 6 + i]));
        }
        for (std::size_t n = 0; n < T && static_cast<std::size_t>(A.stats_time[p].n_reached) > n;
             ++n) {
            mt = std::max(mt, resid(&A.times_x[(p * T + n) * 3], &A.times_cov[(p * T + n) * 6],
                                    A.times_I[p * T + n]));
        }
    }
    std::printf("  %sINFO identity [%s] max |a x b - v exp(-I)|/|v|: planes %.3e, times %.3e "
                "(max |I| %.3e, max |cov| %.3e)\n",
                tag.c_str(), what, mp, mt, imax, amax);
}

// ------------------------------------------------------------------------------------------------
// Criterion I
// ------------------------------------------------------------------------------------------------
void run_case_I(TestReport& rep, CudaContext& ctx, const std::string& name,
                const sl::SlabFieldSource& src, double eps, int N, int M) {
    const std::string tag = "I " + name + " N=" + std::to_string(N) + ": ";
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, eps);
    rep.check(st.ok() && st.labels_available, tag + "stage ok with labels",
              sl::to_string(st.report.status));
    if (!st.flow_ready())
        return;

    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = M;
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = grid.h / 8.0;
    to.planes = {0.25, 0.5, 1.0, 1.5};
    to.times = {0.3, 0.7, 1.2};
    to.covectors = true;
    to.block = 64;

    ll::LagrangianTracker on;
    on.prepare(ctx, st, inj, to);
    const std::vector<const void*> ptr0 = on.device_storage_pointers();
    std::size_t free_pre = 0, free0 = 0, free1 = 0, free2 = 0, total = 0;
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free_pre, &total));
    // The first launch of a kernel instantiation makes the driver reserve per-thread local memory /
    // load its module (a few MiB, observed also with CUDA_MODULE_LOADING=EAGER; not an allocation
    // of the tracker, whose buffers are checked by pointer stability); the allocation contract is
    // therefore asserted between the 1st and the following launches and the delta is printed.
    const ll::LagrangianTrackResult A = on.track(ctx); // copy
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free0, &total));
    const ll::LagrangianTrackResult B = on.track(ctx);
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free1, &total));
    (void)on.track(ctx);
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free2, &total));
    std::printf("  %sINFO free bytes: before 1st track %zu, after 1st %zu (delta %lld = driver first-launch "
                "reservation if non-zero), after 2nd %zu, after 3rd %zu\n",
                tag.c_str(), free_pre, free0,
                static_cast<long long>(free_pre) - static_cast<long long>(free0), free1, free2);
    const std::vector<const void*> ptr1 = on.device_storage_pointers();
    rep.check(ptr0 == ptr1, tag + "device storage pointers stable across track() (no reallocation)");
    rep.check(free0 == free1 && free1 == free2,
              tag + "cudaMemGetInfo free bytes equal across the 1st, 2nd and 3rd track()",
              "free bytes " + std::to_string(free0) + " / " + std::to_string(free1) + " / " +
                  std::to_string(free2) + " (device shared: re-run alone before concluding)");

    rep.check(same_all(A, B), tag + "two launches bitwise identical (all outputs, moments)");

    const ll::LagrangianTrackResult C256 = on.track(ctx, 256);
    rep.check(same_all(A, C256), tag + "block 64 vs 256 bitwise identical");

    ll::TrackOptions to_off = to;
    to_off.covectors = false;
    ll::LagrangianTracker off;
    off.prepare(ctx, st, inj, to_off);
    const ll::LagrangianTrackResult& O = off.track(ctx);
    const bool pos_same = same_vec(A.planes_x23, O.planes_x23) && same_vec(A.planes_tau, O.planes_tau) &&
                          same_vec(A.times_x, O.times_x) && same_vec(A.planes_I, O.planes_I) &&
                          same_vec(A.times_I, O.times_I);
    bool stat_same = A.status_plane == O.status_plane && A.status_time == O.status_time;
    long long nfev_diff = 0;
    for (int p = 0; p < A.Np; ++p) {
        const std::size_t i = static_cast<std::size_t>(p);
        if (A.stats_plane[i].nfev != O.stats_plane[i].nfev ||
            A.stats_time[i].nfev != O.stats_time[i].nfev)
            ++nfev_diff;
    }
    rep.check(pos_same && stat_same && nfev_diff == 0,
              tag + "positions, tau, I, statuses, nfev bitwise identical with covectors on vs off",
              "nfev differing particles " + std::to_string(nfev_diff));

    identity_info(tag, st, A, "tol 1e-8, h_max h/8");
    for (const double div : {16.0, 32.0}) {
        ll::TrackOptions tv = to;
        tv.tol = 1e-10;
        tv.h_max = grid.h / div;
        ll::LagrangianTracker tq;
        tq.prepare(ctx, st, inj, tv);
        const std::string w = "tol 1e-10, h_max h/" + std::to_string(static_cast<int>(div));
        identity_info(tag, st, tq.track(ctx), w.c_str());
    }

    // labels at the seeds (D-1) finite and the injection weights normalized
    double wsum = 0.0;
    bool lab_ok = true;
    for (int p = 0; p < A.Np; ++p) {
        wsum += A.weights[static_cast<std::size_t>(p)];
        lab_ok = lab_ok && std::isfinite(A.labels[2 * static_cast<std::size_t>(p)]) &&
                 std::isfinite(A.labels[2 * static_cast<std::size_t>(p) + 1]);
    }
    rep.check(lab_ok && std::abs(wsum - 1.0) <= 1e-13,
              tag + "D-1 labels finite at the seeds, weights sum to 1", fmt("sum w %.16f", wsum));

    // CSV round trip + downstream reader
    namespace fs = std::filesystem;
    const fs::path dir =
        fs::temp_directory_path() / ("sf36_lagr_gpu_host_" + std::to_string(::getpid()));
    ll::TrackMeta meta;
    meta.field = name;
    meta.eps = eps;
    meta.N = N;
    meta.h = grid.h;
    ll::write_track_outputs(dir.string(), A, meta);
    const std::vector<io::TimeSeriesPoint<double>> raw =
        analysis::read_timeseries_csv<double>((dir / "moments_time.csv").string());
    // Pre-existing behaviour of the reader (not changed here): it skips ONE header line, but
    // CsvTimeSeriesWriter writes "# format_version=N" and then the column header, so the column
    // header line is parsed as an all-zero row 0. Drop exactly that artifact row (verified).
    const bool artifact = raw.size() == A.moments_time.size() + 1 && raw[0].time == 0.0 &&
                          raw[0].active == 0;
    std::vector<io::TimeSeriesPoint<double>> rd;
    if (artifact)
        rd.assign(raw.begin() + 1, raw.end());
    bool csv_ok = artifact && rd.size() == A.moments_time.size() && !rd.empty();
    double maxrel = 0.0;
    for (std::size_t n = 0; csv_ok && n < rd.size(); ++n) {
        csv_ok = csv_ok && rd[n].active == A.moments_time[n].active;
        auto rel = [](double a, double b) {
            return std::abs(a - b) / std::max(1.0, std::abs(b));
        };
        maxrel = std::max(maxrel, rel(rd[n].time, A.moments_time[n].time));
        for (int k = 0; k < 3; ++k) {
            maxrel = std::max(maxrel, rel(rd[n].mean[k], A.moments_time[n].mean[k]));
            maxrel = std::max(maxrel, rel(rd[n].var[k], A.moments_time[n].var[k]));
        }
    }
    rep.check(csv_ok && maxrel <= 1e-14,
              tag + "moments_time.csv read back by analysis::read_timeseries_csv reproduces the "
                    "in-memory moments (rows 1..T; row 0 = reader's header artifact)",
              fmt("max rel %.3e", maxrel) + ", raw rows " + std::to_string(raw.size()));
    const auto rows = analysis::compute_macrodispersion<double>({rd}, 0.25, 1.0);
    rep.check(rows.size() == rd.size(), tag + "compute_macrodispersion runs on the read series",
              "rows " + std::to_string(rows.size()));
    // npy round trip of a record array, exact
    const sl::NpyArray px = sl::read_npy((dir / "planes_x23.npy").string());
    rep.check(px.shape == std::vector<std::size_t>{static_cast<std::size_t>(A.Np), 4, 2} &&
                  same_vec(px.data, A.planes_x23),
              tag + "planes_x23.npy round trip exact, shape (Np, P, 2)");
    std::printf("  %sINFO ok statuses: plane %d/%d, time %d/%d; seed_backflow %lld; "
                "moments_time n=%zu active[0]=%d; kernels plane %.3fs time %.3fs\n",
                tag.c_str(), A.totals_plane.status_hist[0], A.Np, A.totals_time.status_hist[0], A.Np,
                A.seed_backflow, A.moments_time.size(), A.moments_time[0].active,
                A.seconds_plane_kernel, A.seconds_time_kernel);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

} // namespace

int main(int argc, char** argv) {
    // default: C at N = 16, 32 and I at N = 16 (ctest tier); --n N: C at that single N (I unchanged);
    // --full: C at N = 64 on the four fields (M = N = 64 seeds per side), I unchanged.
    std::vector<int> c_grids = {16, 32};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--full")
            c_grids = {64};
        else if (a == "--n" && i + 1 < argc)
            c_grids = {std::atoi(argv[++i])};
        else {
            std::fprintf(stderr, "usage: %s [--n N] [--full]\n", argv[0]);
            return 2;
        }
    }
    TestReport rep;
    CudaContext ctx(0);
    const auto t0 = std::chrono::steady_clock::now();
    auto lap = [&](const char* what) {
        std::printf("  [time] %s: %.1fs elapsed\n", what,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        std::fflush(stdout);
    };

    for (const int N : c_grids) {
        {
            const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(
                ctx, "uniform", N, std::vector<real>(static_cast<std::size_t>(N) * N * N, 0.0));
            run_case_C(rep, ctx, "k = 1", src, 1.0, N);
        }
        {
            const sl::SlabFieldSource src = sl::SlabFieldSource::analytic("generic3d", N);
            run_case_C(rep, ctx, "generic3d eps 0.25", src, 0.25, N);
        }
        {
            sl::ProductionFieldSpec spec;
            spec.N = N;
            spec.sigma2 = 1.0;
            spec.ell = 0.25;
            spec.seed = 3001ULL;
            const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, spec);
            run_case_C(rep, ctx, "gauss s2=1 ell=0.25 seed=3001 eps 0.5", src, 0.5, N);
            run_case_C(rep, ctx, "gauss s2=1 ell=0.25 seed=3001 eps 2.0", src, 2.0, N);
        }
        lap(("C at N=" + std::to_string(N)).c_str());
    }
    {
        const int N = 16;
        sl::ProductionFieldSpec spec;
        spec.N = N;
        spec.sigma2 = 1.0;
        spec.ell = 0.25;
        spec.seed = 3001ULL;
        const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, spec);
        run_case_I(rep, ctx, "gauss eps 0.5", src, 0.5, N, 16);
        lap("I");
    }

    std::printf("\nSUMMARY: %d checks, %d failed -> %s\n", rep.checks, rep.failed,
                rep.overall_pass ? "ALL PASS" : "SOME FAILED");
    return rep.overall_pass ? 0 : 1;
}
