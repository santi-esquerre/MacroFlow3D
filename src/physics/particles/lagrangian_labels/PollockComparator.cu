/**
 * @file PollockComparator.cu
 * @brief SF-37 N3: Pollock lattice tracker (see PollockComparator.cuh). Compiled with --fmad=false
 *        (CMakeLists.txt, like LagrangianTracker.cu).
 */

#include "PollockComparator.cuh"

#include "../../../core/DeviceBuffer.cuh"
#include "../../../runtime/cuda_check.cuh"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace lagrangian_labels {

namespace stt = physics::particles::streamline_tracker;

namespace {

constexpr int kBlock = 64;

__device__ inline double nan_d() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

/// Raw weight: x1 velocity of the RT0 interpolant at the seed, in the cell chosen by the core.
template <class F>
__global__ void pollock_seed_weight_kernel(F f, const double* seeds, int Np, double* weight_raw) {
    const int p = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (p >= Np)
        return;
    const real xi[3] = {seeds[3 * p], seeds[3 * p + 1], seeds[3 * p + 2]};
    const int32_t w0[3] = {0, 0, 0};
    stt::PollockState st{};
    double raw = nan_d();
    if (stt::pollock_init_state(f, xi, w0, st) == stt::kStatusActive) {
        stt::pollock_detail::CellEval ev;
        if (stt::pollock_detail::eval_cell(f, st, ev) == stt::kStatusActive) {
            raw = ev.vp[0];
        }
    }
    weight_raw[p] = raw;
}

/// PLANE pass: pollock_advance_to_x1 for each plane in turn; records unwrapped (x2, x3) and the clock.
template <class F>
__global__ void pollock_plane_kernel(F f, stt::PollockParams prm, const double* seeds, int Np,
                                     const double* planes, int P, std::uint8_t* status,
                                     int* reached, double* x23, double* tt,
                                     std::uint64_t* cells) {
    const int p = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (p >= Np)
        return;
    const double nan = nan_d();
    for (int m = 0; m < P; ++m) {
        x23[(static_cast<std::size_t>(p) * P + m) * 2] = nan;
        x23[(static_cast<std::size_t>(p) * P + m) * 2 + 1] = nan;
        tt[static_cast<std::size_t>(p) * P + m] = nan;
    }
    int n_reached = 0;
    std::uint64_t n_cells = 0;
    const real xi[3] = {seeds[3 * p], seeds[3 * p + 1], seeds[3 * p + 2]};
    const int32_t w0[3] = {0, 0, 0};
    stt::PollockState st{};
    std::uint8_t code = stt::pollock_init_state(f, xi, w0, st);
    for (int m = 0; m < P && code == stt::kStatusActive; ++m) {
        stt::PollockCounters cnt{0u};
        code = stt::pollock_advance_to_x1(f, st, planes[m], prm, cnt);
        n_cells += cnt.cells;
        if (code != stt::kStatusActive)
            break;
        real xu[3];
        stt::pollock_unwrapped_position(f, st, xu);
        x23[(static_cast<std::size_t>(p) * P + m) * 2] = xu[1];
        x23[(static_cast<std::size_t>(p) * P + m) * 2 + 1] = xu[2];
        tt[static_cast<std::size_t>(p) * P + m] = st.t;
        n_reached = m + 1;
    }
    status[p] = code;
    reached[p] = n_reached;
    cells[p] = n_cells;
}

/// TIME pass: fresh init, pollock_advance_to_time for each time in turn; records unwrapped (x1, x2, x3).
template <class F>
__global__ void pollock_time_kernel(F f, stt::PollockParams prm, const double* seeds, int Np,
                                    const double* times, int T, std::uint8_t* status,
                                    int* reached, double* xs, std::uint64_t* cells) {
    const int p = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (p >= Np)
        return;
    const double nan = nan_d();
    for (int n = 0; n < T; ++n) {
        for (int a = 0; a < 3; ++a)
            xs[(static_cast<std::size_t>(p) * T + n) * 3 + a] = nan;
    }
    int n_reached = 0;
    std::uint64_t n_cells = 0;
    const real xi[3] = {seeds[3 * p], seeds[3 * p + 1], seeds[3 * p + 2]};
    const int32_t w0[3] = {0, 0, 0};
    stt::PollockState st{};
    std::uint8_t code = stt::pollock_init_state(f, xi, w0, st);
    for (int n = 0; n < T && code == stt::kStatusActive; ++n) {
        stt::PollockCounters cnt{0u};
        code = stt::pollock_advance_to_time(f, st, times[n], prm, cnt);
        n_cells += cnt.cells;
        if (code != stt::kStatusActive)
            break;
        real xu[3];
        stt::pollock_unwrapped_position(f, st, xu);
        for (int a = 0; a < 3; ++a)
            xs[(static_cast<std::size_t>(p) * T + n) * 3 + a] = xu[a];
        n_reached = n + 1;
    }
    status[p] = code;
    reached[p] = n_reached;
    cells[p] = n_cells;
}

/// Weighted mean and biased variance of val(p) over the selected particles (index order). The same
/// definition as the SF-36 moments (finalize_track_result).
template <class Sel, class Val>
void weighted_mean_var(const std::vector<double>& w, std::size_t Np, Sel sel, Val val, double& mean,
                       double& var, int& count) {
    double W = 0.0, S = 0.0;
    count = 0;
    for (std::size_t p = 0; p < Np; ++p) {
        if (!sel(p))
            continue;
        ++count;
        W += w[p];
        S += w[p] * val(p);
    }
    if (count == 0 || !(W > 0.0)) {
        mean = var = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    mean = S / W;
    double V = 0.0;
    for (std::size_t p = 0; p < Np; ++p) {
        if (!sel(p))
            continue;
        const double d = val(p) - mean;
        V += w[p] * d * d;
    }
    var = V / W;
}

template <class T> void upload(DeviceBuffer<T>& dst, const std::vector<T>& src) {
    if (src.empty())
        return;
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T> void download(std::vector<T>& dst, const DeviceBuffer<T>& src) {
    if (dst.empty())
        return;
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(dst.data(), src.data(), dst.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void validate(const InjectionSpec& inj, const std::vector<double>& planes,
              const std::vector<double>& times, const stt::PollockParams& prm) {
    auto bad = [](const std::string& m) {
        throw std::invalid_argument("lagrangian_labels::track_pollock_lattice: " + m);
    };
    if (!std::isfinite(inj.X0))
        bad("injection X0 must be finite");
    if (inj.M < 1)
        bad("injection M must be >= 1");
    if (!(std::isfinite(inj.w2a) && std::isfinite(inj.w2b) && inj.w2a < inj.w2b))
        bad("injection window requires finite w2a < w2b");
    if (!(std::isfinite(inj.w3a) && std::isfinite(inj.w3b) && inj.w3a < inj.w3b))
        bad("injection window requires finite w3a < w3b");
    if (prm.max_cells_per_call < 1)
        bad("max_cells_per_call must be >= 1");
    if (planes.empty() && times.empty())
        bad("at least one plane or one snapshot time is required");
    double prev = inj.X0;
    for (double X : planes) {
        if (!std::isfinite(X) || !(X > prev))
            bad("planes must be finite, strictly increasing and > X0");
        prev = X;
    }
    prev = 0.0;
    for (double t : times) {
        if (!std::isfinite(t) || !(t > prev))
            bad("times must be finite, strictly increasing and > 0");
        prev = t;
    }
}

/// Shared body; F = PeriodicFaceFluxView is the SF-37 periodic comparator (code path unchanged),
/// F = NonPeriodicXFaceFluxView the SF-41 medium (b) mode (exit counts, optional zero-weight lattice).
template <class F>
PollockLatticeResult track_pollock_lattice_impl(CudaContext& ctx, const F& f, const InjectionSpec& inj,
                                                const std::vector<double>& planes,
                                                const std::vector<double>& times,
                                                const stt::PollockParams& prm,
                                                bool allow_no_positive_weight) {
    validate(inj, planes, times, prm);
    PollockLatticeResult r;
    r.injection = inj;
    r.planes = planes;
    r.times = times;
    r.max_cells_per_call = prm.max_cells_per_call;
    r.P = static_cast<int>(planes.size());
    r.T = static_cast<int>(times.size());
    r.seeds = make_seed_lattice(inj);
    r.Np = inj.M * inj.M;
    const std::size_t Np = static_cast<std::size_t>(r.Np);
    const std::size_t P = static_cast<std::size_t>(r.P), T = static_cast<std::size_t>(r.T);
    const unsigned blocks = static_cast<unsigned>((Np + kBlock - 1) / kBlock);

    // ---- device allocations (once per call) -------------------------------------------------
    DeviceBuffer<double> d_seeds(Np * 3), d_raw(Np);
    DeviceBuffer<double> d_planes(P), d_times(T);
    DeviceBuffer<std::uint8_t> d_stat_p(Np), d_stat_t(Np);
    DeviceBuffer<int> d_reach_p(Np), d_reach_t(Np);
    DeviceBuffer<std::uint64_t> d_cells_p(Np), d_cells_t(Np);
    DeviceBuffer<double> d_x23(Np * P * 2), d_tt(Np * P), d_xs(Np * T * 3);
    upload(d_seeds, r.seeds);
    upload(d_planes, planes);
    upload(d_times, times);
    // SF-33 C2: legacy-stream copies must land before ctx-stream work.
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());

    // ---- kernels -------------------------------------------------------------------------------
    cudaStream_t s = ctx.cuda_stream();
    pollock_seed_weight_kernel<<<blocks, kBlock, 0, s>>>(f, d_seeds.data(), r.Np, d_raw.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    auto t0 = std::chrono::steady_clock::now();
    if (P > 0) {
        pollock_plane_kernel<<<blocks, kBlock, 0, s>>>(
            f, prm, d_seeds.data(), r.Np, d_planes.data(), r.P, d_stat_p.data(), d_reach_p.data(),
            d_x23.data(), d_tt.data(), d_cells_p.data());
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    ctx.synchronize();
    r.seconds_plane_kernel = seconds_since(t0);
    t0 = std::chrono::steady_clock::now();
    if (T > 0) {
        pollock_time_kernel<<<blocks, kBlock, 0, s>>>(f, prm, d_seeds.data(), r.Np, d_times.data(),
                                                      r.T, d_stat_t.data(), d_reach_t.data(),
                                                      d_xs.data(), d_cells_t.data());
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    ctx.synchronize();
    r.seconds_time_kernel = seconds_since(t0);

    // ---- downloads -----------------------------------------------------------------------------
    t0 = std::chrono::steady_clock::now();
    r.weight_raw.assign(Np, 0.0);
    r.status_plane.assign(Np, 0);
    r.status_time.assign(Np, 0);
    r.reached_plane.assign(Np, 0);
    r.reached_time.assign(Np, 0);
    r.cells.assign(Np * 2, 0);
    r.planes_x23.assign(Np * P * 2, 0.0);
    r.planes_t.assign(Np * P, 0.0);
    r.times_x.assign(Np * T * 3, 0.0);
    std::vector<std::uint64_t> cp(Np, 0), ct(Np, 0);
    download(r.weight_raw, d_raw);
    if (P > 0) {
        download(r.status_plane, d_stat_p);
        download(r.reached_plane, d_reach_p);
        download(cp, d_cells_p);
        download(r.planes_x23, d_x23);
        download(r.planes_t, d_tt);
    }
    if (T > 0) {
        download(r.status_time, d_stat_t);
        download(r.reached_time, d_reach_t);
        download(ct, d_cells_t);
        download(r.times_x, d_xs);
    }
    for (std::size_t p = 0; p < Np; ++p) {
        r.cells[2 * p] = cp[p];
        r.cells[2 * p + 1] = ct[p];
    }
    r.seconds_download = seconds_since(t0);

    // ---- reductions (host, index order) ----------------------------------------------------
    t0 = std::chrono::steady_clock::now();
    auto none_positive = [&] {
        for (double w : r.weight_raw)
            if (std::isfinite(w) && w > 0.0)
                return false;
        return true;
    };
    if (allow_no_positive_weight && none_positive()) {
        r.weights.assign(Np, 0.0); // test-only path: no flux-weighted moments exist (NaN)
        r.seed_nonpos = static_cast<long long>(Np);
    } else {
        normalize_flux_weights(r.weight_raw, r.weights, r.seed_nonpos);
    }
    for (std::size_t p = 0; p < Np; ++p) {
        ++r.status_hist_plane[r.status_plane[p] < kPollockStatusSlots ? r.status_plane[p]
                                                                      : kPollockStatusSlots - 1];
        ++r.status_hist_time[r.status_time[p] < kPollockStatusSlots ? r.status_time[p]
                                                                    : kPollockStatusSlots - 1];
        r.cells_plane += r.cells[2 * p];
        r.cells_time += r.cells[2 * p + 1];
    }
    if constexpr (stt::kNonPeriodicX<F>) {
        // Exits through the box faces: per plane / time, the particles that left before reaching it.
        r.exit_inlet_plane.assign(P, 0);
        r.exit_outlet_plane.assign(P, 0);
        r.exit_inlet_time.assign(T, 0);
        r.exit_outlet_time.assign(T, 0);
        for (std::size_t p = 0; p < Np; ++p) {
            const std::uint8_t sp = r.status_plane[p], st = r.status_time[p];
            if (P > 0 && (sp == stt::kStatusPollockExitInlet || sp == stt::kStatusPollockExitOutlet)) {
                auto& v = (sp == stt::kStatusPollockExitInlet) ? r.exit_inlet_plane : r.exit_outlet_plane;
                ++(sp == stt::kStatusPollockExitInlet ? r.exit_inlet_plane_total
                                                      : r.exit_outlet_plane_total);
                for (std::size_t m = static_cast<std::size_t>(r.reached_plane[p]); m < P; ++m)
                    ++v[m];
            }
            if (T > 0 && (st == stt::kStatusPollockExitInlet || st == stt::kStatusPollockExitOutlet)) {
                auto& v = (st == stt::kStatusPollockExitInlet) ? r.exit_inlet_time : r.exit_outlet_time;
                ++(st == stt::kStatusPollockExitInlet ? r.exit_inlet_time_total
                                                      : r.exit_outlet_time_total);
                for (std::size_t n = static_cast<std::size_t>(r.reached_time[p]); n < T; ++n)
                    ++v[n];
            }
        }
    }
    r.moments_planes.assign(P, PollockPlaneMoments{});
    for (std::size_t m = 0; m < P; ++m) {
        PollockPlaneMoments& pm = r.moments_planes[m];
        pm.x1 = planes[m];
        auto sel = [&](std::size_t p) { return static_cast<std::size_t>(r.reached_plane[p]) > m; };
        int cnt = 0;
        weighted_mean_var(
            r.weights, Np, sel, [&](std::size_t p) { return r.planes_x23[(p * P + m) * 2]; },
            pm.mean_x2, pm.var_x2, cnt);
        weighted_mean_var(
            r.weights, Np, sel, [&](std::size_t p) { return r.planes_x23[(p * P + m) * 2 + 1]; },
            pm.mean_x3, pm.var_x3, cnt);
        weighted_mean_var(
            r.weights, Np, sel, [&](std::size_t p) { return r.planes_t[p * P + m]; }, pm.mean_t,
            pm.var_t, cnt);
        pm.contributing = cnt;
        int non_ok = 0;
        for (std::size_t p = 0; p < Np; ++p)
            if (r.status_plane[p] != stt::kStatusActive)
                ++non_ok;
        pm.non_ok = non_ok;
    }
    r.moments_time.assign(T, io::TimeSeriesPoint<double>{});
    for (std::size_t n = 0; n < T; ++n) {
        io::TimeSeriesPoint<double>& pt = r.moments_time[n];
        pt.time = times[n];
        auto sel = [&](std::size_t p) { return static_cast<std::size_t>(r.reached_time[p]) > n; };
        int cnt = 0;
        for (int k = 0; k < 3; ++k) {
            double m = 0.0, v = 0.0;
            weighted_mean_var(
                r.weights, Np, sel, [&](std::size_t p) { return r.times_x[(p * T + n) * 3 + k]; }, m,
                v, cnt);
            pt.mean[k] = m;
            pt.var[k] = v;
        }
        pt.active = cnt;
    }
    r.seconds_moments = seconds_since(t0);
    return r;
}

} // namespace

PollockLatticeResult track_pollock_lattice(CudaContext& ctx, const stt::PeriodicFaceFluxView& f,
                                           const InjectionSpec& inj,
                                           const std::vector<double>& planes,
                                           const std::vector<double>& times,
                                           const stt::PollockParams& prm) {
    return track_pollock_lattice_impl(ctx, f, inj, planes, times, prm, false);
}

PollockLatticeResult track_pollock_lattice(CudaContext& ctx, const stt::NonPeriodicXFaceFluxView& f,
                                           const InjectionSpec& inj,
                                           const std::vector<double>& planes,
                                           const std::vector<double>& times,
                                           const stt::PollockParams& prm,
                                           const PollockNonPeriodicOptions& opt) {
    return track_pollock_lattice_impl(ctx, f, inj, planes, times, prm, opt.allow_no_positive_weight);
}

} // namespace lagrangian_labels
} // namespace macroflow3d
