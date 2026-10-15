/**
 * @file pollock_tracker_tests.cu
 * @brief SF-32 N3b: fast contract tests of the Pollock-type (RT0) tracker
 *        (`src/physics/particles/streamline_tracker/PollockTracker.cuh`) on the
 *        N0 Stokes face fluxes (`StokesFaceVelocity.cuh`).
 *
 * Standalone ctest-friendly runner (style of
 * `tests/streamline_tracker/pseudo_symplectic_tracker_tests.cu`: printed
 * `[PASS]/[FAIL]` checks, `main` returns 0/1). Fast tier.
 *
 * Checks P1-P6 (+ no allocation, timing record) are PRE-REGISTERED by the
 * SF-32 orchestrator (node N3b specification; orchestration record section
 * 3.3, P3 as RESTATED with the targets x1 = 0.25 and x1 = 0.30) and are
 * implemented verbatim; no gate may be adjusted to make a failing line pass.
 *
 * Fixtures: SF-31 analytic pairs (`analytic_pairs.hpp`) sampled at the cell
 * centres and splined with the SF-28 GPU prefilter; N0 face fluxes on the
 * Pollock grid Delta = h (n = N); 256 seeds on the face x1 = 0 by
 * inject_box(0,0,0, 0,1,1, 0, 256) with a fixed seed; engine call order
 * configure, bind_fluxes, bind_particles, inject_box, ensure_tracking,
 * prepare, step_to_x1 / step.
 *
 * Cases:
 *   p1_uniform_exact, p2_pair_A_exact, p3_pair_B_ladder,
 *   p4_host_core_vs_gpu, p5_determinism, p6_stagnation (+ degenerate inputs),
 *   no_allocation, timing_record (information only).
 */

#include "analytic_pairs.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "src/physics/particles/streamline_tracker/PollockTracker.cuh"
#include "src/physics/particles/streamline_tracker/StokesFaceVelocity.cuh"
#include "src/physics/particles/streamline_tracker/StreamlineTrackerCommon.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::interpolation;
namespace st = macroflow3d::physics::particles::streamline_tracker;
using macroflow3d::physics::particles::ParticlesSoA;
using namespace sf31_tests;

namespace {

// ============================================================================
// Bookkeeping
// ============================================================================

struct TestReport {
    bool overall_pass = true;
    int checks = 0;
    int fails = 0;

    void check(bool cond, const std::string& name, const std::string& detail = "") {
        ++checks;
        if (!cond)
            ++fails;
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(), detail.empty() ? "" : "  ",
                    detail.c_str());
        overall_pass = overall_pass && cond;
    }
};

std::string strf(const char* f, ...) {
    char buf[768];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

template <class T> std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    if (n > 0)
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

template <class T> void h2d(T* d, const std::vector<T>& h) {
    if (!h.empty())
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T> bool vec_eq(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

template <class Ex, class F> bool throws_as(F&& f, const char* needle, std::string& msg) {
    try {
        f();
    } catch (const Ex& e) {
        msg = e.what();
        return std::string(e.what()).find(needle) != std::string::npos;
    } catch (const std::exception& e) {
        msg = std::string("WRONG TYPE: ") + e.what();
        return false;
    } catch (...) {
        msg = "WRONG TYPE (non-std)";
        return false;
    }
    msg = "no exception";
    return false;
}

/// 64 fixed points: 4 hand-picked followed by 60 inject_uniform01(seed 20261005, index 0..59).
struct P3 {
    double v[3];
};
std::vector<P3> fixed_points64() {
    std::vector<P3> p = {{{0.1, 0.2, 0.3}}, {{0.35, 0.6, 0.85}}, {{0.7, 0.15, 0.5}}, {{0.9, 0.8, 0.05}}};
    for (uint64_t i = 0; i < 60; ++i) {
        P3 q;
        for (int a = 0; a < 3; ++a)
            q.v[a] = st::inject_uniform01(20261005ULL, i, a);
        p.push_back(q);
    }
    return p;
}

double order2(double coarse, double fine) {
    return std::log2(coarse / fine);
}

// ============================================================================
// Fixtures: splined labels + N0 face fluxes (Delta = h)
// ============================================================================

/// Splined labels of one pair at N^3, their N0 face fluxes on the n = N Pollock
/// grid (device), host labels (host prefilter) and host face arrays (host
/// mirror). Not movable (views point into owned buffers).
struct FluxSet {
    int pair = 0;
    int N = 0;
    Grid3D g;
    PeriodicTricubicBSplineWorkspace ws1, ws2;
    std::vector<real> hc1, hc2; ///< host-prefilter coefficients
    st::SplineLabelPair dev{}, host{};
    st::StokesFaceFluxWorkspace fws;
    st::PeriodicFaceFluxView dview{};
    std::vector<real> hu, hv, hw; ///< host-mirror face arrays
    st::PeriodicFaceFluxView hview{};
    double e_s1 = 0.0, e_s2 = 0.0; ///< max |s_spline - s_exact| over the 64 fixed points
    double coef_diff = 0.0;         ///< max |host-prefilter coef - GPU coef| (information)

    FluxSet() = default;
    FluxSet(const FluxSet&) = delete;
    FluxSet& operator=(const FluxSet&) = delete;
};

std::unique_ptr<FluxSet> build_fluxes(const CudaContext& ctx, int pair, int N, bool host_mirror) {
    std::unique_ptr<FluxSet> S(new FluxSet());
    S->pair = pair;
    S->N = N;
    const double h = 1.0 / N;
    S->g = Grid3D(N, N, N, h, h, h);
    std::vector<real> s1, s2;
    sample_fluctuations(pair, S->g, s1, s2);
    {
        DeviceBuffer<real> d1(s1.size()), d2(s2.size());
        h2d(d1.data(), s1);
        h2d(d2.data(), s2);
        prefilter_periodic_tricubic_bspline(ctx, S->g, DeviceSpan<const real>(d1.data(), d1.size()), S->ws1);
        prefilter_periodic_tricubic_bspline(ctx, S->g, DeviceSpan<const real>(d2.data(), d2.size()), S->ws2);
        ctx.synchronize();
    }
    real gb1[3], gb2[3];
    pair_gbar(pair, gb1, gb2);
    S->dev = st::make_spline_label_pair(S->ws1.view(), S->ws2.view(), gb1, gb2);
    st::compute_stokes_face_fluxes(ctx.cuda_stream(), S->dev, S->g, S->fws);
    ctx.synchronize();
    S->dview = S->fws.view();

    // Host mirror: host prefilter + make_host_view.
    S->hc1.resize(s1.size());
    S->hc2.resize(s2.size());
    prefilter_periodic_tricubic_bspline_host(S->g, s1.data(), S->hc1.data());
    prefilter_periodic_tricubic_bspline_host(S->g, s2.data(), S->hc2.data());
    {
        const std::vector<real> g1 = d2h(S->ws1.coefficients.data(), S->g.num_cells());
        const std::vector<real> g2 = d2h(S->ws2.coefficients.data(), S->g.num_cells());
        double m = 0.0;
        for (size_t i = 0; i < g1.size(); ++i)
            m = std::max(m, std::max(std::fabs(g1[i] - S->hc1[i]), std::fabs(g2[i] - S->hc2[i])));
        S->coef_diff = m;
    }
    const PeriodicTricubicBSplineView hv1 = make_host_view(S->g, S->hc1.data());
    const PeriodicTricubicBSplineView hv2 = make_host_view(S->g, S->hc2.data());
    S->host = st::make_spline_label_pair(hv1, hv2, gb1, gb2);

    // Spline error at the 64 fixed points (host spline vs analytic fluctuation).
    for (const P3& p : fixed_points64()) {
        real v1, v2, gx, gy, gz;
        evaluate_point(hv1, p.v[0], p.v[1], p.v[2], v1, gx, gy, gz);
        evaluate_point(hv2, p.v[0], p.v[1], p.v[2], v2, gx, gy, gz);
        real e1, e2, ge1[3], ge2[3];
        pair_fluct(pair, p.v[0], p.v[1], p.v[2], e1, ge1, e2, ge2);
        S->e_s1 = std::max(S->e_s1, std::fabs(v1 - e1));
        S->e_s2 = std::max(S->e_s2, std::fabs(v2 - e2));
    }

    if (host_mirror) {
        st::compute_stokes_face_fluxes_host(S->host, S->g, S->hu, S->hv, S->hw);
        S->hview = S->dview;
        S->hview.u = S->hu.data();
        S->hview.v = S->hv.data();
        S->hview.w = S->hw.data();
    }
    std::printf("  [fluxes] pair %s N = %d: E_s(s1) = %.3e, E_s(s2) = %.3e, |coef host - GPU| = %.3e\n",
                pair_name(pair), N, S->e_s1, S->e_s2, S->coef_diff);
    return S;
}

struct DevParticles {
    int n;
    DeviceBuffer<real> x, y, z;
    DeviceBuffer<uint8_t> stt;
    DeviceBuffer<int32_t> wx, wy, wz;
    ParticlesSoA<real> soa;

    explicit DevParticles(int n_) : n(n_), x(n_), y(n_), z(n_), stt(n_), wx(n_), wy(n_), wz(n_) {
        soa.x = x.data();
        soa.y = y.data();
        soa.z = z.data();
        soa.n = n;
        soa.status = stt.data();
        soa.wrapX = wx.data();
        soa.wrapY = wy.data();
        soa.wrapZ = wz.data();
    }
};

struct Snap {
    std::vector<real> x, y, z, clock, rel;
    std::vector<int32_t> wx, wy, wz, cell, cwrap;
    std::vector<uint8_t> stt;
    std::vector<uint32_t> cells;
};

Snap snap(st::PollockTracker& e, DevParticles& P) {
    e.synchronize();
    const size_t n = static_cast<size_t>(P.n);
    Snap s;
    s.x = d2h(P.x.data(), n);
    s.y = d2h(P.y.data(), n);
    s.z = d2h(P.z.data(), n);
    s.wx = d2h(P.wx.data(), n);
    s.wy = d2h(P.wy.data(), n);
    s.wz = d2h(P.wz.data(), n);
    s.stt = d2h(P.stt.data(), n);
    s.clock = d2h(e.clocks(), n);
    s.cells = d2h(e.cell_counts(), n);
    s.cell = d2h(e.state_cells(), 3 * n);
    s.cwrap = d2h(e.state_wraps(), 3 * n);
    s.rel = d2h(e.state_relative(), 3 * n);
    return s;
}

/// Unwrapped SoA position x_u = fma(w, L, x) (L = 1 on every axis).
void snap_xu(const Snap& s, size_t i, double xu[3]) {
    xu[0] = std::fma(static_cast<real>(s.wx[i]), 1.0, s.x[i]);
    xu[1] = std::fma(static_cast<real>(s.wy[i]), 1.0, s.y[i]);
    xu[2] = std::fma(static_cast<real>(s.wz[i]), 1.0, s.z[i]);
}

int count_not(const Snap& s, uint8_t code) {
    int c = 0;
    for (uint8_t v : s.stt)
        c += (v != code);
    return c;
}

const uint64_t kSeed = 20261006ULL;
const int kNp = 256;

/// Contract order: configure, bind_fluxes, bind_particles, inject_box (seeds on
/// x1 = 0), ensure_tracking, prepare.
void setup_engine(st::PollockTracker& e, const st::PeriodicFaceFluxView& f, DevParticles& P) {
    e.configure(st::PollockConfig{});
    e.bind_fluxes(f);
    e.bind_particles(P.soa);
    e.inject_box(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, P.n);
    e.ensure_tracking();
    e.prepare();
}

/// Injected seed positions (host reproduction of the stateless hash is
/// bitwise; downloaded here from the engine arrays right after injection).
std::vector<P3> seeds_after_inject(st::PollockTracker& e, DevParticles& P) {
    e.synchronize();
    const size_t n = static_cast<size_t>(P.n);
    const std::vector<real> x = d2h(P.x.data(), n), y = d2h(P.y.data(), n), z = d2h(P.z.data(), n);
    std::vector<P3> s(n);
    for (size_t i = 0; i < n; ++i) {
        s[i].v[0] = x[i];
        s[i].v[1] = y[i];
        s[i].v[2] = z[i];
    }
    return s;
}

// ============================================================================
// P1: uniform pair (N = 16)
// ============================================================================

void case_p1(const CudaContext& ctx, const FluxSet& U, TestReport& rep) {
    std::printf("\n=== P1 uniform pair (N = 16, n = 16) ===\n");
    DevParticles P(kNp);
    st::PollockTracker e(ctx.cuda_stream(), kSeed);
    setup_engine(e, U.dview, P);
    const std::vector<P3> s0 = seeds_after_inject(e, P);
    e.step_to_x1(1.0);
    const Snap s = snap(e, P);
    int bad_x1 = 0, bad_x23 = 0, bad_clock = 0, bad_cells = 0;
    for (size_t i = 0; i < s.x.size(); ++i) {
        double xu[3];
        snap_xu(s, i, xu);
        bad_x1 += !same_bits(xu[0], 1.0);
        bad_x23 += !(same_bits(xu[1], s0[i].v[1]) && same_bits(xu[2], s0[i].v[2]));
        bad_clock += !same_bits(s.clock[i], 1.0);
        bad_cells += (s.cells[i] != 16u);
    }
    const int bad_st = count_not(s, st::kStatusActive);
    std::printf("  step_to_x1(1.0): x1 != 1 bitwise: %d, x2/x3 changed: %d, clock != 1: %d, cells != 16: %d, "
                "status != 0: %d (of %d)\n",
                bad_x1, bad_x23, bad_clock, bad_cells, bad_st, kNp);
    rep.check(bad_x1 == 0, "P1/step_to_x1_x1_bitwise_1", strf("%d of %d differ (gate 0)", bad_x1, kNp));
    rep.check(bad_x23 == 0, "P1/step_to_x1_x2_x3_bitwise_unchanged", strf("%d of %d differ (gate 0)", bad_x23, kNp));
    rep.check(bad_clock == 0, "P1/step_to_x1_clock_bitwise_1", strf("%d of %d differ (gate 0)", bad_clock, kNp));
    rep.check(bad_cells == 0, "P1/step_to_x1_cells_16", strf("%d of %d differ (gate 0)", bad_cells, kNp));
    rep.check(bad_st == 0, "P1/step_to_x1_status_0", strf("%d of %d nonzero (gate 0)", bad_st, kNp));

    // Fresh prepare (re-injected seeds), two steps of 0.5.
    e.inject_box(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, P.n);
    e.prepare();
    e.step(0.5);
    e.step(0.5);
    const Snap t = snap(e, P);
    int bad_tc = 0;
    double ex1 = 0.0;
    for (size_t i = 0; i < t.x.size(); ++i) {
        double xu[3];
        snap_xu(t, i, xu);
        bad_tc += !same_bits(t.clock[i], 1.0);
        ex1 = std::max(ex1, std::fabs(xu[0] - 1.0));
    }
    const int bad_st2 = count_not(t, st::kStatusActive);
    std::printf("  step(0.5) x2: clock != 1: %d, max |x1_u - 1| = %.3e, status != 0: %d\n", bad_tc, ex1, bad_st2);
    rep.check(bad_tc == 0, "P1/step_time_clock_bitwise_1", strf("%d of %d differ (gate 0)", bad_tc, kNp));
    rep.check(ex1 <= 1e-15, "P1/step_time_x1_1", strf("max |x1_u - 1| = %.3e (gate 1e-15)", ex1));
    rep.check(bad_st2 == 0, "P1/step_time_status_0", strf("%d nonzero", bad_st2));
}

// ============================================================================
// P2: pair A (N = 16, 32)
// ============================================================================

void case_p2(const CudaContext& ctx, const FluxSet& A, TestReport& rep) {
    std::printf("\n=== P2 pair A (N = %d) ===\n", A.N);
    const std::string sfx = strf("_N%d", A.N);
    DevParticles P(kNp);
    st::PollockTracker e(ctx.cuda_stream(), kSeed);
    setup_engine(e, A.dview, P);
    const std::vector<P3> s0 = seeds_after_inject(e, P);
    e.step_to_x1(1.0);
    const Snap s = snap(e, P);
    double m2 = 0.0, m3 = 0.0, mt = 0.0, ml = 0.0;
    for (size_t i = 0; i < s.x.size(); ++i) {
        double xu[3];
        snap_xu(s, i, xu);
        m2 = std::max(m2, std::fabs(xu[1] - s0[i].v[1]));
        m3 = std::max(m3, std::fabs(xu[2] - s0[i].v[2]));
        mt = std::max(mt, std::fabs(s.clock[i] - 1.0));
        ml = std::max(ml, std::fabs(xu[0] - 1.0));
    }
    const int bad1 = count_not(s, st::kStatusActive);
    const double worst = std::max(m2, std::max(m3, mt));
    std::printf("  step_to_x1(1.0): max|dx2| = %.3e, max|dx3| = %.3e, max|tau - 1| = %.3e, max|x1_u - 1| = %.3e, "
                "status != 0: %d\n",
                m2, m3, mt, ml, bad1);
    rep.check(worst <= 1e-13 && bad1 == 0, "P2/one_period_exact" + sfx,
              strf("max(|dx2|, |dx3|, |tau-1|) = %.3e (gate 1e-13), status != 0: %d", worst, bad1));

    e.inject_box(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, P.n);
    e.prepare();
    e.step_to_x1(0.5);
    const Snap h = snap(e, P);
    double mh = 0.0, mlh = 0.0;
    for (size_t i = 0; i < h.x.size(); ++i) {
        double xu[3];
        snap_xu(h, i, xu);
        mh = std::max(mh, std::fabs(xu[1] - s0[i].v[1]));
        mlh = std::max(mlh, std::fabs(xu[0] - 0.5));
    }
    const int bad2 = count_not(h, st::kStatusActive);
    const double gate = 2.0 * A.e_s1 + 1e-13;
    std::printf("  step_to_x1(0.5): max|x2_u - x2_0| = %.3e, E_s(a sin 2 pi x1) = %.3e, gate 2 E_s + 1e-13 = %.3e, "
                "max|x1_u - 0.5| = %.3e, status != 0: %d\n",
                mh, A.e_s1, gate, mlh, bad2);
    rep.check(mh <= gate && bad2 == 0, "P2/half_period_within_spline_error" + sfx,
              strf("max|x2_u - x2_0| = %.3e (gate 2 E_s + 1e-13 = %.3e, E_s = %.3e), status != 0: %d", mh, gate,
                   A.e_s1, bad2));
}

// ============================================================================
// P3: pair B ladder (N = 16, 32, 64, 128), targets 0.25 and 0.30
// ============================================================================

struct Rms {
    double x2 = 0.0, x3 = 0.0, land = 0.0;
    int bad = 0;
};

Rms run_pair_b(const CudaContext& ctx, const FluxSet& B, double target) {
    DevParticles P(kNp);
    st::PollockTracker e(ctx.cuda_stream(), kSeed);
    setup_engine(e, B.dview, P);
    const std::vector<P3> s0 = seeds_after_inject(e, P);
    e.step_to_x1(target);
    const Snap s = snap(e, P);
    Rms r;
    double a2 = 0.0, a3 = 0.0;
    for (size_t i = 0; i < s.x.size(); ++i) {
        double xu[3], ex[3];
        snap_xu(s, i, xu);
        exact_position(kPairB, s0[i].v, target, ex); // c1 = 1: t = x1 displacement (x1_0 = 0)
        a2 += (xu[1] - ex[1]) * (xu[1] - ex[1]);
        a3 += (xu[2] - ex[2]) * (xu[2] - ex[2]);
        r.land = std::max(r.land, std::fabs(xu[0] - target));
    }
    r.x2 = std::sqrt(a2 / s.x.size());
    r.x3 = std::sqrt(a3 / s.x.size());
    r.bad = count_not(s, st::kStatusActive);
    return r;
}

void case_p3(const CudaContext& ctx, const std::vector<std::unique_ptr<FluxSet>>& Bs, TestReport& rep) {
    std::printf("\n=== P3 pair B ladder (a = 0.1, b = 0.08; n = N = 16, 32, 64, 128) ===\n");
    std::printf("  spline floor: E_s at N = 128 = %.3e (max over 64 points of |s_i,spline - s_i,exact|)\n",
                std::max(Bs.back()->e_s1, Bs.back()->e_s2));
    const double targets[2] = {0.25, 0.30};
    for (int ti = 0; ti < 2; ++ti) {
        const double tg = targets[ti];
        const std::string tn = ti == 0 ? "x1_0.25" : "x1_0.30";
        std::vector<Rms> r;
        for (const auto& B : Bs)
            r.push_back(run_pair_b(ctx, *B, tg));
        int bad = 0;
        double land = 0.0;
        for (size_t l = 0; l < r.size(); ++l) {
            std::printf("  target %.2f n = %3d: RMS dx2 = %.3e, RMS dx3 = %.3e, max|x1_u - target| = %.3e, "
                        "status != 0: %d\n",
                        tg, Bs[l]->N, r[l].x2, r[l].x3, r[l].land, r[l].bad);
            bad += r[l].bad;
            land = std::max(land, r[l].land);
        }
        bool mono3 = true, ord3 = true, mono2 = true, ord2 = true;
        std::string o2s, o3s;
        for (size_t l = 0; l + 1 < r.size(); ++l) {
            const double o2 = order2(r[l].x2, r[l + 1].x2);
            const double o3 = order2(r[l].x3, r[l + 1].x3);
            o2s += strf("%s%.2f", l ? ", " : "", o2);
            o3s += strf("%s%.2f", l ? ", " : "", o3);
            mono3 = mono3 && (r[l + 1].x3 < r[l].x3);
            mono2 = mono2 && (r[l + 1].x2 < r[l].x2);
            ord3 = ord3 && (o3 >= 0.8);
            ord2 = ord2 && (o2 >= 0.8);
        }
        std::printf("  target %.2f two-level orders: dx2 [%s], dx3 [%s]\n", tg, o2s.c_str(), o3s.c_str());
        rep.check(bad == 0, "P3/" + tn + "/status_0", strf("%d nonzero over all levels", bad));
        rep.check(mono3 && ord3, "P3/" + tn + "/dx3_monotone_order",
                  strf("RMS dx3 %.3e, %.3e, %.3e, %.3e; orders [%s] (gate: strictly decreasing, every order >= 0.8)",
                       r[0].x3, r[1].x3, r[2].x3, r[3].x3, o3s.c_str()));
        if (ti == 0) {
            std::printf("  [info] target 0.25 dx2 (spline floor, expected order about 4): RMS %.3e, %.3e, %.3e, "
                        "%.3e; orders [%s]\n",
                        r[0].x2, r[1].x2, r[2].x2, r[3].x2, o2s.c_str());
            std::printf("  [info] target 0.25 max|x1_u - 0.25| = %.3e (face target; not gated)\n", land);
        } else {
            rep.check(mono2 && ord2, "P3/" + tn + "/dx2_monotone_order",
                      strf("RMS dx2 %.3e, %.3e, %.3e, %.3e; orders [%s] (gate: strictly decreasing, every order >= "
                           "0.8)",
                           r[0].x2, r[1].x2, r[2].x2, r[3].x2, o2s.c_str()));
            rep.check(land <= 1e-15, "P3/" + tn + "/landing",
                      strf("max|x1_u - 0.30| = %.3e over all levels (gate 1e-15)", land));
        }
    }
}

// ============================================================================
// P4: host core == GPU engine (pair B, N = 32)
// ============================================================================

struct HostRun {
    std::vector<st::PollockState> s;
    std::vector<uint32_t> cells;
    std::vector<uint8_t> code;
};

HostRun host_run(const st::PeriodicFaceFluxView& f, const std::vector<P3>& seeds, double target) {
    HostRun h;
    const st::PollockParams prm{st::PollockConfig{}.max_cells_per_call};
    for (const P3& p : seeds) {
        const real xi[3] = {p.v[0], p.v[1], p.v[2]};
        const int32_t w[3] = {0, 0, 0};
        st::PollockState s{};
        uint8_t c = st::pollock_init_state(f, xi, w, s);
        st::PollockCounters cnt{0u};
        if (c == st::kStatusActive)
            c = st::pollock_advance_to_x1(f, s, target, prm, cnt);
        h.s.push_back(s);
        h.cells.push_back(cnt.cells);
        h.code.push_back(c);
    }
    return h;
}

void compare_host_gpu(const Snap& g, const HostRun& h, const st::PeriodicFaceFluxView& f, const std::string& tag,
                      TestReport& rep) {
    const size_t n = h.s.size();
    double dpos = 0.0, dsoa = 0.0, dclk = 0.0;
    int dcell = 0, dwrap = 0, dcnt = 0, dst = 0;
    for (size_t i = 0; i < n; ++i) {
        st::PollockState gs{};
        for (int a = 0; a < 3; ++a) {
            gs.cell[a] = g.cell[a * n + i];
            gs.w[a] = g.cwrap[a * n + i];
            gs.r[a] = g.rel[a * n + i];
        }
        gs.t = g.clock[i];
        real xg[3], xh[3];
        st::pollock_unwrapped_position(f, gs, xg);
        st::pollock_unwrapped_position(f, h.s[i], xh);
        double xs[3];
        snap_xu(g, i, xs);
        for (int a = 0; a < 3; ++a) {
            dpos = std::max(dpos, std::fabs(xg[a] - xh[a]));
            dsoa = std::max(dsoa, std::fabs(xs[a] - xh[a]));
            dcell += (gs.cell[a] != h.s[i].cell[a]);
            dwrap += (gs.w[a] != h.s[i].w[a]);
        }
        dclk = std::max(dclk, std::fabs(gs.t - h.s[i].t));
        dcnt += (g.cells[i] != h.cells[i]);
        dst += (g.stt[i] != h.code[i]);
    }
    std::printf("  %s: max|state pos diff| = %.3e, max|SoA pos diff| = %.3e, max|clock diff| = %.3e; "
                "differing cells %d, wraps %d, cell counts %d, statuses %d\n",
                tag.c_str(), dpos, dsoa, dclk, dcell, dwrap, dcnt, dst);
    const double m = std::max(dpos, dsoa);
    rep.check(m <= 1e-14 && dclk <= 1e-14, "P4/" + tag + "/positions_clocks",
              strf("max|pos diff| = %.3e, max|clock diff| = %.3e (gate 1e-14 absolute)", m, dclk));
    rep.check(dcell == 0 && dwrap == 0 && dcnt == 0 && dst == 0, "P4/" + tag + "/discrete_identical",
              strf("cells %d, wraps %d, cell counts %d, statuses %d differ (gate 0)", dcell, dwrap, dcnt, dst));
}

void case_p4(const CudaContext& ctx, const FluxSet& B, TestReport& rep) {
    std::printf("\n=== P4 host core vs GPU engine (pair B, N = 32, step_to_x1(1.0)) ===\n");
    DevParticles P(kNp);
    st::PollockTracker e(ctx.cuda_stream(), kSeed);
    setup_engine(e, B.dview, P);
    const std::vector<P3> s0 = seeds_after_inject(e, P);
    e.step_to_x1(1.0);
    const Snap g = snap(e, P);
    {
        double fd = 0.0;
        const std::vector<real> du = d2h(B.dview.u, B.hu.size()), dv = d2h(B.dview.v, B.hv.size()),
                                dw = d2h(B.dview.w, B.hw.size());
        for (size_t i = 0; i < du.size(); ++i)
            fd = std::max(fd, std::max(std::fabs(du[i] - B.hu[i]),
                                       std::max(std::fabs(dv[i] - B.hv[i]), std::fabs(dw[i] - B.hw[i]))));
        std::printf("  [info] max |device face - host-mirror face| = %.3e\n", fd);
        // Host core on the host-mirror face arrays (the specified comparison).
        compare_host_gpu(g, host_run(B.hview, s0, 1.0), B.hview, "host_mirror_faces", rep);
        // Diagnostic split: host core on the downloaded DEVICE face arrays
        // (isolates the tracker core from the flux host/device differences).
        st::PeriodicFaceFluxView dh = B.hview;
        dh.u = du.data();
        dh.v = dv.data();
        dh.w = dw.data();
        compare_host_gpu(g, host_run(dh, s0, 1.0), dh, "device_faces_on_host", rep);
    }
}

// ============================================================================
// P5: determinism
// ============================================================================

void case_p5(const CudaContext& ctx, const FluxSet& B, TestReport& rep) {
    std::printf("\n=== P5 determinism (pair B, N = 32; two fresh engines) ===\n");
    Snap a[2], b[2];
    for (int r = 0; r < 2; ++r) {
        DevParticles P(kNp);
        st::PollockTracker e(ctx.cuda_stream(), kSeed);
        setup_engine(e, B.dview, P);
        e.step_to_x1(1.0);
        a[r] = snap(e, P);
        e.inject_box(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, P.n);
        e.prepare();
        e.step(0.3);
        b[r] = snap(e, P);
    }
    const Snap* s[2][2] = {{&a[0], &a[1]}, {&b[0], &b[1]}};
    const char* nm[2] = {"after_step_to_x1_1.0", "after_step_0.3"};
    for (int c = 0; c < 2; ++c) {
        const Snap& x = *s[c][0];
        const Snap& y = *s[c][1];
        const bool pos = vec_eq(x.x, y.x) && vec_eq(x.y, y.y) && vec_eq(x.z, y.z) && vec_eq(x.rel, y.rel) &&
                         vec_eq(x.cell, y.cell);
        const bool wr = vec_eq(x.wx, y.wx) && vec_eq(x.wy, y.wy) && vec_eq(x.wz, y.wz) && vec_eq(x.cwrap, y.cwrap);
        const bool stt = vec_eq(x.stt, y.stt);
        const bool clk = vec_eq(x.clock, y.clock);
        const bool cnt = vec_eq(x.cells, y.cells);
        const int nz = count_not(x, st::kStatusActive);
        rep.check(pos && wr && stt && clk && cnt, std::string("P5/memcmp_") + nm[c],
                  strf("positions %s, wraps %s, statuses %s, clocks %s, cell counts %s (status != 0 in run: %d)",
                       pos ? "equal" : "DIFFER", wr ? "equal" : "DIFFER", stt ? "equal" : "DIFFER",
                       clk ? "equal" : "DIFFER", cnt ? "equal" : "DIFFER", nz));
    }
}

// ============================================================================
// P6: stagnation path + degenerate inputs
// ============================================================================

void case_p6(const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== P6 stagnation path (hand-built 4^3 face fluxes) ===\n");
    const int n = 4;
    const size_t nc = static_cast<size_t>(n) * n * n;
    std::vector<real> hu(nc), hz(nc, 0.0);
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                hu[i + n * (j + n * k)] = (i == 2) ? -1.0 : 1.0;
    DeviceBuffer<real> du(nc), dv(nc), dw(nc);
    h2d(du.data(), hu);
    h2d(dv.data(), hz);
    h2d(dw.data(), hz);
    st::PeriodicFaceFluxView f{};
    f.u = du.data();
    f.v = dv.data();
    f.w = dw.data();
    f.nx = f.ny = f.nz = n;
    f.dx = f.dy = f.dz = 0.25;
    f.Lx = f.Ly = f.Lz = 1.0;

    const int np = 8;
    {
        DevParticles P(np);
        st::PollockTracker e(ctx.cuda_stream(), kSeed);
        setup_engine(e, f, P);
        e.step_to_x1(1.0);
        const Snap s = snap(e, P);
        int bad = 0;
        double xmin = 1e300, xmax = -1e300, cmin = 1e300, cmax = -1e300;
        bool finite = true;
        uint32_t cmn = 0xffffffffu, cmx = 0u;
        for (int i = 0; i < np; ++i) {
            double xu[3];
            snap_xu(s, i, xu);
            finite = finite && std::isfinite(xu[0]) && std::isfinite(xu[1]) && std::isfinite(xu[2]) &&
                     std::isfinite(s.clock[i]);
            for (int a = 0; a < 3; ++a)
                finite = finite && std::isfinite(s.rel[a * np + i]);
            const bool in = (xu[0] >= 0.25 && xu[0] <= 0.5);
            bad += (s.stt[i] != st::kStatusPollockStagnation) || !in || (s.cells[i] != 1u);
            xmin = std::min(xmin, xu[0]);
            xmax = std::max(xmax, xu[0]);
            cmin = std::min(cmin, s.clock[i]);
            cmax = std::max(cmax, s.clock[i]);
            cmn = std::min(cmn, s.cells[i]);
            cmx = std::max(cmx, s.cells[i]);
        }
        std::printf("  step_to_x1(1.0): status[0] = %u, x1_u in [%.17g, %.17g], clock in [%.17g, %.17g], cells in "
                    "[%u, %u], all finite %s\n",
                    s.stt[0], xmin, xmax, cmin, cmax, cmn, cmx, finite ? "yes" : "no");
        rep.check(bad == 0 && finite, "P6/step_to_x1_stagnation_status_15",
                  strf("%d of %d violate (status 15, 0.25 <= x1_u <= 0.5, cells == 1); x1_u in [%.6g, %.6g], clock "
                       "in [%.6g, %.6g], finite %s",
                       bad, np, xmin, xmax, cmin, cmax, finite ? "yes" : "no"));
    }
    {
        DevParticles P(np);
        st::PollockTracker e(ctx.cuda_stream(), kSeed);
        setup_engine(e, f, P);
        e.step(10.0);
        const Snap s = snap(e, P);
        int bad = 0;
        bool finite = true;
        double xmin = 1e300, xmax = -1e300;
        for (int i = 0; i < np; ++i) {
            double xu[3];
            snap_xu(s, i, xu);
            finite = finite && std::isfinite(xu[0]) && std::isfinite(xu[1]) && std::isfinite(xu[2]) &&
                     std::isfinite(s.clock[i]);
            const bool in = (xu[0] >= 0.25 && xu[0] <= 0.5);
            bad += (s.stt[i] != st::kStatusActive) || !in || !same_bits(s.clock[i], 10.0);
            xmin = std::min(xmin, xu[0]);
            xmax = std::max(xmax, xu[0]);
        }
        std::printf("  step(10.0): status[0] = %u, x1_u in [%.17g, %.17g], clock[0] = %.17g, all finite %s\n",
                    s.stt[0], xmin, xmax, s.clock[0], finite ? "yes" : "no");
        rep.check(bad == 0 && finite, "P6/step_time_active_approach",
                  strf("%d of %d violate (status 0, 0.25 <= x1_u <= 0.5, clock == 10.0 exactly); x1_u in [%.6g, "
                       "%.6g], finite %s",
                       bad, np, xmin, xmax, finite ? "yes" : "no"));
    }
    // Degenerate inputs (header: configure / bind_fluxes -> std::invalid_argument,
    // step before prepare -> std::logic_error), distinct messages.
    {
        std::string m0, m1, m2, m3;
        st::PollockTracker e(ctx.cuda_stream(), kSeed);
        const bool t0 = throws_as<std::invalid_argument>([&] { e.configure(st::PollockConfig{0}); },
                                                         "max_cells_per_call", m0);
        const bool t1 = throws_as<std::invalid_argument>([&] { e.configure(st::PollockConfig{-5}); },
                                                         "max_cells_per_call", m1);
        st::PeriodicFaceFluxView bad = f;
        bad.v = nullptr;
        const bool t2 = throws_as<std::invalid_argument>([&] { e.bind_fluxes(bad); }, "null", m2);
        DevParticles P(np);
        st::PollockTracker e2(ctx.cuda_stream(), kSeed);
        e2.configure(st::PollockConfig{});
        e2.bind_fluxes(f);
        e2.bind_particles(P.soa);
        e2.inject_box(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, P.n);
        e2.ensure_tracking();
        const bool t3 = throws_as<std::logic_error>([&] { e2.step(0.1); }, "before prepare", m3);
        e2.synchronize();
        const bool distinct = (m0 != m2) && (m2 != m3) && (m0 != m3);
        std::printf("  configure(0): \"%s\"\n  configure(-5): \"%s\"\n  bind_fluxes(null v): \"%s\"\n"
                    "  step before prepare: \"%s\"\n",
                    m0.c_str(), m1.c_str(), m2.c_str(), m3.c_str());
        rep.check(t0 && t1, "P6/configure_max_cells_le_0_throws_invalid_argument");
        rep.check(t2, "P6/bind_fluxes_null_throws_invalid_argument");
        rep.check(t3, "P6/step_before_prepare_throws_logic_error");
        rep.check(distinct, "P6/degenerate_messages_distinct");
    }
}

// ============================================================================
// No allocation
// ============================================================================

size_t free_bytes() {
    size_t f = 0, t = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&f, &t));
    return f;
}

void case_no_allocation(const CudaContext& ctx, const FluxSet& B, TestReport& rep) {
    std::printf("\n=== no_allocation (pair B, N = 32) ===\n");
    DevParticles P(kNp);
    st::PollockTracker e(ctx.cuda_stream(), kSeed);
    setup_engine(e, B.dview, P);
    e.step(1.0 / 64); // warm-up
    e.synchronize();
    const size_t f0 = free_bytes();
    for (int i = 0; i < 100; ++i)
        e.step(1.0 / 64);
    e.synchronize();
    const size_t f1 = free_bytes();
    {
        const st::PollockStats ss = e.compute_stats();
        std::printf("  after step block: target %.17g, clock [%.17g, %.17g], active %d of %d, total cells %llu\n",
                    e.target_time(), ss.min_clock, ss.max_clock, ss.n_active, ss.n_particles,
                    static_cast<unsigned long long>(ss.total_cells));
    }
    rep.check(f0 == f1, "no_allocation/cudaMemGetInfo_unchanged_step",
              strf("100 step(1/64): free before %zu, after %zu", f0, f1));

    e.inject_box(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, P.n);
    e.prepare();
    e.step_to_x1(1.0 / 64); // warm-up
    e.synchronize();
    const size_t f2 = free_bytes();
    for (int i = 0; i < 100; ++i)
        e.step_to_x1(static_cast<double>(i + 2) / 64.0);
    e.synchronize();
    const size_t f3 = free_bytes();
    {
        const st::PollockStats ss = e.compute_stats();
        std::printf("  after step_to_x1 block (last target 101/64): clock [%.17g, %.17g], active %d of %d, total "
                    "cells %llu\n",
                    ss.min_clock, ss.max_clock, ss.n_active, ss.n_particles,
                    static_cast<unsigned long long>(ss.total_cells));
    }
    rep.check(f2 == f3, "no_allocation/cudaMemGetInfo_unchanged_step_to_x1",
              strf("100 step_to_x1 (increasing targets): free before %zu, after %zu", f2, f3));
}

// ============================================================================
// Timing record (information only)
// ============================================================================

void case_timing(const CudaContext& ctx, const FluxSet& G) {
    std::printf("\n=== timing_record (information only; pair G, e = 0.03, N = n = 128, 8192 seeds) ===\n");
    DevParticles P(8192);
    st::PollockTracker e(ctx.cuda_stream(), kSeed);
    setup_engine(e, G.dview, P);
    e.synchronize();
    const auto a = std::chrono::steady_clock::now();
    e.step_to_x1(1.0);
    e.synchronize();
    const double ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    const st::PollockStats ss = e.compute_stats();
    std::printf("  step_to_x1(1.0): %.3f ms; active %d, stagnation %d, substep limit %d, non-finite %d; cells "
                "total %llu max %u; tau in [%.6f, %.6f]\n",
                ms, ss.n_active, ss.n_stagnation, ss.n_substep_limit, ss.n_nonfinite,
                static_cast<unsigned long long>(ss.total_cells), ss.max_cells, ss.min_clock, ss.max_clock);
}

} // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    TestReport rep;
    std::vector<std::pair<std::string, double>> times;
    auto timed = [&](const char* name, auto&& fn) {
        const auto a = std::chrono::steady_clock::now();
        try {
            fn();
        } catch (const std::exception& e) {
            rep.check(false, std::string(name) + "/unexpected_exception", e.what());
        }
        times.emplace_back(name, std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
    };
    try {
        CudaContext ctx(0);
        std::unique_ptr<FluxSet> U16, A16, A32, G128;
        std::vector<std::unique_ptr<FluxSet>> B;
        timed("fixtures", [&] {
            std::printf("=== fixtures (SF-28 GPU prefilter + N0 face fluxes, Delta = h) ===\n");
            U16 = build_fluxes(ctx, kPairU, 16, false);
            A16 = build_fluxes(ctx, kPairA, 16, false);
            A32 = build_fluxes(ctx, kPairA, 32, false);
            for (int N : {16, 32, 64, 128})
                B.push_back(build_fluxes(ctx, kPairB, N, N == 32));
            G128 = build_fluxes(ctx, kPairG, 128, false);
        });
        if (U16 && A16 && A32 && B.size() == 4 && G128) {
            timed("P1", [&] { case_p1(ctx, *U16, rep); });
            timed("P2", [&] {
                case_p2(ctx, *A16, rep);
                case_p2(ctx, *A32, rep);
            });
            timed("P3", [&] { case_p3(ctx, B, rep); });
            timed("P4", [&] { case_p4(ctx, *B[1], rep); });
            timed("P5", [&] { case_p5(ctx, *B[1], rep); });
            timed("P6", [&] { case_p6(ctx, rep); });
            timed("no_allocation", [&] { case_no_allocation(ctx, *B[1], rep); });
            timed("timing_record", [&] { case_timing(ctx, *G128); });
        } else {
            rep.check(false, "fixtures/complete");
        }
    } catch (const std::exception& e) {
        rep.check(false, "unexpected_exception", e.what());
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== wall times (information only) ===\n");
    for (const auto& t : times)
        std::printf("  %-20s %8.3f s\n", t.first.c_str(), t.second);
    std::printf("  %-20s %8.3f s\n", "whole executable", wall);
    std::printf("\n%d checks, %d failed, overall %s\n", rep.checks, rep.fails, rep.overall_pass ? "PASS" : "FAIL");
    return rep.overall_pass ? 0 : 1;
}
