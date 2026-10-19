/**
 * @file face_flux_repack_tests.cu
 * @brief SF-37 N3: tests of the CompactMAC -> periodic face repack (C-REPACK) and of the Pollock lattice
 *        tracker (C-POLLOCK-LIB), ctest `lagrangian_face_flux_repack` (<= 60 s, deterministic).
 *
 *  (a) repack bitwise on synthetic CompactMAC arrays (distinct value per unique entry; duplicate planes
 *      copied from plane 0, or poisoned in a second pass to prove they are never read), non-cubic
 *      (6, 5, 4) and cubic 8^3; upload_and_repack equals the direct repack; size validation throws.
 *  (b) k = 1 (direct solve_affine_periodic_flow, N = 16, unit cube): U bitwise 1, V = W bitwise 0; the
 *      repacked faces drive straight lines in the Pollock lattice tracker (M = 8, planes {0.25, .5, 1, 2},
 *      times {.5, 1, 2}): x2, x3 bitwise equal to the seeds, clock t = x1 - X0, x1 = X0 + t, statuses 0,
 *      weights 1/Np, seed_nonpos 0.
 *  (c) smooth non-uniform K (analytic, N = 16): statuses, finite moments, independent host recomputation of
 *      the weights and moments from the per-particle arrays (<= 1e-15 relative), independent host replay
 *      of the SF-32 core (agreement to the platform libm difference), determinism of two calls.
 *  (d) validation errors throw.
 *
 * This TU calls device code only through the library; compiled with --fmad=false like the others.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/blas/fill.cuh"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/physics/particles/lagrangian_labels/FaceFluxRepack.cuh"
#include "src/physics/particles/lagrangian_labels/PollockComparator.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace ll = macroflow3d::lagrangian_labels;
namespace stt = macroflow3d::physics::particles::streamline_tracker;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

struct TestReport {
    bool overall_pass = true;
    int checks = 0, failed = 0;
    void check(bool cond, const std::string& name, const std::string& detail = "") {
        ++checks;
        if (!cond)
            ++failed;
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(), detail.empty() ? "" : "  ",
                    detail.c_str());
        overall_pass = overall_pass && cond;
    }
};

std::string strf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return std::string(buf);
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

template <class T> bool same_bits_vec(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

std::vector<real> d2h(const real* p, std::size_t n) {
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

// ---------------------------------------------------------------------------------------------
// (a) synthetic CompactMAC arrays
// ---------------------------------------------------------------------------------------------

double enc(int axis, int i, int j, int k) {
    return static_cast<double>(((axis * 64 + i) * 64 + j) * 64 + k) + 0.25;
}

struct Compact {
    std::vector<real> U, V, W;
};

/// Unique entries encode (axis, i, j, k); the periodic duplicate planes copy plane 0, or hold `poison`.
Compact make_compact(int nx, int ny, int nz, bool poison_duplicates) {
    const double poison = -777.0;
    Compact c;
    c.U.resize(ll::compact_mac_u_size(nx, ny, nz));
    c.V.resize(ll::compact_mac_v_size(nx, ny, nz));
    c.W.resize(ll::compact_mac_w_size(nx, ny, nz));
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i)
                c.U[static_cast<std::size_t>(i) + static_cast<std::size_t>(nx + 1) * (j + static_cast<std::size_t>(ny) * k)] =
                    (i == nx) ? (poison_duplicates ? poison : enc(0, 0, j, k)) : enc(0, i, j, k);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i < nx; ++i)
                c.V[static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * j +
                    static_cast<std::size_t>(nx) * (ny + 1) * k] =
                    (j == ny) ? (poison_duplicates ? poison : enc(1, i, 0, k)) : enc(1, i, j, k);
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                c.W[static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * j +
                    static_cast<std::size_t>(nx) * ny * k] =
                    (k == nz) ? (poison_duplicates ? poison : enc(2, i, j, 0)) : enc(2, i, j, k);
    return c;
}

struct Periodic {
    std::vector<real> u, v, w;
};

Periodic expected_periodic(int nx, int ny, int nz) {
    Periodic e;
    const std::size_t n = static_cast<std::size_t>(nx) * ny * nz;
    e.u.resize(n);
    e.v.resize(n);
    e.w.resize(n);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t q = static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * (j + static_cast<std::size_t>(ny) * k);
                e.u[q] = enc(0, i, j, k);
                e.v[q] = enc(1, i, j, k);
                e.w[q] = enc(2, i, j, k);
            }
    return e;
}

void upload(DeviceBuffer<real>& d, const std::vector<real>& h) {
    d = DeviceBuffer<real>(h.size());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
}

void case_a_one(TestReport& rep, CudaContext& ctx, int nx, int ny, int nz, bool poison) {
    const std::string tag = strf("(a) %dx%dx%d%s", nx, ny, nz, poison ? " poisoned-duplicates" : "");
    const Compact c = make_compact(nx, ny, nz, poison);
    const Periodic e = expected_periodic(nx, ny, nz);
    const std::size_t n = e.u.size();

    DeviceBuffer<real> dU, dV, dW;
    upload(dU, c.U);
    upload(dV, c.V);
    upload(dW, c.W);
    DeviceBuffer<real> du(n), dv(n), dw(n);
    ll::repack_compact_mac_to_periodic_faces(ctx, nx, ny, nz, dU.span(), dV.span(), dW.span(), du.span(),
                                             dv.span(), dw.span());
    ctx.synchronize();
    const auto gu = d2h(du.data(), n), gv = d2h(dv.data(), n), gw = d2h(dw.data(), n);
    rep.check(same_bits_vec(gu, e.u) && same_bits_vec(gv, e.v) && same_bits_vec(gw, e.w),
              tag + " device repack == unique planes bitwise", strf("n = %zu per component", n));

    ll::PeriodicFaceFluxes pf = ll::PeriodicFaceFluxes::upload_and_repack(ctx, c.U, c.V, c.W, nx, ny, nz, 0.125);
    const stt::PeriodicFaceFluxView f = pf.view();
    const auto hu = d2h(f.u, n), hv = d2h(f.v, n), hw = d2h(f.w, n);
    rep.check(same_bits_vec(hu, gu) && same_bits_vec(hv, gv) && same_bits_vec(hw, gw),
              tag + " upload_and_repack == direct repack bitwise");
    const bool geom = f.nx == nx && f.ny == ny && f.nz == nz && f.dx == 0.125 && f.dy == 0.125 &&
                      f.dz == 0.125 && f.Lx == nx * 0.125 && f.Ly == ny * 0.125 && f.Lz == nz * 0.125;
    rep.check(geom && pf.device_bytes() == 3 * n * sizeof(real), tag + " view geometry (dx = dy = dz = h, L = n h)",
              strf("Lx %.17g Ly %.17g Lz %.17g bytes %zu", f.Lx, f.Ly, f.Lz, pf.device_bytes()));
}

template <class F> bool throws_invalid(F&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void case_a(TestReport& rep, CudaContext& ctx) {
    std::printf("\n=== (a) repack bitwise ===\n");
    case_a_one(rep, ctx, 6, 5, 4, false);
    case_a_one(rep, ctx, 6, 5, 4, true);
    case_a_one(rep, ctx, 8, 8, 8, false);
    case_a_one(rep, ctx, 3, 7, 2, true);

    // size validation
    const int nx = 6, ny = 5, nz = 4;
    Compact c = make_compact(nx, ny, nz, false);
    const std::size_t n = static_cast<std::size_t>(nx) * ny * nz;
    DeviceBuffer<real> dU, dV, dW;
    upload(dU, c.U);
    upload(dV, c.V);
    upload(dW, c.W);
    DeviceBuffer<real> du(n), dv(n), dw(n), dshort(n - 1);
    rep.check(throws_invalid([&] {
                  ll::repack_compact_mac_to_periodic_faces(ctx, nx, ny, nz, DeviceSpan<const real>(dU.data(), c.U.size() - 1),
                                                           dV.span(), dW.span(), du.span(), dv.span(), dw.span());
              }),
              "(a) repack: U one entry short throws");
    rep.check(throws_invalid([&] {
                  ll::repack_compact_mac_to_periodic_faces(ctx, nx, ny, nz, dU.span(), dV.span(), dW.span(), du.span(),
                                                           dshort.span(), dw.span());
              }),
              "(a) repack: output v_p one entry short throws");
    rep.check(throws_invalid([&] {
                  ll::repack_compact_mac_to_periodic_faces(ctx, 0, ny, nz, dU.span(), dV.span(), dW.span(), du.span(),
                                                           dv.span(), dw.span());
              }),
              "(a) repack: nx = 0 throws");
    std::vector<real> shortW = c.W;
    shortW.pop_back();
    rep.check(throws_invalid([&] { (void)ll::PeriodicFaceFluxes::upload_and_repack(ctx, c.U, c.V, shortW, nx, ny, nz, 0.1); }),
              "(a) upload_and_repack: W one entry short throws");
    rep.check(throws_invalid([&] { (void)ll::PeriodicFaceFluxes::upload_and_repack(ctx, c.U, c.V, c.W, nx, ny, nz, 0.0); }),
              "(a) upload_and_repack: h = 0 throws");
    rep.check(throws_invalid([&] { (void)ll::PeriodicFaceFluxes::upload_and_repack(ctx, c.U, c.V, c.W, nx, ny, nz,
                                                                                   std::numeric_limits<double>::quiet_NaN()); }),
              "(a) upload_and_repack: h = NaN throws");
    bool logic = false;
    try {
        (void)ll::PeriodicFaceFluxes().view();
    } catch (const std::logic_error&) {
        logic = true;
    }
    rep.check(logic, "(a) empty PeriodicFaceFluxes::view() throws logic_error");
}

// ---------------------------------------------------------------------------------------------
// flow helpers (SF-19)
// ---------------------------------------------------------------------------------------------

struct FlowFaces {
    std::vector<real> U, V, W;
    physics::AffinePeriodicFlowReport report;
};

physics::AffinePeriodicFlowConfig flow_config() {
    physics::AffinePeriodicFlowConfig cfg;
    cfg.qbar[0] = real{1};
    cfg.qbar[1] = real{0};
    cfg.qbar[2] = real{0};
    cfg.linear.rtol = real{1e-11};
    cfg.linear.max_iter = 2000;
    cfg.linear.check_every = 10;
    cfg.mg = multigrid::MGConfig{};
    return cfg;
}

FlowFaces solve_faces(CudaContext& ctx, int N, const std::vector<real>& hostK) {
    const real h = real{1} / static_cast<real>(N);
    const Grid3D grid(N, N, N, h, h, h);
    DeviceBuffer<real> K;
    upload(K, hostK);
    DeviceBuffer<real> u(ll::compact_mac_u_size(N, N, N)), v(ll::compact_mac_v_size(N, N, N)),
        w(ll::compact_mac_w_size(N, N, N));
    physics::AffinePeriodicFlowWorkspace ws;
    FlowFaces out;
    out.report = physics::solve_affine_periodic_flow(
        ctx, grid, DeviceSpan<const real>(K.span()), flow_config(),
        physics::AffinePeriodicVelocityView{u.span(), v.span(), w.span()}, ws);
    ctx.synchronize();
    out.U = d2h(u.data(), u.size());
    out.V = d2h(v.data(), v.size());
    out.W = d2h(w.data(), w.size());
    return out;
}

// ---------------------------------------------------------------------------------------------
// (b) k = 1
// ---------------------------------------------------------------------------------------------

void case_b(TestReport& rep, CudaContext& ctx) {
    std::printf("\n=== (b) k = 1, N = 16, Pollock lattice on the repacked SF-19 faces ===\n");
    const int N = 16;
    const FlowFaces ff = solve_faces(ctx, N, std::vector<real>(static_cast<std::size_t>(N) * N * N, real{1}));
    double du = 0.0, dvw = 0.0;
    for (double x : ff.U)
        du = std::max(du, std::fabs(x - 1.0));
    for (double x : ff.V)
        dvw = std::max(dvw, std::fabs(x));
    for (double x : ff.W)
        dvw = std::max(dvw, std::fabs(x));
    rep.check(du == 0.0 && dvw == 0.0, "(b) K = 1: U bitwise 1, V = W bitwise 0", strf("max|U-1| = %.3e, max|V|,|W| = %.3e", du, dvw));

    const ll::PeriodicFaceFluxes pf =
        ll::PeriodicFaceFluxes::upload_and_repack(ctx, ff.U, ff.V, ff.W, N, N, N, 1.0 / N);
    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = 8;
    const std::vector<double> planes = {0.25, 0.5, 1.0, 2.0}, times = {0.5, 1.0, 2.0};
    const stt::PollockParams prm{4096};
    const ll::PollockLatticeResult r = ll::track_pollock_lattice(ctx, pf.view(), inj, planes, times, prm);
    const std::size_t Np = 64, P = 4, T = 3;
    rep.check(r.Np == 64 && r.P == 4 && r.T == 3, "(b) sizes", strf("Np %d P %d T %d", r.Np, r.P, r.T));

    int bad_status = 0, bad_reach = 0;
    for (std::size_t p = 0; p < Np; ++p) {
        bad_status += (r.status_plane[p] != 0) + (r.status_time[p] != 0);
        bad_reach += (r.reached_plane[p] != 4) + (r.reached_time[p] != 3);
    }
    rep.check(bad_status == 0 && bad_reach == 0, "(b) all statuses 0 and every plane/time reached",
              strf("status != 0: %d, not reached: %d, hist plane[0] = %lld, time[0] = %lld", bad_status, bad_reach,
                   r.status_hist_plane[0], r.status_hist_time[0]));

    int bad_x23 = 0, bad_clock_bits = 0, bad_x1_1e15 = 0, bad_time_x23 = 0;
    double max_clock = 0.0, max_x1 = 0.0;
    for (std::size_t p = 0; p < Np; ++p) {
        for (std::size_t m = 0; m < P; ++m) {
            bad_x23 += !(same_bits(r.planes_x23[(p * P + m) * 2], r.seeds[3 * p + 1]) &&
                         same_bits(r.planes_x23[(p * P + m) * 2 + 1], r.seeds[3 * p + 2]));
            const double t = r.planes_t[p * P + m];
            const double want = planes[m] - inj.X0;
            bad_clock_bits += !same_bits(t, want);
            max_clock = std::max(max_clock, std::fabs(t - want));
        }
        for (std::size_t n = 0; n < T; ++n) {
            const double* x = &r.times_x[(p * T + n) * 3];
            const double e1 = std::fabs(x[0] - (inj.X0 + times[n]));
            max_x1 = std::max(max_x1, e1);
            bad_x1_1e15 += !(e1 <= 1e-15);
            bad_time_x23 += !(same_bits(x[1], r.seeds[3 * p + 1]) && same_bits(x[2], r.seeds[3 * p + 2]));
        }
    }
    rep.check(bad_x23 == 0, "(b) plane records x2, x3 bitwise equal the seeds", strf("%d of %zu differ", bad_x23, Np * P));
    rep.check(bad_clock_bits == 0, "(b) clock t == x1 - X0 bitwise (face-aligned planes, as SF-32 P1)",
              strf("%d of %zu differ, max |t - (x1 - X0)| = %.3e", bad_clock_bits, Np * P, max_clock));
    rep.check(bad_x1_1e15 == 0, "(b) time records x1 = X0 + t within 1e-15", strf("max |x1 - (X0 + t)| = %.3e", max_x1));
    rep.check(bad_time_x23 == 0, "(b) time records x2, x3 bitwise unchanged", strf("%d of %zu differ", bad_time_x23, Np * T));
    bool w_equal = true;
    for (std::size_t p = 0; p < Np; ++p)
        w_equal = w_equal && same_bits(r.weights[p], 1.0 / 64.0) && same_bits(r.weight_raw[p], 1.0);
    rep.check(w_equal && r.seed_nonpos == 0, "(b) raw weights 1, weights all 1/Np bitwise, seed_nonpos 0",
              strf("seed_nonpos = %lld", r.seed_nonpos));
    bool mom = true;
    double max_mt = 0.0;
    for (std::size_t m = 0; m < P; ++m) {
        const ll::PollockPlaneMoments& pm = r.moments_planes[m];
        mom = mom && pm.contributing == 64 && pm.non_ok == 0 && pm.var_t == 0.0;
        max_mt = std::max(max_mt, std::fabs(pm.mean_t - planes[m]));
    }
    for (std::size_t n = 0; n < T; ++n)
        mom = mom && r.moments_time[n].active == 64 && std::fabs(r.moments_time[n].mean[0] - times[n]) <= 1e-15 &&
              r.moments_time[n].var[0] <= 1e-30;
    rep.check(mom && max_mt <= 1e-15, "(b) moments: contributing 64, non_ok 0, var t = 0, mean t = x1, time mean x1 = t",
              strf("max |mean_t - x1| = %.3e", max_mt));
    // 16 cells per period: the plane pass takes 4 + 4 + 8 + 16 = 32 cell steps to x1 = 2; the time pass 2 x 16 = 32.
    std::uint64_t cpmin = ~0ull, cpmax = 0, ctmin = ~0ull, ctmax = 0;
    for (std::size_t p = 0; p < Np; ++p) {
        cpmin = std::min(cpmin, r.cells[2 * p]);
        cpmax = std::max(cpmax, r.cells[2 * p]);
        ctmin = std::min(ctmin, r.cells[2 * p + 1]);
        ctmax = std::max(ctmax, r.cells[2 * p + 1]);
    }
    rep.check(cpmin == 32 && cpmax == 32 && ctmin == 32 && ctmax == 32 && r.cells_plane == 32 * Np &&
                  r.cells_time == 32 * Np,
              "(b) cell steps per particle: plane pass 32, time pass 32",
              strf("plane [%llu, %llu], time [%llu, %llu], totals %llu / %llu", (unsigned long long)cpmin,
                   (unsigned long long)cpmax, (unsigned long long)ctmin, (unsigned long long)ctmax, r.cells_plane,
                   r.cells_time));
}

// ---------------------------------------------------------------------------------------------
// (c) smooth non-uniform K
// ---------------------------------------------------------------------------------------------

std::vector<real> smooth_K(int N) {
    std::vector<real> K(static_cast<std::size_t>(N) * N * N);
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const double x = (i + 0.5) / N, y = (j + 0.5) / N, z = (k + 0.5) / N;
                const double Y = 0.4 * std::sin(kTwoPi * x) * std::cos(kTwoPi * y) + 0.3 * std::cos(kTwoPi * (y + z)) +
                                 0.2 * std::sin(kTwoPi * (x - z));
                K[static_cast<std::size_t>(i) + static_cast<std::size_t>(N) * (j + static_cast<std::size_t>(N) * k)] = std::exp(Y);
            }
    return K;
}

/// Independent biased weighted moments (written separately from the library code).
struct Mom {
    double mean, var;
    int count;
};
Mom ref_moments(const std::vector<double>& w, const std::vector<double>& val, const std::vector<char>& sel) {
    double W = 0.0, S = 0.0;
    int c = 0;
    for (std::size_t p = 0; p < w.size(); ++p)
        if (sel[p]) {
            W += w[p];
            S += w[p] * val[p];
            ++c;
        }
    Mom m{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), c};
    if (c == 0 || !(W > 0.0))
        return m;
    m.mean = S / W;
    double V = 0.0;
    for (std::size_t p = 0; p < w.size(); ++p)
        if (sel[p]) {
            const double d = val[p] - m.mean;
            V += w[p] * d * d;
        }
    m.var = V / W;
    return m;
}

double rel(double a, double b) {
    const double d = std::fabs(a - b);
    const double s = std::max(std::fabs(a), std::fabs(b));
    return s > 0.0 ? d / s : d;
}

bool results_bitwise_equal(const ll::PollockLatticeResult& a, const ll::PollockLatticeResult& b) {
    bool ok = same_bits_vec(a.seeds, b.seeds) && same_bits_vec(a.weight_raw, b.weight_raw) &&
              same_bits_vec(a.weights, b.weights) && same_bits_vec(a.status_plane, b.status_plane) &&
              same_bits_vec(a.status_time, b.status_time) && same_bits_vec(a.reached_plane, b.reached_plane) &&
              same_bits_vec(a.reached_time, b.reached_time) && same_bits_vec(a.planes_x23, b.planes_x23) &&
              same_bits_vec(a.planes_t, b.planes_t) && same_bits_vec(a.times_x, b.times_x) &&
              same_bits_vec(a.cells, b.cells) && a.seed_nonpos == b.seed_nonpos && a.cells_plane == b.cells_plane &&
              a.cells_time == b.cells_time && a.status_hist_plane == b.status_hist_plane &&
              a.status_hist_time == b.status_hist_time;
    for (std::size_t m = 0; m < a.moments_planes.size() && ok; ++m) {
        const auto &x = a.moments_planes[m], &y = b.moments_planes[m];
        ok = same_bits(x.mean_x2, y.mean_x2) && same_bits(x.var_x2, y.var_x2) && same_bits(x.mean_x3, y.mean_x3) &&
             same_bits(x.var_x3, y.var_x3) && same_bits(x.mean_t, y.mean_t) && same_bits(x.var_t, y.var_t) &&
             x.contributing == y.contributing && x.non_ok == y.non_ok;
    }
    for (std::size_t n = 0; n < a.moments_time.size() && ok; ++n) {
        const auto &x = a.moments_time[n], &y = b.moments_time[n];
        for (int k = 0; k < 3; ++k)
            ok = ok && same_bits(x.mean[k], y.mean[k]) && same_bits(x.var[k], y.var[k]);
        ok = ok && x.active == y.active;
    }
    return ok;
}

void case_c(TestReport& rep, CudaContext& ctx) {
    std::printf("\n=== (c) smooth non-uniform K, N = 16 ===\n");
    const int N = 16;
    const FlowFaces ff = solve_faces(ctx, N, smooth_K(N));
    std::printf("  SF-19 solve: div_max_abs %.3e, G = (%.6e, %.6e, %.6e)\n", (double)ff.report.div_max_abs,
                (double)ff.report.G[0], (double)ff.report.G[1], (double)ff.report.G[2]);
    const ll::PeriodicFaceFluxes pf =
        ll::PeriodicFaceFluxes::upload_and_repack(ctx, ff.U, ff.V, ff.W, N, N, N, 1.0 / N);
    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = 8;
    const std::vector<double> planes = {0.5, 1.0}, times = {1.0};
    const stt::PollockParams prm{100000};
    const ll::PollockLatticeResult r = ll::track_pollock_lattice(ctx, pf.view(), inj, planes, times, prm);
    const std::size_t Np = 64, P = 2, T = 1;

    std::printf("  statuses plane pass:");
    for (int s = 0; s < ll::kPollockStatusSlots; ++s)
        if (r.status_hist_plane[s])
            std::printf(" [%d]=%lld", s, r.status_hist_plane[s]);
    std::printf("; time pass:");
    for (int s = 0; s < ll::kPollockStatusSlots; ++s)
        if (r.status_hist_time[s])
            std::printf(" [%d]=%lld", s, r.status_hist_time[s]);
    std::printf("; cells plane %llu time %llu; seed_nonpos %lld\n", r.cells_plane, r.cells_time, r.seed_nonpos);
    for (std::size_t m = 0; m < P; ++m) {
        const auto& pm = r.moments_planes[m];
        std::printf("  plane %.2f: contributing %d non_ok %d mean(x2,x3,t) = (%.6f, %.6f, %.6f) var = (%.3e, %.3e, %.3e)\n",
                    pm.x1, pm.contributing, pm.non_ok, pm.mean_x2, pm.mean_x3, pm.mean_t, pm.var_x2, pm.var_x3, pm.var_t);
    }
    {
        const auto& pt = r.moments_time[0];
        std::printf("  time %.2f: active %d mean x = (%.6f, %.6f, %.6f) var = (%.3e, %.3e, %.3e)\n", pt.time, pt.active,
                    pt.mean[0], pt.mean[1], pt.mean[2], pt.var[0], pt.var[1], pt.var[2]);
    }

    bool finite = true;
    for (std::size_t m = 0; m < P; ++m) {
        const auto& pm = r.moments_planes[m];
        finite = finite && pm.contributing > 0 && std::isfinite(pm.mean_x2) && std::isfinite(pm.var_x2) &&
                 std::isfinite(pm.mean_x3) && std::isfinite(pm.var_x3) && std::isfinite(pm.mean_t) && std::isfinite(pm.var_t);
    }
    for (int k = 0; k < 3; ++k)
        finite = finite && r.moments_time[0].active > 0 && std::isfinite(r.moments_time[0].mean[k]) && std::isfinite(r.moments_time[0].var[k]);
    rep.check(finite, "(c) moments finite with contributing particles");
    long long nonzero = 0;
    for (int s = 1; s < ll::kPollockStatusSlots; ++s)
        nonzero += r.status_hist_plane[s] + r.status_hist_time[s];
    rep.check(nonzero == 0, "(c) statuses recorded; all active on this smooth field", strf("non-active: %lld", nonzero));

    // weights recomputed independently
    double total = 0.0;
    for (double x : r.weight_raw)
        if (std::isfinite(x) && x > 0.0)
            total += x;
    double max_w = 0.0, wsum = 0.0;
    for (std::size_t p = 0; p < Np; ++p) {
        const double raw = r.weight_raw[p];
        const double want = (std::isfinite(raw) && raw > 0.0) ? raw / total : 0.0;
        max_w = std::max(max_w, std::fabs(r.weights[p] - want));
        wsum += r.weights[p];
    }
    rep.check(max_w == 0.0 && std::fabs(wsum - 1.0) <= 1e-14, "(c) weights = max(raw, 0)/sum(positive raw), sum 1",
              strf("max dev %.3e, |sum - 1| = %.3e", max_w, std::fabs(wsum - 1.0)));

    // moments recomputed from per-particle arrays
    double max_rel = 0.0;
    bool counts = true;
    for (std::size_t m = 0; m < P; ++m) {
        std::vector<char> sel(Np);
        std::vector<double> vx2(Np), vx3(Np), vt(Np);
        for (std::size_t p = 0; p < Np; ++p) {
            sel[p] = std::isfinite(r.planes_x23[(p * P + m) * 2]) ? 1 : 0;
            vx2[p] = r.planes_x23[(p * P + m) * 2];
            vx3[p] = r.planes_x23[(p * P + m) * 2 + 1];
            vt[p] = r.planes_t[p * P + m];
        }
        const Mom a = ref_moments(r.weights, vx2, sel), b = ref_moments(r.weights, vx3, sel), c = ref_moments(r.weights, vt, sel);
        const auto& pm = r.moments_planes[m];
        counts = counts && a.count == pm.contributing;
        max_rel = std::max({max_rel, rel(a.mean, pm.mean_x2), rel(a.var, pm.var_x2), rel(b.mean, pm.mean_x3),
                            rel(b.var, pm.var_x3), rel(c.mean, pm.mean_t), rel(c.var, pm.var_t)});
    }
    {
        std::vector<char> sel(Np);
        for (std::size_t p = 0; p < Np; ++p)
            sel[p] = std::isfinite(r.times_x[p * 3]) ? 1 : 0;
        for (int k = 0; k < 3; ++k) {
            std::vector<double> v(Np);
            for (std::size_t p = 0; p < Np; ++p)
                v[p] = r.times_x[(p * T) * 3 + k];
            const Mom a = ref_moments(r.weights, v, sel);
            counts = counts && a.count == r.moments_time[0].active;
            max_rel = std::max({max_rel, rel(a.mean, r.moments_time[0].mean[k]), rel(a.var, r.moments_time[0].var[k])});
        }
    }
    rep.check(counts && max_rel <= 1e-15, "(c) host moments == independent recomputation (<= 1e-15 relative)",
              strf("max relative difference %.3e", max_rel));

    // independent host replay of the SF-32 core on host copies of the repacked faces
    const stt::PeriodicFaceFluxView fd = pf.view();
    const std::size_t nc = static_cast<std::size_t>(N) * N * N;
    const std::vector<real> hu = d2h(fd.u, nc), hv = d2h(fd.v, nc), hw = d2h(fd.w, nc);
    stt::PeriodicFaceFluxView fh = fd;
    fh.u = hu.data();
    fh.v = hv.data();
    fh.w = hw.data();
    double max_pos = 0.0, max_t = 0.0, max_raw = 0.0;
    std::uint64_t cell_mismatch = 0;
    for (std::size_t p = 0; p < Np; ++p) {
        const real xi[3] = {r.seeds[3 * p], r.seeds[3 * p + 1], r.seeds[3 * p + 2]};
        const int32_t w0[3] = {0, 0, 0};
        stt::PollockState st{};
        if (stt::pollock_init_state(fh, xi, w0, st) != stt::kStatusActive)
            continue;
        stt::pollock_detail::CellEval ev;
        if (stt::pollock_detail::eval_cell(fh, st, ev) == stt::kStatusActive)
            max_raw = std::max(max_raw, std::fabs(ev.vp[0] - r.weight_raw[p]));
        std::uint64_t cells = 0;
        for (std::size_t m = 0; m < P; ++m) {
            stt::PollockCounters cnt{0u};
            if (stt::pollock_advance_to_x1(fh, st, planes[m], prm, cnt) != stt::kStatusActive)
                break;
            cells += cnt.cells;
            real xu[3];
            stt::pollock_unwrapped_position(fh, st, xu);
            max_pos = std::max({max_pos, std::fabs(xu[1] - r.planes_x23[(p * P + m) * 2]),
                                std::fabs(xu[2] - r.planes_x23[(p * P + m) * 2 + 1])});
            max_t = std::max(max_t, std::fabs(st.t - r.planes_t[p * P + m]));
        }
        cell_mismatch += (cells != r.cells[2 * p]);
    }
    rep.check(max_raw == 0.0, "(c) raw weights == host replay of the core's cell evaluation (bitwise)", strf("max dev %.3e", max_raw));
    rep.check(max_pos <= 1e-11 && max_t <= 1e-11 && cell_mismatch == 0,
              "(c) plane records == host replay of the SF-32 core (device vs host libm: <= 1e-11)",
              strf("max |dx23| = %.3e, max |dt| = %.3e, cell-count mismatches %llu", max_pos, max_t,
                   (unsigned long long)cell_mismatch));

    // determinism
    const ll::PollockLatticeResult r2 = ll::track_pollock_lattice(ctx, pf.view(), inj, planes, times, prm);
    rep.check(results_bitwise_equal(r, r2), "(c) two calls bitwise identical (records, statuses, counts, moments)");
}

// ---------------------------------------------------------------------------------------------
// (d) validation
// ---------------------------------------------------------------------------------------------

void case_d(TestReport& rep, CudaContext& ctx) {
    std::printf("\n=== (d) validation errors ===\n");
    const int N = 8;
    const ll::PeriodicFaceFluxes pf = ll::PeriodicFaceFluxes::upload_and_repack(
        ctx, std::vector<real>(ll::compact_mac_u_size(N, N, N), 1.0), std::vector<real>(ll::compact_mac_v_size(N, N, N), 0.0),
        std::vector<real>(ll::compact_mac_w_size(N, N, N), 0.0), N, N, N, 1.0 / N);
    ll::InjectionSpec inj;
    inj.M = 4;
    const stt::PollockParams prm{1000};
    auto call = [&](const ll::InjectionSpec& i, const std::vector<double>& pl, const std::vector<double>& ti,
                    const stt::PollockParams& pr) { (void)ll::track_pollock_lattice(ctx, pf.view(), i, pl, ti, pr); };
    rep.check(throws_invalid([&] { call(inj, {1.0, 0.5}, {}, prm); }), "(d) unsorted planes throw");
    rep.check(throws_invalid([&] { call(inj, {0.5, 0.5}, {}, prm); }), "(d) repeated plane throws");
    rep.check(throws_invalid([&] { call(inj, {}, {1.0, 0.5}, prm); }), "(d) unsorted times throw");
    rep.check(throws_invalid([&] { call(inj, {}, {}, prm); }), "(d) both lists empty throw");
    rep.check(throws_invalid([&] { call(inj, {0.0}, {}, prm); }), "(d) plane <= X0 throws");
    rep.check(throws_invalid([&] { call(inj, {}, {0.0}, prm); }), "(d) time = 0 throws");
    rep.check(throws_invalid([&] { call(inj, {std::numeric_limits<double>::infinity()}, {}, prm); }), "(d) non-finite plane throws");
    rep.check(throws_invalid([&] { call(inj, {0.5}, {}, stt::PollockParams{0}); }), "(d) max_cells_per_call = 0 throws");
    ll::InjectionSpec bad = inj;
    bad.M = 0;
    rep.check(throws_invalid([&] { call(bad, {0.5}, {}, prm); }), "(d) M = 0 throws");
    bad = inj;
    bad.w2a = 1.0;
    bad.w2b = 0.5;
    rep.check(throws_invalid([&] { call(bad, {0.5}, {}, prm); }), "(d) empty window throws");
    // planes only / times only are accepted
    bool ok = true;
    try {
        call(inj, {0.5}, {}, prm);
        call(inj, {}, {0.5}, prm);
    } catch (...) {
        ok = false;
    }
    rep.check(ok, "(d) planes-only and times-only calls run");
}

} // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    TestReport rep;
    CudaContext ctx(0);
    std::printf("lagrangian_face_flux_repack\n");
    case_a(rep, ctx);
    case_b(rep, ctx);
    case_c(rep, ctx);
    case_d(rep, ctx);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n%d checks, %d failed, %.1f s\n", rep.checks, rep.failed, s);
    std::printf("%s\n", rep.overall_pass ? "ALL PASS" : "SOME FAILED");
    return rep.overall_pass ? 0 : 1;
}
