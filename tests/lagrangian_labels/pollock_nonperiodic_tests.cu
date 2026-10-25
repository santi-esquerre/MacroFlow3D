/**
 * @file pollock_nonperiodic_tests.cu
 * @brief SF-41 N3a: tests of the Pollock core and lattice comparator non-periodic in x (medium (b)), ctest
 *        `lagrangian_pollock_nonperiodic` (FAST tier, grids <= 32^3, well under 60 s).
 *
 *  core  host replay of the templated SF-32 core on a NonPeriodicXFaceFluxView: seeds on x1 = 0 / L1 and on an
 *        interior x face, exit_inlet / exit_outlet commit (state kept at the box face), target x1 = L1 reached
 *        on the face, seeds outside the box, no wrap in x.
 *  C4a   k = 1 faces (U = G1 = 1, V = W = 0), 32x8x8 (h = 1/8, L1 = 4): plane pass {1, L1 + 1/2} and time pass
 *        {1, 2 L1}: every particle exit_outlet after the first plane / time, counts exact, records finite and
 *        straight; V100 sizes 64x16x16, 128x16x16 (h = 1/16) under `--only C4`.
 *  C4a'  reversed uniform faces (U = -1) through the test-only zero-weight path: every particle exit_inlet.
 *  mixed mixed-sign shear U(x2) = +1 / -1 by cell bands (div-free): counts exact by construction.
 *  C4b   lobe field (lobe_field_reference.hpp, G1 = 1, a = 0.5, X0 = 0.1, M = 64 lattice, exact face-flux
 *        integrals) at 32x32x32 (h = 1/32, L1 = 1; V100 sizes 64x32x32, 128x32x32 under `--only C4`): every
 *        included seed classified as the independent host Dormand-Prince reference (plane L1/2 reached /
 *        exit_inlet; exit_inlet / exit_outlet by the time T).
 *  equiv the non-periodic comparator equals the periodic one bitwise while the trajectories stay inside the box
 *        (the shared interior arithmetic is untouched).
 *
 * `--only C4` prints the C4_CHECK lines of the pre-registration (RR5) for the V100 sizes and exits 0 iff all
 * pass. Compiled with --fmad=false like the other lagrangian_labels tests.
 */

#include "lobe_field_reference.hpp"

#include "src/core/Scalar.hpp"
#include "src/physics/particles/lagrangian_labels/FaceFluxRepack.cuh"
#include "src/physics/particles/lagrangian_labels/PollockComparator.cuh"
#include "src/runtime/CudaContext.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace ll = macroflow3d::lagrangian_labels;
namespace stt = macroflow3d::physics::particles::streamline_tracker;
namespace lr = macroflow3d::lobe_ref;

namespace {

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

struct Faces {
    std::vector<real> U, V, W;
};

/// Uniform faces: U = u0 everywhere (nx+1 planes), V = W = 0.
Faces uniform_faces(int nx, int ny, int nz, double u0) {
    Faces f;
    f.U.assign(ll::compact_mac_u_size(nx, ny, nz), u0);
    f.V.assign(ll::compact_mac_v_size(nx, ny, nz), 0.0);
    f.W.assign(ll::compact_mac_w_size(nx, ny, nz), 0.0);
    return f;
}

constexpr std::uint8_t kIn = stt::kStatusPollockExitInlet;
constexpr std::uint8_t kOut = stt::kStatusPollockExitOutlet;

bool finite3(const double* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(p[i]))
            return false;
    return true;
}

// ---------------------------------------------------------------------------------------------
// core (host instantiation of the templated SF-32 core)
// ---------------------------------------------------------------------------------------------

void case_core(TestReport& rep) {
    std::printf("\n-- core: NonPeriodicXFaceFluxView on the host\n");
    const int nx = 4, ny = 2, nz = 2;
    const double h = 0.25, L1 = 1.0;
    const stt::PollockParams prm{1000};
    auto view_of = [&](Faces& f) {
        return stt::NonPeriodicXFaceFluxView{f.U.data(), f.V.data(), f.W.data(), nx, ny, nz,
                                             h,          h,          h,          nx * h, ny * h, nz * h};
    };
    auto init = [&](const stt::NonPeriodicXFaceFluxView& v, double x1, stt::PollockState& st) {
        const real xi[3] = {x1, 0.3, 0.3};
        const int32_t w0[3] = {0, 0, 0};
        return stt::pollock_init_state(v, xi, w0, st);
    };
    Faces up = uniform_faces(nx, ny, nz, 1.0), dn = uniform_faces(nx, ny, nz, -1.0);
    const auto vu = view_of(up), vd = view_of(dn);
    stt::PollockCounters cnt{0u};
    stt::PollockState st{};

    // U = +1: seed on x1 = 0 -> cell 0 (r = 0); target L1 is reached on the face, then the outlet.
    rep.check(init(vu, 0.0, st) == stt::kStatusActive && st.cell[0] == 0 && st.r[0] == 0.0 && st.w[0] == 0,
              "core: seed on x1 = 0 belongs to cell 0 (U > 0)");
    uint8_t code = stt::pollock_advance_to_x1(vu, st, L1, prm, cnt);
    real xu[3];
    stt::pollock_unwrapped_position(vu, st, xu);
    rep.check(code == stt::kStatusActive && st.cell[0] == nx - 1 && st.r[0] == h && xu[0] == L1 && st.w[0] == 0,
              "core: target x1 = L1 is reached on the face of cell nx-1 (Active)",
              strf("code %d cell %d r %.17g x %.17g", code, st.cell[0], st.r[0], xu[0]));
    code = stt::pollock_advance_to_x1(vu, st, L1 + 0.5, prm, cnt);
    stt::pollock_unwrapped_position(vu, st, xu);
    rep.check(code == kOut && st.cell[0] == nx - 1 && st.r[0] == h && xu[0] == L1 && st.w[0] == 0 &&
                  st.t == 1.0,
              "core: crossing the upper face of cell nx-1 -> exit_outlet (9), state kept at x1 = L1, no wrap",
              strf("code %d t %.17g x %.17g w %d", code, st.t, xu[0], st.w[0]));

    // U = +1: seed on x1 = L1 -> last cell on its upper face, exits at once.
    stt::PollockCounters c2{0u};
    code = init(vu, L1, st);
    bool ok = code == stt::kStatusActive && st.cell[0] == nx - 1 && st.r[0] == h;
    code = stt::pollock_cell_exit(vu, st, c2);
    rep.check(ok && code == kOut && st.t == 0.0 && c2.cells == 1u,
              "core: seed on x1 = L1 (U > 0) -> cell nx-1, exit_outlet at t = 0");

    // U = -1: seed on x1 = 0 stays in cell 0 and exits at once through the inlet.
    c2 = stt::PollockCounters{0u};
    code = init(vd, 0.0, st);
    ok = code == stt::kStatusActive && st.cell[0] == 0 && st.r[0] == 0.0 && st.w[0] == 0;
    code = stt::pollock_cell_exit(vd, st, c2);
    stt::pollock_unwrapped_position(vd, st, xu);
    rep.check(ok && code == kIn && st.cell[0] == 0 && st.r[0] == 0.0 && xu[0] == 0.0 && st.t == 0.0,
              "core: seed on x1 = 0 (U < 0) stays in cell 0, exit_inlet at t = 0 (no flip to a cell -1)");

    // U = -1: seed at 0.75, time pass to 10: exit_inlet at t = 0.75, position kept at x1 = 0.
    c2 = stt::PollockCounters{0u};
    init(vd, 0.75, st);
    code = stt::pollock_advance_to_time(vd, st, 10.0, prm, c2);
    stt::pollock_unwrapped_position(vd, st, xu);
    rep.check(code == kIn && xu[0] == 0.0 && st.t == 0.75 && st.w[0] == 0,
              "core: backflow in the time driver -> exit_inlet (8) at t = 0.75, state at x1 = 0",
              strf("code %d t %.17g x %.17g", code, st.t, xu[0]));

    // U = -1: seed on x1 = L1 moves inward.
    c2 = stt::PollockCounters{0u};
    code = init(vd, L1, st);
    uint8_t c3 = stt::pollock_advance_to_time(vd, st, 0.25, prm, c2);
    stt::pollock_unwrapped_position(vd, st, xu);
    rep.check(code == stt::kStatusActive && c3 == stt::kStatusActive && xu[0] == 0.75,
              "core: seed on x1 = L1 (U < 0) belongs to cell nx-1 and moves inward");

    // seeds outside the box; interior x-face ownership unchanged
    rep.check(init(vu, -0.1, st) == kIn && init(vu, L1 + 0.1, st) == kOut,
              "core: seeds outside the box return the exit codes (8 below, 9 above)");
    init(vd, 0.5, st);
    const bool low = st.cell[0] == 1 && st.r[0] == h;
    init(vu, 0.5, st);
    rep.check(low && st.cell[0] == 2 && st.r[0] == 0.0,
              "core: interior x face -> cell it moves into (U <= 0: lower cell with r = D)");
}

// ---------------------------------------------------------------------------------------------
// C4a on k = 1 faces (and the reversed / mixed-sign variants)
// ---------------------------------------------------------------------------------------------

/// Returns true iff all checks of the case passed. Prints the two C4_CHECK lines (plane, time).
bool run_c4a(TestReport& rep, CudaContext& ctx, int nx, int ny, int nz, double h) {
    const std::string grid = strf("%dx%dx%d", nx, ny, nz);
    const double L1 = nx * h;
    const int fails0 = rep.failed;
    const ll::InjectionSpec inj{0.5, 16, 0.1, 0.9, 0.1, 0.9};
    const std::vector<double> planes = {1.0, L1 + 0.5}, times = {1.0, 2.0 * L1};
    const int Np = inj.M * inj.M;
    Faces f = uniform_faces(nx, ny, nz, 1.0);
    auto dev = ll::NonPeriodicXFaceFluxes::upload(ctx, f.U, f.V, f.W, nx, ny, nz, h);
    const auto r = ll::track_pollock_lattice(ctx, dev.view(), inj, planes, times, stt::PollockParams{10000000});

    bool all_out = true, straight = true, finite = true, nan_after = true;
    int reach_p = 0, reach_t = 0;
    for (int p = 0; p < Np; ++p) {
        all_out = all_out && r.status_plane[p] == kOut && r.status_time[p] == kOut;
        reach_p += r.reached_plane[p] >= 1 ? 1 : 0;
        reach_t += r.reached_time[p] >= 1 ? 1 : 0;
        const std::size_t q = static_cast<std::size_t>(p);
        straight = straight && same_bits(r.planes_x23[2 * q * 2], r.seeds[3 * q + 1]) &&
                   same_bits(r.planes_x23[2 * q * 2 + 1], r.seeds[3 * q + 2]) &&
                   r.planes_t[q * 2] == planes[0] - inj.X0 &&
                   std::abs(r.times_x[q * 2 * 3] - (inj.X0 + times[0])) <= 1e-15 &&
                   same_bits(r.times_x[q * 2 * 3 + 1], r.seeds[3 * q + 1]) &&
                   same_bits(r.times_x[q * 2 * 3 + 2], r.seeds[3 * q + 2]);
        finite = finite && finite3(&r.planes_x23[q * 2 * 2], 2) && std::isfinite(r.planes_t[q * 2]) &&
                 finite3(&r.times_x[q * 2 * 3], 3);
        nan_after = nan_after && std::isnan(r.planes_x23[(q * 2 + 1) * 2]) &&
                    std::isnan(r.planes_t[q * 2 + 1]) && std::isnan(r.times_x[(q * 2 + 1) * 3]);
    }
    rep.check(all_out, "C4a " + grid + ": every particle exit_outlet in both passes");
    rep.check(r.exit_outlet_plane_total == Np && r.exit_inlet_plane_total == 0 && r.exit_outlet_time_total == Np &&
                  r.exit_inlet_time_total == 0 && r.status_hist_plane[kOut] == Np && r.status_hist_time[kOut] == Np,
              "C4a " + grid + ": exit totals and histogram slot 9 exact (" + std::to_string(Np) + ")");
    rep.check(r.exit_outlet_plane == std::vector<long long>({0, Np}) &&
                  r.exit_outlet_time == std::vector<long long>({0, Np}) &&
                  r.exit_inlet_plane == std::vector<long long>({0, 0}) &&
                  r.exit_inlet_time == std::vector<long long>({0, 0}),
              "C4a " + grid + ": per-plane / per-time exit counts {0, Np}");
    rep.check(reach_p == Np && reach_t == Np && r.moments_planes[0].contributing == Np &&
                  r.moments_planes[1].contributing == 0 && r.moments_time[0].active == Np &&
                  r.moments_time[1].active == 0,
              "C4a " + grid + ": the first plane / time is reached by all, the second by none");
    rep.check(finite && nan_after, "C4a " + grid + ": reached records finite, records after the exit NaN");
    rep.check(straight, "C4a " + grid + ": reached records are the straight line (x2, x3 bitwise, t = x1 - X0)");
    rep.check(r.seed_nonpos == 0 && r.status_hist_plane[14] == 0 && r.status_hist_time[14] == 0 &&
                  r.status_hist_plane[12] == 0 && r.status_hist_time[12] == 0,
              "C4a " + grid + ": no non-positive seed, no substep-limit / non-finite status");
    const bool ok_all = rep.failed == fails0;
    for (int mode = 0; mode < 2; ++mode) {
        const bool plane = mode == 0;
        std::printf("C4_CHECK grid=%s case=C4a method=pollock mode=%s exit_inlet=%lld exit_outlet=%lld reached=%d "
                    "expected=inlet:0,outlet:%d,reached:%d verdict=%s\n",
                    grid.c_str(), plane ? "plane" : "time",
                    plane ? r.exit_inlet_plane_total : r.exit_inlet_time_total,
                    plane ? r.exit_outlet_plane_total : r.exit_outlet_time_total, plane ? reach_p : reach_t, Np, Np,
                    ok_all ? "PASS" : "FAIL");
    }
    return ok_all;
}

void case_reversed(TestReport& rep, CudaContext& ctx) {
    const int nx = 32, ny = 8, nz = 8;
    const double h = 0.125, L1 = nx * h;
    const ll::InjectionSpec inj{0.5, 16, 0.1, 0.9, 0.1, 0.9};
    const int Np = inj.M * inj.M;
    Faces f = uniform_faces(nx, ny, nz, -1.0);
    auto dev = ll::NonPeriodicXFaceFluxes::upload(ctx, f.U, f.V, f.W, nx, ny, nz, h);
    const std::vector<double> planes = {1.0, L1 + 0.5}, times = {1.0, 2.0 * L1};
    bool threw = false;
    try {
        ll::track_pollock_lattice(ctx, dev.view(), inj, planes, times, stt::PollockParams{10000000});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    rep.check(threw, "reversed: production path throws when no seed has a positive flux");
    ll::PollockNonPeriodicOptions opt;
    opt.allow_no_positive_weight = true;
    const auto r = ll::track_pollock_lattice(ctx, dev.view(), inj, planes, times, stt::PollockParams{10000000}, opt);
    bool all_in = true, none_reached = true;
    for (int p = 0; p < Np; ++p) {
        all_in = all_in && r.status_plane[p] == kIn && r.status_time[p] == kIn;
        none_reached = none_reached && r.reached_plane[p] == 0 && r.reached_time[p] == 0;
    }
    rep.check(all_in && none_reached, "reversed (U = -1): every particle exit_inlet, none reaches a plane / time");
    rep.check(r.exit_inlet_plane_total == Np && r.exit_inlet_time_total == Np && r.exit_outlet_plane_total == 0 &&
                  r.exit_outlet_time_total == 0 && r.exit_inlet_plane == std::vector<long long>({Np, Np}) &&
                  r.exit_inlet_time == std::vector<long long>({Np, Np}) && r.status_hist_plane[kIn] == Np,
              "reversed: exit counts exact (all inlet)");
    rep.check(r.seed_nonpos == Np && r.moments_planes[0].contributing == 0 && std::isnan(r.moments_planes[0].mean_x2),
              "reversed: all seeds non-positive (weight 0), empty moments are NaN");
    // exact clock of the exit: the particle at x1 = 0.5 reaches x1 = 0 at t = 0.5 (not recorded by the comparator;
    // the core test checks the state); the cells counter is positive
    rep.check(r.cells_plane > 0 && r.cells_time > 0, "reversed: cell steps were taken");
}

void case_mixed(TestReport& rep, CudaContext& ctx) {
    const int nx = 32, ny = 8, nz = 8;
    const double h = 0.125, L1 = nx * h;
    const ll::InjectionSpec inj{0.5, 16, 0.0, 1.0, 0.0, 1.0};
    const int Np = inj.M * inj.M;
    Faces f = uniform_faces(nx, ny, nz, 0.0);
    auto band = [&](int j) { return ((j / 2) % 2 == 0) ? 1.0 : -1.0; };
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i)
                f.U[static_cast<std::size_t>(i) + static_cast<std::size_t>(nx + 1) * (j + static_cast<std::size_t>(ny) * k)] =
                    band(j);
    auto dev = ll::NonPeriodicXFaceFluxes::upload(ctx, f.U, f.V, f.W, nx, ny, nz, h);
    const std::vector<double> planes = {1.0, L1 + 0.5}, times = {1.0, 2.0 * L1};
    const auto r = ll::track_pollock_lattice(ctx, dev.view(), inj, planes, times, stt::PollockParams{10000000});
    // seed x2 = (m2 + 1/2)/16 lies in the y-cell floor(8 x2) = m2/2 (never on a face)
    long long n_plus = 0, n_minus = 0;
    bool as_expected = true;
    for (int p = 0; p < Np; ++p) {
        const int m2 = p / inj.M;
        const bool plus = band(m2 / 2) > 0.0;
        (plus ? n_plus : n_minus) += 1;
        as_expected = as_expected && r.status_plane[p] == (plus ? kOut : kIn) && r.status_time[p] == (plus ? kOut : kIn) &&
                      r.reached_plane[p] == (plus ? 1 : 0) && r.reached_time[p] == (plus ? 1 : 0);
    }
    rep.check(n_plus == 128 && n_minus == 128 && as_expected,
              "mixed-sign shear: every particle exits through the face its band points to, reach counts as expected");
    rep.check(r.exit_inlet_plane == std::vector<long long>({n_minus, n_minus}) &&
                  r.exit_outlet_plane == std::vector<long long>({0, n_plus}) &&
                  r.exit_inlet_time == std::vector<long long>({n_minus, n_minus}) &&
                  r.exit_outlet_time == std::vector<long long>({0, n_plus}) && r.exit_inlet_plane_total == n_minus &&
                  r.exit_outlet_plane_total == n_plus && r.exit_inlet_time_total == n_minus &&
                  r.exit_outlet_time_total == n_plus,
              "mixed-sign shear: per-plane / per-time exit counts exact");
    rep.check(r.seed_nonpos == n_minus && r.moments_planes[0].contributing == n_plus &&
                  r.moments_time[0].active == n_plus,
              "mixed-sign shear: negative-flux seeds get weight 0 but are traced; moments over the reached particles");
    double w = 0.0;
    for (double x : r.weights)
        w += x;
    rep.check(std::abs(w - 1.0) <= 1e-14, "mixed-sign shear: weights sum to 1 over the positive-flux seeds");
}

// ---------------------------------------------------------------------------------------------
// C4b lobe field
// ---------------------------------------------------------------------------------------------

bool run_c4b(TestReport& rep, CudaContext& ctx, int nx, int ny, int nz, double h) {
    const std::string grid = strf("%dx%dx%d", nx, ny, nz);
    const int fails0 = rep.failed;
    const double L1 = nx * h, X0 = 0.1, Xp = 0.5 * L1;
    const int M = 64;
    const lr::LobeField field; // G1 = 1, a = 0.5
    const lr::CompactMacHost c = lr::lobe_face_fluxes(field, nx, ny, nz, h);
    const lr::LobeLattice lat = lr::make_lobe_lattice(field, L1, X0, M, Xp);
    const double T = lr::lobe_time_T(lat);
    std::printf("  C4b %s: L1 %g, included %d of %d x2-values (inlet %d, outlet %d, plane-reaching %d), T %g, "
                "stream-function drift %.2e, max |div cell| %.2e\n",
                grid.c_str(), L1, lat.n_included / M, M, lat.n_inlet / M, lat.n_outlet / M, lat.n_plane / M, T,
                lat.max_psi_drift, c.max_cell_divergence);
    rep.check(lat.max_psi_drift < 1e-9 && c.max_cell_divergence < 1e-10 && lat.n_inlet > 0 && lat.n_outlet > 0,
              "C4b " + grid + ": reference sane (psi drift < 1e-9, face fluxes divergence-free, both exit kinds present)");
    auto dev = ll::NonPeriodicXFaceFluxes::upload(ctx, c.U, c.V, c.W, nx, ny, nz, h);
    const ll::InjectionSpec inj{X0, M, 0.0, 1.0, 0.0, 1.0};
    const auto r = ll::track_pollock_lattice(ctx, dev.view(), inj, {Xp}, {T}, stt::PollockParams{10000000});
    int mism_p = 0, mism_t = 0, in_p = 0, out_p = 0, reach_p = 0, in_t = 0, out_t = 0, reach_t = 0, odd = 0;
    for (std::size_t p = 0; p < lat.seeds.size(); ++p) {
        const std::uint8_t sp = r.status_plane[p], st = r.status_time[p];
        if (sp != 0 && sp != kIn && sp != kOut)
            ++odd;
        if (st != 0 && st != kIn && st != kOut)
            ++odd;
        const lr::LobeSeed& s = lat.seeds[p];
        if (!s.included)
            continue;
        in_p += sp == kIn;
        out_p += sp == kOut;
        reach_p += r.reached_plane[p] >= 1;
        in_t += st == kIn;
        out_t += st == kOut;
        reach_t += r.reached_time[p] >= 1;
        const bool ref_in = s.ref.exit == lr::ExitKind::inlet;
        // plane pass: Active (plane reached) iff the reference reaches L1/2 before leaving; else exit_inlet
        const bool pl_ok = s.ref.reached_plane ? (sp == 0 && r.reached_plane[p] == 1) : (sp == kIn);
        // time pass: every included seed has left by T (T = 1.25 x the largest outlet exit time)
        const bool tm_ok = ref_in ? (st == kIn) : (st == kOut);
        mism_p += pl_ok ? 0 : 1;
        mism_t += tm_ok ? 0 : 1;
    }
    const int N = lat.n_included;
    rep.check(odd == 0, "C4b " + grid + ": no status other than active / exit_inlet / exit_outlet over the whole lattice");
    rep.check(mism_p == 0 && in_p == lat.n_inlet && reach_p == lat.n_plane && out_p == 0,
              "C4b " + grid + ": plane pass L1/2 classifies every included seed as the reference",
              strf("included %d mismatches %d", N, mism_p));
    rep.check(mism_t == 0 && in_t == lat.n_inlet && out_t == lat.n_outlet && reach_t == 0,
              "C4b " + grid + ": time pass T classifies every included seed as the reference",
              strf("included %d mismatches %d T %g", N, mism_t, T));
    const bool ok_all = rep.failed == fails0;
    std::printf("C4_CHECK grid=%s case=C4b method=pollock mode=plane exit_inlet=%d exit_outlet=%d reached=%d "
                "expected=inlet:%d,outlet:0,reached:%d mismatches=%d verdict=%s\n",
                grid.c_str(), in_p, out_p, reach_p, lat.n_inlet, lat.n_plane, mism_p, ok_all ? "PASS" : "FAIL");
    std::printf("C4_CHECK grid=%s case=C4b method=pollock mode=time exit_inlet=%d exit_outlet=%d reached=%d "
                "expected=inlet:%d,outlet:%d,reached:0 mismatches=%d verdict=%s\n",
                grid.c_str(), in_t, out_t, reach_t, lat.n_inlet, lat.n_outlet, mism_t, ok_all ? "PASS" : "FAIL");
    return ok_all;
}

// ---------------------------------------------------------------------------------------------
// periodic vs non-periodic while inside the box
// ---------------------------------------------------------------------------------------------

void case_equiv(TestReport& rep, CudaContext& ctx) {
    const int nx = 16, ny = 8, nz = 8;
    const double h = 0.125;
    const ll::InjectionSpec inj{0.5, 8, 0.1, 0.9, 0.1, 0.9};
    // smooth non-uniform divergence-free-ish faces: U > 0 with a y-dependence, V small; both views from the same
    // CompactMAC arrays (periodic duplicates at plane 0 for the periodic view; U[nx] unused by the interior).
    Faces f = uniform_faces(nx, ny, nz, 0.0);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i)
                f.U[static_cast<std::size_t>(i) + static_cast<std::size_t>(nx + 1) * (j + static_cast<std::size_t>(ny) * k)] =
                    (i == nx ? 1.0 : 1.0 + 0.3 * std::sin(0.7 * i) + 0.2 * std::cos(1.1 * j + 0.3 * k));
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i < nx; ++i)
                f.V[static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * j + static_cast<std::size_t>(nx) * (ny + 1) * k] =
                    (j == ny) ? 0.05 * std::sin(0.4 * i + 0.9 * k) : 0.05 * std::sin(0.4 * i + 0.9 * k) + 0.02 * std::cos(2.1 * j);
    // make the periodic duplicate plane of V equal to plane 0
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i < nx; ++i)
            f.V[static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * ny + static_cast<std::size_t>(nx) * (ny + 1) * k] =
                f.V[static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * (ny + 1) * k];
    auto per = ll::PeriodicFaceFluxes::upload_and_repack(ctx, f.U, f.V, f.W, nx, ny, nz, h);
    auto np = ll::NonPeriodicXFaceFluxes::upload(ctx, f.U, f.V, f.W, nx, ny, nz, h);
    const std::vector<double> planes = {0.8, 1.2}, times = {0.4, 0.9};
    const stt::PollockParams prm{10000000};
    const auto a = ll::track_pollock_lattice(ctx, per.view(), inj, planes, times, prm);
    const auto b = ll::track_pollock_lattice(ctx, np.view(), inj, planes, times, prm);
    const bool inside = std::all_of(b.status_plane.begin(), b.status_plane.end(), [](std::uint8_t s) { return s == 0; }) &&
                        std::all_of(b.status_time.begin(), b.status_time.end(), [](std::uint8_t s) { return s == 0; });
    rep.check(inside, "equiv: the test trajectories stay inside the box (all statuses active)");
    rep.check(same_bits_vec(a.planes_x23, b.planes_x23) && same_bits_vec(a.planes_t, b.planes_t) &&
                  same_bits_vec(a.times_x, b.times_x) && same_bits_vec(a.weight_raw, b.weight_raw) &&
                  same_bits_vec(a.cells, b.cells) && a.status_plane == b.status_plane && a.status_time == b.status_time,
              "equiv: non-periodic comparator == periodic comparator bitwise while inside the box");
    rep.check(a.exit_inlet_plane.empty() && a.exit_outlet_time.empty() && a.exit_inlet_plane_total == 0 &&
                  b.exit_inlet_plane.size() == 2 && b.exit_outlet_time.size() == 2,
              "equiv: the periodic result leaves the new exit fields empty");
}

} // namespace

int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();
    bool only_c4 = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--only" && i + 1 < argc && std::string(argv[i + 1]) == "C4") {
            only_c4 = true;
            ++i;
        } else {
            std::fprintf(stderr, "usage: %s [--only C4]\n", argv[0]);
            return 2;
        }
    }
    TestReport rep;
    CudaContext ctx(0);
    if (only_c4) {
        std::printf("pollock_nonperiodic_tests --only C4 (RR5, V100 sizes)\n");
        bool ok = true;
        ok = run_c4a(rep, ctx, 64, 16, 16, 1.0 / 16.0) && ok;
        ok = run_c4a(rep, ctx, 128, 16, 16, 1.0 / 16.0) && ok;
        ok = run_c4b(rep, ctx, 64, 32, 32, 1.0 / 32.0) && ok;
        ok = run_c4b(rep, ctx, 128, 32, 32, 1.0 / 32.0) && ok;
        std::printf("C4_VERDICT method=pollock %s (%d checks, %d failed)\n", ok && rep.overall_pass ? "PASS" : "FAIL",
                    rep.checks, rep.failed);
        return ok && rep.overall_pass ? 0 : 1;
    }
    std::printf("lagrangian_pollock_nonperiodic\n");
    case_core(rep);
    run_c4a(rep, ctx, 32, 8, 8, 0.125);
    case_reversed(rep, ctx);
    case_mixed(rep, ctx);
    run_c4b(rep, ctx, 32, 32, 32, 1.0 / 32.0);
    case_equiv(rep, ctx);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n%d checks, %d failed, %.1f s\n", rep.checks, rep.failed, s);
    std::printf("%s\n", rep.overall_pass ? "ALL PASS" : "SOME FAILED");
    return rep.overall_pass ? 0 : 1;
}
