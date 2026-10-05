/**
 * @file reference_rk_tracker_tests.cu
 * @brief SF-31 N4: fast contract tests of the adaptive Runge-Kutta reference
 *        tracker (`src/physics/particles/streamline_tracker/ReferenceRkTracker.cuh`)
 *        and the cross-check between the two SF-31 integrators.
 *
 * Standalone ctest-friendly runner (style of
 * `tests/streamline_tracker/pseudo_symplectic_tracker_tests.cu`: printed
 * `[PASS]/[FAIL]` checks, `main` returns 0/1). Fast tier.
 *
 * Every case, particle set, grid, ladder, tolerance and threshold is
 * PRE-REGISTERED by the SF-31 orchestrator (node N4 specification, readings
 * T4a/T4b/T5/T6 of the orchestration record) and implemented verbatim; none
 * may be adjusted to make a failing line pass.
 *
 * Host-core cases (analytic labels, LabelVelocity<AnalyticLabels>, no spline,
 * no GPU kernels):
 *   rk_tableau_order (T4a), rk_tolerance_ladder (T4a), rk_x1_exact,
 *   rk_position_error, rk_landing, rk_failure_paths, core_ps_vs_rk_cross_check.
 * GPU-engine cases (SF-28 splined labels):
 *   gpu_rk_ladder (T4b), gpu_rk_multi_call_landing, gpu_rk_positive_control, gpu_rk_host_core_agreement,
 *   gpu_rk_no_allocation (T5), gpu_rk_determinism (T6),
 *   gpu_ps_vs_rk_cross_check, rk_validation_errors, timing_record.
 *
 * Note on splined labels: c = grad psi1 x grad psi2 is only C^1 there, so the
 * label drift of the GPU engine does not follow the tolerance; its fitted
 * slope is printed as INFO and not gated (ReferenceRkTracker.cuh section 6).
 */

#include "analytic_pairs.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "src/physics/particles/streamline_tracker/PseudoSymplecticTracker.cuh"
#include "src/physics/particles/streamline_tracker/ReferenceRkTracker.cuh"
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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::interpolation;
using namespace macroflow3d::physics::particles;
using namespace macroflow3d::physics::particles::streamline_tracker;
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
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(),
                    detail.empty() ? "" : "  ", detail.c_str());
        overall_pass = overall_pass && cond;
    }
};

std::string strf(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kHorizon = 1.7; // T of every host case unless stated
const uint64_t kSeed = 12345ULL;

// ============================================================================
// Fixtures and host helpers
// ============================================================================

struct P3 {
    double v[3];
};

const P3 kFixed[4] = {{{0.1, 0.2, 0.3}}, {{0.35, 0.6, 0.85}}, {{0.7, 0.15, 0.5}}, {{0.9, 0.8, 0.05}}};

/// "The 16 host points": P1..P4 followed by inject_uniform01(seed = 20261005, index = 0..11, axis).
std::vector<P3> host_points16() {
    std::vector<P3> p(kFixed, kFixed + 4);
    for (uint64_t i = 0; i < 12; ++i) {
        P3 q;
        for (int a = 0; a < 3; ++a)
            q.v[a] = inject_uniform01(20261005ULL, i, a);
        p.push_back(q);
    }
    return p;
}

void unwrapped(const real xi[3], const int32_t w[3], const real L[3], double xu[3]) {
    for (int d = 0; d < 3; ++d)
        xu[d] = std::fma(static_cast<real>(w[d]), L[d], xi[d]);
}

double maxnorm_diff(const double a[3], const double b[3]) {
    return std::max(std::fabs(a[0] - b[0]), std::max(std::fabs(a[1] - b[1]), std::fabs(a[2] - b[2])));
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

double order2(double coarse, double fine) {
    return std::log2(coarse / fine);
}

/// Least-squares slope of log10(E) against log10(tol).
double ls_slope(const double* tol, const double* e, int n) {
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < n; ++i) {
        const double x = std::log10(tol[i]);
        const double y = std::log10(e[i]);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

/// Label drift max(|psi1(end) - psi1(start)|, |psi2(end) - psi2(start)|) at the
/// unwrapped states (xi0, w0) and (xi1, w1), same evaluator.
template <class E>
double label_drift(const E& lab, const real xi0[3], const int32_t w0[3], const real xi1[3], const int32_t w1[3]) {
    LabelSample a, b;
    lab(xi0, w0, a);
    lab(xi1, w1, b);
    return std::max(std::fabs(b.psi1 - a.psi1), std::fabs(b.psi2 - a.psi2));
}

ReferenceRkParams make_rk_prm(double tol, double dt_max, double min_step = 1e-14, int max_steps = 1000000) {
    const ReferenceRkParams p{tol, dt_max, min_step, max_steps};
    return p;
}

/// Host RK start state: (x0, w = 0, t = 0, h = h0).
void rk_init(RkState& st, const double x0[3], double h0) {
    for (int d = 0; d < 3; ++d) {
        st.xi[d] = x0[d];
        st.w[d] = 0;
    }
    st.t = 0.0;
    st.h = h0;
}

/// Label drift of a host RK state from its start x0 (w = 0).
template <class E> double rk_state_drift(const E& lab, const double x0[3], const RkState& st) {
    const int32_t w0[3] = {0, 0, 0};
    return label_drift(lab, x0, w0, st.xi, st.w);
}

PseudoSymplecticParams make_ps_prm(double tol, int max_iter = 8) {
    PseudoSymplecticParams prm{};
    prm.tol_psi = tol;
    prm.max_newton_iter = max_iter;
    prm.trust_factor = 1.0;
    prm.min_cross_norm = 0.0;
    prm.min_cross_sin2 = kDefaultMinCrossSin2;
    return prm;
}

/// Velocity functor returning NaN everywhere (failure-path case).
struct NaNVelocity {
    real value;
    __host__ __device__ inline void operator()(const real xi[3], real v[3]) const {
        (void)xi;
        v[0] = value;
        v[1] = value;
        v[2] = value;
    }
};

bool rk_state_equal(const RkState& a, const RkState& b) {
    return std::memcmp(&a, &b, sizeof(RkState)) == 0;
}

const double kTolLadder[7] = {1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10};

// Shared results of core_ps_vs_rk_cross_check (used by gpu_ps_vs_rk_cross_check).
double g_D_ds64 = -1.0;
double g_D_ds128 = -1.0;

// ============================================================================
// Case 1: rk_tableau_order (T4a)
// ============================================================================

void case_rk_tableau_order(TestReport& rep) {
    std::printf("\n=== rk_tableau_order (T4a) ===\n");
    const std::vector<P3> pts = host_points16();
    const AnalyticLabels lab = make_analytic_labels(kPairG);
    const LabelVelocity<AnalyticLabels> vel{lab};
    const int ms[4] = {17, 34, 68, 136};
    double drift[4];
    bool ok = true;
    for (int k = 0; k < 4; ++k) {
        const double dt = kHorizon / ms[k];
        const ReferenceRkParams prm = make_rk_prm(1e300, dt, 1e-14);
        drift[k] = 0.0;
        uint32_t acc_max = 0, rej_tot = 0;
        for (const P3& p : pts) {
            RkState st;
            rk_init(st, p.v, dt);
            RkCounters cnt{0u, 0u};
            const uint8_t code = rk_advance_to_time(vel, st, lab.L, kHorizon, prm, cnt);
            ok = ok && code == kStatusActive;
            drift[k] = std::max(drift[k], rk_state_drift(lab, p.v, st));
            acc_max = std::max(acc_max, cnt.accepted);
            rej_tot += cnt.rejected;
        }
        std::printf("  G m=%3d h=T/m=%.6e  max label drift = %.6e  (max accepted per point %u, rejected %u)\n",
                    ms[k], dt, drift[k], acc_max, rej_tot);
    }
    rep.check(ok, "rk_tableau_order/all_runs_active");
    for (int k = 0; k < 3; ++k) {
        const double o = order2(drift[k], drift[k + 1]);
        rep.check(o >= 4.5, strf("rk_tableau_order/order_m%d->m%d", ms[k], ms[k + 1]),
                  strf("%.4f (gate >= 4.5)", o));
    }
}

// ============================================================================
// Case 2: rk_tolerance_ladder (T4a) + rk_x1_exact + rk_position_error
// ============================================================================

void case_rk_tolerance_ladder(TestReport& rep) {
    std::printf("\n=== rk_tolerance_ladder (T4a) + rk_x1_exact + rk_position_error ===\n");
    const std::vector<P3> pts = host_points16();
    const int pairs[3] = {kPairG, kPairH, kPairB};
    for (int pair : pairs) {
        const std::string pn = pair_name(pair);
        const AnalyticLabels lab = make_analytic_labels(pair);
        const LabelVelocity<AnalyticLabels> vel{lab};
        double maxd[7];
        bool all_active = true;
        double x1_err = 0.0;
        double pos_err_1e10 = 0.0;
        for (int k = 0; k < 7; ++k) {
            const double tol = kTolLadder[k];
            const ReferenceRkParams prm = make_rk_prm(tol, 0.5);
            maxd[k] = 0.0;
            double dP[4] = {0, 0, 0, 0};
            uint32_t aP[4] = {0, 0, 0, 0}, rP[4] = {0, 0, 0, 0};
            for (size_t ip = 0; ip < pts.size(); ++ip) {
                RkState st;
                rk_init(st, pts[ip].v, 0.05);
                RkCounters cnt{0u, 0u};
                const uint8_t code = rk_advance_to_time(vel, st, lab.L, kHorizon, prm, cnt);
                if (code != kStatusActive) {
                    all_active = false;
                    std::printf("  pair %s tol=%.0e point %zu returned status %u\n", pn.c_str(), tol, ip,
                                static_cast<unsigned>(code));
                }
                const double d = rk_state_drift(lab, pts[ip].v, st);
                maxd[k] = std::max(maxd[k], d);
                if (ip < 4) {
                    dP[ip] = d;
                    aP[ip] = cnt.accepted;
                    rP[ip] = cnt.rejected;
                }
                double xu[3];
                unwrapped(st.xi, st.w, lab.L, xu);
                if (pair == kPairH || pair == kPairB) {
                    x1_err = std::max(x1_err, std::fabs(xu[0] - pts[ip].v[0] - kHorizon));
                    if (k == 6) {
                        double xe[3];
                        exact_position(pair, pts[ip].v, kHorizon, xe);
                        pos_err_1e10 = std::max(pos_err_1e10, maxnorm_diff(xu, xe));
                    }
                }
            }
            std::printf("  pair %s tol=%.0e  max drift = %.6e  P1..P4 drift: %.7g %.7g %.7g %.7g  "
                        "acc/rej P1..P4: %u/%u %u/%u %u/%u %u/%u\n",
                        pn.c_str(), tol, maxd[k], dP[0], dP[1], dP[2], dP[3], aP[0], rP[0], aP[1], rP[1], aP[2],
                        rP[2], aP[3], rP[3]);
        }
        bool decreasing = true;
        for (int k = 0; k < 6; ++k)
            decreasing = decreasing && maxd[k + 1] < maxd[k];
        const double slope = ls_slope(kTolLadder, maxd, 7);
        rep.check(decreasing, "rk_tolerance_ladder/drift_strictly_decreasing_" + pn);
        rep.check(slope >= 0.8 && slope <= 1.5, "rk_tolerance_ladder/ls_slope_" + pn,
                  strf("%.4f (gate [0.8, 1.5])", slope));
        rep.check(all_active, "rk_tolerance_ladder/all_runs_active_" + pn);
        if (pair == kPairH || pair == kPairB) {
            rep.check(x1_err <= 1e-13, "rk_x1_exact/" + pn,
                      strf("max |x1_u(T) - x1_0 - T| = %.3e over 16 points x 7 tol (gate 1e-13)", x1_err));
            rep.check(pos_err_1e10 <= 1e-8, "rk_position_error/" + pn,
                      strf("max_p |x_u(T) - x_exact(T)|_max at tol 1e-10 = %.3e (gate 1e-8)", pos_err_1e10));
        }
    }
}

// ============================================================================
// Case 3: rk_landing
// ============================================================================

void case_rk_landing(TestReport& rep) {
    std::printf("\n=== rk_landing ===\n");
    const AnalyticLabels lab = make_analytic_labels(kPairG);
    const LabelVelocity<AnalyticLabels> vel{lab};
    const ReferenceRkParams prm = make_rk_prm(1e-8, 0.5);
    bool i_land = true, i_idem = true, i_active = true, ii_land = true, ii_active = true;
    for (int p = 0; p < 4; ++p) {
        RkState st;
        rk_init(st, kFixed[p].v, 0.05);
        RkCounters cnt{0u, 0u};
        i_active = i_active && rk_advance_to_time(vel, st, lab.L, kHorizon, prm, cnt) == kStatusActive;
        i_land = i_land && same_bits(st.t, kHorizon);
        const RkState saved = st;
        const RkCounters c0 = cnt;
        i_active = i_active && rk_advance_to_time(vel, st, lab.L, kHorizon, prm, cnt) == kStatusActive;
        i_idem = i_idem && rk_state_equal(saved, st) && cnt.accepted == c0.accepted && cnt.rejected == c0.rejected;
        std::printf("  P%d (i): t = %.17g, accepted %u, rejected %u\n", p + 1, st.t, cnt.accepted, cnt.rejected);

        RkState s2;
        rk_init(s2, kFixed[p].v, 0.05);
        RkCounters c2{0u, 0u};
        double t_target = 0.0;
        for (int k = 1; k <= 10; ++k) {
            t_target += 0.17;
            ii_active = ii_active && rk_advance_to_time(vel, s2, lab.L, t_target, prm, c2) == kStatusActive;
            if (!same_bits(s2.t, t_target)) {
                ii_land = false;
                std::printf("  P%d (ii) call %d: t = %.17g, target %.17g\n", p + 1, k, s2.t, t_target);
            }
        }
    }
    rep.check(i_active, "rk_landing/i_all_active");
    rep.check(i_land, "rk_landing/i_clock_equals_T_bitwise");
    rep.check(i_idem, "rk_landing/i_second_call_same_target_no_change_no_step");
    rep.check(ii_active, "rk_landing/ii_all_active");
    rep.check(ii_land, "rk_landing/ii_clock_equals_accumulated_target_bitwise_after_each_of_10_calls");
}

// ============================================================================
// Case 4: rk_failure_paths
// ============================================================================

void case_rk_failure_paths(TestReport& rep) {
    std::printf("\n=== rk_failure_paths ===\n");
    const AnalyticLabels lab = make_analytic_labels(kPairG);
    const LabelVelocity<AnalyticLabels> vel{lab};
    {
        const ReferenceRkParams prm = make_rk_prm(1e-8, 0.5, 1e-14, 3);
        RkState st;
        rk_init(st, kFixed[0].v, 0.05);
        RkCounters cnt{0u, 0u};
        const uint8_t code = rk_advance_to_time(vel, st, lab.L, 100.0, prm, cnt);
        std::printf("  (i) status %u, t = %.6e, accepted %u, rejected %u\n", static_cast<unsigned>(code), st.t,
                    cnt.accepted, cnt.rejected);
        rep.check(code == kStatusSubstepLimit, "rk_failure_paths/i_status_substep_limit",
                  strf("status %u", static_cast<unsigned>(code)));
        rep.check(st.t > 0.0 && st.t < 100.0, "rk_failure_paths/i_clock_in_(0,100)", strf("t = %.6e", st.t));
        rep.check(cnt.accepted + cnt.rejected == 3u, "rk_failure_paths/i_accepted+rejected==3",
                  strf("%u + %u", cnt.accepted, cnt.rejected));
    }
    {
        const ReferenceRkParams prm = make_rk_prm(1e-8, 0.5, 1.0);
        RkState st;
        rk_init(st, kFixed[0].v, 0.05);
        RkCounters cnt{0u, 0u};
        const uint8_t code = rk_advance_to_time(vel, st, lab.L, kHorizon, prm, cnt);
        std::printf("  (ii) status %u, t = %.6e\n", static_cast<unsigned>(code), st.t);
        rep.check(code == kStatusStepUnderflow, "rk_failure_paths/ii_status_step_underflow",
                  strf("status %u", static_cast<unsigned>(code)));
    }
    {
        const NaNVelocity nv{kNaN};
        const ReferenceRkParams prm = make_rk_prm(1e-8, 0.5);
        RkState st;
        rk_init(st, kFixed[0].v, 0.05);
        const RkState saved = st;
        RkCounters cnt{0u, 0u};
        const uint8_t code = rk_advance_to_time(nv, st, lab.L, kHorizon, prm, cnt);
        const bool unchanged = same_bits(st.xi[0], saved.xi[0]) && same_bits(st.xi[1], saved.xi[1]) &&
                               same_bits(st.xi[2], saved.xi[2]) && st.w[0] == saved.w[0] && st.w[1] == saved.w[1] &&
                               st.w[2] == saved.w[2];
        std::printf("  (iii) status %u, position (%.17g, %.17g, %.17g)\n", static_cast<unsigned>(code), st.xi[0],
                    st.xi[1], st.xi[2]);
        rep.check(code == kStatusNonFinite, "rk_failure_paths/iii_status_nonfinite",
                  strf("status %u", static_cast<unsigned>(code)));
        rep.check(unchanged, "rk_failure_paths/iii_position_unchanged");
    }
}

// ============================================================================
// Case 5: core_ps_vs_rk_cross_check
// ============================================================================

void case_core_ps_vs_rk_cross_check(TestReport& rep) {
    std::printf("\n=== core_ps_vs_rk_cross_check ===\n");
    const std::vector<P3> pts = host_points16();
    const AnalyticLabels lab = make_analytic_labels(kPairG);
    const LabelVelocity<AnalyticLabels> vel{lab};
    const PseudoSymplecticParams pprm = make_ps_prm(1e-13, 8);
    const ReferenceRkParams rprm = make_rk_prm(1e-12, 0.25);
    const double dss[4] = {1.0 / 16, 1.0 / 32, 1.0 / 64, 1.0 / 128};
    const char* dsn[4] = {"1/16", "1/32", "1/64", "1/128"};
    double D[4];
    bool ok = true;
    for (int k = 0; k < 4; ++k) {
        const double ds = dss[k];
        const int n = static_cast<int>(1.5 / ds);
        D[k] = 0.0;
        double tmin = 1e300, tmax = -1e300;
        for (const P3& p : pts) {
            PanelState ps;
            for (int d = 0; d < 3; ++d) {
                ps.xi[d] = p.v[d];
                ps.w[d] = 0;
            }
            ps.t = 0.0;
            if (evaluate_label_state(lab, ps.xi, ps.w, pprm, ps.at) != kStatusActive) {
                ok = false;
                continue;
            }
            const real p1 = ps.at.psi1, p2 = ps.at.psi2;
            ProjectionCounters pc{0u, 0u};
            for (int i = 0; i < n; ++i) {
                if (advance_panel(lab, ps, p1, p2, ds, pprm, pc) != kStatusActive) {
                    ok = false;
                    break;
                }
            }
            RkState rs;
            rk_init(rs, p.v, 0.025);
            RkCounters rc{0u, 0u};
            ok = ok && rk_advance_to_time(vel, rs, lab.L, ps.t, rprm, rc) == kStatusActive;
            double xp[3], xr[3];
            unwrapped(ps.xi, ps.w, lab.L, xp);
            unwrapped(rs.xi, rs.w, lab.L, xr);
            D[k] = std::max(D[k], maxnorm_diff(xp, xr));
            tmin = std::min(tmin, ps.t);
            tmax = std::max(tmax, ps.t);
        }
        std::printf("  ds=%-5s n=%3d  D = max_p |x_u,PS - x_u,RK|_max = %.6e  (t_p in [%.6f, %.6f])\n", dsn[k], n,
                    D[k], tmin, tmax);
    }
    rep.check(ok, "core_ps_vs_rk_cross_check/all_active");
    for (int k = 0; k < 3; ++k) {
        const double o = order2(D[k], D[k + 1]);
        rep.check(o >= 1.8 && o <= 2.2, std::string("core_ps_vs_rk_cross_check/order_") + dsn[k] + "->" + dsn[k + 1],
                  strf("%.4f (gate [1.8, 2.2])", o));
    }
    g_D_ds64 = D[2];
    g_D_ds128 = D[3];
}

// ============================================================================
// GPU infrastructure
// ============================================================================

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

/// Splined labels of one pair: SF-28 GPU prefilter into two workspaces, device
/// pair and host pair on the downloaded coefficients. Not movable (host views
/// point into hc1/hc2).
struct SplineSet {
    int pair = 0;
    Grid3D g;
    PeriodicTricubicBSplineWorkspace ws1, ws2;
    std::vector<real> hc1, hc2;
    SplineLabelPair dev{}, host{};

    SplineSet() = default;
    SplineSet(const SplineSet&) = delete;
    SplineSet& operator=(const SplineSet&) = delete;
};

std::unique_ptr<SplineSet> build_spline(const CudaContext& ctx, int pair, const Grid3D& g) {
    std::unique_ptr<SplineSet> S(new SplineSet());
    S->pair = pair;
    S->g = g;
    std::vector<real> s1, s2;
    sample_fluctuations(pair, g, s1, s2);
    DeviceBuffer<real> d1(s1.size()), d2(s2.size());
    h2d(d1.data(), s1);
    h2d(d2.data(), s2);
    prefilter_periodic_tricubic_bspline(ctx, g, DeviceSpan<const real>(d1.data(), d1.size()), S->ws1);
    prefilter_periodic_tricubic_bspline(ctx, g, DeviceSpan<const real>(d2.data(), d2.size()), S->ws2);
    ctx.synchronize();
    real gb1[3], gb2[3];
    pair_gbar(pair, gb1, gb2);
    S->dev = make_spline_label_pair(S->ws1.view(), S->ws2.view(), gb1, gb2);
    S->hc1 = d2h(S->ws1.coefficients.data(), g.num_cells());
    S->hc2 = d2h(S->ws2.coefficients.data(), g.num_cells());
    S->host = make_spline_label_pair(make_host_view(g, S->hc1.data()), make_host_view(g, S->hc2.data()),
                                     gb1, gb2);
    std::printf("  [spline] pair %s on %dx%dx%d\n", pair_name(pair), g.nx, g.ny, g.nz);
    return S;
}

struct DevParticles {
    int n;
    DeviceBuffer<real> x, y, z;
    DeviceBuffer<uint8_t> st;
    DeviceBuffer<int32_t> wx, wy, wz;
    ParticlesSoA<real> soa;

    explicit DevParticles(int n_) : n(n_), x(n_), y(n_), z(n_), st(n_), wx(n_), wy(n_), wz(n_) {
        soa.x = x.data();
        soa.y = y.data();
        soa.z = z.data();
        soa.n = n;
        soa.status = st.data();
        soa.wrapX = wx.data();
        soa.wrapY = wy.data();
        soa.wrapZ = wz.data();
    }
};

/// Host snapshot of particle arrays (+ engine arrays when the engine provides them).
struct Snap {
    std::vector<real> x, y, z, clock, h;
    std::vector<int32_t> wx, wy, wz;
    std::vector<uint8_t> st;
    std::vector<uint32_t> acc, rej;
};

void snap_particles(DevParticles& P, Snap& s) {
    const size_t n = static_cast<size_t>(P.n);
    s.x = d2h(P.x.data(), n);
    s.y = d2h(P.y.data(), n);
    s.z = d2h(P.z.data(), n);
    s.wx = d2h(P.wx.data(), n);
    s.wy = d2h(P.wy.data(), n);
    s.wz = d2h(P.wz.data(), n);
    s.st = d2h(P.st.data(), n);
}

Snap snap_rk(ReferenceRkTracker& e, DevParticles& P) {
    e.synchronize();
    const size_t n = static_cast<size_t>(P.n);
    Snap s;
    snap_particles(P, s);
    s.clock = d2h(e.clocks(), n);
    s.h = d2h(e.step_proposals(), n);
    s.acc = d2h(e.accepted_counts(), n);
    s.rej = d2h(e.rejected_counts(), n);
    return s;
}

Snap snap_ps(PseudoSymplecticTracker& e, DevParticles& P) {
    e.synchronize();
    const size_t n = static_cast<size_t>(P.n);
    Snap s;
    snap_particles(P, s);
    s.clock = d2h(e.clocks(), n);
    return s;
}

void snap_pos(const Snap& s, size_t i, real xi[3], int32_t w[3]) {
    xi[0] = s.x[i];
    xi[1] = s.y[i];
    xi[2] = s.z[i];
    w[0] = s.wx[i];
    w[1] = s.wy[i];
    w[2] = s.wz[i];
}

void snap_unwrapped(const Snap& s, size_t i, const real L[3], double xu[3]) {
    real xi[3];
    int32_t w[3];
    snap_pos(s, i, xi, w);
    unwrapped(xi, w, L, xu);
}

/// Max label drift between two snapshots (host spline evaluator, unwrapped states).
double snap_drift(const Snap& s0, const Snap& s1, const SplineLabelPair& lab) {
    double m = 0.0;
    for (size_t i = 0; i < s0.x.size(); ++i) {
        real a[3], b[3];
        int32_t wa[3], wb[3];
        snap_pos(s0, i, a, wa);
        snap_pos(s1, i, b, wb);
        m = std::max(m, label_drift(lab, a, wa, b, wb));
    }
    return m;
}

bool all_status(const Snap& s, uint8_t code) {
    for (uint8_t v : s.st)
        if (v != code)
            return false;
    return true;
}

ReferenceRkConfig make_rk_cfg(double tol, double dt_max) {
    ReferenceRkConfig c{};
    c.tol = tol;
    c.dt_max = dt_max;
    return c;
}

/// Contract order: configure, bind_labels, bind_particles, inject_box, ensure_tracking, prepare.
void setup_rk(ReferenceRkTracker& e, const ReferenceRkConfig& cfg, const SplineSet& S, DevParticles& P) {
    e.configure(cfg);
    e.bind_labels(S.dev);
    e.bind_particles(P.soa);
    e.inject_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0, P.n);
    e.ensure_tracking();
    e.prepare();
}

void setup_ps(PseudoSymplecticTracker& e, const PseudoSymplecticConfig& cfg, const SplineSet& S, DevParticles& P) {
    e.configure(cfg);
    e.bind_labels(S.dev);
    e.bind_particles(P.soa);
    e.inject_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0, P.n);
    e.ensure_tracking();
    e.prepare();
}

template <class T> bool vec_eq(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

struct SplineCache {
    std::unique_ptr<SplineSet> G, B;
};

// Shared result of gpu_rk_ladder (used by gpu_rk_positive_control).
double g_gpu_G_drift_1e4 = -1.0;

// ============================================================================
// Case 6: gpu_rk_ladder (T4b)
// ============================================================================

void case_gpu_rk_ladder(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_rk_ladder (T4b) ===\n");
    const SplineSet* sets[2] = {C.G.get(), C.B.get()};
    for (const SplineSet* Sp : sets) {
        const SplineSet& S = *Sp;
        const std::string pn = pair_name(S.pair);
        double drift[7];
        bool clocks_ok = true, active_ok = true;
        for (int k = 0; k < 7; ++k) {
            const double tol = kTolLadder[k];
            DevParticles P(1024);
            ReferenceRkTracker eng(ctx.cuda_stream(), kSeed);
            setup_rk(eng, make_rk_cfg(tol, 0.5), S, P);
            const Snap s0 = snap_rk(eng, P);
            eng.step(1.7);
            const Snap s = snap_rk(eng, P);
            const real tt = eng.target_time();
            for (size_t i = 0; i < s.clock.size(); ++i) {
                if (!same_bits(s.clock[i], tt)) {
                    clocks_ok = false;
                }
            }
            if (!all_status(s, kStatusActive))
                active_ok = false;
            drift[k] = snap_drift(s0, s, S.host);
            const ReferenceRkStats st = eng.compute_stats();
            std::printf("  pair %s tol=%.0e  max label drift = %.6e  total accepted = %llu  total rejected = %llu  "
                        "n_active = %d  target = %.17g\n",
                        pn.c_str(), tol, drift[k], static_cast<unsigned long long>(st.total_accepted),
                        static_cast<unsigned long long>(st.total_rejected), st.n_active, eng.target_time());
        }
        const double slope = ls_slope(kTolLadder, drift, 7);
        std::printf("  [INFO] pair %s fitted LS slope of log10(drift) vs log10(tol) = %.4f (not gated: C^1 spline "
                    "velocity)\n",
                    pn.c_str(), slope);
        rep.check(clocks_ok, "gpu_rk_ladder/clocks_equal_target_time_bitwise_" + pn);
        rep.check(active_ok, "gpu_rk_ladder/all_particles_active_" + pn);
        rep.check(drift[6] <= drift[0] / 100.0, "gpu_rk_ladder/drift_1e-10<=drift_1e-4/100_" + pn,
                  strf("%.6e <= %.6e", drift[6], drift[0] / 100.0));
        if (S.pair == kPairG)
            g_gpu_G_drift_1e4 = drift[0];
    }
}

// ============================================================================
// Case 6b: gpu_rk_multi_call_landing
// ============================================================================

void case_gpu_rk_multi_call_landing(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_rk_multi_call_landing ===\n");
    DevParticles P(1024);
    ReferenceRkTracker eng(ctx.cuda_stream(), kSeed);
    setup_rk(eng, make_rk_cfg(1e-8, 0.5), *C.G, P);
    bool clocks_ok = true, active_ok = true;
    for (int call = 1; call <= 10; ++call) {
        eng.step(0.17);
        const Snap s = snap_rk(eng, P);
        const real tt = eng.target_time();
        for (size_t i = 0; i < s.clock.size(); ++i) {
            if (!same_bits(s.clock[i], tt))
                clocks_ok = false;
        }
        if (!all_status(s, kStatusActive))
            active_ok = false;
    }
    const ReferenceRkStats st = eng.compute_stats();
    std::printf("  pair G tol=1e-08  final target = %.17g  total accepted = %llu  total rejected = %llu  n_active = %d\n",
                eng.target_time(), static_cast<unsigned long long>(st.total_accepted),
                static_cast<unsigned long long>(st.total_rejected), st.n_active);
    rep.check(clocks_ok, "gpu_rk_multi_call_landing/clocks_equal_target_time_bitwise_after_every_call");
    rep.check(active_ok, "gpu_rk_multi_call_landing/all_particles_active");
}

// ============================================================================
// Case 7: gpu_rk_positive_control
// ============================================================================

void case_gpu_rk_positive_control(TestReport& rep) {
    std::printf("\n=== gpu_rk_positive_control ===\n");
    rep.check(g_gpu_G_drift_1e4 >= 1e-7, "gpu_rk_positive_control/G_drift_tol1e-4>=1e-7",
              strf("%.6e (gate >= 1e-7)", g_gpu_G_drift_1e4));
}

// ============================================================================
// Case 8: gpu_rk_host_core_agreement
// ============================================================================

void case_gpu_rk_host_core_agreement(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_rk_host_core_agreement ===\n");
    const SplineSet& S = *C.G;
    const double dt_max = 1.0 / 64;
    const ReferenceRkConfig cfg = make_rk_cfg(1e300, dt_max);
    DevParticles P(64);
    ReferenceRkTracker eng(ctx.cuda_stream(), kSeed);
    setup_rk(eng, cfg, S, P);
    const Snap s0 = snap_rk(eng, P);
    for (int k = 0; k < 10; ++k)
        eng.step(0.17);
    const Snap s = snap_rk(eng, P);
    const LabelVelocity<SplineLabelPair> vel{S.host};
    const ReferenceRkParams prm = make_rk_prm(cfg.tol, cfg.dt_max, cfg.min_step, cfg.max_steps_per_call);
    bool ok = all_status(s, kStatusActive);
    double dx = 0.0;
    for (size_t i = 0; i < 64; ++i) {
        const double x0[3] = {s0.x[i], s0.y[i], s0.z[i]};
        RkState st;
        rk_init(st, x0, 0.1 * dt_max);
        RkCounters cnt{0u, 0u};
        double tt = 0.0;
        for (int k = 0; k < 10; ++k) {
            tt += 0.17;
            ok = ok && rk_advance_to_time(vel, st, S.host.L, tt, prm, cnt) == kStatusActive;
        }
        double xh[3], xg[3];
        unwrapped(st.xi, st.w, S.host.L, xh);
        snap_unwrapped(s, i, S.dev.L, xg);
        dx = std::max(dx, maxnorm_diff(xh, xg));
    }
    std::printf("  max |x_u GPU - x_u host core| = %.3e\n", dx);
    rep.check(ok, "gpu_rk_host_core_agreement/all_active");
    rep.check(dx <= 1e-11, "gpu_rk_host_core_agreement/positions", strf("%.3e (gate 1e-11)", dx));
}

// ============================================================================
// Case 9: gpu_rk_no_allocation (T5)
// ============================================================================

size_t free_bytes() {
    size_t f = 0, t = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&f, &t));
    return f;
}

void case_gpu_rk_no_allocation(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_rk_no_allocation (T5) ===\n");
    DevParticles P(1024);
    ReferenceRkTracker eng(ctx.cuda_stream(), kSeed);
    setup_rk(eng, make_rk_cfg(1e-8, 0.5), *C.G, P);
    eng.step(0.01);
    eng.synchronize();
    const size_t f0 = free_bytes();
    for (int i = 0; i < 100; ++i)
        eng.step(0.01);
    eng.synchronize();
    const size_t f1 = free_bytes();
    std::printf("  free before = %zu, after = %zu, delta = %lld\n", f0, f1,
                static_cast<long long>(f0) - static_cast<long long>(f1));
    rep.check(f0 == f1, "gpu_rk_no_allocation/cudaMemGetInfo_unchanged", "100 step(0.01)");
}

// ============================================================================
// Case 10: gpu_rk_determinism (T6)
// ============================================================================

void case_gpu_rk_determinism(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_rk_determinism (T6) ===\n");
    Snap r[2];
    for (int run = 0; run < 2; ++run) {
        DevParticles P(1024);
        ReferenceRkTracker eng(ctx.cuda_stream(), kSeed);
        setup_rk(eng, make_rk_cfg(1e-8, 0.5), *C.G, P);
        for (int i = 0; i < 10; ++i)
            eng.step(0.17);
        r[run] = snap_rk(eng, P);
        const ReferenceRkStats st = eng.compute_stats();
        std::printf("  run %d: n_active %d, target %.4f, total accepted %llu, rejected %llu\n", run + 1, st.n_active,
                    eng.target_time(), static_cast<unsigned long long>(st.total_accepted),
                    static_cast<unsigned long long>(st.total_rejected));
    }
    const Snap& a = r[0];
    const Snap& b = r[1];
    rep.check(vec_eq(a.x, b.x) && vec_eq(a.y, b.y) && vec_eq(a.z, b.z), "gpu_rk_determinism/positions_memcmp");
    rep.check(vec_eq(a.wx, b.wx) && vec_eq(a.wy, b.wy) && vec_eq(a.wz, b.wz), "gpu_rk_determinism/wraps_memcmp");
    rep.check(vec_eq(a.st, b.st), "gpu_rk_determinism/status_memcmp");
    rep.check(vec_eq(a.clock, b.clock), "gpu_rk_determinism/clocks_memcmp");
    rep.check(vec_eq(a.h, b.h), "gpu_rk_determinism/step_proposals_memcmp");
    rep.check(vec_eq(a.acc, b.acc) && vec_eq(a.rej, b.rej), "gpu_rk_determinism/accepted_rejected_memcmp");
}

// ============================================================================
// Case 11: gpu_ps_vs_rk_cross_check
// ============================================================================

void case_gpu_ps_vs_rk_cross_check(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_ps_vs_rk_cross_check ===\n");
    const SplineSet& S = *C.G;
    const int n = 256;
    DevParticles PR(n);
    ReferenceRkTracker rk(ctx.cuda_stream(), kSeed);
    setup_rk(rk, make_rk_cfg(1e-12, 1.0 / 128), S, PR);
    const Snap r0 = snap_rk(rk, PR);
    for (int i = 0; i < 20; ++i)
        rk.step(0.05);
    const Snap r1 = snap_rk(rk, PR);
    const double T = rk.target_time();
    const double d_rk = snap_drift(r0, r1, S.host);
    const bool rk_active = all_status(r1, kStatusActive);
    std::printf("  RK: target T = %.17g, d_RK = %.6e, all active = %s\n", T, d_rk, rk_active ? "yes" : "no");
    rep.check(rk_active, "gpu_ps_vs_rk_cross_check/rk_all_active");

    const double dsm[2] = {1.0 / 64, 1.0 / 128};
    const char* dsn[2] = {"1/64", "1/128"};
    const double Dref[2] = {g_D_ds64, g_D_ds128};
    for (int k = 0; k < 2; ++k) {
        DevParticles PP(n);
        PseudoSymplecticTracker ps(ctx.cuda_stream(), kSeed);
        PseudoSymplecticConfig cfg{};
        cfg.ds_max = dsm[k];
        cfg.tol_psi = 1e-13;
        setup_ps(ps, cfg, S, PP);
        const Snap p0 = snap_ps(ps, PP);
        const bool same_start = vec_eq(p0.x, r0.x) && vec_eq(p0.y, r0.y) && vec_eq(p0.z, r0.z);
        for (int i = 0; i < 20; ++i)
            ps.step(0.05);
        const Snap p1 = snap_ps(ps, PP);
        const bool ps_active = all_status(p1, kStatusActive);
        double dcorr = 0.0, dmis = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            double xp[3], xr[3];
            snap_unwrapped(p1, i, S.dev.L, xp);
            snap_unwrapped(r1, i, S.dev.L, xr);
            real xi[3];
            int32_t w[3];
            snap_pos(r1, i, xi, w);
            LabelSample ls;
            S.host(xi, w, ls);
            real c[3];
            cross3(ls.g1, ls.g2, c);
            const double dt = p1.clock[i] - T;
            dmis = std::max(dmis, std::fabs(dt));
            for (int d = 0; d < 3; ++d)
                dcorr = std::max(dcorr, std::fabs(xp[d] - xr[d] - c[d] * dt));
        }
        const double gate = 1.5 * Dref[k] + 10.0 * d_rk;
        std::printf("  ds_max=%-5s D_corr = %.6e, D(core) = %.6e, d_RK = %.6e, max |t_p - T| = %.6e, ps target = "
                    "%.17g\n",
                    dsn[k], dcorr, Dref[k], d_rk, dmis, ps.target_time());
        rep.check(same_start, std::string("gpu_ps_vs_rk_cross_check/same_start_positions_ds") + dsn[k]);
        rep.check(ps_active, std::string("gpu_ps_vs_rk_cross_check/ps_all_active_ds") + dsn[k]);
        rep.check(Dref[k] > 0.0 && dcorr <= gate, std::string("gpu_ps_vs_rk_cross_check/D_corr_ds") + dsn[k],
                  strf("%.6e (gate 1.5 D + 10 d_RK = %.6e)", dcorr, gate));
    }
}

// ============================================================================
// Case 12: rk_validation_errors
// ============================================================================

template <class Ex, class F> bool throws_as(F&& f) {
    try {
        f();
    } catch (const Ex&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void case_rk_validation_errors(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== rk_validation_errors ===\n");
    const SplineSet& S = *C.G;
    const cudaStream_t sm = ctx.cuda_stream();
    struct Bad {
        const char* name;
        ReferenceRkConfig cfg;
    };
    std::vector<Bad> bad;
    {
        ReferenceRkConfig c = make_rk_cfg(1e-8, 0.5);
        c.tol = 0.0;
        bad.push_back({"tol_0", c});
        c.tol = -1e-8;
        bad.push_back({"tol_negative", c});
        c.tol = kNaN;
        bad.push_back({"tol_NaN", c});
    }
    {
        ReferenceRkConfig c = make_rk_cfg(1e-8, 0.5);
        c.dt_max = 0.0;
        bad.push_back({"dt_max_0", c});
        c.dt_max = kNaN;
        bad.push_back({"dt_max_NaN", c});
    }
    {
        ReferenceRkConfig c = make_rk_cfg(1e-8, 0.5);
        c.min_step = -1e-14;
        bad.push_back({"min_step_negative", c});
    }
    {
        ReferenceRkConfig c = make_rk_cfg(1e-8, 0.5);
        c.max_steps_per_call = 0;
        bad.push_back({"max_steps_per_call_0", c});
    }
    for (const Bad& b : bad) {
        ReferenceRkTracker e(sm, kSeed);
        rep.check(throws_as<std::invalid_argument>([&] { e.configure(b.cfg); }),
                  std::string("rk_validation_errors/configure_rejects_") + b.name);
    }
    DevParticles P(8);
    // The real API requires ensure_tracking before prepare; ensure_tracking is
    // attempted (its result is printed) and prepare must throw std::logic_error.
    {
        ReferenceRkTracker e(sm, kSeed);
        e.bind_labels(S.dev);
        e.bind_particles(P.soa);
        const bool et = throws_as<std::logic_error>([&] { e.ensure_tracking(); });
        std::printf("  missing configure: ensure_tracking threw logic_error = %s\n", et ? "yes" : "no");
        rep.check(throws_as<std::logic_error>([&] { e.prepare(); }), "rk_validation_errors/prepare_before_configure");
    }
    {
        ReferenceRkTracker e(sm, kSeed);
        e.configure(make_rk_cfg(1e-8, 0.5));
        e.bind_particles(P.soa);
        const bool et = throws_as<std::logic_error>([&] { e.ensure_tracking(); });
        std::printf("  missing bind_labels: ensure_tracking threw logic_error = %s\n", et ? "yes" : "no");
        rep.check(throws_as<std::logic_error>([&] { e.prepare(); }), "rk_validation_errors/prepare_before_bind_labels");
    }
    {
        ReferenceRkTracker e(sm, kSeed);
        e.configure(make_rk_cfg(1e-8, 0.5));
        e.bind_labels(S.dev);
        const bool et = throws_as<std::logic_error>([&] { e.ensure_tracking(); });
        std::printf("  missing bind_particles: ensure_tracking threw logic_error = %s\n", et ? "yes" : "no");
        rep.check(throws_as<std::logic_error>([&] { e.prepare(); }),
                  "rk_validation_errors/prepare_before_bind_particles");
    }
    {
        ReferenceRkTracker e(sm, kSeed);
        e.configure(make_rk_cfg(1e-8, 0.5));
        e.bind_labels(S.dev);
        e.bind_particles(P.soa);
        e.inject_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0, P.n);
        e.ensure_tracking();
        rep.check(throws_as<std::logic_error>([&] { e.step(0.01); }), "rk_validation_errors/step_before_prepare");
        e.prepare();
        rep.check(throws_as<std::invalid_argument>([&] { e.step(-1.0); }), "rk_validation_errors/step_negative");
        rep.check(throws_as<std::invalid_argument>([&] { e.step(kNaN); }), "rk_validation_errors/step_NaN");
        e.synchronize();
    }
    {
        ParticlesSoA<real> pw = P.soa;
        pw.wrapX = nullptr;
        ReferenceRkTracker e(sm, kSeed);
        e.configure(make_rk_cfg(1e-8, 0.5));
        e.bind_labels(S.dev);
        bool threw = false;
        try {
            e.bind_particles(pw);
        } catch (const std::exception&) {
            threw = true;
        }
        if (!threw) {
            try {
                e.ensure_tracking();
                e.prepare();
            } catch (const std::exception&) {
                threw = true;
            }
        }
        rep.check(threw, "rk_validation_errors/missing_wrap_array_throws", "wrapX = nullptr");
    }
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
        timed("rk_tableau_order", [&] { case_rk_tableau_order(rep); });
        timed("rk_tolerance_ladder+rk_x1_exact+rk_position_error", [&] { case_rk_tolerance_ladder(rep); });
        timed("rk_landing", [&] { case_rk_landing(rep); });
        timed("rk_failure_paths", [&] { case_rk_failure_paths(rep); });
        timed("core_ps_vs_rk_cross_check", [&] { case_core_ps_vs_rk_cross_check(rep); });

        CudaContext ctx(0);
        SplineCache C;
        timed("spline_setup", [&] {
            std::printf("\n=== spline setup (SF-28 GPU prefilter) ===\n");
            const Grid3D g32(32, 32, 32, 1.0 / 32, 1.0 / 32, 1.0 / 32);
            const Grid3D gB(256, 256, 4, 1.0 / 256, 1.0 / 256, 1.0 / 4);
            C.G = build_spline(ctx, kPairG, g32);
            C.B = build_spline(ctx, kPairB, gB);
        });
        if (C.G && C.B) {
            timed("gpu_rk_ladder", [&] { case_gpu_rk_ladder(ctx, C, rep); });
            timed("gpu_rk_multi_call_landing", [&] { case_gpu_rk_multi_call_landing(ctx, C, rep); });
            timed("gpu_rk_positive_control", [&] { case_gpu_rk_positive_control(rep); });
            timed("gpu_rk_host_core_agreement", [&] { case_gpu_rk_host_core_agreement(ctx, C, rep); });
            timed("gpu_rk_no_allocation", [&] { case_gpu_rk_no_allocation(ctx, C, rep); });
            timed("gpu_rk_determinism", [&] { case_gpu_rk_determinism(ctx, C, rep); });
            timed("gpu_ps_vs_rk_cross_check", [&] { case_gpu_ps_vs_rk_cross_check(ctx, C, rep); });
            timed("rk_validation_errors", [&] { case_rk_validation_errors(ctx, C, rep); });
        } else {
            rep.check(false, "spline_setup/complete");
        }
    } catch (const std::exception& e) {
        rep.check(false, "unexpected_exception", e.what());
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== timing_record (information only) ===\n");
    for (const auto& t : times)
        std::printf("  %-52s %8.3f s\n", t.first.c_str(), t.second);
    std::printf("  %-52s %8.3f s\n", "whole executable", wall);
    std::printf("\n%d checks, %d failed, overall %s\n", rep.checks, rep.fails, rep.overall_pass ? "PASS" : "FAIL");
    return rep.overall_pass ? 0 : 1;
}
