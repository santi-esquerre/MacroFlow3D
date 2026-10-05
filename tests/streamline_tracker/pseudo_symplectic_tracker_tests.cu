/**
 * @file pseudo_symplectic_tracker_tests.cu
 * @brief SF-31 N3: fast contract tests of the pseudo-symplectic streamline
 *        tracker (`src/physics/particles/streamline_tracker/PseudoSymplecticTracker.cuh`).
 *
 * Standalone ctest-friendly runner (style of
 * `tests/interpolation/periodic_tricubic_bspline_tests.cu`: printed
 * `[PASS]/[FAIL]` checks, `main` returns 0/1). Fast tier.
 *
 * Every case, particle set, grid, ladder, tolerance and threshold is
 * PRE-REGISTERED by the SF-31 orchestrator (node N3 specification, readings
 * T1..T6 of the orchestration record) and implemented verbatim; none may be
 * adjusted to make a failing line pass.
 *
 * Host-core cases (analytic labels, no spline, no GPU kernels):
 *   core_analytic_selftest, core_order_ladder (T2), core_helix_constant,
 *   core_uniform_exact (T3), core_label_conservation (T1),
 *   core_advance_to_time, core_failure_paths, core_degeneracy_threshold (D-11).
 * GPU-engine cases (SF-28 splined labels):
 *   gpu_label_conservation (T1), gpu_order_ladder (T2), gpu_uniform_exact (T3),
 *   gpu_engine_mode, gpu_host_core_agreement, gpu_no_allocation (T5),
 *   gpu_determinism (T6), gpu_failure_paths, validation_errors, timing_record.
 */

#include "analytic_pairs.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "src/physics/particles/streamline_tracker/PseudoSymplecticTracker.cuh"
#include "src/physics/particles/streamline_tracker/StreamlineTrackerCommon.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cfloat>
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

// ============================================================================
// Fixtures
// ============================================================================

struct P3 {
    double v[3];
};

const P3 kFixed[4] = {{{0.1, 0.2, 0.3}}, {{0.35, 0.6, 0.85}}, {{0.7, 0.15, 0.5}}, {{0.9, 0.8, 0.05}}};

/// P1..P4 followed by 60 points inject_uniform01(seed = 20261005, index = 0..59, axis).
std::vector<P3> host_points64() {
    std::vector<P3> p(kFixed, kFixed + 4);
    for (uint64_t i = 0; i < 60; ++i) {
        P3 q;
        for (int a = 0; a < 3; ++a)
            q.v[a] = inject_uniform01(20261005ULL, i, a);
        p.push_back(q);
    }
    return p;
}

PseudoSymplecticParams make_prm(double tol, int max_iter = 8) {
    PseudoSymplecticParams prm{};
    prm.tol_psi = tol;
    prm.max_newton_iter = max_iter;
    prm.trust_factor = 1.0;
    prm.min_cross_norm = 0.0;
    prm.min_cross_sin2 = kDefaultMinCrossSin2;
    return prm;
}

void unwrapped(const real xi[3], const int32_t w[3], const real L[3], double xu[3]) {
    for (int d = 0; d < 3; ++d)
        xu[d] = std::fma(static_cast<real>(w[d]), L[d], xi[d]);
}

template <class E>
double label_residual(const E& lab, const real xi[3], const int32_t w[3], real p1, real p2) {
    LabelSample s;
    lab(xi, w, s);
    return std::max(std::fabs(s.psi1 - p1), std::fabs(s.psi2 - p2));
}

double maxnorm_diff(const double a[3], const double b[3]) {
    return std::max(std::fabs(a[0] - b[0]), std::max(std::fabs(a[1] - b[1]), std::fabs(a[2] - b[2])));
}

/// Host start state: (x0, w = 0, t = 0), labels evaluated with the core's own test.
template <class E>
uint8_t core_init(const E& lab, const double x0[3], const PseudoSymplecticParams& prm,
                  PanelState& st, real& p1, real& p2) {
    for (int d = 0; d < 3; ++d) {
        st.xi[d] = x0[d];
        st.w[d] = 0;
    }
    st.t = 0.0;
    const uint8_t code = evaluate_label_state(lab, st.xi, st.w, prm, st.at);
    p1 = st.at.psi1;
    p2 = st.at.psi2;
    return code;
}

double order2(double coarse, double fine) {
    return std::log2(coarse / fine);
}

const double kDsLadder[5] = {1.0 / 16, 1.0 / 32, 1.0 / 64, 1.0 / 128, 1.0 / 256};
const char* kDsName[5] = {"1/16", "1/32", "1/64", "1/128", "1/256"};

// Shared result for core_advance_to_time.
double g_core_B_Ex_ds64 = -1.0;

// ============================================================================
// Case 0: core_analytic_selftest (header check)
// ============================================================================

void case_core_analytic_selftest(TestReport& rep) {
    std::printf("\n=== core_analytic_selftest (analytic_pairs.hpp) ===\n");
    const std::vector<P3> pts = host_points64();
    const int pairs[6] = {kPairU, kPairA, kPairH, kPairB, kPairG, kPairD};
    const double step = 1e-6;
    for (int pi = 0; pi < 6; ++pi) {
        const AnalyticLabels lab = make_analytic_labels(pairs[pi]);
        double max_fd = 0.0, max_cdot = 0.0;
        for (const P3& p : pts) {
            const int32_t w[3] = {0, 0, 0};
            LabelSample s;
            lab(p.v, w, s);
            for (int d = 0; d < 3; ++d) {
                real xp[3] = {p.v[0], p.v[1], p.v[2]};
                real xm[3] = {p.v[0], p.v[1], p.v[2]};
                xp[d] += step;
                xm[d] -= step;
                LabelSample sp, sm;
                lab(xp, w, sp);
                lab(xm, w, sm);
                const double fd1 = (sp.psi1 - sm.psi1) / (2.0 * step);
                const double fd2 = (sp.psi2 - sm.psi2) / (2.0 * step);
                max_fd = std::max(max_fd, std::max(std::fabs(fd1 - s.g1[d]), std::fabs(fd2 - s.g2[d])));
            }
            real c[3];
            cross3(s.g1, s.g2, c);
            const double d1 = c[0] * s.g1[0] + c[1] * s.g1[1] + c[2] * s.g1[2];
            const double d2 = c[0] * s.g2[0] + c[1] * s.g2[1] + c[2] * s.g2[2];
            max_cdot = std::max(max_cdot, std::max(std::fabs(d1), std::fabs(d2)));
        }
        const std::string pn = pair_name(pairs[pi]);
        rep.check(max_fd <= 1e-7, "core_analytic_selftest/gradient_fd_" + pn,
                  strf("max |grad - centred FD(h=1e-6)| = %.3e (gate 1e-7)", max_fd));
        rep.check(max_cdot <= 1e-14, "core_analytic_selftest/c_dot_grad_" + pn,
                  strf("max |c . grad psi_i| = %.3e (gate 1e-14)", max_cdot));
    }
    // Wrap consistency of the affine part: (xi, w) and (xi + w, 0) same labels for dyadic xi.
    {
        const AnalyticLabels lab = make_analytic_labels(kPairB);
        const real xi[3] = {0.25, 0.75, 0.5};
        const int32_t w[3] = {3, -2, 1};
        const real xs[3] = {3.25, -1.25, 1.5};
        const int32_t w0[3] = {0, 0, 0};
        LabelSample a, b;
        lab(xi, w, a);
        lab(xs, w0, b);
        const double dd = std::max(std::fabs(a.psi1 - b.psi1), std::fabs(a.psi2 - b.psi2));
        rep.check(dd <= 1e-14, "core_analytic_selftest/affine_unwrap_B",
                  strf("|psi(xi, w) - psi(xi + w, 0)| = %.3e", dd));
    }
    // Exact trajectories satisfy the label invariance (closed forms consistent with the labels).
    {
        double worst = 0.0;
        const int cp[4] = {kPairU, kPairA, kPairH, kPairB};
        for (int k = 0; k < 4; ++k) {
            const AnalyticLabels lab = make_analytic_labels(cp[k]);
            for (const P3& p : pts) {
                const int32_t w0[3] = {0, 0, 0};
                LabelSample s0, s1;
                lab(p.v, w0, s0);
                double x[3];
                exact_position(cp[k], p.v, 1.37, x);
                lab(x, w0, s1);
                worst = std::max(worst, std::max(std::fabs(s1.psi1 - s0.psi1), std::fabs(s1.psi2 - s0.psi2)));
            }
        }
        rep.check(worst <= 1e-14, "core_analytic_selftest/exact_position_on_label_curve",
                  strf("max label change along exact_position (t = 1.37), U/A/H/B: %.3e", worst));
    }
    std::printf("  helix constants: kappa = %.15e, |c| = %.15e\n", helix_kappa(), helix_speed());
}

// ============================================================================
// Case 1: core_order_ladder (T2) + core_helix_constant
// ============================================================================

void case_core_order_ladder(TestReport& rep) {
    std::printf("\n=== core_order_ladder (T2) + core_helix_constant ===\n");
    const std::vector<P3> pts = host_points64();
    const int pairs[3] = {kPairA, kPairH, kPairB};
    const PseudoSymplecticParams prm = make_prm(1e-13);
    const double kap = helix_kappa(), cn = helix_speed();

    for (int pi = 0; pi < 3; ++pi) {
        const int pair = pairs[pi];
        const std::string pn = pair_name(pair);
        const AnalyticLabels lab = make_analytic_labels(pair);
        double max_et[5], max_ex[5];
        std::vector<double> et[5];
        bool all_active = true;
        uint32_t nmax = 0;
        for (int k = 0; k < 5; ++k) {
            const double ds = kDsLadder[k];
            const int n = static_cast<int>(2.0 / ds);
            max_et[k] = 0.0;
            max_ex[k] = 0.0;
            et[k].assign(pts.size(), 0.0);
            for (size_t ip = 0; ip < pts.size(); ++ip) {
                PanelState st;
                real p1, p2;
                if (core_init(lab, pts[ip].v, prm, st, p1, p2) != kStatusActive) {
                    all_active = false;
                    continue;
                }
                ProjectionCounters cnt{0u, 0u};
                for (int i = 0; i < n; ++i) {
                    if (advance_panel(lab, st, p1, p2, ds, prm, cnt) != kStatusActive) {
                        all_active = false;
                        break;
                    }
                }
                nmax = std::max(nmax, cnt.newton_iter_max);
                double xu[3], xe[3];
                unwrapped(st.xi, st.w, lab.L, xu);
                exact_position(pair, pts[ip].v, st.t, xe);
                const double e_t = st.t - (xu[0] - pts[ip].v[0]);
                et[k][ip] = e_t;
                max_et[k] = std::max(max_et[k], std::fabs(e_t));
                max_ex[k] = std::max(max_ex[k], maxnorm_diff(xu, xe));
            }
            std::printf("  pair %s ds=%-5s n=%4d  max|e_t|=%.6e  max E_x=%.6e  P1..P4 e_t: %.7g %.7g "
                        "%.7g %.7g\n",
                        pn.c_str(), kDsName[k], n, max_et[k], max_ex[k], et[k][0], et[k][1],
                        et[k][2], et[k][3]);
        }
        std::printf("  pair %s max Newton iterations = %u\n", pn.c_str(), nmax);
        rep.check(all_active, "core_order_ladder/all_panels_active_" + pn);
        for (int k = 0; k < 4; ++k) {
            const double ot = order2(max_et[k], max_et[k + 1]);
            const double ox = order2(max_ex[k], max_ex[k + 1]);
            rep.check(ot >= 1.8 && ot <= 2.2,
                      "core_order_ladder/order_e_t_" + pn + "_" + kDsName[k] + "->" + kDsName[k + 1],
                      strf("%.4f (gate [1.8, 2.2])", ot));
            rep.check(ox >= 1.8 && ox <= 2.2,
                      "core_order_ladder/order_E_x_" + pn + "_" + kDsName[k] + "->" + kDsName[k + 1],
                      strf("%.4f (gate [1.8, 2.2])", ox));
        }
        if (pair == kPairB)
            g_core_B_Ex_ds64 = max_ex[2];

        if (pair == kPairH) {
            for (int k = 0; k < 5; ++k) {
                const double ds = kDsLadder[k];
                const double pred = 2.0 * kap * kap * ds * ds / (12.0 * cn);
                double rmin = 1e300, rmax = -1e300;
                for (double e : et[k]) {
                    const double r = e / pred;
                    rmin = std::min(rmin, r);
                    rmax = std::max(rmax, r);
                }
                std::printf("  helix ratio e_t / (S kappa^2 ds^2 / (12 |c|)) ds=%-5s: min %.6f max %.6f "
                            "(P1..P4: %.6f %.6f %.6f %.6f)\n",
                            kDsName[k], rmin, rmax, et[k][0] / pred, et[k][1] / pred,
                            et[k][2] / pred, et[k][3] / pred);
                if (k >= 2) {
                    rep.check(rmin >= 0.99 && rmax <= 1.01,
                              std::string("core_helix_constant/ratio_ds_") + kDsName[k],
                              strf("[%.6f, %.6f] (gate [0.99, 1.01])", rmin, rmax));
                }
            }
        }
    }
}

// ============================================================================
// Case 2: core_uniform_exact (T3)
// ============================================================================

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

void case_core_uniform_exact(TestReport& rep) {
    std::printf("\n=== core_uniform_exact (T3) ===\n");
    const AnalyticLabels lab = make_analytic_labels(kPairU);
    const PseudoSymplecticParams prm = make_prm(1e-13);
    const P3 starts[2] = {{{0.125, 0.375, 0.625}}, {{0.984375, 0.015625, 0.5}}};
    for (int s = 0; s < 2; ++s) {
        PanelState st;
        real p1, p2;
        bool ok = core_init(lab, starts[s].v, prm, st, p1, p2) == kStatusActive;
        ProjectionCounters cnt{0u, 0u};
        for (int i = 0; i < 200 && ok; ++i)
            ok = advance_panel(lab, st, p1, p2, 1.0 / 64, prm, cnt) == kStatusActive;
        double xu[3];
        unwrapped(st.xi, st.w, lab.L, xu);
        const double T = 200.0 / 64.0;
        const bool bt = same_bits(st.t, T);
        const bool bx = same_bits(xu[0], starts[s].v[0] + T);
        const bool by = same_bits(xu[1], starts[s].v[1]) && same_bits(st.xi[1], starts[s].v[1]) && st.w[1] == 0;
        const bool bz = same_bits(xu[2], starts[s].v[2]) && same_bits(st.xi[2], starts[s].v[2]) && st.w[2] == 0;
        const bool range = st.xi[0] >= 0.0 && st.xi[0] < 1.0;
        const std::string tag = strf("start%d", s + 1);
        rep.check(ok, "core_uniform_exact/i_all_active_" + tag);
        rep.check(bt, "core_uniform_exact/i_clock_bitwise_" + tag, strf("t = %.17g (expect %.17g)", st.t, T));
        rep.check(bx, "core_uniform_exact/i_x1_bitwise_" + tag,
                  strf("x1_u = %.17g (expect %.17g), wrapX = %d", xu[0], starts[s].v[0] + T, st.w[0]));
        rep.check(by && bz, "core_uniform_exact/i_x2_x3_bitwise_unchanged_" + tag,
                  strf("x2 = %.17g, x3 = %.17g", xu[1], xu[2]));
        rep.check(range, "core_uniform_exact/i_xi0_in_[0,1)_" + tag, strf("xi[0] = %.17g", st.xi[0]));
    }
    {
        const P3 s0 = {{0.3, 0.7, 0.1}};
        PanelState st;
        real p1, p2;
        bool ok = core_init(lab, s0.v, prm, st, p1, p2) == kStatusActive;
        ProjectionCounters cnt{0u, 0u};
        for (int i = 0; i < 250 && ok; ++i)
            ok = advance_panel(lab, st, p1, p2, 0.01, prm, cnt) == kStatusActive;
        double xu[3];
        unwrapped(st.xi, st.w, lab.L, xu);
        const double et = std::fabs(st.t - 2.5);
        const double ex = std::fabs(xu[0] - s0.v[0] - st.t);
        rep.check(ok, "core_uniform_exact/ii_all_active");
        rep.check(et <= 2.5e-13, "core_uniform_exact/ii_clock", strf("|t - 2.5| = %.3e (gate 2.5e-13)", et));
        rep.check(ex <= 2.5e-13, "core_uniform_exact/ii_x1_vs_clock",
                  strf("|x1_u - x1_0 - t| = %.3e (gate 2.5e-13)", ex));
        rep.check(same_bits(xu[1], s0.v[1]) && same_bits(xu[2], s0.v[2]) && st.w[1] == 0 && st.w[2] == 0,
                  "core_uniform_exact/ii_x2_x3_bitwise_unchanged",
                  strf("x2 = %.17g, x3 = %.17g", xu[1], xu[2]));
    }
}

// ============================================================================
// Case 3: core_label_conservation (T1)
// ============================================================================

void case_core_label_conservation(TestReport& rep) {
    std::printf("\n=== core_label_conservation (T1) ===\n");
    const std::vector<P3> pts = host_points64();
    const AnalyticLabels lab = make_analytic_labels(kPairG);
    const double tols[2] = {1e-8, 1e-12};
    for (double tol : tols) {
        const PseudoSymplecticParams prm = make_prm(tol);
        double max_res = 0.0;
        bool ok = true;
        uint32_t nmax = 0, clamps = 0;
        for (const P3& p : pts) {
            PanelState st;
            real p1, p2;
            if (core_init(lab, p.v, prm, st, p1, p2) != kStatusActive) {
                ok = false;
                continue;
            }
            ProjectionCounters cnt{0u, 0u};
            for (int i = 0; i < 256; ++i) {
                if (advance_panel(lab, st, p1, p2, 1.0 / 64, prm, cnt) != kStatusActive) {
                    ok = false;
                    break;
                }
                max_res = std::max(max_res, label_residual(lab, st.xi, st.w, p1, p2));
            }
            nmax = std::max(nmax, cnt.newton_iter_max);
            clamps += cnt.clamp_count;
        }
        const std::string tn = strf("tol%.0e", tol);
        std::printf("  tol_psi=%.0e: max label residual = %.3e, max Newton iterations = %u, clamps = %u\n",
                    tol, max_res, nmax, clamps);
        rep.check(ok, "core_label_conservation/all_panels_succeed_" + tn);
        rep.check(max_res <= tol + 1e-14, "core_label_conservation/residual_" + tn,
                  strf("%.3e (gate %.3e)", max_res, tol + 1e-14));
    }
}

// ============================================================================
// Case 4: core_advance_to_time
// ============================================================================

void case_core_advance_to_time(TestReport& rep) {
    std::printf("\n=== core_advance_to_time ===\n");
    const std::vector<P3> pts = host_points64();
    const AnalyticLabels lab = make_analytic_labels(kPairB);
    const PseudoSymplecticParams prm = make_prm(1e-13);
    struct Cfg {
        double ds, dt;
        const char* name;
    };
    const Cfg cfgs[3] = {{1.0 / 64, 0.01, "ds1/64_dt0.01"}, {1.0 / 64, 0.1, "ds1/64_dt0.1"},
                         {1.0 / 128, 0.037, "ds1/128_dt0.037"}};
    for (const Cfg& c : cfgs) {
        double m_all = 0.0, m1 = 0.0, m2 = 0.0;
        bool ok = true;
        for (const P3& p : pts) {
            PanelState st;
            real p1, p2;
            if (core_init(lab, p.v, prm, st, p1, p2) != kStatusActive) {
                ok = false;
                continue;
            }
            ProjectionCounters cnt{0u, 0u};
            double tt = 0.0;
            for (int call = 1; call <= 200; ++call) {
                tt += c.dt;
                if (advance_to_time(lab, st, p1, p2, tt, c.ds, 100000, prm, cnt) != kStatusActive) {
                    ok = false;
                    break;
                }
                const double mm = std::fabs(st.t - tt);
                m_all = std::max(m_all, mm);
                if (call <= 100)
                    m1 = std::max(m1, mm);
                else
                    m2 = std::max(m2, mm);
            }
        }
        const double bound = 2.5 * c.ds * c.ds;
        std::printf("  %s: max|t - t_target| all = %.4e (calls 1-100 %.4e, 101-200 %.4e), bound %.4e\n",
                    c.name, m_all, m1, m2, bound);
        rep.check(ok, std::string("core_advance_to_time/all_active_") + c.name);
        rep.check(m_all <= bound, std::string("core_advance_to_time/mismatch_bound_") + c.name,
                  strf("%.4e (gate 2.5 ds_max^2 = %.4e)", m_all, bound));
        rep.check(m2 <= 1.5 * m1 + 1e-12, std::string("core_advance_to_time/mismatch_not_growing_") + c.name,
                  strf("late %.4e <= 1.5 x early %.4e + 1e-12", m2, m1));
    }
    {
        double max_ex = 0.0;
        bool ok = true;
        for (const P3& p : pts) {
            PanelState st;
            real p1, p2;
            if (core_init(lab, p.v, prm, st, p1, p2) != kStatusActive) {
                ok = false;
                continue;
            }
            ProjectionCounters cnt{0u, 0u};
            double tt = 0.0;
            for (int call = 1; call <= 40; ++call) {
                tt += 0.05;
                if (advance_to_time(lab, st, p1, p2, tt, 1.0 / 64, 100000, prm, cnt) != kStatusActive) {
                    ok = false;
                    break;
                }
            }
            double xu[3], xe[3];
            unwrapped(st.xi, st.w, lab.L, xu);
            exact_position(kPairB, p.v, st.t, xe);
            max_ex = std::max(max_ex, maxnorm_diff(xu, xe));
        }
        const double gate = 1.5 * g_core_B_Ex_ds64;
        std::printf("  ds1/64_dt0.05 x40: max_p E_x = %.6e; ladder B ds=1/64 max_p E_x = %.6e\n", max_ex,
                    g_core_B_Ex_ds64);
        rep.check(ok, "core_advance_to_time/position_run_all_active");
        rep.check(g_core_B_Ex_ds64 > 0.0 && max_ex <= gate, "core_advance_to_time/position_error",
                  strf("%.6e (gate 1.5 x %.6e = %.6e)", max_ex, g_core_B_Ex_ds64, gate));
    }
}

// ============================================================================
// Case 5: core_failure_paths
// ============================================================================

bool panel_state_equal(const PanelState& a, const PanelState& b) {
    return std::memcmp(a.xi, b.xi, sizeof(a.xi)) == 0 && std::memcmp(a.w, b.w, sizeof(a.w)) == 0 &&
           same_bits(a.t, b.t) && std::memcmp(&a.at, &b.at, sizeof(LabelSample)) == 0;
}

void case_core_failure_paths(TestReport& rep) {
    std::printf("\n=== core_failure_paths ===\n");
    {
        const AnalyticLabels lab = make_analytic_labels(kPairD);
        const PseudoSymplecticParams prm = make_prm(1e-12);
        bool ev = true, pj = true, unchanged = true;
        for (int i = 0; i < 4; ++i) {
            PanelState st;
            real p1, p2;
            const uint8_t c1 = core_init(lab, kFixed[i].v, prm, st, p1, p2);
            real xi[3] = {kFixed[i].v[0], kFixed[i].v[1], kFixed[i].v[2]};
            const int32_t w[3] = {0, 0, 0};
            LabelSample at;
            ProjectionCounters cnt{0u, 0u};
            const uint8_t c2 = project_to_label_curve(lab, xi, w, p1, p2, 1.0 / 64, prm, at, cnt);
            std::printf("  D P%d: evaluate_label_state=%u project_to_label_curve=%u\n", i + 1, c1, c2);
            ev = ev && c1 == kStatusDegenerate;
            pj = pj && c2 == kStatusDegenerate;
            unchanged = unchanged && std::memcmp(xi, kFixed[i].v, sizeof(xi)) == 0;
        }
        rep.check(ev, "core_failure_paths/i_D_evaluate_degenerate");
        rep.check(pj, "core_failure_paths/i_D_project_degenerate");
        rep.check(unchanged, "core_failure_paths/i_D_position_bitwise_unchanged");
    }
    {
        const AnalyticLabels lab = make_analytic_labels(kPairH);
        const PseudoSymplecticParams prm = make_prm(1e-12, 1);
        bool code_ok = true, untouched = true;
        for (int i = 0; i < 4; ++i) {
            PanelState st;
            real p1, p2;
            core_init(lab, kFixed[i].v, prm, st, p1, p2);
            const PanelState saved = st;
            ProjectionCounters cnt{0u, 0u};
            const uint8_t c = advance_panel(lab, st, p1, p2, 0.5, prm, cnt);
            std::printf("  H P%d max_newton_iter=1 ds=0.5: code=%u\n", i + 1, c);
            code_ok = code_ok && c == kStatusNewtonFailed;
            untouched = untouched && panel_state_equal(st, saved);
        }
        rep.check(code_ok, "core_failure_paths/ii_H_newton_failed");
        rep.check(untouched, "core_failure_paths/ii_H_state_bitwise_untouched");
    }
    {
        const AnalyticLabels lab = make_analytic_labels(kPairB);
        const PseudoSymplecticParams prm = make_prm(1e-12);
        bool code_ok = true, t_ok = true;
        double max_res = 0.0;
        for (int i = 0; i < 4; ++i) {
            PanelState st;
            real p1, p2;
            core_init(lab, kFixed[i].v, prm, st, p1, p2);
            ProjectionCounters cnt{0u, 0u};
            const uint8_t c = advance_to_time(lab, st, p1, p2, 1.0, 1.0 / 64, 2, prm, cnt);
            const double r = label_residual(lab, st.xi, st.w, p1, p2);
            std::printf("  B P%d max_panels=2 target 1.0: code=%u t=%.6e residual=%.3e\n", i + 1, c,
                        st.t, r);
            code_ok = code_ok && c == kStatusSubstepLimit;
            t_ok = t_ok && st.t > 0.0 && st.t < 1.0;
            max_res = std::max(max_res, r);
        }
        rep.check(code_ok, "core_failure_paths/iii_B_substep_limit");
        rep.check(t_ok, "core_failure_paths/iii_B_clock_in_(0,1)");
        rep.check(max_res <= 1e-12 + 1e-14, "core_failure_paths/iii_B_label_residual",
                  strf("%.3e (gate %.3e)", max_res, 1e-12 + 1e-14));
    }
}

// ============================================================================
// Case 6: core_degeneracy_threshold (D-11)
// ============================================================================

template <class E>
__global__ void k_eval_state(E lab, PseudoSymplecticParams prm, real x, real y, real z,
                             uint8_t* out) {
    const real xi[3] = {x, y, z};
    const int32_t w[3] = {0, 0, 0};
    LabelSample at;
    out[0] = evaluate_label_state(lab, xi, w, prm, at);
}

uint8_t device_eval_state(const ThetaLabels& lab, const PseudoSymplecticParams& prm, const P3& p) {
    DeviceBuffer<uint8_t> d(1);
    k_eval_state<ThetaLabels><<<1, 1>>>(lab, prm, p.v[0], p.v[1], p.v[2], d.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    uint8_t h = 255;
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(&h, d.data(), 1, cudaMemcpyDeviceToHost));
    return h;
}

void case_core_degeneracy_threshold(TestReport& rep) {
    std::printf("\n=== core_degeneracy_threshold (D-11) ===\n");
    rep.check(kDefaultMinCrossSin2 == 16.0 * DBL_EPSILON, "core_degeneracy_threshold/default_is_16_eps",
              strf("kDefaultMinCrossSin2 = %.17g", kDefaultMinCrossSin2));
    const PseudoSymplecticParams prm_def = make_prm(1e-12);
    PseudoSymplecticParams prm_zero = prm_def;
    prm_zero.min_cross_sin2 = 0.0;
    const P3& p1pt = kFixed[0];

    // theta = 1e-6: active, one panel.
    {
        const double theta = 1e-6;
        const ThetaLabels lab = make_theta_labels(theta);
        PanelState st;
        real p1, p2;
        const uint8_t c = core_init(lab, p1pt.v, prm_def, st, p1, p2);
        rep.check(c == kStatusActive, "core_degeneracy_threshold/theta1e-6_active", strf("code %u", c));
        ProjectionCounters cnt{0u, 0u};
        const double sigma = 1.0 / 64;
        const uint8_t ca = advance_panel(lab, st, p1, p2, sigma, prm_def, cnt);
        double xu[3];
        unwrapped(st.xi, st.w, lab.L, xu);
        const double rx = std::fabs((xu[0] - p1pt.v[0]) - sigma) / sigma;
        const double dyz = std::max(std::fabs(xu[1] - p1pt.v[1]), std::fabs(xu[2] - p1pt.v[2]));
        const double tex = sigma / lab.st;
        const double rt = std::fabs(st.t - tex) / tex;
        rep.check(ca == kStatusActive, "core_degeneracy_threshold/theta1e-6_panel_succeeds", strf("code %u", ca));
        rep.check(rx <= 1e-9, "core_degeneracy_threshold/theta1e-6_x1_advance",
                  strf("relative error %.3e (gate 1e-9)", rx));
        rep.check(dyz <= 1e-9, "core_degeneracy_threshold/theta1e-6_x2_x3_fixed",
                  strf("max |dx2|, |dx3| = %.3e (gate 1e-9)", dyz));
        rep.check(rt <= 1e-9, "core_degeneracy_threshold/theta1e-6_clock",
                  strf("t = %.10e, sigma/sin(theta) = %.10e, relative error %.3e (gate 1e-9)", st.t, tex, rt));
        const uint8_t cd = device_eval_state(lab, prm_def, p1pt);
        rep.check(cd == kStatusActive, "core_degeneracy_threshold/theta1e-6_device_active", strf("code %u", cd));
    }
    // theta = 3e-8: degenerate with the default threshold, active with 0.
    {
        const ThetaLabels lab = make_theta_labels(3e-8);
        LabelSample at;
        const int32_t w[3] = {0, 0, 0};
        const uint8_t cdef = evaluate_label_state(lab, p1pt.v, w, prm_def, at);
        const uint8_t czero = evaluate_label_state(lab, p1pt.v, w, prm_zero, at);
        const double a11 = at.g1[0] * at.g1[0] + at.g1[1] * at.g1[1] + at.g1[2] * at.g1[2];
        const double a12 = at.g1[0] * at.g2[0] + at.g1[1] * at.g2[1] + at.g1[2] * at.g2[2];
        const double a22 = at.g2[0] * at.g2[0] + at.g2[1] * at.g2[1] + at.g2[2] * at.g2[2];
        std::printf("  theta=3e-8 host: det = %.6e, 16 eps a11 a22 = %.6e\n", a11 * a22 - a12 * a12,
                    kDefaultMinCrossSin2 * a11 * a22);
        rep.check(cdef == kStatusDegenerate, "core_degeneracy_threshold/theta3e-8_default_degenerate",
                  strf("code %u", cdef));
        rep.check(czero == kStatusActive, "core_degeneracy_threshold/theta3e-8_zero_threshold_active",
                  strf("code %u", czero));
        const uint8_t ddef = device_eval_state(lab, prm_def, p1pt);
        const uint8_t dzero = device_eval_state(lab, prm_zero, p1pt);
        rep.check(ddef == cdef, "core_degeneracy_threshold/theta3e-8_device_default_same_code",
                  strf("device %u host %u", ddef, cdef));
        rep.check(dzero == czero, "core_degeneracy_threshold/theta3e-8_device_zero_same_code",
                  strf("device %u host %u", dzero, czero));
    }
    // theta = 1e-8: degenerate with the default.
    {
        const ThetaLabels lab = make_theta_labels(1e-8);
        LabelSample at;
        const int32_t w[3] = {0, 0, 0};
        const uint8_t cdef = evaluate_label_state(lab, p1pt.v, w, prm_def, at);
        rep.check(cdef == kStatusDegenerate, "core_degeneracy_threshold/theta1e-8_default_degenerate",
                  strf("code %u", cdef));
        const uint8_t ddef = device_eval_state(lab, prm_def, p1pt);
        rep.check(ddef == cdef, "core_degeneracy_threshold/theta1e-8_device_same_code",
                  strf("device %u host %u", ddef, cdef));
    }
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
    AnalyticLabels exact{};
    double e_interp = 0.0;

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
    S->exact = make_analytic_labels(pair);
    // E_interp: 4096 points inject_uniform01(seed 777), both labels, host, w = 0.
    double e = 0.0;
    for (uint64_t i = 0; i < 4096; ++i) {
        const real x[3] = {inject_uniform01(777ULL, i, 0), inject_uniform01(777ULL, i, 1),
                           inject_uniform01(777ULL, i, 2)};
        const int32_t w[3] = {0, 0, 0};
        LabelSample a, b;
        S->host(x, w, a);
        S->exact(x, w, b);
        e = std::max(e, std::max(std::fabs(a.psi1 - b.psi1), std::fabs(a.psi2 - b.psi2)));
    }
    S->e_interp = e;
    std::printf("  [spline] pair %s on %dx%dx%d: E_interp = %.3e\n", pair_name(pair), g.nx, g.ny, g.nz, e);
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

    /// Positions written by the test (wraps 0, status active).
    void write(const std::vector<P3>& p) {
        std::vector<real> hx(n), hy(n), hz(n);
        for (int i = 0; i < n; ++i) {
            hx[i] = p[i].v[0];
            hy[i] = p[i].v[1];
            hz[i] = p[i].v[2];
        }
        h2d(x.data(), hx);
        h2d(y.data(), hy);
        h2d(z.data(), hz);
        h2d(st.data(), std::vector<uint8_t>(n, kStatusActive));
        const std::vector<int32_t> zero(n, 0);
        h2d(wx.data(), zero);
        h2d(wy.data(), zero);
        h2d(wz.data(), zero);
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    }
};

struct Snap {
    std::vector<real> x, y, z, clock, p1, p2;
    std::vector<int32_t> wx, wy, wz;
    std::vector<uint8_t> st;
    std::vector<uint32_t> fail, clamp, nmax;
};

Snap snap(PseudoSymplecticTracker& e, DevParticles& P) {
    e.synchronize();
    const size_t n = static_cast<size_t>(P.n);
    Snap s;
    s.x = d2h(P.x.data(), n);
    s.y = d2h(P.y.data(), n);
    s.z = d2h(P.z.data(), n);
    s.wx = d2h(P.wx.data(), n);
    s.wy = d2h(P.wy.data(), n);
    s.wz = d2h(P.wz.data(), n);
    s.st = d2h(P.st.data(), n);
    s.clock = d2h(e.clocks(), n);
    s.p1 = d2h(e.psi1_targets(), n);
    s.p2 = d2h(e.psi2_targets(), n);
    s.fail = d2h(e.fail_counts(), n);
    s.clamp = d2h(e.clamp_counts(), n);
    s.nmax = d2h(e.newton_iter_max(), n);
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

/// Max host label residual (host spline pair) over all particles of a snapshot.
double snap_residual(const Snap& s, const SplineSet& S) {
    double m = 0.0;
    for (size_t i = 0; i < s.x.size(); ++i) {
        real xi[3];
        int32_t w[3];
        snap_pos(s, i, xi, w);
        m = std::max(m, label_residual(S.host, xi, w, s.p1[i], s.p2[i]));
    }
    return m;
}

PseudoSymplecticConfig make_cfg(double ds_max, double tol) {
    PseudoSymplecticConfig c{};
    c.ds_max = ds_max;
    c.tol_psi = tol;
    return c;
}

/// Contract order: configure, bind_labels, bind_particles, inject_box (unless
/// positions were written by the test), ensure_tracking, prepare.
void setup_engine(PseudoSymplecticTracker& e, const PseudoSymplecticConfig& cfg, const SplineSet& S,
                  DevParticles& P, bool inject) {
    e.configure(cfg);
    e.bind_labels(S.dev);
    e.bind_particles(P.soa);
    if (inject)
        e.inject_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0, P.n);
    e.ensure_tracking();
    e.prepare();
}

bool all_status(const Snap& s, uint8_t code) {
    for (uint8_t v : s.st)
        if (v != code)
            return false;
    return true;
}

struct SplineCache {
    std::unique_ptr<SplineSet> A, H, B, G, D, U;
    const SplineSet& get(int pair) const {
        switch (pair) {
        case kPairA:
            return *A;
        case kPairH:
            return *H;
        case kPairB:
            return *B;
        case kPairG:
            return *G;
        case kPairD:
            return *D;
        default:
            return *U;
        }
    }
};

const uint64_t kSeed = 12345ULL;

// ============================================================================
// Case 7: gpu_label_conservation (T1)
// ============================================================================

void case_gpu_label_conservation(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_label_conservation (T1) ===\n");
    const int pairs[4] = {kPairA, kPairH, kPairB, kPairG};
    const double tols[2] = {1e-8, 1e-12};
    for (int pair : pairs) {
        const SplineSet& S = C.get(pair);
        const std::string pn = pair_name(pair);
        for (double tol : tols) {
            const std::string tag = pn + strf("_tol%.0e", tol);
            DevParticles P(1024);
            PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
            setup_engine(eng, make_cfg(1.0 / 64, tol), S, P, true);
            const Snap s0 = snap(eng, P);
            double max_res = 0.0, max_an = 0.0;
            Snap s;
            for (int call = 1; call <= 256; ++call) {
                eng.step_arclength(1.0 / 64);
                if (call % 16 == 0) {
                    s = snap(eng, P);
                    max_res = std::max(max_res, snap_residual(s, S));
                    for (size_t i = 0; i < s.x.size(); ++i) {
                        real xi[3], x0[3];
                        int32_t w[3];
                        const int32_t w0[3] = {0, 0, 0};
                        snap_pos(s, i, xi, w);
                        x0[0] = s0.x[i];
                        x0[1] = s0.y[i];
                        x0[2] = s0.z[i];
                        LabelSample a, b;
                        S.exact(xi, w, a);
                        S.exact(x0, w0, b);
                        max_an = std::max(max_an, std::max(std::fabs(a.psi1 - b.psi1), std::fabs(a.psi2 - b.psi2)));
                    }
                }
            }
            const PseudoSymplecticStats st = eng.compute_stats();
            std::printf("  %s: max spline-label residual = %.3e, max analytic-label change = %.3e "
                        "(2 E_interp = %.3e), n_active = %d, total_fail = %llu, max Newton = %u, clamps = %llu, "
                        "clock [%.4f, %.4f]\n",
                        tag.c_str(), max_res, max_an, 2.0 * S.e_interp, st.n_active,
                        static_cast<unsigned long long>(st.total_fail), st.max_newton_iter,
                        static_cast<unsigned long long>(st.total_clamps), st.min_clock, st.max_clock);
            rep.check(max_res <= tol + 1e-14, "gpu_label_conservation/spline_residual_" + tag,
                      strf("%.3e (gate %.3e)", max_res, tol + 1e-14));
            rep.check(st.n_active == 1024 && st.total_fail == 0, "gpu_label_conservation/all_active_no_fail_" + tag,
                      strf("n_active %d total_fail %llu", st.n_active, static_cast<unsigned long long>(st.total_fail)));
            rep.check(max_an <= tol + 2.0 * S.e_interp, "gpu_label_conservation/analytic_labels_" + tag,
                      strf("%.3e (gate tol + 2 E_interp = %.3e)", max_an, tol + 2.0 * S.e_interp));
            if (pair == kPairB) {
                int32_t mwx = 0;
                bool wy = false, wz = false, range = true;
                for (size_t i = 0; i < s.x.size(); ++i) {
                    mwx = std::max(mwx, s.wx[i]);
                    wy = wy || s.wy[i] != 0;
                    wz = wz || s.wz[i] != 0;
                    range = range && s.x[i] >= 0.0 && s.x[i] < 1.0 && s.y[i] >= 0.0 && s.y[i] < 1.0 &&
                            s.z[i] >= 0.0 && s.z[i] < 1.0;
                }
                int nwy = 0, nwz = 0;
                for (size_t i = 0; i < s.x.size(); ++i) {
                    nwy += s.wy[i] != 0;
                    nwz += s.wz[i] != 0;
                }
                rep.check(mwx >= 3, "gpu_label_conservation/coverage_wrapX>=3_" + tag, strf("max wrapX = %d", mwx));
                rep.check(wy, "gpu_label_conservation/coverage_wrapY!=0_" + tag, strf("%d particles", nwy));
                rep.check(wz, "gpu_label_conservation/coverage_wrapZ!=0_" + tag, strf("%d particles", nwz));
                rep.check(range, "gpu_label_conservation/stored_coordinates_in_[0,1)_" + tag);
            }
        }
    }
}

// ============================================================================
// Case 8: gpu_order_ladder (T2)
// ============================================================================

void case_gpu_order_ladder(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_order_ladder (T2) ===\n");
    const int pairs[3] = {kPairA, kPairH, kPairB};
    const double kap = helix_kappa(), cn = helix_speed();
    for (int pair : pairs) {
        const SplineSet& S = C.get(pair);
        const std::string pn = pair_name(pair);
        double max_et[5], max_ex[5], hmin[5], hmax[5];
        bool ok = true;
        for (int k = 0; k < 5; ++k) {
            const double ds = kDsLadder[k];
            const int n = static_cast<int>(2.0 / ds);
            DevParticles P(256);
            PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
            setup_engine(eng, make_cfg(1.0 / 64, 1e-13), S, P, true);
            const Snap s0 = snap(eng, P);
            for (int i = 0; i < n; ++i)
                eng.step_arclength(ds);
            const Snap s = snap(eng, P);
            const PseudoSymplecticStats st = eng.compute_stats();
            ok = ok && st.n_active == 256 && st.total_fail == 0;
            max_et[k] = 0.0;
            max_ex[k] = 0.0;
            hmin[k] = 1e300;
            hmax[k] = -1e300;
            const double pred = 2.0 * kap * kap * ds * ds / (12.0 * cn);
            for (size_t i = 0; i < s.x.size(); ++i) {
                const double x0[3] = {s0.x[i], s0.y[i], s0.z[i]};
                double xu[3], xe[3];
                snap_unwrapped(s, i, S.dev.L, xu);
                exact_position(pair, x0, s.clock[i], xe);
                const double e_t = s.clock[i] - (xu[0] - x0[0]);
                max_et[k] = std::max(max_et[k], std::fabs(e_t));
                max_ex[k] = std::max(max_ex[k], maxnorm_diff(xu, xe));
                hmin[k] = std::min(hmin[k], e_t / pred);
                hmax[k] = std::max(hmax[k], e_t / pred);
            }
            std::printf("  pair %s ds=%-5s n=%4d  max|e_t|=%.6e  max E_x=%.6e  n_active=%d fail=%llu "
                        "maxNewton=%u%s\n",
                        pn.c_str(), kDsName[k], n, max_et[k], max_ex[k], st.n_active,
                        static_cast<unsigned long long>(st.total_fail), st.max_newton_iter,
                        pair == kPairH ? strf("  helix ratio [%.6f, %.6f]", hmin[k], hmax[k]).c_str() : "");
        }
        rep.check(ok, "gpu_order_ladder/all_active_no_fail_" + pn);
        for (int k = 0; k < 4; ++k) {
            const double ot = order2(max_et[k], max_et[k + 1]);
            const double ox = order2(max_ex[k], max_ex[k + 1]);
            rep.check(ot >= 1.8 && ot <= 2.2,
                      "gpu_order_ladder/order_e_t_" + pn + "_" + kDsName[k] + "->" + kDsName[k + 1],
                      strf("%.4f (gate [1.8, 2.2])", ot));
            rep.check(ox >= 1.8 && ox <= 2.2,
                      "gpu_order_ladder/order_E_x_" + pn + "_" + kDsName[k] + "->" + kDsName[k + 1],
                      strf("%.4f (gate [1.8, 2.2])", ox));
        }
        rep.check(2.0 * S.e_interp <= 0.02 * max_ex[4], "gpu_order_ladder/regime_" + pn,
                  strf("2 E_interp = %.3e <= 0.02 x max_p E_x(1/256) = %.3e", 2.0 * S.e_interp, 0.02 * max_ex[4]));
        if (pair == kPairH) {
            for (int k = 2; k < 5; ++k)
                rep.check(hmin[k] >= 0.99 && hmax[k] <= 1.01, std::string("gpu_order_ladder/helix_ratio_ds_") + kDsName[k],
                          strf("[%.6f, %.6f] (gate [0.99, 1.01])", hmin[k], hmax[k]));
        }
    }
}

// ============================================================================
// Case 9: gpu_uniform_exact (T3)
// ============================================================================

void case_gpu_uniform_exact(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_uniform_exact (T3) ===\n");
    const SplineSet& S = C.get(kPairU);
    const std::vector<P3> dyadic = {{{0.984375, 0.015625, 0.5}}, {{0.125, 0.375, 0.625}},
                                    {{0.0, 0.5, 0.25}},          {{0.5, 0.0, 0.75}},
                                    {{0.25, 0.984375, 0.125}},   {{0.75, 0.25, 0.0}},
                                    {{0.015625, 0.125, 0.875}},  {{0.875, 0.625, 0.375}}};
    const double T = 200.0 / 64.0;
    {
        DevParticles P(8);
        P.write(dyadic);
        PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
        setup_engine(eng, make_cfg(1.0 / 64, 1e-13), S, P, false);
        for (int i = 0; i < 200; ++i)
            eng.step_arclength(1.0 / 64);
        const Snap s = snap(eng, P);
        bool bt = true, bx = true, byz = true, bw = true, act = all_status(s, kStatusActive);
        for (size_t i = 0; i < 8; ++i) {
            double xu[3];
            snap_unwrapped(s, i, S.dev.L, xu);
            bt = bt && same_bits(s.clock[i], T);
            bx = bx && same_bits(xu[0], dyadic[i].v[0] + T);
            byz = byz && same_bits(s.y[i], dyadic[i].v[1]) && same_bits(s.z[i], dyadic[i].v[2]) &&
                  s.wy[i] == 0 && s.wz[i] == 0;
            bw = bw && s.wx[i] == static_cast<int32_t>(std::floor(dyadic[i].v[0] + T));
            if (i < 2)
                std::printf("  i: particle %zu: t = %.17g x_u = %.17g wrapX = %d\n", i, s.clock[i], xu[0], s.wx[i]);
        }
        rep.check(act, "gpu_uniform_exact/i_all_active");
        rep.check(bt, "gpu_uniform_exact/i_clock_bitwise", strf("expect %.17g", T));
        rep.check(bx, "gpu_uniform_exact/i_x_unwrapped_bitwise");
        rep.check(byz, "gpu_uniform_exact/i_y_z_bitwise_unchanged");
        rep.check(bw, "gpu_uniform_exact/i_wrapX_expected");
    }
    {
        DevParticles P(8);
        P.write(dyadic);
        PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
        setup_engine(eng, make_cfg(1.0 / 64, 1e-13), S, P, false);
        for (int i = 0; i < 100; ++i)
            eng.step(1.0 / 32);
        const Snap s = snap(eng, P);
        const double T2 = 100.0 / 32.0;
        bool bt = same_bits(eng.target_time(), T2), bx = true, act = all_status(s, kStatusActive);
        for (size_t i = 0; i < 8; ++i) {
            double xu[3];
            snap_unwrapped(s, i, S.dev.L, xu);
            bt = bt && same_bits(s.clock[i], eng.target_time());
            bx = bx && same_bits(xu[0], dyadic[i].v[0] + T2);
        }
        std::printf("  ii: target_time = %.17g, clock[0] = %.17g\n", eng.target_time(), s.clock[0]);
        rep.check(act, "gpu_uniform_exact/ii_all_active");
        rep.check(bt, "gpu_uniform_exact/ii_clock_eq_target_time_bitwise", strf("expect %.17g", T2));
        rep.check(bx, "gpu_uniform_exact/ii_x_unwrapped_bitwise");
    }
    {
        const std::vector<P3> nd = {{{0.3, 0.7, 0.1}},   {{0.1, 0.2, 0.3}},   {{0.35, 0.6, 0.85}},
                                    {{0.7, 0.15, 0.5}},  {{0.9, 0.8, 0.05}},  {{0.33, 0.66, 0.99}},
                                    {{0.07, 0.41, 0.73}}, {{0.61, 0.29, 0.17}}};
        DevParticles P(8);
        P.write(nd);
        PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
        setup_engine(eng, make_cfg(1.0 / 64, 1e-13), S, P, false);
        for (int i = 0; i < 250; ++i)
            eng.step_arclength(0.01);
        const Snap s = snap(eng, P);
        double rt = 0.0, rx = 0.0;
        for (size_t i = 0; i < 8; ++i) {
            double xu[3];
            snap_unwrapped(s, i, S.dev.L, xu);
            rt = std::max(rt, std::fabs(s.clock[i] - 2.5) / 2.5);
            rx = std::max(rx, std::fabs((xu[0] - nd[i].v[0]) - 2.5) / 2.5);
        }
        rep.check(all_status(s, kStatusActive), "gpu_uniform_exact/iii_all_active");
        rep.check(rt <= 1e-13, "gpu_uniform_exact/iii_clock_relative", strf("%.3e (gate 1e-13)", rt));
        rep.check(rx <= 1e-13, "gpu_uniform_exact/iii_x_displacement_relative", strf("%.3e (gate 1e-13)", rx));
    }
}

// ============================================================================
// Case 10: gpu_engine_mode
// ============================================================================

void case_gpu_engine_mode(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_engine_mode ===\n");
    const SplineSet& S = C.get(kPairB);
    const double tol = 1e-13, ds_max = 1.0 / 64;
    DevParticles P(1024);
    PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
    setup_engine(eng, make_cfg(ds_max, tol), S, P, true);
    const Snap s0 = snap(eng, P);
    const int marks[3] = {10, 100, 200};
    double mm[3] = {0.0, 0.0, 0.0};
    int m = 0;
    const double bound = 2.5 * ds_max * ds_max;
    for (int call = 1; call <= 200; ++call) {
        eng.step(0.01);
        if (call == marks[m]) {
            const Snap s = snap(eng, P);
            double mis = 0.0, ex = 0.0;
            for (size_t i = 0; i < s.x.size(); ++i) {
                mis = std::max(mis, std::fabs(s.clock[i] - eng.target_time()));
                const double x0[3] = {s0.x[i], s0.y[i], s0.z[i]};
                double xu[3], xe[3];
                snap_unwrapped(s, i, S.dev.L, xu);
                exact_position(kPairB, x0, s.clock[i], xe);
                ex = std::max(ex, maxnorm_diff(xu, xe));
            }
            const double res = snap_residual(s, S);
            mm[m] = mis;
            std::printf("  after call %3d: target %.4f, max|t_p - target| = %.4e (bound %.4e), residual %.3e, "
                        "max_p E_x = %.6e\n",
                        call, eng.target_time(), mis, bound, res, ex);
            rep.check(mis <= bound, strf("gpu_engine_mode/mismatch_bound_call%d", call),
                      strf("%.4e (gate %.4e)", mis, bound));
            rep.check(res <= tol + 1e-14, strf("gpu_engine_mode/label_residual_call%d", call),
                      strf("%.3e (gate %.3e)", res, tol + 1e-14));
            rep.check(all_status(s, kStatusActive), strf("gpu_engine_mode/all_active_call%d", call));
            ++m;
            if (m == 3)
                break;
        }
    }
    const double ref = std::max(mm[0], mm[1]);
    rep.check(mm[2] <= 1.5 * ref + 1e-12, "gpu_engine_mode/mismatch_not_growing",
              strf("call 200: %.4e <= 1.5 x %.4e + 1e-12", mm[2], ref));
}

// ============================================================================
// Case 11: gpu_host_core_agreement
// ============================================================================

void case_gpu_host_core_agreement(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_host_core_agreement ===\n");
    const SplineSet& S = C.get(kPairB);
    const double tol = 1e-13;
    DevParticles P(64);
    PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
    setup_engine(eng, make_cfg(1.0 / 64, tol), S, P, true);
    const Snap s0 = snap(eng, P);
    for (int i = 0; i < 64; ++i)
        eng.step_arclength(1.0 / 64);
    const Snap s = snap(eng, P);
    const PseudoSymplecticParams prm = make_prm(tol);
    double dx = 0.0, dt = 0.0;
    bool ok = all_status(s, kStatusActive);
    for (size_t i = 0; i < 64; ++i) {
        const double x0[3] = {s0.x[i], s0.y[i], s0.z[i]};
        PanelState st;
        real p1, p2;
        bool hok = core_init(S.host, x0, prm, st, p1, p2) == kStatusActive;
        ProjectionCounters cnt{0u, 0u};
        for (int k = 0; k < 64 && hok; ++k)
            hok = advance_panel(S.host, st, p1, p2, 1.0 / 64, prm, cnt) == kStatusActive;
        ok = ok && hok;
        double xh[3], xg[3];
        unwrapped(st.xi, st.w, S.host.L, xh);
        snap_unwrapped(s, i, S.dev.L, xg);
        dx = std::max(dx, maxnorm_diff(xh, xg));
        dt = std::max(dt, std::fabs(st.t - s.clock[i]));
    }
    std::printf("  max |x_u GPU - x_u host| = %.3e, max |t GPU - t host| = %.3e\n", dx, dt);
    rep.check(ok, "gpu_host_core_agreement/all_active");
    rep.check(dx <= 1e-11, "gpu_host_core_agreement/positions", strf("%.3e (gate 1e-11)", dx));
    rep.check(dt <= 1e-11, "gpu_host_core_agreement/clocks", strf("%.3e (gate 1e-11)", dt));
}

// ============================================================================
// Case 12: gpu_no_allocation (T5)
// ============================================================================

size_t free_bytes() {
    size_t f = 0, t = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&f, &t));
    return f;
}

void case_gpu_no_allocation(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_no_allocation (T5) ===\n");
    const SplineSet& S = C.get(kPairB);
    DevParticles P(1024);
    PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
    setup_engine(eng, make_cfg(1.0 / 64, 1e-13), S, P, true);
    eng.step_arclength(1.0 / 64);
    eng.step(0.01);
    eng.synchronize();
    const size_t f0 = free_bytes();
    for (int i = 0; i < 100; ++i)
        eng.step_arclength(1.0 / 64);
    for (int i = 0; i < 100; ++i)
        eng.step(0.01);
    eng.synchronize();
    const size_t f1 = free_bytes();
    std::printf("  free before = %zu, after = %zu, delta = %lld\n", f0, f1,
                static_cast<long long>(f0) - static_cast<long long>(f1));
    rep.check(f0 == f1, "gpu_no_allocation/cudaMemGetInfo_unchanged", "100 step_arclength + 100 step");
}

// ============================================================================
// Case 13: gpu_determinism (T6)
// ============================================================================

template <class T> bool vec_eq(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

void case_gpu_determinism(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_determinism (T6) ===\n");
    const SplineSet& S = C.get(kPairG);
    Snap r[2];
    for (int run = 0; run < 2; ++run) {
        DevParticles P(1024);
        PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
        setup_engine(eng, make_cfg(1.0 / 64, 1e-12), S, P, true);
        for (int i = 0; i < 64; ++i)
            eng.step_arclength(1.0 / 64);
        for (int i = 0; i < 20; ++i)
            eng.step(0.01);
        r[run] = snap(eng, P);
        const PseudoSymplecticStats st = eng.compute_stats();
        std::printf("  run %d: n_active %d, target %.4f, clock [%.6f, %.6f], max Newton %u, clamps %llu\n", run + 1,
                    st.n_active, eng.target_time(), st.min_clock, st.max_clock, st.max_newton_iter,
                    static_cast<unsigned long long>(st.total_clamps));
    }
    const Snap& a = r[0];
    const Snap& b = r[1];
    rep.check(vec_eq(a.x, b.x) && vec_eq(a.y, b.y) && vec_eq(a.z, b.z), "gpu_determinism/positions_memcmp");
    rep.check(vec_eq(a.wx, b.wx) && vec_eq(a.wy, b.wy) && vec_eq(a.wz, b.wz), "gpu_determinism/wraps_memcmp");
    rep.check(vec_eq(a.st, b.st), "gpu_determinism/status_memcmp");
    rep.check(vec_eq(a.clock, b.clock), "gpu_determinism/clocks_memcmp");
    rep.check(vec_eq(a.p1, b.p1) && vec_eq(a.p2, b.p2), "gpu_determinism/psi_targets_memcmp");
    rep.check(vec_eq(a.fail, b.fail) && vec_eq(a.clamp, b.clamp) && vec_eq(a.nmax, b.nmax),
              "gpu_determinism/counters_memcmp");
    {
        DevParticles P1(1024), P2(1024);
        PseudoSymplecticTracker e1(ctx.cuda_stream(), kSeed), e2(ctx.cuda_stream(), 54321ULL);
        setup_engine(e1, make_cfg(1.0 / 64, 1e-12), S, P1, true);
        setup_engine(e2, make_cfg(1.0 / 64, 1e-12), S, P2, true);
        const Snap s1 = snap(e1, P1);
        const Snap s2 = snap(e2, P2);
        int ndiff = 0;
        for (size_t i = 0; i < s1.x.size(); ++i)
            ndiff += !(same_bits(s1.x[i], s2.x[i]) && same_bits(s1.y[i], s2.y[i]) && same_bits(s1.z[i], s2.z[i]));
        rep.check(ndiff > 0, "gpu_determinism/seed54321_different_start_positions",
                  strf("%d of 1024 start positions differ", ndiff));
    }
}

// ============================================================================
// Case 14: gpu_failure_paths
// ============================================================================

bool snap_all_finite(const Snap& s) {
    for (size_t i = 0; i < s.x.size(); ++i) {
        if (!std::isfinite(s.x[i]) || !std::isfinite(s.y[i]) || !std::isfinite(s.z[i]) ||
            !std::isfinite(s.clock[i]) || !std::isfinite(s.p1[i]) || !std::isfinite(s.p2[i]))
            return false;
    }
    return true;
}

bool same_positions_wraps(const Snap& a, const Snap& b) {
    return vec_eq(a.x, b.x) && vec_eq(a.y, b.y) && vec_eq(a.z, b.z) && vec_eq(a.wx, b.wx) &&
           vec_eq(a.wy, b.wy) && vec_eq(a.wz, b.wz);
}

void case_gpu_failure_paths(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== gpu_failure_paths ===\n");
    {
        const SplineSet& S = C.get(kPairD);
        DevParticles P(256);
        PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
        setup_engine(eng, make_cfg(1.0 / 64, 1e-12), S, P, true);
        const Snap s0 = snap(eng, P);
        eng.step_arclength(1.0 / 64);
        const Snap s1 = snap(eng, P);
        const PseudoSymplecticStats st = eng.compute_stats();
        bool f1 = true;
        for (uint32_t f : s1.fail)
            f1 = f1 && f == 1u;
        eng.step_arclength(1.0 / 64);
        const Snap s2 = snap(eng, P);
        bool f2 = true;
        for (uint32_t f : s2.fail)
            f2 = f2 && f == 1u;
        std::printf("  i D: n_degenerate = %d, n_active = %d, n_nonfinite = %d, total_fail = %llu\n", st.n_degenerate,
                    st.n_active, st.n_nonfinite, static_cast<unsigned long long>(st.total_fail));
        rep.check(all_status(s1, kStatusDegenerate), "gpu_failure_paths/i_D_status_degenerate");
        rep.check(same_positions_wraps(s0, s1), "gpu_failure_paths/i_D_positions_wraps_bitwise_unchanged");
        rep.check(f1, "gpu_failure_paths/i_D_fail_count_1");
        rep.check(st.n_degenerate == 256, "gpu_failure_paths/i_D_stats_n_degenerate_256",
                  strf("%d", st.n_degenerate));
        rep.check(snap_all_finite(s1) && snap_all_finite(s2), "gpu_failure_paths/i_D_no_nan_inf");
        rep.check(f2 && same_positions_wraps(s1, s2) && vec_eq(s1.clock, s2.clock) && vec_eq(s1.st, s2.st),
                  "gpu_failure_paths/i_D_second_call_changes_nothing");
    }
    {
        const SplineSet& S = C.get(kPairH);
        DevParticles P(256);
        PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
        PseudoSymplecticConfig cfg = make_cfg(1.0 / 64, 1e-12);
        cfg.max_newton_iter = 1;
        setup_engine(eng, cfg, S, P, true);
        const Snap s0 = snap(eng, P);
        eng.step_arclength(0.5);
        const Snap s1 = snap(eng, P);
        const double res = snap_residual(s1, S);
        const PseudoSymplecticStats st = eng.compute_stats();
        std::printf("  ii H: n_newton_failed = %d, n_active = %d, residual at stored positions = %.3e\n",
                    st.n_newton_failed, st.n_active, res);
        rep.check(all_status(s1, kStatusNewtonFailed), "gpu_failure_paths/ii_H_status_newton_failed");
        rep.check(same_positions_wraps(s0, s1) && vec_eq(s0.clock, s1.clock),
                  "gpu_failure_paths/ii_H_positions_wraps_clocks_bitwise_unchanged");
        rep.check(res <= 1e-12 + 1e-14, "gpu_failure_paths/ii_H_label_residual",
                  strf("%.3e (gate %.3e)", res, 1e-12 + 1e-14));
    }
    {
        const SplineSet& S = C.get(kPairB);
        DevParticles P(256);
        PseudoSymplecticTracker eng(ctx.cuda_stream(), kSeed);
        PseudoSymplecticConfig cfg = make_cfg(1.0 / 64, 1e-12);
        cfg.max_panels_per_step = 2;
        setup_engine(eng, cfg, S, P, true);
        eng.step(1.0);
        const Snap s1 = snap(eng, P);
        const double res = snap_residual(s1, S);
        bool tin = true;
        for (double t : s1.clock)
            tin = tin && t > 0.0 && t < 1.0;
        const PseudoSymplecticStats st = eng.compute_stats();
        std::printf("  iii B: n_substep_limit = %d, clock [%.6f, %.6f], residual = %.3e\n", st.n_substep_limit,
                    st.min_clock, st.max_clock, res);
        rep.check(all_status(s1, kStatusSubstepLimit), "gpu_failure_paths/iii_B_status_substep_limit");
        rep.check(tin, "gpu_failure_paths/iii_B_clocks_in_(0,1)");
        rep.check(res <= 1e-12 + 1e-14, "gpu_failure_paths/iii_B_label_residual",
                  strf("%.3e (gate %.3e)", res, 1e-12 + 1e-14));
    }
}

// ============================================================================
// Case 15: validation_errors
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

void case_validation_errors(const CudaContext& ctx, const SplineCache& C, TestReport& rep) {
    std::printf("\n=== validation_errors ===\n");
    const SplineSet& S = C.get(kPairU);
    const cudaStream_t sm = ctx.cuda_stream();
    struct Bad {
        const char* name;
        PseudoSymplecticConfig cfg;
    };
    std::vector<Bad> bad;
    {
        PseudoSymplecticConfig c = make_cfg(1.0 / 64, 1e-12);
        c.ds_max = 0.0;
        bad.push_back({"ds_max_0", c});
        c.ds_max = -1.0;
        bad.push_back({"ds_max_negative", c});
        c.ds_max = kNaN;
        bad.push_back({"ds_max_NaN", c});
    }
    {
        PseudoSymplecticConfig c = make_cfg(1.0 / 64, 1e-12);
        c.tol_psi = 0.0;
        bad.push_back({"tol_psi_0", c});
        c.tol_psi = kNaN;
        bad.push_back({"tol_psi_NaN", c});
    }
    {
        PseudoSymplecticConfig c = make_cfg(1.0 / 64, 1e-12);
        c.max_newton_iter = 0;
        bad.push_back({"max_newton_iter_0", c});
    }
    {
        PseudoSymplecticConfig c = make_cfg(1.0 / 64, 1e-12);
        c.trust_factor = 0.0;
        bad.push_back({"trust_factor_0", c});
    }
    {
        PseudoSymplecticConfig c = make_cfg(1.0 / 64, 1e-12);
        c.min_cross_norm = -1.0;
        bad.push_back({"min_cross_norm_negative", c});
    }
    {
        PseudoSymplecticConfig c = make_cfg(1.0 / 64, 1e-12);
        c.max_panels_per_step = 0;
        bad.push_back({"max_panels_per_step_0", c});
    }
    {
        PseudoSymplecticConfig c = make_cfg(1.0 / 64, 1e-12);
        c.min_cross_sin2 = -1.0;
        bad.push_back({"min_cross_sin2_negative", c});
        c.min_cross_sin2 = kNaN;
        bad.push_back({"min_cross_sin2_NaN", c});
    }
    for (const Bad& b : bad) {
        PseudoSymplecticTracker e(sm, kSeed);
        rep.check(throws_as<std::invalid_argument>([&] { e.configure(b.cfg); }),
                  std::string("validation_errors/configure_rejects_") + b.name);
    }
    {
        PseudoSymplecticTracker e(sm, kSeed);
        PseudoSymplecticConfig c = make_cfg(1.0 / 64, 1e-12);
        c.min_cross_sin2 = 0.0;
        bool accepted = true;
        try {
            e.configure(c);
        } catch (...) {
            accepted = false;
        }
        rep.check(accepted, "validation_errors/configure_accepts_min_cross_sin2_0");
    }
    DevParticles P(8);
    {
        PseudoSymplecticTracker e(sm, kSeed);
        e.bind_labels(S.dev);
        e.bind_particles(P.soa);
        rep.check(throws_as<std::logic_error>([&] { e.prepare(); }), "validation_errors/prepare_before_configure");
    }
    {
        PseudoSymplecticTracker e(sm, kSeed);
        e.configure(make_cfg(1.0 / 64, 1e-12));
        e.bind_particles(P.soa);
        rep.check(throws_as<std::logic_error>([&] { e.prepare(); }), "validation_errors/prepare_before_bind_labels");
    }
    {
        PseudoSymplecticTracker e(sm, kSeed);
        e.configure(make_cfg(1.0 / 64, 1e-12));
        e.bind_labels(S.dev);
        rep.check(throws_as<std::logic_error>([&] { e.prepare(); }), "validation_errors/prepare_before_bind_particles");
    }
    {
        PseudoSymplecticTracker e(sm, kSeed);
        e.configure(make_cfg(1.0 / 64, 1e-12));
        e.bind_labels(S.dev);
        e.bind_particles(P.soa);
        e.inject_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0, P.n);
        e.ensure_tracking();
        rep.check(throws_as<std::logic_error>([&] { e.step(0.01); }), "validation_errors/step_before_prepare");
        rep.check(throws_as<std::logic_error>([&] { e.step_arclength(1.0 / 64); }),
                  "validation_errors/step_arclength_before_prepare");
        e.prepare();
        rep.check(throws_as<std::invalid_argument>([&] { e.step(-1.0); }), "validation_errors/step_negative");
        rep.check(throws_as<std::invalid_argument>([&] { e.step(kNaN); }), "validation_errors/step_NaN");
        rep.check(throws_as<std::invalid_argument>([&] { e.step_arclength(0.0); }),
                  "validation_errors/step_arclength_0");
        e.synchronize();
    }
    {
        // bind_particles documents that wraps are checked by inject_box /
        // ensure_tracking / prepare, so the pair bind_particles + ensure_tracking
        // must throw; whether bind_particles alone throws is reported.
        ParticlesSoA<real> pw = P.soa;
        pw.wrapX = nullptr;
        PseudoSymplecticTracker e(sm, kSeed);
        e.configure(make_cfg(1.0 / 64, 1e-12));
        e.bind_labels(S.dev);
        bool bind_threw = false;
        try {
            e.bind_particles(pw);
        } catch (const std::exception&) {
            bind_threw = true;
        }
        bool ensure_threw = bind_threw;
        if (!bind_threw)
            ensure_threw = throws_as<std::invalid_argument>([&] { e.ensure_tracking(); });
        std::printf("  null wrapX: bind_particles threw = %s, ensure_tracking threw = %s\n", bind_threw ? "yes" : "no",
                    ensure_threw ? "yes" : "no");
        rep.check(ensure_threw, "validation_errors/bind_particles_ensure_tracking_null_wrap_throws");
    }
    {
        const SplineSet& G = C.get(kPairG);
        real g1[3], g2[3];
        pair_gbar(kPairU, g1, g2);
        rep.check(throws_as<std::invalid_argument>(
                      [&] { make_spline_label_pair(S.ws1.view(), G.ws2.view(), g1, g2); }),
                  "validation_errors/make_spline_label_pair_different_grids",
                  "16^3 vs 32^3");
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
        timed("core_analytic_selftest", [&] { case_core_analytic_selftest(rep); });
        timed("core_order_ladder+core_helix_constant", [&] { case_core_order_ladder(rep); });
        timed("core_uniform_exact", [&] { case_core_uniform_exact(rep); });
        timed("core_label_conservation", [&] { case_core_label_conservation(rep); });
        timed("core_advance_to_time", [&] { case_core_advance_to_time(rep); });
        timed("core_failure_paths", [&] { case_core_failure_paths(rep); });
        timed("core_degeneracy_threshold", [&] { case_core_degeneracy_threshold(rep); });

        CudaContext ctx(0);
        SplineCache C;
        timed("spline_setup", [&] {
            std::printf("\n=== spline setup (SF-28 GPU prefilter) ===\n");
            const Grid3D gABH(256, 256, 4, 1.0 / 256, 1.0 / 256, 1.0 / 4);
            const Grid3D g32(32, 32, 32, 1.0 / 32, 1.0 / 32, 1.0 / 32);
            const Grid3D g16(16, 16, 16, 1.0 / 16, 1.0 / 16, 1.0 / 16);
            C.A = build_spline(ctx, kPairA, gABH);
            C.H = build_spline(ctx, kPairH, gABH);
            C.B = build_spline(ctx, kPairB, gABH);
            C.G = build_spline(ctx, kPairG, g32);
            C.D = build_spline(ctx, kPairD, g32);
            C.U = build_spline(ctx, kPairU, g16);
        });
        if (C.A && C.H && C.B && C.G && C.D && C.U) {
            timed("gpu_label_conservation", [&] { case_gpu_label_conservation(ctx, C, rep); });
            timed("gpu_order_ladder", [&] { case_gpu_order_ladder(ctx, C, rep); });
            timed("gpu_uniform_exact", [&] { case_gpu_uniform_exact(ctx, C, rep); });
            timed("gpu_engine_mode", [&] { case_gpu_engine_mode(ctx, C, rep); });
            timed("gpu_host_core_agreement", [&] { case_gpu_host_core_agreement(ctx, C, rep); });
            timed("gpu_no_allocation", [&] { case_gpu_no_allocation(ctx, C, rep); });
            timed("gpu_determinism", [&] { case_gpu_determinism(ctx, C, rep); });
            timed("gpu_failure_paths", [&] { case_gpu_failure_paths(ctx, C, rep); });
            timed("validation_errors", [&] { case_validation_errors(ctx, C, rep); });
        } else {
            rep.check(false, "spline_setup/complete");
        }
    } catch (const std::exception& e) {
        rep.check(false, "unexpected_exception", e.what());
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== timing_record (information only) ===\n");
    for (const auto& t : times)
        std::printf("  %-40s %8.3f s\n", t.first.c_str(), t.second);
    std::printf("  %-40s %8.3f s\n", "whole executable", wall);
    std::printf("\n%d checks, %d failed, overall %s\n", rep.checks, rep.fails, rep.overall_pass ? "PASS" : "FAIL");
    return rep.overall_pass ? 0 : 1;
}
