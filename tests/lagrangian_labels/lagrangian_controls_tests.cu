/**
 * @file lagrangian_controls_tests.cu
 * @brief SF-36 N6: exact-control tests of the forward Lagrangian-label tracker
 *        (`lagrangian_tracker_controls`). Independent of the tracker author.
 *
 * Thresholds A1, A2, A3, B, D, E are those of the SF-36 specification ("Acceptance thresholds",
 * pre-registered verbatim at activation; nothing is tuned here):
 *
 *  A1  k = 1: on every plane and snapshot |x2 - x2_0|, |x3 - x3_0| <= 1e-12; tau at plane X equals
 *      X - X0 to 1e-12 relative; covectors equal their initial values to 1e-12; |I| <= 1e-14.
 *  A2  layered k = exp(0.5 cos 2 pi x1) (SlabFieldSource::spectral): transverse positions and
 *      covectors constant to 1e-12; tau and I at each plane against an independent adaptive
 *      Gauss-Kronrod quadrature (<= 1e-13) of 1/v_s1 and div v_s / v_s1 on the same spline,
 *      <= 1e-9 relative.
 *  A3  synthetic failure field (device functor; the SF-35 A4 bands adapted to the FORWARD
 *      direction): per-particle status / n_reached and the pass histograms and counters exactly as
 *      constructed (landings_discarded of the landing_failed band as a lower bound: the number of
 *      discards of the Zeno-like retry loop is not derivable).
 *  B   lester2021 (eps = 1): one period (plane X0 + 1) returns every particle to its seed within
 *      10 tol at tol = 1e-8, all statuses ok.
 *  D   covectors vs central finite differences of the time-tau flow map: observed order >= 1.8 of
 *      the RMS defect |a(tau)^T DPhi_FD - a(0)^T| (and b) on both delta pairs of {8, 4, 2} x 1e-3.
 *  E   Lagrangian identity |a x b - v_s exp(-I)| / |v_s| <= 1e-8 at every plane / snapshot up to
 *      x1 = 4 for every ok particle, tol = 1e-10: generic3d eps 0.25 and gauss eps 1.0 are
 *      blocking; gauss eps 2.0 is recorded with covector-norm percentiles.
 *
 * Diagnostics (recorded, no thresholds; they separate an implementation defect from a regularity
 * limit of the C^2 spline, whose Hessian is only C^0):
 *  DIAG-1  C-infinity analytic field with exact derivatives: identity residual of E at tol 1e-10 /
 *          1e-12, h_max = 1/128.
 *  DIAG-2  identity residual vs h_max in {h/8, h/16, h/32, h/64} (tol 1e-12) on the spline fields
 *          of E (+ gauss 2.0) and on the C-infinity field, with observed orders.
 *  DIAG-3  accuracy of I (h/8 vs h/64) next to the RMS of I and of e_v = 1 - exp(-I).
 *
 * Usage: lagrangian_controls_tests [--n N] [--full] [--only A1|A2|A3|B|D|E|DIAG] [--diag]
 *   default: fast tier (16^3 / 32^3, <= 60 s); --full: 64^3 repeats of A, B, D, E with the full D
 *   settings (256 seeds); --only restricts to one case; --diag adds DIAG to an --only selection.
 *
 * This TU instantiates the tracker's device templates: compiled with --fmad=false.
 */

#include "apps/closure_gate/streamline_integrator.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "src/physics/particles/lagrangian_labels/LagrangianTracker.cuh"
#include "src/physics/particles/lagrangian_labels/LagrangianTrackerImpl.cuh"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#include <set>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;
namespace ll = macroflow3d::lagrangian_labels;
namespace cg = macroflow3d::closure_gate;

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
        std::fflush(stdout);
        overall_pass = overall_pass && cond;
    }
};

std::string fmt(const char* f, double v) {
    char b[200];
    std::snprintf(b, sizeof(b), f, v);
    return b;
}
std::string fmt2(const char* f, double v, double w) {
    char b[240];
    std::snprintf(b, sizeof(b), f, v, w);
    return b;
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Config {
    int n = 0;        // 0: per-case default
    bool full = false;
    bool diag_flag = false;
    std::set<std::string> only;
    bool run(const std::string& c) const {
        if (only.empty())
            return true;
        if (only.count(c))
            return true;
        return c == "DIAG" && diag_flag;
    }
};

using V3 = std::array<double, 3>;
using VFun = std::function<void(const double x[3], double v[3])>;

VFun spline_velocity(const sl::SlabSplineDirectionField& f) {
    return [f](const double x[3], double v[3]) {
        double g[3], k;
        f(x, g, k);
        v[0] = k * g[0];
        v[1] = k * g[1];
        v[2] = k * g[2];
    };
}

/// Initial covectors a = (-v2, v1, 0), b = (-v3/v1, 0, 1) at x.
void initial_covectors(const VFun& vf, const double x[3], double a[3], double b[3]) {
    double v[3];
    vf(x, v);
    a[0] = -v[1];
    a[1] = v[0];
    a[2] = 0.0;
    b[0] = -v[2] / v[0];
    b[1] = 0.0;
    b[2] = 1.0;
}

sl::ProductionStage make_stage(CudaContext& ctx, int N, const sl::SlabFieldSource& src, double eps) {
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    sl::ProductionStageOptions so;
    so.require_inlet_labels = false;
    return sl::build_production_stage(ctx, grid, src, eps, so);
}

sl::SlabFieldSource gauss_source(CudaContext& ctx, int N) {
    sl::ProductionFieldSpec spec;
    spec.N = N;
    spec.sigma2 = 1.0;
    spec.ell = 0.25;
    spec.seed = 3001ULL;
    return sl::SlabFieldSource::gaussian(ctx, spec);
}

ll::LagrangianTrackResult run_spline(CudaContext& ctx, const sl::ProductionStage& st,
                                     const ll::InjectionSpec& inj, const ll::TrackOptions& to) {
    ll::LagrangianTracker tr;
    tr.prepare(ctx, st, inj, to);
    return tr.track(ctx); // copy
}

std::vector<double> pct(std::vector<double> v, const std::vector<double>& q) {
    std::vector<double> out;
    if (v.empty())
        return std::vector<double>(q.size(), std::nan(""));
    std::sort(v.begin(), v.end());
    for (double p : q) {
        const std::size_t i = static_cast<std::size_t>(p * static_cast<double>(v.size() - 1) + 0.5);
        out.push_back(v[i]);
    }
    return out;
}

// ================================================================================================
// Record access helpers (planes and snapshots of ok particles)
// ================================================================================================

/// Calls f(kind, index, particle, x[3], I, tau_or_nan, cov[6] or nullptr) for every landed record of
/// every particle whose pass status is ok.
template <class F> void for_each_record(const ll::LagrangianTrackResult& R, F f) {
    const std::size_t Np = static_cast<std::size_t>(R.Np), P = R.planes.size(), T = R.times.size();
    for (std::size_t p = 0; p < Np; ++p) {
        if (R.status_plane[p] == 0) {
            for (std::size_t m = 0; m < P; ++m) {
                const double x[3] = {R.planes[m], R.planes_x23[(p * P + m) * 2],
                                     R.planes_x23[(p * P + m) * 2 + 1]};
                f(0, m, p, x, R.planes_I[p * P + m], R.planes_tau[p * P + m],
                  R.covectors ? &R.planes_cov[(p * P + m) * 6] : nullptr);
            }
        }
        if (R.status_time[p] == 0) {
            for (std::size_t n = 0; n < T; ++n) {
                f(1, n, p, &R.times_x[(p * T + n) * 3], R.times_I[p * T + n], R.times[n],
                  R.covectors ? &R.times_cov[(p * T + n) * 6] : nullptr);
            }
        }
    }
}

// ================================================================================================
// Identity statistics: |a x b - v exp(-I)| / |v|
// ================================================================================================
struct IdStats {
    double max = 0.0, sumsq = 0.0;
    long long n = 0, nonok = 0;
    double max_I = 0.0, sumsq_I = 0.0, sumsq_ev = 0.0;
    std::vector<double> cnorm; // |a|, |b| of every record
    double rms() const { return n > 0 ? std::sqrt(sumsq / static_cast<double>(n)) : std::nan(""); }
    double rms_I() const { return n > 0 ? std::sqrt(sumsq_I / static_cast<double>(n)) : std::nan(""); }
    double rms_ev() const { return n > 0 ? std::sqrt(sumsq_ev / static_cast<double>(n)) : std::nan(""); }
};

IdStats identity_stats(const ll::LagrangianTrackResult& R, const VFun& vf) {
    IdStats s;
    for (int p = 0; p < R.Np; ++p) {
        if (R.status_plane[static_cast<std::size_t>(p)] != 0)
            ++s.nonok;
        if (R.status_time[static_cast<std::size_t>(p)] != 0)
            ++s.nonok;
    }
    for_each_record(R, [&](int, std::size_t, std::size_t, const double* x, double I, double,
                           const double* cov) {
        double v[3];
        vf(x, v);
        const double a[3] = {cov[0], cov[1], cov[2]}, b[3] = {cov[3], cov[4], cov[5]};
        const double c[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                             a[0] * b[1] - a[1] * b[0]};
        const double e = std::exp(-I);
        const double vn = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        const double d0 = c[0] - v[0] * e, d1 = c[1] - v[1] * e, d2 = c[2] - v[2] * e;
        const double r = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2) / vn;
        s.max = std::max(s.max, r);
        s.sumsq += r * r;
        ++s.n;
        s.max_I = std::max(s.max_I, std::abs(I));
        s.sumsq_I += I * I;
        const double ev = 1.0 - e;
        s.sumsq_ev += ev * ev;
        s.cnorm.push_back(std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]));
        s.cnorm.push_back(std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]));
    });
    return s;
}

// ================================================================================================
// A1: k = 1
// ================================================================================================
void case_A1(TestReport& rep, CudaContext& ctx, int N) {
    const std::string tag = "A1 k=1 N=" + std::to_string(N) + ": ";
    const double t0 = now_s();
    const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(
        ctx, "uniform", N, std::vector<real>(static_cast<std::size_t>(N) * N * N, 0.0));
    const sl::ProductionStage st = make_stage(ctx, N, src, 1.0);
    rep.check(st.flow_ready(), tag + "stage flow ready", sl::to_string(st.report.status));
    if (!st.flow_ready())
        return;
    const double h = 1.0 / N;
    const double X0 = 0.0;
    ll::InjectionSpec inj;
    inj.X0 = X0;
    inj.M = 8;
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = h / 8.0;
    to.planes = {0.25, 0.5, 1.0, 2.0, 4.0};
    to.times = {0.25, 0.5, 1.0, 2.0, 4.0};
    to.covectors = true;
    const ll::LagrangianTrackResult R = run_spline(ctx, st, inj, to);
    const VFun vf = spline_velocity(st.direction_field());

    double dx = 0.0, dtau = 0.0, dcov = 0.0, mI = 0.0, dxt = 0.0, dtt = 0.0;
    long long nrec = 0;
    for_each_record(R, [&](int kind, std::size_t idx, std::size_t p, const double* x, double I,
                           double tau, const double* cov) {
        const double* sd = &R.seeds[3 * p];
        dx = std::max({dx, std::abs(x[1] - sd[1]), std::abs(x[2] - sd[2])});
        double a0[3], b0[3];
        initial_covectors(vf, sd, a0, b0);
        for (int i = 0; i < 3; ++i) {
            dcov = std::max(dcov, std::abs(cov[i] - a0[i]));
            dcov = std::max(dcov, std::abs(cov[3 + i] - b0[i]));
        }
        mI = std::max(mI, std::abs(I));
        if (kind == 0) {
            const double X = R.planes[idx];
            dtau = std::max(dtau, std::abs(tau - (X - X0)) / (X - X0));
        } else {
            dxt = std::max(dxt, std::abs(x[0] - (X0 + R.times[idx])));
            (void)tau;
        }
        ++nrec;
    });
    (void)dtt;
    bool all_ok = R.totals_plane.status_hist[0] == R.Np && R.totals_time.status_hist[0] == R.Np;
    rep.check(all_ok, tag + "all particles ok in both passes",
              "plane ok " + std::to_string(R.totals_plane.status_hist[0]) + "/" + std::to_string(R.Np) +
                  ", time ok " + std::to_string(R.totals_time.status_hist[0]));
    rep.check(dx <= 1e-12, tag + "transverse displacement (planes+snapshots) <= 1e-12",
              fmt("max %.3e", dx) + " over " + std::to_string(nrec) + " records");
    rep.check(dtau <= 1e-12, tag + "tau at plane X equals X - X0, relative <= 1e-12",
              fmt("max rel %.3e", dtau));
    rep.check(dcov <= 1e-12, tag + "covectors equal initial values <= 1e-12", fmt("max %.3e", dcov));
    rep.check(mI <= 1e-14, tag + "|I| <= 1e-14", fmt("max %.3e", mI));
    std::printf("  %sINFO max |x1(t) - (X0 + t)| at snapshots %.3e  [%.1f s]\n", tag.c_str(), dxt,
                now_s() - t0);
}

// ================================================================================================
// A2: layered k = exp(0.5 cos 2 pi x1)
// ================================================================================================

// Gauss-Kronrod G7K15 (QUADPACK)
template <class F> double gk15(const F& f, double a, double b, double& err) {
    static const double xgk[8] = {0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                                  0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                                  0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
                                  0.207784955007898467600689403773245, 0.0};
    static const double wgk[8] = {0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                                  0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                                  0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                                  0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
    static const double wg[4] = {0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                                 0.381830050505118944950369775488975, 0.417959183673469387755102040816327};
    const double c = 0.5 * (a + b), hl = 0.5 * (b - a);
    const double fc = f(c);
    double rk = wgk[7] * fc, rg = wg[3] * fc;
    for (int j = 0; j < 7; ++j) {
        const double dx = hl * xgk[j];
        const double s = f(c - dx) + f(c + dx);
        rk += wgk[j] * s;
        if (j % 2 == 1)
            rg += wg[j / 2] * s;
    }
    err = std::abs((rk - rg) * hl);
    return rk * hl;
}

template <class F> double adaptive_gk(const F& f, double a, double b, double atol, int depth) {
    double err;
    const double r = gk15(f, a, b, err);
    if (err <= atol || depth >= 40)
        return r;
    const double m = 0.5 * (a + b);
    return adaptive_gk(f, a, m, 0.5 * atol, depth + 1) + adaptive_gk(f, m, b, 0.5 * atol, depth + 1);
}

/// Integral over [a, b] split at the half-cell lattice (spline piece boundaries lie on it),
/// adaptive GK15 with absolute tolerance 1e-15 per piece (total <= 1e-13 for the cell counts here).
template <class F> double quad_split(const F& f, double a, double b, double h) {
    double total = 0.0;
    double x = a;
    const double step = 0.5 * h;
    long long j = static_cast<long long>(std::floor(a / step)) + 1;
    while (x < b) {
        const double nxt = std::min(b, static_cast<double>(j) * step);
        if (nxt > x)
            total += adaptive_gk(f, x, nxt, 1e-15, 0);
        x = nxt;
        ++j;
    }
    return total;
}

void case_A2(TestReport& rep, CudaContext& ctx, int N) {
    const std::string tag = "A2 layered N=" + std::to_string(N) + ": ";
    const double t0 = now_s();
    std::vector<real> cells(static_cast<std::size_t>(N) * N * N);
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                cells[static_cast<std::size_t>(i) + static_cast<std::size_t>(N) * (j + static_cast<std::size_t>(N) * k)] =
                    std::cos(kTwoPi * (static_cast<double>(i) + 0.5) / N);
    const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(ctx, "layered", N, cells);
    const sl::ProductionStage st = make_stage(ctx, N, src, 0.5);
    rep.check(st.flow_ready(), tag + "stage flow ready", sl::to_string(st.report.status));
    if (!st.flow_ready())
        return;
    const double h = 1.0 / N;
    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = 8;
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = h / 8.0;
    to.planes = {0.25, 0.5, 1.0, 2.0, 4.0};
    to.times = {0.25, 0.5, 1.0, 2.0, 4.0};
    to.covectors = true;
    const ll::LagrangianTrackResult R = run_spline(ctx, st, inj, to);
    const sl::SlabSplineDirectionField fld = st.direction_field();
    const VFun vf = spline_velocity(fld);

    auto v1_at = [&](double x1, double x2, double x3) {
        double v[3];
        const double x[3] = {x1, x2, x3};
        vf(x, v);
        return v[0];
    };
    auto dI_at = [&](double x1, double x2, double x3) {
        real val, gx, gy, gz, H[6];
        interpolation::evaluate_point_hessian(fld.potential, x1, x2, x3, val, gx, gy, gz, H);
        real y, yx, yy, yz;
        interpolation::evaluate_point(fld.log_conductivity, x1, x2, x3, y, yx, yy, yz);
        const double g1 = fld.G[0] + gx, g2 = fld.G[1] + gy, g3 = fld.G[2] + gz;
        return (yx * g1 + yy * g2 + yz * g3 + H[0] + H[1] + H[2]) / g1;
    };

    struct A2Err {
        double dx = 0.0, dcov = 0.0, rel_tau = 0.0, abs_tau = 0.0, rel_tau_t = 0.0;
        double abs_I_planes = 0.0, abs_I_times = 0.0, abs_I = 0.0, rel_I_spec = 0.0, max_Iref = 0.0, abs_I_over_scale = 0.0;
        long long nrec = 0, n_rel = 0, n_small = 0;
    };
    // |I_ref| below this fraction of max |I_ref| over the records counts as "I_ref = 0" (periodic
    // returns of the layered flow: I(x1 integer) = 0 exactly): relative error undefined there.
    auto evaluate = [&](const ll::LagrangianTrackResult& Rr) {
        A2Err e;
        std::vector<double> iref_all, idiff_all;
        std::vector<std::pair<double, double>> tmp; // (I_ref, |dI|) per record
        for_each_record(Rr, [&](int kind, std::size_t idx, std::size_t p, const double* x, double I,
                                double tau, const double* cov) {
            const double* sd = &Rr.seeds[3 * p];
            e.dx = std::max({e.dx, std::abs(x[1] - sd[1]), std::abs(x[2] - sd[2])});
            if (cov != nullptr) {
                double a0[3], b0[3];
                initial_covectors(vf, sd, a0, b0);
                for (int i = 0; i < 3; ++i) {
                    e.dcov = std::max(e.dcov, std::abs(cov[i] - a0[i]));
                    e.dcov = std::max(e.dcov, std::abs(cov[3 + i] - b0[i]));
                }
            }
            const double tau_ref =
                quad_split([&](double s) { return 1.0 / v1_at(s, sd[1], sd[2]); }, sd[0], x[0], h);
            const double I_ref =
                quad_split([&](double s) { return dI_at(s, sd[1], sd[2]); }, sd[0], x[0], h);
            if (kind == 0) {
                e.rel_tau = std::max(e.rel_tau, std::abs(tau - tau_ref) / std::abs(tau_ref));
                e.abs_tau = std::max(e.abs_tau, std::abs(tau - tau_ref));
            } else {
                e.rel_tau_t = std::max(e.rel_tau_t, std::abs(tau_ref - tau) / tau);
            }
            tmp.emplace_back(I_ref, std::abs(I - I_ref));
            if (kind == 0)
                e.abs_I_planes = std::max(e.abs_I_planes, std::abs(I - I_ref));
            else
                e.abs_I_times = std::max(e.abs_I_times, std::abs(I - I_ref));
            (void)idx;
            ++e.nrec;
        });
        for (const auto& t : tmp)
            e.max_Iref = std::max(e.max_Iref, std::abs(t.first));
        for (const auto& t : tmp) {
            e.abs_I = std::max(e.abs_I, t.second);
            if (std::abs(t.first) >= 1e-3 * e.max_Iref) {
                e.rel_I_spec = std::max(e.rel_I_spec, t.second / std::abs(t.first));
                ++e.n_rel;
            } else {
                ++e.n_small;
            }
        }
        e.abs_I_over_scale = e.max_Iref > 0.0 ? e.abs_I / e.max_Iref : std::nan("");
        return e;
    };
    const A2Err E8 = evaluate(R);
    const double dx = E8.dx, dcov = E8.dcov, rel_tau = E8.rel_tau, abs_tau = E8.abs_tau,
                 rel_tau_t = E8.rel_tau_t, abs_I = E8.abs_I, max_Iref = E8.max_Iref;
    double max_g23 = 0.0;
    const long long nrec = E8.nrec;
    // transverse velocity of the spline flow along the lines (roundoff expectation)
    for (std::size_t p = 0; p < static_cast<std::size_t>(R.Np); ++p)
        for (int q = 0; q < 16; ++q) {
            const double x[3] = {0.25 * q, R.seeds[3 * p + 1], R.seeds[3 * p + 2]};
            double g[3], k;
            fld(x, g, k);
            max_g23 = std::max({max_g23, std::abs(g[1]), std::abs(g[2])});
        }
    const bool all_ok = R.totals_plane.status_hist[0] == R.Np && R.totals_time.status_hist[0] == R.Np;
    rep.check(all_ok, tag + "all particles ok in both passes");
    rep.check(dx <= 1e-12, tag + "transverse positions constant <= 1e-12",
              fmt("max %.3e", dx) + fmt(" (max |g2|,|g3| of the spline flow on the lines: %.3e)", max_g23));
    rep.check(dcov <= 1e-12, tag + "covectors constant <= 1e-12", fmt("max %.3e", dcov));
    rep.check(rel_tau <= 1e-9, tag + "tau at planes vs independent quadrature <= 1e-9 relative",
              fmt("max rel %.3e", rel_tau) + fmt(" (abs %.3e)", abs_tau));
    rep.check(E8.n_rel > 0 && E8.rel_I_spec <= 1e-9,
              tag + "I at planes+snapshots vs independent quadrature <= 1e-9 relative (records with |I_ref| >= 1e-3 max|I_ref|)",
              fmt("max rel %.3e", E8.rel_I_spec) + " over " + std::to_string(E8.n_rel) + " records; " +
                  std::to_string(E8.n_small) + " records with I_ref ~ 0 (periodic returns) excluded from the relative form");
    std::printf("  %sINFO I error by pass: planes max abs %.3e, snapshots max abs %.3e\n", tag.c_str(), E8.abs_I_planes, E8.abs_I_times);
    std::printf("  %sINFO I error: max abs %.3e; max |I_ref| %.3e; abs error / max|I_ref| = %.3e (spec: 1e-9 relative)\n",
                tag.c_str(), abs_I, max_Iref, E8.abs_I_over_scale);
    rep.check(rel_tau_t <= 10.0 * to.tol,
              tag + "snapshots (extra, not a spec threshold): quadrature tau(x1 recorded) = t within 10 tol (positions are error-controlled at tol)",
              fmt("max rel %.3e", rel_tau_t) + fmt(" (limit %.1e)", 10.0 * to.tol));
    // Scaling of the passive-I error with h_max (diagnostic, not a threshold): plane records only,
    // references computed once.
    {
        const std::size_t Pn = R.planes.size(), Npz = static_cast<std::size_t>(R.Np);
        std::vector<double> refI(Npz * Pn), refT(Npz * Pn);
        double mx = 0.0;
        for (std::size_t p = 0; p < Npz; ++p)
            for (std::size_t m = 0; m < Pn; ++m) {
                const double* sd = &R.seeds[3 * p];
                refT[p * Pn + m] = quad_split([&](double s) { return 1.0 / v1_at(s, sd[1], sd[2]); }, sd[0], R.planes[m], h);
                refI[p * Pn + m] = quad_split([&](double s) { return dI_at(s, sd[1], sd[2]); }, sd[0], R.planes[m], h);
                mx = std::max(mx, std::abs(refI[p * Pn + m]));
            }
        double prev = 0.0;
        for (const double dv : {8.0, 16.0, 32.0}) {
            ll::TrackOptions t2 = to;
            t2.tol = 1e-12;
            t2.h_max = h / dv;
            t2.covectors = false;
            t2.times.clear();
            const ll::LagrangianTrackResult R2 = run_spline(ctx, st, inj, t2);
            double eI = 0.0, eT = 0.0, eRel = 0.0;
            for (std::size_t p = 0; p < Npz; ++p)
                for (std::size_t m = 0; m < Pn; ++m) {
                    const double d = std::abs(R2.planes_I[p * Pn + m] - refI[p * Pn + m]);
                    eI = std::max(eI, d);
                    eT = std::max(eT, std::abs(R2.planes_tau[p * Pn + m] - refT[p * Pn + m]) / refT[p * Pn + m]);
                    if (std::abs(refI[p * Pn + m]) >= 1e-3 * mx)
                        eRel = std::max(eRel, d / std::abs(refI[p * Pn + m]));
                }
            std::printf("  %sDIAG-4 tol 1e-12 h_max h/%.0f (planes): I max abs err %.3e (rel-form %.3e)%s, tau rel err %.3e\n",
                        tag.c_str(), dv, eI, eRel, prev > 0.0 ? fmt(" order vs previous %.2f", std::log2(prev / eI)).c_str() : "", eT);
            prev = eI;
        }
    }
    std::printf("  %sINFO %lld records  [%.1f s]\n", tag.c_str(), nrec, now_s() - t0);
}

// ================================================================================================
// A3: synthetic failure field (forward)
// ================================================================================================
// M = 16 lattice, x2 = (m2 + 1/2)/16, bands of two m2 rows (32 seeds); x2 is constant along every
// line (g2 = 0). X0 = 0, planes {0.3, 0.55, 1.1}, times {0.3, 0.55, 1.1}, tol 1e-8, h_max = 1/128,
// max_arclength = 5.
//   m2 {0,1,14,15}: g = e1                                  -> ok (both passes)
//   m2 {2,3}  : g1 = -1                                      -> seed_backflow (both)
//   m2 {4,5}  : g = (1e-3, 0, 6)  (|v| = 6)                  -> plane: cap_exceeded (needs s ~ 3e5 > 5)
//                                                               time : s(t) = 6 t: snapshots 0.3 (1.8),
//                                                               0.55 (3.3) reached, cap before 1.1 (6.6)
//   m2 {6,7}  : g1 = NaN                                     -> stagnation (both)
//   m2 {8,9}  : k = -1                                       -> invalid_conductivity (both)
//   m2 {10,11}: g = e1 except g1 = -1 at x1 == 0.55 exactly  -> plane: plane 1 reached, the landing on
//                                                               0.55 is discarded until h < min_step:
//                                                               landing_failed (n_reached 1);
//                                                               time: the exact equality is not hit: ok
//   m2 {12,13}: pocket in x3: g = (1,0,1), g1 = -1e-3 for z = x3 mod 1 in [7.5/16, 11.5/16):
//               seeds m3 in {7..10} are in the pocket -> seed_backflow; the others move at dx1/dx3 = 1
//               and enter a pocket at x1 <= 0.75 < 1.1 in both passes -> ok with one interior backflow
//               encounter flagged per particle per pass.
constexpr double kTrapX = 0.55;
constexpr double kPocketLo = 7.5 / 16.0, kPocketHi = 11.5 / 16.0;
constexpr int kBandOk = 0, kBandSeed = 2, kBandCap = 4, kBandStag = 6, kBandInvk = 8, kBandLand = 10,
              kBandPocket = 12;

struct FailureField {
    __device__ void base(const double x[3], double g[3], double& k) const {
        k = 1.0;
        g[0] = 1.0;
        g[1] = 0.0;
        g[2] = 0.0;
        const int m2 = static_cast<int>(floor(x[1] * 16.0));
        const int band = m2 / 2 * 2;
        if (band == kBandSeed) {
            g[0] = -1.0;
        } else if (band == kBandCap) {
            g[0] = 1e-3;
            g[2] = 6.0;
        } else if (band == kBandStag) {
            g[0] = __longlong_as_double(0x7ff8000000000000LL);
        } else if (band == kBandInvk) {
            k = -1.0;
        } else if (band == kBandLand) {
            if (x[0] == kTrapX)
                g[0] = -1.0;
        } else if (band == kBandPocket) {
            const double z = x[2] - floor(x[2]);
            g[2] = 1.0;
            g[0] = (z >= kPocketLo && z < kPocketHi) ? -1e-3 : 1.0;
        }
    }
    __device__ void eval(const double x[3], double g[3], double& k) const { base(x, g, k); }
    __device__ void eval_ext(const double x[3], double g[3], double& k, double gl[3],
                             double H[6]) const {
        base(x, g, k);
        gl[0] = gl[1] = gl[2] = 0.0;
        for (int i = 0; i < 6; ++i)
            H[i] = 0.0;
    }
};

struct A3Expect {
    int plane_status, plane_reached, time_status, time_reached;
    bool weight_zero;
    bool encounter;
};

A3Expect a3_expected(int m2, int m3) {
    const int band = m2 / 2 * 2;
    const int S_ok = static_cast<int>(cg::StreamlineStatus::ok);
    const int S_sb = static_cast<int>(cg::StreamlineStatus::seed_backflow);
    const int S_cap = static_cast<int>(cg::StreamlineStatus::cap_exceeded);
    const int S_stag = static_cast<int>(cg::StreamlineStatus::stagnation);
    const int S_lf = static_cast<int>(cg::StreamlineStatus::landing_failed);
    const int S_ik = static_cast<int>(cg::StreamlineStatus::invalid_conductivity);
    switch (band) {
    case kBandSeed: return {S_sb, 0, S_sb, 0, true, false};
    case kBandCap: return {S_cap, 0, S_cap, 2, false, false};
    case kBandStag: return {S_stag, 0, S_stag, 0, true, false};
    case kBandInvk: return {S_ik, 0, S_ik, 0, true, false};
    case kBandLand: return {S_lf, 1, S_ok, 3, false, false};
    case kBandPocket: {
        const double z = (m3 + 0.5) / 16.0;
        if (z >= kPocketLo && z < kPocketHi)
            return {S_sb, 0, S_sb, 0, true, false};
        return {S_ok, 3, S_ok, 3, false, true};
    }
    default: return {S_ok, 3, S_ok, 3, false, false};
    }
}

void case_A3(TestReport& rep, CudaContext& ctx) {
    const std::string tag = "A3 failure field M=16: ";
    const double t0 = now_s();
    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = 16;
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = 1.0 / 128.0;
    to.min_step = 1e-13;
    to.max_arclength = 5.0;
    to.planes = {0.3, kTrapX, 1.1};
    to.times = {0.3, 0.55, 1.1};
    ll::LagrangianTrackerT<FailureField> tr;
    tr.prepare(ctx, FailureField{}, inj, to, nullptr);
    const ll::LagrangianTrackResult& R = tr.track(ctx);

    long long bad_plane = 0, bad_time = 0, bad_weight = 0, bad_enc = 0, bad_nr = 0;
    std::array<long long, 7> want_hist_plane{}, want_hist_time{};
    long long want_seed_bf = 0, want_enc = 0, ldisc_land_min = -1, ldisc_other = 0;
    double wsum = 0.0;
    for (int m2 = 0; m2 < 16; ++m2)
        for (int m3 = 0; m3 < 16; ++m3) {
            const std::size_t p = static_cast<std::size_t>(m2) * 16 + static_cast<std::size_t>(m3);
            const A3Expect e = a3_expected(m2, m3);
            ++want_hist_plane[static_cast<std::size_t>(e.plane_status)];
            ++want_hist_time[static_cast<std::size_t>(e.time_status)];
            if (e.weight_zero)
                ++want_seed_bf;
            if (e.encounter)
                ++want_enc;
            const ll::PassStats &sp = R.stats_plane[p], &stt = R.stats_time[p];
            if (sp.status != e.plane_status)
                ++bad_plane;
            if (stt.status != e.time_status)
                ++bad_time;
            if (sp.n_reached != e.plane_reached || stt.n_reached != e.time_reached)
                ++bad_nr;
            if ((R.weights[p] == 0.0) != e.weight_zero)
                ++bad_weight;
            wsum += R.weights[p];
            if (sp.backflow_encounter != (e.encounter ? 1 : 0) ||
                stt.backflow_encounter != (e.encounter ? 1 : 0))
                ++bad_enc;
            if (m2 / 2 * 2 == kBandLand) {
                if (ldisc_land_min < 0 || sp.landings_discarded < ldisc_land_min)
                    ldisc_land_min = sp.landings_discarded;
                ldisc_other += stt.landings_discarded;
            } else {
                ldisc_other += sp.landings_discarded + stt.landings_discarded;
            }
        }
    // headline totals written out from the construction (88 ok = 64 straight + 24 pocket seeds
    // outside the pocket; 40 seed_backflow = 32 + 8 pocket seeds inside it; 32 per remaining band)
    const long long ok_plane = 88, ok_time = 120;
    auto H = [](const std::array<long long, 7>& a) {
        std::string s;
        for (long long v : a)
            s += std::to_string(v) + " ";
        return s;
    };
    const auto& hp = R.totals_plane.status_hist;
    const auto& ht = R.totals_time.status_hist;
    const bool hist_plane_ok = hp == std::array<long long, 7>{ok_plane, 40, 32, 32, 0, 32, 32};
    const bool hist_time_ok = ht == std::array<long long, 7>{ok_time, 40, 32, 32, 0, 0, 32};
    rep.check(hist_plane_ok && hp == want_hist_plane, tag + "plane-pass status histogram exactly as constructed",
              "got " + H(hp) + "| want 88 40 32 32 0 32 32");
    rep.check(hist_time_ok && ht == want_hist_time, tag + "time-pass status histogram exactly as constructed",
              "got " + H(ht) + "| want 120 40 32 32 0 0 32");
    rep.check(bad_plane == 0 && bad_time == 0 && bad_nr == 0,
              tag + "per-particle status and n_reached (both passes) exactly as constructed",
              "mismatching: plane status " + std::to_string(bad_plane) + ", time status " +
                  std::to_string(bad_time) + ", n_reached " + std::to_string(bad_nr));
    rep.check(R.seed_backflow == 104 && R.seed_backflow == want_seed_bf && bad_weight == 0 &&
                  std::abs(wsum - 1.0) <= 1e-13,
              tag + "seed_backflow = 104 (32 g1<0 + 32 NaN + 32 k<0 + 8 pocket), zero weights exactly there, sum w = 1",
              "seed_backflow " + std::to_string(R.seed_backflow) + fmt(", sum w - 1 = %.2e", wsum - 1.0) +
                  ", weight mismatches " + std::to_string(bad_weight));
    rep.check(bad_enc == 0 && R.totals_plane.backflow_encounters == want_enc &&
                  R.totals_time.backflow_encounters == want_enc && want_enc == 24,
              tag + "backflow_encounters = 24 per pass (one per pocket seed outside the pocket, none elsewhere)",
              "plane " + std::to_string(R.totals_plane.backflow_encounters) + ", time " +
                  std::to_string(R.totals_time.backflow_encounters));
    rep.check(R.totals_plane.recrossings == 0 && R.totals_time.recrossings == 0,
              tag + "recrossings = 0", "plane " + std::to_string(R.totals_plane.recrossings));
    rep.check(ldisc_land_min >= 1 && ldisc_other == 0 && R.totals_plane.landings_discarded >= 32,
              tag + "landings_discarded: >= 1 for every landing_failed seed (LOWER BOUND, the retry count is "
                    "not derivable), exactly 0 for every other particle/pass",
              "min over band " + std::to_string(ldisc_land_min) + ", plane total " +
                  std::to_string(R.totals_plane.landings_discarded) + ", elsewhere " +
                  std::to_string(ldisc_other));
    // moments bookkeeping
    const std::array<int, 3> want_contrib = {120, 88, 88};
    bool mom_ok = R.moments_planes.size() == 3;
    for (std::size_t m = 0; mom_ok && m < 3; ++m)
        mom_ok = R.moments_planes[m].contributing == want_contrib[m] &&
                 R.moments_planes[m].non_ok == 168;
    const std::array<int, 3> want_active = {152, 152, 120};
    bool act_ok = R.moments_time.size() == 3;
    for (std::size_t n = 0; act_ok && n < 3; ++n)
        act_ok = R.moments_time[n].active == want_active[n];
    rep.check(mom_ok && act_ok,
              tag + "moments bookkeeping: plane contributing {120,88,88}, non_ok 168; snapshot active {152,152,120}");
    std::printf("  %sINFO %.1f s\n", tag.c_str(), now_s() - t0);
}

// ================================================================================================
// B: lester2021 closed streamlines
// ================================================================================================
void case_B(TestReport& rep, CudaContext& ctx, int N) {
    const std::string tag = "B lester2021 eps=1 N=" + std::to_string(N) + ": ";
    const double t0 = now_s();
    const sl::SlabFieldSource src = sl::SlabFieldSource::analytic("lester2021", N);
    const sl::ProductionStage st = make_stage(ctx, N, src, 1.0);
    rep.check(st.flow_ready(), tag + "stage flow ready", sl::to_string(st.report.status));
    if (!st.flow_ready())
        return;
    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = 16;
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = (1.0 / N) / 8.0;
    to.planes = {1.0};
    const ll::LagrangianTrackResult R = run_spline(ctx, st, inj, to);
    double ret = 0.0;
    for (int p = 0; p < R.Np; ++p) {
        if (R.status_plane[static_cast<std::size_t>(p)] != 0)
            continue;
        ret = std::max({ret, std::abs(R.planes_x23[2 * static_cast<std::size_t>(p)] - R.seeds[3 * p + 1]),
                        std::abs(R.planes_x23[2 * static_cast<std::size_t>(p) + 1] - R.seeds[3 * p + 2])});
    }
    rep.check(R.totals_plane.status_hist[0] == R.Np, tag + "all statuses ok (plane pass)",
              "ok " + std::to_string(R.totals_plane.status_hist[0]) + "/" + std::to_string(R.Np));
    rep.check(ret <= 10.0 * to.tol, tag + "one-period transverse return <= 10 tol = 1e-7",
              fmt("max %.3e", ret) + fmt(" (limit %.1e)", 10.0 * to.tol));
    std::printf("  %sINFO %.1f s\n", tag.c_str(), now_s() - t0);
}

// ================================================================================================
// E: Lagrangian identity on the spline fields
// ================================================================================================
struct SplineCase {
    std::string name;
    double eps;
    bool blocking;
};

void case_E(TestReport& rep, CudaContext& ctx, int N) {
    const double t0 = now_s();
    const sl::SlabFieldSource gsrc = gauss_source(ctx, N);
    const sl::SlabFieldSource asrc = sl::SlabFieldSource::analytic("generic3d", N);
    struct C {
        std::string name;
        const sl::SlabFieldSource* src;
        double eps;
        bool blocking;
    };
    const std::vector<C> cases = {{"generic3d eps 0.25", &asrc, 0.25, true},
                                  {"gauss s2=1 ell=.25 seed=3001 eps 1.0", &gsrc, 1.0, true},
                                  {"gauss s2=1 ell=.25 seed=3001 eps 2.0", &gsrc, 2.0, false}};
    for (const C& c : cases) {
        const std::string tag = "E " + c.name + " N=" + std::to_string(N) + ": ";
        const sl::ProductionStage st = make_stage(ctx, N, *c.src, c.eps);
        rep.check(st.flow_ready(), tag + "stage flow ready", sl::to_string(st.report.status));
        if (!st.flow_ready())
            continue;
        ll::InjectionSpec inj;
        inj.X0 = 0.0;
        inj.M = 16;
        ll::TrackOptions to;
        to.tol = 1e-10;
        to.h_max = (1.0 / N) / 8.0;
        to.planes = {0.5, 1.0, 2.0, 4.0};
        to.times = {0.5, 1.0, 2.0, 4.0};
        to.covectors = true;
        const ll::LagrangianTrackResult R = run_spline(ctx, st, inj, to);
        const IdStats s = identity_stats(R, spline_velocity(st.direction_field()));
        const std::vector<double> q = pct(s.cnorm, {0.5, 0.9, 0.99, 1.0});
        const std::string det = fmt("max %.3e", s.max) + fmt(", rms %.3e", s.rms()) + " over " +
                                std::to_string(s.n) + " records (non-ok passes " +
                                std::to_string(s.nonok) + ")";
        if (c.blocking) {
            rep.check(s.n > 0 && s.max <= 1e-8,
                      tag + "identity |a x b - v exp(-I)|/|v| <= 1e-8 at tol 1e-10, h_max h/8 [BLOCKING]", det);
        } else {
            std::printf("[RECORD] %sidentity |a x b - v exp(-I)|/|v| (not blocking): %s\n", tag.c_str(), det.c_str());
        }
        std::printf("  %sINFO covector norms |a|,|b| percentiles p50 %.3e p90 %.3e p99 %.3e max %.3e;"
                    " max|I| %.3e rms I %.3e\n",
                    tag.c_str(), q[0], q[1], q[2], q[3], s.max_I, s.rms_I());
    }
    std::printf("  E total %.1f s\n", now_s() - t0);
}

// ================================================================================================
// D: covectors vs central finite differences of the time-tau flow map
// ================================================================================================
struct DOut {
    double rms_a = std::nan(""), rms_b = std::nan("");
    int used = 0;
};

/// One FD level: defect of a(tau)^T DPhi_FD(delta) - a(0)^T (and b), RMS over seeds usable in
/// every run, for each snapshot.
std::vector<DOut> d_level(CudaContext& ctx, const sl::ProductionStage& st, const ll::LagrangianTrackResult& base,
                          const VFun& vf, int M, double h, const std::vector<double>& taus, double delta,
                          double tol, double hmax_div) {
    ll::TrackOptions to;
    to.tol = tol;
    to.h_max = h / hmax_div;
    to.times = taus;
    to.covectors = false;
    std::array<std::array<ll::LagrangianTrackResult, 2>, 3> run; // [j][sign 0:+ 1:-]
    for (int j = 0; j < 3; ++j)
        for (int sg = 0; sg < 2; ++sg) {
            const double d = sg == 0 ? delta : -delta;
            ll::InjectionSpec inj;
            inj.M = M;
            inj.X0 = 0.0;
            if (j == 0)
                inj.X0 = d;
            if (j == 1) {
                inj.w2a = d;
                inj.w2b = 1.0 + d;
            }
            if (j == 2) {
                inj.w3a = d;
                inj.w3b = 1.0 + d;
            }
            run[static_cast<std::size_t>(j)][static_cast<std::size_t>(sg)] = run_spline(ctx, st, inj, to);
        }
    const std::size_t T = taus.size();
    std::vector<DOut> out(T);
    std::vector<double> sa(T, 0.0), sb(T, 0.0);
    std::vector<int> cnt(T, 0);
    for (int p = 0; p < base.Np; ++p) {
        const std::size_t pz = static_cast<std::size_t>(p);
        bool ok = base.status_time[pz] == 0;
        for (int j = 0; j < 3; ++j)
            for (int sg = 0; sg < 2; ++sg)
                ok = ok && run[static_cast<std::size_t>(j)][static_cast<std::size_t>(sg)].status_time[pz] == 0;
        if (!ok)
            continue;
        double a0[3], b0[3];
        initial_covectors(vf, &base.seeds[3 * pz], a0, b0);
        for (std::size_t n = 0; n < T; ++n) {
            double J[3][3];
            for (int j = 0; j < 3; ++j) {
                const auto& rp = run[static_cast<std::size_t>(j)][0];
                const auto& rm = run[static_cast<std::size_t>(j)][1];
                const double dj = rp.seeds[3 * pz + static_cast<std::size_t>(j)] -
                                  rm.seeds[3 * pz + static_cast<std::size_t>(j)];
                for (int i = 0; i < 3; ++i)
                    J[i][j] = (rp.times_x[(pz * T + n) * 3 + static_cast<std::size_t>(i)] -
                               rm.times_x[(pz * T + n) * 3 + static_cast<std::size_t>(i)]) / dj;
            }
            const double* cov = &base.times_cov[(pz * T + n) * 6];
            double da = 0.0, db = 0.0;
            for (int j = 0; j < 3; ++j) {
                double ra = 0.0, rb = 0.0;
                for (int i = 0; i < 3; ++i) {
                    ra += cov[i] * J[i][j];
                    rb += cov[3 + i] * J[i][j];
                }
                da += (ra - a0[j]) * (ra - a0[j]);
                db += (rb - b0[j]) * (rb - b0[j]);
            }
            sa[n] += da;
            sb[n] += db;
            ++cnt[n];
        }
    }
    for (std::size_t n = 0; n < T; ++n) {
        out[n].used = cnt[n];
        if (cnt[n] > 0) {
            out[n].rms_a = std::sqrt(sa[n] / cnt[n]);
            out[n].rms_b = std::sqrt(sb[n] / cnt[n]);
        }
    }
    return out;
}

void case_D(TestReport& rep, CudaContext& ctx, int N, int M, const std::string& mode_note, bool extra) {
    const double t0 = now_s();
    const sl::SlabFieldSource gsrc = gauss_source(ctx, N);
    const sl::SlabFieldSource asrc = sl::SlabFieldSource::analytic("generic3d", N);
    struct C {
        std::string name;
        const sl::SlabFieldSource* src;
        double eps;
    };
    const std::vector<C> cases = {{"generic3d eps 0.25", &asrc, 0.25},
                                  {"gauss s2=1 ell=.25 seed=3001 eps 1.0", &gsrc, 1.0}};
    const std::vector<double> taus = {0.5, 2.0};
    const std::array<double, 3> deltas = {8e-3, 4e-3, 2e-3};
    const double h = 1.0 / N;
    for (const C& c : cases) {
        const std::string tag = "D " + c.name + " N=" + std::to_string(N) + " seeds=" +
                                std::to_string(M * M) + ": ";
        const sl::ProductionStage st = make_stage(ctx, N, *c.src, c.eps);
        rep.check(st.flow_ready(), tag + "stage flow ready", sl::to_string(st.report.status));
        if (!st.flow_ready())
            continue;
        const VFun vf = spline_velocity(st.direction_field());
        ll::InjectionSpec inj0;
        inj0.M = M;
        ll::TrackOptions tb;
        tb.tol = 1e-12;
        tb.h_max = h / 32.0;
        tb.times = taus;
        tb.covectors = true;
        const ll::LagrangianTrackResult base = run_spline(ctx, st, inj0, tb);
        std::vector<std::vector<DOut>> lev;
        for (double d : deltas)
            lev.push_back(d_level(ctx, st, base, vf, M, h, taus, d, 1e-12, 32.0));
        for (std::size_t n = 0; n < taus.size(); ++n) {
            double ord[2][2]; // [pair][a,b]
            for (int pr = 0; pr < 2; ++pr) {
                const DOut &x = lev[static_cast<std::size_t>(pr)][n], &y = lev[static_cast<std::size_t>(pr) + 1][n];
                ord[pr][0] = std::log2(x.rms_a / y.rms_a);
                ord[pr][1] = std::log2(x.rms_b / y.rms_b);
            }
            const double omin = std::min({ord[0][0], ord[0][1], ord[1][0], ord[1][1]});
            char b[600];
            std::snprintf(b, sizeof(b),
                          "rms_a (8,4,2)e-3 = %.3e %.3e %.3e | rms_b = %.3e %.3e %.3e | orders a: %.3f %.3f, b: %.3f %.3f"
                          " | seeds used %d",
                          lev[0][n].rms_a, lev[1][n].rms_a, lev[2][n].rms_a, lev[0][n].rms_b, lev[1][n].rms_b,
                          lev[2][n].rms_b, ord[0][0], ord[1][0], ord[0][1], ord[1][1], lev[2][n].used);
            rep.check(omin >= 1.8, tag + fmt("tau=%.1f: observed order >= 1.8 on both delta pairs (a and b)", taus[n]),
                      std::string(b) + fmt(" | min order %.3f", omin));
        }
        std::printf("  %sINFO tol 1e-12, h_max h/32; %s\n", tag.c_str(), mode_note.c_str());
        if (extra) {
            // DIAG-5 (recorded): smaller deltas, to see whether the pre-asymptotic orders reach 2.
            std::vector<std::vector<DOut>> lv2;
            const std::array<double, 4> d2 = {1e-3, 5e-4, 2.5e-4, 1.25e-4};
            for (double d : d2)
                lv2.push_back(d_level(ctx, st, base, vf, M, h, taus, d, 1e-12, 32.0));
            for (std::size_t n = 0; n < taus.size(); ++n) {
                std::string l = "rms_a (1,.5,.25,.125)e-3 =";
                for (const auto& v : lv2)
                    l += fmt(" %.3e", v[n].rms_a);
                l += " | orders a:";
                for (std::size_t i = 0; i + 1 < lv2.size(); ++i)
                    l += fmt(" %.2f", std::log2(lv2[i][n].rms_a / lv2[i + 1][n].rms_a));
                l += " | rms_b =";
                for (const auto& v : lv2)
                    l += fmt(" %.3e", v[n].rms_b);
                l += " | orders b:";
                for (std::size_t i = 0; i + 1 < lv2.size(); ++i)
                    l += fmt(" %.2f", std::log2(lv2[i][n].rms_b / lv2[i + 1][n].rms_b));
                std::printf("[RECORD] DIAG-5 %stau=%.1f: %s\n", tag.c_str(), taus[n], l.c_str());
            }
        }
    }
    std::printf("  D total %.1f s\n", now_s() - t0);
}

// ================================================================================================
// C-infinity analytic field with exact derivatives (DIAG-1/2)
// ================================================================================================
struct SmoothField {
    double A = 0.05, B = 0.05, c = 0.5;
    __host__ __device__ double phi(const double x[3]) const {
        const double w = 6.283185307179586476925286766559;
        return x[0] + A * sin(w * x[1]) * cos(w * x[0]) + B * sin(w * x[2]) * sin(w * x[0]);
    }
    __host__ __device__ double lnk(const double x[3]) const {
        const double w = 6.283185307179586476925286766559;
        return c * cos(w * (x[0] + x[1] + x[2]));
    }
    __host__ __device__ void eval_ext(const double x[3], double g[3], double& k, double gl[3],
                                      double H[6]) const {
        const double w = 6.283185307179586476925286766559;
        const double s1 = sin(w * x[0]), c1 = cos(w * x[0]), s2 = sin(w * x[1]), c2 = cos(w * x[1]),
                     s3 = sin(w * x[2]), c3 = cos(w * x[2]);
        g[0] = 1.0 - A * w * s2 * s1 + B * w * s3 * c1;
        g[1] = A * w * c2 * c1;
        g[2] = B * w * c3 * s1;
        H[0] = -A * w * w * s2 * c1 - B * w * w * s3 * s1;
        H[1] = -A * w * w * s2 * c1;
        H[2] = -B * w * w * s3 * s1;
        H[3] = -A * w * w * c2 * s1;
        H[4] = B * w * w * c3 * c1;
        H[5] = 0.0;
        const double sd = sin(w * (x[0] + x[1] + x[2]));
        k = exp(c * cos(w * (x[0] + x[1] + x[2])));
        gl[0] = gl[1] = gl[2] = -c * w * sd;
    }
    __host__ __device__ void eval(const double x[3], double g[3], double& k) const {
        double gl[3], H[6];
        eval_ext(x, g, k, gl, H);
    }
};

VFun smooth_velocity(const SmoothField& f) {
    return [f](const double x[3], double v[3]) {
        double g[3], k;
        f.eval(x, g, k);
        v[0] = k * g[0];
        v[1] = k * g[1];
        v[2] = k * g[2];
    };
}

/// Central-FD check of g = grad phi, H = Hess phi, grad ln k of SmoothField at test points.
double smooth_derivative_check(const SmoothField& f, double& max_g, double& max_H, double& max_gl) {
    const double d = 1e-4;
    max_g = max_H = max_gl = 0.0;
    const double pts[4][3] = {{0.13, 0.41, 0.77}, {0.62, 0.05, 0.31}, {0.9, 0.55, 0.2}, {0.27, 0.83, 0.64}};
    for (const auto& x0 : pts) {
        double g[3], k, gl[3], H[6];
        f.eval_ext(x0, g, k, gl, H);
        for (int i = 0; i < 3; ++i) {
            double xp[3] = {x0[0], x0[1], x0[2]}, xm[3] = {x0[0], x0[1], x0[2]};
            xp[i] += d;
            xm[i] -= d;
            max_g = std::max(max_g, std::abs((f.phi(xp) - f.phi(xm)) / (2 * d) - g[i]));
            max_gl = std::max(max_gl, std::abs((f.lnk(xp) - f.lnk(xm)) / (2 * d) - gl[i]));
            double gp[3], gm[3], kk, glp[3], Hp[6];
            f.eval_ext(xp, gp, kk, glp, Hp);
            f.eval_ext(xm, gm, kk, glp, Hp);
            // H_ij = d g_i / d x_j etc.: compare the three columns
            const int idx[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};
            for (int r = 0; r < 3; ++r)
                max_H = std::max(max_H, std::abs((gp[r] - gm[r]) / (2 * d) - H[idx[r][i]]));
        }
    }
    return std::max({max_g, max_H, max_gl});
}

ll::LagrangianTrackResult run_smooth(CudaContext& ctx, const SmoothField& f, int M, double tol, double hmax) {
    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = M;
    ll::TrackOptions to;
    to.tol = tol;
    to.h_max = hmax;
    to.planes = {0.5, 1.0, 2.0, 4.0};
    to.times = {0.5, 1.0, 2.0, 4.0};
    to.covectors = true;
    ll::LagrangianTrackerT<SmoothField> tr;
    tr.prepare(ctx, f, inj, to, nullptr);
    return tr.track(ctx);
}

void diag1(TestReport& rep, CudaContext& ctx) {
    const double t0 = now_s();
    const SmoothField f;
    double mg, mH, mgl;
    smooth_derivative_check(f, mg, mH, mgl);
    const bool dok = mg < 1e-6 && mH < 1e-5 && mgl < 1e-6;
    rep.check(dok, "DIAG-1 derivative check (central FD, d=1e-4) of g, H, grad ln k of the analytic field",
              fmt("max |g - FD| %.2e", mg) + fmt(", |H - FD(g)| %.2e", mH) + fmt(", |grad ln k - FD| %.2e", mgl));
    const VFun vf = smooth_velocity(f);
    for (double tol : {1e-10, 1e-12}) {
        const ll::LagrangianTrackResult R = run_smooth(ctx, f, 16, tol, 1.0 / 128.0);
        const IdStats s = identity_stats(R, vf);
        const std::vector<double> q = pct(s.cnorm, {0.5, 0.99, 1.0});
        std::printf("[RECORD] DIAG-1 C-inf field (A=B=0.05, c=0.5), h_max=1/128, tol %.0e: identity residual max %.3e "
                    "rms %.3e over %lld records (non-ok passes %lld); max|I| %.3e; |cov| p50 %.3e p99 %.3e max %.3e\n",
                    tol, s.max, s.rms(), s.n, s.nonok, s.max_I, q[0], q[1], q[2]);
    }
    std::printf("  DIAG-1 %.1f s\n", now_s() - t0);
}

void diag2_3(TestReport& rep, CudaContext& ctx, int N) {
    (void)rep;
    const double t0 = now_s();
    const sl::SlabFieldSource gsrc = gauss_source(ctx, N);
    const sl::SlabFieldSource asrc = sl::SlabFieldSource::analytic("generic3d", N);
    struct C {
        std::string name;
        const sl::SlabFieldSource* src;
        double eps;
    };
    const std::vector<C> cases = {{"generic3d eps 0.25", &asrc, 0.25},
                                  {"gauss eps 1.0", &gsrc, 1.0},
                                  {"gauss eps 2.0", &gsrc, 2.0}};
    const double h = 1.0 / N;
    const std::array<double, 4> divs = {8.0, 16.0, 32.0, 64.0};
    for (const C& c : cases) {
        const std::string tag = "DIAG-2 " + c.name + " N=" + std::to_string(N) + ": ";
        const sl::ProductionStage st = make_stage(ctx, N, *c.src, c.eps);
        if (!st.flow_ready()) {
            std::printf("[RECORD] %sstage not ready\n", tag.c_str());
            continue;
        }
        const VFun vf = spline_velocity(st.direction_field());
        std::vector<ll::LagrangianTrackResult> res;
        std::vector<IdStats> st_id;
        for (double dv : divs) {
            ll::InjectionSpec inj;
            inj.M = 16;
            ll::TrackOptions to;
            to.tol = 1e-12;
            to.h_max = h / dv;
            to.planes = {0.5, 1.0, 2.0, 4.0};
            to.times = {0.5, 1.0, 2.0, 4.0};
            to.covectors = true;
            res.push_back(run_spline(ctx, st, inj, to));
            std::printf("  %skernels h/%.0f: plane %.2f s, time %.2f s\n", tag.c_str(), dv, res.back().seconds_plane_kernel,
                        res.back().seconds_time_kernel);
            st_id.push_back(identity_stats(res.back(), vf));
        }
        std::string line = "E residual (max / rms) at tol 1e-12 for h_max = h/8,h/16,h/32,h/64: ";
        for (std::size_t i = 0; i < divs.size(); ++i)
            line += fmt2("%.3e/%.3e  ", st_id[i].max, st_id[i].rms());
        line += "| observed orders (max):";
        for (std::size_t i = 0; i + 1 < divs.size(); ++i)
            line += fmt(" %.2f", std::log2(st_id[i].max / st_id[i + 1].max));
        line += " | (rms):";
        for (std::size_t i = 0; i + 1 < divs.size(); ++i)
            line += fmt(" %.2f", std::log2(st_id[i].rms() / st_id[i + 1].rms()));
        std::printf("[RECORD] %s%s\n", tag.c_str(), line.c_str());
        // DIAG-3: I accuracy, h/8 vs h/64 (planes and snapshots)
        const ll::LagrangianTrackResult &r8 = res.front(), &r64 = res.back();
        double dmax = 0.0, dsq = 0.0, dmax_t = 0.0;
        long long n = 0;
        const std::size_t Np = static_cast<std::size_t>(r8.Np), P = r8.planes.size(), T = r8.times.size();
        for (std::size_t p = 0; p < Np; ++p) {
            if (r8.status_plane[p] == 0 && r64.status_plane[p] == 0)
                for (std::size_t m = 0; m < P; ++m) {
                    const double d = std::abs(r8.planes_I[p * P + m] - r64.planes_I[p * P + m]);
                    dmax = std::max(dmax, d);
                    dsq += d * d;
                    ++n;
                }
            if (r8.status_time[p] == 0 && r64.status_time[p] == 0)
                for (std::size_t m = 0; m < T; ++m) {
                    const double d = std::abs(r8.times_I[p * T + m] - r64.times_I[p * T + m]);
                    dmax_t = std::max(dmax_t, d);
                }
        }
        std::printf("[RECORD] DIAG-3 %s N=%d: |I(h/8) - I(h/64)| over plane records: max %.3e rms %.3e (snapshots max %.3e);"
                    " RMS of I (h/64) %.3e, max|I| %.3e; RMS of e_v = 1 - exp(-I) %.3e\n",
                    c.name.c_str(), N, dmax, n > 0 ? std::sqrt(dsq / static_cast<double>(n)) : std::nan(""), dmax_t,
                    st_id.back().rms_I(), st_id.back().max_I, st_id.back().rms_ev());
    }
    // C-infinity field, same h_max values
    {
        const SmoothField f;
        const VFun vf = smooth_velocity(f);
        std::vector<IdStats> ids;
        for (double dv : divs)
            ids.push_back(identity_stats(run_smooth(ctx, f, 16, 1e-12, h / dv), vf));
        std::string line = "DIAG-2 C-inf field, tol 1e-12, E residual (max / rms) for h_max = h/8..h/64 (h=1/" +
                           std::to_string(N) + "): ";
        for (std::size_t i = 0; i < divs.size(); ++i)
            line += fmt2("%.3e/%.3e  ", ids[i].max, ids[i].rms());
        line += "| observed orders (max):";
        for (std::size_t i = 0; i + 1 < divs.size(); ++i)
            line += fmt(" %.2f", std::log2(ids[i].max / ids[i + 1].max));
        std::printf("[RECORD] %s\n", line.c_str());
    }
    std::printf("  DIAG-2/3 %.1f s\n", now_s() - t0);
}

} // namespace

int main(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--full")
            cfg.full = true;
        else if (a == "--diag")
            cfg.diag_flag = true;
        else if (a == "--n" && i + 1 < argc)
            cfg.n = std::atoi(argv[++i]);
        else if (a == "--only" && i + 1 < argc)
            cfg.only.insert(argv[++i]);
        else {
            std::fprintf(stderr, "usage: %s [--n N] [--full] [--only A1|A2|A3|B|D|E|DIAG] [--diag]\n", argv[0]);
            return 2;
        }
    }
    // default (fast) tier: 16^3 / 32^3 (A1, A2, E, D at 16^3; B at 32^3), --full: 64^3 (A, B, D, E)
    const int nA1 = cfg.n > 0 ? cfg.n : (cfg.full ? 64 : 16);
    const int nA2 = cfg.n > 0 ? cfg.n : (cfg.full ? 64 : 32);
    const int nB = cfg.n > 0 ? cfg.n : (cfg.full ? 64 : 32);
    const int nE = cfg.n > 0 ? cfg.n : (cfg.full ? 64 : 16);
    const int nD = cfg.n > 0 ? cfg.n : (cfg.full ? 64 : 32);
    const int nDiag = cfg.n > 0 ? cfg.n : 32;
    const int MD = cfg.full ? 16 : 6; // 256 seeds (spec) / 36 seeds (fast tier)
    const bool heavy_diag = cfg.full || cfg.diag_flag || cfg.only.count("DIAG") > 0;

    TestReport rep;
    CudaContext ctx(0);
    const double t0 = now_s();
    std::printf("lagrangian_tracker_controls (%s mode)\n", cfg.full ? "FULL" : "default");
    auto lap = [&](const char* w) { std::printf("  [time] %s: %.1f s elapsed\n", w, now_s() - t0); std::fflush(stdout); };

    if (cfg.run("A1")) {
        case_A1(rep, ctx, nA1);
        lap("A1");
    }
    if (cfg.run("A2")) {
        case_A2(rep, ctx, nA2);
        lap("A2");
    }
    if (cfg.run("A3")) {
        case_A3(rep, ctx);
        lap("A3");
    }
    if (cfg.run("B")) {
        case_B(rep, ctx, nB);
        lap("B");
    }
    if (cfg.run("E")) {
        case_E(rep, ctx, nE);
        lap("E");
    }
    if (cfg.run("D")) {
        case_D(rep, ctx, nD, MD,
               cfg.full ? "FULL D: 256 seeds" : "default tier: 36 seeds (6x6 lattice) at this N instead of the spec's 256 seeds at 64^3 (see --full)",
               heavy_diag);
        lap("D");
    }
    if (cfg.run("DIAG")) {
        diag1(rep, ctx);
        if (heavy_diag)
            diag2_3(rep, ctx, nDiag);
        else
            std::printf("  DIAG-2/3/5 (h_max sweeps, small-delta FD) are skipped in the fast tier: run with --diag, --only DIAG or --full\n");
        lap("DIAG");
    }

    std::printf("\nSUMMARY: %d checks, %d failed -> %s  (%.1f s)\n", rep.checks, rep.failed,
                rep.overall_pass ? "ALL PASS" : "SOME FAILED", now_s() - t0);
    return rep.overall_pass ? 0 : 1;
}
