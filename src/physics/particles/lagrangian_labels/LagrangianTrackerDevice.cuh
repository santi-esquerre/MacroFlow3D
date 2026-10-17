#pragma once

/**
 * @file LagrangianTrackerDevice.cuh
 * @brief SF-36 N4: device templates of the forward Lagrangian-label tracker (DP5(4) stepper
 *        templated on the state dimension, plane pass, time pass, kernels).
 *
 * NUMERICAL CONTRACT: every translation unit that instantiates anything from this header MUST be
 * compiled with `--fmad=false` (as SlabTraceGpu.cu), so that device multiplications/additions are
 * IEEE-rounded exactly like the host SF-30 integrator (criterion C: positions bitwise equal to
 * `trace_to_plane`).
 *
 * ## Traced field (device functor contract)
 *
 * A `Field` provides two `__device__` const methods, both at an UNWRAPPED point `x` (the functor
 * owns periodicity):
 *
 *   void eval(const double x[3], double g[3], double& k) const;
 *   void eval_ext(const double x[3], double g[3], double& k, double grad_lnk[3],
 *                 double H[6]) const;
 *
 * `g` is the direction field (v = k g), `grad_lnk` = grad(ln k) and `H` = (xx, yy, zz, xy, xz, yz)
 * the Hessian of the potential fluctuation s_h (g = G + grad s_h, hence div g = trace H and
 * div v = k (grad ln k . g + trace H)). `eval_ext` must return exactly the `g`, `k` of `eval`.
 *
 * ## Equations (sigma = +1; u = g/|g|; w(a) = a - (u.a) u)
 *
 * Arclength  : dx/ds = u, dtau/ds = 1/(k|g|), dI/ds = (grad ln k . g + tr H)/|g|,
 *              da/ds = -H w(a)/|g|            (same for b)
 * Plane x1   : dx2/dx1 = g2/g1, dx3/dx1 = g3/g1, ds/dx1 = |g|/g1, dtau/dx1 = 1/(k g1),
 *              dI/dx1 = (grad ln k . g + tr H)/g1, da/dx1 = -H w(a)/g1      (needs g1 > 0)
 * Time tau   : dx/dtau = k g, ds/dtau = k|g|, dI/dtau = k (grad ln k . g + tr H),
 *              da/dtau = -k H w(a)
 *
 * ## State layouts (D = 5 without covectors, 11 with)
 *
 *   arclength : (x1, x2, x3, tau, I, [a1 a2 a3 b1 b2 b3])   independent variable s
 *   plane land: (x2, x3, s, tau, I, [a b])                  independent variable x1
 *   time land : (x1, x2, x3, s, I, [a b])                   independent variable tau
 *
 * The error norm of every step runs over the leading position components only (3 for the
 * arclength and the time landing, 2 for the plane landing), exactly as SF-30; tau, I, s and the
 * covectors ride passively, so positions, tau and statuses are bitwise independent of D (the
 * per-component stage expressions are those of `trace_dev::dp5_step`).
 */

#include "../../../numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "../../streamfunctions/inlet_slab/SlabTraceDevice.cuh"
#include "LagrangianTrackerTypes.hpp"

#include <cstdint>

namespace macroflow3d {
namespace lagrangian_labels {

namespace tdev = streamfunctions::inlet_slab::trace_dev;
using tdev::EvalCode;
using tdev::LandingOutcome;

/// Device spline field of the cube production stage. `eval` is bitwise identical to
/// `inlet_slab::DeviceSplineDirectionField::operator()`; `eval_ext` adds grad(ln k) and the
/// Hessian of the potential fluctuation (value/gradient of evaluate_point_hessian are bitwise equal
/// to evaluate_point).
struct SplineLagrangianField {
    interpolation::PeriodicTricubicBSplineView potential{};
    interpolation::PeriodicTricubicBSplineView log_conductivity{};
    double G[3] = {0.0, 0.0, 0.0};

    __device__ void eval(const double x[3], double g[3], double& k) const {
        real v = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
        interpolation::evaluate_point(potential, x[0], x[1], x[2], v, gx, gy, gz);
        g[0] = G[0] + gx;
        g[1] = G[1] + gy;
        g[2] = G[2] + gz;
        real y = 0.0, yx = 0.0, yy = 0.0, yz = 0.0;
        interpolation::evaluate_point(log_conductivity, x[0], x[1], x[2], y, yx, yy, yz);
        k = exp(y);
    }

    __device__ void eval_ext(const double x[3], double g[3], double& k, double grad_lnk[3],
                             double H[6]) const {
        real v = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
        interpolation::evaluate_point_hessian(potential, x[0], x[1], x[2], v, gx, gy, gz, H);
        g[0] = G[0] + gx;
        g[1] = G[1] + gy;
        g[2] = G[2] + gz;
        real y = 0.0, yx = 0.0, yy = 0.0, yz = 0.0;
        interpolation::evaluate_point(log_conductivity, x[0], x[1], x[2], y, yx, yy, yz);
        k = exp(y);
        grad_lnk[0] = yx;
        grad_lnk[1] = yy;
        grad_lnk[2] = yz;
    }
};

namespace ldev {

__device__ inline double dnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// ------------------------------------------------------------------------------------------
// RHS helpers
// ------------------------------------------------------------------------------------------

/// Common validated evaluation. Returns good iff g finite, |g| > 0 finite, k finite > 0 (the same
/// checks and order as trace_dev::ArclengthRhs).
template <class Field>
__device__ inline EvalCode eval_checked(const Field& field, const double x[3], double g[3],
                                        double& k, double gl[3], double H[6], double& gn) {
    g[0] = g[1] = g[2] = 0.0;
    k = 0.0;
    field.eval_ext(x, g, k, gl, H);
    if (!tdev::finite3(g)) {
        return EvalCode::invalid_direction;
    }
    gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    if (!(gn > 0.0) || !isfinite(gn)) {
        return EvalCode::invalid_direction;
    }
    if (!isfinite(k) || !(k > 0.0)) {
        return EvalCode::invalid_conductivity;
    }
    return EvalCode::good;
}

/// divergence factor d = grad ln k . g + tr H (div v = k d).
__device__ inline double div_factor(const double g[3], const double gl[3], const double H[6]) {
    return gl[0] * g[0] + gl[1] * g[1] + gl[2] * g[2] + (H[0] + H[1] + H[2]);
}

/// out = H w(a), w(a) = a - (u.a) u, u = g/gn.
__device__ inline void hess_w(const double H[6], const double g[3], double gn, const double a[3],
                              double out[3]) {
    const double u0 = g[0] / gn, u1 = g[1] / gn, u2 = g[2] / gn;
    const double ua = u0 * a[0] + u1 * a[1] + u2 * a[2];
    const double w0 = a[0] - ua * u0, w1 = a[1] - ua * u1, w2 = a[2] - ua * u2;
    out[0] = H[0] * w0 + H[3] * w1 + H[4] * w2;
    out[1] = H[3] * w0 + H[1] * w1 + H[5] * w2;
    out[2] = H[4] * w0 + H[5] * w1 + H[2] * w2;
}

/// Writes the covector part dy[5..10] = -scale * (H w(a), H w(b)); `inv` multiplies as a
/// divisor-free factor chosen by the caller (1/gn, 1/g1 or k).
template <bool DIV>
__device__ inline void cov_rhs(const double H[6], const double g[3], double gn, const double* y,
                               double fac, double* dy) {
    const double a[3] = {y[5], y[6], y[7]};
    const double b[3] = {y[8], y[9], y[10]};
    double Ha[3], Hb[3];
    hess_w(H, g, gn, a, Ha);
    hess_w(H, g, gn, b, Hb);
    if (DIV) {
        dy[5] = -Ha[0] / fac;
        dy[6] = -Ha[1] / fac;
        dy[7] = -Ha[2] / fac;
        dy[8] = -Hb[0] / fac;
        dy[9] = -Hb[1] / fac;
        dy[10] = -Hb[2] / fac;
    } else {
        dy[5] = -fac * Ha[0];
        dy[6] = -fac * Ha[1];
        dy[7] = -fac * Ha[2];
        dy[8] = -fac * Hb[0];
        dy[9] = -fac * Hb[1];
        dy[10] = -fac * Hb[2];
    }
}

/// Arclength RHS; y = (x1, x2, x3, tau, I, [a, b]).
template <class Field, int D> struct ArclengthRhs {
    const Field& field;
    long long& nfev;
    double last_g1 = 0.0;
    double last_g1_hat = 0.0;

    __device__ EvalCode operator()(double /*s*/, const double y[D], double dy[D]) {
        const double x[3] = {y[0], y[1], y[2]};
        double g[3], k, gl[3], H[6], gn;
        const EvalCode c = eval_checked(field, x, g, k, gl, H, gn);
        ++nfev;
        if (c != EvalCode::good) {
            return c;
        }
        dy[0] = g[0] / gn;
        dy[1] = g[1] / gn;
        dy[2] = g[2] / gn;
        dy[3] = 1.0 / (k * gn);
        dy[4] = div_factor(g, gl, H) / gn;
        if constexpr (D > 5) {
            cov_rhs<true>(H, g, gn, y, gn, dy);
        }
        last_g1 = g[0];
        last_g1_hat = g[0] / gn;
        return EvalCode::good;
    }
};

/// Plane-landing RHS (independent x1); y = (x2, x3, s, tau, I, [a, b]).
template <class Field, int D> struct PlaneLandingRhs {
    const Field& field;
    long long& nfev;

    __device__ EvalCode operator()(double x1, const double y[D], double dy[D]) {
        const double x[3] = {x1, y[0], y[1]};
        double g[3], k, gl[3], H[6], gn;
        const EvalCode c = eval_checked(field, x, g, k, gl, H, gn);
        ++nfev;
        if (c != EvalCode::good) {
            return c;
        }
        if (!(g[0] > 0.0)) {
            return EvalCode::nonpositive_g1;
        }
        dy[0] = g[1] / g[0];
        dy[1] = g[2] / g[0];
        dy[2] = 1.0 * gn / g[0];
        dy[3] = 1.0 / (k * g[0]);
        dy[4] = div_factor(g, gl, H) / g[0];
        if constexpr (D > 5) {
            cov_rhs<true>(H, g, gn, y, g[0], dy);
        }
        return EvalCode::good;
    }
};

/// Time-landing RHS (independent tau); y = (x1, x2, x3, s, I, [a, b]).
template <class Field, int D> struct TimeLandingRhs {
    const Field& field;
    long long& nfev;

    __device__ EvalCode operator()(double /*tau*/, const double y[D], double dy[D]) {
        const double x[3] = {y[0], y[1], y[2]};
        double g[3], k, gl[3], H[6], gn;
        const EvalCode c = eval_checked(field, x, g, k, gl, H, gn);
        ++nfev;
        if (c != EvalCode::good) {
            return c;
        }
        dy[0] = k * g[0];
        dy[1] = k * g[1];
        dy[2] = k * g[2];
        dy[3] = k * gn;
        dy[4] = k * div_factor(g, gl, H);
        if constexpr (D > 5) {
            cov_rhs<false>(H, g, gn, y, k, dy);
        }
        return EvalCode::good;
    }
};

// ------------------------------------------------------------------------------------------
// DP5(4) step (copy of trace_dev::dp5_step with the state dimension D)
// ------------------------------------------------------------------------------------------
template <int D, class Rhs>
__device__ EvalCode dp5_step(Rhs& rhs, double t, double t_end, const double y[D],
                             const double k1[D], double h, int n_err, double tol, double y5[D],
                             double k7[D], double& err) {
    using C = tdev::DP5;
    double k2[D], k3[D], k4[D], k5[D], k6[D], yt[D];
    EvalCode code;

    for (int i = 0; i < D; ++i)
        yt[i] = y[i] + h * (C::a21 * k1[i]);
    if ((code = rhs(t + C::c2 * h, yt, k2)) != EvalCode::good)
        return code;

    for (int i = 0; i < D; ++i)
        yt[i] = y[i] + h * (C::a31 * k1[i] + C::a32 * k2[i]);
    if ((code = rhs(t + C::c3 * h, yt, k3)) != EvalCode::good)
        return code;

    for (int i = 0; i < D; ++i)
        yt[i] = y[i] + h * (C::a41 * k1[i] + C::a42 * k2[i] + C::a43 * k3[i]);
    if ((code = rhs(t + C::c4 * h, yt, k4)) != EvalCode::good)
        return code;

    for (int i = 0; i < D; ++i)
        yt[i] = y[i] + h * (C::a51 * k1[i] + C::a52 * k2[i] + C::a53 * k3[i] + C::a54 * k4[i]);
    if ((code = rhs(t + C::c5 * h, yt, k5)) != EvalCode::good)
        return code;

    for (int i = 0; i < D; ++i)
        yt[i] = y[i] + h * (C::a61 * k1[i] + C::a62 * k2[i] + C::a63 * k3[i] + C::a64 * k4[i] +
                            C::a65 * k5[i]);
    if ((code = rhs(t_end, yt, k6)) != EvalCode::good)
        return code;

    for (int i = 0; i < D; ++i)
        y5[i] = y[i] + h * (C::b1 * k1[i] + C::b3 * k3[i] + C::b4 * k4[i] + C::b5 * k5[i] +
                            C::b6 * k6[i]);
    if ((code = rhs(t_end, y5, k7)) != EvalCode::good)
        return code;

    err = 0.0;
    for (int i = 0; i < n_err; ++i) {
        const double e = h * (C::e1 * k1[i] + C::e3 * k3[i] + C::e4 * k4[i] + C::e5 * k5[i] +
                              C::e6 * k6[i] + C::e7 * k7[i]);
        err = tdev::hmax(err, fabs(e) / tol);
    }
    return EvalCode::good;
}

// ------------------------------------------------------------------------------------------
// Landings
// ------------------------------------------------------------------------------------------

/// Henon landing on x1 = X from (x1_a, ya = (x2, x3, s, tau, I, ..)) (copy of
/// trace_dev::land_on_plane, state dimension D).
template <int D, class Field>
__device__ LandingOutcome land_on_plane(const Field& field, const PassOptions& opt, double x1_a,
                                        const double ya[D], double X, double y_out[D],
                                        PassStats& st) {
    PlaneLandingRhs<Field, D> rhs{field, st.nfev};
    double t = x1_a;
    double y[D];
    for (int i = 0; i < D; ++i)
        y[i] = ya[i];
    double k1[D];
    EvalCode code = rhs(t, y, k1);
    if (code != EvalCode::good) {
        return tdev::map_code(code);
    }
    const double dir = (X - t) >= 0.0 ? 1.0 : -1.0;
    double dt_mag = tdev::hmin(opt.h_max, fabs(X - t));
    long long guard = 0;
    while (true) {
        const double remaining = X - t;
        bool last = false;
        double dt = dir * dt_mag;
        if (dt_mag >= fabs(remaining)) {
            dt = remaining;
            last = true;
        }
        const double t_end = last ? X : t + dt;
        double y5[D], k7[D], err = 0.0;
        code = dp5_step<D>(rhs, t, t_end, y, k1, dt, 2, opt.tol, y5, k7, err);
        if (code != EvalCode::good) {
            return tdev::map_code(code);
        }
        const double fac = tdev::step_factor(err);
        if (err <= 1.0) {
            ++st.landing_steps;
            for (int i = 0; i < D; ++i) {
                y[i] = y5[i];
                k1[i] = k7[i];
            }
            if (last) {
                for (int i = 0; i < D; ++i)
                    y_out[i] = y[i];
                return LandingOutcome::landed;
            }
            t = t_end;
            dt_mag = tdev::hmin(opt.h_max, fabs(dt) * fac);
        } else {
            ++st.landing_rejected;
            dt_mag = fabs(dt) * fac;
            if (dt_mag < opt.min_step) {
                return LandingOutcome::underflow;
            }
        }
        if (++guard > 100000000LL) {
            return LandingOutcome::underflow;
        }
    }
}

/// Henon landing on tau = T from (tau_a, ya = (x1, x2, x3, s, I, ..)). Sub-steps are bounded in
/// arclength: tau-step <= h_max / (k|g|) at the step start; the underflow test compares the
/// arclength-equivalent step (tau-step * k|g|) with min_step. n_err = 3 on the positions; the last
/// stage sits exactly at T.
template <int D, class Field>
__device__ LandingOutcome land_on_time(const Field& field, const PassOptions& opt, double tau_a,
                                       const double ya[D], double T, double y_out[D],
                                       PassStats& st) {
    TimeLandingRhs<Field, D> rhs{field, st.nfev};
    double t = tau_a;
    double y[D];
    for (int i = 0; i < D; ++i)
        y[i] = ya[i];
    double k1[D];
    EvalCode code = rhs(t, y, k1);
    if (code != EvalCode::good) {
        return tdev::map_code(code);
    }
    double dt_mag = tdev::hmin(T - t, opt.h_max / k1[3]);
    long long guard = 0;
    while (true) {
        const double remaining = T - t;
        bool last = false;
        double dt = dt_mag;
        if (dt_mag >= remaining) {
            dt = remaining;
            last = true;
        }
        const double t_end = last ? T : t + dt;
        double y5[D], k7[D], err = 0.0;
        code = dp5_step<D>(rhs, t, t_end, y, k1, dt, 3, opt.tol, y5, k7, err);
        if (code != EvalCode::good) {
            return tdev::map_code(code);
        }
        const double fac = tdev::step_factor(err);
        if (err <= 1.0) {
            ++st.landing_steps;
            for (int i = 0; i < D; ++i) {
                y[i] = y5[i];
                k1[i] = k7[i];
            }
            if (last) {
                for (int i = 0; i < D; ++i)
                    y_out[i] = y[i];
                return LandingOutcome::landed;
            }
            t = t_end;
            dt_mag = tdev::hmin(opt.h_max / k1[3], fabs(dt) * fac);
        } else {
            ++st.landing_rejected;
            dt_mag = fabs(dt) * fac;
            if (dt_mag * k1[3] < opt.min_step) {
                return LandingOutcome::underflow;
            }
        }
        if (++guard > 100000000LL) {
            return LandingOutcome::underflow;
        }
    }
}

// ------------------------------------------------------------------------------------------
// Per-particle output slices
// ------------------------------------------------------------------------------------------

/// Output slice of one particle in the plane pass (pointers into the global arrays).
struct PlaneOut {
    double* x23;      ///< P * 2 (x2, x3 per plane)
    double* tau;      ///< P
    double* I;        ///< P
    double* cov;      ///< P * 6 or nullptr
};
/// Output slice of one particle in the time pass.
struct TimeOut {
    double* x;   ///< T * 3
    double* I;   ///< T
    double* cov; ///< T * 6 or nullptr
};

/// Initial state at the seed: (x, tau = 0, I = 0, a = (-v2, v1, 0), b = (-v3/v1, 0, 1)).
template <class Field, int D>
__device__ inline void init_state(const Field& field, const double seed[3], double y[D]) {
    y[0] = seed[0];
    y[1] = seed[1];
    y[2] = seed[2];
    y[3] = 0.0;
    y[4] = 0.0;
    if constexpr (D > 5) {
        double g[3] = {0.0, 0.0, 0.0}, k = 0.0;
        field.eval(seed, g, k);
        const double v1 = k * g[0], v2 = k * g[1], v3 = k * g[2];
        y[5] = -v2;
        y[6] = v1;
        y[7] = 0.0;
        y[8] = -v3 / v1;
        y[9] = 0.0;
        y[10] = 1.0;
    }
}

// ------------------------------------------------------------------------------------------
// PLANE pass
// ------------------------------------------------------------------------------------------
template <int D, class Field>
__device__ void plane_pass(const Field& field, const double seed[3], const double* planes, int P,
                           const PassOptions& opt, PlaneOut out, PassStats& st) {
    double y[D];
    init_state<Field, D>(field, seed, y);
    double s = 0.0;
    double f[D];
    ArclengthRhs<Field, D> rhs{field, st.nfev};

    EvalCode code = rhs(0.0, y, f);
    if (code != EvalCode::good) {
        st.status = tdev::status_from(code);
        return;
    }
    st.min_g1_hat = rhs.last_g1_hat;
    if (!(rhs.last_g1 > 0.0)) {
        st.status = static_cast<int>(tdev::StreamlineStatus::seed_backflow);
        return;
    }

    int m = 0;
    double X = planes[0];
    double h = 0.1 * opt.h_max;

    while (true) {
        double y5[D], k7[D], err = 0.0;
        code = dp5_step<D>(rhs, s, s + h, y, f, h, 3, opt.tol, y5, k7, err);
        if (code != EvalCode::good) {
            st.status = tdev::status_from(code);
            st.final_s = s;
            return;
        }
        const double g1_end = rhs.last_g1;
        const double g1_hat_end = rhs.last_g1_hat;
        const double fac = tdev::step_factor(err);

        if (!(err <= 1.0)) {
            ++st.rejected_steps;
            h = h * fac;
            if (h < opt.min_step) {
                st.status = static_cast<int>(tdev::StreamlineStatus::step_underflow);
                st.final_s = s;
                return;
            }
            continue;
        }

        const bool crossing = 1.0 * (y5[0] - X) >= 0.0;
        if (crossing) {
            double ya[D];
            ya[0] = y[1];
            ya[1] = y[2];
            ya[2] = s;
            for (int i = 3; i < D; ++i)
                ya[i] = y[i];
            double yl[D];
            const LandingOutcome lo = land_on_plane<D>(field, opt, y[0], ya, X, yl, st);
            if (lo == LandingOutcome::backflow) {
                ++st.landings_discarded;
                h = 0.5 * h;
                if (h < opt.min_step) {
                    st.status = static_cast<int>(tdev::StreamlineStatus::landing_failed);
                    st.final_s = s;
                    return;
                }
                continue;
            }
            if (lo == LandingOutcome::invalid_direction) {
                st.status = static_cast<int>(tdev::StreamlineStatus::stagnation);
                st.final_s = s;
                return;
            }
            if (lo == LandingOutcome::invalid_conductivity) {
                st.status = static_cast<int>(tdev::StreamlineStatus::invalid_conductivity);
                st.final_s = s;
                return;
            }
            if (lo == LandingOutcome::underflow) {
                st.status = static_cast<int>(tdev::StreamlineStatus::landing_failed);
                st.final_s = s;
                return;
            }
            if (yl[2] - 0.0 > opt.max_arclength) {
                st.status = static_cast<int>(tdev::StreamlineStatus::cap_exceeded);
                st.final_s = s;
                return;
            }
            y[0] = X;
            y[1] = yl[0];
            y[2] = yl[1];
            for (int i = 3; i < D; ++i)
                y[i] = yl[i];
            s = yl[2];
            out.x23[2 * m] = y[1];
            out.x23[2 * m + 1] = y[2];
            out.tau[m] = y[3];
            out.I[m] = y[4];
            if constexpr (D > 5) {
                if (out.cov != nullptr) {
                    for (int i = 0; i < 6; ++i)
                        out.cov[6 * m + i] = y[5 + i];
                }
            }
            st.n_reached = m + 1;
            st.final_s = s;
            code = rhs(s, y, f); // FSAL invalidated by the landing
            if (code != EvalCode::good) {
                st.status = tdev::status_from(code);
                return;
            }
            st.min_g1_hat = tdev::hmin(st.min_g1_hat, rhs.last_g1_hat);
            if (!(rhs.last_g1 > 0.0)) {
                st.backflow_encounter = 1;
            }
            if (m == P - 1) {
                st.status = static_cast<int>(tdev::StreamlineStatus::ok);
                return;
            }
            ++m;
            X = planes[m];
            h = tdev::hmin(opt.h_max, h * fac);
            continue;
        }

        // Accept the arclength step.
        ++st.accepted_steps;
        if (m > 0) {
            const double Xl = planes[m - 1];
            if (y[0] >= Xl && y5[0] < Xl) {
                ++st.recrossings;
            }
        }
        for (int i = 0; i < D; ++i) {
            y[i] = y5[i];
            f[i] = k7[i];
        }
        s += h;
        st.min_g1_hat = tdev::hmin(st.min_g1_hat, g1_hat_end);
        if (!(g1_end > 0.0)) {
            st.backflow_encounter = 1;
        }
        if (s - 0.0 > opt.max_arclength) {
            st.status = static_cast<int>(tdev::StreamlineStatus::cap_exceeded);
            st.final_s = s;
            return;
        }
        h = tdev::hmin(opt.h_max, h * fac);
    }
}

// ------------------------------------------------------------------------------------------
// TIME pass
// ------------------------------------------------------------------------------------------
template <int D, class Field>
__device__ void time_pass(const Field& field, const double seed[3], const double* times, int T,
                          const PassOptions& opt, TimeOut out, PassStats& st) {
    double y[D];
    init_state<Field, D>(field, seed, y);
    double s = 0.0;
    double f[D];
    ArclengthRhs<Field, D> rhs{field, st.nfev};

    EvalCode code = rhs(0.0, y, f);
    if (code != EvalCode::good) {
        st.status = tdev::status_from(code);
        return;
    }
    st.min_g1_hat = rhs.last_g1_hat;
    if (!(rhs.last_g1 > 0.0)) {
        st.status = static_cast<int>(tdev::StreamlineStatus::seed_backflow);
        return;
    }

    int n = 0;
    double Tn = times[0];
    double h = 0.1 * opt.h_max;

    while (true) {
        double y5[D], k7[D], err = 0.0;
        code = dp5_step<D>(rhs, s, s + h, y, f, h, 3, opt.tol, y5, k7, err);
        if (code != EvalCode::good) {
            st.status = tdev::status_from(code);
            st.final_s = s;
            return;
        }
        const double g1_end = rhs.last_g1;
        const double g1_hat_end = rhs.last_g1_hat;
        const double fac = tdev::step_factor(err);

        if (!(err <= 1.0)) {
            ++st.rejected_steps;
            h = h * fac;
            if (h < opt.min_step) {
                st.status = static_cast<int>(tdev::StreamlineStatus::step_underflow);
                st.final_s = s;
                return;
            }
            continue;
        }

        const bool crossing = y5[3] >= Tn;
        if (crossing) {
            double ya[D];
            ya[0] = y[0];
            ya[1] = y[1];
            ya[2] = y[2];
            ya[3] = s;
            for (int i = 4; i < D; ++i)
                ya[i] = y[i];
            double yl[D];
            const LandingOutcome lo = land_on_time<D>(field, opt, y[3], ya, Tn, yl, st);
            if (lo == LandingOutcome::invalid_direction || lo == LandingOutcome::backflow) {
                st.status = static_cast<int>(tdev::StreamlineStatus::stagnation);
                st.final_s = s;
                return;
            }
            if (lo == LandingOutcome::invalid_conductivity) {
                st.status = static_cast<int>(tdev::StreamlineStatus::invalid_conductivity);
                st.final_s = s;
                return;
            }
            if (lo == LandingOutcome::underflow) {
                st.status = static_cast<int>(tdev::StreamlineStatus::landing_failed);
                st.final_s = s;
                return;
            }
            if (yl[3] - 0.0 > opt.max_arclength) {
                st.status = static_cast<int>(tdev::StreamlineStatus::cap_exceeded);
                st.final_s = s;
                return;
            }
            y[0] = yl[0];
            y[1] = yl[1];
            y[2] = yl[2];
            y[3] = Tn; // SET, never accumulated
            for (int i = 4; i < D; ++i)
                y[i] = yl[i];
            s = yl[3];
            out.x[3 * n] = y[0];
            out.x[3 * n + 1] = y[1];
            out.x[3 * n + 2] = y[2];
            out.I[n] = y[4];
            if constexpr (D > 5) {
                if (out.cov != nullptr) {
                    for (int i = 0; i < 6; ++i)
                        out.cov[6 * n + i] = y[5 + i];
                }
            }
            st.n_reached = n + 1;
            st.final_s = s;
            code = rhs(s, y, f);
            if (code != EvalCode::good) {
                st.status = tdev::status_from(code);
                return;
            }
            st.min_g1_hat = tdev::hmin(st.min_g1_hat, rhs.last_g1_hat);
            if (!(rhs.last_g1 > 0.0)) {
                st.backflow_encounter = 1;
            }
            if (n == T - 1) {
                st.status = static_cast<int>(tdev::StreamlineStatus::ok);
                return;
            }
            ++n;
            Tn = times[n];
            h = tdev::hmin(opt.h_max, h * fac);
            continue;
        }

        ++st.accepted_steps;
        for (int i = 0; i < D; ++i) {
            y[i] = y5[i];
            f[i] = k7[i];
        }
        s += h;
        st.min_g1_hat = tdev::hmin(st.min_g1_hat, g1_hat_end);
        if (!(g1_end > 0.0)) {
            st.backflow_encounter = 1;
        }
        if (s - 0.0 > opt.max_arclength) {
            st.status = static_cast<int>(tdev::StreamlineStatus::cap_exceeded);
            st.final_s = s;
            return;
        }
        h = tdev::hmin(opt.h_max, h * fac);
    }
}

// ------------------------------------------------------------------------------------------
// Kernels (one thread per particle)
// ------------------------------------------------------------------------------------------

struct KernelArgs {
    int Np = 0;
    const double* seeds = nullptr; ///< Np * 3
    const double* planes = nullptr;
    int P = 0;
    const double* times = nullptr;
    int T = 0;
    PassOptions opt{};
    PassStats* stats_plane = nullptr; ///< Np
    PassStats* stats_time = nullptr;  ///< Np
    double* planes_x23 = nullptr;     ///< Np * P * 2
    double* planes_tau = nullptr;     ///< Np * P
    double* planes_I = nullptr;       ///< Np * P
    double* planes_cov = nullptr;     ///< Np * P * 6 or nullptr
    double* times_x = nullptr;        ///< Np * T * 3
    double* times_I = nullptr;        ///< Np * T
    double* times_cov = nullptr;      ///< Np * T * 6 or nullptr
};

template <int D, class Field>
__global__ void plane_kernel(const Field field, const KernelArgs a) {
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= a.Np) {
        return;
    }
    const std::size_t Pz = static_cast<std::size_t>(a.P);
    const std::size_t pz = static_cast<std::size_t>(p);
    for (std::size_t m = 0; m < Pz; ++m) {
        a.planes_x23[(pz * Pz + m) * 2] = dnan();
        a.planes_x23[(pz * Pz + m) * 2 + 1] = dnan();
        a.planes_tau[pz * Pz + m] = dnan();
        a.planes_I[pz * Pz + m] = dnan();
        if (a.planes_cov != nullptr) {
            for (int i = 0; i < 6; ++i)
                a.planes_cov[(pz * Pz + m) * 6 + i] = dnan();
        }
    }
    PassStats st;
    const double seed[3] = {a.seeds[3 * pz], a.seeds[3 * pz + 1], a.seeds[3 * pz + 2]};
    PlaneOut out{a.planes_x23 + pz * Pz * 2, a.planes_tau + pz * Pz, a.planes_I + pz * Pz,
                 a.planes_cov != nullptr ? a.planes_cov + pz * Pz * 6 : nullptr};
    plane_pass<D>(field, seed, a.planes, a.P, a.opt, out, st);
    a.stats_plane[p] = st;
}

template <int D, class Field>
__global__ void time_kernel(const Field field, const KernelArgs a) {
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= a.Np) {
        return;
    }
    const std::size_t Tz = static_cast<std::size_t>(a.T);
    const std::size_t pz = static_cast<std::size_t>(p);
    for (std::size_t n = 0; n < Tz; ++n) {
        a.times_x[(pz * Tz + n) * 3] = dnan();
        a.times_x[(pz * Tz + n) * 3 + 1] = dnan();
        a.times_x[(pz * Tz + n) * 3 + 2] = dnan();
        a.times_I[pz * Tz + n] = dnan();
        if (a.times_cov != nullptr) {
            for (int i = 0; i < 6; ++i)
                a.times_cov[(pz * Tz + n) * 6 + i] = dnan();
        }
    }
    PassStats st;
    const double seed[3] = {a.seeds[3 * pz], a.seeds[3 * pz + 1], a.seeds[3 * pz + 2]};
    TimeOut out{a.times_x + pz * Tz * 3, a.times_I + pz * Tz,
                a.times_cov != nullptr ? a.times_cov + pz * Tz * 6 : nullptr};
    time_pass<D>(field, seed, a.times, a.T, a.opt, out, st);
    a.stats_time[p] = st;
}

/// Raw flux k g1 at the seeds (injection weights, before clamping/normalization) and the labels'
/// input is separate. Writes NaN when the evaluation is not valid.
template <class Field>
__global__ void flux_kernel(const Field field, int Np, const double* seeds, double* flux) {
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= Np) {
        return;
    }
    const double x[3] = {seeds[3 * p], seeds[3 * p + 1], seeds[3 * p + 2]};
    double g[3] = {0.0, 0.0, 0.0}, k = 0.0;
    field.eval(x, g, k);
    flux[p] = k * g[0];
}

} // namespace ldev
} // namespace lagrangian_labels
} // namespace macroflow3d
