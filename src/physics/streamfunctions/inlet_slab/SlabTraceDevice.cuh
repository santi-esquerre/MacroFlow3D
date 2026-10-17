#pragma once

/**
 * @file SlabTraceDevice.cuh
 * @brief Device integrator namespace `trace_dev`: device copies of the frozen SF-30 routines
 *        (closure_gate DP5(4) step, Henon landing on a plane, trace_to_plane).
 *
 * Moved verbatim from SlabTraceGpu.cu (SF-36 N1) so that the SF-36 forward Lagrangian tracker can
 * reuse them; SF-35 outputs must stay byte-identical.
 *
 * NUMERICAL CONTRACT: any translation unit that includes this header and compiles device code
 * from it MUST be compiled with `--fmad=false` (as SlabTraceGpu.cu is, see CMakeLists.txt) so that
 * device multiplications/additions are IEEE-rounded exactly like the host SF-30 integrator.
 * The functions are `__device__` templates/inlines, so there is no ODR issue when several
 * translation units include this header.
 */

#include "apps/closure_gate/streamline_integrator.hpp"

#include <cmath>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace trace_dev {

using closure_gate::StreamlineStatus;

// Device copy of closure_gate::detail::DP5 (same constexpr expressions).
struct DP5 {
    static constexpr double c2 = 1.0 / 5.0, c3 = 3.0 / 10.0, c4 = 4.0 / 5.0, c5 = 8.0 / 9.0;
    static constexpr double a21 = 1.0 / 5.0;
    static constexpr double a31 = 3.0 / 40.0, a32 = 9.0 / 40.0;
    static constexpr double a41 = 44.0 / 45.0, a42 = -56.0 / 15.0, a43 = 32.0 / 9.0;
    static constexpr double a51 = 19372.0 / 6561.0, a52 = -25360.0 / 2187.0,
                            a53 = 64448.0 / 6561.0, a54 = -212.0 / 729.0;
    static constexpr double a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0, a63 = 46732.0 / 5247.0,
                            a64 = 49.0 / 176.0, a65 = -5103.0 / 18656.0;
    static constexpr double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0,
                            b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;
    static constexpr double e1 = 71.0 / 57600.0, e3 = -71.0 / 16695.0, e4 = 71.0 / 1920.0,
                            e5 = -17253.0 / 339200.0, e6 = 22.0 / 525.0, e7 = -1.0 / 40.0;
};

enum class EvalCode { good, invalid_direction, invalid_conductivity, nonpositive_g1 };
enum class LandingOutcome { landed, backflow, invalid_direction, invalid_conductivity, underflow };

constexpr int kDim = 4;

/// Device mirror of the StreamlineResult fields used by the oracle.
struct DevResult {
    int status = 0;
    int periods_completed = 0;
    bool backflow_encounter = false;
    double min_g1_hat = 0.0;
    double backflow_arclength = 0.0;
    long long accepted_steps = 0, rejected_steps = 0, landing_steps = 0, landing_rejected = 0;
    long long landings_discarded = 0, field_evaluations = 0;
    double rec_x2 = 0.0, rec_x3 = 0.0; ///< landed (x2, x3) at the plane
};

__device__ inline bool finite3(const double g[3]) {
    return isfinite(g[0]) && isfinite(g[1]) && isfinite(g[2]);
}

// std::max(a, b) = (a < b) ? b : a ; std::min(a, b) = (b < a) ? b : a (host semantics).
__device__ inline double hmax(double a, double b) {
    return (a < b) ? b : a;
}
__device__ inline double hmin(double a, double b) {
    return (b < a) ? b : a;
}

template <class Field> struct ArclengthRhs {
    const Field& field;
    double sigma;
    long long& nfev;
    double last_g1 = 0.0;
    double last_g1_hat = 0.0;

    __device__ EvalCode operator()(double /*s*/, const double y[kDim], double dy[kDim]) {
        const double x[3] = {y[0], y[1], y[2]};
        double g[3] = {0.0, 0.0, 0.0};
        double k = 0.0;
        field(x, g, k);
        ++nfev;
        if (!finite3(g)) {
            return EvalCode::invalid_direction;
        }
        const double gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (!(gn > 0.0) || !isfinite(gn)) {
            return EvalCode::invalid_direction;
        }
        if (!isfinite(k) || !(k > 0.0)) {
            return EvalCode::invalid_conductivity;
        }
        dy[0] = sigma * g[0] / gn;
        dy[1] = sigma * g[1] / gn;
        dy[2] = sigma * g[2] / gn;
        dy[3] = 1.0 / (k * gn);
        last_g1 = g[0];
        last_g1_hat = g[0] / gn;
        return EvalCode::good;
    }
};

template <class Field> struct LandingRhs {
    const Field& field;
    double sigma;
    long long& nfev;

    __device__ EvalCode operator()(double x1, const double y[kDim], double dy[kDim]) {
        const double x[3] = {x1, y[0], y[1]};
        double g[3] = {0.0, 0.0, 0.0};
        double k = 0.0;
        field(x, g, k);
        ++nfev;
        if (!finite3(g)) {
            return EvalCode::invalid_direction;
        }
        const double gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (!(gn > 0.0) || !isfinite(gn)) {
            return EvalCode::invalid_direction;
        }
        if (!isfinite(k) || !(k > 0.0)) {
            return EvalCode::invalid_conductivity;
        }
        if (!(g[0] > 0.0)) {
            return EvalCode::nonpositive_g1;
        }
        dy[0] = g[1] / g[0];
        dy[1] = g[2] / g[0];
        dy[2] = sigma * gn / g[0];
        dy[3] = sigma / (k * g[0]);
        return EvalCode::good;
    }
};

template <class Rhs>
__device__ EvalCode dp5_step(Rhs& rhs, double t, double t_end, const double y[kDim],
                             const double k1[kDim], double h, int n_err, double tol,
                             double y5[kDim], double k7[kDim], double& err) {
    using C = DP5;
    double k2[kDim], k3[kDim], k4[kDim], k5[kDim], k6[kDim], yt[kDim];
    EvalCode code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a21 * k1[i]);
    if ((code = rhs(t + C::c2 * h, yt, k2)) != EvalCode::good)
        return code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a31 * k1[i] + C::a32 * k2[i]);
    if ((code = rhs(t + C::c3 * h, yt, k3)) != EvalCode::good)
        return code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a41 * k1[i] + C::a42 * k2[i] + C::a43 * k3[i]);
    if ((code = rhs(t + C::c4 * h, yt, k4)) != EvalCode::good)
        return code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a51 * k1[i] + C::a52 * k2[i] + C::a53 * k3[i] + C::a54 * k4[i]);
    if ((code = rhs(t + C::c5 * h, yt, k5)) != EvalCode::good)
        return code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a61 * k1[i] + C::a62 * k2[i] + C::a63 * k3[i] + C::a64 * k4[i] +
                            C::a65 * k5[i]);
    if ((code = rhs(t_end, yt, k6)) != EvalCode::good)
        return code;

    for (int i = 0; i < kDim; ++i)
        y5[i] = y[i] + h * (C::b1 * k1[i] + C::b3 * k3[i] + C::b4 * k4[i] + C::b5 * k5[i] +
                            C::b6 * k6[i]);
    if ((code = rhs(t_end, y5, k7)) != EvalCode::good)
        return code;

    err = 0.0;
    for (int i = 0; i < n_err; ++i) {
        const double e = h * (C::e1 * k1[i] + C::e3 * k3[i] + C::e4 * k4[i] + C::e5 * k5[i] +
                              C::e6 * k6[i] + C::e7 * k7[i]);
        err = hmax(err, fabs(e) / tol);
    }
    return EvalCode::good;
}

__device__ inline double step_factor(double err) {
    if (!(err > 0.0)) {
        return 5.0;
    }
    return hmin(5.0, hmax(0.2, 0.9 * pow(err, -0.2)));
}

__device__ inline int status_from(EvalCode c) {
    return c == EvalCode::invalid_conductivity
               ? static_cast<int>(StreamlineStatus::invalid_conductivity)
               : static_cast<int>(StreamlineStatus::stagnation);
}

__device__ inline LandingOutcome map_code(EvalCode c) {
    switch (c) {
    case EvalCode::nonpositive_g1:
        return LandingOutcome::backflow;
    case EvalCode::invalid_conductivity:
        return LandingOutcome::invalid_conductivity;
    default:
        return LandingOutcome::invalid_direction;
    }
}

template <class Field>
__device__ LandingOutcome land_on_plane(const Field& field, const closure_gate::IntegratorOptions& opt,
                                        double x1_a, const double ya[kDim], double X,
                                        double y_out[kDim], DevResult& res) {
    LandingRhs<Field> rhs{field, static_cast<double>(opt.sigma), res.field_evaluations};
    double t = x1_a;
    double y[kDim] = {ya[0], ya[1], ya[2], ya[3]};
    double k1[kDim];
    EvalCode code = rhs(t, y, k1);
    if (code != EvalCode::good) {
        return map_code(code);
    }
    const double dir = (X - t) >= 0.0 ? 1.0 : -1.0;
    double dt_mag = hmin(opt.h_max, fabs(X - t));
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
        double y5[kDim], k7[kDim], err = 0.0;
        code = dp5_step(rhs, t, t_end, y, k1, dt, 2, opt.tol, y5, k7, err);
        if (code != EvalCode::good) {
            return map_code(code);
        }
        const double fac = step_factor(err);
        if (err <= 1.0) {
            ++res.landing_steps;
            for (int i = 0; i < kDim; ++i) {
                y[i] = y5[i];
                k1[i] = k7[i];
            }
            if (last) {
                for (int i = 0; i < kDim; ++i)
                    y_out[i] = y[i];
                return LandingOutcome::landed;
            }
            t = t_end;
            dt_mag = hmin(opt.h_max, fabs(dt) * fac);
        } else {
            ++res.landing_rejected;
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

/// Device copy of inlet_slab::trace_to_plane (options pre-validated on the host).
template <class Field>
__device__ DevResult trace_to_plane(const Field& field, const double seed[3], double X,
                                    const closure_gate::IntegratorOptions& opt) {
    DevResult res;
    const double sigma = static_cast<double>(opt.sigma);
    ArclengthRhs<Field> rhs{field, sigma, res.field_evaluations};

    double y[kDim] = {seed[0], seed[1], seed[2], 0.0};
    double s = 0.0;
    double f[kDim];

    EvalCode code = rhs(0.0, y, f);
    if (code != EvalCode::good) {
        res.status = status_from(code);
        return res;
    }
    res.min_g1_hat = rhs.last_g1_hat;
    if (!(rhs.last_g1 > 0.0)) {
        res.status = static_cast<int>(StreamlineStatus::seed_backflow);
        return res;
    }

    const double s_plane = 0.0;
    double h = 0.1 * opt.h_max;

    while (true) {
        double y5[kDim], k7[kDim], err = 0.0;
        code = dp5_step(rhs, s, s + h, y, f, h, 3, opt.tol, y5, k7, err);
        if (code != EvalCode::good) {
            res.status = status_from(code);
            return res;
        }
        const double g1_end = rhs.last_g1;
        const double g1_hat_end = rhs.last_g1_hat;
        const double fac = step_factor(err);

        if (!(err <= 1.0)) {
            ++res.rejected_steps;
            h = h * fac;
            if (h < opt.min_step) {
                res.status = static_cast<int>(StreamlineStatus::step_underflow);
                return res;
            }
            continue;
        }

        const bool crossing = sigma * (y5[0] - X) >= 0.0;
        if (crossing) {
            const double ya[kDim] = {y[1], y[2], s, y[3]};
            double yl[kDim];
            const LandingOutcome out = land_on_plane(field, opt, y[0], ya, X, yl, res);
            if (out == LandingOutcome::backflow) {
                ++res.landings_discarded;
                h = 0.5 * h;
                if (h < opt.min_step) {
                    res.status = static_cast<int>(StreamlineStatus::landing_failed);
                    return res;
                }
                continue;
            }
            if (out == LandingOutcome::invalid_direction) {
                res.status = static_cast<int>(StreamlineStatus::stagnation);
                return res;
            }
            if (out == LandingOutcome::invalid_conductivity) {
                res.status = static_cast<int>(StreamlineStatus::invalid_conductivity);
                return res;
            }
            if (out == LandingOutcome::underflow) {
                res.status = static_cast<int>(StreamlineStatus::landing_failed);
                return res;
            }
            if (yl[2] - s_plane > opt.max_arclength_per_period) {
                res.status = static_cast<int>(StreamlineStatus::cap_exceeded);
                return res;
            }
            y[0] = X;
            y[1] = yl[0];
            y[2] = yl[1];
            y[3] = yl[3];
            s = yl[2];
            res.rec_x2 = y[1];
            res.rec_x3 = y[2];
            res.periods_completed = 1;
            code = rhs(s, y, f);
            if (code != EvalCode::good) {
                res.status = status_from(code);
                return res;
            }
            res.min_g1_hat = hmin(res.min_g1_hat, rhs.last_g1_hat);
            if (!(rhs.last_g1 > 0.0)) {
                res.backflow_encounter = true;
            }
            res.status = static_cast<int>(StreamlineStatus::ok);
            return res;
        }

        ++res.accepted_steps;
        for (int i = 0; i < kDim; ++i) {
            y[i] = y5[i];
            f[i] = k7[i];
        }
        s += h;
        res.min_g1_hat = hmin(res.min_g1_hat, g1_hat_end);
        if (!(g1_end > 0.0)) {
            res.backflow_encounter = true;
            res.backflow_arclength += h;
        }
        if (s - s_plane > opt.max_arclength_per_period) {
            res.status = static_cast<int>(StreamlineStatus::cap_exceeded);
            return res;
        }
        h = hmin(opt.h_max, h * fac);
    }
}

} // namespace trace_dev

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
