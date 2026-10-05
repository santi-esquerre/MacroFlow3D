#pragma once

/**
 * @file streamline_integrator.hpp
 * @brief SF-30 oracle streamline integrator: first-return map of the face
 *        `x1 = x1_0` under a direction field `g(x)` (header-only, host-only
 *        C++17; no CUDA include).
 *
 * This is the label-independent instrument of the SF-30 streamline-closure
 * gate. It integrates the integral curves of a direction field `g` (for the
 * SF-19 Darcy flow `g = G + grad h_tilde`, `v = K g`) and records where each
 * curve returns to the planes `x1 = x1_0 + sigma * n`. It is deliberately
 * independent of any streamfunction/label construction and of the future
 * tracker (an oracle must not share the tracker's construction).
 *
 * ## Field functor contract
 *
 * The integrator is a template on a callable `field` with the signature
 *
 *   void field(const double x[3], double g[3], double& k) const;
 *
 * which, at the UNWRAPPED position `x` (any finite reals; the functor owns
 * periodicity), writes the direction field `g` and a conductivity `k`
 * (strictly positive; analytic test fields may return 1). The functor must
 * be callable concurrently from several threads (const, no shared mutable
 * state). The integrator never reduces coordinates.
 *
 * ## Curves and method
 *
 * Arclength form, direction `sigma = +1` (forward) or `-1` (backward):
 *
 *   dx/ds   = sigma * g / |g|
 *   dtau/ds = 1 / (k |g|)            (travel time; porosity 1)
 *
 * Dormand-Prince 5(4) with FSAL (standard coefficients), adaptive step:
 *   - error norm `err = max_{i=1..3} |x5_i - x4_i| / tol`, ONE absolute
 *     tolerance `tol` on the positions (units of the period, L = 1);
 *     arclength and travel time use the same steps and do NOT enter the
 *     step control;
 *   - accept iff `err <= 1`; `h_new = h * clamp(0.9 err^(-1/5), 0.2, 5)`
 *     (`err = 0` -> factor 5), `h_new <= h_max`; first step `0.1 h_max`;
 *   - no regularization anywhere: `|g|` is never floored, no epsilon added.
 *
 * ## First return (Henon's device)
 *
 * Return planes `X_n = x1_0 + sigma * n`, `n = 1..n_periods`, on the
 * unwrapped `x1`. An accepted arclength step that takes `x1` from strictly
 * before `X_n` to at-or-beyond it (in the direction `sigma`) is NOT kept;
 * instead the curve is landed on the plane from the pre-crossing point with
 * `x1` as the independent variable:
 *
 *   d(x2)/dx1 = g2/g1,  d(x3)/dx1 = g3/g1,
 *   ds/dx1 = sigma |g|/g1,  dtau/dx1 = sigma / (k g1)
 *
 * integrated with the same DP5(4) and `tol` on `(x2, x3)` (sub-steps
 * `<= h_max`), the last sub-step ending exactly at `X_n` (the landed `x1` is
 * SET to `X_n`, never accumulated). The landing requires `g1 > 0` at every
 * stage; otherwise it is discarded, the arclength step is halved from the
 * pre-crossing point and the arclength integration continues. FSAL is
 * invalidated by a landing: the field is re-evaluated at the landed point.
 *
 * ## Statuses and counters
 *
 * See `StreamlineStatus` and `StreamlineResult`. Records made at earlier
 * planes stay valid when a later period fails (`periods_completed`).
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace macroflow3d {
namespace closure_gate {

/// Terminal status of one streamline.
enum class StreamlineStatus : int {
    ok = 0,              ///< all requested periods landed
    seed_backflow = 1,   ///< g1 <= 0 at the seed: not in the domain of the section map; not integrated
    cap_exceeded = 2,    ///< arclength since the previous plane exceeded max_arclength_per_period
    stagnation = 3,      ///< |g| zero or a component of g non-finite at some stage
    step_underflow = 4,  ///< arclength step fell below min_step outside a landing
    landing_failed = 5,  ///< landing could not be completed above min_step (tangency)
    invalid_conductivity = 6, ///< k non-finite or k <= 0 at some stage (functor contract violation)
};

inline const char* to_string(StreamlineStatus s) {
    switch (s) {
    case StreamlineStatus::ok: return "ok";
    case StreamlineStatus::seed_backflow: return "seed_backflow";
    case StreamlineStatus::cap_exceeded: return "cap_exceeded";
    case StreamlineStatus::stagnation: return "stagnation";
    case StreamlineStatus::step_underflow: return "step_underflow";
    case StreamlineStatus::landing_failed: return "landing_failed";
    case StreamlineStatus::invalid_conductivity: return "invalid_conductivity";
    }
    return "unknown";
}

/// Integrator options. `h_max` has no default on purpose (0 is rejected):
/// the caller must pass it explicitly (the closure gate passes the grid
/// spacing so that a step never spans more than one spline cell).
struct IntegratorOptions {
    double tol = 1e-8;                       ///< absolute position tolerance (L = 1)
    double h_max = 0.0;                      ///< maximum step (arclength and landing x1); REQUIRED
    double min_step = 1e-13;                 ///< underflow threshold of the step
    double max_arclength_per_period = 100.0; ///< cap on arclength between consecutive planes
    int n_periods = 1;                       ///< number of return planes to reach
    int sigma = +1;                          ///< +1 forward (along g), -1 backward (along -g)
};

/// Landed state at one return plane.
struct PlaneRecord {
    double x1 = 0.0;  ///< exactly x1_0 + sigma * n
    double x2 = 0.0;  ///< unwrapped
    double x3 = 0.0;  ///< unwrapped
    double s = 0.0;   ///< arclength from the seed
    double tau = 0.0; ///< travel time from the seed
};

/// Result of one streamline. Counters include every period attempted.
struct StreamlineResult {
    StreamlineStatus status = StreamlineStatus::ok;
    int periods_completed = 0;
    std::vector<PlaneRecord> records; ///< size == periods_completed

    bool backflow_encounter = false;  ///< any accepted point (seed, step end, landed point) with g1 <= 0
    double min_g1_hat = 0.0;          ///< min of g1/|g| over accepted points (incl. the seed)
    double backflow_arclength = 0.0;  ///< sum of accepted arclength steps whose end point has g1 <= 0

    long long accepted_steps = 0;     ///< accepted arclength steps
    long long rejected_steps = 0;     ///< rejected arclength steps (err > 1)
    long long landing_steps = 0;      ///< accepted landing sub-steps
    long long landing_rejected = 0;   ///< rejected landing sub-steps (err > 1)
    long long landings_discarded = 0; ///< landings discarded because g1 <= 0 at a stage
    long long field_evaluations = 0;  ///< every functor call

    double final_x[3] = {0.0, 0.0, 0.0}; ///< last accepted/landed state
    double final_s = 0.0;
    double final_tau = 0.0;
};

namespace detail {

// Dormand-Prince 5(4) coefficients (Dormand & Prince 1980; Hairer, Norsett,
// Wanner, "Solving ODEs I", Table 5.2).
struct DP5 {
    static constexpr double c2 = 1.0 / 5.0, c3 = 3.0 / 10.0, c4 = 4.0 / 5.0, c5 = 8.0 / 9.0;
    static constexpr double a21 = 1.0 / 5.0;
    static constexpr double a31 = 3.0 / 40.0, a32 = 9.0 / 40.0;
    static constexpr double a41 = 44.0 / 45.0, a42 = -56.0 / 15.0, a43 = 32.0 / 9.0;
    static constexpr double a51 = 19372.0 / 6561.0, a52 = -25360.0 / 2187.0,
                            a53 = 64448.0 / 6561.0, a54 = -212.0 / 729.0;
    static constexpr double a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0, a63 = 46732.0 / 5247.0,
                            a64 = 49.0 / 176.0, a65 = -5103.0 / 18656.0;
    // 5th-order weights (= row 7, FSAL).
    static constexpr double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0,
                            b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;
    // Error weights e = b(5th) - b(4th).
    static constexpr double e1 = 71.0 / 57600.0, e3 = -71.0 / 16695.0, e4 = 71.0 / 1920.0,
                            e5 = -17253.0 / 339200.0, e6 = 22.0 / 525.0, e7 = -1.0 / 40.0;
};

enum class EvalCode { good, invalid_direction, invalid_conductivity, nonpositive_g1 };

constexpr int kDim = 4;

inline bool finite3(const double g[3]) {
    return std::isfinite(g[0]) && std::isfinite(g[1]) && std::isfinite(g[2]);
}

// Arclength-form RHS; state y = (x1, x2, x3, tau), independent variable s.
template <class Field>
struct ArclengthRhs {
    const Field& field;
    double sigma;
    long long& nfev;
    double last_g1 = 0.0;
    double last_g1_hat = 0.0;

    EvalCode operator()(double /*s*/, const double y[kDim], double dy[kDim]) {
        const double x[3] = {y[0], y[1], y[2]};
        double g[3] = {0.0, 0.0, 0.0};
        double k = 0.0;
        field(x, g, k);
        ++nfev;
        if (!finite3(g)) {
            return EvalCode::invalid_direction;
        }
        const double gn = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (!(gn > 0.0) || !std::isfinite(gn)) {
            return EvalCode::invalid_direction;
        }
        if (!std::isfinite(k) || !(k > 0.0)) {
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

// Landing RHS with x1 as independent variable; state y = (x2, x3, s, tau).
template <class Field>
struct LandingRhs {
    const Field& field;
    double sigma;
    long long& nfev;

    EvalCode operator()(double x1, const double y[kDim], double dy[kDim]) {
        const double x[3] = {x1, y[0], y[1]};
        double g[3] = {0.0, 0.0, 0.0};
        double k = 0.0;
        field(x, g, k);
        ++nfev;
        if (!finite3(g)) {
            return EvalCode::invalid_direction;
        }
        const double gn = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (!(gn > 0.0) || !std::isfinite(gn)) {
            return EvalCode::invalid_direction;
        }
        if (!std::isfinite(k) || !(k > 0.0)) {
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

// One DP5(4) step from (t, y) with FSAL derivative k1 and step h. The last
// stage is evaluated at `t_end` (== t + h, passed explicitly so that a
// landing's final stage sits EXACTLY on the plane). The error is the max
// over the first `n_err` components of |y5 - y4| / tol.
template <class Rhs>
EvalCode dp5_step(Rhs& rhs, double t, double t_end, const double y[kDim],
                  const double k1[kDim], double h, int n_err, double tol, double y5[kDim],
                  double k7[kDim], double& err) {
    using C = DP5;
    double k2[kDim], k3[kDim], k4[kDim], k5[kDim], k6[kDim], yt[kDim];
    EvalCode code;

    for (int i = 0; i < kDim; ++i) yt[i] = y[i] + h * (C::a21 * k1[i]);
    if ((code = rhs(t + C::c2 * h, yt, k2)) != EvalCode::good) return code;

    for (int i = 0; i < kDim; ++i) yt[i] = y[i] + h * (C::a31 * k1[i] + C::a32 * k2[i]);
    if ((code = rhs(t + C::c3 * h, yt, k3)) != EvalCode::good) return code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a41 * k1[i] + C::a42 * k2[i] + C::a43 * k3[i]);
    if ((code = rhs(t + C::c4 * h, yt, k4)) != EvalCode::good) return code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a51 * k1[i] + C::a52 * k2[i] + C::a53 * k3[i] + C::a54 * k4[i]);
    if ((code = rhs(t + C::c5 * h, yt, k5)) != EvalCode::good) return code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a61 * k1[i] + C::a62 * k2[i] + C::a63 * k3[i] + C::a64 * k4[i] +
                            C::a65 * k5[i]);
    if ((code = rhs(t_end, yt, k6)) != EvalCode::good) return code;

    for (int i = 0; i < kDim; ++i)
        y5[i] = y[i] + h * (C::b1 * k1[i] + C::b3 * k3[i] + C::b4 * k4[i] + C::b5 * k5[i] +
                            C::b6 * k6[i]);
    if ((code = rhs(t_end, y5, k7)) != EvalCode::good) return code;

    err = 0.0;
    for (int i = 0; i < n_err; ++i) {
        const double e = h * (C::e1 * k1[i] + C::e3 * k3[i] + C::e4 * k4[i] + C::e5 * k5[i] +
                              C::e6 * k6[i] + C::e7 * k7[i]);
        err = std::max(err, std::abs(e) / tol);
    }
    return EvalCode::good;
}

inline double step_factor(double err) {
    if (!(err > 0.0)) {
        return 5.0; // err == 0 guard (NaN never reaches here: derivatives are validated finite)
    }
    return std::min(5.0, std::max(0.2, 0.9 * std::pow(err, -0.2)));
}

inline StreamlineStatus status_from(EvalCode c) {
    return c == EvalCode::invalid_conductivity ? StreamlineStatus::invalid_conductivity
                                               : StreamlineStatus::stagnation;
}

enum class LandingOutcome { landed, backflow, invalid_direction, invalid_conductivity, underflow };

// Lands from (x1_a, ya = (x2, x3, s, tau)) onto x1 = X. Writes the landed
// (x2, x3, s, tau) into y_out on success.
template <class Field>
LandingOutcome land_on_plane(const Field& field, const IntegratorOptions& opt, double x1_a,
                             const double ya[kDim], double X, double y_out[kDim],
                             StreamlineResult& res) {
    LandingRhs<Field> rhs{field, static_cast<double>(opt.sigma), res.field_evaluations};
    double t = x1_a;
    double y[kDim] = {ya[0], ya[1], ya[2], ya[3]};
    double k1[kDim];
    EvalCode code = rhs(t, y, k1);
    auto map_code = [](EvalCode c) {
        switch (c) {
        case EvalCode::nonpositive_g1: return LandingOutcome::backflow;
        case EvalCode::invalid_conductivity: return LandingOutcome::invalid_conductivity;
        default: return LandingOutcome::invalid_direction;
        }
    };
    if (code != EvalCode::good) {
        return map_code(code);
    }
    const double dir = (X - t) >= 0.0 ? 1.0 : -1.0;
    double dt_mag = std::min(opt.h_max, std::abs(X - t));
    long long guard = 0;
    while (true) {
        const double remaining = X - t;
        bool last = false;
        double dt = dir * dt_mag;
        if (dt_mag >= std::abs(remaining)) {
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
                for (int i = 0; i < kDim; ++i) y_out[i] = y[i];
                return LandingOutcome::landed;
            }
            t = t_end;
            dt_mag = std::min(opt.h_max, std::abs(dt) * fac);
        } else {
            ++res.landing_rejected;
            dt_mag = std::abs(dt) * fac;
            if (dt_mag < opt.min_step) {
                return LandingOutcome::underflow;
            }
        }
        if (++guard > 100000000LL) { // unreachable in practice; never silently loop forever
            return LandingOutcome::underflow;
        }
    }
}

inline void validate_options(const IntegratorOptions& opt) {
    if (!(opt.tol > 0.0) || !std::isfinite(opt.tol)) {
        throw std::invalid_argument("closure_gate: tol must be finite and > 0");
    }
    if (!(opt.h_max > 0.0) || !std::isfinite(opt.h_max) || !(opt.h_max < 1.0)) {
        throw std::invalid_argument(
            "closure_gate: h_max must be set explicitly, finite, 0 < h_max < 1 (period)");
    }
    if (!(opt.min_step > 0.0) || !(opt.min_step < opt.h_max)) {
        throw std::invalid_argument("closure_gate: min_step must satisfy 0 < min_step < h_max");
    }
    if (!(opt.max_arclength_per_period > 0.0)) {
        throw std::invalid_argument("closure_gate: max_arclength_per_period must be > 0");
    }
    if (opt.n_periods < 1) {
        throw std::invalid_argument("closure_gate: n_periods must be >= 1");
    }
    if (opt.sigma != 1 && opt.sigma != -1) {
        throw std::invalid_argument("closure_gate: sigma must be +1 or -1");
    }
}

} // namespace detail

/**
 * Integrates one streamline from `seed` (unwrapped) and records its returns
 * to the planes `x1 = seed[0] + sigma * n`, `n = 1..opt.n_periods`. Pure
 * function of (field, seed, opt): no shared state, deterministic.
 */
template <class Field>
StreamlineResult integrate_streamline(const Field& field, const std::array<double, 3>& seed,
                                      const IntegratorOptions& opt) {
    using namespace detail;
    validate_options(opt);
    if (!std::isfinite(seed[0]) || !std::isfinite(seed[1]) || !std::isfinite(seed[2])) {
        throw std::invalid_argument("closure_gate: seed coordinates must be finite");
    }

    StreamlineResult res;
    res.records.reserve(static_cast<std::size_t>(opt.n_periods));
    const double sigma = static_cast<double>(opt.sigma);
    ArclengthRhs<Field> rhs{field, sigma, res.field_evaluations};

    double y[kDim] = {seed[0], seed[1], seed[2], 0.0}; // (x1, x2, x3, tau)
    double s = 0.0;
    double f[kDim];

    auto finish = [&](StreamlineStatus st) {
        res.status = st;
        res.final_x[0] = y[0];
        res.final_x[1] = y[1];
        res.final_x[2] = y[2];
        res.final_s = s;
        res.final_tau = y[3];
        return res;
    };

    EvalCode code = rhs(0.0, y, f);
    if (code != EvalCode::good) {
        return finish(status_from(code));
    }
    res.min_g1_hat = rhs.last_g1_hat;
    if (!(rhs.last_g1 > 0.0)) {
        // Not in the domain of the section map (both directions use g1 itself).
        // No step is taken; min_g1_hat carries the seed's g1/|g| (<= 0).
        return finish(StreamlineStatus::seed_backflow);
    }

    const double x1_0 = seed[0];
    int n = 1;
    double X = x1_0 + sigma * static_cast<double>(n);
    double s_plane = 0.0;
    double h = 0.1 * opt.h_max;

    while (true) {
        double y5[kDim], k7[kDim], err = 0.0;
        code = dp5_step(rhs, s, s + h, y, f, h, 3, opt.tol, y5, k7, err);
        if (code != EvalCode::good) {
            return finish(status_from(code));
        }
        const double g1_end = rhs.last_g1;
        const double g1_hat_end = rhs.last_g1_hat;
        const double fac = step_factor(err);

        if (!(err <= 1.0)) {
            ++res.rejected_steps;
            h = h * fac;
            if (h < opt.min_step) {
                return finish(StreamlineStatus::step_underflow);
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
                    return finish(StreamlineStatus::landing_failed);
                }
                continue;
            }
            if (out == LandingOutcome::invalid_direction) {
                return finish(StreamlineStatus::stagnation);
            }
            if (out == LandingOutcome::invalid_conductivity) {
                return finish(StreamlineStatus::invalid_conductivity);
            }
            if (out == LandingOutcome::underflow) {
                return finish(StreamlineStatus::landing_failed);
            }
            // Landed: state is replaced by the landed point (x1 SET to X).
            if (yl[2] - s_plane > opt.max_arclength_per_period) {
                return finish(StreamlineStatus::cap_exceeded);
            }
            y[0] = X;
            y[1] = yl[0];
            y[2] = yl[1];
            y[3] = yl[3];
            s = yl[2];
            PlaneRecord rec;
            rec.x1 = X;
            rec.x2 = y[1];
            rec.x3 = y[2];
            rec.s = s;
            rec.tau = y[3];
            res.records.push_back(rec);
            res.periods_completed = n;
            // FSAL invalidated: re-evaluate the field at the landed point.
            code = rhs(s, y, f);
            if (code != EvalCode::good) {
                return finish(status_from(code));
            }
            res.min_g1_hat = std::min(res.min_g1_hat, rhs.last_g1_hat);
            if (!(rhs.last_g1 > 0.0)) {
                res.backflow_encounter = true;
            }
            if (n == opt.n_periods) {
                return finish(StreamlineStatus::ok);
            }
            ++n;
            X = x1_0 + sigma * static_cast<double>(n);
            s_plane = s;
            h = std::min(opt.h_max, h * fac);
            continue;
        }

        // Accept the arclength step.
        ++res.accepted_steps;
        for (int i = 0; i < kDim; ++i) {
            y[i] = y5[i];
            f[i] = k7[i];
        }
        s += h;
        res.min_g1_hat = std::min(res.min_g1_hat, g1_hat_end);
        if (!(g1_end > 0.0)) {
            res.backflow_encounter = true;
            res.backflow_arclength += h;
        }
        if (s - s_plane > opt.max_arclength_per_period) {
            return finish(StreamlineStatus::cap_exceeded);
        }
        h = std::min(opt.h_max, h * fac);
    }
}

/**
 * Integrates every seed with `n_threads` std::threads (atomic work index).
 * Result `i` belongs to seed `i` and is independent of the thread count and
 * of scheduling (each streamline is a pure function of its seed). An
 * exception thrown for any seed is rethrown after all threads joined.
 */
template <class Field>
std::vector<StreamlineResult> integrate_streamlines(const Field& field,
                                                    const std::vector<std::array<double, 3>>& seeds,
                                                    const IntegratorOptions& opt, int n_threads) {
    detail::validate_options(opt);
    if (n_threads < 1) {
        throw std::invalid_argument("closure_gate: n_threads must be >= 1");
    }
    std::vector<StreamlineResult> results(seeds.size());
    if (seeds.empty()) {
        return results;
    }
    const int workers =
        static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(n_threads), seeds.size()));
    std::atomic<std::size_t> next{0};
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(workers));
    auto work = [&](int w) {
        try {
            while (true) {
                const std::size_t i = next.fetch_add(1);
                if (i >= seeds.size()) {
                    break;
                }
                results[i] = integrate_streamline(field, seeds[i], opt);
            }
        } catch (...) {
            errors[static_cast<std::size_t>(w)] = std::current_exception();
        }
    };
    if (workers == 1) {
        work(0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(workers));
        for (int w = 0; w < workers; ++w) {
            pool.emplace_back(work, w);
        }
        for (auto& t : pool) {
            t.join();
        }
    }
    for (const auto& e : errors) {
        if (e) {
            std::rethrow_exception(e);
        }
    }
    return results;
}

} // namespace closure_gate
} // namespace macroflow3d
