#pragma once

/**
 * @file SlabOracle.cuh
 * @brief SF-33 N3: production oracle of the inlet-slab labels: backward streamline tracing of every
 *        slab vertex to the inlet plane x1 = 0 with the SF-30 integrator (host), D-1 inlet labels
 *        at the feet (GPU), and the forward round trip per plane.
 *
 * Authority: SF-33 specification item 7b and acceptance (d); understanding record section 5
 * ("Production oracle"); prototype `cases.compute_oracle` / `tracing.trace_to_inlet` of the SF-29
 * artifact. The oracle shares NOTHING with the solver: it is the solver-independent instrument
 * (labels transported from the inlet along the actual streamlines of the traced flow).
 *
 * trace_to_plane(field, seed, X, opt): a reproduction of closure_gate::integrate_streamline whose
 * single return plane is an arbitrary X (integrate_streamline only lands on x1_0 + sigma n). It
 * reuses closure_gate::detail::{ArclengthRhs, dp5_step, land_on_plane, step_factor, status_from,
 * validate_options} unchanged; the loop body is the integrate_streamline loop with n_periods = 1
 * and X given. Requirements: opt.n_periods == 1, X finite, sigma (X - seed[0]) > 0. For
 * X = seed[0] + sigma the result is bitwise identical to integrate_streamline (tested).
 *
 * compute_oracle: planes j = 1..N, every vertex (j/N, m2/N, m3/N) is traced backward (sigma = -1)
 * to X = 0; the feet (x2, x3) are unwrapped; each foot is re-traced forward (sigma = +1) to
 * X = j/N and the max-norm distance to the start vertex is the round trip. Plane 0 = the inlet
 * labels at the vertices. Labels at the feet / vertices by InletLabels::evaluate_labels (GPU).
 * Any non-`ok` streamline (backward or forward) flags its plane; its foot and labels are NaN
 * (nothing clamped) and the oracle status is oracle_roundtrip_fail. A plane whose round trip
 * exceeds opt.max_roundtrip (default +inf: disabled) is flagged the same way.
 *
 * Step bound h_max (SlabOracleOptions::h_max; default grid.h = the SF-30 convention). Measured
 * finding (tests/inlet_slab/slab_production_tests.cu, generic3d eps = 0.25, 16^3): the round trip
 * is limited by h_max, not by tol. The direction field is the gradient of a C^2 cubic spline, so
 * g'' jumps at the knots; the DP5(4) error estimate does not see those jumps and the error
 * committed by the steps that straddle a knot scales ~ h_max^3 (max round trip 1.3e-6 / 8.8e-8
 * / 6.6e-9 / 1.2e-9 / 1.6e-10 at h_max = h, h/4, h/8, h/16, h/32 with tol 1e-8; tol 1e-8 and 1e-10
 * give bitwise identical labels at h/16). The round trip therefore certifies the integration only
 * together with the h_max used; callers needing a round trip <= 1e-8 at coarse N must reduce
 * h_max explicitly (the choice is the caller's and must be logged).
 *
 * Threads: std::thread pool with an atomic work index (as closure_gate::integrate_streamlines);
 * each streamline is a pure function of its seed and the per-plane reductions run sequentially in
 * index order, so every output is bitwise independent of the thread count.
 *
 * Allocation (documented contract): the host tracing allocates only its result storage (the
 * per-seed foot / round-trip / counter vectors, allocated once per call, and the one-element
 * `records` vector of each StreamlineResult inside trace_to_plane); compute_oracle allocates the
 * device foot arrays (2 N^3 doubles) once per call. The GPU label kernel allocates nothing (see
 * InletLabels.cuh).
 */

#include "../../../core/DeviceSpan.cuh"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "InletLabels.cuh"
#include "InletSlabGrid.cuh"

#include "apps/closure_gate/streamline_integrator.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

/**
 * One streamline from `seed` (unwrapped) to its first crossing of the plane x1 = X in the
 * direction opt.sigma, landed exactly on X (Henon). Pure function of (field, seed, X, opt).
 */
template <class Field>
closure_gate::StreamlineResult trace_to_plane(const Field& field, const std::array<double, 3>& seed,
                                              double X,
                                              const closure_gate::IntegratorOptions& opt) {
    using namespace closure_gate;
    using namespace closure_gate::detail;
    validate_options(opt);
    if (opt.n_periods != 1) {
        throw std::invalid_argument("trace_to_plane: opt.n_periods must be 1 (single plane X)");
    }
    if (!std::isfinite(seed[0]) || !std::isfinite(seed[1]) || !std::isfinite(seed[2])) {
        throw std::invalid_argument("trace_to_plane: seed coordinates must be finite");
    }
    if (!std::isfinite(X) || !(static_cast<double>(opt.sigma) * (X - seed[0]) > 0.0)) {
        throw std::invalid_argument("trace_to_plane: X must be finite and strictly ahead of the "
                                    "seed (sigma (X - x1_0) > 0)");
    }

    StreamlineResult res;
    res.records.reserve(1);
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
        return finish(StreamlineStatus::seed_backflow);
    }

    const double s_plane = 0.0;
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
            res.periods_completed = 1;
            // FSAL invalidated: re-evaluate the field at the landed point (as
            // integrate_streamline).
            code = rhs(s, y, f);
            if (code != EvalCode::good) {
                return finish(status_from(code));
            }
            res.min_g1_hat = std::min(res.min_g1_hat, rhs.last_g1_hat);
            if (!(rhs.last_g1 > 0.0)) {
                res.backflow_encounter = true;
            }
            return finish(StreamlineStatus::ok);
        }

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

// ------------------------------------------------------------------------------------------------
// Oracle
// ------------------------------------------------------------------------------------------------

enum class SlabOracleStatus { ok = 0, oracle_roundtrip_fail = 1 };
const char* to_string(SlabOracleStatus s);

struct SlabOracleOptions {
    double tol = 1e-8;  ///< integrator position tolerance (both directions)
    double h_max = 0.0; ///< 0: grid.h (SF-30 convention); sets the round-trip level (header)
    double min_step = 1e-13;
    double max_arclength = 100.0;
    int threads = 1;
    double max_roundtrip = std::numeric_limits<double>::infinity(); ///< flag threshold (inf: off)
};

struct SlabOraclePlane {
    int j = 0;
    double x1 = 0.0;
    long long nfev = 0;     ///< backward field evaluations (sum over the plane)
    long long nfev_rt = 0;  ///< forward (round-trip) field evaluations
    double roundtrip = 0.0; ///< max-norm, over the plane's ok streamlines
    long long back_non_ok = 0;
    long long fwd_non_ok = 0;
    long long backflow_encounters = 0; ///< ok streamlines that met g1 <= 0 (informative)
    bool flagged = false;
};

struct SlabOracleResult {
    SlabOracleStatus status = SlabOracleStatus::ok;
    double tol = 0.0;
    std::vector<SlabOraclePlane> planes; ///< j = 0..N (plane 0: inlet data, nothing traced)
    std::vector<double> foot_y, foot_z;  ///< N^3, planes 1..N in the unknown layout (NaN if non-ok)
    std::vector<double> roundtrip;       ///< per seed (NaN if non-ok)
    double max_roundtrip = 0.0;
    long long non_ok = 0;
    double seconds_trace = 0.0;
    double seconds_labels = 0.0;
    /// Per-plane table (one line per plane) + a summary line.
    std::string table() const;
};

namespace detail {

/// Parallel for over [0, n) with an atomic work index; exceptions are rethrown after the join.
template <class Body> void oracle_parallel_for(std::size_t n, int threads, const Body& body) {
    if (threads < 1) {
        throw std::invalid_argument("compute_oracle: threads must be >= 1");
    }
    const int workers =
        static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(threads), n));
    if (workers <= 1) {
        for (std::size_t i = 0; i < n; ++i)
            body(i);
        return;
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(workers));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers));
    for (int w = 0; w < workers; ++w) {
        pool.emplace_back([&, w] {
            try {
                while (true) {
                    const std::size_t i = next.fetch_add(1);
                    if (i >= n)
                        break;
                    body(i);
                }
            } catch (...) {
                errors[static_cast<std::size_t>(w)] = std::current_exception();
            }
        });
    }
    for (auto& t : pool)
        t.join();
    for (const auto& e : errors)
        if (e)
            std::rethrow_exception(e);
}

/// GPU part: labels at the inlet vertices (plane 0) and at the feet (planes 1..N) into psi1/psi2
/// (full arrays). Allocates the device foot arrays once; one device synchronization after the
/// legacy-stream uploads (SF-33 C2) and one synchronization at the end.
void oracle_evaluate_labels(CudaContext& ctx, const InletSlabGrid& grid, const InletLabels& labels,
                            const std::vector<double>& foot_y, const std::vector<double>& foot_z,
                            DeviceSpan<real> psi1, DeviceSpan<real> psi2);

} // namespace detail

/**
 * Oracle labels psi_or (FULL labels, affine part included) on planes 0..N into psi1_out /
 * psi2_out ((N+1) N^2 each, device). `labels` must be device-prepared. Returns the per-plane table
 * and status; never clamps.
 */
template <class Field>
SlabOracleResult compute_oracle(CudaContext& ctx, const InletSlabGrid& grid, const Field& field,
                                const InletLabels& labels, const SlabOracleOptions& opt,
                                DeviceSpan<real> psi1_out, DeviceSpan<real> psi2_out) {
    require_valid_grid(grid, "compute_oracle");
    if (psi1_out.size() != grid.full_size() || psi2_out.size() != grid.full_size()) {
        throw std::invalid_argument("compute_oracle: psi outputs must have (N+1) N^2 entries");
    }
    if (!labels.device_prepared()) {
        throw std::logic_error("compute_oracle: inlet labels not device-prepared");
    }
    const int N = grid.n;
    const std::size_t n2 = grid.plane_size();
    const std::size_t total = grid.field_size();

    closure_gate::IntegratorOptions base;
    base.tol = opt.tol;
    base.h_max = opt.h_max > 0.0 ? opt.h_max : grid.h;
    base.min_step = opt.min_step;
    base.max_arclength_per_period = opt.max_arclength;
    base.n_periods = 1;
    closure_gate::detail::validate_options(base);

    SlabOracleResult R;
    R.tol = opt.tol;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    R.foot_y.assign(total, nan);
    R.foot_z.assign(total, nan);
    R.roundtrip.assign(total, nan);
    std::vector<long long> nfev_b(total, 0), nfev_f(total, 0);
    std::vector<std::uint8_t> st_b(total, 0), st_f(total, 0), bfe(total, 0);

    const auto t0 = std::chrono::steady_clock::now();
    detail::oracle_parallel_for(total, opt.threads, [&](std::size_t i) {
        const int j = static_cast<int>(i / n2) + 1;
        const std::size_t rem = i % n2;
        const int m2 = static_cast<int>(rem / static_cast<std::size_t>(N));
        const int m3 = static_cast<int>(rem % static_cast<std::size_t>(N));
        const std::array<double, 3> seed = {grid.coord(j), grid.coord(m2), grid.coord(m3)};
        closure_gate::IntegratorOptions ob = base;
        ob.sigma = -1;
        const closure_gate::StreamlineResult rb = trace_to_plane(field, seed, 0.0, ob);
        nfev_b[i] = rb.field_evaluations;
        st_b[i] = static_cast<std::uint8_t>(rb.status);
        if (rb.status != closure_gate::StreamlineStatus::ok) {
            return;
        }
        const double fy = rb.records[0].x2, fz = rb.records[0].x3;
        closure_gate::IntegratorOptions of = base;
        of.sigma = +1;
        const closure_gate::StreamlineResult rf = trace_to_plane(field, {0.0, fy, fz}, seed[0], of);
        nfev_f[i] = rf.field_evaluations;
        st_f[i] = static_cast<std::uint8_t>(rf.status);
        bfe[i] = (rb.backflow_encounter || rf.backflow_encounter) ? 1 : 0;
        if (rf.status != closure_gate::StreamlineStatus::ok) {
            return;
        }
        R.foot_y[i] = fy;
        R.foot_z[i] = fz;
        R.roundtrip[i] =
            std::max(std::abs(rf.records[0].x2 - seed[1]), std::abs(rf.records[0].x3 - seed[2]));
    });
    R.seconds_trace = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // per-plane table (sequential, index order: independent of the thread count)
    R.planes.resize(static_cast<std::size_t>(N) + 1);
    R.planes[0].j = 0;
    R.planes[0].x1 = 0.0;
    for (int j = 1; j <= N; ++j) {
        SlabOraclePlane& P = R.planes[static_cast<std::size_t>(j)];
        P.j = j;
        P.x1 = grid.coord(j);
        for (std::size_t r = 0; r < n2; ++r) {
            const std::size_t i = static_cast<std::size_t>(j - 1) * n2 + r;
            P.nfev += nfev_b[i];
            P.nfev_rt += nfev_f[i];
            if (st_b[i] != 0) {
                ++P.back_non_ok;
            } else if (st_f[i] != 0) {
                ++P.fwd_non_ok;
            } else {
                P.roundtrip = std::max(P.roundtrip, R.roundtrip[i]);
                if (bfe[i])
                    ++P.backflow_encounters;
            }
        }
        P.flagged = P.back_non_ok > 0 || P.fwd_non_ok > 0 || P.roundtrip > opt.max_roundtrip;
        R.non_ok += P.back_non_ok + P.fwd_non_ok;
        R.max_roundtrip = std::max(R.max_roundtrip, P.roundtrip);
        if (P.flagged)
            R.status = SlabOracleStatus::oracle_roundtrip_fail;
    }

    const auto t1 = std::chrono::steady_clock::now();
    detail::oracle_evaluate_labels(ctx, grid, labels, R.foot_y, R.foot_z, psi1_out, psi2_out);
    R.seconds_labels = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    return R;
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
