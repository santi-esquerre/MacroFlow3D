#pragma once

/**
 * @file ReferenceRkTracker.cuh
 * @brief SF-31 adaptive Runge-Kutta reference tracker: Dormand-Prince 5(4)
 *        integrator core (host + device) and GPU engine.
 * @ingroup physics_particles
 *
 * SF-31 (Lester eq.(14) roadmap), DAG node N2. Authoritative contract: the
 * increment specification
 * `docs/plans/active/lester-eq14/increments/SF-31-pseudo-symplectic-tracker-core.md`
 * and its orchestration record (sections 3.1, 3.7, 3.8; decision D-7).
 *
 * ---------------------------------------------------------------------------
 * 1) ODE
 * ---------------------------------------------------------------------------
 *
 *    dx/dt = c(x),     c = grad psi1 x grad psi2,
 *
 *  on the labels of StreamlineTrackerCommon.cuh (affine part + SF-28 periodic
 *  spline). The velocity depends on the label gradients only, which are
 *  periodic, so the velocity functor is evaluated at the wrapped position with
 *  zero wrap counts (LabelVelocity). Positions are kept wrapped in [0, L) with
 *  integer wrap counters (wrap_position of N0) after every accepted step.
 *
 * ---------------------------------------------------------------------------
 * 2) Pair, controller, error norm
 * ---------------------------------------------------------------------------
 *
 *  Dormand-Prince 5(4) with FSAL (Dormand & Prince 1980; Hairer, Norsett,
 *  Wanner, "Solving ODEs I", Table 5.2). The coefficients are those of the
 *  already reviewed SF-30 oracle (`apps/closure_gate/streamline_integrator.hpp`,
 *  struct detail::DP5), copied as plain `real` constants into
 *  dp54_trial_step; that header is NOT included (host-only instrument of
 *  another increment; an oracle must not share the tracker's construction).
 *  The propagated solution is the 5th-order one (local extrapolation);
 *  x5 - x4 = h sum_i e_i k_i with e = b(5th) - b(4th).
 *
 *  Error norm: ABSOLUTE, max over the three coordinates,
 *      err = max_i |x5_i - x4_i| / tol          (tol in length units).
 *  Accept iff err <= 1. Step-size controller (rk_advance_to_time):
 *      accepted, not clipped:  h_new = min(dt_max, h fac(err)),
 *      fac(err) = 5 if err == 0, else clamp(0.9 err^(-1/5), 0.2, 5);
 *      rejected:               h_new = h max(0.2, 0.9 err^(-1/5)).
 *  The first proposal of a particle is 0.1 dt_max (engine prepare()).
 *
 * ---------------------------------------------------------------------------
 * 3) Exact landing
 * ---------------------------------------------------------------------------
 *
 *  rk_advance_to_time integrates to t_target. A step whose proposal reaches or
 *  passes the target is clipped to h = t_target - t; when that clipped step
 *  is accepted the clock is SET to t_target (never accumulated), so after a
 *  successful call t == t_target bitwise. A clipped accepted step leaves the
 *  proposal unchanged (the proposal is a property of the solution, not of
 *  the output grid). FSAL is used inside one call; k1 is re-evaluated once at
 *  the start of each call.
 *
 * ---------------------------------------------------------------------------
 * 4) No label projection, on purpose
 * ---------------------------------------------------------------------------
 *
 *  This integrator does NOT project onto the label level set, does not
 *  regularize |c| and has no floor on the velocity. Its label drift is the
 *  quantity SF-32 compares against the pseudo-symplectic tracker (Lester et
 *  al. 2023, section 5.3), so it must not conserve the labels by construction.
 *
 * ---------------------------------------------------------------------------
 * 5) Status codes it can set (StreamlineTrackerCommon.cuh)
 * ---------------------------------------------------------------------------
 *
 *  kStatusSubstepLimit  (12)  max_steps_per_call trial steps (accepted +
 *                             rejected) in one call without reaching t_target;
 *  kStatusStepUnderflow (13)  an unclipped proposal below min_step;
 *  kStatusNonFinite     (14)  a stage position or velocity is not finite.
 *  On any failure the state (position, wraps, clock) is the last accepted one.
 *
 * ---------------------------------------------------------------------------
 * 6) Caveat on splined labels
 * ---------------------------------------------------------------------------
 *
 *  The labels are C^2 piecewise cubics, so c = grad psi1 x grad psi2 is only
 *  C^1 across spline knots. A 5th-order pair loses order at every knot
 *  crossing and the embedded estimate no longer controls the true error: on
 *  splined labels the label drift does NOT follow the tolerance (it is limited
 *  by the smoothness of the interpolant, not by the integrator). On analytic
 *  (smooth) labels the drift scales about like tol. See the SF-31 bitacora
 *  (orchestrator prototype row of 2026-10-05) and orchestration record
 *  section 9. This is a property of the interpolant, not a defect.
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Scalar.hpp"
#include "../par2_adapter/par2_views.hpp"
#include "StreamlineTrackerCommon.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace streamline_tracker {

// ============================================================================
// Velocity functor concept and the label adapter
// ============================================================================
//
// A velocity functor V is a POD type with
//   __host__ __device__ void operator()(const real xi[3], real v[3]) const;
// pure in xi, allocating nothing.

/// Velocity c = grad psi1 x grad psi2 of a label evaluator E (N0 concept),
/// evaluated at (xi, w = 0): the gradients do not depend on the wrap counts.
template <class E> struct LabelVelocity {
    E labels;

    __host__ __device__ inline void operator()(const real xi[3], real v[3]) const {
        const int32_t w0[3] = {0, 0, 0};
        LabelSample s;
        labels(xi, w0, s);
        cross3(s.g1, s.g2, v);
    }
};

// ============================================================================
// Integrator core (host + device)
// ============================================================================

/// Parameters of the core (POD, passed by value into kernels).
struct ReferenceRkParams {
    real tol;               ///< absolute tolerance on the positions (length units)
    real dt_max;            ///< maximum step (time units)
    real min_step;          ///< underflow threshold of an unclipped step
    int max_steps_per_call; ///< guard on trial steps (accepted + rejected) per advance call
};

/// Per-particle integrator state.
struct RkState {
    real xi[3];   ///< wrapped position
    int32_t w[3]; ///< wrap counts
    real t;       ///< clock
    real h;       ///< current step proposal
};

/// Trial-step counters of one or more advance calls.
struct RkCounters {
    uint32_t accepted;
    uint32_t rejected;
};

namespace rk_detail {

__host__ __device__ inline bool finite3(const real a[3]) {
    return isfinite(a[0]) && isfinite(a[1]) && isfinite(a[2]);
}

/// Accepted-step growth factor: 5 if err == 0, else clamp(0.9 err^(-1/5), 0.2, 5).
__host__ __device__ inline real accept_factor(real err) {
    if (err == static_cast<real>(0.0)) {
        return static_cast<real>(5.0);
    }
    real f = static_cast<real>(0.9) * pow(err, static_cast<real>(-0.2));
    if (f < static_cast<real>(0.2))
        f = static_cast<real>(0.2);
    if (f > static_cast<real>(5.0))
        f = static_cast<real>(5.0);
    return f;
}

/// Rejected-step shrink factor: max(0.2, 0.9 err^(-1/5)) (err > 1 here).
__host__ __device__ inline real reject_factor(real err) {
    const real f = static_cast<real>(0.9) * pow(err, static_cast<real>(-0.2));
    return f < static_cast<real>(0.2) ? static_cast<real>(0.2) : f;
}

} // namespace rk_detail

/**
 * @brief One Dormand-Prince 5(4) trial step of size h from x with k1 = V(x).
 *
 * Writes the 5th-order solution x5, k7 = V(x5) (FSAL) and
 * err_abs = max_i |x5_i - x4_i|. Returns false (outputs unspecified) if a stage
 * position or a stage velocity is not finite.
 */
template <class V>
__host__ __device__ inline bool dp54_trial_step(const V& vel, const real x[3], const real k1[3],
                                                real h, real x5[3], real k7[3], real& err_abs) {
    // Dormand-Prince 5(4) tableau (identical to apps/closure_gate/streamline_integrator.hpp).
    const real a21 = 1.0 / 5.0;
    const real a31 = 3.0 / 40.0, a32 = 9.0 / 40.0;
    const real a41 = 44.0 / 45.0, a42 = -56.0 / 15.0, a43 = 32.0 / 9.0;
    const real a51 = 19372.0 / 6561.0, a52 = -25360.0 / 2187.0, a53 = 64448.0 / 6561.0,
               a54 = -212.0 / 729.0;
    const real a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0, a63 = 46732.0 / 5247.0,
               a64 = 49.0 / 176.0, a65 = -5103.0 / 18656.0;
    // 5th-order weights (= row 7, FSAL); b2 = 0.
    const real b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0,
               b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;
    // Error weights e = b(5th) - b(4th); e2 = 0.
    const real e1 = 71.0 / 57600.0, e3 = -71.0 / 16695.0, e4 = 71.0 / 1920.0,
               e5 = -17253.0 / 339200.0, e6 = 22.0 / 525.0, e7 = -1.0 / 40.0;

    real k2[3], k3[3], k4[3], k5[3], k6[3], y[3];

    for (int i = 0; i < 3; ++i)
        y[i] = x[i] + h * (a21 * k1[i]);
    if (!rk_detail::finite3(y))
        return false;
    vel(y, k2);
    if (!rk_detail::finite3(k2))
        return false;

    for (int i = 0; i < 3; ++i)
        y[i] = x[i] + h * (a31 * k1[i] + a32 * k2[i]);
    if (!rk_detail::finite3(y))
        return false;
    vel(y, k3);
    if (!rk_detail::finite3(k3))
        return false;

    for (int i = 0; i < 3; ++i)
        y[i] = x[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    if (!rk_detail::finite3(y))
        return false;
    vel(y, k4);
    if (!rk_detail::finite3(k4))
        return false;

    for (int i = 0; i < 3; ++i)
        y[i] = x[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
    if (!rk_detail::finite3(y))
        return false;
    vel(y, k5);
    if (!rk_detail::finite3(k5))
        return false;

    for (int i = 0; i < 3; ++i)
        y[i] = x[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
    if (!rk_detail::finite3(y))
        return false;
    vel(y, k6);
    if (!rk_detail::finite3(k6))
        return false;

    for (int i = 0; i < 3; ++i)
        x5[i] = x[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i]);
    if (!rk_detail::finite3(x5))
        return false;
    vel(x5, k7);
    if (!rk_detail::finite3(k7))
        return false;

    real e = static_cast<real>(0.0);
    for (int i = 0; i < 3; ++i) {
        const real d =
            fabs(h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]));
        if (d > e)
            e = d;
    }
    err_abs = e;
    return true;
}

/**
 * @brief Advance st to t_target (file header sections 2-3).
 *
 * Returns kStatusActive on success (then st.t == t_target bitwise, or st is
 * untouched when st.t >= t_target already), or kStatusSubstepLimit /
 * kStatusStepUnderflow / kStatusNonFinite; on failure position, wraps and
 * clock are those of the last accepted step. cnt is incremented (not reset).
 * L are the periods used by wrap_position.
 */
template <class V>
__host__ __device__ inline uint8_t rk_advance_to_time(const V& vel, RkState& st, const real L[3],
                                                      real t_target, const ReferenceRkParams& prm,
                                                      RkCounters& cnt) {
    if (!(st.t < t_target)) {
        return kStatusActive;
    }
    real k1[3];
    vel(st.xi, k1); // once per call; FSAL inside the call (a non-finite k1 fails the first trial)
    int steps = 0;
    while (st.t < t_target) {
        if (steps == prm.max_steps_per_call) {
            return kStatusSubstepLimit;
        }
        const real rem = t_target - st.t;
        real h;
        bool clipped;
        if (st.h >= rem) {
            h = rem;
            clipped = true;
        } else {
            h = st.h;
            clipped = false;
        }
        if (!clipped && !(h >= prm.min_step)) {
            return kStatusStepUnderflow;
        }
        real x5[3], k7[3];
        real err_abs = static_cast<real>(0.0);
        const bool ok = dp54_trial_step(vel, st.xi, k1, h, x5, k7, err_abs);
        ++steps;
        if (!ok) {
            return kStatusNonFinite;
        }
        const real err = err_abs / prm.tol;
        if (err <= static_cast<real>(1.0)) {
            for (int i = 0; i < 3; ++i) {
                st.xi[i] = x5[i];
                k1[i] = k7[i];
            }
            ++cnt.accepted;
            st.t = clipped ? t_target : st.t + h; // landing time SET, never accumulated
            if (!clipped) {
                const real hn = h * rk_detail::accept_factor(err);
                st.h = hn < prm.dt_max ? hn : prm.dt_max;
            }
            wrap_position(st.xi, st.w, L);
        } else {
            ++cnt.rejected;
            st.h = h * rk_detail::reject_factor(err);
        }
    }
    return kStatusActive;
}

// ============================================================================
// GPU engine
// ============================================================================

/// Engine configuration. tol and dt_max are REQUIRED (> 0, finite).
struct ReferenceRkConfig {
    real tol = 0.0;
    real dt_max = 0.0;
    real min_step = 1e-14;
    int max_steps_per_call = 1000000;
};

/// Aggregate report (integer aggregates, plus min/max clock and step proposal
/// over all particles; 0 when there are no particles).
struct ReferenceRkStats {
    int n_particles = 0;
    int n_active = 0;
    int n_step_underflow = 0;
    int n_substep_limit = 0;
    int n_nonfinite = 0;
    int n_other = 0; ///< any other nonzero status
    uint64_t total_accepted = 0;
    uint64_t total_rejected = 0;
    uint32_t max_accepted = 0;
    uint32_t max_rejected = 0;
    real min_clock = 0.0;
    real max_clock = 0.0;
    real min_step_proposal = 0.0;
    real max_step_proposal = 0.0;
};

/**
 * @brief GPU engine of the RK reference (EnsembleRunner engine contract).
 *
 * Call order: configure, bind_labels and bind_particles (any order; a re-bind
 * or re-configure invalidates ensure_tracking/prepare), inject_box (needs
 * particles), ensure_tracking (needs all three), prepare, then step. Calls out
 * of order throw std::logic_error.
 *
 * Per-particle clocks t_p, step proposals h_p and accepted/rejected counters
 * live in engine-owned buffers sized in prepare(). step(dt) advances the
 * engine target time by dt and integrates every active particle (status == 0)
 * to it with rk_advance_to_time; a successful particle lands on the target
 * time exactly. A failure code is written to the particle status, and the
 * particle is no longer advanced.
 *
 * step() launches one kernel: no allocation, no host synchronization, no
 * printf. One thread per particle, fixed block size, no atomics, no
 * reductions: results are bitwise deterministic.
 */
class ReferenceRkTracker {
  public:
    explicit ReferenceRkTracker(cudaStream_t stream, uint64_t inject_seed = 0);

    ReferenceRkTracker(const ReferenceRkTracker&) = delete;
    ReferenceRkTracker& operator=(const ReferenceRkTracker&) = delete;
    ReferenceRkTracker(ReferenceRkTracker&&) noexcept = default;
    ReferenceRkTracker& operator=(ReferenceRkTracker&&) noexcept = default;
    ~ReferenceRkTracker() = default;

    /// Validates (std::invalid_argument, distinct messages): tol and dt_max
    /// finite and > 0; min_step finite and >= 0; max_steps_per_call >= 1.
    void configure(const ReferenceRkConfig& cfg);
    void bind_labels(const SplineLabelPair& labels);
    /// Throws std::invalid_argument if n < 0, a position/status pointer is
    /// null, or a wrap array is null (wrap arrays are REQUIRED).
    void bind_particles(ParticlesSoA<real>& p);
    /// Delegates to streamline_tracker::inject_box with the engine seed.
    void inject_box(real x0, real y0, real z0, real x1, real y1, real z1, int first, int count);
    void ensure_tracking();
    /// May allocate: clocks = 0, proposals = 0.1 dt_max, counters = 0,
    /// target time = 0. Stream-ordered; does not synchronize.
    void prepare();
    /// Throws std::invalid_argument for non-finite or negative dt.
    void step(real dt);
    void synchronize();

    ConstParticlesSoA<real> particles() const;
    /// Delegates to streamline_tracker::compute_unwrapped with the label periods.
    void compute_unwrapped(UnwrappedSoA<real>& uw, cudaStream_t stream);

    const real* clocks() const { return clock_.data(); }
    const real* step_proposals() const { return h_.data(); }
    const uint32_t* accepted_counts() const { return accepted_.data(); }
    const uint32_t* rejected_counts() const { return rejected_.data(); }
    real target_time() const { return t_target_; }

    /// Synchronizes the stream and downloads the per-particle arrays. NOT for
    /// the hot loop.
    ReferenceRkStats compute_stats();

    void set_stream(cudaStream_t s) { stream_ = s; }
    void set_inject_seed(uint64_t seed) { inject_seed_ = seed; }

  private:
    cudaStream_t stream_ = nullptr;
    uint64_t inject_seed_ = 0;

    ReferenceRkConfig cfg_{};
    SplineLabelPair labels_{};
    ParticlesSoA<real> parts_{};

    bool configured_ = false;
    bool labels_bound_ = false;
    bool particles_bound_ = false;
    bool tracking_ready_ = false;
    bool prepared_ = false;

    real t_target_ = 0.0;

    DeviceBuffer<real> clock_;
    DeviceBuffer<real> h_;
    DeviceBuffer<uint32_t> accepted_;
    DeviceBuffer<uint32_t> rejected_;
};

} // namespace streamline_tracker
} // namespace particles
} // namespace physics
} // namespace macroflow3d
