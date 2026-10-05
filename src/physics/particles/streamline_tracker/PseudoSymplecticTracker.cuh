#pragma once

/**
 * @file PseudoSymplecticTracker.cuh
 * @brief SF-31 pseudo-symplectic streamline tracker: label-curve projection
 *        integrator core (host + device) and GPU engine.
 * @ingroup physics_particles
 *
 * SF-31 (Lester eq.(14) roadmap), DAG node N1. Authoritative numerical
 * contract: the increment specification
 * `docs/plans/active/lester-eq14/increments/SF-31-pseudo-symplectic-tracker-core.md`
 * and its orchestration record (sections 3.1-3.6, 3.8; decisions D-2..D-6, D-9).
 *
 * ---------------------------------------------------------------------------
 * 0) What this is (and what it is not)
 * ---------------------------------------------------------------------------
 *
 *  Each particle carries the two label values psi1_0, psi2_0 sampled at its
 *  position by prepare() and is kept on the curve
 *  {psi1 = psi1_0, psi2 = psi2_0}, which is the streamline of
 *  c = grad psi1 x grad psi2 through its start point. The labels are those of
 *  the bound evaluator (StreamlineTrackerCommon.cuh section 1); a tracker that
 *  conserves labels conserves whatever labels it is given, correct or not.
 *
 *  The method is a PROJECTION method chosen by the project (arclength
 *  predictor + least-norm Newton projection onto the label curve + Simpson
 *  clock). It is NOT the inverse-function algorithm of Lester et al. (2023)
 *  section 5.4 (inversion of (x2, x3) = X(psi1, psi2, x1) and t = int dx1/v1);
 *  it needs neither inverse functions nor v1 > 0.
 *
 * ---------------------------------------------------------------------------
 * 1) Projection (project_to_label_curve): 2x2 least-norm Newton
 * ---------------------------------------------------------------------------
 *
 *  r = (psi1(x) - psi1_0, psi2(x) - psi2_0),  J = [g1; g2] (2x3),
 *  delta = J^T (J J^T)^{-1} (-r):
 *
 *    for it = 0, 1, 2, ...
 *        s = labels(xi, w);  r1 = s.psi1 - psi1_0;  r2 = s.psi2 - psi2_0
 *        a11 = g1.g1; a12 = g1.g2; a22 = g2.g2; det = a11 a22 - a12^2   (= |g1 x g2|^2)
 *        r1, r2 or det not finite              -> kStatusNonFinite
 *        not (det > min_cross_norm^2)          -> kStatusDegenerate
 *        max(|r1|, |r2|) <= tol_psi            -> at = s; newton_iter_max = max(., it); active
 *        it == max_newton_iter                 -> kStatusNewtonFailed
 *        lam1 = (-r1 a22 + r2 a12) / det;  lam2 = (r1 a12 - r2 a11) / det
 *        delta = lam1 g1 + lam2 g2
 *        |delta| > trust_radius: delta *= trust_radius / |delta|; ++clamp_count
 *        xi += delta
 *
 *  det(J J^T) = |c|^2 is NEVER floored, regularized or offset: nothing is
 *  added to it and no epsilon appears (AGENTS.md hard rule on
 *  |grad psi1 x grad psi2| denominators). min_cross_norm (default 0) is a
 *  FAILURE threshold, not a regularization. The degeneracy test runs on every
 *  evaluation, including the converged one.
 *
 *  tol_psi is ABSOLUTE, in label units, applied to each label separately
 *  (max-norm of r). No per-label scaling (decision D-2).
 *
 * ---------------------------------------------------------------------------
 * 2) Panel of length sigma (advance_panel): two projected half-steps
 * ---------------------------------------------------------------------------
 *
 *  Requires st.at = labels at (st.xi, st.w), finite and non-degenerate.
 *
 *    saved = st;  h = sigma/2;  c0 = at.g1 x at.g2;  n0 = |c0|
 *    xi += h c0 / n0;  project (trust_radius = trust_factor h)   failure -> st = saved
 *    cm = at.g1 x at.g2;  nm = |cm|
 *    xi += h cm / nm;  project                                   failure -> st = saved
 *    n1 = |at.g1 x at.g2|
 *    t += sigma ((1/n0 + 4/nm + 1/n1) / 6)        (Simpson of 1/|c| on three on-curve points)
 *    wrap_position(xi, w, labels.L)
 *
 *  The clock is written so that n0 = nm = n1 = 1 gives exactly sigma. The
 *  wrap reduction runs after convergence and only re-represents the same
 *  unwrapped point (xi + w L) up to the rounding of that representation.
 *
 *  Expected accuracy (orchestration record 3.6): after projection the particle
 *  is on the label curve to tol_psi; the along-curve error is global order 2
 *  in sigma (the panel advances sigma (1 - kappa^2 sigma^2 / 12 + ...) of true
 *  arc while the clock integrates over the nominal sigma). Straight
 *  streamlines are exact up to roundoff.
 *
 * ---------------------------------------------------------------------------
 * 3) Advance to a target time (advance_to_time): the step(dt) semantics
 * ---------------------------------------------------------------------------
 *
 *    panels = 0
 *    loop
 *        tau = t_target - st.t;  not (tau > 0) -> active (done)
 *        n = |at.g1 x at.g2|
 *        tau n >= ds_max:  sigma = ds_max (full panel)   else sigma = tau n (last panel)
 *        panels == max_panels -> kStatusSubstepLimit
 *        advance_panel(sigma); failure -> return its code
 *        ++panels; last -> active
 *
 *  Per-particle BANKED clock (decision D-6): there is no landing iteration.
 *  After the final partial panel the particle clock st.t differs from t_target
 *  by a second-order amount of either sign; positions returned after step(dt)
 *  are at the per-particle clock t_p, within O(ds_max^2) of target_time(). The
 *  next call starts from the real t_p, so the mismatch does not accumulate.
 *
 * ---------------------------------------------------------------------------
 * 4) Failure policy (decision D-5) and status codes (decision D-9)
 * ---------------------------------------------------------------------------
 *
 *  A position with max|r| > tol_psi is never committed. A failed panel leaves
 *  the PanelState bitwise as it was (position, wraps, clock, at). There is no
 *  retry with a smaller step and no fallback path; the caller chooses ds.
 *  In the engine a failure writes the code to the particle status, increments
 *  the particle's fail counter, and freezes the particle (status != 0 is never
 *  advanced again). Codes this module can set (StreamlineTrackerCommon.cuh):
 *
 *    kStatusNewtonFailed (10)  projection did not reach tol_psi in max_newton_iter updates
 *    kStatusDegenerate   (11)  det = |c|^2 not > min_cross_norm^2 (incl. |c| = 0)
 *    kStatusSubstepLimit (12)  step(dt) needed more than max_panels_per_step panels
 *    kStatusNonFinite    (14)  a label residual or det is not finite
 *
 * ---------------------------------------------------------------------------
 * 5) Engine (PseudoSymplecticTracker)
 * ---------------------------------------------------------------------------
 *
 *  Implements the EnsembleRunner engine contract (bind_particles, inject_box,
 *  ensure_tracking, prepare, step, particles(), compute_unwrapped,
 *  synchronize); NOT wired into the runner in SF-31. Triply periodic only;
 *  periods are those of the bound labels (labels.L). Every kernel is one
 *  thread per particle, fixed block size, no reductions, no atomics, so the
 *  result is bitwise deterministic for fixed inputs on one platform.
 *  step() and step_arclength() only launch kernels: no allocation, no host
 *  synchronization, no printf. Buffers come from prepare() (grow-only).
 *
 *  Each kernel call, per particle with status 0: load (xi, w, t, psi1_0,
 *  psi2_0); evaluate the labels at (xi, w) with the non-finite / degeneracy
 *  test of section 1 (no residual test: the stored state is a committed one);
 *  run advance_panel or advance_to_time; on failure set status and ++fail
 *  count; always write back the last accepted (xi, w, t); merge counters
 *  (newton_iter_max = max, clamp_count +=).
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
// Integrator core (host + device, templated on the label evaluator)
// ============================================================================

/// Integrator parameters (POD, passed by value into kernels).
struct PseudoSymplecticParams {
    real tol_psi;        ///< absolute tolerance, label units, on max(|r1|, |r2|)
    int max_newton_iter; ///< maximum number of Newton updates in one projection
    real trust_factor;   ///< each update is clamped to |delta| <= trust_factor * (predictor length)
    real min_cross_norm; ///< FAILURE threshold on |c| (default 0); not a regularization
};

/// Counters accumulated by projections (caller initializes to zero).
struct ProjectionCounters {
    uint32_t newton_iter_max; ///< max number of Newton updates of a converged projection
    uint32_t clamp_count;     ///< number of trust-clamped Newton updates
};

/// State of one particle on its label curve. `at` = labels and gradients at (xi, w).
struct PanelState {
    real xi[3];
    int32_t w[3];
    real t;
    LabelSample at;
};

/**
 * @brief Evaluate the labels at (xi, w) and apply the non-finite and
 *        degeneracy tests of the projection (no residual test, no update).
 *
 * Returns kStatusNonFinite if psi1, psi2 or det is not finite,
 * kStatusDegenerate if not (det > min_cross_norm^2), else kStatusActive with
 * `at` written. On failure `at` is still written (diagnostic only).
 */
template <class E>
__host__ __device__ inline uint8_t evaluate_label_state(const E& labels, const real xi[3],
                                                        const int32_t w[3],
                                                        const PseudoSymplecticParams& prm,
                                                        LabelSample& at) {
    labels(xi, w, at);
    const real a11 = at.g1[0] * at.g1[0] + at.g1[1] * at.g1[1] + at.g1[2] * at.g1[2];
    const real a12 = at.g1[0] * at.g2[0] + at.g1[1] * at.g2[1] + at.g1[2] * at.g2[2];
    const real a22 = at.g2[0] * at.g2[0] + at.g2[1] * at.g2[1] + at.g2[2] * at.g2[2];
    const real det = a11 * a22 - a12 * a12;
    if (!isfinite(at.psi1) || !isfinite(at.psi2) || !isfinite(det)) {
        return kStatusNonFinite;
    }
    if (!(det > prm.min_cross_norm * prm.min_cross_norm)) {
        return kStatusDegenerate;
    }
    return kStatusActive;
}

/**
 * @brief 2x2 least-norm Newton projection of xi onto {psi1 = psi1_0,
 *        psi2 = psi2_0} (file header section 1), wraps w held fixed.
 *
 * On success returns kStatusActive, xi is the converged point and `at` the
 * labels there. On failure returns the code; xi holds the last iterate (the
 * caller restores its saved state) and `at` is NOT written.
 */
template <class E>
__host__ __device__ inline uint8_t
project_to_label_curve(const E& labels, real xi[3], const int32_t w[3], real psi1_0, real psi2_0,
                       real trust_radius, const PseudoSymplecticParams& prm, LabelSample& at,
                       ProjectionCounters& cnt) {
    const real thr = prm.min_cross_norm * prm.min_cross_norm;
    for (int it = 0;; ++it) {
        LabelSample s;
        labels(xi, w, s);
        const real r1 = s.psi1 - psi1_0;
        const real r2 = s.psi2 - psi2_0;
        const real a11 = s.g1[0] * s.g1[0] + s.g1[1] * s.g1[1] + s.g1[2] * s.g1[2];
        const real a12 = s.g1[0] * s.g2[0] + s.g1[1] * s.g2[1] + s.g1[2] * s.g2[2];
        const real a22 = s.g2[0] * s.g2[0] + s.g2[1] * s.g2[1] + s.g2[2] * s.g2[2];
        const real det = a11 * a22 - a12 * a12; // = |g1 x g2|^2; never floored
        if (!isfinite(r1) || !isfinite(r2) || !isfinite(det)) {
            return kStatusNonFinite;
        }
        if (!(det > thr)) {
            return kStatusDegenerate;
        }
        if (fmax(fabs(r1), fabs(r2)) <= prm.tol_psi) {
            at = s;
            const uint32_t used = static_cast<uint32_t>(it);
            if (used > cnt.newton_iter_max) {
                cnt.newton_iter_max = used;
            }
            return kStatusActive;
        }
        if (it == prm.max_newton_iter) {
            return kStatusNewtonFailed;
        }
        const real lam1 = (-r1 * a22 + r2 * a12) / det;
        const real lam2 = (r1 * a12 - r2 * a11) / det;
        real delta[3];
        for (int d = 0; d < 3; ++d) {
            delta[d] = lam1 * s.g1[d] + lam2 * s.g2[d];
        }
        const real nd = norm3(delta);
        if (nd > trust_radius) {
            const real scale = trust_radius / nd;
            for (int d = 0; d < 3; ++d) {
                delta[d] *= scale;
            }
            ++cnt.clamp_count;
        }
        for (int d = 0; d < 3; ++d) {
            xi[d] += delta[d];
        }
    }
}

/**
 * @brief One panel of arclength sigma: two projected half-steps, Simpson
 *        clock, periodic wrap (file header section 2).
 *
 * Requires st.at current, finite and non-degenerate. On failure returns the
 * code and st is bitwise unchanged.
 */
template <class E>
__host__ __device__ inline uint8_t advance_panel(const E& labels, PanelState& st, real psi1_0,
                                                 real psi2_0, real sigma,
                                                 const PseudoSymplecticParams& prm,
                                                 ProjectionCounters& cnt) {
    const PanelState saved = st;
    const real h = sigma / static_cast<real>(2.0);
    const real trust_radius = prm.trust_factor * h;

    real c0[3];
    cross3(st.at.g1, st.at.g2, c0);
    const real n0 = norm3(c0);
    for (int d = 0; d < 3; ++d) {
        st.xi[d] += h * c0[d] / n0;
    }
    uint8_t code =
        project_to_label_curve(labels, st.xi, st.w, psi1_0, psi2_0, trust_radius, prm, st.at, cnt);
    if (code != kStatusActive) {
        st = saved;
        return code;
    }

    real cm[3];
    cross3(st.at.g1, st.at.g2, cm);
    const real nm = norm3(cm);
    for (int d = 0; d < 3; ++d) {
        st.xi[d] += h * cm[d] / nm;
    }
    code =
        project_to_label_curve(labels, st.xi, st.w, psi1_0, psi2_0, trust_radius, prm, st.at, cnt);
    if (code != kStatusActive) {
        st = saved;
        return code;
    }

    real c1[3];
    cross3(st.at.g1, st.at.g2, c1);
    const real n1 = norm3(c1);
    const real one = static_cast<real>(1.0);
    st.t += sigma * ((one / n0 + static_cast<real>(4.0) / nm + one / n1) / static_cast<real>(6.0));
    wrap_position(st.xi, st.w, labels.L);
    return kStatusActive;
}

/**
 * @brief Advance st to the target time t_target with full panels of length
 *        ds_max and one final partial panel (file header section 3).
 *
 * Banked clock: on return st.t is within a second-order amount of t_target
 * (either sign). Returns kStatusActive, kStatusSubstepLimit (max_panels
 * panels taken, more needed) or the code of a failed panel; st always holds
 * the last accepted state.
 */
template <class E>
__host__ __device__ inline uint8_t
advance_to_time(const E& labels, PanelState& st, real psi1_0, real psi2_0, real t_target,
                real ds_max, int max_panels, const PseudoSymplecticParams& prm,
                ProjectionCounters& cnt) {
    int panels = 0;
    for (;;) {
        const real tau = t_target - st.t;
        if (!(tau > static_cast<real>(0.0))) {
            return kStatusActive;
        }
        real c[3];
        cross3(st.at.g1, st.at.g2, c);
        const real n = norm3(c);
        real sigma;
        bool last;
        if (tau * n >= ds_max) {
            sigma = ds_max;
            last = false;
        } else {
            sigma = tau * n;
            last = true;
        }
        if (panels == max_panels) {
            return kStatusSubstepLimit;
        }
        const uint8_t code = advance_panel(labels, st, psi1_0, psi2_0, sigma, prm, cnt);
        if (code != kStatusActive) {
            return code;
        }
        ++panels;
        if (last) {
            return kStatusActive;
        }
    }
}

// ============================================================================
// GPU engine
// ============================================================================

/// Engine configuration (validated by PseudoSymplecticTracker::configure).
struct PseudoSymplecticConfig {
    real ds_max = 0.0;  ///< REQUIRED (> 0): panel length used by step(dt)
    real tol_psi = 0.0; ///< REQUIRED (> 0): absolute, label units, max-norm
    int max_newton_iter = 8;
    real trust_factor = 1.0;
    real min_cross_norm = 0.0; ///< failure threshold on |c|, not a regularization
    int max_panels_per_step = 100000;
};

/// Aggregate report (integer aggregates only, plus min/max clock over all particles).
struct PseudoSymplecticStats {
    int n_particles = 0;
    int n_active = 0;
    int n_newton_failed = 0;
    int n_degenerate = 0;
    int n_substep_limit = 0;
    int n_nonfinite = 0;
    int n_other = 0; ///< any other nonzero status (not set by this engine)
    uint64_t total_fail = 0;
    uint32_t max_fail = 0;
    uint32_t max_newton_iter = 0;
    uint64_t total_clamps = 0;
    real min_clock = 0.0; ///< over all particles; 0 if n_particles == 0
    real max_clock = 0.0;
};

/**
 * @brief GPU pseudo-symplectic tracker engine (file header section 5).
 *
 * Call order: configure, bind_labels, bind_particles, inject_box (or caller
 * positions), ensure_tracking, prepare, then step / step_arclength.
 * bind_labels / bind_particles invalidate a previous prepare().
 * Movable, not copyable.
 */
class PseudoSymplecticTracker {
  public:
    explicit PseudoSymplecticTracker(cudaStream_t stream, uint64_t inject_seed = 0);
    ~PseudoSymplecticTracker() = default;

    PseudoSymplecticTracker(const PseudoSymplecticTracker&) = delete;
    PseudoSymplecticTracker& operator=(const PseudoSymplecticTracker&) = delete;
    PseudoSymplecticTracker(PseudoSymplecticTracker&&) noexcept = default;
    PseudoSymplecticTracker& operator=(PseudoSymplecticTracker&&) noexcept = default;

    /// Validates; throws std::invalid_argument (distinct messages).
    void configure(const PseudoSymplecticConfig& cfg);

    /// Stored by value; the coefficients (device memory) must outlive the engine.
    /// Throws std::invalid_argument for null coefficients or a non-finite / <= 0 period.
    void bind_labels(const SplineLabelPair& labels);

    /// Caller-owned device arrays. Throws std::invalid_argument for a null
    /// x/y/z/status pointer or n < 0. Wrap arrays are REQUIRED but checked by
    /// inject_box, ensure_tracking and prepare.
    void bind_particles(ParticlesSoA<real>& p);

    /// Deterministic hash injection (StreamlineTrackerCommon inject_box) with
    /// the engine seed. Throws std::logic_error if no particles are bound.
    void inject_box(real x0, real y0, real z0, real x1, real y1, real z1, int first, int count);

    /// Contract no-op; throws std::invalid_argument if wrap arrays are missing.
    void ensure_tracking();

    /// May allocate (grow-only). Samples psi1_0, psi2_0 at the current
    /// positions, sets clocks = 0, counters = 0, target time = 0. May be called
    /// again for a new realization. Throws std::logic_error if configure,
    /// bind_labels or bind_particles has not been called.
    void prepare();

    /// Target time += dt; every active particle advances to it (banked clock).
    /// Throws std::invalid_argument for non-finite or negative dt,
    /// std::logic_error before prepare(). Kernel launch only.
    void step(real dt);

    /// One panel of length ds for every active particle; target time untouched.
    /// Throws std::invalid_argument for non-finite or <= 0 ds,
    /// std::logic_error before prepare(). Kernel launch only.
    void step_arclength(real ds);

    void synchronize();

    ConstParticlesSoA<real> particles() const;

    /// x_u = x + wrap L with the label periods. Throws std::logic_error if
    /// labels or particles are not bound.
    void compute_unwrapped(UnwrappedSoA<real>& uw, cudaStream_t stream);

    // Read-only device pointers, valid after prepare() (null before).
    const real* clocks() const { return clock_.data(); }
    const real* psi1_targets() const { return psi1_0_.data(); }
    const real* psi2_targets() const { return psi2_0_.data(); }
    const uint32_t* fail_counts() const { return fail_count_.data(); }
    const uint32_t* clamp_counts() const { return clamp_count_.data(); }
    const uint32_t* newton_iter_max() const { return newton_iter_max_.data(); }
    real target_time() const { return t_target_; }

    /// Synchronizes the stream and copies to the host. NOT for the hot loop.
    /// Throws std::logic_error before prepare().
    PseudoSymplecticStats compute_stats();

    void set_stream(cudaStream_t stream) { stream_ = stream; }
    void set_inject_seed(uint64_t seed) { inject_seed_ = seed; }

  private:
    void require_prepared(const char* who) const;
    PseudoSymplecticParams params() const;

    cudaStream_t stream_;
    uint64_t inject_seed_;

    PseudoSymplecticConfig cfg_{};
    SplineLabelPair labels_{};
    ParticlesSoA<real> p_{};
    bool configured_ = false;
    bool labels_bound_ = false;
    bool particles_bound_ = false;
    bool prepared_ = false;
    int prepared_n_ = 0;
    real t_target_ = 0.0;

    DeviceBuffer<real> psi1_0_;
    DeviceBuffer<real> psi2_0_;
    DeviceBuffer<real> clock_;
    DeviceBuffer<uint32_t> fail_count_;
    DeviceBuffer<uint32_t> clamp_count_;
    DeviceBuffer<uint32_t> newton_iter_max_;
};

} // namespace streamline_tracker
} // namespace particles
} // namespace physics
} // namespace macroflow3d
