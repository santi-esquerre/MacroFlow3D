#pragma once

/**
 * @file return_map.cuh
 * @brief SF-32 N2a/N2b: one-period return map of the `spurious_spreading`
 *        instrument -- per-particle cores (host + device) and GPU kernels for
 *        the SF-31 pseudo-symplectic tracker, the SF-31 DP5(4) RK reference
 *        and the SF-32 Pollock (RT0) tracker on the Stokes face fluxes.
 *
 * ---------------------------------------------------------------------------
 * 1) Observable (Lester et al. 2023, eqs. 34-36; understanding.md 3.4)
 * ---------------------------------------------------------------------------
 *
 *  Labels psi_i = gbar_i . x_u + s_i (affine + SF-28 periodic spline,
 *  label_routes.hpp). The surrogate flow c = grad psi1 x grad psi2 has closed
 *  streamlines by construction (affine + periodic labels with c1 > 0), so the
 *  exact streamline through a seed x_0 on the face x1 = 0 returns to
 *  (x1_0 + 1, x2_0, x3_0) after one period. For every seed and tracker:
 *
 *    advance until the unwrapped x1 first reaches x1_0 + 1 (= 1, x1_0 = 0),
 *    land on it (|x1_u - 1| <= 1e-12, section 2), and record
 *      delta_x2   = x2_u - x2_0,  delta_x3 = x3_u - x3_0       (eq. 34; unwrapped)
 *      delta_psi_i = psi_i(landing, unwrapped) - psi_i(seed)   (eq. 35; spline labels)
 *      tau        = clock at the landing                        (eq. 36 uses <tau>)
 *
 *  The exact answer of delta_x and delta_psi is 0: everything measured is a
 *  NUMERICAL error of the tracker on this surrogate (understanding.md 8).
 *  delta_psi_i is evaluated with SplineLabelPair at the final (xi, w) for
 *  every tracker (for the pseudo-symplectic tracker it is its own residual,
 *  <= tol_psi by construction of a committed state).
 *
 * ---------------------------------------------------------------------------
 * 2) Landing procedures and their limits
 * ---------------------------------------------------------------------------
 *
 *  pseudo_symplectic (SF-31 cores, unmodified): from (xi = seed, w = 0,
 *  t = 0), evaluate_label_state gives psi1_0, psi2_0; full panels
 *  advance_panel(ds) (ds = ds_ratio h) are taken until the panel end has
 *  x1_u >= 1; `saved` is the last state before the crossing. Landing:
 *  bisection on the panel length sigma in (0, ds]: lo = 0, hi = ds,
 *  best = the crossing state; each trial is advance_panel(mid) from `saved`;
 *  x1_u(trial) >= 1 -> hi = mid, best = trial, else lo = mid; stop when
 *  |x1_u(best) - 1| <= 1e-12 or after 60 trials. The final state is `best`,
 *  a projected state (label residual <= tol_psi). The clock of `best` is the
 *  Simpson clock of a panel of length sigma (second order in sigma, SF-31).
 *
 *  rk (SF-31 DP5(4) core rk_advance_to_time, unmodified; LabelVelocity of the
 *  spline pair; ReferenceRkParams{tol, dt_max (ABSOLUTE, default 0.25;
 *  decision D-2 of the SF-32 orchestration record: a cap h/2 left the DP5(4)
 *  controller inactive over the whole tolerance ladder), min_step = 1e-14,
 *  max_steps_per_call = 1e6}; first proposal 0.1 dt_max):
 *  chunks of duration EQUAL to dt_max (a shorter chunk would re-cap the step
 *  at every chunk end; rk_advance_to_time to st.t + dt_max) until
 *  x1_u >= 1; `saved` is the state before the crossing chunk. Landing:
 *  bisection on the chunk's target time t* in (saved.t, saved.t + dt_max];
 *  each trial re-integrates the chunk from `saved` with
 *  rk_advance_to_time(t*); same acceptance/stop rule as above. LIMIT OF THE
 *  INSTRUMENT: a trial re-integrates the chunk and the controller then clips
 *  its last step at t*, so the landing state differs from a dense-output
 *  evaluation of the original chunk by a clipped-step error, which is within
 *  the controller's accepted local error (tol); the SF-31 core has no dense
 *  output and is reused unmodified (no Henon device either).
 *
 *  pollock (SF-32 N1 core pollock_init_state + pollock_advance_to_x1,
 *  unmodified; face fluxes of StokesFaceVelocity on the Pollock grid
 *  Delta = m h, n = N / m cells per axis): the seed (0, x2_0, x3_0), w = 0,
 *  is converted to a cell state (face ownership of the core), then
 *  pollock_advance_to_x1(1) crosses cells semi-analytically; the target
 *  x1 = 1 is an x-face of the grid (n Delta = 1 exactly), so the core lands
 *  ON it exactly: no bisection (land_iters = 0, land_err = |x1_u - 1| as
 *  computed, expected 0). The unwrapped position is the core's
 *  pollock_unwrapped_position; the labels are evaluated at
 *  xi = cell Delta + r (reduced to [0, L) with the wrap moved into w).
 *  tau = the Pollock clock; count = cells crossed. Status codes: 0, 12
 *  (max_cells_per_call), 14 (non-finite), 15 (stagnation, no fallback); on a
 *  non-zero status the state is the last committed exit.
 *
 *  The landing tolerance 1e-12 is on x1 only; land_err = |x1_u - 1| of the
 *  final state is always written and the maximum over seeds and the maximum
 *  number of bisection trials are reported in summary.json, so a seed that
 *  did not reach 1e-12 within 60 trials is visible (its status stays 0: the
 *  state is a valid tracker state, only the landing is reported as inexact).
 *
 *  count: pseudo_symplectic = panels of the main loop up to and including
 *  the crossing panel (landing trials excluded); rk = accepted DP5(4) steps of
 *  the main loop up to and including the crossing chunk (landing trials
 *  excluded); pollock = cells crossed (committed full cell steps).
 *
 *  Failures (no fallback, no retry, no epsilon): a failing advance_panel /
 *  rk_advance_to_time (main loop or landing trial) writes its SF-31 status
 *  code (10-14) and the particle keeps its last accepted state; the main loop
 *  guard (max_panels / max_chunks, default 1e7) gives kStatusSubstepLimit.
 *  The deltas of a non-ok seed are written as nan by the host.
 *
 * ---------------------------------------------------------------------------
 * 3) Determinism
 * ---------------------------------------------------------------------------
 *
 *  One thread per particle, fixed block size, no atomics, no reductions, no
 *  shared state: the per-particle result depends only on (labels, seed
 *  position, parameters), so outputs are bitwise reproducible on one
 *  platform. The cores are __host__ __device__ (a host mirror reproduces a
 *  particle up to FMA contraction differences).
 *
 * ---------------------------------------------------------------------------
 * 4) Output schema (dag.json output_schema) -- written by the host
 * ---------------------------------------------------------------------------
 *
 *  seeds.csv:  seed_index,x2_0,x3_0,status,delta_x2,delta_x3,delta_psi1,
 *              delta_psi2,tau,count,land_err   (%.17g; nan deltas if status != 0)
 *  summary.json: field, tracker, level, n_seeds, n_ok, status_counts, stats,
 *              flux_weighted_tau_mean, D22_uniform, D33_uniform, D22_flux,
 *              D33_flux, divergence_max_rel, min_abs_c_grid, landing, config,
 *              wall_seconds (last).
 */

#include "src/core/Scalar.hpp"
#include "src/physics/particles/streamline_tracker/PollockTracker.cuh"
#include "src/physics/particles/streamline_tracker/PseudoSymplecticTracker.cuh"
#include "src/physics/particles/streamline_tracker/ReferenceRkTracker.cuh"
#include "src/physics/particles/streamline_tracker/StreamlineTrackerCommon.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace spurious_spreading {

using macroflow3d::real;
namespace stt = macroflow3d::physics::particles::streamline_tracker;

inline constexpr real kLandingTol = 1e-12; ///< |x1_u - target| landing tolerance
inline constexpr int kMaxLandingTrials = 60;
inline constexpr int kReturnMapBlock = 256;

/// Per-particle result of a return map (POD).
struct ReturnMapResult {
    uint8_t status;           ///< 0 ok, else SF-31 status code
    real x_u[3];              ///< unwrapped final position (xi + w L)
    real tau;                 ///< clock of the final state
    unsigned long long count; ///< panels (ps) / accepted steps (rk) of the main loop
    int land_iters;           ///< bisection trials of the landing
    real psi_seed[2];         ///< labels at the seed
    real psi_end[2];          ///< labels at the final (xi, w)
};

/// Device output arrays (SoA, n entries each).
struct ReturnMapOut {
    uint8_t* status;
    real* x1u;
    real* x2u;
    real* x3u;
    real* tau;
    unsigned long long* count;
    int* land_iters;
    real* dpsi1;
    real* dpsi2;
};

__host__ __device__ inline real unwrapped_coord(const real xi[3], const int32_t w[3],
                                                const real L[3], int d) {
    return fma(static_cast<real>(w[d]), L[d], xi[d]);
}

template <class E>
__host__ __device__ inline void finalize_result(const E& labels, const real xi[3],
                                                const int32_t w[3], real t, ReturnMapResult& r) {
    for (int d = 0; d < 3; ++d)
        r.x_u[d] = unwrapped_coord(xi, w, labels.L, d);
    stt::LabelSample s;
    labels(xi, w, s);
    r.psi_end[0] = s.psi1;
    r.psi_end[1] = s.psi2;
    r.tau = t;
}

// ===========================================================================
// pseudo-symplectic return map (section 2)
// ===========================================================================

template <class E>
__host__ __device__ inline void
ps_return_map_one(const E& labels, real x2_0, real x3_0, const stt::PseudoSymplecticParams& prm,
                  real ds, long long max_panels, real target, ReturnMapResult& r) {
    r.status = stt::kStatusActive;
    r.count = 0ULL;
    r.land_iters = 0;
    stt::PanelState st{};
    st.xi[0] = static_cast<real>(0.0);
    st.xi[1] = x2_0;
    st.xi[2] = x3_0;
    st.w[0] = st.w[1] = st.w[2] = 0;
    st.t = static_cast<real>(0.0);
    stt::ProjectionCounters cnt{0u, 0u};
    uint8_t code = stt::evaluate_label_state(labels, st.xi, st.w, prm, st.at);
    r.psi_seed[0] = st.at.psi1;
    r.psi_seed[1] = st.at.psi2;
    if (code != stt::kStatusActive) {
        r.status = code;
        finalize_result(labels, st.xi, st.w, st.t, r);
        return;
    }
    const real psi1_0 = st.at.psi1;
    const real psi2_0 = st.at.psi2;

    stt::PanelState saved = st;
    long long panels = 0;
    for (;;) {
        if (panels >= max_panels) {
            r.status = stt::kStatusSubstepLimit;
            r.count = static_cast<unsigned long long>(panels);
            finalize_result(labels, st.xi, st.w, st.t, r);
            return;
        }
        saved = st;
        code = stt::advance_panel(labels, st, psi1_0, psi2_0, ds, prm, cnt);
        if (code != stt::kStatusActive) { // st == saved (bitwise) after a failed panel
            r.status = code;
            r.count = static_cast<unsigned long long>(panels);
            finalize_result(labels, st.xi, st.w, st.t, r);
            return;
        }
        ++panels;
        if (unwrapped_coord(st.xi, st.w, labels.L, 0) >= target)
            break;
    }
    r.count = static_cast<unsigned long long>(panels);

    // Landing: bisection on the panel length from the saved pre-crossing state.
    stt::PanelState best = st;
    real lo = static_cast<real>(0.0);
    real hi = ds;
    int it = 0;
    while (it < kMaxLandingTrials &&
           !(fabs(unwrapped_coord(best.xi, best.w, labels.L, 0) - target) <= kLandingTol)) {
        ++it;
        const real mid = static_cast<real>(0.5) * (lo + hi);
        stt::PanelState trial = saved;
        stt::ProjectionCounters ct{0u, 0u};
        code = stt::advance_panel(labels, trial, psi1_0, psi2_0, mid, prm, ct);
        if (code != stt::kStatusActive) {
            r.status = code;
            break;
        }
        if (unwrapped_coord(trial.xi, trial.w, labels.L, 0) >= target) {
            hi = mid;
            best = trial;
        } else {
            lo = mid;
        }
    }
    r.land_iters = it;
    finalize_result(labels, best.xi, best.w, best.t, r);
}

// ===========================================================================
// RK return map (section 2)
// ===========================================================================

template <class E>
__host__ __device__ inline void
rk_return_map_one(const E& labels, real x2_0, real x3_0, const stt::ReferenceRkParams& prm,
                  real chunk, long long max_chunks, real target, ReturnMapResult& r) {
    r.status = stt::kStatusActive;
    r.count = 0ULL;
    r.land_iters = 0;
    stt::RkState st{};
    st.xi[0] = static_cast<real>(0.0);
    st.xi[1] = x2_0;
    st.xi[2] = x3_0;
    st.w[0] = st.w[1] = st.w[2] = 0;
    st.t = static_cast<real>(0.0);
    st.h = static_cast<real>(0.1) * prm.dt_max;
    {
        stt::LabelSample s0;
        labels(st.xi, st.w, s0);
        r.psi_seed[0] = s0.psi1;
        r.psi_seed[1] = s0.psi2;
    }
    const stt::LabelVelocity<E> vel{labels};
    stt::RkCounters cnt{0u, 0u};

    stt::RkState saved = st;
    real t_hi = static_cast<real>(0.0);
    long long chunks = 0;
    for (;;) {
        if (chunks >= max_chunks) {
            r.status = stt::kStatusSubstepLimit;
            r.count = cnt.accepted;
            finalize_result(labels, st.xi, st.w, st.t, r);
            return;
        }
        saved = st;
        t_hi = st.t + chunk;
        const uint8_t code = stt::rk_advance_to_time(vel, st, labels.L, t_hi, prm, cnt);
        if (code != stt::kStatusActive) { // st = last accepted step
            r.status = code;
            r.count = cnt.accepted;
            finalize_result(labels, st.xi, st.w, st.t, r);
            return;
        }
        ++chunks;
        if (unwrapped_coord(st.xi, st.w, labels.L, 0) >= target)
            break;
    }
    r.count = cnt.accepted;

    // Landing: bisection on the chunk's target time from the saved state.
    stt::RkState best = st;
    real lo = saved.t;
    real hi = t_hi;
    int it = 0;
    while (it < kMaxLandingTrials &&
           !(fabs(unwrapped_coord(best.xi, best.w, labels.L, 0) - target) <= kLandingTol)) {
        ++it;
        const real mid = static_cast<real>(0.5) * (lo + hi);
        stt::RkState trial = saved;
        stt::RkCounters ct{0u, 0u};
        const uint8_t code = stt::rk_advance_to_time(vel, trial, labels.L, mid, prm, ct);
        if (code != stt::kStatusActive) {
            r.status = code;
            break;
        }
        if (unwrapped_coord(trial.xi, trial.w, labels.L, 0) >= target) {
            hi = mid;
            best = trial;
        } else {
            lo = mid;
        }
    }
    r.land_iters = it;
    finalize_result(labels, best.xi, best.w, best.t, r);
}

// ===========================================================================
// Pollock return map (section 2)
// ===========================================================================

template <class E>
__host__ __device__ inline void
pollock_return_map_one(const E& labels, const stt::PeriodicFaceFluxView& f, real x2_0, real x3_0,
                       const stt::PollockParams& prm, real target, ReturnMapResult& r) {
    r.status = stt::kStatusActive;
    r.count = 0ULL;
    r.land_iters = 0;
    const real xi0[3] = {static_cast<real>(0.0), x2_0, x3_0};
    const int32_t w0[3] = {0, 0, 0};
    {
        stt::LabelSample s0;
        labels(xi0, w0, s0);
        r.psi_seed[0] = s0.psi1;
        r.psi_seed[1] = s0.psi2;
    }
    stt::PollockState st{};
    uint8_t code = stt::pollock_init_state(f, xi0, w0, st);
    if (code != stt::kStatusActive) {
        r.status = code;
        finalize_result(labels, xi0, w0, static_cast<real>(0.0), r);
        return;
    }
    stt::PollockCounters cnt{0u};
    code = stt::pollock_advance_to_x1(f, st, target, prm, cnt);
    r.status = code;
    r.count = static_cast<unsigned long long>(cnt.cells);
    // Wrapped label coordinates of the final (or last committed) state:
    // xi = cell D + r in [0, L] (cell in [0, n), r in [0, D]); the image
    // xi == L is moved to 0 with the period carried by w (same point).
    real xi[3];
    int32_t w[3];
    for (int a = 0; a < 3; ++a) {
        const real D = a == 0 ? f.dx : (a == 1 ? f.dy : f.dz);
        const real L = a == 0 ? f.Lx : (a == 1 ? f.Ly : f.Lz);
        xi[a] = fma(static_cast<real>(st.cell[a]), D, st.r[a]);
        w[a] = st.w[a];
        if (xi[a] >= L) {
            xi[a] -= L;
            w[a] += 1;
        }
    }
    finalize_result(labels, xi, w, st.t, r);
    // The unwrapped position is the core's own (fma(cell + w n, D, r)).
    stt::pollock_unwrapped_position(f, st, r.x_u);
}

// ===========================================================================
// Kernels (one thread per particle; deterministic)
// ===========================================================================

__device__ inline void store_result(const ReturnMapResult& r, int p, const ReturnMapOut& out) {
    out.status[p] = r.status;
    out.x1u[p] = r.x_u[0];
    out.x2u[p] = r.x_u[1];
    out.x3u[p] = r.x_u[2];
    out.tau[p] = r.tau;
    out.count[p] = r.count;
    out.land_iters[p] = r.land_iters;
    out.dpsi1[p] = r.psi_end[0] - r.psi_seed[0];
    out.dpsi2[p] = r.psi_end[1] - r.psi_seed[1];
}

template <class E>
__global__ void ps_return_map_kernel(E labels, stt::PseudoSymplecticParams prm, real ds,
                                     long long max_panels, real target, const real* x2_0,
                                     const real* x3_0, int n, ReturnMapOut out) {
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= n)
        return;
    ReturnMapResult r;
    ps_return_map_one(labels, x2_0[p], x3_0[p], prm, ds, max_panels, target, r);
    store_result(r, p, out);
}

template <class E>
__global__ void rk_return_map_kernel(E labels, stt::ReferenceRkParams prm, real chunk,
                                     long long max_chunks, real target, const real* x2_0,
                                     const real* x3_0, int n, ReturnMapOut out) {
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= n)
        return;
    ReturnMapResult r;
    rk_return_map_one(labels, x2_0[p], x3_0[p], prm, chunk, max_chunks, target, r);
    store_result(r, p, out);
}

template <class E>
__global__ void pollock_return_map_kernel(E labels, stt::PeriodicFaceFluxView fluxes,
                                          stt::PollockParams prm, real target, const real* x2_0,
                                          const real* x3_0, int n, ReturnMapOut out) {
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= n)
        return;
    ReturnMapResult r;
    pollock_return_map_one(labels, fluxes, x2_0[p], x3_0[p], prm, target, r);
    store_result(r, p, out);
}

/// c1(seed) = (grad psi1 x grad psi2)_1 at (0, x2_0, x3_0), w = 0: the flux weights.
template <class E>
__global__ void seed_c1_kernel(E labels, const real* x2_0, const real* x3_0, int n, real* c1) {
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= n)
        return;
    const real xi[3] = {static_cast<real>(0.0), x2_0[p], x3_0[p]};
    const int32_t w[3] = {0, 0, 0};
    stt::LabelSample s;
    labels(xi, w, s);
    real c[3];
    stt::cross3(s.g1, s.g2, c);
    c1[p] = c[0];
}

} // namespace spurious_spreading
