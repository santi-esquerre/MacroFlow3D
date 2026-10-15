#pragma once

/**
 * @file PollockTracker.cuh
 * @brief SF-32 Pollock-type (RT0, cellwise-linear) semi-analytical cell tracker
 *        on periodic MAC face averages: host/device core and GPU engine.
 * @ingroup physics_particles
 *
 * SF-32 (Lester eq.(14) roadmap), DAG node N1. Authoritative contract: the
 * increment specification
 * `docs/plans/active/lester-eq14/increments/SF-32-reference-trackers-and-scalings.md`
 * and its orchestration record (section 3.3, including the CORRECTED paragraph
 * on the well-conditioned exit forms). This is the conventional
 * reconstruction-based tracker of Lester et al. (2023) section 5.2
 * (Pollock 1988). It consumes the face averages of StokesFaceVelocity.cuh
 * (PeriodicFaceFluxView) and decides no physical claim.
 *
 * ---------------------------------------------------------------------------
 * 1) Velocity model (RT0 / Pollock)
 * ---------------------------------------------------------------------------
 *
 *  Grid of the PeriodicFaceFluxView: n_a cells per axis a, spacing D_a,
 *  period L_a = n_a D_a, all indices periodic (mod n_a). In cell (i, j, k) with
 *  lower corner x_- and face averages u_- = u[i,j,k], u_+ = u[i+1,j,k]
 *  (likewise v on y-faces, w on z-faces):
 *
 *    v_x(x) = u_- + A_x (x - x_-),   A_x = (u_+ - u_-) / D_x
 *
 *  (each component depends on its own coordinate only). The particle velocity
 *  at relative coordinate r in [0, D] is evaluated as v_p = fma(A, r, u_-),
 *  except r == D, where the interpolant equals u_+ and u_+ is returned exactly
 *  (an exact evaluation of the same interpolant at its endpoint, not a
 *  tolerance: it makes the face value seen from both sides of a face the same
 *  double).
 *
 * ---------------------------------------------------------------------------
 * 2) Per-axis exit: WELL-CONDITIONED forms (the contract)
 * ---------------------------------------------------------------------------
 *
 *    v_p == 0                     -> no exit on this axis (t = +inf)
 *    d = (v_p > 0) ? (D - r) : (0 - r)          signed distance to the candidate face
 *    z = A d / v_p                              (v_f = v_p + A d exactly for the interpolant)
 *    not (1 + z > 0)              -> no exit: the velocity vanishes before the face
 *    t = (A == 0) ? d / v_p : log1p(z) / A
 *
 *    position after time t:   r(t) = r + v_p t               if A == 0
 *                             r(t) = r + v_p expm1(A t) / A  otherwise
 *                             r(t) = r                       if v_p == 0 (exact; avoids 0 * inf)
 *
 *  Why: the textbook forms t = ln(v_f / v_p) / A and
 *  x(t) = x_- + (v_p e^{A t} - u_-) / A are algebraically identical but lose
 *  every digit when u_+ and u_- are equal up to roundoff (A ~ 1e-14 is then a
 *  rounding residual; the orchestrator's prototype produced a garbage exit time
 *  on pair A this way). The log1p / expm1 forms need NO threshold on A (a
 *  threshold would be an epsilon device, forbidden by AGENTS.md) and tend
 *  smoothly to the linear forms as A -> 0. A == 0 is tested by exact equality
 *  only. The products in r(t) and v_p use explicit fma so that host and device
 *  perform the same roundings (log1p / expm1 themselves come from different
 *  math libraries on host and device and may differ by an ulp).
 *
 * ---------------------------------------------------------------------------
 * 3) Cell step and state
 * ---------------------------------------------------------------------------
 *
 *  State (PollockState): cell index cell[a] in [0, n_a), integer cell-wrap
 *  counter w[a], relative position r[a] in [0, D_a], clock t. Unwrapped
 *  position x_u[a] = (cell[a] + w[a] n_a) D_a + r[a], evaluated as
 *  fma(cell[a] + w[a] n_a, D_a, r[a]) (the integer index is converted to
 *  double exactly).
 *
 *    t_e = min(t_x, t_y, t_z)
 *    t_e == +inf  -> kStatusPollockStagnation (15); nothing committed
 *    otherwise    -> every axis moves by the expm1 form; on every axis with
 *                    t_a == t_e (exact ties advance all of them) the
 *                    coordinate is SET to the face (r = 0 in the new cell when
 *                    moving +, r = D when moving -) and cell[a] moves by +-1
 *                    mod n_a (w[a] += 1 when the index wraps past n_a - 1,
 *                    -= 1 below 0); t += t_e; ++cells.
 *
 *  Rounding guard (an addition to the record's 3.3, reported by N1): on the
 *  non-exit axes the exact position for 0 <= t <= t_e lies in the closed cell
 *  [0, D]; the computed expm1 value can leave it by an ulp, which would make
 *  the next distance d have the wrong sign and the next exit time negative.
 *  The computed r is therefore projected on [0, D] (r < 0 -> 0, r > D -> D).
 *  No tolerance is involved; a particle left exactly on a face it is moving
 *  towards leaves the cell at the next step with t = 0 (log1p(0) = 0).
 *
 *  Face ownership: a particle exactly on a face belongs to the cell it is
 *  moving into (sign of the face velocity component). The exit logic gives
 *  this automatically. pollock_init_state applies it to wrapped input
 *  positions; a zero face velocity there is not an error (the particle does
 *  not move across that face): the lower-index cell (the one below the face)
 *  is chosen, with r = D.
 *
 * ---------------------------------------------------------------------------
 * 4) Drivers
 * ---------------------------------------------------------------------------
 *
 *  pollock_advance_to_x1(target): first crossing of x_u[0] = target from
 *  below. If x_u[0] >= target on entry, returns immediately (nothing to do).
 *  Per cell, with X_+ the unwrapped coordinate of the cell's upper x-face:
 *    - if v_x > 0 and X_+ >= target the crossing may happen in this cell:
 *        * X_+ == target and x is an exit axis (t_x == t_e): the full cell
 *          step is taken and lands on the face exactly (stop);
 *        * otherwise t* = log1p(A_x (target - x_u) / v_x) / A_x (or the linear
 *          form) if 1 + A_x (target - x_u) / v_x > 0; when x is an exit axis
 *          t* is bounded by t_e (rounding guard: mathematically t* <= t_x);
 *          if t* <= t_e: partial step, x SET to target (r[0] = target - corner),
 *          the other axes by the expm1 form (stop);
 *    - otherwise (crossing not in this cell): t_e == +inf -> 15, else full step.
 *  Backflow in x1 is followed naturally (exits through the lower x-face).
 *
 *  pollock_advance_to_time(t_target): full cell steps while
 *  t + t_e <= t_target, then the partial step r(t_target - t) on every axis
 *  (no face reached) and t = t_target exactly. A cell with no exit (t_e = +inf)
 *  is NOT a failure in this driver: the particle approaches the zero-velocity
 *  point exponentially (expm1(A t) -> -1 with A < 0, finite), t reaches
 *  t_target, status active.
 *
 *  Guards (both drivers): more than max_cells_per_call cell steps needed ->
 *  kStatusSubstepLimit (12; at most max_cells_per_call steps are taken);
 *  a non-finite face value, velocity, exit time or position ->
 *  kStatusNonFinite (14). On a non-active return the state holds the last
 *  committed exit (nothing past it).
 *
 * ---------------------------------------------------------------------------
 * 5) Status codes
 * ---------------------------------------------------------------------------
 *
 *  SF-31 codes (StreamlineTrackerCommon.cuh) 0, 10..14, plus
 *  kStatusPollockStagnation = 15 (defined here): no axis of the current cell
 *  has a finite exit time (the RT0 velocity vanishes inside the cell on every
 *  moving axis), reported by pollock_cell_exit and pollock_advance_to_x1.
 *  The Pollock tracker never sets 10, 11 or 13.
 *
 * ---------------------------------------------------------------------------
 * 6) GPU engine, determinism, memory
 * ---------------------------------------------------------------------------
 *
 *  One thread per particle, fixed block size, no atomics, no reductions, no
 *  printf: bitwise deterministic for fixed inputs on one platform. The engine
 *  owns, per particle, cell[3], w[3] (int32), r[3], clock (double) and the
 *  cumulative cell count (uint32), i.e. 3*4 + 3*4 + 3*8 + 8 + 4 = 60 bytes,
 *  grown on demand by prepare() and never shrunk. step / step_to_x1 launch one
 *  kernel each: no allocation, no host synchronization. After each kernel the
 *  bound ParticlesSoA receives xi = cell D + r reduced to [0, L) by
 *  wrap_position, with the wrap counters of the SF-31 bookkeeping; the engine
 *  state (not the SoA) is authoritative between calls.
 *  No epsilon, threshold on A, regularization or fallback appears anywhere.
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Scalar.hpp"
#include "../par2_adapter/par2_views.hpp"
#include "StokesFaceVelocity.cuh"
#include "StreamlineTrackerCommon.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace streamline_tracker {

/// No axis of the current RT0 cell has a finite exit time (header section 5).
inline constexpr uint8_t kStatusPollockStagnation = 15;

// ============================================================================
// Core types
// ============================================================================

struct PollockParams {
    int max_cells_per_call; ///< >= 1
};

/// Particle state (header section 3).
struct PollockState {
    int32_t cell[3]; ///< cell index in [0, n_a)
    int32_t w[3];    ///< integer cell-wrap counters
    real r[3];       ///< position relative to the cell's lower corner, in [0, D_a]
    real t;          ///< clock
};

struct PollockCounters {
    uint32_t cells; ///< cell steps taken (incremented by every committed cell step)
};

namespace pollock_detail {

__host__ __device__ inline real inf() {
    return static_cast<real>(INFINITY);
}

__host__ __device__ inline int axis_n(const PeriodicFaceFluxView& f, int a) {
    return a == 0 ? f.nx : (a == 1 ? f.ny : f.nz);
}

__host__ __device__ inline real axis_d(const PeriodicFaceFluxView& f, int a) {
    return a == 0 ? f.dx : (a == 1 ? f.dy : f.dz);
}

__host__ __device__ inline real axis_l(const PeriodicFaceFluxView& f, int a) {
    return a == 0 ? f.Lx : (a == 1 ? f.Ly : f.Lz);
}

/// Face value of axis a on the a-face at index c[a] (lower face of cell c).
__host__ __device__ inline real face_value(const PeriodicFaceFluxView& f, int a, int i, int j,
                                           int k) {
    const size_t idx = stokes_detail::lin(i, j, k, f.nx, f.ny);
    return a == 0 ? f.u[idx] : (a == 1 ? f.v[idx] : f.w[idx]);
}

/// Unwrapped cell index (cell + w n) as an exact double.
__host__ __device__ inline real unwrapped_index(const PollockState& st, int a, int n) {
    return static_cast<real>(st.cell[a]) + static_cast<real>(st.w[a]) * static_cast<real>(n);
}

/// Per-cell evaluation: interpolant data and per-axis exits.
struct CellEval {
    real D[3];
    real A[3];   ///< velocity gradient per axis
    real vp[3];  ///< particle velocity per axis
    real t[3];   ///< exit time per axis (+inf: no exit)
    int side[3]; ///< +1 / -1 exit direction, 0 no exit
    real te;     ///< min exit time
};

/// RT0 evaluation of the current cell (header sections 1-2). Returns
/// kStatusNonFinite if a face value, velocity or exit time is not finite /
/// is NaN, else kStatusActive.
__host__ __device__ inline uint8_t eval_cell(const PeriodicFaceFluxView& f, const PollockState& st,
                                             CellEval& ev) {
    const real INF = inf();
    ev.te = INF;
    for (int a = 0; a < 3; ++a) {
        const int n = axis_n(f, a);
        const real D = axis_d(f, a);
        int c[3] = {st.cell[0], st.cell[1], st.cell[2]};
        const real um = face_value(f, a, c[0], c[1], c[2]);
        c[a] = (c[a] + 1 == n) ? 0 : c[a] + 1;
        const real up = face_value(f, a, c[0], c[1], c[2]);
        if (!isfinite(um) || !isfinite(up)) {
            return kStatusNonFinite;
        }
        const real A = (up - um) / D;
        const real r = st.r[a];
        const real v = (r == D) ? up : fma(A, r, um);
        ev.D[a] = D;
        ev.A[a] = A;
        ev.vp[a] = v;
        ev.t[a] = INF;
        ev.side[a] = 0;
        if (!isfinite(v)) {
            return kStatusNonFinite;
        }
        if (v == static_cast<real>(0.0)) {
            continue;
        }
        const int side = (v > static_cast<real>(0.0)) ? 1 : -1;
        const real d = (side > 0) ? (D - r) : (static_cast<real>(0.0) - r);
        real ta;
        if (A == static_cast<real>(0.0)) {
            ta = d / v;
        } else {
            const real z = A * d / v;
            if (!(static_cast<real>(1.0) + z > static_cast<real>(0.0))) {
                continue;
            }
            ta = log1p(z) / A;
        }
        if (isnan(ta)) {
            return kStatusNonFinite;
        }
        ev.t[a] = ta;
        ev.side[a] = side;
        if (ta < ev.te) {
            ev.te = ta;
        }
    }
    return kStatusActive;
}

/// Relative position on axis a after time t (header section 2), projected on [0, D].
__host__ __device__ inline real move_axis(const CellEval& ev, const PollockState& st, int a,
                                          real t) {
    const real v = ev.vp[a];
    const real A = ev.A[a];
    const real r = st.r[a];
    real x;
    if (v == static_cast<real>(0.0)) {
        x = r;
    } else if (A == static_cast<real>(0.0)) {
        x = fma(v, t, r);
    } else {
        x = fma(v, expm1(A * t) / A, r);
    }
    if (x < static_cast<real>(0.0)) {
        x = static_cast<real>(0.0);
    }
    if (x > ev.D[a]) {
        x = ev.D[a];
    }
    return x;
}

/// Commit the full cell step of duration ev.te (te finite). Returns
/// kStatusNonFinite (nothing committed) if a new coordinate or the clock is
/// not finite.
__host__ __device__ inline uint8_t commit_full(const PeriodicFaceFluxView& f, const CellEval& ev,
                                               PollockState& st, PollockCounters& cnt) {
    PollockState nx = st;
    for (int a = 0; a < 3; ++a) {
        if (ev.side[a] != 0 && ev.t[a] == ev.te) {
            const int n = axis_n(f, a);
            if (ev.side[a] > 0) {
                nx.r[a] = static_cast<real>(0.0);
                nx.cell[a] = st.cell[a] + 1;
                if (nx.cell[a] == n) {
                    nx.cell[a] = 0;
                    nx.w[a] += 1;
                }
            } else {
                nx.r[a] = ev.D[a];
                nx.cell[a] = st.cell[a] - 1;
                if (nx.cell[a] < 0) {
                    nx.cell[a] = n - 1;
                    nx.w[a] -= 1;
                }
            }
        } else {
            nx.r[a] = move_axis(ev, st, a, ev.te);
        }
    }
    nx.t = st.t + ev.te;
    if (!isfinite(nx.r[0]) || !isfinite(nx.r[1]) || !isfinite(nx.r[2]) || !isfinite(nx.t)) {
        return kStatusNonFinite;
    }
    st = nx;
    ++cnt.cells;
    return kStatusActive;
}

/// Commit a partial step of duration t (no face reached). Returns kStatusNonFinite
/// (nothing committed) on a non-finite result.
__host__ __device__ inline uint8_t commit_partial(const CellEval& ev, PollockState& st, real t,
                                                  real t_new) {
    PollockState nx = st;
    for (int a = 0; a < 3; ++a) {
        nx.r[a] = move_axis(ev, st, a, t);
    }
    nx.t = t_new;
    if (!isfinite(nx.r[0]) || !isfinite(nx.r[1]) || !isfinite(nx.r[2]) || !isfinite(nx.t)) {
        return kStatusNonFinite;
    }
    st = nx;
    return kStatusActive;
}

} // namespace pollock_detail

// ============================================================================
// Core API (host and device)
// ============================================================================

/// x_u[a] = fma(cell[a] + w[a] n_a, D_a, r[a]) (header section 3).
__host__ __device__ inline void pollock_unwrapped_position(const PeriodicFaceFluxView& f,
                                                           const PollockState& st, real x_u[3]) {
    for (int a = 0; a < 3; ++a) {
        const int n = pollock_detail::axis_n(f, a);
        x_u[a] =
            fma(pollock_detail::unwrapped_index(st, a, n), pollock_detail::axis_d(f, a), st.r[a]);
    }
}

/**
 * @brief State from a wrapped position (xi, w) of the SF-31 bookkeeping
 *        (x_u = xi + w L). Face ownership as in header section 3. t = 0.
 *
 * Returns kStatusNonFinite if a coordinate or a face value consulted is not
 * finite (st is then unspecified), else kStatusActive (a zero face velocity
 * is not an error: the lower-index cell is chosen).
 */
__host__ __device__ inline uint8_t pollock_init_state(const PeriodicFaceFluxView& f,
                                                      const real xi[3], const int32_t w[3],
                                                      PollockState& st) {
    st.t = static_cast<real>(0.0);
    for (int a = 0; a < 3; ++a) {
        if (!isfinite(xi[a])) {
            return kStatusNonFinite;
        }
        const int n = pollock_detail::axis_n(f, a);
        const real D = pollock_detail::axis_d(f, a);
        real q = floor(xi[a] / D);
        real r = fma(-q, D, xi[a]);
        if (r < static_cast<real>(0.0)) {
            r += D;
            q -= static_cast<real>(1.0);
        }
        if (r >= D) {
            r -= D;
            q += static_cast<real>(1.0);
        }
        // q = cell index in the image of xi; split into [0, n) and periods.
        const real nn = static_cast<real>(n);
        real p = floor(q / nn);
        real c = q - p * nn;
        if (c < static_cast<real>(0.0)) {
            c += nn;
            p -= static_cast<real>(1.0);
        }
        if (c >= nn) {
            c -= nn;
            p += static_cast<real>(1.0);
        }
        st.cell[a] = static_cast<int32_t>(c);
        st.w[a] = w[a] + static_cast<int32_t>(p);
        st.r[a] = r;
    }
    // Face ownership: on a lower face (r == 0) with a non-positive face
    // velocity, the particle belongs to the lower-index cell (r = D).
    for (int a = 0; a < 3; ++a) {
        if (st.r[a] != static_cast<real>(0.0)) {
            continue;
        }
        const real fv = pollock_detail::face_value(f, a, st.cell[0], st.cell[1], st.cell[2]);
        if (!isfinite(fv)) {
            return kStatusNonFinite;
        }
        if (fv <= static_cast<real>(0.0)) {
            const int n = pollock_detail::axis_n(f, a);
            st.r[a] = pollock_detail::axis_d(f, a);
            st.cell[a] -= 1;
            if (st.cell[a] < 0) {
                st.cell[a] = n - 1;
                st.w[a] -= 1;
            }
        }
    }
    return kStatusActive;
}

/// One cell step or stagnation (header section 3). Nothing is committed on a
/// non-active return.
__host__ __device__ inline uint8_t pollock_cell_exit(const PeriodicFaceFluxView& f,
                                                     PollockState& st, PollockCounters& cnt) {
    pollock_detail::CellEval ev;
    const uint8_t code = pollock_detail::eval_cell(f, st, ev);
    if (code != kStatusActive) {
        return code;
    }
    if (ev.te == pollock_detail::inf()) {
        return kStatusPollockStagnation;
    }
    return pollock_detail::commit_full(f, ev, st, cnt);
}

/// First crossing of the unwrapped x1 = x1_target from below (header section 4).
__host__ __device__ inline uint8_t pollock_advance_to_x1(const PeriodicFaceFluxView& f,
                                                         PollockState& st, real x1_target_unwrapped,
                                                         const PollockParams& prm,
                                                         PollockCounters& cnt) {
    const real target = x1_target_unwrapped;
    if (!isfinite(target)) {
        return kStatusNonFinite;
    }
    {
        real xu[3];
        pollock_unwrapped_position(f, st, xu);
        if (!(xu[0] < target)) {
            return kStatusActive;
        }
    }
    int taken = 0;
    for (;;) {
        pollock_detail::CellEval ev;
        const uint8_t code = pollock_detail::eval_cell(f, st, ev);
        if (code != kStatusActive) {
            return code;
        }
        const real idx0 = pollock_detail::unwrapped_index(st, 0, f.nx);
        const real corner = idx0 * f.dx;
        const real x_plus = (idx0 + static_cast<real>(1.0)) * f.dx;
        const real vx = ev.vp[0];
        const bool x_exits = (ev.side[0] != 0) && (ev.t[0] == ev.te);
        if (vx > static_cast<real>(0.0) && x_plus >= target) {
            if (x_plus == target && x_exits) {
                if (taken >= prm.max_cells_per_call) {
                    return kStatusSubstepLimit;
                }
                return pollock_detail::commit_full(f, ev, st, cnt);
            }
            // Target strictly inside this cell along x (or the x-face is the
            // target but another axis exits first: then t* > t_e below).
            const real xp = fma(idx0, f.dx, st.r[0]);
            const real dist = target - xp;
            const real A = ev.A[0];
            bool reach = true;
            real ts;
            if (A == static_cast<real>(0.0)) {
                ts = dist / vx;
            } else {
                const real z = A * dist / vx;
                reach = (static_cast<real>(1.0) + z > static_cast<real>(0.0));
                ts = reach ? log1p(z) / A : pollock_detail::inf();
            }
            if (isnan(ts)) {
                return kStatusNonFinite;
            }
            if (reach) {
                if (x_exits && ts > ev.te) {
                    ts = ev.te; // rounding guard: mathematically t* <= t_x (header 4)
                }
                if (ts <= ev.te) {
                    const uint8_t pc = pollock_detail::commit_partial(ev, st, ts, st.t + ts);
                    if (pc != kStatusActive) {
                        return pc;
                    }
                    st.r[0] = target - corner;
                    return kStatusActive;
                }
            }
        }
        if (ev.te == pollock_detail::inf()) {
            return kStatusPollockStagnation;
        }
        if (taken >= prm.max_cells_per_call) {
            return kStatusSubstepLimit;
        }
        const uint8_t fc = pollock_detail::commit_full(f, ev, st, cnt);
        if (fc != kStatusActive) {
            return fc;
        }
        ++taken;
    }
}

/// Advance to the clock t_target (header section 4). t_target <= st.t: no-op.
__host__ __device__ inline uint8_t pollock_advance_to_time(const PeriodicFaceFluxView& f,
                                                           PollockState& st, real t_target,
                                                           const PollockParams& prm,
                                                           PollockCounters& cnt) {
    if (!isfinite(t_target)) {
        return kStatusNonFinite;
    }
    int taken = 0;
    for (;;) {
        if (!(st.t < t_target)) {
            return kStatusActive;
        }
        pollock_detail::CellEval ev;
        const uint8_t code = pollock_detail::eval_cell(f, st, ev);
        if (code != kStatusActive) {
            return code;
        }
        if (ev.te == pollock_detail::inf() || !(st.t + ev.te <= t_target)) {
            return pollock_detail::commit_partial(ev, st, t_target - st.t, t_target);
        }
        if (taken >= prm.max_cells_per_call) {
            return kStatusSubstepLimit;
        }
        const uint8_t fc = pollock_detail::commit_full(f, ev, st, cnt);
        if (fc != kStatusActive) {
            return fc;
        }
        ++taken;
    }
}

// ============================================================================
// GPU engine
// ============================================================================

struct PollockConfig {
    int max_cells_per_call = 10000000; ///< per particle and per call (>= 1)
};

/// Aggregate report (synchronizing; not for the hot loop).
struct PollockStats {
    int n_particles = 0;
    int n_active = 0;
    int n_stagnation = 0;
    int n_substep_limit = 0;
    int n_nonfinite = 0;
    int n_other = 0; ///< any other nonzero status (not set by this engine)
    uint64_t total_cells = 0;
    uint32_t max_cells = 0;
    real min_clock = 0.0; ///< over all particles; 0 if n_particles == 0
    real max_clock = 0.0;
};

/**
 * @brief GPU Pollock tracker engine (header section 6).
 *
 * Call order: configure, bind_fluxes, bind_particles, inject_box (or caller
 * positions + wraps), ensure_tracking, prepare, then step / step_to_x1.
 * bind_fluxes / bind_particles invalidate a previous prepare().
 * Movable, not copyable.
 */
class PollockTracker {
  public:
    explicit PollockTracker(cudaStream_t stream, uint64_t inject_seed = 0);
    ~PollockTracker() = default;

    PollockTracker(const PollockTracker&) = delete;
    PollockTracker& operator=(const PollockTracker&) = delete;
    PollockTracker(PollockTracker&&) noexcept = default;
    PollockTracker& operator=(PollockTracker&&) noexcept = default;

    /// Throws std::invalid_argument if max_cells_per_call < 1.
    void configure(const PollockConfig& cfg);

    /// Device face arrays, stored by value; the buffers must outlive the engine.
    /// Throws std::invalid_argument (distinct messages) for a null face
    /// pointer, a cell count < 1, a non-finite / <= 0 spacing or period.
    void bind_fluxes(const PeriodicFaceFluxView& fluxes);

    /// Caller-owned device arrays. Throws std::invalid_argument for a null
    /// x/y/z/status pointer or n < 0. Wrap arrays are REQUIRED (checked by
    /// inject_box, ensure_tracking and prepare).
    void bind_particles(ParticlesSoA<real>& p);

    /// StreamlineTrackerCommon inject_box with the engine seed. Throws
    /// std::logic_error if no particles are bound.
    void inject_box(real x0, real y0, real z0, real x1, real y1, real z1, int first, int count);

    /// Contract no-op; throws std::invalid_argument if wrap arrays are missing.
    void ensure_tracking();

    /// May allocate (grow-only). Converts the bound wrapped positions + wraps
    /// into engine states (pollock_init_state; a non-finite input sets status
    /// 14), clocks = 0, cell counts = 0, target time = 0. The bound positions
    /// are not modified. Throws std::logic_error if configure,
    /// bind_fluxes or bind_particles has not been called.
    void prepare();

    /// Target time += dt; every active particle: pollock_advance_to_time.
    /// Throws std::invalid_argument for non-finite or negative dt,
    /// std::logic_error before prepare(). Kernel launch only.
    void step(real dt);

    /// Every active particle: pollock_advance_to_x1(x1_target_unwrapped).
    /// Throws std::invalid_argument for a non-finite target, std::logic_error
    /// before prepare(). Kernel launch only.
    void step_to_x1(real x1_target_unwrapped);

    void synchronize();

    ConstParticlesSoA<real> particles() const;

    /// x_u = x + wrap L with the flux-grid periods. Throws std::logic_error if
    /// fluxes or particles are not bound.
    void compute_unwrapped(UnwrappedSoA<real>& uw, cudaStream_t stream);

    // Read-only device pointers, valid after prepare() (null before).
    const real* clocks() const { return clock_.data(); }
    const uint32_t* cell_counts() const { return cell_count_.data(); }
    /// Engine state arrays (axis a at offset a * n): cell, cell-wrap, relative position.
    const int32_t* state_cells() const { return cell_.data(); }
    const int32_t* state_wraps() const { return wrap_.data(); }
    const real* state_relative() const { return rel_.data(); }
    real target_time() const { return t_target_; }

    /// Synchronizes the stream and copies to the host. NOT for the hot loop.
    /// Throws std::logic_error before prepare().
    PollockStats compute_stats();

    void set_stream(cudaStream_t stream) { stream_ = stream; }
    void set_inject_seed(uint64_t seed) { inject_seed_ = seed; }

  private:
    void require_prepared(const char* who) const;

    cudaStream_t stream_;
    uint64_t inject_seed_;

    PollockConfig cfg_{};
    PeriodicFaceFluxView fluxes_{};
    ParticlesSoA<real> p_{};
    bool configured_ = false;
    bool fluxes_bound_ = false;
    bool particles_bound_ = false;
    bool prepared_ = false;
    int prepared_n_ = 0;
    real t_target_ = 0.0;

    DeviceBuffer<int32_t> cell_;        ///< 3 n
    DeviceBuffer<int32_t> wrap_;        ///< 3 n
    DeviceBuffer<real> rel_;            ///< 3 n
    DeviceBuffer<real> clock_;          ///< n
    DeviceBuffer<uint32_t> cell_count_; ///< n (cumulative since prepare)
};

} // namespace streamline_tracker
} // namespace particles
} // namespace physics
} // namespace macroflow3d
