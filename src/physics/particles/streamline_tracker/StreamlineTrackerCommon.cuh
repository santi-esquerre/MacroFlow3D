#pragma once

/**
 * @file StreamlineTrackerCommon.cuh
 * @brief Shared label evaluator and particle bookkeeping of the SF-31
 *        streamline trackers (pseudo-symplectic tracker and RK reference).
 * @ingroup physics_particles
 *
 * SF-31 (Lester eq.(14) roadmap), DAG node N0. The authoritative record of the
 * numerical contract is the increment specification
 * `docs/plans/active/lester-eq14/increments/SF-31-pseudo-symplectic-tracker-core.md`
 * and its orchestration record (sections 3.1, 3.8; decisions D-8, D-9, D-10).
 * Nothing in this file decides a physical claim: a tracker that conserves
 * labels conserves whatever labels it is given.
 *
 * ---------------------------------------------------------------------------
 * 1) Labels (the mathematical object)
 * ---------------------------------------------------------------------------
 *
 *    psi_i(x) = gbar_i . x_u + s_i(x),          i = 1, 2,
 *    grad psi_i = gbar_i + grad s_i,
 *    x_u = xi + w o L   (componentwise: x_u[d] = xi[d] + w[d] L[d]).
 *
 *  gbar_i is a constant 3-vector, s_i the SF-28 periodic tricubic B-spline of a
 *  triply periodic fluctuation (both splines on the same grid and periods),
 *  xi the position in some periodic image (normally [0, L), any nearby value is
 *  legal), w the integer wrap counts and L the periods of the spline grid.
 *  The velocity consumed by the trackers is c = grad psi1 x grad psi2.
 *
 *  Why the evaluator takes (xi, w) and not a wrapped point alone: the affine
 *  part makes the labels NON-periodic. Moving one period along axis d changes
 *  psi_i by gbar_i[d] L[d], so a wrapped point does not determine the label
 *  value. The spline is evaluated at xi (full precision near the particle's
 *  cell; SF-28 reduces any finite coordinate to [0, L) itself) and the affine
 *  part at the unwrapped coordinate xi[d] + w[d] L[d].
 *
 *  Floating-point order of SplineLabelPair::operator() (fixed; host and
 *  device perform the same roundings for the affine part):
 *
 *    xu[d] = fma(w[d], L[d], xi[d])                         (one rounding)
 *    psi1  = fma(g1[2], xu[2], fma(g1[1], xu[1], g1[0]*xu[0])) + s1(xi)
 *
 *  i.e. the affine sum first in the order 0, 1, 2, then the spline value; the
 *  same for psi2. With gbar1 = (0,1,0) and a zero spline, psi1 == xu[1]
 *  exactly. The spline part itself may differ between host and device by FMA
 *  contraction inside SF-28 (accepted SF-28 contract: normwise 1e-13).
 *
 *  Scope: triply periodic labels only. Labels non-periodic in x1 (inlet-
 *  anchored Darcy labels on a long domain, SF-29) are OUT OF SCOPE here.
 *  No epsilon, clamp or regularization appears anywhere in this module.
 *
 * ---------------------------------------------------------------------------
 * 2) Label-evaluator concept (what the integrator cores of N1/N2 template on)
 * ---------------------------------------------------------------------------
 *
 *  Any POD type E with
 *
 *    real L[3];   // periods used by wrap_position and the unwrapped bookkeeping
 *    __host__ __device__ void operator()(const real xi[3], const int32_t w[3],
 *                                        LabelSample& out) const;
 *
 *  operator() writes psi1, psi2 and both gradients at the point (xi, w). It
 *  must allocate nothing and must not depend on any state other than its
 *  members (pure function of (xi, w)). SplineLabelPair is the production
 *  model of the concept (GPU kernels, passed by value); tests instantiate the
 *  integrator cores on the host with analytic evaluators.
 *
 * ---------------------------------------------------------------------------
 * 3) Wrap rule (wrap_position)
 * ---------------------------------------------------------------------------
 *
 *  Per axis d (the SF-28 reduction, with the subtraction fused):
 *
 *    q   = floor(xi / L)
 *    xi  = fma(-q, L, xi)                    (one rounding, host == device)
 *    if xi <  0:  xi += L, q -= 1            (rounding guards only)
 *    if xi >= L:  xi -= L, q += 1
 *    w  += q
 *
 *  The two guards are those of SF-28 in the opposite order: testing "< 0"
 *  first guarantees the closed result xi in [0, L) even when the "< 0" guard
 *  produces xi + L == L by rounding (tiny negative xi); the second guard then
 *  maps it to 0. xi_out + w_out L == xi_in holds whenever it is representable.
 *
 * ---------------------------------------------------------------------------
 * 4) Status codes (uint8_t particle status array)
 * ---------------------------------------------------------------------------
 *
 *  0 = active; any nonzero value = the particle is no longer advanced. The
 *  legacy PSPTA engine uses 2 for "exited"; codes 1 and 2 are not reused.
 *  See the kStatus* constants below.
 *
 * ---------------------------------------------------------------------------
 * 5) Injection (inject_box; decision D-8)
 * ---------------------------------------------------------------------------
 *
 *  For particle index p (global index in the arrays) and axis a in {0,1,2}:
 *
 *    h = mix64(seed ^ mix64(p ^ salt[a]))            (inject_hash)
 *    u = (h >> 11) * 2^-53                           (inject_uniform01; u in [0,1),
 *                                                     53 random bits)
 *    x = x0 + u * (x1 - x0)                          (inject_coordinate)
 *
 *  evaluated as x = fma(u, x1 - x0, x0): one rounding of the product-sum, the
 *  same on host and device, so a host reproduction of the positions is
 *  bitwise. mix64 is the SplitMix64 / Murmur3 finalizer; salt[a] are three
 *  fixed, distinct 64-bit constants. Deterministic in (seed, p) only (not in
 *  launch configuration). x0 == x1 gives x == x0 bitwise (u * 0 = +0; the
 *  only exception is x0 = -0.0, which yields +0.0, equal as a value).
 */

#include "../../../core/Scalar.hpp"
#include "../../../numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "../par2_adapter/par2_views.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace streamline_tracker {

// ============================================================================
// Labels
// ============================================================================

/// Values and gradients of the two labels at one point (POD).
struct LabelSample {
    real psi1, psi2;
    real g1[3], g2[3]; ///< grad psi1, grad psi2
};

/**
 * @brief Production label evaluator: constant affine gradient + SF-28 periodic
 *        tricubic spline per label (POD, passed by value into kernels).
 *
 * Build it with make_spline_label_pair (validating). L holds the common
 * periods (Lx, Ly, Lz) of the two spline views. Coefficients may be device
 * (workspace.view()) or host (make_host_view) memory; operator() must be
 * called on the side that owns them.
 */
struct SplineLabelPair {
    interpolation::PeriodicTricubicBSplineView s1, s2;
    real gbar1[3], gbar2[3];
    real L[3];

    __host__ __device__ inline void operator()(const real xi[3], const int32_t w[3],
                                               LabelSample& out) const {
        real xu[3];
        for (int d = 0; d < 3; ++d) {
            xu[d] = fma(static_cast<real>(w[d]), L[d], xi[d]);
        }

        real v1, gx1, gy1, gz1;
        real v2, gx2, gy2, gz2;
        interpolation::evaluate_point(s1, xi[0], xi[1], xi[2], v1, gx1, gy1, gz1);
        interpolation::evaluate_point(s2, xi[0], xi[1], xi[2], v2, gx2, gy2, gz2);

        const real a1 = fma(gbar1[2], xu[2], fma(gbar1[1], xu[1], gbar1[0] * xu[0]));
        const real a2 = fma(gbar2[2], xu[2], fma(gbar2[1], xu[1], gbar2[0] * xu[0]));
        out.psi1 = a1 + v1;
        out.psi2 = a2 + v2;

        out.g1[0] = gbar1[0] + gx1;
        out.g1[1] = gbar1[1] + gy1;
        out.g1[2] = gbar1[2] + gz1;
        out.g2[0] = gbar2[0] + gx2;
        out.g2[1] = gbar2[1] + gy2;
        out.g2[2] = gbar2[2] + gz2;
    }
};

/**
 * @brief Validating factory (host).
 *
 * Throws std::invalid_argument with distinct messages when:
 *  - the s1 or the s2 coefficient pointer is null;
 *  - the two views differ in (nx, ny, nz);
 *  - the two views differ in spacings (hx, hy, hz);
 *  - the two views differ in periods (Lx, Ly, Lz);
 *  - a gbar1 or a gbar2 component is not finite.
 * Comparisons are exact (both splines must come from the same grid).
 */
SplineLabelPair make_spline_label_pair(const interpolation::PeriodicTricubicBSplineView& s1,
                                       const interpolation::PeriodicTricubicBSplineView& s2,
                                       const real gbar1[3], const real gbar2[3]);

// ============================================================================
// Small vector helpers
// ============================================================================

/// c = a x b (c must not alias a or b).
__host__ __device__ inline void cross3(const real a[3], const real b[3], real c[3]) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

/// Euclidean norm |a| (plain sqrt of the sum of squares; no scaling).
__host__ __device__ inline real norm3(const real a[3]) {
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

// ============================================================================
// Periodic wrap with integer bookkeeping (file header section 3)
// ============================================================================

/// Reduce xi[d] to [0, L[d]) and add the removed whole periods to w[d].
/// xi must be finite and L[d] finite and > 0 (not checked here).
__host__ __device__ inline void wrap_position(real xi[3], int32_t w[3], const real L[3]) {
    for (int d = 0; d < 3; ++d) {
        const real Ld = L[d];
        real q = floor(xi[d] / Ld);
        real x = fma(-q, Ld, xi[d]);
        if (x < static_cast<real>(0.0)) {
            x += Ld;
            q -= static_cast<real>(1.0);
        }
        if (x >= Ld) {
            x -= Ld;
            q += static_cast<real>(1.0);
        }
        xi[d] = x;
        w[d] += static_cast<int32_t>(q);
    }
}

// ============================================================================
// Status codes (uint8_t; file header section 4; decision D-9)
// ============================================================================

inline constexpr uint8_t kStatusActive = 0;
inline constexpr uint8_t kStatusNewtonFailed = 10;  ///< projection did not reach tol_psi within max_iter
inline constexpr uint8_t kStatusDegenerate = 11;    ///< |grad psi1 x grad psi2| zero/below threshold or non-finite
inline constexpr uint8_t kStatusSubstepLimit = 12;  ///< per-call substep guard exceeded
inline constexpr uint8_t kStatusStepUnderflow = 13; ///< RK step below min_step
inline constexpr uint8_t kStatusNonFinite = 14;     ///< a non-finite position or velocity appeared

// ============================================================================
// Deterministic injection hash (file header section 5; decision D-8)
// ============================================================================

/// Per-axis salts (x, y, z). Fixed; changing them changes every injection.
inline constexpr uint64_t kInjectSaltX = 0xA1B2C3D4E5F60718ULL;
inline constexpr uint64_t kInjectSaltY = 0x5DEECE66D1F2E3C4ULL;
inline constexpr uint64_t kInjectSaltZ = 0xC2B2AE3D27D4EB4FULL;

/// 64-bit finalizer (Murmur3 fmix64 constants; bijective).
__host__ __device__ inline uint64_t inject_mix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

/// h = mix64(seed ^ mix64(index ^ salt[axis])), axis in {0, 1, 2}.
__host__ __device__ inline uint64_t inject_hash(uint64_t seed, uint64_t index, int axis) {
    const uint64_t salt = axis == 0 ? kInjectSaltX : (axis == 1 ? kInjectSaltY : kInjectSaltZ);
    return inject_mix64(seed ^ inject_mix64(index ^ salt));
}

/// u = (h >> 11) * 2^-53, uniform on the 2^53 doubles k 2^-53 in [0, 1).
__host__ __device__ inline real inject_uniform01(uint64_t seed, uint64_t index, int axis) {
    const real two_m53 = static_cast<real>(1.0) / static_cast<real>(9007199254740992.0); // 2^-53
    return static_cast<real>(inject_hash(seed, index, axis) >> 11) * two_m53;
}

/// x = x0 + u * (x1 - x0), evaluated as fma(u, x1 - x0, x0) (same rounding on host and device).
__host__ __device__ inline real inject_coordinate(real x0, real x1, real u) {
    return fma(u, x1 - x0, x0);
}

// ============================================================================
// Host API (kernels in StreamlineTrackerCommon.cu)
// ============================================================================

/**
 * @brief Uniform injection in the axis-aligned box [p0, p1] for particles
 *        [first, first + count).
 *
 * Position of particle p on axis a: inject_coordinate(lo_a, hi_a,
 * inject_uniform01(seed, p, a)). Sets status = kStatusActive and zeroes the
 * three wrap counters of those particles. Launches on `stream`, allocates
 * nothing, does not synchronize. count == 0 is a no-op.
 *
 * Throws std::invalid_argument (distinct messages) if x/y/z/status or any
 * wrap pointer is null (wrap arrays are REQUIRED: the domain is triply
 * periodic), if first < 0, count < 0 or first + count > p.n, if a box bound is
 * not finite, or if p1 < p0 on an axis.
 */
void inject_box(cudaStream_t stream, const ParticlesSoA<real>& p, real x0, real y0, real z0,
                real x1, real y1, real z1, int first, int count, uint64_t seed);

/**
 * @brief x_u = x + wrapX * Lx (same for y, z) for particles [0, p.n) into uw.
 *
 * Evaluated as fma(wrap, L, x) (same rounding as the unwrapped coordinate of
 * SplineLabelPair). Launches on `stream`, allocates nothing, does not
 * synchronize. p.n == 0 is a no-op after validation.
 *
 * Throws std::invalid_argument (distinct messages) if x/y/z/status or any
 * wrap pointer is null, if p.n < 0, if a period is not finite or not > 0, if
 * uw is not valid, or if uw.capacity < p.n.
 */
void compute_unwrapped(cudaStream_t stream, const ConstParticlesSoA<real>& p, const real L[3],
                       const UnwrappedSoA<real>& uw);

} // namespace streamline_tracker
} // namespace particles
} // namespace physics
} // namespace macroflow3d
