#pragma once

/**
 * @file analytic_pairs.hpp
 * @brief SF-31 test fixtures: analytic label pairs (host + device) for the
 *        streamline-tracker contract tests (pseudo-symplectic tracker, N3;
 *        RK reference, N4).
 *
 * All pairs: L = (1, 1, 1), gbar1 = (0, 1, 0), gbar2 = (0, 0, 1) (pair D:
 * gbar2 = gbar1), labels psi1 = gbar1 . x_u + s1(xi), psi2 = gbar2 . x_u + s2(xi)
 * with x_u = xi + w L (evaluated as fma(w, L, xi), the rounding of
 * SplineLabelPair). Constants a = 0.1, b = 0.08, e = 0.03.
 *
 *   U  s1 = 0,                         s2 = 0
 *   A  s1 = a sin(2 pi x1),            s2 = 0
 *   H  s1 = a sin(2 pi x1),            s2 = a cos(2 pi x1)
 *   B  s1 = a sin(2 pi x1),            s2 = b sin(2 pi x2)
 *   G  s1 = e [sin(2 pi x1) cos(2 pi x3) + 0.5 sin(2 pi (x2 + x3))]
 *      s2 = e [cos(2 pi x1) sin(2 pi x2) + 0.5 cos(2 pi (x1 - x3))]
 *   D  s1 = s2 = a sin(2 pi x1), gbar2 = gbar1 (psi2 = psi1: degenerate)
 *
 * In U, A, H, B the velocity c = grad psi1 x grad psi2 has c1 = 1, so the
 * exact travel time equals the unwrapped x1 displacement. Exact unwrapped
 * motion (exact_position) for U, A, H, B; G has no closed form.
 *
 * ThetaLabels: psi1 = y_u, psi2 = cos(theta) y_u + sin(theta) z_u (degeneracy
 * threshold case), c = (sin theta, 0, 0).
 *
 * Plain C++17 + CUDA qualifiers; no heavy headers (nvcc 11.4).
 */

#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/particles/streamline_tracker/StreamlineTrackerCommon.cuh"

#include <cmath>
#include <cstdint>
#include <vector>

namespace sf31_tests {

using macroflow3d::real;
using macroflow3d::physics::particles::streamline_tracker::LabelSample;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kA = 0.1;
constexpr double kB = 0.08;
constexpr double kE = 0.03;

enum PairId : int { kPairU = 0, kPairA = 1, kPairH = 2, kPairB = 3, kPairG = 4, kPairD = 5 };

inline const char* pair_name(int p) {
    switch (p) {
    case kPairU:
        return "U";
    case kPairA:
        return "A";
    case kPairH:
        return "H";
    case kPairB:
        return "B";
    case kPairG:
        return "G";
    case kPairD:
        return "D";
    default:
        return "?";
    }
}

/// gbar1 and gbar2 of a pair (pair D: gbar2 = gbar1).
inline void pair_gbar(int pair, real g1[3], real g2[3]) {
    g1[0] = 0.0;
    g1[1] = 1.0;
    g1[2] = 0.0;
    g2[0] = 0.0;
    g2[1] = (pair == kPairD) ? 1.0 : 0.0;
    g2[2] = (pair == kPairD) ? 0.0 : 1.0;
}

/// Periodic fluctuations s1, s2 and their gradients at (x, y, z).
__host__ __device__ inline void pair_fluct(int pair, real x, real y, real z, real& s1, real g1[3],
                                           real& s2, real g2[3]) {
    for (int d = 0; d < 3; ++d) {
        g1[d] = 0.0;
        g2[d] = 0.0;
    }
    s1 = 0.0;
    s2 = 0.0;
    if (pair == kPairU) {
        return;
    }
    if (pair == kPairG) {
        const real sx = sin(kTwoPi * x), cx = cos(kTwoPi * x);
        const real sy = sin(kTwoPi * y), cy = cos(kTwoPi * y);
        const real sz = sin(kTwoPi * z), cz = cos(kTwoPi * z);
        const real syz = sin(kTwoPi * (y + z)), cyz = cos(kTwoPi * (y + z));
        const real sxz = sin(kTwoPi * (x - z)), cxz = cos(kTwoPi * (x - z));
        s1 = kE * (sx * cz + 0.5 * syz);
        g1[0] = kE * kTwoPi * cx * cz;
        g1[1] = kE * 0.5 * kTwoPi * cyz;
        g1[2] = kE * (-kTwoPi * sx * sz + 0.5 * kTwoPi * cyz);
        s2 = kE * (cx * sy + 0.5 * cxz);
        g2[0] = kE * (-kTwoPi * sx * sy - 0.5 * kTwoPi * sxz);
        g2[1] = kE * kTwoPi * cx * cy;
        g2[2] = kE * 0.5 * kTwoPi * sxz;
        return;
    }
    // A, H, B, D share s1 = a sin(2 pi x).
    const real sx = sin(kTwoPi * x), cx = cos(kTwoPi * x);
    s1 = kA * sx;
    g1[0] = kA * kTwoPi * cx;
    if (pair == kPairH) {
        s2 = kA * cx;
        g2[0] = -kA * kTwoPi * sx;
    } else if (pair == kPairB) {
        s2 = kB * sin(kTwoPi * y);
        g2[1] = kB * kTwoPi * cos(kTwoPi * y);
    } else if (pair == kPairD) {
        s2 = s1;
        g2[0] = g1[0];
    }
}

/**
 * @brief Analytic label evaluator (POD; satisfies the label-evaluator concept
 *        of StreamlineTrackerCommon.cuh). Fluctuations at xi, affine part at
 *        x_u = fma(w, L, xi).
 */
struct AnalyticLabels {
    real L[3];
    real gbar1[3], gbar2[3];
    int pair;

    __host__ __device__ inline void operator()(const real xi[3], const int32_t w[3],
                                               LabelSample& out) const {
        real xu[3];
        for (int d = 0; d < 3; ++d) {
            xu[d] = fma(static_cast<real>(w[d]), L[d], xi[d]);
        }
        real s1, s2, gs1[3], gs2[3];
        pair_fluct(pair, xi[0], xi[1], xi[2], s1, gs1, s2, gs2);
        const real a1 = fma(gbar1[2], xu[2], fma(gbar1[1], xu[1], gbar1[0] * xu[0]));
        const real a2 = fma(gbar2[2], xu[2], fma(gbar2[1], xu[1], gbar2[0] * xu[0]));
        out.psi1 = a1 + s1;
        out.psi2 = a2 + s2;
        for (int d = 0; d < 3; ++d) {
            out.g1[d] = gbar1[d] + gs1[d];
            out.g2[d] = gbar2[d] + gs2[d];
        }
    }
};

inline AnalyticLabels make_analytic_labels(int pair) {
    AnalyticLabels e{};
    e.L[0] = 1.0;
    e.L[1] = 1.0;
    e.L[2] = 1.0;
    pair_gbar(pair, e.gbar1, e.gbar2);
    e.pair = pair;
    return e;
}

/**
 * @brief psi1 = y_u, psi2 = cos(theta) y_u + sin(theta) z_u (POD). cos and sin
 *        are computed once on the host and stored, so host and device see the
 *        same gradients bitwise.
 */
struct ThetaLabels {
    real L[3];
    real ct, st;

    __host__ __device__ inline void operator()(const real xi[3], const int32_t w[3],
                                               LabelSample& out) const {
        const real yu = fma(static_cast<real>(w[1]), L[1], xi[1]);
        const real zu = fma(static_cast<real>(w[2]), L[2], xi[2]);
        out.psi1 = yu;
        out.psi2 = fma(st, zu, ct * yu);
        out.g1[0] = 0.0;
        out.g1[1] = 1.0;
        out.g1[2] = 0.0;
        out.g2[0] = 0.0;
        out.g2[1] = ct;
        out.g2[2] = st;
    }
};

inline ThetaLabels make_theta_labels(double theta) {
    ThetaLabels e{};
    e.L[0] = 1.0;
    e.L[1] = 1.0;
    e.L[2] = 1.0;
    e.ct = std::cos(theta);
    e.st = std::sin(theta);
    return e;
}

/**
 * @brief Exact unwrapped position at time t from the unwrapped start x0 for
 *        pairs U, A, H, B (c1 = 1). Returns false for pairs without a closed form.
 */
inline bool exact_position(int pair, const double x0[3], double t, double x[3]) {
    x[0] = x0[0] + t;
    x[1] = x0[1];
    x[2] = x0[2];
    if (pair == kPairU) {
        return true;
    }
    if (pair != kPairA && pair != kPairH && pair != kPairB) {
        return false;
    }
    x[1] = x0[1] - kA * (std::sin(kTwoPi * x[0]) - std::sin(kTwoPi * x0[0]));
    if (pair == kPairH) {
        x[2] = x0[2] - kA * (std::cos(kTwoPi * x[0]) - std::cos(kTwoPi * x0[0]));
    } else if (pair == kPairB) {
        x[2] = x0[2] - kB * (std::sin(kTwoPi * x[1]) - std::sin(kTwoPi * x0[1]));
    }
    return true;
}

/// Fluctuations s1, s2 sampled at the cell centres ((i + 1/2) h) of g, layout i + nx (j + ny k).
inline void sample_fluctuations(int pair, const macroflow3d::Grid3D& g, std::vector<real>& s1,
                                std::vector<real>& s2) {
    s1.assign(g.num_cells(), 0.0);
    s2.assign(g.num_cells(), 0.0);
    for (int k = 0; k < g.nz; ++k) {
        for (int j = 0; j < g.ny; ++j) {
            for (int i = 0; i < g.nx; ++i) {
                const real x = (static_cast<real>(i) + 0.5) * g.dx;
                const real y = (static_cast<real>(j) + 0.5) * g.dy;
                const real z = (static_cast<real>(k) + 0.5) * g.dz;
                real v1, v2, gg1[3], gg2[3];
                pair_fluct(pair, x, y, z, v1, gg1, v2, gg2);
                s1[g.idx(i, j, k)] = v1;
                s2[g.idx(i, j, k)] = v2;
            }
        }
    }
}

/// Helix (pair H) constants: curvature kappa and speed |c| (both constant).
inline double helix_kappa() {
    return 4.0 * kPi * kPi * kA / (1.0 + 4.0 * kPi * kPi * kA * kA);
}
inline double helix_speed() {
    return std::sqrt(1.0 + 4.0 * kPi * kPi * kA * kA);
}

} // namespace sf31_tests
