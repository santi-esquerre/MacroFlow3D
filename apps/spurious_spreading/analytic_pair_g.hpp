#pragma once

/**
 * @file analytic_pair_g.hpp
 * @brief SF-32 N2a: analytic label pairs of the `spurious_spreading`
 *        instrument, with a configurable amplitude (host + device).
 *
 * The formulas are copied from the SF-31 test fixture
 * `tests/streamline_tracker/analytic_pairs.hpp` (apps must not include test
 * headers); there the amplitudes are fixed constants (a = 0.1, b = 0.08,
 * e = 0.03), here they are parameters. All pairs have L = (1, 1, 1),
 * gbar1 = (0, 1, 0), gbar2 = (0, 0, 1) (vbar = 1) and
 * psi_i = gbar_i . x_u + s_i(x):
 *
 *   U  s1 = 0,                         s2 = 0                    (amplitude must be 0)
 *   A  s1 = a sin(2 pi x1),            s2 = 0                    (a = amplitude; SF-31: 0.1)
 *   B  s1 = a sin(2 pi x1),            s2 = b sin(2 pi x2)       (a = amplitude, b = amplitude_b;
 * SF-31: 0.1, 0.08) G  s1 = e [sin(2 pi x1) cos(2 pi x3) + 0.5 sin(2 pi (x2 + x3))] s2 = e [cos(2
 * pi x1) sin(2 pi x2) + 0.5 cos(2 pi (x1 - x3))]   (e = amplitude; SF-31: 0.03, SF-32: 0.05)
 *
 * In the instrument these functions are only SAMPLED at the cell centres
 * ((i + 1/2) h) of the label grid; the labels actually tracked are the SF-28
 * splines of those samples (understanding.md 3.1, 3.5), so the closed forms
 * serve for sampling and for the exact `min |c|` over the cell centres that
 * is printed next to the spline's (a check of the sampling, not an input).
 *
 * The gradients are those of the closed forms (used only for the exact
 * `min |c|`). Plain C++17 + CUDA qualifiers; no heavy headers (nvcc 11.4).
 */

#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h> // __host__ / __device__ (this header is used from .cu files only)

namespace spurious_spreading {

enum class AnalyticPair : int { U = 0, A = 1, B = 2, G = 3 };

inline const char* analytic_pair_name(AnalyticPair p) {
    switch (p) {
    case AnalyticPair::U:
        return "U";
    case AnalyticPair::A:
        return "A";
    case AnalyticPair::B:
        return "B";
    case AnalyticPair::G:
        return "G";
    }
    return "?";
}

inline AnalyticPair analytic_pair_from_name(const std::string& s) {
    if (s == "U")
        return AnalyticPair::U;
    if (s == "A")
        return AnalyticPair::A;
    if (s == "B")
        return AnalyticPair::B;
    if (s == "G")
        return AnalyticPair::G;
    throw std::invalid_argument("unknown analytic pair '" + s + "' (expected U, A, B or G)");
}

/// Pair parameters (POD).
struct AnalyticPairParams {
    AnalyticPair pair;
    double amplitude;   ///< a (A, B) or e (G); must be 0 for U
    double amplitude_b; ///< b of pair B (ignored otherwise)
};

/// Periodic fluctuations s1, s2 and their gradients at (x, y, z).
__host__ __device__ inline void analytic_pair_fluct(const AnalyticPairParams& prm, double x,
                                                    double y, double z, double& s1, double g1[3],
                                                    double& s2, double g2[3]) {
    const double two_pi = 2.0 * 3.141592653589793238462643383279502884;
    for (int d = 0; d < 3; ++d) {
        g1[d] = 0.0;
        g2[d] = 0.0;
    }
    s1 = 0.0;
    s2 = 0.0;
    switch (prm.pair) {
    case AnalyticPair::U:
        return;
    case AnalyticPair::G: {
        const double e = prm.amplitude;
        const double sx = sin(two_pi * x), cx = cos(two_pi * x);
        const double sy = sin(two_pi * y), cy = cos(two_pi * y);
        const double sz = sin(two_pi * z), cz = cos(two_pi * z);
        const double syz = sin(two_pi * (y + z)), cyz = cos(two_pi * (y + z));
        const double sxz = sin(two_pi * (x - z)), cxz = cos(two_pi * (x - z));
        s1 = e * (sx * cz + 0.5 * syz);
        g1[0] = e * two_pi * cx * cz;
        g1[1] = e * 0.5 * two_pi * cyz;
        g1[2] = e * (-two_pi * sx * sz + 0.5 * two_pi * cyz);
        s2 = e * (cx * sy + 0.5 * cxz);
        g2[0] = e * (-two_pi * sx * sy - 0.5 * two_pi * sxz);
        g2[1] = e * two_pi * cx * cy;
        g2[2] = e * 0.5 * two_pi * sxz;
        return;
    }
    case AnalyticPair::A:
    case AnalyticPair::B: {
        const double a = prm.amplitude;
        s1 = a * sin(two_pi * x);
        g1[0] = a * two_pi * cos(two_pi * x);
        if (prm.pair == AnalyticPair::B) {
            const double b = prm.amplitude_b;
            s2 = b * sin(two_pi * y);
            g2[1] = b * two_pi * cos(two_pi * y);
        }
        return;
    }
    }
}

/// |grad psi1 x grad psi2| of the closed form at (x, y, z) (gbar1 = e2, gbar2 = e3).
inline double analytic_pair_abs_c(const AnalyticPairParams& prm, double x, double y, double z) {
    double s1, s2, g1[3], g2[3];
    analytic_pair_fluct(prm, x, y, z, s1, g1, s2, g2);
    g1[1] += 1.0;
    g2[2] += 1.0;
    const double c0 = g1[1] * g2[2] - g1[2] * g2[1];
    const double c1 = g1[2] * g2[0] - g1[0] * g2[2];
    const double c2 = g1[0] * g2[1] - g1[1] * g2[0];
    return std::sqrt(c0 * c0 + c1 * c1 + c2 * c2);
}

} // namespace spurious_spreading
