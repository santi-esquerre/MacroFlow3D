"""First-order transverse dispersivity of SF-43's plane estimand IN THE SIMULATED BOX (decision H3, H17; review section 2.1, 3.b).

The SF-18 generator is torus-periodic on the flow grid (`src/physics/streamfunctions/PeriodicGaussianField.cuh`; medium (b) calls it on
the nx x ny x nz grid, so the field repeats every L1 along the mean flow as well as every W transversally). At first order the
transverse velocity is u2 = -U (k1 k2 / k^2) Y_hat on the discrete lattice k = 2 pi (m1/L1, m2/W, m3/W), so along the mean path
    C22(s) = U^2 sum_k (k1 k2 / k^2)^2 S_d(k) cos(k1 s),     S_d(k) = S(k) (2 pi)^3 / (L1 W^2),
    S(k) = sigma^2 lambda^3 / (8 pi^(3/2)) exp(-k^2 lambda^2 / 4)            (C_Y(r) = sigma^2 exp(-(r/lambda)^2)),
    X22(d) = 2 sum_k U^2 (k1 k2/k^2)^2 S_d(k) (1 - cos(k1 d)) / k1^2  (k1 != 0; the k1 = 0 terms carry no transverse velocity),
in units lambda = 1, U = 1, sigma^2 = 1. The Dirichlet faces of medium (b) are not modelled (they perturb the last 5-10 lambda).
The estimand is half the OLS slope of X22 over the plane list. Compare with `firstorder_window.py` (unbounded medium).

Usage: python3 firstorder_box.py
"""
import numpy as np


def x22_box(d, L1, W, kmax=12.0):
    m1 = np.arange(1, int(np.ceil(kmax * L1 / (2 * np.pi))) + 1)
    m2 = np.arange(-int(np.ceil(kmax * W / (2 * np.pi))), int(np.ceil(kmax * W / (2 * np.pi))) + 1)
    k1 = 2 * np.pi * m1 / L1
    k2 = 2 * np.pi * m2 / W
    K1, K2, K3 = np.meshgrid(k1, k2, k2, indexing="ij")
    k2sq = K1 ** 2 + K2 ** 2 + K3 ** 2
    S = (1.0 / (8 * np.pi ** 1.5)) * np.exp(-k2sq / 4.0) * (2 * np.pi) ** 3 / (L1 * W * W)
    A = (K1 * K2 / k2sq) ** 2 * S  # transverse spectral weight per mode (x2 component)
    w = 2.0 * A.sum(axis=(1, 2))  # factor 2: +-k1 (m1 < 0 mirrors m1 > 0)
    d = np.atleast_1d(np.asarray(d, dtype=float))
    return np.array([2.0 * np.sum(w * (1 - np.cos(k1 * dd)) / k1 ** 2) for dd in d])


def half_slope(lo, hi, L1, W, step=1.0):
    d = np.arange(lo, hi + 0.5 * step, step)
    return 0.5 * np.polyfit(d, x22_box(d, L1, W), 1)[0]


if __name__ == "__main__":
    cases = [("production W 24, L1 104 (sigma^2 0.25, 1, 2.25)", 24.0, 104.0, (25, 94)),
             ("production W 12, L1 104 (sigma^2 4)", 12.0, 104.0, (25, 94)),
             ("width arm W 48, L1 104", 48.0, 104.0, (25, 94)),
             ("width arm W 24, L1 104 at sigma^2 4", 24.0, 104.0, (25, 94)),
             ("distance arm W 24, L1 208, window [25, 94]", 24.0, 208.0, (25, 94)),
             ("distance arm W 24, L1 208, window [94, 194]", 24.0, 208.0, (94, 194)),
             ("transverse periodicity only (L1 = 2000), W 24", 24.0, 2000.0, (25, 94))]
    for name, W, L1, (lo, hi) in cases:
        a = half_slope(lo, hi, L1, W)
        extra = ""
        if (lo, hi) == (25, 94):
            extra = "; halves [25, 59.5] %+.3e, [59.5, 94] %+.3e" % (half_slope(25, 59.5, L1, W, 0.5), half_slope(59.5, 94, L1, W, 0.5))
        print("%-52s alpha_T^FO,box / (sigma^2 lambda) on [%g, %g] = %+.3e%s" % (name, lo, hi, a, extra))
    # check against the unbounded continuum at short distance (large box)
    d = np.array([1.0, 2.0, 4.0, 8.0])
    print("X22 at d = 1, 2, 4, 8 in a 400 x 120 x 120 box:", np.round(x22_box(d, 400.0, 120.0), 5), "(unbounded continuum tends to 1/3)")
