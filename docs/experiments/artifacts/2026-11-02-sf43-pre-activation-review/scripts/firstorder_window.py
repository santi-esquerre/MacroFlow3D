"""First-order (linear) purely advective transverse dispersivity of SF-43's plane estimand, Gaussian covariance, 3D isotropic.

alpha_T(tau) / (sigma^2 lambda) = (tau / 2) int_0^1 mu^2 (1 - mu^2) exp(-tau^2 mu^2) dmu   (SF-39 pre-activation review, section 5.1;
re-derived in wave-number form by the SF-43 review's theory agent), tau = travel distance / lambda; X_22(tau) = 2 int_0^tau alpha_T.
SF-43's estimand is half the ordinary least-squares slope of X_22 over the plane list d = 25, 26, ..., 94 lambda (spacing 1 lambda);
this is the unbounded-domain, one-particle value (periodic boxes and per-realization centring make it smaller, so it is an upper value
of the first-order effective estimand: a conservative null for decision H3). Also: the two halves and the distance window.

Usage: python3 firstorder_window.py
"""
import numpy as np
from scipy.integrate import quad


def alpha_t(tau):
    return 0.5 * tau * quad(lambda m: m * m * (1 - m * m) * np.exp(-tau * tau * m * m), 0.0, 1.0, limit=200)[0]


def x22(d):
    return 2.0 * quad(alpha_t, 0.0, d, limit=400)[0]


def window_alpha(lo, hi, step=1.0):
    d = np.arange(lo, hi + 0.5 * step, step)
    return 0.5 * np.polyfit(d, np.array([x22(v) for v in d]), 1)[0]


if __name__ == "__main__":
    w = {}
    for lo, hi, step in ((25, 94, 1.0), (25, 59.5, 0.5), (59.5, 94, 0.5), (94, 194, 1.0)):
        w[(lo, hi)] = window_alpha(lo, hi, step)
        print("window [%g, %g] lambda: alpha_T^FO / (sigma^2 lambda) = %.4e   (local alpha_T at the ends: %.3e, %.3e)"
              % (lo, hi, w[(lo, hi)], alpha_t(lo), alpha_t(hi)))
    print("ratio first half / second half = %.2f ; ratio [25,94] / [94,194] = %.2f"
          % (w[(25, 59.5)] / w[(59.5, 94)], w[(25, 94)] / w[(94, 194)]))
    for s2 in (0.25, 1.0, 2.25, 4.0):
        print("sigma^2 = %-5g alpha_T^FO[25,94] = %.3e lambda" % (s2, s2 * w[(25, 94)]))
