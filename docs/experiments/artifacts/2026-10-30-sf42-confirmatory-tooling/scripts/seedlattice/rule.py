"""Exact reproduction of the lagrangian_ensemble seed lattice and its RR12 seed-on-face check.

Production code mirrored (base 3164e4f):
  seeds  LagrangianTracker.cu make_seed_lattice:
         s = wa + (wb - wa) * ((double(m) + 0.5) / M)           (binary64, this operation order)
  RR12   EnsembleRunner.cu find_seed_on_face, n_transverse = ny:
         v = double(ny) * s;  on face  <=>  !(|v - round(v)| > 1e-6)   (NaN counts as on face)
  window --window w2a,w2b,w3a,w3b parsed by strtod from the decimal strings (Python float() is the same
         correctly rounded conversion); need w2a < w2b and w3a < w3b; --np-side M in [1, 4096].
The seed coordinates of the two axes are independent (axis 2 depends on m2 only, axis 3 on m3 only), so the
check is exact on two length-M vectors. Probe order of the runner: p = m2 * M + m3 ascending, axis 2 then 3.
"""
import math
import sys

import numpy as np

TOL = 1e-6
M_MAX = 4096


def parse_window(window):
    """'w2a,w2b,w3a,w3b' (the runner's strings) -> 4 floats; same validity rules as the runner."""
    parts = window.split(",") if isinstance(window, str) else list(window)
    if len(parts) != 4:
        raise ValueError("window: expected w2a,w2b,w3a,w3b")
    w = [float(x) for x in parts]
    if not all(math.isfinite(x) for x in w):
        raise ValueError("window: values must be finite")
    if not (w[0] < w[1]) or not (w[2] < w[3]):
        raise ValueError("window: need w2a < w2b and w3a < w3b")
    return w


def _round_half_away(v):
    """std::round on a float64 array (exact: the fraction v - floor(v) is exactly representable)."""
    a = np.abs(v)
    f = np.floor(a)
    return np.copysign(np.where(a - f >= 0.5, f + 1.0, f), v)


def axis_seeds(wa, wb, M):
    """s_m = wa + (wb - wa) * ((double(m) + 0.5) / M), m = 0..M-1."""
    m = np.arange(M, dtype=np.float64)
    return wa + (wb - wa) * ((m + 0.5) / float(M))


def axis_on_face(wa, wb, M, ny):
    """Boolean vector: seed m of this axis fails RR12 (NaN counts as on face, as in the runner)."""
    with np.errstate(all="ignore"):
        v = float(ny) * axis_seeds(wa, wb, M)
        return ~(np.abs(v - _round_half_away(v)) > TOL)


def on_face(ny, window, M):
    """-> (True, (m2, m3, axis)) with the runner's FIRST offending seed (axis 2 or 3), else (False, None)."""
    if not (1 <= M <= M_MAX):
        raise ValueError("M must be in [1, %d]" % M_MAX)
    w2a, w2b, w3a, w3b = parse_window(window)
    b2 = np.flatnonzero(axis_on_face(w2a, w2b, M, ny))
    b3 = np.flatnonzero(axis_on_face(w3a, w3b, M, ny))
    cand = []  # (p, axis, m2, m3): the runner scans p ascending, axis 2 before 3
    if b2.size:
        cand.append((int(b2[0]) * M, 2, int(b2[0]), 0))
    if b3.size:
        cand.append((int(b3[0]), 3, 0, int(b3[0])))  # smallest p with a bad m3 is m2 = 0
    if not cand:
        return False, None
    _, axis, m2, m3 = min(cand)
    return True, (m2, m3, axis)


def admissible(ny, window, M_range):
    """Sorted list of M in M_range (iterable of ints) that pass RR12 on grid ny."""
    return [M for M in M_range if not on_face(ny, window, M)[0]]


def admissible_all(nys, window, M_range):
    """M passing on EVERY ny in nys."""
    return [M for M in M_range if not any(on_face(ny, window, M)[0] for ny in nys)]


def v2(n):
    return (n & -n).bit_length() - 1


def line(ny, window, M):
    """The probe's output line for this case (seedlattice_probe prints the same format)."""
    bad, where = on_face(ny, window, M)
    return "%d %s %d %s" % (ny, window, M, "ON_FACE m2=%d m3=%d axis=%d" % where if bad else "OK")


if __name__ == "__main__":
    print(line(int(sys.argv[1]), sys.argv[2], int(sys.argv[3])))
