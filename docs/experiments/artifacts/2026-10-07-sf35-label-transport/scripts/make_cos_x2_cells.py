#!/usr/bin/env python3
"""Cell-sample inputs for the SF-35 positive controls (driver option --cells).

  make_cos_x2_cells.py N out.npy           Y_cell = cos(2 pi (j + 1/2) h), h = 1/N   (A2: k = exp(0.5 cos 2 pi x2) at --eps 0.5)
  make_cos_x2_cells.py N out.npy --zeros   Y_cell = 0                                  (A1: k = 1)

Grid3D layout: flat index i + N (j + N k) with x1 (index i) fastest, i.e. a C-order array of shape
(N, N, N) indexed [k, j, i].  Y depends on the x2 index j only.  float64 .npy.
"""
import argparse
import math
import sys

import numpy as np


def make(n, zeros=False):
    j = np.arange(n, dtype=np.float64)
    y2 = np.zeros(n) if zeros else np.cos(2.0 * math.pi * (j + 0.5) / n)
    # [k, j, i]: value depends on j only
    return np.ascontiguousarray(np.broadcast_to(y2[None, :, None], (n, n, n)), dtype=np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("N", type=int, help="cells per direction")
    ap.add_argument("out", help="output .npy path")
    ap.add_argument("--zeros", action="store_true", help="write Y = 0 (the k = 1 control) instead of cos(2 pi x2)")
    a = ap.parse_args(argv)
    if a.N < 2:
        print("N must be >= 2", file=sys.stderr)
        return 2
    import os
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    arr = make(a.N, a.zeros)
    np.save(a.out, arr)
    flat = arr.reshape(-1)
    # layout self-check: flat index i + N (j + N k) -> value of j
    n = a.N
    i, jj, k = 1, 3 % n, 2 % n
    expect = 0.0 if a.zeros else math.cos(2.0 * math.pi * (jj + 0.5) / n)
    ok = abs(flat[i + n * (jj + n * k)] - expect) <= 1e-15
    print("wrote %s shape=%s %s layout_check=%s" % (a.out, arr.shape, "zeros" if a.zeros else "cos(2 pi x2)", ok))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
