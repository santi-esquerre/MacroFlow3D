#!/usr/bin/env python3
"""SF-41 RR8: write the pre-registered block-bootstrap index matrix of threshold F (D_P standard errors).

  make_bootstrap_blocks_f.py [--out PATH]

The matrix is numpy.random.default_rng(20261025).integers(0, 4, size=(2000, 4)) saved as little-endian int64 (2000, 4)
(analysis/bootstrap_blocks_f.npy); its sha256 is the sha256 of m.astype('<i8').tobytes() and is committed in
analysis/bootstrap_blocks_f.sha256.  f_verdict.py LOADS the committed file and refuses a sha256 different from
PREREGISTERED_SHA256 below (it never regenerates).  Python 3 + numpy only.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

SEED = 20261025
B = 2000
NBLOCKS = 4
PREREGISTERED_SHA256 = "bf45a8cb390d691b7c83a8db72964919d498dd719f98d1e652c8327179e8d0fd"  # numpy 2.5.0 (PCG64 stream)


def matrix():
    return np.random.default_rng(SEED).integers(0, NBLOCKS, size=(B, NBLOCKS)).astype("<i8")


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(here), "analysis", "bootstrap_blocks_f.npy"))
    ap.add_argument("--print-sha", action="store_true")
    args = ap.parse_args(argv)
    m = matrix()
    sha = hashlib.sha256(m.tobytes()).hexdigest()
    print("shape=%s dtype=%s sha256=%s numpy=%s" % (m.shape, m.dtype, sha, np.__version__))
    if args.print_sha:
        return 0
    if sha != PREREGISTERED_SHA256:
        print("ERROR: sha256 differs from the pre-registered %s (numpy %s)" % (PREREGISTERED_SHA256, np.__version__))
        return 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, m)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
