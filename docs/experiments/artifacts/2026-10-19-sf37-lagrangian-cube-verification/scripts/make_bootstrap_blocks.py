#!/usr/bin/env python3
"""SF-37: write the pre-registered block-bootstrap index matrix.

  make_bootstrap_blocks.py [--out PATH]

The matrix is numpy.random.default_rng(20261017).integers(0, 16, size=(2000, 16)) saved as little-endian
int64 (2000, 16).  The sha256 of m.astype('<i8').tobytes() must equal the value pre-registered in the SF-37
activation row (16c1a5b9...af39); the script refuses to write anything else.  The digests LOAD the committed
file (analysis/bootstrap_blocks.npy); they never regenerate it.  Python 3 + numpy only.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

SEED = 20261017
B = 2000
NBLOCKS = 16
PREREGISTERED_SHA256 = "16c1a5b9d9fe062576d5f18fc9d54127db76472c7fbe64493201d8cffc66af39"


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(here), "analysis", "bootstrap_blocks.npy"))
    args = ap.parse_args(argv)
    m = np.random.default_rng(SEED).integers(0, NBLOCKS, size=(B, NBLOCKS)).astype("<i8")
    sha = hashlib.sha256(m.astype("<i8").tobytes()).hexdigest()
    print("shape=%s dtype=%s sha256=%s" % (m.shape, m.dtype, sha))
    if sha != PREREGISTERED_SHA256:
        print("ERROR: sha256 differs from the pre-registered %s (numpy %s)" % (PREREGISTERED_SHA256, np.__version__))
        return 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, m)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
