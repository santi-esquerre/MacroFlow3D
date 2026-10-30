"""Runs seedlattice_probe (production find_seed_on_face) on data/cases.txt and compares every line with the
Python predictor and with the committed fixture data/probe_output.txt.  Exit 0: identical; 1: any mismatch.
Usage (scripts root):  python3 -W error -m seedlattice.verify_probe <path to seedlattice_probe>
"""
import subprocess
import sys

from .make_cases import CASES, FIXTURE
from .rule import line


def read_cases():
    with open(CASES) as f:
        return [tuple(l.split()) for l in f if l.strip() and not l.startswith("#")]


def predictions():
    return [line(int(ny), w, int(M)) for ny, w, M in read_cases()]


def main(probe):
    got = subprocess.run([probe, CASES], check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.splitlines()
    want = predictions()
    with open(FIXTURE) as f:
        fixture = f.read().splitlines()
    bad = [(i, g, w) for i, (g, w) in enumerate(zip(got, want)) if g != w]
    if len(got) != len(want):
        bad.append((-1, "%d probe lines" % len(got), "%d predictions" % len(want)))
    for i, g, w in bad[:20]:
        print("MISMATCH case %d: probe '%s' vs predictor '%s'" % (i, g, w))
    if got != fixture:
        print("FIXTURE DIFFERS from the probe output (regenerate: see make_cases.py)")
        return 1
    n_face = sum(1 for g in got if "ON_FACE" in g)
    print("seedlattice_probe: %d cases (%d ON_FACE, %d OK), %d mismatches" % (len(got), n_face, len(got) - n_face, len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
