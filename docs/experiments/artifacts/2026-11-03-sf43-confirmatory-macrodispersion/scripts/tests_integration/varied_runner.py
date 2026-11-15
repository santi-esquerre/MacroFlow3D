"""The fake runners of ``sf43camp/tests_executor/fake_runner.py`` with seed- and arm-dependent variation of the synthetic outputs.

    python3 varied_runner.py consumer <lagrangian_ensemble args>      |      python3 varied_runner.py par2 config.yaml

The unchanged fake writes the SAME longitudinal numbers for every seed and every resolution (``alpha_LP`` has zero variance, a ladder has no
differences), so the analysis (correctly) cannot compute a correlation or a ladder bound on it.  This wrapper multiplies, deterministically from the
field seed and the arm id, the longitudinal clock variance of the consumer passes (``c_tau``), the transverse coefficient (``a_T``) and the
Par2 plane variances.  The numbers remain synthetic: the integration test asserts wiring, never a value.
"""
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_DIR = os.path.join(HERE, "..", "sf43camp", "tests_executor")
sys.path.insert(0, os.path.abspath(FAKE_DIR))

import fake_runner as F  # noqa: E402  (also puts the SF-42 scripts on sys.path)
from sf42red.tests import synth  # noqa: E402


def jitter(seed, arm, scale):
    a = ((int(seed) * 7919) % 101) / 100.0 - 0.5          # seed part in [-0.5, 0.5]
    b = (zlib.crc32(arm.encode()) % 7) / 7.0 - 0.5         # arm part
    return 1.0 + scale * a + 0.6 * scale * b


def arm_of_out(path):
    """``.../segments/<arm>/<unit>/out`` -> arm id."""
    parts = os.path.normpath(path).split(os.sep)
    arm = parts[parts.index("segments") + 1] if "segments" in parts else "?"
    return arm[:-2] if arm.startswith("sm_det_") else arm        # the two members of a determinism pair must write identical outputs


_write_pass = synth.write_pass


def write_pass(root, **kw):
    arm = arm_of_out(root)
    seed = kw.get("seed", 777)
    kw["c_tau"] = kw.get("c_tau", 0.02) * jitter(seed, arm + "L", 0.8)
    kw["a_T"] = kw.get("a_T", 0.03) * jitter(seed, arm + "T", 0.4)
    return _write_pass(root, **kw)


_write_par2 = synth.write_par2


def write_par2(path, rows):
    arm = arm_of_out(os.getcwd())
    r = int(os.path.basename(path).split("_")[1])
    s0 = int(os.path.basename(os.path.normpath(os.getcwd())).rsplit("_s", 1)[1])
    f = jitter(s0 + r, arm, 0.8)
    out = []
    for row in rows:
        row = list(row)
        for i in (7, 8):                 # per-plane variances around 1/12: scale the part above the injection variance
            row[i] = (row[i] - 1.0 / 12) * f + 1.0 / 12
        for i in (4, 12, 13):
            row[i] = row[i] * f
        out.append(row)
    return _write_par2(path, out)


synth.write_pass = write_pass
synth.write_par2 = write_par2

if __name__ == "__main__":
    sys.exit({"consumer": F.consumer, "par2": F.par2}[sys.argv[1]](sys.argv[2:]))
