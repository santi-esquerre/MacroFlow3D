"""RR3(iii) byte check on synthetic trees (a multi-pass layout and its single-pass twin)."""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

from sf39 import bytecheck as B

PASS_FILES = {
    "planes_x23.npy": b"\x93NUMPY-planes",
    "moments_planes.csv": b"x,var\n1,2\n",
    "moments_time.csv": b"t,var\n1,2\n",
    "exits_planes.csv": b"p,n\n1,0\n",
    "exits_time.csv": b"t,n\n1,0\n",
    "track.json": b'{"times": [1, 2]}\n',
    "twin/moments_planes.csv": b"x,var\n1,3\n",
    "twin/moments_time.csv": b"t,var\n1,3\n",
}
ROOT_FILES = {"field.json": b'{"var": 1}\n', "flow_stats.json": b'{"qbar": 0.5}\n', "solver_b.json": b'{"verdict": "converged"}\n',
              "pollock/pollock_weights.npy": b"\x93NUMPY-w", "pollock/pollock.json": b'{"times": [1]}\n'}
STATS = {"stats/realization_0000_timeseries.csv": b"t,m\n1,1\n", "stats_twin/realization_0000_timeseries.csv": b"t,m\n1,2\n"}
PASS_STATUS = {"tracker_plane_pass": {"status_hist": [10, 0]}, "tracker_time_pass": {"status_hist": [10, 0]}, "Np": 10000,
               "seed_backflow": 0}
POLLOCK = {"pollock": {"seed_nonpos": 0}}


def put(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else json.dumps(data, sort_keys=True).encode())


def make(root, multi):
    R = os.path.join(root, "realizations", "r_0000")
    for n, d in ROOT_FILES.items():
        put(os.path.join(R, n), d)
    P = os.path.join(R, "hmax_8") if multi else R
    for n, d in PASS_FILES.items():
        put(os.path.join(P, n), d)
    if multi:  # the other passes exist only in the multi run
        put(os.path.join(R, "hmax_16", "planes_x23.npy"), b"other")
        put(os.path.join(R, "hmax_16", "track.json"), b"{}")
        put(os.path.join(R, "status.json"), dict(Np=10000, seed_backflow=0, hmax_div=[8, 16, 32], **POLLOCK))
        put(os.path.join(P, "status.json"), dict(PASS_STATUS, hmax_div=8))
    else:
        put(os.path.join(R, "status.json"), dict(PASS_STATUS, **POLLOCK))
    for n, d in STATS.items():
        put(os.path.join(root, "hmax_8", n) if multi else os.path.join(root, n), d)
    for r in (R,):
        put(os.path.join(r, "DONE"), b"sig-multi" if multi else b"sig-single")
        put(os.path.join(r, "timings.json"), b'{"s": 1.5}' if multi else b'{"s": 2.5}')


class TestBytecheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.m = os.path.join(self.tmp, "multi")
        self.s = os.path.join(self.tmp, "single")
        make(self.m, True)
        make(self.s, False)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_check(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = B.main([self.m, self.s])
        return rc, buf.getvalue()

    def test_identical_excluding_done_and_timings(self):
        rc, out = self.run_check()
        self.assertEqual(rc, 0, out)
        self.assertIn("BYTECHECK IDENTICAL", out)

    def test_each_mutation_is_detected(self):
        R = "realizations/r_0000/"
        muts = [
            ("m", R + "hmax_8/planes_x23.npy"), ("s", R + "track.json"), ("s", R + "twin/moments_time.csv"),
            ("m", R + "field.json"), ("s", R + "pollock/pollock_weights.npy"), ("m", "hmax_8/stats/realization_0000_timeseries.csv"),
            ("s", "stats_twin/realization_0000_timeseries.csv"), ("m", R + "solver_b.json"),
        ]
        for base, rel in muts:
            with self.subTest(rel=rel, base=base):
                self.tearDown()
                self.setUp()
                p = os.path.join(self.m if base == "m" else self.s, rel)
                with open(p, "rb") as f:
                    data = f.read()
                put(p, data + b"x")
                rc, out = self.run_check()
                self.assertEqual(rc, 1, out)
                self.assertIn("DIFF", out)

    def test_missing_and_extra_files(self):
        os.remove(os.path.join(self.s, "realizations/r_0000/exits_time.csv"))
        rc, out = self.run_check()
        self.assertEqual(rc, 1)
        self.assertIn("exits_time.csv", out)
        self.tearDown()
        self.setUp()
        put(os.path.join(self.m, "realizations/r_0000/hmax_8/exits_extra.csv"), b"z")
        self.assertEqual(self.run_check()[0], 1)
        self.tearDown()
        self.setUp()
        put(os.path.join(self.m, "hmax_8/stats/realization_0001_timeseries.csv"), b"z")
        self.assertEqual(self.run_check()[0], 1)

    def test_status_keys_at_their_multi_pass_location(self):
        R = os.path.join(self.m, "realizations/r_0000")
        # tracker block / Np read at hmax_8/status.json
        put(os.path.join(R, "hmax_8/status.json"), dict(PASS_STATUS, Np=9999, hmax_div=8))
        rc, out = self.run_check()
        self.assertEqual(rc, 1)
        self.assertIn("'Np' differs", out)
        # a difference in the root status.json of a key owned by hmax_8 is NOT what is compared
        put(os.path.join(R, "hmax_8/status.json"), dict(PASS_STATUS, hmax_div=8))
        put(os.path.join(R, "status.json"), dict(Np=1, seed_backflow=5, hmax_div=[8, 16, 32], **POLLOCK))
        self.assertEqual(self.run_check()[0], 0)
        # pollock read at the root
        put(os.path.join(R, "status.json"), dict(Np=1, seed_backflow=5, hmax_div=[8, 16, 32], pollock={"seed_nonpos": 3}))
        rc, out = self.run_check()
        self.assertEqual(rc, 1)
        self.assertIn("'pollock' differs", out)

    def test_unmapped_single_key_and_missing_key(self):
        S = os.path.join(self.s, "realizations/r_0000/status.json")
        put(S, dict(PASS_STATUS, extra=1, **POLLOCK))
        rc, out = self.run_check()
        self.assertEqual(rc, 1)
        self.assertIn("no multi-pass mapping", out)
        put(S, dict(Np=10000, pollock=POLLOCK["pollock"]))
        rc, out = self.run_check()
        self.assertEqual(rc, 1)
        self.assertIn("absent from the single-pass", out)

    def test_realization_set_and_usage(self):
        shutil.rmtree(os.path.join(self.s, "realizations/r_0000"))
        self.assertEqual(self.run_check()[0], 1)
        self.assertEqual(B.main(["/nonexistent-a", "/nonexistent-b"]), 2)
        self.assertEqual(B.main([]), 2)


if __name__ == "__main__":
    unittest.main()
