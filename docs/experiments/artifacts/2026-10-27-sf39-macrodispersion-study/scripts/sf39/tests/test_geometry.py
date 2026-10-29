"""C-GEOM tests: arm table, plane strings (sha256 of section 2), runner-parser mirror, Par2 pilot configs."""

import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

from sf39 import geometry as G

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[2]
ART = HERE.parents[3]
REPO = HERE.parents[7]

# preregistration section 1, `--ell` column (normative strings)
ELL = {
    "s1_A": "0.0390625", "s1_B": "0.0390625", "s1_n05": "0.078125", "s1_n10": "0.078125", "s1_n20": "0.078125",
    "s1_tw": "0.0390625",
    "s225_A": "0.041666666666666664", "s225_B": "0.041666666666666664", "s225_n08": "0.125", "s225_n16": "0.125",
    "s225_n32": "0.125", "s225_tw": "0.041666666666666664",
    "s4_A": "0.083333333333333329", "s4_B": "0.083333333333333329", "s4_n16": "0.25", "s4_n32": "0.25",
    "s4_n64": "0.25", "s4_tw": "0.083333333333333329", "s4_w24": "0.041666666666666664",
    "s4_w12": "0.083333333333333329", "s4_w12c": "0.083333333333333329",
    # s025 (addendum 2026-10-29, A1): the s1 geometry
    "s025_A": "0.0390625", "s025_B": "0.0390625", "s025_n05": "0.078125", "s025_n10": "0.078125", "s025_n20": "0.078125",
    "s025_tw": "0.0390625",
}
GRID = {
    "s1_A": "2048,256,256", "s1_B": "2048,256,256", "s1_n05": "1024,64,64", "s1_n10": "2048,128,128",
    "s1_n20": "4096,256,256", "s1_tw": "1024,128,128", "s225_A": "3328,384,384", "s225_B": "3328,384,384",
    "s225_n08": "1664,64,64", "s225_n16": "3328,128,128", "s225_n32": "6656,256,256", "s225_tw": "1664,192,192",
    "s4_A": "3328,384,384", "s4_B": "3328,384,384", "s4_n16": "1664,64,64", "s4_n32": "3328,128,128",
    "s4_n64": "6656,256,256", "s4_tw": "1664,192,192", "s4_w24": "1664,384,384", "s4_w12": "1664,192,192",
    "s4_w12c": "1664,192,192",
    "s025_A": "2048,256,256", "s025_B": "2048,256,256", "s025_n05": "1024,64,64", "s025_n10": "2048,128,128",
    "s025_n20": "4096,256,256", "s025_tw": "1024,128,128",
}


def opt(arm, name):
    f = G.flags(arm)
    return f[f.index(name) + 1]


class TestArmTable(unittest.TestCase):
    def test_table_values(self):
        for a, ell in ELL.items():
            self.assertEqual(opt(a, "--ell"), ell, a)
            self.assertEqual(opt(a, "--grid"), GRID[a], a)

    def test_amplitudes_and_lambda_over_h(self):
        for a in ELL:
            c = a.split("_")[0]
            self.assertEqual(opt(a, "--eps"), {"s025": "0.5", "s1": "1", "s225": "1.5", "s4": "2"}[c])
            self.assertEqual(opt(a, "--sigma2"), "1")
        lh = {"s1_A": 10, "s1_n05": 5, "s1_n10": 10, "s1_n20": 20, "s1_tw": 5, "s225_A": 16, "s225_n08": 8,
              "s225_n16": 16, "s225_n32": 32, "s225_tw": 8, "s4_A": 32, "s4_n16": 16, "s4_n32": 32, "s4_n64": 64,
              "s4_tw": 16, "s4_w24": 16, "s4_w12": 16, "s4_w12c": 16,
              "s025_A": 10, "s025_n05": 5, "s025_n10": 10, "s025_n20": 20, "s025_tw": 5}
        for a, v in lh.items():
            self.assertEqual(G.ARMS[a].lh, v, a)
            self.assertEqual(float(opt(a, "--ell")) * G.ARMS[a].grid[1], v, a)

    def test_hmax_and_window(self):
        for a in ELL:
            self.assertEqual(opt(a, "--hmax-div"), "8,16,32" if a.endswith("_A") else "8", a)
        self.assertEqual(opt("s4_w24", "--window"), "0.3,0.7,0.3,0.7")
        self.assertEqual(opt("s4_w12", "--window"), G.W12)
        self.assertEqual(opt("s4_w12c", "--window"), G.W12)
        self.assertEqual([float(x) for x in G.W12.split(",")], [0.36666666666666664, 0.6333333333333334] * 2)
        for a in ELL:
            if a not in ("s4_w24", "s4_w12", "s4_w12c"):
                self.assertEqual(opt(a, "--window"), "0.1,0.9,0.1,0.9")

    def test_common_flags_and_excluded(self):
        for a in G.ARMS:
            f = G.flags(a)
            for x in ("--out", "--seed0", "--nr"):
                self.assertNotIn(x, f, a)
            for x, v in (("--medium", "b"), ("--np-side", "100"), ("--times", "auto"), ("--tol", "1e-8"),
                         ("--twin", "on"), ("--pollock-bridge", "on")):
                self.assertEqual(opt(a, x), v)
            self.assertIn("--fused-splines", f)
            self.assertIn("--single-launch", f)
            for x in ("--x0", "--max-arclength", "--head-mapped-host", "--pcg-rtol"):
                self.assertNotIn(x, f)

    def test_planes_per_arm_and_check_arms(self):
        want = {"s1_A": "P1", "s1_B": "P1", "s1_tw": "P1", "s225_A": "P2", "s225_B": "P2", "s225_tw": "P2",
                "s4_A": "P3", "s4_B": "P3", "s4_tw": "P3", "s4_w12": "P3", "s4_w12c": "P3", "s1_n05": "P4", "s1_n10": "P4",
                "s1_n20": "P4", "s225_n08": "P5", "s225_n16": "P5", "s225_n32": "P5", "s4_n16": "P6",
                "s4_n32": "P6", "s4_n64": "P6", "s4_w24": "P7",
                "s025_A": "P1", "s025_B": "P1", "s025_tw": "P1", "s025_n05": "P4", "s025_n10": "P4", "s025_n20": "P4"}
        for a, p in want.items():
            pid, s = G.plane_list_of(a)
            self.assertEqual(pid, p)
            self.assertEqual(hashlib.sha256(s.encode()).hexdigest(), G.PLANE_SHA256[p], a)
            self.assertEqual(opt(a, "--planes"), s)
        self.assertEqual(len(G.SOLVE_CHECK_ARMS), 18)
        # the s025 arms are appended last: the order of every earlier check arm is unchanged; s025 adds chk_{n05,n10,n20,tw} (no _A / _B)
        self.assertEqual(G.SOLVE_CHECK_ARMS[-4:], ["chk_s025_n05", "chk_s025_n10", "chk_s025_n20", "chk_s025_tw"])
        self.assertEqual(G.SOLVE_CHECK_ARMS[:14], ["chk_s1_n05", "chk_s1_n10", "chk_s1_n20", "chk_s1_tw", "chk_s225_n08", "chk_s225_n16",
                                                    "chk_s225_n32", "chk_s225_tw", "chk_s4_n16", "chk_s4_n32", "chk_s4_n64", "chk_s4_tw",
                                                    "chk_s4_w24", "chk_s4_w12"])
        for a in ("chk_s025_A", "chk_s025_B"):
            self.assertNotIn(a, G.ARMS)
        self.assertEqual(G.plane_list_of("P1")[0], "P1")
        self.assertEqual(next(a for a in G.ARMS.values() if a.plist == "P1" and not a.id.startswith("chk_")).id, "s1_A")
        # the continuation arm is never a check arm (the checks part already ran)
        self.assertNotIn("chk_s4_w12c", G.ARMS)
        self.assertEqual(G.ARMS["s4_w12c"], G.ARMS["s4_w12"]._replace(id="s4_w12c"))
        self.assertEqual(G.ARMS["chk_s4_A"].hmax, "8,16,32")
        self.assertEqual(G.ARMS["chk_s4_A1"].hmax, "8")
        for a in G.SOLVE_CHECK_ARMS:
            self.assertEqual(G.ARMS[a].hmax, "8")
            self.assertEqual(opt(a, "--planes"), opt(a[4:], "--planes"))
        self.assertEqual(G.ARMS["chk_s4_A"]._replace(id="s4_A", hmax="8,16,32"), G.ARMS["s4_A"])


class TestPlanes(unittest.TestCase):
    def test_seven_sha256(self):
        self.assertEqual(G.check(), [])

    def test_structure(self):
        for a in ("s1_A", "s225_A", "s4_A", "s4_w24", "s1_n05"):
            nx, ny, _ = G.ARMS[a].grid
            lh = G.ARMS[a].lh
            ell, L1 = lh / ny, nx / ny
            x0 = 5.0 * ell
            v = G.plane_values(nx, ny, lh)
            self.assertEqual(v[-1], L1 - 5.0 * ell)  # Xb exactly, last
            Xa = x0 + (L1 - 10.0 * ell) / 2.0
            self.assertIn(Xa, v)
            self.assertTrue(all(b > a_ for a_, b in zip(v, v[1:])))
            self.assertTrue(v[0] > x0 and v[-1] < L1)
            self.assertIn(x0 + (L1 - 10.0 * ell) / 4.0, v)

    def test_merge_keeps_canonical(self):
        # s225: Xa and Xb coincide with lattice planes within rounding; the canonical doubles must survive
        nx, ny, lh = 3328, 384, 16
        ell, L1 = lh / ny, nx / ny
        v = G.plane_values(nx, ny, lh)
        self.assertEqual(v[-1], L1 - 5.0 * ell)
        self.assertIn(5.0 * ell + (L1 - 10.0 * ell) / 2.0, v)


class TestParserMirror(unittest.TestCase):
    def test_every_arm_accepted(self):
        for a in G.ARMS:
            self.assertEqual(G.validate(a), [], a)

    def _with(self, **kw):
        base = G.ARMS["s1_A"]
        G.ARMS["_bad"] = base._replace(id="_bad", **kw)
        try:
            return G.validate("_bad")
        finally:
            del G.ARMS["_bad"]

    def test_rejections(self):
        self.assertTrue(any("same" in x or "equal" in x for x in self._with(grid=(2048, 256, 128))))
        self.assertTrue(any("even" in x for x in self._with(grid=(2049, 256, 256))))
        self.assertTrue(any(">= 4" in x for x in self._with(grid=(2, 2, 2))))
        self.assertTrue(any("window" in x for x in self._with(window="0.9,0.1,0.1,0.9")))
        self.assertTrue(any("hmax" in x for x in self._with(hmax="8,8")))
        self.assertTrue(any("ell string" in x for x in self._with(ell_str="0.04")))

    def test_seed_on_face_detected(self):
        # ny = 100, window 0.1..0.9, M = 100: x = 0.1 + 0.008 (m + 1/2), 100 x = 10 + 0.8 (m + 1/2); m = 2 gives 12
        bad = self._with(grid=(1000, 100, 100), lh=10, ell_str="0.10000000000000001")
        self.assertTrue(any("RR12" in x for x in bad), bad)

    def test_planes_must_exceed_x0_and_stay_below_L1(self):
        # lh/ny such that L1 <= 10 ell: nx = 10 ny
        bad = self._with(grid=(2560, 256, 256), lh=256 + 0, ell_str=G.fmt17(256 / 256))
        self.assertTrue(bad)


class TestCli(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONPATH=str(SCRIPTS))
        return subprocess.run([sys.executable, "-m", "sf39.geometry", *args], capture_output=True, text=True, env=env)

    def test_flags_planes_check(self):
        r = self.run_cli("flags", "s4_A")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.strip().split(" "), G.flags("s4_A"))
        r = self.run_cli("planes", "P3")
        self.assertEqual(hashlib.sha256(r.stdout.rstrip("\n").encode()).hexdigest(), G.PLANE_SHA256["P3"])
        r = self.run_cli("planes", "s4_w24")
        self.assertEqual(hashlib.sha256(r.stdout.rstrip("\n").encode()).hexdigest(), G.PLANE_SHA256["P7"])
        self.assertEqual(self.run_cli("check").returncode, 0)
        self.assertEqual(self.run_cli("flags", "nope").returncode, 2)


class TestPar2Configs(unittest.TestCase):
    def test_s025_config_is_the_generator_output_for_the_s1_source(self):
        from sf39 import par2
        src = (REPO / "apps" / "config_pipeline_par2_sf38_s1.yaml").read_text(encoding="utf-8")
        new = (ART / "configs" / "par2_s025.yaml").read_text(encoding="utf-8")
        self.assertEqual(new, par2.make_pilot_config(src, "025"))
        self.assertEqual(par2.cfg_values(new)["sigma2"], 0.25)
        self.assertFalse((REPO / "apps" / "config_pipeline_par2_sf38_s025.yaml").exists())  # no apps/ file is added

    def test_only_declared_lines_differ(self):
        for c in ("1", "225", "4"):
            orig = (REPO / "apps" / ("config_pipeline_par2_sf38_s%s.yaml" % c)).read_text(encoding="utf-8").split("\n")
            new = (ART / "configs" / ("par2_s%s.yaml" % c)).read_text(encoding="utf-8").split("\n")
            self.assertEqual(len(orig), len(new))
            header_end = orig.index("")  # first blank line: end of the header comment block
            allowed = {
                "  seed: 7001": "  seed: 9001",
                "    NR: 10": "    NR: 20",
                "  output_dir: ./output/sf38_par2_s%s" % c: "  output_dir: raw/par2_s%s" % c,
            }
            seen = set()
            for i, (o, n) in enumerate(zip(orig, new)):
                if o == n:
                    continue
                if i < header_end:
                    self.assertTrue(o.startswith("#") and n.startswith("#"), (c, i))
                else:
                    self.assertEqual(allowed.get(o), n, (c, i, o, n))
                    seen.add(o)
            self.assertEqual(seen, set(allowed), c)
            # the changed values are the only ones
            self.assertIn("  seed: 123456789", new)  # transport.seed untouched
            self.assertTrue(any("preregistration.md" in x for x in new[:header_end]))


if __name__ == "__main__":
    unittest.main()
