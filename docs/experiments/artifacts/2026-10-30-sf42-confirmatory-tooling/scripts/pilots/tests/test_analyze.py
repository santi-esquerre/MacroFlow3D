import json
import math
import os
import tempfile
import unittest

import numpy as np

from mcstat import costmodel as cm
from pilots import analyze as an
from pilots import cost


def cons_rows(vals, A=None, B=None, basis="pos", weights="flux"):
    rows = []
    for r, v in enumerate(vals):
        a = v if A is None else A[r]
        b = v if B is None else B[r]
        row = {"code": "consumer", "family": "plane", "weights": weights, "basis": basis, "set": "perplane", "r": str(r)}
        for e in an.EST:
            row[e], row[e + "_A"], row[e + "_B"] = str(v), str(a), str(b)
        rows.append(row)
    return rows


class TestParticleShare(unittest.TestCase):
    def test_component_matches_definition(self):
        v = [1.0, 2.0, 4.0, 3.0]
        A = [1.2, 1.9, 4.1, 3.3]
        B = [0.8, 2.1, 3.9, 2.7]
        c = an.component_rows(cons_rows(v, A, B), "alpha_TP", "pos")
        self.assertAlmostEqual(c["component"], np.mean((np.array(A) - np.array(B)) ** 2) / 4)
        self.assertAlmostEqual(c["variance"], np.var(v, ddof=1))
        self.assertAlmostEqual(c["fraction"], c["component"] / c["variance"])

    def test_exact_inverse_square_law(self):
        Ms = [64, 128, 192]
        comps = [4.0e-6 / m ** 2 for m in Ms]
        adm = list(range(64, 1025, 64))
        out, fit = an.project_M(Ms, comps, V=1.0e-9, adm=adm)
        self.assertAlmostEqual(fit["exponent"], -2.0, places=9)
        self.assertAlmostEqual(fit["k_fixed"], 4.0e-6, places=12)
        # share(M) = 4e-6 / M^2 / 1e-9 = 4000 / M^2 : <= 0.25 needs M >= 126.5 -> 128 ; <= 0.10 needs M >= 200 -> 256
        self.assertEqual(out[0.25]["M_fixed"], 128)
        self.assertEqual(out[0.10]["M_fixed"], 256)
        self.assertEqual(out[0.25]["M_fit"], 128)

    def test_unreachable_threshold_is_none(self):
        out, _ = an.project_M([64, 128], [1.0, 0.25], V=1e-9, adm=[64, 128])
        self.assertIsNone(out[0.25]["M_fixed"])


class TestPairing(unittest.TestCase):
    def rows(self, code, vals):
        return [{"code": code, "family": "plane", "weights": "unif", "basis": "pos", "set": "perplane", "r": str(r), "alpha_LP": str(v)}
                for r, v in enumerate(vals)]

    def test_correlated_pairs_and_plan(self):
        rng = np.random.default_rng(7)
        z = rng.standard_normal(40)
        L = 0.20 * (1 + 0.15 * z)
        P = 0.20 * (1 + 0.15 * (0.8 * z + 0.6 * rng.standard_normal(40)))
        out = an.pairing(self.rows("consumer", L), self.rows("par2", P))
        self.assertEqual(out["n_pairs"], 40)
        self.assertAlmostEqual(out["corr"], np.corrcoef(L, P)[0, 1], places=12)
        self.assertLess(out["corr_lower80"], out["corr"])
        self.assertTrue(out["plan_lower"]["found"] and out["plan_zero"]["found"])
        self.assertLess(out["plan_lower"]["n"], out["plan_zero"]["n"])  # pairing reduces the planning n
        self.assertEqual(out["plan_zero"]["corr_used"], 0.0)

    def test_unpaired_seeds_are_dropped(self):
        out = an.pairing(self.rows("consumer", [1.0, 1.1, 0.9, 1.05, 1.02]), self.rows("par2", [1.0, 1.1, 0.9, 1.05]))
        self.assertEqual((out["n_pairs"], out["n_consumer"], out["n_par2"]), (4, 5, 4))


class TestScaleAndCost(unittest.TestCase):
    def test_scale_n_floor_and_cap(self):
        self.assertEqual(an.scale_n(1100, 4.476e-3, 1.206e-4), 30)
        self.assertEqual(an.scale_n(110, 1.0, 1.0), 110)
        self.assertEqual(an.scale_n(1000, 1.0, 0.5), 250)
        self.assertEqual(an.scale_n(30, 1.0, 2.0), 30)  # never raised above the original count

    def test_points_and_plan(self):
        with tempfile.TemporaryDirectory() as td:
            for job, rid, code, nx, ny, amp, nr, wall, ex in (("J1", "c1", "consumer", 520, 120, "1", 1, 30.0, 0),
                                                              ("J1", "p1", "par2", 1040, 240, "1", 1, 50.0, 0),
                                                              ("J1", "bad", "consumer", 520, 120, "1", 1, 1.0, 1)):
                d = os.path.join(td, job, rid)
                os.makedirs(d)
                with open(os.path.join(d, "RUN.json"), "w") as f:
                    json.dump({"id": rid, "code": code, "nx": nx, "ny": ny, "nz": ny, "amp": amp, "nr": nr, "M": 96, "wall_s": wall, "exit": ex}, f)
            pts = cost.points_from_runs(td)
        self.assertEqual(sorted(p.arm.label for p in pts), ["J1/c1", "J1/p1"])
        c = [p for p in pts if p.arm.code == "lagrangian"][0]
        self.assertEqual(c.arm.particles, 96.0 ** 2)
        self.assertEqual([p.arm.code for p in pts if p.arm.family == "par2"], [cost.PAR2_TAG])
        rows = cost.plan({a: 100 for a in cost.AMPS}, {a: 10 for a in cost.AMPS})
        fam = {}
        for r in rows:
            fam[r["family"]] = fam.get(r["family"], 0) + 1
        self.assertEqual(fam, {"production": 4, "E_h": 8, "E_W": 10, "hmax": 4, "gate5_consumer": 4, "gate5_par2": 4, "g7_par2": 2, "distance": 2})
        # production counts of the planning table
        self.assertEqual([r["n"] for r in rows if r["family"] == "production"], [1100, 1650, 110, 30])
        # variant B: floor 30 on production and E_W only
        rb = cost.plan({a: 100 for a in cost.AMPS}, {a: 10 for a in cost.AMPS},
                       scale=lambda fam, amp, n: an.scale_n(n, 10.0, 1.0) if fam in ("production", "E_W") else n)
        self.assertEqual([r["n"] for r in rb if r["family"] == "production"], [30, 30, 30, 30])
        self.assertEqual([r["n"] for r in rb if r["family"] == "E_h"], [r["n"] for r in rows if r["family"] == "E_h"])

    def test_realization_seconds_basis(self):
        c = {"code": "consumer", "realizations": [{"seconds": {"total": 10.0}}, {"seconds": {"total": 12.5}}]}
        p = {"code": "par2", "par2_stage_times": [{"flow_stage_s": 3.0, "transport_s": 4.0}, {"flow_stage_s": 2.0, "transport_s": 5.0}]}
        self.assertEqual(cost.realization_seconds(c), 22.5)
        self.assertEqual(cost.realization_seconds(p), 14.0)

    def test_project_linear_in_n(self):
        pts = []
        for i, (nx, s2) in enumerate(((512, 0.25), (1024, 1.0), (2048, 2.25), (2048, 4.0), (1024, 4.0), (512, 1.0), (4096, 1.0))):
            arm = cm.lagrangian_arm(nx, 64, 64, s2, 1e4, (8,), 70, "a%d" % i)
            pts.append(cm.Point(arm, 10, 10 * (2.0 + 1e-6 * nx * 64 * 64 * (1 + 0.1 * s2)), "t"))
        m = cm.CostModel(pts)
        rows = [{"family": "f", "amp": "1", "n": 100, "arm": pts[0].arm}, {"family": "f", "amp": "1", "n": 200, "arm": pts[0].arm}]
        res = cost.project(m, rows, reserve=0.05)
        self.assertAlmostEqual(res["rows"][1]["gpu_h"], 2 * res["rows"][0]["gpu_h"])
        self.assertAlmostEqual(res["total_gpu_h"], 1.05 * res["subtotal_gpu_h"])
        self.assertAlmostEqual(res["rows"][0]["gpu_h_per_100"], res["rows"][0]["gpu_h"])


if __name__ == "__main__":
    unittest.main()
