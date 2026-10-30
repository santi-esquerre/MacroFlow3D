import csv
import json
import os
import tempfile
import unittest

import numpy as np

from pilots import report
from pilots.tests.test_analyze import cons_rows


def write_table(path, rows):
    cols = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


class TestReportJ2(unittest.TestCase):
    def test_j2_recovers_inverse_square_and_conclusion(self):
        rng = np.random.default_rng(3)
        n = 6
        base = 0.02 + 0.01 * rng.standard_normal(n)
        Ms = (64, 128, 192)
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "J2", "reduced"))
            for M in Ms:
                # half-lattice estimates differ by +-d/M (so (aA-aB)^2/4 = d^2/M^2... exactly an inverse-square law)
                d = 1.9e-2 * 64.0 / M
                A, B = base + d, base - d
                rid = "c_s1_M%d" % M
                os.makedirs(os.path.join(td, "J2", rid))
                with open(os.path.join(td, "J2", rid, "RUN.json"), "w") as f:
                    json.dump({"id": rid, "amp": "1", "exit": 0, "M": M, "nx": 1040, "ny": 240, "nz": 240, "lh": 10}, f)
                rows = cons_rows(list(base), list(A), list(B)) + cons_rows(list(base), list(A), list(B), basis="disp")
                write_table(os.path.join(td, "J2", "reduced", rid + ".csv"), rows)
            sd = {k: {e + "_" + b: {"sd": 0.01, "mean": 0.02, "n": 50} for e in ("alpha_TP", "alpha_TAP", "alpha_LP") for b in ("pos", "disp")}
                  for k in ("s025", "s1", "s225", "s4")}
            with open(os.path.join(td, "hostref.json"), "w") as f:
                json.dump({"pcg": {}, "planning_sd": sd}, f)
            out = os.path.join(td, "out")
            os.makedirs(out)
            res = report.j2(td, out)
            self.assertTrue(os.path.exists(os.path.join(out, "j2_components.csv")))
        r = res["1"]["estimators"]["alpha_TP/pos"]
        self.assertAlmostEqual(r["exponent"], -2.0, places=9)
        comp64 = (1.9e-2) ** 2  # mean (A-B)^2/4 = d^2
        self.assertAlmostEqual(r["by_M"][64]["component"], comp64, places=15)
        # share(M) = 3.61 (64/M)^2: <= 0.25 needs M >= 243.2 -> 256 (admissible multiples of 64); <= 0.10 needs M >= 384.5 -> 448
        self.assertEqual(r["M_0.25_fixed"], 256)
        self.assertEqual(res["1"]["M_0.25_pos"], 256)
        self.assertEqual(r["M_0.10_fixed"], 448)


class TestReportJ1Rule(unittest.TestCase):
    def test_three_times_rule(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "J1", "c1"))
            os.makedirs(os.path.join(td, "J1", "c2"))
            for rid, it in (("c1", 90), ("c2", 100)):
                rec = {"id": rid, "code": "consumer", "nx": 520, "ny": 120, "nz": 120, "lh": 5, "amp": "1", "M": 96, "seed0": 1, "exit": 0,
                       "wall_s": 10.0, "peak_gpu_mib_own_pid": 5000,
                       "realizations": [{"iterations": it, "verdict": "converged", "seconds": {"stage": 1.0},
                                         "mem_lines_stage": ["MEM_PHASE phase=S2 exact_bytes=1048576 x"], "mem_line_bridge": ""}]}
                with open(os.path.join(td, "J1", rid, "RUN.json"), "w") as f:
                    json.dump(rec, f)
            ref = {k: {"production": {"median": 30}} for k in ("s025", "s1", "s225", "s4")}
            with open(os.path.join(td, "hostref.json"), "w") as f:
                json.dump({"pcg": ref, "planning_sd": {}}, f)
            out = os.path.join(td, "out")
            os.makedirs(out)
            rows = {r["id"]: r for r in report.j1(td, out)}
        self.assertFalse(rows["c1"]["different_grid_required"])      # 90 = 3.0 x 30: not above 3x
        self.assertTrue(rows["c2"]["different_grid_required"])       # 100 > 90
        self.assertAlmostEqual(rows["c1"]["runner_exact_bytes_max_mib"], 1.0)


if __name__ == "__main__":
    unittest.main()
