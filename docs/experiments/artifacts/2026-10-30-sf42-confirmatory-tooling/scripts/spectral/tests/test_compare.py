"""SF-42 N3: the comparison pipeline, the acceptance-E verdict and the summary (no GPU: the FV dumps are fabricated)."""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from spectral import compare, darcy, sf18


class Verdict(unittest.TestCase):
    ns = [16, 32, 40, 64, 128, 256]

    def ladder(self, p=2.0, c=1.0):
        return [c * n ** -p for n in self.ns]

    def test_second_order_ladder_passes(self):
        err = self.ladder()
        v = compare.verdict_E(self.ns, err, [e * 1e-3 for e in err])
        self.assertTrue(v["pass"])
        self.assertAlmostEqual(v["order_finest_pair"], 2.0, places=12)

    def test_non_monotone_fails(self):
        err = self.ladder()
        err[3] = err[2] * 1.5
        v = compare.verdict_E(self.ns, err, [e * 1e-3 for e in err])
        self.assertFalse(v["monotone"])
        self.assertFalse(v["pass"])

    def test_low_order_over_the_finest_pair_fails(self):
        err = self.ladder()
        err[-1] = err[-2] * 0.5 ** 1.2  # order 1.2 over the finest pair
        v = compare.verdict_E(self.ns, err, [e * 1e-3 for e in err])
        self.assertFalse(v["order_ok"])
        self.assertFalse(v["pass"])

    def test_unconverged_reference_fails(self):
        err = self.ladder()
        sc = [e * 1e-3 for e in err]
        sc[-1] = err[-1] / 5.0  # only 5 times below the finest FV error
        v = compare.verdict_E(self.ns, err, sc)
        self.assertFalse(v["reference_ok"])
        self.assertAlmostEqual(v["min_ratio_fv_over_selfconv"], 5.0)
        self.assertFalse(v["pass"])

    def test_single_level_rejected(self):
        with self.assertRaises(ValueError):
            compare.verdict_E([16], [1.0], [1e-9])


class Pipeline(unittest.TestCase):
    """Fabricated FV dumps = (1 + 0.5 / N^2) x the reference faces: the measured error must be 0.5 / N^2."""

    seed = 7
    sigma2 = 0.25
    ns = (16, 24)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def fabricate(self):
        fv_root = os.path.join(self.tmp, "fv")
        faces, summ = compare.solve_reference(self.sigma2, self.seed, 32, self.ns, os.path.join(self.tmp, "scratch32"), 32, 1e-13, 500, 2)
        for N in self.ns:
            d = os.path.join(fv_root, "%s_seed%d" % (compare.tag(self.sigma2), self.seed), "N%d" % N)
            os.makedirs(d)
            f = sf18.generate(N, seed=self.seed)
            np.save(os.path.join(d, "Y.npy"), f.y)
            for c, a in zip("UVW", faces[N]):
                np.save(os.path.join(d, "%s.npy" % c), a * (1.0 + 0.5 / N ** 2))
            rep = {"n": N, "pcg_rtol": 1e-12, "all_correctors_converged": True,
                   "correctors": [{"iterations": 7, "converged": True}] * 3, "G": summ["G"], "K_eff": summ["K_eff"],
                   "sf18": {"applied_scale": f.applied_scale, "raw_variance": f.raw_variance},
                   "seconds": {"flow": 0.1}}
            with open(os.path.join(d, "report.json"), "w") as fh:
                json.dump(rep, fh)
        return fv_root

    def test_end_to_end_recovers_the_fabricated_error(self):
        fv_root = self.fabricate()
        out = os.path.join(self.tmp, "out")
        compare.run_case(self.sigma2, self.seed, self.ns, 16, fv_root, out, 32, 1e-13, 500, 2, keep_ref_faces=False)
        with open(os.path.join(out, "cases", "s0p25_seed7.json")) as fh:
            case = json.load(fh)
        self.assertFalse(os.path.exists(os.path.join(out, "ref", "s0p25_seed7")))  # the reference cache is dropped
        for lev in case["levels"]:
            N = lev["N"]
            self.assertAlmostEqual(lev["fv_error"]["rel_l2_all"], 0.5 / N ** 2, delta=1e-4 / N ** 2)
            self.assertAlmostEqual(lev["fv_error"]["rel_l2_x"], 0.5 / N ** 2, delta=1e-4 / N ** 2)
            self.assertLess(lev["ref_selfconv"]["rel_l2_all"], 1e-2)  # plumbing only: M = 16 (band 1/2: |m| <= 4) against 32
            self.assertLess(lev["generator_match"]["max_abs_dY_over_std"], 1e-13)
            self.assertLess(lev["field_variance_outside_fv_modes"], 1e-15)  # modes the FV grid does not carry
        self.assertAlmostEqual(case["orders_all"][0], 2.0, delta=1e-3)
        self.assertLess(case["applied_scale_spread_rel"], 1e-10)
        s = compare.summarize(out)
        self.assertTrue(s["sigma2"]["0.25"]["seeds"]["7"]["verdict"]["monotone"])
        self.assertTrue(s["sigma2"]["0.25"]["seeds"]["7"]["verdict"]["order_ok"])
        self.assertTrue(os.path.exists(os.path.join(out, "summary.md")))
        # restart safety: a second call is a no-op on a finished case
        mtime = os.path.getmtime(os.path.join(out, "cases", "s0p25_seed7.json"))
        compare.run_case(self.sigma2, self.seed, self.ns, 16, fv_root, out, 32, 1e-13, 500, 2, keep_ref_faces=False)
        self.assertEqual(mtime, os.path.getmtime(os.path.join(out, "cases", "s0p25_seed7.json")))

    def test_wrong_shapes_are_rejected(self):
        fv_root = self.fabricate()
        d = os.path.join(fv_root, "s0p25_seed7", "N16")
        np.save(os.path.join(d, "U.npy"), np.zeros((8, 8, 8)))
        with self.assertRaises(ValueError):
            compare.load_fv(d)

    def test_scan_reports_the_reference_self_convergence(self):
        out = os.path.join(self.tmp, "scan")
        res = compare.scan(self.sigma2, self.seed, [16, 24, 32], [8, 16], out, 16, 1e-12, 500, 2, 0.5)
        self.assertEqual(len(res["rows"]), 4)
        e = {(r["M_from"], r["N"]): r["rel_l2_all"] for r in res["rows"]}
        self.assertLess(e[(24, 16)], e[(16, 16)])  # spectral convergence: the 24 -> 32 step is smaller than 16 -> 24
        self.assertTrue(os.path.exists(os.path.join(out, "scan_s0p25_seed7.json")))

    def test_ref_m_parser(self):
        self.assertEqual(compare.parse_ref_m("0.25:64,4:128"), {0.25: 64, 4.0: 128})


if __name__ == "__main__":
    unittest.main()
