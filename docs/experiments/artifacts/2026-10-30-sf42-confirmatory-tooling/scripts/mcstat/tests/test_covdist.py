import json
import os
import unittest

import numpy as np
from scipy import stats

from mcstat import covdist as cd
from mcstat.tests._paths import A


class MixtureFamily(unittest.TestCase):
    def test_closed_form_moments_match_the_targets(self):
        for skew, ek in [(1.94, 4.69), (1.55, 4.52), (0.5, 0.0), (1.0, -0.99), (2.0, 2.02), (1.0, 20.0), (3.0, 7.1)]:
            p, v = cd.mixture_params(skew, ek)
            self.assertTrue(0.0 < p < 0.5 and 0.0 <= v < 1.0)
            s, k = cd.mixture_moments(p, v)
            self.assertAlmostEqual(float(s), skew, delta=1e-9)
            self.assertAlmostEqual(float(k), ek, delta=1e-9)

    def test_sample_moments_of_the_stress_family(self):
        d = cd.get_dist("stress")
        x = d.draw(np.random.default_rng(5), (2_000_000,))
        self.assertAlmostEqual(float(x.mean()), 0.0, delta=0.005)
        self.assertAlmostEqual(float(x.var()), 1.0, delta=0.005)
        self.assertAlmostEqual(float(stats.skew(x)), d.skew, delta=0.03)  # sampling sd of the sample skewness about 0.006
        self.assertAlmostEqual(float(stats.kurtosis(x)), d.ekurt, delta=0.15)  # about 0.04 for this tail weight
        self.assertGreater(d.skew, 1.9)
        self.assertGreater(d.ekurt, 4.6)

    def test_reflection_has_the_opposite_skewness(self):
        d, m = cd.get_dist("stress"), cd.get_dist("stress-")
        self.assertEqual(m.name, "stress-")
        self.assertAlmostEqual(m.skew, -d.skew, places=14)
        self.assertAlmostEqual(m.ekurt, d.ekurt, places=14)
        x = m.draw(np.random.default_rng(2), (1_000_000,))
        self.assertAlmostEqual(float(stats.skew(x)), -d.skew, delta=0.03)
        self.assertAlmostEqual(float(x.mean()), 0.0, delta=0.01)

    def test_normal_and_unknown(self):
        x = cd.get_dist("normal").draw(np.random.default_rng(0), (1_000_000,))
        self.assertAlmostEqual(float(x.var()), 1.0, delta=0.005)
        with self.assertRaises(ValueError):
            cd.get_dist("laplace")

    def test_infeasible_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            cd.mixture_params(1.5, 0.1)  # below kurtosis = skew^2 - 2
        with self.assertRaises(ValueError):
            cd.mixture_params(-1.0, 3.0)
        with self.assertRaises(ValueError):
            cd.mixture_params(0.0, 3.0)

    def test_the_stress_pair_is_beyond_johnson_su(self):
        # the reason for the mixture family: the bound found lies below the lognormal line, where Johnson SU does not exist
        s = cd.load_stress()["envelope"]
        w = 1.2
        for _ in range(60):  # lognormal line: skew = (w + 2) sqrt(w - 1)
            w -= ((w + 2.0) * np.sqrt(w - 1.0) - s["skew"]) / (np.sqrt(w - 1.0) + (w + 2.0) / (2.0 * np.sqrt(w - 1.0)))
        self.assertLess(s["ekurt"], w ** 4 + 2 * w ** 3 + 3 * w ** 2 - 6)


class StressMoments(unittest.TestCase):
    def test_committed_json_is_reproduced_from_the_committed_sf39_tables(self):
        st = cd.load_stress()
        again = cd.stress_from_sf39(A, st["confidence"], st["B"], st["seed"])
        self.assertEqual(len(again["series"]), len(st["series"]))
        self.assertEqual(again["files_sha256"], st["files_sha256"])
        # same numpy: the seeded bootstrap reproduces the committed numbers; another numpy may change the bootstrap stream, whose
        # effect on a 95 % quantile of 10^4 resamples is about 0.02 (the tolerance still catches a wrong column or filter)
        same = st["computed_with"]["numpy"].split(".")[:2] == np.__version__.split(".")[:2]
        tol = 1e-12 if same else 0.1
        for a, d in st["amplitudes"].items():
            for k in ("skew", "ekurt"):
                self.assertAlmostEqual(again["amplitudes"][a][k], d[k], delta=tol)
        for k in ("skew", "ekurt"):
            self.assertAlmostEqual(again["envelope"][k], st["envelope"][k], delta=tol)

    def test_envelope_dominates_every_amplitude_and_every_series(self):
        st = cd.load_stress()
        self.assertEqual(st["confidence"], 0.95)
        for a, d in st["amplitudes"].items():
            self.assertLessEqual(d["skew"], st["envelope"]["skew"])
            self.assertLessEqual(d["ekurt"], st["envelope"]["ekurt"])
        self.assertEqual(set(st["amplitudes"]), set(cd.AMPLITUDES))
        for r in st["series"]:
            self.assertLessEqual(r["skew_ucb"], st["amplitudes"][r["amplitude"]]["skew"] + 1e-15)
            self.assertLessEqual(r["ekurt_ucb"], st["amplitudes"][r["amplitude"]]["ekurt"] + 1e-15)

    def test_all_pairs_are_realizable_and_named_members_exist(self):
        st = cd.load_stress()
        for a in cd.AMPLITUDES:
            d = cd.get_dist("stress_" + a)
            self.assertAlmostEqual(d.skew, st["amplitudes"][a]["skew"], places=12)
        with self.assertRaises(ValueError):
            cd.get_dist("stress_s9")

    def test_source_columns_are_the_sf39_analysis_columns(self):
        with open(os.path.join(A, "scripts", "sf39", "analyze.py")) as f:
            src = f.read()
        for col in cd.SF39_COLUMNS.values():
            self.assertIn('"%s"' % col, src)


if __name__ == "__main__":
    unittest.main()
