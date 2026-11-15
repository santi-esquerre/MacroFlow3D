"""post hoc (2026-11-04) self-checks of the A1-POSTHOC scripts (no campaign data needed).

    cd <artifact>/scripts/posthoc && python3 -W error -m unittest test_posthoc
"""
import json
import os
import unittest

import numpy as np

import common
import n6_local_alphaT as n6
from n1_compare import compare


class Posthoc(unittest.TestCase):
    def test_posthoc_prereg_differs_only_in_g7_responses(self):
        with open(common.PREREG) as f:
            a = json.load(f)
        with open(os.path.join(common.HERE, "prereg_posthoc_g7.json")) as f:
            b = json.load(f)
        self.assertEqual([k for k in a if a[k] != b[k]], ["g7"])
        self.assertEqual([k for k in a["g7"] if a["g7"][k] != b["g7"][k]], ["responses"])
        self.assertEqual(b["g7"]["responses"], ["alpha_TP:disp", "alpha_TAP:disp", "alpha_TP", "alpha_LP"])
        self.assertEqual(set(a), set(b))

    def test_alpha_local_linear_curve(self):
        d = np.arange(25.0, 195.0)
        S = np.stack([0.002 * d + 3.0, 0.004 * d - 1.0])  # slope 0.002, 0.004 -> alpha_T = slope / 2
        for lo, hi in ((25, 94), (59, 76), (161, 178)):
            np.testing.assert_allclose(n6.alpha_local(d, S, lo, hi), [0.001, 0.002], rtol=1e-12)

    def test_fit_zeta_recovers_exponential(self):
        d = np.arange(25.0, 195.0)
        y = 0.5 + 7.0 * (1.0 - np.exp(-d / 150.0))
        z, s0, sp, ssr = n6.fit_zeta(d, y)
        self.assertAlmostEqual(z / 150.0, 1.0, delta=0.02)
        self.assertAlmostEqual(s0, 0.5, delta=0.05)
        self.assertAlmostEqual(sp, 7.0, delta=0.2)

    def test_compare_equal_and_different(self):
        a = {"x": [1.0, 2.0, "nan"], "s": "ok"}
        self.assertEqual(compare(a, json.loads(json.dumps(a)))[0], 0.0)
        w, _, _, mism, _ = compare({"x": 1.0, "s": "a"}, {"x": 1.0 + 1e-6, "s": "b"})
        self.assertAlmostEqual(w, 1e-6, delta=1e-9)
        self.assertEqual(mism, 1)


if __name__ == "__main__":
    unittest.main()
