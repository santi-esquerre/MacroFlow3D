import math
import unittest

import numpy as np

from sf43cov import mcpath  # noqa: F401
from mcstat import splitplot as sp

from sf43ana import g7

EFFECTS = {"G": 0.30, "I": -0.10, "D": 0.04, "R": 0.02, "GI": 0.06, "GD": -0.03, "GR": 0.05}  # log-scale half-effects


def log_mean(g, i, d, r):
    """log of the cell mean: mu + eG g + eI i + eD d + eR r + eGI g i + eGD g d + eGR g r (on the fraction g i d r = 1 the aliased
    interactions g i = d r etc. coincide, so eGI is also the effect of D x R)."""
    return (math.log(2.0) + EFFECTS["G"] * g + EFFECTS["I"] * i + EFFECTS["D"] * d + EFFECTS["R"] * r
            + EFFECTS["GI"] * g * i + EFFECTS["GD"] * g * d + EFFECTS["GR"] * g * r)


def factorial_data(n=12, seed0=100, field_sd=0.0):
    """Arms ``g=+1,i,d,r`` per cell of the fraction; the per-seed values are ``exp(log_mean) * f_s`` with a per-seed field factor
    ``f_s`` common to the cells of a seed (mean 1)."""
    rng = np.random.default_rng(5)
    arms, cells = {}, []
    for g, base in ((1, seed0), (-1, seed0 + 1000)):
        f = np.exp(field_sd * rng.standard_normal(n))
        f = f / f.mean()
        seeds = np.arange(base, base + n)
        for (i, d, r) in sp.cells_of(g):
            name = "g%+d_i%+d_d%+d_r%+d" % (g, i, d, r)
            arms[name] = {"seed": seeds, "cols": {"alpha_LP": math.exp(log_mean(g, i, d, r)) * f}}
            cells.append({"g": g, "i": i, "d": d, "r": r, "arm": name})
    return {"arms": arms}, cells


class Factorial(unittest.TestCase):
    def test_known_effects_and_aliases(self):
        data, cells = factorial_data(field_sd=0.2)
        prereg = {"g7": {"factorial": {"s1": cells}, "responses": ["alpha_LP"], "log_bootstrap": 200}}
        res = g7.run(prereg, data)["factorial"]["s1"]["alpha_LP"]
        le = res["log_effects"]
        # E = (1/4) sum_c w_c log ybar_c over the 8 cells: twice the coefficient of the +-1 coded term (the common field factor
        # is a constant per generator, equal to 1 after normalisation, and cancels in every contrast)
        for word, coef in EFFECTS.items():
            self.assertAlmostEqual(le[word]["estimate"], 2.0 * coef, places=10, msg=word)
            self.assertAlmostEqual(le[word]["ratio"], math.exp(2.0 * coef), places=10)
        self.assertEqual({w: le[w]["alias"] for w in EFFECTS}, {"G": "IDR", "I": "GDR", "D": "GIR", "R": "GID", "GI": "DR", "GD": "IR", "GR": "ID"})
        self.assertEqual(sp.aliases("GI"), "DR")
        # within-seed (sub-plot) contrasts exist for I, D, R per generator; the stratum labels are reported
        self.assertEqual(res["effects"]["G"]["stratum"], "whole-plot")
        self.assertEqual(res["effects"]["I"]["stratum"], "sub-plot")
        self.assertEqual(set(res["simple"]), {"g=%+d,%s" % (g, f) for g in (1, -1) for f in "IDR"})
        self.assertEqual(len(res["cells"]), 8)
        self.assertEqual(res["seeds"]["1"]["n_seeds"], 12)

    def test_noise_free_effects_on_the_linear_scale_and_welch_for_g(self):
        data, cells = factorial_data(field_sd=0.0)
        res = g7.run({"g7": {"factorial": {"s1": cells}, "responses": ["alpha_LP"], "log_bootstrap": 0}}, data)["factorial"]["s1"]["alpha_LP"]
        # no noise: sub-plot contrasts are constant per seed, so the one-sample t is degenerate (infinite) and the estimates exact
        ysum = {k: math.exp(log_mean(*k)) for k in sp.design_cells()}
        want_I = sum(c[1] * v for c, v in ysum.items()) / 4.0
        self.assertAlmostEqual(res["effects"]["I"]["estimate"], want_I, places=12)
        want_G = (sum(v for c, v in ysum.items() if c[0] == 1) - sum(v for c, v in ysum.items() if c[0] == -1)) / 4.0
        self.assertAlmostEqual(res["effects"]["G"]["estimate"], want_G, places=12)
        self.assertNotIn("log_effects", res)

    def test_incomplete_seeds_are_dropped_per_generator_and_a_bad_design_is_reported(self):
        data, cells = factorial_data(n=10, field_sd=0.1)
        data["arms"][cells[0]["arm"]]["cols"]["alpha_LP"][2] = np.nan
        res = g7.run({"g7": {"factorial": {"s1": cells}, "responses": ["alpha_LP"], "log_bootstrap": 0}}, data)["factorial"]["s1"]["alpha_LP"]
        self.assertEqual(res["seeds"][str(cells[0]["g"])]["n_seeds"], 9)
        bad = [dict(c) for c in cells]
        bad[0]["i"] = -bad[0]["i"]  # leaves the fraction g i d r = +1
        res = g7.run({"g7": {"factorial": {"s1": bad}, "responses": ["alpha_LP"]}}, data)["factorial"]["s1"]["alpha_LP"]
        self.assertIn("fraction", res["note"])


class Twins(unittest.TestCase):
    def test_three_pairwise_log_ratio_contrasts(self):
        rng = np.random.default_rng(1)
        n = 80
        base = 1.0 + 0.3 * rng.standard_normal(n)
        eps = lambda: 1 + 0.01 * rng.standard_normal(n)
        seeds = np.arange(n)
        arms = {"C": {"seed": seeds, "cols": {"alpha_LP": base * 1.00 * eps(), "alpha_TP": 0.02 * base * eps()}},
                "K": {"seed": seeds[::-1].copy(), "cols": {"alpha_LP": (base * 1.05 * eps())[::-1].copy(), "alpha_TP": (0.021 * base * eps())[::-1].copy()}},
                "P": {"seed": seeds, "cols": {"alpha_LP": base * 1.20 * eps(), "alpha_TP": 0.03 * base * eps()}}}
        res = g7.run({"g7": {"twins": {"s1": {"consumer": "C", "pollock": "K", "par2": "P"}}}}, {"arms": arms})["twins"]["s1"]
        lp = res["alpha_LP"]
        self.assertAlmostEqual(lp["consumer/pollock"]["ratio"], 1 / 1.05, delta=0.01)
        self.assertAlmostEqual(lp["pollock/par2"]["ratio"], 1.05 / 1.20, delta=0.01)
        self.assertAlmostEqual(lp["consumer/par2"]["ratio"], 1 / 1.20, delta=0.01)
        c = lp["consumer/par2"]
        self.assertAlmostEqual(c["log_ratio"], math.log(c["ratio"]), places=12)
        self.assertLess(c["ratio_lo"], c["ratio"])
        self.assertGreater(c["ratio_hi"], c["ratio"])
        # the log ratios add up exactly (ratio of means is multiplicative)
        self.assertAlmostEqual(lp["consumer/pollock"]["log_ratio"] + lp["pollock/par2"]["log_ratio"], c["log_ratio"], places=12)
        self.assertAlmostEqual(res["alpha_TP"]["consumer/par2"]["ratio"], 0.02 / 0.03, delta=0.02)
        self.assertEqual(lp["consumer/pollock"]["meaning"], "velocity representation")

    def test_missing_arm_is_noted_not_raised(self):
        seeds = np.arange(10)
        arms = {"C": {"seed": seeds, "cols": {"alpha_LP": np.ones(10)}}, "P": {"seed": seeds, "cols": {"alpha_LP": np.ones(10)}}}
        res = g7.run({"g7": {"twins": {"s1": {"consumer": "C", "pollock": "K", "par2": "P"}}, "responses": ["alpha_LP"]}}, {"arms": arms})
        r = res["twins"]["s1"]["alpha_LP"]
        self.assertIn("unavailable", r["consumer/pollock"]["note"])
        self.assertIn("unavailable", r["pollock/par2"]["note"])
        self.assertAlmostEqual(r["consumer/par2"]["ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
