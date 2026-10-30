import math
import unittest

import numpy as np

from mcstat import splitplot as sp


def make_data(coef, n_plus=14, n_minus=11, seed=0, field_sd=0.0, noise_sd=0.0, log_scale=False):
    """y = mu + a_G g + a_I i + a_D d + a_R r + a_GI g i + ... + seed (field) effect common to the 4 cells + noise.

    ``coef``: dict of coded-term coefficients (keys like 'G', 'I', 'GI'); the generator-level seeds differ (unpaired)."""
    rng = np.random.default_rng(seed)
    data = {}
    for g, n in ((1, n_plus), (-1, n_minus)):
        field = rng.normal(0.0, field_sd, size=n)
        data[g] = {}
        for (i, d, r) in sp.cells_of(g):
            lv = {"G": g, "I": i, "D": d, "R": r}
            eta = coef.get("mu", 0.0)
            for word, a in coef.items():
                if word != "mu":
                    eta = eta + a * np.prod([lv[f] for f in word])
            y = eta + field + rng.normal(0.0, noise_sd, size=n)
            data[g][(i, d, r)] = np.exp(y) if log_scale else y
    return data


class DesignAndAliases(unittest.TestCase):
    def test_fraction_and_defining_relation(self):
        cells = sp.design_cells()
        self.assertEqual(len(cells), 8)
        self.assertTrue(all(g * i * d * r == 1 for g, i, d, r in cells))
        self.assertEqual(len(set(cells)), 8)
        for g in (1, -1):
            self.assertEqual(len(sp.cells_of(g)), 4)
            self.assertTrue(all(i * d * r == g for i, d, r in sp.cells_of(g)))

    def test_alias_structure_resolution_iv(self):
        self.assertEqual({w: sp.aliases(w) for w in ("GI", "GD", "GR")}, {"GI": "DR", "GD": "IR", "GR": "ID"})
        self.assertEqual({w: sp.aliases(w) for w in ("G", "I", "D", "R")}, {"G": "IDR", "I": "GDR", "D": "GIR", "R": "GID"})
        self.assertEqual(sp.aliases("GIDR"), "1")
        self.assertEqual(sp.aliases(""), "GIDR")
        # resolution IV: no main effect is aliased with a main effect or a two-factor interaction
        for w in "GIDR":
            self.assertEqual(len(sp.aliases(w)), 3)

    def test_effect_contrasts_are_orthogonal_balanced_and_alias_consistent(self):
        tab = sp.effect_table()
        W = np.array([w for _, _, _, w in tab])
        np.testing.assert_allclose(W @ W.T, 8 * np.eye(7))  # the 7 estimable effects are orthogonal
        np.testing.assert_allclose(W.sum(axis=1), 0)
        cells = sp.design_cells()
        col = {c: k for k, c in enumerate(cells)}
        # the contrast of GI equals the contrast of its alias DR on the 8 cells of the fraction
        w_gi = dict((lbl, w) for lbl, _, _, w in tab)["GI"]
        w_dr = np.array([c[2] * c[3] for c in cells], dtype=float)
        np.testing.assert_allclose(w_gi, w_dr)
        self.assertEqual(len(col), 8)

    def test_design_rows(self):
        rows = sp.design([101, 102, 103], [201, 202])
        self.assertEqual(len(rows), 4 * 5)
        self.assertEqual(sum(1 for r in rows if r["seed"] == 102), 4)  # every seed runs the 4 cells of its generator
        self.assertTrue(all(r["g"] * r["i"] * r["d"] * r["r"] == 1 for r in rows))
        self.assertEqual({r["generator"] for r in rows if r["seed"] < 200}, {"sf18_periodic"})
        self.assertEqual({r["generator"] for r in rows if r["seed"] > 200}, {"par2_random_spectral"})


class Analysis(unittest.TestCase):
    COEF = {"mu": 2.0, "G": 0.40, "I": 0.15, "D": -0.05, "R": 0.0, "GI": 0.12, "GD": 0.0, "GR": -0.08}

    def test_noise_free_recovery_of_every_effect(self):
        out = sp.analyze(make_data(self.COEF), log_bootstrap=0)
        eff = out["effects"]
        # E = difference of mean cell values at the high and low level = 2 * coefficient; GI is aliased with DR, GD with IR, GR with ID
        for word, a in (("G", 0.40), ("I", 0.15), ("D", -0.05), ("R", 0.0), ("GI", 0.12), ("GD", 0.0), ("GR", -0.08)):
            self.assertAlmostEqual(eff[word]["estimate"], 2 * a, places=12, msg=word)
        self.assertEqual(eff["GI"]["alias"], "DR")
        self.assertEqual(out["aliases"]["G"], "IDR")
        self.assertEqual(eff["G"]["stratum"], "whole-plot")
        self.assertEqual(eff["I"]["stratum"], "sub-plot")
        self.assertEqual(out["n"], {1: 14, -1: 11})

    def test_with_field_and_cell_noise_subplot_effects_are_sharp_and_whole_plot_is_noisy(self):
        out = sp.analyze(make_data(self.COEF, field_sd=0.5, noise_sd=0.02, seed=5), log_bootstrap=0)
        eff = out["effects"]
        for word, a in (("I", 0.15), ("GI", 0.12), ("GR", -0.08)):
            self.assertLess(abs(eff[word]["estimate"] - 2 * a), 4 * eff[word]["se"] + 1e-12, msg=word)
            self.assertLess(eff[word]["p"], 1e-3, msg=word)  # true nonzero effects are detected within the sub-plot stratum
        for word in ("R", "GD"):
            self.assertGreater(eff[word]["p"], 1e-3, msg=word)
        # the field-to-field variance is in the whole-plot stratum only
        self.assertGreater(eff["G"]["se"], 10 * eff["I"]["se"])
        self.assertTrue(eff["G"]["lo"] < eff["G"]["estimate"] < eff["G"]["hi"])

    def test_simple_effects_are_one_sample_t_on_per_seed_contrasts(self):
        data = make_data(self.COEF, field_sd=0.4, noise_sd=0.05, seed=6)
        out = sp.analyze(data, log_bootstrap=0)
        se = out["simple"][(1, "I")]
        # per-seed contrast of I within generator +1 (cells with i = +1 minus i = -1)
        cells = sp.cells_of(1)
        Y = np.column_stack([data[1][c] for c in cells])
        x = np.array([c[0] for c in cells])
        c = Y[:, x > 0].mean(axis=1) - Y[:, x < 0].mean(axis=1)
        self.assertAlmostEqual(se.mean, float(c.mean()), places=13)
        self.assertAlmostEqual(se.t, float(c.mean() / (c.std(ddof=1) / math.sqrt(c.size))), places=11)
        self.assertEqual(se.dof, c.size - 1)
        # within generator +1 the simple I effect is 2 (a_I + a_GI) = 0.54, within -1 it is 2 (a_I - a_GI) = 0.06
        self.assertAlmostEqual(se.mean, 2 * (0.15 + 0.12), delta=4 * se.sd / math.sqrt(se.n))
        self.assertAlmostEqual(out["simple"][(-1, "I")].mean, 2 * (0.15 - 0.12), delta=4 * out["simple"][(-1, "I")].sd / math.sqrt(11))

    def test_log_scale_effects_and_bootstrap_ci(self):
        coef = {"mu": 0.5, "G": 0.20, "I": 0.10, "GI": 0.05}
        data = make_data(coef, field_sd=0.05, noise_sd=0.01, seed=8, log_scale=True)
        out = sp.analyze(data, log_bootstrap=400, seed=3)
        le = out["log_effects"]
        for word, a in (("G", 0.20), ("I", 0.10), ("GI", 0.05), ("D", 0.0)):
            tol = 0.08 if word == "G" else 0.03  # G carries the field-to-field (whole-plot) variance
            self.assertLess(abs(le[word]["estimate"] - 2 * a), tol, msg=word)
            self.assertLessEqual(le[word]["lo"], le[word]["estimate"] + 1e-12)
            self.assertGreaterEqual(le[word]["hi"], le[word]["estimate"] - 1e-12)
        self.assertAlmostEqual(le["I"]["ratio"], math.exp(le["I"]["estimate"]), places=14)
        self.assertEqual(le["GD"]["alias"], "IR")
        # reproducible
        again = sp.analyze(data, log_bootstrap=400, seed=3)["log_effects"]
        self.assertEqual(again["I"]["lo"], le["I"]["lo"])

    def test_validation(self):
        data = make_data(self.COEF)
        bad = {1: dict(data[1]), -1: dict(data[-1])}
        del bad[1][sp.cells_of(1)[0]]
        with self.assertRaises(ValueError):
            sp.analyze(bad, log_bootstrap=0)
        neg = make_data({"mu": -1.0})
        with self.assertRaises(ValueError):
            sp.analyze(neg, log_bootstrap=10)  # cell means must be positive on the log scale


if __name__ == "__main__":
    unittest.main()
