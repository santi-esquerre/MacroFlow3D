"""Reproduction of deterministic SF-39 numbers from committed data (A/analysis/summary.json and A/analysis/reduced/*.csv).

Tolerances.  Quantile-free quantities (difference, standard error, degrees of freedom, integer-dof Student intervals) are
reproduced at 1e-9 relative or better (the integer-dof intervals at 1e-15).  The Welch bounds use a t quantile at NON-integer degrees of
freedom: the committed numbers were computed on the host with scipy 1.11.4, whose ``t.ppf`` is off by 1.4e-9 relative there (its
cdf at the returned quantile misses the level by 2.3e-10; scipy 1.18 returns 1.68546250814216 with a cdf residual of 0), so the bounds
are compared at ``RTOL_Q = 5e-9`` relative to the bound.  The refined quantile of ``mcstat`` (``_util.t_isf``) is accurate to ~1e-14
on every scipy, so on the host it differs from the committed record by that ~1.3e-9 (it passes through ``RTOL_Q``); the recorded
values are NOT reproduced exactly on any stack.

The ratio-consistent contrast ``L - 0.9 P`` has no full-precision committed reference (the record gives 4 decimals,
[-0.0466, -0.0014]): it is checked at 4 decimals against the record and at 1e-12 relative against a Welch interval computed inside the
test from the SF-39 means, standard deviations and sample sizes, independently of ``mcstat``.
"""
import csv
import json
import math
import os
import unittest

import numpy as np
from scipy import stats

from mcstat import intervals as iv
from mcstat.tests._paths import A

ANALYSIS = os.path.join(A, "analysis")
AMPS = ("s025", "s1", "s225", "s4")
EST = ("alpha_TAM", "alpha_TP", "alpha_LAM")
RTOL = 1e-9
RTOL_Q = 5e-9  # non-integer-dof t quantile, host scipy 1.11 against scipy >= 1.12 (see the module docstring)


def summary():
    with open(os.path.join(ANALYSIS, "summary.json")) as fh:
        return json.load(fh)


def close(test, got, want, rtol=RTOL, msg=None):
    test.assertLessEqual(abs(float(got) - float(want)), rtol * max(abs(float(want)), 1e-300), msg=msg or "%r vs %r" % (got, want))


class Gate5Welch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S = summary()

    def test_s025_primary_interval_published_values(self):
        C = self.S["amplitudes"]["s025"]["C"]
        L, P = C["lagrangian"]["alpha_LAM"], C["par2"]["alpha_x"]
        w = iv.welch_from_summary(L["mean"], L["sd"], L["n"], P["mean"], P["sd"], P["n"], 0.90)
        close(self, w.lo, -0.07546642238833463, RTOL_Q)
        close(self, w.hi, -0.02655991524453609, RTOL_Q)

    def test_every_amplitude_primary_interval_from_the_recorded_means_and_sds(self):
        for amp in AMPS:
            C = self.S["amplitudes"][amp]["C"]
            L, P = C["lagrangian"]["alpha_LAM"], C["par2"]["alpha_x"]
            w = iv.welch_from_summary(L["mean"], L["sd"], L["n"], P["mean"], P["sd"], P["n"], 0.90)
            g = C["gate5"]["primary"]
            close(self, w.diff, g["diff"], 1e-12, msg=amp)
            close(self, w.nu, g["nu"], 1e-12, msg=amp)
            close(self, w.lo, g["lo"], RTOL_Q, msg=amp)
            close(self, w.hi, g["hi"], RTOL_Q, msg=amp)

    def test_ratio_consistent_contrast_of_the_post_review_evaluation(self):
        C = self.S["amplitudes"]["s025"]["C"]
        L, P = C["lagrangian"]["alpha_LAM"], C["par2"]["alpha_x"]
        w = iv.welch_contrast(L["mean"], L["sd"], L["n"], P["mean"], P["sd"], P["n"], 1.0, -0.9, 0.90)
        # the record (note): [-0.0466, -0.0014] at 4 decimals, nu = 40.94; there is no full-precision committed reference
        self.assertEqual(round(float(w.lo), 4), -0.0466)
        self.assertEqual(round(float(w.hi), 4), -0.0014)
        self.assertAlmostEqual(float(w.nu), 40.94, delta=5.1e-3)
        self.assertLess(float(w.hi), 0.0)  # the contrast reads `disagree` (the whole interval below zero)
        # 1e-12 relative against a hand Welch (contrast L - 0.9 P, Welch-Satterthwaite dof, two-sided 90 %, Student quantile)
        v1, v2 = L["sd"] ** 2 / L["n"], 0.81 * P["sd"] ** 2 / P["n"]
        diff, se = L["mean"] - 0.9 * P["mean"], math.sqrt(v1 + v2)
        nu = (v1 + v2) ** 2 / (v1 ** 2 / (L["n"] - 1) + v2 ** 2 / (P["n"] - 1))
        q = float(stats.t.isf(0.05, nu))
        for _ in range(2):  # Newton on sf(q) - 0.05: scipy < 1.12 `t.isf` is only accurate to ~1e-9 at non-integer dof
            q += (float(stats.t.sf(q, nu)) - 0.05) / float(stats.t.pdf(q, nu))
        close(self, w.diff, diff, 1e-12)
        close(self, w.se, se, 1e-12)
        close(self, w.nu, nu, 1e-12)
        close(self, w.lo, diff - q * se, 1e-12)
        close(self, w.hi, diff + q * se, 1e-12)


class F4Ratios(unittest.TestCase):
    """max(|tw_ci.lo|, |tw_ci.hi|) / m of the twin-difference interval against the margin m (F4 production verdict).

    The substantive reproductions are ``tw_inside`` (the verdict) and ``tw_ci`` recomputed from the reduced per-realization tables;
    the headline multiples are the values quoted in the note."""

    @classmethod
    def setUpClass(cls):
        cls.S = summary()

    def test_tw_inside_verdict_of_all_amplitudes_and_estimators(self):
        n = 0
        for amp in AMPS:
            for est in EST:
                f4 = self.S["amplitudes"][amp]["A"][est]["F4"]
                lo, hi, m = f4["tw_ci"]["lo"], f4["tw_ci"]["hi"], f4["m"]
                # tw_inside is the statement ratio <= 2^1.5
                self.assertEqual(bool(iv.margin_multiple(lo, hi, m) <= 2.0 ** 1.5), bool(f4["tw_inside"]), msg="%s %s" % (amp, est))
                n += 1
        self.assertEqual(n, 12)

    def test_headline_values_of_the_note(self):
        a = self.S["amplitudes"]
        for amp, est, want in (("s4", "alpha_TAM", 2.52), ("s225", "alpha_TAM", 2.11), ("s4", "alpha_TP", 1.43)):
            f4 = a[amp]["A"][est]["F4"]
            got = iv.margin_multiple(f4["tw_ci"]["lo"], f4["tw_ci"]["hi"], f4["m"])
            self.assertEqual(float("%.3g" % got), want, msg="%s %s: %.6f" % (amp, est, got))
        # every other transverse cell is at most 1.43
        others = []
        for amp in AMPS:
            for est in ("alpha_TAM", "alpha_TP"):
                if (amp, est) in (("s4", "alpha_TAM"), ("s225", "alpha_TAM")):
                    continue
                f4 = a[amp]["A"][est]["F4"]
                others.append(iv.margin_multiple(f4["tw_ci"]["lo"], f4["tw_ci"]["hi"], f4["m"]))
        self.assertLessEqual(max(others), 1.43 + 5e-3)

    def test_twin_interval_from_the_reduced_per_realization_tables(self):
        """D_tw = production minus twin per seed (reduced/<amp>_prod_twinpairs.csv and <amp>_tw.csv, flux weights): the Student
        interval of ``mcstat`` equals the recorded ``F4.tw_ci`` (mean, lo, hi, n)."""
        key = {"alpha_TAM": "win.alpha_TAM", "alpha_TP": "tp.alpha_TP", "alpha_LAM": "win.alpha_e1"}

        def read(name, col):
            out = {}
            with open(os.path.join(ANALYSIS, "reduced", name + ".csv")) as fh:
                for row in csv.DictReader(fh):
                    if row["method"] == "flux":
                        out[int(row["seed"])] = float(row[col]) if row[col] != "" else float("nan")
            return out

        for amp in AMPS:
            for est in EST:
                a, b = read(amp + "_prod_twinpairs", key[est]), read(amp + "_tw", key[est])
                seeds = sorted(set(a) & set(b))
                d = np.array([a[s] - b[s] for s in seeds])
                r = iv.student_interval(d, 0.90)
                f = self.S["amplitudes"][amp]["A"][est]["F4"]["tw_ci"]
                self.assertEqual(len(seeds), f["n"], msg="%s %s" % (amp, est))
                scale = max(abs(f["lo"]), abs(f["hi"]))
                for got, want in ((r.mean, f["mean"]), (r.lo, f["lo"]), (r.hi, f["hi"])):
                    self.assertLessEqual(abs(float(got) - want), 1e-12 * scale, msg="%s %s" % (amp, est))


class Par2PerRealization(unittest.TestCase):
    """Par2 per-realization values are host-only data (the committed record has only means and standard deviations)."""

    @unittest.skipUnless(os.environ.get("MCSTAT_PAR2_ALPHAX_S025"),
                         "host-only data: set MCSTAT_PAR2_ALPHAX_S025 to a text file with the 25 per-realization Par2 alpha_x values of s025")
    def test_s025_gate5_from_per_realization_values(self):
        with open(os.environ["MCSTAT_PAR2_ALPHAX_S025"]) as fh:
            p = np.array([float(t) for t in fh.read().split()])
        S = summary()["amplitudes"]["s025"]["C"]
        rec = S["par2"]["alpha_x"]
        self.assertEqual(p.size, rec["n"])
        close(self, p.mean(), rec["mean"])
        close(self, p.std(ddof=1), rec["sd"])
        # Gate 5 from the recorded consumer summary and the per-realization Par2 array (the consumer's Gate 5 values use a common
        # frozen window and are not committed per realization either)
        L = S["lagrangian"]["alpha_LAM"]
        w = iv.welch_from_summary(L["mean"], L["sd"], L["n"], p.mean(), p.std(ddof=1), p.size, 0.90)
        g = S["gate5"]["primary"]
        close(self, w.lo, g["lo"], RTOL_Q)
        close(self, w.hi, g["hi"], RTOL_Q)


if __name__ == "__main__":
    unittest.main()
