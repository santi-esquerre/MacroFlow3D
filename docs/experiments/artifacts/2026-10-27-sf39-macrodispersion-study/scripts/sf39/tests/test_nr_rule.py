"""Tests of sf39.power and nr_rule.py (C-NR): synthetic known answers and Monte Carlo validation of the TOST power."""
import contextlib
import io
import json
import math
import os
import re
import tempfile
import unittest

import numpy as np
from scipy import stats

import nr_rule
from sf39 import power as P


def mc_tost_power(N, margin, sigma, n=200000, seed=7):
    rng = np.random.default_rng(seed)
    dbar = rng.normal(0.0, sigma / math.sqrt(N), n)
    s = sigma * np.sqrt(rng.chisquare(N - 1, n) / (N - 1))
    t = stats.t.ppf(0.95, N - 1)
    return float(np.mean(np.abs(dbar) + t * s / math.sqrt(N) <= margin))


def arr(sd, n=20, mean=0.0):
    """n values with exact sample mean `mean` and exact ddof=1 sd `sd` (alternating signs)."""
    x = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    x = x / np.std(x, ddof=1)
    return (mean + sd * x).tolist()


def est(sd_f=0.01, sd_tw=0.01, sd_h=0.01, sd_prod=0.01, mean_prod=0.1, mean_2x=0.1, mean_h16=0.1):
    return {"D_f": arr(sd_f), "D_tw": arr(sd_tw), "d16_8": arr(sd_h), "prod": arr(sd_prod, mean=mean_prod),
            "mean_2x": mean_2x, "mean_h16": mean_h16}


class SUpper(unittest.TestCase):
    def test_factor_n20(self):
        self.assertAlmostEqual(P.s_upper(1.0, 20), 1.17697, places=5)
        self.assertAlmostEqual(P.s_upper(2.5, 20), 2.5 * 1.17697, places=4)


class TostPower(unittest.TestCase):
    def test_against_monte_carlo(self):
        # (N, margin/sigma): from low to high power, N = 20 .. 221, plus a large ratio (unclipped quad returned 0 there)
        for N, r in [(20, 0.45), (20, 0.6), (50, 0.3), (100, 0.22), (221, 0.14), (221, 0.15), (20, 5.0)]:
            with self.subTest(N=N, r=r):
                self.assertAlmostEqual(P.tost_power(N, r, 1.0), mc_tost_power(N, r, 1.0), delta=4e-3)

    def test_scale_invariance_and_degenerate(self):
        self.assertAlmostEqual(P.tost_power(40, 0.3, 1.0), P.tost_power(40, 0.003, 0.01), places=9)
        self.assertGreater(P.tost_power(20, 100.0, 1.0), 0.999999)
        self.assertEqual(P.tost_power(20, 0.0, 1.0), 0.0)
        self.assertEqual(P.tost_power(20, 1.0, 0.0), 1.0)

    def test_monotone_in_n(self):
        # for power below ~1e-9 (N = 20, margin/sigma 0.1) the function dips with N (lower tail of S at small d.o.f.);
        # monotone from N = 60 on, which covers the 0.80 crossing that the bisection of ladder_n relies on
        for r in (0.1, 0.25, 0.5):
            p = [P.tost_power(N, r, 1.0) for N in range(60, 501, 20)]
            self.assertTrue(all(b >= a - 1e-12 for a, b in zip(p, p[1:])), p)


class Ladder(unittest.TestCase):
    def test_floor_and_cap(self):
        self.assertEqual(P.ladder_n(1e-6, 20, 0.1), (20, False))
        self.assertEqual(P.ladder_n(10.0, 20, 0.1), (500, True))

    def test_smallest_n_with_power(self):
        for sd, m in [(1.0, 0.5), (1.0, 0.25), (0.3, 0.05)]:
            n, capped = P.ladder_n(sd, 20, m)
            su = P.s_upper(sd, 20)
            self.assertFalse(capped)
            self.assertGreaterEqual(P.tost_power(n, m, su), 0.80)
            if n > 20:
                self.assertLess(P.tost_power(n - 1, m, su), 0.80)
            # linear scan agrees with the bisection
            lin = next(N for N in range(20, 501) if P.tost_power(N, m, su) >= 0.80)
            self.assertEqual(n, lin)

    def test_margin_scaling_shrinks_n(self):
        # twin margin 2^1.5 m is looser than m, so N is not larger
        self.assertLessEqual(P.ladder_n(1.0, 20, 2 ** 1.5 * 0.2)[0], P.ladder_n(1.0, 20, 0.2)[0])


class NrB(unittest.TestCase):
    def test_hand_checked(self):
        # s=1, mean 5 -> target 0.5; s_U = 1.17697; t_{.975,23} sU/sqrt(24) = 0.49699 <= 0.5 < t_{.975,22} sU/sqrt(23) = 0.50898
        self.assertEqual(P.nr_b(1.0, 20, 5.0), (24, False))
        self.assertEqual(P.nr_b(1.0, 20, -5.0), (24, False))  # |mean|

    def test_floor_absolute_target_and_cap(self):
        self.assertEqual(P.nr_b(1e-6, 20, 1.0), (20, False))
        # mean 0 -> target 2.5e-3: s = 0.1 needs N ~ 8e3 > 500
        self.assertEqual(P.nr_b(0.1, 20, 0.0), (500, True))
        n, c = P.nr_b(0.002, 20, 0.0)  # absolute floor active (0.1*0 < 2.5e-3)
        self.assertFalse(c)
        self.assertLessEqual(stats.t.ppf(0.975, n - 1) * P.s_upper(0.002, 20) / math.sqrt(n), 2.5e-3)


class Rules(unittest.TestCase):
    def test_margin(self):
        self.assertEqual(P.margin_t(0.0), 1.25e-3)
        self.assertAlmostEqual(P.margin_t(-1.0), 0.05)
        self.assertEqual(P.margin_t(0.01), 1.25e-3)

    def test_derive(self):
        d = P.derive(dict(N_narrow=30, N_twin=25, N_hmax=20, N_R=40))
        self.assertEqual((d["N_A"], d["N_run"], d["par2_NR"]), (20, 40, 40))
        d = P.derive(dict(N_narrow=30, N_twin=90, N_hmax=60, N_R=40))
        self.assertEqual((d["N_A"], d["N_run"], d["par2_NR"]), (60, 90, 40))
        d = P.derive(dict(N_narrow=30, N_twin=20, N_hmax=20, N_R=20))
        self.assertEqual((d["N_A"], d["N_run"]), (20, 20))

    def test_amplitude_ns_takes_max_over_estimators(self):
        a = est(sd_f=1e-4, sd_tw=1e-4, sd_h=1e-4, sd_prod=1e-4)  # everything at the floor
        b = est(sd_f=0.01, sd_tw=0.02, sd_h=0.015, sd_prod=0.02, mean_prod=0.2, mean_2x=0.2, mean_h16=0.2)
        n, cap = P.amplitude_ns({"alpha_TAM": a, "alpha_TP": b})
        self.assertEqual(cap, [])
        m = P.margin_t(0.2)
        self.assertEqual(n["N_narrow"], P.ladder_n(0.01, 20, m)[0])
        self.assertEqual(n["N_twin"], P.ladder_n(0.02, 20, 2 ** 1.5 * m)[0])
        self.assertEqual(n["N_hmax"], P.ladder_n(0.015, 20, m)[0])
        self.assertEqual(n["N_R"], P.nr_b(0.02, 20, 0.2)[0])
        self.assertEqual(n["N_A"], max(20, n["N_hmax"]))
        self.assertEqual(n["N_run"], max(n["N_R"], n["N_A"], n["N_twin"]))
        self.assertEqual(n["par2_NR"], n["N_R"])
        swapped, _ = P.amplitude_ns({"alpha_TAM": b, "alpha_TP": a})
        self.assertEqual(n, swapped)

    def test_margin_uses_min_of_2x_and_production_means(self):
        a = est(sd_f=0.02, mean_prod=0.1, mean_2x=0.4)  # margins 5e-3 (prod) and 0.02 (2x): min = prod
        n1, _ = P.amplitude_ns({"alpha_TAM": a, "alpha_TP": a})
        self.assertEqual(n1["N_narrow"], P.ladder_n(0.02, 20, P.margin_t(0.1))[0])
        b = est(sd_f=0.02, mean_prod=0.4, mean_2x=0.1)
        n2, _ = P.amplitude_ns({"alpha_TAM": b, "alpha_TP": b})
        self.assertEqual(n2["N_narrow"], n1["N_narrow"])

    def test_cap_flags_propagate(self):
        big = est(sd_f=10.0, sd_tw=10.0, sd_h=10.0, sd_prod=10.0)
        n, cap = P.amplitude_ns({"alpha_TAM": big, "alpha_TP": big})
        self.assertEqual({k: n[k] for k in ("N_narrow", "N_twin", "N_hmax", "N_R", "N_A", "N_run", "par2_NR")},
                         dict.fromkeys(("N_narrow", "N_twin", "N_hmax", "N_R", "N_A", "N_run", "par2_NR"), 500))
        self.assertEqual(set(cap), {"N_narrow", "N_twin", "N_hmax", "N_R", "N_A", "N_run", "par2_NR"})


def flat_costs(a=2.0, b=1.0, other=1.0):
    c = {}
    for amp in P.AMPLITUDES:
        c[f"{amp}_A"], c[f"{amp}_B"], c[f"{amp}_tw"], c[f"{amp}_par2"] = a, b, other, other
        for arm in P.NARROW_ARMS[amp]:
            c[arm] = other
    c.update({"s4_w24": other, "s4_w12": other})
    return c


def big_n():
    return {amp: dict(N_narrow=200, N_twin=150, N_hmax=100, N_R=400) for amp in P.AMPLITUDES}


class Allocate(unittest.TestCase):
    def test_a_and_b_costs_are_distinct(self):
        n = {"s1": P.derive(dict(N_narrow=20, N_twin=20, N_hmax=20, N_R=100))}
        done = {}
        # s1: A 20 realizations, B 80, twin 20, par2 100, three narrow rungs 20 each
        self.assertAlmostEqual(P.projected_gpu_h(n, flat_costs(a=3600.0, b=1.0), done), 20.0 + 260 / 3600.0)
        self.assertAlmostEqual(P.projected_gpu_h(n, flat_costs(a=1.0, b=3600.0), done), 80.0 + 200 / 3600.0)

    def test_under_budget_unchanged(self):
        r = P.allocate(big_n(), flat_costs(), {}, 1e9)
        self.assertEqual((r["f"], r["s4_ladders_to_20"], r["owner"]), (1.0, False, False))
        self.assertEqual(r["N"]["s4"]["N_narrow"], 200)

    def test_s4_ladders_first_then_common_factor(self):
        n, costs, done = big_n(), flat_costs(), {}
        full = P.allocate(n, costs, done, 1e9)["projected_gpu_h"]
        # budget that only the s4-ladder reduction needs
        s4n = {**n, "s4": {**n["s4"], "N_narrow": 20, "N_twin": 20, "N_hmax": 20}}
        p_s4 = P.projected_gpu_h({a: P.derive(s4n[a]) for a in s4n}, costs, done)
        self.assertLess(p_s4, full)
        r = P.allocate(n, costs, done, p_s4 + 1e-9)
        self.assertTrue(r["s4_ladders_to_20"])
        self.assertEqual(r["f"], 1.0)
        self.assertEqual([r["N"]["s4"][k] for k in ("N_narrow", "N_twin", "N_hmax")], [20, 20, 20])
        self.assertEqual(r["N"]["s1"]["N_narrow"], 200)
        self.assertEqual(r["N"]["s4"]["N_R"], 400)
        # smaller budget: common factor on everything remaining, floors of 20, s4 ladders stay at 20
        r = P.allocate(n, costs, done, p_s4 * 0.5)
        self.assertTrue(r["s4_ladders_to_20"])
        self.assertLess(r["f"], 1.0)
        self.assertLessEqual(r["projected_gpu_h"], p_s4 * 0.5)
        k = round(r["f"] * 100)
        self.assertEqual(r["N"]["s1"]["N_narrow"], max(20, -(-k * 200 // 100)))
        self.assertEqual(r["N"]["s1"]["N_R"], max(20, -(-k * 400 // 100)))
        self.assertEqual(r["N"]["s1"]["par2_NR"], r["N"]["s1"]["N_R"])
        self.assertEqual(r["N"]["s4"]["N_twin"], 20)
        # f is the largest multiple of 0.01 that fits
        n_up = {a: P._scale(P.derive(r0), k + 1) for a, r0 in
                {a: {kk: (20 if a == "s4" and kk in ("N_narrow", "N_twin", "N_hmax") else n[a][kk])
                     for kk in P.LADDER_KEYS} for a in n}.items()}
        self.assertGreater(P.projected_gpu_h(n_up, costs, done), p_s4 * 0.5)

    def test_width_arms_and_reference_floor_untouched(self):
        n = big_n()
        r = P.allocate(n, flat_costs(), {}, 1.0)
        counts = P.arm_counts(r["N"]["s4"], "s4")
        self.assertEqual(counts["s4_w24"], 30)
        self.assertEqual(counts["s4_w12"], 30)
        self.assertEqual(counts["s4_tw"], 30)  # max(N_twin = 20, 30)
        self.assertEqual(counts["s4_n16"], 30)
        self.assertEqual(counts["s4_n32"], 20)
        self.assertNotIn("s1_w24", P.arm_counts(r["N"]["s1"], "s1"))

    def test_owner_when_nothing_fits(self):
        r = P.allocate(big_n(), flat_costs(), {}, 0.0)
        self.assertTrue(r["owner"])
        self.assertEqual(r["f"], 0.01)
        self.assertEqual(r["N"]["s1"]["N_narrow"], 20)  # floor of 20
        self.assertEqual(r["N"]["s1"]["N_run"], 20)

    def test_done_realizations_are_free(self):
        costs = flat_costs()
        n = {a: P.derive(dict(N_narrow=20, N_twin=20, N_hmax=20, N_R=20)) for a in ("s1",)}
        done = dict.fromkeys(P.arm_counts(n["s1"], "s1"), 1000)
        self.assertEqual(P.projected_gpu_h(n, costs, done), 0.0)

    def test_integer_ceil(self):
        # float 0.07*100 = 7.000000000000001 would round up to 8
        self.assertEqual(P._scale(dict(N_narrow=100, N_twin=100, N_hmax=100, N_R=500), 7)["N_narrow"], 20)
        self.assertEqual(P._scale(dict(N_narrow=100, N_twin=100, N_hmax=100, N_R=500), 29)["N_R"], 145)
        self.assertEqual(P._scale(dict(N_narrow=100, N_twin=100, N_hmax=100, N_R=500), 29)["N_narrow"], 29)
        self.assertEqual(P._scale(dict(N_narrow=100, N_twin=100, N_hmax=100, N_R=500), 57)["N_hmax"], 57)


BASE3 = ("s1", "s225", "s4")  # the amplitudes of the original three-amplitude input


def synth_pilot(seed=3, scale=0.01, amps=BASE3):
    rng = np.random.default_rng(seed)
    out = {}
    for i, amp in enumerate(amps):
        out[amp] = {}
        for e in P.ESTIMATORS:
            out[amp][e] = {"D_f": (scale * rng.normal(size=20)).tolist(), "D_tw": (scale * rng.normal(size=20)).tolist(),
                           "d16_8": (scale * rng.normal(size=20)).tolist(),
                           "prod": (0.05 * (i + 1) + scale * rng.normal(size=20)).tolist(),
                           "mean_2x": 0.05 * (i + 1), "mean_h16": 0.05 * (i + 1)}
    return out


FORBIDDEN = re.compile(r"mean|sd|std|margin|verdict|alpha|pilot_|D_f|D_tw", re.I)
LINE_N = re.compile(r"^(s025|s1|s225|s4) (N_narrow|N_twin|N_hmax|N_R|N_A|N_run|par2_NR)=(\d+|>500)$")


def run_cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = nr_rule.main(argv)
    return rc, buf.getvalue()


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pilot = os.path.join(self.tmp.name, "pilot.json")
        self.out = os.path.join(self.tmp.name, "nr.json")

    def load(self):
        with open(self.out) as f:
            return json.load(f)

    def write(self, name, obj):
        p = os.path.join(self.tmp.name, name)
        with open(p, "w") as f:
            json.dump(obj, f)
        return p

    def test_prints_only_n_values(self):
        self.write("pilot.json", synth_pilot())
        rc, text = run_cli([self.pilot, "--out", self.out])
        self.assertEqual(rc, 0)
        lines = text.strip().splitlines()
        self.assertEqual(len(lines), 21)
        for ln in lines:
            self.assertRegex(ln, LINE_N)
        self.assertIsNone(FORBIDDEN.search(re.sub(r"N_narrow|N_twin|N_hmax|N_R|N_A|N_run|par2_NR", "", text)), text)
        res = self.load()
        self.assertEqual(sorted(res), sorted(BASE3))
        for amp in BASE3:
            for k in nr_rule.KEYS:
                self.assertIsInstance(res[amp][k], int)
                self.assertIn(f"{amp} {k}={res[amp][k]}", text)
            self.assertEqual(res[amp]["N_run"], max(res[amp]["N_R"], res[amp]["N_A"], res[amp]["N_twin"]))

    def test_original_three_amplitude_input_gives_the_original_output(self):
        # reference: what the loop over ("s1", "s225", "s4") always produced (P.amplitude_ns per amplitude, in that order)
        pilot = synth_pilot()
        self.write("pilot.json", pilot)
        rc, text = run_cli([self.pilot, "--out", self.out])
        self.assertEqual(rc, 0)
        want_lines, want_json = [], {}
        for amp in BASE3:
            n, capped = P.amplitude_ns(pilot[amp])
            want_lines += [f"{amp} {k}={'>500' if k in capped else n[k]}" for k in nr_rule.KEYS]
            want_json[amp] = {**n, "capped": capped}
        self.assertEqual(text.strip().splitlines(), want_lines)
        self.assertEqual(self.load(), want_json)
        # the registry order is used whatever the key order of the file
        self.write("pilot.json", {amp: pilot[amp] for amp in reversed(BASE3)})
        rc, text2 = run_cli([self.pilot, "--out", self.out])
        self.assertEqual((rc, text2), (0, text))

    def test_s025_alone_and_four_amplitudes(self):
        full = synth_pilot(amps=("s025", *BASE3))
        self.write("pilot.json", {"s025": full["s025"]})
        rc, text = run_cli([self.pilot, "--out", self.out])
        self.assertEqual(rc, 0)
        lines = text.strip().splitlines()
        self.assertEqual(len(lines), 7)
        self.assertTrue(all(ln.startswith("s025 ") for ln in lines))
        self.assertEqual(sorted(self.load()), ["s025"])
        n, _ = P.amplitude_ns(full["s025"])
        self.assertEqual(self.load()["s025"]["N_R"], n["N_R"])
        # the amplitudes are independent: s025 in a four-amplitude input gets the N of its own pilot, the others are unchanged
        self.write("pilot.json", full)
        rc, text4 = run_cli([self.pilot, "--out", self.out])
        self.assertEqual(rc, 0)
        self.assertEqual(text4.strip().splitlines()[:7], lines)
        self.assertEqual(sorted(self.load()), ["s025", "s1", "s225", "s4"])
        self.assertEqual([ln.split(" ")[0] for ln in text4.strip().splitlines()], ["s025"] * 7 + ["s1"] * 7 + ["s225"] * 7 + ["s4"] * 7)

    def test_unknown_amplitude_is_refused(self):
        self.write("pilot.json", {"s9": synth_pilot(amps=("s1",))["s1"]})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            nr_rule.main([self.pilot, "--out", self.out])
        self.assertFalse(os.path.exists(self.out))

    def test_cap_marker(self):
        pilot = synth_pilot(scale=1.0)  # sd >> margins
        self.write("pilot.json", pilot)
        rc, text = run_cli([self.pilot, "--out", self.out])
        self.assertIn(">500", text)
        for ln in text.strip().splitlines():
            self.assertRegex(ln, LINE_N)
        res = self.load()
        self.assertEqual(res["s1"]["N_R"], 500)
        self.assertIn("N_R", res["s1"]["capped"])

    def test_budget_mode(self):
        self.write("pilot.json", synth_pilot())
        costs = self.write("costs.json", flat_costs())
        rc, text = run_cli([self.pilot, "--out", self.out, "--budget-gpu-h", "0.01", "--costs", costs])
        self.assertEqual(rc, 0)
        lines = text.strip().splitlines()
        self.assertRegex(lines[0], r"^budget_gpu_h=\S+ projected_gpu_h_unscaled=\S+ s4_ladders_to_20=(yes|no) f=\S+ "
                                   r"projected_gpu_h=\S+ owner=(yes|no)$")
        for ln in lines[1:]:
            self.assertRegex(ln, LINE_N)
        self.assertIsNone(FORBIDDEN.search(lines[0].replace("projected_gpu_h", "").replace("budget_gpu_h", "")), lines[0])
        res = self.load()
        self.assertIn("f", res)
        self.assertEqual(res["N"]["s1"]["N_run"], max(res["N"]["s1"][k] for k in ("N_R", "N_A", "N_twin")))

    def test_budget_args_go_together(self):
        self.write("pilot.json", synth_pilot())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            nr_rule.main([self.pilot, "--out", self.out, "--budget-gpu-h", "1"])


if __name__ == "__main__":
    unittest.main()
