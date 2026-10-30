"""End-to-end tests of sf39.analyze on a fake complete campaign (fixtures_analysis; known answers from the closed forms of fixtures_runner).

Run (about two minutes): `cd A/scripts && python3 -m unittest discover -s . -t . -p 'test_analyze*.py'`.
The F10 reference/width counts (30 in the campaign) are patched to 9 (`power.REF_FLOOR`, `power.WIDTH_ARMS`, and the s4_w12 + s4_w12c split
`analyze.W12_PARTS`, 13 + 17 in the campaign, 5 + 4 here) to keep the fixture small.
"""
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from sf39 import analyze as an
from sf39 import campaign as cp
from sf39 import decide, power, stats
from sf39.tests import fixtures_analysis as fa
from sf39.tests import fixtures_campaign as fc
from sf39.tests import fixtures_par2 as fp
from sf39.tests import fixtures_runner as fx

N = fa.N_PILOT
REAL_W12 = an.W12_PARTS     # captured at import, before the Study patches it with the small fixture split
B = fa.BUDGET
SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../A/scripts


def close(tc, a, b, rtol=1e-10, atol=1e-13, msg=""):
    tc.assertTrue(abs(a - b) <= atol + rtol * abs(b), "%s %.17g != %.17g" % (msg, a, b))


def slurp(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run_quiet(*a, **k):
    err = io.StringIO()
    rc = an.run(*a, n_pilot=N, err=err, **k)
    return rc, err.getvalue()


def tf_of(arm, amp):
    """(tf, qf) of an arm as built (s1 uses identity role factors)."""
    with mock.patch.dict(fc.ROLE_F, fa.IDENT if amp == "s1" else {}):
        return fc.arm_factors(arm, amp)


def qb(g, qf=1.0):
    return (0.97 + 0.01 * g) * qf


def wvar_unif(v):
    v = np.asarray(v, float)
    return float(np.mean((v - v.mean()) ** 2))


class Study(unittest.TestCase):
    """One fake complete campaign and one analysis run, shared by the read-only tests."""

    @classmethod
    def setUpClass(cls):
        cls.patches = [mock.patch.object(power, "REF_FLOOR", fa.REF_FLOOR),
                       mock.patch.dict(power.WIDTH_ARMS, {"s4_w24": fa.REF_FLOOR, "s4_w12": fa.REF_FLOOR}),
                       mock.patch.object(an, "W12_PARTS", fa.W12_PARTS)]
        for p in cls.patches:
            p.start()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = os.path.join(cls.tmp.name, "out")
        cls.art = os.path.join(cls.tmp.name, "A")
        cls.dest = os.path.join(cls.tmp.name, "analysis")
        os.makedirs(cls.out)
        fa.build_complete(cls.out, cls.art, SCRIPTS)
        cls.rc, cls.err = run_quiet(cls.out, cls.art, cls.dest)
        with open(os.path.join(cls.dest, "summary.json")) as f:
            cls.S = json.load(f)
        cls.frz = an.load_frozen(cls.out)  # the original freeze merged with checkpoint/s025/tau_b_star_s025.json

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        for p in cls.patches:
            p.stop()

    # -- helpers ---------------------------------------------------------------------------------
    def tau(self, amp):
        return self.frz["amplitudes"][amp]["tau_b_star"]

    def A(self, amp):
        return cp.arm_ids(amp)["A"]

    def exp_tam(self, amp, arm, g, U, gen_index=None):
        tf, _ = tf_of(arm, amp)
        return fc.expected_alpha_TAM(g if gen_index is None else gen_index, tf, U, self.tau(amp) * fx.ELL / U)

    def exp_tp(self, amp, arm, g, shift=(0.0, 0.0)):
        return fc.expected_alpha_TP(g, tf_of(arm, amp)[0], shift)

    def csv(self, name):
        return read_csv(os.path.join(self.dest, "reduced", name + ".csv"))

    def flux(self, name):
        """{seed: row} of the flux rows of a reduced CSV."""
        return {int(r["seed"]): r for r in self.csv(name) if r["method"] == "flux"}

    # -- run -------------------------------------------------------------------------------------
    def test_run_ok_and_files(self):
        self.assertEqual(self.rc, 0, self.err)
        self.assertIn("ANALYSIS_DONE", self.err)
        for f in ("summary.json", "decisions.csv", "decisions.md", "manifest.txt"):
            self.assertTrue(os.path.isfile(os.path.join(self.dest, f)), f)
        self.assertFalse(os.path.exists(os.path.join(self.dest, "PARTIAL.txt")))
        self.assertFalse(self.S["partial"])
        self.assertEqual(self.S["missing"], [])

    # -- set sizes, seeds, U rule ------------------------------------------------------------------
    def expected_sets(self, amp):
        """{csv name: (seeds, U, hmax_div, arm for the closed form, gen indices)} per the section 3 rules."""
        b, a = B[amp], cp.arm_ids(amp)
        mean_q = lambda n, qf=1.0: float(np.mean([qb(g, qf) for g in range(n)]))  # noqa: E731
        s = {amp + "_prod": (b["N_R"], mean_q(b["N_R"]), 8, a["A"]),
             amp + "_prod_twinpairs": (b["N_twin"], mean_q(b["N_twin"]), 8, a["A"]),
             amp + "_tw": (b["N_twin"], mean_q(b["N_twin"], tf_of(a["tw"], amp)[1]), 8, a["tw"])}
        for arm in a["narrow"]:
            s[arm] = (b["N_narrow"], mean_q(b["N_narrow"], tf_of(arm, amp)[1]), 8, arm)
        for d in cp.HMAX_DIVS:
            s["%s_A_hmax%d" % (amp, d)] = (b["N_hmax"], mean_q(b["N_hmax"]), d, a["A"])
        return {k: (list(range(n)), U, d, arm) for k, (n, U, d, arm) in s.items()}

    def test_set_sizes_seeds_and_U(self):
        for amp in cp.AMPS:
            for name, (gs, U, d, arm) in self.expected_sets(amp).items():
                rows = self.csv(name)
                self.assertEqual(len(rows), 7 * len(gs), name)  # one row per realization and method
                self.assertEqual(sorted({int(r["seed"]) for r in rows}), [5001 + g for g in gs], name)
                self.assertEqual({r["method"] for r in rows}, {"flux", "unif", "expI", "full", "pollock", "ixc", "ixp"})
                self.assertEqual({int(r["hmax_div"]) for r in rows}, {d}, name)
                for r in rows:
                    close(self, float(r["U"]), U, 1e-12, msg=name)
                    close(self, float(r["tau_b_star"]), self.tau(amp), 1e-12, msg=name)
        # F10 sets: s4 only, each arm its own standalone U
        b = B["s4"]
        for name, seeds, gens, arm in (("s4_w24", [7101 + r for r in range(9)], [fa.WIDTH_G0 + r for r in range(9)], "s4_tw"),
                                       ("s4_w12", [7201 + r for r in range(5)] + [7215 + r for r in range(4)], [fa.WIDTH_G0 + 50 + r for r in range(9)], "s4_tw"),
                                       ("s4_tw_first30", [5001 + g for g in range(9)], list(range(9)), "s4_tw"),
                                       ("s4_n16_first30", [5001 + g for g in range(9)], list(range(9)), "s4_n16")):
            rows = self.csv(name)
            self.assertEqual(sorted({int(r["seed"]) for r in rows}), seeds, name)
            U = float(np.mean([qb(g, tf_of(arm, "s4")[1]) for g in gens]))
            for r in rows:
                close(self, float(r["U"]), U, 1e-12, msg=name)
        self.assertNotIn("F10", self.S["amplitudes"]["s1"])
        self.assertEqual(self.S["amplitudes"]["s4"]["F10"]["w24_vs_tw"]["n_wide"], 9)
        self.assertEqual(self.S["amplitudes"]["s4"]["F10"]["w12_vs_n16"]["n_reference"], 9)
        self.assertEqual(self.S["amplitudes"]["s4"]["F10"]["w12_vs_n16"]["n_wide"], 9)
        self.assertEqual(b["N_R"], self.S["amplitudes"]["s4"]["n"]["N_R"])

    def test_known_answers_per_realization(self):
        for amp in cp.AMPS:
            for name, (gs, U, d, arm) in self.expected_sets(amp).items():
                rows = self.flux(name)
                shift = fx.DISP[d] if name.endswith(("hmax16", "hmax32")) else (0.0, 0.0)
                for g in gs:
                    r = rows[5001 + g]
                    close(self, float(r["win.alpha_TAM"]), self.exp_tam(amp, arm, g, U), 1e-9, 1e-12, "%s g=%d TAM" % (name, g))
                    close(self, float(r["tp.alpha_TP"]), self.exp_tp(amp, arm, g, shift), 1e-9, 1e-12, "%s g=%d TP" % (name, g))
                    close(self, float(r["qbar"]), qb(g, tf_of(arm, amp)[1]), 1e-12, msg="%s g=%d qbar" % (name, g))

    def test_production_means_and_B(self):
        for amp in cp.AMPS:
            b = B[amp]
            U = float(np.mean([qb(g) for g in range(b["N_R"])]))
            want = {"alpha_TAM": [self.exp_tam(amp, self.A(amp), g, U) for g in range(b["N_R"])],
                    "alpha_TP": [self.exp_tp(amp, self.A(amp), g) for g in range(b["N_R"])]}
            for est, v in want.items():
                P = self.S["amplitudes"][amp]["production"][est]
                close(self, P["mean"], float(np.mean(v)), 1e-9, 1e-12)
                self.assertEqual(P["n"], b["N_R"])
                m, lo, hi, n, sd = stats.t_interval(v, 0.90)
                close(self, P["lo"], lo, 1e-8, 1e-11)
                close(self, P["hi"], hi, 1e-8, 1e-11)
                self.assertIn("bca_lo", P)  # BCa beside the Student CI
                Bk = self.S["amplitudes"][amp]["B"][est]
                _, hw, target, met = stats.halfwidth_check(v, stats.halfwidth_target(an.KIND[est], float(np.mean(v))))
                close(self, Bk["halfwidth_95"], hw, 1e-8, 1e-11)
                self.assertEqual(Bk["met"], met)

    # -- U rule on paired differences (known answer + mutation) -------------------------------------------
    def dtw_ok(self, S, amp):
        """True iff the production-minus-twin paired difference matches its closed form with each side's own U over the pair seeds."""
        b, a = B[amp], cp.arm_ids(amp)
        n = b["N_twin"]
        u_p = float(np.mean([qb(g) for g in range(n)]))
        qft = tf_of(a["tw"], amp)[1]
        u_t = float(np.mean([qb(g, qft) for g in range(n)]))
        ok = True
        for est in ("alpha_TAM", "alpha_TP"):
            if est == "alpha_TAM":
                want = [self.exp_tam(amp, a["A"], g, u_p) - self.exp_tam(amp, a["tw"], g, u_t) for g in range(n)]
            else:
                want = [self.exp_tp(amp, a["A"], g) - self.exp_tp(amp, a["tw"], g) for g in range(n)]
            got = S["A"][est]["trend"]["D_tw"]["ci"]
            ok &= got["n"] == n and abs(got["mean"] - float(np.mean(want))) <= 1e-12 + 1e-9 * abs(float(np.mean(want)))
        return bool(ok)

    def test_paired_U_rule_known_answer_and_mutations(self):
        for amp in cp.AMPS:
            self.assertTrue(self.dtw_ok(self.S["amplitudes"][amp], amp), amp)
        # mutation test: a paired side with the wrong U must fail the closed-form check (smallest amplitude, s225)
        amp = "s225"
        frz = self.frz

        def analyze_with(pair_u):
            C = an.Ctx(self.out, self.art, os.path.join(self.tmp.name, "mut"), frz, None, N, False)
            C.eff = an.effective_counts(C, B)
            with mock.patch.object(an, "pair_u", pair_u):
                return an.analyze_amp(C, amp)

        n = B[amp]["N_twin"]
        allprod = cp.production_records(self.out, amp)
        u_all = cp.u_standalone([x for x in allprod if x.seed - cp.PILOT_SEED0 < B[amp]["N_R"]])  # standalone production U, not the pair U
        base = analyze_with(cp.u_paired)
        self.assertTrue(self.dtw_ok(base, amp))
        wrong_prod = lambda p, o: (u_all, cp.u_paired(p, o)[1], cp.u_paired(p, o)[2])  # noqa: E731
        wrong_twin = lambda p, o: (cp.u_paired(p, o)[0], cp.u_paired(p, o)[0], cp.u_paired(p, o)[2])  # noqa: E731
        self.assertNotEqual(u_all, cp.u_paired(allprod[:n], allprod[:n])[0])
        self.assertFalse(self.dtw_ok(analyze_with(wrong_prod), amp), "production side with the wrong U went undetected")
        self.assertFalse(self.dtw_ok(analyze_with(wrong_twin), amp), "twin side with the wrong U went undetected")

    def test_hmax_ladder_known_answer(self):
        for amp in cp.AMPS:
            b = B[amp]
            a = self.A(amp)
            h = self.S["amplitudes"][amp]["A"]
            tam = [0.0] * b["N_hmax"]  # same times_x in every pass: the window alpha does not change
            tp = [self.exp_tp(amp, a, g, fx.DISP[16]) - self.exp_tp(amp, a, g, fx.DISP[8]) for g in range(b["N_hmax"])]
            close(self, h["alpha_TAM"]["trend"]["d16_8"]["ci"]["mean"], 0.0, atol=1e-12)
            close(self, h["alpha_TP"]["trend"]["d16_8"]["ci"]["mean"], float(np.mean(tp)), 1e-8, 1e-12)
            tp32 = [self.exp_tp(amp, a, g, fx.DISP[32]) - self.exp_tp(amp, a, g, fx.DISP[16]) for g in range(b["N_hmax"])]
            close(self, h["alpha_TP"]["trend"]["d32_16"]["ci"]["mean"], float(np.mean(tp32)), 1e-8, 1e-12)
            m16 = stats.margin("TP", float(np.mean([self.exp_tp(amp, a, g, fx.DISP[16]) for g in range(b["N_hmax"])])))
            close(self, h["alpha_TP"]["margins"]["m_hmax"], m16, 1e-9)
            self.assertEqual(h["alpha_TAM"]["hmax_verdict"], "converged")  # an exactly-zero difference: degenerate CI inside the margin

    # -- verdict / label plumbing ---------------------------------------------------------------------------
    def test_end_to_end_verdicts_and_labels(self):
        s1, s225, s4 = (self.S["amplitudes"][k] for k in ("s1", "s225", "s4"))
        # s1: every rung and the twin equal the production content -> converged by construction, physical (positive)
        for est in ("alpha_TAM", "alpha_TP", "alpha_LAM"):
            self.assertEqual(s1["A"][est]["narrow_verdict"], "converged", est)
            self.assertEqual(s1["A"][est]["F4_verdict"], "converged", est)
            self.assertEqual(s1["A"][est]["hmax_verdict"], "converged", est)
            self.assertTrue(s1["A"][est]["F4"]["order_ok"])
        for est in ("alpha_TAM", "alpha_TP"):
            self.assertEqual(s1["D"][est]["label"], "physical (positive)")
            self.assertEqual(s1["D"][est]["branch"], 4)
        self.assertEqual(s1["D"]["amplitude_label"], "physical (positive)")
        # s225 / s4: designed role factors -> the narrow ladder is not converged, D_f above 0 -> branch 1
        for S in (s225, s4):
            for est in ("alpha_TAM", "alpha_TP"):
                self.assertEqual(S["A"][est]["narrow_verdict"], "not converged", est)
                self.assertEqual(S["A"][est]["F4_verdict"], "not converged", est)
                d = S["D"][est]
                self.assertEqual((d["label"], d["branch"], d["ladder"], d["difference"]),
                                 ("unresolved (grid-limited, lower bound)", 1, "h", "D_f"))
                self.assertGreater(S["A"][est]["trend"]["D_f"]["ci"]["lo"], 0)
            self.assertEqual(S["D"]["amplitude_label"], "unresolved (grid-limited, lower bound)")
            self.assertEqual(S["A"]["alpha_TAM"]["hmax_verdict"], "converged")
        # richardson guard recorded for every estimator; the rung means of s225 are monotone
        self.assertIn("p_obs", s225["A"]["alpha_TAM"]["richardson"])

    def synth_ladder(self, df_shift, dtw_shift, prod_mean=0.2, mean_2x=0.2, seed=3):
        rng = np.random.RandomState(seed)
        n = 20
        noise = lambda sd: rng.normal(0.0, sd, n)  # noqa: E731
        vp = prod_mean + noise(0.01)
        v2 = mean_2x + noise(0.01)
        return dict(vals_prod=vp, vals_half=prod_mean + noise(0.01), vals_1x=prod_mean + noise(0.01), vals_2x=v2,
                    Dc=df_shift + noise(1e-4), Df=df_shift + noise(1e-4), Dtw=dtw_shift + noise(1e-4), vals_prod_tw=vp,
                    d16_8=noise(1e-4), d32_16=noise(1e-4), vals_h16=prod_mean + noise(0.01))

    def test_ladder_block_converged(self):
        blk, rd = an.ladder_block("alpha_TAM", **self.synth_ladder(0.0, 0.0))
        self.assertEqual((blk["narrow_verdict"], blk["F4_verdict"], blk["hmax_verdict"]), ("converged", "converged", "converged"))
        self.assertEqual((rd["label"], rd["branch"]), ("physical (positive)", 4))

    def test_ladder_block_not_converged(self):
        blk, rd = an.ladder_block("alpha_TAM", **self.synth_ladder(0.05, 0.05))
        self.assertEqual((blk["narrow_verdict"], blk["F4_verdict"]), ("not converged", "not converged"))
        self.assertEqual((rd["label"], rd["branch"], rd["ladder"], rd["difference"]), ("unresolved (grid-limited, lower bound)", 1, "h", "D_f"))
        # LAM has no rule D
        _, rl = an.ladder_block("alpha_LAM", **self.synth_ladder(0.05, 0.05))
        self.assertIsNone(rl)

    def test_ladder_block_numerical_uses_the_2x_values(self):
        # D_f and D_tw below 0 (not converged); the 2x-rung values (finer setting of D_f) contain 0, the production values do not
        kw = self.synth_ladder(-0.05, -0.05, prod_mean=0.2, mean_2x=0.0)
        kw["vals_2x"] = np.linspace(-0.01, 0.01, 20)
        blk, rd = an.ladder_block("alpha_TP", **kw)
        self.assertEqual(blk["F4_verdict"], "not converged")
        self.assertEqual((rd["label"], rd["branch"], rd["ladder"], rd["difference"]), ("numerical", 2, "h", "D_f"))
        kw["vals_2x"] = 0.2 + np.linspace(-0.01, 0.01, 20)  # a finer-setting CI far from 0: branch 3 instead
        blk, rd = an.ladder_block("alpha_TP", **kw)
        self.assertEqual((rd["label"], rd["branch"]), ("unresolved (grid-limited, upper bound)", 3))

    # -- Gate 5 -------------------------------------------------------------------------------------------------
    def lag_uc(self, amp, n):
        tbm = self.frz["amplitudes"][amp]["gate5"]["twin"]["t_b_min"]
        us = []
        for g in range(n):
            gg = fx.gen(g)
            mx = [float(np.mean([fx.X0 + fx.U0 * t + gg["c1"][i] * np.sqrt(t) for i in range(fx.NP) if gg["t_end"][i] > t])) for t in fx.TIMES]
            us.append((np.interp(tbm, fx.TIMES, mx) - fx.X0) / tbm)
        return float(np.mean(us))

    def lag_alpha(self, g, tau, U):
        """alpha_LAM of the uniform-weight window set at t_hi = tau ell / U: Var(c1 | records at j_hi)/(2 ell U) (variance linear in t)."""
        t_hi = tau * fx.ELL / U
        j = next(k for k, t in enumerate(fx.TIMES) if t >= t_hi)
        gg = fx.gen(g)
        idx = [i for i in range(fx.NP) if gg["t_end"][i] > fx.TIMES[j]]
        return wvar_unif(gg["c1"][idx]) / (2.0 * fx.ELL * U)

    def test_gate5_known_answers(self):
        for amp in cp.AMPS:
            n = B[amp]["N_R"]
            g5 = self.frz["amplitudes"][amp]["gate5"]
            uc = self.lag_uc(amp, n)
            L = np.array([self.lag_alpha(g, g5["tau_c"], uc) for g in range(n)])
            uc_p = float(np.mean([fa.par2_u(g, amp) for g in range(n)]))
            P = np.array([fa.par2_a(g)[0] / (2.0 * fp.LAM * uc_p) for g in range(n)])
            u_fv = float(np.mean([qb(g) for g in range(n)]))
            tau_fv = min(u_fv * g5["twin"]["t_b_min"] / g5["twin"]["ell"], g5["par2"]["tau_b_star"])
            L_fv = np.array([self.lag_alpha(g, tau_fv, u_fv) for g in range(n)])
            sig2 = np.array([fa.par2_var(g) for g in range(n)])
            want = decide.gate5_verdict(L, P, L_fv, sig2, an.SIGMA2_NOMINAL[amp])
            C = self.S["amplitudes"][amp]["C"]
            close(self, C["lagrangian"]["U_c"], uc, 1e-9)
            close(self, C["par2"]["U_c"], uc_p, 1e-10)
            for key in ("primary", "fv_qbar", "ancova"):
                got = C["gate5"][key]
                for f in ("diff", "lo", "hi", "margin"):
                    close(self, got[f], want[key][f], 1e-7, 1e-11, "%s %s %s" % (amp, key, f))
                self.assertEqual(got["verdict"], want[key]["verdict"])
                self.assertIn("bca_lo", got)
            close(self, C["gate5"]["primary"]["margin"], 0.10 * float(np.mean(P)), 1e-10)
            close(self, C["lagrangian"]["alpha_LAM"]["mean"], float(np.mean(L)), 1e-9)
            close(self, C["par2"]["alpha_x"]["mean"], float(np.mean(P)), 1e-9)
            close(self, C["gate5"]["ancova"]["a"], stats.ancova_adjusted(P, sig2, an.SIGMA2_NOMINAL[amp]).a, 1e-9)
            t_hi = g5["tau_c"] * fp.LAM / uc_p
            late = sum(1 for g in range(N, n) if fa.par2_tb_step(g) * fp.DT < t_hi)
            self.assertEqual((C["par2"]["late_beyond_pilot"], C["par2"]["n_beyond_pilot"]), (late, n - N))
            # the frozen tau_c and t_b,min are used, not recomputed
            close(self, C["frozen"]["tau_c"], g5["tau_c"], 1e-14)
            close(self, C["lagrangian_fv_qbar"]["tau_c"], tau_fv, 1e-12)

    # -- F7, F8, F10, F11, E: plumbing against the reduced CSVs -------------------------------------------------------
    def test_f7(self):
        for amp in cp.AMPS:
            rows = self.flux(amp + "_prod")
            d = [float(r["tp.delta_half"]) for _, r in sorted(rows.items())]
            want = stats.delta_half(d)
            got = self.S["amplitudes"][amp]["F7"]
            for k in ("mean", "lo", "hi"):
                close(self, got[k], want[k], 1e-9, 1e-13)
            self.assertEqual(got["decreasing"], want["decreasing"])

    def test_f8(self):
        for amp in cp.AMPS:
            rows = [r for r in self.csv("rho_" + amp) if r["pair"] == "8-16" and int(r["plane"]) == fx.IB]
            self.assertEqual(len(rows), B[amp]["N_hmax"])
            rho = np.array([float(r["rho"]) for r in rows])
            F = self.S["amplitudes"][amp]["F8"]
            rd = F["pairs"]["8-16"]["readings"]
            close(self, rd["p95"], float(np.percentile(rho, 95)), 1e-12)
            close(self, rd["median"], float(np.median(rho)), 1e-12)
            m = F["m_alpha_TP_hmax"]
            S = F["S"]
            self.assertEqual(rd["pathwise_resolved"], bool(np.percentile(rho, 95) <= 0.1 * m / S))
            S_rows = float(np.mean([float(r["S_r"]) for r in rows]))
            close(self, S, S_rows, 1e-12)
            self.assertEqual(len(self.csv("rho_" + amp)), B[amp]["N_hmax"] * 3 * len(fx.PLANES))  # 3 pairs x planes
            self.assertEqual(set(F["pairs"]), {"8-16", "16-32", "8-32"})
            self.assertTrue(F["gating_label"])

    def test_f10(self):
        F = self.S["amplitudes"]["s4"]["F10"]
        for label, wide, ref in (("w24_vs_tw", "s4_w24", "s4_tw_first30"), ("w12_vs_n16", "s4_w12", "s4_n16_first30")):
            w, r = self.flux(wide), self.flux(ref)
            for est, col in (("alpha_TAM", "win.alpha_TAM"), ("alpha_TP", "tp.alpha_TP")):
                a_ = [float(x[col]) for _, x in sorted(w.items())]
                b_ = [float(x[col]) for _, x in sorted(r.items())]
                want = stats.width_check(a_, b_)
                got = F[label][est]
                close(self, got["diff"], want["diff"], 1e-9, 1e-13)
                close(self, got["margin"], 0.10 * abs(float(np.mean(a_))), 1e-10)
                self.assertEqual(got["label"], want["label"])
                self.assertIn("bca_lo", got)

    def test_f11(self):
        for amp in cp.AMPS:
            F = self.S["amplitudes"][amp]["F11"]
            rows = self.csv(amp + "_prod")
            by = {}
            for r in rows:
                if r["method"] in ("ixp", "ixc"):
                    by.setdefault(int(r["seed"]), {})[r["method"]] = r
            for est, col in (("alpha_TAM", "win.alpha_TAM"), ("alpha_TP", "tp.alpha_TP"), ("alpha_Tfixed", "win.alpha_Tfixed")):
                d = [float(v["ixp"][col]) - float(v["ixc"][col]) for _, v in sorted(by.items())]
                m, lo, hi, n, sd = stats.t_interval(d, 0.90)
                got = F["delta_pollock"]["production"][est]["ci"]
                close(self, got["mean"], m, 1e-9, 1e-13)
                close(self, got["lo"], lo, 1e-8, 1e-12)
                self.assertEqual(got["n"], B[amp]["N_R"])
            self.assertEqual(set(F["delta_pollock"]), {"production", "narrow_half", "narrow_1x", "narrow_2x", "twin"})
            self.assertEqual(set(F["diff_in_diff"]), {"narrow_half_minus_1x", "narrow_1x_minus_2x", "twin_minus_production"})
            # difference in differences = per-seed difference of the two Delta_P (seeds aligned)
            tw = {int(r["seed"]): r for r in self.csv(amp + "_tw") if r["method"] in ("ixp",)}
            twc = {int(r["seed"]): r for r in self.csv(amp + "_tw") if r["method"] in ("ixc",)}
            pr_p = {int(r["seed"]): r for r in self.csv(amp + "_prod_twinpairs") if r["method"] == "ixp"}
            pr_c = {int(r["seed"]): r for r in self.csv(amp + "_prod_twinpairs") if r["method"] == "ixc"}
            col = "win.alpha_TAM"
            dd = [(float(tw[s][col]) - float(twc[s][col])) - (float(pr_p[s][col]) - float(pr_c[s][col])) for s in sorted(tw)]
            close(self, F["diff_in_diff"]["twin_minus_production"]["alpha_TAM"]["ci"]["mean"], float(np.mean(dd)), 1e-8, 1e-12)
            ub = F["upper_bound_fixed_axis_alpha_T"]
            self.assertEqual(ub["alpha_Tfixed"]["sided"], "upper")
            self.assertEqual(ub["BdD_value"], an.BD_BOUND.get(amp))

    def test_s025_nominal_variance_and_the_one_sided_lower_limit(self):
        self.assertEqual((an.SIGMA2_NOMINAL["s025"], an.BD_BOUND["s025"]), (0.25, 1e-3))
        self.assertEqual(self.S["amplitudes"]["s025"]["C"]["par2"]["var_lnK"]["nominal"], 0.25)
        for amp in cp.AMPS:
            ub = self.S["amplitudes"][amp]["F11"]["upper_bound_fixed_axis_alpha_T"]
            if amp != "s025":  # the summary subtrees of the existing amplitudes get no new key
                self.assertEqual(sorted(ub), ["BdD_value", "alpha_Tfixed", "below_BdD", "conditions"], amp)
                continue
            self.assertEqual(sorted(ub), ["BdD_value", "alpha_Tfixed", "alpha_Tfixed_lower", "below_BdD", "conditions", "exceeds_BdD"])
            self.assertEqual(ub["BdD_value"], 1e-3)
            fx_ = [float(r["win.alpha_Tfixed"]) for _, r in sorted(self.flux("s025_prod").items())]
            _, lo, hi, n, _ = stats.t_interval(fx_, 0.95, "lower")
            lw = ub["alpha_Tfixed_lower"]
            self.assertEqual((lw["sided"], lw["level"], lw["n"], lw["hi"]), ("lower", 0.95, B["s025"]["N_R"], "inf"))
            close(self, lw["lo"], lo, 1e-12)
            self.assertEqual(ub["exceeds_BdD"], bool(lo > 1e-3))
            # the existing upper-limit entry is the one-sided 95 % upper limit, as at the other amplitudes
            self.assertEqual((ub["alpha_Tfixed"]["sided"], ub["alpha_Tfixed"]["level"]), ("upper", 0.95))
            self.assertEqual(ub["below_BdD"], bool(stats.t_interval(fx_, 0.95, "upper")[2] < 1e-3))

    def test_existing_amplitudes_are_reproduced_without_s025(self):
        """Addendum A6 at fixture level: the analysis over four amplitudes reproduces, per `regress_existing`, the s1 / s225 / s4 outputs
        of the analysis over the three original ones (same data, same budget file; only the amplitude registry differs); and
        `--amps s1,s225,s4` gives exactly those outputs, also while an s025 arm is incomplete (the contingency)."""
        from sf39 import regress_existing as rg
        three = ("s1", "s225", "s4")
        ref = os.path.join(self.tmp.name, "analysis_three")
        with mock.patch.object(cp, "AMPS", three):  # what the analysis was before s025 existed
            rc, err = run_quiet(self.out, self.art, ref)
        self.assertEqual(rc, 0, err)
        self.assertEqual(sorted(json.loads(slurp(os.path.join(ref, "summary.json")))["amplitudes"]), sorted(three))
        fails, adds, notes = rg.compare(ref, self.dest)
        self.assertEqual(fails, [])
        self.assertTrue(any("s025" in a for a in adds), adds)  # the additions are the s025 rows / files
        self.assertTrue(os.path.isfile(os.path.join(self.dest, "reduced", "s025_prod.csv")))
        self.assertFalse(os.path.exists(os.path.join(ref, "reduced", "s025_prod.csv")))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(rg.main(["--ref", ref, "--new", self.dest]), 0)
        self.assertIn("IDENTICAL", out.getvalue())
        self.assertIn(hashlib.sha256(slurp(os.path.join(ref, "manifest.txt"), "rb")).hexdigest(), out.getvalue())
        # and the comparison is not vacuous: one changed value in an s1 file of the new analysis is found
        bad = os.path.join(self.tmp.name, "analysis_bad")
        shutil.copytree(self.dest, bad)
        pth = os.path.join(bad, "reduced", "s1_prod.csv")
        with open(pth) as f:
            t = f.read()
        with open(pth, "w") as f:
            f.write(t + "x")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertNotEqual(rg.main(["--ref", ref, "--new", bad]), 0)
        # contingency: an s025 arm is incomplete and its frozen entry is absent; the default analysis refuses, `amps` does not need them
        done = os.path.join(cp.arm_dir(self.out, "s025_n10"), "realizations", "r_0003", "DONE")
        ck25 = os.path.join(self.out, "checkpoint", "s025")
        opt = os.path.join(self.tmp.name, "analysis_option")
        os.rename(done, done + ".off")
        os.rename(ck25, ck25 + ".off")
        try:
            rc_def, err_def = run_quiet(self.out, self.art, os.path.join(self.tmp.name, "analysis_default_refused"))
            rc_opt, err_opt = run_quiet(self.out, self.art, opt, amps=three)
        finally:
            os.rename(done + ".off", done)
            os.rename(ck25 + ".off", ck25)
        self.assertEqual(rc_def, 2, err_def)
        self.assertIn("no frozen entry for s025", err_def)
        self.assertEqual(rc_opt, 0, err_opt)
        files = sorted(os.path.relpath(os.path.join(r, f), ref) for r, _, fs in os.walk(ref) for f in fs)
        self.assertEqual(files, sorted(os.path.relpath(os.path.join(r, f), opt) for r, _, fs in os.walk(opt) for f in fs))
        for pth in files:  # every output byte-identical to the analysis of the original registry
            self.assertEqual(slurp(os.path.join(ref, pth), "rb"), slurp(os.path.join(opt, pth), "rb"), pth)
        self.assertFalse(json.loads(slurp(os.path.join(opt, "summary.json")))["partial"])
        self.assertEqual(rg.compare(opt, self.dest)[0], [])

    def test_a4_readings_only_for_s025_and_consistent_with_the_A_block(self):
        for amp in cp.AMPS:
            self.assertEqual("A4_readings" in self.S["amplitudes"][amp], amp == "s025", amp)
        R = self.S["amplitudes"]["s025"]["A4_readings"]
        self.assertEqual(sorted(R), ["alpha_LAM", "alpha_TAM", "alpha_TP"])
        t15 = 2.0 ** 1.5
        for est, r in R.items():
            A = self.S["amplitudes"]["s025"]["A"][est]
            mp, m = A["margins"]["mean_production"], A["margins"]["m_h"]
            close(self, r["production_mean"], mp)
            close(self, r["m_over_alpha"]["m"], m / abs(mp))
            close(self, r["m_over_alpha"]["twin_band"], t15 * m / abs(mp))
            excl = []
            for key, name, scale in (("D_f_frac", "D_f", 1.0), ("D_tw_frac", "D_tw", t15), ("d16_8_frac", "d16_8", 1.0)):
                c = A["trend"][name]["ci"]  # the CI of the A block, not recomputed
                d = scale * mp
                lo, hi = sorted((c["lo"] / d, c["hi"] / d))
                close(self, r[key]["mean"], c["mean"] / d)
                close(self, r[key]["lo"], lo)
                close(self, r[key]["hi"], hi)
                if c["lo"] > 0 or c["hi"] < 0:
                    excl.append({"D_f": "D_f", "D_tw": "D_tw/2^1.5", "d16_8": "d16_8"}[name])
            self.assertEqual([t.split(":")[0] for t in r["trend_detected"]], excl, est)
        # D_f_frac against the reduced CSVs (an independent path): (n20 - n10) over the production mean, alpha_TAM
        n20, n10, prod = self.flux("s025_n20"), self.flux("s025_n10"), self.flux("s025_prod")
        d_f = [float(n20[sd]["win.alpha_TAM"]) - float(n10[sd]["win.alpha_TAM"]) for sd in sorted(n20) if sd in n10]
        close(self, R["alpha_TAM"]["D_f_frac"]["mean"], float(np.mean(d_f)) / float(np.mean([float(r_["win.alpha_TAM"]) for r_ in prod.values()])), 1e-9)

    def test_a4_notes_on_the_s025_gate4_rows_only(self):
        rows = read_csv(os.path.join(self.dest, "decisions.csv"))
        R = self.S["amplitudes"]["s025"]["A4_readings"]
        g4 = [r_ for r_ in rows if r_["decision_point"] == "Gate 4 label (D)"]
        self.assertEqual([(r_["amplitude"], r_["estimator"]) for r_ in g4],
                         [(a, e) for a in cp.AMPS for e in ("alpha_TAM", "alpha_TP")])
        for r_ in g4:
            if r_["amplitude"] == "s025":
                self.assertTrue(r_["note"].endswith(an.a4_note(R[r_["estimator"]]["trend_detected"])), r_["note"])
                self.assertIn("refinement trend", r_["note"])
            else:
                self.assertNotIn("refinement trend", r_["note"])
        self.assertFalse([r_ for r_ in rows if r_["amplitude"] != "s025" and "refinement trend" in r_["note"]])

    def test_f7_not_available_is_listed_and_the_finite_case_is_unchanged(self):
        bud = an.load_budget(self.art)
        orig = an.column

        def column(res, key, mode=None):
            v = orig(res, key, mode)
            if key == "tp.delta_half":
                v = np.array(v, dtype=float)
                v[0] = np.nan
            return v

        C = an.Ctx(self.out, self.art, None, self.frz, None, N, False)
        C.eff = an.effective_counts(C, bud)
        with mock.patch.object(an, "column", column):
            S = an.analyze_amp(C, "s1")
        self.assertEqual(S["F7"], {"available": False})
        rows = [r_ for _, r_ in C.rows if r_["decision_point"] == "F7 Delta_half"]
        self.assertEqual(len(rows), 1)
        r_ = rows[0]
        self.assertEqual((r_["amplitude"], r_["estimator"], r_["verdict"], r_["ci"], r_["n"]),
                         ("s1", "alpha_TP", "not available", None, B["s1"]["N_R"]))
        self.assertIn("not finite in 1 of %d realizations" % B["s1"]["N_R"], r_["note"])
        # finite: the row of the full analysis is the unchanged one (a verdict, a CI, no 'not available')
        fin = [r2 for r2 in read_csv(os.path.join(self.dest, "decisions.csv")) if r2["decision_point"] == "F7 Delta_half"]
        self.assertEqual([r2["amplitude"] for r2 in fin], list(cp.AMPS))
        self.assertTrue(all(r2["verdict"] in ("decreasing", "not shown decreasing") and r2["ci_lo"] for r2 in fin))

    def test_gate5_refuses_a_par2_config_variance_that_is_not_the_nominal(self):
        art = self.art_copy()
        cfg = os.path.join(art, "configs", "par2_s025.yaml")
        with open(cfg) as f:
            t = f.read()
        self.assertIn("  sigma2: 0.25\n", t)
        with open(cfg, "w") as f:
            f.write(t.replace("  sigma2: 0.25\n", "  sigma2: 1.0\n"))
        dest = os.path.join(self.tmp.name, "d_sigma2")
        rc, err = run_quiet(self.out, art, dest, amps=("s025",))
        self.assertEqual(rc, 1, err)
        self.assertIn("ANALYSIS_FAILED ReduceError", err)
        self.assertIn("stochastic.sigma2 1.0 differs from the nominal variance 0.25", err)
        self.assertFalse(os.path.exists(os.path.join(dest, "summary.json")))

    def test_recorded_E(self):
        for amp in cp.AMPS:
            E = self.S["amplitudes"][amp]["E"]
            rows = self.flux(amp + "_prod")
            ev = [float(r["ev_p50"]) for _, r in sorted(rows.items())]
            close(self, E["context"]["ev"]["ev_p50"]["median"], float(np.median(ev)), 1e-12)
            close(self, E["context"]["ev"]["ev_p100"]["max"], max(float(r["ev_p100"]) for r in rows.values()), 1e-12)
            for est, col in (("alpha_TAM", "win.alpha_TAM"), ("alpha_TP", "tp.alpha_TP"), ("alpha_LAM", "win.alpha_e1")):
                a_ = [float(rows[s][col]) for s in sorted(rows)]
                A_ = [float(rows[s]["split.%s.A" % est]) for s in sorted(rows)]
                B_ = [float(rows[s]["split.%s.B" % est]) for s in sorted(rows)]
                want = stats.splithalf_fraction(A_, B_, a_)
                close(self, E["split_half"][est]["fraction"], want["fraction"], 1e-9)
                self.assertEqual(E["split_half"][est]["flag"], want["flag"])
            for mode in ("flux", "unif", "expI", "full", "pollock", "ixc", "ixp"):
                self.assertIn(mode, E["sensitivities"]["alpha_TAM"])
            self.assertNotIn("full", E["sensitivities"]["alpha_TP"])  # the full population has no plane estimator
            self.assertEqual(len(E["sigma_T2_mean"]["x1"]), len(fx.PLANES))
            self.assertEqual(max(E["sigma_T2_mean"]["n"]), B[amp]["N_R"])
            for k in ("exits_jhi", "excluded_fraction", "flow_stats", "t_b", "plane_hist_mean"):
                self.assertIn(k, E["context"])
        D = self.S["amplitudes"]["s225"]["A"]["alpha_TAM"]["trend"]["D_f"]["dist"]  # per-ladder distribution and largest realization
        self.assertIn(D["argmax_seed"], [5001 + g for g in range(B["s225"]["N_narrow"])])
        nr = self.S["amplitudes"]["s225"]["A"]["alpha_TAM"]["narrow_rung_vs_production"]
        self.assertEqual((nr["n_a"], nr["n_b"]), (B["s225"]["N_narrow"], B["s225"]["N_R"]))

    def test_series_in_main_run(self):
        for amp in cp.AMPS:
            ptr = self.S["amplitudes"][amp]["alpha_t"]
            self.assertEqual((ptr["file"], ptr["n_real"], ptr["n_snapshots"]), ("series/alpha_t_%s.csv" % amp, B[amp]["N_R"], len(fx.TIMES)))
            rows = read_csv(os.path.join(self.dest, ptr["file"]))
            self.assertEqual(len(rows), 4 * len(fx.TIMES))
            self.assertEqual({r["mode"] for r in rows}, set(an.SERIES_MODES))
            U = float(np.mean([qb(g) for g in range(B[amp]["N_R"])]))
            fl = [r for r in rows if r["mode"] == "flux"]
            for j, r in enumerate(fl):
                close(self, float(r["t"]), fx.TIMES[j], 1e-14)
                close(self, float(r["tau"]), U * fx.TIMES[j] / fx.ELL, 1e-12)
                self.assertEqual(int(r["n_real"]), B[amp]["N_R"])
            last = []
            for g in range(B[amp]["N_R"]):
                gg = fx.gen(g)
                idx = [i for i in range(fx.NP) if gg["t_end"][i] > fx.TIMES[-1]]
                last.append(fc._wvar([gg["c1"][i] for i in idx], [gg["w"][i] for i in idx]) / (2 * fx.ELL * U))
            close(self, float(fl[-1]["alpha_L_fixed_mean"]), float(np.mean(last)), 1e-9)
        # the series are in the manifest
        self.assertIn("series/alpha_t_s1.csv", slurp(os.path.join(self.dest, "manifest.txt")))

    # -- decision table, serialization, manifest, determinism ----------------------------------------------------------------
    def test_decision_table(self):
        rows = read_csv(os.path.join(self.dest, "decisions.csv"))
        self.assertEqual(list(rows[0]), ["tier", "amplitude", "estimator", "decision_point", "verdict", "ci_lo", "ci_hi", "n", "note"])
        order = [r["tier"] for r in rows]
        self.assertEqual(order, sorted(order, key=an.TIERS.index))
        self.assertEqual(set(order), set(an.TIERS))
        for amp in cp.AMPS:
            sel = [r for r in rows if r["amplitude"] == amp]
            cnt = lambda t: sum(1 for r in sel if r["tier"] == t)  # noqa: E731
            self.assertEqual((cnt("primary"), cnt("key secondary"), cnt("estimation")), (6, 15, 26), amp)
            self.assertEqual(cnt("exploratory"), 4 if amp == "s4" else 0)
        self.assertEqual(sum(1 for r in rows if r["tier"] == "process"), 1)
        lab = [r for r in rows if r["decision_point"] == "Gate 4 amplitude label (D)"]
        self.assertEqual([r["amplitude"] for r in lab], list(cp.AMPS))  # s025 first: rows of the existing amplitudes keep their order
        self.assertEqual([r["verdict"] for r in lab], ["unresolved (grid-limited, lower bound)", "physical (positive)"]
                         + ["unresolved (grid-limited, lower bound)"] * 2)
        md = slurp(os.path.join(self.dest, "decisions.md"))
        for t in an.TIERS:
            self.assertIn("## " + t, md)
        self.assertNotIn("PARTIAL", md)

    def test_jdump_is_deterministic_and_strict(self):
        text = an.jdump({"b": 1, "a": float("nan"), "c": 0.1, "d": [np.float64(1.0 / 3.0), True, None], "e": float("-inf"), 2: {"z": np.int64(3)}})
        self.assertEqual(json.loads(text), {"2": {"z": 3}, "a": None, "b": 1, "c": 0.1, "d": [1.0 / 3.0, True, None], "e": "-inf"})
        self.assertIn('"c": 0.10000000000000001', text)
        self.assertLess(text.index('"a"'), text.index('"b"'))
        raw = slurp(os.path.join(self.dest, "summary.json"))
        json.loads(raw, parse_constant=lambda c: self.fail("non-strict JSON constant %s" % c))

    def test_manifest_lists_every_file_with_sha256(self):
        lines = [ln for ln in slurp(os.path.join(self.dest, "manifest.txt")).splitlines() if ln]
        listed = {}
        for ln in lines:
            sha, path = ln.split("  ", 1)
            listed[path] = sha
        on_disk = sorted(os.path.relpath(os.path.join(r, f), self.dest) for r, _, fs in os.walk(self.dest) for f in fs if f != "manifest.txt")
        self.assertEqual(sorted(listed), on_disk)
        for p, sha in listed.items():
            self.assertEqual(hashlib.sha256(slurp(os.path.join(self.dest, p), "rb")).hexdigest(), sha, p)

    def test_deterministic_rerun(self):
        dest2 = os.path.join(self.tmp.name, "analysis2")
        rc, err = run_quiet(self.out, self.art, dest2)
        self.assertEqual(rc, 0, err)
        files = sorted(os.path.relpath(os.path.join(r, f), self.dest) for r, _, fs in os.walk(self.dest) for f in fs)
        self.assertEqual(files, sorted(os.path.relpath(os.path.join(r, f), dest2) for r, _, fs in os.walk(dest2) for f in fs))
        for p in files:
            self.assertEqual(slurp(os.path.join(self.dest, p), "rb"), slurp(os.path.join(dest2, p), "rb"), p)

    # -- refusals and partial -----------------------------------------------------------------------------------------------
    def art_copy(self, mutate=None):
        d = os.path.join(self.tmp.name, "A_%d" % len(os.listdir(self.tmp.name)))
        shutil.copytree(self.art, d)
        if mutate:
            with open(os.path.join(d, "configs", "budget.json")) as f:
                b = json.load(f)
            mutate(b)
            with open(os.path.join(d, "configs", "budget.json"), "w") as f:
                json.dump(b, f)
        return d

    def test_refuses_without_budget(self):
        art = self.art_copy()
        os.remove(os.path.join(art, "configs", "budget.json"))
        dest = os.path.join(self.tmp.name, "d_nobudget")
        rc, err = run_quiet(self.out, art, dest)
        self.assertEqual(rc, 2)
        self.assertIn("ANALYSIS_REFUSED", err)
        self.assertIn("budget", err)
        self.assertFalse(os.path.exists(dest))

    def test_refuses_invalid_budget(self):
        art = self.art_copy(lambda b: b["s1"].pop("N_hmax"))
        rc, err = run_quiet(self.out, art, os.path.join(self.tmp.name, "d_badbudget"))
        self.assertEqual(rc, 2)
        self.assertIn("s1.N_hmax", err)

    def test_refuses_without_frozen_file(self):
        empty = os.path.join(self.tmp.name, "empty_out")
        os.makedirs(empty)
        dest = os.path.join(self.tmp.name, "d_nofrozen")
        rc, err = run_quiet(empty, self.art, dest)
        self.assertEqual(rc, 2)
        self.assertIn("frozen", err)
        self.assertFalse(os.path.exists(dest))

    def test_refuses_budget_without_s025(self):
        art = self.art_copy(lambda b: b.pop("s025"))
        dest = os.path.join(self.tmp.name, "d_nos025")
        rc, err = run_quiet(self.out, art, dest)
        self.assertEqual(rc, 2)
        self.assertIn("ANALYSIS_REFUSED", err)
        self.assertIn("s025.N_narrow=None", err)
        self.assertFalse(os.path.exists(dest))

    def inflate(self, b):
        b["s1"].update(N_narrow=9, N_run=11, par2_NR=11)

    def test_refuses_missing_realizations_and_lists_them(self):
        art = self.art_copy(self.inflate)
        dest = os.path.join(self.tmp.name, "d_missing")
        rc, err = run_quiet(self.out, art, dest)
        self.assertEqual(rc, 2)
        self.assertIn("ANALYSIS_INCOMPLETE", err)
        for item in ("s1_n05: realizations not DONE: 8", "s1_n10: realizations not DONE: 8", "s1_n20: realizations not DONE: 8",
                     "s1_B: realizations not DONE: 3", "par2 s1: 10 realizations, budget asks for 11"):
            self.assertIn(item, err)
        self.assertNotIn("s225", err)
        self.assertFalse(os.path.exists(dest))  # nothing analysed, nothing written

    def test_allow_partial_labels_every_output(self):
        art = self.art_copy(self.inflate)
        dest = os.path.join(self.tmp.name, "d_partial")
        rc, err = run_quiet(self.out, art, dest, allow_partial=True)
        self.assertEqual(rc, 0, err)
        self.assertIn("ANALYSIS_DONE PARTIAL", err)
        S = json.loads(slurp(os.path.join(dest, "summary.json")))
        self.assertTrue(S["partial"])
        self.assertTrue(S["missing"])
        self.assertEqual(S["amplitudes"]["s1"]["n"]["N_narrow"], 8)  # effective count: the complete prefix
        self.assertEqual(S["amplitudes"]["s1"]["n"]["N_R"], B["s1"]["N_R"])
        self.assertTrue(os.path.isfile(os.path.join(dest, "PARTIAL.txt")))
        self.assertTrue(slurp(os.path.join(dest, "manifest.txt")).startswith("# PARTIAL"))
        self.assertIn("PARTIAL", slurp(os.path.join(dest, "decisions.md")).splitlines()[0])
        notes = [r["note"] for r in read_csv(os.path.join(dest, "decisions.csv"))]
        self.assertTrue(all(n.startswith("PARTIAL: ") for n in notes))
        # the partial analysis of the complete part equals the full one
        a1 = S["amplitudes"]["s1"]["A"]["alpha_TAM"]["trend"]["D_f"]["ci"]["mean"]
        self.assertEqual(a1, self.S["amplitudes"]["s1"]["A"]["alpha_TAM"]["trend"]["D_f"]["ci"]["mean"])

    # -- F10 (ii): s4_w12 + s4_w12c wide set (owner decision 2026-10-28) ------------------------------------------------------------
    @contextlib.contextmanager
    def w12_layout(self, n12, n12c, seeds12c=None):
        """The REAL W12_PARTS split with minimal DONE realizations (field.json, flow_stats.json, DONE) in s4_w12 (n12) and s4_w12c (n12c);
        the fixture's own s4_w12 / s4_w12c directories are moved aside and restored."""
        moved = []
        try:
            for arm, n, s0s in (("s4_w12", n12, None), ("s4_w12c", n12c, seeds12c)):
                d = cp.arm_dir(self.out, arm)
                if os.path.isdir(d):
                    os.rename(d, d + ".bak")
                    moved.append(d)
                for r in range(n):
                    R = os.path.join(d, "realizations", "r_%04d" % r)
                    os.makedirs(R)
                    seed = (s0s[r] if s0s else {"s4_w12": 7201, "s4_w12c": 7215}[arm] + r)
                    for name, body in (("field.json", {"seed": seed}), ("flow_stats.json", {"qbar": 1.0})):
                        with open(os.path.join(R, name), "w") as f:
                            json.dump(body, f)
                    open(os.path.join(R, "DONE"), "w").close()
            with mock.patch.object(an, "W12_PARTS", REAL_W12):
                yield an.Ctx(self.out, self.art, None, self.frz, None, N, False)
        finally:
            for arm in ("s4_w12", "s4_w12c"):
                d = cp.arm_dir(self.out, arm)
                shutil.rmtree(d, ignore_errors=True)
                if d in moved:
                    os.rename(d + ".bak", d)

    def test_w12_real_split_constants(self):
        self.assertEqual(REAL_W12, (("s4_w12", 7201, 13), ("s4_w12c", 7215, 17)))
        self.assertEqual(sum(n for _, _, n in REAL_W12), 30)
        self.assertEqual(an.W12_EXCLUDED_SEEDS, [7214])

    def test_w12_split_complete_is_not_refused(self):
        bud = an.load_budget(self.art)
        with self.w12_layout(13, 17) as C:
            self.assertEqual(an.missing_items(C, bud), [])
            seeds = [x.seed for x in an.w12_records(C)]
            self.assertEqual(seeds, list(range(7201, 7214)) + list(range(7215, 7232)))
            self.assertNotIn(7214, seeds)
            self.assertEqual(an.effective_counts(C, bud)["s4"]["n_w12"], 30)

    def test_w12_split_refusals_are_precise(self):
        bud = an.load_budget(self.art)
        with self.w12_layout(12, 17) as C:  # s4_w12 only 12 DONE
            self.assertEqual(an.missing_items(C, bud), ["s4_w12: realizations not DONE: 12"])
        with self.w12_layout(13, 16) as C:  # s4_w12c short by one
            self.assertEqual(an.missing_items(C, bud), ["s4_w12c: realizations not DONE: 16"])
        wrong = [7215 + r for r in range(17)]
        wrong[2] = 7214  # the excluded seed must not be accepted in the continuation arm
        with self.w12_layout(13, 17, wrong) as C:
            self.assertEqual(an.missing_items(C, bud), ["s4_w12c: seed != 7215 + r for r = 2"])
        with self.w12_layout(13, 16):  # through run(): refused, nothing written
            dest = os.path.join(self.tmp.name, "d_w12c_short")
            rc, err = run_quiet(self.out, self.art, dest)
            self.assertEqual(rc, 2)
            self.assertIn("MISSING s4_w12c: realizations not DONE: 16", err)
            self.assertFalse(os.path.exists(dest))

    def test_w12_split_partial_prefixes(self):
        bud = an.load_budget(self.art)
        with self.w12_layout(11, 3) as C:
            C.partial = True
            self.assertEqual(an.effective_counts(C, bud)["s4"]["n_w12"], 14)
            self.assertEqual([x.seed for x in an.w12_records(C)], list(range(7201, 7212)) + [7215, 7216, 7217])

    def test_w12_exclusion_is_disclosed_and_not_partial(self):
        F = self.S["amplitudes"]["s4"]["F10"]
        w = F["w12_vs_n16"]
        self.assertEqual(w["n_wide"], 9)
        self.assertEqual(w["wide_arms"], ["s4_w12", "s4_w12c"])
        self.assertEqual(w["wide_seeds"], [7201 + r for r in range(5)] + [7215 + r for r in range(4)])
        self.assertEqual(w["excluded_seeds"], [7214])
        self.assertIn("owner decision 2026-10-28", w["excluded_reason"])
        self.assertIn("s4_w12c seeds 7215-7231", w["excluded_reason"])
        self.assertNotIn("excluded_seeds", F["w24_vs_tw"])
        dec = read_csv(os.path.join(self.dest, "decisions.csv"))
        self.assertEqual(sum(r["decision_point"].startswith("F10 width") for r in dec), 2 * 2)  # 2 checks x (TAM, TP)
        for r in dec:
            if r["decision_point"] == "F10 width check w12_vs_n16":
                self.assertIn("seed 7214 excluded", r["note"])
            if r["decision_point"].startswith("F10 width"):
                self.assertFalse(r["note"].startswith("PARTIAL"))
            if r["decision_point"] == "F10 width check w24_vs_tw":
                self.assertNotIn("7214", r["note"])
        self.assertFalse(self.S["partial"])

    def test_window_error_is_a_failure_not_a_clip(self):
        frz = json.loads(json.dumps(self.frz))
        for amp in cp.AMPS:
            frz["amplitudes"][amp]["tau_b_star"] *= 100.0  # t_hi far beyond the last snapshot
        C = an.Ctx(self.out, self.art, os.path.join(self.tmp.name, "d_win"), frz, None, N, False)
        C.eff = an.effective_counts(C, B)
        from sf39 import io as sio
        with self.assertRaises(sio.WindowError):
            an.analyze_amp(C, "s1")


class Series(unittest.TestCase):
    """alpha(t) / sigma_T^2(t) series on small single-arm trees (linear variance growth: closed forms at snapshots with a constant set)."""

    ARM, AMP, NR, U = "s225_n16", "s225", 3, 0.9
    J0 = 32  # t = 8.25 > every exit time of the fixture: the particle set is constant from snapshot 31 on

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = self.tmp.name
        fa.build_single(self.out, self.ARM, self.AMP, self.NR, 0, 5001)
        self.tf = fc.arm_factors(self.ARM, self.AMP)[0]

    def recs(self):
        return cp.done_records(self.out, self.ARM)

    def expected(self, j, mode, tf=None, theta=0.0):
        """Closed-form (L_fixed, T_fixed, L_aligned, TAM, sigma_T2) ensemble means at snapshot j (set = records at j)."""
        tf = self.tf if tf is None else tf
        rows = []
        for r in range(self.NR):
            g = fx.gen(r, theta)
            pol = mode.startswith("pollock")
            end = g["p_end"] if pol else g["t_end"]
            idx = [i for i in range(fx.NP) if end[i] > fx.TIMES[j]]
            c2, c3 = (g["pc2"], g["pc3"]) if pol else (tf * g["c2"], tf * g["c3"])
            w = [1.0] * len(idx) if mode in ("unif", "pollock_unif") else [float((g["pw"] if pol else g["w"])[i]) for i in idx]
            v = lambda a: fc._wvar([float(a[i]) for i in idx], w)  # noqa: E731
            c1 = g["c1"]
            f = 2.0 * fx.ELL * self.U
            if theta == 0.0:
                vf = (v(c1), v(c2), v(c3))
                vt = (v(c1), v(c2), v(c3))
            else:
                th = theta
                vf = (v(np.cos(th) * c1 - np.sin(th) * c2), v(np.sin(th) * c1 + np.cos(th) * c2), v(c3))
                vt = (v(c1), v(c2), v(c3))
            rows.append([vf[0] / f, 0.5 * (vf[1] + vf[2]) / f, vt[0] / f, 0.5 * (vt[1] + vt[2]) / f, fx.TIMES[j] * 0.5 * (vf[1] + vf[2])])
        return np.array(rows)

    def rows_of(self, rows, mode):
        return [r for r in rows if r["mode"] == mode]

    def check(self, rows, mode, j, theta=0.0, tf=None):
        want = self.expected(j, mode, tf, theta)
        r = self.rows_of(rows, mode)[j]
        for k, name in enumerate(an.SERIES_COLS):
            close(self, r[name + "_mean"], float(want[:, k].mean()), 1e-9, 1e-12, "%s j=%d %s" % (mode, j, name))
            close(self, r[name + "_se"], float(want[:, k].std(ddof=1) / np.sqrt(self.NR)), 1e-7, 1e-12, "%s j=%d %s se" % (mode, j, name))

    def test_linear_growth_gives_constant_alpha_interior_and_one_sided_ends(self):
        rows, n_real, n = an.alpha_series_rows(self.recs(), self.U)
        self.assertEqual((n_real, n), (self.NR, len(fx.TIMES)))
        self.assertEqual(len(rows), 4 * n)
        for mode in an.SERIES_MODES:
            for j in list(range(self.J0, n - 1)):  # interior, constant set: central differences
                self.check(rows, mode, j)
            self.check(rows, mode, n - 1)  # last snapshot: one-sided
            self.check(rows, mode, 0)      # first snapshot: one-sided (same set at snapshots 0 and 1)
            r0 = self.rows_of(rows, mode)[0]
            self.assertEqual((r0["j"], r0["n_real"]), (0, self.NR))
            close(self, r0["t"], fx.TIMES[0], 1e-14)
            close(self, r0["tau"], self.U * fx.TIMES[0] / fx.ELL, 1e-14)
        # constant in time where the set is constant (interior values equal the last one)
        fl = self.rows_of(rows, "flux")
        close(self, fl[self.J0]["alpha_L_fixed_mean"], fl[-1]["alpha_L_fixed_mean"], 1e-9)
        # the modes differ (flux vs uniform weights; Pollock positions)
        self.assertNotAlmostEqual(fl[-1]["alpha_T_fixed_mean"], self.rows_of(rows, "unif")[-1]["alpha_T_fixed_mean"], places=6)
        self.assertNotAlmostEqual(fl[-1]["alpha_T_fixed_mean"], self.rows_of(rows, "pollock")[-1]["alpha_T_fixed_mean"], places=6)

    def test_aligned_frame_differs_from_fixed_when_tilted(self):
        arm2 = "s1_n10"
        fx.build_run(cp.arm_dir(self.out, arm2), nr=2, theta=0.3, pollock=True)
        recs = cp.done_records(self.out, arm2)
        self.NR_save, self.NR = self.NR, 2
        try:
            rows, _, n = an.alpha_series_rows(recs, self.U)
            for j in (self.J0, n - 1):
                self.check(rows, "flux", j, theta=0.3, tf=1.0)
            r = self.rows_of(rows, "flux")[-1]
            self.assertGreater(abs(r["alpha_T_fixed_mean"] - r["alpha_TAM_mean"]), 1e-4)
            self.assertGreater(abs(r["alpha_L_fixed_mean"] - r["alpha_L_aligned_mean"]), 1e-4)
        finally:
            self.NR = self.NR_save

    def edit_times(self, r, fn):
        R = os.path.join(cp.arm_dir(self.out, self.ARM), "realizations", "r_%04d" % r)
        for path in (os.path.join(R, "track.json"), os.path.join(R, "pollock", "pollock.json")):
            with open(path) as f:
                d = json.load(f)
            d["times"] = fn(d["times"])
            with open(path, "w") as f:
                json.dump(d, f)

    def test_common_prefix_truncation(self):
        self.edit_times(1, lambda t: t[:40])  # realization 1 has only 40 snapshots
        rows, n_real, n = an.alpha_series_rows(self.recs(), self.U)
        self.assertEqual((n_real, n), (self.NR, 40))
        for mode in an.SERIES_MODES:
            sel = self.rows_of(rows, mode)
            self.assertEqual([r["j"] for r in sel], list(range(40)))
            self.assertEqual({r["n_real"] for r in sel}, {self.NR})
            close(self, sel[-1]["t"], fx.TIMES[39], 1e-14)
            self.check(rows, mode, 39)  # constant set from snapshot 31 on: the one-sided end of realization 1 gives the same value
            self.check(rows, mode, self.J0)

    def test_times_mismatch_raises(self):
        from sf39 import io as sio

        def bump(t):
            t = list(t)
            t[10] *= 1.0 + 1e-6
            return t

        self.edit_times(1, bump)
        with self.assertRaises(sio.ReduceError) as cm:
            an.alpha_series_rows(self.recs(), self.U)
        self.assertIn("snapshot times differ", str(cm.exception))


class Amps(unittest.TestCase):
    """The `--amps` contingency: registry order, refusals, the budget requirement, the CLI."""

    def test_check_amps(self):
        self.assertEqual(an.check_amps(None), tuple(cp.AMPS))
        self.assertEqual(an.check_amps(("s4", "s1", "s225")), ("s1", "s225", "s4"))
        self.assertEqual(an.check_amps(("s025",)), ("s025",))
        for bad in ((), ("s9",), ("s1", "s1"), ("s1", "s9")):
            with self.assertRaises(an.Refused):
                an.check_amps(bad)

    def test_budget_requires_only_the_listed_amplitudes(self):
        with tempfile.TemporaryDirectory() as t:
            os.makedirs(os.path.join(t, "configs"))
            keys = dict(N_narrow=20, N_twin=20, N_hmax=20, N_R=20, N_A=20, N_run=20, par2_NR=20)
            with open(os.path.join(t, "configs", "budget.json"), "w") as f:
                json.dump({"s1": keys, "s225": keys, "s4": keys}, f)
            self.assertEqual(sorted(an.load_budget(t, ("s1", "s225", "s4"))), ["s1", "s225", "s4"])
            with self.assertRaises(an.Refused) as cm:
                an.load_budget(t)
            self.assertIn("s025.N_narrow=None", str(cm.exception))

    def test_cli_refuses_an_unknown_amplitude_before_reading_anything(self):
        with tempfile.TemporaryDirectory() as t:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = an.main(["--out-root", t, "--artifact", t, "--dest", os.path.join(t, "d"), "--amps", "s1,s9"])
            self.assertEqual(rc, 2)
            self.assertIn("ANALYSIS_REFUSED", err.getvalue())
            self.assertFalse(os.path.exists(os.path.join(t, "d")))

    def test_committed_par2_configs_have_the_nominal_variance(self):
        from pathlib import Path
        from sf39 import par2
        cfgs = Path(__file__).resolve().parents[3] / "configs"
        for amp in cp.AMPS:
            v = par2.cfg_values((cfgs / ("par2_s%s.yaml" % cp.PAR2_ID[amp])).read_text(encoding="utf-8"))
            self.assertEqual(v["sigma2"], an.SIGMA2_NOMINAL[amp], amp)
        self.assertEqual(fc.SIGMA2_NOMINAL, an.SIGMA2_NOMINAL)  # the fixtures' own table


class A4Pure(unittest.TestCase):
    """`a4_readings` / `a4_note` on synthetic A blocks (a ratio of means is a ratio of the CIs already computed)."""

    @staticmethod
    def blk(mp, m, f, tw, h):
        ci_ = lambda t: {"mean": t[0], "lo": t[1], "hi": t[2]}  # noqa: E731
        return {"margins": {"mean_production": mp, "m_h": m},
                "trend": {"D_f": {"ci": ci_(f)}, "D_tw": {"ci": ci_(tw)}, "d16_8": {"ci": ci_(h)}}}

    def test_fractions_and_trend_strings(self):
        r = an.a4_readings(self.blk(0.01, 0.0005, (-0.0003, -0.0005, -0.0001), (0.0002, -0.0004, 0.0008), (0.0002, 0.0001, 0.0003)))
        self.assertAlmostEqual(r["m_over_alpha"]["m"], 0.05)
        self.assertAlmostEqual(r["m_over_alpha"]["twin_band"], 2.0 ** 1.5 * 0.05)
        self.assertAlmostEqual(r["D_f_frac"]["mean"], -0.03)
        self.assertAlmostEqual(r["D_f_frac"]["lo"], -0.05)
        self.assertAlmostEqual(r["D_f_frac"]["hi"], -0.01)
        self.assertAlmostEqual(r["D_tw_frac"]["mean"], 0.0002 / (2.0 ** 1.5 * 0.01))  # D_tw / 2^1.5 over the production mean
        self.assertAlmostEqual(r["d16_8_frac"]["hi"], 0.03)
        # D_f and d16_8 exclude 0, D_tw does not
        self.assertEqual(r["trend_detected"], ["D_f: -3 % [-5, -1]", "d16_8: 2 % [1, 3]"])
        self.assertEqual(an.a4_note(r["trend_detected"]), "; refinement trend detected: D_f: -3 % [-5, -1]; d16_8: 2 % [1, 3]")

    def test_no_trend(self):
        r = an.a4_readings(self.blk(0.01, 0.0005, (0.0, -0.001, 0.001), (0.0, -0.001, 0.001), (0.0, -0.001, 0.001)))
        self.assertEqual(r["trend_detected"], [])
        self.assertEqual(an.a4_note(r["trend_detected"]), "; no refinement trend detected")

    def test_negative_and_zero_production_mean(self):
        r = an.a4_readings(self.blk(-0.01, 0.0005, (0.0003, 0.0001, 0.0005), (0.0, -0.001, 0.001), (0.0, -0.001, 0.001)))
        self.assertAlmostEqual(r["m_over_alpha"]["m"], 0.05)  # over |alpha|
        self.assertAlmostEqual(r["D_f_frac"]["mean"], -0.03)  # over the signed mean; the ends are reordered (lo <= hi)
        self.assertAlmostEqual(r["D_f_frac"]["lo"], -0.05)
        self.assertAlmostEqual(r["D_f_frac"]["hi"], -0.01)
        self.assertEqual(r["trend_detected"], ["D_f: -3 % [-5, -1]"])
        z = an.a4_readings(self.blk(0.0, 0.0005, (0.0003, 0.0001, 0.0005), (0.0, -0.001, 0.001), (0.0, -0.001, 0.001)))
        self.assertTrue(np.isnan(z["m_over_alpha"]["m"]) and np.isnan(z["D_f_frac"]["mean"]))
        self.assertEqual(len(z["trend_detected"]), 1)  # the CI still excludes 0; the percentages are NaN
        self.assertTrue(z["trend_detected"][0].startswith("D_f: nan %"))


class LoadFrozen(unittest.TestCase):
    """`load_frozen`: the original freeze merged with checkpoint/<amp>/tau_b_star_<amp>.json; refusals (exit 2 in `run`)."""

    @staticmethod
    def frz(amps, schema="sf39-tau_b_star-v1", n_pilot=20):
        return {"schema": schema, "n_pilot": n_pilot, "amplitudes": {a: {"tau_b_star": 1.0 + i} for i, a in enumerate(amps)}}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = self.tmp.name

    def put(self, rel, obj):
        p = os.path.join(self.out, "checkpoint", rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(obj, f)

    def test_merge(self):
        self.put("tau_b_star.json", self.frz(("s1", "s225", "s4")))
        self.put("s025/tau_b_star_s025.json", self.frz(("s025",)))
        f = an.load_frozen(self.out)
        self.assertEqual(sorted(f["amplitudes"]), ["s025", "s1", "s225", "s4"])
        self.assertEqual(f["amplitudes"]["s025"], {"tau_b_star": 1.0})
        self.assertEqual((f["schema"], f["n_pilot"]), ("sf39-tau_b_star-v1", 20))

    def test_refuses_schema_mismatch_in_either_file(self):
        self.put("tau_b_star.json", self.frz(("s1", "s225", "s4")))
        self.put("s025/tau_b_star_s025.json", self.frz(("s025",), schema="sf39-tau_b_star-v2"))
        with self.assertRaises(an.Refused) as cm:
            an.load_frozen(self.out)
        self.assertIn("unexpected schema 'sf39-tau_b_star-v2'", str(cm.exception))
        self.put("s025/tau_b_star_s025.json", self.frz(("s025",)))
        self.put("tau_b_star.json", self.frz(("s1", "s225", "s4"), schema="x"))
        with self.assertRaises(an.Refused):
            an.load_frozen(self.out)

    def test_refuses_an_amplitude_present_in_both(self):
        self.put("tau_b_star.json", self.frz(("s1", "s225", "s4")))
        self.put("s025/tau_b_star_s025.json", self.frz(("s025", "s1")))
        with self.assertRaises(an.Refused) as cm:
            an.load_frozen(self.out)
        self.assertIn("present in both", str(cm.exception))
        self.assertIn("s1", str(cm.exception))
        # the four-amplitude original file plus an s025 file is the same refusal
        self.put("tau_b_star.json", self.frz(("s025", "s1", "s225", "s4")))
        self.put("s025/tau_b_star_s025.json", self.frz(("s025",)))
        with self.assertRaises(an.Refused):
            an.load_frozen(self.out)

    def test_refuses_without_the_original_file_or_without_s025(self):
        with self.assertRaises(an.Refused) as cm:
            an.load_frozen(self.out)
        self.assertIn("no frozen file", str(cm.exception))
        self.put("tau_b_star.json", self.frz(("s1", "s225", "s4")))
        with self.assertRaises(an.Refused) as cm:
            an.load_frozen(self.out)
        self.assertIn("no frozen entry for s025", str(cm.exception))
        with mock.patch.object(cp, "AMPS", ("s1", "s225", "s4")):  # a three-amplitude registry needs no s025 file
            self.assertEqual(sorted(an.load_frozen(self.out)["amplitudes"]), ["s1", "s225", "s4"])

    def test_refuses_a_different_n_pilot(self):
        self.put("tau_b_star.json", self.frz(("s1", "s225", "s4")))
        self.put("s025/tau_b_star_s025.json", self.frz(("s025",), n_pilot=7))
        with self.assertRaises(an.Refused) as cm:
            an.load_frozen(self.out)
        self.assertIn("n_pilot 7 differs from 20", str(cm.exception))

    def test_amps_restrict_the_entries_required(self):
        self.put("tau_b_star.json", self.frz(("s1", "s225", "s4")))
        self.assertEqual(sorted(an.load_frozen(self.out, ("s1", "s225", "s4"))["amplitudes"]), ["s1", "s225", "s4"])
        self.assertEqual(sorted(an.load_frozen(self.out, ("s1",))["amplitudes"]), ["s1", "s225", "s4"])  # the file is what it is
        with self.assertRaises(an.Refused):
            an.load_frozen(self.out, ("s025", "s1"))
        self.put("s025/tau_b_star_s025.json", self.frz(("s025", "s4")))  # not read when s025 is not analysed
        self.assertEqual(sorted(an.load_frozen(self.out, ("s1", "s225", "s4"))["amplitudes"]), ["s1", "s225", "s4"])

    def test_run_exits_2_on_a_refusal(self):
        self.put("tau_b_star.json", self.frz(("s1", "s225", "s4")))
        self.put("s025/tau_b_star_s025.json", self.frz(("s025", "s4")))
        err = io.StringIO()
        rc = an.run(self.out, self.out, os.path.join(self.out, "dest"), err=err)
        self.assertEqual(rc, 2)
        self.assertIn("ANALYSIS_REFUSED", err.getvalue())


if __name__ == "__main__":
    unittest.main()
