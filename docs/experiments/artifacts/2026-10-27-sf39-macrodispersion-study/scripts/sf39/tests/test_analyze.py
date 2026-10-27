"""End-to-end tests of sf39.analyze on a fake complete campaign (fixtures_analysis; known answers from the closed forms of fixtures_runner).

Run (about two minutes): `cd A/scripts && python3 -m unittest discover -s . -t . -p 'test_analyze*.py'`.
The F10 reference/width counts (30 in the campaign) are patched to 9 (`power.REF_FLOOR`, `power.WIDTH_ARMS`) to keep the fixture small.
"""
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
                       mock.patch.dict(power.WIDTH_ARMS, {"s4_w24": fa.REF_FLOOR, "s4_w12": fa.REF_FLOOR})]
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
        with open(os.path.join(cls.out, "checkpoint", "tau_b_star.json")) as f:
            cls.frz = json.load(f)

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
                                       ("s4_w12", [7201 + r for r in range(9)], [fa.WIDTH_G0 + 50 + r for r in range(9)], "s4_tw"),
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
        self.assertEqual([r["verdict"] for r in lab], ["physical (positive)"] + ["unresolved (grid-limited, lower bound)"] * 2)
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

    def test_window_error_is_a_failure_not_a_clip(self):
        frz = json.loads(json.dumps(self.frz))
        for amp in cp.AMPS:
            frz["amplitudes"][amp]["tau_b_star"] *= 100.0  # t_hi far beyond the last snapshot
        C = an.Ctx(self.out, self.art, os.path.join(self.tmp.name, "d_win"), frz, None, N, False)
        C.eff = an.effective_counts(C, B)
        from sf39 import io as sio
        with self.assertRaises(sio.WindowError):
            an.analyze_amp(C, "s1")


if __name__ == "__main__":
    unittest.main()
