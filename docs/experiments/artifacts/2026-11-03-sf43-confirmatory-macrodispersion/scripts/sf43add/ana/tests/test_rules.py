"""The pre-registered rules on in-memory views (no files): every J1 / J3 / J9 class, the gates, the precedence and the J8 trigger."""
import json
import unittest
import zlib

import numpy as np

import sf43ana  # noqa: F401

from sf43add.ana import rules, trigger
from sf43add.ana.data import LP, TP_FAR, TP_MAIN, View, load_addendum
from sf43add.ana.tests import synth

ADD = load_addendum(synth.ADD_PATH)
SEEDS = np.arange(434001, 434061)
REPRO_OK = {"evaluated": True, "passed": True}


def noise(label, n, sd=0.004):
    return 1 + sd * np.random.default_rng(zlib.crc32(label.encode())).standard_normal(n)


def par2(label, factor, seeds=SEEDS, complete=True):
    n = seeds.size
    common = 1 + 0.1 * np.random.default_rng(99).standard_normal(n)
    v = 8.0 * common * factor * noise(label, n)
    nl = np.full(n, 10000.0)
    if not complete:
        nl[: int(0.3 * n)] -= 50  # 30 % of the realizations incomplete, mean completeness 0.9985
    return View(label, seeds, {LP: v}, {"n_last": nl, "n_traced": np.full(n, 10000.0)})


def j1_views(h3=1.0, d=1.0, dh3=1.0, h2=1.0, pol=1.16, complete=True, base10=True):
    v = {"s4_g5p": par2("s4_g5p", 1.0), "s4_g5pol": par2("pol", pol), "j1_H3": par2("H3", h3, complete=complete), "j1_D": par2("D", d),
         "j1_DH3": par2("DH3", dh3), "j1_H2": par2("H2", h2)}
    if base10:
        v["j1_base10"] = par2("s4_g5p", 1.0, SEEDS[:10])
    return v


class J1(unittest.TestCase):
    def run_j1(self, repro=REPRO_OK, **kw):
        return rules.evaluate_j1(ADD, j1_views(**kw), repro, "n60")

    def test_horizonte_dominante(self):
        r = self.run_j1(h3=1.12, d=1.0, dh3=1.13, pol=1.14)
        self.assertEqual((r["status"], r["class"]), ("classified", "horizonte dominante"))
        self.assertAlmostEqual(r["quantities"]["r_H"]["ratio"], 1.12, delta=0.01)
        self.assertTrue(all(c["truth"] for c in r["conditions"]["horizonte dominante"]))

    def test_paso_dominante(self):
        r = self.run_j1(h3=1.0, d=1.08, dh3=1.08, pol=1.16)
        self.assertEqual(r["class"], "paso dominante")
        self.assertFalse(r["conditions"]["horizonte dominante"][0]["truth"])

    def test_mixto(self):
        r = self.run_j1(h3=1.06, d=1.06, dh3=1.10, pol=1.12)
        self.assertEqual(r["class"], "mixto")
        self.assertFalse(r["conditions"]["paso dominante"][1]["truth"])  # r_H < 1.05 fails: that is why class 2 does not match

    def test_abierto_fall_through_with_and_without_the_override_condition(self):
        r = self.run_j1(h3=1.0, d=1.0, dh3=1.0, pol=1.16)  # nothing matched; q_DH3 lower limit above 1.10 holds but changes nothing
        self.assertEqual((r["class"], r["class_before_override"]), ("abierto", "abierto"))
        self.assertTrue(r["override"]["holds"])
        self.assertFalse(r["override"]["applied"])
        r = self.run_j1(h3=1.0, d=1.0, dh3=1.0, pol=1.0)
        self.assertEqual((r["class"], r["class_before_override"]), ("abierto", "abierto"))
        self.assertFalse(r["override"]["holds"])

    def test_amendment_a1_override_over_class_2(self):
        r = self.run_j1(h3=1.0, d=1.08, dh3=1.0, pol=1.16)  # class 2 matches (r_D 1.08, r_H 1.0) but ci(q_DH3).lo = 1.16 > 1.10
        self.assertTrue(all(c["truth"] for c in r["conditions"]["paso dominante"]))
        self.assertEqual((r["status"], r["class"], r["class_before_override"]), ("classified", "abierto", "paso dominante"))
        self.assertTrue(r["override"]["holds"] and r["override"]["applied"])
        self.assertEqual(r["override"]["conditions"][0]["lhs"], "ci(q_DH3).lo")
        self.assertGreater(r["override"]["conditions"][0]["value"], 1.10)
        self.assertIn("Amendment A1", r["reason"])
        self.assertIn("paso dominante", r["reason"])
        self.assertIn("Amendment A1", r["override"]["note"])

    def test_amendment_a1_override_over_class_1(self):
        r = self.run_j1(h3=1.12, d=1.0, dh3=1.0, pol=1.14)  # class 1 matches (q_H3 1.02) but q_DH3 = 1.14
        self.assertTrue(all(c["truth"] for c in r["conditions"]["horizonte dominante"]))
        self.assertEqual((r["class"], r["class_before_override"]), ("abierto", "horizonte dominante"))
        self.assertTrue(r["override"]["applied"])

    def test_no_override_when_the_lower_limit_is_not_above_1_10(self):
        r = self.run_j1(h3=1.12, d=1.0, dh3=1.03, pol=1.12)  # class 1; q_DH3 = 1.087, ci lower limit below 1.10
        self.assertLessEqual(r["quantities"]["q_DH3"]["ci90"][0], 1.10)
        self.assertEqual((r["class"], r["class_before_override"]), ("horizonte dominante", "horizonte dominante"))
        self.assertEqual((r["override"]["holds"], r["override"]["applied"]), (False, False))
        self.assertNotIn("Amendment A1", r["reason"])
        r = self.run_j1(h3=1.0, d=1.08, dh3=1.08, pol=1.16)  # class 2; q_DH3 = 1.074
        self.assertEqual((r["class"], r["class_before_override"], r["override"]["applied"]), ("paso dominante", "paso dominante", False))

    def test_override_is_read_from_the_json_not_hard_coded(self):
        add = json.loads(json.dumps(ADD))
        add["j1"]["classes"][-1]["override_if"] = [["ci(q_DH3).lo", ">", 1.30]]  # a threshold the data do not reach
        r = rules.evaluate_j1(add, j1_views(h3=1.0, d=1.08, dh3=1.0, pol=1.16), REPRO_OK, "n60")
        self.assertEqual((r["class"], r["override"]["applied"]), ("paso dominante", False))
        del add["j1"]["classes"][-1]["override_if"]  # no override clause at all: first match stands
        r = rules.evaluate_j1(add, j1_views(h3=1.0, d=1.08, dh3=1.0, pol=1.16), REPRO_OK, "n60")
        self.assertEqual((r["class"], r["override"]["holds"], r["override"]["conditions"]), ("paso dominante", False, []))

    def test_the_override_is_only_in_the_classification_path(self):
        kw = dict(h3=1.12, d=1.0, dh3=1.0, pol=1.14)  # would be overridden
        r = self.run_j1({"evaluated": True, "passed": False}, **kw)
        self.assertEqual((r["class"], r["class_before_override"], r["override"]["applied"]), (None, None, False))
        r = self.run_j1(complete=False, **kw)  # manipulation failure: classes 1-3 are not assigned, nothing to override
        self.assertEqual((r["class"], r["class_before_override"], r["override"]["applied"]), ("abierto (horizonte no saturado)", None, False))
        v = j1_views(**kw)
        del v["j1_DH3"]  # q_DH3 not evaluable: status "not evaluable", the override condition is undetermined
        r = rules.evaluate_j1(ADD, v, REPRO_OK, "n60")
        self.assertEqual((r["status"], r["class"], r["override"]["holds"]), ("not evaluable", None, False))
        self.assertIsNone(r["override"]["conditions"][0]["truth"])

    def test_first_match_wins_when_two_classes_hold(self):
        r = self.run_j1(h3=1.12, d=1.0, dh3=1.12, pol=1.12)
        self.assertEqual(r["class"], "horizonte dominante")
        self.assertTrue(all(c["truth"] for c in r["conditions"]["horizonte dominante"]))

    def test_conditions_are_closed_at_their_thresholds(self):
        Q = {"r_H": {"evaluable": True, "ratio": 1.10, "ci90": [1.0, 1.2]}, "q": {"evaluable": True, "ratio": 1.0, "ci90": [1 / 1.10, 1.10]}}
        self.assertTrue(rules.eval_cond(Q, "r_H", ">=", 1.10)["truth"])
        self.assertFalse(rules.eval_cond(Q, "r_H", ">", 1.10)["truth"])
        self.assertTrue(rules.eval_cond(Q, "ci(q)", "within", [1 / 1.10, 1.10])["truth"])
        self.assertFalse(rules.eval_cond(Q, "ci(r_H).lo", "<", 1.0)["truth"])
        self.assertAlmostEqual(rules.eval_cond(Q, "abs(r_H-1)", "<=", 0.2)["value"], 0.1)
        self.assertIsNone(rules.eval_cond(Q, "r_X", ">=", 1.0)["truth"])  # an unknown / not evaluable quantity is "undetermined"

    def test_manipulation_failure_gives_abierto_horizonte_no_saturado(self):
        r = self.run_j1(h3=1.12, d=1.0, dh3=1.12, pol=1.14, complete=False)
        self.assertFalse(r["manipulation"]["passed"])
        self.assertEqual((r["status"], r["class"]), ("classified", "abierto (horizonte no saturado)"))
        self.assertLess(r["manipulation"]["mean"], 0.999)

    def test_manipulation_thresholds(self):
        spec = ADD["j1"]["manipulation"]
        comp = {"evaluable": True, "n": 60, "mean": 0.999, "fraction_complete": 0.90}
        self.assertTrue(rules.manipulation(spec, comp)["passed"])
        self.assertFalse(rules.manipulation(spec, dict(comp, mean=0.9989))["passed"])
        self.assertFalse(rules.manipulation(spec, dict(comp, fraction_complete=0.8999))["passed"])

    def test_reproduction_gate(self):
        for rp, why in (({"evaluated": True, "passed": False}, "failed"), ({"evaluated": False}, "not evaluated"), (None, "not evaluated")):
            r = self.run_j1(rp, h3=1.12, d=1.0, dh3=1.13, pol=1.14)
            self.assertEqual((r["status"], r["class"]), ("not classified", None))
            self.assertIn(why, r["reason"])
            self.assertTrue(r["quantities"]["r_H"]["evaluable"])  # everything else is still computed and reported

    def test_missing_arm_is_not_evaluable_not_a_crash(self):
        v = j1_views(h3=1.0)
        del v["j1_D"]
        r = rules.evaluate_j1(ADD, v, REPRO_OK, "n60")
        self.assertEqual((r["status"], r["class"]), ("not evaluable", None))
        self.assertFalse(r["quantities"]["r_D"]["evaluable"])
        self.assertTrue(r["quantities"]["r_H"]["evaluable"])
        self.assertIsNone(r["conditions"]["paso dominante"][0]["truth"])

    def test_too_few_common_seeds(self):
        v = j1_views()
        v["j1_D"] = v["j1_D"].where(434001, 434002)
        self.assertFalse(rules.evaluate_j1(ADD, v, REPRO_OK, "n60")["quantities"]["r_D"]["evaluable"])

    def test_representativity_and_completeness_are_reported(self):
        r = self.run_j1(pol=1.164)
        rep = r["representativity"]
        self.assertAlmostEqual(rep["result"]["ratio"], 1.164, delta=0.01)
        self.assertTrue(rep["frozen_ratio_in_bootstrap_t_ci"])
        c = r["completeness"]["j1_H3"]
        self.assertEqual((c["n"], c["fraction_complete"], c["mean"]), (60, 1.0, 1.0))
        self.assertEqual(sorted(c["quantiles"]), sorted(str(q) for q in rules.QUANTILES))

    def test_j8_trigger_inputs_list_the_thresholds_inside_an_interval(self):
        Q = {"r_H": {"evaluable": True, "ci90": [1.04, 1.06]}, "r_D": {"evaluable": True, "ci90": [0.99, 1.0]},
             "q_H3": {"evaluable": True, "ci90": [0.95, 1.2]}, "q_DH3": {"evaluable": False, "reason": "arm not available"}}
        ins = rules.j8_inputs(ADD, {"quantities": Q})["quantities"]
        self.assertEqual({k: v["contained"] for k, v in ins.items()}, {"r_H": [1.05], "r_D": [], "q_H3": [1.1], "q_DH3": []})
        self.assertTrue(rules.j8_inputs(ADD, {"quantities": Q})["any_contained"])
        r = self.run_j1(h3=1.0, d=1.0, dh3=1.0, pol=1.0)
        ins = rules.j8_inputs(ADD, r)
        self.assertEqual({k: v["contained"] for k, v in ins["quantities"].items()}, {"r_H": [], "r_D": [], "q_H3": [], "q_DH3": []})
        self.assertFalse(ins["any_contained"])
        self.assertTrue(ins["quantities"]["r_H"]["thresholds"] == [1.03, 1.05, 1.1])


def tp_view(label, far_over_main=1.02, main=1.0, n=100, seeds=None):
    seeds = np.arange(436001, 436001 + n) if seeds is None else seeds
    m = seeds.size
    common = 1 + 0.1 * np.random.default_rng(5).standard_normal(m)
    base = 0.006 * common * main
    return View(label, seeds, {TP_MAIN: base * noise(label + "m", m), TP_FAR: base * far_over_main * noise(label + "f", m)}, {})


def j3_views(j3=(1.0, 1.02), j9=(1.02,), n9=150, s1=(1.0, 1.02), present=True):
    v = {"s1_dist": tp_view("s1", s1[1], s1[0])}
    if present:
        v["j3_l5"] = tp_view("j3", j3[1], j3[0])
    v["j9_w48"] = tp_view("j9", j9[0], 1.0, n9, np.arange(438001, 438001 + n9))
    return v


class J3(unittest.TestCase):
    def test_equivalent(self):
        r = rules.evaluate_j3(ADD, j3_views())
        self.assertEqual(r["class"], "lambda/h = 5 equivalente")
        self.assertTrue(all(c["truth"] for c in r["conditions"]))
        self.assertEqual(len(r["conditions"]), 3)

    def test_not_equivalent_when_a_level_differs(self):
        r = rules.evaluate_j3(ADD, j3_views(j3=(1.05, 1.071)))
        self.assertEqual(r["class"], "lambda/h = 5 no equivalente")
        self.assertFalse(r["conditions"][0]["truth"])  # m_main ~ 1.05 is outside [0.97, 1.03]

    def test_not_equivalent_when_only_the_ratio_of_ratios_is_outside(self):
        r = rules.evaluate_j3(ADD, j3_views(j3=(1.0, 1.045)))  # m_main 1.00 and m_far 1.025 are within their margins, m_far/m_main 1.025 is not
        self.assertEqual([c["truth"] for c in r["conditions"]], [True, True, False])
        self.assertEqual(r["class"], "lambda/h = 5 no equivalente")

    def test_missing_arm(self):
        r = rules.evaluate_j3(ADD, j3_views(present=False))
        self.assertEqual(r["class"], "lambda/h = 5 no equivalente")
        self.assertIn("missing", r["reason"])
        self.assertFalse(r["quantities"]["m_main"]["evaluable"])


class J9(unittest.TestCase):
    def run_j9(self, **kw):
        v = j3_views(**kw)
        return rules.evaluate_j9(ADD, v, rules.evaluate_j3(ADD, v))

    def test_meseta_robusta(self):
        r = self.run_j9(j9=(1.02,))
        self.assertEqual(r["class"], "meseta robusta al ancho")
        self.assertIsNone(r["j3_qualification"])
        self.assertAlmostEqual(r["quantities"]["F48"]["ratio"], 1.02, delta=0.01)
        self.assertAlmostEqual(r["quantities"]["F48/F5"]["ratio"], 1.0, delta=0.01)
        self.assertAlmostEqual(r["quantities"]["F24"]["ratio"], 1.02, delta=0.01)
        self.assertEqual(r["F24_frozen"]["ratio"], 1.0208991755164611)
        self.assertEqual(sorted(r["descriptive_levels"]), ["W48/W24 far", "W48/W24 main"])

    def test_decae_con_el_ancho(self):
        self.assertEqual(self.run_j9(j9=(0.85,))["class"], "decae con el ancho")
        self.assertEqual(self.run_j9(j9=(0.89,))["class"], "decae con el ancho")  # IC90(F48) pokes below 0.90, so class 1 fails and hi < 0.95 holds

    def test_first_match_order(self):
        r = self.run_j9(j9=(0.93,))  # hi < 0.95 holds too, but class 1 (within [0.90, 1.10] twice) comes first
        self.assertEqual(r["class"], "meseta robusta al ancho")
        self.assertTrue(r["conditions"]["decae con el ancho"][0]["truth"])

    def test_inconcluso(self):
        self.assertEqual(self.run_j9(j9=(1.12,))["class"], "inconcluso")
        r = self.run_j9(j9=(1.09,), j3=(1.0, 0.95))  # F48 within [0.90, 1.10] but F48/F5 ~ 1.15 is not, and hi >= 0.95
        self.assertEqual(r["class"], "inconcluso")
        self.assertTrue(r["conditions"]["meseta robusta al ancho"][0]["truth"])
        self.assertFalse(r["conditions"]["meseta robusta al ancho"][1]["truth"])

    def test_inconcluso_por_N_takes_precedence(self):
        r = self.run_j9(j9=(1.02,), n9=99)  # would be meseta robusta with 150 realizations
        self.assertEqual((r["class"], r["n_completed"]), ("inconcluso por N", 99))
        self.assertEqual(self.run_j9(j9=(1.02,), n9=100)["class"], "meseta robusta al ancho")

    def test_j3_qualification_is_attached_not_a_class_change(self):
        r = self.run_j9(j3=(1.07, 1.09), j9=(1.02,))
        self.assertIn("caveat", r["j3_qualification"])
        self.assertFalse(r["j3_equivalent"])

    def test_missing_j3_gives_inconcluso_not_a_crash(self):
        r = self.run_j9(present=False, j9=(1.02,))
        self.assertEqual(r["class"], "inconcluso")  # F48/F5 is not evaluable, so the plateau cannot be established
        self.assertFalse(r["quantities"]["F48/F5"]["evaluable"])
        self.assertIsNotNone(r["j3_qualification"])
        self.assertEqual(self.run_j9(present=False, j9=(0.85,))["class"], "decae con el ancho")  # that class needs F48 alone


class J8Trigger(unittest.TestCase):
    def summary(self, passed=True, contained=True):
        return {"config": {"budget_gpu_h_cap": 8.0, "hard_end_utc": "2026-11-07T03:00:00Z"}, "reproduction": {"passed": passed},
                "j8_trigger_inputs": {"quantities": {"r_H": {"contained": [1.05] if contained else []}, "r_D": {"contained": []}}}}

    def test_run(self):
        run, c = trigger.evaluate(self.summary(), 6.0, 0.9, "2026-11-06T20:00:00Z")
        self.assertTrue(run)
        self.assertEqual([x[0] for x in c], list("abcd"))

    def test_each_condition_can_skip(self):
        for kw, args, label in ((dict(passed=False), (6.0, 0.9, "2026-11-06T20:00:00Z"), "a"), (dict(contained=False), (6.0, 0.9, "2026-11-06T20:00:00Z"), "b"),
                                (dict(), (7.0, 0.9, "2026-11-06T20:00:00Z"), "c"), (dict(), (6.0, 0.9, "2026-11-07T03:00:01Z"), "d")):
            run, c = trigger.evaluate(self.summary(**kw), *args)
            self.assertFalse(run, label)
            self.assertEqual([x[0] for x in c if not x[1]], [label])

    def test_cost_boundary_and_hard_end_are_inclusive(self):
        self.assertTrue(trigger.evaluate(self.summary(), 6.8, 1.0, "2026-11-07T03:00:00Z")[0])  # 6.8 + 1.2 = 8.0 <= 8
        self.assertFalse(trigger.evaluate(self.summary(), 6.81, 1.0, "2026-11-07T03:00:00Z")[0])
        self.assertTrue(trigger.evaluate(self.summary(), 1.0, 1.0, "2026-11-07T04:00:00+01:00")[0])

    def test_unevaluated_j1_skips(self):
        s = self.summary()
        del s["j8_trigger_inputs"]
        s["reproduction"] = {"evaluated": False}
        self.assertFalse(trigger.evaluate(s, 1.0, 1.0, "2026-11-06T00:00:00Z")[0])


if __name__ == "__main__":
    unittest.main()
