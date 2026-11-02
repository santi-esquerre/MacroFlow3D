import json
import math
import os
import tempfile
import unittest

import numpy as np
from scipy import stats

from mcstat import coverage as cv
from mcstat import covdist, covproc


def small_cells():
    """Fast cells of three procedure families (plain, bootstrap-t, shared-draw contrast) for the engine tests."""
    return [
        cv.make_cell("sampling", "sampling:student", "", "normal", 30, {"method": "student"}, target=2, m=100, chunk=200),
        cv.make_cell("sampling", "sampling:boott", "", "stress", 20, {"method": "boott", "B": 1000}, target=0.3, m=1, chunk=40),
        cv.make_cell("contrast", "contrast:unpaired", "", "stress-", 20, {"kind": "unpaired", "kappa": 5.0, "B": 1000}, target=0.3,
                     m=10, chunk=50, share=True),
    ]


def counts_of(summary):
    return {c["id"]: {r: [(e["misses"], e["R"]) for e in es] for r, es in c["rows"].items()} for c in summary["cells"]}


class Tails(unittest.TestCase):
    def test_holm_sequence_and_derived_tails(self):
        a = covproc.holm_levels()
        np.testing.assert_allclose(a, [0.05 / 7, 0.05 / 6, 0.05 / 5, 0.05 / 4, 0.05 / 3, 0.05 / 2, 0.05], rtol=1e-14)
        s, c = covproc.sampling_tails(), covproc.component_tails()
        self.assertAlmostEqual(s.min(), 3.5714e-3, delta=1e-7)  # a / 2 at the worst Holm level
        self.assertAlmostEqual(c.min(), 8.9286e-4, delta=1e-8)  # a / (2 K), K = 4
        self.assertTrue(np.any(np.abs(c - 0.05 / 7 / 6) < 1e-15))  # K = 3 worst, 1.19e-3
        self.assertEqual(len(c), 13)  # 14 (K = 3, 4) values, one shared: a_5 / 6 = a_4 / 8
        self.assertAlmostEqual(float(covproc.holm_levels().min()), 7.1429e-3, delta=1e-7)  # TOST worst
        self.assertEqual(covproc.boot_size(c.min() / 2.0, 10), 23000)  # E_h resolves t / 2 with 10 resamples beyond the quantile

    def test_replicates_per_cell_follow_the_175_rule(self):
        cells = cv.profile_full()
        self.assertGreater(len(cells), 300)
        self.assertEqual(len({c["id"] for c in cells}), len(cells))
        for c in cells:
            self.assertEqual(c["R"] % c["m"], 0)
            self.assertEqual(c["chunk"] % c["m"], 0)
            self.assertGreaterEqual(c["R"] * min(c["tails"]), cv.TARGET - 1e-9, c["id"])
        by = {(c["proc"], c["n"]): c for c in cells}
        self.assertEqual(by[("sampling", 30)]["R"], 49000)  # 175 / 3.571e-3
        self.assertEqual(by[("contrast", 100)]["R"], 196000)  # 175 / 8.929e-4
        self.assertTrue(24500 <= by[("tost", 30)]["R"] < 24500 + 1000)  # 175 / 7.143e-3, rounded up to the batch size
        self.assertEqual({c["n"] for c in cells if c["proc"] == "sampling"}, set(cv.SAMPLING_NS))
        self.assertEqual({c["n"] for c in cells if c["proc"] == "tost"}, set(cv.GATE5_NS))
        self.assertTrue(all(c["report_only"] for c in cells if c["proc"] == "bca"))


class Statistics(unittest.TestCase):
    def test_clopper_pearson_known_values(self):
        for R in (3, 59, 1000):
            self.assertAlmostEqual(cv.cp_upper(0, R), 1.0 - 0.05 ** (1.0 / R), places=14)  # the "rule of three" family
        self.assertAlmostEqual(cv.cp_upper(0, 3), 0.6316, places=4)
        self.assertAlmostEqual(cv.cp_upper(1, 10), 0.3942, places=4)  # one-sided 95 % limit of 1 out of 10
        self.assertEqual(cv.cp_upper(10, 10), 1.0)
        for k, R in ((3, 50), (175, 49000), (0, 24500)):
            p = cv.cp_upper(k, R)
            self.assertAlmostEqual(float(stats.binom.cdf(k, R, p)), 0.05, places=9)  # defining equation
        self.assertGreater(cv.cp_upper(175, 49000) / (175 / 49000), 1.10)  # an exactly nominal procedure: about 1.13
        self.assertLess(cv.cp_upper(175, 49000) / (175 / 49000), 1.25)

    def test_design_effect(self):
        rng = np.random.default_rng(0)
        iid = rng.binomial(25, 0.01, size=(4000, 3))
        self.assertLess(cv.design_effect(iid, 25).max(), 1.15)
        clustered = 25 * rng.binomial(1, 0.01, size=(4000, 3))  # all or none of a batch fails
        self.assertGreater(cv.design_effect(clustered, 25).min(), 15.0)
        np.testing.assert_array_equal(cv.design_effect(iid, 1), np.ones(3))
        np.testing.assert_array_equal(cv.design_effect(np.zeros((10, 3), dtype=int), 25), np.ones(3))

    def test_verdict_rule_on_synthetic_counts(self):
        cell = {"tails": [0.05 / 14], "m": 1, "report_only": False}
        agg = lambda k: {"R": 49000, "rows": {"r": np.array([k])}, "sumsq": {}, "nbatch": 0}
        e175 = cv.evaluate_cell(cell, agg(175))["r"][0]
        e210 = cv.evaluate_cell(cell, agg(210))["r"][0]
        self.assertTrue(e175["pass"] and e175["powered"])
        self.assertAlmostEqual(e175["ucl_over_tail"], 1.13, delta=0.01)
        self.assertFalse(e210["pass"])
        self.assertGreater(e210["ucl_over_tail"], 1.25)
        rep = cv.evaluate_cell(dict(cell, report_only=True), agg(210))["r"][0]
        self.assertIsNone(rep["pass"])  # BCa: reported, never a verdict
        batches = 25 * (np.arange(1960) < 7).astype(int)[:, None]  # 7 of 1960 batches fail completely: 175 misses, all clustered
        shared = cv.evaluate_cell(dict(cell, m=25), {"R": 49000, "rows": {"r": batches.sum(axis=0)}, "sumsq": {"r": (batches ** 2).sum(axis=0)},
                                                    "nbatch": 1960})["r"][0]
        self.assertGreater(shared["design_effect"], 1.0)  # clustered misses widen the interval
        self.assertGreater(shared["ucl"], shared["ucl_naive"])

    def test_cost_projection(self):
        cells = small_cells()
        spr = {c["id"]: 0.001 for c in cells}
        pr = cv.project(cells, spr, workers=80, host_factor=1.5, efficiency=0.9)
        core = sum(c["R"] for c in cells) * 0.001
        self.assertAlmostEqual(pr["core_seconds_local"], core, places=9)
        self.assertAlmostEqual(pr["wall_hours_host"], core * 1.5 / (80 * 0.9) / 3600.0, places=12)


class NominalFailProbability(unittest.TestCase):
    """False-fail probability of the G5.3 criterion for an exactly nominal procedure (replaces the former '1e-3' statement)."""

    def test_threshold_is_the_smallest_failing_miss_count(self):
        for R, tau in ((49000, 1 / 280.0), (196000, 0.05 / 56.0), (5000, 0.05)):
            k = cv.fail_threshold(R, tau)
            self.assertGreater(cv.cp_upper(k, R), cv.CRITERION * tau)
            self.assertLessEqual(cv.cp_upper(k - 1, R), cv.CRITERION * tau)

    def test_reference_values_sampling_tails_at_49000(self):
        r = cv.nominal_fail_probability(49000, covproc.sampling_tails())
        self.assertEqual(r["kstar"][0], 195)  # against 175 expected misses at the smallest tail 1/280
        np.testing.assert_allclose(r["p_tail"][:5], [0.0717, 0.0459, 0.0203, 0.0064, 0.0008], atol=6e-5)
        self.assertAlmostEqual(r["p_any"], 0.0819, delta=1e-3)  # exact; the fixed-seed simulation below agrees within its error
        self.assertGreaterEqual(r["p_any"], max(r["p_tail"]))
        self.assertLessEqual(r["p_any"], sum(r["p_tail"]))
        self.assertAlmostEqual(1.0 - (1.0 - r["p_any"]) ** 3, 0.2261, delta=2e-3)
        c = cv.nominal_fail_probability(196000, covproc.component_tails())
        self.assertEqual(c["kstar"][0], 195)
        self.assertAlmostEqual(c["p_any"], 0.084, delta=2e-3)

    def test_dynamic_programme_against_multinomial_simulation(self):
        R, tails = 49000, covproc.sampling_tails()
        r = cv.nominal_fail_probability(R, tails)
        S = 200000
        edges = np.concatenate([[0.0], r["tails"]])
        pv = np.diff(np.concatenate([edges, [1.0]]))
        N = np.cumsum(np.random.default_rng(12345).multinomial(R, pv, size=S)[:, :-1], axis=1)  # nested counts at the sorted tails
        sim = float((N >= np.array(r["kstar"])).any(axis=1).mean())
        se = math.sqrt(sim * (1.0 - sim) / S)
        self.assertLess(abs(sim - r["p_any"]), 4.0 * se, (sim, r["p_any"], se))

    def test_dynamic_programme_against_exhaustive_enumeration(self):
        # tiny case, every multinomial outcome: R = 6 replicates, tails 0.2 and 0.5, a lenient criterion so that both can fail
        R, tails, conf, crit = 6, [0.2, 0.5], 0.95, 1.0
        ks = [cv.fail_threshold(R, t, conf, crit) for t in tails]
        pv = [0.2, 0.3, 0.5]
        total = 0.0
        for a in range(R + 1):
            for b in range(R + 1 - a):
                c = R - a - b
                if a >= ks[0] or a + b >= ks[1]:
                    total += math.factorial(R) / (math.factorial(a) * math.factorial(b) * math.factorial(c)) * pv[0] ** a * pv[1] ** b * pv[2] ** c
        r = cv.nominal_fail_probability(R, tails, conf, crit, target=0.0, window_sd=50.0)
        self.assertEqual(r["kstar"], ks)
        self.assertAlmostEqual(r["p_any"], total, places=12)

    def test_untested_tails_are_skipped_and_the_table_lists_distinct_plan_sets(self):
        self.assertEqual(cv.nominal_fail_probability(100, [0.05])["tails"], [])  # 5 expected misses: not judged
        cells = [cv.make_cell("sampling", "sampling:student", "", d, n, {"method": "student"}, m=1000) for d in ("normal", "stress") for n in (20, 30)]
        cells.append(cv.make_cell("bca", "bca:mean", "", "normal", 20, {"B": 199}, m=10, report_only=True))
        tab = cv.nominal_fail_table(cells)
        self.assertEqual(len(tab), 1)  # one distinct (R, tails); the BCa cell is not judged
        self.assertEqual((tab[0]["R"], tab[0]["cells"]), (49000, 4))


class IncompleteGroups(unittest.TestCase):
    def synthetic(self, dists):
        e = lambda: {"tail": 0.01, "misses": 100, "R": 17500, "rate": 0.0057, "ucl_over_tail": 0.9, "rate_over_tail": 0.6, "pass": True}
        return [{"label": "p", "proc": "p", "scen": "", "dist": d, "n": 30, "report_only": False, "rows": {"r": [e()]}} for d in dists]

    def plan(self, dists, n=30):
        return [{"label": "p", "scen": "", "dist": d, "n": n} for d in dists]

    def test_group_with_a_missing_distribution_is_not_qualified(self):
        full = cv.make_verdicts(self.synthetic(["normal", "stress"]), self.plan(["normal", "stress"]))
        self.assertTrue(full[0]["qualified"])
        part = cv.make_verdicts(self.synthetic(["normal"]), self.plan(["normal", "stress"]))
        self.assertIsNone(part[0]["qualified"])
        self.assertEqual(part[0]["incomplete_dists"], ["stress"])
        self.assertTrue(cv.make_verdicts(self.synthetic(["normal"]))[0]["qualified"])  # without a plan: the former behaviour
        bad = self.synthetic(["normal"])
        bad[0]["rows"]["r"][0]["pass"] = False
        bad[0]["rows"]["r"][0]["tail"] = 0.01
        self.assertIs(cv.make_verdicts(bad, self.plan(["normal", "stress"]))[0]["qualified"], False)  # a failure is final

    def test_summary_marks_unfinished_groups(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cells = [cv.make_cell("sampling", "sampling:student", "", d, 30, {"method": "student"}, m=1000, chunk=25000) for d in ("normal", "stress")]
        cells.append(cv.make_cell("sampling", "sampling:student", "", "normal", 50, {"method": "student"}, m=1000, chunk=25000))
        out = os.path.join(tmp.name, "o")
        cv.init_plan(out, cells, 11, "test")
        cv.run(out, 1, max_chunks=2 + 1, log=lambda s: None)  # n=30 normal (2 chunks) + the first chunk of n=30 stress
        s = cv.summarize(out)
        self.assertEqual(s["cells_complete"], 1)
        self.assertEqual({tuple(g) for g in s["incomplete_groups"]}, {("sampling:student", "", 30), ("sampling:student", "", 50)})
        self.assertTrue(all(x["incomplete_dists"] == ["stress"] and x["qualified"] is not True for x in s["verdicts"]))
        with open(os.path.join(out, "summary.md")) as f:
            md = f.read()
        matrix = md.split("## Qualification matrix")[1].split("## Verdict list")[0]
        self.assertNotIn("ALL", matrix.split("Entry:")[1].split("\n", 1)[1])  # the legend mentions ALL; the table rows must not
        self.assertIn("inc", matrix)
        self.assertIn("INCOMPLETE", md.split("## Verdict list")[1].split("## ")[0])
        self.assertIn("## False-fail probability of an exactly nominal procedure", md)


class NominalCoverage(unittest.TestCase):
    """Small-scale runs reproduce the nominal rates within Monte Carlo error where the procedure is exact or close to it."""

    def run_cell(self, cell, R):
        cell = dict(cell, R=R)
        rows, extra = cv.run_cell_batches(cell, 3, 0, R // cell["m"])
        return {k: v.sum(axis=0) for k, v in rows.items()}, {k: v.sum(axis=0) for k, v in extra.items()}

    def check(self, misses, R, tails, idx, nsd=4.5, extra=0.0):
        for i in idx:
            t = tails[i]
            self.assertLessEqual(abs(misses[i] / R - t), nsd * math.sqrt(t * (1 - t) / R) + extra, "tail %g: %g" % (t, misses[i] / R))

    def test_student_on_normal_data_is_exact(self):
        cell = cv.make_cell("sampling", "s", "", "normal", 30, {"method": "student"}, m=1000)
        rows, extra = self.run_cell(cell, 40000)
        tails = covproc.sampling_tails()
        for r in ("lower", "upper"):
            self.check(rows[r], 40000, tails, (4, 5, 6, 7))
        self.assertEqual(int(extra["unresolved"].sum()), 0)

    def test_student_under_the_stress_family_fails_where_it_should(self):
        cell = cv.make_cell("sampling", "s", "", "stress", 30, {"method": "student"}, m=1000)
        rows, _ = self.run_cell(cell, 20000)
        tails = covproc.sampling_tails()
        self.assertGreater(rows["upper"][0] / 20000, 3.0 * tails[0])  # known: right-skewed data, n = 30, tail 3.6e-3
        self.assertLess(rows["lower"][0] / 20000, tails[0])  # the other side is conservative

    def test_bootstrap_t_on_normal_data(self):
        cell = cv.make_cell("sampling", "s", "", "normal", 30, {"method": "boott", "B": 1000}, m=1)
        rows, _ = self.run_cell(cell, 3000)
        tails = covproc.sampling_tails()
        for r in ("lower", "upper"):
            self.check(rows[r], 3000, tails, (6, 7), extra=0.012)  # bootstrap-t is second-order accurate, not exact

    def test_tost_type_one_error_at_the_boundary(self):
        cell = cv.make_cell("tost", "t", "", "normal", 100, {"rho": 0.5, "cv": 0.2}, m=1000)
        rows, _ = self.run_cell(cell, 20000)
        tails = covproc.holm_levels()
        for r in ("ratio_hi", "ratio_lo"):
            self.check(rows[r], 20000, tails, (3, 4, 5, 6), extra=0.003)

    def test_pipeline_is_conservative_without_and_with_biases(self):
        for K, ip, b in ((3, 1, 0.0), (3, 1, 2.0), (4, 2, 2.0)):
            pp = {"N": 50, "nb": 25, "K": K, "ip": ip, "order": 1.0, "b_h": b, "b_int": b, "b_w": b, "sampling": "student",
                  "B_c": 2000, "B_l": 3000, "B_s": 1000}
            cell = cv.make_cell("pipeline", "p", "", "normal", 50, pp, m=10, share=True)
            rows, extra = self.run_cell(cell, 100)
            self.assertEqual(rows["RL"].shape, (7,))
            self.assertLessEqual(int(rows["RL"].max()), 2)  # the union bound is loose: a few misses in 100 at most (tail 0.05)
            self.assertEqual(int(extra["nonfinite"].sum()), 0)

    def test_the_biased_scenario_is_anti_conservative_without_the_budget(self):
        # the pipeline scenario "bias 2 + 2 + 2 production standard errors": a sampling limit that ignores the budget falsely
        # rejects "alpha <= 0" almost always, so the near-zero miss rates of the full pipeline are the budget at work
        from mcstat import budget as bd
        N, s_p = 50, 50 ** -0.5
        x = 6.0 * s_p + np.random.default_rng(0).standard_normal((2000, N))
        naive = (bd.StudentMean(x).lower(covproc.holm_levels() / 2.0) > 0.0).mean(axis=0)
        self.assertGreater(float(naive.min()), 0.95)

    def test_ladder_bound_reports_its_branch(self):
        base = {"scenario": "power", "order": 2.0, "ip": 1, "kappa": 60.0, "B": 2000}
        rows, extra = self.run_cell(cv.make_cell("ladder", "l", "", "normal", 30, base, m=10, share=True), 100)
        self.assertEqual(int(rows["bound"].max()), 0)
        self.assertGreater(int(extra["order_branch"][-1]), 90)  # large signal: the order branch
        rows, extra = self.run_cell(cv.make_cell("ladder", "l", "", "normal", 30, dict(base, scenario="nonmono"), m=10, share=True), 100)
        self.assertEqual(int(rows["bound"].max()), 0)
        self.assertEqual(int(extra["order_branch"].sum()), 0)  # non-monotone: never a real order, always the fallback

    def test_positive_control_a_too_small_bound_is_detected(self):
        params = {"scenario": "power", "order": 1.0, "ip": 1, "kappa": 60.0, "B": 3000, "safety": 0.4}  # bound about 0.4 of the bias
        rows, _ = self.run_cell(cv.make_cell("ladder", "l", "", "normal", 30, params, m=10, share=True), 100)
        self.assertGreater(int(rows["bound"].min()), 95)  # misses nearly every replicate at every tail

    def test_unresolved_bootstrap_tails_are_counted(self):
        tiny = cv.make_cell("sampling", "s", "", "normal", 20, {"method": "boott", "B": 100}, m=1)
        _, extra = self.run_cell(tiny, 50)
        self.assertEqual(int(extra["unresolved"][0]), 50)  # tail 3.6e-3 < 1 / (B + 1): the limit is infinite, never optimistic
        ok = cv.make_cell("sampling", "s", "", "normal", 20, {"method": "boott", "B": covproc.boot_size(0.05 / 14, 10)}, m=1)
        _, extra = self.run_cell(ok, 20)
        self.assertEqual(int(extra["unresolved"].sum()), 0)

    def test_bca_is_reported_not_judged(self):
        cell = cv.make_cell("bca", "b", "", "normal", 20, {"B": 199}, m=10, report_only=True)
        rows, extra = self.run_cell(cell, 40)
        self.assertEqual(rows["lower"].shape, (2,))
        self.assertTrue(cell["report_only"])


class Engine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def full_run(self, name, seed=11, workers=1):
        out = self.path(name)
        cv.init_plan(out, small_cells(), seed, "test")
        cv.run(out, workers, log=lambda s: None)
        return out, cv.summarize(out)

    def test_interrupted_run_resumes_to_the_identical_result(self):
        out_a, sa = self.full_run("a")
        out_b = self.path("b")
        cv.init_plan(out_b, small_cells(), 11, "test")
        cv.run(out_b, 1, max_chunks=4, log=lambda s: None)  # interrupted
        partial = cv.summarize(out_b)
        self.assertTrue(partial["incomplete"])
        self.assertLess(partial["cells_complete"], partial["cells_total"])
        self.assertEqual(sum(len(fs) for _, _, fs in os.walk(os.path.join(out_b, "chunks"))), 4)
        cv.run(out_b, 1, log=lambda s: None)  # resume
        sb = cv.summarize(out_b)
        self.assertEqual(sb["incomplete"], [])
        self.assertEqual(counts_of(sa), counts_of(sb))
        self.assertEqual(cv.run(out_b, 1, log=lambda s: None), 0)  # nothing left to do
        self.assertTrue(os.path.exists(os.path.join(out_b, "summary.md")))

    def test_results_do_not_depend_on_the_number_of_workers(self):
        _, s1 = self.full_run("w1")
        _, s2 = self.full_run("w2", workers=2)
        self.assertEqual(counts_of(s1), counts_of(s2))

    def test_seed_reproducibility(self):
        _, a = self.full_run("s1", seed=5)
        _, b = self.full_run("s2", seed=5)
        _, c = self.full_run("s3", seed=6)
        self.assertEqual(counts_of(a), counts_of(b))
        self.assertNotEqual(counts_of(a), counts_of(c))

    def test_a_different_plan_in_the_same_directory_is_rejected(self):
        out, _ = self.full_run("p")
        with self.assertRaises(ValueError):
            cv.init_plan(out, small_cells()[:2], 11, "test")
        with self.assertRaises(ValueError):
            cv.init_plan(out, small_cells(), 12, "test")
        cv.init_plan(out, small_cells(), 11, "test")  # the same plan is accepted

    def test_streams_depend_on_cell_and_batch_only(self):
        x = cv.batch_rng(1, "cell-a", 0).random(3)
        self.assertTrue(np.array_equal(x, cv.batch_rng(1, "cell-a", 0).random(3)))
        for other in (cv.batch_rng(1, "cell-a", 1), cv.batch_rng(1, "cell-b", 0), cv.batch_rng(2, "cell-a", 0)):
            self.assertFalse(np.array_equal(x, other.random(3)))

    def test_results_do_not_depend_on_the_chunking(self):
        cells = small_cells()
        for c in cells:
            c["chunk"] = c["m"] * (1 + (c["chunk"] // c["m"]) // 3)  # other chunk sizes, same plan otherwise
        out = self.path("rechunked")
        cv.init_plan(out, cells, 11, "test")
        cv.run(out, 1, log=lambda s: None)
        _, ref = self.full_run("ref")
        self.assertEqual(counts_of(ref), counts_of(cv.summarize(out)))

    def test_cli_smoke_profile_end_to_end(self):
        out = self.path("smoke")
        self.assertEqual(cv.main(["run", "--profile", "smoke", "--out", out, "--workers", "1", "--target", "0.3"]), 0)
        self.assertEqual(cv.main(["summarize", "--out", out]), 0)
        with open(os.path.join(out, "summary.json")) as f:
            s = json.load(f)
        self.assertEqual(s["cells_complete"], s["cells_total"])
        procs = {c["proc"] for c in s["cells"]}
        self.assertEqual(procs, set(covproc.PROCS))  # every procedure family ran
        with open(os.path.join(out, "summary.md")) as f:
            md = f.read()
        for text in ("Tail levels", "Qualification matrix", "Verdict list", "Failures", "UCL / tail by cell", "bca:mean"):
            self.assertIn(text, md)
        bca = [v for v in s["verdicts"] if v["label"] == "bca:mean"]
        self.assertTrue(bca and all(v["qualified"] is None for v in bca))


if __name__ == "__main__":
    unittest.main()
