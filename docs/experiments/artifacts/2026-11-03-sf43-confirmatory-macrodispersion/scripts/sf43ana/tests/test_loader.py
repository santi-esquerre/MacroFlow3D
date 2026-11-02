import glob
import io
import json
import os
import tempfile
import unittest

import numpy as np

import sf43ana  # noqa: F401
from sf42red import reduce as red
from sf42red.tests import synth
from sf43ana import config
from sf43ana.loader import Ledger, LoadError, TableCache, build_data, common_prefix, effective_order, load_all, read_ledger
from sf43ana.tests import fixtures as fx


def flat_mean(a, w, basis, est):
    return {"alpha_TP": 0.006, "alpha_TAP": 0.0055, "alpha_LP": 0.5}.get(est, 0.0) + (1e-4 if a["id"] == "l5" else 0.0)


class Base(unittest.TestCase):
    def build(self, **kw):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pr = fx.prereg_dict()
        self.paths = fx.write_world(self._tmp.name, self.pr, flat_mean, **kw)
        self.prereg = config.load_prereg(self.paths["prereg"])
        return load_all(self.prereg, self.paths["campaign"], self.paths["reduced"])


class PrefixAndReplacement(Base):
    def test_full_prefix(self):
        arms, proc, _ = self.build()
        a = arms["p1"]
        self.assertEqual(a.N, 40)
        self.assertEqual(list(a.seeds), list(range(1000, 1040)))
        self.assertEqual(a.info["status"], "ok")
        self.assertIn("par2_1", arms)  # both codes are loaded
        self.assertEqual(arms["par2_1"].N, 10)
        self.assertTrue(np.all(np.isfinite(a.get("main", "disp", "alpha_TP"))))

    def test_par2_seed_is_segment_seed0_plus_r(self):
        arms, _, _ = self.build()
        a = arms["par2_1"]
        self.assertEqual(list(a.seeds), list(range(7000, 7010)))
        tab = TableCache(self.paths["reduced"], self.prereg, read_ledger(os.path.join(self.paths["campaign"], "ledger.jsonl")))
        t = tab.table("par2_1", a.segments[0], "main")
        row = next(iter(t.values()))
        self.assertEqual(row["seed"], "nan")  # the table has no seed for Par2 rows
        self.assertEqual(sorted({k[-1] for k in t}), list(range(7000, 7005)))  # the first of two segments (seed0 7000)

    def test_prefix_stops_at_first_incomplete_even_if_later_seeds_are_fine(self):
        arms, _, _ = self.build(fail=[("p1", 1030)])
        a = arms["p1"]
        self.assertEqual(a.N, 30)
        self.assertEqual(a.info["stopped_at"]["seed"], 1030)
        self.assertIn("unusable_solve", a.info["stopped_at"]["reason"])
        self.assertEqual(arms["l5"].N, 40)  # other arms unaffected (ledger per source arm)

    def test_below_floor(self):
        arms, _, _ = self.build(fail=[("p1", 1005)])
        self.assertEqual(arms["p1"].N, 5)
        self.assertEqual(arms["p1"].info["status"], "below_floor")
        self.assertFalse(arms["p1"].usable)

    def test_group_wide_replacement_takes_the_position_of_the_replaced_seed(self):
        arms, _, _ = self.build(fail=[("p1", 1005), ("l5", 1005)], replace=[("g1", 1005, 1900)])
        for aid in ("p1", "l5", "l20"):
            a = arms[aid]
            self.assertEqual(a.N, 40, aid)
            self.assertEqual(int(a.seeds[5]), 1900, aid)
            self.assertNotIn(1005, a.seeds)
            self.assertEqual(a.info["replacements_applied"], [{"seed": 1005, "replaced_by": 1900}])
        self.assertEqual(common_prefix([arms["p1"], arms["l5"], arms["l20"]]), 40)

    def test_replacement_chain_and_failed_reserve(self):
        arms, _, _ = self.build(fail=[("p1", 1005), ("p1", 1900)], replace=[("g1", 1005, 1900), ("g1", 1900, 1901)])
        self.assertEqual(int(arms["p1"].seeds[5]), 1901)
        arms, _, _ = self.build(fail=[("p1", 1900)], replace=[("g1", 1005, 1900)])  # the reserve itself failed, not replaced
        self.assertEqual(arms["p1"].N, 5)

    def test_effective_order_errors(self):
        self.assertEqual(effective_order([1, 2, 3], {2: 9})[0], [1, 9, 3])
        with self.assertRaises(LoadError):
            effective_order([1, 2, 3], {2: 3})  # the reserve is already in the order
        with self.assertRaises(LoadError):
            effective_order([1, 2], {1: 5, 5: 1})  # cycle

    def test_missing_table_ends_the_prefix_and_is_reported(self):
        self.build()
        seg2 = sorted(glob.glob(os.path.join(self.paths["reduced"], "p1", "*")))[1]
        os.remove(glob.glob(os.path.join(seg2, "*d25-94.csv"))[0])
        arms, _, _ = load_all(self.prereg, self.paths["campaign"], self.paths["reduced"])
        self.assertEqual(arms["p1"].N, 25)  # the first 25 of the 50 ledger seeds (40 + reserves) sit in the first segment
        self.assertEqual(len(arms["p1"].info["missing_tables"]), 1)
        self.assertIn("p1__u", arms["p1"].info["missing_tables"][0])  # the executor's naming <arm>__<segdir>__d<lo>-<hi>.csv

    def test_source_and_hmax_selection_two_entries_sharing_the_production_arm(self):
        arms, _, _ = self.build()
        h16, h8, p1 = arms["h16"], arms["h8"], arms["p1"]
        self.assertEqual((h16.N, h8.N), (20, 20))
        self.assertTrue(np.array_equal(h16.seeds, p1.seeds[:20]))
        self.assertEqual(h16.cfg["source"], "p1")
        # rows of the same table, different hmax_div: values come from different (noise) streams
        self.assertFalse(np.allclose(h16.get("main", "disp", "alpha_TP"), h8.get("main", "disp", "alpha_TP")))
        self.assertFalse(np.allclose(h16.get("main", "disp", "alpha_TP"), p1.get("main", "disp", "alpha_TP", 20)))

    def test_windows_are_separate_tables(self):
        arms, _, _ = self.build()
        a = arms["p1"]
        self.assertEqual(a.get("h1", "pos", "alpha_TP").shape, (40,))
        self.assertTrue(np.all(np.isfinite(a.get("far", "pos", "alpha_LP"))))

    def test_torn_last_ledger_line_is_ignored_but_not_a_bad_middle_line(self):
        self.build()
        lp = os.path.join(self.paths["campaign"], "ledger.jsonl")
        with open(lp, "a") as f:
            f.write('{"kind": "unit_end", "arm": "p1", "compl')
        n = read_ledger(lp).n
        self.assertGreater(n, 5)
        with open(lp) as f:
            lines = f.read().split("\n")
        lines.insert(1, "{bad")
        with open(lp, "w") as f:
            f.write("\n".join(lines))
        with self.assertRaises(LoadError):
            read_ledger(lp)

    def test_data_contract(self):
        arms, _, ledger = self.build(fail=[("p1", 1005)], replace=[("g1", 1005, 1900)])
        d = build_data(self.prereg, arms, ledger)
        self.assertEqual(sorted(d), ["arms", "coverage_summary", "halted", "ledger", "sources", "unreduced"])
        p = d["arms"]["p1"]
        self.assertEqual(p["N"], 40)
        self.assertEqual(sorted(p["cols"])[:4], ["U_r", "U_r:disp", "alpha_LP", "alpha_LP:disp"])
        self.assertEqual(len(p["verdict"]), 40)
        self.assertEqual(d["sources"]["p1"], ["p1", "h16", "h8"])
        failed = [r for r in d["ledger"] if r["class"] == "unusable_solve" and r["arm"] == "p1"]
        self.assertEqual([r["seed"] for r in failed], [1005])
        rep = [r for r in d["ledger"] if r["arm"] == "p1" and r["seed"] == 1900]
        self.assertEqual(rep[0]["replaces"], 1005)
        self.assertTrue(np.array_equal(p["seed"], arms["p1"].seeds))


class LedgerDrivenCuts(Base):
    """The engine's ledger vocabulary (sf43camp/lanes.py): seed records, unreduced, reest."""

    def build(self, mutate=None, **kw):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pr = fx.prereg_dict()
        if mutate:
            mutate(self.pr)
        self.paths = fx.write_world(self._tmp.name, self.pr, flat_mean, **kw)
        self.prereg = config.load_prereg(self.paths["prereg"])
        return load_all(self.prereg, self.paths["campaign"], self.paths["reduced"])

    @staticmethod
    def grow_p1(pr, n_max=60):
        a = next(x for x in pr["arms"] if x["id"] == "p1")
        a.update(seeds=list(range(1000, 1000 + n_max)), n_max=n_max, reest=True)

    def test_window_stem_is_the_executors_rule(self):
        from sf43ana.loader import window_stem
        self.assertEqual([window_stem(*b) for b in ([25.0, 94.0], [25, 59.5], [59.5, 94], [94, 194.0])], ["d25-94", "d25-59.5", "d59.5-94", "d94-194"])

    def test_replacement_record_key_is_replaced_by(self):
        self.build(fail=[("p1", 1005)], replace=[("g1", 1005, 1900)])
        lp = os.path.join(self.paths["campaign"], "ledger.jsonl")
        with open(lp) as f:
            rec = [json.loads(x) for x in f if '"replacement"' in x]
        self.assertEqual(rec, [{"kind": "replacement", "group": "g1", "seed": 1005, "replaced_by": 1900, "arm": "?"}])
        self.assertEqual(read_ledger(lp).repl, {"g1": {1005: 1900}})

    def test_seed_records_carry_verdict_and_class(self):
        arms, proc, ledger = self.build(fail=[("p1", 1005)])
        r = ledger.seeds[("p1", 1005)]
        self.assertEqual((r["class"], r["verdict"]), ("unusable_solve", "stagnated"))
        self.assertEqual(ledger.seeds[("p1", 1006)]["verdict"], "converged")
        self.assertEqual(proc["verdict_unknown"], 0)
        self.assertEqual((arms["p1"].N, set(arms["p1"].verdicts)), (5, {"converged"}))

    def test_verdict_outside_the_ok_vocabulary_ends_the_prefix(self):
        self.build()
        lp = os.path.join(self.paths["campaign"], "ledger.jsonl")
        with open(lp) as f:
            recs = [json.loads(x) for x in f]
        for r in recs:
            if r["kind"] == "seed" and r["arm"] == "p1" and r["seed"] == 1010:
                r["verdict"] = "stagnated"
        with open(lp, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
        arms, _, _ = load_all(self.prereg, self.paths["campaign"], self.paths["reduced"])
        self.assertEqual(arms["p1"].N, 10)
        self.assertIn("verdict 'stagnated'", arms["p1"].info["stopped_at"]["reason"])

    def test_unreduced_segment_is_missing_explicitly(self):
        arms, proc, _ = self.build(unreduced=[("p1", 1030)])
        a = arms["p1"]
        seg2_first = a.info["stopped_at"]
        self.assertIsNotNone(seg2_first)
        self.assertIn("unreduced segment", seg2_first["reason"])
        self.assertLess(a.N, 40)
        self.assertEqual(list(proc["unreduced"]), ["p1/" + next(iter(proc["unreduced"])).split("/")[1]])
        self.assertEqual(arms["l5"].N, 40)  # other arms unaffected
        self.assertEqual(arms["h16"].N, 20)

    def test_unreduced_then_reduced_in_a_later_session_is_not_missing(self):
        self.build(unreduced=[("p1", 1030)])
        lp = os.path.join(self.paths["campaign"], "ledger.jsonl")
        with open(lp) as f:
            recs = [json.loads(x) for x in f]
        un = next(r for r in recs if r["kind"] == "unreduced")
        self.assertEqual(read_ledger(lp).unreduced.keys(), {(un["arm"], os.path.basename(un["segment"]))})
        with open(lp, "a") as f:
            f.write(json.dumps({"kind": "reduced", "arm": un["arm"], "segment": un["segment"], "seeds": un["seeds"]}) + "\n")
        self.assertEqual(read_ledger(lp).unreduced, {})

    def test_reest_cut_is_n_new_for_entries_that_follow_it_else_n_conf(self):
        arms, proc, _ = self.build(mutate=self.grow_p1, reest={"p1": 50})
        self.assertEqual((arms["p1"].N, arms["p1"].info["n_cut"], arms["p1"].info["reest_n_new"]), (50, 50, 50))
        self.assertEqual(arms["l5"].N, 40)  # an entry without reest keeps its planned n_conf
        self.assertEqual(arms["h16"].N, 20)
        self.assertEqual(proc["reest_n_new"], {"p1": 50})
        arms, _, _ = self.build(mutate=self.grow_p1)  # no reest record: the planned n_conf
        self.assertEqual((arms["p1"].N, arms["p1"].info["n_cut"]), (40, 40))
        arms, _, _ = self.build(mutate=self.grow_p1, reest={"p1": 200})  # beyond the seeds listed: clipped
        self.assertEqual((arms["p1"].N, arms["p1"].info["n_cut"]), (60, 60))
        arms, _, _ = self.build(mutate=self.grow_p1, reest={"p1": 10})  # never below n_conf
        self.assertEqual(arms["p1"].N, 40)

    def test_prefix_stops_inside_the_extension_positions(self):
        arms, _, _ = self.build(mutate=self.grow_p1, reest={"p1": 50}, fail=[("p1", 1045)])
        self.assertEqual(arms["p1"].N, 45)
        self.assertEqual(arms["p1"].info["stopped_at"]["seed"], 1045)

    def test_n_max_validation(self):
        pr = fx.prereg_dict()
        pr["arms"][0]["n_max"] = 99  # more than the seeds listed
        with tempfile.TemporaryDirectory() as tmp:
            paths = fx.write_world(tmp, pr, flat_mean)
            with self.assertRaises(config.ConfigError):
                config.load_prereg(paths["prereg"])


class RealReducerTable(unittest.TestCase):
    def test_table_written_by_sf42red_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "arm")
            synth.write_pass(root, r=0, seed=777)
            out = os.path.join(tmp, "x", "xa__seg__d25-94.csv")
            os.makedirs(os.path.dirname(out))
            red.reduce_all([root], [], [], None, None, 25.0, 94.0, None, out, root=tmp, warn=io.StringIO())
            pr = {"windows": {"w": [25.0, 94.0]}, "reduced_pattern": "x/xa__seg__d{lo:g}-{hi:g}.csv"}
            tc = TableCache(tmp, pr, Ledger())
            rows = tc.table("x", "seg", "w")
            row = rows[("consumer", "plane", "flux", "disp", "perplane", -1, 777)]
            self.assertAlmostEqual(float(row["alpha_TP"]), 0.03, places=9)
            self.assertAlmostEqual(float(row["U_r"]), 2.0, places=9)


if __name__ == "__main__":
    unittest.main()
