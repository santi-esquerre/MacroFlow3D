import io
import os
import tempfile
import unittest

import numpy as np

import sf43ana  # noqa: F401
from sf42red import reduce as red
from sf42red.tests import synth
from sf43ana import config
from sf43ana.loader import LoadError, TableCache, common_prefix, effective_order, load_all
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
        arms, proc = self.build()
        a = arms["p1"]
        self.assertEqual(a.N, 40)
        self.assertEqual(list(a.seeds), list(range(1000, 1040)))
        self.assertEqual(a.info["status"], "ok")
        self.assertEqual(proc["skipped_arms_other_code"], ["par2_1"])
        self.assertTrue(np.all(np.isfinite(a.get("main", "disp", "alpha_TP"))))

    def test_prefix_stops_at_first_incomplete_even_if_later_seeds_are_fine(self):
        arms, _ = self.build(fail=[("p1", 1030)])
        a = arms["p1"]
        self.assertEqual(a.N, 30)
        self.assertEqual(a.info["stopped_at"]["seed"], 1030)
        self.assertIn("verdict", a.info["stopped_at"]["reason"])
        self.assertEqual(arms["l5"].N, 40)  # other arms unaffected (ledger per source arm)

    def test_below_floor(self):
        arms, _ = self.build(fail=[("p1", 1005)])
        self.assertEqual(arms["p1"].N, 5)
        self.assertEqual(arms["p1"].info["status"], "below_floor")
        self.assertFalse(arms["p1"].usable)

    def test_group_wide_replacement_takes_the_position_of_the_replaced_seed(self):
        arms, _ = self.build(fail=[("p1", 1005), ("l5", 1005)], replace=[("g1", 1005, 1900)])
        for aid in ("p1", "l5", "l20"):
            a = arms[aid]
            self.assertEqual(a.N, 40, aid)
            self.assertEqual(int(a.seeds[5]), 1900, aid)
            self.assertNotIn(1005, a.seeds)
            self.assertEqual(a.info["replacements_applied"], [{"seed": 1005, "replaced_by": 1900}])
        self.assertEqual(common_prefix([arms["p1"], arms["l5"], arms["l20"]]), 40)

    def test_replacement_chain_and_failed_reserve(self):
        arms, _ = self.build(fail=[("p1", 1005), ("p1", 1900)], replace=[("g1", 1005, 1900), ("g1", 1900, 1901)])
        self.assertEqual(int(arms["p1"].seeds[5]), 1901)
        arms, _ = self.build(fail=[("p1", 1900)], replace=[("g1", 1005, 1900)])  # the reserve itself failed, not replaced
        self.assertEqual(arms["p1"].N, 5)

    def test_effective_order_errors(self):
        self.assertEqual(effective_order([1, 2, 3], {2: 9})[0], [1, 9, 3])
        with self.assertRaises(LoadError):
            effective_order([1, 2, 3], {2: 3})  # the reserve is already in the order
        with self.assertRaises(LoadError):
            effective_order([1, 2], {1: 5, 5: 1})  # cycle

    def test_missing_table_ends_the_prefix_and_is_reported(self):
        self.build()
        os.remove(os.path.join(self.paths["reduced"], "p1", "s2", "main.csv"))
        arms, _ = load_all(self.prereg, self.paths["campaign"], self.paths["reduced"])
        self.assertEqual(arms["p1"].N, 25)  # the first 25 of the 50 ledger seeds (40 + reserves) sit in segment s1
        self.assertEqual(len(arms["p1"].info["missing_tables"]), 1)

    def test_source_and_hmax_selection(self):
        arms, _ = self.build()
        h16, h8, p1 = arms["h16"], arms["h8"], arms["p1"]
        self.assertEqual((h16.N, h8.N), (20, 20))
        self.assertTrue(np.array_equal(h16.seeds, p1.seeds[:20]))
        # rows of the same table, different hmax_div: values come from different (noise) streams
        self.assertFalse(np.allclose(h16.get("main", "disp", "alpha_TP"), h8.get("main", "disp", "alpha_TP")))
        self.assertFalse(np.allclose(h16.get("main", "disp", "alpha_TP"), p1.get("main", "disp", "alpha_TP", 20)))

    def test_windows_are_separate_tables(self):
        arms, _ = self.build()
        a = arms["p1"]
        self.assertEqual(a.get("h1", "pos", "alpha_TP").shape, (40,))
        self.assertTrue(np.all(np.isfinite(a.get("far", "pos", "alpha_LP"))))


class RealReducerTable(unittest.TestCase):
    def test_table_written_by_sf42red_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "arm")
            synth.write_pass(root, r=0, seed=777)
            out = os.path.join(tmp, "t.csv")
            red.reduce_all([root], [], [], None, None, 25.0, 94.0, None, out, root=tmp, warn=io.StringIO())
            rows = TableCache(tmp, dict(config.DEFAULTS, reduced_pattern="t.csv")).rows(
                {"source": "x", "hmax_div": -1, "code": "consumer"}, "s", "w")
            self.assertEqual(set(rows), {(777, "pos"), (777, "disp")})
            self.assertAlmostEqual(rows[(777, "disp")]["alpha_TP"], 0.03, places=9)
            self.assertAlmostEqual(rows[(777, "disp")]["U_r"], 2.0, places=9)


if __name__ == "__main__":
    unittest.main()
