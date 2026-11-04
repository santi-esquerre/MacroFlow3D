"""Seed registry of addendum 1: generation, the frozen check_seeds on frozen + new blocks, reuse and transport-rule checks."""
import copy
import os
import unittest
from unittest import mock

from sf43add.camp import plan as P
from sf43add.camp import seeds as S


class Registry(unittest.TestCase):
    def setUp(self):
        self.reg = S.build_registry()

    def test_passes_and_is_committed(self):
        lines = S.check(self.reg)
        self.assertTrue(all(x.startswith("PASS") for x in lines), lines)
        with open(S.REGISTRY) as f:
            self.assertEqual(f.read(), S.dumps(self.reg), "run `python3 -m sf43add.camp check-seeds` and commit the registry")

    def test_new_blocks_and_reuse(self):
        groups = {b["group"]: b for b in self.reg["blocks"]}
        self.assertEqual(set(groups), {"a1_j9", "a1_j1_base", "a1_j1_H3", "a1_j1_D", "a1_j1_DH3", "a1_j1_H2", "a1_j3", "a1_j8_H3", "a1_j8_D", "a1_j8_DH3"})
        j9 = groups["a1_j9"]
        self.assertEqual((j9["start"], j9["count"], j9["end"], j9["reserves"]), (438001, 150, 438150, {"start": 438151, "count": 15}))
        self.assertEqual({e["id"]: (e["arms"], e["first"], e["last"], e["pairs_with_group"]) for e in self.reg["paired_reuse"]},
                         {"reuse_production_g_s4": (["j1_base10", "j1_H3", "j1_D", "j1_DH3", "j1_H2"], 434001, 434060, "g_s4"),
                          "reuse_j8_conditional_g_s4": (["j8_H3", "j8_D", "j8_DH3"], 434061, 434100, "g_s4"),
                          "reuse_production_g_d_s1": (["j3_l5"], 436001, 436100, "g_d_s1")})
        self.assertTrue(all(e["purpose"] and e["transport"] for e in self.reg["paired_reuse"]))
        # every new range is disjoint from the frozen blocks (the reuse is declared, not hidden in a block)
        fz = P.jload(os.path.join(P.ART, "seeds_registry.json"))
        for b in self.reg["blocks"]:
            for f in fz["blocks"]:
                self.assertTrue(b["end"] < f["start"] or b["start"] > f["end"], (b["group"], f["group"]))

    def test_overlap_with_a_frozen_block_fails(self):
        reg = copy.deepcopy(self.reg)
        j9 = next(b for b in reg["blocks"] if b["group"] == "a1_j9")
        j9["start"], j9["end"] = 434001, 434150                                         # the J9 block moved onto g_s4
        with self.assertRaisesRegex(S.SeedsError, "overlap"):
            S.check(reg)

    def test_forbidden_range_fails(self):
        reg = copy.deepcopy(self.reg)
        b = next(b for b in reg["blocks"] if b["group"] == "a1_j3")
        b["start"], b["end"] = 420601, 420605                                           # SF-42 seeds
        with self.assertRaisesRegex(S.SeedsError, "forbidden"):
            S.check(reg)

    def test_reuse_outside_its_frozen_block_fails(self):
        reg = copy.deepcopy(self.reg)
        next(e for e in reg["paired_reuse"] if e["id"] == "reuse_j8_conditional_g_s4")["last"] = 434401        # beyond g_s4 (434001-434400)
        with self.assertRaisesRegex(S.SeedsError, "not inside frozen block"):
            S.check(reg)

    def test_other_transport_rule_fails(self):
        real = P.build_arms

        def other(frozen, pre):
            made, root = real(frozen, pre)
            made["j1_H3"]["par2"]["transport_seed"] = 123456789
            return made, root
        with mock.patch.object(S, "build_arms", other):
            with self.assertRaisesRegex(S.SeedsError, "transport_seed"):
                S.check(self.reg)

    def test_the_frozen_check_is_the_one_used(self):
        with mock.patch.object(S.B, "check_seeds", side_effect=ValueError("boom")) as m:
            with self.assertRaisesRegex(S.SeedsError, "boom"):
                S.check(self.reg)
        merged = m.call_args[0][0]
        self.assertEqual(len(merged["blocks"]), 12 + len(self.reg["blocks"]))                 # the 12 frozen blocks + the new ones
        self.assertEqual(len(merged["smoke_blocks"]), 9)


if __name__ == "__main__":
    unittest.main()
