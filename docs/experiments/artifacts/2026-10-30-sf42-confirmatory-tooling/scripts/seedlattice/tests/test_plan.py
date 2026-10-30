import json
import unittest

from seedlattice.plan import build_arms, per_amplitude, window_string
from seedlattice.rule import on_face


class TestPlan(unittest.TestCase):
    def test_window_strings(self):
        self.assertEqual(window_string(24, "19.2"), "0.1,0.9,0.1,0.9")
        self.assertEqual(window_string(48, "19.2"), "0.3,0.7,0.3,0.7")
        self.assertEqual(window_string(12, "9.6"), "0.1,0.9,0.1,0.9")
        self.assertEqual(window_string(24, "9.6"), "0.3,0.7,0.3,0.7")
        self.assertEqual(window_string(24, 24), "0,1,0,1")
        with self.assertRaises(ValueError):
            window_string(7, 1)  # 3/7 is not a short decimal

    def test_arms_follow_the_sf43_table(self):
        grids = {(a["amp"], a["family"], a["nx"], a["ny"]) for a in build_arms()}
        for g in (("0.25", "production", 1040, 240), ("2.25", "production", 1664, 384), ("4", "production", 3328, 384),
                  ("0.25", "E_h", 2080, 480), ("1", "E_W", 1040, 480), ("2.25", "gate5", 1248, 288),
                  ("4", "gate5", 1664, 192), ("4", "E_W", 1664, 384)):
            self.assertIn(g, grids)

    def test_flagged_m_pass_on_every_arm_and_in_json(self):
        arms = build_arms()
        res = per_amplitude(arms)
        for amp, r in res.items():
            for name, v in r["variants"].items():
                for f in v["flagged"].values():
                    if f is not None:
                        self.assertIn(f["M"], v["admissible_on_every_arm"])
                for c in v["per_arm_check"]:
                    self.assertTrue(all(c["flagged_passes"].values()), (amp, name, c))
            prod = next(a for a in arms if a["amp"] == amp and a["family"] == "production")
            for f in r["production_arm_only"]["flagged"].values():
                self.assertFalse(on_face(prod["ny"], prod["window"], f["M"])[0])
        json.dumps(res)

    def test_known_plan_results(self):
        res = per_amplitude(build_arms())
        self.assertEqual([v["M"] for v in res["4"]["variants"]["all arms"]["flagged"].values()], [58, 100, 202])
        self.assertEqual(res["1"]["variants"]["all arms"]["admissible_on_every_arm"], [128])
        for name, v in res["2.25"]["variants"].items():
            self.assertEqual(v["admissible_on_every_arm"], [], name)  # lh = 10 arms (5 | ny) vs the 3 | ny production grid


if __name__ == "__main__":
    unittest.main()
