import json
import os
import unittest

from pilots import heldout

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, "..", "..", ".."))           # artifact root
REPO = os.path.normpath(os.path.join(R, "..", "..", "..", ".."))      # docs/experiments/artifacts/<name> -> repository root


class TestHeldout(unittest.TestCase):
    def test_j1b_points_are_held_out_and_match_the_committed_json(self):
        rows = heldout.heldout(os.path.join(R, "pilots"), REPO)
        self.assertEqual(sorted(r["label"] for r in rows), ["J1B/c_s1_2080x240", "J1B/c_s225_416x96", "J1B/c_s225_832x384"])
        for r in rows:
            self.assertEqual(r["code"], "lagrangian")
            self.assertGreater(r["observed_s"], 0.0)
            self.assertAlmostEqual(r["rel_refit_model"], (r["refit_model_s"] - r["observed_s"]) / r["observed_s"], places=12)
        with open(os.path.join(R, "pilots", "j1b_heldout.json")) as f:
            committed = {x["label"]: x for x in json.load(f)["rows"]}
        for r in rows:
            for k in ("observed_s", "sf39_model_s", "refit_model_s"):
                self.assertAlmostEqual(r[k] / committed[r["label"]][k], 1.0, places=9)

    def test_j1b_is_not_in_the_committed_fit(self):
        with open(os.path.join(R, "pilots", "cost_projection.json")) as f:
            labels = [x["label"] for x in json.load(f)["new_points_residuals_refit"]]
        self.assertFalse([x for x in labels if x.startswith("J1B/")])


if __name__ == "__main__":
    unittest.main()
