import os
import unittest

from seedlattice.make_cases import FIXTURE
from seedlattice.verify_probe import predictions, read_cases


class TestProbeFixture(unittest.TestCase):
    """The fixture is the output of the production find_seed_on_face (seedlattice_probe) on data/cases.txt."""

    def test_predictor_matches_production_fixture(self):
        self.assertTrue(os.path.exists(FIXTURE), "regenerate the fixture: see make_cases.py")
        with open(FIXTURE) as f:
            fixture = f.read().splitlines()
        want = predictions()
        self.assertEqual(len(fixture), len(want))
        bad = [(i, g, w) for i, (g, w) in enumerate(zip(fixture, want)) if g != w]
        self.assertEqual(bad, [])

    def test_sample_covers_required_cases(self):
        cases = {(int(a), w, int(m)) for a, w, m in read_cases()}
        self.assertIn((384, "0.1,0.9,0.1,0.9", 128), cases)
        for ny in (256, 384, 64, 128, 192):
            self.assertIn((ny, "0.1,0.9,0.1,0.9", 100), cases)
        self.assertTrue(any(w == "0,1,0,1" for _, w, _ in cases))
        self.assertTrue(any(w == "0.3,0.7,0.3,0.7" for _, w, _ in cases))
        out = predictions()
        self.assertTrue(any("ON_FACE" in l for l in out) and any(l.endswith(" OK") for l in out))


if __name__ == "__main__":
    unittest.main()
