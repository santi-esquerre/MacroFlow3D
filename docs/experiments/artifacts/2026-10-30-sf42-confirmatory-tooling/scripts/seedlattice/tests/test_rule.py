import math
import unittest

import numpy as np

from seedlattice.rule import (_round_half_away, admissible, admissible_all, axis_seeds, line, on_face, parse_window, v2)

D = "0.1,0.9,0.1,0.9"
FULL = "0,1,0,1"


class TestRule(unittest.TestCase):
    def test_full_plane_v2_rule(self):
        # derived: ny*(2m+1)/(2M) integer for some m  <=>  v2(M) < v2(ny); the 1e-6 tolerance never matters (spacing >= 1/8192)
        for ny in range(2, 521, 2):
            for M in range(1, 65):
                self.assertEqual(on_face(ny, FULL, M)[0], v2(M) < v2(ny), (ny, M))

    def test_384_128_fails(self):
        bad, where = on_face(384, D, 128)
        self.assertTrue(bad)
        self.assertEqual(where, (0, 1, 3))  # p = 1 precedes (m2 = 1, m3 = 0): the runner reports the lowest p

    def test_powers_of_two_with_window_01_09(self):
        # the SF-42 spec's "powers of two fail whenever 3 divides ny" holds for the SF-43 grids only as listed here
        # (it is false in general, e.g. ny = 96 admits M = 128 and ny = 384 admits M = 512): the exact outcome is arithmetic
        for M in (8, 16, 32, 64, 128, 256):
            self.assertTrue(on_face(384, D, M)[0], M)
        self.assertFalse(on_face(384, D, 512)[0])
        self.assertFalse(on_face(96, D, 128)[0])
        for ny in (240, 480):  # 5 | ny as well: small powers of two fail, large ones (>= 64 / 128) pass
            self.assertTrue(on_face(ny, D, 32)[0])
        self.assertEqual([M for M in (32, 64, 128) if not on_face(240, D, M)[0]], [64, 128])
        self.assertEqual([M for M in (64, 128, 256) if not on_face(480, D, M)[0]], [128, 256])

    def test_sf39_m100_passes(self):
        for ny in (256, 384, 64, 128, 192):
            self.assertFalse(on_face(ny, D, 100)[0], ny)
        self.assertFalse(on_face(384, "0.3,0.7,0.3,0.7", 100)[0])
        w12 = "0.36666666666666664,0.63333333333333336,0.36666666666666664,0.63333333333333336"
        self.assertFalse(on_face(192, w12, 100)[0])

    def test_seed_formula_is_binary64_in_runner_order(self):
        M = 7
        want = [0.1 + (0.9 - 0.1) * ((float(m) + 0.5) / float(M)) for m in range(M)]
        self.assertEqual(list(axis_seeds(0.1, 0.9, M)), want)  # exact equality, not allclose

    def test_first_offender_order(self):
        # axis 3 offender at (m2 = 0, m3 > 0) precedes an axis 2 offender at m2 >= 1; at p = 0 axis 2 is tested first
        self.assertEqual(on_face(16, FULL, 2), (True, (0, 0, 2)))
        self.assertEqual(on_face(384, "0,1,0.1,0.9", 128)[0], True)

    def test_overflow_and_nan_count_as_on_face(self):
        # finite window whose width overflows: positions are inf/NaN, !(NaN > tol) -> on face (as in the runner)
        self.assertEqual(on_face(16, "-1e308,1e308,0.1,0.9", 4), (True, (0, 0, 2)))
        self.assertTrue(on_face(16, "-1e308,1e308,-1e308,1e308", 1)[0])

    def test_round_half_away_matches_std_round(self):
        v = np.array([2.5, -2.5, 0.5, -0.5, 0.49999999999999994, 1.5, -0.2, 3.0, 1e15 + 0.5])
        want = [3.0, -3.0, 1.0, -1.0, 0.0, 2.0, -0.0, 3.0, 1e15 + 1.0]
        self.assertEqual(list(_round_half_away(v)), want)

    def test_input_validation(self):
        for bad in ("0.1,0.9,0.1", "0.9,0.1,0.1,0.9", "0.1,0.9,0.5,0.5", "nan,1,0,1", "0,inf,0,1", "a,b,c,d"):
            with self.assertRaises(ValueError, msg=bad):
                parse_window(bad)
        for M in (0, 4097, -1):
            with self.assertRaises(ValueError):
                on_face(16, D, M)

    def test_admissible(self):
        self.assertEqual(admissible(16, FULL, range(1, 9)), [])  # needs v2(M) >= v2(16) = 4
        self.assertEqual(admissible(8, FULL, range(1, 17)), [8, 16])
        self.assertEqual(admissible_all([8, 16], FULL, range(1, 33)), [16, 32])
        self.assertEqual(line(384, D, 100), "384 0.1,0.9,0.1,0.9 100 OK")


if __name__ == "__main__":
    unittest.main()
