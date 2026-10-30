"""SF-42 N3: the Python SF-18 generator against committed GPU dumps and structural invariants."""
import json
import os
import unittest

import numpy as np

from spectral import sf18

DATA = os.path.join(os.path.dirname(__file__), "data")


class GpuDump(unittest.TestCase):
    """data/gpu_Y_N<N>_seed<seed>.npy: Y of `sf42_periodic_flow --n N --eps 1 --seed <seed>` (sigma2 = 1, ell = 1/4)."""

    def test_matches_gpu_to_roundoff(self):
        with open(os.path.join(DATA, "gpu_sf18_meta.json")) as fh:
            meta = json.load(fh)
        self.assertGreaterEqual(len(meta["dumps"]), 2)
        for d in meta["dumps"]:
            y_gpu = np.load(os.path.join(DATA, d["file"]))
            f = sf18.generate(d["n"], sigma2=1.0, ell=0.25, seed=d["seed"])
            self.assertEqual(y_gpu.shape, f.y.shape)
            dy = np.max(np.abs(f.y - y_gpu)) / np.std(y_gpu)
            self.assertLess(dy, 1e-13, "max|dY|/std = %.2e for N=%d" % (dy, d["n"]))
            self.assertLess(abs(f.applied_scale / d["applied_scale"] - 1.0), 1e-13)
            self.assertLess(abs(f.raw_variance / d["raw_variance"] - 1.0), 1e-13)
            self.assertEqual(f.active_mode_count, d["active_mode_count"])


class Structure(unittest.TestCase):
    def test_splitmix64_known_answer(self):
        # first output of the reference splitmix64 stream with state 0
        out = sf18._splitmix64(np.zeros(1, dtype=np.uint64))
        self.assertEqual(int(out[0]), 0xE220A8397B1DCDAF)

    def test_variance_and_mean(self):
        f = sf18.generate(16, sigma2=2.5, seed=3)
        self.assertLess(abs(f.y.var() / 2.5 - 1.0), 1e-12)
        self.assertLess(abs(f.y.mean()), 1e-12)
        g = sf18.generate(16, sigma2=2.5, seed=3, normalize=False)
        self.assertEqual(g.applied_scale, 1.0)
        np.testing.assert_allclose(g.y * f.applied_scale, f.y, rtol=1e-13, atol=1e-14)

    def test_deterministic_and_seed_sensitive(self):
        a = sf18.generate(8, seed=11).y
        np.testing.assert_array_equal(a, sf18.generate(8, seed=11).y)
        self.assertGreater(np.max(np.abs(a - sf18.generate(8, seed=12).y)), 1e-3)

    def test_odd_extent_rejected(self):
        with self.assertRaises(ValueError):
            sf18.generate(9, seed=1)

    def test_refinement_keeps_the_coefficients_of_shared_modes(self):
        a, b = sf18.generate(8, seed=5), sf18.generate(16, seed=5)
        n_checked = 0
        for mx in range(0, 4):
            for my in range(-3, 4):
                for mz in range(-3, 4):
                    ca, cb = a.c[mz % 8, my % 8, mx], b.c[mz % 16, my % 16, mx]
                    if mx == 0 and my == 0 and mz == 0:
                        self.assertEqual((ca, cb), (0, 0))
                        continue
                    self.assertLessEqual(abs(ca - cb), 1e-12 * abs(ca))
                    n_checked += 1
        self.assertEqual(n_checked, 4 * 49 - 1)

    def test_field_on_finer_grids(self):
        f = sf18.generate(16, seed=420001)
        np.testing.assert_allclose(sf18.field_on_grid(f, 16, offset=0.5), f.y, rtol=0, atol=1e-13)
        # the vertices of a 4N grid contain the cell centres (j + 1/2)/N at index 4 j + 2
        v = sf18.field_on_grid(f, 64, offset=0.0)
        np.testing.assert_allclose(v[2::4, 2::4, 2::4], f.y, rtol=0, atol=1e-13)
        # an anisotropic target agrees with the cubic one on the shared points (x: 32, y: 16, z: 24 vertices)
        w = sf18.field_on_grid(f, (32, 16, 24), offset=0.0)
        c = sf18.field_on_grid(f, 96, offset=0.0)
        np.testing.assert_allclose(w, c[::4, ::6, ::3], rtol=0, atol=1e-13)  # (z, y, x)
        # a coarser target keeps the modes |m| <= M/2 - 1 with their coefficients: the generator of that grid, unnormalized
        np.testing.assert_allclose(sf18.field_on_grid(f, 8, offset=0.5, scale=1.0), sf18.generate(8, seed=420001, normalize=False).y,
                                   rtol=0, atol=1e-13)

    def test_truncated_mode_variance_matches_the_direct_difference(self):
        # ell = 0.1 keeps the modes 8..15 well above round-off
        big = sf18.generate(32, seed=9, ell=0.1, normalize=False)
        small = sf18.generate(16, seed=9, ell=0.1, normalize=False)
        d = sf18.field_on_grid(big, 64, offset=0.0) - sf18.field_on_grid(small, 64, offset=0.0)
        expected = sf18.mode_variance_outside(big, 16)
        self.assertGreater(expected, 1e-4)
        self.assertLess(abs(np.mean(d ** 2) / expected - 1.0), 1e-9)
        self.assertEqual(sf18.mode_variance_outside(big, 64), 0.0)


if __name__ == "__main__":
    unittest.main()
