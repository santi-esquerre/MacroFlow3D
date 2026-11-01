import unittest

from pilots import design as d
from seedlattice import rule


class TestGeometry(unittest.TestCase):
    def test_planes_window_d25_to_94(self):
        pl = d.planes(1040, 240, 10)
        self.assertEqual(len(pl), 70)
        e = 10 / 240
        x = [float(s) for s in pl]
        self.assertAlmostEqual((x[0] - 5 * e) / e, 25.0, places=9)
        self.assertAlmostEqual((x[-1] - 5 * e) / e, 94.0, places=9)
        self.assertTrue(all(b > a for a, b in zip(x, x[1:])))
        self.assertLess(x[-1], 1040 / 240)  # inside the box

    def test_short_box_rejected(self):
        with self.assertRaises(ValueError):
            d.planes(480, 120, 5)  # 96 lambda: planes up to d = 94 do not leave the 5-lambda buffer

    def test_windows(self):
        self.assertEqual(d.window(1040, 240, 10, "1"), "0.1,0.9,0.1,0.9")       # 19.2 of 24 lambda
        self.assertEqual(d.window(1040, 480, 10, "1"), "0.3,0.7,0.3,0.7")       # 19.2 of 48
        self.assertEqual(d.window(832, 192, 8, "4"), "0.3,0.7,0.3,0.7")         # 9.6 of 24
        with self.assertRaises(ValueError):
            d.window(832, 96, 8, "1")                                            # 19.2 > 12

    def test_mg_levels_matches_sf39_par2(self):
        self.assertEqual(d.mg_levels(2048, 256, 256), 6)   # SF-39 par2_s1.yaml
        self.assertEqual(d.mg_levels(2496, 288, 288), 5)   # par2_s225 / s4

    def test_par2_time_is_sf39_rule(self):
        # SF-39 s1: dt = 12.5, dx = 5, U_nom = 100/10240  ->  U dt / dx = 0.0244140625
        self.assertAlmostEqual(12.5 * (100 / 10240) / 5, 0.0244140625, places=15)
        dt, n = d.par2_time(1040, 240, 10)
        self.assertAlmostEqual(dt, 0.0244140625 / 240, places=18)
        L1, ell = 1040 / 240, 10 / 240
        self.assertGreaterEqual(n * dt, 1.2 * (L1 - 10 * ell))
        self.assertLess((n - 1) * dt, 1.2 * (L1 - 10 * ell))


class TestRunTables(unittest.TestCase):
    def test_j2_distinct_admissible_lattices(self):
        runs = d.j2_runs()
        self.assertEqual(len(runs), 12)
        for amp in d.EPS:
            ms = [r.M for r in runs if r.amp == amp]
            self.assertEqual(len(set(ms)), 3, amp)
            self.assertEqual(ms, sorted(ms))
            r0 = [r for r in runs if r.amp == amp][0]
            for m in ms:
                self.assertFalse(rule.on_face(r0.ny, d.window(r0.nx, r0.ny, r0.lh, amp), m)[0], (amp, m))
        self.assertEqual([r.M for r in runs if r.amp == "0.25"], [64, 128, 192])   # ny = 240: multiples of 64 only
        self.assertEqual([r.M for r in runs if r.amp == "4"], [58, 100, 202])

    def test_j2_same_field_seeds_across_M(self):
        for amp in d.EPS:
            self.assertEqual(len({(r.seed0, r.nr) for r in d.j2_runs() if r.amp == amp}), 1)

    def test_seed_ranges_disjoint_and_reserved(self):
        for job, lo, hi in (("J1", 420601, 420699), ("J1B", 420901, 420999), ("J2", 420701, 420799), ("J3", 420801, 420899)):
            for r in d.JOBS[job]():
                self.assertTrue(lo <= r.seed0 and r.seed0 + r.nr - 1 <= hi, (r.id, r.seed0))
        self.assertEqual(d.PAR2_SEED, 420800001)

    def test_j1_and_j3_lattices_pass_the_rule(self):
        for r in d.j1_runs() + d.j3_runs():
            if r.code == "consumer":
                self.assertFalse(rule.on_face(r.ny, d.window(r.nx, r.ny, r.lh, r.amp), r.M)[0], r.id)

    def test_j1b_is_the_complement_of_j1_and_sf39_in_the_sf43_arms(self):
        """Every SF-43 planning grid (seedlattice.plan arms) neither run by SF-39 nor in J1 is in J1B, at the highest sigma^2 planned on it."""
        from seedlattice import plan
        sf39 = {(3328, 384, 16), (3328, 384, 32), (1664, 384, 16)}     # SF-39 s225_A (distance), s4_A (production), s4_w24 (E_W at 4)
        order = list(d.EPS)
        top = {}
        for a in plan.build_arms():
            k = (a["nx"], a["ny"], a["lh"])
            top[k] = max(top.get(k, a["amp"]), a["amp"], key=order.index)
        want = sorted((k + (amp,)) for k, amp in top.items() if k not in sf39 and k not in {g[:3] for g in d.J1_GRIDS})
        self.assertEqual(sorted(d.J1B_GRIDS), want)
        for k, amp in top.items():
            if k in {g[:3] for g in d.J1_GRIDS}:        # J1 already ran each of its grids at the highest planned sigma^2
                self.assertEqual([g[3] for g in d.J1_GRIDS if g[:3] == k], [amp], k)

    def test_j1b_runs(self):
        runs = d.j1b_runs()
        self.assertEqual([(r.nx, r.ny, r.amp, r.nr, r.code, r.job) for r in runs],
                         [(416, 96, "2.25", 1, "consumer", "J1B"), (832, 384, "2.25", 1, "consumer", "J1B"), (2080, 240, "1", 1, "consumer", "J1B")])
        self.assertEqual([r.seed0 for r in runs], [420901, 420902, 420903])
        self.assertEqual([r.M for r in runs], [100, 100, 128])
        self.assertEqual(len({r.id for r in runs}), 3)
        for r in runs:
            self.assertFalse(rule.on_face(r.ny, d.window(r.nx, r.ny, r.lh, r.amp), r.M)[0], r.id)
            a = d.consumer_args(r, "/o")
            self.assertEqual(len(a[a.index("--planes") + 1].split(",")), 70)
            self.assertEqual(a[a.index("--nr") + 1], "1")

    def test_j1b_seeds_disjoint_from_every_other_sf42_range(self):
        used = [r.seed0 + k for job in ("J1", "J2", "J3") for r in d.JOBS[job]() for k in range(r.nr)] + [d.PAR2_SEED]
        mine = {r.seed0 for r in d.j1b_runs()}
        self.assertFalse(mine & set(used))
        self.assertTrue(all(420901 <= s <= 420999 for s in mine))     # clear of 420001-420099 (spectral), 420201-420299 (R1), 4206xx-4208xx

    def test_j1_par2_uses_the_consumer_seed_of_the_grid(self):
        runs = {r.id: r for r in d.j1_runs()}
        self.assertEqual(runs["p_s225_1248x288"].seed0, runs["c_s225_1248x288"].seed0)

    def test_j3b_is_par2_only_with_longer_horizon(self):
        runs = d.j3b_runs()
        self.assertEqual([r.amp for r in runs], ["1", "2.25", "4"])
        self.assertTrue(all(r.code == "par2" and r.job == "J3B" and r.hz == 3.6 for r in runs))
        base = {r.id: r for r in d.j3_runs() if r.code == "par2"}
        for r in runs:
            b = base[r.id[:-3]]
            self.assertEqual((r.seed0, r.nr, r.nx, r.ny), (b.seed0, b.nr, b.nx, b.ny))     # same fields as J3
            self.assertEqual(d.par2_time(r.nx, r.ny, r.lh, 1.2), d.par2_time(b.nx, b.ny, b.lh))
            self.assertGreater(d.par2_time(r.nx, r.ny, r.lh, r.hz)[1], 2.9 * d.par2_time(b.nx, b.ny, b.lh)[1])
        self.assertIn("n_steps: %d" % d.par2_time(runs[2].nx, runs[2].ny, runs[2].lh, 3.6)[1], d.par2_yaml(runs[2]))

    def test_consumer_args_and_par2_yaml(self):
        r = [x for x in d.j3_runs() if x.id == "c_s4_1664x192"][0]
        a = d.consumer_args(r, "/o")
        self.assertEqual(a[a.index("--np-side") + 1], "100")
        self.assertEqual(len(a[a.index("--planes") + 1].split(",")), 70)
        self.assertEqual(a[a.index("--hmax-div") + 1], "8")
        y = d.par2_yaml([x for x in d.j3_runs() if x.id == "p_s4_1664x192"][0])
        for key in ("field_source: sf18_medium_b", "seed0: 420831", "cg_rtol: 1.0e-10", "seed: 420800001", "plane_crossings",
                    "inject_window: [0.1, 0.9, 0.1, 0.9]", "NR: 8"):
            self.assertIn(key, y)
        self.assertEqual(y.count("planes: ["), 1)


if __name__ == "__main__":
    unittest.main()
