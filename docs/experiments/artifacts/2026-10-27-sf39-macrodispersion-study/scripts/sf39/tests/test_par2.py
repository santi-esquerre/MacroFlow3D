"""Known-answer tests of sf39.par2 (C-PAR2): t_b, U_c, alpha, pooled indexing and seeds, config generators on the real SF-38 files, FIELDSTAT."""
import contextlib
import difflib
import io
import os
import tempfile
import unittest

import numpy as np

from sf39 import par2
from sf39.tests import fixtures_par2 as fx

REAL = {"1": "config_pipeline_par2_sf38_s1.yaml", "225": "config_pipeline_par2_sf38_s225.yaml", "4": "config_pipeline_par2_sf38_s4.yaml"}


def line_diff(a, b):
    """(removed, added) lines of a -> b."""
    rem, add = [], []
    for d in difflib.ndiff(a.split("\n"), b.split("\n")):
        if d.startswith("- "):
            rem.append(d[2:])
        elif d.startswith("+ "):
            add.append(d[2:])
    return rem, add


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()


class ConfigReader(unittest.TestCase):
    def test_real_files(self):
        exp = {"1": (2048, 50.0, 250.0, 1.0, 957), "225": (2496, 60.0, 300.0, 2.25, 1167), "4": (2496, 120.0, 600.0, 4.0, 1108)}
        for c, (nx, lam, inj, s2, ev) in exp.items():
            v = par2.cfg_values(fx.find_apps_cfg(REAL[c]))
            self.assertEqual((v["nx"], v["dx"], v["lam"], v["inject_x"], v["sigma2"], v["snapshots_every"]), (nx, 5.0, lam, inj, s2, ev))
            self.assertEqual(v["Lx"], nx * 5.0)
            self.assertEqual((v["dt"], v["field_seed"], v["transport_seed"], v["nr"], v["sample_every"]), (12.5, 7001, 123456789, 10, 10))
            self.assertEqual(v["inject_x"], 5 * lam)

    def test_section_aware_and_comments(self):
        t = "a:\n  seed: 1  # c\nb:\n  seed: 2\n  c:\n    seed: 3\n# seed: 9\n"
        self.assertEqual(par2.parse_cfg(t), {"a.seed": "1", "b.seed": "2", "b.c.seed": "3"})
        self.assertEqual(par2.set_cfg_key(t, "b.seed", 5), "a:\n  seed: 1  # c\nb:\n  seed: 5\n  c:\n    seed: 3\n# seed: 9\n")
        self.assertEqual(par2.set_cfg_key(t, "a.seed", 7).split("\n")[1], "  seed: 7  # c")
        with self.assertRaises(ValueError):
            par2.set_cfg_key(t, "a.nope", 1)
        with self.assertRaises(ValueError):
            par2.parse_cfg("a:\n  k: 1\n  k: 2\n")
        with self.assertRaises(ValueError):
            par2.cfg_values("grid:\n  nx: 3\n")


class Generators(unittest.TestCase):
    def test_pilot_real_files_only_intended_lines_change(self):
        for c, name in REAL.items():
            src = fx.find_apps_cfg(name)
            out = par2.make_pilot_config(src, c)
            rem, add = line_diff(src, out)
            self.assertEqual(len(rem), 4)  # title + three keys
            self.assertTrue(rem[0].startswith("# MacroFlow3D") and "SF-38 (s%s)" % c in rem[0])
            self.assertEqual(rem[1:], ["  seed: 7001", "    NR: 10", "  output_dir: ./output/sf38_par2_s%s" % c])
            keys = [a for a in add if not a.startswith("#")]
            self.assertEqual(keys, ["  seed: 9001", "    NR: 20", "  output_dir: raw/par2_s%s" % c])
            self.assertIn("SF-39 pilot (s%s)" % c, "\n".join(add))  # updated header; every other added line is a comment
            v = par2.cfg_values(out)
            self.assertEqual((v["field_seed"], v["nr"], v["output_dir"], v["transport_seed"]), (9001, 20, "raw/par2_s%s" % c, 123456789))
            # the non-comment lines are the source's, in order, but for the three keys
            a = [l for l in src.split("\n") if not l.lstrip().startswith("#")]
            b = [l for l in out.split("\n") if not l.lstrip().startswith("#")]
            self.assertEqual(len(a), len(b))
            self.assertEqual(sum(x != y for x, y in zip(a, b)), 3)

    def test_original_amplitudes_header_is_byte_unchanged(self):
        # the generation for s1 / s225 / s4 is the one that produced the committed configs: a golden of the inserted header block
        for c, name in REAL.items():
            src = fx.find_apps_cfg(name)
            out = par2.make_pilot_config(src, c).split("\n")
            self.assertEqual(out[3:9], [
                "# SF-39 (docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/preregistration.md, sec. 1): copia de",
                "# apps/config_pipeline_par2_sf38_s%s.yaml generada por `python3 -m sf39.par2 mkpilot`; solo cambian" % c,
                "# stochastic.seed (7001 -> 9001), analysis.macrodispersion.NR (10 -> 20, piloto) y output.output_dir (-> raw/par2_s%s)." % c,
                "# El texto SF-38 que sigue (semillas 7001-7010, NR = 10, piloto N_R = 10) es heredado y queda superado por esas claves.",
                "#", src.split("\n")[3]])
            self.assertEqual(out[1], "# MacroFlow3D \u2014 Par2 RWPT baseline, SF-39 pilot (s%s)" % c)

    def test_pilot_s025_from_the_s1_source_only_intended_lines_change(self):
        src = fx.find_apps_cfg(REAL["1"])
        out = par2.make_pilot_config(src, "025")
        rem, add = line_diff(src, out)
        self.assertEqual(len(rem), 5)  # title + four keys
        self.assertTrue(rem[0].startswith("# MacroFlow3D") and "SF-38 (s1)" in rem[0])
        self.assertEqual(rem[1:], ["  sigma2: 1.0", "  seed: 7001", "    NR: 10", "  output_dir: ./output/sf38_par2_s1"])
        self.assertEqual([a for a in add if not a.startswith("#")],
                         ["  sigma2: 0.25", "  seed: 9001", "    NR: 20", "  output_dir: raw/par2_s025"])
        head = "\n".join(a for a in add if a.startswith("#"))
        self.assertIn("SF-39 pilot (s025)", head)
        self.assertIn("apps/config_pipeline_par2_sf38_s1.yaml", head)  # names the s1 source
        self.assertIn("stochastic.sigma2 (1.0 -> 0.25", head)  # and the variance change
        v = par2.cfg_values(out)
        self.assertEqual((v["sigma2"], v["field_seed"], v["nr"], v["output_dir"], v["transport_seed"]), (0.25, 9001, 20, "raw/par2_s025", 123456789))
        a = [l for l in src.split("\n") if not l.lstrip().startswith("#")]
        b = [l for l in out.split("\n") if not l.lstrip().startswith("#")]
        self.assertEqual(len(a), len(b))
        self.assertEqual(sum(x != y for x, y in zip(a, b)), 4)  # everything else byte-identical (domain, dt, injection, snapshots)

    def test_ext_s025(self):
        pilot = par2.make_pilot_config(fx.find_apps_cfg(REAL["1"]), "025")
        ext = par2.make_ext_config(pilot, 57)
        rem, add = line_diff(pilot, ext)
        self.assertEqual(rem, ["  seed: 9001", "  seed: 123456789", "    NR: 20", "  output_dir: raw/par2_s025"])
        self.assertEqual([a for a in add if not a.startswith("#")],
                         ["  seed: 9021", "  seed: 123476789", "    NR: 37", "  output_dir: raw/par2_s025_ext"])
        self.assertEqual(par2.check_ext_consistency(pilot, ext), [])
        v = par2.cfg_values(ext)
        self.assertEqual((v["sigma2"], v["field_seed"], v["transport_seed"], v["nr"], v["output_dir"]),
                         (0.25, 9021, 123476789, 37, "raw/par2_s025_ext"))

    def test_s025_rules(self):
        self.assertEqual(par2.AMPLITUDES, ("025", "1", "225", "4"))
        with self.assertRaises(ValueError):  # the s025 source is the s1 config only
            par2.make_pilot_config(fx.find_apps_cfg(REAL["225"]), "025")
        with self.assertRaises(ValueError):  # and s1 still needs its own title
            par2.make_pilot_config(fx.find_apps_cfg(REAL["225"]), "1")
        pilot = par2.make_pilot_config(fx.find_apps_cfg(REAL["1"]), "025")
        with self.assertRaises(ValueError):
            par2.make_ext_config(pilot.replace("raw/par2_s025", "raw/par2_s0250"), 30)

    def test_ext_from_real_pilot(self):
        for c, name in REAL.items():
            pilot = par2.make_pilot_config(fx.find_apps_cfg(name), c)
            ext = par2.make_ext_config(pilot, 57)
            rem, add = line_diff(pilot, ext)
            self.assertEqual(rem, ["  seed: 9001", "  seed: 123456789", "    NR: 20", "  output_dir: raw/par2_s%s" % c])
            self.assertEqual([a for a in add if not a.startswith("#")],
                             ["  seed: 9021", "  seed: 123476789", "    NR: 37", "  output_dir: raw/par2_s%s_ext" % c])
            hdr = [a for a in add if a.startswith("# SF-39 Par2 EXTENSION")]
            self.assertEqual(len(hdr), 1)
            self.assertTrue(ext.startswith(hdr[0] + "\n"))
            self.assertEqual(par2.check_ext_consistency(pilot, ext), [])
            v = par2.cfg_values(ext)
            self.assertEqual((v["field_seed"], v["transport_seed"], v["nr"], v["output_dir"]), (9021, 123476789, 37, "raw/par2_s%s_ext" % c))

    def test_rules(self):
        pilot = par2.make_pilot_config(fx.find_apps_cfg(REAL["1"]), "1")
        for n in (20, 5):
            with self.assertRaises(ValueError):
                par2.make_ext_config(pilot, n)
        with self.assertRaises(ValueError):
            par2.make_ext_config(fx.SYN_CFG.replace("raw/par2_s1", "out/x"), 30)
        with self.assertRaises(ValueError):
            par2.make_pilot_config(fx.find_apps_cfg(REAL["1"]), "3")
        with self.assertRaises(ValueError):  # title names s1 but c = 4
            par2.make_pilot_config(fx.find_apps_cfg(REAL["1"]), "4")
        ext = par2.make_ext_config(pilot, 30)
        bad = par2.check_ext_consistency(pilot, ext.replace("seed: 123476789", "seed: 123456789").replace("nx: 2048", "nx: 2000"))
        self.assertEqual(len(bad), 2)


class Series(TmpCase):
    def test_uc_linear_mean_x(self):
        t = 125.0 * np.arange(1, 41)
        us = [0.5, 0.7, 0.9]
        ss = []
        for r, u in enumerate(us):
            fx.write_series(self.d, r, t, mean_x=lambda tt, u=u: fx.INJ + u * tt)
            ss.append(par2.read_series(os.path.join(self.d, "stats", "realization_%04d_timeseries.csv" % r)))
        tb = 1234.5  # between samples: linear interpolation of a linear function is exact
        uc, per = par2.uc_mean(ss, tb, fx.INJ)
        self.assertAlmostEqual(uc, 0.7, places=12)
        for p, u in zip(per, us):
            self.assertAlmostEqual(p, u, places=12)
        self.assertAlmostEqual(par2.tau_b_star(uc, tb, 20.0), 0.7 * tb / 20.0, places=12)
        with self.assertRaises(ValueError):  # outside the series: error, never clipped
            par2.uc_mean(ss, 125.0 * 41, fx.INJ)
        with self.assertRaises(ValueError):
            par2.uc_mean(ss, 100.0, fx.INJ)
        self.assertAlmostEqual(par2.uc_mean(ss, tb, fx.INJ + 10.0)[0], 0.7 - 10.0 / tb, places=12)

    def test_alpha_linear_variance_exact(self):
        t = 125.0 * np.arange(1, 41)
        fx.write_series(self.d, 0, t, var=(lambda tt: 3.0 * tt + 11.0, lambda tt: 0.6 * tt + 2.0, 5.0))
        s = par2.read_series(os.path.join(self.d, "stats", "realization_0000_timeseries.csv"))
        lam, U = 20.0, 0.8
        t_lo, t_hi = par2.window(2.0, lam, U)  # tau_c lam / U = 50
        self.assertAlmostEqual(t_hi, 50.0, places=12)
        self.assertAlmostEqual(t_lo, 37.5, places=12)
        with self.assertRaises(ValueError):  # window before the first sample (t = 125)
            par2.alpha_window(s, lam, U, t_lo, t_hi)
        t_lo, t_hi = par2.window(40.0, lam, U)  # [750, 1000]
        a = par2.alpha_window(s, lam, U, t_lo, t_hi)
        self.assertAlmostEqual(a["x"], 3.0 / (2 * lam * U), places=12)
        self.assertAlmostEqual(a["y"], 0.6 / (2 * lam * U), places=12)
        self.assertAlmostEqual(a["z"], 0.0, places=12)
        with self.assertRaises(ValueError):
            par2.alpha_window(s, lam, U, 5000.0, 5100.0)  # beyond t_last = 5000

    def test_interpolation_quadratic_known_answer(self):
        s = {"t": np.array([0.0, 10.0, 20.0]), "var_x": np.array([0.0, 100.0, 400.0]), "var_y": np.zeros(3), "var_z": np.zeros(3)}
        self.assertAlmostEqual(par2.interp(s, "var_x", 5.0), 50.0)
        self.assertAlmostEqual(par2.interp(s, "var_x", 15.0), 250.0)
        a = par2.alpha_window(s, 1.0, 1.0, 5.0, 15.0)
        self.assertAlmostEqual(a["x"], (250.0 - 50.0) / (2 * 10.0))

    def test_twin_series_format_generic(self):
        t = 125.0 * np.arange(1, 41)
        fx.write_series(self.d, 0, t, mean_x=lambda tt: 3.0 + 0.5 * tt, sub="stats_twin")
        s = par2.series_of({"dir": self.d, "r": 0}, "stats_twin")
        self.assertAlmostEqual(par2.uc_mean([s], 1000.0, 3.0)[0], 0.5, places=12)

    def test_bad_series(self):
        os.makedirs(self.d + "/stats")
        with open(self.d + "/stats/realization_0000_timeseries.csv", "w") as f:
            f.write("# format_version=2\nt,mean_x\n1,2\n")
        with self.assertRaises(ValueError):
            par2.read_series(self.d + "/stats/realization_0000_timeseries.csv")


class Breakthrough(TmpCase):
    def tb(self, r):
        return par2.t_b_last_clean({"dir": self.d, "r": r}, fx.LX, fx.LAM, fx.DT)

    def test_last_clean_known_answers(self):
        # threshold = 320 - 5 * 20 = 220 (inclusive: a particle AT 220 is dirty)
        fx.write_snapshots(self.d, 0, {10: [10.0, 20.0], 20: [100.0, 219.9], 30: [100.0, 220.0], 40: [100.0]})  # clean again at 40
        fx.write_snapshots(self.d, 1, {10: [10.0], 20: [11.0]})  # never reaches
        fx.write_snapshots(self.d, 2, {10: [250.0], 20: [10.0]})  # first already dirty
        fx.write_snapshots(self.d, 3, {9: [1.0], 10: [2.0], 100: [221.0]})
        fx.write_snapshots(self.d, 4, {10: [], 20: [300.0]})  # an empty snapshot is clean
        self.assertEqual(self.tb(0), (20, 250.0, 4, True))  # last snapshot before the FIRST dirty one: 20 * 12.5
        self.assertEqual(self.tb(1), (20, 250.0, 2, False))
        self.assertEqual(self.tb(2), (None, None, 2, True))
        self.assertEqual(self.tb(3), (10, 125.0, 3, True))
        self.assertEqual(self.tb(4), (10, 125.0, 2, True))
        x = par2.read_snapshot_x(os.path.join(self.d, "snapshots/r_0000/step_00000020.csv"))
        self.assertEqual(x.tolist(), [100.0, 219.9])
        with self.assertRaises(ValueError):
            self.tb(7)

    def test_tbmin_over_first_20_and_late_count(self):
        tbs = [40 + 10 * (g % 4) for g in range(26)]  # steps 40..70 -> t_b 500..875
        tbs[7] = 30  # pilot minimum: 30 * 12.5 = 375
        tbs[22] = 20  # extension realization earlier than the pilot minimum: must not enter t_b,min
        tbs[24] = 30  # equals the pilot minimum
        pd, ed = fx.make_tree(self.d, 20, 6, tbs, lambda g: 0.6)
        tab = par2.t_b_table(par2.pooled(pd, ed), fx.LX, fx.LAM, fx.DT)
        self.assertEqual([r["g"] for r in tab], list(range(26)))
        self.assertEqual((tab[7]["t_b"], tab[7]["n_snapshots"], tab[22]["t_b"]), (375.0, 5, 250.0))
        self.assertEqual(par2.t_b_min(tab), 375.0)
        # t_hi = 400: extension g=22 (250) and g=24 (375) are late; pilot g=7 (375) is late (reported separately)
        self.assertEqual(par2.late_count(tab, 400.0), (2, 6, 1))
        self.assertEqual(par2.late_count(tab, 375.0), (1, 6, 0))  # strict "<"
        with self.assertRaises(ValueError):
            par2.t_b_min(tab[:19])
        fx.write_snapshots(pd, 3, {10: [250.0]})  # a pilot realization with no clean snapshot: t_b,min undefined
        with self.assertRaises(ValueError):
            par2.t_b_min(par2.t_b_table(par2.pooled(pd, ed), fx.LX, fx.LAM, fx.DT))


class Pooled(TmpCase):
    def test_indexing_and_seeds(self):
        pd, ed = fx.make_tree(self.d, 20, 3, [40] * 23, lambda g: 0.6)
        es = par2.pooled(pd, ed)
        self.assertEqual([e["g"] for e in es], list(range(23)))
        self.assertEqual([e["r"] for e in es[18:23]], [18, 19, 0, 1, 2])
        self.assertEqual([e["dir"] for e in es[19:21]], [pd, ed])
        for e in es:  # one uninterrupted launch of N_R = 23
            self.assertEqual(e["field_seed"], 9001 + e["g"])
            self.assertEqual(e["injection_seed"], 123456789 + 1000 * e["g"])
        self.assertEqual(es[20]["field_seed"], 9021)  # == the extension config's stochastic.seed
        self.assertEqual(es[20]["injection_seed"], par2.TRANSPORT_SEED_EXT)
        self.assertEqual(len(par2.pooled(pd)), 20)

    def test_refuses_gaps(self):
        pd, ed = os.path.join(self.d, "p"), os.path.join(self.d, "e")
        t = 125.0 * np.arange(1, 5)
        for r in range(19):
            fx.write_series(pd, r, t)
        fx.write_series(ed, 0, t)
        with self.assertRaises(ValueError):  # pilot incomplete before the extension
            par2.pooled(pd, ed)
        fx.write_series(pd, 20, t)  # index gap inside a directory
        with self.assertRaises(ValueError):
            par2.pooled(pd)

    def test_wall_times(self):
        t = 125.0 * np.arange(1, 5)
        for r, mt in enumerate((1000.0, 1010.0, 1035.0)):
            fx.write_series(self.d, r, t)
            os.utime(os.path.join(self.d, "stats", "realization_%04d_timeseries.csv" % r), (mt, mt))
        self.assertEqual(par2.wall_times(self.d), [(0, 1000.0, None), (1, 1010.0, 10.0), (2, 1035.0, 25.0)])
        self.assertEqual(par2.wall_times(self.d, 990.0)[0][2], 10.0)


class FieldStat(unittest.TestCase):
    def test_parse(self):
        txt = ("some banner\n"
               "FIELDSTAT r=0 seed=9001 n=16777216 mean_lnK=1.234500000000e-03 var_lnK=9.912345678901e-01 Kmin=1.1000e-01 Kmax=9.2000e+00 Kmean=1.4000e+00\n"
               "FIELDSTAT r=1 seed=9002 n=16777216 mean_lnK=-2.0e-03 var_lnK=1.010000000000e+00 Kmin=1.2000e-01 Kmax=8.2000e+00 Kmean=1.5000e+00\n"
               "done\n")
        d = par2.parse_fieldstat(txt)
        self.assertEqual(sorted(d), [9001, 9002])
        self.assertEqual((d[9001]["r"], d[9001]["n"]), (0, 16777216))
        self.assertAlmostEqual(d[9001]["var_lnK"], 0.9912345678901)
        self.assertAlmostEqual(d[9002]["mean_lnK"], -2.0e-3)
        self.assertAlmostEqual(d[9002]["Kmax"], 8.2)
        with self.assertRaises(ValueError):
            par2.parse_fieldstat(txt + txt.splitlines()[1] + "\n")
        self.assertEqual(par2.parse_fieldstat("nothing\n"), {})


class Cli(TmpCase):
    def run_cli(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(par2.main(list(args)), 0)
        return buf.getvalue()

    def test_pipeline_tb_uc_alpha(self):
        tbs = [40 + 10 * (g % 3) for g in range(23)]
        tbs[5] = 30  # t_b,min = 375
        tbs[21] = 20  # late extension realization (250)
        uof = lambda g: 0.5 + 0.01 * (g % 2)
        pd, ed = fx.make_tree(self.d, 20, 3, tbs, uof, lambda g: (2.0, 0.4, 0.1))
        cfg = os.path.join(self.d, "cfg.yaml")
        with open(cfg, "w") as f:
            f.write(fx.SYN_CFG)
        out = self.run_cli("tb", "--cfg", cfg, pd, ed)
        self.assertIn("TB g=5 step=30 t_b=375 n_snapshots=5 reached=1", out)
        self.assertIn("TB_MIN t_b_min=375 n_pilot=20 Lx=320 threshold_x=220", out)
        uc = self.run_cli("uc", "--cfg", cfg, "--first", "20", pd, ed)
        u20 = np.mean([uof(g) for g in range(20)])
        self.assertIn("UC U_c=%.12e" % u20, uc)
        self.assertIn("tau_b_star=%.12e" % (u20 * 375.0 / 20.0), uc)
        u23 = np.mean([uof(g) for g in range(23)])  # final U_c over all 23
        tau_c = 700.0 * u23 / 20.0  # W_c = [525, 700]
        al = self.run_cli("alpha", "--cfg", cfg, "--tau-c", repr(float(tau_c)), pd, ed)
        mean = al.split("ALPHA_MEAN")[1]
        self.assertIn("alpha_x=%.12e" % (2.0 / (2 * 20.0 * u23)), mean)
        self.assertIn("alpha_T=%.12e" % (0.5 * (0.4 + 0.1) / (2 * 20.0 * u23)), mean)
        late_beyond = sum(tbs[g] * 12.5 < 700.0 for g in range(20, 23))
        late_pilot = sum(tbs[g] * 12.5 < 700.0 for g in range(20))
        self.assertEqual((late_beyond, late_pilot), (2, 15))
        self.assertIn("LATE beyond_pilot=%d/3 pilot=%d" % (late_beyond, late_pilot), al)

    def test_mk_commands(self):
        src = os.path.join(self.d, "s1.yaml")
        with open(src, "w") as f:
            f.write(fx.find_apps_cfg(REAL["1"]))
        pil = self.run_cli("mkpilot", src, "1")
        self.assertEqual(pil, par2.make_pilot_config(fx.find_apps_cfg(REAL["1"]), "1"))
        pf = os.path.join(self.d, "pilot.yaml")
        with open(pf, "w") as f:
            f.write(pil)
        self.assertEqual(self.run_cli("mkext", pf, "100"), par2.make_ext_config(pil, 100))
        pil25 = self.run_cli("mkpilot", src, "025")  # s025 is generated from the s1 source file
        self.assertEqual(pil25, par2.make_pilot_config(fx.find_apps_cfg(REAL["1"]), "025"))
        pf25 = os.path.join(self.d, "pilot25.yaml")
        with open(pf25, "w") as f:
            f.write(pil25)
        self.assertEqual(self.run_cli("mkext", pf25, "100"), par2.make_ext_config(pil25, 100))
        self.assertIn("raw/par2_s025_ext", self.run_cli("mkext", pf25, "100"))
        fl = os.path.join(self.d, "fs.log")
        with open(fl, "w") as f:
            f.write("FIELDSTAT r=0 seed=9001 n=8 mean_lnK=1.0e-03 var_lnK=9.9e-01 Kmin=1.0e-01 Kmax=9.0e+00 Kmean=1.4e+00\n")
        self.assertIn("seed=9001 r=0 n=8 var_lnK=9.900000000000e-01", self.run_cli("fieldstat", fl))


if __name__ == "__main__":
    unittest.main()
