"""Tests of sf39.campaign and the blinded checkpoint (sf39.checkpoint) on a fake campaign (pilot count 6 through the private `n_pilot` hook;
the CLI default is 20).  Run (tens of seconds): `cd A/scripts && python3 -m unittest discover -s . -t . -p 'test_checkpoint*.py'`."""
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from sf39 import campaign as cp
from sf39 import checkpoint as ck
from sf39 import io as sio
from sf39 import par2, power
from sf39.tests import fixtures_campaign as fc
from sf39.tests import fixtures_par2 as fp
from sf39.tests import fixtures_runner as fx

N = 6
SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../A/scripts
FORBIDDEN = re.compile(r"\b(mean|sd|std|margin|verdict|alpha\w*|qbar|U_c|variance)\b", re.I)


def close(tc, a, b, rtol=1e-11, atol=1e-13):
    tc.assertTrue(abs(a - b) <= atol + rtol * abs(b), "%.17g != %.17g" % (a, b))


def rd(path):
    with open(path) as f:
        return f.read()


def load(path):
    with open(path) as f:
        return json.load(f)


def run_quiet(*a, **k):
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = ck.run(*a, **k)
    return rc, err.getvalue()


class Campaign(unittest.TestCase):
    """One fake pilot, one checkpoint run, shared by the read-only tests."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = os.path.join(cls.tmp.name, "out")
        cls.art = os.path.join(cls.tmp.name, "A")
        os.makedirs(cls.out)
        fc.build_campaign(cls.out, cls.art, N, SCRIPTS)
        cls.rc, cls.err = run_quiet(cls.out, cls.art, n_pilot=N)
        cls.ckd = os.path.join(cls.out, "checkpoint")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # -- helpers ---------------------------------------------------------------------------------
    def U(self, arm, amp):
        _, qf = fc.arm_factors(arm, amp)
        return float(np.mean([0.97 + 0.01 * r for r in range(N)])) * qf

    def tau(self):
        tbm = fc.TB_FACTOR * (fx.XB - fx.X0) / fx.U0
        u_prod = float(np.mean([0.97 + 0.01 * r for r in range(N)]))
        return tbm, u_prod, u_prod * tbm / fx.ELL

    def expected_est(self, arm, amp, div=8):
        """{key: [per r]} of an arm/pass from the closed forms, with the frozen tau and the arm's own U (A: U_prod)."""
        tf, _ = fc.arm_factors(arm, amp)
        U = self.U(arm, amp)
        t_hi = self.tau()[2] * fx.ELL / U
        shift = fx.DISP[div] if cp.is_multipass(arm) else (0.0, 0.0)
        return {"alpha_TAM": [fc.expected_alpha_TAM(r, tf, U, t_hi) for r in range(N)],
                "alpha_TP": [fc.expected_alpha_TP(r, tf, shift) for r in range(N)]}

    # -- run ---------------------------------------------------------------------------------------
    def test_run_ok_and_files(self):
        self.assertEqual(self.rc, 0, self.err)
        for f in ("tau_b_star.json", "costs.json", "bytes.json", "checkpoint_report.txt", "checkpoint_report.json",
                  "sealed/estimates.csv", "sealed/nr_input.json", "sealed/nr_out.json", "sealed/nr_stdout.txt"):
            self.assertTrue(os.path.isfile(os.path.join(self.ckd, f)), f)
        # nothing outside checkpoint/ was written by the checkpoint
        self.assertEqual(sorted(os.listdir(self.out)), ["checkpoint", "par2", "pilot", "runs"])

    # -- freeze ------------------------------------------------------------------------------------
    def test_freeze_known_values(self):
        frz = load(os.path.join(self.ckd, "tau_b_star.json"))
        tbm, u_prod, tau = self.tau()
        tb0 = (fx.XB - fx.X0) / fx.U0
        for amp in cp.AMPS:
            e = frz["amplitudes"][amp]
            self.assertEqual(e["seeds"], [5001 + r for r in range(N)])
            for r, t in enumerate(e["t_b_r"]):
                close(self, t, tb0 * (fc.TB_FACTOR if r == fc.TB_R else 1.0))
            close(self, e["t_b_min"], tbm)
            close(self, e["U_prod"], u_prod)
            close(self, e["ell"], fx.ELL)
            close(self, e["tau_b_star"], tau)

    def test_gate5_frozen_times(self):
        frz = load(os.path.join(self.ckd, "tau_b_star.json"))
        tbm = self.tau()[0]
        for amp in cp.AMPS:
            g5 = frz["amplitudes"][amp]["gate5"]
            # Lagrangian twin: independent mean_x(t) of the uniform series, np.interp at t_b,min, anchor x0 = 5 ell
            us = []
            for r in range(N):
                g = fx.gen(r)
                mx = [float(np.mean([fx.X0 + fx.U0 * t + g["c1"][i] * np.sqrt(t) for i in range(fx.NP) if g["t_end"][i] > t]))
                      for t in fx.TIMES]
                us.append((np.interp(tbm, fx.TIMES, mx) - fx.X0) / tbm)
            uc = float(np.mean(us))
            close(self, g5["twin"]["U_c"], uc)
            close(self, g5["twin"]["anchor_x0"], fx.X0)
            close(self, g5["twin"]["tau_b_star"], uc * tbm / fx.ELL)
            # Par2: t_b,min = min step x dt over the first N, U_c from series mean_x = INJ + u t, anchor inject_x, / corr_length
            tbp = min(fc.PAR2_TB_STEPS[:N]) * fp.DT
            ucp = float(np.mean([fc.PAR2_U[g] * (1.0 + 0.1 * fc.AMP_F[amp]) for g in range(N)]))
            p = g5["par2"]
            close(self, p["t_b_min"], tbp)
            close(self, p["U_c"], ucp)
            close(self, p["corr_length"], fp.LAM)
            close(self, p["inject_x"], fp.INJ)
            close(self, p["tau_b_star"], ucp * tbp / fp.LAM)
            close(self, g5["tau_c"], min(uc * tbm / fx.ELL, ucp * tbp / fp.LAM))

    # -- sealed estimates and nr_rule ----------------------------------------------------------------
    def test_nr_input_known_paired_differences(self):
        nr_in = load(os.path.join(self.ckd, "sealed", "nr_input.json"))
        for amp in cp.AMPS:
            a = cp.arm_ids(amp)
            n05, n10, n20 = a["narrow"]
            e8, e16 = self.expected_est(a["A"], amp, 8), self.expected_est(a["A"], amp, 16)
            etw, e1, e2 = self.expected_est(a["tw"], amp), self.expected_est(n10, amp), self.expected_est(n20, amp)
            for key in power.ESTIMATORS:
                d = nr_in[amp][key]
                want = {"D_f": np.array(e2[key]) - np.array(e1[key]), "D_tw": np.array(e8[key]) - np.array(etw[key]),
                        "d16_8": np.array(e16[key]) - np.array(e8[key]), "prod": np.array(e8[key])}
                for k, v in want.items():
                    self.assertEqual(len(d[k]), N)
                    np.testing.assert_allclose(d[k], v, rtol=1e-9, atol=1e-12, err_msg="%s %s %s" % (amp, key, k))
                close(self, d["mean_2x"], float(np.mean(e2[key])), 1e-9)
                close(self, d["mean_h16"], float(np.mean(e16[key])), 1e-9)
        # sanity: the designed differences are non-trivial; the TAM h16-h8 difference is exactly 0 (same times_x) and TP is not
        d = nr_in["s1"]
        self.assertTrue(np.ptp(d["alpha_TAM"]["D_f"]) > 0 and np.max(np.abs(d["alpha_TAM"]["D_tw"])) > 0)
        np.testing.assert_allclose(d["alpha_TAM"]["d16_8"], 0.0, atol=1e-13)
        self.assertTrue(np.max(np.abs(d["alpha_TP"]["d16_8"])) > 1e-6)

    def test_estimates_csv_has_every_arm_pass_and_realization(self):
        import csv
        with open(os.path.join(self.ckd, "sealed", "estimates.csv")) as f:
            rows = list(csv.DictReader(f))
        got = {(r["arm"], int(r["hmax_div"]), int(r["r"])) for r in rows}
        want = set()
        for arm in cp.all_pilot_arms():
            for d in ((8, 16) if cp.is_multipass(arm) else (8,)):
                want |= {(arm, d, r) for r in range(N)}
        self.assertEqual(got, want)
        for r in rows:  # U rule: A passes share U_prod; every other arm its own pilot mean
            amp = [a for a in cp.AMPS if r["arm"].startswith(a + "_")][0]
            close(self, float(r["U"]), self.U(r["arm"], amp), 1e-12)
            for k in ("alpha_TAM", "alpha_TP", "alpha_LAM"):
                self.assertTrue(np.isfinite(float(r[k])))

    def test_nr_out_matches_power(self):
        nr_in = load(os.path.join(self.ckd, "sealed", "nr_input.json"))
        out = load(os.path.join(self.ckd, "sealed", "nr_out.json"))
        for amp in cp.AMPS:
            n, capped = power.amplitude_ns(nr_in[amp])
            self.assertEqual({k: out[amp][k] for k in n}, n)
            self.assertEqual(out[amp]["capped"], capped)
        lines = rd(os.path.join(self.ckd, "sealed", "nr_stdout.txt")).split("\n")
        self.assertIn("s1 N_narrow=%d" % out["s1"]["N_narrow"], lines)
        self.assertTrue(all(ck.NR_LINE.match(ln) for ln in lines if ln))
        self.assertEqual(sum(1 for ln in lines if ln), 3 * len(("N_narrow", "N_twin", "N_hmax", "N_R", "N_A", "N_run", "par2_NR")))

    # -- costs, bytes, projections -------------------------------------------------------------------
    def expected_costs(self):
        c = {}
        for i, arm in enumerate(cp.all_pilot_arms()):
            c[arm] = fc.arm_total_s(arm, N) / N
        for amp in cp.AMPS:
            a = cp.arm_ids(amp)
            c[a["B"]] = fc.A_TOTAL0 + (N - 1) / 2.0 - (fc.P16 + fc.P32)
            c[amp + "_par2"] = fc.PAR2_STEP_S + fc.FS_WALL
        c["s4_w24"] = c["s4_B"] * 0.5
        c["s4_w12"] = c["s4_tw"]
        return c

    def test_costs(self):
        got = load(os.path.join(self.ckd, "costs.json"))
        want = self.expected_costs()
        self.assertEqual(sorted(got), sorted(want))
        for k in want:
            close(self, got[k], want[k], 1e-12)
        self.assertEqual(cp.cell_count("s4_w24") * 2, cp.cell_count("s4_B"))  # the cell-count ratio of the task is 1/2

    def test_par2_cost_fallback_to_walls(self):
        raw = os.path.join(self.out, "par2", "raw", "par2_s1")
        files = [os.path.join(raw, "stats", "realization_%04d_timeseries.csv" % r) for r in range(N)]
        old = [os.stat(f).st_mtime for f in files]
        try:
            for f in files:
                os.utime(f, (1.0e9, 1.0e9))  # zero intervals: unavailable
            per, src = ck._par2_cost(self.out, "s1", N, ck.read_walls(self.out))
            close(self, per, 900.0 / N + fc.FS_WALL, 1e-12)
            self.assertIn("walls.tsv", src)
        finally:
            for f, t in zip(files, old):
                os.utime(f, (t, t))

    def test_bytes(self):
        got = load(os.path.join(self.ckd, "bytes.json"))
        for arm in cp.all_pilot_arms():
            tot = sum(fc.tree_bytes(os.path.join(cp.arm_dir(self.out, arm), "realizations", "r_%04d" % r)) for r in range(N))
            self.assertEqual(got[arm], int(round(tot / N)), arm)
        for amp in cp.AMPS:
            a = cp.arm_ids(amp)
            ex = []
            for r in range(N):
                R = os.path.join(cp.arm_dir(self.out, a["A"]), "realizations", "r_%04d" % r)
                ex.append(fc.tree_bytes(R) - fc.tree_bytes(R + "/hmax_16") - fc.tree_bytes(R + "/hmax_32"))
            self.assertEqual(got[a["B"]], int(round(float(np.mean(ex)))))
            self.assertLess(got[a["B"]], got[a["A"]])
            self.assertEqual(got[amp + "_par2"], int(round(fc.tree_bytes(os.path.join(self.out, "par2", "raw", "par2_s" + cp.PAR2_ID[amp])) / N)))
        self.assertEqual(got["s4_w24"], got["s4_B"])
        self.assertEqual(got["s4_w12"], got["s4_tw"])

    def test_projections_and_report(self):
        rep = load(os.path.join(self.ckd, "checkpoint_report.json"))
        out = load(os.path.join(self.ckd, "sealed", "nr_out.json"))
        costs, by = load(os.path.join(self.ckd, "costs.json")), load(os.path.join(self.ckd, "bytes.json"))
        done = {}
        for amp in cp.AMPS:
            for arm in cp.pilot_arms(amp) + (amp + "_par2",):
                done[arm] = N
        gpu_h, nb = 0.0, 0
        for amp in cp.AMPS:
            for arm, cnt in power.arm_counts(out[amp], amp).items():
                gpu_h += max(0, cnt - done.get(arm, 0)) * costs[arm] / 3600.0
                nb += cnt * by[arm]
        close(self, rep["projected_gpu_h"], gpu_h, 1e-12)
        close(self, rep["pilot_wall_over_gpu_busy"], fc.WALL_FACTOR, 1e-12)  # superseded wall records (last wins) are ignored
        close(self, rep["projected_wall_h"], gpu_h * fc.WALL_FACTOR, 1e-12)
        self.assertEqual(rep["projected_bytes"], nb)
        self.assertEqual(rep["n_pilot"], N)
        self.assertEqual(rep["gpu_s_per_realization"]["s4_B"]["kind"], "estimate")
        self.assertEqual(rep["gpu_s_per_realization"]["s4_w24"]["kind"], "estimate")
        self.assertEqual(rep["gpu_s_per_realization"]["s4_w12"]["kind"], "measured")
        self.assertEqual(rep["gpu_s_per_realization"]["s1_A"]["kind"], "measured")
        tbm, u_prod, tau = self.tau()
        for amp in cp.AMPS:
            close(self, rep["tau_b_star"][amp], tau, 1e-12)
            self.assertEqual(rep["N"][amp]["N_run"], out[amp]["N_run"])

    # -- blinding (RR4) ------------------------------------------------------------------------------
    def test_report_is_blinded(self):
        text = rd(os.path.join(self.ckd, "checkpoint_report.txt"))
        rep = load(os.path.join(self.ckd, "checkpoint_report.json"))
        self.assertIsNone(FORBIDDEN.search(text), FORBIDDEN.search(text))
        self.assertIsNone(FORBIDDEN.search(json.dumps(rep)), "forbidden key/word in report json")
        allowed_float = [("tau_b_star", None), ("gpu_s_per_realization", "value"), ("projected_gpu_h",), ("pilot_wall_over_gpu_busy",),
                         ("projected_wall_h",)]

        def walk(x, path):
            if isinstance(x, dict):
                for k, v in x.items():
                    walk(v, path + (k,))
            elif isinstance(x, list):
                for v in x:
                    walk(v, path + ("[]",))
            elif isinstance(x, float):
                ok = any(path[0] == a[0] and (len(a) == 1 or a[1] is None or path[-1] == a[1]) for a in allowed_float)
                self.assertTrue(ok, "float outside the allowed report fields: %s=%r" % ("/".join(path), x))

        walk(rep, ())
        # no sealed or frozen plume number leaks: tokens of the text never equal U, U_prod, U_c, qbar or any pilot estimate at 6 digits
        frz = load(os.path.join(self.ckd, "tau_b_star.json"))
        secret = []
        for amp in cp.AMPS:
            e = frz["amplitudes"][amp]
            secret += [e["U_prod"], e["gate5"]["twin"]["U_c"], e["gate5"]["par2"]["U_c"], e["t_b_min"], e["gate5"]["tau_c"],
                       e["gate5"]["twin"]["tau_b_star"], e["gate5"]["par2"]["tau_b_star"]]
        import csv
        with open(os.path.join(self.ckd, "sealed", "estimates.csv")) as f:
            for r in csv.DictReader(f):
                secret += [float(r[k]) for k in ("qbar", "U", "alpha_TAM", "alpha_TP", "alpha_LAM")]
        toks = set(re.findall(r"(?<![\w.])[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", text))
        leaks = sorted({"%.6g" % v for v in secret} & toks)
        self.assertEqual(leaks, [], "report contains values of the sealed/frozen set")
        # the tau_b* that IS allowed is there
        self.assertIn("TAU_B_STAR s1=%.12g" % rep["tau_b_star"]["s1"], text)

    # -- write-once, completeness ------------------------------------------------------------------------
    def test_rerun_identical_is_accepted(self):
        before = rd(os.path.join(self.ckd, "tau_b_star.json"))
        rc, err = run_quiet(self.out, self.art, n_pilot=N)
        self.assertEqual(rc, 0, err)
        self.assertEqual(rd(os.path.join(self.ckd, "tau_b_star.json")), before)
        self.assertIn("identical", err)

    def test_write_once_refuses_different_content(self):
        path = os.path.join(self.ckd, "tau_b_star.json")
        orig = rd(path)
        frz = json.loads(orig)
        frz["amplitudes"]["s1"]["tau_b_star"] *= 1.0000001
        altered = ck.dumps(frz)
        try:
            with open(path, "w") as f:
                f.write(altered)
            rc, err = run_quiet(self.out, self.art, n_pilot=N)
            self.assertEqual(rc, 2)
            self.assertIn("CHECKPOINT_REFUSED", err)
            self.assertEqual(rd(path), altered)  # never overwritten
        finally:
            with open(path, "w") as f:
                f.write(orig)

    def test_incomplete_pilot_refused_and_listed(self):
        done = os.path.join(cp.arm_dir(self.out, "s4_n32"), "realizations", "r_0003", "DONE")
        os.rename(done, done + ".off")
        report = os.path.join(self.ckd, "checkpoint_report.txt")
        m0 = os.stat(report).st_mtime_ns
        try:
            rc, err = run_quiet(self.out, self.art, n_pilot=N)
            self.assertEqual(rc, 2)
            self.assertIn("CHECKPOINT_INCOMPLETE", err)
            self.assertIn("s4_n32: realizations not DONE: 3", err)
            self.assertEqual(os.stat(report).st_mtime_ns, m0)
        finally:
            os.rename(done + ".off", done)

    def test_incomplete_par2_and_walls_listed(self):
        raw = os.path.join(self.out, "par2", "raw", "par2_s225", "stats", "realization_0004_timeseries.csv")
        fs = os.path.join(self.out, "par2", "field_stats_par2_s4.txt")
        walls = os.path.join(self.out, "pilot", "walls.tsv")
        txt, w = rd(fs), rd(walls)
        os.rename(raw, raw + ".off")
        try:
            with open(fs, "w") as f:
                f.write("\n".join(txt.splitlines()[:-1]) + "\n")
            with open(walls, "w") as f:
                f.write("\n".join(ln for ln in w.splitlines() if not ln.startswith("s1_tw:")) + "\n")
            miss = ck.missing_items(self.out, self.art, N)
        finally:
            os.rename(raw + ".off", raw)
            with open(fs, "w") as f:
                f.write(txt)
            with open(walls, "w") as f:
                f.write(w)
        joined = "\n".join(miss)
        self.assertIn("par2_s225: series missing for r = 4", joined)
        self.assertIn("5 FIELDSTAT lines, expected 6", joined)
        self.assertIn("no record for s1_tw:s5001:n%d" % N, joined)
        self.assertEqual(len(miss), 3)
        self.assertEqual(ck.missing_items(self.out, self.art, N), [])


class DefaultPilotCount(unittest.TestCase):
    def test_cli_default_is_20_and_empty_root_writes_nothing(self):
        self.assertEqual(ck.N_PILOT, 20)
        with tempfile.TemporaryDirectory() as t:
            out, art = os.path.join(t, "out"), os.path.join(t, "A")
            os.makedirs(out)
            os.makedirs(art)
            env = dict(os.environ, PYTHONPATH=SCRIPTS)
            p = subprocess.run([sys.executable, "-m", "sf39.checkpoint", "--out-root", out, "--artifact", art], capture_output=True,
                               text=True, env=env)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("CHECKPOINT_INCOMPLETE", p.stderr)
            self.assertIn("s1_A: realizations not DONE: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19", p.stderr)
            self.assertEqual(os.listdir(out), [])  # no partial checkpoint

    def test_pilot_of_six_is_incomplete_for_the_cli(self):
        with tempfile.TemporaryDirectory() as t:
            out, art = os.path.join(t, "out"), os.path.join(t, "A")
            os.makedirs(out)
            fc.build_arm(out, "s1_A", "s1", 2)
            miss = ck.missing_items(out, art, ck.N_PILOT)
            self.assertIn("s1_A: realizations not DONE: " + ",".join(str(r) for r in range(2, 20)), miss)


class CampaignHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = cls.tmp.name
        fc.build_arm(cls.out, "s1_A", "s1", 4)
        fc.build_arm(cls.out, "s1_tw", "s1", 4)
        os.remove(os.path.join(cp.arm_dir(cls.out, "s1_A"), "realizations", "r_0001", "DONE"))
        # directory B: seeds continue after A (5005..), single pass
        fc.build_arm(cls.out, "s1_B", "s1", 2)
        for r in range(2):  # rename r_0000/1 of B to global seeds 5005/5006 by rewriting field.json
            p = os.path.join(cp.arm_dir(cls.out, "s1_B"), "realizations", "r_%04d" % r, "field.json")
            d = load(p)
            d["seed"] = 5005 + r
            with open(p, "w") as f:
                json.dump(d, f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tables(self):
        cp.check_tables()
        self.assertEqual(len(cp.all_pilot_arms()), 15)
        self.assertEqual(cp.pilot_arms("s4"), ("s4_A", "s4_n16", "s4_n32", "s4_n64", "s4_tw"))
        self.assertTrue(cp.is_multipass("s225_A") and not cp.is_multipass("s225_B") and not cp.is_multipass("s4_tw"))
        self.assertEqual(cp.pilot_walls_id("s1_A", 20), "s1_A:s5001:n20")

    def test_done_discovery_skips_unfinished_and_reads_seed(self):
        recs = cp.done_records(self.out, "s1_A")
        self.assertEqual([x.r for x in recs], [0, 2, 3])
        self.assertEqual([x.seed for x in recs], [5001, 5003, 5004])
        self.assertEqual(cp.done_records(self.out, "s1_n05"), [])  # absent arm directory
        with self.assertRaises(sio.MissingData):
            cp.pilot_records(self.out, "s1_A", 4)

    def test_pass_root_rule(self):
        a = cp.done_records(self.out, "s1_A")[0]
        tw = cp.done_records(self.out, "s1_tw")[0]
        self.assertEqual(cp.pass_root_of(a, 16), os.path.join(a.rdir, "hmax_16"))
        self.assertEqual(cp.pass_root_of(tw, 8), tw.rdir)
        with self.assertRaises(sio.ReduceError):
            cp.pass_root_of(tw, 16)
        self.assertTrue(cp.twin_series_path(self.out, a).endswith("s1_A/hmax_8/stats_twin/realization_0000_timeseries.csv"))
        self.assertTrue(cp.twin_series_path(self.out, tw).endswith("s1_tw/stats_twin/realization_0000_timeseries.csv"))

    def test_u_rules(self):
        a = cp.done_records(self.out, "s1_A")  # r 0, 2, 3
        tw = cp.done_records(self.out, "s1_tw")  # r 0..3, qbar x 1.01
        q = lambda r: 0.97 + 0.01 * r
        close(self, cp.u_standalone(a), np.mean([q(0), q(2), q(3)]))
        ua, ub, seeds = cp.u_paired(a, tw)
        self.assertEqual(seeds, [5001, 5003, 5004])
        close(self, ua, np.mean([q(0), q(2), q(3)]))
        close(self, ub, 1.01 * np.mean([q(0), q(2), q(3)]))  # the twin side averages ITS qbar over the pair's seeds only
        with self.assertRaises(sio.ReduceError):
            cp.u_paired(a, cp.done_records(self.out, "s1_B"))

    def test_production_pooling_by_seed(self):
        recs = cp.production_records(self.out, "s1")
        self.assertEqual([x.seed for x in recs], [5001, 5003, 5004, 5005, 5006])
        self.assertEqual([x.arm for x in recs], ["s1_A"] * 3 + ["s1_B"] * 2)
        self.assertEqual([x.seed for x in cp.production_records(self.out, "s1", n_r=4)], [5001, 5003, 5004])  # g < 4
        real = cp.realization(recs[3])  # B: single pass root
        self.assertEqual(real.pass_.root, recs[3].rdir)

    def test_estimates_and_pairing(self):
        recs = cp.done_records(self.out, "s1_A")
        tau = (0.9 * (fx.XB - fx.X0) / fx.U0) * 0.97 / fx.ELL
        U = cp.u_standalone(recs)
        e8 = cp.reduce_arm(recs, 8, tau, U)
        e16 = cp.reduce_arm(recs, 16, tau, U)
        self.assertEqual(sorted(e8), [5001, 5003, 5004])
        t_hi = tau * fx.ELL / U
        for x in recs:
            close(self, e8[x.seed]["alpha_TAM"], fc.expected_alpha_TAM(x.r, fc.AMP_F["s1"], U, t_hi), 1e-10)
            close(self, e8[x.seed]["alpha_TP"], fc.expected_alpha_TP(x.r, fc.AMP_F["s1"]), 1e-10)
            close(self, e16[x.seed]["alpha_TP"], fc.expected_alpha_TP(x.r, fc.AMP_F["s1"], fx.DISP[16]), 1e-10)
        sd, a, b = cp.paired(e16, e8, "alpha_TP")
        self.assertEqual(list(sd), [5001, 5003, 5004])
        np.testing.assert_allclose(a - b, [e16[s]["alpha_TP"] - e8[s]["alpha_TP"] for s in sd])
        # pairing restricts to common seeds
        only = {s: e8[s] for s in (5003, 5004)}
        self.assertEqual(list(cp.paired(only, e8, "alpha_TAM")[0]), [5003, 5004])
        # a window that does not fit stops with an error (never clipped)
        with self.assertRaises(sio.WindowError):
            cp.estimates(cp.realization(recs[0], 8), tau * 5.0, U)
        full = cp.estimates(cp.realization(recs[0], 8), tau, U, mode="full")
        self.assertTrue(np.isnan(full["alpha_TP"]) and np.isfinite(full["alpha_TAM"]))


if __name__ == "__main__":
    unittest.main()
