import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from scipy import stats

from sf43cov import mcpath  # noqa: F401
from sf43cov import __main__ as cli
from sf43cov import procs

from mcstat import coverage as cov
from mcstat import covdist, covproc
from sf43ana import gate5 as g5

TAILS3 = np.array([0.0125, 0.025, 0.05])
SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return f(*a, **k)


def _student_proc(shift):
    """One-sided Student tests of the mean of ``dist + shift`` (true mean ``shift``) against H0 mean >= 0 / <= 0: exactly nominal for
    shift 0 on normal data, anti-conservative on the lo side for shift > 0."""
    def run(p, dist, rng, m):
        x = dist.draw(rng, (m, p["n"])) + shift
        ph, pl = g5.onesample_p(x, "student")
        return {"hi": (ph[:, None] <= TAILS3).sum(0).astype(np.int64), "lo": (pl[:, None] <= TAILS3).sum(0).astype(np.int64)}, {}
    return run


class Qualify(unittest.TestCase):
    def setUp(self):
        covproc.PROCS["t_exact"] = covproc.Proc(lambda p: TAILS3, _student_proc(0.0))
        covproc.PROCS["t_biased"] = covproc.Proc(lambda p: TAILS3, _student_proc(0.15))

    def tearDown(self):
        covproc.PROCS.pop("t_exact", None)
        covproc.PROCS.pop("t_biased", None)

    def test_exact_student_passes_and_a_biased_test_fails(self):
        with tempfile.TemporaryDirectory() as out:
            # 350 expected misses at the smallest tail: the false-fail probability of an exactly nominal row is about 0.003
            cells = [cov.make_cell(p, p, "", "normal", 100, {}, target=350, m=100, chunk=7000) for p in ("t_exact", "t_biased")]
            cov.init_plan(out, cells, 20261103, "test")
            cov.run(out, workers=1, log=lambda s: None)
            s = cov.summarize(out, write=False)
        v = {(x["label"], x["row"]): x for x in s["verdicts"]}
        self.assertTrue(v[("t_exact", "hi")]["qualified"])
        self.assertTrue(v[("t_exact", "lo")]["qualified"])
        self.assertTrue(v[("t_biased", "hi")]["qualified"])  # a positive shift makes the hi side conservative
        self.assertFalse(v[("t_biased", "lo")]["qualified"])  # and the lo side anti-conservative: 1.5 standard errors of shift at n = 100
        cell = {c["label"]: c for c in s["cells"]}
        for e in cell["t_exact"]["rows"]["lo"]:
            self.assertLess(abs(e["rate_over_tail"] - 1.0), 0.15)
            self.assertLessEqual(e["ucl_over_tail"], 1.25)
        self.assertGreater(min(e["rate_over_tail"] for e in cell["t_biased"]["rows"]["lo"]), 2.0)

    def test_chunking_and_restart_do_not_change_the_counts(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for out, chunk in ((a, 1400), (b, 14000)):
                cov.init_plan(out, [cov.make_cell("t_exact", "t_exact", "", "stress", 50, {}, target=175, m=100, chunk=chunk)], 5, "test")
            cov.run(a, workers=1, max_chunks=3, log=lambda s: None)  # interrupted
            cov.run(a, workers=1, log=lambda s: None)  # resumed
            cov.run(b, workers=1, log=lambda s: None)
            ca, cb = (cov.aggregate_cell(o, cov.load_plan(o)["cells"][0]) for o in (a, b))
        self.assertEqual({k: v.tolist() for k, v in ca["rows"].items()}, {k: v.tolist() for k, v in cb["rows"].items()})

    def test_a_different_plan_in_the_same_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as out:
            c = cov.make_cell("t_exact", "t_exact", "", "normal", 50, {}, target=175, m=100)
            cov.init_plan(out, [c], 1, "test")
            with self.assertRaises(ValueError):
                cov.init_plan(out, [c], 2, "test")


class Fieller(unittest.TestCase):
    def test_tails_are_the_holm_levels_and_their_halves(self):
        t = procs.fieller_tails({"family_m": 8, "fwer": 0.05})
        self.assertAlmostEqual(t.min(), 0.05 / 16)
        self.assertAlmostEqual(t.max(), 0.05)
        self.assertEqual(len(t), 12)  # 16 values, 4 coincide (0.025, 0.0125, 0.05/6, 0.05/8)
        for a in g5.holm_levels(8, 0.05):
            self.assertTrue(np.any(np.abs(t - a) < 1e-15) and np.any(np.abs(t - a / 2) < 1e-15))

    def test_tie_tolerance_counts_an_exact_rank_p_at_the_tail(self):
        t = procs.fieller_tails({"family_m": 8, "fwer": 0.05})
        p = np.array([240.0 / 33600.0, 241.0 / 33600.0])  # 240 / 33600 is the tail 0.05 / 7 up to representation
        c = procs._counts(p, t)
        i = int(np.argmin(np.abs(t - 0.05 / 7)))
        self.assertEqual(c[i], 1)

    def test_rows_and_exactness_of_student_on_normal_and_its_failure_under_stress(self):
        with tempfile.TemporaryDirectory() as out:
            rc = quiet(cli.main, ["plan", "--out", out, "--ns", "100", "--dists", "normal,stress", "--report-ns", "", "--B", "0", "--target", "350",
                           "--batch", "100"])
            self.assertEqual(rc, 0)
            plan = cov.load_plan(out)
            self.assertEqual([c["R"] for c in plan["cells"]], [112000, 112000])
            self.assertEqual(plan["cells"][0]["tails"][0], 0.003125)
            quiet(cli.main, ["run", "--out", out, "--ns", "100", "--dists", "normal,stress", "--report-ns", "", "--B", "0", "--target", "350",
                      "--batch", "100", "--workers", "1"])
            s = quiet(cli.summarize, out)
        self.assertEqual(set(s["cells"][0]["rows"]), {"student/hi", "student/lo", "hall/hi", "hall/lo"})
        rows = {(c["dist"], r): e for c in s["cells"] for r, e in c["rows"].items()}
        for r in ("student/hi", "student/lo"):
            self.assertTrue(all(e["pass"] for e in rows[("normal", r)]), r)  # exact on normal
        self.assertTrue(any(e["pass"] is False for r in ("student/hi", "student/lo") for e in rows[("stress", r)]))  # not under stress
        # the summary is readable by gate5: no bootstrap rows -> boott is not simulated -> not qualified; the selection never crashes
        q = g5.qualification(s, 8, 0.05)
        self.assertFalse(q["boott"][100]["qualified"])
        self.assertIn("not simulated", " ".join(q["boott"][100]["problems"]))
        self.assertIn("first_qualified_by_n", s["sf43cov"])
        self.assertTrue(s["sf43cov"]["nominal_false_fail"]["112000"]["p_any_one_row"] < 0.02)

    def test_workers_register_the_procedure_and_the_counts_do_not_depend_on_them(self):
        argv = ["--ns", "60", "--dists", "stress", "--report-ns", "", "--B", "199", "--target", "2", "--batch", "10", "--family-m", "4"]
        res = []
        for w in (1, 2):
            with tempfile.TemporaryDirectory() as out:
                quiet(cli.main, ["plan", "--out", out] + argv)
                quiet(cli.main, ["run", "--out", out, "--workers", str(w)] + argv)
                plan = cov.load_plan(out)
                agg = cov.aggregate_cell(out, plan["cells"][0])
                res.append({k: v.tolist() for k, v in agg["rows"].items()})
                self.assertIn("boott/hi", agg["rows"])
                self.assertEqual(plan["cells"][0]["params"]["B"], 199)
        self.assertEqual(res[0], res[1])

    def test_report_block_and_plan_hash_are_deterministic(self):
        ns = ["--ns", "100,150", "--report-ns", "150"]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            h = []
            for out in (a, b):
                quiet(cli.main, ["plan", "--out", out] + ns)
                h.append(cov.load_plan(out)["plan_hash"])
            cells = cov.load_plan(a)["cells"]
        self.assertEqual(h[0], h[1])
        self.assertEqual(len(cells), 3 * 2 + 2)
        self.assertEqual(sorted(c["dist"] for c in cells if c["report_only"]), ["mix:2.73:7.59", "mix:2.73:7.59-"])
        self.assertTrue(all(c["R"] == 56000 for c in cells))
        d = covdist.get_dist("mix:2.73:7.59-")
        self.assertAlmostEqual(d.skew, -2.73)
        self.assertAlmostEqual(d.ekurt, 7.59)


@unittest.skipUnless(shutil.which("bash"), "bash not available")
class RunScript(unittest.TestCase):
    def test_run_cov_sh_is_resumable_and_idempotent(self):
        args = ["--workers", "2", "--ns", "60", "--dists", "stress", "--report-ns", "", "--B", "199", "--target", "2", "--batch", "10",
                "--family-m", "4"]
        with tempfile.TemporaryDirectory() as out:
            first = subprocess.run(["bash", os.path.join(SCRIPTS, "sf43cov", "run_cov.sh"), out] + args, capture_output=True, text=True, timeout=240)
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            self.assertIn("smoke check passed", first.stdout)
            self.assertTrue(os.path.exists(os.path.join(out, "summary.json")))
            with open(os.path.join(out, "summary.json")) as f:
                a = f.read()
            second = subprocess.run(["bash", os.path.join(SCRIPTS, "sf43cov", "run_cov.sh"), out] + args, capture_output=True, text=True, timeout=240)
            self.assertEqual(second.returncode, 0, second.stderr[-2000:])
            self.assertIn("0 of 1 chunks pending", second.stderr + second.stdout)  # nothing is recomputed
            with open(os.path.join(out, "summary.json")) as f:
                self.assertEqual(a, f.read())
            bad = subprocess.run(["bash", os.path.join(SCRIPTS, "sf43cov", "run_cov.sh"), out] + args + ["--seed", "1"], capture_output=True, text=True, timeout=240)
            self.assertNotEqual(bad.returncode, 0)  # another plan in the same directory is refused
            self.assertIn("different plan", bad.stdout + bad.stderr)


if __name__ == "__main__":
    unittest.main()
