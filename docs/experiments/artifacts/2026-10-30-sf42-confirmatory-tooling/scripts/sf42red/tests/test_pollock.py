"""SF-43 `--pollock` option of the reducer and the matching rule of prune (synthetic arrays in the format of EnsembleRunner.cu)."""
import contextlib
import csv
import glob
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from sf42red import __main__ as cli
from sf42red import prune as pr
from sf42red import reduce as red
from sf42red.tests import synth

ELL, X0 = 0.25, 1.25
SKIP = {"code", "source", "note"}


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def slurp(path):
    with open(path, "rb") as f:
        return f.read()


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.out = os.path.join(self.root, "table.csv")

    def reduce(self, arm, out=None, **kw):
        out = out or self.out
        red.reduce_all([arm], [], [], None, None, 25.0, 94.0, None, out, self.root, **kw)
        return read(out)


class TestPollockRows(Base):
    def test_identical_arrays_give_the_consumer_rows(self):
        arm = os.path.join(self.root, "arm")
        proot = synth.write_pass(arm, ell=ELL, x0=X0, drift=(0.3, -0.1), miss={3: 5, 10: 6}, flux_weights=np.arange(1, 65.0))
        synth.write_pollock(proot, proot)
        rows = self.reduce(arm, pollock=True)
        cons = [r for r in rows if r["code"] == "consumer" and r["family"] == "plane" and r["weights"] != "expI"]
        poll = [r for r in rows if r["code"] == "pollock"]
        self.assertEqual((len(cons), len(poll)), (8, 8))  # flux/unif x pos/disp x perplane/common; no expI, no time family
        self.assertTrue(all(r["family"] == "plane" for r in poll))
        key = lambda r: (r["weights"], r["basis"], r["set"])  # noqa: E731
        cons = {key(r): r for r in cons}
        for p in poll:
            c = cons[key(p)]
            self.assertEqual(p["source"], "arm/realizations/r_0000/pollock")
            for col in p:
                if col not in SKIP:
                    self.assertEqual(p[col], c[col], (key(p), col))  # the same estimator functions: identical text, halves included
            self.assertNotEqual(p["alpha_TP"], "")
            self.assertNotEqual(p["alpha_TP_A"], "")
        self.assertEqual({r["seed"] for r in poll}, {"777"})
        self.assertEqual({r["hmax_div"] for r in poll}, {"-1"})
        planes = read(self.out[:-4] + ".planes.csv")
        self.assertEqual(sorted({(r["code"], r["weights"]) for r in planes}),
                         [("consumer", "expI"), ("consumer", "flux"), ("consumer", "unif"), ("pollock", "flux"), ("pollock", "unif")])

    def test_known_answer_differs_from_the_consumer(self):
        arm = os.path.join(self.root, "arm")
        proot = synth.write_pass(arm, ell=ELL, x0=X0, a_T=0.03, U0=2.0, c_tau=0.02)
        twin = synth.write_pass(os.path.join(self.root, "other"), ell=ELL, x0=X0, a_T=0.05, U0=3.0, c_tau=0.01)  # the Pollock cloud
        synth.write_pollock(proot, proot, planes_x23=np.load(os.path.join(twin, "planes_x23.npy")),
                            planes_t=np.load(os.path.join(twin, "planes_tau.npy")))
        rows = self.reduce(arm, pollock=True)
        for r in rows:
            if r["family"] != "plane" or r["weights"] == "expI":
                continue
            want = (0.05, 3.0) if r["code"] == "pollock" else (0.03, 2.0)
            self.assertAlmostEqual(float(r["alpha_TP"]), want[0], delta=1e-12, msg=r["code"])
            self.assertAlmostEqual(float(r["U_r"]), want[1], delta=1e-12, msg=r["code"])
            if r["code"] == "pollock":
                self.assertAlmostEqual(float(r["alpha_LP"]), 0.5 * 9.0 * 0.01 / ELL, delta=1e-12)

    def test_zero_weight_seed_and_nan_records_are_excluded(self):
        arm = os.path.join(self.root, "arm")
        proot = synth.write_pass(arm, ell=ELL, x0=X0)
        x, t = np.load(os.path.join(proot, "planes_x23.npy")), np.load(os.path.join(proot, "planes_tau.npy"))
        w = np.full(64, 1.0 / 63)
        w[5] = 0.0  # a seed with a non-positive RT0 weight: tracked, weight 0 -> outside the sets
        x[9, 4:, :] = np.nan  # stopped at plane 4 (status 15 in the bridge): NaN from there on
        t[9, 4:] = np.nan
        synth.write_pollock(proot, proot, weights=w, planes_x23=x, planes_t=t, status_plane=np.where(np.arange(64) == 9, 15, 0).astype(np.uint8))
        rows = [r for r in self.reduce(arm, pollock=True) if r["code"] == "pollock" and r["weights"] == "flux" and r["basis"] == "pos"]
        by = {r["set"]: r for r in rows}
        self.assertEqual(by["perplane"]["n_traced"], "63")
        self.assertEqual(by["perplane"]["n_first"], "63")  # first window plane (d = 25, index 2) before the stop
        self.assertEqual(by["perplane"]["n_last"], "62")
        self.assertEqual(by["common"]["n_common"], "62")
        self.assertEqual(by["perplane"]["set_differs"], "1")

    def test_option_off_is_byte_identical_whether_or_not_pollock_arrays_exist(self):
        a = os.path.join(self.root, "a")
        proot = synth.write_pass(os.path.join(a, "arm"), ell=ELL, x0=X0)
        red.reduce_all([os.path.join(a, "arm")], [], [], None, None, 25.0, 94.0, None, os.path.join(a, "t.csv"), a)
        synth.write_pollock(proot, proot)
        red.reduce_all([os.path.join(a, "arm")], [], [], None, None, 25.0, 94.0, None, os.path.join(a, "t2.csv"), a)
        self.assertEqual(slurp(os.path.join(a, "t.csv")), slurp(os.path.join(a, "t2.csv")))
        self.assertEqual(slurp(os.path.join(a, "t.planes.csv")), slurp(os.path.join(a, "t2.planes.csv")))
        self.assertNotIn("pollock", slurp(os.path.join(a, "t2.csv.manifest.json")).decode())
        self.assertNotIn(b"pollock", slurp(os.path.join(a, "t2.csv")))

    def test_multipass_labels_and_single_pass_label(self):
        arm = os.path.join(self.root, "arm")
        p8 = synth.write_pass(arm, ell=ELL, x0=X0, hmax=8)
        synth.write_pass(arm, ell=ELL, x0=X0, hmax=16)
        synth.write_pollock(os.path.dirname(p8), p8)
        rows = [r for r in self.reduce(arm, pollock=True) if r["code"] == "pollock"]
        self.assertEqual(sorted({r["hmax_div"] for r in rows}), ["16", "8"])
        self.assertEqual({r["source"] for r in rows}, {"arm/realizations/r_0000/pollock"})
        one = os.path.join(self.root, "one")
        proot = synth.write_pass(one, ell=ELL, x0=X0)
        synth.write_pollock(proot, proot)
        rows = [r for r in self.reduce(one, pollock=True, single_pass_hmax=8) if r["code"] == "pollock"]
        self.assertEqual({r["hmax_div"] for r in rows}, {"8"})

    def test_missing_or_inconsistent_pollock_data_is_an_error(self):
        arm = os.path.join(self.root, "arm")
        proot = synth.write_pass(arm, ell=ELL, x0=X0)
        with self.assertRaises(red.MissingData):
            self.reduce(arm, pollock=True)  # no pollock/ directory
        d = synth.write_pollock(proot, proot, seeds_equal=False)
        with self.assertRaisesRegex(red.ReduceError, "seeds_equal_tracker"):
            self.reduce(arm, pollock=True)
        shutil.rmtree(d)
        synth.write_pollock(proot, proot, planes=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        with self.assertRaisesRegex(red.ReduceError, "planes differ"):
            self.reduce(arm, pollock=True)
        shutil.rmtree(d)
        synth.write_pollock(proot, proot, planes_t=np.zeros((64, 7)))
        with self.assertRaisesRegex(red.ReduceError, "shapes"):
            self.reduce(arm, pollock=True)
        shutil.rmtree(d)
        synth.write_pollock(proot, proot, weights=np.ones(60))
        with self.assertRaisesRegex(red.ReduceError, "pollock_weights"):
            self.reduce(arm, pollock=True)

    def test_cli_flag(self):
        arm = os.path.join(self.root, "arm")
        proot = synth.write_pass(arm, ell=ELL, x0=X0)
        synth.write_pollock(proot, proot)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["reduce", "--consumer", arm, "--out", self.out, "--root", self.root, "--pollock"]), 0)
        self.assertIn("pollock", {r["code"] for r in read(self.out)})
        shutil.rmtree(os.path.join(proot, "pollock"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["reduce", "--consumer", arm, "--out", self.out, "--root", self.root]), 0)
        self.assertNotIn("pollock", {r["code"] for r in read(self.out)})
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["reduce", "--consumer", arm, "--out", self.out, "--root", self.root, "--pollock"]), 2)


class TestPrunePollock(Base):
    def build(self, pollock_rows):
        self.arm = os.path.join(self.root, "arm")
        for r in range(3):
            proot = synth.write_pass(self.arm, r=r, seed=900 + r)
            synth.write_pollock(proot, proot)
        self.reduce(self.arm, pollock=pollock_rows)
        self.retained = os.path.join(self.root, "retained.txt")
        with open(self.retained, "w") as f:
            f.write("arm/realizations/r_0001\n")

    def prune(self, execute=False, **kw):
        lines = []
        pr.prune([self.arm], self.retained, self.out, self.root, execute, os.path.join(self.root, "log.csv") if execute else None,
                 out=lines.append, min_retained=0, **kw)
        return "\n".join(lines)

    def remaining(self):
        return sorted(os.path.relpath(p, self.root) for p in glob.glob(os.path.join(self.arm, "**", "pollock_*.npy"), recursive=True))

    def test_pollock_arrays_are_deleted_only_when_their_rows_are_in_the_table(self):
        self.build(pollock_rows=True)
        text = self.prune(execute=True, keep_arrays=("seeds", "weights", "planes_x23", "planes_tau"))
        self.assertNotIn("unreduced", text)
        self.assertEqual([p for p in self.remaining() if "r_0000" in p or "r_0002" in p], [])  # non-retained: deleted
        self.assertEqual(len([p for p in self.remaining() if "r_0001" in p]), 7)  # retained: kept whole
        self.assertTrue(os.path.isfile(os.path.join(self.arm, "realizations", "r_0000", "pollock", "pollock.json")))  # JSON never touched

    def test_unreduced_pollock_arrays_are_kept_and_reported(self):
        self.build(pollock_rows=False)
        text = self.prune(execute=True)
        self.assertIn("14 unreduced pollock array(s) kept", text)
        self.assertEqual(len(self.remaining()), 21)  # all three realizations keep their seven Pollock arrays
        self.assertEqual(len(glob.glob(os.path.join(self.arm, "realizations", "r_0000", "*.npy"))), 0)  # the consumer ones are gone
        with open(os.path.join(self.root, "log.csv")) as f:
            rows = list(csv.DictReader(ln for ln in f if not ln.startswith("#")))
        self.assertTrue(all(r["action"] == "keep" for r in rows if "pollock_" in r["path"]))

    def test_all_nan_pollock_rows_do_not_license_deletion(self):
        self.build(pollock_rows=True)
        rows = read(self.out)
        for r in rows:
            if r["code"] == "pollock":
                r["alpha_TP"] = r["alpha_TAP"] = r["alpha_LP"] = "nan"
        with open(self.out, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        man = self.out + ".manifest.json"
        with open(man) as f:
            m = json.load(f)
        m["tables"][0]["sha256"] = red.sha256_file(self.out)
        with open(man, "w") as f:
            json.dump(m, f)
        self.assertIn("14 unreduced", self.prune())


if __name__ == "__main__":
    unittest.main()
