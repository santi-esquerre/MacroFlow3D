"""Fidelity of the data path on the FROZEN, already published SF-43 results (``A/results``): the addendum's loading + statistics must reproduce
numbers of ``results/analysis/summary.json``; the reproduction check must accept a self-comparison and reject a 1e-9 perturbation.  The committed
archives of three arms are extracted into a scratch directory outside the repository."""
import contextlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest

import sf43ana  # noqa: F401
from mcstat.gate5 import gate5_paired

from sf43add.ana import __main__ as cli
from sf43add.ana import rules
from sf43add.ana.data import LP, TP_FAR, TP_MAIN, Frozen, load_addendum, load_frozen, pair
from sf43add.ana.tests import synth

RED = os.path.join(synth.RESULTS, "reduced")
MANIFEST = os.path.join(RED, "freeze.reduced.sha256")
LEDGER = os.path.join(synth.RESULTS, "ledger.jsonl")
SUMMARY = os.path.join(synth.RESULTS, "analysis", "summary.json")


@unittest.skipUnless(os.path.isfile(SUMMARY) and os.path.isdir(os.path.join(RED, "tables")), "the committed SF-43 results are not in this checkout")
class FrozenResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = os.path.join(cls.tmp.name, "reduced")
        for arm in ("s4_g5p", "s4_l16", "s1_dist"):
            for kind in ("tables", "planes"):
                with tarfile.open(os.path.join(RED, kind, arm + ".tar.xz")) as t:
                    t.extractall(cls.dir, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        cls.add = load_addendum(synth.ADD_PATH)
        cls.fro = Frozen(cls.dir, MANIFEST, cls.add["frozen_reduced"]["manifest_sha256"])
        cls.P, cls.v, cls.ledger = load_frozen(synth.PREREG_PATH, cls.fro, LEDGER)
        with open(SUMMARY) as f:
            cls.summary = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_arms_have_their_planned_sizes_and_every_file_read_is_verified(self):
        self.assertEqual({a: self.v[a].N for a in self.v}, {"s4_g5p": 250, "s4_g5pol": 250, "s1_dist": 100})
        self.assertGreater(len(self.fro.verified), 60)
        self.assertEqual(self.fro.manifest_sha256, "250395ed5319052ddf1e5338e0d081d60610a41e85e1f66252b828daeb492dd6")

    def test_pollock_over_par2_at_sigma4_reproduces_the_published_decomposition(self):
        want = self.summary["pollock"]["decomposition"]["s4"]["alpha_LP"]["pollock/par2"]
        _, x, y, info = pair(self.v["s4_g5pol"], LP, self.v["s4_g5p"], LP)
        g = gate5_paired(x, y, level=0.90, method="student")
        self.assertEqual((x.size, info["n_common"]), (250, 250))
        self.assertEqual(want["n"], 250)
        for got, w in ((g.ratio, want["ratio"]), (g.ratio_lo, want["ratio_lo"]), (g.ratio_hi, want["ratio_hi"])):
            self.assertLessEqual(abs(got - w) / abs(w), 1e-9)
        # the same through the addendum's own paired_ratio (Student interval reported beside the bootstrap-t one)
        r = rules.ratio(self.v["s4_g5pol"], LP, self.v["s4_g5p"], LP)
        self.assertLessEqual(abs(r["ratio"] - want["ratio"]) / want["ratio"], 1e-9)
        self.assertLessEqual(abs(r["student_ci90"][0] - want["ratio_lo"]) / want["ratio_lo"], 1e-9)
        self.assertLessEqual(abs(r["student_ci90"][1] - want["ratio_hi"]) / want["ratio_hi"], 1e-9)
        self.assertAlmostEqual(want["ratio"], 1.164, places=3)

    def test_far_over_main_of_s1_dist_reproduces_the_published_point_estimate(self):
        want = self.summary["estimands"]["s1"]["distance"]["ratio_far_to_main"]["alpha_TP/disp"]["ratio"]
        self.assertEqual(want, 1.0208991755164611)
        r = rules.ratio(self.v["s1_dist"], TP_FAR, self.v["s1_dist"], TP_MAIN)
        self.assertEqual(r["n"], 100)
        self.assertLessEqual(abs(r["ratio"] - want) / want, 1e-9)

    def test_completeness_columns_are_read_from_the_frozen_par2_tables(self):
        c = rules.completeness(self.v["s4_g5p"])
        self.assertTrue(c["evaluable"])
        self.assertEqual(c["n"], 250)
        self.assertGreater(c["mean"], 0.9)
        self.assertLessEqual(c["mean"], 1.0)

    def campaign(self, perturb):
        """A scratch 'j1_base10' campaign: the frozen s4_g5p tables of seeds 434001-434010 copied under the new name."""
        d = os.path.join(self.tmp.name, "camp_%s" % perturb)
        seg = "u000226_s434001"
        dst = os.path.join(d, "reduced", "j1_base10", seg)
        os.makedirs(dst)
        src = os.path.join(self.dir, "s4_g5p", seg)
        for f in os.listdir(src):
            if not f.endswith(".planes.csv"):
                shutil.copy(os.path.join(src, f), os.path.join(dst, f.replace("s4_g5p__", "j1_base10__")))
        if perturb:
            p = os.path.join(dst, "j1_base10__%s__d25-94.csv" % seg)
            with open(p) as f:
                lines = f.read().split("\n")
            head = lines[0].split(",")
            row = lines[1].split(",")
            i = head.index("alpha_LP")
            row[i] = repr(float(row[i]) * (1 + 1e-9))
            lines[1] = ",".join(row)
            with open(p, "w") as f:
                f.write("\n".join(lines))
        return d

    def run_repro(self, camp):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = cli.main(["repro", "--campaign", camp, "--arm", "j1_base10", "--seeds", "434001-434010", "--frozen-reduced", self.dir,
                           "--frozen-manifest", MANIFEST])
        return rc, out.getvalue().strip()

    def test_repro_self_comparison_passes(self):
        rc, line = self.run_repro(self.campaign(False))
        self.assertEqual((rc, line), (0, "REPRO PASS 10 30 0 csv_sha_equal=10/10"))

    def test_repro_rejects_a_one_part_in_1e9_perturbation(self):
        rc, line = self.run_repro(self.campaign(True))
        self.assertEqual(rc, 1)
        self.assertEqual(line, "REPRO FAIL 10 30 1e-09 csv_sha_equal=10/10")


if __name__ == "__main__":
    unittest.main()
