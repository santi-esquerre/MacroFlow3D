"""Pruning safety tests (temporary directories only)."""
import csv
import glob
import hashlib
import os
import shutil
import json
import tempfile
import unittest

from sf42red import prune as pr
from sf42red import reduce as red
from sf42red.tests import synth


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def npys(root):
    return sorted(glob.glob(os.path.join(root, "**", "*.npy"), recursive=True))


class PruneBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.build(self.root)

    def build(self, root):
        """armA: r_0000..r_0002 (single pass); armB: r_0000 with passes hmax_8 and hmax_16; reduced table + manifest in root."""
        self.armA, self.armB = os.path.join(root, "armA"), os.path.join(root, "armB")
        for r in range(3):
            synth.write_pass(self.armA, r=r, seed=900 + r)
        synth.write_pass(self.armB, r=0, hmax=8)
        synth.write_pass(self.armB, r=0, hmax=16)
        self.table = os.path.join(root, "table.csv")
        red.reduce_all([self.armA, self.armB], [], [], None, None, 25.0, 94.0, None, self.table, root)
        self.retained = os.path.join(root, "retained.txt")
        with open(self.retained, "w") as f:
            f.write("# pre-registered subset\narmA/realizations/r_0001\n")
        self.log = os.path.join(root, "prune_log.csv")

    def run_prune(self, **kw):
        # the small fixtures retain fewer than decision G8's 5 per arm: min_retained=0 here, the G8 rule is tested in TestG8Minimum
        args = dict(arms=[self.armA, self.armB], retained_file=self.retained, table=self.table, root=self.root, out=lambda s: None,
                    min_retained=0)
        args.update(kw)
        return pr.prune(**args)


class TestPrune(PruneBase):
    def test_dry_run_deletes_nothing_and_lists_plan(self):
        before = npys(self.root)
        planned = self.run_prune()
        self.assertEqual(npys(self.root), before)
        self.assertFalse(os.path.exists(self.log))
        names = {os.path.relpath(p, self.root).split(os.sep)[1] for p in planned}
        self.assertEqual(names, {"realizations"})
        self.assertTrue(all("r_0001" not in p for p in planned))
        self.assertEqual(len(planned), len(before) - 9)  # retained r_0001 keeps its 9 arrays

    def test_execute_keeps_retained_and_all_non_npy_and_logs_sha256(self):
        everything = sorted(p for p in glob.glob(os.path.join(self.root, "**", "*"), recursive=True) if os.path.isfile(p))
        before_sha = {p: sha(p) for p in everything if p.endswith(".npy")}
        deleted = self.run_prune(execute=True, log=self.log)
        remaining = npys(self.root)
        self.assertEqual(sorted(remaining), sorted(p for p in before_sha if "r_0001" in p))
        for p in everything:
            if not p.endswith(".npy"):
                self.assertTrue(os.path.exists(p), p)  # CSV, JSON, DONE, manifest untouched
        with open(self.log) as f:
            rows = list(csv.DictReader(ln for ln in f if not ln.startswith("#")))
        self.assertEqual(len(rows), len(before_sha))  # every .npy recorded, kept or deleted
        for row in rows:
            p = os.path.join(self.root, row["path"])
            self.assertEqual(row["sha256"], before_sha[p])
            self.assertEqual(row["action"], "keep" if "r_0001" in p else "delete")
        self.assertEqual(len(deleted), len(before_sha) - len(remaining))
        with open(self.log) as f:
            self.assertTrue(f.read().rstrip().endswith("bytes"))

    def test_execute_needs_a_log(self):
        with self.assertRaises(pr.PruneRefused):
            self.run_prune(execute=True)
        self.assertEqual(len(npys(self.root)), 9 * 5)

    def test_guard_refuses_sf39_campaign_paths(self):
        guarded = tempfile.TemporaryDirectory()
        self.addCleanup(guarded.cleanup)
        g = os.path.join(os.path.realpath(guarded.name), "sf39_campaign")
        self.build(g)  # a complete, valid, reduced tree whose path contains the guard string
        n = len(npys(g))
        with self.assertRaises(pr.PruneRefused) as cm:
            pr.prune([os.path.join(g, "armA")], os.path.join(g, "retained.txt"), os.path.join(g, "table.csv"), g, execute=True,
                     log=os.path.join(self.root, "l.csv"), out=lambda s: None, min_retained=0)
        self.assertIn("sf39_campaign", str(cm.exception))
        self.assertEqual(len(npys(g)), n)
        # guard on the realpath of a symlink pointing into a guarded tree
        self.build(self.root)
        link = os.path.join(self.root, "innocent")
        os.symlink(g, link)
        with self.assertRaises(pr.PruneRefused):
            pr.prune([os.path.join(link, "armA")], self.retained, self.table, self.root, out=lambda s: None, min_retained=0)
        # guard on a retained entry
        bad = os.path.join(self.root, "ret2.txt")
        with open(bad, "w") as f:
            f.write("sf39_campaign/armA/realizations/r_0001\n")
        with self.assertRaises(pr.PruneRefused):
            self.run_prune(retained_file=bad)

    def test_verification_failure_refuses(self):
        before = npys(self.root)
        victim = os.path.join(self.armA, "realizations", "r_0002", "track.json")  # a recorded input changed after the reduction
        with open(victim, "a") as f:
            f.write(" ")
        with self.assertRaises(pr.PruneRefused) as cm:
            self.run_prune(execute=True, log=self.log)
        self.assertIn("differ", str(cm.exception))
        self.assertEqual(npys(self.root), before)

    def test_changed_array_refuses(self):
        a = os.path.join(self.armA, "realizations", "r_0000", "weights.npy")
        with open(a, "ab") as f:
            f.write(b"\0")
        with self.assertRaises(pr.PruneRefused):
            self.run_prune()

    def test_tampered_table_refuses(self):
        with open(self.table, "a") as f:
            f.write("\n")
        with self.assertRaises(pr.PruneRefused) as cm:
            self.run_prune()
        self.assertIn("manifest", str(cm.exception))

    def test_missing_manifest_refuses(self):
        os.remove(self.table + ".manifest.json")
        with self.assertRaises(pr.PruneRefused):
            self.run_prune()

    def test_table_must_cover_every_realization(self):
        table2 = os.path.join(self.root, "partial.csv")
        red.reduce_all([self.armA], [], [], None, None, 25.0, 94.0, None, table2, self.root)  # armB not reduced
        with self.assertRaises(pr.PruneRefused) as cm:
            self.run_prune(table=table2, execute=True, log=self.log)
        self.assertIn("no row for", str(cm.exception))
        self.assertEqual(len(npys(self.root)), 9 * 5)

    def test_unknown_retained_entry_refuses(self):
        with open(self.retained, "a") as f:
            f.write("armA/realizations/r_0042\n")
        with self.assertRaises(pr.PruneRefused):
            self.run_prune()

    def test_not_done_realization_refuses(self):
        os.remove(os.path.join(self.armA, "realizations", "r_0002", "DONE"))
        with self.assertRaises(pr.PruneRefused):
            self.run_prune()

    def test_retained_pass_directories_of_a_realization_are_kept_together(self):
        with open(self.retained, "w") as f:
            f.write("armB/realizations/r_0000\n")
        self.run_prune(execute=True, log=self.log)
        self.assertEqual(len(npys(os.path.join(self.armB))), 18)  # both passes of the retained realization
        self.assertEqual(npys(self.armA), [])


class TestG8Minimum(PruneBase):
    """Decision G8: at least 5 retained realizations per arm (an arm with fewer realizations is retained whole)."""

    def write_retained(self, *entries):
        with open(self.retained, "w") as f:
            f.write("".join(e + "\n" for e in entries))

    def refused(self, **kw):
        before = npys(self.root)
        with self.assertRaises(pr.PruneRefused) as cm:
            self.run_prune(execute=True, log=self.log, **kw)
        self.assertEqual(npys(self.root), before)
        self.assertFalse(os.path.exists(self.log))  # refused before anything is written or deleted
        return str(cm.exception)

    def test_default_minimum_is_the_g8_number(self):
        self.assertEqual(pr.MIN_RETAINED, 5)

    def test_empty_retained_list_refuses_by_default(self):
        self.write_retained("# nothing retained")
        self.assertIn("G8", self.refused(min_retained=pr.MIN_RETAINED))

    def test_default_refuses_a_partial_retained_subset(self):
        self.assertIn("G8", self.refused(min_retained=pr.MIN_RETAINED))  # armA keeps 1 of 3 (needs 3), armB keeps 0 of 1 (needs 1)

    def test_every_arm_needs_its_own_retained_realizations(self):
        self.write_retained("armA/realizations/r_0000", "armA/realizations/r_0001")
        msg = self.refused(min_retained=2)  # armA is fine, armB retains none
        self.assertIn("armB", msg)

    def test_arms_smaller_than_the_minimum_must_be_retained_whole_and_then_nothing_is_deleted(self):
        self.write_retained(*["armA/realizations/r_000%d" % i for i in range(3)], "armB/realizations/r_0000")
        deleted = self.run_prune(execute=True, log=self.log, min_retained=pr.MIN_RETAINED)
        self.assertEqual(deleted, [])
        self.assertEqual(len(npys(self.root)), 9 * 5)

    def test_a_met_minimum_deletes_only_the_rest(self):
        self.write_retained("armA/realizations/r_0000", "armA/realizations/r_0002", "armB/realizations/r_0000")
        deleted = self.run_prune(execute=True, log=self.log, min_retained=2)
        self.assertEqual(len(deleted), 9)  # armA r_0001 only
        self.assertTrue(all("r_0001" in p for p in deleted))


class TestSymlinksAndNaN(PruneBase):
    def refused(self, match):
        before = npys(self.root)
        with self.assertRaises(pr.PruneRefused) as cm:
            self.run_prune(execute=True, log=self.log)
        self.assertIn(match, str(cm.exception))
        self.assertEqual(npys(self.root), before)
        self.assertFalse(os.path.exists(self.log))

    def outside(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        return os.path.realpath(d.name)

    def test_symlinked_realization_directory_refuses(self):
        out = self.outside()
        R = os.path.join(self.armA, "realizations", "r_0002")
        moved = os.path.join(out, "r_0002")
        shutil.move(R, moved)
        os.symlink(moved, R)  # every recorded input still exists and matches: only the symlink guard can stop this
        self.assertEqual(len(npys(moved)), 9)
        self.refused("symlink")
        self.assertEqual(len(npys(moved)), 9)

    def test_symlinked_realizations_directory_refuses(self):
        out = self.outside()
        rd = os.path.join(self.armB, "realizations")
        moved = os.path.join(out, "realizations")
        shutil.move(rd, moved)
        os.symlink(moved, rd)
        self.refused("symlink")
        self.assertEqual(len(npys(moved)), 18)

    def test_symlinked_pass_directory_below_a_realization_refuses(self):
        out = self.outside()
        P = os.path.join(self.armB, "realizations", "r_0000", "hmax_8")
        moved = os.path.join(out, "hmax_8")
        shutil.move(P, moved)
        os.symlink(moved, P)
        self.refused("symlinked directory")
        self.assertEqual(len(npys(moved)), 9)

    def test_symlinked_npy_resolving_outside_refuses(self):
        out = self.outside()
        f = os.path.join(self.armA, "realizations", "r_0002", "weights.npy")
        moved = os.path.join(out, "weights.npy")
        shutil.move(f, moved)
        os.symlink(moved, f)
        self.refused("resolves outside")
        self.assertTrue(os.path.isfile(moved))

    def test_symlinked_npy_resolving_into_the_guarded_tree_refuses(self):
        out = self.outside()
        g = os.path.join(out, "sf39_campaign")
        os.makedirs(g)
        f = os.path.join(self.armA, "realizations", "r_0002", "weights.npy")
        moved = os.path.join(g, "weights.npy")
        shutil.move(f, moved)
        os.symlink(moved, f)
        self.refused("sf39_campaign")
        self.assertTrue(os.path.isfile(moved))

    def test_realization_whose_reduced_rows_are_all_nan_is_not_verified(self):
        src = os.path.relpath(os.path.join(self.armA, "realizations", "r_0002"), self.root)
        with open(self.table) as f:
            rows = list(csv.reader(f))
        head = rows[0]
        est = [head.index(e) for e in ("alpha_TP", "alpha_TAP", "alpha_LP")]
        n = 0
        for r in rows[1:]:
            if r[head.index("source")] == src and r[head.index("code")] == "consumer" and r[head.index("family")] == "plane":
                for i in est:
                    r[i] = "nan"
                n += 1
        self.assertGreater(n, 0)
        with open(self.table, "w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
        man_path = self.table + ".manifest.json"  # keep the manifest consistent with the edited table: only the NaN rule can refuse
        with open(man_path) as f:
            man = json.load(f)
        for t in man["tables"]:
            if t["path"] == os.path.basename(self.table):
                t["sha256"] = sha(self.table)
        with open(man_path, "w") as f:
            json.dump(man, f)
        self.refused("all NaN")


if __name__ == "__main__":
    unittest.main()
