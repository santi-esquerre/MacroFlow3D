"""Dry-run tests of run_sf39.sh (`SF39_DRYRUN=1`: every command is printed, nothing is run, written or verified): selectors, the s025 arm
lists, the checkpoint cases, the tree-tied check ids and the budget requirements of `extend`.  Seconds; needs bash and python3."""
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
ART = SCRIPTS.parent
SH = SCRIPTS / "run_sf39.sh"
REV = "0123456789abcdef0123456789abcdef01234567"
KEYS = dict(N_narrow=20, N_twin=20, N_hmax=20, N_R=20, N_A=20, N_run=20, par2_NR=20)
ORIGINAL_BUDGET = {"s1": dict(KEYS, N_R=50, N_run=50, par2_NR=50), "s225": dict(KEYS, N_twin=26, N_R=62, N_run=62, par2_NR=62),
                   "s4": dict(KEYS, N_R=71, N_run=71, par2_NR=71)}


class Dry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out_root = os.path.join(cls.tmp.name, "root")
        cls.budget = os.path.join(cls.tmp.name, "budget.json")
        cls.write_budget(ORIGINAL_BUDGET)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def write_budget(cls, obj):
        with open(cls.budget, "w") as f:
            json.dump(obj, f)

    def run_sh(self, *args, budget=True):
        env = dict(os.environ, SF39_DRYRUN="1", SF39_COMMIT=REV, SF39_OUT_ROOT=self.out_root)
        env.pop("PYTHONPATH", None)
        if budget:
            env["SF39_BUDGET"] = self.budget
        p = subprocess.run(["bash", str(SH), "build/v100-release", *args], capture_output=True, text=True, env=env)
        return p.returncode, p.stdout, p.stderr

    def cmds(self, *args):
        rc, out, err = self.run_sh(*args)
        self.assertEqual(rc, 0, err)
        return [ln for ln in out.splitlines() if ln.startswith("DRYRUN [")]

    @staticmethod
    def ids(lines):
        return [re.match(r"DRYRUN \[([^\]]+)\]", ln).group(1) for ln in lines]

    # -- selectors ------------------------------------------------------------------------------------------
    def test_selectors(self):
        for args in (("checks", "s025"), ("width", "s1"), ("analysis", "s025"), ("checkpoint", "s1"), ("checkpoint", "s4"),
                     ("checkpoint", "par2"), ("pilot", "s9"), ("extend", "s9"), ("pilot", "s0250")):
            rc, out, err = self.run_sh(*args)
            self.assertEqual(rc, 2, args)
            self.assertIn("usage:", err)
            self.assertIn("s025|s1|s225|s4|par2", err)
            self.assertNotIn("DRYRUN [", out)

    # -- checkpoint -----------------------------------------------------------------------------------------
    def test_checkpoint_without_selector_is_the_original_command(self):
        lines = self.cmds("checkpoint")
        want = "DRYRUN [checkpoint] .: python3 -m sf39.checkpoint --out-root %s --artifact %s" % (self.out_root, ART)
        self.assertEqual(lines, [want])  # no --amps / --subdir: the original three amplitudes, never four

    def test_checkpoint_s025_is_a_separate_case(self):
        lines = self.cmds("checkpoint", "s025")
        want = "DRYRUN [checkpoint_s025] .: python3 -m sf39.checkpoint --out-root %s --artifact %s --amps s025 --subdir s025" % (
            self.out_root, ART)
        self.assertEqual(lines, [want])

    # -- pilot ----------------------------------------------------------------------------------------------
    def test_pilot_s025_arms(self):
        lines = self.cmds("pilot", "s025")
        self.assertEqual(self.ids(lines), ["s025_A:s5001:n20", "s025_n05:s5001:n20", "s025_n10:s5001:n20", "s025_n20:s5001:n20",
                                           "s025_tw:s5001:n20"])
        for ln in lines:
            self.assertRegex(ln, r"--sigma2 1 --eps 0\.5 ")
            self.assertRegex(ln, r" --seed0 5001 --nr 20 --out %s/runs/s025_" % re.escape(self.out_root))
        self.assertIn("--hmax-div 8,16,32 ", lines[0])
        self.assertIn("--grid 2048,256,256 ", lines[0])
        self.assertIn("--grid 1024,64,64 ", lines[1])
        self.assertIn("--grid 4096,256,256 ", lines[3])
        self.assertIn("--grid 1024,128,128 ", lines[4])

    def test_pilot_default_order_and_par2(self):
        ids = self.ids(self.cmds("pilot"))
        arms = [i for i in ids if ":" in i]
        self.assertEqual([a.split("_")[0] for a in arms[::5]], ["s025", "s1", "s225", "s4"])
        self.assertEqual(ids[-8:], ["par2_s025", "fs_par2_s025", "par2_s1", "fs_par2_s1", "par2_s225", "fs_par2_s225", "par2_s4", "fs_par2_s4"])
        p2 = self.cmds("pilot", "par2")
        self.assertEqual(self.ids(p2), ids[-8:])
        self.assertTrue(p2[0].endswith("/configs/par2_s025.yaml"))
        self.assertTrue(p2[1].endswith("/configs/par2_s025.yaml --nr 20"))

    # -- checks ---------------------------------------------------------------------------------------------
    def test_checks_tie_the_tree_dependent_ids_to_the_revision(self):
        ids = self.ids(self.cmds("checks"))
        rev = REV[:24]
        for name in ("unittest", "geometry", "smoke_reduce", "signtable"):
            self.assertIn("%s:%s" % (name, rev), ids)
            self.assertNotIn(name, ids)
        # regress (RR2) and the byte check (RR3 iii) keep their ids: their records are reused, not re-run
        for i in ("regress_head", "regress_compare", "byte_multi", "byte_single", "bytecheck"):
            self.assertIn(i, ids)
        solve = [i for i in ids if i.startswith("chk_")]
        self.assertEqual(len(solve), 18)
        self.assertEqual(solve[-4:], ["chk_s025_n05:s4901:n1", "chk_s025_n10:s4901:n1", "chk_s025_n20:s4901:n1", "chk_s025_tw:s4901:n1"])
        self.assertEqual(solve[:14], [i for i in solve[:14] if not i.startswith("chk_s025")])

    def test_checks_id_follows_the_revision_source(self):
        env = dict(os.environ, SF39_DRYRUN="1", SF39_COMMIT="abcdef0123456789-dirty extra", SF39_OUT_ROOT=self.out_root)
        p = subprocess.run(["bash", str(SH), "build/v100-release", "checks"], capture_output=True, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("DRYRUN [unittest:abcdef0123456789-dirty_e]", p.stdout)  # file-name characters, 24 at most

    # -- extend ---------------------------------------------------------------------------------------------
    def test_extend_needs_a_complete_entry_for_what_it_runs(self):
        try:
            # the committed situation: no s025 entry yet
            self.write_budget(ORIGINAL_BUDGET)
            for sel in ("s1", "s225", "s4"):
                self.assertTrue(self.cmds("extend", sel), sel)
            for args in (("extend", "s025"), ("extend",), ("extend", "par2")):
                rc, out, err = self.run_sh(*args)
                self.assertEqual(rc, 2, args)
                self.assertNotIn("DRYRUN [", out)
                self.assertIn("amplitude s025 missing", out)
                self.assertIn("refusing", err)
            # with the s025 entry everything resolves; s025 alone needs only its own entry
            self.write_budget({**ORIGINAL_BUDGET, "s025": KEYS})
            lines = self.cmds("extend", "s025")
            self.assertEqual(self.ids(lines), ["s025_A:s5001:n20", "s025_n05:s5001:n20", "s025_n10:s5001:n20", "s025_n20:s5001:n20",
                                               "s025_tw:s5001:n20"])  # N_run = N_A: no directory B
            self.assertTrue(self.cmds("extend"))
            # an entry that is present is validated even when another amplitude is selected
            self.write_budget({**ORIGINAL_BUDGET, "s025": dict(KEYS, N_narrow=19)})
            rc, out, err = self.run_sh("extend", "s1")
            self.assertEqual(rc, 2)
            self.assertIn("s025.N_narrow < 20", out)
            self.write_budget({**ORIGINAL_BUDGET, "s025": {k: v for k, v in KEYS.items() if k != "N_hmax"}})
            self.assertEqual(self.run_sh("extend", "s225")[0], 2)
        finally:
            self.write_budget(ORIGINAL_BUDGET)

    def test_extend_s025_directory_b_and_floor_30_is_s4_only(self):
        try:
            self.write_budget({**ORIGINAL_BUDGET, "s025": dict(KEYS, N_R=33, N_run=33, par2_NR=33)})
            ids = self.ids(self.cmds("extend", "s025"))
            self.assertEqual(ids[:2], ["s025_A:s5001:n20", "s025_B:s5021:n13"])
            self.assertEqual(ids[2:], ["s025_n05:s5001:n20", "s025_n10:s5001:n20", "s025_n20:s5001:n20", "s025_tw:s5001:n20"])
            self.write_budget({**ORIGINAL_BUDGET, "s4": dict(KEYS, N_R=40, N_run=40, par2_NR=40)})
            ids = self.ids(self.cmds("extend", "s4"))
            self.assertIn("s4_n16:s5001:n30", ids)  # the floor of 30 stays s4-only
            self.assertIn("s4_tw:s5001:n30", ids)
            self.write_budget({**ORIGINAL_BUDGET, "s1": dict(KEYS, N_R=40, N_run=40, par2_NR=40)})
            self.assertIn("s1_n05:s5001:n20", self.ids(self.cmds("extend", "s1")))
        finally:
            self.write_budget(ORIGINAL_BUDGET)


if __name__ == "__main__":
    unittest.main()
