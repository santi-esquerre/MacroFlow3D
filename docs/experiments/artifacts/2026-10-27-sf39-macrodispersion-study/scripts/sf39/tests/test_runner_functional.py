"""Restart behaviour of run_sf39.sh with FAKE binaries (instant, no CUDA, nothing outside a temp directory): the campaign state after the
original runs (old unprefixed check ids, a recorded Par2 case) and a re-invocation with a new scripts tree.  Seconds; needs bash and python3.

  * the ids of the tree-dependent checks carry the scripts revision: a new revision re-runs exactly those, the old records stay, the same
    revision re-runs nothing, recorded regress / byte-check / chk_s1..s4 ids are skipped and the new chk_s025_* ids run;
  * a skipped Par2 field-statistics case never rewrites par2/field_stats_*.txt (and says why when that file is missing);
  * a real `checks` run refuses a scripts revision that does not identify the tree (unknown, blank, `-dirty`; an enclosing repository's
    HEAD is not this tree's) before the part starts: the part directory is left byte-identical;
  * the sealed files of the separate checkpoint case are hashed into checkpoint/sha256.txt like the original ones.
"""
import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sf39 import geometry

SCRIPTS = Path(__file__).resolve().parents[2]
SH = SCRIPTS / "run_sf39.sh"

GIT_MIN = (1, 7, 2)  # `git -c` (the oldest feature used here); exercised against git 1.8.3.1, the version of the V100 host


def git_support():
    """(ok, why not): git is installed and at least GIT_MIN.  The tests that need it are skipped, with this reason, otherwise."""
    exe = shutil.which("git")
    if not exe:
        return False, "needs git (not found)"
    out = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    if not m:
        return False, "needs git >= %s; cannot read its version from %r" % (".".join(map(str, GIT_MIN)), out.strip())
    if tuple(int(x) for x in m.groups()) < GIT_MIN:
        return False, "needs git >= %s, found %s" % (".".join(map(str, GIT_MIN)), ".".join(m.groups()))
    return True, ""


GIT_OK, GIT_WHY = git_support()

RUNNER = """#!/usr/bin/env bash
out=""; nr=0; s0=0
while [ $# -gt 0 ]; do
  case "$1" in --out) out=$2; shift ;; --nr) nr=$2; shift ;; --seed0) s0=$2; shift ;; esac
  shift
done
for ((r = 0; r < nr; r++)); do
  d=$out/realizations/$(printf 'r_%04d' $r)
  mkdir -p "$d"
  echo '{"verdict": "converged"}' > "$d/solver_b.json"
  echo "{\\"seed\\": $((s0 + r))}" > "$d/field.json"
  : > "$d/DONE"
done
echo "STATUS fake runner nr=$nr"
"""
PIPELINE = """#!/usr/bin/env bash
nr=$(awk '/^ *NR:/ {print $2}' "$1")
od=$(awk '/^ *output_dir:/ {print $2}' "$1")
mkdir -p "$od/stats"
for ((r = 0; r < nr; r++)); do : > "$od/stats/$(printf 'realization_%04d_timeseries.csv' $r)"; done
"""
FIELD_STATS = """#!/usr/bin/env bash
nr=20
while [ $# -gt 0 ]; do [ "$1" = "--nr" ] && nr=$2; shift; done
for ((r = 0; r < nr; r++)); do echo "FIELDSTAT r=$r seed=$((9001 + r)) n=8 mean_lnK=0 var_lnK=1.0e-01 Kmin=1 Kmax=2 Kmean=1"; done
"""
FAKE_TIME = """#!/usr/bin/env bash
[ "$1" = "-v" ] && shift
if [ "$1" = "-o" ]; then echo "fake time" > "$2"; shift 2; fi
exec "$@"
"""
FAKE_PY = """#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "unittest" ]; then echo "Ran 999 tests in 0.1s"; echo "OK"; exit 0; fi
if [ "$1" = "-m" ] && [ "$2" = "sf39.reduce" ]; then echo "fake reduce $3 ok"; exit 0; fi
if [ "$1" = "-m" ] && [ "$2" = "sf39.checkpoint" ]; then
  shift 2; out=""; sub=""
  while [ $# -gt 0 ]; do case "$1" in --out-root) out=$2; shift ;; --subdir) sub=$2; shift ;; esac; shift; done
  d=$out/checkpoint${sub:+/$sub}
  mkdir -p "$d/sealed"
  echo '{}' > "$d/tau_b_star${sub:+_$sub}.json"; echo "estimates $sub" > "$d/sealed/estimates.csv"; echo '{}' > "$d/sealed/nr_out.json"
  exit 0
fi
exec python3 "$@"
"""


def put(path, text, exe=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if exe:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


class Fake(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        self.bin, self.root, self.home = t / "bin", t / "root", t / "home"
        for name, text in (("lagrangian_ensemble", RUNNER), ("macroflow3d_pipeline", PIPELINE), ("par2_field_stats", FIELD_STATS),
                           ("inlet_slab", "#!/usr/bin/env bash\n"), ("lagrangian_medium_ab", "#!/usr/bin/env bash\n")):
            put(self.bin / name, text, exe=True)
        put(t / "faketime.sh", FAKE_TIME, exe=True)
        put(t / "fakepy.sh", FAKE_PY, exe=True)
        md5 = subprocess.run("md5sum lagrangian_ensemble lagrangian_medium_ab macroflow3d_pipeline inlet_slab par2_field_stats", shell=True,
                             cwd=self.bin, capture_output=True, text=True, check=True).stdout
        put(self.root / "build" / "binaries.md5", md5)
        put(self.home / "regress" / "refs" / "1e40cfe" / "out" / "REGRESS_DONE", "")
        self.seed_checks()
        self.seed_par2()

    def seed_checks(self):
        """The checks part as the campaign left it: old ids without a revision, rc 0, and the artifacts the part reads back."""
        c = self.root / "checks"
        put(c / "regress_head" / "out" / "REGRESS_DONE", "")
        rows = ["unittest", "geometry", "regress_head", "regress_compare", "bytecheck", "smoke_reduce", "signtable", "byte_multi", "byte_single"]
        rows += ["%s:s4901:n1" % a for a in geometry.SOLVE_CHECK_ARMS if "s025" not in a]
        put(c / "exit_codes.tsv", "".join("%s\t0\n" % r for r in rows))
        put(c / "logs" / "unittest.log", "Ran 232 tests in 88.4s\nOK\n")
        put(c / "logs" / "geometry.log", "GEOMETRY_CHECK OK\n")
        put(c / "logs" / "regress_compare.log", "REGRESS IDENTICAL\n")
        put(c / "logs" / "bytecheck.log", "BYTECHECK OK\n")
        for d in ("byte_multi", "byte_single"):
            put(c / "runs" / d / "realizations" / "r_0000" / "DONE", "")
        for a in geometry.SOLVE_CHECK_ARMS:
            if "s025" not in a:
                put(c / "runs" / a / "realizations" / "r_0000" / "solver_b.json", '{"verdict": "converged"}')
                put(c / "runs" / a / "realizations" / "r_0000" / "DONE", "")

    def seed_par2(self):
        """A recorded Par2 s1 pilot and its field-statistics file, whose content differs from what the saved log would give."""
        put(self.root / "pilot" / "exit_codes.tsv", "par2_s1\t0\nfs_par2_s1\t0\n")
        log = "".join("FIELDSTAT r=%d seed=%d n=8 mean_lnK=0 var_lnK=7.7e-01 Kmin=1 Kmax=2 Kmean=1\n" % (r, 9001 + r) for r in range(20))
        put(self.root / "pilot" / "logs" / "fs_par2_s1.log", log)
        put(self.root / "par2" / "field_stats_par2_s1.txt", log.replace("7.7e-01", "5.5e-01"))
        for r in range(20):
            put(self.root / "par2" / "raw" / "par2_s1" / "stats" / ("realization_%04d_timeseries.csv" % r), "")

    def run_script(self, script, rev, *args):
        """A real (non-dry) run of `script`; rev None = SF39_COMMIT unset.  No GIT_* variable leaks into the script's repository lookups."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k not in ("SF39_DRYRUN", "SF39_COMMIT")}
        env.update(SF39_OUT_ROOT=str(self.root), SF39_TIME_BIN=str(Path(self.tmp.name) / "faketime.sh"), SF39_REGRESS_HOME=str(self.home),
                   PYTHON=str(Path(self.tmp.name) / "fakepy.sh"))
        if rev is not None:
            env["SF39_COMMIT"] = rev
        p = subprocess.run(["bash", str(script), str(self.bin), *args], capture_output=True, text=True, env=env)
        return p.returncode, p.stdout, p.stderr

    def run_sh(self, rev, *args):
        return self.run_script(SH, rev, *args)

    def copy_tree(self, root):
        """A copy of the scripts at <root>/docs/experiments/artifacts/x/scripts (REPO of that copy = <root>); returns its run_sf39.sh."""
        copy = Path(root) / "docs" / "experiments" / "artifacts" / "x" / "scripts"
        shutil.copytree(SCRIPTS, copy, ignore=shutil.ignore_patterns("__pycache__", "tests"))
        return copy / "run_sf39.sh"

    @staticmethod
    def snapshot(d):
        return {str(p.relative_to(d)): p.read_bytes() for p in sorted(Path(d).rglob("*")) if p.is_file()}

    def ran(self, out):
        return re.findall(r"CASE_RESULT run=(\S+) ", out)

    def records(self):
        return [ln.split("\t")[0] for ln in (self.root / "checks" / "exit_codes.tsv").read_text().splitlines()]

    def test_new_tree_reruns_only_the_tree_dependent_checks(self):
        old = self.records()
        rc, out, err = self.run_sh("1111111111aa", "checks")
        self.assertEqual(rc, 0, out + err)
        want = ["unittest:1111111111aa", "geometry:1111111111aa", "chk_s025_n05:s4901:n1", "chk_s025_n10:s4901:n1", "chk_s025_n20:s4901:n1",
                "chk_s025_tw:s4901:n1", "smoke_reduce:1111111111aa", "signtable:1111111111aa"]
        self.assertEqual(self.ran(out), want)
        for skipped in ("regress_head", "regress_compare", "bytecheck", "byte_multi", "byte_single", "chk_s1_n05:s4901:n1", "chk_s4_w12:s4901:n1"):
            self.assertIn("SKIP_DONE run=%s rc=0" % skipped, out)
        for name in ("unittest", "geometry", "regress", "bytecheck", "smoke_reduce", "signtable", "solve_chk_s025_tw", "solve_chk_s4_w12"):
            self.assertRegex(out, r"(?m)^CHECK_%s rc=0 " % name)
        self.assertIn("CHECK_solve_chk_s025_n05 rc=0 verdict=converged", out)
        self.assertIn("PART_DONE part=checks rc=0", out)
        # the old records stay (appended to, never rewritten)
        self.assertEqual(self.records()[:len(old)], old)
        logs = {p.name for p in (self.root / "checks" / "logs").iterdir()}
        self.assertTrue({"unittest_1111111111aa.log", "geometry_1111111111aa.log", "unittest.log", "geometry.log"} <= logs)
        # the same tree again: nothing runs
        rc, out2, err = self.run_sh("1111111111aa", "checks")
        self.assertEqual((rc, self.ran(out2)), (0, []))
        # a new tree: exactly the four tree-dependent ids run again
        rc, out3, err = self.run_sh("2222222222bb", "checks")
        self.assertEqual((rc, self.ran(out3)), (0, ["unittest:2222222222bb", "geometry:2222222222bb", "smoke_reduce:2222222222bb",
                                                    "signtable:2222222222bb"]))

    def git(self, cwd, *args):
        """Run git in `cwd` (a working directory, never `git -C`: the V100 host has git 1.8.3).  Only features of git 1.7.2 and older are used (an unknown `-c` key such as commit.gpgsign is ignored by old git)."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, env=env,
                              check=True, cwd=str(cwd)).stdout.strip()

    def assertRefusedUntouched(self, rc, out, err, needle="does not identify the tree"):
        self.assertEqual(rc, 2, out + err)
        self.assertIn(needle, err)
        self.assertEqual(self.ran(out), [])
        self.assertEqual(self.snapshot(self.root / "checks"), self.before)  # results.txt, PART_EXIT, run.env, logs, records: byte-identical
        self.assertFalse((self.root / "checks" / ".build_digest").exists())

    def seed_part_files(self):
        c = self.root / "checks"
        put(c / "results.txt", "PART_BEGIN old\nPART_DONE part=checks rc=0\n")
        put(c / "PART_EXIT", "0\n")
        put(c / "run.env", "OLD_RUN_ENV=1\n")
        self.before = self.snapshot(c)

    def test_a_revision_that_does_not_identify_the_tree_refuses_before_the_part_starts(self):
        self.seed_part_files()
        bad = {"blank": "   ", "tab": "\t \t", "dirty": "abcdef012345-dirty", "unknown": "unknown", "dirty full": "0123456789abcdef-dirty"}
        for what, rev in bad.items():
            rc, out, err = self.run_sh(rev, "checks")
            self.assertRefusedUntouched(rc, out, err)
        # the other parts do not need a tree revision
        self.assertEqual(self.run_sh("abcdef012345-dirty", "pilot", "par2")[0], 0)
        # a clean revision is accepted (and starts the part)
        rc, out, err = self.run_sh("abcdef012345", "checks")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("unittest:abcdef012345", self.ran(out))

    def test_unknown_revision_refuses(self):
        # a copy of the scripts outside any repository, no SF39_COMMIT, no .mf3d_source_rev: the ids cannot be tied to a tree
        self.seed_part_files()
        script = self.copy_tree(Path(self.tmp.name) / "tree")
        self.assertRefusedUntouched(*self.run_script(script, None, "checks"))

    def test_the_mirror_revision_file_is_the_second_source(self):
        self.seed_part_files()
        tree = Path(self.tmp.name) / "tree"
        script = self.copy_tree(tree)
        put(tree / ".mf3d_source_rev", "0123456789ab-dirty\n")
        self.assertRefusedUntouched(*self.run_script(script, None, "checks"))
        put(tree / ".mf3d_source_rev", "0123456789ab\n")
        rc, out, err = self.run_script(script, None, "checks")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("unittest:0123456789ab", self.ran(out))

    @unittest.skipUnless(GIT_OK, GIT_WHY)
    def test_an_enclosing_repository_is_not_this_tree(self):
        self.seed_part_files()
        outer = Path(self.tmp.name) / "outer"
        outer.mkdir()
        self.git(outer, "init", "-q")
        self.git(outer, "commit", "-q", "--allow-empty", "-m", "x")
        script = self.copy_tree(outer / "inner")  # REPO = outer/inner: inside a repository whose top level is outer
        self.assertRefusedUntouched(*self.run_script(script, None, "checks"))

    @unittest.skipUnless(GIT_OK, GIT_WHY)
    def test_the_repository_head_is_used_when_the_tree_is_the_top_level(self):
        repo = Path(self.tmp.name) / "repo"
        repo.mkdir()
        self.git(repo, "init", "-q")
        script = self.copy_tree(repo)
        self.seed_part_files()
        # no commit yet: HEAD is unborn, there is no revision
        self.assertRefusedUntouched(*self.run_script(script, None, "checks"))
        self.git(repo, "commit", "-q", "--allow-empty", "-m", "x")
        head = self.git(repo, "rev-parse", "HEAD")
        rc, out, err = self.run_script(script, None, "checks")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("unittest:" + head[:24], self.ran(out))

    def test_sealed_files_of_a_separate_checkpoint_case_are_hashed(self):
        for sel in ((), ("s025",)):
            rc, out, err = self.run_sh("abcdef012345", "checkpoint", *sel)
            self.assertEqual(rc, 0, out + err)
        lines = (self.root / "checkpoint" / "sha256.txt").read_text().splitlines()
        hashed = {ln.split("  ", 1)[1]: ln.split("  ", 1)[0] for ln in lines}
        for rel in ("checkpoint/tau_b_star.json", "checkpoint/sealed/estimates.csv", "checkpoint/s025/tau_b_star_s025.json",
                    "checkpoint/s025/sealed/estimates.csv", "checkpoint/s025/sealed/nr_out.json"):
            self.assertIn(rel, hashed)
            self.assertEqual(hashed[rel], hashlib.sha256((self.root / rel).read_bytes()).hexdigest(), rel)
        self.assertNotIn("checkpoint/results.txt", hashed)

    def test_a_skipped_field_stats_case_with_a_missing_file_says_why(self):
        # fs_par2_s225 is recorded (rc 0, final) but its file is gone: it is not regenerated; the failure names the cause and the remedy
        put(self.root / "pilot" / "exit_codes.tsv", "par2_s1\t0\nfs_par2_s1\t0\npar2_s225\t0\nfs_par2_s225\t0\n")
        for r in range(20):
            put(self.root / "par2" / "raw" / "par2_s225" / "stats" / ("realization_%04d_timeseries.csv" % r), "")
        rc, out, err = self.run_sh("abcdef012345", "pilot", "par2")
        self.assertEqual(rc, 1, out + err)
        self.assertRegex(out, r"FIELDSTAT_FAILED id=fs_par2_s225 \(rc=0, 0 of 20 lines; \S+field_stats_par2_s225\.txt is missing, "
                              r"the recorded rc 0 is final .*remove the fs_par2_s225 record from exit_codes\.tsv to regenerate it\)")
        self.assertIn("fieldstat_lines=0/20", out)
        self.assertIn("PART_DONE part=pilot rc=1", out)
        self.assertNotIn("No such file", out + err)  # the redirect comes first: no shell noise for the missing file
        self.assertFalse((self.root / "par2" / "field_stats_par2_s225.txt").exists())  # and nothing is regenerated

    def test_skipped_par2_case_does_not_rewrite_field_stats(self):
        fs = self.root / "par2" / "field_stats_par2_s1.txt"
        before = fs.read_bytes()
        rc, out, err = self.run_sh("1111111111aa", "pilot", "par2")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("SKIP_DONE run=fs_par2_s1 rc=0", out)
        self.assertEqual(fs.read_bytes(), before)  # recorded rc 0: the analysis input is never re-derived from the log
        self.assertIn("fieldstat_lines=20/20", out)
        for c in ("025", "225", "4"):  # the other cases ran and wrote their own file
            f = self.root / "par2" / ("field_stats_par2_s%s.txt" % c)
            self.assertEqual(len(f.read_text().splitlines()), 20, c)
        self.assertEqual(self.ran(out), ["par2_s025", "fs_par2_s025", "par2_s225", "fs_par2_s225", "par2_s4", "fs_par2_s4"])


class GitSupport(unittest.TestCase):
    """`git_support`: the git-dependent tests are skipped, with a stated reason, when git is absent or too old for them."""

    def fake_git(self, banner):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        if banner is not None:
            put(Path(d) / "git", "#!/bin/sh\necho '%s'\n" % banner, exe=True)
        return d

    def support(self, banner):
        with mock.patch.dict(os.environ, {"PATH": self.fake_git(banner)}):
            return git_support()

    def test_versions(self):
        self.assertEqual(self.support("git version 1.8.3.1"), (True, ""))  # the V100 host
        self.assertEqual(self.support("git version 1.7.2"), (True, ""))
        self.assertEqual(self.support("git version 2.39.5 (Apple Git-154)"), (True, ""))
        ok, why = self.support("git version 1.6.5.2")
        self.assertFalse(ok)
        self.assertIn("needs git >= 1.7.2, found 1.6.5", why)
        ok, why = self.support("git: not a version")
        self.assertFalse(ok)
        self.assertIn("cannot read its version", why)
        ok, why = self.support(None)
        self.assertEqual((ok, why), (False, "needs git (not found)"))


if __name__ == "__main__":
    unittest.main()
