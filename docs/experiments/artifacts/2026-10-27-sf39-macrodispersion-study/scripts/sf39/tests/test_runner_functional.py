"""Restart behaviour of run_sf39.sh with FAKE binaries (instant, no CUDA, nothing outside a temp directory): the campaign state after the
original runs (old unprefixed check ids, a recorded Par2 case) and a re-invocation with a new scripts tree.  Seconds; needs bash and python3.

  * the ids of the tree-dependent checks carry the scripts revision: a new revision re-runs exactly those, the old records stay, the same
    revision re-runs nothing, recorded regress / byte-check / chk_s1..s4 ids are skipped and the new chk_s025_* ids run;
  * a skipped Par2 field-statistics case never rewrites par2/field_stats_*.txt.
"""
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from sf39 import geometry

SCRIPTS = Path(__file__).resolve().parents[2]
SH = SCRIPTS / "run_sf39.sh"

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

    def run_sh(self, rev, *args):
        env = dict(os.environ, SF39_OUT_ROOT=str(self.root), SF39_TIME_BIN=str(Path(self.tmp.name) / "faketime.sh"),
                   SF39_REGRESS_HOME=str(self.home), SF39_COMMIT=rev, PYTHON=str(Path(self.tmp.name) / "fakepy.sh"))
        env.pop("SF39_DRYRUN", None)
        p = subprocess.run(["bash", str(SH), str(self.bin), *args], capture_output=True, text=True, env=env)
        return p.returncode, p.stdout, p.stderr

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

    def test_unknown_revision_refuses(self):
        # a copy of the scripts outside any git repository, no SF39_COMMIT, no .mf3d_source_rev: the ids cannot be tied to a tree
        t = Path(self.tmp.name)
        copy = t / "tree" / "docs" / "experiments" / "artifacts" / "x" / "scripts"
        shutil.copytree(SCRIPTS, copy, ignore=shutil.ignore_patterns("__pycache__", "tests"))
        env = dict(os.environ, SF39_OUT_ROOT=str(self.root), SF39_TIME_BIN=str(t / "faketime.sh"), SF39_REGRESS_HOME=str(self.home),
                   PYTHON=str(t / "fakepy.sh"))
        for k in ("SF39_DRYRUN", "SF39_COMMIT"):
            env.pop(k, None)
        p = subprocess.run(["bash", str(copy / "run_sf39.sh"), str(self.bin), "checks"], capture_output=True, text=True, env=env)
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("scripts revision is unknown", p.stderr)
        self.assertEqual(self.ran(p.stdout), [])

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


if __name__ == "__main__":
    unittest.main()
