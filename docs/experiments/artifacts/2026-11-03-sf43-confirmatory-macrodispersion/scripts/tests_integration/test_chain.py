"""END-TO-END test of the WHOLE SF-43 toolchain on CPU (no GPU, no data):

    sf43plan (the real generator, smoke universe, sizes scaled by ``D.SMOKE``) -> sf43camp ``all --smoke`` (the real engine, the real
    ``RunnerExecutor`` and the fake runner binaries of ``sf43camp/tests_executor/fake_runner.py``, varied by seed and arm) -> reduce -> prune -> ``recompute43`` ->
    ``reconcile43`` -> ``analyse_sf43.sh --smoke`` (a copy of the wrapper in a scratch repository tree) -> ``summary.json`` ->
    ``check-finite`` STRUCTURE_OK.

The plan keeps every arm family of the study (production with three-pass first seeds, E_h ladders, width cells, Gate 5 twin + Par2, G7 cells,
distance, positive control, determinism pairs); only the runner dictionaries are replaced by tiny ones (a 208x4x4 grid, 100 particles) because
the fake runners ignore the numerics, and the Par2 time-window keys are dropped (the fake writes no time series).  Injected, and asserted through
the analysis: one UNUSABLE SOLVE (group-wide replacement), one UNREDUCED segment (the reduction of the last segment of one ladder arm always
fails), one blinded RE-ESTIMATION (a production arm grows from n_conf to n_max).  The fake data are synthetic: NO statistical value is asserted,
only wiring (every family hypothesis evaluated, pairs found by seed, the cuts the ledger dictates, the blocks that must exist).  ~2 min.
"""
import contextlib
import io
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from sf43camp import campaign, lanes
from sf43plan import __main__ as GEN
from sf43plan import build as B
from sf43plan import design as D
from sf43ana.tests.test_gate5 import fake_summary

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ART = os.path.dirname(SCRIPTS)
SF42 = os.path.normpath(os.path.join(ART, "..", "2026-10-30-sf42-confirmatory-tooling", "scripts"))
FAKE = os.path.join(HERE, "varied_runner.py")        # sf43camp/tests_executor/fake_runner.py with seed- and arm-dependent variation of its numbers

PLANES_D = [25, 30, 40, 50, 60, 70, 80, 94, 120, 150, 194]
WINDOW = "0.375,0.625,0.375,0.625"
SIZES = {"n": 6, "n_slow": 6, "reest_n_conf": 8, "reest_n_max": 16, "reest_fraction": 0.5, "hmax_first_n": 4, "unit": 2}
REPLACED_SEED = 439104       # smoke_s1 block (start 439101), position 3
FLAKY_ARM, FLAKY_SEED = "sm_s1_eh5", 439106
REEST_ARM = "sm_s025_prod"
AMPS = ("s025", "s1", "s225", "s4")


def tiny_plan(plan, flaky_arm, flaky_seed, bg_argv, wrapper):
    """The generated plan with tiny runner dictionaries, the stub background job, the scratch wrapper and the flaky executor."""
    plan["binaries"] = {}
    plan["disk_min_free_gb"] = 0
    plan["poll_s"] = 0.1
    plan["grace_minutes"] = 1
    plan["executor"] = "tests_integration.executor:FlakyExecutor"
    plan["flaky"] = {"arm": flaky_arm, "seed": flaky_seed}
    plan["min_retained"] = SIZES["n"]        # the fake runner writes no moment CSVs (the real one does): keep the raw arrays of every non-primary realization;
                                             # the primary arm that grows past n still exercises the selective pruning
    sm = plan["smoke"]
    for a in sm["arms"]:
        run = a[a["runner"]]
        run.update(grid=[208, 4, 4], ell=0.25, window=WINDOW, planes_d=PLANES_D)
        if a["runner"] == "consumer":
            run["np_side"] = 8
        else:
            run["n_particles"] = 100
            if "snapshots_every" in run:
                run["snapshots_every"] = 100
        a["reduce"].pop("U_time", None)            # the fake Par2 runner writes no time series
        a["reduce"].pop("par2_exit_x", None)
        a["est_seconds"] = 1.0
    sm["background"] = [{"name": "coverage", "nice": 19, "argv": bg_argv}]
    sm["background_wait_minutes"] = 0.5
    sm["analysis"] = {"command": ["bash", wrapper, "{campaign_dir}", "{build_dir}", "--smoke"], "sha256_list": {}}
    sm["budget_gpu_h"] = 50
    return plan


def jload(path):
    with open(path) as f:
        return json.load(f)


class Chain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        keep = os.environ.get("SF43_INTEGRATION_DIR")      # debugging: keep the scratch tree in this (empty) directory
        cls._t = None if keep else tempfile.TemporaryDirectory()
        cls.root = os.path.realpath(keep or cls._t.name)
        if cls._t:
            cls.addClassCleanup(cls._t.cleanup)
        cls.sig = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        for s, h in cls.sig.items():
            cls.addClassCleanup(signal.signal, s, h)
        cls.gen()
        cls.run_engine()

    # ------------------------------------------------------------------------------------------------------------------------------
    @classmethod
    def gen(cls):
        """sf43plan with the smoke sizes of SIZES; a scratch repository tree holding a COPY of the wrapper, the smoke pre-registration and
        symlinks to the packages (the wrapper finds its prereg and packages relative to itself)."""
        root = cls.root
        cls.out = os.path.join(root, "gen")
        cov = os.path.join(root, "cov.json")
        with open(cov, "w") as f:
            json.dump({"fixture": 1}, f)
        # Gate 5 coverage of the smoke job: Hall and bootstrap-t qualified at the grid size 4 for the base families and the J3B-bound family
        summ = fake_summary({"student": {4: False}, "hall": {4: True}, "boott": {4: True}}, ns=(4,))
        extra = []
        for c in summ["cells"]:
            if c["dist"] == "normal":
                for d in ("mix:2.73:7.59", "mix:2.73:7.59-"):
                    extra.append(dict(c, dist=d, rows=json.loads(json.dumps(c["rows"]))))
        summ["cells"] += extra
        cls.cov_summary = os.path.join(root, "cov_summary.json")
        with open(cls.cov_summary, "w") as f:
            json.dump(summ, f)
        cls.camp = os.path.join(root, "camp")
        with mock.patch.dict(D.SMOKE, SIZES), contextlib.redirect_stdout(io.StringIO()):
            GEN.main(["generate", "--variant", "A", "--campaign-dir", cls.camp, "--binaries", "placeholder", "--coverage-summary", cov,
                      "--coverage-qualification", cov, "--out", cls.out])
        cls.cdir = cls.camp + "_smoke"
        # the scratch repository: <T>/docs/experiments/artifacts/<A>/{analyse_sf43.sh, prereg_smoke.json, scripts/<pkg> -> symlink}
        art_t = os.path.join(root, "repo", os.path.relpath(ART, B.REPO))
        os.makedirs(os.path.join(art_t, "scripts"))
        for pkg in ("sf43ana", "sf43cov", "sf43camp", "recompute43", "reconcile43"):
            os.symlink(os.path.join(SCRIPTS, pkg), os.path.join(art_t, "scripts", pkg))
        os.symlink(os.path.dirname(SF42), os.path.join(os.path.dirname(art_t), os.path.basename(os.path.dirname(SF42))))
        cls.wrapper = os.path.join(art_t, "scripts", "analyse_sf43.sh")
        shutil.copy(os.path.join(SCRIPTS, "analyse_sf43.sh"), cls.wrapper)
        shutil.copy(os.path.join(cls.out, "prereg_smoke.json"), os.path.join(art_t, "prereg_smoke.json"))
        cls.prereg = jload(os.path.join(cls.out, "prereg_smoke.json"))
        cls.plan_gen = jload(os.path.join(cls.out, "plan.json"))
        bg = [sys.executable, "-c", "import os,shutil,sys; os.makedirs(sys.argv[1]); shutil.copy(sys.argv[2], os.path.join(sys.argv[1], 'summary.json'))",
              "{campaign_dir}/coverage", cls.cov_summary]
        plan = tiny_plan(json.loads(json.dumps(cls.plan_gen)), FLAKY_ARM, FLAKY_SEED, bg, cls.wrapper)
        cls.plan_file = os.path.join(root, "plan.json")
        with open(cls.plan_file, "w") as f:
            json.dump(plan, f)

    @classmethod
    def run_engine(cls):
        root = cls.root
        cls.build_dir = os.path.join(root, "build")
        os.makedirs(cls.build_dir)
        for exe, mode in (("lagrangian_ensemble", "consumer"), ("macroflow3d_pipeline", "par2")):
            p = os.path.join(cls.build_dir, exe)
            with open(p, "w") as f:
                f.write('#!/bin/sh\nexec %s %s %s "$@"\n' % (sys.executable, FAKE, mode))
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
        smi = os.path.join(root, "bin", "nvidia-smi")        # the engine reads the free memory of its GPUs: an unreadable query = enough (this host may have a GPU)
        os.makedirs(os.path.dirname(smi))
        with open(smi, "w") as f:
            f.write("#!/bin/sh\nexit 1\n")
        os.chmod(smi, 0o755)
        env = mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,0", "PYTHONPATH": SCRIPTS + os.pathsep + SF42,
                                          "PATH": os.path.dirname(smi) + os.pathsep + os.environ["PATH"], "FAKE_NOCONV_SEED": str(REPLACED_SEED)})
        env.start()
        cls.addClassCleanup(env.stop)
        for name, val in (("PRUNE_S", 0.5), ("REDUCE_BACKOFF_S", (0.0, 0.0, 0.0))):
            m = mock.patch.object(lanes, name, val)
            m.start()
            cls.addClassCleanup(m.stop)
        t0 = time.time()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cls.rc = campaign.main([cls.build_dir, "all", "--plan", cls.plan_file, "--smoke"])
        cls.engine_log = buf.getvalue()
        cls.seconds = time.time() - t0
        with open(os.path.join(cls.cdir, "ledger.jsonl")) as f:
            cls.ledger = [json.loads(x) for x in f if x.strip()]
        with open(os.path.join(cls.cdir, "analysis.log")) as f:
            cls.analysis_log = f.read()
        cls.summary_path = os.path.join(cls.cdir, "analysis", "summary.json")
        cls.S = jload(cls.summary_path) if os.path.isfile(cls.summary_path) else None

    def report(self):
        p = os.path.join(self.cdir, "REPORT.txt")
        if not os.path.exists(p):
            return "(no REPORT)\n" + self.engine_log[-3000:]
        with open(p) as f:
            return f.read()

    def kind(self, k):
        return [r for r in self.ledger if r["kind"] == k]

    # ------------------------------------------------------------------------------------------------------------------------------
    def test_engine_and_hooks_finished(self):
        self.assertIn(self.rc, (campaign.DONE, campaign.PARTIAL), self.report())
        self.assertFalse(os.path.exists(os.path.join(self.cdir, "ANALYSIS_FAILED")), self.report() + self.analysis_log[-2000:])
        for n in ("recompute", "reconcile"):
            with open(self.cdir + ".%s.txt" % n) as f:
                last = f.read().rstrip().splitlines()[-1]
            self.assertIn(last, ("RECOMPUTE PASS", "RECONCILE OK"), n)
        self.assertEqual([r["rc"] for r in self.kind("background_end")], [0])
        self.assertTrue(os.path.isfile(os.path.join(self.cdir, "coverage", "summary.json")))

    def test_smoke_wrapper_prints_only_check_finite(self):
        """analysis.log holds the wrapper's stdout and stderr: ONLY the check-finite lines (counts, STRUCTURE_OK)."""
        lines = [x for x in self.analysis_log.splitlines() if x.strip()]
        self.assertEqual(lines[-1], "STRUCTURE_OK", self.analysis_log[-2000:])
        import re
        for ln in lines:
            self.assertRegex(ln, r"^(section \w+: finite=\d+ nonfinite=\d+ other=\d+|family: hypotheses=\d+ m=\d+|estimands: nonfinite_means=\d+|process: warnings_count=\d+ errors_count=\d+|STRUCTURE_OK)$")
        for word in ("physical", "rejected", "numerical", "unresolved", "alpha", "p="):
            self.assertNotIn(word, self.analysis_log)
        self.assertIsNone(re.search(r"\d\.\d", self.analysis_log))                 # no decimal number at all
        out = os.path.join(self.cdir, "analysis")
        self.assertTrue(os.path.isfile(os.path.join(out, "run.stdout.txt")) and os.path.isfile(os.path.join(out, "run.stderr.txt")))
        self.assertIsNotNone(self.S)

    def test_production_mode_wrapper_prints_the_check_finite_after_the_run(self):
        """Production branch of the wrapper (no --smoke, ``-W default``) on a copy of the smoke campaign: ``wrote ...`` then the check-finite
        output, exit 0; the summary records process.errors_count == 0 and the warnings of the run (counts only in check-finite)."""
        art_t = os.path.dirname(os.path.dirname(self.wrapper))
        shutil.copy(os.path.join(art_t, "prereg_smoke.json"), os.path.join(art_t, "prereg.json"))
        self.addCleanup(os.remove, os.path.join(art_t, "prereg.json"))
        cdir2 = os.path.join(self.root, "camp_prod")
        shutil.copytree(self.cdir, cdir2)
        r = subprocess.run(["bash", self.wrapper, cdir2, self.build_dir], capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        lines = [x for x in r.stdout.splitlines() if x.strip()]
        self.assertRegex(lines[0], r"^wrote .*summary\.json$")
        self.assertEqual(lines[-1], "STRUCTURE_OK")
        self.assertRegex(r.stdout, r"process: warnings_count=\d+ errors_count=0")
        S2 = jload(os.path.join(cdir2, "analysis", "summary.json"))
        self.assertEqual((S2["process"]["errors"], S2["process"]["errors_count"]), ([], 0))
        self.assertEqual(S2["process"]["warnings_count"], sum(w["count"] for ws in S2["process"]["warnings"].values() for w in ws))

    def test_every_family_hypothesis_is_evaluated(self):
        fam = self.S["family"]
        self.assertEqual(self.S["process"]["errors_count"], 0)                    # no hypothesis was isolated as an error
        self.assertEqual((fam["info"]["m"], len(fam["hypotheses"])), (8, 8))
        for h in fam["hypotheses"]:
            self.assertTrue(h["testable"], h)                                      # no p = 1 from wiring (floors, arms, pairs, coverage)
            self.assertLess(h["p"], 1.0 + 1e-12)
        self.assertEqual(sorted(h["id"] for h in fam["hypotheses"]), sorted(D.PRIMARY_FAMILY))
        for amp in AMPS:
            for basis in ("disp", "pos"):
                for est in ("alpha_TP", "alpha_TAP"):
                    self.assertTrue(self.S["gate4"][amp]["bases"][basis]["estimators"][est]["testable"], (amp, basis, est))

    def test_gate5_pairs_are_found_by_seed(self):
        g5 = self.S["gate5"]["amplitudes"]
        for amp in AMPS:
            a = g5[amp]
            self.assertEqual(a["pairing"]["n_nonfinite_dropped"] if "n_nonfinite_dropped" in a["pairing"] else 0, 0)
            self.assertEqual((a["pairing"]["n_only_a"], a["pairing"]["n_only_b"]), (0, 0), amp)
            self.assertEqual(a["n"], SIZES["n"], amp)
            self.assertIsNotNone(a["procedure"], amp)                              # the coverage summary of the smoke job qualified a procedure
            self.assertEqual(a["reporting"]["confirmatory_outcome"]["decided"], True)
            self.assertIn("Holm TOST", a["reporting"]["confirmatory_outcome"]["what"])
            self.assertIn("descriptive", a["reporting"]["descriptive_verdict"]["what"])
        arms = self.S["process"]["arms"]
        pair = (arms["sm_s1_g5c"]["N"], arms["sm_s1_g5p"]["N"])
        self.assertEqual(pair, (SIZES["n"], SIZES["n"]))

    def test_replacement_unusable_solve_is_group_wide(self):
        rp = self.kind("replacement")
        self.assertEqual([(r["group"], r["seed"]) for r in rp], [("smoke_s1", REPLACED_SEED)])
        reserve = rp[0]["replaced_by"]
        # the first arm that fails the seed triggers the replacement; the pending seed is DROPPED from the other arms of the group (not run, not failed)
        bad = {r["arm"] for r in self.kind("seed") if r["seed"] == REPLACED_SEED and r["class"] == "unusable_solve"}
        self.assertEqual(bad, {rp[0]["arm"]})
        arms = self.S["process"]["arms"]
        for aid in ("sm_s1_prod", "sm_s1_eh20", "sm_s1_g5c", "sm_s1_g5p", "sm_s1_h8"):
            applied = arms[aid]["replacements_applied"]
            if arms[aid]["n_cut"] > 3:
                self.assertEqual(applied, [{"seed": REPLACED_SEED, "replaced_by": reserve}], aid)
        self.assertEqual(arms["sm_s1_prod"]["N"], SIZES["n"])
        failed = self.S["failures"]["arms"][rp[0]["arm"]]
        self.assertEqual(failed["failed_seeds"], [REPLACED_SEED])
        self.assertEqual(failed["replacements"], [{"failed_seed": REPLACED_SEED, "replacement_seed": reserve}])
        other = self.S["failures"]["arms"]["sm_s1_prod"]               # the replacement is disclosed for the arms that never ran the seed as well
        self.assertEqual((other["failed_seeds"], other["replacements"], other["replacements_without_failure"]),
                         ([], [{"failed_seed": REPLACED_SEED, "replacement_seed": reserve}], [REPLACED_SEED]))

    def test_unreduced_segment_is_missing_explicitly(self):
        un = self.kind("unreduced")
        self.assertEqual([r["arm"] for r in un], [FLAKY_ARM])
        self.assertIn(FLAKY_SEED, un[0]["seeds"])
        a = self.S["process"]["arms"][FLAKY_ARM]
        self.assertIn("unreduced segment", a["stopped_at"]["reason"])
        self.assertEqual(a["N"], a["stopped_at"]["position"])
        self.assertGreaterEqual(a["N"], 2)
        self.assertLess(a["N"], SIZES["n"])
        self.assertEqual(list(self.S["process"]["loader"]["unreduced"]), ["%s/%s" % (FLAKY_ARM, os.path.basename(un[0]["segment"]))])
        self.assertEqual(self.S["failures"]["unreduced_segments"], self.S["process"]["loader"]["unreduced"])
        self.assertTrue(os.path.exists(os.path.join(self.cdir, "ANALYSIS_BLOCKED_UNREDUCED")))
        h = {x["id"]: x for x in self.S["family"]["hypotheses"]}["robust:s1"]
        self.assertTrue(h["testable"])                                             # the ladder prefix is shorter, still above the floor
        self.assertEqual(self.S["process"]["arms"]["sm_s1_eh20"]["N"], SIZES["n"])  # other arms unaffected

    def test_reestimation_extends_the_production_cut_and_only_that(self):
        re = self.kind("reest")
        self.assertEqual([(r["arm"], r["n_old"], r["n_new"]) for r in re], [(REEST_ARM, SIZES["reest_n_conf"], SIZES["reest_n_max"])])
        arms = self.S["process"]["arms"]
        for aid in (REEST_ARM, "sm_s025_prodpol"):
            self.assertEqual((arms[aid]["n_conf"], arms[aid]["n_cut"], arms[aid]["N"]), (SIZES["reest_n_conf"], SIZES["reest_n_max"], SIZES["reest_n_max"]), aid)
        self.assertEqual(self.S["process"]["loader"]["reest_n_new"], {REEST_ARM: SIZES["reest_n_max"]})
        self.assertEqual(arms["sm_s025_g5c"]["N"], SIZES["n"])                    # the Gate 5 twin of the same engine arm keeps its planned size
        self.assertEqual(arms["sm_s025_h8"]["N"], 2)
        self.assertEqual(self.S["estimands"]["s025"]["N"], SIZES["reest_n_max"])
        self.assertEqual(arms["sm_s1_prod"]["n_cut"], SIZES["n"])                  # not re-estimated

    def test_window_stems_match_the_executors_files(self):
        """Every reduced table the engine wrote is found under the stem rule of the loader for the windows of the plan."""
        import glob
        from sf43ana.loader import window_stem
        wins = {window_stem(*w) for w in self.prereg["windows"].values()}
        stems = {os.path.basename(p).rsplit("__", 1)[1][:-4] for p in glob.glob(os.path.join(self.cdir, "reduced", "*", "*", "*.csv"))
                 if not p.endswith(".planes.csv")}
        self.assertEqual(stems, wins)
        self.assertIn("window", self.S["estimands"]["s225"]["distance"])
        d = self.S["estimands"]["s225"]["distance"]
        self.assertGreaterEqual(d["N"], 2)
        self.assertEqual(sorted(self.S["estimands"]["s225"]["halves_no_claim"]), ["h1", "h2"])
        self.assertTrue(all(h["alpha_TP/disp"]["n"] >= 2 for h in self.S["estimands"]["s225"]["halves_no_claim"].values()))

    def test_pollock_contrast_and_decomposition_exist(self):
        P = self.S["pollock"]
        self.assertNotIn("error", P)
        self.assertIn("outside the error budget and the Holm family", P["label"])
        for amp in AMPS:
            C = P["contrast"][amp]
            self.assertEqual(sorted(C), sorted("%s/%s" % (e, b) for e in ("alpha_TP", "alpha_TAP", "alpha_LP", "U_r") for b in ("disp", "pos")))
            for key, c in C.items():
                self.assertGreaterEqual(c["n"], 2, (amp, key))
                self.assertIn("difference", c)
            D_ = P["decomposition"][amp]["alpha_LP"]
            self.assertEqual(sorted(D_), ["consumer/par2", "consumer/pollock", "pollock/par2"])
            self.assertEqual(D_["consumer/pollock"]["pairing"]["n_common"], SIZES["n"], amp)
            self.assertEqual(D_["pollock/par2"]["pairing"]["n_common"], SIZES["n"], amp)
            self.assertEqual(self.S["g7"]["twins"][amp]["alpha_LP"]["consumer/pollock"]["pairing"]["n_common"], SIZES["n"], amp)
        self.assertEqual(P["contrast"]["s025"]["alpha_TP/disp"]["n"], SIZES["reest_n_max"])        # the production analysis set after the re-estimation

    def test_descriptive_blocks_present_without_errors(self):
        for sec in ("gate5", "g7", "secondaries", "failures", "positive_control", "pollock"):
            self.assertIn(sec, self.S, sec)
        text = json.dumps(self.S)
        self.assertNotIn('"error"', text)
        pc = self.S["positive_control"]
        self.assertIn("R", pc)
        self.assertIn(pc["R"]["status"], ("ok", "inconclusive", "failed"))
        self.assertEqual(sorted(self.S["g7"]["factorial"]), ["s025"])
        with open(os.path.join(self.cdir, "analysis", "run.stderr.txt")) as f:
            self.assertTrue(os.path.isdir(os.path.join(self.cdir, "analysis", "figures")) or "FIGURES_SKIPPED" in f.read())

    def test_determinism_pairs_and_selective_pruning(self):
        with open(os.path.join(self.cdir, "DETERMINISM.txt")) as f:
            det = f.read()
        self.assertNotIn("FAIL", det)
        self.assertIn("PASS", det)
        self.assertEqual(self.kind("pruned") != [], True)


class RealPlansDryRun(unittest.TestCase):
    """``run_sf43.sh ... all --dry-run`` and ``--dry-run --smoke`` on the plans the generator writes for the real design (no process, no file)."""

    def test_dry_runs(self):
        with tempfile.TemporaryDirectory() as d:
            cov = os.path.join(d, "c.json")
            with open(cov, "w") as f:
                json.dump({}, f)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                GEN.main(["generate", "--variant", "A", "--campaign-dir", os.path.join(d, "camp"), "--binaries", "placeholder",
                          "--coverage-summary", cov, "--coverage-qualification", cov, "--out", d])
            plan = os.path.join(d, "plan.json")
            for extra, tag in (([], "main"), (["--smoke"], "smoke")):
                r = subprocess.run(["bash", os.path.join(SCRIPTS, "run_sf43.sh"), "build", "all", "--dry-run", "--plan", plan] + extra,
                                   capture_output=True, text=True, timeout=300, env=dict(os.environ, REMOTE_GPU_COUNT_GRANTED="2"))
                self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
                line = next((x for x in r.stdout.splitlines() if "DRYRUN_SUMMARY" in x), None)
                self.assertIsNotNone(line, r.stdout[-500:])
                self.assertIn("complete=True", line)
                print("\nDRYRUN_SUMMARY (%s) %s" % (tag, line.split("DRYRUN_SUMMARY ", 1)[1]))


if __name__ == "__main__":
    unittest.main()
