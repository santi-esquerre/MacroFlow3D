"""Tests of ``sf43camp.runner_executor`` with fake runner binaries (run from ``scripts/``:
``python3 -W error -m unittest discover -s sf43camp/tests_executor``).  The module and the helpers are loaded by path (the package
marker of ``sf43camp`` belongs to the engine core)."""
import copy
import csv
import glob
import importlib.util
import itertools
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rx = _load("sf43_runner_executor", os.path.join(HERE, "..", "runner_executor.py"))
my = _load("sf43_miniyaml", os.path.join(HERE, "miniyaml.py"))

ELL = 0.25
PLANES_D = [25, 30, 40, 60, 94, 120, 150, 194]
WINDOW = "0.375,0.625,0.375,0.625"


def consumer_arm(**over):
    c = {"grid": [208, 4, 4], "eps": 1.5, "ell": ELL, "window": WINDOW, "np_side": 8, "planes_d": PLANES_D, "hmax_div": [8],
         "twin": True, "pollock_bridge": True, "tol": "1e-8", "times": "auto"}
    c.update(over)
    return {"id": "c_s225", "family": "production", "amplitude": 2.25, "runner": "consumer", "raw_gb": 1.5, "consumer": c,
            "reduce": {"code": "consumer", "weights": ["flux"], "bases": ["pos"], "d_windows": [[25, 94], [94, 194]]}}


def par2_arm(**over):
    p = {"grid": [208, 4, 4], "eps": 1.5, "ell": ELL, "window": WINDOW, "planes_d": PLANES_D, "n_particles": 10000,
         "cg_max_iter": 2000, "cg_rtol": "1.0e-10", "hz": 3.6}
    p.update(over)
    return {"id": "p_s225", "family": "gate5", "amplitude": 2.25, "runner": "par2", "raw_gb": 0.1, "par2": p,
            "reduce": {"code": "par2", "weights": ["unif"], "bases": ["pos"], "d_windows": [[25, 94]]}}


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = os.path.realpath(self._tmp.name)
        self.build = os.path.join(self.root, "build")
        os.makedirs(self.build)
        for exe, mode in (("lagrangian_ensemble", "consumer"), ("macroflow3d_pipeline", "par2")):
            p = os.path.join(self.build, exe)
            with open(p, "w") as f:
                f.write('#!/bin/sh\nexec %s %s %s "$@"\n' % (sys.executable, os.path.join(HERE, "fake_runner.py"), mode))
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
        self.ex = rx.RunnerExecutor()

    def seg(self, name="seg0"):
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        return d

    def launch(self, arm, seeds, seg, **env):
        """Write the command's files, run it like the core would (stdout+stderr in log_path), return (exit code, log text)."""
        cmd = self.ex.command(arm, seeds, seg, self.build)
        for path, text in cmd["files"].items():
            with open(path, "w") as f:
                f.write(text)
        e = dict(os.environ, **cmd["env"])
        e.update({k: str(v) for k, v in env.items()})
        with open(cmd["log_path"], "w") as log:
            rc = subprocess.run(cmd["argv"], cwd=cmd["cwd"], env=e, stdout=log, stderr=subprocess.STDOUT).returncode
        with open(cmd["log_path"]) as f:
            return rc, f.read()


class TestConsumerCommand(Base):
    def test_production_command_line(self):
        cmd = self.ex.command(consumer_arm(), [1000, 1001, 1002], "/s/seg", "/b")
        planes = ",".join(repr(5.0 * ELL + d * ELL) for d in PLANES_D)
        self.assertEqual(cmd["argv"], [
            "/b/lagrangian_ensemble", "--medium", "b", "--grid", "208,4,4", "--sigma2", "1", "--eps", "1.5", "--ell", "0.25",
            "--window", WINDOW, "--np-side", "8", "--planes", planes, "--times", "auto", "--tol", "1e-8", "--hmax-div", "8",
            "--fused-splines", "--single-launch", "--twin", "on", "--pollock-bridge", "on", "--seed0", "1000", "--nr", "3",
            "--out", "/s/seg/out"])
        self.assertEqual((cmd["cwd"], cmd["log_path"], cmd["env"]), ("/s/seg", "/s/seg/run.log", {}))
        self.assertEqual(sorted(cmd["files"]), ["/s/seg/seedmap.json", "/s/seg/segment.json"])
        self.assertEqual(json.loads(cmd["files"]["/s/seg/seedmap.json"]), {"runner": "consumer", "seed0": 1000, "nr": 3,
                                                                          "seeds": {"0": 1000, "1": 1001, "2": 1002}})
        self.assertEqual(json.loads(cmd["files"]["/s/seg/segment.json"])["seed0"], 1000)

    def test_never_sets_a_gpu(self):
        for arm in (consumer_arm(), par2_arm()):
            cmd = self.ex.command(arm, [1, 2], "/s", "/b")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", cmd["env"])
            self.assertFalse(any("CUDA_VISIBLE_DEVICES" in a for a in cmd["argv"]))

    def test_ladder_rung(self):
        arm = consumer_arm(grid=[416, 96, 96], eps=0.5, ell=4 / 96, np_side=24, window="0.2,0.8,0.2,0.8", planes_d=[25, 94])
        a = self.ex.command(arm, [7], "/s", "/b")["argv"]
        self.assertEqual(a[a.index("--grid") + 1], "416,96,96")
        self.assertEqual(a[a.index("--eps") + 1], "0.5")
        self.assertEqual(float(a[a.index("--ell") + 1]), 4 / 96)
        self.assertEqual(a[a.index("--np-side") + 1], "24")
        pl = [float(x) for x in a[a.index("--planes") + 1].split(",")]
        self.assertEqual(pl, [5.0 * (4 / 96) + 25 * (4 / 96), 5.0 * (4 / 96) + 94 * (4 / 96)])
        self.assertEqual((a[a.index("--seed0") + 1], a[a.index("--nr") + 1]), ("7", "1"))

    def test_hmax_ladder_and_switches(self):
        a = self.ex.command(consumer_arm(hmax_div=[8, 16, 32]), [1, 2], "/s", "/b")["argv"]
        self.assertEqual(a[a.index("--hmax-div") + 1], "8,16,32")
        a = self.ex.command(consumer_arm(twin=False, pollock_bridge=False), [1], "/s", "/b")["argv"]
        self.assertEqual((a[a.index("--twin") + 1], a[a.index("--pollock-bridge") + 1]), ("off", "off"))

    def test_hmax_div_first_selects_the_pass_list_by_position(self):
        arm = consumer_arm(hmax_div_first={"n": 30, "divs": [8, 16, 32]})
        arm["seeds"] = {"start": 1000, "count": 200}
        div = lambda seeds: (lambda a: a[a.index("--hmax-div") + 1])(self.ex.command(arm, seeds, "/s", "/b")["argv"])  # noqa: E731
        self.assertEqual(div(list(range(1000, 1010))), "8,16,32")
        self.assertEqual(div(list(range(1020, 1030))), "8,16,32")  # the last unit inside the first 30 positions
        self.assertEqual(div(list(range(1030, 1040))), "8")  # later positions: the arm's own hmax_div
        self.assertEqual(div([1900]), "8")  # a reserve seed (outside the seed range) runs single-pass
        self.assertEqual(div([1100, 1101]), "8")  # an extension seed
        with self.assertRaises(ValueError):  # a unit straddling position n cannot happen with a unit_size that divides n
            div(list(range(1025, 1035)))
        self.assertEqual(json.loads(self.ex.command(arm, [1000, 1001], "/s", "/b")["files"]["/s/segment.json"])["hmax_div"], [8, 16, 32])

    def test_hmax_div_first_selects_the_pass_list_by_position(self):
        arm = consumer_arm(hmax_div_first={"n": 30, "divs": [8, 16, 32]})
        arm["seeds"] = {"start": 1000, "count": 200}
        div = lambda seeds: (lambda a: a[a.index("--hmax-div") + 1])(self.ex.command(arm, seeds, "/s", "/b")["argv"])  # noqa: E731
        self.assertEqual(div(list(range(1000, 1010))), "8,16,32")
        self.assertEqual(div(list(range(1020, 1030))), "8,16,32")  # the last unit inside the first 30 positions
        self.assertEqual(div(list(range(1030, 1040))), "8")  # later positions: the arm's own hmax_div
        self.assertEqual(div([1900]), "8")  # a reserve seed (outside the seed range) runs single-pass
        self.assertEqual(div([1100, 1101]), "8")  # an extension seed
        with self.assertRaises(ValueError):  # a unit straddling position n cannot happen with a unit_size that divides n
            div(list(range(1025, 1035)))
        self.assertEqual(json.loads(self.ex.command(arm, [1000, 1001], "/s", "/b")["files"]["/s/segment.json"])["hmax_div"], [8, 16, 32])

    def test_seeds_must_be_consecutive_and_keys_present(self):
        for bad in ([], [1, 3], [2, 1]):
            with self.assertRaises(ValueError):
                self.ex.command(consumer_arm(), bad, "/s", "/b")
        arm = consumer_arm()
        del arm["consumer"]["np_side"]
        with self.assertRaises(KeyError):
            self.ex.command(arm, [1], "/s", "/b")

    def test_planes_match_between_codes(self):
        c = self.ex.command(consumer_arm(), [1], "/s", "/b")["argv"]
        cfg = my.load(self.ex.command(par2_arm(), [1], "/s", "/b")["files"]["/s/config.yaml"])
        self.assertEqual([float(x) for x in c[c.index("--planes") + 1].split(",")], cfg["analysis"]["plane_crossings"]["planes"])


class TestPar2Config(Base):
    def cfg(self, arm, seeds=(5000, 5001, 5002)):
        text = self.ex.command(arm, list(seeds), "/s", "/b")["files"]["/s/config.yaml"]
        return my.load(text)

    def test_command(self):
        cmd = self.ex.command(par2_arm(), [5000, 5001], "/s", "/b")
        self.assertEqual(cmd["argv"], ["stdbuf", "-oL", "/b/macroflow3d_pipeline", "config.yaml"])
        self.assertEqual((cmd["cwd"], cmd["log_path"], cmd["env"]), ("/s", "/s/run.log", {}))
        self.assertEqual(sorted(cmd["files"]), ["/s/config.yaml", "/s/seedmap.json", "/s/segment.json"])
        self.assertEqual(json.loads(cmd["files"]["/s/seedmap.json"])["seeds"], {"0": 5000, "1": 5001})

    def test_matched_gate5_configuration(self):
        cfg = self.cfg(par2_arm())
        self.assertEqual(my.validate_par2(cfg), [])
        st, g, tr, fl = cfg["stochastic"], cfg["grid"], cfg["transport"], cfg["flow"]
        self.assertEqual(st["field_source"], "sf18_medium_b")
        self.assertEqual(st["sf18_medium_b"], {"sigma2": 1, "eps": 1.5, "seed0": 5000})
        self.assertEqual(st["K_mean"], 1.0)
        self.assertEqual(g["dx"], 1.0 / g["ny"])
        self.assertEqual(tr["inject_window"], [0.375, 0.625, 0.375, 0.625])
        self.assertEqual(tr["inject_x"], 5 * ELL)
        dt = 0.0244140625 * g["dx"]
        self.assertEqual(tr["dt"], dt)
        self.assertEqual(tr["n_steps"], math.ceil(3.6 * (208 * g["dx"] - 10 * ELL) / dt))
        self.assertEqual(tr["seed"], 420800001 + 1000 * 5000)
        self.assertEqual(fl["cg_rtol"], 1.0e-10)
        self.assertEqual(fl["cg_max_iter"], 2000)
        self.assertEqual(fl["bc"]["west"]["value"], 208 * (1.0 / 4))
        self.assertEqual(cfg["analysis"]["macrodispersion"]["NR"], 3)
        self.assertNotIn("snapshots", cfg["analysis"])

    def test_transport_seed_is_segmentation_invariant(self):
        a, b = self.cfg(par2_arm(), [100, 101, 102, 103]), self.cfg(par2_arm(), [102, 103])
        # field seed 102 is realization 2 of the first unit and realization 0 of the second: same stream
        self.assertEqual(a["transport"]["seed"] + 1000 * 2, b["transport"]["seed"] + 1000 * 0)

    def test_g7_cells(self):
        base = self.cfg(par2_arm())
        cells = 0
        for G, I, D, R in itertools.product([0, 1], repeat=4):
            over = {}
            if G:
                over.update(generator="random_spectral", n_modes=10000)
            if I:
                over["inject"] = "full"
            if D:
                over["dt_half"] = True
            if R:
                over["cg_rtol"] = "1.0e-6"
            cfg = self.cfg(par2_arm(**over))
            self.assertEqual(my.validate_par2(cfg), [], (G, I, D, R))
            cells += 1
            self.assertEqual(("field_source" in cfg["stochastic"], "sf18_medium_b" in cfg["stochastic"]), (not G, not G))
            if G:
                self.assertEqual((cfg["stochastic"]["sigma2"], cfg["stochastic"]["n_modes"], cfg["stochastic"]["seed"]), (2.25, 10000, 5000))
            else:
                self.assertEqual(cfg["stochastic"]["field_source"], "sf18_medium_b")
            self.assertEqual("inject_window" in cfg["transport"], not I)
            self.assertEqual(cfg["transport"]["dt"], base["transport"]["dt"] / (2 if D else 1))
            self.assertEqual(cfg["transport"]["n_steps"], base["transport"]["n_steps"] * (2 if D else 1))
            self.assertEqual(cfg["flow"]["cg_rtol"], 1.0e-6 if R else 1.0e-10)
            rest = copy.deepcopy(cfg)  # every other key equals the matched configuration's
            for k in ("field_source", "sf18_medium_b", "sigma2", "n_modes", "seed"):
                rest["stochastic"].pop(k, None)
            for k in ("dt", "n_steps", "inject_window", "output_every"):
                rest["transport"].pop(k, None)
            rest["flow"].pop("cg_rtol")
            ref = copy.deepcopy(base)
            for k in ("field_source", "sf18_medium_b", "sigma2", "n_modes", "seed"):
                ref["stochastic"].pop(k, None)
            for k in ("dt", "n_steps", "inject_window", "output_every"):
                ref["transport"].pop(k, None)
            ref["flow"].pop("cg_rtol")
            self.assertEqual(rest, ref)
        self.assertEqual(cells, 16)  # the 8-cell resolution-IV fraction is a subset

    def test_g_needs_n_modes_and_options_are_checked(self):
        with self.assertRaises(KeyError):
            self.cfg(par2_arm(generator="random_spectral"))
        with self.assertRaises(ValueError):
            self.cfg(par2_arm(generator="x"))
        with self.assertRaises(ValueError):
            self.cfg(par2_arm(inject="half"))

    def test_snapshots_only_when_requested(self):
        cfg = self.cfg(par2_arm(snapshots_every=500))
        self.assertEqual(cfg["analysis"]["snapshots"], {"enabled": True, "every": 500})
        self.assertEqual(my.validate_par2(cfg), [])

    def test_the_test_validator_rejects_what_the_real_one_rejects(self):
        cfg = self.cfg(par2_arm())
        bad = copy.deepcopy(cfg)
        bad["grid"]["dx"] = 0.25000001
        self.assertTrue(my.validate_par2(bad))
        bad = copy.deepcopy(cfg)
        bad["analysis"]["plane_crossings"]["planes"][-1] = 52.5
        self.assertTrue(my.validate_par2(bad))
        bad = copy.deepcopy(cfg)
        bad["stochastic"]["K_mean"] = 2.0
        self.assertTrue(my.validate_par2(bad))
        bad = copy.deepcopy(cfg)
        bad["transport"]["bogus"] = 1
        self.assertTrue(my.validate_par2(bad))


class TestClassify(Base):
    def cls(self, arm, text, code, seeds=(10, 11, 12)):
        return self.ex.classify(arm, list(seeds), self.seg("empty"), code, text)

    def test_oom_texts(self):
        for text in ("ERROR: CUDA error at /x/Stage.cu:88 - out of memory",
                     "ERROR: cuFFT error at /x/PeriodicGaussianField.cu:120 - code 2",
                     "ERROR: cuBLAS error at /x/CudaContext.cu:14 - code 1", "ERROR: cuBLAS error at /x/CudaContext.cu:14 - code 3",
                     "out of memory", "Failed to allocate RNG states", "Failed to allocate corner velocity buffers",
                     "Failed to allocate drift correction buffers", "Failed to allocate status array",
                     "Failed to allocate wrap counter arrays", "terminate called after throwing 'std::bad_alloc'"):
            for arm in (consumer_arm(), par2_arm()):
                r = self.cls(arm, "some progress\n" + text + "\n", 1)
                self.assertEqual(r["cls"], "oom", text)
                self.assertEqual(r["failed_seed"], 10)  # nothing complete: the first seed is the failing one

    def test_not_oom_cublas_other_codes(self):
        self.assertEqual(self.cls(consumer_arm(), "cuBLAS error at a.cu:1 - code 5", 1)["cls"], "other_error")

    def test_host_oom_killer(self):
        self.assertEqual(self.cls(consumer_arm(), "", 137)["cls"], "oom")
        self.assertEqual(self.cls(par2_arm(), "", -9)["cls"], "oom")

    def test_config_errors(self):
        for code in (2, 3, 7):
            self.assertEqual(self.cls(consumer_arm(), "ERROR: usage", code)["cls"], "config_error")

    def test_disk_full(self):
        for text in ("write_track_outputs: write failed for planes_x23.npy", "write_track_outputs: cannot write moments_time.csv"):
            self.assertEqual(self.cls(consumer_arm(), text, 1)["cls"], "disk_full")

    def test_unknown_error(self):
        self.assertEqual(self.cls(consumer_arm(), "segfault-ish", 139)["cls"], "other_error")
        self.assertEqual(self.cls(par2_arm(), "Config validation failed", 1)["cls"], "other_error")

    def test_exit_zero_with_missing_outputs_is_not_ok(self):
        r = self.cls(consumer_arm(), "RUN_DONE nr=3", 0)
        self.assertEqual(r["cls"], "other_error")

    def test_consumer_ok_and_fallback_is_kept(self):
        arm, seg = consumer_arm(), self.seg()
        rc, log = self.launch(arm, [4000, 4001, 4002], seg, FAKE_FALLBACK_AT=1)
        self.assertEqual(rc, 0)
        r = self.ex.classify(arm, [4000, 4001, 4002], seg, rc, log)
        self.assertEqual((r["cls"], r["completed_seeds"], r["failed_seed"]), ("ok", [4000, 4001, 4002], None))
        self.assertEqual(r["verdicts"], {4000: "converged", 4001: "fallback", 4002: "converged"})
        self.assertEqual(r["seconds"], {4000: 10.0, 4001: 11.0, 4002: 12.0})

    def test_exit_4_failed_seed_is_seed0_plus_done(self):
        arm, seg = consumer_arm(), self.seg()
        rc, log = self.launch(arm, [4000, 4001, 4002, 4003], seg, FAKE_FAIL4_AT=2)
        self.assertEqual(rc, 4)
        r = self.ex.classify(arm, [4000, 4001, 4002, 4003], seg, rc, log)
        self.assertEqual((r["cls"], r["completed_seeds"], r["failed_seed"]), ("unusable_solve", [4000, 4001], 4002))
        self.assertEqual(r["verdicts"][4002], "not_converged_beyond_fallback")
        self.assertEqual(r["verdicts"][4000], "converged")
        self.assertEqual(sorted(r["seconds"]), [4000, 4001])

    def test_exit_4_without_a_stage_failed_line_uses_n_done(self):
        arm, seg = consumer_arm(), self.seg()
        self.launch(arm, [4000, 4001, 4002, 4003], seg, FAKE_FAIL4_AT=1)
        r = self.ex.classify(arm, [4000, 4001, 4002, 4003], seg, 4, "")
        self.assertEqual((r["cls"], r["completed_seeds"], r["failed_seed"]), ("unusable_solve", [4000], 4001))

    def test_consumer_oom_mid_unit(self):
        arm, seg = consumer_arm(), self.seg()
        rc, log = self.launch(arm, [4000, 4001, 4002], seg, FAKE_ERR_AT=2, FAKE_ERR_TEXT="ERROR: CUDA error at E.cu:9 - out of memory")
        r = self.ex.classify(arm, [4000, 4001, 4002], seg, rc, log)
        self.assertEqual((rc, r["cls"], r["completed_seeds"], r["failed_seed"]), (1, "oom", [4000, 4001], 4002))

    def test_par2_ok_and_converged_no(self):
        arm, seg = par2_arm(), self.seg()
        rc, log = self.launch(arm, [7000, 7001, 7002], seg)
        r = self.ex.classify(arm, [7000, 7001, 7002], seg, rc, log)
        self.assertEqual((rc, r["cls"], r["completed_seeds"], r["failed_seed"]), (0, "ok", [7000, 7001, 7002], None))
        self.assertEqual(set(r["verdicts"].values()), {"converged"})
        self.assertEqual(sorted(r["seconds"]), [7000, 7001, 7002])
        seg2 = self.seg("seg2")
        rc, log = self.launch(arm, [7000, 7001, 7002], seg2, FAKE_NOCONV_AT=1)
        self.assertEqual(rc, 0)
        r = self.ex.classify(arm, [7000, 7001, 7002], seg2, rc, log)
        self.assertEqual((r["cls"], r["completed_seeds"], r["failed_seed"]), ("unusable_solve", [7000, 7002], 7001))
        self.assertEqual(r["verdicts"], {7000: "converged", 7001: "not_converged_beyond_fallback", 7002: "converged"})
        self.assertEqual(r["failed_seeds"], [7001])
        self.assertEqual(self.ex.completed(arm, seg2), [7000, 7002])

    def test_par2_solver_records_carry_iterations_and_the_printed_field_seed(self):
        arm, seg = par2_arm(), self.seg()
        rc, log = self.launch(arm, [7000, 7001], seg)
        r = self.ex.classify(arm, [7000, 7001], seg, rc, log)
        self.assertEqual({s: (v["iterations"], v["field_seed"], v["verdict"]) for s, v in r["solver"].items()},
                         {7000: (40, 7000, "converged"), 7001: (41, 7001, "converged")})
        self.assertNotIn("note", r)
        # Par2's own generator (G7 factor G) prints `K generation (seed=`; the seed is the same seed0 + r
        arm2, seg2 = par2_arm(generator="random_spectral", n_modes=10000), self.seg("g")
        rc, log = self.launch(arm2, [7000, 7001], seg2)
        self.assertIn("[4] K generation (seed=7001)", log)
        r = self.ex.classify(arm2, [7000, 7001], seg2, rc, log)
        self.assertEqual({s: v["field_seed"] for s, v in r["solver"].items()}, {7000: 7000, 7001: 7001})
        # a printed field seed that disagrees with the seed map is reported
        r = self.ex.classify(arm2, [7005, 7006], seg2, rc, log)
        self.assertTrue(r.get("note"))

    def test_par2_converged_no_is_usable_only_within_the_fallback_tolerance(self):
        arm, seg = par2_arm(), self.seg()
        rc, log = self.launch(arm, [7000, 7001, 7002], seg, FAKE_FALLBACK_AT=1)
        r = self.ex.classify(arm, [7000, 7001, 7002], seg, rc, log)
        self.assertEqual((r["cls"], r["completed_seeds"], r["failed_seed"]), ("ok", [7000, 7001, 7002], None))
        self.assertEqual(r["verdicts"], {7000: "converged", 7001: "fallback", 7002: "converged"})
        sol = rx.par2_solves
        self.assertEqual(sol("Converged=NO  iters=2000  res=1.00e+00 \u2192 9.99e-09")[0]["verdict"], "fallback")
        self.assertEqual(sol("Converged=NO  iters=2000  res=1.00e+00 -> 1.01e-08")[0]["verdict"], "not_converged_beyond_fallback")
        self.assertEqual(sol("Converged=NO  iters=2000  res=1.00e+00 -> nan")[0]["verdict"], "non_finite")
        self.assertEqual(sol("Converged=NO")[0]["verdict"], "not_converged_beyond_fallback")  # nothing printed: not usable
        self.assertEqual(sol("Converged=YES  iters=3  res=1.00e+00 -> 1.00e-12")[0]["verdict"], "converged")

    def test_par2_oom_mid_unit_is_not_unusable(self):
        arm, seg = par2_arm(), self.seg()
        rc, log = self.launch(arm, [7000, 7001, 7002], seg, FAKE_ERR_AT=2, FAKE_ERR_TEXT="Failed to allocate status array", FAKE_NOCONV_AT=0)
        r = self.ex.classify(arm, [7000, 7001, 7002], seg, rc, log)
        self.assertEqual((r["cls"], r["completed_seeds"], r["failed_seed"]), ("oom", [7001], 7000))


class TestCompleted(Base):
    def test_partial_consumer_segment(self):
        arm, seg = consumer_arm(), self.seg()
        self.launch(arm, [4000, 4001, 4002, 4003], seg, FAKE_FAIL4_AT=2)
        os.makedirs(os.path.join(seg, "out", "realizations", "r_0003"))  # an unfinished directory (no DONE) is not complete
        os.remove(os.path.join(seg, "out", "realizations", "r_0001", "DONE"))
        self.assertEqual(self.ex.completed(arm, seg), [4000])

    def test_missing_segment(self):
        self.assertEqual(self.ex.completed(consumer_arm(), os.path.join(self.root, "nothing")), [])
        self.assertEqual(self.ex.completed(par2_arm(), os.path.join(self.root, "nothing")), [])

    def test_partial_par2_segment(self):
        arm, seg = par2_arm(), self.seg()
        self.launch(arm, [7000, 7001, 7002], seg, FAKE_ERR_AT=2)
        self.assertEqual(self.ex.completed(arm, seg), [7000, 7001])
        p = os.path.join(seg, "out", "crossings", "realization_0001_planes.csv")  # a truncated CSV is not complete
        with open(p) as f:
            lines = f.readlines()
        with open(p, "w") as f:
            f.writelines(lines[:-2])
        self.assertEqual(self.ex.completed(arm, seg), [7000])


class TestReducePrune(Base):
    @classmethod
    def setUpClass(cls):
        cls._t = tempfile.TemporaryDirectory()
        cls.root = os.path.realpath(cls._t.name)
        cls.build = os.path.join(cls.root, "build")
        os.makedirs(cls.build)
        for exe, mode in (("lagrangian_ensemble", "consumer"), ("macroflow3d_pipeline", "par2")):
            p = os.path.join(cls.build, exe)
            with open(p, "w") as f:
                f.write('#!/bin/sh\nexec %s %s %s "$@"\n' % (sys.executable, os.path.join(HERE, "fake_runner.py"), mode))
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
        cls.ex = rx.RunnerExecutor()
        cls.seeds = list(range(3000, 3006))
        cls.arm = consumer_arm(hmax_div=[8, 16])
        cls.seg = os.path.join(cls.root, "seg_c")
        os.makedirs(cls.seg)
        t = Base("launch")
        t.ex, t.build = cls.ex, cls.build
        cls.rc, _ = t.launch(cls.arm, cls.seeds, cls.seg)
        cls.out = os.path.join(cls.root, "reduced")
        cls.red = cls.ex.reduce(cls.arm, cls.seg, cls.out)
        cls.parm = par2_arm()
        cls.pseg = os.path.join(cls.root, "seg_p")
        os.makedirs(cls.pseg)
        cls.prc, _ = t.launch(cls.parm, cls.seeds, cls.pseg)
        cls.pred = cls.ex.reduce(cls.parm, cls.pseg, os.path.join(cls.root, "reduced_p"))

    @classmethod
    def tearDownClass(cls):
        cls._t.cleanup()

    def setUp(self):  # the class-level fixtures replace Base.setUp
        pass

    def test_run_ok(self):
        self.assertEqual((self.rc, self.prc), (0, 0))

    def test_reduce_writes_one_table_per_window_with_manifest(self):
        r = self.red
        self.assertEqual(len(r["tables"]), 2)
        self.assertTrue(r["tables"][0].endswith("c_s225__seg_c__d25-94.csv"))
        self.assertTrue(r["tables"][1].endswith("c_s225__seg_c__d94-194.csv"))
        for t, pl, m in zip(r["tables"], r["planes"], r["manifests"]):
            for p in (t, pl, m):
                self.assertTrue(os.path.isfile(p), p)
            self.assertEqual(r["manifest_sha256"][os.path.basename(m)], rx._red.sha256_file(m))
            with open(m) as f:
                man = json.load(f)
            self.assertEqual(man["tables"][0]["sha256"], rx._red.sha256_file(t))
        with open(r["tables"][0]) as f:
            rows = list(csv.DictReader(f))
        sel = lambda code: [x for x in rows if x["code"] == code and x["family"] == "plane" and x["weights"] == "flux"  # noqa: E731
                            and x["basis"] == "pos" and x["set"] == "perplane"]
        plane = sel("consumer")
        self.assertEqual(len(plane), 6 * 2)  # 6 realizations x 2 passes
        self.assertEqual({x["hmax_div"] for x in plane}, {"8", "16"})
        self.assertEqual({x["d_lo"] for x in plane}, {"25"})
        # the Pollock bridge of the same realizations (pollock_bridge on by default): one set of rows per pass label, seed = the field seed
        pol = sel("pollock")
        self.assertEqual(len(pol), 6 * 2)
        self.assertEqual({(x["r"], x["seed"]) for x in pol}, {(str(r), str(3000 + r)) for r in range(6)})
        self.assertEqual({x["hmax_div"] for x in pol}, {"8", "16"})
        self.assertEqual({x["source"] for x in pol}, {"out/realizations/r_%04d/pollock" % r for r in range(6)})
        self.assertTrue(all(x["alpha_TP"] not in ("", "nan") for x in pol))
        self.assertTrue(all(x["alpha_TP"] != y["alpha_TP"] for x, y in zip(pol, plane) if x["r"] == y["r"]))  # the fake's bridge is another cloud
        self.assertEqual({x["weights"] for x in rows if x["code"] == "pollock"}, {"flux", "unif"})  # no expI, no time family
        self.assertEqual({x["family"] for x in rows if x["code"] == "pollock"}, {"plane"})

    def test_consumer_rows_carry_the_tracker_counts_of_their_own_pass(self):
        """H17: ``reduce --tracker-counts`` for consumer arms; the fake runner writes recrossings = seed % 5 + d // 8, backflow = seed % 3 and
        seed_backflow = seed % 2 in the track.json of pass d; Np = 8 x 8 = 64 traced (the fake's weights are uniform)."""
        with open(self.red["tables"][0]) as f:
            rows = list(csv.DictReader(f))
        cons = [x for x in rows if x["code"] == "consumer" and x["family"] == "plane"]
        self.assertTrue(cons)
        for x in cons:
            seed, d = int(x["seed"]), int(x["hmax_div"])
            self.assertEqual(int(x["n_traced"]), 64)
            self.assertAlmostEqual(float(x["recrossings_per_traced"]), (seed % 5 + d // 8) / 64.0, delta=1e-15)
            self.assertAlmostEqual(float(x["frac_backflow"]), (seed % 3) / 64.0, delta=1e-15)
            self.assertEqual(int(x["seed_backflow"]), seed % 2)
        for x in rows:
            if x["code"] != "consumer" or x["family"] != "plane":
                self.assertEqual(x["recrossings_per_traced"], "")
        with open(self.pred["tables"][0]) as f:  # Par2 tables: no tracker columns
            self.assertNotIn("frac_backflow", next(csv.reader(f)))

    def test_pollock_rows_are_off_when_the_arm_has_no_bridge(self):
        arm = consumer_arm(hmax_div=[8], pollock_bridge=False)
        seg = os.path.join(self.root, "seg_nopol")
        os.makedirs(seg)
        self.assertEqual(self.launch(arm, [3000, 3001], seg)[0], 0)
        self.assertFalse(os.path.exists(os.path.join(seg, "out", "realizations", "r_0000", "pollock")))
        red = self.ex.reduce(arm, seg, os.path.join(self.root, "reduced_nopol"))
        with open(red["tables"][0]) as f:
            self.assertEqual({x["code"] for x in csv.DictReader(f)}, {"consumer"})

    def test_par2_reduce_and_prune_is_a_noop(self):
        with open(self.pred["tables"][0]) as f:
            rows = [x for x in csv.DictReader(f) if x["family"] == "plane"]
        self.assertEqual({x["code"] for x in rows}, {"par2"})
        self.assertEqual(len({x["r"] for x in rows}), 6)
        res = self.ex.prune(self.parm, self.pseg, self.pred, [3000])
        self.assertEqual(res["deleted_files"], 0)

    def test_blinded_sd_is_flat_takes_reduce_info_dicts_and_is_shift_invariant(self):
        sd, n = self.ex.blinded_sd(self.arm, [self.red])  # the engine passes the reduce-result dicts
        self.assertEqual(n, 6)  # 6 seeds; window [25, 94], pass h/8, basis disp, flux weights: one row per seed
        self.assertLessEqual(set(sd), {"alpha_TP", "alpha_TAP", "alpha_LP"})
        self.assertIn("alpha_TP", sd)
        self.assertTrue(all(isinstance(v, float) and v >= 0 for v in sd.values()))
        self.assertGreater(sd["alpha_TP"], 0.0)  # synthetic a_T depends on the seed
        # one table path is accepted as well, and the second window or pass never leaks into the selection
        self.assertEqual(self.ex.blinded_sd(self.arm, [self.red["tables"][0]]), (sd, n))
        self.assertEqual(self.ex.blinded_sd(self.arm, [{"tables": self.red["tables"][:1]}]), (sd, n))
        self.assertEqual(self.ex.blinded_sd(self.arm, [{"tables": self.red["tables"][1:]}])[0], {})  # window [94, 194] only
        # shifting every estimator by a constant leaves the output unchanged: no location statistic enters the result
        shifted = []
        for t in self.red["tables"]:
            with open(t, newline="") as f:
                rd = csv.DictReader(f)
                rows, cols = list(rd), rd.fieldnames
            for row in rows:
                for e in ("alpha_TP", "alpha_TAP", "alpha_LP"):
                    if row[e] not in ("", "nan"):
                        row[e] = repr(float(row[e]) + 123.5)
            p = t + ".shift.csv"
            with open(p, "w", newline="") as f:
                wr = csv.DictWriter(f, cols, lineterminator="\n")
                wr.writeheader()
                wr.writerows(rows)
            shifted.append(p)
        sd2, n2 = self.ex.blinded_sd(self.arm, [{"tables": shifted}])
        self.assertEqual((set(sd2), n2), (set(sd), n))
        for k, v in sd.items():
            self.assertAlmostEqual(v, sd2[k], places=9)

    def test_blinded_sd_reads_consumer_rows_only_never_pollock(self):
        sd, n = self.ex.blinded_sd(self.arm, [self.red])
        bumped = []
        for t in self.red["tables"]:
            with open(t, newline="") as f:
                rd = csv.DictReader(f)
                rows, cols = list(rd), rd.fieldnames
            self.assertIn("pollock", {x["code"] for x in rows})
            for row in rows:
                if row["code"] == "pollock":  # wildly different Pollock estimates, even a Pollock-only seed: nothing may change
                    for e in ("alpha_TP", "alpha_TAP", "alpha_LP"):
                        row[e] = "1e6"
                    row["seed"] = str(int(row["seed"]) + 500)
            p = t + ".pollock.csv"
            with open(p, "w", newline="") as f:
                wr = csv.DictWriter(f, cols, lineterminator="\n")
                wr.writeheader()
                wr.writerows(rows)
            bumped.append(p)
        self.assertEqual(self.ex.blinded_sd(self.arm, [{"tables": bumped}]), (sd, n))

    def test_blinded_sd_selection_follows_the_arm_and_the_blind_override(self):
        pos = dict(self.arm, reduce=dict(self.arm["reduce"], blind={"basis": "pos", "hmax_div": 16}))
        sd, n = self.ex.blinded_sd(pos, [self.red])
        self.assertEqual(n, 6)
        self.assertNotEqual(sd["alpha_TP"], self.ex.blinded_sd(self.arm, [self.red])[0]["alpha_TP"])  # another basis, another SD
        sd_p, n_p = self.ex.blinded_sd(self.parm, [self.pred])  # Par2: code par2, uniform weights, hmax_div -1, seeds from the rows
        self.assertEqual(n_p, 6)
        self.assertIn("alpha_TP", sd_p)

    def test_blinded_sd_needs_four_values(self):
        small = os.path.join(self.root, "small.csv")
        with open(self.red["tables"][0], newline="") as f:
            rd = csv.DictReader(f)
            rows, cols = [x for x in rd if int(x["r"]) < 2], rd.fieldnames
        with open(small, "w", newline="") as f:
            wr = csv.DictWriter(f, cols, lineterminator="\n")
            wr.writeheader()
            wr.writerows(rows)
        sd, n = self.ex.blinded_sd(self.arm, [{"tables": [small]}])
        self.assertEqual(sd, {})  # counted, no SD
        self.assertEqual(n, 2)

    def test_par2_reduced_rows_carry_the_seed(self):
        with open(self.pred["tables"][0]) as f:
            rows = [x for x in csv.DictReader(f) if x["family"] == "plane"]
        self.assertEqual({(int(x["r"]), int(x["seed"])) for x in rows}, {(r, 3000 + r) for r in range(6)})

    def test_zz_prune_with_arm_level_retained_list(self):
        npy = lambda: sorted(glob.glob(os.path.join(self.seg, "out", "**", "*.npy"), recursive=True))  # noqa: E731
        before, others = npy(), sorted(p for p in glob.glob(os.path.join(self.seg, "out", "**", "*"), recursive=True)
                                       if os.path.isfile(p) and not p.endswith(".npy"))
        # arm-level list: two seeds of this segment and one of another segment (ignored); none needs to reach decision G8's 5 here
        res = self.ex.prune(self.arm, self.seg, self.red, [3001, 3004, 99999])
        self.assertEqual(res["retained"], [3001, 3004])
        left = npy()
        self.assertTrue(left and all("r_0001" in p or "r_0004" in p for p in left))
        self.assertEqual(len(before) - len(left), res["deleted_files"])
        self.assertGreater(res["deleted_bytes"], 0)
        self.assertEqual(sorted(p for p in glob.glob(os.path.join(self.seg, "out", "**", "*"), recursive=True)
                                if os.path.isfile(p) and not p.endswith(".npy")), others)  # nothing but .npy is touched
        self.assertTrue(os.path.isfile(res["log"]) and len(res["log_sha256"]) == 64)
        # the retained realizations keep every array of every pass; the reduced tables are untouched
        self.assertEqual(len([p for p in left if "r_0001" in p]), 2 * 9 + 7)  # + the seven arrays of its pollock/ directory
        self.assertEqual([p for p in left if "pollock_" in p and "r_0001" not in p and "r_0004" not in p], [])  # reduced and not retained: gone
        for t in self.red["tables"]:
            self.assertTrue(os.path.isfile(t))


class TestSelectivePrune(Base):
    def test_primary_arm_keeps_the_plane_arrays_of_every_realization(self):
        arm = dict(consumer_arm(hmax_div=[8, 16]), primary=True)
        seg = self.seg("p")
        self.assertEqual(self.launch(arm, [3000, 3001, 3002, 3003], seg)[0], 0)
        red = self.ex.reduce(arm, seg, os.path.join(self.root, "red"))
        res = self.ex.prune(arm, seg, red, [3001])
        left = glob.glob(os.path.join(seg, "out", "**", "*.npy"), recursive=True)
        names = lambda sel: {os.path.basename(x)[:-4] for x in left if sel(x)}  # noqa: E731
        self.assertEqual(res["kept_arrays"], ["seeds", "weights", "planes_x23", "planes_tau"])
        for r in (0, 2, 3):  # not retained: the four plane arrays stay in both passes, everything else goes
            self.assertEqual(names(lambda x, r=r: "r_%04d" % r in x), set(rx.KEEP_ARRAYS))
            self.assertEqual(len([x for x in left if "r_%04d" % r in x]), 2 * 4)
        self.assertEqual(len(names(lambda x: "r_0001" in x)), 9 + 7)  # the retained realization keeps all nine arrays and its seven pollock_ ones
        self.assertEqual(names(lambda x: "pollock_" in x and "r_0001" not in x), set())  # the others' Pollock arrays were reduced, then deleted
        self.assertGreater(res["deleted_files"], 0)
        # a non-primary arm keeps only the retained realizations (all arrays)
        arm2, seg2 = consumer_arm(), self.seg("q")
        self.launch(arm2, [3000, 3001], seg2)
        red2 = self.ex.reduce(arm2, seg2, os.path.join(self.root, "red2"))
        self.assertEqual(self.ex.prune(arm2, seg2, red2, [3001])["kept_arrays"], [])
        self.assertEqual({os.path.basename(x) for x in glob.glob(os.path.join(seg2, "out", "**", "r_0000", "*.npy"), recursive=True)}, set())

    def test_recompute_inputs_survive_for_the_displacement_basis(self):
        arm = dict(consumer_arm(), primary=True)
        seg = self.seg("p")
        self.launch(arm, [3000, 3001], seg)
        red = self.ex.reduce(arm, seg, os.path.join(self.root, "red"))
        self.ex.prune(arm, seg, red, [])
        for r in (0, 1):
            d = os.path.join(seg, "out", "realizations", "r_%04d" % r)
            self.assertEqual(sorted(x for x in os.listdir(d) if x.endswith(".npy")), ["planes_tau.npy", "planes_x23.npy", "seeds.npy", "weights.npy"])

    def test_par2_prune_deletes_regular_snapshots_keeps_crossings_and_retained_snapshots(self):
        arm = par2_arm(snapshots_every=100)
        seg = self.seg("s")
        self.assertEqual(self.launch(arm, [7000, 7001, 7002], seg)[0], 0)
        red = self.ex.reduce(arm, seg, os.path.join(self.root, "red"))
        snaps = lambda: sorted(glob.glob(os.path.join(seg, "out", "snapshots", "*", "*.csv")))  # noqa: E731
        self.assertEqual(len(snaps()), 3)
        res = self.ex.prune(arm, seg, red, [7001, 99999])
        self.assertEqual([os.path.basename(os.path.dirname(x)) for x in snaps()], ["r_0001"])
        self.assertEqual((res["deleted_files"], res["retained"]), (2, [7001]))
        self.assertEqual(len(glob.glob(os.path.join(seg, "out", "crossings", "*.csv"))), 3)
        # an altered input refuses before anything is deleted
        arm2, seg2 = par2_arm(snapshots_every=100), self.seg("s2")
        self.launch(arm2, [7000, 7001], seg2)
        red2 = self.ex.reduce(arm2, seg2, os.path.join(self.root, "red2"))
        with open(os.path.join(seg2, "out", "crossings", "realization_0000_planes.csv"), "a") as f:
            f.write("# tampered\n")
        with self.assertRaises(Exception):
            self.ex.prune(arm2, seg2, red2, [])
        self.assertEqual(len(glob.glob(os.path.join(seg2, "out", "snapshots", "*", "*.csv"))), 2)


class TestSmall(unittest.TestCase):
    def test_expected_raw_gb(self):
        self.assertEqual(rx.RunnerExecutor().expected_raw_gb({"raw_gb": 1.25}, 8), 10.0)

    def test_prune_refuses_without_a_reduced_table(self):
        with tempfile.TemporaryDirectory() as td:
            seg = os.path.join(td, "s")
            os.makedirs(os.path.join(seg, "out", "realizations"))
            with self.assertRaises(Exception):
                rx.RunnerExecutor().prune(consumer_arm(), seg, {"tables": [os.path.join(td, "none.csv")]}, [1])


if __name__ == "__main__":
    unittest.main()
