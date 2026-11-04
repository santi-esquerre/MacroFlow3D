"""Plan generator: determinism, arm derivation (exact factor), schema (frozen loader), hashes, costs, drops, J8.  No GPU, no data.
Run from `scripts/`:  python3 -W error -m unittest discover -s sf43add/camp/tests -t .
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from sf43add.camp import cli
from sf43add.camp import plan as P
from sf43camp.plan import load_plan, utc_epoch

FROZEN = P.jload(os.path.join(P.ART, "plan.json"))
PRE = P.jload(os.path.join(P.ART, "prereg_addendum.json"))
ORDER = ["j1_base10", "j1_H3", "j1_D", "j1_DH3", "j1_H2", "j3_l5", "j9_w48"]
SCRIPTS = P.SCRIPTS
SF42 = os.path.normpath(os.path.join(P.ART, "..", "2026-10-30-sf42-confirmatory-tooling", "scripts"))


def build(**kw):
    kw.setdefault("allow_missing", True)      # sf43add/ana is the sibling node's package: absent until integration
    return P.build_plan(**kw)


class Derivation(unittest.TestCase):
    def test_two_runs_are_byte_identical(self):
        self.assertEqual(P.dumps(build()), P.dumps(build()))
        with tempfile.TemporaryDirectory() as t:
            outs = []
            for i in range(2):
                o = os.path.join(t, "p%d.json" % i)
                env = dict(os.environ, PYTHONPATH=SCRIPTS + os.pathsep + SF42)
                subprocess.run([sys.executable, "-W", "error", "-m", "sf43add.camp", "plan", "--allow-missing", "--out", o], check=True, cwd=SCRIPTS, env=env,
                               stdout=subprocess.DEVNULL)
                with open(o, "rb") as f:
                    outs.append(f.read())
            self.assertEqual(outs[0], outs[1])
            self.assertEqual(outs[0].decode(), P.dumps(build()))

    def test_arms_are_exact_copies_but_for_the_factor(self):
        plan = build()
        froz = {a["id"]: a for a in FROZEN["arms"]}
        self.assertEqual([a["id"] for a in plan["arms"]], ORDER)
        j3 = {"grid": [1040, 120, 120]}
        j9 = {"grid": [1040, 240, 240], "ell": 0.020833333333333332, "window": "0.3,0.7,0.3,0.7"}
        want = {"j1_base10": ("s4_g5p", {}), "j1_H3": ("s4_g5p", {"hz": 10.8}), "j1_D": ("s4_g5p", {"dt_half": True}),
                "j1_DH3": ("s4_g5p", {"hz": 10.8, "dt_half": True}), "j1_H2": ("s4_g5p", {"hz": 7.2}), "j3_l5": ("s1_dist", j3), "j9_w48": ("s1_dist", j9)}
        for a in plan["arms"]:
            root, fac = want[a["id"]]
            o, r = froz[root], a["runner"]
            self.assertEqual(r, o["runner"])
            exp = copy.deepcopy(o[r])
            exp.update(fac)
            self.assertEqual(json.dumps(a[r], sort_keys=True), json.dumps(exp, sort_keys=True), a["id"])      # canonical text: 1 != 1.0
            self.assertEqual(json.dumps(a["reduce"], sort_keys=True), json.dumps(o["reduce"], sort_keys=True), a["id"])
            self.assertEqual((a["family"], a["amplitude"]), (o["family"], o["amplitude"]))
            self.assertEqual((a["n_conf"], a["n_max"], a["unit_size"], a["primary"], a["no_reestimation"], a["heavy"]),
                             (a["seeds"]["count"], a["seeds"]["count"], 10, False, True, False))
        # the Par2 knobs the executor maps to the pipeline stay those of s4_g5p (snapshots, rtol, transport stream)
        h3 = next(a for a in plan["arms"] if a["id"] == "j1_H3")["par2"]
        self.assertEqual((h3["snapshots_every"], h3["cg_rtol"], h3["transport_seed"], h3["dt_coeff"]), (3000, "1.0e-10", 420800001, 0.0244140625))
        j9a = next(a for a in plan["arms"] if a["id"] == "j9_w48")["consumer"]
        self.assertEqual((j9a["planes_d"], j9a["np_side"], j9a["hmax_div"], j9a["twin"], j9a["pollock_bridge"]),
                         (froz["s1_dist"]["consumer"]["planes_d"], 128, [8], True, True))

    def test_groups_priorities_seeds_retained(self):
        plan = build()
        self.assertEqual([a["priority"] for a in plan["arms"]], list(range(7)))
        self.assertEqual(len({a["group"] for a in plan["arms"]}), 7)                    # one arm per CRN group (engine rule)
        self.assertEqual(set(plan["groups"]), {a["group"] for a in plan["arms"]})
        self.assertEqual({a["id"]: (a["seeds"]["start"], a["seeds"]["count"]) for a in plan["arms"]},
                         {"j1_base10": (434001, 10), "j1_H3": (434001, 60), "j1_D": (434001, 60), "j1_DH3": (434001, 60), "j1_H2": (434001, 60),
                          "j3_l5": (436001, 100), "j9_w48": (438001, 150)})
        for a in plan["arms"]:
            self.assertEqual(a["retained"], [a["seeds"]["start"], a["seeds"]["start"] + 1])
            self.assertEqual(plan["groups"][a["group"]]["reserves"], next(s["reserves"] for s in PRE["arms"] if s["id"] == a["id"]))

    def test_assertion_catches_an_injected_extra_difference(self):
        real = P.apply_factor

        def sneaky(run, factor):
            real(run, factor)
            run["np_side" if "np_side" in run else "cg_max_iter"] = 7            # a difference the factor does not name
        with mock.patch.object(P, "apply_factor", sneaky):
            with self.assertRaisesRegex(P.AddendumPlanError, "difference"):
                P.build_arms(FROZEN, PRE)
        a = copy.deepcopy(next(x for x in FROZEN["arms"] if x["id"] == "s4_g5p"))
        b = copy.deepcopy(a)
        b["id"] = "x"
        P.check_arm(a, b, {})
        b["reduce"]["blind"]["weights"] = "flux"
        with self.assertRaisesRegex(P.AddendumPlanError, "reduce"):
            P.check_arm(a, b, {})
        b = copy.deepcopy(a)
        b["id"] = "x"
        b["par2"]["hz"] = 3.6 + 0.0                                              # a no-op factor value is not a difference: the check says so
        with self.assertRaisesRegex(P.AddendumPlanError, "difference"):
            P.check_arm(a, b, {"hz": 3.6})
        b["par2"]["hz"] = 1
        a["par2"]["hz"] = 1.0
        with self.assertRaisesRegex(P.AddendumPlanError, "difference"):          # 1 vs 1.0 is a difference of the text of the plan
            P.check_arm(a, b, {})

    def test_provisional_costs(self):
        got = {a["id"]: (a["est_seconds"], a["raw_gb"], a["min_free_mib"]) for a in build()["arms"]}
        self.assertEqual(got, {"j1_base10": (17.5, 0.065, 10000), "j1_H3": (52.5, 0.195, 10000), "j1_D": (35.0, 0.13, 10000),
                               "j1_DH3": (105.0, 0.39, 10000), "j1_H2": (35.0, 0.13, 10000), "j3_l5": (6.8, 0.561, 9000), "j9_w48": (27.0, 0.561, 9000)})


class Schema(unittest.TestCase):
    def test_frozen_loader_validates_all_three_forms(self):
        with tempfile.TemporaryDirectory() as t:
            for job, name in (("production", "p.json"), ("j8", "j8.json")):
                path = os.path.join(t, name)
                with open(path, "w") as f:
                    f.write(P.dumps(build(job=job)))
                p = load_plan(path)
                self.assertEqual(len(p["arms"]), 7 if job == "production" else 3)
                if job == "production":
                    s = load_plan(path, smoke=True, now=1.0e9)
                    self.assertEqual([a["id"] for a in s["arms"]], ["j1_base10", "j1_H3", "j1_DH3", "j3_l5", "j9_w48"])
                    self.assertEqual(s["campaign_dir"], "/home/sesquerre/sf43_addendum_smoke")
                    self.assertEqual(s["cutoff"], 1.0e9 + 240 * 60)
                    self.assertEqual(s["budget_gpu_h"], 1.0)

    def test_top_level_and_hooks(self):
        plan = build()
        self.assertEqual((plan["campaign"], plan["schema_version"], plan["cutoff_utc"], plan["grace_minutes"], plan["stage1_max_hours"], plan["budget_gpu_h"],
                          plan["disk_min_free_gb"], plan["campaign_dir"], plan["min_retained"], plan["failure"], plan["executor"]),
                         ("sf43-addendum-1", 1, "2026-11-07T02:30:00Z", 30, 0, 7.2, 100, "/home/sesquerre/sf43_addendum", 2,
                          {"halt_rate": 0.25, "min_for_rate": 20}, "sf43camp.runner_executor:RunnerExecutor"))
        self.assertLessEqual(utc_epoch(plan["cutoff_utc"]) + 60 * plan["grace_minutes"], utc_epoch("2026-11-07T03:00:00Z"))
        for k in ("blinded_reestimation", "background", "lane_b_min_free_mib", "determinism_pairs"):
            self.assertNotIn(k, plan)
        self.assertEqual(plan["binaries"], P.jload(os.path.join(P.ART, "binaries.json")))
        a = "docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion"
        self.assertEqual(" ".join(plan["analysis"]["command"]),
                         "python3 -W default -m sf43add.ana run --prereg %s/prereg_addendum.json --campaign {campaign_dir} --frozen-reduced "
                         "/home/sesquerre/sf43_campaign/reduced --frozen-manifest /home/sesquerre/sf43_campaign.freeze.reduced.sha256 --frozen-ledger "
                         "/home/sesquerre/sf43_campaign/ledger.jsonl --frozen-segments /home/sesquerre/sf43_campaign/segments --out {campaign_dir}/analysis" % a)
        self.assertEqual(" ".join(plan["smoke"]["analysis"]["command"]),
                         "python3 -W default -m sf43add.ana check --prereg %s/prereg_addendum.json --campaign {campaign_dir} --out {campaign_dir}/analysis" % a)
        self.assertEqual(" ".join(plan["recompute"]["command"]), "python3 -W error -m recompute43 run --campaign {campaign_dir} --reduced "
                         "{campaign_dir}/reduced --out {campaign_dir}.recompute.txt")
        self.assertEqual(" ".join(plan["reconcile"]["command"]), "python3 -W error -m reconcile43 run --campaign {campaign_dir} --out {campaign_dir}.reconcile.txt")

    def test_smoke_section(self):
        s = build()["smoke"]
        self.assertEqual({a["id"]: (a["seeds"]["count"], a["retained"], a["unit_size"], a["n_conf"], a["n_max"]) for a in s["arms"]},
                         {"j1_base10": (1, [434001], 1, 1, 1), "j1_H3": (2, [434001, 434002], 2, 2, 2), "j1_DH3": (2, [434001, 434002], 2, 2, 2),
                          "j3_l5": (1, [436001], 1, 1, 1), "j9_w48": (1, [438001], 1, 1, 1)})
        full = {a["id"]: a for a in build()["arms"]}
        for a in s["arms"]:        # same runner dict and reduce block as the production arm
            for k in (a["runner"], "reduce", "group", "priority"):
                self.assertEqual(a[k], full[a["id"]][k])
        self.assertEqual(set(s["groups"]), {a["group"] for a in s["arms"]})
        self.assertEqual(s["analysis"]["sha256_list"], build()["analysis"]["sha256_list"])


class Hashes(unittest.TestCase):
    def test_list_is_what_is_on_disk(self):
        h = build()["analysis"]["sha256_list"]
        base = "docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/"
        self.assertTrue(set(FROZEN["analysis"]["sha256_list"]) <= set(h))
        for p, v in FROZEN["analysis"]["sha256_list"].items():
            self.assertEqual(h[p], v, p)                                   # the frozen files are unchanged
        for p in (base + "prereg_addendum.json", base + "prereg.json", base + "scripts/sf43add/__init__.py", base + "scripts/sf43add/camp/plan.py",
                  base + "scripts/sf43camp/runner_executor.py", base + "scripts/sf43ana/loader.py", base + "scripts/recompute43/driver.py"):
            self.assertIn(p, h)
        self.assertFalse([p for p in h if "/tests" in p or "__pycache__" in p])
        for p, v in h.items():
            self.assertEqual(P.sha256_file(os.path.join(P.REPO, p)), v, p)

    def test_missing_file_fails_unless_allowed(self):
        with mock.patch.object(P, "REQUIRED", P.REQUIRED + ("sf43add/ana/not_there.py",)):
            with self.assertRaisesRegex(P.AddendumPlanError, "not_there.py"):
                P.build_plan(allow_missing=False)
            plan = P.build_plan(allow_missing=True)
            self.assertTrue(any(m.endswith("sf43add/ana/not_there.py") for m in plan["_provenance"]["missing_allowed"]))

    def test_a_changed_frozen_file_is_an_error(self):
        fz = copy.deepcopy(FROZEN)
        k = sorted(fz["analysis"]["sha256_list"])[0]
        fz["analysis"]["sha256_list"][k] = "0" * 64
        with self.assertRaisesRegex(P.AddendumPlanError, "frozen file changed"):
            P.analysis_hashes(fz, True)


class Costs(unittest.TestCase):
    def costs(self, drop=()):
        # exact line t = 10 + 5 m (m = steps relative to s4_g5p) on base (m 1), H3 (m 3), DH3 (m 6)
        arm = lambda m, **k: dict({"seconds_per_realization": 10.0 + 5.0 * m, "raw_gb_per_realization": 0.01 * m, "peak_mib": None}, **k)  # noqa: E731
        return {"arms": {"j1_base10": arm(1), "j1_H3": arm(3), "j1_DH3": arm(6), "j3_l5": {"seconds_per_realization": 8.0, "raw_gb_per_realization": 0.5, "peak_mib": 7000.0},
                         "j9_w48": {"seconds_per_realization": 40.0, "raw_gb_per_realization": 0.4, "peak_mib": 12000.0}}, "drop": list(drop), "sha256": "ab" * 32}

    def test_finalized_values(self):
        plan = build(costs=self.costs(), budget_gpu_h=6.1, disk_min_free_gb=250)
        self.assertEqual((plan["budget_gpu_h"], plan["disk_min_free_gb"]), (6.1, 250))
        got = {a["id"]: (a["est_seconds"], a["raw_gb"], a["min_free_mib"]) for a in plan["arms"]}
        self.assertEqual(got["j1_base10"], (round(1.25 * 15.0, 1), 0.01, 10000))   # Par2: no peak -> the frozen origin's 10000
        self.assertEqual(got["j1_H3"], (round(1.25 * 25.0, 1), 0.03, 10000))
        self.assertEqual(got["j1_DH3"], (round(1.25 * 40.0, 1), 0.06, 10000))
        self.assertEqual(got["j1_D"], (round(1.25 * 20.0, 1), 0.02, 10000))     # unmeasured: the fitted line at m = 2
        self.assertEqual(got["j1_H2"], (round(1.25 * 20.0, 1), 0.02, 10000))
        self.assertEqual(got["j3_l5"], (10.0, 0.5, 9000))                       # max(1.25 x 7000 = 8750, origin 9000)
        self.assertEqual(got["j9_w48"], (50.0, 0.4, 15000))                     # 1.25 x 12000
        self.assertEqual(plan["_provenance"]["costs_sha256"], "ab" * 32)

    def test_smoke_section_ignores_costs(self):
        self.assertEqual(build()["smoke"], build(costs=self.costs())["smoke"])

    def test_drops(self):
        plan = build(costs=self.costs(drop=["j9_w48"]), drop=["J1"])
        self.assertEqual([a["id"] for a in plan["arms"]], ["j3_l5"])
        self.assertEqual(set(plan["groups"]), {"a1_j3"})
        self.assertEqual(plan["_provenance"]["dropped"], ["j1_D", "j1_DH3", "j1_H2", "j1_H3", "j1_base10", "j9_w48"])
        with self.assertRaisesRegex(P.AddendumPlanError, "no such arm"):
            build(drop=["j7_nothing"])
        with self.assertRaisesRegex(P.AddendumPlanError, "every arm"):
            build(drop=["J1", "J3", "J9"])

    def test_linear_fit(self):
        self.assertAlmostEqual(P.linear_fit([(1, 15.0), (3, 25.0), (6, 40.0)], 2), 20.0)
        self.assertAlmostEqual(P.linear_fit([(1, 20.0)], 3), 60.0)                       # one point: proportional
        self.assertAlmostEqual(P.linear_fit([(1, 20.0), (3, 30.0)], 6), 45.0)            # a = 15, b = 5
        self.assertAlmostEqual(P.linear_fit([(1, 20.0), (3, 18.0)], 2), 20.0 * 2)         # decreasing: proportional bound max(y/m) m


class J8(unittest.TestCase):
    def test_j8_plan(self):
        plan = build(job="j8")
        self.assertEqual((plan["campaign"], plan["campaign_dir"], "smoke" in plan), ("sf43-addendum-1-j8", "/home/sesquerre/sf43_addendum_j8", False))
        self.assertEqual([a["id"] for a in plan["arms"]], ["j8_H3", "j8_D", "j8_DH3"])
        self.assertEqual({a["id"]: (a["seeds"], a["retained"], a["priority"]) for a in plan["arms"]},
                         {i: ({"start": 434061, "count": 40}, [434061, 434062], 0) for i in ("j8_H3", "j8_D", "j8_DH3")})
        prod = {a["id"]: a for a in build()["arms"]}
        for a, b in (("j8_H3", "j1_H3"), ("j8_D", "j1_D"), ("j8_DH3", "j1_DH3")):
            x = next(t for t in plan["arms"] if t["id"] == a)
            self.assertEqual(x["par2"], prod[b]["par2"])                              # identical settings, more seeds
            self.assertEqual(x["reduce"], prod[b]["reduce"])
        self.assertIn("--j8-campaign", plan["analysis"]["command"])
        i = plan["analysis"]["command"].index("--campaign")
        self.assertEqual(plan["analysis"]["command"][i:i + 4], ["--campaign", "/home/sesquerre/sf43_addendum", "--j8-campaign", "{campaign_dir}"])
        # J8 costs come from the measured J1 arms
        c = {"arms": {"j1_H3": {"seconds_per_realization": 20.0}, "j1_D": {"seconds_per_realization": 12.0}, "j1_DH3": {"seconds_per_realization": 30.0}}}
        self.assertEqual([a["est_seconds"] for a in build(job="j8", costs=c)["arms"]], [25.0, 15.0, 37.5])


class Binaries(unittest.TestCase):
    NEW = {"lagrangian_ensemble": "4099aa1aa7ce1ac5cdb0482552fdcb0f", "macroflow3d_pipeline": "de66d6a20e15ccbc1c429a51f5a06a87"}

    def write(self, t, d):
        path = os.path.join(t, "binaries_x.json")
        with open(path, "w") as f:
            json.dump(d, f)
        return path

    def test_default_is_binaries_json(self):
        plan = build()
        self.assertEqual(plan["binaries"], P.jload(os.path.join(P.ART, "binaries.json")))
        pv = plan["_provenance"]
        self.assertEqual((pv["binaries_path"], pv["binaries_sha256"]), (P.ART_REL + "/binaries.json", P.sha256_file(os.path.join(P.ART, "binaries.json"))))

    def test_other_file_with_provenance_key_is_used(self):
        with tempfile.TemporaryDirectory() as t:
            path = self.write(t, dict(self.NEW, _provenance="rebuilt at the addendum tree", _note={"x": 1}, extra="ignored"))
            for job in ("production", "j8"):
                plan = build(job=job, binaries_path=path)
                self.assertEqual(plan["binaries"], self.NEW)                                   # only the two md5 keys, no `_provenance`
                self.assertEqual((plan["_provenance"]["binaries_path"], plan["_provenance"]["binaries_sha256"]), (os.path.abspath(path),
                                                                                                                  P.sha256_file(path)))
                self.assertEqual(plan["arms"], build(job=job)["arms"])                       # nothing else moves
                pf = os.path.join(t, "plan_%s.json" % job)
                with open(pf, "w") as f:
                    f.write(P.dumps(plan))
                self.assertEqual(load_plan(pf)["binaries"], self.NEW)
            with mock.patch("sys.stdout"):
                o = os.path.join(t, "o.json")
                self.assertEqual(cli.main(["plan", "--allow-missing", "--binaries", path, "--out", o]), 0)
            self.assertEqual(P.jload(o)["binaries"], self.NEW)

    def test_invalid_or_missing_md5_fails(self):
        with tempfile.TemporaryDirectory() as t:
            for bad in (dict(self.NEW, macroflow3d_pipeline="de66d6a20e15ccbc1c429a51f5a0"), dict(self.NEW, lagrangian_ensemble="4099AA1AA7CE1AC5CDB0482552FDCB0F"),
                        dict(self.NEW, lagrangian_ensemble="z" * 32), dict(self.NEW, lagrangian_ensemble=12345), {"lagrangian_ensemble": self.NEW["lagrangian_ensemble"]}, []):
                path = self.write(t, bad)
                with self.assertRaisesRegex(P.AddendumPlanError, "md5|JSON object"):
                    build(binaries_path=path)
                with mock.patch("sys.stderr"):
                    self.assertEqual(cli.main(["plan", "--allow-missing", "--binaries", path, "--out", os.path.join(t, "never.json")]), 1)
                self.assertFalse(os.path.exists(os.path.join(t, "never.json")))


class Committed(unittest.TestCase):
    def test_committed_plans_are_current(self):
        """The committed plans are what the generator gives now.  Fails (by design) when a file the plan hashes changed, e.g. after the sibling
        package sf43add.ana is integrated: regenerate with `python3 -m sf43add.camp plan [--job j8] --binaries binaries_addendum.json` (and `finalize` after S0) and commit."""
        for job, name in (("production", "plan_addendum.json"), ("j8", "plan_addendum_j8.json")):
            with open(os.path.join(P.ART, name)) as f:
                self.assertEqual(f.read(), P.dumps(build(job=job, binaries_path=os.path.join(P.ART, "binaries_addendum.json"))), "%s is stale: regenerate it (see this docstring)" % name)


class Cli(unittest.TestCase):
    def test_finalize_needs_everything(self):
        for argv in (["finalize"], ["finalize", "--costs", "x.json", "--budget-gpu-h", "5"]):
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as e:
                cli.main(argv)
            self.assertEqual(e.exception.code, 2)

    def test_finalize_writes_the_plan(self):
        with tempfile.TemporaryDirectory() as t:
            c, o = os.path.join(t, "costs.json"), os.path.join(t, "plan.json")
            with open(c, "w") as f:
                json.dump({"arms": {"j3_l5": {"seconds_per_realization": 8.0, "peak_mib": 7000}}, "drop": ["j9_w48"]}, f)
            with mock.patch("sys.stdout"):
                self.assertEqual(cli.main(["finalize", "--costs", c, "--budget-gpu-h", "6.5", "--disk-min-free-gb", "140", "--allow-missing", "--out", o]), 0)
            p = load_plan(o)
            self.assertEqual((p["budget_gpu_h"], p["disk_min_free_gb"], "j9_w48" in [a["id"] for a in p["arms"]]), (6.5, 140, False))
            self.assertEqual(next(a for a in p["arms"] if a["id"] == "j3_l5")["est_seconds"], 10.0)


if __name__ == "__main__":
    unittest.main()
