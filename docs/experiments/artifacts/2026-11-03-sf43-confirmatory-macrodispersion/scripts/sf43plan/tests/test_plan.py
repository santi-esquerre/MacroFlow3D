"""Tests of the SF-43 plan generator: the generated files validate with the engine and the analysis, seeds are disjoint, CRN groups and
Gate 5 pairs are consistent, the runner configurations equal the SF-42 pilots', and an engine dry run of the plan completes.  ~15 s."""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest

from sf43camp.plan import load_plan
from sf43ana.config import load_prereg
from sf43plan import build as B
from sf43plan import design as D
from sf43plan import __main__ as CLI

S = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # <A>/scripts
SF42 = os.path.normpath(os.path.join(S, "..", "..", "2026-10-30-sf42-confirmatory-tooling", "scripts"))
if SF42 not in sys.path:
    sys.path.insert(0, SF42)

from mcstat import sizing  # noqa: E402
from pilots import design as pilots  # noqa: E402
from seedlattice import rule  # noqa: E402
from sf43camp import runner_executor as RE  # noqa: E402

# the forbidden ranges, written out independently of design.py
FORBIDDEN = [(1, 99999), (240000, 240999), (420000, 420999)]
MD5S = {"lagrangian_ensemble": "a" * 32, "macroflow3d_pipeline": "b" * 32}


def load(path):
    with open(path) as f:
        return json.load(f)


def read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def generate(out, binaries="placeholder", campaign_dir="sf43_campaign"):
    old = sys.stderr, sys.stdout
    with open(os.devnull, "w") as dn:
        sys.stderr = sys.stdout = dn
        try:
            CLI.main(["generate", "--variant", "A", "--campaign-dir", campaign_dir, "--binaries", binaries, "--out", out])
        finally:
            sys.stderr, sys.stdout = old


class Generated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = cls.tmp.name
        generate(cls.out)
        cls.plan = load(os.path.join(cls.out, "plan.json"))
        cls.smoke = cls.plan["smoke"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.out, name)

    def executor(self):
        """RunnerExecutor: constructed bare at the base, as Class(plan, build_dir) by the engine contract."""
        try:
            return RE.RunnerExecutor()
        except TypeError:
            return RE.RunnerExecutor(load_plan(self.path("plan.json")), "build")

    # ---------------------------------------------------------------------------------------------------------- validation
    def test_plan_validates_main_and_smoke(self):
        p = load_plan(self.path("plan.json"))
        self.assertEqual(len(p["arms"]), 44)
        s = load_plan(self.path("plan.json"), smoke=True, now=1.0e9)
        self.assertEqual(s["campaign_dir"], "sf43_campaign_smoke")
        self.assertNotEqual(s["campaign_dir"], p["campaign_dir"])
        self.assertEqual(s["cutoff"], 1000000000 + 60 * D.SMOKE["cutoff_minutes"])
        self.assertEqual(len(s["arms"]), 40)

    def test_prereg_validates(self):
        for name, narms, floor in (("prereg.json", 55, None), ("prereg_smoke.json", 47, 2)):
            r = load_prereg(self.path(name))
            self.assertEqual(len(r.arms), narms)
            self.assertEqual(r.family_ids(), D.PRIMARY_FAMILY)        # a FIXED family of m = 8
            if floor:
                self.assertTrue(all(a["floor"] == floor for a in r.arms.values()))
        r = load_prereg(self.path("prereg.json"))
        self.assertEqual(r.sigma2("s4"), 4.0)
        self.assertAlmostEqual(r.null_value("s1"), 7.992e-5)
        self.assertEqual(r.raw["bases"]["primary"], "disp")

    def test_prereg_matches_engine_outputs(self):
        """Reduced tables are reduced/<arm>/<segment>/<arm>__<segment>__d<lo>-<hi>.csv; ok verdicts are the engine's; hmax_div labels."""
        r = load("%s/prereg.json" % self.out)
        for label, (lo, hi) in r["windows"].items():
            self.assertEqual(r["reduced_pattern"].format(arm="A", segment="S", window=label), "A/S/A__S__d%g-%g.csv" % (lo, hi))
        self.assertEqual(r["primary_window"], "d25-94")
        self.assertEqual(r["ok_verdicts"], ["converged", "fallback"])
        arms = {a["id"]: a for a in r["arms"]}
        plan = {a["id"]: a for a in self.plan["arms"]}
        for aid, a in arms.items():
            e = plan[a.get("source", aid)]
            has_first = "hmax_div_first" in e.get("consumer", {})
            if a["role"] == "h_max_pass":
                self.assertEqual(a["hmax_div"], 16 if aid.endswith("_h16") else 8)
            else:
                self.assertEqual(a["hmax_div"], 8 if has_first else -1, aid)
        for lo, hi in ([25, 94], [25, 59.5], [59.5, 94]):
            self.assertIn([lo, hi], plan["s225_prod"]["reduce"]["d_windows"])
        self.assertIn([94, 194], plan["s4_dist"]["reduce"]["d_windows"])

    def test_prereg_references_plan_arms(self):
        ids = {a["id"] for a in self.plan["arms"]}
        for a in load(self.path("prereg.json"))["arms"]:
            self.assertIn(a.get("source", a["id"]), ids)
        sids = {a["id"] for a in self.smoke["arms"]}
        for a in load(self.path("prereg_smoke.json"))["arms"]:
            self.assertIn(a.get("source", a["id"]), sids)

    def test_prereg_seeds_are_the_plan_seeds(self):
        arms = {a["id"]: a for a in self.plan["arms"]}
        for a in load(self.path("prereg.json"))["arms"]:
            e = arms[a.get("source", a["id"])]
            if a["role"] in ("h_max_pass", "gate5_consumer_twin"):
                self.assertEqual(a["seeds"], list(range(e["seeds"]["start"], e["seeds"]["start"] + len(a["seeds"]))))
            else:
                self.assertEqual(a["seeds"], list(range(e["seeds"]["start"], e["seeds"]["start"] + e["seeds"]["count"])))
            r = self.plan["groups"][a["group"]]["reserves"]
            self.assertEqual(a["reserves"], list(range(r["start"], r["start"] + r["count"])))

    # ---------------------------------------------------------------------------------------------------------- seeds
    def seed_spans(self, plan_like):
        """(first, last, label) of every group's arm seed blocks and reserves, from the plan itself."""
        spans = {}
        for a in plan_like["arms"]:
            s, c = a["seeds"]["start"], a["seeds"]["count"]
            lo, hi = spans.get(a["group"], (s, s + c - 1))
            spans[a["group"]] = (min(lo, s), max(hi, s + c - 1))
        out = []
        for g, (lo, hi) in spans.items():
            out.append((lo, hi, g))
            r = plan_like["groups"][g]["reserves"]
            out.append((r["start"], r["start"] + r["count"] - 1, g + " reserves"))
        return out

    def test_seed_blocks_disjoint_and_not_forbidden(self):
        spans = self.seed_spans(self.plan) + self.seed_spans(self.smoke)
        spans.sort()
        for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
            self.assertLess(a1, b0, "%s overlaps %s" % (an, bn))
        for lo, hi, name in spans:
            self.assertTrue(430000 <= lo and hi <= 439999, name)
            for f0, f1 in FORBIDDEN:
                self.assertTrue(hi < f0 or lo > f1, "%s overlaps the forbidden range %d-%d" % (name, f0, f1))
            # Par2 transport seeds B + 1000 s of the Par2 arms: disjoint from the SF-42 series 420800001 + 1000 r (r < 100) and SF-39's
            t0, t1 = 420800001 + 1000 * lo, 420800001 + 1000 * hi
            for b0, b1 in ((420800001, 420800001 + 99000), (123456789, 123456789 + 99000), (123476789, 123476789 + 99000)):
                self.assertTrue(t1 < b0 or t0 > b1)
        self.assertTrue(B.check_seeds(load(self.path("seeds_registry.json"))))

    def test_registry_matches_plan(self):
        reg = load(self.path("seeds_registry.json"))
        blocks = {b["group"]: b for b in reg["blocks"] + reg["smoke_blocks"]}
        for plan_like in (self.plan, self.smoke):
            for g, d in plan_like["groups"].items():
                self.assertEqual(blocks[g]["reserves"], d["reserves"])
            for lo, hi, name in self.seed_spans(plan_like):
                if not name.endswith("reserves"):
                    self.assertEqual((blocks[name]["start"], blocks[name]["end"]), (lo, hi))
        for g, b in blocks.items():
            self.assertEqual(b["reserves"]["start"], b["end"] + 1)

    def test_check_seeds_detects_overlap_and_forbidden(self):
        reg = json.loads(json.dumps(load(self.path("seeds_registry.json"))))
        reg["blocks"][1]["start"] = reg["blocks"][0]["start"] + 5
        reg["blocks"][1]["end"] = reg["blocks"][1]["start"] + 399
        with self.assertRaises(ValueError):
            B.check_seeds(reg)
        reg = json.loads(json.dumps(load(self.path("seeds_registry.json"))))
        reg["blocks"][0].update(start=420500, end=420899, reserves={"start": 420900, "count": 5})
        with self.assertRaises(ValueError):
            B.check_seeds(reg)

    # ---------------------------------------------------------------------------------------------------------- CRN groups, pairs, units
    def test_crn_groups_share_start_and_priority(self):
        for plan_like in (self.plan, self.smoke):
            g = {}
            for a in plan_like["arms"]:
                k = g.setdefault(a["group"], (a["seeds"]["start"], a["priority"]))
                self.assertEqual(k, (a["seeds"]["start"], a["priority"]), a["id"])

    def test_priorities_follow_h13(self):
        byid = {a["id"]: a for a in self.plan["arms"]}
        for a in self.plan["arms"]:
            want = {"positive_control": 2, "distance": 3, "g7_par2": 3}.get(a["family"], 1)
            self.assertEqual(a["priority"], want, a["id"])
        self.assertEqual(byid["s4_l4"]["priority"], 1)                  # the sigma^2 = 4 rung feeds the primary family

    def test_gate5_pairs_share_seeds(self):
        byid = {a["id"]: a for a in self.plan["arms"]}
        twin = {"s025": "s025_prod", "s1": "s1_prod", "s225": "s225_c12", "s4": "s4_l16"}
        for amp, c in twin.items():
            p, k = byid[amp + "_g5p"], byid[c]
            self.assertEqual((p["group"], p["seeds"]["start"]), (k["group"], k["seeds"]["start"]))
            self.assertEqual(p["n_conf"], {"s025": 200, "s1": 200, "s225": 200, "s4": 250}[amp])
            self.assertLessEqual(p["n_conf"], k["n_max"])
            self.assertEqual(p["est_seconds"] > 0, True)
        pre = {a["id"]: a for a in load(self.path("prereg.json"))["arms"]}
        for amp, c in twin.items():
            self.assertEqual(pre[amp + "_g5c"]["seeds"], pre[amp + "_g5p"]["seeds"])
            self.assertEqual(pre[amp + "_g5c"]["n_conf"], pre[amp + "_g5p"]["n_conf"])

    def test_unit_size_divides_30_on_production_arms(self):
        for plan_like, n in ((self.plan, 30), (self.smoke, D.SMOKE["hmax_first_n"])):
            for a in plan_like["arms"]:
                h = a.get("consumer", {}).get("hmax_div_first")
                if a["family"] == "production":
                    self.assertEqual(h["n"], n)
                    self.assertEqual(h["divs"], [8, 16, 32])
                    self.assertEqual(h["n"] % a["unit_size"], 0, a["id"])
                else:
                    self.assertIsNone(h)
        for a in self.plan["arms"]:
            self.assertGreaterEqual(a["unit_size"], 5)
            self.assertEqual(a["n_conf"] % a["unit_size"], 0, a["id"])

    def test_reserves_fractions(self):
        for g, d in self.plan["groups"].items():
            n = max(a["seeds"]["count"] for a in self.plan["arms"] if a["group"] == g)
            frac = d["reserves"]["count"] / n
            self.assertGreaterEqual(frac, 0.10 if g.endswith("s4") else 0.05, g)
        for g, d in self.smoke["groups"].items():
            self.assertGreaterEqual(d["reserves"]["count"], 2)

    # ---------------------------------------------------------------------------------------------------------- design content
    def test_geometry_matches_the_specified_grids(self):
        want = {"s025_prod": (1040, 240, 240), "s025_eh5": (520, 120, 120), "s025_eh20": (2080, 480, 480), "s025_w48": (1040, 480, 480),
                "s1_dist": (2080, 240, 240), "s225_prod": (1664, 384, 384), "s225_eh4": (416, 96, 96), "s225_eh8": (832, 192, 192),
                "s225_w48_8": (832, 384, 384), "s225_w48_12": (1248, 576, 576), "s225_c12": (1248, 288, 288), "s225_dist": (3328, 384, 384),
                "s4_prod": (3328, 384, 384), "s4_l4": (416, 48, 48), "s4_l8": (832, 96, 96), "s4_l16": (1664, 192, 192),
                "s4_w24_8": (832, 192, 192), "s4_w24_16": (1664, 384, 384), "s4_dist": (3328, 192, 192), "pc_s004": (1040, 240, 240)}
        nn = {"s025_prod": (200, 400), "s025_eh5": (100, 100), "s025_eh20": (100, 100), "s025_w48": (150, 300), "s1_dist": (100, 100),
              "s225_prod": (200, 400), "s225_eh4": (150, 150), "s225_eh8": (150, 150), "s225_w48_8": (150, 300), "s225_w48_12": (150, 300),
              "s225_c12": (200, 200), "s225_dist": (100, 100), "s4_prod": (200, 400), "s4_l4": (100, 100), "s4_l8": (150, 150),
              "s4_l16": (250, 250), "s4_w24_8": (150, 300), "s4_w24_16": (150, 300), "s4_dist": (100, 100), "pc_s004": (30, 30),
              "s225_g5p": (200, 200), "s4_g5p": (250, 250), "s025_g5p": (200, 200), "s1_g5p": (200, 200)}
        byid = {a["id"]: a for a in self.plan["arms"]}
        for i, g in want.items():
            self.assertEqual(byid[i]["consumer"]["grid"], list(g), i)
        for i, n in nn.items():
            self.assertEqual((byid[i]["n_conf"], byid[i]["n_max"]), n, i)
        self.assertEqual(byid["s225_g5p"]["par2"]["grid"], [1248, 288, 288])
        self.assertEqual(byid["s4_g5p"]["par2"]["grid"], [1664, 192, 192])
        self.assertEqual(byid["s1_dist"]["consumer"]["planes_d"][-1], 194)
        self.assertEqual(byid["s1_dist"]["reduce"]["d_windows"], [[25, 94], [94, 194]])
        self.assertEqual(byid["pc_s004"]["consumer"]["eps"], 0.2)
        self.assertEqual(sum(a["family"] == "g7_par2" for a in self.plan["arms"]), 16)
        self.assertTrue(all(a["n_conf"] == 60 for a in self.plan["arms"] if a["family"] == "g7_par2"))

    def test_lattice_m_is_admissible_on_every_consumer_arm(self):
        for a in self.plan["arms"] + self.smoke["arms"]:
            if a["runner"] != "consumer":
                continue
            c = a["consumer"]
            self.assertFalse(rule.on_face(c["grid"][1], c["window"], c["np_side"])[0], a["id"])
            self.assertEqual(c["np_side"], 128 if a["amplitude"] in (0.25, 1.0, 0.04) else 100)

    def test_runner_configurations_equal_the_sf42_pilots(self):
        EPS = {0.25: "0.25", 1.0: "1", 2.25: "2.25", 4.0: "4"}
        for a in self.plan["arms"]:
            if a["amplitude"] not in EPS or a["family"] == "distance":
                continue
            c = a["consumer"] if a["runner"] == "consumer" else a["par2"]
            nx, ny, nz = c["grid"]
            lh = round(c["ell"] * ny)
            amp = EPS[a["amplitude"]]
            self.assertEqual(pilots.EPS[amp], c["eps"])
            self.assertEqual(c["window"], pilots.window(nx, ny, lh, amp), a["id"])
            self.assertEqual(c["ell"], pilots.ell(ny, lh))
            self.assertEqual([float(x) for x in pilots.planes(nx, ny, lh)], RE.planes_of(c["ell"], c["planes_d"], c["x0_ell"]), a["id"])
            if a["runner"] == "par2":
                dt, n = pilots.par2_time(nx, ny, lh, 3.6)
                want = (dt / 2.0, 2 * n) if c["dt_half"] else (dt, n)                  # factor D of G7: dt / 2 and exactly twice the steps
                self.assertEqual(RE.par2_time(nx, ny, c["ell"], c["hz"], c["dt_coeff"], c["dt_half"]), want, a["id"])
                self.assertEqual(RE.mg_levels(nx, ny, nz), pilots.mg_levels(nx, ny, nz))
                self.assertEqual((c["n_particles"], c["cg_max_iter"]), (10000, 2000))
                self.assertEqual(c["cg_rtol"], "1.0e-6" if a["id"].endswith("L") and a["family"] == "g7_par2" else "1.0e-10")
                self.assertEqual(c["transport_seed"], pilots.PAR2_SEED)
            else:
                argv = self.executor().command(a, [a["seeds"]["start"]], "/tmp/seg", "build")["argv"]
                self.assertEqual(argv[argv.index("--tol") + 1], "1e-8")
                self.assertEqual(argv[argv.index("--hmax-div") + 1], "8,16,32" if "hmax_div_first" in c else "8")
                if "hmax_div_first" in c:             # later positions run the single production pass
                    late = self.executor().command(a, [a["seeds"]["start"] + 30], "/tmp/seg", "build")["argv"]
                    self.assertEqual(late[late.index("--hmax-div") + 1], "8")
                self.assertEqual(argv[argv.index("--twin") + 1], "on")
                self.assertEqual(argv[argv.index("--pollock-bridge") + 1], "on")
                self.assertEqual([float(x) for x in argv[argv.index("--planes") + 1].split(",")], [float(x) for x in pilots.planes(nx, ny, lh)], a["id"])
                self.assertEqual(float(argv[argv.index("--ell") + 1]), pilots.ell(ny, lh))

    def test_par2_yaml_and_g7_factors(self):
        byid = {a["id"]: a for a in self.plan["arms"]}
        y = RE.par2_yaml(byid["s025_g5p"], list(range(431001, 431021)))
        self.assertIn("field_source: sf18_medium_b", y)
        self.assertIn("every: %d" % byid["s025_g5p"]["par2"]["snapshots_every"], y)
        for rid, (g, i, d, r) in B.G7_CELLS.items():
            a = byid[rid]
            self.assertEqual(g * i * d * r, 1)
            p = a["par2"]
            self.assertEqual(p["generator"], "sf18" if g > 0 else "random_spectral")
            self.assertEqual(p["inject"], "window" if i > 0 else "full")
            self.assertEqual(p["dt_half"], d > 0)
            self.assertEqual(p["cg_rtol"], "1.0e-10" if r > 0 else "1.0e-6")
            y = RE.par2_yaml(a, [a["seeds"]["start"], a["seeds"]["start"] + 1])
            self.assertEqual("inject_window" in y, i > 0)
            self.assertEqual("field_source" in y, g > 0)
            self.assertNotIn("snapshots", y)
        self.assertEqual(len({B.G7_CELLS[r][1:] for r in B.G7_CELLS if r.startswith("g7_s1_S")}), 4)

    def test_memory_classes(self):
        for a in self.plan["arms"]:
            near = a["id"] in ("s025_eh20", "s1_eh20", "s225_w48_12", "s225_dist", "s4_prod")
            half = a["id"] in ("s025_w48", "s1_w48", "s225_prod", "s4_w24_16")
            if near or half:
                self.assertEqual((a["heavy"], a["min_free_mib"]), (True, 30000 if near else 17000), a["id"])
            else:
                self.assertFalse(a["heavy"], a["id"])
                self.assertLess(a["min_free_mib"], 17000, a["id"])
            peak = D.PEAK_MIB.get((a["runner"][0], a["consumer" if a["runner"] == "consumer" else "par2"]["grid"][0],
                                   a["consumer" if a["runner"] == "consumer" else "par2"]["grid"][1]))
            if peak:
                self.assertGreaterEqual(a["min_free_mib"], peak, a["id"])

    def test_engine_fields(self):
        blinded = set(self.plan["blinded_reestimation"]["arms"])
        for a in self.plan["arms"]:
            self.assertEqual(a["primary"], a["family"] == "production")
            self.assertEqual(a["no_reestimation"], a["id"] not in blinded, a["id"])
            if a["id"] in blinded:
                self.assertIn(a["family"], ("production", "E_W"))
                self.assertEqual(a["n_max"], 2 * a["n_conf"])
            else:
                self.assertEqual(a["n_max"], a["n_conf"])
        self.assertEqual(self.plan["executor"], "sf43camp.runner_executor:RunnerExecutor")
        self.assertEqual(self.plan["lane_b_min_free_mib"], 17000)
        self.assertNotIn("sd_plan_ub", self.plan["blinded_reestimation"])
        for a in self.plan["arms"]:
            if a["family"] in ("gate5_par2", "gate5_consumer"):
                self.assertNotIn(a["id"], blinded)
        self.assertEqual(sorted(blinded), sorted(["s025_prod", "s025_w48", "s1_prod", "s1_w48", "s225_prod", "s225_w48_8", "s225_w48_12",
                                                  "s4_prod", "s4_w24_8", "s4_w24_16"]))

    def test_blinded_targets(self):
        from sf43camp import guards
        br = self.plan["blinded_reestimation"]
        n_conf = {a["id"]: a["n_conf"] for a in self.plan["arms"]}
        for aid in br["arms"]:
            self.assertEqual(set(br["hw_target"][aid]), {"alpha_TP", "alpha_TAP"})
            amp = next(a["amplitude"] for a in self.plan["arms"] if a["id"] == aid)
            key = {0.25: "s025", 1.0: "s1", 2.25: "s225", 4.0: "s4"}[amp]
            sd0, n0 = D.SD_PLAN[key]
            for est, hw in br["hw_target"][aid].items():
                self.assertEqual(hw, D.TARGETS[key])
                # H9: inert at the planning SD, and a several-fold SD (up to 20 times) asks for more than n_conf
                self.assertLessEqual(guards.n_required(sd0, n0, hw), n_conf[aid] // 4, (aid, est))
                self.assertGreater(guards.n_required(20 * sd0, n0, hw), n_conf[aid], (aid, est))
        s = self.smoke["blinded_reestimation"]
        self.assertEqual(s["arms"], ["sm_s025_prod"])
        self.assertEqual(self.smoke["arms"][0]["n_max"], D.SMOKE["reest_n_max"])
        # smoke: the scaled target makes the planning-size SD ask for more than n_max (so n_target grows to n_max)
        sd0, n0 = D.SD_PLAN["s025"]
        self.assertGreater(guards.n_required(sd0, 6, s["hw_target"]["sm_s025_prod"]["alpha_TP"]), D.SMOKE["reest_n_max"])

    def test_costs_against_design_cost_variant_a(self):
        ents = B.production_entries()
        arms_h, extra_h = B.total_gpu_h(ents)
        self.assertAlmostEqual(arms_h + extra_h, 59.6, delta=0.15)         # design_cost.py variant A: 59.6 GPU-h planning
        self.assertEqual(round(extra_h, 2), 1.88)
        # the only differences from design_cost: the 2.25 rung 416x96 has 150 (design_cost 100)
        self.assertAlmostEqual(50 * 5.1 / 3600.0, 0.0708, places=3)

    def test_analysis_hashes_and_command(self):
        import hashlib
        h = self.plan["analysis"]["sha256_list"]
        self.assertGreater(len(h), 20)
        repo = B.REPO
        for rel, want in h.items():
            self.assertFalse(os.path.isabs(rel))
            self.assertNotIn("/tests/", rel)
            with open(os.path.join(repo, rel), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), want)
        self.assertTrue(any(k.endswith("sf43ana/gate4.py") for k in h))
        self.assertEqual(self.plan["analysis"]["command"][:2], ["bash", B.ART_REL + "/scripts/analyse_sf43.sh"])
        self.assertEqual(self.plan["analysis"]["command"][2:], ["{campaign_dir}", "{build_dir}"])

    def test_smoke_section(self):
        s = self.smoke
        ids = [a["id"] for a in s["arms"]]
        self.assertEqual(len(set(ids)), len(ids))
        keys = {(a["amplitude"], a["consumer" if a["runner"] == "consumer" else "par2"]["grid"][0],
                 a["consumer" if a["runner"] == "consumer" else "par2"]["grid"][1], a["runner"]) for a in s["arms"] if not a["id"].startswith("sm_det")
                and a["family"] != "g7_par2"}
        prod = {(a["amplitude"], a["consumer" if a["runner"] == "consumer" else "par2"]["grid"][0],
                 a["consumer" if a["runner"] == "consumer" else "par2"]["grid"][1], a["runner"]) for a in self.plan["arms"] if a["family"] != "g7_par2"}
        self.assertEqual(keys, prod)                                         # one smoke arm per distinct (amplitude, grid, runner)
        for a in s["arms"]:
            if a["family"] == "production":
                self.assertEqual(a["consumer"]["hmax_div_first"]["n"], 2)
                self.assertGreaterEqual(a["n_conf"], 2)
            self.assertLessEqual(a["n_max"], 14)
            if a["est_seconds"] > 100 and a["family"] != "production":
                self.assertEqual(a["n_conf"], 1, a["id"])
        pairs = s["determinism_pairs"]
        self.assertEqual(len(pairs), 2)
        byid = {a["id"]: a for a in s["arms"]}
        for p in pairs:
            a, b = byid[p[0]], byid[p[1]]
            self.assertEqual((a["lane"], b["lane"]), ("A", "B"))
            self.assertEqual((a["seeds"], a["priority"], a["group"]), (b["seeds"], b["priority"], b["group"]))
            self.assertEqual(a.get("consumer", a.get("par2")), b.get("consumer", b.get("par2")))
            self.assertEqual(a["priority"], 0)
        self.assertEqual({byid[p[0]]["runner"] for p in pairs}, {"consumer", "par2"})
        bg = s["background"][0]
        self.assertEqual((bg["name"], bg["nice"]), ("coverage", 19))
        self.assertEqual(s["background_wait_minutes"], 300)
        self.assertTrue(bg["argv"][1].endswith("sf43cov/run_cov.sh"))
        self.assertIn("32", bg["argv"])
        self.assertTrue(os.path.isfile(os.path.join(B.REPO, bg["argv"][1])))
        self.assertGreater(s["budget_gpu_h"], 0)
        self.assertLess(s["budget_gpu_h"], 10)
        self.assertEqual(s["blinded_reestimation"]["fraction"], D.SMOKE["reest_fraction"])
        self.assertNotIn("determinism_pairs", self.plan)

    def test_smoke_prereg_covers_every_code_path(self):
        r = load(self.path("prereg_smoke.json"))
        self.assertEqual(len(r["g7"]["factorial"]["s025"]), 8)
        self.assertEqual(set(r["gate5"]["amplitudes"]), {"s025", "s1", "s225", "s4"})
        # arms with a single realization exist (below_floor paths) although the pre-registration needs n_conf >= 2
        plan_n = {a["id"]: a["n_conf"] for a in self.smoke["arms"]}
        pre_n = {a["id"]: a["n_conf"] for a in r["arms"]}
        self.assertTrue(any(plan_n[i] == 1 and pre_n[i] == 2 for i in pre_n if i in plan_n))

    def test_placeholder_is_flagged_and_real_md5_is_not(self):
        self.assertIn("PLACEHOLDER", self.plan["binaries_note"])
        self.assertIn("PLACEHOLDER", read(self.path("design_table.md")))
        with tempfile.TemporaryDirectory() as d:
            bj = os.path.join(d, "md5.json")
            with open(bj, "w") as f:
                json.dump(MD5S, f)
            generate(os.path.join(d, "o"), binaries=bj)
            p = load(os.path.join(d, "o", "plan.json"))
            self.assertEqual(p["binaries"], MD5S)
            self.assertNotIn("binaries_note", p)
            self.assertNotIn("PLACEHOLDER", read(os.path.join(d, "o", "design_table.md")))
            self.assertEqual(load_plan(os.path.join(d, "o", "plan.json"))["binaries"], MD5S)

    def test_deterministic_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            generate(os.path.join(d, "a"))
            generate(os.path.join(d, "b"))
            for name in sorted(os.listdir(os.path.join(d, "a"))):
                self.assertEqual(read(os.path.join(d, "a", name), "rb"), read(os.path.join(d, "b", name), "rb"), name)
                self.assertEqual(read(os.path.join(d, "a", name), "rb"), read(self.path(name), "rb"), name)

    # ---------------------------------------------------------------------------------------------------------- engine dry run
    def dry_run(self, *extra):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([S, SF42]), REMOTE_GPU_COUNT_GRANTED="2")
        r = subprocess.run([sys.executable, "-W", "error", "-m", "sf43camp", "build", "all", "--plan", self.path("plan.json"), "--dry-run",
                            "--artifact", os.path.dirname(S)] + list(extra), cwd=S, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout[-500:] + r.stderr[-500:])
        m = re.search(r"DRYRUN_SUMMARY lanes=(\d+) complete=(\w+) stopping=(\S+) stage1_h=([\d.]+) stage2_h=([\d.]+) total_h=([\d.]+) gpu_h=([\d.]+) "
                      r"budget_gpu_h=(\S+) end=(\S+) cutoff=(\S+)", r.stdout)
        self.assertIsNotNone(m, r.stdout[-500:])
        print("\nDRYRUN_SUMMARY%s" % (" (smoke)" if extra else ""), m.group(0)[len("DRYRUN_SUMMARY "):])
        return {"complete": m.group(2) == "True", "stage1": float(m.group(4)), "stage2": float(m.group(5)), "wall": float(m.group(6)),
                "gpu_h": float(m.group(7)), "end": m.group(9), "cutoff": m.group(10)}

    def test_engine_dry_run_main(self):
        d = self.dry_run()
        self.assertTrue(d["complete"])
        self.assertLess(d["end"], d["cutoff"])
        self.assertLess(d["gpu_h"], D.BUDGET_GPU_H)
        # the sum of est_seconds over the n_conf realizations is the design_cost.py planning figure minus the three-pass extras
        work = sum(a["n_conf"] * a["est_seconds"] for a in self.plan["arms"]) / 3600.0
        self.assertAlmostEqual(work, 57.77, delta=0.1)
        self.assertGreaterEqual(d["gpu_h"], work)

    def test_engine_dry_run_smoke(self):
        d = self.dry_run("--smoke")
        self.assertTrue(d["complete"])
        self.assertLess(d["gpu_h"], self.smoke["budget_gpu_h"])
        self.assertLess(d["end"], d["cutoff"])


if __name__ == "__main__":
    unittest.main()
