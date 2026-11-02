"""Engine scenarios on a virtual clock with the FakeExecutor (hours of campaign time in milliseconds)."""
import os
import unittest
from unittest import mock

from sf43camp import lanes
from sf43camp.campaign import DONE, PARTIAL, STOPPED, ANALYSIS_FAILED, REFUSED

from .fx import H, T0, Sim, arm, mk_plan, tmpdir


def kinds(sim, k):
    return [r for r in sim.records() if r["kind"] == k]


def spawns(sim, gpu=None):
    return [s for s in sim.ex.spawned if gpu is None or s[2] == gpu]


class Stages(unittest.TestCase):
    def test_lane_b_heavy_only_and_released_when_heavy_queue_empties(self):
        p = mk_plan(tmpdir(self), [arm("H", n=10, est=100), arm("L", heavy=False, n=40, est=100)])
        s = Sim(p)
        self.assertEqual(s.run(), DONE)
        self.assertTrue(all(a == "H" for a, _, _, _ in spawns(s, "0")))  # lane B = second granted id, heavy only
        self.assertTrue(any(a == "L" for a, _, _, _ in spawns(s, "1")))
        self.assertEqual(len(kinds(s, "lane_release")), 1)
        self.assertIn("LANE_B_RELEASED gpu=0 utc=", s.log())
        rel = kinds(s, "lane_release")[0]["t"]
        self.assertTrue(all(t <= rel for _, _, _, t in spawns(s, "0")))  # nothing on B after the release
        gpu_h = s.state().gpu_s / H
        end = s.t - T0
        self.assertAlmostEqual(gpu_h, (2 * (rel - T0) + (s.t - rel)) / H, delta=0.1)  # 2 x stage 1 + stage 2
        self.assertGreater(end / H, 0)

    def test_stage1_cap_never_crossed(self):
        # units of 7 h: B starts at 0, 7, 14, 21 (ends 28); the next would end at 35 > 30 -> not started, B leaves
        p = mk_plan(tmpdir(self), [arm("H", n=100, est=7 * H / 5)], budget_gpu_h=1e6)
        s = Sim(p)
        self.assertEqual(s.run(), DONE)
        b = spawns(s, "0")
        self.assertEqual(len(b), 4)
        self.assertLessEqual(max(t for *_, t in b) + 7 * H, T0 + 30 * H)
        self.assertLess(kinds(s, "lane_release")[0]["t"], T0 + 30 * H)
        self.assertGreater(len(spawns(s, "1")), 4)  # lane A went on alone

    def test_single_gpu_fallback_runs_everything_on_lane_a(self):
        p = mk_plan(tmpdir(self), [arm("H", n=10), arm("L", heavy=False, n=10)])
        s = Sim(p)
        self.assertEqual(s.run(gpus=("7",)), DONE)
        self.assertEqual({g for _, _, g, _ in s.ex.spawned}, {"7"})
        self.assertEqual(kinds(s, "lane_release"), [])
        self.assertEqual(len(s.state().arms["L"].done), 10)

    def test_resume_after_stage1_runs_one_lane_and_keeps_stage1_start(self):
        p = mk_plan(tmpdir(self), [arm("H", n=100, est=7 * H / 5)])
        s = Sim(p)
        s.kill_at = T0 + 40 * H
        self.assertEqual(s.run(), "killed")
        start = kinds(s, "stage1_start")[0]["t"]
        s.kill_at = None
        n = len(s.ex.spawned)
        self.assertEqual(s.run(resume=True), DONE)
        self.assertIn("RESUME_SINGLE_LANE", s.log())
        self.assertEqual({g for _, _, g, _ in s.ex.spawned[n:]}, {"1"})
        self.assertEqual([r["t"] for r in kinds(s, "stage1_start")], [start])

    def test_memory_and_largest_first(self):
        p = mk_plan(tmpdir(self), [arm("H17", n=10, mem=17000), arm("H30", n=10, mem=30000)])
        s = Sim(p)
        self.assertEqual(s.run(gpu_free_mib=lambda g: 20000.0 if g == "0" else 40000.0), DONE)
        self.assertEqual(s.ex.spawned[0][0], "H30")  # lane A prefers the largest min_free_mib
        self.assertTrue(all(a == "H17" for a, _, _, _ in spawns(s, "0")))  # B never gets a 30 GB unit
        self.assertEqual(len(s.state().arms["H30"].done), 10)


class Failures(unittest.TestCase):
    def test_oom_requeue_with_backoff_is_not_a_failure(self):
        p = mk_plan(tmpdir(self), [arm("H", n=10, est=100), arm("L", heavy=False, n=10, est=100)], failure={"halt_rate": 0.01, "min_for_rate": 1})
        s = Sim(p, events={("H", 100): ["oom", "oom"]})
        self.assertEqual(s.run(), DONE)
        st = s.state()
        self.assertEqual(st.arms["H"].oom_total, 2)
        self.assertEqual(st.arms["H"].failed, set())
        self.assertIsNone(st.arms["H"].halted)  # a 1 % halt threshold is not tripped by OOMs
        starts = [t for a, sd, _, t in s.ex.spawned if a == "H" and sd[0] == 100]
        self.assertEqual(len(starts), 3)
        self.assertGreaterEqual(starts[1] - starts[0], 50 + 600)  # unit ends after half a seed (50 s), then 10 min
        self.assertGreaterEqual(starts[2] - starts[1], 50 + 1200)  # then 20 min
        self.assertEqual(s.log().count("OOM arm=H"), 2)
        self.assertEqual(len(st.arms["H"].done), 10)

    def test_unusable_solve_replaces_the_seed_in_every_arm_of_the_group(self):
        p = mk_plan(tmpdir(self), [arm("A1", n=10), arm("A2", n=10), arm("S", n=5, heavy=False), arm("B", group="g2", n=10, start=200)])
        s = Sim(p, events={("A1", 103): ["unusable_solve"]})
        self.assertEqual(s.run(), DONE)
        st = s.state()
        rep = kinds(s, "replacement")
        self.assertEqual([(r["group"], r["seed"], r["replaced_by"]) for r in rep], [("g1", 103, 900)])
        self.assertEqual(st.arms["A1"].failed, {103})
        for a in ("A1", "A2", "S"):
            self.assertIn(900, st.arms[a].done)
        self.assertIn(103, st.arms["A2"].dropped | st.arms["A2"].done)
        self.assertEqual(st.arms["B"].done, set(range(200, 210)))  # other groups untouched
        self.assertNotIn(900, st.arms["B"].done)
        self.assertEqual(st.groups["g1"].next_reserve, 1)
        self.assertNotIn(950, st.arms["B"].done)
        # the failed unit's tail (104) is re-queued from the first incomplete seed
        self.assertIn(104, st.arms["A1"].done)

    def test_config_error_halts_only_that_arm(self):
        p = mk_plan(tmpdir(self), [arm("A1", n=10), arm("A2", n=10)])
        s = Sim(p, events={("A1", 100): ["config_error"]})
        self.assertEqual(s.run(), PARTIAL)
        st = s.state()
        self.assertEqual(st.arms["A1"].halted, "config_error")
        self.assertEqual(len(st.arms["A2"].done), 10)
        self.assertTrue(os.path.exists(os.path.join(p["campaign_dir"], "DONE_PARTIAL")))

    def test_other_error_retried_once_then_halts(self):
        p = mk_plan(tmpdir(self), [arm("A1", n=10)])
        s = Sim(p, events={("A1", 100): ["other_error"]})
        self.assertEqual(s.run(), DONE)
        p2 = mk_plan(tmpdir(self), [arm("A1", n=10)])
        s2 = Sim(p2, events={("A1", 100): ["other_error", "other_error"]})
        self.assertEqual(s2.run(), PARTIAL)
        self.assertIn("other_error twice", s2.state().arms["A1"].halted)

    def test_failure_rate_halts_arm(self):
        p = mk_plan(tmpdir(self), [arm("A1", n=20, unit=1)], failure={"halt_rate": 0.1, "min_for_rate": 4})
        s = Sim(p, events={("A1", x): ["unusable_solve"] for x in (101, 102, 103)})
        self.assertEqual(s.run(), PARTIAL)
        self.assertIn("failure rate", s.state().arms["A1"].halted)

    def test_lane_crash_is_logged_and_other_lane_continues(self):
        p = mk_plan(tmpdir(self), [arm("H", n=20, est=100)])
        s = Sim(p)
        orig = s.ex.classify
        calls = []

        def bad(arm_, seeds, seg, rc, text):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return orig(arm_, seeds, seg, rc, text)

        s.ex.classify = bad
        self.assertEqual(s.run(), PARTIAL)  # the wrapper exit reflects the crash
        self.assertIn("LANE_CRASH", s.log())
        self.assertEqual(len(kinds(s, "lane_crash")), 1)
        self.assertGreater(len(s.state().arms["H"].done), 5)


class Guards(unittest.TestCase):
    def test_budget_stops_before_exceeding_and_analysis_still_runs(self):
        p = mk_plan(tmpdir(self), [arm("H", n=40, est=H / 5), arm("L", heavy=False, n=40, est=H / 5)], budget_gpu_h=6.0)
        s = Sim(p)
        self.assertEqual(s.run(), STOPPED)
        gpu_h = s.state().gpu_s / H
        self.assertLessEqual(gpu_h, 6.0 + 0.1)
        self.assertGreater(gpu_h, 4.0)
        self.assertEqual([c[0] for c in s.cmds], ["an", "rc", "rec"])
        self.assertIn("STOP reason=budget", s.log())
        self.assertTrue(os.path.exists(os.path.join(p["campaign_dir"], "STOPPED")))
        self.assertIn("PROJECTION", s.log())

    def test_disk_guard_prunes_reduced_segments_then_continues(self):
        p = mk_plan(tmpdir(self), [arm("H", n=50, unit=5, est=60, raw=1.0)], disk_min_free_gb=10)
        s = Sim(p)
        with mock.patch.object(lanes, "PRUNE_S", 1e12):  # only the guard prunes
            self.assertEqual(s.run(gpus=("1",), disk_free_gb=lambda d: 40.0 - s.ex.disk_used_gb()), DONE)
        self.assertGreater(len(s.ex.pruned), 0)
        self.assertIn("DISK_GUARD", s.log())
        self.assertNotIn("STOP reason=disk", s.log())

    def test_disk_guard_stops_when_pruning_cannot_help(self):
        p = mk_plan(tmpdir(self), [arm("H", n=30, unit=5, est=60, raw=1.0)], disk_min_free_gb=10)
        s = Sim(p)
        self.assertEqual(s.run(gpus=("1",), disk_free_gb=lambda d: 12.0 - s.ex.disk_used_gb() * 0.0 - len(s.ex.spawned) * 3.0), STOPPED)
        self.assertIn("STOP reason=disk", s.log())
        self.assertEqual([c[0] for c in s.cmds], ["an", "rc", "rec"])
        self.assertLess(len(s.ex.spawned), 6)

    def test_cutoff_with_grace_terminates_inflight_unit_and_keeps_completed_seeds(self):
        # 5 seeds x 24 min = 2 h per unit; cutoff at 3 h, grace 30 min: unit 2 (2 h..4 h) is cut at 3.5 h with 3 seeds done
        p = mk_plan(tmpdir(self), [arm("H", n=40, est=1440.0)], cutoff_h=3.0)
        s = Sim(p)
        self.assertEqual(s.run(gpus=("1",)), PARTIAL)
        ends = kinds(s, "unit_end")
        self.assertEqual([e["cls"] for e in ends], ["ok", "terminated"])
        self.assertEqual(len(ends[1]["completed"]), 3)
        self.assertEqual(len(spawns(s)), 2)  # nothing claimed after the cutoff
        self.assertLessEqual(s.t, T0 + 3.5 * H + 120)
        self.assertEqual([c[0] for c in s.cmds], ["an", "rc", "rec"])
        self.assertEqual(len(s.state().arms["H"].done), 8)

    def test_inflight_unit_may_finish_inside_the_grace(self):
        p = mk_plan(tmpdir(self), [arm("H", n=40, est=780.0)], cutoff_h=3.0)  # units of 65 min start at 0, 65, 130; the last ends at 195 min: after the cutoff, inside the grace
        s = Sim(p)
        self.assertEqual(s.run(gpus=("1",)), PARTIAL)
        self.assertEqual([e["cls"] for e in kinds(s, "unit_end")], ["ok", "ok", "ok"])


class Resume(unittest.TestCase):
    def test_kill_mid_unit_then_resume_reruns_only_incomplete_seeds_in_new_segment(self):
        p = mk_plan(tmpdir(self), [arm("A", n=20, est=100), arm("L", heavy=False, n=10, est=100)])
        s = Sim(p)
        s.kill_at = T0 + 800  # mid-unit on both lanes
        self.assertEqual(s.run(), "killed")
        first = len(s.ex.spawned)
        start = kinds(s, "stage1_start")[0]["t"]
        s.t += 600
        s.kill_at = None
        self.assertEqual(s.run(resume=True), DONE)
        ends = kinds(s, "unit_end")
        seen = {}
        for e in ends:
            for sd in e["completed"]:
                seen[(e["arm"], sd)] = seen.get((e["arm"], sd), 0) + 1
        self.assertTrue(all(v == 1 for v in seen.values()), "a completed seed ran twice")
        st = s.state()
        self.assertEqual(st.arms["A"].done, set(range(100, 120)))
        self.assertEqual(st.arms["L"].done, set(range(100, 110)))
        self.assertTrue(any(e["cls"] == "recovered" and e["completed"] for e in ends))
        segs = [c["segment"] for c in kinds(s, "claim")]
        self.assertEqual(len(segs), len(set(segs)))  # every (re-)run in a new segment directory
        rec = [e for e in ends if e["cls"] == "recovered"][0]
        claim = [c for c in kinds(s, "claim") if c["unit"] == rec["unit"]][0]
        again = [sd for a, sd, _, t in s.ex.spawned[first:] if a == claim["arm"] and set(sd) & set(claim["seeds"])]
        self.assertEqual(again[0][0], rec["next_seed"])  # from the first incomplete seed
        self.assertEqual([r["t"] for r in kinds(s, "stage1_start")], [start])
        self.assertEqual(len(kinds(s, "campaign_start")), 1)

    def test_stale_state_and_other_plan_are_refused(self):
        p = mk_plan(tmpdir(self), [arm("A", n=5)])
        s = Sim(p)
        self.assertEqual(s.run(), DONE)
        self.assertEqual(s.run(), REFUSED)  # ledger exists, no --resume
        s.plan["plan_sha256"] = "other"
        self.assertEqual(s.run(resume=True), REFUSED)
        self.assertIn("REFUSED stale state", s.log())

    def test_second_launcher_is_refused(self):
        p = mk_plan(tmpdir(self), [arm("A", n=5)])
        import fcntl
        os.makedirs(p["campaign_dir"], exist_ok=True)
        f = open(os.path.join(p["campaign_dir"], "LOCK"), "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        self.addCleanup(f.close)
        self.assertEqual(Sim(p).run(), REFUSED)


class Reestimation(unittest.TestCase):
    def plan(self, hw=1.0, **kw):
        a = arm("P", n=10, count=20, est=100)
        return mk_plan(tmpdir(self), [a, arm("G5", group="g2", n=10, start=200, no_reestimation=True)], blinded_reestimation={
            "fraction": 0.5, "max_factor": 2.0, "arms": ["P"], "hw_target": {"P": {"e": hw}}}, **kw)

    def test_n_raised_once_sds_only_other_arms_untouched(self):
        s = Sim(self.plan(), sd_obs={"P": {"e": 1.2}, "G5": {"e": 99.0}})
        self.assertEqual(s.run(), DONE)
        st = s.state()
        self.assertEqual(s.ex.blinded_calls[0][0], "P")
        self.assertEqual(len(s.ex.blinded_calls), 1)  # once, only the flagged arm
        r = kinds(s, "reest")[0]
        self.assertGreater(r["n_new"], 10)
        self.assertLessEqual(r["n_new"], 20)
        self.assertEqual(len(st.arms["P"].done), r["n_new"])
        self.assertEqual(len(st.arms["G5"].done), 10)
        self.assertEqual(set(r), {"kind", "arm", "sd", "n_obs", "n_req", "n_old", "n_new", "text", "t", "utc"})  # SDs and N only
        self.assertEqual((r["n_obs"], r["n_req"], r["n_new"]), (5, 16, 16))  # H9: ceil((t_{.95,4} sd_UB80 / hw)^2) at sd 1.2, n 5, hw 1
        self.assertIn("REESTIMATE arm=P", s.log())

    def test_never_lowered_and_no_growth_when_sd_below_plan(self):
        s = Sim(self.plan(hw=100.0), sd_obs={"P": {"e": 0.5}})
        self.assertEqual(s.run(), DONE)
        self.assertEqual(kinds(s, "reest")[0]["n_new"], 10)
        self.assertEqual(len(s.state().arms["P"].done), 10)

    def test_triggers_at_the_fraction_of_n_conf(self):
        s = Sim(self.plan(), sd_obs={"P": {"e": 1.2}})
        s.run()
        t_re = kinds(s, "reest")[0]["t"]
        red = [r for r in kinds(s, "reduced") if r["arm"] == "P"]
        self.assertEqual(sum(len(r["seeds"]) for r in red if r["t"] <= t_re), 5)  # fraction 0.5 x n_conf 10

    def test_resize_failure_is_recorded_not_fatal(self):
        s = Sim(self.plan())
        s.ex.blinded_sd = mock.Mock(side_effect=RuntimeError("no tables"))
        self.assertEqual(s.run(), DONE)
        self.assertIn("REESTIMATE_ERROR", s.log())
        self.assertEqual(len(s.state().arms["P"].done), 10)


class Analysis(unittest.TestCase):
    def test_runs_after_every_end_and_failures_give_12(self):
        p = mk_plan(tmpdir(self), [arm("A", n=5)])
        s = Sim(p)
        s.rc["rc"] = 1
        self.assertEqual(s.run(), ANALYSIS_FAILED)
        self.assertEqual([c[0] for c in s.cmds], ["an", "rc"])  # stops at the failing step
        self.assertEqual(s.cmds[0][1], p["campaign_dir"])  # placeholder substituted
        self.assertTrue(os.path.exists(os.path.join(p["campaign_dir"], "ANALYSIS_FAILED")))
        self.assertTrue(os.path.exists(os.path.join(p["campaign_dir"], "REPORT.txt")))

    def test_analysis_file_sha_mismatch_blocks_analysis(self):
        d = tmpdir(self)
        f = os.path.join(d, "an.py")
        with open(f, "w") as fh:
            fh.write("x")
        p = mk_plan(os.path.join(d, "c"), [arm("A", n=5)])
        p["analysis"]["sha256_list"] = {f: "0" * 64}
        s = Sim(p)
        self.assertEqual(s.run(), ANALYSIS_FAILED)
        self.assertEqual(s.cmds, [])

    def test_report_counts(self):
        p = mk_plan(tmpdir(self), [arm("A1", n=10), arm("A2", n=10)])
        s = Sim(p, events={("A1", 103): ["unusable_solve"], ("A2", 100): ["oom"]})
        s.run()
        with open(os.path.join(p["campaign_dir"], "REPORT.txt")) as fh:
            rep = fh.read()
        self.assertIn("gpu_h_used", rep)
        self.assertIn("replacement", rep)
        self.assertRegex(rep, r"A2\s+10\s+10\s+\d+\s+0\s+1")


if __name__ == "__main__":
    unittest.main()
