"""Corrective-preparation scenarios (executability review X1, X5, m3, m10, m13, F4, F7; coordinator addendum A, B): ordering and lane
rules, lane-B release, reduction retries, retention, solver/seed records, lane pins, determinism, background jobs, peak memory, plan
validation.  Fake executor, virtual clock."""
import json
import os
import sys
import unittest

from sf43camp.campaign import DONE, lane_b_class_mib
from sf43camp.lanes import Lane
from sf43camp.plan import DEFAULT_EXECUTOR, PlanError, load_plan
from sf43camp.scheduler import State

from .fx import T0, Sim, arm, mk_plan, tmpdir


def kinds(sim, k):
    return [r for r in sim.records() if r["kind"] == k]


def order(st):
    return [(u.arm, u.seeds[0]) for u in st.candidates()]


class Ordering(unittest.TestCase):
    def test_lane_a_takes_the_highest_pending_priority_before_any_near_empty_preference(self):
        p = mk_plan("x", [arm("L0", heavy=False, n=10, mem=8000), arm("H1", group="g2", start=200, n=10, prio=1, mem=30000)])
        st = State(p)
        self.assertEqual(st.pick(Lane("A", "1", False), 0, None, None)[0].arm, "L0")  # priority 0 light before priority 1 heavy (m3)
        self.assertEqual(st.pick(Lane("B", "0", True), 0, None, None)[0].arm, "H1")  # B: heavy only

    def test_near_empty_first_only_within_the_priority(self):
        p = mk_plan("x", [arm("H17", n=10, mem=17000), arm("H30", n=10, mem=30000), arm("H30b", group="g2", start=200, n=10, prio=1, mem=40000)])
        st = State(p)
        self.assertEqual(st.pick(Lane("A", "1", False), 0, None, None)[0].arm, "H30")
        self.assertEqual(st.pick(Lane("B", "0", True), 0, None, None)[0].arm, "H17")  # B: queue order, not the largest memory

    def test_progress_fraction_uses_n_conf_and_extension_seeds_are_priority_2_units(self):
        a = arm("P", n=10, count=20, prio=1)
        p = mk_plan("x", [a, arm("Q", group="g2", start=200, n=10, prio=1)])
        st = State(p)
        before = order(st)
        self.assertEqual(before, [("P", 100), ("Q", 200), ("P", 105), ("Q", 205)])
        key = lambda s, arm_id, seed: next(u.key for u in s.candidates() if (u.arm, u.seeds[0]) == (arm_id, seed))  # noqa: E731
        k0 = key(st, "P", 105)
        st.apply({"kind": "reest", "arm": "P", "n_new": 20, "utc": "u", "text": ""})
        self.assertEqual(key(st, "P", 105), k0)  # the group size is n_conf, not the re-sized n_target
        self.assertEqual(order(st), [("P", 100), ("Q", 200), ("P", 105), ("Q", 205), ("P", 110), ("P", 115)])
        self.assertEqual((key(st, "P", 110)[0], key(st, "P", 110)[1]), (2, 1.0))  # extension: priority 2, fraction on n_conf
        st.apply({"kind": "reest", "arm": "Q", "n_new": 10, "utc": "u", "text": ""})  # no growth: nothing changes
        self.assertEqual(len(order(st)), 6)

    def test_extension_units_never_mix_with_planned_seeds(self):
        p = mk_plan("x", [arm("P", n=10, count=20, unit=4)])  # n_conf 10 is not a multiple of 4
        st = State(p)
        st.arms["P"].done |= set(range(100, 108))
        st.apply({"kind": "reest", "arm": "P", "n_new": 16, "utc": "u", "text": ""})
        us = st.candidates()
        self.assertEqual([(u.seeds, u.key[0]) for u in us], [([108, 109], 0), ([110, 111], 2), ([112, 113, 114, 115], 2)])

    def test_group_adjacency_survives_the_priority_rule(self):
        p = mk_plan("x", [arm("A", n=10), arm("B", n=10, heavy=False), arm("C", group="g2", start=200, n=10, prio=1)])
        self.assertEqual(order(State(p)), [("A", 100), ("B", 100), ("A", 105), ("B", 105), ("C", 200), ("C", 205)])


class LaneB(unittest.TestCase):
    def test_class_units_wait_for_memory_but_near_empty_units_are_never_waited_for(self):
        p = mk_plan("x", [arm("H30", n=5, mem=30000)])
        st, B = State(p), Lane("B", "0", True, 17000)
        self.assertEqual(st.pick(B, 0, lambda: 100.0, None), (None, None))  # not its class, does not fit: nothing -> release
        self.assertEqual(st.pick(B, 0, lambda: 40000.0, None)[0].arm, "H30")  # fits the live memory: taken
        p = mk_plan("x", [arm("H17", n=5, mem=17000), arm("H30", n=5, mem=30000, group="g2", start=200)])
        self.assertEqual(State(p).pick(B, 0, lambda: 100.0, None), (None, 120.0))  # its class waits
        self.assertEqual(State(p).pick(Lane("B", "0", True), 0, lambda: 100.0, None), (None, 120.0))  # no threshold: every heavy unit

    def test_lane_b_class_threshold_sources(self):
        self.assertEqual(lane_b_class_mib({"lane_b_min_free_mib": 17000}, {"REMOTE_GPU_MIN_FREE_MIB_2": "9"}), 17000)
        self.assertEqual(lane_b_class_mib({}, {"REMOTE_GPU_MIN_FREE_MIB_2": "17000", "REMOTE_GPU_MIN_FREE_MIB": "30000"}), 17000)
        self.assertEqual(lane_b_class_mib({}, {"REMOTE_GPU_MIN_FREE_MIB": "30000"}), 30000)
        self.assertIsNone(lane_b_class_mib({}, {}))

    def test_lane_b_is_released_when_only_near_empty_units_remain_and_logs_it(self):
        p = mk_plan(tmpdir(self), [arm("H17", n=20, mem=17000, est=100), arm("H30", n=20, mem=30000, est=2000, group="g2", start=200)],
                    lane_b_min_free_mib=17000)
        s = Sim(p)
        self.assertEqual(s.run(gpu_free_mib=lambda g: 20000.0 if g == "0" else 40000.0), DONE)
        b = [x for x in s.ex.spawned if x[2] == "0"]
        self.assertTrue(b and all(a == "H17" for a, *_ in b))  # B never gets a near-empty unit it cannot hold
        rel = kinds(s, "lane_release")[0]["t"]
        h30_end = max(t for a, _, g, t in s.ex.spawned if a == "H30") + 5 * 2000
        self.assertLess(rel, h30_end - 3 * 2000)  # it left long before the near-empty work was done: it did not wait for it
        self.assertIn("LANE_B_RELEASED gpu=0 utc=", s.log())
        self.assertTrue(all(t <= rel for _, _, g, t in s.ex.spawned if g == "0"))

    def test_lane_b_leaves_at_the_cap_even_while_waiting(self):
        p = mk_plan(tmpdir(self), [arm("H17", n=100, mem=17000, est=7 * 3600 / 5)], lane_b_min_free_mib=17000)  # 140 h of work for lane A
        s = Sim(p)
        s.run(gpu_free_mib=lambda g: 100.0 if g == "0" else 40000.0)  # B's GPU never has memory: it waits, holding the GPU, until the cap
        rel = kinds(s, "lane_release")
        self.assertEqual(len(rel), 1)
        self.assertTrue(T0 + 23 * 3600.0 <= rel[0]["t"] <= T0 + 30 * 3600.0)  # when the next unit (7 h) could no longer end inside the cap
        self.assertEqual([x for x in s.ex.spawned if x[2] == "0"], [])


class ThirdParty(unittest.TestCase):
    """Owner decision of 2026-11-02 (session): a colleague's compute process on lane B's GPU releases that GPU at the next claim boundary."""

    def plan(self):
        return mk_plan(tmpdir(self), [arm("H", n=40, mem=1000, est=1000)])

    def test_present_from_the_start_lane_b_never_claims_and_lane_a_finishes(self):
        s, seen = Sim(self.plan()), []
        self.assertEqual(s.run(third_party=lambda g: seen.append(g) or ([4242, 4243] if g == "0" else [])), DONE)
        self.assertEqual([x for x in s.ex.spawned if x[2] == "0"], [])
        self.assertIn("LANE_B_RELEASED gpu=0 reason=third_party pids=2 utc=", s.log())
        rec = kinds(s, "lane_release")
        self.assertEqual([(r["gpu"], r["reason"], r["pids"]) for r in rec], [("0", "third_party", 2)])
        self.assertEqual(set(seen), {"0"})  # lane A is never checked
        self.assertEqual(len(s.state().arms["H"].done), 40)

    def test_appearing_mid_run_the_unit_in_flight_finishes_first_and_B_never_comes_back(self):
        s = Sim(self.plan())
        t_on = T0 + 1500.0
        self.assertEqual(s.run(third_party=lambda g: [7] if g == "0" and s.t >= t_on else []), DONE)
        b = [x for x in s.ex.spawned if x[2] == "0"]
        rel = kinds(s, "lane_release")
        self.assertTrue(b and len(rel) == 1 and rel[0]["reason"] == "third_party")
        self.assertTrue(all(t <= t_on for *_, t in b))  # nothing claimed after the process appeared
        ends = [r for r in s.records() if r["kind"] == "unit_end" and r["lane"] == "B"]
        self.assertTrue(ends and all(r["cls"] == "ok" for r in ends))  # the running unit was not killed
        self.assertGreaterEqual(rel[0]["t"], max(r["t"] for r in ends))
        s.run(resume=True)  # a resumed session does not bring B back
        self.assertIn("RESUME_SINGLE_LANE", s.log())

    def test_no_third_party_changes_nothing(self):
        s = Sim(self.plan())
        self.assertEqual(s.run(third_party=lambda g: []), DONE)
        self.assertTrue([x for x in s.ex.spawned if x[2] == "0"])
        self.assertNotIn("reason=third_party", s.log())

    def test_nvidia_reader_excludes_the_engines_descendants_and_an_unreadable_query_means_none(self):
        import subprocess
        from unittest import mock
        from sf43camp import procs
        kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: (kid.kill(), kid.wait()))
        d = tmpdir(self)
        smi = os.path.join(d, "nvidia-smi")
        with open(smi, "w") as f:  # fake nvidia-smi: GPU 1 has our child and the test runner's parent, GPU 0 a foreign pid
            f.write("#!%s\nimport os, sys\nif os.path.exists(%r): sys.exit(1)\n" % (sys.executable, os.path.join(d, "fail")))
            f.write("a = ' '.join(sys.argv)\nif 'query-gpu=index,uuid' in a: print('0, GPU-a\\n1, GPU-b')\n")
            f.write("elif 'query-compute-apps' in a: print('GPU-b, %d\\nGPU-b, %d\\nGPU-a, 99999')\n" % (kid.pid, os.getppid()))
        os.chmod(smi, 0o755)
        with mock.patch.dict(os.environ, {"PATH": d + os.pathsep + os.environ["PATH"]}):
            self.assertEqual(procs.third_party_pids("1"), [os.getppid()])  # the engine's own child is not third-party
            self.assertEqual(procs.third_party_pids("0"), [99999])
            self.assertEqual(procs.third_party_pids("7"), [])  # unknown GPU
            open(os.path.join(d, "fail"), "w").close()
            self.assertEqual(procs.third_party_pids("1"), [])  # unreadable = assume none


class Reductions(unittest.TestCase):
    def test_a_failed_reduction_is_retried_and_nothing_is_blocked(self):
        p = mk_plan(tmpdir(self), [arm("A", n=10, est=2000)])
        s = Sim(p)
        orig, calls = s.ex.reduce, {}

        def flaky(a, seg, out):
            calls[seg] = calls.get(seg, 0) + 1
            if calls[seg] <= 2:
                raise RuntimeError("I/O error")
            return orig(a, seg, out)
        s.ex.reduce = flaky
        self.assertEqual(s.run(), DONE)
        self.assertEqual(len(kinds(s, "reduced")), 2)
        self.assertIn("REDUCE_ERROR arm=A", s.log())
        self.assertIn("REDUCE_RETRY arm=A", s.log())
        self.assertFalse(os.path.exists(os.path.join(p["campaign_dir"], "ANALYSIS_BLOCKED_UNREDUCED")))
        self.assertEqual(kinds(s, "unreduced"), [])

    def test_unreduced_segments_block_nothing_but_are_listed_and_the_analysis_still_runs(self):
        p = mk_plan(tmpdir(self), [arm("A", n=10, est=2000)])
        s = Sim(p)

        def dead(a, seg, out):
            if seg.endswith("_s100"):
                raise RuntimeError("disk full")
            return {"tables": [out + "/t.csv"]}
        s.ex.reduce = dead
        self.assertEqual(s.run(), DONE)
        un = kinds(s, "unreduced")
        self.assertEqual([(r["arm"], r["seeds"]) for r in un], [("A", [100, 101, 102, 103, 104])])
        blocked = os.path.join(p["campaign_dir"], "ANALYSIS_BLOCKED_UNREDUCED")
        with open(blocked) as f:
            self.assertIn("seeds [100, 101, 102, 103, 104]", f.read())
        self.assertTrue(s.cmds)  # the analysis ran
        self.assertIn("REDUCE_GAVE_UP", s.log())
        self.assertIn("FINAL_REDUCE arm=A", s.log())
        with open(os.path.join(p["campaign_dir"], "REPORT.txt")) as f:
            rep = f.read()
        self.assertIn("UNREDUCED", rep)
        self.assertRegex(rep, r"A segments/A/u\d+_s100 seeds \[100, 101, 102, 103, 104\]")
        # a resumed session retries them from scratch and clears the marker
        s.ex.reduce = lambda a, seg, out: {"tables": [out + "/t.csv"]}
        s.run(resume=True)
        self.assertFalse(os.path.exists(blocked))


class Retention(unittest.TestCase):
    def test_retain_set_positions_rank_and_replacements(self):
        p = mk_plan("x", [arm("A", n=10, retained=(103,))])
        st = State(p)
        a = st.arms["A"]
        self.assertEqual(sorted(st.retain_set(a, 5)), [100, 101, 102, 103, 104])
        st.apply({"kind": "replacement", "group": "g1", "seed": 103, "replaced_by": 900, "arm": "A", "utc": "u"})
        self.assertEqual(st.order_of(a)[:5], [100, 101, 102, 900, 104])
        self.assertEqual(sorted(st.retain_set(a, 5)), [100, 101, 102, 103, 104, 900])  # the replacement of a retained seed is retained
        a.done |= {100, 101, 102, 104, 105, 106, 107, 108, 109}  # position 3 (the reserve) is not done: the rank rule adds the next ones
        self.assertEqual(sorted(st.retain_set(a, 5)), [100, 101, 102, 103, 104, 105, 900])

    def test_every_pruned_segment_keeps_the_first_five_and_the_replacement_of_a_retained_seed(self):
        p = mk_plan(tmpdir(self), [arm("A", n=20, est=2000, retained=(103,)), arm("B", n=20, est=2000, retained=(), heavy=False)])
        s = Sim(p, events={("A", 103): ["unusable_solve"]})
        self.assertEqual(s.run(), DONE)
        kept = {}
        for arm_id, seg, keep in s.ex.pruned:
            kept.setdefault(arm_id, set()).update(keep)
        self.assertTrue(s.ex.pruned)
        need = {100, 101, 102, 104, 900}  # the first five positions: 103 is replaced by the reserve 900 (also the replacement of a retained seed)
        self.assertTrue(need <= kept["A"] and need <= kept["B"])
        for arm_id in "AB":  # a few extra seeds may be kept while a position was still open (rank rule), never more than a handful
            self.assertLessEqual(len(kept[arm_id]), 8)
            self.assertFalse(any(x >= 107 and x != 900 for x in kept[arm_id]))
        rep = open(os.path.join(p["campaign_dir"], "REPORT.txt")).read()
        self.assertRegex(rep, r"retention.*\n(  \S+ +\d+ / \d+\n)+")

    def test_retention_shortfall_is_an_event(self):
        p = mk_plan(tmpdir(self), [arm("A", n=5, est=2000), arm("Z", n=20, est=2000, heavy=False, group="g2", start=200)])
        s = Sim(p)
        s.ex.prune = lambda a, seg, info, keep: {"retained": []}  # a pruner that kept nothing
        s.run()
        self.assertIn("RETENTION_SHORT arm=A", s.log())


class Records(unittest.TestCase):
    def test_seed_records_carry_verdict_class_solver_numbers_and_the_replacement_key(self):
        p = mk_plan(tmpdir(self), [arm("A1", n=10), arm("A2", n=10)])
        s = Sim(p, events={("A1", 103): ["unusable_solve"]})
        self.assertEqual(s.run(), DONE)
        seeds = kinds(s, "seed")
        self.assertEqual({(r["arm"], r["seed"]) for r in seeds if r["arm"] == "A1"}, {("A1", x) for x in [100, 101, 102, 104, 105, 106, 107, 108, 109, 900]} | {("A1", 103)})
        bad = [r for r in seeds if r["class"] == "unusable_solve"]
        self.assertEqual([(r["arm"], r["seed"]) for r in bad], [("A1", 103)])
        ok = next(r for r in seeds if r["arm"] == "A2" and r["seed"] == 101)
        self.assertEqual((ok["verdict"], ok["class"], ok["group"], ok["iterations"], ok["max_div"], ok["seed0"]), ("converged", "ok", "g1", 30, 2.0e-08, 100))
        self.assertRegex(ok["segment"], r"^u\d+_s100$")
        self.assertTrue(ok["segment_path"].startswith("segments/A2/"))
        self.assertEqual([(r["seed"], r["replaced_by"], r["group"]) for r in kinds(s, "replacement")], [(103, 900, "g1")])

    def test_peak_memory_is_in_the_ledger_and_the_report(self):
        p = mk_plan(tmpdir(self), [arm("A", n=10, mem=17000), arm("L", n=5, heavy=False, mem=8000)])
        s = Sim(p)
        s.run()
        self.assertEqual({r["peak_mib"] for r in kinds(s, "unit_end") if r["arm"] == "A"}, {17000})
        rep = open(os.path.join(p["campaign_dir"], "REPORT.txt")).read()
        self.assertRegex(rep, r"\nA +\d+ +10 +0 +0 +0 +0 +17000 ")
        self.assertRegex(rep, r"\nL +\d+ +5 +0 +0 +0 +0 +8000 ")


class Pins(unittest.TestCase):
    def test_pinned_arms_stay_on_their_lane(self):
        p = mk_plan(tmpdir(self), [arm("X", n=10, lane="B"), arm("Y", n=10, lane="A", group="g2", start=200)])
        s = Sim(p)
        self.assertEqual(s.run(), DONE)
        self.assertEqual({(a, g) for a, _, g, _ in s.ex.spawned}, {("X", "0"), ("Y", "1")})
        self.assertNotIn("LANE_PIN_IGNORED", s.log())

    def test_a_pin_to_a_missing_lane_is_ignored_and_logged_once(self):
        p = mk_plan(tmpdir(self), [arm("X", n=10, lane="B"), arm("Y", n=10, lane="A", group="g2", start=200)])
        s = Sim(p)
        self.assertEqual(s.run(gpus=("7",)), DONE)
        self.assertEqual({g for _, _, g, _ in s.ex.spawned}, {"7"})
        self.assertEqual(s.log().count("LANE_PIN_IGNORED arm=X"), 1)
        self.assertNotIn("LANE_PIN_IGNORED arm=Y", s.log())

    def test_a_pin_to_a_released_lane_is_ignored(self):
        p = mk_plan(tmpdir(self), [arm("Z", n=10, lane="B", est=7 * 3600 / 5), arm("W", n=100, group="g2", start=200, est=7 * 3600 / 5)])
        s = Sim(p)
        self.assertEqual(s.run(), DONE)  # lane B leaves at the stage-1 cap; whatever is still pinned to it goes to lane A
        self.assertEqual(len(s.state().arms["Z"].done), 10)


class Background(unittest.TestCase):
    def py(self, code):
        return [sys.executable, "-c", code]

    def test_job_runs_with_no_gpu_next_to_the_units_and_is_waited_for_and_reported(self):
        d = tmpdir(self)
        p = mk_plan(os.path.join(d, "c"), [arm("A", n=5)], background_wait_minutes=1,
                    background=[{"name": "cov", "argv": self.py("import os,sys; print(repr(os.environ['CUDA_VISIBLE_DEVICES']), os.environ['X'], sys.argv[1:])") + ["{campaign_dir}"],
                                 "nice": 19, "env": {"X": "1"}}])
        s = Sim(p)
        self.assertEqual(s.run(), DONE)
        cdir = p["campaign_dir"]
        with open(os.path.join(cdir, "background", "cov.log")) as f:
            self.assertEqual(f.read().strip(), "'' 1 ['%s']" % cdir)
        self.assertEqual(open(os.path.join(cdir, "background", "cov.done")).read().strip(), "0")
        self.assertEqual([r["rc"] for r in kinds(s, "background_end")], [0])
        self.assertEqual(len(kinds(s, "background_start")), 1)
        rep = open(os.path.join(cdir, "REPORT.txt")).read()
        self.assertRegex(rep, r"background jobs:\n  cov +done rc=0")
        # resume: the done file is respected
        s.run(resume=True)
        self.assertIn("BACKGROUND_SKIPPED name=cov", s.log())
        self.assertEqual(len(kinds(s, "background_start")), 1)

    def test_only_a_job_that_exits_by_itself_with_rc0_writes_the_done_file(self):
        import signal
        import time
        from sf43camp.background import Background
        d = tmpdir(self)
        p = {"background": [{"name": "k", "argv": self.py("import time; time.sleep(60)")}, {"name": "f", "argv": self.py("import sys; sys.exit(3)")},
                            {"name": "ok", "argv": self.py("pass")}]}
        recs = []
        b = Background(p, "bd", d, lambda m: None)
        b.start(lambda **r: recs.append(r))
        os.kill(b.procs["k"].pid, signal.SIGTERM)  # killed from outside (the group kill of a cancel)
        for _ in range(100):
            b.poll(lambda **r: recs.append(r))
            if not b.procs:
                break
            time.sleep(0.1)

        def done(n):
            return os.path.exists(b.done_file(n))
        self.assertEqual((done("k"), done("f"), done("ok")), (False, False, True))
        self.assertEqual({r["name"]: r["rc"] for r in recs if r["kind"] == "background_end"}, {"k": -15, "f": 3, "ok": 0})
        # terminated by the engine: no done file either
        q = {"background": [{"name": "t", "argv": self.py("import time; time.sleep(60)")}]}
        b2 = Background(q, "bd", d, lambda m: None)
        b2.start(lambda **r: None)
        b2.terminate(lambda **r: None)
        self.assertFalse(done("t"))
        # resume: killed / failed jobs restart, the finished one is skipped
        b3, started = Background({"background": p["background"] + q["background"]}, "bd", d, lambda m: None), []
        b3.start(lambda **r: started.append(r["name"]))
        self.assertEqual(sorted(started), ["f", "k", "t"])
        b3.terminate(lambda **r: None)

    def test_default_does_not_wait_and_the_job_is_terminated_and_reported(self):
        p = mk_plan(tmpdir(self), [arm("A", n=5)],
                    background=[{"name": "long", "argv": self.py("import time; time.sleep(60)")}])
        s = Sim(p)
        self.assertEqual(s.run(), DONE)
        rep = open(os.path.join(p["campaign_dir"], "REPORT.txt")).read()
        self.assertRegex(rep, r"long +terminated")
        self.assertFalse(os.path.exists(os.path.join(p["campaign_dir"], "background", "long.done")))  # not done: a resume starts it again
        self.assertEqual(kinds(s, "background_end")[0]["text"], "terminated")


class PlanValidation(unittest.TestCase):
    def test_executor_defaults_to_the_runner_executor(self):
        self.assertEqual(mk_plan("x", [arm("A")])["executor"], DEFAULT_EXECUTOR)
        self.assertEqual(DEFAULT_EXECUTOR, "sf43camp.runner_executor:RunnerExecutor")
        self.assertEqual(mk_plan("x", [arm("A")])["min_retained"], 5)

    def test_flags_pairs_and_background(self):
        mk_plan("x", [arm("A", lane="A", primary=True)])
        for bad in (dict(lane="C"), dict(primary="yes"), dict(no_reestimation=1)):
            with self.assertRaises(PlanError):
                mk_plan("x", [arm("A", **bad)])
        with self.assertRaises(PlanError):  # identical runner dicts and seeds are required
            mk_plan("x", [arm("A"), arm("B", consumer={"a": 1})], determinism_pairs=[["A", "B"]])
        with self.assertRaises(PlanError):
            mk_plan("x", [arm("A")], determinism_pairs=[["A", "Z"]])
        mk_plan("x", [arm("A"), arm("B")], determinism_pairs=[["A", "B"]])
        with self.assertRaises(PlanError):
            mk_plan("x", [arm("A")], background=[{"name": "bad name", "argv": ["x"]}])
        with self.assertRaises(PlanError):
            mk_plan("x", [arm("A")], background=[{"name": "n", "argv": []}])

    def test_reestimation_needs_targets_and_refuses_no_reestimation_arms(self):
        ok = {"fraction": 0.25, "max_factor": 2.0, "arms": ["A"], "hw_target": {"A": {"alpha_TP": 0.001}}}
        mk_plan("x", [arm("A")], blinded_reestimation=ok)
        for bad in (dict(ok, hw_target={}), dict(ok, hw_target={"A": {"alpha_TP": 0}}), dict(ok, hw_target={"A": {}})):
            with self.assertRaises(PlanError):
                mk_plan("x", [arm("A")], blinded_reestimation=bad)
        with self.assertRaises(PlanError):
            mk_plan("x", [arm("A", no_reestimation=True)], blinded_reestimation=ok)

    def test_hmax_div_first_needs_unit_size_dividing_n(self):
        def a(unit, n=30):
            x = arm("A", n=60, unit=unit, count=60)
            x["consumer"] = {"hmax_div": [8], "hmax_div_first": {"n": n, "divs": [8, 16, 32]}}
            return x
        mk_plan("x", [a(10)])
        mk_plan("x", [a(5)])
        with self.assertRaises(PlanError):
            mk_plan("x", [a(8)])
        with self.assertRaises(PlanError):
            mk_plan("x", [a(10, n=0)])

    def test_example_plan_has_the_new_fields(self):
        f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plan_example.json")
        p = load_plan(f)
        self.assertEqual(p["executor"], DEFAULT_EXECUTOR)
        self.assertTrue(p["arms"][0]["primary"])
        self.assertIn("hw_target", p["blinded_reestimation"])


if __name__ == "__main__":
    unittest.main()
