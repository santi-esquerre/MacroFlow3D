"""Plan validation, queue order, lane pick rules, pure guard functions."""
import os
import unittest

from sf43camp import guards
from sf43camp.lanes import Lane
from sf43camp.ledger import Ledger
from sf43camp.plan import PlanError, load_plan
from sf43camp.scheduler import State

from .fx import arm, mk_plan, tmpdir


def order(st):
    return [(u.arm, u.seeds[0]) for u in st.candidates()]


class Order(unittest.TestCase):
    def test_crn_blocks_adjacent_and_short_arms_have_no_later_units(self):
        p = mk_plan("x", [arm("A1", n=10), arm("A2", n=10), arm("S", n=5), arm("B1", group="g2", n=10, start=200)])
        self.assertEqual(order(State(p)), [("A1", 100), ("A2", 100), ("S", 100), ("B1", 200), ("A1", 105), ("A2", 105), ("B1", 205)])

    def test_groups_interleave_by_progress_fraction(self):
        p = mk_plan("x", [arm("A", n=20), arm("B", group="g2", n=10, start=200)])
        self.assertEqual(order(State(p)), [("A", 100), ("B", 200), ("A", 105), ("A", 110), ("B", 205), ("A", 115)])

    def test_priority_class_first(self):
        p = mk_plan("x", [arm("A", n=10, prio=1), arm("B", group="g2", n=10, start=200)])
        self.assertEqual(order(State(p)), [("B", 200), ("B", 205), ("A", 100), ("A", 105)])

    def test_validation_rejects(self):
        for bad in (dict(n=10, count=5), dict(unit=0)):
            with self.assertRaises(PlanError):
                mk_plan("x", [arm("A", **bad)])
        with self.assertRaises(PlanError):  # CRN group arms must share the seed start
            mk_plan("x", [arm("A"), arm("B", start=101)])
        with self.assertRaises(PlanError):  # duplicate id
            mk_plan("x", [arm("A"), arm("A")])
        with self.assertRaises(PlanError):  # reserves inside the seed range
            mk_plan("x", [arm("A")], groups={"g1": {"reserves": {"start": 105, "count": 3}}})


class Example(unittest.TestCase):
    def test_plan_example_and_its_smoke_section_validate(self):
        f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plan_example.json")
        p = load_plan(f)
        self.assertEqual(len(p["arms"]), 4)
        s = load_plan(f, smoke=True, now=1_800_000_000.0)
        self.assertEqual((len(s["arms"]), s["campaign_dir"], s["cutoff_utc"]), (2, "sf43_campaign_smoke", "2027-01-15T08:30:00Z"))
        self.assertNotEqual(p["plan_sha256"], s["plan_sha256"])


class Pick(unittest.TestCase):
    def setUp(self):
        self.p = mk_plan("x", [arm("L", heavy=False, n=10), arm("H17", n=10, mem=17000), arm("H30", n=10, mem=30000)])
        self.st = State(self.p)
        self.A, self.B = Lane("A", "1", False), Lane("B", "0", True)

    def test_lane_a_heavy_first_largest_memory(self):
        u, _ = self.st.pick(self.A, 0, None, None)
        self.assertEqual(u.arm, "H30")

    def test_lane_b_only_heavy_and_memory_gate(self):
        u, _ = self.st.pick(self.B, 0, lambda: 20000.0, None)
        self.assertEqual(u.arm, "H17")
        for a in ("H17", "H30"):
            self.st.arms[a].done |= set(range(100, 110))
        self.assertEqual(self.st.pick(self.B, 0, None, None), (None, None))  # heavy queue empty -> nothing
        u, _ = self.st.pick(self.A, 0, None, None)
        self.assertEqual(u.arm, "L")

    def test_memory_wait_and_cap(self):
        u, wake = self.st.pick(self.B, 0, lambda: 100.0, None)
        self.assertEqual((u, wake), (None, 120.0))
        # a unit that would end after the cap is never claimed by lane B: 5 seeds x 100 s
        self.assertEqual(self.st.pick(self.B, 0, None, 499.0), (None, None))
        self.assertIsNotNone(self.st.pick(self.B, 0, None, 501.0)[0])

    def test_oom_backoff_gate(self):
        for a in ("H17", "H30"):
            self.st.arms[a].not_before[100] = 600.0
            self.st.arms[a].not_before[105] = 600.0
        u, wake = self.st.pick(self.B, 0, None, None)
        self.assertEqual((u, wake), (None, 600.0))


class Guards(unittest.TestCase):
    def test_backoff(self):
        self.assertEqual([guards.oom_backoff_s(n) for n in (1, 2, 3, 4, 5, 9)], [600, 1200, 2400, 3600, 3600, 3600])

    def test_resize_up_only_clipped_and_capped(self):
        ub = {"e": 1.0}
        self.assertEqual(guards.resize({"e": 0.1}, 20, ub, 10, 2.0, 100), (1.0, 10))  # lower SD never lowers N
        f, n = guards.resize({"e": 5.0}, 20, ub, 10, 2.0, 100)
        self.assertEqual((f, n), (2.0, 20))  # clipped at max_factor
        self.assertEqual(guards.resize({"e": 5.0}, 20, ub, 10, 2.0, 15)[1], 15)  # capped by the seed range / n_max
        f, n = guards.resize({"e": 1.0, "other": 99.0}, 40, {"e": 1.0}, 10, 2.0, 100)  # only planned estimators count
        self.assertTrue(1.0 < f < 2.0 and 10 < n < 20)

    def test_budget_and_disk(self):
        self.assertTrue(guards.budget_ok(1, 1, 1, 3))
        self.assertFalse(guards.budget_ok(1, 1, 1.01, 3))
        self.assertTrue(guards.disk_ok(15, 10, 5))
        self.assertFalse(guards.disk_ok(14.9, 10, 5))


class LedgerTorn(unittest.TestCase):
    def test_torn_tail_ignored_and_not_swallowing_the_next_record(self):
        p = os.path.join(tmpdir(self), "l.jsonl")
        led = Ledger(p, lambda: 1.0)
        led.append({"kind": "a"})
        with open(p, "a") as f:
            f.write('{"kind": "tor')  # SIGKILL mid-write
        self.assertEqual([r["kind"] for r in led.records()], ["a"])
        led.append({"kind": "b"})
        self.assertEqual([r["kind"] for r in led.records()], ["a", "b"])
        led.append({"kind": "c"})
        with open(p) as f:
            self.assertEqual(sum(1 for ln in f if ln.count("tor") == 0 and not ln.strip()), 0)


if __name__ == "__main__":
    unittest.main()
