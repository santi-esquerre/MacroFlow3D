"""reconcile43 on synthetic campaign trees with the real run-log grammar (runner `R=<r> FIELD/B_SOLVER/STAGE` lines, Par2 pipeline
`SF-18 ... (seed=S` / `Converged=YES|NO` lines), `solver_b.json`, `DONE` and `ledger.jsonl`."""
import contextlib
import io
import json
import os
import tempfile
import unittest

from reconcile43 import driver, rule
from reconcile43.__main__ import main


def b_solver(r, converged=1, rel="1.000000e-11", bal="1.000000e-11", fb=0):
    return ("B_SOLVER r=%d converged=%d iterations=50 initial=1.000000e+07 final=1.000000e-04 rel=%s qbar=1.000000000000e+00 "
            "Q_in=1.000000000000e+00 Q_out=1.000000000001e+00 balance=%s max_div=2.000000e-08 fallback=%d" % (r, converged, rel, bal, fb))


CONV = dict(converged=1)
FALLBACK = dict(converged=0, rel="9.000000e-10", bal="2.000000e-09", fb=1)
BAD = dict(converged=0, rel="3.390000e-07", bal="2.640000e-07")


class Camp:
    def __init__(self, root):
        self.C = root
        self.ledger = []

    def consumer(self, arm, seg, seed0, specs, group=None, with_json=True, omit_field=None, stage_lie=None):
        d = os.path.join(self.C, "arms", arm, seg)
        os.makedirs(d, exist_ok=True)
        lines = ["R=0 some banner"]
        for r, spec in enumerate(specs):
            seed = seed0 + r
            line = b_solver(r, **spec)
            v, _ = rule.verdict_of(rule.parse_solver_line(line)[1])
            usable = v in rule.USABLE
            if omit_field != r:
                lines.append("R=%d FIELD gaussian (SF-18) N=128 grid=64x16x16 sigma2=1 ell=1 seed=%d normalize_variance=1" % (r, seed))
            lines.append("R=%d %s" % (r, line))
            lines.append("R=%d STAGE medium=b status=%s grid=64x16x16 mg_levels=4 solver_converged=%d solver_usable=%d flow_ready=1 seconds=1.0" % (
                r, stage_lie or v, spec["converged"], int(usable)))
            self.ledger.append(dict(arm=arm, segment=seg, seed=seed, verdict=v, seed0=seed0, **{"class": "ok" if usable else "unusable_solve"},
                                    **({"group": group} if group else {})))
            if not usable:
                lines.append("R=%d STAGE_FAILED status=solver_failed verdict=%s" % (r, v))
                break  # the runner stops at an unusable solve and writes no solver_b.json
            rd = os.path.join(d, "out", "realizations", "r_%04d" % r)
            os.makedirs(rd, exist_ok=True)
            if with_json:
                with open(os.path.join(rd, "solver_b.json"), "w") as f:
                    json.dump({"r": r, "seed": seed, "verdict": v, "fallback": v == "fallback", "usable": True}, f)
            with open(os.path.join(rd, "DONE"), "w") as f:
                f.write("sig\n")
        self._write(d, lines)

    def _write(self, d, lines):
        with open(os.path.join(d, "run.log"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def par2(self, arm, seg, seeds_yes_no, group=None, drop_conv=None):
        d = os.path.join(self.C, "arms", arm, seg)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.yaml"), "w") as f:
            f.write("grid: {}\n")
        lines = ["[1] Loading config: config.yaml"]
        for k, (seed, yes) in enumerate(seeds_yes_no):
            lines += ["  Realization %d / %d" % (k + 1, len(seeds_yes_no)),
                      "  [4] SF-18 medium (b) field (seed=%d, sigma2=1.0000e+00, corr_length=4.1667e-02, eps=5.0000e-01); K=exp(eps*Y)" % seed,
                      "  [5] Solving head (mg_cg)"]
            if drop_conv != seed:
                lines.append("       Converged=%s  iters=40  res=1.66e+05 → 1.65e-05" % ("YES" if yes else "NO"))
            self.ledger.append(dict(arm=arm, segment=seg, seed=seed, verdict="converged" if yes else "not_converged",
                                    **{"class": "ok" if yes else "unusable_solve"}, **({"group": group} if group else {})))
        self._write(d, lines)

    def replacement(self, seed, by, group=None, arm=None):
        self.ledger.append({"kind": "replacement", "seed": seed, "replaced_by": by, **({"group": group} if group else {"arm": arm})})

    def finish(self):
        with open(os.path.join(self.C, "ledger.jsonl"), "w") as f:
            for rec in self.ledger:
                f.write(json.dumps(rec) + "\n")

    def run(self):
        self.finish()
        buf = io.StringIO()
        code = driver.run(self.C, buf)
        return code, buf.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.c = Camp(self._t.name)


class Passing(Base):
    def test_consistent_campaign(self):
        self.c.consumer("a1", "s0", 5001, [CONV, FALLBACK, BAD], group="g1")  # seed 5003 unusable, the segment ends there
        self.c.consumer("a2", "s0", 5001, [CONV, FALLBACK, CONV], group="g1")  # same seed converged in the paired arm
        self.c.par2("p1", "s0", [(5001, True), (5002, True)], group="g1")
        self.c.replacement(5003, 5101, group="g1")
        code, rep = self.c.run()
        self.assertEqual(code, 0, rep)
        self.assertIn("RECONCILE OK", rep)
        self.assertIn("segments consumer=2 par2=1 solves=8", rep)
        self.assertIn("TOTAL converged=5 fallback=2 not_converged_beyond_fallback=1", rep)
        self.assertIn("UNUSABLE 1 solves; REPLACEMENTS 1; reserve seeds not yet run 1", rep)
        self.assertIn("FINDINGS conflicts=0 unmapped=0 unparsed=0 json=0 ledger=0 replacement=0 rule=0", rep)

    def test_par2_not_converged_needs_class_and_replacement(self):
        self.c.par2("p1", "s0", [(7001, True), (7002, False)])
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("REPLACEMENT unusable solve p1 seed 7002", rep)
        self.c.replacement(7002, 7100, arm="p1")
        code, rep = self.c.run()
        self.assertEqual(code, 0, rep)

    def test_seed0_maps_a_line_without_field(self):
        self.c.consumer("a1", "s0", 5001, [CONV, CONV], omit_field=1)
        code, rep = self.c.run()  # the ledger records carry seed0
        self.assertEqual(code, 0, rep)


class Detects(Base):
    def test_json_verdict_conflict(self):
        self.c.consumer("a1", "s0", 5001, [CONV, CONV])
        p = os.path.join(self.c.C, "arms", "a1", "s0", "out", "realizations", "r_0001", "solver_b.json")
        with open(p, "w") as f:
            json.dump({"r": 1, "seed": 5002, "verdict": "fallback", "fallback": True}, f)
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("JSON a1/s0 r_0001 seed 5002: verdict log=converged json=fallback", rep)
        self.assertIn("RECONCILE MISMATCH", rep)

    def test_ledger_verdict_conflict_and_class(self):
        self.c.consumer("a1", "s0", 5001, [CONV, FALLBACK])
        self.c.ledger[1]["verdict"] = "converged"
        self.c.ledger[0]["class"] = "unusable_solve"
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("verdict log=fallback ledger=converged", rep)
        self.assertIn("class unusable_solve but the solve is usable", rep)

    def test_unmapped_line(self):
        self.c.consumer("a1", "s0", 5001, [CONV, CONV], omit_field=1)
        for rec in self.c.ledger:
            rec.pop("seed0")
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("UNMAPPED a1/s0/run.log:", rep)
        self.assertIn("ledger verdict converged without a log line", rep)

    def test_missing_replacement_and_replacement_without_unusable_solve(self):
        self.c.consumer("a1", "s0", 5001, [CONV, BAD], group="g1")
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("REPLACEMENT unusable solve a1 seed 5002 (group g1) has no replacement", rep)
        self.c.replacement(5002, 5101, group="g1")
        self.c.replacement(5001, 5102, group="g1")  # seed 5001 converged: nothing to replace
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("replacement of seed 5001 (group g1) with no unusable solve of that seed", rep)
        self.assertNotIn("has no replacement", rep)

    def test_reserve_seed_used_twice_is_a_conflict(self):
        self.c.consumer("a1", "s0", 5001, [BAD], group="g1")
        self.c.consumer("a1", "s1", 5010, [BAD], group="g1")
        self.c.replacement(5001, 5101, group="g1")
        self.c.replacement(5010, 5101, group="g1")
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("CONFLICT reserve seed 5101 used by 2 replacements in group g1", rep)

    def test_repeat_identical_ok_different_conflict(self):
        self.c.consumer("a1", "s0", 5001, [CONV])
        self.c.consumer("a1", "s0b", 5001, [CONV])
        for rec in self.c.ledger:
            rec["segment"] = "s0"
        code, rep = self.c.run()
        self.assertEqual(code, 0, rep)
        self.c.consumer("a1", "s0c", 5001, [dict(converged=1, bal="1.000000e-12")])
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("CONFLICT a1 seed 5001", rep)

    def test_stage_line_disagrees_with_recomputed_rule(self):
        self.c.consumer("a1", "s0", 5001, [CONV], stage_lie="fallback")
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("RULE ", rep)

    def test_json_missing_for_done_realization(self):
        self.c.consumer("a1", "s0", 5001, [CONV], with_json=False)
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("DONE realization without solver_b.json", rep)

    def test_par2_seed_without_converged_line(self):
        self.c.par2("p1", "s0", [(7001, True), (7002, True)], drop_conv=7002)
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("UNPARSED p1/s0/run.log:", rep)

    def test_logged_solve_without_ledger_record(self):
        self.c.consumer("a1", "s0", 5001, [CONV, CONV])
        self.c.ledger.pop()
        code, rep = self.c.run()
        self.assertEqual(code, 1)
        self.assertIn("logged solve (converged) without ledger record", rep)


class Cli(Base):
    def test_empty_campaign_is_code_2_and_out_inside_campaign_refused(self):
        self.assertEqual(self.c.run()[0], 2)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            main(["run", "--campaign", self.c.C, "--out", os.path.join(self.c.C, "report.txt")])

    def test_cli_writes_report(self):
        self.c.consumer("a1", "s0", 5001, [CONV])
        self.c.finish()
        out = os.path.join(self._t.name + "_report.txt")
        self.addCleanup(lambda: os.path.exists(out) and os.remove(out))
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = main(["run", "--campaign", self.c.C, "--out", out])
        self.assertEqual(code, 0)
        self.assertEqual(buf.getvalue().strip(), "RECONCILE OK")


class Rule(unittest.TestCase):
    def test_production_rule(self):
        v = lambda **kw: rule.verdict_of(rule.parse_solver_line(b_solver(0, **kw))[1])[0]  # noqa: E731
        self.assertEqual(v(), "converged")
        self.assertEqual(v(bal="1.100000e-08"), "converged_balance_above_1e-8")
        self.assertEqual(v(converged=0, rel="1.000000e-08", bal="1.000000e-08"), "fallback")
        self.assertEqual(v(converged=0, rel="1.100000e-08", bal="1.000000e-12"), "not_converged_beyond_fallback")
        self.assertEqual(v(rel="nan"), "non_finite")
        cut = b_solver(0)[:b_solver(0).index("balance=") + 15]
        self.assertEqual(rule.verdict_of(rule.parse_solver_line(cut)[1])[0], "unparsed_truncated")


if __name__ == "__main__":
    unittest.main()
