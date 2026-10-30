"""Tests of the solver-verdict reconciliation: synthetic logs for the parser and rule, and the committed SF-39 job logs (acceptance H)."""
import contextlib
import io
import json
import os
import tempfile
import unittest

from reconcile import solver as rs

HERE = os.path.dirname(os.path.abspath(__file__))
JOB_LOGS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "2026-10-27-sf39-macrodispersion-study", "logs", "job_logs"))


def line(r, converged=1, iters=50, rel="1.000000e-11", bal="1.000000e-11", cut=None, max_div="2.000000e-08", fb=0, indent="    "):
    s = ("%sR=%d B_SOLVER r=%d converged=%d iterations=%d initial=1.000000e+07 final=1.000000e-04 rel=%s qbar=1.000000000000e+00 "
         "Q_in=1.000000000000e+00 Q_out=1.000000000001e+00 balance=%s" % (indent, r, r, converged, iters, rel, bal))
    if max_div is not None:
        s += " max_div=%s fallback=%d" % (max_div, fb)
    return s[:cut] if cut else s


def write(d, name, lines):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    return p


class Rule(unittest.TestCase):
    def v(self, **kw):
        _, fields = rs.parse_solver_line(line(0, **kw))
        return rs.verdict_of(fields)[0]

    def test_production_rule(self):
        self.assertEqual(self.v(), "converged")
        self.assertEqual(self.v(bal="1.000000e-08"), "converged")  # <= 1e-8
        self.assertEqual(self.v(bal="1.100000e-08"), "converged_balance_above_1e-8")
        self.assertEqual(self.v(converged=0, rel="9.000000e-09", bal="9.000000e-09"), "fallback")
        self.assertEqual(self.v(converged=0, rel="1.000000e-08", bal="1.000000e-08"), "fallback")
        self.assertEqual(self.v(converged=0, rel="1.100000e-08", bal="1.000000e-12"), "not_converged_beyond_fallback")
        self.assertEqual(self.v(converged=0, rel="1.000000e-12", bal="1.100000e-08"), "not_converged_beyond_fallback")
        self.assertEqual(self.v(rel="nan"), "non_finite")
        self.assertEqual(self.v(bal="inf"), "non_finite")
        self.assertEqual(self.v(max_div="nan"), "non_finite")

    def test_truncated_lines(self):
        _, f = rs.parse_solver_line(line(3, max_div=None))
        self.assertNotIn("max_div", f)
        self.assertEqual(rs.verdict_of(f), ("converged", ["max_div_missing"]))
        # cut inside the balance number: that field is NOT trusted, the record is unparsed
        full = line(3, max_div=None)
        cut = full[:full.index("balance=") + len("balance=1.0000")]
        _, f = rs.parse_solver_line(cut)
        self.assertNotIn("balance", f)
        self.assertEqual(rs.verdict_of(f)[0], "unparsed_truncated")
        # cut before Q_out
        _, f = rs.parse_solver_line(full[:full.index(" Q_out")])
        self.assertEqual(rs.verdict_of(f)[0], "unparsed_truncated")
        self.assertIsNone(rs.parse_solver_line("    R=0 STATUS ok"))

    def test_exponent_cut_is_not_a_complete_number(self):
        full = line(3, max_div=None)
        _, f = rs.parse_solver_line(full[:full.index("balance=") + len("balance=1.000000e-1")] )
        self.assertNotIn("balance", f)  # '1.000000e-1' could have been 'e-12'
        _, f = rs.parse_solver_line(full)
        self.assertEqual(f["balance"], "1.000000e-11")


class Mapping(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.d = self._t.name

    def tearDown(self):
        self._t.cleanup()

    def parse(self, *names):
        return rs.parse_logs([os.path.join(self.d, n) for n in names])

    def test_header_gives_arm_and_seed(self):
        write(self.d, "a.log", ["+ [x_B:s5021:n3] .: cmd", "  CASE_RESULT run=x_B:s5021:n3 rc=0 wall=1s", line(0), line(1, converged=0, rel="5.000000e-10"),
                                line(2), "RUN id=x_B:s5021:n3 rc=0 wall=1s realizations_done=3/3"])
        solves, conflicts, unmapped = self.parse("a.log")
        self.assertEqual(sorted(solves), [("x_B", 5021), ("x_B", 5022), ("x_B", 5023)])
        self.assertEqual([s.verdict for s in solves.values()], ["converged", "fallback", "converged"])
        self.assertEqual((conflicts, unmapped), ([], []))

    def test_lines_before_a_run_summary_take_its_arm(self):
        write(self.d, "w.log", [line(0), line(1), "RUN id=y_w:s7101:n30 rc=0 wall=1s", "+ [other] .: x"])
        solves, _, unmapped = self.parse("w.log")
        self.assertEqual(sorted(solves), [("y_w", 7101), ("y_w", 7102)])
        self.assertEqual(unmapped, [])

    def test_repeat_with_more_fields_replaces_and_orphan_is_flagged(self):
        write(self.d, "e.log", ["+ [x_A:s5001:n2] .", line(0, max_div=None), line(1, max_div=None), "RUN id=x_A:s5001:n2 rc=0 wall=1s",
                                line(0, indent=""), "RECORDED part_seconds=1"])  # unindented full re-print of r=0
        solves, conflicts, unmapped = self.parse("e.log")
        self.assertEqual(unmapped, [])
        self.assertIn("max_div", solves[("x_A", 5001)].fields)  # the longer record was adopted
        self.assertNotIn("max_div", solves[("x_A", 5002)].fields)
        write(self.d, "o.log", ["+ [x_A:s5001:n1] .", line(0), "RUN id=x_A:s5001:n1 rc=0", line(7, converged=0, rel="1.000000e-03")])
        solves, _, unmapped = self.parse("o.log")
        self.assertEqual([(u[1], u[2]) for u in unmapped], [(7, "not_converged_beyond_fallback")])

    def test_conflicting_duplicates_are_reported(self):
        write(self.d, "a.log", ["+ [x_A:s5001:n1] .", line(0)])
        write(self.d, "b.log", ["+ [x_A:s5001:n1] .", line(0, iters=51)])
        solves, conflicts, _ = self.parse("a.log", "b.log")
        self.assertEqual(len(solves), 1)
        self.assertEqual([(c[0], c[1]) for c in conflicts], [("x_A", 5001)])

    def test_identical_duplicates_across_logs_count_once(self):
        write(self.d, "a.log", ["+ [x_A:s5001:n1] .", line(0)])
        write(self.d, "b.log", ["+ [x_A:s5001:n1] .", line(0)])
        solves, conflicts, _ = self.parse("a.log", "b.log")
        self.assertEqual((len(solves), conflicts), (1, []))

    def test_unseeded_runs_and_non_run_headers(self):
        write(self.d, "c.log", ["+ [byte_multi] .: cmd", "  CASE_RESULT run=byte_multi rc=0", line(0), "RUN id=byte_multi rc=0",
                                "+ [unittest:61a019e0bb82] .: python3", line(0, converged=0, rel="1.000000e-03")])
        solves, _, unmapped = self.parse("c.log")
        self.assertEqual(list(solves), [("byte_multi", "r0")])
        self.assertEqual(len(unmapped), 1)  # after a non-run header the line has no invocation

    def test_cli_exit_codes(self):
        write(self.d, "a.log", ["+ [s4_B:s5021:n3] .", line(0), line(1), line(2)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(rs.main(["--logs", self.d]), 0)
            self.assertEqual(rs.main(["--logs", self.d, "--check-sf39"]), 1)  # the four expected solves are missing
        self.assertIn("EXPECTED_MISSING s4_B r=2 seed=5023 fallback", buf.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            os.remove(os.path.join(self.d, "a.log"))
            self.assertEqual(rs.main(["--logs", self.d]), 2)

    def test_host_crosscheck_reads_solver_b_json(self):
        write(self.d, "a.log", ["+ [x_A:s5001:n3] .", line(0), line(1, converged=0, rel="5.000000e-10", fb=1), line(2)])
        solves, _, _ = self.parse("a.log")
        root = os.path.join(self.d, "runs")
        for r, (verdict, fb) in enumerate((("converged", False), ("fallback", True), ("converged", False))):
            p = os.path.join(root, "x_A", "realizations", "r_%04d" % r)
            os.makedirs(p)
            with open(os.path.join(p, "solver_b.json"), "w") as f:
                json.dump({"r": r, "seed": 5001 + r, "verdict": verdict, "fallback": fb}, f)
        n, bad, absent, census = rs.host_crosscheck(solves, root)
        self.assertEqual((n, bad, absent, dict(census)), (3, [], [], {"converged": 2, "fallback": 1}))
        with open(os.path.join(root, "x_A", "realizations", "r_0002", "solver_b.json"), "w") as f:
            json.dump({"r": 2, "seed": 5003, "verdict": "not_converged_beyond_fallback", "fallback": False}, f)
        n, bad, absent, census = rs.host_crosscheck(solves, root)
        self.assertEqual(len(bad), 2)  # verdict differs, and a non-converged json the logs call converged


@unittest.skipUnless(os.path.isdir(JOB_LOGS), "committed SF-39 job logs not found")
class CommittedSF39Logs(unittest.TestCase):
    """Acceptance H: the four fallback / non-converged solves of SF-39 are reproduced from the committed job logs."""

    @classmethod
    def setUpClass(cls):
        paths = [os.path.join(JOB_LOGS, n) for n in os.listdir(JOB_LOGS) if n.endswith(".log")]
        cls.solves, cls.conflicts, cls.unmapped = rs.parse_logs(paths)

    def test_exactly_the_four_non_converged_solves(self):
        got = sorted((s.arm, s.r, s.seed, s.verdict) for s in rs.non_converged(self.solves))
        self.assertEqual(got, sorted(rs.SF39_EXPECTED))
        self.assertEqual(sorted(rs.SF39_EXPECTED), [("s4_B", 2, 5023, "fallback"), ("s4_tw", 22, 5023, "fallback"),
                                                    ("s4_w12", 4, 7205, "fallback"), ("s4_w12", 13, 7214, "not_converged_beyond_fallback")])

    def test_everything_else_is_converged_and_nothing_is_ambiguous(self):
        self.assertEqual(self.conflicts, [])
        self.assertEqual(self.unmapped, [])
        c = {}
        for s in self.solves.values():
            c[s.verdict] = c.get(s.verdict, 0) + 1
        self.assertEqual(c, {"converged": 631, "fallback": 3, "not_converged_beyond_fallback": 1})

    def test_counts_per_arm(self):
        per = {a: sum(c.values()) for a, c in rs.counts_per_arm(self.solves).items()}
        for arm, n in (("s4_B", 51), ("s225_B", 42), ("s1_B", 30), ("s025_B", 5), ("s4_w12", 14), ("s4_w12c", 17), ("s4_w24", 30), ("s4_tw", 30)):
            self.assertEqual(per[arm], n, arm)
        c = rs.counts_per_arm(self.solves)
        self.assertEqual(c["s4_B"]["fallback"], 1)
        self.assertEqual(c["s4_w12"]["not_converged_beyond_fallback"], 1)

    def test_seed_is_seed0_plus_r(self):
        s = self.solves[("s4_B", 5023)]
        self.assertEqual((s.r, s.seed, s.fields["converged"], s.fields["iterations"]), (2, 5023, "0", "2000"))

    def test_cli_check_sf39(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = rs.main(["--logs", JOB_LOGS, "--check-sf39"])
        self.assertEqual(code, 0, buf.getvalue())
        self.assertIn("SF39_EXPECTATION MATCH", buf.getvalue())
        self.assertIn("RECONCILE OK", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
