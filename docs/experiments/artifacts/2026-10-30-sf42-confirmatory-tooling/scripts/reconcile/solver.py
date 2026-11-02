"""Reconcile the medium (b) solver verdicts with the committed job logs (decision 2026-10-29 G8; the SF-39 completeness error).

Rule recomputed from each `B_SOLVER` line (the production rule, `DirichletLongBoxStage.cu` `make_dirichlet_solver_record`):
  non_finite                        any of rel, balance, max_div, qbar, Q_in, Q_out not finite
  converged                         converged=1 and balance <= 1e-8
  converged_balance_above_1e-8      converged=1 and balance > 1e-8
  fallback                          converged=0 and rel <= 1e-8 and balance <= 1e-8
  not_converged_beyond_fallback     otherwise
Job logs (`scripts/remote` logs of `run_sf39.sh`) have two defects handled here, both reported, never silently patched:
  * lines are cut at about 200 characters and indented, so `max_div=` and `fallback=` are often absent: finiteness is checked on the
    fields present and `max_div_missing` is counted; a field cut inside its number (not the full `%.6e` / `%.12e` format) is
    treated as absent, and a record without a complete `converged`, `rel` or `balance` is `unparsed_truncated`;
  * the arm and seed are not on the line: `R=<n>`/`r=<n>` is the index inside one runner invocation `<arm>:s<seed0>:n<nr>` and the
    seed is seed0 + r.  The invocation is taken from the preceding header `+ [<arm>:s<seed0>:n<nr>]` / `CASE_RESULT run=...`, or,
    for lines that precede no header (a log whose head was cut), from the following `RUN id=<arm>:s<seed0>:n<nr>` summary line.
    Runs without a seed in their id (the byte-check runs `byte_multi`, `byte_single`) have seed `?` and are keyed by r.
    A line mapped to no invocation is `unmapped` unless it is an exact repeat of a mapped solve (the logs re-print the first lines).
Identical records of the same (arm, seed) are counted once; differing records are reported as conflicts.

CLI:  python3 -m reconcile.solver --logs <A>/logs/job_logs [--check-sf39] [--report FILE]
                                  [--runs-root $HOME/sf39_campaign/runs]      (host: read-only cross-check against solver_b.json)
Exit: 0 consistent; 1 a mismatch (--check-sf39 expectation, solver_b.json disagreement, conflict, or unparsed/unmapped solve);
2 nothing parsed.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict

TOL = 1e-8
VERDICTS = ("converged", "converged_balance_above_1e-8", "fallback", "not_converged_beyond_fallback", "non_finite",
            "unparsed_truncated")

# the four non-converged medium (b) solves of SF-39 (note section "Post-review", acceptance H of SF-42): (arm, r, seed, verdict)
SF39_EXPECTED = (("s4_B", 2, 5023, "fallback"), ("s4_tw", 22, 5023, "fallback"), ("s4_w12", 4, 7205, "fallback"),
                 ("s4_w12", 13, 7214, "not_converged_beyond_fallback"))

_E6 = r"(?:[-+]?\d\.\d{6}e[+-]\d{2,3}|[-+]?nan|[-+]?inf)"
_E12 = r"(?:[-+]?\d\.\d{12}e[+-]\d{2,3}|[-+]?nan|[-+]?inf)"
_FIELDS = (("converged", r"[01]"), ("iterations", r"\d+"), ("initial", _E6), ("final", _E6), ("rel", _E6), ("qbar", _E12),
           ("Q_in", _E12), ("Q_out", _E12), ("balance", _E6), ("max_div", _E6), ("fallback", r"[01]"))
_SOLVER = re.compile(r"B_SOLVER r=(\d+)(?P<rest>.*)$")
_HEADER = re.compile(r"^\s*\+ \[([A-Za-z0-9_]+)(?::s(\d+):n(\d+))?\]")
_ANY_HEADER = re.compile(r"^\s*\+ \[")
_CASE = re.compile(r"CASE_RESULT run=([A-Za-z0-9_]+)(?::s(\d+):n(\d+))?(?=\s)")
_RUN = re.compile(r"^RUN id=([A-Za-z0-9_]+):s(\d+):n(\d+)\b")


def parse_solver_line(line):
    """(r, fields{name: str}) of a B_SOLVER line, or None.  A field is kept only if it has the full number format and is
    followed by a space or the end of the line, and only the leading run of consecutive complete fields is kept."""
    m = _SOLVER.search(line)
    if not m:
        return None
    rest, fields = m.group("rest"), {}
    for name, pat in _FIELDS:
        fm = re.match(r"\s+%s=(%s)(?=\s|$)" % (name, pat), rest)
        if not fm:
            break
        fields[name] = fm.group(1)
        rest = rest[fm.end():]
    return int(m.group(1)), fields


def verdict_of(fields):
    """(verdict, notes) from parsed fields (production rule); a record needing converged/rel/balance that are missing is
    `unparsed_truncated`.  notes: list of strings (e.g. max_div_missing)."""
    need = ("converged", "rel", "balance", "qbar", "Q_in", "Q_out")
    if any(k not in fields for k in need):
        return "unparsed_truncated", ["missing %s" % ",".join(k for k in need if k not in fields)]
    notes = []
    if "max_div" not in fields:
        notes.append("max_div_missing")
    fin = all(math.isfinite(float(fields[k])) for k in ("rel", "balance", "qbar", "Q_in", "Q_out") + (("max_div",) if "max_div" in fields else ()))
    conv, rel, bal = fields["converged"] == "1", float(fields["rel"]), float(fields["balance"])
    if not fin:
        return "non_finite", notes
    if conv:
        return ("converged" if bal <= TOL else "converged_balance_above_1e-8"), notes
    if rel <= TOL and bal <= TOL:
        return "fallback", notes
    return "not_converged_beyond_fallback", notes


class Solve:
    def __init__(self, arm, r, seed, fields, source):
        self.arm, self.r, self.seed, self.fields, self.source = arm, r, seed, fields, source
        self.key = (arm, seed if seed is not None else "r%d" % r)
        self.verdict, self.notes = verdict_of(fields)
        self.logged_fallback = fields.get("fallback")

    def payload(self):
        return tuple(sorted(self.fields.items()))


def parse_logs(paths):
    """(solves OrderedDict[(arm, seed)] -> Solve, conflicts list, unmapped list of (source, r, verdict))."""
    solves, conflicts, unmapped = OrderedDict(), [], []

    def add(arm, seed0, r, fields, src):
        s = Solve(arm, r, None if seed0 is None else seed0 + r, fields, src)
        old = solves.get(s.key)
        if old is None:
            solves[s.key] = s
        elif old.payload() != s.payload():
            # a re-print may be shorter (cut earlier): same prefix of fields is a repeat, anything else a conflict
            a, b = dict(old.fields), dict(s.fields)
            common = set(a) & set(b)
            if any(a[k] != b[k] for k in common):
                conflicts.append((arm, s.seed, old.source, src))
            elif len(b) > len(a):
                solves[s.key] = s
        return s

    for path in sorted(paths):
        ctx, pending = None, []
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
        base = os.path.basename(path)

        def flush_orphans():
            """A solve line with no invocation is accepted only as a repeat of mapped solves (same r, all common fields equal,
            at least 8 of them); if exactly one mapped solve matches and the repeat has more fields (the cut was shorter there),
            the longer record replaces it."""
            for lineno, r, fields in pending:
                hits = [t for t in solves.values() if t.r == r and len(set(t.fields) & set(fields)) >= 8
                        and all(t.fields[k] == fields[k] for k in set(t.fields) & set(fields))]
                if not hits:
                    unmapped.append(("%s:%d" % (base, lineno), r, verdict_of(fields)[0]))
                elif len(hits) == 1 and len(fields) > len(hits[0].fields):
                    solves[hits[0].key] = Solve(hits[0].arm, hits[0].r, hits[0].seed, fields, "%s:%d" % (base, lineno))
            pending.clear()

        for lineno, line in enumerate(lines, 1):
            m = _RUN.match(line)
            if m:
                arm, s0 = m.group(1), (int(m.group(2)) if m.group(2) else None)
                for ln, r, fields in pending:  # solve lines printed before their summary (cut log head)
                    add(arm, s0, r, fields, "%s:%d" % (base, ln))
                pending.clear()
                ctx = None
                continue
            m = _HEADER.match(line) or _CASE.search(line)
            if m:
                flush_orphans()
                ctx = (m.group(1), int(m.group(2)) if m.group(2) else None)
                continue
            if _ANY_HEADER.match(line):
                flush_orphans()
                ctx = None
                continue
            p = parse_solver_line(line)
            if p is None:
                continue
            r, fields = p
            if ctx is not None:
                add(ctx[0], ctx[1], r, fields, "%s:%d" % (base, lineno))
            else:
                pending.append((lineno, r, fields))
        flush_orphans()
    return solves, conflicts, unmapped


def non_converged(solves):
    return [s for s in solves.values() if s.verdict != "converged"]


def counts_per_arm(solves):
    out = OrderedDict()
    for (arm, _), s in sorted(solves.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        out.setdefault(arm, Counter())[s.verdict] += 1
    return out


def host_crosscheck(solves, runs_root):
    """Read-only comparison of the log-derived verdicts with every `<runs_root>/<arm>/realizations/r_*/solver_b.json` (matched by
    arm and `seed`; r in the json is the index inside the arm directory).  Also a census of all jsons: a json whose verdict is not
    `converged` and that the logs do not show as non-converged is a mismatch.
    Returns (n_checked, mismatches [(arm, seed, why)], absent [(arm, seed)], census Counter of json verdicts)."""
    index, census = {}, Counter()
    for p in glob.glob(os.path.join(runs_root, "*", "realizations", "r_*", "solver_b.json")):
        arm = p.split(os.sep)[-4]
        with open(p) as f:
            j = json.load(f)
        index[(arm, int(j["seed"]))] = j
        census[j.get("verdict")] += 1
    n, bad, absent = 0, [], []
    for (arm, seed), s in solves.items():
        if s.seed is None:  # unseeded byte-check runs live outside runs/
            continue
        hit = index.get((arm, seed))
        if hit is None:
            absent.append((arm, seed))
            continue
        n += 1
        why = []
        if hit.get("verdict") != s.verdict:
            why.append("verdict log=%s json=%s" % (s.verdict, hit.get("verdict")))
        if s.logged_fallback is not None and bool(int(s.logged_fallback)) != bool(hit.get("fallback")):
            why.append("fallback log=%s json=%s" % (s.logged_fallback, hit.get("fallback")))
        if why:
            bad.append((arm, seed, "; ".join(why)))
    for (arm, seed), j in sorted(index.items()):
        if j.get("verdict") != "converged" and ((arm, seed) not in solves or solves[(arm, seed)].verdict == "converged"):
            bad.append((arm, seed, "solver_b.json verdict %s not shown as non-converged by the logs" % j.get("verdict")))
    return n, bad, absent, census


def report(solves, conflicts, unmapped, out, expect=None, host=None):
    P = lambda *a: print(*a, file=out)
    ncs = non_converged(solves)
    P("RECONCILE solves=%d arms=%d non_converged=%d conflicts=%d unmapped=%d" % (
        len(solves), len({a for a, _ in solves}), len(ncs), len(conflicts), len(unmapped)))
    P("")
    P("NON_CONVERGED (every solve whose recomputed verdict is not `converged`)")
    P("%-8s %4s %6s  %-31s %9s %6s %12s %12s  %s" % ("arm", "r", "seed", "verdict", "converged", "iters", "rel", "balance", "notes"))
    for s in ncs:
        P("%-8s %4d %6s  %-31s %9s %6s %12s %12s  %s" % (s.arm, s.r, "?" if s.seed is None else s.seed, s.verdict, s.fields.get("converged", "?"),
                                                           s.fields.get("iterations", "?"), s.fields.get("rel", "?"),
                                                           s.fields.get("balance", "?"), ",".join(s.notes) or "-"))
    if not ncs:
        P("(none)")
    P("")
    P("COUNTS_PER_ARM")
    P("%-12s %6s %s" % ("arm", "solves", " ".join("%s" % v for v in VERDICTS)))
    tot = Counter()
    for arm, c in counts_per_arm(solves).items():
        tot.update(c)
        P("%-12s %6d %s" % (arm, sum(c.values()), " ".join(str(c.get(v, 0)) for v in VERDICTS)))
    P("%-12s %6d %s" % ("TOTAL", sum(tot.values()), " ".join(str(tot.get(v, 0)) for v in VERDICTS)))
    nmd = sum("max_div_missing" in s.notes for s in solves.values())
    P("max_div missing (line cut) in %d of %d solves; their finiteness is checked on rel, balance, qbar, Q_in, Q_out" % (nmd, len(solves)))
    for arm, seed, a, b in conflicts:
        P("CONFLICT %s seed %d: %s vs %s" % (arm, seed, a, b))
    for src, r, v in unmapped:
        P("UNMAPPED %s r=%d verdict=%s" % (src, r, v))
    ok = not conflicts and not unmapped and not tot.get("unparsed_truncated")
    if expect is not None:
        got = {(s.arm, s.r, s.seed, s.verdict) for s in ncs}
        want = set(expect)
        for e in sorted(want - got):
            P("EXPECTED_MISSING %s r=%d seed=%d %s" % e)
        for e in sorted(got - want):
            P("UNEXPECTED_NON_CONVERGED %s r=%d seed=%d %s" % e)
        match = got == want
        P("SF39_EXPECTATION %s (%d expected non-converged solves)" % ("MATCH" if match else "MISMATCH", len(want)))
        ok = ok and match
    if host is not None:
        n, bad, absent, census = host
        for arm, seed, why in bad:
            P("HOST_MISMATCH %s seed %d: %s" % (arm, seed, why))
        P("HOST_CENSUS solver_b.json verdicts over all runs: %s" % ", ".join("%s=%d" % kv for kv in sorted(census.items(), key=str)))
        P("HOST_CHECK checked=%d mismatches=%d logged_solves_without_solver_b_json=%d" % (n, len(bad), len(absent)))
        ok = ok and not bad
    P("RECONCILE %s" % ("OK" if ok else "DIFFERENCES"))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="reconcile.solver", description=__doc__.split("\n")[0])
    ap.add_argument("--logs", required=True, help="directory of job logs (*.log) or a single log file")
    ap.add_argument("--check-sf39", action="store_true", help="require exactly the four SF-39 non-converged solves")
    ap.add_argument("--runs-root", default=None, help="host mode: $HOME/sf39_campaign/runs (read only)")
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    paths = sorted(glob.glob(os.path.join(a.logs, "*.log"))) if os.path.isdir(a.logs) else [a.logs]
    solves, conflicts, unmapped = parse_logs(paths)
    if not solves:
        print("no B_SOLVER lines mapped in %s" % a.logs, file=sys.stderr)
        return 2
    host = host_crosscheck(solves, a.runs_root) if a.runs_root else None
    out = open(a.report, "w") if a.report else sys.stdout
    try:
        ok = report(solves, conflicts, unmapped, out, SF39_EXPECTED if a.check_sf39 else None, host)
    finally:
        if a.report:
            out.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
