"""Solver-verdict reconciliation of an SF-43 campaign (decision 2026-10-29 G8; acceptance F).

CAMPAIGN LAYOUT (the engine's: sf43camp/README.md; read only)
  <C>/segments/<arm_id>/<segment>/run.log      stdout+stderr of ONE runner invocation (a segment, `u<unit>_s<seed0>`)
  <C>/segments/<arm_id>/<segment>/config.yaml  present iff the segment is a Par2 run (otherwise a `lagrangian_ensemble` run)
  <C>/segments/<arm_id>/<segment>/seedmap.json {seed0, nr, seeds: {r: seed0 + r}} written by the executor (both codes)
  <C>/segments/<arm_id>/<segment>/out/**/realizations/r_XXXX/{solver_b.json,DONE}    consumer realizations (r = index in the invocation)
  <C>/ledger.jsonl                         the engine's ledger, one JSON object per line; only these kinds are read (every other kind of
                                           the engine - claim, unit_end, hb, reduced, ... - is ignored):
      seed record     {"kind": "seed", "arm", "segment" (basename), "seed", "verdict", "class", "group", "seed0", ["iterations",
                      "max_div", "rel", "balance", "field_seed"]}.  verdict: the consumer's solver verdict, or for Par2 `converged`
                      (Converged=YES), `fallback` (NO with final/initial residual <= 1e-8, usable) or `not_converged_beyond_fallback`;
                      class: `ok` or `unusable_solve` (only the LAST record of an (arm, seed) is compared); `group`: the paired group a
                      replacement is applied to group-wide; `seed0`: the segment's --seed0 (cross-check and fallback of the seed mapping)
      replacement     {"kind": "replacement", "arm", "group", "seed": <replaced seed>, "replaced_by": <reserve seed>}

WHAT IS CHECKED
  1. every consumer `B_SOLVER` line is mapped to (arm, seed): seed = the seed printed by the same realization's `R=<r> FIELD ... seed=`
     line (else seed0 + r when the ledger gives seed0); a line mapped to no seed is UNMAPPED.  The verdict is recomputed from the
     fields (rule.py) and compared with the runner's own `R=<r> STAGE medium=b status=<verdict> ... solver_usable=<0|1>` line.
  2. every Par2 `Converged=YES|NO  iters=N  res=<initial> -> <final>` line is mapped to the seed of the preceding field line: the
     SF-18 medium (b) line `[4] SF-18 ... (seed=S` or Par2's own generator `[4] K generation (seed=S)` (G7 factor G), and the seed
     is cross-checked against the segment's seed map (the r-th Converged line is seed0 + r); a seed line with no Converged line (and
     a Converged line with no seed line and no seed map) is UNPARSED / UNMAPPED.  The verdict is `converged` (YES), `fallback`
     (NO, final/initial <= 1e-8) or `not_converged_beyond_fallback` (the pipeline prints no balance).
  3. repeated records of one (arm, seed) must be identical (an OOM retry re-runs a deterministic solve); a difference is a CONFLICT.
  4. `solver_b.json` of every consumer realization: seed, r, verdict and fallback equal the log's; a json with no log line, and a DONE
     realization without json, are mismatches.  (An unusable solve stops the runner before it writes a json, so none is expected.)
  5. the ledger: its last record of every (arm, seed) has the log's verdict and segment, `class == unusable_solve` iff the solve is
     not usable (consumer verdict not `converged`/`fallback`; Par2 `Converged=NO`); logged solves without ledger record, and ledger
     records (with a verdict) without log line, are mismatches.
  6. replacements, group-wide: every unusable solve of any arm of a group has a replacement of its seed in that group, and every
     replacement of a seed has at least one unusable solve of that seed in some arm of its group; a reserve seed used twice in a
     group is a conflict.  (A reserve seed not yet run is counted, not a mismatch.)
Exit: 0 `RECONCILE OK`, 1 `RECONCILE MISMATCH`, 2 nothing parsed.
"""
import glob
import json
import math
import os
import re
from collections import Counter, OrderedDict

from . import rule

FIELD_RE = re.compile(r"^R=(\d+) FIELD .*?\bseed=(\d+)")
STAGE_RE = re.compile(r"^R=(\d+) STAGE medium=b status=(\S+) .*?\bsolver_usable=([01])")
PFX_RE = re.compile(r"^R=(\d+) ")
P2_SEED_RE = re.compile(r"\[4\] (?:SF-18 .*?|K generation )\(seed=(\d+)")
P2_CONV_RE = re.compile(r"Converged=(YES|NO)\b(?:\s+iters=(\d+)\s+res=(\S+)\s+\S+\s+(\S+))?")


def par2_verdict(m):
    """Par2 solve verdict of a `Converged=` match (the rule of sf43camp.runner_executor.par2_solves, written again here)."""
    if m.group(1) == "YES":
        return "converged"
    try:
        r0, r1 = float(m.group(3)), float(m.group(4))
    except (TypeError, ValueError):
        return "not_converged_beyond_fallback"
    if not (math.isfinite(r0) and math.isfinite(r1)):
        return "non_finite"
    return "fallback" if (r1 / r0 if r0 > 0.0 else 0.0) <= rule.TOL else "not_converged_beyond_fallback"


class Solve:
    def __init__(self, code, arm, segment, r, seed, verdict, usable, payload, source, fallback=None, stage=None):
        self.code, self.arm, self.segment, self.r, self.seed = code, arm, segment, r, seed
        self.verdict, self.usable, self.payload, self.source = verdict, usable, payload, source
        self.fallback, self.stage = fallback, stage


class Findings:
    """Anomalies by kind (each a one-line string); `counts` is what the report summarises."""

    def __init__(self):
        self.items = OrderedDict()

    def add(self, kind, text):
        self.items.setdefault(kind, []).append(text)

    def n(self, *kinds):
        return sum(len(self.items.get(k, [])) for k in kinds) if kinds else sum(len(v) for v in self.items.values())


def read_lines(path):
    with open(path, errors="replace") as f:
        return f.read().splitlines()


def parse_consumer_log(arm, seg, path, seed0, fnd):
    out, cur_seed, last = [], {}, {}
    for no, line in enumerate(read_lines(path), 1):
        src = "%s/%s/run.log:%d" % (arm, seg, no)
        m = FIELD_RE.match(line)
        if m:
            cur_seed[int(m.group(1))] = int(m.group(2))
            continue
        m = STAGE_RE.match(line)
        if m:
            r = int(m.group(1))
            if r in last:
                last[r].stage = (m.group(2), m.group(3) == "1")
            continue
        p = rule.parse_solver_line(line)
        if p is None:
            continue
        r, fields = p
        pm = PFX_RE.match(line)
        if pm and int(pm.group(1)) != r:
            fnd.add("conflict", "%s: line prefix R=%s but B_SOLVER r=%d" % (src, pm.group(1), r))
        seed = cur_seed.get(r, None if seed0 is None else seed0 + r)
        verdict, _ = rule.verdict_of(fields)
        if seed is None:
            fnd.add("unmapped", "%s: B_SOLVER r=%d verdict=%s has no FIELD seed line and no seed0" % (src, r, verdict))
            continue
        if seed0 is not None and seed != seed0 + r:
            fnd.add("conflict", "%s: FIELD seed %d != seed0 + r = %d" % (src, seed, seed0 + r))
        s = Solve("consumer", arm, seg, r, seed, verdict, verdict in rule.USABLE, tuple(sorted(fields.items())), src,
                  fallback=fields.get("fallback"))
        last[r] = s
        out.append(s)
    return out


def parse_par2_log(arm, seg, path, seed0, fnd):
    out, pending, k = [], None, 0
    for no, line in enumerate(read_lines(path), 1):
        src = "%s/%s/run.log:%d" % (arm, seg, no)
        m = P2_SEED_RE.search(line)
        if m:
            if pending is not None:
                fnd.add("unparsed", "%s: seed %d has no Converged line before the next seed" % (pending[1], pending[0]))
            pending = (int(m.group(1)), src)
            continue
        m = P2_CONV_RE.search(line)
        if m:
            mapped = None if seed0 is None else seed0 + k  # the r-th Converged line of the invocation is seed0 + r
            k += 1
            if pending is None and mapped is None:
                fnd.add("unmapped", "%s: Converged=%s without a preceding field seed line and without a seed map" % (src, m.group(1)))
                continue
            seed = pending[0] if pending is not None else mapped
            if pending is not None and mapped is not None and pending[0] != mapped:
                fnd.add("conflict", "%s: field seed %d != seed map seed0 + r = %d" % (src, pending[0], mapped))
            v = par2_verdict(m)
            out.append(Solve("par2", arm, seg, k - 1, seed, v, v in rule.USABLE, (("converged", m.group(1)), ("iterations", m.group(2))), src))
            pending = None
    if pending is not None:
        fnd.add("unparsed", "%s: seed %d has no Converged line" % (pending[1], pending[0]))
    return out


def read_ledger(path, fnd):
    seeds, repl = OrderedDict(), []
    if not os.path.isfile(path):
        fnd.add("ledger", "ledger %s is missing" % path)
        return seeds, repl, {}
    seed0 = {}
    for no, line in enumerate(read_lines(path), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec.get("kind") == "replacement":
                repl.append((rec.get("group") or rec.get("arm"), int(rec["seed"]), int(rec["replaced_by"]), no))
            elif rec.get("kind", "seed") == "seed":
                key = (rec["arm"], int(rec["seed"]))
                seeds.setdefault(key, []).append(rec)
                if rec.get("seed0") is not None:
                    seed0[(rec["arm"], os.path.basename(rec["segment"]))] = int(rec["seed0"])
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            fnd.add("ledger", "ledger line %d unreadable (%s: %s)" % (no, type(e).__name__, e))
    return seeds, repl, seed0


def check_jsons(seg_dir, arm, seg, solves, fnd):
    """Cross-check solver_b.json / DONE of a consumer segment against its log solves."""
    by_key = {s.seed: s for s in solves}
    n = 0
    for rd in sorted(glob.glob(os.path.join(seg_dir, "out", "**", "realizations", "r_*"), recursive=True)):
        if not re.fullmatch(r"r_\d{4}", os.path.basename(rd)):
            continue
        jp, rn = os.path.join(rd, "solver_b.json"), int(os.path.basename(rd)[2:])
        who = "%s/%s r_%04d" % (arm, seg, rn)
        if not os.path.isfile(jp):
            if os.path.isfile(os.path.join(rd, "DONE")):
                fnd.add("json", "%s: DONE realization without solver_b.json" % who)
            continue
        with open(jp) as f:
            j = json.load(f)
        n += 1
        s = by_key.get(int(j["seed"]))
        if s is None:
            fnd.add("json", "%s: solver_b.json seed %s has no B_SOLVER line in run.log" % (who, j["seed"]))
            continue
        why = []
        if int(j["r"]) != s.r or rn != s.r:
            why.append("r json=%s dir=%d log=%d" % (j["r"], rn, s.r))
        if j.get("verdict") != s.verdict:
            why.append("verdict log=%s json=%s" % (s.verdict, j.get("verdict")))
        if s.fallback is not None and bool(int(s.fallback)) != bool(j.get("fallback")):
            why.append("fallback log=%s json=%s" % (s.fallback, j.get("fallback")))
        if why:
            fnd.add("json", "%s seed %s: %s" % (who, j["seed"], "; ".join(why)))
    return n


def run(campaign, out):
    P = lambda *a: print(*a, file=out)  # noqa: E731
    fnd = Findings()
    seeds_l, repl, seed0 = read_ledger(os.path.join(campaign, "ledger.jsonl"), fnd)
    solves, n_json, n_seg = OrderedDict(), 0, Counter()
    for lp in sorted(glob.glob(os.path.join(campaign, "segments", "*", "*", "run.log"))):
        sd = os.path.dirname(lp)
        arm, seg = os.path.basename(os.path.dirname(sd)), os.path.basename(sd)
        par2 = os.path.isfile(os.path.join(sd, "config.yaml"))
        n_seg["par2" if par2 else "consumer"] += 1
        s0 = seed0.get((arm, seg))
        if os.path.isfile(os.path.join(sd, "seedmap.json")):  # the executor's seed map is the primary source of seed0
            with open(os.path.join(sd, "seedmap.json")) as f:
                m0 = int(json.load(f)["seed0"])
            if s0 is not None and s0 != m0:
                fnd.add("conflict", "%s/%s: ledger seed0 %d != seedmap.json seed0 %d" % (arm, seg, s0, m0))
            s0 = m0
        seg_solves = parse_par2_log(arm, seg, lp, s0, fnd) if par2 else parse_consumer_log(arm, seg, lp, s0, fnd)
        if not par2:
            n_json += check_jsons(sd, arm, seg, seg_solves, fnd)
            for s in seg_solves:
                if s.stage is not None and (s.stage[0] != s.verdict or s.stage[1] != s.usable):
                    fnd.add("rule", "%s: recomputed verdict %s (usable=%d) but STAGE line status=%s solver_usable=%d" % (
                        s.source, s.verdict, s.usable, s.stage[0], s.stage[1]))
        for s in seg_solves:
            key = (s.arm, s.seed)
            old = solves.get(key)
            if old is None:
                solves[key] = s
            elif old.payload != s.payload or old.verdict != s.verdict:
                fnd.add("conflict", "%s seed %d: %s (%s) vs %s (%s)" % (s.arm, s.seed, old.source, old.verdict, s.source, s.verdict))
    for s in solves.values():
        if s.verdict == "unparsed_truncated":
            fnd.add("unparsed", "%s: B_SOLVER record without complete converged/rel/balance/qbar/Q_in/Q_out" % s.source)
    # ledger vs logs
    group_of = {k: (v[-1].get("group") or k[0]) for k, v in seeds_l.items()}
    for key, s in solves.items():
        recs = seeds_l.get(key)
        if not recs:
            fnd.add("ledger", "%s seed %d: logged solve (%s) without ledger record" % (key[0], key[1], s.verdict))
            continue
        rec = recs[-1]
        lv = rec.get("verdict")
        if lv != s.verdict:
            fnd.add("ledger", "%s seed %d: verdict log=%s ledger=%s" % (key[0], key[1], s.verdict, rec.get("verdict")))
        if os.path.basename(str(rec.get("segment"))) != s.segment:
            fnd.add("ledger", "%s seed %d: segment log=%s ledger=%s" % (key[0], key[1], s.segment, rec.get("segment")))
        if (rec.get("class") == "unusable_solve") == s.usable:
            fnd.add("ledger", "%s seed %d: class %s but the solve is %s" % (key[0], key[1], rec.get("class"), "usable" if s.usable else "unusable"))
        pay = dict(s.payload)  # solver numbers the ledger carries (SF-43 item 10) against the log's (the log prints %.6e)
        if rec.get("iterations") is not None and pay.get("iterations") not in (None, "") and int(rec["iterations"]) != int(pay["iterations"]):
            fnd.add("ledger", "%s seed %d: iterations log=%s ledger=%s" % (key[0], key[1], pay["iterations"], rec["iterations"]))
        if rec.get("max_div") is not None and pay.get("max_div") is not None:
            lg, lv = float(pay["max_div"]), float(rec["max_div"])
            if abs(lg - lv) > 1e-5 * max(abs(lg), abs(lv)):
                fnd.add("ledger", "%s seed %d: max_div log=%s ledger=%s" % (key[0], key[1], pay["max_div"], rec["max_div"]))
    for key, recs in seeds_l.items():
        if key not in solves and recs[-1].get("verdict"):
            fnd.add("ledger", "%s seed %d: ledger verdict %s without a log line" % (key[0], key[1], recs[-1]["verdict"]))
    # replacements, group-wide
    unusable = [(group_of.get(k, k[0]), k[1], k[0]) for k, s in solves.items() if not s.usable]
    repl_keys = {(g, s) for g, s, _, _ in repl}
    for g, seed, arm in unusable:
        if (g, seed) not in repl_keys:
            fnd.add("replacement", "unusable solve %s seed %d (group %s) has no replacement" % (arm, seed, g))
    unus_keys = {(g, s) for g, s, _ in unusable}
    used = Counter()
    for g, seed, new, no in repl:
        if (g, seed) not in unus_keys:
            fnd.add("replacement", "ledger line %d: replacement of seed %d (group %s) with no unusable solve of that seed" % (no, seed, g))
        if new == seed:
            fnd.add("replacement", "ledger line %d: seed %d replaced by itself" % (no, seed))
        used[(g, new)] += 1
    for (g, new), c in used.items():
        if c > 1:
            fnd.add("conflict", "reserve seed %d used by %d replacements in group %s" % (new, c, g))
    reserve_not_run = sum(1 for g, _, new, _ in repl if not any(k[1] == new and group_of.get(k, k[0]) == g for k in solves))
    # report
    if not solves:
        P("RECONCILE43 campaign=%s: no solver record parsed" % campaign)
        P("RECONCILE MISMATCH")
        return 2
    by_arm = OrderedDict()
    for (arm, _), s in sorted(solves.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        by_arm.setdefault((s.code, arm), Counter())[s.verdict] += 1
    tot = Counter()
    for c in by_arm.values():
        tot.update(c)
    P("RECONCILE43 campaign=%s segments consumer=%d par2=%d solves=%d (consumer %d, par2 %d) arms=%d solver_b_json=%d ledger_seed_records=%d "
      "replacements=%d" % (campaign, n_seg["consumer"], n_seg["par2"], len(solves), sum(s.code == "consumer" for s in solves.values()),
                           sum(s.code == "par2" for s in solves.values()), len(by_arm), n_json, len(seeds_l), len(repl)))
    P("")
    P("COUNTS_BY_VERDICT (code arm: verdict=count)")
    for (code, arm), c in by_arm.items():
        P("  %-8s %-14s %s" % (code, arm, " ".join("%s=%d" % kv for kv in sorted(c.items()))))
    P("  TOTAL %s" % " ".join("%s=%d" % kv for kv in sorted(tot.items())))
    P("UNUSABLE %d solves; REPLACEMENTS %d; reserve seeds not yet run %d" % (len(unusable), len(repl), reserve_not_run))
    P("")
    for kind, items in fnd.items.items():
        for t in items:
            P("%s %s" % (kind.upper(), t))
    P("FINDINGS conflicts=%d unmapped=%d unparsed=%d json=%d ledger=%d replacement=%d rule=%d" % (
        fnd.n("conflict"), fnd.n("unmapped"), fnd.n("unparsed"), fnd.n("json"), fnd.n("ledger"), fnd.n("replacement"), fnd.n("rule")))
    ok = fnd.n() == 0
    P("RECONCILE %s" % ("OK" if ok else "MISMATCH"))
    return 0 if ok else 1
