"""Campaign state (folded from the ledger) and the queue.

Every state change goes through State.apply(record): the live engine appends a record to the ledger and applies it, a resume replays
the ledger through the same function.  Decisions (halts, replacements, re-sizing) are records of their own.

Queue order of a unit: (priority class, progress fraction of its CRN group, group order, anchor seed, replacement flag, arm order).
The progress fraction is (first seed - group start) / group size, so the units of one seed block of every arm in a group are adjacent
and groups of one priority class advance together (a cutoff cuts every group's N evenly).
"""
MEM_RETRY_S = 120.0


class ArmState:
    def __init__(self, spec, order):
        self.spec, self.id, self.order = spec, spec["id"], order
        self.start = spec["seeds"]["start"]
        self.n_target = spec["n_conf"]
        self.planned = list(range(self.start, self.start + self.n_target))
        self.done, self.failed, self.dropped, self.claimed, self.reduced = set(), set(), set(), set(), set()
        self.halted = None
        self.not_before = {}  # first seed of a unit -> epoch (OOM backoff; not persisted)
        self.oom_count, self.oom_total = {}, 0  # oom_count: first seed of the re-queued unit -> OOMs in a row
        self.other_seen = set()
        self.attempted = 0
        self.secs = []
        self.segments = []  # {segment, seeds, reduced, pruned}
        self.reest_done = False

    def block(self, s):
        c = self.spec["seeds"]["count"]
        return (s - self.start) // self.spec["unit_size"] if self.start <= s < self.start + c else ("r", s)

    def pending(self):
        x = self.done | self.failed | self.dropped | self.claimed
        return sorted(s for s in self.planned if s not in x)

    def rate(self):
        """Seconds per realization: mean of the last 20 measured, else the plan's estimate."""
        return sum(self.secs[-20:]) / len(self.secs[-20:]) if self.secs else self.spec["est_seconds"]


class Group:
    def __init__(self, gid, order, res):
        self.id, self.order, self.res = gid, order, res
        self.start, self.arms, self.next_reserve, self.repl = None, [], 0, {}  # repl: replacement seed -> replaced seed

    @property
    def size(self):
        return max((a.n_target for a in self.arms), default=1)


class Unit:
    def __init__(self, arm, seeds, key):
        self.arm, self.seeds, self.key = arm, seeds, key


class State:
    def __init__(self, plan):
        self.plan = plan
        self.arms = {a["id"]: ArmState(a, i) for i, a in enumerate(plan["arms"])}
        self.groups = {}
        for i, (g, d) in enumerate(plan["groups"].items()):
            self.groups[g] = Group(g, i, d["reserves"])
        for a in self.arms.values():
            g = self.groups[a.spec["group"]]
            g.arms.append(a)
            g.start = a.start
        self.stage1_start = None
        self.lane_b_released = False
        self.gpu_s = 0.0
        self.unit_n = 0
        self.open = {}  # unit_id -> claim record without a unit_end
        self.events = []  # (utc, kind, text) for the report
        self.sessions = 0

    def apply(self, r):
        k = r["kind"]
        a = self.arms.get(r.get("arm"))
        if k == "claim":
            self.unit_n = max(self.unit_n, r["unit"])
            self.open[r["unit"]] = r
            a.claimed |= set(r["seeds"])
        elif k == "unit_end":
            c = self.open.pop(r["unit"], None)
            if c:
                a.claimed -= set(c["seeds"])
            a.done |= set(r["completed"])
            a.secs += [float(v) for v in r.get("seconds", {}).values()]
            a.attempted += len(r["completed"])
            if r["cls"] == "unusable_solve" and r.get("failed_seed") is not None:
                a.failed.add(r["failed_seed"])
                a.attempted += 1
            if r["cls"] == "other_error" and r.get("next_seed") is not None:
                a.other_seen.add(r["next_seed"])
            if r["cls"] == "oom":
                a.oom_total += 1
                a.oom_count[r["next_seed"]] = a.oom_count.get(r["next_seed"], 0) + 1
            elif r["cls"] == "ok":
                a.oom_count.pop(r["segment_seed0"], None)
            if r["completed"]:
                a.segments.append({"segment": r["segment"], "seeds": list(r["completed"]), "reduced": False, "pruned": False})
        elif k == "replacement":
            g = self.groups[r["group"]]
            g.next_reserve += 1
            g.repl[r["replacement"]] = r["seed"]
            for b in g.arms:
                if r["seed"] in b.planned and not b.halted:
                    if r["seed"] not in b.done | b.claimed | b.failed:
                        b.dropped.add(r["seed"])
                    b.planned.append(r["replacement"])
            self.events.append((r["utc"], "replacement", "group %s seed %d -> %d (failed in %s)" % (r["group"], r["seed"], r["replacement"], r["arm"])))
        elif k == "arm_halt":
            a.halted = r["reason"]
            self.events.append((r["utc"], "arm_halt", "%s: %s" % (a.id, r["reason"])))
        elif k == "reest":
            a.planned += range(a.start + a.n_target, a.start + r["n_new"])
            a.n_target = r["n_new"]
            a.reest_done = True
            self.events.append((r["utc"], "reest", r.get("text", "")))
        elif k == "reduced":
            a.reduced |= set(r["seeds"])
            for s in a.segments:
                if s["segment"] == r["segment"]:
                    s["reduced"], s["info"] = True, r["reduced"]
        elif k == "pruned":
            for s in a.segments:
                if s["segment"] == r["segment"]:
                    s["pruned"] = True
        elif k == "stage1_start" and self.stage1_start is None:
            self.stage1_start = r["t"]
        elif k == "lane_release":
            self.lane_b_released = True
        elif k == "hb":
            self.gpu_s = max(self.gpu_s, r["gpu_s"])
        elif k == "session_start":
            self.sessions += 1
        elif k in ("guard", "lane_crash"):
            self.events.append((r["utc"], k, r.get("text", "")))

    def est(self, a, nr):
        return a.rate() * nr

    def complete(self):
        return all(not a.pending() and not a.claimed and not a.halted for a in self.arms.values())

    def candidates(self):
        out = []
        for a in self.arms.values():
            if a.halted:
                continue
            g = self.groups[a.spec["group"]]
            run = []
            for s in a.pending() + [None]:
                if run and (s is None or s != run[-1] + 1 or a.block(s) != a.block(run[0]) or len(run) >= a.spec["unit_size"]):
                    s0 = run[0]
                    anchor, rep = (g.repl[s0], 1) if s0 in g.repl else (s0, 0)
                    out.append(Unit(a.id, run, (a.spec["priority"], (anchor - g.start) / g.size, g.order, anchor, rep, a.order)))
                    run = []
                if s is not None:
                    run.append(s)
        out.sort(key=lambda u: u.key)
        return out

    @staticmethod
    def _free(cache, fn):
        """Free GPU memory (MiB), queried once per pick and only if a unit needs the comparison; unknown (None) counts as enough."""
        if not cache:
            v = fn() if fn else None
            cache.append(float("inf") if v is None else v)
        return cache[0]

    def pick(self, lane, now, free_fn, cap):
        """Next unit for the lane -> (Unit, None) | (None, wake epoch) | (None, None) when the lane has nothing it could ever claim.

        Lane B (heavy_only) skips units whose expected end would pass `cap`.  Lane A takes heavy units first (largest min_free_mib
        within the priority class), then light ones in queue order."""
        ok, waits, free = [], [], []
        for u in self.candidates():
            a = self.arms[u.arm]
            if lane.heavy_only and not a.spec["heavy"]:
                continue
            nb = a.not_before.get(u.seeds[0], 0.0)
            if cap is not None and max(now, nb) + self.est(a, len(u.seeds)) > cap:
                continue
            if nb > now:
                waits.append(nb)
            elif a.spec["min_free_mib"] > self._free(free, free_fn):
                waits.append(now + MEM_RETRY_S)
            else:
                ok.append(u)
        heavy = [u for u in ok if self.arms[u.arm].spec["heavy"]]
        if heavy and not lane.heavy_only:
            return min(heavy, key=lambda u: (u.key[0], -self.arms[u.arm].spec["min_free_mib"], u.key)), None
        return (ok[0], None) if ok else (None, min(waits) if waits else None)
