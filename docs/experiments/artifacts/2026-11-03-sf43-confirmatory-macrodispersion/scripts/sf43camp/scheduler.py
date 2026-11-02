"""Campaign state (folded from the ledger) and the queue.

Every state change goes through State.apply(record): the live engine appends a record to the ledger and applies it, a resume replays
the ledger through the same function.  Decisions (halts, replacements, re-sizing) are records of their own.

Queue order of a unit: (priority class, progress fraction of its CRN group, group order, anchor seed, replacement flag, arm order).
The progress fraction is (first seed - group start) / group size with the group size = the largest `n_conf` of the group (not the
re-sized `n_target`), so the units of one seed block of every arm in a group are adjacent and groups of one priority class advance
together (a cutoff cuts every group's N evenly).  The extension seeds a blinded re-estimation adds (positions >= n_conf) are separate
units of priority `EXT_PRIORITY` (2) or the arm's own priority if that is lower in rank.

Lane pick: among the claimable units of the highest pending priority, lane B (heavy only) takes the first in queue order, lane A
prefers the largest `min_free_mib` (near-empty GPU first), then queue order.  Lane B never waits for units outside its class
(heavy and `min_free_mib` <= the lane's launch threshold): it only takes a near-empty unit when its live free memory allows.
"""
MEM_RETRY_S = 120.0
EXT_PRIORITY = 2


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
        self.peak_mib = None  # largest per-unit device memory the runner reported (MEM_PHASE device_used / exact_bytes)

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
        return max((a.spec["n_conf"] for a in self.arms), default=1)


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
        self.lane_names = {"A", "B"}  # lanes that can claim now (the engine updates it); a pin to another lane is ignored

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
            if r.get("peak_mib") is not None:
                a.peak_mib = max(a.peak_mib or 0.0, float(r["peak_mib"]))
            a.attempted += len(r["completed"])
            if r["cls"] == "unusable_solve":
                fs = set(r.get("failed_seeds") or []) | ({r["failed_seed"]} if r.get("failed_seed") is not None else set())
                a.failed |= fs
                a.attempted += len(fs)
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
            g.repl[r["replaced_by"]] = r["seed"]
            for b in g.arms:
                if r["seed"] in b.planned and not b.halted:
                    if r["seed"] not in b.done | b.claimed | b.failed:
                        b.dropped.add(r["seed"])
                    b.planned.append(r["replaced_by"])
            self.events.append((r["utc"], "replacement", "group %s seed %d -> %d (failed in %s)" % (r["group"], r["seed"], r["replaced_by"], r["arm"])))
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
                    s["pruned"], s["prune_info"] = True, r.get("info") or {}
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

    def is_ext(self, a, s):
        """A seed of the arm's own range beyond its n_conf: an extension added by the blinded re-estimation."""
        return a.start + a.spec["n_conf"] <= s < a.start + a.spec["seeds"]["count"]

    def candidates(self):
        out = []
        for a in self.arms.values():
            if a.halted:
                continue
            g = self.groups[a.spec["group"]]
            run = []
            for s in a.pending() + [None]:
                if run and (s is None or s != run[-1] + 1 or a.block(s) != a.block(run[0]) or len(run) >= a.spec["unit_size"]
                            or self.is_ext(a, s) != self.is_ext(a, run[0])):
                    s0 = run[0]
                    anchor, rep = (g.repl[s0], 1) if s0 in g.repl else (s0, 0)
                    prio = max(a.spec["priority"], EXT_PRIORITY) if self.is_ext(a, s0) else a.spec["priority"]
                    out.append(Unit(a.id, run, (prio, (anchor - g.start) / g.size, g.order, anchor, rep, a.order)))
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

        Lane B (heavy_only) skips units whose expected end would pass `cap`, and only waits (memory, OOM back-off) for units of its
        class (`lane.class_mib`: heavy and min_free_mib <= the threshold; None = every heavy unit); a unit outside its class is
        claimed only when it fits the live free memory at once, and is never waited for.  Among the claimable units only those of the
        highest pending priority are considered; lane B takes the first in queue order, lane A the one with the largest min_free_mib
        (near-empty GPU first), then queue order."""
        ok, waits, free = [], [], []
        for u in self.candidates():
            a = self.arms[u.arm]
            pin = a.spec.get("lane")
            if pin and pin in self.lane_names and pin != lane.name:
                continue  # pinned to the other lane (smoke test: determinism pairs); ignored if that lane does not exist
            if lane.heavy_only and not a.spec["heavy"] and pin != lane.name:
                continue
            in_class = not lane.heavy_only or lane.class_mib is None or a.spec["min_free_mib"] <= lane.class_mib
            nb = a.not_before.get(u.seeds[0], 0.0)
            if cap is not None and max(now, nb) + self.est(a, len(u.seeds)) > cap:
                continue
            if nb > now:
                if in_class:
                    waits.append(nb)
            elif a.spec["min_free_mib"] > self._free(free, free_fn):
                if in_class:
                    waits.append(now + MEM_RETRY_S)
            else:
                ok.append(u)
        if not ok:
            return None, (min(waits) if waits else None)
        top = min(u.key[0] for u in ok)
        ok = [u for u in ok if u.key[0] == top]
        if lane.heavy_only:
            return ok[0], None
        return min(ok, key=lambda u: (-self.arms[u.arm].spec["min_free_mib"], u.key)), None

    # retention (G8, H21) ----------------------------------------------------------------------------------------------------
    def order_of(self, a):
        """Effective seed order of the arm's first n_conf positions: a replaced seed's position holds its (chained) replacement."""
        fwd = {v: k for k, v in self.groups[a.spec["group"]].repl.items()}  # replaced seed -> reserve
        out = []
        for s in range(a.start, a.start + a.spec["n_conf"]):
            while s in fwd:
                s = fwd[s]
            out.append(s)
        return out

    def retain_set(self, a, k):
        """Seeds whose raw arrays must survive pruning: the configured human-review sample (and the replacement of any of them that
        was replaced), the first k positions of the effective order, and the first k COMPLETED seeds in that order.  Pruning decides
        per segment at its time: the position rule makes the choice independent of the order in which segments complete, the rank
        rule covers a position that never completes."""
        fwd = {v: x for x, v in self.groups[a.spec["group"]].repl.items()}
        keep = set(a.spec["retained"])
        for s in list(keep):
            while s in fwd:
                s = fwd[s]
                keep.add(s)
        order = self.order_of(a)
        return keep | set(order[:k]) | set([s for s in order if s in a.done][:k])
