"""Simulated executor and runner processes on an injectable clock: the dry run (durations = the plan's est_seconds) and the tests.

events: {(arm id, seed): [cls, ...]}, one class consumed each time a unit reaches that seed (the seed fails after half its duration);
durations: {(arm id, seed): seconds}; sd_obs: {arm id: {estimator: sd}}; reduce_fail: {(arm id, segment basename)}.
"""


class FakeProc:
    def __init__(self, fx, seg):
        self.fx, self.seg = fx, seg

    def poll(self):
        r = self.fx.segs[self.seg]
        if r["killed"] is not None:
            return -15
        if self.fx.clock() >= r["t0"] + r["end"]:
            return 0 if r["cls"] == "ok" else 3
        return None

    def terminate(self, grace=0.0):
        r = self.fx.segs[self.seg]
        if r["killed"] is None and self.poll() is None:
            r["killed"] = self.fx.clock()


class FakeExecutor:
    def __init__(self, plan, clock, events=None, durations=None, sd_obs=None, reduce_fail=()):
        self.arms = {a["id"]: a for a in plan["arms"]}
        self.clock = clock
        self.events = {k: list(v) for k, v in (events or {}).items()}
        self.durations, self.sd_obs, self.reduce_fail = durations or {}, sd_obs or {}, set(reduce_fail)
        self.segs = {}
        self.spawned, self.reduced, self.pruned, self.blinded_calls = [], [], [], []

    # executor interface ---------------------------------------------------------------------------------------------------
    def command(self, arm, seeds, segment_dir, build_dir):
        return {"argv": ["fake", arm["id"], str(seeds[0]), str(len(seeds))], "cwd": None, "env": {}, "files": {},
                "log_path": segment_dir + "/runner.log", "sim": (arm["id"], list(seeds), segment_dir)}

    def classify(self, arm, seeds, segment_dir, exit_code, log_text):
        r = self.segs[segment_dir]
        return {"cls": r["cls"], "completed_seeds": [s for s, _ in r["items"]], "failed_seed": r["failed"],
                "verdicts": {s: "converged" for s in seeds},
                "solver": {s: {"verdict": "converged", "iterations": 30, "rel": 1e-12, "balance": 1e-10, "max_div": 2.0e-08} for s in seeds}, "seconds": {s: self.durations.get((arm["id"], s), arm["est_seconds"]) for s in seeds},
                "peak_mib": arm["min_free_mib"]}

    def completed(self, arm, segment_dir):
        r = self.segs.get(segment_dir)
        if r is None:
            return []
        e = (r["killed"] if r["killed"] is not None else self.clock()) - r["t0"]
        return [s for s, end in r["items"] if end <= e + 1e-9]

    def reduce(self, arm, segment_dir, out_dir):
        if (arm["id"], segment_dir.rsplit("/", 1)[-1]) in self.reduce_fail:
            raise RuntimeError("reduce failed")
        self.reduced.append((arm["id"], segment_dir))
        return {"tables": [out_dir + "/table.csv"], "sha256": {"table.csv": "0" * 64}}

    def prune(self, arm, segment_dir, reduced, retained_seeds):
        self.pruned.append((arm["id"], segment_dir, list(retained_seeds)))
        return {"retained": list(retained_seeds)}

    def blinded_sd(self, arm, reduced_tables):
        self.blinded_calls.append((arm["id"], len(reduced_tables)))
        return dict(self.sd_obs.get(arm["id"], {}))

    def expected_raw_gb(self, arm, n_seeds):
        return arm["raw_gb"] * n_seeds

    # simulated process ----------------------------------------------------------------------------------------------------
    def spawn(self, cmd, gpu):
        arm, seeds, seg = cmd["sim"]
        a = self.arms[arm]
        t, items, cls, failed = 0.0, [], "ok", None
        for s in seeds:
            d = self.durations.get((arm, s), a["est_seconds"])
            ev = self.events.get((arm, s))
            if ev:
                cls = ev.pop(0)
                failed = s if cls == "unusable_solve" else None
                t += d * 0.5
                break
            t += d
            items.append((s, t))
        self.segs[seg] = {"t0": self.clock(), "end": t, "items": items, "cls": cls, "failed": failed, "killed": None, "arm": arm}
        self.spawned.append((arm, list(seeds), gpu, self.clock()))
        return FakeProc(self, seg)

    def disk_used_gb(self):
        """Raw output on disk: every spawned segment (running or finished) not yet pruned."""
        gone = {s for _, s, _ in self.pruned}
        return sum(self.arms[r["arm"]]["raw_gb"] * len(r["items"]) for s, r in self.segs.items() if s not in gone)
