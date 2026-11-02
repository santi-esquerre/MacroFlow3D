"""Executor of the integration test: the real ``RunnerExecutor`` whose ``reduce`` fails for one (arm, seed) named by ``plan["flaky"]``
({"arm": id, "seed": s}): every attempt of the segment that contains the seed raises, so the engine ends with an ``unreduced`` record."""
import json
import os

from sf42red import reduce as _red
from sf43camp.runner_executor import RunnerExecutor


class FlakyExecutor(RunnerExecutor):
    def reduce(self, arm, segment_dir, out_dir):
        f = (self.plan or {}).get("flaky") or {}
        if f.get("arm") == arm["id"]:
            with open(os.path.join(segment_dir, "seedmap.json")) as fh:
                seeds = [int(s) for s in json.load(fh)["seeds"].values()]
            if f["seed"] in seeds:
                raise _red.ReduceError("injected reduction failure for %s seed %d" % (arm["id"], f["seed"]))
        return super().reduce(arm, segment_dir, out_dir)
