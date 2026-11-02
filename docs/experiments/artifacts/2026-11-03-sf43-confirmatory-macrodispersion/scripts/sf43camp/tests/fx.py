"""Test harness: plan builder, virtual clock, simulated 'kill -9' of the launcher, ledger replay."""
import gc
import json
import os
import tempfile

from sf43camp import campaign
from sf43camp.fake import FakeExecutor
from sf43camp.ledger import Ledger
from sf43camp.plan import utc_iso, validate
from sf43camp.scheduler import State

T0 = 1_800_000_000  # virtual start (epoch s)
H = 3600.0


def arm(i, group="g1", heavy=True, n=10, unit=5, est=100.0, prio=0, start=100, count=None, mem=1000, raw=1.0, retained=()):
    return {"id": i, "family": "x", "amplitude": 1.0, "runner": "consumer", "priority": prio, "heavy": heavy, "min_free_mib": mem, "group": group,
            "seeds": {"start": start, "count": count or n}, "n_conf": n, "n_max": count or n, "unit_size": unit, "est_seconds": est, "raw_gb": raw,
            "consumer": {}, "reduce": {}, "retained": list(retained)}


def mk_plan(cdir, arms, cutoff_h=1000.0, **over):
    groups = sorted({a["group"] for a in arms})
    p = {"campaign": "t", "schema_version": 1, "cutoff_utc": utc_iso(T0 + cutoff_h * H), "grace_minutes": 30, "stage1_max_hours": 30,
         "budget_gpu_h": 1e6, "disk_min_free_gb": 0, "campaign_dir": cdir, "binaries": {}, "analysis": {"command": ["an", "{campaign_dir}"], "sha256_list": {}},
         "recompute": {"command": ["rc"]}, "reconcile": {"command": ["rec"]}, "failure": {"halt_rate": 0.5, "min_for_rate": 1000},
         "groups": {g: {"reserves": {"start": 900 + 50 * k, "count": 5}} for k, g in enumerate(groups)}, "arms": arms, "plan_sha256": "abc"}
    p.update(over)
    validate(p)
    return p


class Kill(Exception):
    pass


class Sim:
    """Virtual clock; run() = one launcher session on it."""

    def __init__(self, plan, **ex_kw):
        self.t = float(T0)
        self.plan = plan
        self.ex = FakeExecutor(plan, lambda: self.t, **ex_kw)
        self.lines, self.cmds, self.kill_at, self.rc = [], [], None, {}

    def sleep(self, s):
        self.t += s
        if self.kill_at is not None and self.t >= self.kill_at:
            raise Kill()

    def runner(self, argv, log_path):
        self.cmds.append(argv)
        return self.rc.get(argv[0], 0)

    def run(self, gpus=("1", "0"), resume=False, **kw):
        kw.setdefault("disk_free_gb", lambda p: 1e9)
        kw.setdefault("gpu_free_mib", lambda g: None)
        try:
            return campaign.run_campaign(self.plan, "bd", self.ex, resume=resume, gpus=list(gpus), clock=lambda: self.t, sleep=self.sleep,
                                         spawn=self.ex.spawn, threaded=False, log=self.lines.append, poll_s=60.0, runner=self.runner, **kw)
        except Kill:
            for r in self.ex.segs.values():  # the runners die with the launcher
                if r["killed"] is None and self.t < r["t0"] + r["end"]:
                    r["killed"] = self.t
            gc.collect()  # release the LOCK of the 'dead' launcher
            return "killed"

    def records(self):
        return Ledger(os.path.join(self.plan["campaign_dir"], "ledger.jsonl"), None).records()

    def state(self):
        st = State(self.plan)
        for r in self.records():
            st.apply(r)
        return st

    def log(self):
        return "\n".join(self.lines)


def tmpdir(case):
    d = tempfile.TemporaryDirectory()
    case.addCleanup(d.cleanup)
    return d.name


def write_json(path, d):
    with open(path, "w") as f:
        json.dump(d, f)
