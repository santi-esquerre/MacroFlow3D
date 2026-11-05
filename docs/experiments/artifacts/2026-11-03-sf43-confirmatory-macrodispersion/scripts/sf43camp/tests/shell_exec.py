"""Executor for the functional test of run_sf43.sh: drives the fake runner binary <build_dir>/fake_runner (real subprocesses)."""
import os
import shutil


class Executor:
    def __init__(self, plan, build_dir):
        self.plan, self.build_dir = plan, build_dir

    def command(self, arm, seeds, segment_dir, build_dir):
        return {"argv": [os.path.join(build_dir, "fake_runner"), arm["id"], str(seeds[0]), str(len(seeds)), segment_dir], "cwd": None,
                "env": {}, "files": {segment_dir + "/cfg.txt": "arm=%s\n" % arm["id"]}, "log_path": segment_dir + "/runner.log"}

    def completed(self, arm, segment_dir):
        out = []
        for d in sorted(os.listdir(segment_dir)) if os.path.isdir(segment_dir) else []:
            if d.startswith("r_") and os.path.exists(os.path.join(segment_dir, d, "DONE")):
                out.append(int(d[2:]))
        return out

    def classify(self, arm, seeds, segment_dir, exit_code, log_text):
        done = self.completed(arm, segment_dir)
        cls = {0: "ok", 4: "unusable_solve", 9: "oom"}.get(exit_code, "other_error")
        return {"cls": cls, "completed_seeds": done, "failed_seed": None, "verdicts": {}, "seconds": {s: 0.1 for s in done}}

    def reduce(self, arm, segment_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "table.txt"), "w") as f:
            f.write(" ".join(map(str, self.completed(arm, segment_dir))))
        return {"tables": [os.path.join(out_dir, "table.txt")]}

    def prune(self, arm, segment_dir, reduced, retained_seeds):
        for s in self.completed(arm, segment_dir):
            if s not in retained_seeds:
                shutil.rmtree(os.path.join(segment_dir, "r_%d" % s))
        return {"retained": retained_seeds}

    def blinded_sd(self, arm, reduced_tables):
        return {}

    def expected_raw_gb(self, arm, n_seeds):
        return 0.0
