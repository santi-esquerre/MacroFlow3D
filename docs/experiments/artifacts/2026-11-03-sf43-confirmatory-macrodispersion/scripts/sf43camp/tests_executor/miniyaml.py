"""Tiny YAML-subset loader and Par2 config validator for the executor tests (PyYAML is not on the host stack).

``validate_par2`` mirrors the rules of ``src/io/config/ConfigYaml.cpp`` (unknown keys) and ``ConfigValidator.hpp`` (sf18_medium_b, grid,
transport, inject_window, plane crossings, macrodispersion) that the generated Par2 configurations must satisfy.
"""


def _scalar(t):
    t = t.strip()
    if t in ("true", "false"):
        return t == "true"
    for f in (int, float):
        try:
            return f(t)
        except ValueError:
            pass
    return t.strip("'\"")


def _split(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += ch in "{["
        depth -= ch in "}]"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def _flow(t):
    t = t.strip()
    if t.startswith("{"):
        d = {}
        for item in _split(t[1:-1]):
            k, v = item.split(":", 1)
            d[k.strip()] = _flow(v)
        return d
    if t.startswith("["):
        return [_flow(x) for x in _split(t[1:-1])]
    return _scalar(t)


def load(text):
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack[-1][0] >= ind:
            stack.pop()
        if val.strip() == "":
            d = {}
            stack[-1][1][key] = d
            stack.append((ind, d))
        else:
            stack[-1][1][key] = _flow(val)
    return root


KNOWN = {
    "": {"run_mode", "grid", "stochastic", "flow", "transport", "analysis", "diagnostics", "output"},
    "grid": {"nx", "ny", "nz", "dx"},
    "stochastic": {"sigma2", "corr_length", "n_modes", "covariance_type", "seed", "K_mean", "field_source", "sf18_medium_b"},
    "flow": {"solver", "mg_levels", "mg_pre_smooth", "mg_post_smooth", "mg_coarse_iters", "mg_max_cycles", "cg_max_iter", "cg_rtol",
             "cg_check_every", "rtol", "verify_velocity", "pin", "pin_first_cell", "bc"},
    "transport": {"n_particles", "dt", "n_steps", "porosity", "diffusion", "alpha_l", "alpha_t", "seed", "output_every",
                  "snapshot_every", "inject_x", "method", "inject_window"},
    "analysis": {"macrodispersion", "snapshots", "plane_crossings"},
    "analysis.macrodispersion": {"enabled", "NR", "lambda", "vmean_norm", "sample_every", "var_estimator"},
    "analysis.snapshots": {"enabled", "every", "legacy_format", "include_time", "include_status", "include_wrap_counts",
                           "include_unwrapped", "stride", "max_particles", "precision"},
    "analysis.plane_crossings": {"enabled", "dump_injection", "planes"},
    "output": {"output_dir", "save_K", "save_head", "save_velocity", "save_particles", "format", "dump_field"},
    "stochastic.sf18_medium_b": {"sigma2", "eps", "seed0"},
}


def validate_par2(cfg):
    """List of error strings (empty = the real validator would accept the configuration)."""
    errs = []

    def keys(d, path):
        for k in d:
            if path in KNOWN and k not in KNOWN[path]:
                errs.append("unknown key %s.%s" % (path, k))
    keys(cfg, "")
    for sec in ("grid", "stochastic", "flow", "transport", "analysis", "output"):
        keys(cfg.get(sec, {}), sec)
    for sub in ("macrodispersion", "snapshots", "plane_crossings"):
        if sub in cfg.get("analysis", {}):
            keys(cfg["analysis"][sub], "analysis." + sub)
    g, st, tr, an = cfg["grid"], cfg["stochastic"], cfg["transport"], cfg["analysis"]
    sf18 = st.get("field_source") == "sf18_medium_b"
    if "field_source" in st and not sf18:
        errs.append("field_source unknown")
    if sf18 != ("sf18_medium_b" in st):
        errs.append("sf18_medium_b block and field_source must go together")
    if st.get("covariance_type") != 1:
        errs.append("covariance_type must be 1")
    if sf18:
        m = st["sf18_medium_b"]
        keys(m, "stochastic.sf18_medium_b")
        if set(m) != {"sigma2", "eps", "seed0"}:
            errs.append("sf18_medium_b needs sigma2, eps, seed0")
        if not m["eps"] > 0:
            errs.append("eps must be > 0")
        if st["K_mean"] != 1.0:
            errs.append("K_mean must be 1.0")
        if any(g[k] % 2 for k in ("nx", "ny", "nz")):
            errs.append("grid must be even")
        if g["dx"] != 1.0 / g["ny"]:
            errs.append("dx must equal 1/ny exactly")
    if tr.get("method", "par2") != "par2":
        errs.append("method must be par2")
    for k in ("n_particles", "dt", "n_steps", "porosity", "output_every"):
        if not tr[k] > 0:
            errs.append("transport.%s must be > 0" % k)
    if "inject_window" in tr:
        w = tr["inject_window"]
        ok = lambda a, b: 0 <= a < b <= 1  # noqa: E731
        if len(w) != 4 or not (ok(w[0], w[1]) and ok(w[2], w[3])):
            errs.append("inject_window invalid")
    pc = an.get("plane_crossings", {})
    if pc.get("enabled"):
        pl, Lx = pc["planes"], g["nx"] * g["dx"]
        if not pl or any(b <= a for a, b in zip(pl, pl[1:])):
            errs.append("planes must be non-empty and strictly increasing")
        elif not pl[0] > tr["inject_x"] or not pl[-1] < Lx:
            errs.append("first plane must be > inject_x and last < Lx")
    mac = an["macrodispersion"]
    if mac.get("enabled") and (mac["NR"] < 1 or mac["lambda"] <= 0 or mac["vmean_norm"] <= 0 or mac["sample_every"] <= 0
                               or mac["var_estimator"] not in ("biased", "unbiased")):
        errs.append("macrodispersion invalid")
    if an.get("snapshots", {}).get("enabled") and not an["snapshots"]["every"] > 0:
        errs.append("snapshots.every must be > 0")
    return errs
