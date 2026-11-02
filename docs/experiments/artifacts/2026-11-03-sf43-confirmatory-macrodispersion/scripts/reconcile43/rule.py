"""`B_SOLVER` grammar and verdict rule (copy of the generic part of SF-42 `reconcile/solver.py`, itself the production rule of
`DirichletLongBoxStage.cu` `make_dirichlet_solver_record`).  Self-standing: no import from the SF-42 tree.

  non_finite                        any of rel, balance, max_div, qbar, Q_in, Q_out not finite
  converged                         converged=1 and balance <= 1e-8
  converged_balance_above_1e-8      converged=1 and balance > 1e-8
  fallback                          converged=0 and rel <= 1e-8 and balance <= 1e-8
  not_converged_beyond_fallback     otherwise
A solve is USABLE iff its verdict is `converged` or `fallback`.  A field cut inside its number (not the full `%.6e` / `%.12e`
format) is treated as absent, and a record without a complete converged, rel, balance, qbar, Q_in and Q_out is `unparsed_truncated`
(the SF-42 rule for cut job logs; the per-segment `run.log` of SF-43 is not cut, so a truncated record is an anomaly).
"""
import math
import re

TOL = 1e-8
USABLE = ("converged", "fallback")
VERDICTS = ("converged", "converged_balance_above_1e-8", "fallback", "not_converged_beyond_fallback", "non_finite", "unparsed_truncated")

_E6 = r"(?:[-+]?\d\.\d{6}e[+-]\d{2,3}|[-+]?nan|[-+]?inf)"
_E12 = r"(?:[-+]?\d\.\d{12}e[+-]\d{2,3}|[-+]?nan|[-+]?inf)"
_FIELDS = (("converged", r"[01]"), ("iterations", r"\d+"), ("initial", _E6), ("final", _E6), ("rel", _E6), ("qbar", _E12),
           ("Q_in", _E12), ("Q_out", _E12), ("balance", _E6), ("max_div", _E6), ("fallback", r"[01]"))
_SOLVER = re.compile(r"B_SOLVER r=(\d+)(?P<rest>.*)$")


def parse_solver_line(line):
    """(r, fields{name: str}) of a B_SOLVER line, or None.  Only the leading run of consecutive complete fields is kept."""
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
    """(verdict, notes) from parsed fields."""
    need = ("converged", "rel", "balance", "qbar", "Q_in", "Q_out")
    if any(k not in fields for k in need):
        return "unparsed_truncated", ["missing %s" % ",".join(k for k in need if k not in fields)]
    notes = [] if "max_div" in fields else ["max_div_missing"]
    keys = ("rel", "balance", "qbar", "Q_in", "Q_out") + (("max_div",) if "max_div" in fields else ())
    if not all(math.isfinite(float(fields[k])) for k in keys):
        return "non_finite", notes
    rel, bal = float(fields["rel"]), float(fields["balance"])
    if fields["converged"] == "1":
        return ("converged" if bal <= TOL else "converged_balance_above_1e-8"), notes
    return ("fallback" if rel <= TOL and bal <= TOL else "not_converged_beyond_fallback"), notes
