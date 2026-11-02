"""SF-43 pre-activation review: common admissible seed-lattice sizes M per amplitude (decision H10).

Uses SF-42's exact reproduction of the runner's seed lattice and RR12 seed-on-face check (`seedlattice.rule`, verified against the
production code in SF-42 section C: 1575 cases, 0 mismatches). An M is admissible for an amplitude if it passes on every arm grid of
that amplitude with the amplitude's fixed source (window centred, source S in lambda, width W in lambda; the grid's ny = W * lambda/h).

Usage: python3 lattice_check.py --sf42-scripts <path to docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts>
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf42-scripts", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.sf42_scripts)
    from seedlattice import plan, rule

    def common(arms, S, rng=range(40, 301)):
        adm = None
        for W, lh in arms:
            s = set(rule.admissible(int(round(W * lh)), plan.window_string(W, S), rng))
            adm = s if adm is None else adm & s
        return sorted(adm)

    cases = [
        ("0.25 and 1", 19.2, [(24, 10), (24, 5), (24, 20), (48, 10)]),
        ("2.25 at 19.2", 19.2, [(24, 16), (24, 4), (24, 8), (48, 8), (24, 10), (48, 10), (24, 12)]),
        ("2.25 at 19.08", 19.08, [(24, 16), (24, 4), (24, 8), (48, 8), (24, 10), (48, 10), (24, 12)]),
        ("2.25 at 19.32", 19.32, [(24, 16), (24, 4), (24, 8), (48, 8), (24, 10), (48, 10), (24, 12)]),
        ("4 at 9.6 (with the lambda/h 4 rung)", 9.6, [(12, 32), (12, 8), (12, 16), (24, 8), (24, 16), (12, 4)]),
    ]
    for name, S, arms in cases:
        c = common(arms, S)
        print("%-38s source %.2f lambda: %3d admissible M in [40, 300]; M=100 %s; M=128 %s; first: %s"
              % (name, S, len(c), "yes" if 100 in c else "no", "yes" if 128 in c else "no", c[:8]))


if __name__ == "__main__":
    main()
