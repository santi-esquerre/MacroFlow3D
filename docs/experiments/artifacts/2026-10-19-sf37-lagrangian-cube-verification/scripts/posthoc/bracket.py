# Post-hoc: O6(N) and Pollock(N) sigma_T^2 relative to O6(512) at far planes (do they bracket the fine value?)
import csv, sys
AN = sys.argv[1] + "/analysis"
V = [r for r in csv.DictReader(l for l in open(f"{AN}/v3/v3b_cells.csv") if not l.startswith("#"))]
key = {(r['eps_code'], r['seed'], r['N'], float(r['x1'])): r for r in V}
for x in (4.0, 16.0):
  for e in ("10", "15", "20"):
    for s in ("3101", "3102", "3103"):
        ref = float(key[(e, s, "512", x)]['sigma_T2_O6'])
        out = []
        for N in ("64", "128", "256", "512"):
            r = key[(e, s, N, x)]
            o = float(r['sigma_T2_O6']) - ref; p = float(r['sigma_T2_pollock']) - ref
            out.append(f"N{N}: O6 {o/ref*100:+.2f}% P {p/ref*100:+.2f}%")
        print(f"x1={x:g} eps{e} s{s}: " + " | ".join(out))
