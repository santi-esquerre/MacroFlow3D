# Post-hoc reading of committed SF-37 CSVs: covector finite-time rates vs travel, growth exponents, V3b trajectory part vs x1.
import csv, sys, math, collections
AN = sys.argv[1] + "/analysis"
def rows(p): return [r for r in csv.DictReader(l for l in open(p) if not l.startswith("#"))]
# 1) covector rates: median over seeds of rate_a_p50 per plane, setting primary tol1e-8_h8, N 256/512
R = rows(f"{AN}/v4v5/v5_rates.csv")
print("settings:", sorted({r['setting'] for r in R}))
for e in ("10", "15", "20"):
    for N in ("256", "512"):
        line = []
        for s in ("3101", "3102", "3103"):
            sel = sorted([r for r in R if r['eps_code']==e and r['N']==N and r['seed']==s and r['setting']=='tol1e-8_h8'], key=lambda r: float(r['plane']))
            line.append(" ".join(f"{float(r['plane']):g}:{float(r['rate_a_p50']):.3f}" for r in sel))
        print(f"eps{e} N{N} rate_a_p50 (log|a|/|a0| / tau) per plane:\n   " + "\n   ".join(line))
# median log-stretch L = rate*tau: need tau; use plane ~ tau (mean tau ~ x1). print rate*x1
print()
for e in ("10", "15", "20"):
    sel = [r for r in R if r['eps_code']==e and r['N']=='512' and r['setting']=='tol1e-8_h8']
    by = collections.defaultdict(list)
    for r in sel: by[float(r['plane'])].append(float(r['rate_a_p50']) * float(r['plane']))
    print(f"eps{e} N512 median log-stretch ~ rate_p50*x1 (3 seeds):", " ".join(f"{p:g}:[{', '.join(f'{v:.2f}' for v in by[p])}]" for p in sorted(by)))
# 2) V3b trajectory part vs x1 per N (seed-avg of abs)
V = rows(f"{AN}/v3/v3b_cells.csv")
print()
for e in ("10", "15", "20"):
    for N in ("64", "128", "256", "512"):
        sel = [r for r in V if r['eps_code']==e and r['N']==N]
        by = collections.defaultdict(list)
        for r in sel: by[float(r['x1'])].append(float(r['D_trajectory_part']))
        print(f"eps{e} N{N:>3} traj part (3 seeds) per x1:", " ".join(f"{x:g}:{max(by[x], key=abs):+.1e}" for x in sorted(by)))
