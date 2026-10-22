# Growth exponents at N = 512 (log2 ratio per doubling): var_tau(x1), transverse growth G(x1) = sigma_T2(x1) - sigma_T2(0.25), var_x(t).
import csv, sys, math
A = sys.argv[1]
def rd(e, s, f):
    return {float(r[list(r)[0]]): r for r in csv.DictReader(l for l in open(f"{A}/raw/v2/v2_e{e}_s{s}_N512/{f}") if not l.startswith("#"))}
for e in ("10", "15", "20"):
    for s in ("3101", "3102", "3103"):
        p = rd(e, s, "moments_planes.csv"); t = rd(e, s, "moments_time.csv")
        vt = lambda x: float(p[x]["var_tau"]); st = lambda x: (float(p[x]["var_x2"]) + float(p[x]["var_x3"])) / 2
        G = lambda x: st(x) - st(0.25); vx = lambda x: float(t[x]["var_x"])
        f = lambda a, b: math.log2(b / a) if a > 0 and b > 0 else float("nan")
        print(f"eps{e} s{s}: var_tau exps 2->4 {f(vt(2),vt(4)):.2f} 4->8 {f(vt(4),vt(8)):.2f} 8->16 {f(vt(8),vt(16)):.2f} | "
              f"G_T exps 4->8 {f(G(4),G(8)):.2f} 8->16 {f(G(8),G(16)):.2f} | var_x(t) exps 4->8 {f(vx(4),vx(8)):.2f} 8->16 {f(vx(8),vx(16)):.2f} "
              f"| var_x2/var_x3 at 16: {float(p[16.0]['var_x2']):.3f}/{float(p[16.0]['var_x3']):.3f}")
