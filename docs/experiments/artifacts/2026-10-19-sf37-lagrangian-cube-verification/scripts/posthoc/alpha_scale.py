# Post-hoc (not pre-registered): grid errors expressed as alpha-equivalent rates, lambda = 0.25 (cube units).
# alpha_T-like rate over x1 in [x_a, 16]: (sigma_T2(16) - sigma_T2(x_a)) / (2 (16 - x_a)) / lambda.
# alpha_L-like rate over t in [8, 16]: (var_x1(16) - var_x1(8)) / (2 (16 - 8)) / lambda  (mean velocity ~1).
import csv, sys
A = sys.argv[1]; lam = 0.25
def mp(e, s, N, f="moments_planes.csv"):
    return {float(r[list(r)[0]]): r for r in csv.DictReader(l for l in open(f"{A}/raw/v2/v2_e{e}_s{s}_N{N}/{f}") if not l.startswith("#"))}
V = {(r['eps_code'], r['seed'], r['N'], float(r['x1'])): r for r in csv.DictReader(l for l in open(f"{A}/analysis/v3/v3b_cells.csv") if not l.startswith("#"))}
print("eps seed | aT_cube(512) [lambda] | O6 err N64/128/256 | Pollock err N64/128/256 (vs O6 512) | aL_cube(512) | O6 aL rel err N64/128/256")
for e in ("10", "15", "20"):
    for s in ("3101", "3102", "3103"):
        st = lambda N, x: (float(mp(e, s, N)[x]["var_x2"]) + float(mp(e, s, N)[x]["var_x3"])) / 2
        rate = lambda a, b: (b - a) / (2 * (16 - 4)) / lam
        ref = rate(st(512, 4.0), st(512, 16.0))
        o6 = [rate(st(N, 4.0), st(N, 16.0)) - ref for N in (64, 128, 256)]
        pol = [rate(float(V[(e, s, str(N), 4.0)]['sigma_T2_pollock']), float(V[(e, s, str(N), 16.0)]['sigma_T2_pollock'])) - ref for N in (64, 128, 256)]
        vx = lambda N, t: float(mp(e, s, N, "moments_time.csv")[t]["var_x"])
        aL = lambda N: (vx(N, 16.0) - vx(N, 8.0)) / (2 * 8) / lam
        aLr = aL(512)
        print(f"{e} {s} | {ref:.4f} | " + " ".join(f"{x:+.1e}" for x in o6) + " | " + " ".join(f"{x:+.1e}" for x in pol)
              + f" | {aLr:.3f} | " + " ".join(f"{(aL(N)-aLr)/aLr*100:+.1f}%" for N in (64, 128, 256)))
