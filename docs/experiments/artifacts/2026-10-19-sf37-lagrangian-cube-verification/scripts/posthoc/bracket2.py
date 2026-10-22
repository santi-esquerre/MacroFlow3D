import csv, sys, collections
AN = sys.argv[1] + "/analysis"
V = [r for r in csv.DictReader(l for l in open(f"{AN}/v3/v3b_cells.csv") if not l.startswith("#"))]
key = {(r['eps_code'], r['seed'], r['N'], float(r['x1'])): r for r in V}
cnt = collections.Counter(); sgn = collections.Counter()
for (e, s, N, x), r in key.items():
    if N == "512": continue
    ref = float(key[(e, s, "512", x)]['sigma_T2_O6'])
    o = float(r['sigma_T2_O6']) - ref; p = float(r['sigma_T2_pollock']) - ref
    cnt[(N, abs(p) < abs(o))] += 1
    sgn[(N, "O6<ref" if o < 0 else "O6>=ref")] += 1
for N in ("64", "128", "256"):
    print(f"N{N}: Pollock closer to O6(512) than O6(N) in {cnt[(N, True)]}/{cnt[(N, True)]+cnt[(N, False)]} cells; O6(N)<O6(512) in {sgn[(N,'O6<ref')]}")
