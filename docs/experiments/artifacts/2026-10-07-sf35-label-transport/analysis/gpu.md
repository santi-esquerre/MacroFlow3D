# SF-35 F: host vs GPU tracer

| N | max\|dpsi1\| | max\|dpsi2\| | NaN mismatches | status host / gpu | per-plane nfev and status histograms | <= 1e-12 | note |
|---|---|---|---|---|---|---|---|
| 128 | 0.000e+00 | 0.000e+00 | 0 | ok / ok | identical | pass |  |
| 256 | 0.000e+00 | 0.000e+00 | 0 | ok / ok | identical | pass |  |

## Wall times (driver `timing`, seconds)

| N | trace host | trace gpu | total host | total gpu | trace speedup host/gpu |
|---|---|---|---|---|---|
| 128 | 176.56 | 20.63 | 178.97 | 22.99 | 8.56 |
| 256 | 2996.68 | 499.55 | 3010.62 | 513.45 | 6.00 |

**Verdict F (max |dpsi| host vs gpu <= 1e-12 at the grids run): pass**
