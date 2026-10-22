# Feasibility table (Recorded, no threshold)

Inputs (measured, `sf40_campaign/mem/d3328_i.log`, `d3328_i.apps.csv`, `results.txt`): n = 490733568 cells (3328x384x384), S2 exact bytes = 29765452393 (60.6550 B/cell), context C = 487 MiB, PID peak = 28885 MiB (identical in `d3328_ii`).

- measured B/cell at the peak phase, exact bytes: 60.6550 (S2 exact / n)
- measured B/cell at the peak, PID basis: 60.6795 ((PID peak - C) * 2^20 / n; includes the unmodeled overhead)
- measured unmodeled overhead U at 3328x384x384 = PID peak - C - S2 exact / 2^20 = 11.45 MiB
- measured at 512^3 (`d512`): PID peak 8249 MiB, S2 exact 7761.27 MiB, overhead 0.73 MiB

Derived (extrapolation, not measured): peak(n') = C + U + b n' with b = exact B/cell and U held at its 3328 value;
cells_max = (30000 - C - U) * 2^20 / b = 5.10009e+08 cells; nx_max (ny = nz = 384, nx a multiple of 64 so that the automatic hierarchy keeps 7 levels) = 3456.
The linear model ignores the (m - n) capacity term and the scratch growth with nx (both below 0.1 B/cell), and any change of the PCG iteration count; it is not a prediction for a box that was not run.

| sigma^2 | lambda/h | measured box (lambda) | measured cells | measured PID peak MiB | derived cells_max | derived nx_max | derived Lx (lambda) | B&dD cells (derived) | B&dD need MiB (derived) | B&dD |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 416 x 48 x 48 | 490733568 | 28885 | 5.1e+08 | 3456 | 432 | 6.872e+07 | 4473.5 | fits |
| 2.25 | 16 | 208 x 24 x 24 | 490733568 | 28885 | 5.1e+08 | 3456 | 216 | 5.498e+08 | 32299.1 | does not fit |
| 4 | 32 | 104 x 12 x 12 | 490733568 | 28885 | 5.1e+08 | 3456 | 108 | 4.398e+09 | 254904.0 | does not fit |

B&dD domain = 204.8 x 25.6 x 25.6 lambda (audit `docs/experiments/2026-10-22-lagrangian-memory-audit.md` section 9); cells = product of the extents times (lambda/h)^3 (square transverse section, h = 1/ny).
The (sigma^2, lambda/h) pairs are the SF-37 values; this file makes no statement about them.

