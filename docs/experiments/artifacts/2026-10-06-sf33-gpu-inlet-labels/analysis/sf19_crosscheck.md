# SF-33 N6 — SF-19 cross-check (spec step 8)

On the SF-29 closure-probe field `gauss` (`eps = 0.25`, analytic `ln k`, triply periodic), SF-19 is solved at
16 / 24 / 32 (`inlet_slab --sf19-crosscheck`, `Y_cells` = `ln k` at the cell centres from
`export_proto.py --crosscheck`), and its inlet data are compared with the prototype's spectral reference
(`cases.load_case`). Recorded, no threshold (the spec asks for the differences and their observed order).

Job `sf33-crosscheck` (`scripts/run_crosscheck.sh build/v100-release`), GPU 0, 2026-10-06T22:01:32Z-22:01:46Z,
exit 0; logs `logs/crosscheck/N{16,24,32}.log`, JSON `raw/crosscheck/N{16,24,32}.json`, table
`raw/crosscheck/crosscheck.md` (`compare_proto.py --crosscheck`). All three `STATUS ok`.

## Quantities

- `v1 rms_rel`, `v1 max_rel`: SF-19 inlet U-face `v1` (face flux / face area at `x1 = 0`) vs the spectral **point**
  value `v1(0, (m2+1/2)h, (m3+1/2)h)` (`v1_face_ref`), relative to the reference rms.
- `v1 rms_rel_avg`: the same SF-19 face values vs the 3x3 Gauss-Legendre **face average** of the spectral `v1`
  (`v1_faceavg_ref`, the quantity an SF-19 face flux discretizes).
- `vperp rms`, `vperp max`: `(v2, v3)` of the SF-28 spline flow `k_v (G + grad s_h)` at the inlet vertices vs the
  spectral `(v2, v3)(0, m2 h, m3 h)` (`vperp_vertex_ref`); absolute, reference rms 6.663e-02.

## Table and observed orders

| N | v1 rms_rel | v1 max_rel | v1 rms_rel_avg | vperp rms | vperp max |
|---|---|---|---|---|---|
| 16 | 3.993e-03 | 9.292e-03 | 3.557e-03 | 2.678e-03 | 5.806e-03 |
| 24 | 1.778e-03 | 4.142e-03 | 1.592e-03 | 1.169e-03 | 2.571e-03 |
| 32 | 1.001e-03 | 2.387e-03 | 8.977e-04 | 6.538e-04 | 1.436e-03 |

| quantity | order 16->24 | order 24->32 |
|---|---|---|
| v1 rms_rel (vs point) | 1.99 | 2.00 |
| v1 max_rel (vs point) | 1.99 | 1.92 |
| v1 rms_rel_avg (vs face average) | 1.98 | 1.99 |
| vperp rms (vertex) | 2.04 | 2.02 |
| vperp max (vertex) | 2.01 | 2.02 |

Additional SETUP line of the same runs (SF-19 U-face `v1` vs the spline flow `k g1` at the face centres, i.e. the
internal consistency of the two SF-19-derived inlet velocities): rms_rel 6.127e-03 / 2.730e-03 / 1.537e-03,
max_rel 1.436e-02 / 6.338e-03 / 3.581e-03 at 16 / 24 / 32 (orders 1.99, 2.00 and 2.02, 1.98).

SF-19 solve facts (SETUP lines): MG levels 3 / 3 / 4, converged; `G = (0.99117, 1.48e-4, -4.45e-3)`,
`(0.99052, 2.32e-4, -4.51e-3)`, `(0.99028, 2.62e-4, -4.53e-3)` at 16 / 24 / 32; mean flux `(1, ~1e-19, ~1e-19)`;
`div_max` 4.5e-13 / 8.4e-13 / 2.4e-12; inlet `min v1` 0.650 / 0.639 / 0.635 (> 0, no backflow), `Q0 - 1` <= 6.7e-16;
spline potential GPU vs host prefilter 2.1e-15 / 2.4e-15 / 2.9e-15. Wall time ~1 s per grid (`TIMING`).

## Reading

Both SF-19-derived inlet inputs converge to the prototype's spectral reference at second order (orders 1.92-2.04
on both pairs, point and face-average references alike): the expected order of the 2nd-order SF-19 flux and of
the velocity it induces, and the bound the spec anticipates for the production ladder ("limited by the 2nd-order
SF-19 inlet velocity"). At 32^3 the relative inlet `v1` difference is ~1e-3 (rms) / 2.4e-3 (max) and the vertex
`v_perp` difference ~1% of its rms (6.5e-4 / 6.66e-2).
