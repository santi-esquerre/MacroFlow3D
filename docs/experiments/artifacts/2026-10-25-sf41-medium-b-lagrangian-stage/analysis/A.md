Tier `full`; checks in `A_full.log`: gated 79 (PASS 79, FAIL 0), recorded 54; `A_VERDICT` line of the log: `A_VERDICT all_pass=1 checks_pass=79 checks_fail=0`.

#### Gated checks grouped by name (worst row over boxes and sides; `min` for orders, `max` otherwise)

| check | rows | worst | value | worst at (box/side) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| memory_report_formula_mismatch | 1 | max | 0.000e+00 | 32x16x16/device | 0.000e+00 | PASS | - |
| node_interp_rel | 4 | max | 1.776e-15 | 32x16x16/host | 1.000e-12 | PASS | 562.91 |
| pou_value_abs | 6 | max | 3.109e-15 | 32x16x16/host | 1.000e-14 | PASS | 3.22 |
| repro_cubic_value_rel | 4 | max | 1.331e-15 | 128x16x16/host | 1.000e-12 | PASS | 751.41 |
| repro_cubic_gradient_rel | 4 | max | 5.291e-14 | 128x16x16/host | 1.000e-12 | PASS | 18.90 |
| repro_quad_value_rel | 4 | max | 1.938e-15 | 64x16x16/device | 1.000e-12 | PASS | 515.91 |
| repro_quad_gradient_rel | 4 | max | 1.005e-13 | 128x16x16/host | 1.000e-12 | PASS | 9.95 |
| ladder_min_order_ge_val_max | 2 | min | 3.902e+00 | 32x16x16..64x32x32..128x64x64/host | 3.800e+00 | PASS | 1.03 |
| ladder_min_order_ge_val_rms | 2 | min | 4.364e+00 | 32x16x16..64x32x32..128x64x64/host | 3.800e+00 | PASS | 1.15 |
| ladder_min_order_ge_grad_max | 2 | min | 2.909e+00 | 32x16x16..64x32x32..128x64x64/host | 2.800e+00 | PASS | 1.04 |
| ladder_min_order_ge_grad_rms | 2 | min | 3.373e+00 | 32x16x16..64x32x32..128x64x64/host | 2.800e+00 | PASS | 1.20 |
| ladder_min_order_ge_hess_max | 2 | min | 1.918e+00 | 32x16x16..64x32x32..128x64x64/host | 1.800e+00 | PASS | 1.07 |
| cpugpu_full_value_rel | 2 | max | 2.126e-15 | 64x32x32/both | 1.000e-13 | PASS | 47.04 |
| cpugpu_evalonly_value_rel | 2 | max | 5.002e-16 | 32x16x16/both | 1.000e-13 | PASS | 199.91 |
| cpugpu_evalonly_gradient_rel | 2 | max | 3.366e-15 | 64x32x32/both | 1.000e-13 | PASS | 29.71 |
| cpugpu_evalonly_hessian_rel | 2 | max | 6.021e-14 | 64x32x32/both | 1.000e-13 | PASS | 1.66 |
| hessian_path_bitwise_mismatches | 2 | max | 0.000e+00 | 64x32x32/host | 0.000e+00 | PASS | - |
| noalloc_eval_free_bytes_delta | 1 | max | 0.000e+00 | 64x32x32/device | 0.000e+00 | PASS | - |
| prefilter_repeat_workspace_reallocations | 1 | max | 0.000e+00 | 64x32x32/device | 0.000e+00 | PASS | - |
| shift_value_maxabs | 4 | max | 0.000e+00 | 32x16x16/host | 1.000e-14 | PASS | - |
| shift_gradient_maxabs | 4 | max | 0.000e+00 | 32x16x16/host | 1.000e-14 | PASS | - |
| shift_hessian_maxabs | 4 | max | 0.000e+00 | 32x16x16/host | 1.000e-14 | PASS | - |
| domain_margin_nonfinite_or_outside | 4 | max | 0.000e+00 | 32x16x16/host | 0.000e+00 | PASS | - |
| domain_margin_vs_independent_value_rel | 4 | max | 8.318e-16 | 64x32x32/host | 1.000e-12 | PASS | 1202.16 |
| domain_margin_vs_independent_gradient_rel | 4 | max | 1.454e-15 | 64x32x32/device | 1.000e-12 | PASS | 687.73 |
| domain_outside_reported_without_read | 4 | max | 0.000e+00 | 32x16x16/host | 0.000e+00 | PASS | - |
| domain_half_open_violations | 2 | max | 0.000e+00 | 32x16x16/host | 0.000e+00 | PASS | - |

margin = gate / worst value for upper-bound gates, worst value / gate for order gates; `-` for exact-zero gates or zero values.

#### Order ladder `f = exp(x1) sin(3 x1) (1 + 0.05 S)` on 32x16x16, 64x32x32, 128x64x64: minimum over the pairs 16->32, 32->64 (`A_LADDER_ORDER`, host; device equal: yes)

| quantity | predicted (PRE, A_ladder emulation) | measured min | measured - predicted (derived) | gate | margin over gate (derived) |
|---|---|---|---|---|---|
| val_max | 3.9024 | 3.901686 | -0.0007 | 3.8 | 0.1017 |
| val_rms | 4.3659 | 4.363914 | -0.0020 | 3.8 | 0.5639 |
| grad_max | 2.9042 | 2.908731 | +0.0045 | 2.8 | 0.1087 |
| grad_rms | 3.3858 | 3.373425 | -0.0124 | 2.8 | 0.5734 |
| hess_max | 1.9154 | 1.917651 | +0.0023 | 1.8 | 0.1177 |

Per pair (host):

| pair | val_max | val_rms | grad_max | grad_rms | hess_max |
|---|---|---|---|---|---|
| 16->32 | 3.901686 | 4.363914 | 2.908731 | 3.373425 | 1.917651 |
| 32->64 | 3.975464 | 4.428238 | 2.968985 | 3.435354 | 1.970022 |

#### Recorded: x-only sub-ladder `g = exp(x1) sin(3 x1)`, `L1 = 2`, boxes `nx x 4 x 4` with 16/32/64/128 cells per unit length (host; device equal: yes)

| pair | val_max | val_rms | grad_max | grad_rms | hess_max |
|---|---|---|---|---|---|
| 16->32 | 3.900789 | 4.372389 | 2.905061 | 3.380338 | 1.915694 |
| 32->64 | 3.96186 | 4.480636 | 2.962639 | 3.458511 | 1.966126 |
| 64->128 | 3.98698 | 4.517801 | 2.985752 | 3.511095 | 1.986263 |

Predicted 64->128 orders (SPEC) vs measured: val_max 3.98 vs 3.98698; grad_max 2.98 vs 2.985752; hess_max 1.99 vs 1.986263.

#### Recorded: measured over emulated error per level (`A_EMUL`, host; device lines equal the host lines: yes)

| quantity | level h | measured | emulated | measured / emulated |
|---|---|---|---|---|
| val_max | 1/16 | 2.147676e-03 | 2.175690e-03 | 0.987124 |
| val_max | 1/32 | 1.436959e-04 | 1.455010e-04 | 0.987594 |
| val_max | 1/64 | 9.135043e-06 | 9.360220e-06 | 0.975943 |
| val_rms | 1/16 | 3.141618e-04 | 2.951800e-04 | 1.064306 |
| val_rms | 1/32 | 1.525754e-05 | 1.431620e-05 | 1.065753 |
| val_rms | 1/64 | 7.086828e-07 | 6.477280e-07 | 1.094106 |
| grad_max | 1/16 | 1.191266e-01 | 1.201610e-01 | 0.991392 |
| grad_max | 1/32 | 1.586330e-02 | 1.605180e-02 | 0.988257 |
| grad_max | 1/64 | 2.026002e-03 | 2.060180e-03 | 0.983410 |
| grad_rms | 1/16 | 2.169646e-02 | 2.047930e-02 | 1.059434 |
| grad_rms | 1/32 | 2.093566e-03 | 1.959180e-03 | 1.068593 |
| grad_rms | 1/64 | 1.935272e-04 | 1.797730e-04 | 1.076509 |
| hess_max | 1/16 | 3.990829e+00 | 4.022630e+00 | 0.992094 |
| hess_max | 1/32 | 1.056313e+00 | 1.066380e+00 | 0.990560 |
| hess_max | 1/64 | 2.696230e-01 | 2.730160e-01 | 0.987572 |

derived: measured/emulated over the 15 host rows: min 0.975943, max 1.094106.

#### Recorded (ungated) checks (`A_CHECK ... verdict=RECORDED`, host and device rows)

| check | rows | worst | value | worst at (box/side) |
|---|---|---|---|---|
| pou_gradient_abs_ps | 6 | max | 1.075e-13 | 128x64x64/host |
| pou_value_abs_margin | 6 | max | 2.220e-15 | 64x32x32/host |
| pou_gradient_abs_margin | 6 | max | 8.096e-14 | 128x64x64/host |
| repro_cubic_hessian_rel | 4 | max | 2.517e-12 | 128x16x16/host |
| repro_quad_hessian_rel | 4 | max | 7.386e-12 | 128x16x16/host |
| cpugpu_coeff_diff_rel | 2 | max | 3.458e-14 | 64x32x32/both |
| cpugpu_full_gradient_rel_interior | 2 | max | 5.239e-15 | 64x32x32/both |
| cpugpu_full_hessian_rel_interior | 2 | max | 1.797e-13 | 64x32x32/both |
| cpugpu_full_gradient_rel_nearface | 2 | max | 1.836e-14 | 64x32x32/both |
| cpugpu_full_hessian_rel_nearface | 2 | max | 3.716e-13 | 64x32x32/both |
| cpugpu_full_gradient_rel_margin | 2 | max | 3.833e-14 | 64x32x32/both |
| cpugpu_full_hessian_rel_margin | 2 | max | 9.555e-13 | 64x32x32/both |
| shift_memcmp_mismatched_pairs | 4 | max | 0.000e+00 | 32x16x16/host |

`A_memcheck.log` (compute-sanitizer memcheck, `--only domain`): `A_VERDICT all_pass=1 checks_pass=18 checks_fail=0`; ERROR SUMMARY: 0 errors.
`A_MEMORY` (32x16x16): `A_MEMORY box=32x16x16 coefficient_bytes=73728 spectrum_bytes=82944 line_table_bytes=224 cufft_work_area_bytes=165888 total_device_bytes=322784`.
No-allocation record: `A_NOALLOC free_before=33560199168 free_after=33560199168` (`cudaMemGetInfo` free bytes before / after).
