`C_full.log`: 144 `C_CHECK` rows, all `verdict=PASS`: yes; log line `C_VERDICT all_pass=1`.

#### C1 (gated rows grouped by name; worst row; grids/paths in the worst-at column)

| check | rows | worst | value | worst at (grid/path) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| C1_f_samples_zero_exactly | 2 | max | 0.000e+00 | 64x16x16/host | <=0e+00 | PASS | - |
| C1_coefficients_zero_exactly | 2 | max | 0.000e+00 | 64x16x16/host | <=0e+00 | PASS | - |
| C1_v_equals_G1_e1 | 2 | max | 0.000e+00 | 64x16x16/host | <=0e+00 | PASS | - |
| C1_particles_not_ok | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C1_transverse_displacement | 4 | max | 0.000e+00 | 64x16x16/unfused | <=1e-12 | PASS | - |
| C1_tau_vs_dx1_over_G1_rel | 4 | max | 0.000e+00 | 64x16x16/unfused | <=1e-09 | PASS | - |
| C1_x1_at_times_vs_X0_plus_G1_t_rel | 4 | max | 0.000e+00 | 64x16x16/unfused | <=1e-09 | PASS | - |
| C1_abs_I | 4 | max | 0.000e+00 | 64x16x16/unfused | <=1e-14 | PASS | - |
| C1_covectors_constant | 4 | max | 0.000e+00 | 64x16x16/unfused | <=1e-12 | PASS | - |
| C1_exit_statuses | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C1_fused_equals_unfused_bitwise | 2 | max | 0.000e+00 | 64x16x16/fused | ==0 | PASS | - |

#### C2 (gated rows grouped by name; worst row; grids/paths in the worst-at column)

| check | rows | worst | value | worst at (grid/path) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| C2_particles_not_ok | 4 | max | 0.000e+00 | 64x32x32/unfused | ==0 | PASS | - |
| C2_transverse_positions_constant | 4 | max | 5.551e-17 | 128x32x32/unfused | <=1e-12 | PASS | 18014.40 |
| C2_covectors_constant | 4 | max | 7.494e-16 | 128x32x32/unfused | <=1e-12 | PASS | 1334.48 |
| C2_tau_vs_quadrature_rel | 4 | max | 5.023e-11 | 64x32x32/unfused | <=1e-09 | PASS | 19.91 |
| C2_fused_equals_unfused_bitwise | 2 | max | 0.000e+00 | 64x32x32/fused | ==0 | PASS | - |

#### C3 (gated rows grouped by name; worst row; grids/paths in the worst-at column)

| check | rows | worst | value | worst at (grid/path) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| C3_random_shift_plus_1_violations_1e-14 | 6 | max | 0.000e+00 | 64x16x16/host | ==0 | PASS | - |
| C3_random_fused_equals_unfused_eval_bitwise | 2 | max | 0.000e+00 | 64x16x16/fused | ==0 | PASS | - |
| C3_dyadic_shift_plus_1_violations_1e-14 | 6 | max | 0.000e+00 | 64x16x16/host | ==0 | PASS | - |
| C3_dyadic_fused_equals_unfused_eval_bitwise | 2 | max | 0.000e+00 | 64x16x16/fused | ==0 | PASS | - |
| C3_domain_report_outside_false_NaN_no_read | 2 | max | 0.000e+00 | 64x16x16/host+fused | ==0 | PASS | - |

#### C4a (gated rows grouped by name; worst row; grids/paths in the worst-at column)

| check | rows | worst | value | worst at (grid/path) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| C4a_particles_not_exit_outlet | 6 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_first_plane_time_not_reached_by_all | 6 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_reached_records_nonfinite_or_after_exit_not_NaN | 6 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_exit_counts_inexact | 6 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_fused_equals_unfused_bitwise | 2 | max | 0.000e+00 | 64x16x16/fused | ==0 | PASS | - |
| C4a_straddle_statuses | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_plane_on_L1_statuses | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_plane_just_beyond_statuses | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_stage_outside_domain_hmax_2h_statuses | 4 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |
| C4a_stage_outside_domain_exercised_rejected_steps | 2 | max | 3.072e+03 | 64x16x16/fused+unfused | >0 | PASS | - |
| C4a_track_json_and_exit_files | 2 | max | 0.000e+00 | 64x16x16/unfused | ==0 | PASS | - |

#### C4b (gated rows grouped by name; worst row; grids/paths in the worst-at column)

| check | rows | worst | value | worst at (grid/path) | gate | verdict | margin (derived) |
|---|---|---|---|---|---|---|---|
| C4b_plane_pass_misclassified_seeds | 6 | max | 0.000e+00 | 64x32x32/unfused | ==0 | PASS | - |
| C4b_time_pass_misclassified_seeds | 6 | max | 0.000e+00 | 64x32x32/unfused | ==0 | PASS | - |
| C4b_plane_counts_inexact | 6 | max | 0.000e+00 | 64x32x32/unfused | ==0 | PASS | - |
| C4b_time_counts_inexact | 6 | max | 0.000e+00 | 64x32x32/unfused | ==0 | PASS | - |
| C4b_fused_equals_unfused_bitwise | 2 | max | 0.000e+00 | 64x32x32/fused | ==0 | PASS | - |

#### C2 `tau` against the independent quadrature (pre-activation emulation: about 5e-11; gate 1e-9)

| grid | path | tau rel error | gate |
|---|---|---|---|
| 64x32x32 | unfused | 5.022843e-11 | <=1e-09 |
| 64x32x32 | fused | 5.022843e-11 | <=1e-09 |
| 128x32x32 | unfused | 5.022803e-11 | <=1e-09 |
| 128x32x32 | fused | 5.022803e-11 | <=1e-09 |

#### Recorded `C_RECORD` lines (no threshold)

```
C_RECORD name=C2_I_vs_quadrature grid=64x32x32 path=unfused max_abs_err=1.466e-07 max_abs_Iref=8.659e-04 q_exact=0.98677408375415
C_RECORD name=C2_I_vs_quadrature grid=64x32x32 path=fused max_abs_err=1.466e-07 max_abs_Iref=8.659e-04 q_exact=0.98677408375415
C_RECORD name=C2_I_vs_quadrature grid=128x32x32 path=unfused max_abs_err=1.481e-06 max_abs_Iref=9.678e-04 q_exact=0.943348864398513
C_RECORD name=C2_I_vs_quadrature grid=128x32x32 path=fused max_abs_err=1.481e-06 max_abs_Iref=9.678e-04 q_exact=0.943348864398513
C_RECORD name=C3_random_shift grid=64x16x16 path=host max_abs=6.661e-16 max_rel=6.944e-16 bitwise_equal_points=918 of 4096
C_RECORD name=C3_random_shift grid=64x16x16 path=device max_abs=6.661e-16 max_rel=6.944e-16 bitwise_equal_points=918 of 4096
C_RECORD name=C3_random_shift grid=64x16x16 path=fused max_abs=6.661e-16 max_rel=6.944e-16 bitwise_equal_points=918 of 4096
C_RECORD name=C3_random_host_vs_device_v grid=64x16x16 path=device max_abs=4.441e-16
C_RECORD name=C3_dyadic_shift grid=64x16x16 path=host max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_shift grid=64x16x16 path=device max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_shift grid=64x16x16 path=fused max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_host_vs_device_v grid=64x16x16 path=device max_abs=2.220e-16
C_RECORD name=C3_random_shift grid=128x16x16 path=host max_abs=8.882e-16 max_rel=6.360e-16 bitwise_equal_points=920 of 4096
C_RECORD name=C3_random_shift grid=128x16x16 path=device max_abs=8.882e-16 max_rel=6.360e-16 bitwise_equal_points=920 of 4096
C_RECORD name=C3_random_shift grid=128x16x16 path=fused max_abs=8.882e-16 max_rel=6.360e-16 bitwise_equal_points=920 of 4096
C_RECORD name=C3_random_host_vs_device_v grid=128x16x16 path=device max_abs=2.220e-16
C_RECORD name=C3_dyadic_shift grid=128x16x16 path=host max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_shift grid=128x16x16 path=device max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_shift grid=128x16x16 path=fused max_abs=0.000e+00 max_rel=0.000e+00 bitwise_equal_points=2048 of 2048
C_RECORD name=C3_dyadic_host_vs_device_v grid=128x16x16 path=device max_abs=2.220e-16
C_RECORD name=C4b_reference grid=64x32x32 path=reference included=2560 inlet=768 outlet=1792 plane_reaching=1792 T=3 psi_drift=1.95e-13
C_RECORD name=C4b_counts grid=64x32x32 path=unfused included=2560 plane:inlet=768 reached=1792 outlet=0 time:inlet=768 outlet=1792
C_RECORD name=C4b_counts grid=64x32x32 path=fused included=2560 plane:inlet=768 reached=1792 outlet=0 time:inlet=768 outlet=1792
C_RECORD name=C4b_reference grid=128x32x32 path=reference included=2560 inlet=768 outlet=1792 plane_reaching=1792 T=5.5 psi_drift=1.95e-13
C_RECORD name=C4b_counts grid=128x32x32 path=unfused included=2560 plane:inlet=768 reached=1792 outlet=0 time:inlet=768 outlet=1792
C_RECORD name=C4b_counts grid=128x32x32 path=fused included=2560 plane:inlet=768 reached=1792 outlet=0 time:inlet=768 outlet=1792
C_RECORD name=C4_reversed_head grid=64x16x16 path=unfused outcome="exception: lagrangian_labels: no injected seed has positive flux"
C_RECORD name=C4_reversed_head grid=64x16x16 path=fused outcome="exception: lagrangian_labels: no injected seed has positive flux"
```

#### C3 against the design emulation (max abs about 1.1e-15, max rel about 3.7e-15 on random points)

| grid | path | max abs | max rel | bitwise-equal points (of 4096) |
|---|---|---|---|---|
| 64x16x16 | host | 6.661e-16 | 6.944e-16 | 918 |
| 64x16x16 | device | 6.661e-16 | 6.944e-16 | 918 |
| 64x16x16 | fused | 6.661e-16 | 6.944e-16 | 918 |
| 128x16x16 | host | 8.882e-16 | 6.360e-16 | 920 |
| 128x16x16 | device | 8.882e-16 | 6.360e-16 | 920 |
| 128x16x16 | fused | 8.882e-16 | 6.360e-16 | 920 |

#### C4a Pollock non-periodic checks (`pollock_nonperiodic_tests --only C4`, `C4_pollock.log`)

| grid | case | mode | exit_inlet | exit_outlet | reached | expected | mismatches | verdict |
|---|---|---|---|---|---|---|---|---|
| 64x16x16 | C4a | plane | 0 | 256 | 256 | inlet:0,outlet:256,reached:256 | - | PASS |
| 64x16x16 | C4a | time | 0 | 256 | 256 | inlet:0,outlet:256,reached:256 | - | PASS |
| 128x16x16 | C4a | plane | 0 | 256 | 256 | inlet:0,outlet:256,reached:256 | - | PASS |
| 128x16x16 | C4a | time | 0 | 256 | 256 | inlet:0,outlet:256,reached:256 | - | PASS |
| 64x32x32 | C4b | plane | 768 | 0 | 1792 | inlet:768,outlet:0,reached:1792 | 0 | PASS |
| 64x32x32 | C4b | time | 768 | 1792 | 0 | inlet:768,outlet:1792,reached:0 | 0 | PASS |
| 128x32x32 | C4b | plane | 768 | 0 | 1792 | inlet:768,outlet:0,reached:1792 | 0 | PASS |
| 128x32x32 | C4b | time | 768 | 1792 | 0 | inlet:768,outlet:1792,reached:0 | 0 | PASS |

log line: `C4_VERDICT method=pollock PASS (22 checks, 0 failed)`.

#### C5 solver on the C1 / C2 media through the lean stage (`C5.log`)

| case | size | converged | iterations | initial | final | rel | balance | max head rel diff | pass |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 64x16x16 | 1 | 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 1 |
| C1 | 128x16x16 | 1 | 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 1 |
| C2 | 64x32x32 | 1 | 10 | 3.775619e+02 | 3.009352e-08 | 7.970487e-11 | 2.067940e-13 | 1.404551e-13 | 1 |
| C2 | 128x32x32 | 1 | 20 | 5.206390e+02 | 1.143529e-12 | 2.196396e-15 | 8.709027e-14 | 6.676744e-16 | 1 |

re-run verdict lines are in the note (`c_verdict.py`).
