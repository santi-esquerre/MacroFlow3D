ctest (`final/ctest_times.tsv`): 46 tests, all Passed; sum of per-test seconds = 3064.98 s (derived); ctest 'Total Test time (real)' 3065.06 s (`final/ctest.log`).

| test | status | seconds |
|---|---|---|
| lagrangian_tracker_gpu_host | Passed | 13.79 |
| lagrangian_long_box | Passed | 33.50 |
| lagrangian_dirichlet_medium | Passed | 4.30 |
| lagrangian_ensemble | Passed | 47.68 |
| lagrangian_medium_ab | Passed | 56.66 |

Longest tests: streamfunction_operator_tests 1335.05 s; streamfunction_picard_fixed 419.31 s; streamfunction_gmres 261.46 s.

`IDENTICAL` lines in `final/results.txt` by class: binary 1, r 269, trio 172; `DIFF`/`MISSING` lines: 0.

Exit codes head vs base for the 20-run R set and the trio (`final/exit_codes.tsv`, `sf38_base_refs/exit_codes.tsv`):

| run | head rc | base rc |
|---|---|---|
| trio_config_pspta_small | 0 | 0 |
| trio_config_streamfunctions_homogeneous | 0 | 0 |
| trio_config_streamfunctions_continuation | 0 | 0 |
| r_trace_N32_e0.5_host | 0 | 0 |
| r_trace_N32_e0.5_gpu | 0 | 0 |
| r_trace_N32_e1.0_host | 21 | 21 |
| r_trace_N32_e1.0_gpu | 21 | 21 |
| r_trace_N64_e0.5_host | 0 | 0 |
| r_trace_N64_e0.5_gpu | 0 | 0 |
| r_trace_N64_e1.0_host | 0 | 0 |
| r_trace_N64_e1.0_gpu | 0 | 0 |
| r_trace_N64_e2.0_gpu | 21 | 21 |
| r_prod | 0 | 0 |
| r_fwd_smoke_g3d_n16 | 0 | 0 |
| r_fwd_g3d_n32_cov | 0 | 0 |
| r_fwd_gauss_n64_e1_cov | 0 | 0 |
| r_fwd_gauss_n64_e2 | 0 | 0 |
| r_fwd_gauss_n64_e2_tol | 0 | 0 |
| r_fwd_gauss_n40_e1 | 0 | 0 |
| r_fwd_gauss_n80_e2 | 0 | 0 |
| r_neutral_gauss_n64_e1_cov | 0 | 0 |
| r_neutral_gauss_n64_e2 | 0 | 0 |
| r_neutral_gauss_n64_e2_tol | 0 | 0 |
