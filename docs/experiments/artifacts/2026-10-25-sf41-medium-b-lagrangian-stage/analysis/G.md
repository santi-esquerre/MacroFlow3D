```
G_CTEST exit=0 tests=52 failed=0 wall_s=3182 PASS (summary: 100% tests passed, 0 tests failed out of 52)
G_TRIO IDENTICAL (SF-27 trio byte-compare clean vs /home/sesquerre/sf41_base_refs/trio; manifest keys timestamp build.git_hash dropped)
G_RSET IDENTICAL (R set byte-identical to /home/sesquerre/sf41_base_refs/r; SF-40 compare rules)
G_BYTES name=runner_medium_a files_ref=67 files_cand=67 identical=67 diff=0 missing=0 extra_allowed=0 extra_unexpected=0 excluded=2/2 IDENTICAL
G_BYTES name=medium_ab files_ref=22 files_cand=24 identical=22 diff=0 missing=0 extra_allowed=2 extra_unexpected=0 excluded=2/2 IDENTICAL
G_BYTES_EXTRA name=medium_ab path=realizations/r_0000/exits_b.csv
G_BYTES_EXTRA name=medium_ab path=realizations/r_0001/exits_b.csv
G_VERDICT PASS
```

| item | count (derived from `results.txt` lines) |
|---|---|
| `IDENTICAL trio:` items (SF-27 trio) | 172 |
| `IDENTICAL r:` items (SF-38 20-run R set) | 269 |
| `IDENTICAL binary:md5_differs` item (R set, head != base binary) | 1 |
| R set total (`r:` + `binary:`) | 270 |
| `IDENTICAL g_runner_a:` items (G(ii), the rc line) | 1 |
| `IDENTICAL g_ab:` items (G(iii), the rc line) | 1 |
| all `IDENTICAL` lines | 444 |
| DIFF / MISSING / EXTRA lines | 0 |
| `ctest_times.tsv` rows | 52 |
| rows `Passed` | 52 |
| sum of per-test seconds | 3181.58 |

binary line: `IDENTICAL binary:md5_differs head=ded79b8ad5ad019d81e0dc8bea1d980b base=aae63d2868be1627e80c16a3a41eb479`

#### Named ctest entries (`ctest_times.tsv`)

| test | status | seconds |
|---|---|---|
| streamfunction_operator_tests | Passed | 1350.08 |
| interpolation_periodic_tricubic | Passed | 4.04 |
| interpolation_hessian | Passed | 2.86 |
| interpolation_nonperiodic_x | Passed | 3.87 |
| inlet_slab_trace_gpu | Passed | 33.33 |
| lagrangian_tracker_gpu_host | Passed | 13.83 |
| lagrangian_cross_representation | Passed | 5.73 |
| lagrangian_dirichlet_long_box_stage | Passed | 10.04 |
| lagrangian_pollock_nonperiodic | Passed | 3.38 |
| lagrangian_dirichlet_controls | Passed | 6.21 |
| lagrangian_ensemble | Passed | 48.28 |
| lagrangian_ensemble_medium_b | Passed | 41.76 |
| lagrangian_medium_ab | Passed | 62.23 |

Five longest tests: streamfunction_operator_tests 1350.08 s; streamfunction_picard_fixed 429.78 s; streamfunction_gmres 262.86 s; streamfunction_continuation 163.78 s; streamfunction_anderson 144.79 s.

#### Base reference runs (`sf41_base_refs`) and head runs of the byte-compare sets

`BASEREFS_RECORDED r_runs=20 trio_runs=3 g_runs=2 recorded_in_exit_codes=25 unexpected_exit_codes=0`
base reference runs in `exit_codes.tsv`: 25; non-zero (run: rc) in the base part: r_trace_N32_e1.0_gpu: 21, r_trace_N32_e1.0_host: 21, r_trace_N64_e2.0_gpu: 21; in the head `final` part: r_trace_N32_e1.0_gpu: 21, r_trace_N32_e1.0_host: 21, r_trace_N64_e2.0_gpu: 21 (26 head runs); same non-zero set and codes: yes; every head run name except `ctest` also a base run: yes (derived).
