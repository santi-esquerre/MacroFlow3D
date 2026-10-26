#### Jobs (every file of `raw/job_logs/`, status from `raw/job_status/`; `window` = finished minus started, derived)

| job | evidence | started (UTC) | finished (UTC) | window s | finished exit | status STATE/EXIT_CODE | TREE echo |
|---|---|---|---|---|---|---|---|
| sf41-abrerun | yes | 2026-10-25T02:39:21Z | 2026-10-25T03:25:13Z | 2752 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-basebuild | yes | 2026-10-24T23:44:20Z | 2026-10-24T23:47:11Z | 171 | 0 | succeeded/0 | 2cf48c1 BASE=2cf48c19dac641e1ce8c823c5cf2d6165ac66757 |
| sf41-baserefs | yes | 2026-10-25T01:47:25Z | 2026-10-25T01:51:01Z | 216 | 0 | succeeded/0 | 0256bb0 SCRIPTS_ONLY_FOR_BASEREFS BASE=2cf48c1 |
| sf41-build | yes | 2026-10-25T02:10:44Z | 2026-10-25T02:16:10Z | 326 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-controls | yes | 2026-10-25T02:16:40Z | 2026-10-25T02:20:03Z | 203 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-cross | yes | 2026-10-25T02:20:40Z | 2026-10-25T02:21:14Z | 34 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-evladder | yes | 2026-10-25T02:29:56Z | 2026-10-25T02:38:52Z | 536 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-final | yes | 2026-10-25T03:25:41Z | 2026-10-25T04:22:29Z | 3408 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |
| sf41-pre-e | no (pre-check) | 2026-10-25T01:30:22Z | 2026-10-25T01:40:58Z | 636 | 0 | succeeded/0 | - |
| sf41-pre-n1-full | no (pre-check) | 2026-10-25T00:08:34Z | 2026-10-25T00:09:32Z | 58 | 0 | succeeded/0 | - |
| sf41-pre-n1b | no (pre-check) | 2026-10-25T00:41:29Z | 2026-10-25T00:41:50Z | 21 | 0 | succeeded/0 | - |
| sf41-pre-n2 | no (pre-check) | 2026-10-25T00:21:20Z | 2026-10-25T00:25:49Z | 269 | 0 | succeeded/0 | - |
| sf41-pre-n2c-clean | no (pre-check) | 2026-10-25T01:24:11Z | 2026-10-25T01:28:59Z | 288 | 0 | succeeded/0 | - |
| sf41-pre-n2c-d | no (pre-check) | 2026-10-25T01:16:33Z | 2026-10-25T01:18:49Z | 136 | 0 | succeeded/0 | - |
| sf41-pre-n2c-diag | no (pre-check) | 2026-10-25T01:12:30Z | 2026-10-25T01:15:07Z | 157 | 0 | succeeded/0 | - |
| sf41-pre-n3b | no (pre-check) | 2026-10-25T00:52:43Z | 2026-10-25T00:57:52Z | 309 | 0 | succeeded/0 | - |
| sf41-prodgrid | yes | 2026-10-25T02:21:43Z | 2026-10-25T02:29:19Z | 456 | 0 | succeeded/0 | 269c658687816788fdaf65f9d5b1307d87376b6a |

#### Parts (start = `run.env` DATE_UTC; PART_EXIT; runs = rows of `exit_codes.tsv`; part_seconds / part_runs_seconds = `RECORDED` line of `results.txt`)

| part | start (UTC) | PART_EXIT | runs | non-zero rc | part_seconds | part_runs_seconds | SF41_COMMIT | BINARIES_CHECK |
|---|---|---|---|---|---|---|---|---|
| baserefs | 2026-10-25T01:47:26Z | 0 | 25 | 21 | 215 | 196 | 2cf48c1 | base binaries: md5 recorded only |
| build | 2026-10-25T02:10:45Z | 0 | 2 | - | 325 | 311 | 269c658 | not_applicable(build part) |
| controls | 2026-10-25T02:16:44Z | 0 | 8 | - | 203 | 172 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| cross | 2026-10-25T02:20:45Z | 0 | 1 | - | 33 | 15 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| prodgrid | 2026-10-25T02:21:47Z | 0 | 4 | - | 456 | 424 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| evladder | 2026-10-25T02:30:01Z | 0 | 24 | - | 535 | 450 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| abrerun | 2026-10-25T02:39:26Z | 0 | 2 | - | 2751 | 2728 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| final | 2026-10-25T03:25:45Z | 0 | 26 | 21 | 3407 | 3380 | 269c658 | ok(50 binaries, md5file_sha256=82751b5ebafa457d) |
| build attempt 1 (superseded, not counted) | 2026-10-25T02:02:40Z | 1 | 2 | 1 | 325 | 314 | 269c658 | - |

derived (GPU-hours = wall hours: one GPU, one job at a time): sum of `part_seconds` over baserefs + build (attempt 2) + the six head parts = 7925 s = 2.201 h; sum of `part_runs_seconds` = 7676 s = 2.132 h; adding the base build job window (`sf41-basebuild`, 2026-10-24T23:44:20Z to 2026-10-24T23:47:11Z = 171 s) gives 8096 s = 2.249 h; the superseded build attempt 1 (325 s) is not counted.

| part | GPU-hours (part_seconds / 3600, derived) |
|---|---|
| baserefs | 0.060 |
| build | 0.090 |
| controls | 0.056 |
| cross | 0.009 |
| prodgrid | 0.127 |
| evladder | 0.149 |
| abrerun | 0.764 |
| final | 0.946 |
| build attempt 1 (superseded, not counted) | 0.090 |

#### Per-run wall seconds (`walls.tsv`; the 24 runs of `evladder` are summarised in F.md)

| part | run | wall s | end (UTC) |
|---|---|---|---|
| controls | A_full | 5 | 2026-10-25T02:17:03Z |
| controls | A_memcheck | 10 | 2026-10-25T02:17:15Z |
| controls | C_full | 12 | 2026-10-25T02:17:29Z |
| controls | C4_pollock | 4 | 2026-10-25T02:17:36Z |
| controls | C5 | 4 | 2026-10-25T02:17:42Z |
| controls | D | 130 | 2026-10-25T02:19:56Z |
| controls | MEMMODEL_a | 3 | 2026-10-25T02:19:59Z |
| controls | MEMMODEL_b | 4 | 2026-10-25T02:20:03Z |
| cross | B | 15 | 2026-10-25T02:21:14Z |
| prodgrid | E1 | 35 | 2026-10-25T02:22:37Z |
| prodgrid | E2 | 112 | 2026-10-25T02:24:34Z |
| prodgrid | E3 | 131 | 2026-10-25T02:26:49Z |
| prodgrid | E4 | 146 | 2026-10-25T02:29:19Z |
| abrerun | ab_s1 | 1110 | 2026-10-25T02:58:11Z |
| abrerun | ab_s4 | 1618 | 2026-10-25T03:25:13Z |
| final | ctest | 3182 | 2026-10-25T04:19:01Z |
| final | trio_config_pspta_small | 5 | 2026-10-25T04:19:06Z |
| final | trio_config_streamfunctions_homogeneous | 3 | 2026-10-25T04:19:09Z |
| final | trio_config_streamfunctions_continuation | 3 | 2026-10-25T04:19:13Z |
| final | r_trace_N32_e0.5_host | 6 | 2026-10-25T04:19:19Z |
| final | r_trace_N32_e0.5_gpu | 5 | 2026-10-25T04:19:24Z |
| final | r_trace_N32_e1.0_host | 6 | 2026-10-25T04:19:30Z |
| final | r_trace_N32_e1.0_gpu | 6 | 2026-10-25T04:19:36Z |
| final | r_trace_N64_e0.5_host | 16 | 2026-10-25T04:19:52Z |
| final | r_trace_N64_e0.5_gpu | 6 | 2026-10-25T04:19:58Z |
| final | r_trace_N64_e1.0_host | 17 | 2026-10-25T04:20:16Z |
| final | r_trace_N64_e1.0_gpu | 8 | 2026-10-25T04:20:24Z |
| final | r_trace_N64_e2.0_gpu | 8 | 2026-10-25T04:20:32Z |
| final | r_prod | 17 | 2026-10-25T04:20:49Z |
| final | r_fwd_smoke_g3d_n16 | 4 | 2026-10-25T04:20:53Z |
| final | r_fwd_g3d_n32_cov | 4 | 2026-10-25T04:20:58Z |
| final | r_fwd_gauss_n64_e1_cov | 7 | 2026-10-25T04:21:05Z |
| final | r_fwd_gauss_n64_e2 | 7 | 2026-10-25T04:21:12Z |
| final | r_fwd_gauss_n64_e2_tol | 7 | 2026-10-25T04:21:19Z |
| final | r_fwd_gauss_n40_e1 | 5 | 2026-10-25T04:21:25Z |
| final | r_fwd_gauss_n80_e2 | 6 | 2026-10-25T04:21:31Z |
| final | r_neutral_gauss_n64_e1_cov | 7 | 2026-10-25T04:21:38Z |
| final | r_neutral_gauss_n64_e2 | 7 | 2026-10-25T04:21:45Z |
| final | r_neutral_gauss_n64_e2_tol | 8 | 2026-10-25T04:21:53Z |
| final | g_runner_a | 16 | 2026-10-25T04:22:12Z |
| final | g_ab | 14 | 2026-10-25T04:22:29Z |

#### Binary md5 (head: `controls/run.env`, identical in every head part: yes; base: `sf41_base_refs/run.env`)

| binary | head md5 | base md5 |
|---|---|---|
| dirichlet_controls_tests | dce34ae54d23dff7a2becd7442fca778 | - |
| dirichlet_long_box_stage_tests | 13b99e8cf39e52134a1066a17cbaa9e0 | - |
| inlet_slab | ded79b8ad5ad019d81e0dc8bea1d980b | aae63d2868be1627e80c16a3a41eb479 |
| interpolation_nonperiodic_x_tests | 7e37f248d7391bf9d23902df1b182240 | - |
| lagrangian_ensemble | 0c8caaf6b31c15fe9abf3558b6a8050d | 59ac27b53610b881b14fe730128080ba |
| lagrangian_medium_ab | f5a80df650ab6d086dfc66a61e8a729c | a53800587a2b5b75e5e289f6f5cf37ee |
| macroflow3d_pipeline | 516d516dc759a12bace29b0d50908e70 | fba69968c801cc651695c80b971da5a0 |
| pollock_nonperiodic_tests | e08de5d9cc2c39050a817a1a44e8b33e | - |
| sf41_cross_representation_tests | d54b7f0c655b9f8eae2fa6b4194bcf3a | - |

`build/binaries.md5` (attempt-2 build part) lists 50 binaries; its md5 for the 9 binaries above equal the `BIN_MD5_*` of the head `run.env`.

Host: `localhost.localdomain`, GPU `Tesla V100-PCIE-32GB`, driver 470.239.06, CUDA (nvidia-smi) 11.4, nvcc 11.4, gcc (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2); `TREE_src_digest=821e4feec2e8895d392c754c049da0a1044172a453097b5fb8f5c099f8168243` (`TREE_git=absent`).
derived clock offset: `run.env` DATE_UTC 2026-10-25T02:16:44Z minus the `nvidia-smi` banner time 2026-10-24 23:16:55 = 3.00 h.
