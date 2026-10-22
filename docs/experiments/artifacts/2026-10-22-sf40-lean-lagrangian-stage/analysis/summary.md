# SF-40 derived tables (all values from `raw/`)

## A / R: `sf40_campaign/bytecmp/results.txt`

Totals: IDENTICAL 1317, DIFF 0, MISSING 0, SKIP 0; last line `BYTECMP_ALL_IDENTICAL`.

| set | cases | IDENTICAL items | DIFF/MISSING |
|---|---:|---:|---:|
| LEAN | 31 | 539 | 0 |
| L6 | 31 | 508 | 0 |
| R (default path, SF-35/36/37) | 20 runs | 269 | 0 |

R items by group: fwd 118, neutral 95, prod32 2, trace 54.
`binary` item: `IDENTICAL binary:md5_differs head=95febc0b40245fac1f4b19af12d569bd base=ac806f1b1d8d5076eac4ad97e6f4ffc7`.

## C: exact bytes and S+ closed form

Source: `sf40_campaign/mem/{d512,d3328_i,d3328_ii}.log`; `C_VERDICT` lines of `mem/results.txt`.

### d512: 512x512x512

n = 134217728, m = face_max = 134479872 (m - n = 262144), levels = 8, n_l (l >= 1) = 16777216,2097152,262144,32768,4096,512,64, sum_{l>=1} n_l = 19173952.
S_scr items: scratch_pcg_d_scalar=8, scratch_pcg_mean_zero=16391, scratch_rhs_mean_zero=16391, scratch_rhs_means=32, scratch_mg_mean_zero_l0=16391, scratch_mg_mean_zero_l1=16391, scratch_mg_mean_zero_l2=16391, scratch_mg_mean_zero_l3=16391, scratch_mg_mean_zero_l4=16391, scratch_mg_mean_zero_l5=16391, scratch_mg_mean_zero_l6=9, scratch_mg_mean_zero_l7=9, scratch_reduce_blocks=2048; S_scr = 133234.
closed form = 8 [7 134217728 + 4 19173952] + 32 (262144) + 133234 = 8138281074; reported S2 exact = 8138281074; difference 0.
S2 = 60.6349 B/cell = 7761.27 MiB (nominal audit level S+ 60.571 B/cell; excess 0.0635 B/cell: rule-7 capacity 32 (m - n)/n = 0.0625, scratch 0.00099).

| phase | exact B | B/cell | MiB | cufft_work B | items (name:bytes) |
|---|---:|---:|---:|---:|---|
| F | 3229616136 | 24.0625 | 3080.00 | 1077936128 | y:1073741824, sf18_spectrum:1077936128, sf18_mode_counter:8, sf18_reduce_blocks:2048 |
| S1 | 5381292032 | 40.0938 | 5132.00 | 2155872256 | lnk:1073741824, coef_logk:1073741824, prefilter_spectrum:1077936128 |
| S2 | 8138281074 | 60.6349 | 7761.27 | 0 | pcg_r:1075838976, pcg_p:1075838976, corrector:1075838976, mg_l*:... (4910630912 bytes), scratch_*: 133234 |
| S3 | 6729240552 | 50.1367 | 6417.50 | 2155872256 | h_tilde:1075838976, coef_logk:1073741824, labels_device:272109544, coef_pot:1073741824, prefilter_spectrum:1077936128 |
| T | 2464157784 | 18.3594 | 2350.00 | 0 | coef_interleaved:2147483648, labels_device:272109544, tracker_seeds:1572864, tracker_planes:56, tracker_times:56, tracker_stats_plane:5767168, tracker_stats_time:5767168, tracker_planes_x23:7340032, tracker_planes_tau:3670016, tracker_planes_I:3670016, tracker_times_x:11010048, tracker_times_I:3670016, tracker_prepare_transient:2097152 |

S2 MG items: mg_l0_x:1075838976, mg_l0_b:1073741824, mg_l0_r:1073741824, mg_l0_coef:1073741824, mg_l1_x:134217728, mg_l1_b:134217728, mg_l1_r:134217728, mg_l1_coef:134217728, mg_l2_x:16777216, mg_l2_b:16777216, mg_l2_r:16777216, mg_l2_coef:16777216, mg_l3_x:2097152, mg_l3_b:2097152, mg_l3_r:2097152, mg_l3_coef:2097152, mg_l4_x:262144, mg_l4_b:262144, mg_l4_r:262144, mg_l4_coef:262144, mg_l5_x:32768, mg_l5_b:32768, mg_l5_r:32768, mg_l5_coef:32768, mg_l6_x:4096, mg_l6_b:4096, mg_l6_r:4096, mg_l6_coef:4096, mg_l7_x:512, mg_l7_b:512, mg_l7_r:512, mg_l7_coef:512

### d3328_i: 3328x384x384

n = 490733568, m = face_max = 492011520 (m - n = 1277952), levels = 7, n_l (l >= 1) = 61341696,7667712,958464,119808,14976,1872, sum_{l>=1} n_l = 70104528.
S_scr items: scratch_pcg_d_scalar=8, scratch_pcg_mean_zero=16391, scratch_rhs_mean_zero=16391, scratch_rhs_means=32, scratch_mg_mean_zero_l0=16391, scratch_mg_mean_zero_l1=16391, scratch_mg_mean_zero_l2=16391, scratch_mg_mean_zero_l3=16391, scratch_mg_mean_zero_l4=16391, scratch_mg_mean_zero_l5=16391, scratch_mg_mean_zero_l6=9, scratch_reduce_blocks=2048; S_scr = 133225.
closed form = 8 [7 490733568 + 4 70104528] + 32 (1277952) + 133225 = 29765452393; reported S2 exact = 29765452393; difference 0.
S2 = 60.6550 B/cell = 28386.55 MiB (nominal audit level S+ 60.571 B/cell; excess 0.0836 B/cell: rule-7 capacity 32 (m - n)/n = 0.0833, scratch 0.00027).

| phase | exact B | B/cell | MiB | cufft_work B | items (name:bytes) |
|---|---:|---:|---:|---:|---|
| F | 11782326280 | 24.0096 | 11236.50 | 3928227840 | y:3925868544, sf18_spectrum:3928227840, sf18_mode_counter:8, sf18_reduce_blocks:2048 |
| S1 | 19636420608 | 40.0144 | 18726.75 | 7856455680 | lnk:3925868544, coef_logk:3925868544, prefilter_spectrum:3928227840 |
| S2 | 29765452393 | 60.6550 | 28386.55 | 0 | pcg_r:3936092160, pcg_p:3936092160, corrector:3936092160, mg_l*:... (17957042688 bytes), scratch_*: 133225 |
| S3 | 23775677416 | 48.4493 | 22674.25 | 7856455680 | h_tilde:3936092160, coef_logk:3925868544, labels_device:203164648, coef_pot:3925868544, prefilter_spectrum:3928227840 |
| T | 8060101832 | 16.4246 | 7686.71 | 0 | coef_interleaved:7851737088, labels_device:203164648, tracker_seeds:240000, tracker_planes:72, tracker_times:8, tracker_stats_plane:880000, tracker_stats_time:880000, tracker_planes_x23:1440000, tracker_planes_tau:720000, tracker_planes_I:720000, tracker_times_x:8, tracker_times_I:8, tracker_prepare_transient:320000 |

S2 MG items: mg_l0_x:3936092160, mg_l0_b:3925868544, mg_l0_r:3925868544, mg_l0_coef:3925868544, mg_l1_x:490733568, mg_l1_b:490733568, mg_l1_r:490733568, mg_l1_coef:490733568, mg_l2_x:61341696, mg_l2_b:61341696, mg_l2_r:61341696, mg_l2_coef:61341696, mg_l3_x:7667712, mg_l3_b:7667712, mg_l3_r:7667712, mg_l3_coef:7667712, mg_l4_x:958464, mg_l4_b:958464, mg_l4_r:958464, mg_l4_coef:958464, mg_l5_x:119808, mg_l5_b:119808, mg_l5_r:119808, mg_l5_coef:119808, mg_l6_x:14976, mg_l6_b:14976, mg_l6_r:14976, mg_l6_coef:14976

### d3328_ii: 3328x384x384

n = 490733568, m = face_max = 492011520 (m - n = 1277952), levels = 7, n_l (l >= 1) = 61341696,7667712,958464,119808,14976,1872, sum_{l>=1} n_l = 70104528.
S_scr items: scratch_pcg_d_scalar=8, scratch_pcg_mean_zero=16391, scratch_rhs_mean_zero=16391, scratch_rhs_means=32, scratch_mg_mean_zero_l0=16391, scratch_mg_mean_zero_l1=16391, scratch_mg_mean_zero_l2=16391, scratch_mg_mean_zero_l3=16391, scratch_mg_mean_zero_l4=16391, scratch_mg_mean_zero_l5=16391, scratch_mg_mean_zero_l6=9, scratch_reduce_blocks=2048; S_scr = 133225.
closed form = 8 [7 490733568 + 4 70104528] + 32 (1277952) + 133225 = 29765452393; reported S2 exact = 29765452393; difference 0.
S2 = 60.6550 B/cell = 28386.55 MiB (nominal audit level S+ 60.571 B/cell; excess 0.0836 B/cell: rule-7 capacity 32 (m - n)/n = 0.0833, scratch 0.00027).

| phase | exact B | B/cell | MiB | cufft_work B | items (name:bytes) |
|---|---:|---:|---:|---:|---|
| F | 11782326280 | 24.0096 | 11236.50 | 3928227840 | y:3925868544, sf18_spectrum:3928227840, sf18_mode_counter:8, sf18_reduce_blocks:2048 |
| S1 | 19636420608 | 40.0144 | 18726.75 | 7856455680 | lnk:3925868544, coef_logk:3925868544, prefilter_spectrum:3928227840 |
| S2 | 29765452393 | 60.6550 | 28386.55 | 0 | pcg_r:3936092160, pcg_p:3936092160, corrector:3936092160, mg_l*:... (17957042688 bytes), scratch_*: 133225 |
| S3 | 23572512768 | 48.0353 | 22480.50 | 7856455680 | h_tilde:3936092160, coef_logk:3925868544, coef_pot:3925868544, prefilter_spectrum:3928227840 |
| T | 7856377176 | 16.0095 | 7492.43 | 0 | coef_interleaved:7851737088, tracker_seeds:240000, tracker_planes:64, tracker_times:8, tracker_stats_plane:880000, tracker_stats_time:880000, tracker_planes_x23:1280000, tracker_planes_tau:640000, tracker_planes_I:640000, tracker_times_x:8, tracker_times_I:8, tracker_prepare_transient:80000 |

S2 MG items: mg_l0_x:3936092160, mg_l0_b:3925868544, mg_l0_r:3925868544, mg_l0_coef:3925868544, mg_l1_x:490733568, mg_l1_b:490733568, mg_l1_r:490733568, mg_l1_coef:490733568, mg_l2_x:61341696, mg_l2_b:61341696, mg_l2_r:61341696, mg_l2_coef:61341696, mg_l3_x:7667712, mg_l3_b:7667712, mg_l3_r:7667712, mg_l3_coef:7667712, mg_l4_x:958464, mg_l4_b:958464, mg_l4_r:958464, mg_l4_coef:958464, mg_l5_x:119808, mg_l5_b:119808, mg_l5_r:119808, mg_l5_coef:119808, mg_l6_x:14976, mg_l6_b:14976, mg_l6_r:14976, mg_l6_coef:14976

cuFFT work areas (`controls/long_box_cufft3328.log`): `CUFFT_SIZE nx=3328 ny=384 nz=384 gen_z2d=3928227840 prefilter_d2z=3928227840 prefilter_z2d=3928227840 levels=7`

## D / E: `mem/results.txt`, `mem/*.log`, `mem/*.time`

| run | exit | PID samples | PID peak MiB | device-wide peak MiB | E_max MiB | C MiB | peak - E_max - C MiB | host max RSS kB | wall (time -v) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| d512 | 0 | 1409 | 8249 | 8255 | 7761.270 | 487 | 0.730 | 5867076 | 4:44.61 |
| d3328_i | 0 | 847 | 28885 | 28895 | 28386.547 | 487 | 11.453 | 19802104 | 2:53.40 |
| d3328_ii | 0 | 865 | 28885 | 28895 | 28386.547 | 487 | 11.453 | 19802128 | 2:57.13 |

Host RSS per cell (RSS kB * 1024 / n); the context probe (`mem/probe1.time`) has RSS 605832 kB:

| run | n | RSS B/cell | (RSS - probe RSS) B/cell |
|---|---:|---:|---:|
| d512 | 134217728 | 44.762 | 40.140 |
| d512_bhatdev | 134217728 | 36.840 | 32.218 |
| d3328_i | 490733568 | 41.320 | 40.056 |
| d3328_ii | 490733568 | 41.321 | 40.056 |

S vs S+ at 512^3 (`mem/d512.log` = S+ mapped b_hat, `mem/d512_bhatdev.log` = S, device b_hat):

| variant | stage built in s | S2 (lean SF-19) s | S2 exact B |
|---|---:|---:|---:|
| S+ (mapped) | 42.345 | 28.640221 | 8138281074 |
| S (device) | 39.119 | 26.326715 | 9212022898 |

S2 time difference S+ - S = 2.313506 s; stage time difference = 3.226 s; derived scaling of the S2 difference by cells to 3328x384x384 (x3.6562) = 8.46 s (derived, not measured).

### d512 log excerpts

```
SETUP STAGE field=gaussian eps=2 N=512 status=ok
SETUP   SF-19: mg_levels=8 converged=1 G=(4.949880468826e-01, 1.671592385253e-02, -2.136218098537e-02) mean_flux=(1.000e+00, 1.377e-17, -1.973e-17) div_max=7.110e-09 div_rms=1.409e-10
SETUP   SF-19 corrector x: status=converged its=20 rel_residual=2.422e-12 final_residual=2.801e-06
SETUP   SF-19 corrector y: status=converged its=20 rel_residual=3.227e-12 final_residual=3.635e-06
SETUP   SF-19 corrector z: status=converged its=20 rel_residual=2.368e-12 final_residual=2.663e-06
SETUP   inlet: vmin=3.314756e-02 mean(U plane 0)=0.999999999999980 (mean-1=-1.976e-14) Q0-1=-1.110e-16 v_rms=n/a (lean stage: vertex vD not built)
SETUP stage eps=2 built in 42.345s (lean stage, Grid3D 512x512x512, bhat=mapped)
BACKFLOW field=gaussian eps=2 N=512 inlet_vmin=3.314756e-02 labels_available=1 seed_backflow=0 backflow_enc_plane=0 backflow_enc_time=0 (inlet backflow is recorded, not an exit condition)
TSTATUS pass=plane hist=65536,0,0,0,0,0,0 landings_discarded=0 backflow_enc=0 nfev=30141104694
TSTATUS pass=time hist=65536,0,0,0,0,0,0 landings_discarded=0 backflow_enc=0 nfev=24063432822
TIMING prepare=0.046s stage=42.345s track=240.205s write=0.031s total=282.629s
STATUS ok
```
S2 (lean SF-19) seconds = 28.640; stage phase seconds: F=0.878, S1=4.885, S2=28.640, S3=4.441, T=0.046.

TPLANE lines: 7, distinct n_ok values: 65536.

### d3328_i log excerpts

```
SETUP STAGE field=gaussian eps=1.5 N=3328 status=ok
SETUP   SF-19: mg_levels=7 converged=1 G=(6.910724104955e-01, -2.384750082888e-04, -2.758685846119e-03) mean_flux=(1.000e+00, -8.155e-19, -2.595e-18) div_max=1.835e-08 div_rms=3.228e-11
SETUP   SF-19 corrector x: status=converged its=30 rel_residual=8.367e-14 final_residual=9.096e-07
SETUP   SF-19 corrector y: status=converged its=30 rel_residual=6.495e-14 final_residual=7.265e-07
SETUP   SF-19 corrector z: status=converged its=30 rel_residual=7.528e-14 final_residual=8.108e-07
SETUP   inlet: vmin=5.043988e-03 mean(U plane 0)=1.000000000000015 (mean-1=1.465e-14) Q0-1=2.220e-16 v_rms=n/a (lean stage: vertex vD not built)
SETUP stage eps=1.5 built in 161.794s (lean stage, Grid3D 3328x384x384, bhat=mapped)
BACKFLOW field=gaussian eps=1.5 N=3328 inlet_vmin=5.043988e-03 labels_available=1 seed_backflow=0 backflow_enc_plane=45 backflow_enc_time=0 (inlet backflow is recorded, not an exit condition)
TSTATUS pass=plane hist=10000,0,0,0,0,0,0 landings_discarded=0 backflow_enc=45 nfev=1814622238
TSTATUS pass=time hist=10000,0,0,0,0,0,0 landings_discarded=0 backflow_enc=0 nfev=0
TIMING prepare=0.009s stage=161.794s track=9.768s total=171.572s
STATUS ok
```
S2 (lean SF-19) seconds = 118.559; stage phase seconds: F=3.575, S1=17.745, S2=118.559, S3=18.372, T=0.009.

TPLANE lines: 9, distinct n_ok values: 10000.

### d3328_ii log excerpts

```
SETUP STAGE field=gaussian eps=2 N=3328 status=inlet_backflow
SETUP   SF-19: mg_levels=7 converged=1 G=(5.289836709515e-01, -2.414639831579e-03, -6.931413789823e-03) mean_flux=(1.000e+00, 1.038e-18, 1.035e-17) div_max=3.683e-06 div_rms=6.761e-09
SETUP   SF-19 corrector x: status=converged its=30 rel_residual=8.941e-12 final_residual=2.829e-04
SETUP   SF-19 corrector y: status=converged its=30 rel_residual=8.486e-12 final_residual=2.546e-04
SETUP   SF-19 corrector z: status=converged its=30 rel_residual=4.614e-12 final_residual=1.558e-04
SETUP   inlet: vmin=-4.297129e-02 mean(U plane 0)=0.999999999998595 (mean-1=-1.405e-12) Q0-1=nan v_rms=n/a (lean stage: vertex vD not built)
SETUP stage eps=2 built in 164.740s (lean stage, Grid3D 3328x384x384, bhat=mapped)
BACKFLOW field=gaussian eps=2 N=3328 inlet_vmin=-4.297129e-02 labels_available=0 seed_backflow=3 backflow_enc_plane=1014 backflow_enc_time=0 (inlet backflow is recorded, not an exit condition)
TSTATUS pass=plane hist=9997,3,0,0,0,0,0 landings_discarded=0 backflow_enc=1014 nfev=1956460652
TSTATUS pass=time hist=10000,0,0,0,0,0,0 landings_discarded=0 backflow_enc=0 nfev=0
TIMING prepare=0.008s stage=164.739s track=10.364s total=175.113s
STATUS ok
```
S2 (lean SF-19) seconds = 119.513; stage phase seconds: F=3.828, S1=17.660, S2=119.513, S3=21.322, T=0.008.

TPLANE lines: 8, distinct n_ok values: 9997.

Recorded lines of `mem/results.txt`:

```
PROBE run=probe1 exit=0 pid=162671 pid_peak_mib=487 device_used_bytes=514850816 driver=11040 runtime=11040 status=OK -
PROBE run=probe2 exit=0 pid=162737 pid_peak_mib=487 device_used_bytes=514850816 driver=11040 runtime=11040 status=OK -
PROBE run=probe3 exit=0 pid=162949 pid_peak_mib=487 device_used_bytes=514850816 driver=11040 runtime=11040 status=OK -
C_CONTEXT mib=487 (minimum PID peak over 3 evaluable of 3 probes)
RECORDED warm_probe pid_peak_mib=489 device_used_bytes=516947968 status=OK
D_VERDICT run=d512 pid_peak_mib=8249 E_max_mib=7761.270 C_mib=487 limit_mib=8481.108 exit=0 PASS
D_ALT run=d512 limit_alt_mib=8495.718 ((E_max + C) * 1.03; reported, not the verdict) would_be=PASS
D3328_VERDICT run=d3328_i exit=0 pid_peak_mib=28885 limit_mib=30000 exit_class=ok PASS
E_VERDICT run=d3328_i correctors=converged,30,8.367e-14 converged,30,6.495e-14 converged,30,7.528e-14 converged=1 levels=7 coarsest=52x6x6 PASS
D3328_VERDICT run=d3328_ii exit=0 pid_peak_mib=28885 limit_mib=30000 exit_class=ok PASS
E_VERDICT run=d3328_ii correctors=converged,30,8.941e-12 converged,30,8.486e-12 converged,30,4.614e-12 converged=1 levels=7 coarsest=52x6x6 PASS
RECORDED E_d512 correctors=x,converged,20,2.422e-12 y,converged,20,3.227e-12 z,converged,20,2.368e-12 converged=1 mg_levels=8
RECORDED run=d512 exit=0 host_max_rss_kb=5867076 wall_s=284.61 free_b_before=[Mem: 134590115840 3198115840 26595344384 214081536 104796655616 130536898560]
RECORDED run=d512 phase_seconds=F:0.877824,S1:4.885465,S2:28.640221,S3:4.440621,T:0.046457 exact_bytes=F:3229616136,S1:5381292032,S2:8138281074,S3:6729240552,T:2464157784
RECORDED run=d512_bhatdev exit=0 host_max_rss_kb=4828676 wall_s=280.99 free_b_before=[Mem: 134590115840 3194224640 26565799936 214073344 104830091264 130540818432]
RECORDED run=d512_bhatdev phase_seconds=F:0.916373,S1:4.145161,S2:26.326715,S3:4.423882,T:0.045935 exact_bytes=F:3229616136,S1:5381292032,S2:9212022898,S3:6729240552,T:2464157784
RECORDED run=d3328_i exit=0 host_max_rss_kb=19802104 wall_s=173.4 free_b_before=[Mem: 134590115840 3196506112 26530435072 214073344 104863174656 130538504192]
RECORDED run=d3328_i phase_seconds=F:3.575474,S1:17.744869,S2:118.559475,S3:18.372186,T:0.00885 exact_bytes=F:11782326280,S1:19636420608,S2:29765452393,S3:23775677416,T:8060101832
RECORDED run=d3328_ii exit=0 host_max_rss_kb=19802128 wall_s=177.13 free_b_before=[Mem: 134590115840 3195129856 26531270656 214073344 104863715328 130539909120]
RECORDED run=d3328_ii phase_seconds=F:3.828071,S1:17.659594,S2:119.512592,S3:21.322091,T:0.007708 exact_bytes=F:11782326280,S1:19636420608,S2:29765452393,S3:23572512768,T:7856377176
RECORDED max_cells_exit0 n=490733568 run=d3328_i
RECORDED S_vs_S+ run=512 S2_seconds_S+(mapped)=28.640221 S2_seconds_S(device)=26.326715 wall_s_S+=284.61 wall_s_S=280.99
RECORDED versions driver=11040 runtime=11040
```

## F: `controls/long_box_full.log`

```
CHECK F1_64x16x16_M16_Keff_equals_I PASS  max 0.000e+00
CHECK F1_64x16x16_M16_mean_flux_equals_e1 PASS  max 0.000e+00
CHECK F1_64x16x16_M16_all_particles_ok PASS
CHECK F1_64x16x16_M16_transverse_displacement_le_1e-12 PASS  max 0.000e+00 over 2048 records
CHECK F1_64x16x16_M16_tau_equals_X_rel_le_1e-12 PASS  max rel 0.000e+00
CHECK F1_64x16x16_M16_covectors_equal_initial_le_1e-12 PASS  max 0.000e+00
CHECK F1_64x16x16_M16_abs_I_le_1e-14 PASS  max 0.000e+00
  F1_64x16x16_M16 INFO max |x1(t) - (X0 + t)| at snapshots 0.000e+00  [0.1 s]
CHECK F1_128x16x16_M16_Keff_equals_I PASS  max 0.000e+00
CHECK F1_128x16x16_M16_mean_flux_equals_e1 PASS  max 0.000e+00
CHECK F1_128x16x16_M16_all_particles_ok PASS
CHECK F1_128x16x16_M16_transverse_displacement_le_1e-12 PASS  max 0.000e+00 over 2048 records
CHECK F1_128x16x16_M16_tau_equals_X_rel_le_1e-12 PASS  max rel 0.000e+00
CHECK F1_128x16x16_M16_covectors_equal_initial_le_1e-12 PASS  max 0.000e+00
CHECK F1_128x16x16_M16_abs_I_le_1e-14 PASS  max 0.000e+00
  F1_128x16x16_M16 INFO max |x1(t) - (X0 + t)| at snapshots 0.000e+00  [0.1 s]
CHECK F2_64x32x32_M16_all_particles_ok PASS
CHECK F2_64x32x32_M16_transverse_positions_constant_le_1e-12 PASS  max 4.614e-16
CHECK F2_64x32x32_M16_covectors_constant_le_1e-12 PASS  max 2.985e-14
CHECK F2_64x32x32_M16_tau_vs_quadrature_rel_le_1e-9 PASS  max rel 4.771e-10
[RECORD] F2_64x32x32_M16 I vs independent quadrature (no threshold): max abs err 2.046e-06, max |I_ref| 3.208e-03 over 2048 records  [1.8 s]
CHECK F2_128x32x32_M16_all_particles_ok PASS
CHECK F2_128x32x32_M16_transverse_positions_constant_le_1e-12 PASS  max 4.441e-16
CHECK F2_128x32x32_M16_covectors_constant_le_1e-12 PASS  max 2.962e-14
CHECK F2_128x32x32_M16_tau_vs_quadrature_rel_le_1e-9 PASS  max rel 4.771e-10
[RECORD] F2_128x32x32_M16 I vs independent quadrature (no threshold): max abs err 2.046e-06, max |I_ref| 3.208e-03 over 2048 records  [1.8 s]
CHECK F3_128x32x32_4096pts_shift_plus_1_in_x2_x3_le_1e-14 PASS  max abs 6.661e-15 max rel 4.256e-15 violating 0
  F3_128x32x32_4096pts [1.9 s]
CHECK F4_64cube_to_128x64x64_M64_stages_ready PASS
CHECK F4_64cube_to_128x64x64_M64_G_normwise_rel_le_1e-8 PASS  2.669e-16 (entrywise max 1.273e-15)
CHECK F4_64cube_to_128x64x64_M64_Keff_normwise_rel_le_1e-8 PASS  7.181e-16 (entrywise max 1.542e-15)
  F4_64cube_to_128x64x64_M64 plane 0.25: sigma_T^2 8.463948098205e-02 / 8.463948098205e-02  mean_tau 2.498728640224e-01 / 2.498728640224e-01  var_tau 3.100101238752e-02 / 3.100101238752e-02  (box / cube)
  F4_64cube_to_128x64x64_M64 plane 0.5: sigma_T^2 8.727087390719e-02 / 8.727087390719e-02  mean_tau 4.997599567382e-01 / 4.997599567382e-01  var_tau 1.330273516793e-01 / 1.330273516793e-01  (box / cube)
  F4_64cube_to_128x64x64_M64 plane 1: sigma_T^2 7.928835930526e-02 / 7.928835930526e-02  mean_tau 9.997755995826e-01 / 9.997755995826e-01  var_tau 4.834125739508e-01 / 4.834125739508e-01  (box / cube)
  F4_64cube_to_128x64x64_M64 plane 2: sigma_T^2 8.040569730849e-02 / 8.040569730849e-02  mean_tau 1.999898627226e+00 / 1.999898627226e+00  var_tau 1.885368918056e+00 / 1.885368918056e+00  (box / cube)
  F4_64cube_to_128x64x64_M64 plane 4: sigma_T^2 8.426914518650e-02 / 8.426914518650e-02  mean_tau 4.001519761210e+00 / 4.001519761210e+00  var_tau 7.069937830803e+00 / 7.069937830803e+00  (box / cube)
CHECK F4_64cube_to_128x64x64_M64_plane_stats_rel_le_1e-8 PASS  max rel 1.649e-15
  F4_64cube_to_128x64x64_M64 [4.6 s]
```

## L6: `sf40_campaign/l6/l6_timings.tsv`

```
variant	rc	wall_s	plane_kernel_s	time_kernel_s	single_kernel_s	download_s	moments_s	kernel_total_s	nfev_plane	nfev_time	nfev_total	speedup_vs_default
default	0	168	76.884415211	72.344823551		0.020349192	0.045646538	149.22923876200002	15073317414	12093882732	27167200146	1.0000
fused	0	118	53.24040734	45.25902733		0.020442439	0.045751417	98.49943467	15073317414	12093882732	27167200146	1.5150
single	0	161			142.638341413	0.020615035	0.04555116	142.638341413	15073317414	12093882732	27167200146	1.0462
both	0	116			96.337586389	0.020606309	0.045645347	96.337586389	15073317414	12093882732	27167200146	1.5490
RECORDED L6_SPEEDUP variant=fused kernel_total_s=98.49943467 default_kernel_total_s=149.22923876200002 speedup=1.5150 launches=2
RECORDED L6_SPEEDUP variant=single kernel_total_s=142.638341413 default_kernel_total_s=149.22923876200002 speedup=1.0462 launches=1
RECORDED L6_SPEEDUP variant=both kernel_total_s=96.337586389 default_kernel_total_s=149.22923876200002 speedup=1.5490 launches=1
```
Derived per-pass ratios default / fused: plane 1.4441, time 1.5985 (plane 76.884415211 / 53.240407340 s, time 72.344823551 / 45.259027330 s).

## G: `par2/results.txt`, `sf40_base_refs/results.txt`

PID peaks MiB (head job): base_par2_a=22633, head_par2=19567; base references: par2_a=22633, par2_b=22633.
Config grid (apps/config_pipeline_par2.yaml at H1): 2048 x 256 x 256 = 134217728 cells; 24 B/cell x cells = 3221225472 B = 3072.00 MiB.

base - head PID peak = 22633 - 19567 = 3066 MiB (expected 24 B/cell: 3072.00 MiB; difference -6.00 MiB).

## R / G byte items

SF-27 trio (`final/results.txt`): config_pspta_small 111, config_streamfunctions_continuation 31, config_streamfunctions_homogeneous 30; last verdict line `TRIO_ALL_IDENTICAL`.
Par2 head vs base (`par2/results.txt`): IDENTICAL file items 407, other lines starting DIFF/MISSING: 0; verdict `PAR2_ALL_IDENTICAL`.

## R: `final/ctest.log`

```
100% tests passed, 0 tests failed out of 43
Total Test time (real) = 2968.36 sec
```
Tests listed: 43. SF-37 wall (spec): 2901.55 s.

