Settings echoed by each log (`RESOLVED` line), e.g. E4: `RESOLVED grid=6656x256x256 L1=26 lambda=0.25 planes(auto)=1.5,1.75,2.25,3.25,5.25,9.25,17.25,24.75 times(auto)=n*lambda/2 up to (L1-10 lambda)/qbar, qbar measured per realization medium=b x0=1.25`

#### Solve, memory sampler, wall (verdict `E_RUN` fields: solve mode, PID peak against 30000 MiB; sampler `pid_peak.py`)

| run | grid | eps | ell | solve | converged | iterations | rel | balance | max_div | fallback | PID peak MiB | device-wide peak MiB | samples | sampler status | wall s | host max RSS kB | stage s | E_RUN state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 2048x256x256 | 1 | 0.0390625 | primary | 1 | 20 | 3.409160e-11 | 2.348053e-11 | 3.922635e-09 | 0 | 8251 | 8255 | 160 | OK | 33.69 | 4858940 | 13.870 | PASS |
| E2 | 3328x384x384 | 1.5 | 0.041666666666666664 | primary | 1 | 40 | 1.725197e-11 | 5.513091e-12 | 1.522198e-08 | 0 | 28855 | 28859 | 536 | OK | 110.28 | 16021444 | 67.512 | PASS |
| E3 | 3328x384x384 | 2 | 0.08333333333333333 | primary | 1 | 50 | 1.607008e-12 | 6.261729e-12 | 1.112806e-07 | 0 | 28855 | 28859 | 627 | OK | 129.16 | 16021512 | 74.824 | PASS |
| E4 | 6656x256x256 | 2 | 0.25 | primary | 1 | 90 | 1.263955e-11 | 4.164330e-11 | 6.374646e-08 | 0 | 25701 | 25705 | 703 | OK | 143.81 | 14314552 | 90.314 | PASS |

#### Per-phase exact bytes (`MEM_PHASE`; MiB = bytes / 1048576, derived; B/cell = S2 / cells, derived; context+overhead = PID peak - S2 MiB, derived)

| run | max phase | S2 exact B | S2 MiB | S2 B/cell | phases (MiB): F, S1, S2, V, H, S3, T, B | PID peak - S2 MiB | headroom to 30000 MiB | every phase <= S2 |
|---|---|---|---|---|---|---|---|---|
| E1 | S2 | 8129757192 | 7753.141 | 60.571 | F=3074.002,S1=0.000,S2=7753.141,V=5129.508,H=1024.000,S3=6178.062,T=2157.136,B=4100.500 | 497.859 | 21749.0 | yes |
| E2 | S2 | 29724424712 | 28347.420 | 60.571 | F=11236.502,S1=0.000,S2=28347.420,V=18742.883,H=3744.000,S3=22545.096,T=7585.045,B=14986.875 | 507.580 | 1145.0 | yes |
| E3 | S2 | 29724424712 | 28347.420 | 60.571 | F=11236.502,S1=0.000,S2=28347.420,V=18742.883,H=3744.000,S3=22545.096,T=7531.333,B=14986.875 | 507.580 | 1145.0 | yes |
| E4 | S2 | 26421710856 | 25197.707 | 60.571 | F=9986.002,S1=0.000,S2=25197.707,V=16667.508,H=3328.000,S3=20056.098,T=6703.183,B=13325.500 | 503.293 | 4299.0 | yes |

#### Closed-form memory model (`controls` MEMMODEL_a / MEMMODEL_b, printed before the runs) against the measured exact bytes (derived comparison)

| run | model | F | S2 | V | H | S3 |
|---|---|---|---|---|---|---|
| E2 | MEMMODEL_a | F equal | S2 equal | V equal | H equal | S3 equal |
| E3 | MEMMODEL_a | F equal | S2 equal | V equal | H equal | S3 equal |
| E4 | MEMMODEL_b | F equal | S2 equal | V equal | H equal | S3 equal |

#### Tracking, exits and non-ok (`STATUS`, `TEXIT`, `TSTATUS`, `POLLOCK`, `BACKFLOW` lines of the logs)

| run | tracker non_ok plane | non_ok time | seed_backflow | exit_inlet plane | exit_outlet plane | exit_inlet time | exit_outlet time | TSTATUS time hist | Pollock exit_inlet plane | Pollock exit_outlet plane | Pollock exit_inlet time | Pollock exit_outlet time | Pollock weighted_not_active | inlet_vmin | backflow_enc plane | backflow_enc time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 0 | 0 | 0 | 0 | 0 | 0 | 3695 | 6305,0,0,0,0,0,0 | 0 | 0 | 0 | 3560 | 3560 | 3.441734e-02 | 0 | 0 |
| E2 | 0 | 0 | 0 | 0 | 0 | 0 | 4142 | 5858,0,0,0,0,0,0 | 0 | 0 | 0 | 4157 | 4157 | 5.287898e-03 | 78 | 72 |
| E3 | 0 | 0 | 0 | 0 | 0 | 0 | 3957 | 6043,0,0,0,0,0,0 | 0 | 0 | 0 | 3961 | 3961 | 2.267710e-03 | 843 | 674 |
| E4 | 0 | 0 | 0 | 0 | 0 | 0 | 1910 | 8090,0,0,0,0,0,0 | 0 | 0 | 0 | 1896 | 1896 | 1.023875e-02 | 0 | 0 |

#### `flow_stats.json` values (`FLOWSTATS` line; `theta_2`, `theta_3` = tilt of the interior mean face-flux vector)

| run | qbar | x_faces_nonpositive | speed/qbar percentiles (0, 0.1, 1, 50 %) | interior planes | mean_u | mean_v | mean_w | theta_2 | theta_3 |
|---|---|---|---|---|---|---|---|---|---|
| E1 | 1.181348938810e+00 | 0/134283264 | 1.818938e-02,7.232117e-02,1.337163e-01,8.356550e-01 | 50..1998 | 1.181348938834e+00 | -1.135347269871e-03 | 2.249141258075e-03 | -9.610597538112e-04 | 1.903873162811e-03 |
| E2 | 1.449649782898e+00 | 2230/490881024 | 1.995558e-03,1.497091e-02,3.909465e-02,6.689606e-01 | 80..3248 | 1.449649782893e+00 | -1.365187113382e-03 | 6.605289720408e-03 | -9.417355322050e-04 | 4.556441208865e-03 |
| E3 | 1.899552783704e+00 | 173142/490881024 | 3.666271e-04,2.540001e-03,9.565589e-03,4.944097e-01 | 160..3168 | 1.899552783710e+00 | 6.483659625399e-03 | 2.938580790799e-02 | 3.413242581245e-03 | 1.546862205015e-02 |
| E4 | 1.480233352885e+00 | 0/436273152 | 4.282347e-04,1.667436e-03,7.329138e-03,5.343201e-01 | 320..6336 | 1.480233352925e+00 | -4.711448413967e-02 | -4.585585028417e-02 | -3.181835032224e-02 | -3.096889389965e-02 |

Mapped-host re-run: files with `mapped` in the E part directory: none; `E_RUN` lines carry `(device_head)` only.

```
E_RUN run=E1 exit=0 solve=primary rel=3.409160e-11 balance=2.370176e-11 pid_peak_mib=8251 limit_mib=30000 mem=OK phases_le_S2=1 state=PASS
E_VERDICT id=E1 grid=2048x256x256 PASS (device_head)
E_RUN run=E2 exit=0 solve=primary rel=1.725197e-11 balance=5.518605e-12 pid_peak_mib=28855 limit_mib=30000 mem=OK phases_le_S2=1 state=PASS
E_VERDICT id=E2 grid=3328x384x384 PASS (device_head)
E_RUN run=E3 exit=0 solve=primary rel=1.607008e-12 balance=6.317254e-12 pid_peak_mib=28855 limit_mib=30000 mem=OK phases_le_S2=1 state=PASS
E_VERDICT id=E3 grid=3328x384x384 PASS (device_head)
E_RUN run=E4 exit=0 solve=primary rel=1.263955e-11 balance=4.188526e-11 pid_peak_mib=25701 limit_mib=30000 mem=OK phases_le_S2=1 state=PASS
E_VERDICT id=E4 grid=6656x256x256 PASS (device_head)
E_VERDICT ALL PASS (E1=PASS,E2=PASS,E3=PASS,E4=PASS)
```
