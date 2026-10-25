Compared by `d_verdict.py` from the `D.log` lines; campaign lines copied from `results.txt`:

```
D_VERDICT run=D exit=0 sizes=[128x32x32:head=1,bsolver=1,lines_equal=1,mapped=1; 512x64x64:head=1,bsolver=1,lines_equal=1,mapped=1; 3328x256x256:head=1,bsolver=1,lines_equal=1,mapped=na] PASS
```

#### `B_SOLVER` fields (reference `DirichletMedium`; the lean stage line is string-equal: column 2)

| size | lean line equal | iterations | initial | final | rel | qbar | Q_in | Q_out | balance | max_div |
|---|---|---|---|---|---|---|---|---|---|---|
| 128x32x32 | yes | 30 | 1.434840e+05 | 6.496539e-08 | 4.527710e-13 | 1.756410271345e+00 | 1.756410271345e+00 | 1.756410271344e+00 | 6.552326e-13 | 1.888051e-10 |
| 512x64x64 | yes | 40 | 9.307687e+05 | 1.604320e-05 | 1.723651e-11 | 1.853115550126e+00 | 1.853115550126e+00 | 1.853115550200e+00 | 4.027468e-11 | 1.336864e-08 |
| 3328x256x256 | yes | 50 | 6.724494e+06 | 6.526221e-04 | 9.705149e-11 | 1.002469787644e+00 | 1.002469787644e+00 | 1.002469787885e+00 | 2.404573e-10 | 1.034116e-07 |

`D_C0_COMPARE size=3328x256x256 identical_fields=11/11 differing=[-] (environment control vs SF-38 C0 line)`

#### Lean-stage `MEM_PHASE` exact bytes in the D process (order of the block: 128 device, 128 mapped, 512 device, 512 mapped, 3328 device; B/cell = S2 / cells, derived)

| size | head | cells | S2 exact B | S2 MiB | S2 B/cell | max phase | phases (exact bytes) |
|---|---|---|---|---|---|---|---|
| 128x32x32 | device head | 131072 | 7938056 | 7.570 | 60.563 | S2 | F=0,S1=0,S2=7938056,V=5337088,H=1048576 |
| 128x32x32 | mapped-host head | 131072 | 6889480 | 6.570 | 52.563 | S2 | F=0,S1=0,S2=6889480,V=5337088,H=1048576 |
| 512x64x64 | device head | 2097152 | 127025160 | 121.141 | 60.570 | S2 | F=0,S1=0,S2=127025160,V=84516864,H=16777216 |
| 512x64x64 | mapped-host head | 2097152 | 110247944 | 105.141 | 52.570 | S2 | F=0,S1=0,S2=110247944,V=84516864,H=16777216 |
| 3328x256x256 | device head | 218103808 | 13210855432 | 12598.854 | 60.571 | S2 | F=0,S1=0,S2=13210855432,V=8739364864,H=1744830464 |

#### `D.log` program checks: 40 `CHECK` lines, PASS 40, other 0; `D_CHECK` lines and the closed-form S2 checks

```
DIAG D_128x32x32_device_vs_ref sizes=131072/131072(expected 131072) differing=0 first=-1 (i=-1 j=-1 k=-1) maxabs=0.000e+00
CHECK D_128x32x32_S2_closed_form PASS  S2=7938056 closed=7938056 (60.563 B/cell, scratch=8)
DIAG D_128x32x32_mapped_vs_ref sizes=131072/131072(expected 131072) differing=0 first=-1 (i=-1 j=-1 k=-1) maxabs=0.000e+00
CHECK D_128x32x32_mapped_S2_closed_form PASS  S2=6889480 closed=6889480 (52.563 B/cell, scratch=8)
D_CHECK size=128x32x32 head_identical=1 bsolver_identical=1 mapped_identical=1
DIAG D_512x64x64_device_vs_ref sizes=2097152/2097152(expected 2097152) differing=0 first=-1 (i=-1 j=-1 k=-1) maxabs=0.000e+00
CHECK D_512x64x64_S2_closed_form PASS  S2=127025160 closed=127025160 (60.570 B/cell, scratch=8)
DIAG D_512x64x64_mapped_vs_ref sizes=2097152/2097152(expected 2097152) differing=0 first=-1 (i=-1 j=-1 k=-1) maxabs=0.000e+00
CHECK D_512x64x64_mapped_S2_closed_form PASS  S2=110247944 closed=110247944 (52.570 B/cell, scratch=8)
D_CHECK size=512x64x64 head_identical=1 bsolver_identical=1 mapped_identical=1
DIAG D_3328x256x256_device_vs_ref sizes=218103808/218103808(expected 218103808) differing=0 first=-1 (i=-1 j=-1 k=-1) maxabs=0.000e+00
CHECK D_3328x256x256_S2_closed_form PASS  S2=13210855432 closed=13210855432 (60.571 B/cell, scratch=8)
D_CHECK size=3328x256x256 head_identical=1 bsolver_identical=1 mapped_identical=na
```

`D_MEM run=D pid_peak_mib=19755 dev_peak_mib=19759 samples=631 sampler=OK reasons=- (recorded, not gated)`
`run=D exit=0 wall_s=128.26 host_max_rss_kb=10873636 free_b_before=[Mem: 134590115840 3132481536 22223757312 382025728 109233876992 130434596864]`
`run=D pid_peak_mib=19755 dev_peak_mib=19759 samples=631 max_exact_mib=12598.854 tree_pid=144387 status=OK reasons=-`
