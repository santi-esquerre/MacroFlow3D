# SF-36 controls (A, B, C, D, E1/E2, I at the campaign grids)

Run environment (`run.env`): COMMIT=e81b11833412149027643293d7e11ff989643189; HOST=localhost.localdomain; DATE_UTC=2026-10-17T22:33:46Z; BUILD=/home/sesquerre/MacroFlow3D-SF-36/build/v100-release; PART=controls; DRYRUN=0; BASE_REFS=/home/sesquerre/sf36_base_refs; CUDA_VISIBLE_DEVICES=0; GPU=Tesla V100-PCIE-32GB

## Exit codes

| run | exit code |
|---|---|
| controls_full | 0 |
| controls_diag | 0 |
| gpu_host_full | 0 |

## lagrangian_controls_tests --full (`controls_full.log`)

PASS 31, FAIL 0, RECORD 15.

Final lines: `SUMMARY: 31 checks, 0 failed -> ALL PASS  (101.2 s)`

| kind | check / detail |
|---|---|
| PASS | A1 k=1 N=64: stage flow ready  ok |
| PASS | A1 k=1 N=64: all particles ok in both passes  plane ok 64/64, time ok 64 |
| PASS | A1 k=1 N=64: transverse displacement (planes+snapshots) <= 1e-12  max 0.000e+00 over 640 records |
| PASS | A1 k=1 N=64: tau at plane X equals X - X0, relative <= 1e-12  max rel 0.000e+00 |
| PASS | A1 k=1 N=64: covectors equal initial values <= 1e-12  max 0.000e+00 |
| PASS | A1 k=1 N=64: /I/ <= 1e-14  max 0.000e+00 |
| PASS | A2 layered N=64: stage flow ready  ok |
| PASS | A2 layered N=64: all particles ok in both passes |
| PASS | A2 layered N=64: transverse positions constant <= 1e-12  max 9.714e-17 (max /g2/,/g3/ of the spline flow on the lines: 1.133e-15) |
| PASS | A2 layered N=64: covectors constant <= 1e-12  max 1.195e-13 |
| PASS | A2 layered N=64: tau at planes vs independent quadrature <= 1e-9 relative  max rel 5.649e-11 (abs 1.412e-11) |
| RECORD | A2 layered N=64: I at planes+snapshots vs independent quadrature (no threshold; original 1e-9 relative criterion not met, O(h_max^~2) on the C^2 spline): max rel 7.407e-04 over 256 records with /I_ref/ >= 1e-3 max/I_ref/ (384 records with I_ref ~ 0 excluded fr |
| PASS | A2 layered N=64: snapshots (extra, not a spec threshold): quadrature tau(x1 recorded) = t within 10 tol (positions are error-controlled at tol)  max rel 8.418e-11 (limit 1.0e-07) |
| PASS | A3 failure field M=16: plane-pass status histogram exactly as constructed  got 88 40 32 32 0 32 32 / want 88 40 32 32 0 32 32 |
| PASS | A3 failure field M=16: time-pass status histogram exactly as constructed  got 120 40 32 32 0 0 32 / want 120 40 32 32 0 0 32 |
| PASS | A3 failure field M=16: per-particle status and n_reached (both passes) exactly as constructed  mismatching: plane status 0, time status 0, n_reached 0 |
| PASS | A3 failure field M=16: seed_backflow = 104 (32 g1<0 + 32 NaN + 32 k<0 + 8 pocket), zero weights exactly there, sum w = 1  seed_backflow 104, sum w - 1 = 1.11e-15, weight mismatches 0 |
| PASS | A3 failure field M=16: backflow_encounters = 24 per pass (one per pocket seed outside the pocket, none elsewhere)  plane 24, time 24 |
| PASS | A3 failure field M=16: recrossings = 0  plane 0 |
| PASS | A3 failure field M=16: landings_discarded: >= 1 for every landing_failed seed (LOWER BOUND, the retry count is not derivable), exactly 0 for every other particle/pass  min over band 37, plane total 1184, elsewhere 0 |
| PASS | A3 failure field M=16: moments bookkeeping: plane contributing {120,88,88}, non_ok 168; snapshot active {152,152,120} |
| PASS | B lester2021 eps=1 N=64: stage flow ready  ok |
| PASS | B lester2021 eps=1 N=64: all statuses ok (plane pass)  ok 256/256 |
| PASS | B lester2021 eps=1 N=64: one-period transverse return <= 10 tol = 1e-7  max 3.966e-11 (limit 1.0e-07) |
| PASS | E1 derivative check (central FD, d=1e-4) of g, H, grad ln k of the analytic field  max /g - FD/ 2.21e-08, /H - FD(g)/ 1.12e-07, /grad ln k - FD/ 2.06e-07 |
| PASS | E1 C-inf analytic field (Phi = x1 + .05 sin2pi x2 cos2pi x1 + .05 sin2pi x3 sin2pi x1, k = exp(.5 cos2pi(x1+x2+x3))): identity /a x b - v exp(-I)///v/ <= 1e-8 at tol 1e-10, h_max 1/128, planes+times {.5,1,2,4} [BLOCKING]  max 5.286e-11 (threshold 1e-8), rms 8. |
| RECORD | E2 generic3d eps 0.25 N=64: identity /a x b - v exp(-I)///v/ at tol 1e-10, h_max h/8 (no threshold; original 1e-8 criterion not met, O(h_max^~2) on the C^2 spline): max 6.680e-06, rms 1.110e-06 over 2048 records (non-ok passes 0) |
| RECORD | E2 gauss s2=1 ell=.25 seed=3001 eps 1.0 N=64: identity /a x b - v exp(-I)///v/ at tol 1e-10, h_max h/8 (no threshold; original 1e-8 criterion not met, O(h_max^~2) on the C^2 spline): max 4.610e-05, rms 5.728e-06 over 2048 records (non-ok passes 0) |
| RECORD | E2 gauss s2=1 ell=.25 seed=3001 eps 2.0 N=64: identity /a x b - v exp(-I)///v/ at tol 1e-10, h_max h/8 (no threshold; original 1e-8 criterion not met, O(h_max^~2) on the C^2 spline): max 1.542e-03, rms 7.282e-05 over 2048 records (non-ok passes 0) |
| PASS | D generic3d eps 0.25 N=64 seeds=256: stage flow ready  ok |
| PASS | D generic3d eps 0.25 N=64 seeds=256: tau=0.5: observed order >= 1.8 on both delta pairs (a and b)  rms_a (1,.5,.25)e-3 = 4.339e-06 1.149e-06 2.870e-07 / rms_b = 3.946e-06 1.037e-06 2.689e-07 / orders a: 1.917 2.001, b: 1.927 1.948 / seeds used 256 / min order  |
| RECORD | D generic3d eps 0.25 N=64 seeds=256: pre-registered ladder tau=0.5 (criterion as originally registered: met): rms_a (8,4,2)e-3 = 2.327e-04 6.109e-05 1.620e-05 / rms_b = 2.367e-04 6.008e-05 1.530e-05 / orders a: 1.929 1.915, b: 1.978 1.973 / seeds used 256 / mi |
| PASS | D generic3d eps 0.25 N=64 seeds=256: tau=2.0: observed order >= 1.8 on both delta pairs (a and b)  rms_a (1,.5,.25)e-3 = 7.204e-06 1.871e-06 4.725e-07 / rms_b = 5.225e-06 1.384e-06 3.586e-07 / orders a: 1.945 1.985, b: 1.917 1.948 / seeds used 256 / min order  |
| RECORD | D generic3d eps 0.25 N=64 seeds=256: pre-registered ladder tau=2.0 (criterion as originally registered: met): rms_a (8,4,2)e-3 = 3.830e-04 1.002e-04 2.700e-05 / rms_b = 2.831e-04 7.418e-05 1.958e-05 / orders a: 1.934 1.892, b: 1.932 1.922 / seeds used 256 / mi |
| PASS | D gauss s2=1 ell=.25 seed=3001 eps 1.0 N=64 seeds=256: stage flow ready  ok |
| PASS | D gauss s2=1 ell=.25 seed=3001 eps 1.0 N=64 seeds=256: tau=0.5: observed order >= 1.8 on both delta pairs (a and b)  rms_a (1,.5,.25)e-3 = 1.886e-03 4.702e-04 1.176e-04 / rms_b = 7.774e-04 1.961e-04 4.911e-05 / orders a: 2.004 2.000, b: 1.987 1.997 / seeds use |
| RECORD | D gauss s2=1 ell=.25 seed=3001 eps 1.0 N=64 seeds=256: pre-registered ladder tau=0.5 (criterion as originally registered: met): rms_a (8,4,2)e-3 = 1.197e-01 3.021e-02 7.568e-03 / rms_b = 4.646e-02 1.212e-02 3.071e-03 / orders a: 1.986 1.997, b: 1.939 1.980 / s |
| PASS | D gauss s2=1 ell=.25 seed=3001 eps 1.0 N=64 seeds=256: tau=2.0: observed order >= 1.8 on both delta pairs (a and b)  rms_a (1,.5,.25)e-3 = 4.905e-01 1.199e-01 2.953e-02 / rms_b = 1.862e-01 4.624e-02 1.147e-02 / orders a: 2.033 2.021, b: 2.010 2.011 / seeds use |
| RECORD | D gauss s2=1 ell=.25 seed=3001 eps 1.0 N=64 seeds=256: pre-registered ladder tau=2.0 (criterion as originally registered: not met): rms_a (8,4,2)e-3 = 5.221e+00 3.146e+00 1.605e+00 / rms_b = 2.611e+00 1.086e+00 5.852e-01 / orders a: 0.731 0.971, b: 1.266 0.892 |
| RECORD | E2 ladder generic3d eps 0.25 N=64: E residual (max / rms) at tol 1e-12 for h_max = h/8,h/16,h/32,h/64: 6.680e-06/1.105e-06  9.703e-07/1.952e-07  2.553e-07/4.855e-08  5.778e-08/1.163e-08  / observed orders (max): 2.78 1.93 2.14 / (rms): 2.50 2.01 2.06 |
| RECORD | DIAG-3 generic3d eps 0.25 N=64: /I(h/8) - I(h/64)/ over plane records: max 6.674e-06 rms 1.119e-06 (snapshots max 6.366e-06); RMS of I (h/64) 5.621e-04, max/I/ 3.019e-03; RMS of e_v = 1 - exp(-I) 5.622e-04 |
| RECORD | E2 ladder gauss eps 1.0 N=64: E residual (max / rms) at tol 1e-12 for h_max = h/8,h/16,h/32,h/64: 1.868e-05/3.315e-06  7.949e-06/1.315e-06  2.695e-06/3.181e-07  8.695e-07/8.506e-08  / observed orders (max): 1.23 1.56 1.63 / (rms): 1.33 2.05 1.90 |
| RECORD | DIAG-3 gauss eps 1.0 N=64: /I(h/8) - I(h/64)/ over plane records: max 1.319e-05 rms 2.799e-06 (snapshots max 2.007e-05); RMS of I (h/64) 1.361e-02, max/I/ 9.454e-02; RMS of e_v = 1 - exp(-I) 1.363e-02 |
| RECORD | E2 ladder gauss eps 2.0 N=64: E residual (max / rms) at tol 1e-12 for h_max = h/8,h/16,h/32,h/64: 1.004e-03/3.632e-05  2.729e-04/1.341e-05  9.551e-05/4.667e-06  3.307e-05/1.333e-06  / observed orders (max): 1.88 1.51 1.53 / (rms): 1.44 1.52 1.81 |
| RECORD | DIAG-3 gauss eps 2.0 N=64: /I(h/8) - I(h/64)/ over plane records: max 2.444e-05 rms 5.877e-06 (snapshots max 2.987e-05); RMS of I (h/64) 5.137e-02, max/I/ 2.759e-01; RMS of e_v = 1 - exp(-I) 5.199e-02 |
| RECORD | E2 ladder C-inf field, tol 1e-12, E residual (max / rms) for h_max = h/8..h/64 (h=1/64): 1.645e-13/1.845e-14  1.997e-13/2.407e-14  2.947e-13/3.051e-14  4.335e-13/4.581e-14  / observed orders (max): -0.28 -0.56 -0.56 |

## lagrangian_controls_tests --diag (`controls_diag.log`)

PASS 31, FAIL 0, RECORD 11.

Final lines: `SUMMARY: 31 checks, 0 failed -> ALL PASS  (37.3 s)`

| kind | check / detail |
|---|---|
| PASS | A1 k=1 N=16: stage flow ready  ok |
| PASS | A1 k=1 N=16: all particles ok in both passes  plane ok 64/64, time ok 64 |
| PASS | A1 k=1 N=16: transverse displacement (planes+snapshots) <= 1e-12  max 0.000e+00 over 640 records |
| PASS | A1 k=1 N=16: tau at plane X equals X - X0, relative <= 1e-12  max rel 0.000e+00 |
| PASS | A1 k=1 N=16: covectors equal initial values <= 1e-12  max 0.000e+00 |
| PASS | A1 k=1 N=16: /I/ <= 1e-14  max 0.000e+00 |
| PASS | A2 layered N=32: stage flow ready  ok |
| PASS | A2 layered N=32: all particles ok in both passes |
| PASS | A2 layered N=32: transverse positions constant <= 1e-12  max 1.318e-16 (max /g2/,/g3/ of the spline flow on the lines: 7.216e-16) |
| PASS | A2 layered N=32: covectors constant <= 1e-12  max 3.132e-14 |
| PASS | A2 layered N=32: tau at planes vs independent quadrature <= 1e-9 relative  max rel 4.771e-10 (abs 1.191e-10) |
| RECORD | A2 layered N=32: I at planes+snapshots vs independent quadrature (no threshold; original 1e-9 relative criterion not met, O(h_max^~2) on the C^2 spline): max rel 1.160e-03 over 256 records with /I_ref/ >= 1e-3 max/I_ref/ (384 records with I_ref ~ 0 excluded fr |
| PASS | A2 layered N=32: snapshots (extra, not a spec threshold): quadrature tau(x1 recorded) = t within 10 tol (positions are error-controlled at tol)  max rel 1.789e-09 (limit 1.0e-07) |
| PASS | A3 failure field M=16: plane-pass status histogram exactly as constructed  got 88 40 32 32 0 32 32 / want 88 40 32 32 0 32 32 |
| PASS | A3 failure field M=16: time-pass status histogram exactly as constructed  got 120 40 32 32 0 0 32 / want 120 40 32 32 0 0 32 |
| PASS | A3 failure field M=16: per-particle status and n_reached (both passes) exactly as constructed  mismatching: plane status 0, time status 0, n_reached 0 |
| PASS | A3 failure field M=16: seed_backflow = 104 (32 g1<0 + 32 NaN + 32 k<0 + 8 pocket), zero weights exactly there, sum w = 1  seed_backflow 104, sum w - 1 = 1.11e-15, weight mismatches 0 |
| PASS | A3 failure field M=16: backflow_encounters = 24 per pass (one per pocket seed outside the pocket, none elsewhere)  plane 24, time 24 |
| PASS | A3 failure field M=16: recrossings = 0  plane 0 |
| PASS | A3 failure field M=16: landings_discarded: >= 1 for every landing_failed seed (LOWER BOUND, the retry count is not derivable), exactly 0 for every other particle/pass  min over band 37, plane total 1184, elsewhere 0 |
| PASS | A3 failure field M=16: moments bookkeeping: plane contributing {120,88,88}, non_ok 168; snapshot active {152,152,120} |
| PASS | B lester2021 eps=1 N=32: stage flow ready  ok |
| PASS | B lester2021 eps=1 N=32: all statuses ok (plane pass)  ok 256/256 |
| PASS | B lester2021 eps=1 N=32: one-period transverse return <= 10 tol = 1e-7  max 6.215e-10 (limit 1.0e-07) |
| PASS | E1 derivative check (central FD, d=1e-4) of g, H, grad ln k of the analytic field  max /g - FD/ 2.21e-08, /H - FD(g)/ 1.12e-07, /grad ln k - FD/ 2.06e-07 |
| PASS | E1 C-inf analytic field (Phi = x1 + .05 sin2pi x2 cos2pi x1 + .05 sin2pi x3 sin2pi x1, k = exp(.5 cos2pi(x1+x2+x3))): identity /a x b - v exp(-I)///v/ <= 1e-8 at tol 1e-10, h_max 1/128, planes+times {.5,1,2,4} [BLOCKING]  max 5.286e-11 (threshold 1e-8), rms 8. |
| RECORD | E2 generic3d eps 0.25 N=16: identity /a x b - v exp(-I)///v/ at tol 1e-10, h_max h/8 (no threshold; original 1e-8 criterion not met, O(h_max^~2) on the C^2 spline): max 1.985e-04, rms 3.050e-05 over 2048 records (non-ok passes 0) |
| RECORD | E2 gauss s2=1 ell=.25 seed=3001 eps 1.0 N=16: identity /a x b - v exp(-I)///v/ at tol 1e-10, h_max h/8 (no threshold; original 1e-8 criterion not met, O(h_max^~2) on the C^2 spline): max 4.201e-04, rms 6.473e-05 over 2048 records (non-ok passes 0) |
| RECORD | E2 gauss s2=1 ell=.25 seed=3001 eps 2.0 N=16: identity /a x b - v exp(-I)///v/ at tol 1e-10, h_max h/8 (no threshold; original 1e-8 criterion not met, O(h_max^~2) on the C^2 spline): max 2.714e-01, rms 7.326e-03 over 2048 records (non-ok passes 0) |
| PASS | D generic3d eps 0.25 N=32 seeds=16: stage flow ready  ok |
| PASS | D generic3d eps 0.25 N=32 seeds=16: tau=0.5: observed order >= 1.8 on both delta pairs (a and b)  rms_a (1,.5,.25)e-3 = 5.881e-06 1.550e-06 3.786e-07 / rms_b = 3.947e-06 9.755e-07 2.385e-07 / orders a: 1.924 2.033, b: 2.017 2.032 / seeds used 16 / min order 1. |
| PASS | D generic3d eps 0.25 N=32 seeds=16: tau=2.0: observed order >= 1.8 on both delta pairs (a and b)  rms_a (1,.5,.25)e-3 = 8.138e-06 2.059e-06 5.207e-07 / rms_b = 6.810e-06 1.687e-06 4.341e-07 / orders a: 1.983 1.984, b: 2.013 1.959 / seeds used 16 / min order 1. |
| PASS | D gauss s2=1 ell=.25 seed=3001 eps 1.0 N=32 seeds=16: stage flow ready  ok |
| PASS | D gauss s2=1 ell=.25 seed=3001 eps 1.0 N=32 seeds=16: tau=0.5: observed order >= 1.8 on both delta pairs (a and b)  rms_a (1,.5,.25)e-3 = 1.864e-03 4.674e-04 1.162e-04 / rms_b = 3.046e-03 7.931e-04 1.994e-04 / orders a: 1.996 2.008, b: 1.941 1.992 / seeds used |
| PASS | D gauss s2=1 ell=.25 seed=3001 eps 1.0 N=32 seeds=16: tau=2.0: observed order >= 1.8 on both delta pairs (a and b)  rms_a (1,.5,.25)e-3 = 1.728e-01 4.669e-02 1.195e-02 / rms_b = 1.036e-01 2.837e-02 7.230e-03 / orders a: 1.888 1.966, b: 1.868 1.972 / seeds used |
| RECORD | E2 ladder generic3d eps 0.25 N=32: E residual (max / rms) at tol 1e-12 for h_max = h/8,h/16,h/32,h/64: 1.983e-05/3.078e-06  8.707e-06/1.197e-06  1.346e-06/2.809e-07  3.851e-07/6.796e-08  / observed orders (max): 1.19 2.69 1.81 / (rms): 1.36 2.09 2.05 |
| RECORD | DIAG-3 generic3d eps 0.25 N=32: /I(h/8) - I(h/64)/ over plane records: max 1.981e-05 rms 2.874e-06 (snapshots max 1.866e-05); RMS of I (h/64) 2.332e-03, max/I/ 1.319e-02; RMS of e_v = 1 - exp(-I) 2.334e-03 |
| RECORD | E2 ladder gauss eps 1.0 N=32: E residual (max / rms) at tol 1e-12 for h_max = h/8,h/16,h/32,h/64: 3.991e-05/6.768e-06  2.410e-05/3.092e-06  9.594e-06/1.505e-06  3.779e-06/4.398e-07  / observed orders (max): 0.73 1.33 1.34 / (rms): 1.13 1.04 1.77 |
| RECORD | DIAG-3 gauss eps 1.0 N=32: /I(h/8) - I(h/64)/ over plane records: max 2.574e-05 rms 5.273e-06 (snapshots max 4.324e-05); RMS of I (h/64) 5.485e-02, max/I/ 3.854e-01; RMS of e_v = 1 - exp(-I) 5.572e-02 |
| RECORD | E2 ladder gauss eps 2.0 N=32: E residual (max / rms) at tol 1e-12 for h_max = h/8,h/16,h/32,h/64: 1.879e-03/5.946e-05  1.858e-03/5.811e-05  1.166e-04/1.112e-05  1.392e-04/5.621e-06  / observed orders (max): 0.02 3.99 -0.26 / (rms): 0.03 2.39 0.98 |
| RECORD | DIAG-3 gauss eps 2.0 N=32: /I(h/8) - I(h/64)/ over plane records: max 3.245e-05 rms 7.694e-06 (snapshots max 3.614e-05); RMS of I (h/64) 2.127e-01, max/I/ 1.342e+00; RMS of e_v = 1 - exp(-I) 2.664e-01 |
| RECORD | E2 ladder C-inf field, tol 1e-12, E residual (max / rms) for h_max = h/8..h/64 (h=1/32): 1.733e-12/2.598e-13  1.645e-13/1.845e-14  1.997e-13/2.407e-14  2.947e-13/3.051e-14  / observed orders (max): 3.40 -0.28 -0.56 |

## lagrangian_gpu_host_tests --full (`gpu_host_full.log`)

PASS 26, FAIL 0, RECORD 0.

Final lines: `SUMMARY: 26 checks, 0 failed -> ALL PASS`

| kind | check / detail |
|---|---|
| PASS | k = 1 N=64: stage flow ready  ok |
| PASS | k = 1 N=64: per-particle status identical to host trace_to_plane  differing 0 of 4096 |
| PASS | k = 1 N=64: max /dx/ (x2, x3 at the plane) <= 1e-12  max/dx/ 0.000e+00 over 4096 ok particles |
| PASS | k = 1 N=64: per-particle nfev identical (same step sequences)  differing step sequences 0 |
| PASS | generic3d eps 0.25 N=64: stage flow ready  ok |
| PASS | generic3d eps 0.25 N=64: per-particle status identical to host trace_to_plane  differing 0 of 4096 |
| PASS | generic3d eps 0.25 N=64: max /dx/ (x2, x3 at the plane) <= 1e-12  max/dx/ 0.000e+00 over 4096 ok particles |
| PASS | generic3d eps 0.25 N=64: per-particle nfev identical (same step sequences)  differing step sequences 0 |
| PASS | gauss s2=1 ell=0.25 seed=3001 eps 0.5 N=64: stage flow ready  ok |
| PASS | gauss s2=1 ell=0.25 seed=3001 eps 0.5 N=64: per-particle status identical to host trace_to_plane  differing 0 of 4096 |
| PASS | gauss s2=1 ell=0.25 seed=3001 eps 0.5 N=64: max /dx/ (x2, x3 at the plane) <= 1e-12  max/dx/ 0.000e+00 over 4096 ok particles |
| PASS | gauss s2=1 ell=0.25 seed=3001 eps 0.5 N=64: per-particle nfev identical (same step sequences)  differing step sequences 0 |
| PASS | gauss s2=1 ell=0.25 seed=3001 eps 2.0 N=64: stage flow ready  ok |
| PASS | gauss s2=1 ell=0.25 seed=3001 eps 2.0 N=64: per-particle status identical to host trace_to_plane  differing 0 of 4096 |
| PASS | gauss s2=1 ell=0.25 seed=3001 eps 2.0 N=64: max /dx/ (x2, x3 at the plane) <= 1e-12  max/dx/ 0.000e+00 over 4096 ok particles |
| PASS | gauss s2=1 ell=0.25 seed=3001 eps 2.0 N=64: per-particle nfev identical (same step sequences)  differing step sequences 0 |
| PASS | I gauss eps 0.5 N=16: stage ok with labels  ok |
| PASS | I gauss eps 0.5 N=16: device storage pointers stable across track() (no reallocation) |
| PASS | I gauss eps 0.5 N=16: cudaMemGetInfo free bytes equal across the 1st, 2nd and 3rd track()  free bytes 33405009920 / 33405009920 / 33405009920 (device shared: re-run alone before concluding) |
| PASS | I gauss eps 0.5 N=16: two launches bitwise identical (all outputs, moments) |
| PASS | I gauss eps 0.5 N=16: block 64 vs 256 bitwise identical |
| PASS | I gauss eps 0.5 N=16: positions, tau, I, statuses, nfev bitwise identical with covectors on vs off  nfev differing particles 0 |
| PASS | I gauss eps 0.5 N=16: D-1 labels finite at the seeds, weights sum to 1  sum w 0.9999999999999994 |
| PASS | I gauss eps 0.5 N=16: moments_time.csv read back by analysis::read_timeseries_csv reproduces the in-memory moments (rows 1..T; row 0 = reader's header artifact)  max rel 4.163e-17, raw rows 4 |
| PASS | I gauss eps 0.5 N=16: compute_macrodispersion runs on the read series  rows 3 |
| PASS | I gauss eps 0.5 N=16: planes_x23.npy round trip exact, shape (Np, P, 2) |
