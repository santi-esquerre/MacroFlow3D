# SF-37 digest V3a / V3b (D5, D6)

- python 3.11.7, numpy 1.26.4, scipy 1.11.4
- bootstrap matrix: `analysis/bootstrap_blocks.npy`, sha256 `16c1a5b9d9fe062576d5f18fc9d54127db76472c7fbe64493201d8cffc66af39` (verified against the pre-registered value)
- command: `digest_v3.py /home/sesquerre/sf37_campaign /home/sesquerre/sf37_campaign/analysis/v3 --design full`
- design: `full`; V3a runs 3 (eps, seed) combinations, V3b 9
- criterion C by `../MacroFlow3D-SF-37/docs/experiments/artifacts/2026-10-07-sf35-label-transport/scripts/orders_digest.py` invoked in place

## V3a - confinement against the verified gridded labels (eps = 1)

Rule (RR6): per new seed SF-35 criterion C on the labels at [64, 128, 256] decides whether they are a verified reference. Where verified, the UNWEIGHTED RMS of |psi_grid(x_p) - psi_p| at the grid planes is `concordant` iff it decreases on every refinement. `reference not verified` (C is not accept) and `reference not available` (inlet backflow) still record the numbers.

| eps_code | seed | C verdict | script verdict | note |
|---|---|---|---|---|
| 10 | 3101 | accept | accept |  |
| 10 | 3102 | accept | accept |  |
| 10 | 3103 | accept | accept |  |

### Classification per seed and plane

| eps_code | seed | x1 | C | reference | RMS decreases | classification | RMS N=64 | RMS N=128 | RMS N=256 |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 3101 | 0.25 | accept | reference verified | concordant | concordant | 1.49605e-06 | 1.02295e-07 | 8.21171e-09 |
| 10 | 3101 | 0.5 | accept | reference verified | concordant | concordant | 1.20981e-05 | 5.31622e-07 | 3.227e-08 |
| 10 | 3101 | 0.75 | accept | reference verified | concordant | concordant | 1.00859e-05 | 4.37171e-07 | 2.68071e-08 |
| 10 | 3101 | 1 | accept | reference verified | concordant | concordant | 6.34511e-06 | 2.84229e-07 | 1.90358e-08 |
| 10 | 3102 | 0.25 | accept | reference verified | concordant | concordant | 1.55427e-06 | 9.60586e-08 | 7.24422e-09 |
| 10 | 3102 | 0.5 | accept | reference verified | concordant | concordant | 3.72919e-06 | 1.91001e-07 | 1.28441e-08 |
| 10 | 3102 | 0.75 | accept | reference verified | concordant | concordant | 1.67389e-05 | 7.05465e-07 | 2.91614e-08 |
| 10 | 3102 | 1 | accept | reference verified | concordant | concordant | 1.93747e-05 | 7.51696e-07 | 3.17125e-08 |
| 10 | 3103 | 0.25 | accept | reference verified | concordant | concordant | 2.83821e-06 | 1.52243e-07 | 1.02665e-08 |
| 10 | 3103 | 0.5 | accept | reference verified | concordant | concordant | 4.85838e-06 | 2.29153e-07 | 1.46751e-08 |
| 10 | 3103 | 0.75 | accept | reference verified | concordant | concordant | 1.79189e-06 | 1.30468e-07 | 1.08643e-08 |
| 10 | 3103 | 1 | accept | reference verified | concordant | concordant | 1.7015e-06 | 1.42049e-07 | 1.20659e-08 |

Classification counts: 12 concordant of 12 (seed, plane) cells; the other cells carry their reference state or `not concordant` / `not_evaluable`.

### D5

Interpolation proxy (RMS over the 2N lattice of |U_N interpolated - U_2N|), V3a RMS as a multiple of it, the fraction of particles above 10 x proxy, flux-weighted RMS and per-label RMS: `v3a_cells.csv`; constructor exit code and max round trip: `v3a_constructor.csv`; stage identity trace vs forward (G, inlet_vmin, v_rms at print precision; absent quantities are stated): `v3a_stage_identity.csv`; control x1 = 0: `v3a_control_x0.csv`.

| eps_code | seed | N | G | inlet_vmin | v_rms |
|---|---|---|---|---|---|
| 10 | 3101 | 64 | identical | identical | identical |
| 10 | 3101 | 128 | identical | identical | identical |
| 10 | 3101 | 256 | identical | identical | identical |
| 10 | 3102 | 64 | identical | identical | identical |
| 10 | 3102 | 128 | identical | identical | identical |
| 10 | 3102 | 256 | identical | identical | identical |
| 10 | 3103 | 64 | identical | identical | identical |
| 10 | 3103 | 128 | identical | identical | identical |
| 10 | 3103 | 256 | identical | identical | identical |

## V3b - Pollock (RT0 on the SF-19 faces) against the Lagrangian O6

Rule (RR7): `concordant` iff |D_P| decreases on every refinement 64 -> 128 -> 256 -> 512 and |D_P(N_max)| <= max(0.05 sigma_T^2(O6, N_max), 2 SE_D); else `discordant`; missing data `not_evaluable`. Not blocking; a `discordant` at any amplitude is reported to the owner.

| eps_code | seed | x1 | classification | reason | threshold | D_P N=64 | D_P N=128 | D_P N=256 | D_P N=512 |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 3101 | 0.25 | discordant | \|D_P\| does not decrease on every refinement | 0.00423309 | -6.68029e-06 | -2.62186e-06 | -1.59634e-06 | -8.67882e-05 |
| 10 | 3101 | 0.5 | discordant | \|D_P\| does not decrease on every refinement | 0.00436566 | 3.05538e-06 | 2.21744e-07 | -5.44212e-07 | -7.25974e-05 |
| 10 | 3101 | 1 | discordant | \|D_P\| does not decrease on every refinement | 0.00396687 | 2.61821e-05 | 5.45693e-06 | 5.11721e-07 | -8.2197e-05 |
| 10 | 3101 | 2 | discordant | \|D_P\| does not decrease on every refinement | 0.00402489 | 5.6038e-05 | 1.26388e-05 | 2.25038e-06 | -8.04494e-05 |
| 10 | 3101 | 4 | discordant | \|D_P\| does not decrease on every refinement | 0.00422221 | 0.000101871 | 2.44967e-05 | 4.80994e-06 | -8.7586e-05 |
| 10 | 3101 | 8 | discordant | \|D_P\| does not decrease on every refinement | 0.00477158 | 0.000241078 | 5.7334e-05 | 1.28214e-05 | -0.000135515 |
| 10 | 3101 | 16 | discordant | \|D_P\| does not decrease on every refinement | 0.00556186 | 0.000501838 | 8.85237e-05 | 3.10173e-05 | -0.000185983 |
| 10 | 3102 | 0.25 | discordant | \|D_P\| does not decrease on every refinement | 0.00380346 | 1.94595e-05 | 5.07163e-06 | 1.58738e-06 | -2.46827e-05 |
| 10 | 3102 | 0.5 | discordant | \|D_P\| does not decrease on every refinement | 0.00344199 | 3.64698e-05 | 8.59813e-06 | 2.10651e-06 | -1.76719e-05 |
| 10 | 3102 | 1 | discordant | \|D_P\| does not decrease on every refinement | 0.00368444 | 4.3709e-05 | 1.11997e-05 | 3.46508e-06 | -2.2239e-05 |
| 10 | 3102 | 2 | discordant | \|D_P\| does not decrease on every refinement | 0.00387904 | 7.71526e-05 | 1.90196e-05 | 5.50756e-06 | -1.97356e-05 |
| 10 | 3102 | 4 | discordant | \|D_P\| does not decrease on every refinement | 0.00457723 | 0.000134507 | 3.18419e-05 | 8.45938e-06 | -1.35874e-05 |
| 10 | 3102 | 8 | concordant | \|D_P\| decreases on every refinement and \|D_P(512)\| = 1.246e-06 <= 0.00674 | 0.00673977 | 0.000288331 | 6.69501e-05 | 1.70469e-05 | -1.24553e-06 |
| 10 | 3102 | 16 | concordant | \|D_P\| decreases on every refinement and \|D_P(512)\| = 1.586e-05 <= 0.0108 | 0.0108031 | 0.00178656 | 0.000445724 | 0.000104401 | -1.58576e-05 |
| 10 | 3103 | 0.25 | discordant | \|D_P\| does not decrease on every refinement | 0.00375968 | -5.25594e-06 | -1.67094e-06 | -8.93076e-07 | -1.62893e-05 |
| 10 | 3103 | 0.5 | discordant | \|D_P\| does not decrease on every refinement | 0.0037116 | -2.28364e-06 | -8.5976e-07 | -6.57357e-07 | -1.7894e-05 |
| 10 | 3103 | 1 | discordant | \|D_P\| does not decrease on every refinement | 0.00379698 | -6.30654e-07 | -1.04433e-08 | -3.18572e-09 | -2.53239e-05 |
| 10 | 3103 | 2 | discordant | \|D_P\| does not decrease on every refinement | 0.00381388 | 1.24223e-05 | 3.07092e-06 | 7.53129e-07 | -3.01948e-05 |
| 10 | 3103 | 4 | discordant | \|D_P\| does not decrease on every refinement | 0.00387932 | 3.05211e-05 | 7.96554e-06 | 1.92314e-06 | -4.10213e-05 |
| 10 | 3103 | 8 | discordant | \|D_P\| does not decrease on every refinement | 0.00412063 | 4.59215e-05 | 9.99273e-06 | 1.23597e-06 | -6.08018e-05 |
| 10 | 3103 | 16 | discordant | \|D_P\| does not decrease on every refinement | 0.00497722 | -1.22365e-05 | -3.28891e-06 | -4.33739e-06 | -7.14134e-05 |
| 15 | 3101 | 0.25 | discordant | \|D_P\| does not decrease on every refinement | 0.00426736 | -2.48315e-07 | -6.78672e-07 | -8.27598e-07 | -0.000161912 |
| 15 | 3101 | 0.5 | discordant | \|D_P\| does not decrease on every refinement | 0.00476896 | 3.71811e-05 | 9.63259e-06 | 2.66824e-06 | -0.000145642 |
| 15 | 3101 | 1 | discordant | \|D_P\| does not decrease on every refinement | 0.00388358 | 0.00010239 | 2.52904e-05 | 5.39574e-06 | -0.000152154 |
| 15 | 3101 | 2 | discordant | \|D_P\| does not decrease on every refinement | 0.00411892 | 0.000209849 | 5.13373e-05 | 1.15162e-05 | -0.000172022 |
| 15 | 3101 | 4 | discordant | \|D_P\| does not decrease on every refinement | 0.00489928 | 0.000473789 | 0.000115321 | 2.80715e-05 | -0.000274883 |
| 15 | 3101 | 8 | discordant | \|D_P\| does not decrease on every refinement | 0.00654591 | 0.0012035 | 0.000291217 | 6.61102e-05 | -0.000487591 |
| 15 | 3101 | 16 | discordant | \|D_P\| does not decrease on every refinement | 0.00891902 | 0.00382047 | 0.000866925 | 0.00017034 | -0.000623326 |
| 15 | 3102 | 0.25 | discordant | \|D_P\| does not decrease on every refinement | 0.00316057 | 6.61021e-05 | 1.60335e-05 | 3.50558e-06 | -4.72565e-05 |
| 15 | 3102 | 0.5 | discordant | \|D_P\| does not decrease on every refinement | 0.00299072 | 9.98264e-05 | 2.39639e-05 | 4.51314e-06 | -5.03831e-05 |
| 15 | 3102 | 1 | discordant | \|D_P\| does not decrease on every refinement | 0.00290427 | 0.00010605 | 2.71965e-05 | 7.25972e-06 | -2.59419e-05 |
| 15 | 3102 | 2 | discordant | \|D_P\| does not decrease on every refinement | 0.00344076 | 0.000234805 | 5.87343e-05 | 1.4687e-05 | -2.15992e-05 |
| 15 | 3102 | 4 | concordant | \|D_P\| decreases on every refinement and \|D_P(512)\| = 2.63e-05 <= 0.005116 | 0.00511628 | 0.000533302 | 0.000128228 | 3.11764e-05 | -2.63018e-05 |
| 15 | 3102 | 8 | concordant | \|D_P\| decreases on every refinement and \|D_P(512)\| = 0.0001324 <= 0.00827 | 0.00827005 | 0.00204326 | 0.000467694 | 0.000137751 | 0.000132422 |
| 15 | 3102 | 16 | concordant | \|D_P\| decreases on every refinement and \|D_P(512)\| = 0.0001279 <= 0.01512 | 0.0151198 | 0.0125868 | 0.0038922 | 0.000944981 | 0.00012793 |
| 15 | 3103 | 0.25 | discordant | \|D_P\| does not decrease on every refinement | 0.00358365 | -7.42342e-06 | -2.61445e-06 | -1.38787e-06 | -4.46579e-05 |
| 15 | 3103 | 0.5 | discordant | \|D_P\| does not decrease on every refinement | 0.0035951 | 6.35716e-06 | 6.20605e-07 | -3.38741e-07 | -4.56868e-05 |
| 15 | 3103 | 1 | discordant | \|D_P\| does not decrease on every refinement | 0.00360925 | 1.5851e-05 | 3.72858e-06 | 8.77309e-07 | -6.59339e-05 |
| 15 | 3103 | 2 | discordant | \|D_P\| does not decrease on every refinement | 0.00368984 | 5.06378e-05 | 1.13435e-05 | 2.09613e-06 | -8.44591e-05 |
| 15 | 3103 | 4 | discordant | \|D_P\| does not decrease on every refinement | 0.00394766 | 5.91716e-05 | 1.31121e-05 | 1.49233e-06 | -0.000112508 |
| 15 | 3103 | 8 | discordant | \|D_P\| does not decrease on every refinement | 0.0047349 | -5.38576e-05 | -2.32599e-05 | -7.74462e-06 | -0.000150721 |
| 15 | 3103 | 16 | discordant | \|D_P\| does not decrease on every refinement | 0.00656553 | -0.000186732 | -7.81923e-05 | -1.66232e-05 | -0.000230898 |
| 20 | 3101 | 0.25 | discordant | \|D_P\| does not decrease on every refinement | 0.00431051 | 2.11773e-05 | 4.09189e-06 | -9.42509e-08 | -0.000234681 |
| 20 | 3101 | 0.5 | discordant | \|D_P\| does not decrease on every refinement | 0.00527114 | 8.14036e-05 | 2.1668e-05 | 6.28822e-06 | -0.000226797 |
| 20 | 3101 | 1 | discordant | \|D_P\| does not decrease on every refinement | 0.00390168 | 0.000271153 | 6.6861e-05 | 1.45219e-05 | -0.000239137 |
| 20 | 3101 | 2 | discordant | \|D_P\| does not decrease on every refinement | 0.00454104 | 0.000539835 | 0.000134448 | 2.98199e-05 | -0.000345834 |
| 20 | 3101 | 4 | discordant | \|D_P\| does not decrease on every refinement | 0.00610996 | 0.00135517 | 0.000322463 | 7.81281e-05 | -0.000592532 |
| 20 | 3101 | 8 | discordant | \|D_P\| does not decrease on every refinement | 0.00897601 | 0.00384945 | 0.000886222 | 0.000218636 | -0.000899604 |
| 20 | 3101 | 16 | concordant | \|D_P\| decreases on every refinement and \|D_P(512)\| = 0.0006342 <= 0.01496 | 0.0149594 | 0.0121738 | 0.00250444 | 0.000785867 | -0.000634238 |
| 20 | 3102 | 0.25 | discordant | \|D_P\| does not decrease on every refinement | 0.00242836 | 0.000130153 | 3.04433e-05 | 5.21288e-06 | -7.48043e-05 |
| 20 | 3102 | 0.5 | discordant | \|D_P\| does not decrease on every refinement | 0.00254921 | 0.00018284 | 4.24537e-05 | 7.22483e-06 | -0.000100389 |
| 20 | 3102 | 1 | concordant | \|D_P\| decreases on every refinement and \|D_P(512)\| = 1.56e-05 <= 0.002111 | 0.00211062 | 0.000291635 | 7.45318e-05 | 1.726e-05 | -1.56046e-05 |
| 20 | 3102 | 2 | discordant | \|D_P\| does not decrease on every refinement | 0.00301856 | 0.000580267 | 0.000147263 | 3.44832e-05 | -9.0796e-05 |
| 20 | 3102 | 4 | discordant | \|D_P\| does not decrease on every refinement | 0.00469986 | 0.001118 | 0.000265742 | 5.6667e-05 | 5.84048e-05 |
| 20 | 3102 | 8 | concordant | \|D_P\| decreases on every refinement and \|D_P(512)\| = 0.0002542 <= 0.007428 | 0.007428 | 0.00574904 | 0.00171312 | 0.000382803 | 0.000254232 |
| 20 | 3102 | 16 | concordant | \|D_P\| decreases on every refinement and \|D_P(512)\| = 0.000331 <= 0.01638 | 0.0163825 | 0.0354511 | 0.00808397 | 0.00132455 | -0.00033099 |
| 20 | 3103 | 0.25 | discordant | \|D_P\| does not decrease on every refinement | 0.00344989 | -3.46232e-06 | -1.99166e-06 | -1.47294e-06 | -7.22739e-05 |
| 20 | 3103 | 0.5 | discordant | \|D_P\| does not decrease on every refinement | 0.00355548 | 3.41321e-05 | 8.12445e-06 | 1.37705e-06 | -7.12407e-05 |
| 20 | 3103 | 1 | discordant | \|D_P\| does not decrease on every refinement | 0.00347108 | 6.82007e-05 | 1.76095e-05 | 4.62999e-06 | -0.000109784 |
| 20 | 3103 | 2 | discordant | \|D_P\| does not decrease on every refinement | 0.00362538 | 9.78526e-05 | 2.47237e-05 | 5.70725e-06 | -0.000140546 |
| 20 | 3103 | 4 | discordant | \|D_P\| does not decrease on every refinement | 0.004071 | 7.79951e-06 | -2.97277e-06 | -4.69127e-06 | -0.000207105 |
| 20 | 3103 | 8 | discordant | \|D_P\| does not decrease on every refinement | 0.00535179 | 1.7053e-06 | -1.4312e-06 | -4.43786e-06 | -0.000315932 |
| 20 | 3103 | 16 | discordant | \|D_P\| does not decrease on every refinement | 0.00834132 | -0.000379858 | -9.98196e-05 | -2.35465e-05 | -0.00023577 |

Cells: 54 discordant, 0 not_evaluable, 9 concordant of 63.

### D6

Decomposition (weight part, trajectory part, each with its paired SE_D), D_P on the common contributors, signed D_P per N: `v3b_cells.csv`; noise-limited flag per refinement: `v3b_noise_limited.csv`; slope of D_P against x1 and its scaling with N: `v3b_slopes.csv`; Pollock non-active count and weight per pass: `v3b_pollock_status.csv`; time-snapshot analogues (var x1, x2, x3 at each t): `v3b_time_snapshots.csv`.

### Consistency (recorded)

| eps_code | seed | N | O6 files vs v2 run | Pollock moments worst rel. diff | <= 1e-12 | non_ok column | status |
|---|---|---|---|---|---|---|---|
| 10 | 3101 | 64 | bitwise identical | 3.48646e-16 | yes | yes | ok |
| 10 | 3101 | 128 | bitwise identical | 4.87788e-16 | yes | yes | ok |
| 10 | 3101 | 256 | bitwise identical | 3.88984e-16 | yes | yes | ok |
| 10 | 3101 | 512 | bitwise identical | 4.14972e-16 | yes | yes | ok |
| 10 | 3102 | 64 | bitwise identical | 4.11562e-16 | yes | yes | ok |
| 10 | 3102 | 128 | bitwise identical | 3.93545e-16 | yes | yes | ok |
| 10 | 3102 | 256 | bitwise identical | 3.49697e-16 | yes | yes | ok |
| 10 | 3102 | 512 | bitwise identical | 3.82704e-16 | yes | yes | ok |
| 10 | 3103 | 64 | bitwise identical | 3.5473e-16 | yes | yes | ok |
| 10 | 3103 | 128 | bitwise identical | 3.53127e-16 | yes | yes | ok |
| 10 | 3103 | 256 | bitwise identical | 3.66663e-16 | yes | yes | ok |
| 10 | 3103 | 512 | bitwise identical | 3.53011e-16 | yes | yes | ok |
| 15 | 3101 | 64 | bitwise identical | 5.16627e-16 | yes | yes | ok |
| 15 | 3101 | 128 | bitwise identical | 3.98544e-16 | yes | yes | ok |
| 15 | 3101 | 256 | bitwise identical | 5.29346e-16 | yes | yes | ok |
| 15 | 3101 | 512 | bitwise identical | 3.98621e-16 | yes | yes | ok |
| 15 | 3102 | 64 | bitwise identical | 5.06784e-16 | yes | yes | ok |
| 15 | 3102 | 128 | bitwise identical | 3.83773e-16 | yes | yes | ok |
| 15 | 3102 | 256 | bitwise identical | 3.9528e-16 | yes | yes | ok |
| 15 | 3102 | 512 | bitwise identical | 3.5542e-16 | yes | yes | ok |
| 15 | 3103 | 64 | bitwise identical | 4.57328e-16 | yes | yes | ok |
| 15 | 3103 | 128 | bitwise identical | 4.11277e-16 | yes | yes | ok |
| 15 | 3103 | 256 | bitwise identical | 4.39021e-16 | yes | yes | ok |
| 15 | 3103 | 512 | bitwise identical | 5.49061e-16 | yes | yes | ok |
| 20 | 3101 | 64 | bitwise identical | 4.16322e-16 | yes | yes | ok |
| 20 | 3101 | 128 | bitwise identical | 4.44006e-16 | yes | yes | ok |
| 20 | 3101 | 256 | bitwise identical | 3.56572e-16 | yes | yes | ok |
| 20 | 3101 | 512 | bitwise identical | 4.39272e-16 | yes | yes | ok |
| 20 | 3102 | 64 | bitwise identical | 4.76455e-16 | yes | yes | ok |
| 20 | 3102 | 128 | bitwise identical | 3.5132e-16 | yes | yes | ok |
| 20 | 3102 | 256 | bitwise identical | 5.263e-16 | yes | yes | ok |
| 20 | 3102 | 512 | bitwise identical | 2.49327e-16 | yes | yes | ok |
| 20 | 3103 | 64 | bitwise identical | 4.16719e-16 | yes | yes | ok |
| 20 | 3103 | 128 | bitwise identical | 4.09834e-16 | yes | yes | ok |
| 20 | 3103 | 256 | bitwise identical | 3.27444e-16 | yes | yes | ok |
| 20 | 3103 | 512 | bitwise identical | 3.68864e-16 | yes | yes | ok |

## Input manifest

1137 input files (path, sha256, bytes in `input_manifest_v3.csv`); sha256 of the manifest listing: `86751c5b0593e9cbdceb113834ad87350dcab5cdc327bcb29cf2cbba3525dd41`.
