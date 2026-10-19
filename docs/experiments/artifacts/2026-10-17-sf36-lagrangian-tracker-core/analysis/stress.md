# SF-36 inlet-backflow stress field (N5 finding; informative)

Run environment (`run.env`): COMMIT=e81b11833412149027643293d7e11ff989643189; HOST=localhost.localdomain; DATE_UTC=2026-10-17T22:56:37Z; BUILD=/home/sesquerre/MacroFlow3D-SF-36/build/v100-release; PART=stress; DRYRUN=0; BASE_REFS=/home/sesquerre/sf36_base_refs; CUDA_VISIBLE_DEVICES=0; GPU=Tesla V100-PCIE-32GB

Exit codes (124 = did not terminate within the `timeout`; 22 = `track_fail`; 0 = ok):

| run | exit code |
|---|---|
| backflow_stress_n16 | 22 |

```
  STRESS_RESULT rc=22 wall=1228s timeout=1800s (terminated)
```

### `backflow_stress_n16.log` (23 lines)

```
TRACK field=gaussian eps=4 N=16 h=0.0625 x0=0 M=16 Np=256 planes=0.5,1 times=0.5,1 tol=1.0e-08 h_max=7.812500e-03 (h/8) min_step=1.0e-13 max_arclength=14 covectors=0 labels_available=0 inlet_vmin=-8.241447e-01
BACKFLOW field=gaussian eps=4 N=16 inlet_vmin=-8.241447e-01 labels_available=0 seed_backflow=0 backflow_enc_plane=254 backflow_enc_time=90 (inlet backflow is recorded, not an exit condition)
TPLANE m=0 x1=0.5 n_ok=208 mean2=5.277602075684e-01 mean3=3.915793251716e-01 var2=3.335946894451e-02 var3=3.254648161945e-02 mean_tau=9.333682536160e+01 var_tau=1.535102730159e+05 ev_rms=2.069184e+121 ev_max=4.228849e+122
TPLANE m=1 x1=1 n_ok=2 mean2=5.471555259782e-01 mean3=8.178022865934e-01 var2=5.169083485890e-05 var3=2.994323592218e-06 mean_tau=2.065209966599e+00 var_tau=6.212148280041e+00 ev_rms=2.958191e+00 ev_max=2.978173e+00
TTIME n=0 t=0.5 active=256 mean1=2.245329916706e-01 mean2=6.948878722672e-01 mean3=3.142501206255e-01 var1=4.535512001837e-02 var2=1.815640644155e-02 var3=4.404871679713e-02
TTIME n=1 t=1 active=256 mean1=2.327401324034e-01 mean2=6.925196117285e-01 mean3=3.116181398198e-01 var1=4.863793253697e-02 var2=2.419460477608e-02 var3=4.265895032793e-02
TSTATUS pass=plane hist=2,0,254,0,0,0,0 landings_discarded=0 backflow_enc=254 nfev=17548989856
TSTATUS pass=time hist=256,0,0,0,0,0,0 landings_discarded=0 backflow_enc=90 nfev=584714
STATUS track_fail
```
