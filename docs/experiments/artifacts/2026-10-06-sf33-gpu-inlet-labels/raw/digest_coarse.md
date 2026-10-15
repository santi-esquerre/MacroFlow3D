### ladder_0.25/N32.log

- STAGE_END field=gaussian eps=0.25 N=32 cand=i1o4 status=converged its=18 r_F=1.073e-14 r_out=4.281e-16 -> accepted
- LINEAR solves=18 its_total=257 its_max=50 its_median=9.5 t_linear_total=1.75s non_converged=0
- COARSE build count=18 K=4096 kl=639 ku=639 t_assembly mean=0.027s max=0.028s | t_lu mean=0.579s max=0.638s total=10.4s | t_cond mean=0.035s | rcond_est min=1.606e-05
- COARSE apply applications_total=275 t_apply_avg mean=6.002ms max=6.626ms | t_host_solve_avg mean=5.090ms max=5.725ms
- host max RSS (/usr/bin/time -v) = 680116 kB = 0.65 GiB

### ladder_0.25/N64.log

- STAGE_END field=gaussian eps=0.25 N=64 cand=i1o4 status=linear_failure its=14 r_F=2.854e-02 r_out=9.583e-05 -> FAILED
- STAGE_END field=gaussian eps=0.125 N=64 cand=i1o4 status=converged its=23 r_F=2.323e-14 r_out=4.251e-16 -> accepted
- STAGE_END field=gaussian eps=0.25 N=64 cand=i1o4 status=converged its=23 r_F=5.839e-14 r_out=1.256e-15 -> accepted
- LINEAR solves=60 its_total=4253 its_max=1000 its_median=3.5 t_linear_total=231.25s non_converged=1
- COARSE build count=60 K=16384 kl=1279 ku=1279 t_assembly mean=0.138s max=0.161s | t_lu mean=10.392s max=12.714s total=623.5s | t_cond mean=0.321s | rcond_est min=2.488e-06
- COARSE apply applications_total=4337 t_apply_avg mean=49.517ms max=54.855ms | t_host_solve_avg mean=47.417ms max=52.782ms
- host max RSS (/usr/bin/time -v) = 1167356 kB = 1.11 GiB

### ladder_0.25/N128.log

- STAGE_END field=gaussian eps=0.25 N=128 cand=i1o4 status=linear_failure its=13 r_F=1.557e-01 r_out=7.972e-05 -> FAILED
- STAGE_END field=gaussian eps=0.125 N=128 cand=i1o4 status=stagnation its=20 r_F=1.081e-02 r_out=8.371e-06 -> FAILED
- STAGE_END field=gaussian eps=0.0625 N=128 cand=i1o4 status=stagnation its=20 r_F=5.732e-03 r_out=5.509e-07 -> FAILED
- STAGE_END field=gaussian eps=0.03125 N=128 cand=i1o4 status=stagnation its=20 r_F=2.941e-03 r_out=3.975e-07 -> FAILED
- STAGE_END field=gaussian eps=0.015625 N=128 cand=i1o4 status=stagnation its=20 r_F=1.485e-03 r_out=4.880e-08 -> FAILED
- STAGE_END field=gaussian eps=0.25 N=128 cand=i1o4 status=linear_failure its=13 r_F=1.557e-01 r_out=7.972e-05 (final attempt) -> FAILED
- LINEAR solves=106 its_total=1226 its_max=300 its_median=2 t_linear_total=546.52s non_converged=2
- COARSE build count=106 K=65536 kl=2559 ku=2559 t_assembly mean=1.037s max=1.245s | t_lu mean=274.680s max=343.839s total=29116.1s | t_cond mean=3.358s | rcond_est min=4.307e-06
- COARSE apply applications_total=1336 t_apply_avg mean=393.018ms max=424.945ms | t_host_solve_avg mean=384.611ms max=415.582ms
- host max RSS (/usr/bin/time -v) = 5021672 kB = 4.79 GiB

### prod32_05/N32.log

- STAGE_END field=gaussian eps=0.25 N=32 cand=i1o4 status=converged its=18 r_F=1.073e-14 r_out=4.281e-16 -> accepted
- STAGE_END field=gaussian eps=0.5 N=32 cand=i1o4 status=converged its=19 r_F=4.714e-14 r_out=1.285e-15 -> accepted
- LINEAR solves=37 its_total=6288 its_max=2000 its_median=20 t_linear_total=45.36s non_converged=0
- COARSE build count=37 K=4096 kl=639 ku=639 t_assembly mean=0.025s max=0.027s | t_lu mean=0.604s max=0.662s total=22.4s | t_cond mean=0.035s | rcond_est min=1.158e-06
- COARSE apply applications_total=6372 t_apply_avg mean=6.134ms max=7.727ms | t_host_solve_avg mean=5.258ms max=6.808ms
- host max RSS (/usr/bin/time -v) = 678632 kB = 0.65 GiB

### oracle32/N16.log

- STAGE_END field=cells eps=0.25 N=16 cand=i1o4 status=converged its=12 r_F=1.587e-15 r_out=6.778e-17 -> accepted
- STAGE_END field=cells eps=0.5 N=16 cand=i1o4 status=converged its=12 r_F=2.555e-15 r_out=1.762e-16 -> accepted
- STAGE_END field=cells eps=1 N=16 cand=i1o4 status=converged its=12 r_F=3.762e-15 r_out=2.487e-16 -> accepted
- LINEAR solves=36 its_total=208 its_max=22 its_median=4 t_linear_total=0.15s non_converged=0
- COARSE build count=36 K=1024 kl=319 ku=319 t_assembly mean=0.014s max=0.016s | t_lu mean=0.027s max=0.031s total=1.0s | t_cond mean=0.003s | rcond_est min=9.898e-05
- COARSE apply applications_total=244 t_apply_avg mean=0.619ms max=0.645ms | t_host_solve_avg mean=0.358ms max=0.378ms
- host max RSS (/usr/bin/time -v) = 624756 kB = 0.60 GiB

### oracle32/N24.log

- STAGE_END field=cells eps=0.25 N=24 cand=i1o4 status=converged its=15 r_F=3.945e-15 r_out=1.467e-16 -> accepted
- STAGE_END field=cells eps=0.5 N=24 cand=i1o4 status=converged its=15 r_F=3.882e-15 r_out=1.611e-16 -> accepted
- STAGE_END field=cells eps=1 N=24 cand=i1o4 status=converged its=15 r_F=6.310e-15 r_out=3.231e-16 -> accepted
- LINEAR solves=45 its_total=343 its_max=50 its_median=4 t_linear_total=0.91s non_converged=0
- COARSE build count=45 K=2304 kl=479 ku=479 t_assembly mean=0.009s max=0.009s | t_lu mean=0.166s max=0.178s total=7.5s | t_cond mean=0.011s | rcond_est min=3.327e-05
- COARSE apply applications_total=388 t_apply_avg mean=2.046ms max=2.085ms | t_host_solve_avg mean=1.422ms max=1.446ms
- host max RSS (/usr/bin/time -v) = 649464 kB = 0.62 GiB

### oracle32/N32.log

- STAGE_END field=cells eps=0.25 N=32 cand=i1o4 status=converged its=18 r_F=4.848e-15 r_out=1.545e-16 -> accepted
- STAGE_END field=cells eps=0.5 N=32 cand=i1o4 status=converged its=18 r_F=6.005e-15 r_out=2.278e-16 -> accepted
- STAGE_END field=cells eps=1 N=32 cand=i1o4 status=converged its=18 r_F=9.745e-15 r_out=4.184e-16 -> accepted
- LINEAR solves=54 its_total=566 its_max=91 its_median=4 t_linear_total=4.06s non_converged=0
- COARSE build count=54 K=4096 kl=639 ku=639 t_assembly mean=0.025s max=0.027s | t_lu mean=0.573s max=0.620s total=30.9s | t_cond mean=0.034s | rcond_est min=1.646e-05
- COARSE apply applications_total=620 t_apply_avg mean=6.081ms max=6.376ms | t_host_solve_avg mean=5.210ms max=5.470ms
- host max RSS (/usr/bin/time -v) = 678672 kB = 0.65 GiB
