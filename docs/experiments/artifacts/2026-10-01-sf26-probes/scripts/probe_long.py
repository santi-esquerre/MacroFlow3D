import sys; sys.argv=['x','24','8','1.0','0.1125']
src=open('solver_probe_gauss.py').read().split('print(f"n={n}')[0]
exec(src)
import time
print(f"n={n} ell/h={ellh} sigma2={sigma2} lambda={lam}: long-horizon probes (same-index)",flush=True)
t0=time.time(); hpt=pseudo_time(False,tau_end=20.0,tol=1e-8); print(f"[SAME] pseudo-time Euler to tau=20: {[(a,f'{b:.1e}') for a,b in hpt]} ({time.time()-t0:.0f}s)",flush=True)
t0=time.time(); hp=picard(False,omega=0.5,iters=3000); print(f"[SAME] Picard w=0.5 (3000): @500={hp[min(500,len(hp)-1)]:.2e} @1500={hp[min(1500,len(hp)-1)]:.2e} final={hp[-1]:.2e} its={len(hp)} ({time.time()-t0:.0f}s)",flush=True)
t0=time.time(); ha=anderson(False,m=8,omega=0.5,iters=3000); print(f"[SAME] Anderson m=8 w=0.5 (3000): final={ha[-1]:.2e} min={min(ha):.2e} its={len(ha)} ({time.time()-t0:.0f}s)",flush=True)
