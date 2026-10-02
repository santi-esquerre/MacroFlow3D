import sys, time, numpy as np
n,ellh,sigma2,lam,epsv=int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]); seed=int(sys.argv[6]) if len(sys.argv)>6 else 7
sys.argv=['x',str(n),str(ellh),str(sigma2),str(lam),str(seed)]
src=open('solver_probe_gauss.py').read().split('print(f"n={n}')[0].replace("eps=1e-2","eps=EPSV")
g={'EPSV':epsv}; exec(src,g)
t0=time.time(); ha=g['anderson'](False,m=8,omega=0.5,iters=2000,tol=1e-9)
print(f"[ellh] n={n} ell/h={ellh} L/ell={n/ellh:.1f} sigma2={sigma2} lambda={lam} eps={epsv} seed={seed}: Anderson m=8 w=0.5 (2000 its) final r_F={ha[-1]:.3e} min={min(ha):.3e} @500={ha[min(500,len(ha)-1)]:.2e} @1000={ha[min(1000,len(ha)-1)]:.2e} ({time.time()-t0:.0f}s)",flush=True)
