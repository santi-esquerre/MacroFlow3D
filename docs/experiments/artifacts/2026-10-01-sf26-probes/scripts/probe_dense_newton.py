# Discriminating probe: is the eta=1 residual floor on a rough field a METHOD/rate limit or a DISCRETE INCONSISTENCY?
# (a) Anderson to its floor; (b) dense FD Jacobian + min-norm lstsq Newton with Armijo from that state; (c) singular-value summary.
import sys, time
n,ellh,sigma2,lam=int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]); seed=int(sys.argv[5]) if len(sys.argv)>5 else 7
sys.argv=['x',str(n),str(ellh),str(sigma2),str(lam),str(seed)]
src=open('solver_probe_gauss.py').read().split('print(f"n={n}')[0]
exec(src)
import numpy as np
print(f"[dense-newton] n={n} ell/h={ellh} sigma2={sigma2} lambda={lam} seed={seed} 2N={2*N}",flush=True)
t0=time.time(); ha=anderson(False,m=8,omega=0.5,iters=1500)
print(f"[dense-newton] Anderson floor: final r_F={ha[-1]:.3e} min={min(ha):.3e} its={len(ha)} ({time.time()-t0:.0f}s)",flush=True)
# rebuild the Anderson end state (anderson() does not return it): rerun deterministic to capture x -> cheaper: re-implement quickly
def anderson_state(crossed,m=8,omega=0.5,iters=1500):
    xv=np.zeros(2*N); dX=[]; dG=[]; xprev=None; fprev=None
    def G(xv):
        u1=xv[:N].reshape(q.shape); u2=xv[N:].reshape(q.shape); S1,S2=sources(u1,u2,crossed)
        v1=solveA(rhs1-q*S1); v2=solveA(rhs2-q*S2)
        return np.concatenate([P((1-omega)*u1+omega*v1).ravel(),P((1-omega)*u2+omega*v2).ravel()])
    for k in range(iters):
        g=G(xv); f=g-xv
        if xprev is not None:
            dX.append(xv-xprev); dG.append(f-fprev)
            if len(dX)>m: dX.pop(0); dG.pop(0)
        xprev=xv.copy(); fprev=f.copy()
        if dG:
            DG=np.array(dG).T; gamma,_,_,_=np.linalg.lstsq(DG,f,rcond=None)
            xv=g-np.array([dX[i]+dG[i] for i in range(len(dX))]).T@gamma
        else: xv=g
        xv[:N]-=xv[:N].mean(); xv[N:]-=xv[N:].mean()
    return xv
x=anderson_state(False)
def Fv(v):
    a,b=resid(v[:N].reshape(q.shape),v[N:].reshape(q.shape),False); return np.concatenate([a.ravel(),b.ravel()])
def rFv(F): return rF(F[:N].reshape(q.shape),F[N:].reshape(q.shape))
F=Fv(x); print(f"[dense-newton] start state r_F={rFv(F):.3e}",flush=True)
for step in range(6):
    t0=time.time(); Jm=np.zeros((2*N,2*N)); delta=1e-6
    for j in range(2*N):
        e=np.zeros(2*N); e[j]=delta; Jm[:,j]=(Fv(x+e)-Fv(x-e))/(2*delta)
    for blk in range(2):
        e=np.zeros(2*N); e[blk*N:(blk+1)*N]=1/np.sqrt(N); Jm+=np.outer(e,e)
    tj=time.time()-t0; t0=time.time()
    d,res,rank,sv=np.linalg.lstsq(Jm,-F,rcond=1e-12)
    rel=sv/sv[0]
    print(f"[dense-newton] step {step}: J assembled ({tj:.0f}s), lstsq ({time.time()-t0:.0f}s) rank={rank} smallest rel sv={np.array2string(rel[-6:],precision=2)} #<1e-3={(rel<1e-3).sum()} #<1e-4={(rel<1e-4).sum()} #<1e-6={(rel<1e-6).sum()}  |J d + F|/|F| (lin. residual)={np.linalg.norm(Jm@d+F)/np.linalg.norm(F):.2e}",flush=True)
    f0=np.linalg.norm(F); alpha=1.0
    for _ in range(25):
        xn=x+alpha*d; xn[:N]-=xn[:N].mean(); xn[N:]-=xn[N:].mean(); Fn=Fv(xn)
        if np.linalg.norm(Fn)<=(1-1e-4*alpha)*f0: break
        alpha*=0.5
    x=xn; F=Fn
    print(f"[dense-newton] step {step}: alpha={alpha} r_F={rFv(F):.3e} |d|={np.linalg.norm(d):.3e}",flush=True)
    if rFv(F)<1e-11: break
print("[dense-newton] done",flush=True)
