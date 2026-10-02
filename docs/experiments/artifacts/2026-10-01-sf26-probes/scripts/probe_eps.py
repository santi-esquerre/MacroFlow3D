import sys, time
sys.argv=['x','24','8','1.0','0.1125']
src=open('solver_probe_gauss.py').read().split('print(f"n={n}')[0]
exec(src)
import numpy as np
def sources_eps(u1,u2,epsv):
    G1=[g1b[i]+grad(u1)[i] for i in range(3)]; G2=[g2b[i]+grad(u2)[i] for i in range(3)]
    B=[H(u2,G1)[i]-H(u1,G2)[i] for i in range(3)]
    c=cross(G1,G2); d=dot(c,c)+epsv**2
    return dot(cross(B,G1),c)/d, dot(cross(B,G2),c)/d
def resid_eps(u1,u2,epsv):
    S1,S2=sources_eps(u1,u2,epsv); return A(u1)-P(rhs1-q*S1), A(u2)-P(rhs2-q*S2)
def anderson_eps(epsv,m=8,omega=0.5,iters=1500,tol=1e-9):
    xv=np.zeros(2*N); hist=[]; dX=[]; dG=[]; xprev=None; fprev=None
    def G(xv):
        u1=xv[:N].reshape(q.shape); u2=xv[N:].reshape(q.shape); S1,S2=sources_eps(u1,u2,epsv)
        v1=solveA(rhs1-q*S1); v2=solveA(rhs2-q*S2)
        return np.concatenate([P((1-omega)*u1+omega*v1).ravel(),P((1-omega)*u2+omega*v2).ravel()])
    for k in range(iters):
        u1=xv[:N].reshape(q.shape); u2=xv[N:].reshape(q.shape); F1,F2=resid_eps(u1,u2,epsv); r=rF(F1,F2); hist.append(r)
        if r<tol or not np.isfinite(r): break
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
    return hist
print(f"[eps-sweep] n={n} ell/h={ellh} sigma2={sigma2} lambda={lam} seed={seed} (same-index, Anderson m=8 w=0.5, 1500 its)",flush=True)
for epsv in (1e-2,3e-3,1e-3,0.0):
    t0=time.time(); h_=anderson_eps(epsv); print(f"[eps-sweep] eps={epsv:g}: its={len(h_)} final r_F={h_[-1]:.3e} min={min(h_):.3e} @200={h_[min(200,len(h_)-1)]:.2e} @800={h_[min(800,len(h_)-1)]:.2e} ({time.time()-t0:.0f}s)",flush=True)
