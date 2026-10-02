# Does the eta=1 residual floor come from the 2nd-order FD truncation structure?  Compare, on the SAME rough Gaussian field:
#   (FD)  the code's discretization (divergence form, harmonic faces, centered gradients/Hessians)
#   (SP)  a pseudo-spectral discretization of  L_i = lap(psi_i) - grad(lnK).grad(psi_i) = S_i  (all derivatives by FFT)
# Methods: Anderson to its floor, then dense min-norm Newton (exact linear solve) from that state.  eps = 0.
import sys, time, numpy as np
n,ellh,sigma2,lam=int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]); seed=int(sys.argv[5]) if len(sys.argv)>5 else 7
dense = (len(sys.argv)>6 and sys.argv[6]=='dense')
sys.argv=['x',str(n),str(ellh),str(sigma2),str(lam),str(seed)]
src=open('solver_probe_gauss.py').read().split('print(f"n={n}')[0].replace("eps=1e-2","eps=0.0")
exec(src)
# ---------- pseudo-spectral machinery ----------
kv=2*np.pi*np.fft.fftfreq(n,d=h); kv_odd=kv.copy()
if n%2==0: kv_odd[n//2]=0.0
KX=np.meshgrid(kv_odd,kv_odd,kv_odd,indexing='ij'); KE=np.meshgrid(kv,kv,kv,indexing='ij')
k2s=KE[0]**2+KE[1]**2+KE[2]**2; k2inv=np.where(k2s>0,1.0/np.where(k2s>0,k2s,1),0.0)
def sgrad(u):
    uh=np.fft.fftn(u); return [np.real(np.fft.ifftn(1j*KX[a]*uh)) for a in range(3)]
def shess(u):
    uh=np.fft.fftn(u); Hm={}
    for a in range(3):
        for b in range(a,3):
            ka=KE[a] if a==b else KX[a]; kb=KE[b] if a==b else KX[b]
            Hm[(a,b)]=np.real(np.fft.ifftn(-ka*kb*uh)); Hm[(b,a)]=Hm[(a,b)]
    return Hm
def slap(u): return np.real(np.fft.ifftn(-k2s*np.fft.fftn(u)))
gY=sgrad(lam*Y)
def sp_resid(u1,u2):
    d1=sgrad(u1); d2=sgrad(u2); G1=[g1b[i]+d1[i] for i in range(3)]; G2=[g2b[i]+d2[i] for i in range(3)]
    H1=shess(u1); H2=shess(u2)
    B=[sum(H2[(i,j)]*G1[j] for j in range(3))-sum(H1[(i,j)]*G2[j] for j in range(3)) for i in range(3)]
    c=cross(G1,G2); d=dot(c,c)
    S1=dot(cross(B,G1),c)/d; S2=dot(cross(B,G2),c)/d
    R1=slap(u1)-dot(gY,G1)-S1; R2=slap(u2)-dot(gY,G2)-S2
    return P(-q*R1), P(-q*R2)          # F_i = -q (L_i - S_i): same scaling as the FD residual A u - rhs + q S
def fd_resid(u1,u2): return resid(u1,u2,False)
def vec(a,b): return np.concatenate([a.ravel(),b.ravel()])
def unvec(x): return x[:N].reshape(q.shape), x[N:].reshape(q.shape)
def rFv(F): a,b=unvec(F); return rF(a,b)
# preconditioned residual maps for Anderson
def fd_map(x,omega=0.5):
    u1,u2=unvec(x); S1,S2=sources(u1,u2,False); v1=solveA(rhs1-q*S1); v2=solveA(rhs2-q*S2)
    return vec(P((1-omega)*u1+omega*v1),P((1-omega)*u2+omega*v2))
def sp_map(x,omega=0.5):
    u1,u2=unvec(x); F1,F2=sp_resid(u1,u2)       # F = -q(L-S); Laplacian-preconditioned correction: du = lap^-1( (L-S) ) ~ -lap^-1(F/q)
    c1=np.real(np.fft.ifftn(-k2inv*np.fft.fftn(-F1/q))); c2=np.real(np.fft.ifftn(-k2inv*np.fft.fftn(-F2/q)))
    return vec(P(u1-omega*c1),P(u2-omega*c2))
def anderson_generic(gmap,rfun,m=8,iters=1500,tol=1e-13):
    x=np.zeros(2*N); hist=[]; dX=[]; dG=[]; xp=None; fp=None; best=(1e99,None)
    for k in range(iters):
        F1,F2=rfun(*unvec(x)); r=rF(F1,F2); hist.append(r)
        if np.isfinite(r) and r<best[0]: best=(r,x.copy())
        if r<tol or not np.isfinite(r): break
        g=gmap(x); f=g-x
        if xp is not None:
            dX.append(x-xp); dG.append(f-fp)
            if len(dX)>m: dX.pop(0); dG.pop(0)
        xp=x.copy(); fp=f.copy()
        if dG:
            DG=np.array(dG).T; gam,_,_,_=np.linalg.lstsq(DG,f,rcond=None)
            x=g-np.array([dX[i]+dG[i] for i in range(len(dX))]).T@gam
        else: x=g
        x[:N]-=x[:N].mean(); x[N:]-=x[N:].mean()
    return hist,best[1]
def dense_newton(rfun,x,label,steps=8):
    def Fv(v): a,b=rfun(*unvec(v)); return vec(a,b)
    F=Fv(x)
    for s in range(steps):
        t0=time.time(); J=np.zeros((2*N,2*N)); dl=1e-6
        for j in range(2*N):
            e=np.zeros(2*N); e[j]=dl; J[:,j]=(Fv(x+e)-Fv(x-e))/(2*dl)
        for blk in range(2):
            e=np.zeros(2*N); e[blk*N:(blk+1)*N]=1/np.sqrt(N); J+=np.outer(e,e)
        d,_,rank,sv=np.linalg.lstsq(J,-F,rcond=1e-13); rel=sv/sv[0]
        f0=np.linalg.norm(F); al=1.0
        for _ in range(30):
            xn=x+al*d; xn[:N]-=xn[:N].mean(); xn[N:]-=xn[N:].mean(); Fn=Fv(xn)
            if np.linalg.norm(Fn)<=(1-1e-4*al)*f0: break
            al*=0.5
        x=xn; F=Fn
        print(f"[{label}] newton step {s}: rank={rank} #sv<1e-3={(rel<1e-3).sum()} #<1e-5={(rel<1e-5).sum()} min_rel_sv={rel[-1]:.1e} |d|={np.linalg.norm(d):.2e} alpha={al:g} r_F={rFv(F):.3e} ({time.time()-t0:.0f}s)",flush=True)
        if rFv(F)<1e-12: break
    return x
print(f"[spectral-probe] n={n} ell/h={ellh} L/ell={n/ellh:.1f} sigma2={sigma2} lambda={lam} seed={seed} Kmax/Kmin={K.max()/K.min():.1f} dense={dense}",flush=True)
for label,gmap,rfun in (("FD",fd_map,fd_resid),("SP",sp_map,sp_resid)):
    t0=time.time(); hist,xb=anderson_generic(gmap,rfun)
    print(f"[{label}] Anderson m=8 w=0.5: its={len(hist)} final={hist[-1]:.3e} min={min(hist):.3e} @200={hist[min(200,len(hist)-1)]:.2e} @800={hist[min(800,len(hist)-1)]:.2e} ({time.time()-t0:.0f}s)",flush=True)
    if dense and xb is not None and min(hist)>1e-12: dense_newton(rfun,xb,label)
print("[spectral-probe] done",flush=True)
