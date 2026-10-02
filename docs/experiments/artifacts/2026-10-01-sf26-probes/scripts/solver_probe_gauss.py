import numpy as np, sys, time, scipy.sparse as sp, scipy.sparse.linalg as spl
n=int(sys.argv[1]); ellh=float(sys.argv[2]); sigma2=float(sys.argv[3]); lam=float(sys.argv[4]); seed=int(sys.argv[5]) if len(sys.argv)>5 else 7
L=1.0; h=L/n; ell=ellh*h
x=(np.arange(n)+0.5)*h; X1,X2,X3=np.meshgrid(x,x,x,indexing='ij')
def roll(a,s,ax): return np.roll(a,-s,axis=ax)
# periodic Gaussian-covariance field C(r)=sigma2 exp(-(r/ell)^2): spectral synthesis
rng=np.random.default_rng(seed)
kx=2*np.pi*np.fft.fftfreq(n,d=h); K1,K2,K3=np.meshgrid(kx,kx,kx,indexing='ij'); k2=K1**2+K2**2+K3**2
spec=np.exp(-k2*ell**2/8.0)   # sqrt of power spectrum ~ exp(-k^2 ell^2/4)
noise=np.fft.fftn(rng.standard_normal((n,n,n)))
Y=np.real(np.fft.ifftn(noise*spec)); Y-=Y.mean(); Y*=np.sqrt(sigma2)/Y.std()
K=np.exp(lam*Y); q=1/K
def harm(a,b): return 2*a*b/(a+b)
qf=[harm(q,roll(q,1,ax)) for ax in range(3)]
def A(u):
    out=np.zeros_like(u)
    for ax in range(3):
        fl=qf[ax]*(roll(u,1,ax)-u)/h; out+=-(fl-roll(fl,-1,ax))/h
    return out
def divqg(g):
    out=np.zeros_like(q)
    for ax in range(3): out+=g[ax]*(qf[ax]-roll(qf[ax],-1,ax))/h
    return out
def grad(u): return [(roll(u,1,ax)-roll(u,-1,ax))/(2*h) for ax in range(3)]
def d2(u,a,b):
    if a==b: return (roll(u,1,a)-2*u+roll(u,-1,a))/h**2
    return (roll(roll(u,1,a),1,b)-roll(roll(u,1,a),-1,b)-roll(roll(u,-1,a),1,b)+roll(roll(u,-1,a),-1,b))/(4*h*h)
def H(u,g): return [sum(d2(u,i,j)*g[j] for j in range(3)) for i in range(3)]
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dot(a,b): return sum(a[i]*b[i] for i in range(3))
P=lambda w: w-w.mean()
g1b=[0,1.0,0]; g2b=[0,0,1.0]; N=n**3; eps=1e-2  # epsilon 1e-2 like the fixtures, v_rms~1
def sources(u1,u2,crossed):
    G1=[g1b[i]+grad(u1)[i] for i in range(3)]; G2=[g2b[i]+grad(u2)[i] for i in range(3)]
    B=[H(u2,G1)[i]-H(u1,G2)[i] for i in range(3)]
    c=cross(G1,G2); d=dot(c,c)+eps**2
    S1=dot(cross(B,G1),c)/d; S2=dot(cross(B,G2),c)/d
    return (S2,S1) if crossed else (S1,S2)
rhs1=P(divqg(g1b)); rhs2=P(divqg(g2b))
def resid(u1,u2,crossed):
    S1,S2=sources(u1,u2,crossed); return A(u1)-P(rhs1-q*S1), A(u2)-P(rhs2-q*S2)
def rF(F1,F2):
    qr=np.sqrt((q**2).mean()); return np.sqrt(((F1**2).mean()/qr**2+(F2**2).mean()/qr**2)/2)*L
idx=np.arange(N).reshape(n,n,n); rows=[];cols=[];vals=[]
for ax in range(3):
    qp=qf[ax]; qm=np.roll(qf[ax],1,axis=ax); ip=np.roll(idx,-1,axis=ax); im=np.roll(idx,1,axis=ax)
    rows+=[idx.ravel()]*3; cols+=[idx.ravel(),ip.ravel(),im.ravel()]; vals+=[((qp+qm)/h**2).ravel(),(-qp/h**2).ravel(),(-qm/h**2).ravel()]
A_sp=sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(N,N)); ones=np.ones(N)/np.sqrt(N)
pin=sp.csr_matrix(([float(A_sp.diagonal().mean())],([0],[0])),shape=(N,N))
lu=spl.splu((A_sp+pin).tocsc())
def solveA(b):
    bb=b.ravel()-b.mean(); xx=lu.solve(bb); return (xx-xx.mean()).reshape(b.shape)
def picard(crossed,omega=0.25,iters=1500,tol=1e-6):
    u1=np.zeros_like(q); u2=np.zeros_like(q); hist=[]
    for k in range(iters):
        F1,F2=resid(u1,u2,crossed); r=rF(F1,F2); hist.append(r)
        if r<tol or not np.isfinite(r): break
        S1,S2=sources(u1,u2,crossed)
        v1=solveA(rhs1-q*S1); v2=solveA(rhs2-q*S2)
        u1=P((1-omega)*u1+omega*v1); u2=P((1-omega)*u2+omega*v2)
    return hist
def anderson(crossed,m=5,omega=0.25,iters=800,tol=1e-6):
    xv=np.zeros(2*N); hist=[]; dX=[]; dG=[]; xprev=None; fprev=None
    def G(xv):
        u1=xv[:N].reshape(q.shape); u2=xv[N:].reshape(q.shape); S1,S2=sources(u1,u2,crossed)
        v1=solveA(rhs1-q*S1); v2=solveA(rhs2-q*S2)
        return np.concatenate([P((1-omega)*u1+omega*v1).ravel(),P((1-omega)*u2+omega*v2).ravel()])
    for k in range(iters):
        u1=xv[:N].reshape(q.shape); u2=xv[N:].reshape(q.shape); F1,F2=resid(u1,u2,crossed); r=rF(F1,F2); hist.append(r)
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
def pseudo_time(crossed,tau_end,tol=1e-8):
    u1=np.zeros_like(q); u2=np.zeros_like(q); hist=[]; dt=0.9*h*h/6; steps=int(tau_end/dt); t=0
    for s in range(steps):
        F1,F2=resid(u1,u2,crossed); r=rF(F1,F2)
        if s%max(1,steps//8)==0: hist.append((round(t,3),r))
        if r<tol or not np.isfinite(r): hist.append((round(t,3),r)); break
        u1=P(u1-dt*K*F1); u2=P(u2-dt*K*F2); t+=dt
    return hist
print(f"n={n} ell/h={ellh} sigma2={sigma2} lambda={lam} seed={seed} Ystd={Y.std():.3f} Kmax/Kmin={K.max()/K.min():.2f}")
for crossed in (False,True):
    lab="CROSSED" if crossed else "SAME-INDEX"
    t0=time.time(); hp=picard(crossed); print(f"[{lab}] Picard w=0.25 (tol 1e-6, max 1500): its={len(hp)} r_F @50={hp[min(50,len(hp)-1)]:.2e} @200={hp[min(200,len(hp)-1)]:.2e} @500={hp[min(500,len(hp)-1)]:.2e} final={hp[-1]:.2e} ({time.time()-t0:.0f}s)",flush=True)
    t0=time.time(); ha=anderson(crossed); print(f"[{lab}] Anderson m=5: its={len(ha)} final={ha[-1]:.2e} min={min(ha):.2e} ({time.time()-t0:.0f}s)",flush=True)
    t0=time.time(); hpt=pseudo_time(crossed,tau_end=1.5); print(f"[{lab}] pseudo-time Euler to tau=1.5: {[(a,f'{b:.1e}') for a,b in hpt]} ({time.time()-t0:.0f}s)",flush=True)
