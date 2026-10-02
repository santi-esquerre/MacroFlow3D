import numpy as np, sys, time, scipy.sparse as sp, scipy.sparse.linalg as spl
n=int(sys.argv[1]) if len(sys.argv)>1 else 12
amp=float(sys.argv[2]) if len(sys.argv)>2 else 1.0   # scales Phi/Theta amplitudes (0.1*amp) and k1 log-amplitude
src=open('jac_spectrum.py').read().split('spectrum("same-index @exact pair"')[0]
src=src.replace("aP,aT=0.1,0.1","aP,aT=0.1*amp,0.1*amp").replace("k1=np.exp(0.7*np.sin(2*np.pi*X1)+0.3*np.cos(4*np.pi*X1))","k1=np.exp(amp*(0.7*np.sin(2*np.pi*X1)+0.3*np.cos(4*np.pi*X1)))")
exec(src)
N=n**3
# sparse A (periodic 7-point, harmonic q faces) for the Picard block solves
def build_A():
    idx=np.arange(N).reshape(n,n,n); rows=[];cols=[];vals=[]
    for ax in range(3):
        qp=qf[ax]; qm=np.roll(qf[ax],1,axis=ax)
        ip=np.roll(idx,-1,axis=ax); im=np.roll(idx,1,axis=ax)
        rows+= [idx.ravel()]*3; cols+=[idx.ravel(),ip.ravel(),im.ravel()]
        vals+=[((qp+qm)/h**2).ravel(),(-qp/h**2).ravel(),(-qm/h**2).ravel()]
    A=sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(N,N))
    # pin nullspace via mean-zero: add ones outer (rank-1) -> SPD on mean-zero space
    return A
A_sp=build_A(); ones=np.ones(N)/np.sqrt(N)
A_reg=A_sp+sp.csr_matrix(np.outer(ones,ones))
lu=spl.splu(A_reg.tocsc())
def solveA(b):
    bb=b.ravel()-b.mean(); x=lu.solve(bb); return (x-x.mean()).reshape(b.shape)
def sources(u1,u2,crossed):
    G1=[g1b[i]+grad(u1)[i] for i in range(3)]; G2=[g2b[i]+grad(u2)[i] for i in range(3)]
    B=[H(u2,G1)[i]-H(u1,G2)[i] for i in range(3)]
    c=cross(G1,G2); d=dot(c,c)
    S1=dot(cross(B,G1),c)/d; S2=dot(cross(B,G2),c)/d
    return (S2,S1) if crossed else (S1,S2)
rhs1=P(divqg(g1b)); rhs2=P(divqg(g2b))
def resid(u1,u2,crossed):
    S1,S2=sources(u1,u2,crossed); return A(u1)-P(rhs1-q*S1), A(u2)-P(rhs2-q*S2)
def rF(F1,F2):
    qr=np.sqrt((q**2).mean()); return np.sqrt(((F1**2).mean()/qr**2+(F2**2).mean()/qr**2)/2)*L
def picard(crossed,omega=0.25,iters=3000,tol=1e-10):
    u1=np.zeros_like(q); u2=np.zeros_like(q); hist=[]
    for k in range(iters):
        F1,F2=resid(u1,u2,crossed); r=rF(F1,F2); hist.append(r)
        if r<tol: break
        S1,S2=sources(u1,u2,crossed)
        v1=solveA(rhs1-q*S1).reshape(u1.shape); v2=solveA(rhs2-q*S2).reshape(u1.shape)
        u1=P((1-omega)*u1+omega*v1); u2=P((1-omega)*u2+omega*v2)
    return hist
def anderson(crossed,m=5,omega=0.25,iters=1500,tol=1e-10):
    # Walker-Ni AA on the damped Picard map G(u)
    x=np.zeros(2*N); hist=[]; dX=[]; dG=[]; gprev=None; xprev=None
    def G(x):
        u1=x[:N].reshape(q.shape); u2=x[N:].reshape(q.shape)
        S1,S2=sources(u1,u2,crossed)
        v1=solveA(rhs1-q*S1); v2=solveA(rhs2-q*S2)
        return np.concatenate([P(((1-omega)*u1).ravel().reshape(q.shape)+omega*v1.reshape(q.shape)).ravel(),P(((1-omega)*u2).reshape(q.shape)+omega*v2.reshape(q.shape)).ravel()])
    for k in range(iters):
        u1=x[:N].reshape(q.shape); u2=x[N:].reshape(q.shape); F1,F2=resid(u1,u2,crossed); r=rF(F1,F2); hist.append(r)
        if r<tol: break
        g=G(x); f=g-x
        if xprev is not None:
            dX.append(x-xprev); dG.append(f-fprev)
            if len(dX)>m: dX.pop(0); dG.pop(0)
        xprev=x.copy(); fprev=f.copy()
        if dG:
            DG=np.array(dG).T; gamma,_,_,_=np.linalg.lstsq(DG,f,rcond=None)
            x=g-np.array([ (dX[i]+dG[i]) for i in range(len(dX))]).T@gamma
        else: x=g
        x[:N]-=x[:N].mean(); x[N:]-=x[N:].mean()
    return hist
def newton_full(crossed,iters=40,tol=1e-12):
    u1=np.zeros_like(q); u2=np.zeros_like(q); hist=[]
    def Fv(v): a,b=resid(v[:N].reshape(q.shape),v[N:].reshape(q.shape),crossed); return np.concatenate([a.ravel(),b.ravel()])
    x=np.zeros(2*N)
    for k in range(iters):
        F=Fv(x); r=rF(F[:N].reshape(q.shape),F[N:].reshape(q.shape)); hist.append(r)
        if r<tol: break
        Jm=np.zeros((2*N,2*N)); delta=1e-6
        for j in range(2*N):
            e=np.zeros(2*N); e[j]=delta; Jm[:,j]=(Fv(x+e)-Fv(x-e))/(2*delta)
        for blk in range(2):
            e=np.zeros(2*N); e[blk*N:(blk+1)*N]=1/np.sqrt(N); Jm+=np.outer(e,e)
        d,_,rank,sv=np.linalg.lstsq(Jm,-F,rcond=1e-10)   # min-norm step on the near-singular J
        # Armijo
        alpha=1.0; f0=np.linalg.norm(F)
        for _ in range(20):
            xn=x+alpha*d; xn[:N]-=xn[:N].mean(); xn[N:]-=xn[N:].mean()
            if np.linalg.norm(Fv(xn))<=(1-1e-4*alpha)*f0: break
            alpha*=0.5
        x=xn; hist.append(('step',k,alpha,rank))
    return hist
def pseudo_time(crossed,tau_end,tol=1e-10):
    u1=np.zeros_like(q); u2=np.zeros_like(q); hist=[]
    # explicit Euler on du/dtau = -k F ; stability: k*A ~ unit-coefficient Laplacian -> dt <= h^2/6 (safety 0.9)
    dt=0.9*h*h/6; steps=int(tau_end/dt); K=1/q; t=0
    for s in range(steps):
        F1,F2=resid(u1,u2,crossed)
        if s%max(1,steps//10)==0 or s==steps-1:
            hist.append((round(t,4),rF(F1,F2)))
        if rF(F1,F2)<tol: hist.append((round(t,4),rF(F1,F2))); break
        u1=P(u1-dt*K*F1); u2=P(u2-dt*K*F2); t+=dt
    return hist
for crossed in (False,True):
    lab="CROSSED" if crossed else "SAME-INDEX"
    t0=time.time(); hp=picard(crossed); print(f"[{lab}] damped Picard w=0.25: its={len(hp)} r_F: start={hp[0]:.2e} @100={hp[min(100,len(hp)-1)]:.2e} @500={hp[min(500,len(hp)-1)]:.2e} final={hp[-1]:.2e} ({time.time()-t0:.0f}s)")
    t0=time.time(); ha=anderson(crossed); print(f"[{lab}] Anderson m=5 on damped Picard: its={len(ha)} final r_F={ha[-1]:.2e} min={min(ha):.2e} ({time.time()-t0:.0f}s)")
    t0=time.time(); hn=newton_full(crossed); rs=[x for x in hn if not isinstance(x,tuple)]; print(f"[{lab}] Newton (dense, min-norm lstsq, Armijo): its={len(rs)} r_F per step: {[f'{v:.1e}' for v in rs]} ({time.time()-t0:.0f}s)")
    t0=time.time(); hpt=pseudo_time(crossed, tau_end=2.0); print(f"[{lab}] explicit pseudo-time (Euler, dt=0.9h^2/6) to tau=2: {[(a,f'{b:.1e}') for a,b in hpt]} ({time.time()-t0:.0f}s)")
