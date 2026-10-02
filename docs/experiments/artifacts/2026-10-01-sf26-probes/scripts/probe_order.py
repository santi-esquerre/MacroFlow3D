# Feasibility study: eta=1 residual floor vs discretization ORDER, resolution (ell/h) and amplitude (lambda*sigma),
# same continuum Gaussian field, lambda-continuation with warm starts, Anderson on a Laplacian-preconditioned residual map.
# disc: fd2  = the code's scheme (divergence form, harmonic faces, centered 2nd-order gradients/Hessians)
#       fd2c = 2nd-order "consistent" non-divergence form (same stencil family on both sides of L_i = S_i)
#       fd4  = 4th-order centered non-divergence form
#       sp   = pseudo-spectral non-divergence form
import sys, os, time, numpy as np
n=int(sys.argv[1]); disc=sys.argv[2]; sigma2=float(sys.argv[3]); seed=int(sys.argv[4]) if len(sys.argv)>4 else 7
NGEN=int(sys.argv[5]) if len(sys.argv)>5 else 32
ITERS=int(os.environ.get('ITERS','800')); NSTAGES=int(os.environ.get('NSTAGES','10'))
L=1.0; h=L/n; N=n**3; ELL=0.25*L
def sh(u,s,ax): return np.roll(u,-s,axis=ax)
P=lambda w: w-w.mean()
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
# ---- same continuum field at every n: synthesize at NGEN, band-limited upsampling by spectral zero padding ----
rng=np.random.default_rng(seed); hg=L/NGEN
kg=2*np.pi*np.fft.fftfreq(NGEN,d=hg); G1_,G2_,G3_=np.meshgrid(kg,kg,kg,indexing='ij')
Yhat=np.fft.fftn(rng.standard_normal((NGEN,NGEN,NGEN)))*np.exp(-(G1_**2+G2_**2+G3_**2)*ELL**2/8.0)
Y0=np.real(np.fft.ifftn(Yhat)); m0=Y0.mean(); s0=Y0.std()
def upsample(Yh,nf,nt):
    out=np.zeros((nt,nt,nt),dtype=complex); half=nf//2
    dst=lambda i: slice(0,half) if i==0 else slice(nt-half,nt)
    srcs=lambda i: slice(0,half) if i==0 else slice(nf-half,nf)
    for a in (0,1):
        for b in (0,1):
            for c in (0,1): out[dst(a),dst(b),dst(c)]=Yh[srcs(a),srcs(b),srcs(c)]
    return np.real(np.fft.ifftn(out))*(nt**3/nf**3)
Yraw=Y0 if n==NGEN else upsample(Yhat,NGEN,n)
Y=(Yraw-m0)*np.sqrt(sigma2)/s0
# ---- derivative families ----
def D2(u): return [(sh(u,1,a)-sh(u,-1,a))/(2*h) for a in range(3)]
def H2(u):
    Hm={}
    for a in range(3):
        Hm[(a,a)]=(sh(u,1,a)-2*u+sh(u,-1,a))/h**2
        for b in range(a+1,3):
            Hm[(a,b)]=(sh(sh(u,1,a),1,b)-sh(sh(u,1,a),-1,b)-sh(sh(u,-1,a),1,b)+sh(sh(u,-1,a),-1,b))/(4*h*h); Hm[(b,a)]=Hm[(a,b)]
    return Hm
def d4(u,a): return (-sh(u,2,a)+8*sh(u,1,a)-8*sh(u,-1,a)+sh(u,-2,a))/(12*h)
def D4(u): return [d4(u,a) for a in range(3)]
def H4(u):
    Hm={}
    for a in range(3):
        Hm[(a,a)]=(-sh(u,2,a)+16*sh(u,1,a)-30*u+16*sh(u,-1,a)-sh(u,-2,a))/(12*h*h)
        for b in range(a+1,3):
            Hm[(a,b)]=d4(d4(u,b),a); Hm[(b,a)]=Hm[(a,b)]
    return Hm
kv=2*np.pi*np.fft.fftfreq(n,d=h); ko=kv.copy()
if n%2==0: ko[n//2]=0.0
KO=np.meshgrid(ko,ko,ko,indexing='ij'); KE=np.meshgrid(kv,kv,kv,indexing='ij')
k2=KE[0]**2+KE[1]**2+KE[2]**2; k2inv=np.where(k2>0,1.0/np.where(k2>0,k2,1.0),0.0)
def DS(u):
    uh=np.fft.fftn(u); return [np.real(np.fft.ifftn(1j*KO[a]*uh)) for a in range(3)]
def HS(u):
    uh=np.fft.fftn(u); Hm={}
    for a in range(3):
        Hm[(a,a)]=np.real(np.fft.ifftn(-KE[a]*KE[a]*uh))
        for b in range(a+1,3):
            Hm[(a,b)]=np.real(np.fft.ifftn(-KO[a]*KO[b]*uh)); Hm[(b,a)]=Hm[(a,b)]
    return Hm
FAM={'fd2':(D2,H2),'fd2c':(D2,H2),'fd4':(D4,H4),'sp':(DS,HS)}
Dop,Hop=FAM[disc]
g1b=[0.0,1.0,0.0]; g2b=[0.0,0.0,1.0]
def parts(u1,u2):
    d1=Dop(u1); d2=Dop(u2); G1=[g1b[i]+d1[i] for i in range(3)]; G2=[g2b[i]+d2[i] for i in range(3)]
    Ha=Hop(u1); Hb=Hop(u2)
    B=[sum(Hb[(i,j)]*G2x for j,G2x in enumerate(G1))-sum(Ha[(i,j)]*G1x for j,G1x in enumerate(G2)) for i in range(3)]
    c=cross(G1,G2); d=dot(c,c)
    S1=dot(cross(B,G1),c)/d; S2=dot(cross(B,G2),c)/d
    lap1=Ha[(0,0)]+Ha[(1,1)]+Ha[(2,2)]; lap2=Hb[(0,0)]+Hb[(1,1)]+Hb[(2,2)]
    return G1,G2,S1,S2,lap1,lap2,np.sqrt(d)
state={}
def set_lambda(lam):
    q=np.exp(-lam*Y); state['q']=q; state['lam']=lam; state['qr']=np.sqrt((q**2).mean())
    if disc=='fd2':
        qf=[2*q*sh(q,1,a)/(q+sh(q,1,a)) for a in range(3)]; state['qf']=qf
        state['rhs']=[P(sum(g[a]*(qf[a]-sh(qf[a],-1,a))/h for a in range(3))) for g in (g1b,g2b)]
    else:
        state['gY']=Dop(lam*Y)
def A(u):
    qf=state['qf']; out=np.zeros_like(u)
    for a in range(3):
        fl=qf[a]*(sh(u,1,a)-u)/h; out+=-(fl-sh(fl,-1,a))/h
    return out
def resid(u1,u2):
    q=state['q']; G1,G2,S1,S2,lap1,lap2,cm=parts(u1,u2)
    if disc=='fd2':
        F1=A(u1)-P(state['rhs'][0]-q*S1); F2=A(u2)-P(state['rhs'][1]-q*S2)
    else:
        gY=state['gY']; F1=P(-q*(lap1-dot(gY,G1)-S1)); F2=P(-q*(lap2-dot(gY,G2)-S2))
    return F1,F2,cm
def rF(F1,F2): return np.sqrt(((F1**2).mean()+(F2**2).mean())/(2*state['qr']**2))*L
def lapinv(f): return np.real(np.fft.ifftn(-k2inv*np.fft.fftn(f)))
def stage(x,iters,m=8,omega=0.5):
    hist=[]; dX=[]; dG=[]; xp=None; fp=None; best=(np.inf,x.copy(),np.nan); last_improve=0
    for k in range(iters):
        u1=x[:N].reshape(Y.shape); u2=x[N:].reshape(Y.shape)
        F1,F2,cm=resid(u1,u2); r=rF(F1,F2); hist.append(r)
        if not np.isfinite(r): break
        if r<0.99*best[0]: last_improve=k
        if r<best[0]: best=(r,x.copy(),float(cm.min()))
        if r<1e-12 or (k>=200 and k-last_improve>150): break
        q=state['q']; g=np.concatenate([P(u1-omega*lapinv(-F1/q)).ravel(),P(u2-omega*lapinv(-F2/q)).ravel()]); f=g-x
        if xp is not None:
            dX.append(x-xp); dG.append(f-fp)
            if len(dX)>m: dX.pop(0); dG.pop(0)
        xp=x.copy(); fp=f.copy()
        if dG:
            DG=np.array(dG).T; gam,_,_,_=np.linalg.lstsq(DG,f,rcond=None)
            x=g-np.array([dX[i]+dG[i] for i in range(len(dX))]).T@gam
        else: x=g
        x[:N]-=x[:N].mean(); x[N:]-=x[N:].mean()
    return best,len(hist),hist[0]
print(f"[order] n={n} disc={disc} sigma2={sigma2} seed={seed} ell/h={ELL/h:.0f} L/ell={L/ELL:.0f} Ystd={Y.std():.4f} ITERS={ITERS}",flush=True)
x=np.zeros(2*N); t00=time.time()
for lam in [round(0.1*(i+1),1) for i in range(NSTAGES)]:
    set_lambda(lam); t0=time.time(); (rb,xb,cmin),its,r0=stage(x,ITERS)
    print(f"[order] n={n} disc={disc} lam={lam:.1f} lam*sigma={lam*np.sqrt(sigma2):.2f} Kmax/Kmin={np.exp(lam*(Y.max()-Y.min())):.1f} r_F_start={r0:.2e} floor={rb:.3e} its={its} min|c|={cmin:.3f} ({time.time()-t0:.0f}s)",flush=True)
    if np.isfinite(rb): x=xb
print(f"[order] n={n} disc={disc} done ({time.time()-t00:.0f}s)",flush=True)
