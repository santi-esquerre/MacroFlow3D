import numpy as np, sys, time
n=int(sys.argv[1]) if len(sys.argv)>1 else 10
L=1.0; h=L/n
x=(np.arange(n)+0.5)*h
X1,X2,X3=np.meshgrid(x,x,x,indexing='ij')
def roll(a,s,ax): return np.roll(a,-s,axis=ax)  # roll(a,+1,ax)[i]=a[i+1]
# exact pair
aP,aT=0.1,0.1
Phi=lambda z: aP*np.sin(2*np.pi*z); dPhi=lambda z: aP*2*np.pi*np.cos(2*np.pi*z)
Tht=lambda y: aT*np.cos(2*np.pi*y); dTht=lambda y: -aT*2*np.pi*np.sin(2*np.pi*y)
k1=np.exp(0.7*np.sin(2*np.pi*X1)+0.3*np.cos(4*np.pi*X1))
f=1-dPhi(X3)*dTht(X2)
K=k1*f; q=1/K
def harm(a,b): return 2*a*b/(a+b)
qf=[harm(q,roll(q,1,ax)) for ax in range(3)]   # face between i and i+1 along ax
def A(u):
    out=np.zeros_like(u)
    for ax in range(3):
        fl=qf[ax]*(roll(u,1,ax)-u)/h          # flux at + face
        out+= -(fl-roll(fl,-1,ax))/h
    return out
def divqg(g):  # div_h(q gbar)
    out=np.zeros_like(q)
    for ax in range(3):
        out+= g[ax]*(qf[ax]-roll(qf[ax],-1,ax))/h
    return out
def grad(u):  return [(roll(u,1,ax)-roll(u,-1,ax))/(2*h) for ax in range(3)]
def d2(u,a,b):
    if a==b: return (roll(u,1,a)-2*u+roll(u,-1,a))/h**2
    return (roll(roll(u,1,a),1,b)-roll(roll(u,1,a),-1,b)-roll(roll(u,-1,a),1,b)+roll(roll(u,-1,a),-1,b))/(4*h*h)
def H(u,g): return [sum(d2(u,i,j)*g[j] for j in range(3)) for i in range(3)]
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dot(a,b): return sum(a[i]*b[i] for i in range(3))
P=lambda w: w-w.mean()
g1b=[0,1.0,0]; g2b=[0,0,1.0]
def F(u1,u2,crossed=False,eta=1.0):
    G1=[g1b[i]+grad(u1)[i] for i in range(3)]; G2=[g2b[i]+grad(u2)[i] for i in range(3)]
    B=[H(u2,G1)[i]-H(u1,G2)[i] for i in range(3)]
    c=cross(G1,G2); d=dot(c,c)
    S1=dot(cross(B,G1),c)/d; S2=dot(cross(B,G2),c)/d
    if crossed: S1,S2=S2,S1
    F1=A(u1)-P(divqg(g1b)-eta*q*S1); F2=A(u2)-P(divqg(g2b)-eta*q*S2)
    return F1,F2
u1e=P(Phi(X3)); u2e=P(Tht(X2))
def rF(F1,F2):
    qr=np.sqrt((q**2).mean()); return np.sqrt(((F1**2).mean()/qr**2+(F2**2).mean()/qr**2)/2)*L
F1,F2=F(u1e,u2e); print(f"n={n} same-index r_F(exact pair)={rF(F1,F2):.3e}  Linf={max(abs(F1).max(),abs(F2).max()):.3e}")
F1c,F2c=F(u1e,u2e,crossed=True); print(f"n={n} crossed    r_F(exact pair)={rF(F1c,F2c):.3e}")
def jac(u1,u2,crossed=False,delta=1e-6):
    N=n**3; J=np.zeros((2*N,2*N))
    base=np.concatenate([u1.ravel(),u2.ravel()])
    def Fv(v): a,b=F(v[:N].reshape(u1.shape),v[N:].reshape(u1.shape),crossed); return np.concatenate([a.ravel(),b.ravel()])
    for j in range(2*N):
        e=np.zeros(2*N); e[j]=delta
        J[:,j]=(Fv(base+e)-Fv(base-e))/(2*delta)
    return J
def spectrum(label,u1,u2,crossed=False):
    t=time.time(); J=jac(u1,u2,crossed); s=np.linalg.svd(J,compute_uv=False)
    s=np.sort(s); smax=s[-1]
    rel=s/smax
    print(f"[{label}] 2N={2*n**3} smax={smax:.3e} smallest 8 rel sv: {np.array2string(rel[:8],precision=2)}  count<1e-8:{(rel<1e-8).sum()} <1e-6:{(rel<1e-6).sum()} <1e-4:{(rel<1e-4).sum()} <1e-3:{(rel<1e-3).sum()} <1e-2:{(rel<1e-2).sum()}  ({time.time()-t:.0f}s)")
spectrum("same-index @exact pair",u1e,u2e)
spectrum("same-index @u=0 (non-solution)",0*u1e,0*u2e)
spectrum("crossed @exact pair",u1e,u2e,crossed=True)
# gauge-tangent check: delta = (dH/dpsi2, -dH/dpsi1) for H = eps*sin(2 pi psi1) sin(2 pi psi2)
psi1=X2+u1e; psi2=X3+u2e
dH1=2*np.pi*np.cos(2*np.pi*psi1)*np.sin(2*np.pi*psi2); dH2=2*np.pi*np.sin(2*np.pi*psi1)*np.cos(2*np.pi*psi2)
t1=P(dH2); t2=P(-dH1); nrm=np.sqrt((t1**2).mean()+(t2**2).mean())
for eps in [1e-3,1e-4]:
    a,b=F(u1e+eps*t1,u2e+eps*t2); print(f"gauge tangent step eps={eps}: r_F={rF(a,b):.3e} (base {rF(F1,F2):.3e}); |J t|/|t| ~ {np.sqrt(((a-F1)**2).mean()+((b-F2)**2).mean())/(eps*nrm):.3e}")
# generic random direction for comparison
rng=np.random.default_rng(0); r1=P(rng.standard_normal(u1e.shape)); r2=P(rng.standard_normal(u1e.shape)); nr=np.sqrt((r1**2).mean()+(r2**2).mean())
eps=1e-4; a,b=F(u1e+eps*r1,u2e+eps*r2); print(f"random direction: |J r|/|r| ~ {np.sqrt(((a-F1)**2).mean()+((b-F2)**2).mean())/(eps*nr):.3e}")
