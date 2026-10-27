# First-order (Dagan) purely advective macrodispersivities, 3D isotropic, C_Y(r)=s2*exp(-(r/lam)^2), s2=1, units lambda.
import numpy as np
from scipy import integrate, special
def gT(x):   # int_{-1}^{1} mu(1-mu^2) sin(x mu) dmu
    if x<0.05: return 2*x/15
    s,c=np.sin(x),np.cos(x); return 2*(-2*s/x**2-6*c/x**3+6*s/x**4)
def gL(x):   # int_{-1}^{1} (1-mu^2)^2 sin(x mu)/mu dmu
    if x<0.05: return 16*x/15
    s,c=np.sin(x),np.cos(x); return 2*special.sici(x)[0]+2*(s/x**2+c/x+6*c/x**3-6*s/x**4)
def aT(tau):
    f=lambda k: k*np.exp(-k*k/4)*np.pi*gT(k*tau)
    v,_=integrate.quad(f,0,14,limit=4000); return v/(8*np.pi**1.5)
def aL(tau):
    f=lambda k: k*np.exp(-k*k/4)*2*np.pi*gL(k*tau)
    v,_=integrate.quad(f,0,14,limit=4000); return v/(8*np.pi**1.5)
print("first-order alpha_L(inf)/(s2 lam) = sqrt(pi)/2 =",np.sqrt(np.pi)/2)
for tau in [0.5,1,2,3,5,10,15,20,30,40,50,75,100,150,200]:
    t=aT(tau); l=aL(tau)
    print(f"tau={tau:6.1f} aL/s2={l:.5f} aT/s2={t:+.3e} aT*tau^2={t*tau**2:+.4f}")
def win(tb):
    v,_=integrate.quad(aT,0.75*tb,tb,limit=400); return v/(0.25*tb)
for tb in [10,15,20,30,50,80,100,190]:
    print(f"t_b={tb}: windowed aT/s2 = {win(tb):+.3e}")
