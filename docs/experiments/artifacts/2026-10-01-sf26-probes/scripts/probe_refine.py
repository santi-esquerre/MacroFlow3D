# Same continuum (band-limited at n0^3) field at n0^3 and 2n0^3: does the eta=1 Anderson floor drop with h?
import sys, time, numpy as np
n0,ellh0,sigma2,lam=int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]); seed=int(sys.argv[5]) if len(sys.argv)>5 else 7
import scipy.sparse as sp, scipy.sparse.linalg as spl
def run(n,Y):
    L=1.0; h=L/n
    g={}
    src=open('solver_probe_gauss.py').read().split('print(f"n={n}')[0]
    # strip field generation: replace the Y synthesis block by the supplied Y
    src=src.replace("Y=np.real(np.fft.ifftn(noise*spec)); Y-=Y.mean(); Y*=np.sqrt(sigma2)/Y.std()","Y=Y_IN")
    sys.argv=['x',str(n),str(ellh0*n/n0),str(sigma2),str(lam),str(seed)]
    g['Y_IN']=Y
    exec(src,g)
    t0=time.time(); ha=g['anderson'](False,m=8,omega=0.5,iters=2000)
    print(f"[refine] n={n} ell/h={ellh0*n/n0:.0f} Ystd={Y.std():.3f}: Anderson m=8 w=0.5 (2000 its) final r_F={ha[-1]:.3e} min={min(ha):.3e} @500={ha[min(500,len(ha)-1)]:.3e} @1000={ha[min(1000,len(ha)-1)]:.3e} ({time.time()-t0:.0f}s)",flush=True)
# base field at n0 (same synthesis as solver_probe_gauss.py)
L=1.0; h=L/n0; ell=ellh0*h
rng=np.random.default_rng(seed)
kx=2*np.pi*np.fft.fftfreq(n0,d=h); K1,K2,K3=np.meshgrid(kx,kx,kx,indexing='ij'); k2=K1**2+K2**2+K3**2
spec=np.exp(-k2*ell**2/8.0); noise=np.fft.fftn(rng.standard_normal((n0,n0,n0)))
Yhat=noise*spec; Y0=np.real(np.fft.ifftn(Yhat)); m=Y0.mean(); s=Y0.std(); Y0=(Y0-m)*np.sqrt(sigma2)/s
# exact band-limited upsampling to 2n0 via zero padding of the SAME spectrum (then same affine normalization)
def upsample(Yh,n_from,n_to):
    out=np.zeros((n_to,n_to,n_to),dtype=complex); half=n_from//2
    sl=lambda i: slice(0,half) if i==0 else slice(n_to-half,n_to)
    src_sl=lambda i: slice(0,half) if i==0 else slice(n_from-half,n_from)
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                out[sl(a),sl(b),sl(c)]=Yh[src_sl(a),src_sl(b),src_sl(c)]
    return np.real(np.fft.ifftn(out))*(n_to**3/n_from**3)
Y1=(upsample(Yhat,n0,2*n0)-m)*np.sqrt(sigma2)/s
# sanity: the 2n0 field restricted to the n0 cell centres must match (cell centres differ: (i+0.5)h vs (j+0.5)h/2 -> compare via spectral check instead)
print(f"[refine] field check: Y0 std={Y0.std():.4f} Y1 std={Y1.std():.4f} Y1 mean={Y1.mean():.2e}",flush=True)
run(n0,Y0); run(2*n0,Y1)
