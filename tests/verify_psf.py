"""LAYER 1: the PSF model and its derivatives."""
import numpy as np, sys
from spotsolve import psf
rng=np.random.default_rng(0)
fail=0
def chk(name, ok, detail=""):
    global fail
    print(("  PASS  " if ok else "  FAIL  ")+name+("  "+detail if detail else ""))
    if not ok: fail+=1

# --- 1.1 flux normalisation: A is TOTAL FLUX (sum over an infinite grid) ---
for sig in (0.8,1.2,2.0):
    yy,xx=np.mgrid[0:61,0:61]; yy=yy*1.;xx=xx*1.
    m=psf.model(psf.pack(0.0,[1000.],[30.0],[30.0]),yy,xx,sig)
    chk("flux normalisation sigma=%.1f"%sig, abs(m.sum()-1000.)<1e-6, "sum=%.6f"%m.sum())
# sub-pixel centre must not change total flux
m1=psf.model(psf.pack(0.,[1000.],[30.0],[30.0]),yy,xx,1.2).sum()
m2=psf.model(psf.pack(0.,[1000.],[30.37],[30.62]),yy,xx,1.2).sum()
chk("flux invariant to subpixel shift", abs(m1-m2)<1e-6, "%.6f vs %.6f"%(m1,m2))

# --- 1.2 peak_factor ---
yy,xx=np.mgrid[0:21,0:21]; yy=yy*1.;xx=xx*1.
m=psf.model(psf.pack(0.,[1.0],[10.0],[10.0]),yy,xx,1.2)
chk("peak_factor == on-centre pixel value", abs(m[10,10]-psf.peak_factor(1.2))<1e-12,
    "%.8f vs %.8f"%(m[10,10],psf.peak_factor(1.2)))

# --- 1.3 analytic jacobian vs central finite differences ---
def fd_jac(theta, yy, xx, sig, h=1e-6):
    J=np.empty(yy.shape+(len(theta),))
    for i in range(len(theta)):
        tp=theta.copy(); tm=theta.copy()
        step=h*max(abs(theta[i]),1.0)
        tp[i]+=step; tm[i]-=step
        J[...,i]=(psf.model(tp,yy,xx,sig)-psf.model(tm,yy,xx,sig))/(2*step)
    return J
worst=0.
for trial in range(30):
    K=int(rng.integers(1,4))
    th=psf.pack(rng.uniform(1,20), rng.uniform(50,2000,K),
                rng.uniform(3,17,K), rng.uniform(3,17,K))
    sig=float(rng.uniform(0.7,2.2))
    Ja=psf.jac(th,yy,xx,sig); Jf=fd_jac(th,yy,xx,sig)
    rel=np.abs(Ja-Jf).max()/max(np.abs(Jf).max(),1e-30)
    worst=max(worst,rel)
chk("psf.jac == finite differences", worst<2e-6, "worst rel err %.2e"%worst)

# --- 1.4 model_and_jac consistent with model / jac ---
worst_m=worst_j=0.
for trial in range(20):
    K=int(rng.integers(0,4))
    th=psf.pack(rng.uniform(1,20), rng.uniform(50,2000,K), rng.uniform(3,17,K), rng.uniform(3,17,K)) if K else np.array([rng.uniform(1,20)])
    sig=float(rng.uniform(0.7,2.2)); halo=rng.uniform(0,5,yy.shape)
    m,J=psf.model_and_jac(th,yy,xx,sig,halo)
    worst_m=max(worst_m,np.abs(m-psf.model(th,yy,xx,sig,halo)).max())
    worst_j=max(worst_j,np.abs(J-psf.jac(th,yy,xx,sig,halo)).max())
chk("model_and_jac model agrees with model()", worst_m<1e-10, "%.2e"%worst_m)
chk("model_and_jac jac agrees with jac()",     worst_j<1e-10, "%.2e"%worst_j)

# --- 1.5 halo is additive and jacobian-free ---
th=psf.pack(3.,[500.],[10.2],[9.7]); halo=rng.uniform(0,5,yy.shape)
chk("halo enters model additively",
    np.abs(psf.model(th,yy,xx,1.2,halo)-(psf.model(th,yy,xx,1.2)+halo)).max()<1e-12)
chk("halo does not enter the jacobian",
    np.abs(psf.jac(th,yy,xx,1.2,halo)-psf.jac(th,yy,xx,1.2)).max()<1e-12)

# --- 1.6 free-sigma model/jac ---
th=psf.pack(3.,[500.,800.],[10.2,7.1],[9.7,12.3]); ths=np.append(th,1.35)
chk("model_free_sigma == model at same sigma",
    np.abs(psf.model_free_sigma(ths,yy,xx)-psf.model(th,yy,xx,1.35)).max()<1e-12)
def fd_jac_fs(t,h=1e-6):
    J=np.empty(yy.shape+(len(t),))
    for i in range(len(t)):
        tp=t.copy();tm=t.copy();st=h*max(abs(t[i]),1.0)
        tp[i]+=st;tm[i]-=st
        J[...,i]=(psf.model_free_sigma(tp,yy,xx)-psf.model_free_sigma(tm,yy,xx))/(2*st)
    return J
Ja=psf.jac_free_sigma(ths,yy,xx); Jf=fd_jac_fs(ths)
rel=np.abs(Ja-Jf).max()/np.abs(Jf).max()
chk("jac_free_sigma == finite differences", rel<2e-6, "rel %.2e"%rel)

# --- 1.7 pack/unpack round trip ---
b,A,cy,cx=4.,np.array([1.,2.,3.]),np.array([4.,5.,6.]),np.array([7.,8.,9.])
b2,A2,cy2,cx2=psf.unpack(psf.pack(b,A,cy,cx))
chk("pack/unpack round trip", b2==b and np.array_equal(A2,A) and np.array_equal(cy2,cy) and np.array_equal(cx2,cx))

# --- 1.8 grid convention: model must respect (y,x) = (row,col) ordering ---
yy2,xx2=np.mgrid[0:15,0:25]; yy2=yy2*1.;xx2=xx2*1.
m=psf.model(psf.pack(0.,[1000.],[3.0],[20.0]),yy2,xx2,1.2)
i,j=np.unravel_index(np.argmax(m),m.shape)
chk("(y,x) maps to (row,col)", (i,j)==(3,20), "peak at (%d,%d)"%(i,j))
print("\n1: %d failure(s)"%fail); sys.exit(1 if fail else 0)
