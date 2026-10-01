"""LAYER 1: the PSF model."""
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

# --- 1.3 per-emitter widths ---
th=psf.pack(3.,[500.,800.],[10.2,7.1],[9.7,12.3])
chk("model_var_sigma == model when sigmas match",
    np.abs(psf.model_var_sigma(psf.pack_var_sigma(3.,[500.,800.],[10.2,7.1],[9.7,12.3],[1.35,1.35]),yy,xx)
           - psf.model(th,yy,xx,1.35)).max()<1e-12)
# --- 1.4 pack/unpack round trip ---
b,A,cy,cx=4.,np.array([1.,2.,3.]),np.array([4.,5.,6.]),np.array([7.,8.,9.])
b2,A2,cy2,cx2=psf.unpack(psf.pack(b,A,cy,cx))
chk("pack/unpack round trip", b2==b and np.array_equal(A2,A) and np.array_equal(cy2,cy) and np.array_equal(cx2,cx))

# --- 1.5 grid convention: model must respect (y,x) = (row,col) ordering ---
yy2,xx2=np.mgrid[0:15,0:25]; yy2=yy2*1.;xx2=xx2*1.
m=psf.model(psf.pack(0.,[1000.],[3.0],[20.0]),yy2,xx2,1.2)
i,j=np.unravel_index(np.argmax(m),m.shape)
chk("(y,x) maps to (row,col)", (i,j)==(3,20), "peak at (%d,%d)"%(i,j))
print("\n1: %d failure(s)"%fail); sys.exit(1 if fail else 0)
