"""LAYER 5: calibration of the search itself -- null rate, split rate, flux recovery."""
import numpy as np, sys
import psf, msearch, metrics
SIG=1.2; h=w=11
lam, A_s = 0.034, 1400.0
BG=4.0

# --- 5.1 pure background: how often is an emitter invented? ---
n=400; born=0; Ks=[]
for s in range(n):
    d=np.random.default_rng(9000+s).poisson(np.full((h,w),BG)).astype(float)
    r=msearch.search_patch(d,SIG,np.empty((0,2)),np.empty((0,)),lam=lam,A_s=A_s,b0=BG,k_max=8)
    Ks.append(len(r.amplitudes)); born += len(r.amplitudes)>0
Ks=np.array(Ks)
print("5.1  pure background, %d patches of %dx%d at bg=%.0f e-" % (n,h,w,BG))
print("     patches with >=1 emitter invented: %d/%d = %.3f" % (born,n,born/n))
print("     K distribution: %s" % np.bincount(Ks).tolist())

# --- 5.2 one true emitter: is K recovered, and is flux unbiased? ---
print("\n5.2  ONE true emitter, K recovered and flux/position bias")
print("     %8s %8s %8s %8s %10s %10s" % ("A_true","frac K=1","frac K>1","frac K=0","flux bias","pos rmse"))
for A_true in (200.,400.,900.,1400.,1900.):
    K1=Ksplit=K0=0; fl=[]; pe=[]
    m=200
    for s in range(m):
        rg=np.random.default_rng(7000+s)
        cy,cx=rg.uniform(4.5,6.5,2)
        th=psf.pack(BG,[A_true],[cy],[cx])
        yy,xx=np.mgrid[0:h,0:w]
        d=rg.poisson(psf.model(th,yy*1.,xx*1.,SIG)).astype(float)
        r=msearch.search_patch(d,SIG,np.empty((0,2)),np.empty((0,)),lam=lam,A_s=A_s,b0=BG,k_max=8)
        K=len(r.amplitudes)
        if K==1:
            K1+=1; fl.append(r.amplitudes[0]/A_true-1.0)
            pe.append(np.hypot(r.positions[0,0]-cy, r.positions[0,1]-cx))
        elif K==0: K0+=1
        else: Ksplit+=1
    print("     %8.0f %8.3f %8.3f %8.3f %+10.4f %10.4f"
          % (A_true, K1/m, Ksplit/m, K0/m,
             np.mean(fl) if fl else np.nan, np.sqrt(np.mean(np.square(pe))) if pe else np.nan))

# --- 5.3 CRLB comparison for the isolated case ---
print("\n5.3  position RMSE vs Cramer-Rao bound (isolated emitter, K forced to 1)")
print("     %8s %10s %10s %7s" % ("A_true","rmse","CRLB","ratio"))
for A_true in (400.,900.,1900.):
    errs=[]; crlbs=[]
    for s in range(200):
        rg=np.random.default_rng(8000+s)
        cy,cx=rg.uniform(4.5,6.5,2)
        th=psf.pack(BG,[A_true],[cy],[cx])
        yy,xx=np.mgrid[0:h,0:w]; yy=yy*1.;xx=xx*1.
        d=rg.poisson(psf.model(th,yy,xx,SIG)).astype(float)
        r=msearch.search_patch(d,SIG,np.array([[cy,cx]]),np.array([A_true]),lam=lam,A_s=A_s,
                               b0=BG,k_max=1,enable=())
        errs.append(np.hypot(r.positions[0,0]-cy,r.positions[0,1]-cx))
        m_,J_=psf.model_and_jac(th,yy,xx,SIG)
        F=J_.reshape(-1,4).T@((1.0/np.maximum(m_.reshape(-1),1e-9))[:,None]*J_.reshape(-1,4))
        C=np.linalg.inv(F); crlbs.append(np.sqrt(C[2,2]+C[3,3]))
    print("     %8.0f %10.4f %10.4f %7.2f"%(A_true,np.sqrt(np.mean(np.square(errs))),np.mean(crlbs),
                                            np.sqrt(np.mean(np.square(errs)))/np.mean(crlbs)))
