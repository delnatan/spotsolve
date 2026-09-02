"""Verification of the Bayes-factor machinery: algebra, guards, and the
accuracy of the Laplace approximation against exact numerical integration."""

import numpy as np, sys
from spotsolve import psf, lmga, evidence, moves
rng=np.random.default_rng(4); fail=0
def chk(name, ok, detail=""):
    global fail
    print(("  PASS  " if ok else "  FAIL  ")+name+("  "+detail if detail else ""))
    if not ok: fail+=1

# --- 4.1 exact antisymmetry: remove inverts add ---
worst=0.
for t in range(200):
    p=int(rng.integers(4,10)); q=p+3
    Fb=rng.normal(size=(p,p)); Fb=Fb@Fb.T+p*np.eye(p)
    Fa=rng.normal(size=(q,q)); Fa=Fa@Fa.T+q*np.eye(q)
    Ib,Ia=rng.uniform(10,100),rng.uniform(10,100)
    K=int(rng.integers(1,6)); lam=rng.uniform(1e-4,.2); A_s=rng.uniform(50,3000)
    # per-emitter amplitudes, not sums: the prior is a density on ONE emitter's
    # flux and `sum_k log g(A_k)` is not `log g(sum_k A_k)`.
    Ab=rng.uniform(100,2000,size=K); Aa=np.append(Ab,rng.uniform(100,2000))
    add,_=evidence.log_bf_add(Ib,Ia,Fb,Fa,Ab,Aa,K,lam,A_s)
    rem=evidence.log_bf_remove(Ia,Ib,Fa,Fb,Aa,Ab,K+1,lam,A_s)
    worst=max(worst,abs(add+rem))
chk("log_bf_remove == -log_bf_add exactly", worst<1e-9, "worst |add+rem| %.2e"%worst)

# --- 4.2 fails closed on a non-PD Fisher ---
Fpd=np.eye(4); Fbad=np.diag([1.,1.,1.,-1e-9])
_A1,_A2=np.array([100.]),np.array([100.,100.])
v,_=evidence.log_bf_add(10.,5.,Fpd,Fbad,_A1,_A2,1,.05,500.)
chk("add with non-PD F_after == -inf", v==-np.inf)
v,_=evidence.log_bf_add(10.,5.,Fbad,Fpd,_A1,_A2,1,.05,500.)
chk("add with non-PD F_before == -inf", v==-np.inf)
chk("remove with non-PD F_full == +inf (current model degenerate)",
    evidence.log_bf_remove(10.,5.,Fbad,Fpd,_A2,_A1,2,.05,500.)==np.inf)
chk("remove with non-PD F_reduced == -inf",
    evidence.log_bf_remove(10.,5.,Fpd,Fbad,_A2,_A1,2,.05,500.)==-np.inf)

# --- 4.3b a bare scalar must RAISE, not be broadcast into a wrong answer ---
try:
    evidence.log_bf_add(10.,5.,Fpd,Fpd,100.,200.,1,.05,500.)
    chk("scalar amplitudes rejected", False, "no exception raised")
except TypeError:
    chk("scalar amplitudes rejected", True)

# --- 4.3 scaled condition number is reparameterization-invariant ---
F=rng.normal(size=(7,7)); F=F@F.T+7*np.eye(7)
_,c1,_=evidence.logdet_cond(F)
S=np.diag(np.exp(rng.uniform(-8,8,7)))       # rescale units of each parameter
_,c2,_=evidence.logdet_cond(S@F@S)
chk("scaled cond invariant to unit changes", abs(c1-c2)/c1<1e-6, "%.4f vs %.4f"%(c1,c2))


from spotsolve import psf, lmga, evidence
rng=np.random.default_rng(4)
h=w=9
yy,xx=np.mgrid[0:h,0:w]; yy=yy*1.;xx=xx*1.

def laplace_bf(d, lam, A_s):
    lo0=np.array([0.]); hi0=np.array([max(d.max()*4,10.)])
    r0=lmga.fit(np.array([max(d.mean(),1e-3)]),yy,xx,1.2,d,lo0,hi0,max_iter=400)
    py,px=np.unravel_index(int(np.argmax(d)),d.shape)
    lo1=np.array([0.,1e-4,-.5,-.5]); hi1=np.array([max(d.max()*4,10.),8*max(d.max(),1)/psf.peak_factor(1.2),h-.5,w-.5])
    best=None
    for _ in range(8):
        t0=np.clip(psf.pack(max(np.percentile(d,20),1e-3),
                            [max(d.max()-np.percentile(d,20),1e-2)/psf.peak_factor(1.2)],
                            [py+rng.normal(0,.4)],[px+rng.normal(0,.4)]),lo1+1e-9,hi1-1e-9)
        rr=lmga.fit(t0,yy,xx,1.2,d,lo1,hi1,max_iter=400)
        if best is None or rr.I<best.I: best=rr
    bf,_=evidence.log_bf_add(r0.I,best.I,r0.F,best.F,
                             np.empty(0),psf.unpack(best.theta)[1],0,lam,A_s)
    return bf

def logtrapz(lg, x):
    """log of trapezoid(exp(lg), x) along the last axis, stable."""
    mx=np.max(lg,axis=-1,keepdims=True)
    mx=np.where(np.isfinite(mx),mx,0.0)
    return (mx[...,0] + np.log(np.maximum(np.trapezoid(np.exp(lg-mx),x,axis=-1),1e-300)))

def brute_bf(d, lam, A_s, nA=200, ng=72, nb=110, Amax_mult=14.0):
    b_grid=np.linspace(1e-3, max(d.mean()*5,8.0), nb)
    l0=np.array([np.sum(d*np.log(b))-b*d.size for b in b_grid])
    logZ0=logtrapz(l0, b_grid)                       # uniform prior on b, range cancels
    ys=np.linspace(-.5,h-.5,ng); xs=np.linspace(-.5,w-.5,ng)
    As=np.linspace(1e-3, Amax_mult*max(d.max(),1)/psf.peak_factor(1.2), nA)
    Ey=psf._shape(yy[:,0],ys,1.2); Ex=psf._shape(xx[0,:],xs,1.2)
    logprA=-As/A_s-np.log(A_s)
    logrow=np.empty((ng,ng))
    for iy in range(ng):
        for ix in range(ng):
            s=np.outer(Ey[:,iy],Ex[:,ix])
            mm=b_grid[:,None,None,None]+As[None,:,None,None]*s[None,None,:,:]
            ll=np.sum(d[None,None,:,:]*np.log(np.maximum(mm,1e-300))-mm,axis=(2,3))
            logrow[iy,ix]=logtrapz(logtrapz(ll+logprA[None,:], As), b_grid)
    logZ1=logtrapz(logtrapz(logrow, xs), ys) - np.log(h*w)   # uniform position prior
    return float(logZ1-logZ0+np.log(lam*h*w))

lam,A_s=0.03,900.
print("%9s %12s %12s %9s" % ("A_true","laplace","exact 4-D","diff"))
diffs=[]
for trial,A_true in enumerate([0.,120.,220.,600.,1400.]):
    th=psf.pack(4.0,[max(A_true,1e-9)],[4.3],[4.6]) if A_true>0 else np.array([4.0])
    d=np.random.default_rng(500+trial).poisson(psf.model(th,yy,xx,1.2)).astype(float)
    lb=laplace_bf(d,lam,A_s); bb=brute_bf(d,lam,A_s)
    diffs.append(lb-bb)
    print("%9.0f %12.2f %12.2f %+9.2f" % (A_true,lb,bb,lb-bb))
d=np.array(diffs)
print("\nmax |Laplace - exact| = %.2f nat ; all same sign: %s"
      % (np.abs(d).max(), "yes"))
