"""LAYER 3: coordinate conventions, patch geometry, halo, global model."""
import numpy as np, sys
from spotsolve import psf
from spotsolve.deprecated import calibrate, patches as patch_mod
rng=np.random.default_rng(3); fail=0
def chk(name, ok, detail=""):
    global fail
    print(("  PASS  " if ok else "  FAIL  ")+name+("  "+detail if detail else ""))
    if not ok: fail+=1

H,W,SIG=48,52,1.2

# --- 3.1 render_model == psf.model on the full grid (same convention) ---
P=np.array([[10.3,20.7],[31.2,8.4],[40.9,45.1]]); A=np.array([900.,1500.,700.])
yy,xx=np.mgrid[0:H,0:W]; yy=yy*1.;xx=xx*1.
ref=psf.model(psf.pack(5.0,A,P[:,0],P[:,1]),yy,xx,SIG)
got=calibrate.render_model(P,A,SIG,(H,W),5.0)
chk("render_model == psf.model (truncation error only)",
    np.abs(got-ref).max()<1e-3, "max abs diff %.2e (truncate=4 sigma)"%np.abs(got-ref).max())
chk("render_model flux is conserved to <0.1%",
    abs((got-5.0).sum()-A.sum())/A.sum()<1e-3, "%.5f vs %.1f"%((got-5.0).sum(),A.sum()))

# --- 3.2 patch_grids origin: grid[0,0] <-> global (y0,x0) ---
P2=np.array([[20.4,25.6]])
ps=patch_mod.build_patches(P2,SIG,(H,W),k_max=8); p=ps[0]
gy,gx=patch_mod.patch_grids(p)
chk("patch_grids is 0-based and matches the slice shape",
    gy.shape==p.shape and gy[0,0]==0 and gx[0,0]==0)
# render globally, slice; vs render locally at local coords -> must agree
glob=calibrate.render_model(P2,np.array([1000.]),SIG,(H,W),0.0)[p.y0:p.y1,p.x0:p.x1]
loc=psf.model(psf.pack(0.,[1000.],[P2[0,0]-p.y0],[P2[0,1]-p.x0]),gy,gx,SIG)
chk("local patch coords == global coords minus (y0,x0)",
    np.abs(glob-loc).max()<1e-3, "max diff %.2e"%np.abs(glob-loc).max())

# --- 3.3 halo image uses the same origin ---
Pall=np.array([[20.4,25.6],[20.0,31.0],[26.5,26.0]]); Aall=np.array([1000.,800.,1200.])
ps=patch_mod.build_patches(Pall,SIG,(H,W),k_max=8)
for p in ps:
    if p.frozen_indices.size:
        gy,gx=patch_mod.patch_grids(p)
        halo=patch_mod.build_halo_image(Pall,Aall,p.frozen_indices,SIG,gy,gx,p.y0,p.x0)
        ref_h=calibrate.render_model(Pall[p.frozen_indices],Aall[p.frozen_indices],SIG,(H,W),0.0)[p.y0:p.y1,p.x0:p.x1]
        chk("halo aligns with the global render",
            np.abs(halo-ref_h).max()<1e-3, "max diff %.2e"%np.abs(halo-ref_h).max())
        break
else:
    chk("halo aligns with the global render", False, "no frozen emitters produced -- test vacuous")

# --- 3.4 partition: every emitter free in exactly one patch ---
bad_part=bad_cover=0
for t in range(60):
    n=int(rng.integers(1,60)); P3=np.stack([rng.uniform(2,H-2,n),rng.uniform(2,W-2,n)],1)
    ps=patch_mod.build_patches(P3,SIG,(H,W),k_max=8)
    free=np.concatenate([p.indices for p in ps]) if ps else np.array([],int)
    if sorted(free.tolist())!=list(range(n)): bad_part+=1
    for p in ps:
        if np.intersect1d(p.indices,p.frozen_indices).size: bad_cover+=1
        if len(p.indices)>8: bad_cover+=1
        # every free emitter must lie inside its own bbox
        yy_,xx_=P3[p.indices,0],P3[p.indices,1]
        if not (np.all(yy_>=p.y0)and np.all(yy_<=p.y1)and np.all(xx_>=p.x0)and np.all(xx_<=p.x1)): bad_cover+=1
chk("free emitters partition the set exactly once", bad_part==0, "%d/60 bad"%bad_part)
chk("free/frozen disjoint, k_max honoured, emitters inside own bbox", bad_cover==0, "%d violations"%bad_cover)

# --- 3.5 the halo must capture every neighbour that matters ---
# any emitter NOT free and NOT frozen must contribute negligibly inside the bbox
worst_leak=0.
for t in range(40):
    n=int(rng.integers(5,60)); P3=np.stack([rng.uniform(2,H-2,n),rng.uniform(2,W-2,n)],1)
    A3=rng.uniform(500,2000,n)
    ps=patch_mod.build_patches(P3,SIG,(H,W),k_max=8)
    for p in ps:
        known=set(p.indices.tolist())|set(p.frozen_indices.tolist())
        rest=np.array([i for i in range(n) if i not in known],int)
        if rest.size==0: continue
        leak=calibrate.render_model(P3[rest],A3[rest],SIG,(H,W),0.0)[p.y0:p.y1,p.x0:p.x1].max()
        worst_leak=max(worst_leak,leak)
chk("unmodelled neighbours leak < 0.05 e- into a patch", worst_leak<0.05,
    "worst leak %.3f e- (bkg is ~4 e-)"%worst_leak)

# --- 3.6 patch bbox stays inside the image ---
bad=0
for t in range(40):
    n=int(rng.integers(1,40)); P3=np.stack([rng.uniform(0,H-1,n),rng.uniform(0,W-1,n)],1)
    for p in patch_mod.build_patches(P3,SIG,(H,W),k_max=8):
        if p.y0<0 or p.x0<0 or p.y1>H or p.x1>W or p.y1<=p.y0 or p.x1<=p.x0: bad+=1
chk("patch bboxes are valid and in-frame", bad==0, "%d bad"%bad)
print("\n3: %d failure(s)"%fail); sys.exit(1 if fail else 0)
