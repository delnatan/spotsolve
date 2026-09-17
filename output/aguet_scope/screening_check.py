import ast,json,time,warnings
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import scipy.stats as stats
source=Path('/Users/delnatan/Projects/github/spotfitlm/spotfitlm/utils.py').read_text()
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='PointSourceDetector2D')
namespace=dict(np=np,ndi=ndi,stats=stats)
exec(compile(ast.Module(body=[cls],type_ignores=[]),'<spotfitlm detector>','exec'),namespace)
Detector=namespace['PointSourceDetector2D']
def prepare(det,alpha):
 n=det.kernel_size;k=stats.norm.isf(alpha/2)
 a=(n-1)/(n-3)*det.C[0,0];b=k*k/(2*(n-1))
 df=(n-1)*(a+b)**2/(a*a+b*b)
 cutoff=k+stats.t.isf(alpha,df)*np.sqrt((a+b)/n)
 return cutoff

def screen(det,f,cutoff):
 g=det._g[det._g.shape[0]//2];side=len(g);n=det.kernel_size
 fg=ndi.convolve1d(ndi.convolve1d(f,g,axis=0),g,axis=1)
 fu=ndi.uniform_filter(f,size=side)*n
 fu2=ndi.uniform_filter(f*f,size=side)*n
 A=(fg-det._gsum*fu/n)/(det._g2sum-det._gsum**2/n)
 c=(fu-det._gsum*A)/n
 rss=A*A*det._g2sum-2*A*(fg-c*det._gsum)+fu2-2*c*fu+n*c*c
 return (rss>0)&(A>cutoff*np.sqrt(np.maximum(rss,0)/(n-1)))

rng=np.random.default_rng(1542);y,x=np.indices((128,128))
images=[]
for seed in range(10):
 mean=np.full((128,128),20.)
 for _ in range(15):
  yy,xx=rng.uniform(0,128,2);s=rng.uniform(.8,2.4)
  mean+=rng.uniform(5,100)*np.exp(-((y-yy)**2+(x-xx)**2)/(2*s*s))
 images.append(rng.poisson(mean).astype(float))
rows=[]
with warnings.catch_warnings():
 warnings.simplefilter('ignore',RuntimeWarning)
 for sigma in (.8,1.45,2.4):
  det=Detector(sigma)
  for alpha in (.01,.05,.1):
   cutoff=prepare(det,alpha);different=0
   for f in images+[np.zeros((128,128))]:
    different+=np.count_nonzero(det.detect_significant_pixels(f,alpha)!=screen(det,f,cutoff))
   rows.append(dict(sigma=sigma,alpha=alpha,pixel_decision_differences=int(different)))
 det=Detector(1.45);cutoff=prepare(det,.05);f=images[0]
 times={}
 for name,call in [('reference',lambda:det.detect_significant_pixels(f)),('prototype',lambda:screen(det,f,cutoff))]:
  values=[]
  for _ in range(7):
   t=time.perf_counter();call();values.append(time.perf_counter()-t)
  times[name]=float(np.median(values))
report=dict(reference_revision='e0f4036de786ac3a14c15c5a353e60231fc030a6',scope='Screening only; not a complete detector or fitter',seed=1542,shape=[128,128],cases=99,pixels=99*128*128,rows=rows,seconds=times)
out=Path('output/aguet_scope/screening_check.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
