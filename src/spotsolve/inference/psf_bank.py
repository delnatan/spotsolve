"""Positive, differentiable calibrated pixel-response tables.

Samples are pixel-integrated responses at (depth, lateral offset y/x), not
point intensities. Interpolate their logarithms with a tensor cubic B-spline.
This preserves positivity and C2 derivatives. No ROI-dependent normalization
or extrapolation is allowed. Calibration normalization is supplied by caller.
"""
from dataclasses import dataclass, field
import hashlib

import numpy as np
from scipy.ndimage import spline_filter


def _weights(t):
    w=np.stack([(1-t)**3,3*t**3-6*t*t+4,-3*t**3+3*t*t+3*t+1,t**3])/6
    d=np.stack([-.5*(1-t)**2,1.5*t*t-2*t,-1.5*t*t+t+.5,.5*t*t])
    return w,d


@dataclass(frozen=True)
class PixelPSFBank:
    depth_um: np.ndarray
    offsets_px: np.ndarray
    responses: np.ndarray
    coefficients: np.ndarray = field(init=False,repr=False,compare=False)

    def __post_init__(self):
        z,xy,data=(np.array(v,dtype=float,copy=True) for v in
                   (self.depth_um,self.offsets_px,self.responses))
        if (z.ndim!=1 or xy.ndim!=1 or len(z)<4 or len(xy)<4
                or data.shape!=(len(z),len(xy),len(xy))
                or not all(np.all(np.isfinite(v)) for v in (z,xy,data))
                or np.any(data<=0)):
            raise ValueError('finite positive responses on (depth,y,x) grid required')
        for axis in (z,xy):
            if np.any(np.diff(axis)<=0) or not np.allclose(np.diff(axis),axis[1]-axis[0],rtol=1e-8,atol=1e-12):
                raise ValueError('PSF axes must be strictly increasing and uniformly spaced')
        coefficients=spline_filter(np.pad(np.log(data),2,mode='reflect'),order=3,mode='mirror')
        for name,value in [('depth_um',z),('offsets_px',xy),('responses',data),('coefficients',coefficients)]:
            value.setflags(write=False)
            object.__setattr__(self,name,value)

    @property
    def fingerprint(self):
        digest=hashlib.sha256()
        for value in (self.depth_um,self.offsets_px,self.responses):
            digest.update(str(value.shape).encode());digest.update(value.tobytes())
        return digest.hexdigest()

    def evaluate_offsets(self,dy,dx,depth_um):
        """Return response and derivatives w.r.t. offset-y, offset-x, depth."""
        z,y,x=np.broadcast_arrays(depth_um,dy,dx)
        coords=[];weights=[];derivatives=[]
        for a,axis in zip((z,y,x),(self.depth_um,self.offsets_px,self.offsets_px)):
            if np.any(~np.isfinite(a)) or np.any(a<axis[0]-1e-10) or np.any(a>axis[-1]+1e-10):
                raise ValueError('requested PSF offset/depth lies outside calibration')
            step=axis[1]-axis[0]
            u=(np.clip(a,axis[0],axis[-1])-axis[0])/step+2
            base=np.floor(u).astype(int)
            w,d=_weights(u-base)
            coords.append(base-1);weights.append(w);derivatives.append(d/step)
        log_value=np.zeros(z.shape);gradient=[np.zeros(z.shape) for _ in range(3)]
        for iz in range(4):
            for iy in range(4):
                for ix in range(4):
                    inds=(iz,iy,ix)
                    c=self.coefficients[coords[0]+iz,coords[1]+iy,coords[2]+ix]
                    w=[weights[j][inds[j]] for j in range(3)]
                    log_value+=c*w[0]*w[1]*w[2]
                    for j in range(3):
                        gradient[j]+=c*derivatives[j][inds[j]]*w[(j+1)%3]*w[(j+2)%3]
        response=np.exp(log_value)
        return response,response*gradient[1],response*gradient[2],response*gradient[0]

    def evaluate(self,grid,y,x,depth_um):
        response,dy,dx,dz=self.evaluate_offsets(grid.yy-y,grid.xx-x,depth_um)
        return response,-dy,-dx,dz
