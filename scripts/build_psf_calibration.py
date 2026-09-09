#!/usr/bin/env python3
"""Build a pixel-response calibration from the existing psfkit confocal optics.

Same optical settings as data/sim_out. This is not a calibration of the real
bead movies. A sliding camera-pixel integral supplies subpixel response samples.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter


def build_calibration(*,oversample=7,depth_step=.05,extent_px=22,z_range=.6):
    from psfkit import ConfocalOptics,compute_confocal_psf
    if oversample<3 or oversample%2!=1:
        raise ValueError('oversample must be odd and >=3')
    optics=ConfocalOptics.from_andor_bc43(magnification=100.,wavelength_exc=.488,
        wavelength_em=.525,na=1.4,ni=1.515,ns=1.334)
    nz=2*int(np.ceil(z_range/depth_step))+1
    size=(2*extent_px+3)*oversample
    fine=compute_confocal_psf(optics,shape=(nz,size,size),
        spacing=(depth_step,.085/oversample,.085/oversample),vectorial=True,normalize=None)
    responses=uniform_filter(fine,size=(1,oversample,oversample),mode='constant')
    center=size//2;radius=extent_px*oversample
    responses=responses[:,center-radius:center+radius+1,center-radius:center+radius+1]
    # In-focus reference is a fixed centered calibration aperture, never an ROI.
    reference=responses[nz//2,::oversample,::oversample].sum()
    responses/=reference
    floor=1e-15*responses.max()
    clipped=int(np.sum(responses<floor))
    responses=np.maximum(responses,floor)
    return dict(depth_um=(np.arange(nz)-nz//2)*depth_step,
                offsets_px=np.arange(-radius,radius+1)/oversample,responses=responses),dict(
        instrument='psfkit BC43 confocal',pixel_size_um=.085,oversample=oversample,
        depth_step_um=depth_step,extent_px=extent_px,normalization='fixed centered focus aperture',
        integration='odd midpoint quadrature, sliding camera-pixel average',
        calibration_floor=float(floor),floored_samples=clipped,
        real_camera_calibration=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('reference/defocus-model/psf_calibration.npz'))
    p.add_argument('--oversample',type=int,default=7)
    p.add_argument('--depth-step',type=float,default=.05)
    a=p.parse_args()
    arrays,meta=build_calibration(oversample=a.oversample,depth_step=a.depth_step)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**arrays)
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Saved',a.output,arrays['responses'].shape,flush=True)


if __name__=='__main__':main()
