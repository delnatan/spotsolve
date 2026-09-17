"""Reproducible sparse baseline benchmark against the original spotfitlm sources.

Compiles the reference C batch fitter in a temporary directory. Reference
timings exclude pandas construction; native timings include Localizations.
Synthetic data use the reference's sampled Gaussian and Poisson observations.
"""
import argparse
import ctypes
import json
from pathlib import Path
import platform
import subprocess
import sysconfig
import tempfile
from time import perf_counter

import numpy as np

from make_aguet_fixture import load_detector
from spotsolve import localize_aguet_stack
from spotsolve.metrics import match


def timed(fn, repeats=5):
    fn()
    times = []
    for _ in range(repeats):
        start = perf_counter()
        result = fn()
        times.append(perf_counter() - start)
    return float(np.median(times)), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[2]/'spotfitlm')
    parser.add_argument('--out', type=Path, default=Path('output/aguet_benchmark/results.json'))
    args = parser.parse_args()
    root = args.reference.resolve()
    rng = np.random.default_rng(19037)
    sigma = 1.45
    y, x = np.indices((128, 128))
    grid = np.stack(np.meshgrid(np.arange(20, 120, 28), np.arange(20, 120, 28)), axis=-1).reshape(-1, 2)
    truth, frames = [], []
    for _ in range(24):
        positions = grid + rng.uniform(-3, 3, grid.shape)
        mean = np.full(y.shape, 20.)
        for (yy, xx), peak in zip(positions, rng.uniform(7, 100, len(positions))):
            mean += peak*np.exp(-((y-yy)**2+(x-xx)**2)/(2*sigma*sigma))
        frames.append(rng.poisson(mean).astype(float))
        truth.append(positions)
    stack = np.stack(frames)
    roi = (y < 64) & (x < 64)
    detector = load_detector(root)(sigma)
    report = dict(seed=19037, shape=list(stack.shape), sigma=sigma, significance=.05,
                  reference_revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                  platform=platform.platform(), cases={})
    with tempfile.TemporaryDirectory() as tmp:
        libpath = Path(tmp)/'reference.so'
        flags = ['-dynamiclib','-undefined','dynamic_lookup'] if platform.system() == 'Darwin' else ['-shared','-fPIC']
        sources = [root/'c_src'/f'{name}.c' for name in
                   ('gfit','glm_core','user_funcs','objective_funcs','matrix_operations')]
        subprocess.run(['cc','-O2','-ffp-contract=off',*flags,'-I',sysconfig.get_path('include'),
                        *map(str,sources),'-lm','-o',str(libpath)],check=True)
        fit = ctypes.CDLL(str(libpath)).fit_symmetric_gaussian
        array = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        indices = np.ctypeslib.ndpointer(dtype=np.intc,flags='C_CONTIGUOUS')
        fit.argtypes = [array,indices,indices,ctypes.c_int,ctypes.c_int,ctypes.c_double,
                       ctypes.c_int,ctypes.c_int,ctypes.c_int,array]
        fit.restype = None

        def reference(mask):
            results = []
            for frame in stack:
                yy, xx = detector.detect_spots(frame)
                if mask is not None:
                    selected = mask[yy,xx]
                    yy, xx = yy[selected], xx[selected]
                out = np.empty((len(yy),12))
                fit(frame,yy.astype(np.intc),xx.astype(np.intc),128,128,sigma,len(yy),9,50,out)
                keep = (out[:,11]>=0)&(out[:,1]>0)&(out[:,4]>4)&(out[:,4]<124)&(out[:,5]>4)&(out[:,5]<124)
                results.append(out[keep])
            return results

        for label, mask in [('full',None),('roi',roi)]:
            ref_time, ref = timed(lambda: reference(mask))
            one_time, one = timed(lambda: localize_aguet_stack(stack,sigma,roi=mask,n_threads=1))
            five_time, five = timed(lambda: localize_aguet_stack(stack,sigma,roi=mask,n_threads=5))
            maximum_error = 0.
            for a,b,c in zip(ref,one,five):
                np.testing.assert_allclose(a[:,[5,4]],b.positions,rtol=1e-6,atol=1e-6)
                np.testing.assert_allclose(a[:,8],b.fit_sigma,rtol=1e-6,atol=1e-6)
                np.testing.assert_allclose(a[:,[7,6]],b.se[:,1:],rtol=1e-5,atol=1e-6)
                for name in ('positions','amplitudes','fit_sigma','se','sigma_se'):
                    np.testing.assert_array_equal(getattr(b,name),getattr(c,name))
                if len(b): maximum_error=max(maximum_error,float(np.max(np.abs(a[:,[5,4]]-b.positions))))
            report['cases'][label] = dict(reference_seconds=ref_time,native_serial_seconds=one_time,
                native_5_threads_seconds=five_time,counts=[len(r) for r in one],
                max_reference_position_difference=maximum_error,
                processed_pixels_per_frame=one[0].info['processed_pixels'])
            print(label,report['cases'][label],flush=True)
    sparse = localize_aguet_stack(stack,sigma,n_threads=5)
    scores = [match(t,r.positions,radius=1.) for t,r in zip(truth,sparse)]
    empty = rng.poisson(20.,(24,128,128)).astype(float)
    report['accuracy'] = dict(truth=sum(len(t) for t in truth),detected=sum(len(r) for r in sparse),
        matched=sum(s.n_matched for s in scores),
        empty_frame_counts=[len(r) for r in localize_aguet_stack(empty,sigma,n_threads=5)])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(report['accuracy'])
    print(args.out)


if __name__ == '__main__':
    main()
