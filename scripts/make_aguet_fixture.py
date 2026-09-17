"""Freeze reference outputs from the user's spotfitlm checkout (requires cc).

This is a development tool, not a runtime dependency. It compiles the original
C fitter in a temporary directory and reads the detector class without loading
spotfitlm's installed extension or its DataFrame dependencies.
"""
import argparse
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import warnings

import numpy as np
import scipy.ndimage as ndi
import scipy.stats as stats


WRAPPER = r'''
#include "glm_core.h"
#include "objective_funcs.h"
#include "user_funcs.h"
#include <math.h>
int reference_fit(double *data, int side, double sigma, int limit, double *out, double *cov) {
    coord_data coords = meshgrid2d(side);
    double lo=data[0], hi=data[0];
    for (int i=1; i<side*side; i++) { if(data[i]<lo) lo=data[i]; if(data[i]>hi) hi=data[i]; }
    double p[5] = {0,0,sigma,hi-lo,lo};
    for (int i=0; i<25; i++) cov[i]=NAN;
    OptimizerResult r=dglm_der(&poisson_nll,&poisson_nll_grad,&poisson_nll_appx_hess,
        &symmetric_gaussian,&symmetric_gaussian_deriv,&symmetric_gaussian_poisson_nll_hess,
        p,data,5,side*side,limit,cov,&coords,0);
    for(int i=0;i<5;i++) out[i]=p[i];
    out[5]=r.num_iterations;
    double model[side*side];
    symmetric_gaussian(p,5,side*side,&coords,model);
    out[6]=poisson_nll(model,data,side*side);
    free_coord_data(&coords);
    return r.return_code;
}
'''


def load_detector(root):
    """Read the reference class without importing its optional UI dependencies."""
    text = (root / "spotfitlm/utils.py").read_text()
    detector = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.ClassDef) and node.name == "PointSourceDetector2D")
    namespace = dict(np=np, ndi=ndi, stats=stats)
    exec(compile(ast.Module(body=[detector], type_ignores=[]), str(root / "spotfitlm/utils.py"), "exec"), namespace)
    return namespace["PointSourceDetector2D"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[2] / "spotfitlm")
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/09_aguet.json"))
    args = parser.parse_args()
    root = args.reference.resolve()
    Detector = load_detector(root)
    rng = np.random.default_rng(8932)
    y, x = np.indices((48, 53))
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for sigma in (0.8, 1.45, 2.4):
            mean = np.full(y.shape, 20.)
            for yy, xx, amp in [(12.3, 13.7, 60), (31.8, 34.1, 100), (1.1, 47.2, 50), (20.4, 45.3, 12)]:
                mean += amp * np.exp(-((y-yy)**2+(x-xx)**2)/(2*sigma**2))
            data = rng.poisson(mean).astype(float)
            d = Detector(sigma)
            seeds = np.array(d.detect_spots(data)).T
            frames.append(dict(sigma=sigma, significance=.05, data=data.tolist(), seeds=seeds.tolist()))
        for value in (0., 20.):
            data = np.full((20, 23), value)
            seeds = np.array(Detector(1.2).detect_spots(data)).T
            frames.append(dict(sigma=1.2, significance=.05, data=data.tolist(), seeds=seeds.tolist()))
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "wrapper.c").write_text(WRAPPER)
        libpath = tmp / "reference.so"
        sources = [root / "c_src" / (name + ".c") for name in
                   ("glm_core", "user_funcs", "objective_funcs", "matrix_operations")]
        flags = ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-fPIC"]
        subprocess.run(["cc", "-O2", "-ffp-contract=off", *flags, "-I", str(root / "c_src"),
                        str(tmp / "wrapper.c"), *map(str, sources), "-lm", "-o", str(libpath)], check=True)
        fn = ctypes.CDLL(str(libpath)).reference_fit
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        fn.argtypes = [array, ctypes.c_int, ctypes.c_double, ctypes.c_int, array, array]
        fn.restype = ctypes.c_int
        for sigma in (.8, 1.3, 2.0):
            for amplitude in (15., 60., 200.):
                side = 11
                y, x = np.indices((side, side)) - side//2
                mean = 20 + amplitude * np.exp(-((x-.23)**2+(y+.31)**2)/(2*sigma*sigma))
                data = np.maximum(rng.poisson(mean).astype(float), 1.)
                for limit in (1, 100):
                    out, cov = np.empty(7), np.empty((5, 5))
                    status = fn(data, side, 1.3, limit, out, cov)
                    cases.append(dict(data=data.tolist(), sigma0=1.3, itermax=limit, status=status,
                                      theta=out[:5].tolist(), iterations=int(out[5]), objective=float(out[6]),
                                      covariance=cov.tolist() if np.isfinite(cov).all() else None))
    files = ["spotfitlm/utils.py", "spotfitlm/fitters.py", "c_src/gfit.c", "c_src/glm_core.c",
             "c_src/user_funcs.c", "c_src/objective_funcs.c", "c_src/matrix_operations.c"]
    report = dict(reference="spotfitlm", revision=subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
        seed=8932, candidate_frames=frames, fits=cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(frames)} candidate frames and {len(cases)} fits to {args.out}")


if __name__ == "__main__":
    main()
