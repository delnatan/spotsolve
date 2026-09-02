"""LAYER 7: physical in-focus PSF-width filtering."""

import sys

import numpy as np

from spotsolve import infocus, psf
from spotsolve.structs import DetectResult

fail = 0


def chk(name, ok, detail=""):
    global fail
    print(("  PASS  " if ok else "  FAIL  ") + name
          + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


def synthetic_case():
    sigma = 1.2
    gain = 2.3
    offset = 100.0
    shape = (64, 64)
    pos = np.array([[18.5, 18.5], [18.5, 43.5], [43.5, 31.5]])
    amp = np.array([1500.0, 900.0, 1400.0])
    sig = np.array([sigma, 0.65 * sigma, 1.8 * sigma])
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]] * 1.0
    clean = psf.model_var_sigma(
        psf.pack_var_sigma(25.0, amp, pos[:, 0], pos[:, 1], sig), yy, xx)
    raw = clean * gain + offset
    initial = DetectResult(
        positions=pos, amplitudes=amp, sigma=sigma,
        lam=len(amp) / float(np.prod(shape)), A_s=float(np.mean(amp)),
        gain=gain, background=np.full(shape, 25.0),
        n_outer_passes=0, model_image=clean, residual=np.zeros(shape),
        se=np.full((len(amp), 3), np.nan))
    return raw, initial, offset, gain


raw, initial, offset, gain = synthetic_case()
res = infocus.filter_in_focus(raw, initial, offset=offset, gain=gain,
                              sigma_ratio_min=0.8, sigma_ratio_max=1.2,
                              impl="py")

chk("physical filter keeps only the in-focus source",
    len(res.amplitudes) == 1,
    f"N={len(res.amplitudes)}")
chk("kept source remains near its true position",
    np.linalg.norm(res.positions[0] - initial.positions[0]) < 0.05,
    f"pos={res.positions[0]}")
chk("kept source carries a plausible diagnostic sigma",
    abs(res.sigma_ratio[0] - 1.0) < 0.05,
    f"sigma_ratio={res.sigma_ratio[0]:.3f}")
reasons = sorted(str(r) for r in res.width_rejects["reason"])
chk("reject reasons distinguish narrow and wide",
    reasons == ["too_narrow", "too_wide"],
    f"reasons={reasons}")
chk("narrow source is not rendered back as a kept emitter",
    not np.any(np.linalg.norm(res.positions - initial.positions[1], axis=1) < 1.0))

try:
    import spotsolve_rs  # noqa: F401
except Exception:
    chk("Rust in-focus filter parity skipped; extension not installed", True)
else:
    rs = infocus.filter_in_focus(raw, initial, offset=offset, gain=gain,
                                 sigma_ratio_min=0.8, sigma_ratio_max=1.2,
                                 impl="rs")
    chk("Rust and Python keep the same count",
        len(rs.amplitudes) == len(res.amplitudes))
    chk("Rust and Python agree on kept position",
        np.max(np.abs(rs.positions - res.positions)) < 1e-8,
        f"max diff {np.max(np.abs(rs.positions - res.positions)):.2e}")
    chk("Rust and Python agree on diagnostic sigma",
        np.max(np.abs(rs.sigma_ratio - res.sigma_ratio)) < 1e-8,
        f"max diff {np.max(np.abs(rs.sigma_ratio - res.sigma_ratio)):.2e}")


print(f"\n7: {fail} failure(s)")
sys.exit(1 if fail else 0)
