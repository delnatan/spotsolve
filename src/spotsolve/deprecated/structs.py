"""Data structures shared across the pipeline.

Kept separate from the algorithms so the contracts between modules are
readable in one place. Nothing here imports anything from the pipeline, so
this module is safe to import from anywhere.

Parameter vector layout
-----------------------
Every fit works on a flat `theta` of length 3K+1 for K emitters:

    theta = [b, A_0, y_0, x_0, A_1, y_1, x_1, ..., A_{K-1}, y_{K-1}, x_{K-1}]

`b` is one background shared by the patch. `A_k` is TOTAL FLUX, not peak
height -- the pixel-integrated Gaussian sums to A over all pixels, so a
peak-height guess must be divided by `psf.peak_factor(sigma)` before it can
be used as an A. Positions are in the patch's LOCAL pixel coordinates.

Units
-----
All intensities are PHOTOELECTRONS, d_e = (adu - offset) / gain. The
Poisson weighting W = 1/m used by the optimizer and by every Fisher matrix
is only valid in those units.
"""

from dataclasses import dataclass, field

import numpy as np


WIDTH_REJECT_DTYPE = np.dtype([
    ("source_index", np.int64),
    ("y", np.float64),
    ("x", np.float64),
    ("flux", np.float64),
    ("sigma", np.float64),
    ("sigma_ratio", np.float64),
    ("reason", "U12"),
])
"""One object the model fitted but the reporting band does not accept.

Emitted by `spotsolve.localize` and its reference `box.localize_boxes`.
`reason` is `"too_narrow"` or `"too_wide"`, or `"edge"` for an out-of-band fit
the frame border cuts; `source_index` indexes every fit of the frame, in the
order the search returned them. It lives here rather than in `core` because it is a contract
between modules, which is what this file is for.
"""


def width_reject_records(source, positions, amplitudes, sigmas, sigma, reason):
    """`WIDTH_REJECT_DTYPE` rows for objects the reporting band refused."""
    rec = np.empty(len(source), dtype=WIDTH_REJECT_DTYPE)
    rec["source_index"] = source
    rec["y"] = positions[:, 0] if len(positions) else np.empty(0)
    rec["x"] = positions[:, 1] if len(positions) else np.empty(0)
    rec["flux"] = amplitudes
    rec["sigma"] = sigmas
    rec["sigma_ratio"] = sigmas / sigma if len(sigmas) else sigmas
    rec["reason"] = reason
    return rec


@dataclass
class Patch:
    """A group of emitters fitted jointly, plus its pixel bounding box.

    `indices` are the FREE emitters (indices into the global arrays);
    `frozen_indices` are neighbours just outside the group, held fixed and
    folded into the model as a constant halo so flux is not double-counted
    at patch borders.
    """

    indices: np.ndarray
    frozen_indices: np.ndarray
    y0: int          # bbox origin, inclusive
    x0: int
    y1: int          # bbox end, exclusive
    x1: int

    @property
    def shape(self):
        return (self.y1 - self.y0, self.x1 - self.x0)


@dataclass
class FitResult:
    """Outcome of one bounded Levenberg-Marquardt fit at fixed K."""

    theta: np.ndarray
    I: float              # Poisson I-divergence at the solution
    F: np.ndarray         # (p,p) Fisher information at the solution
    n_iter: int
    converged: bool       # gradient converged (NOT merely "stopped")
    stalled: bool = False  # lambda saturated without an improving step


@dataclass
class DetectResult:
    """Outcome of a full detection run over one image."""

    positions: np.ndarray      # (N,2) global (y,x)
    amplitudes: np.ndarray     # (N,) total flux, photoelectrons
    sigma: float
    lam: float                 # detections per px^2
    A_s: float                 # mean detected flux, photoelectrons
    gain: float                # g_eff used, ADU per photoelectron
    background: np.ndarray     # (H,W) background surface, photoelectrons/px
    n_outer_passes: int
    model_image: np.ndarray
    residual: np.ndarray
    se: np.ndarray = None      # (N,3) CRLB (SE_A, SE_y, SE_x) where available
    history: list = field(default_factory=list)   # per-pass convergence record
    fit_sigma: np.ndarray = None
    """Per-emitter fitted sigma, aligned with `positions`.

    A RESULT, not a diagnostic: with a free width these are the widths that
    produced the reported positions, amplitudes and CRLBs, and `sigma_ratio`
    is a per-emitter defocus readout: a plain maximum-likelihood width.
    """
    sigma_ratio: np.ndarray = None
    width_rejects: np.ndarray = None
    width_filter: dict = field(default_factory=dict)
