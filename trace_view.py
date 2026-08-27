"""Render a recorded detection trace as a step-by-step movie.

The trace (tracer.py) is a flat event list; this module turns it into frames.
Each frame shows the SAME four global panels -- data, model, normalized
residual, and the score-test map that audit.py thresholds -- so that what
changes between two frames is only what the algorithm changed. The bottom row
is the detail for whichever step the frame is about:

  * a `seed` event shows the LoG-filtered normalized residual the candidates
    were picked from, with every candidate drawn at the stage it died: raw
    local maximum, dropped at the border, dropped for sitting within sigma of
    an existing emitter, or accepted.
  * a `move` / `proposals` event zooms on the active patch and draws the
    configuration before and after, with the log Bayes factor of every
    proposal that was considered -- including the ones the conditioning guard
    blocked, which are otherwise invisible from outside.

The live configuration
----------------------
A sweep is Jacobi: every patch is searched against the same starting
configuration and the global arrays are rebuilt only at the end. So during a
sweep there is no single "current model" in the code at all. The frames show
the one the ACTIVE PATCH sees: the sweep's starting configuration with that
patch's own emitters replaced by its in-progress theta. That is the model the
next proposal is scored against, which is what the viewer needs in order to
judge the decision being made.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import audit
import calibrate
import psf

__all__ = ["Timeline", "select_frames", "render_frames", "write_movie",
           "write_log", "storyboard"]


def _in_box(pos, e):
    """Mask of `pos` lying inside event `e`'s fit region (pixel convention:
    pixel i covers [i-0.5, i+0.5))."""
    p = np.atleast_2d(np.asarray(pos, float))
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    return ((p[:, 0] >= e["y0"] - 0.5) & (p[:, 0] < e["y1"] - 0.5)
            & (p[:, 1] >= e["x0"] - 0.5) & (p[:, 1] < e["x1"] - 0.5))


def _theta_to_global(theta, y0, x0):
    """(positions, amplitudes) in image coordinates from a patch-local theta."""
    _, A, cy, cx = psf.unpack(theta)
    if A.size == 0:
        return np.empty((0, 2)), np.empty((0,))
    return np.stack([cy + y0, cx + x0], axis=1), A


class Timeline:
    """Replays a trace, carrying the live configuration from frame to frame.

    The trace records what each loop level DECIDED; the configuration those
    decisions apply to lives across events, so it has to be reconstructed
    here rather than stored on every event (which would multiply the trace
    size by the emitter count).
    """

    def __init__(self, rec):
        self.rec = rec
        init = next(e for e in rec.events if e["kind"] == "init")
        self.d_e = init["d_e"]
        self.raw = init["raw"]
        self.sigma = init["sigma"]
        self.shape = self.d_e.shape
        self.frames = []
        self._build()

    def _build(self):
        pos = np.empty((0, 2))
        amp = np.empty((0,))
        bg = 0.0
        d_e = self.d_e
        patches = []          # boxes of the sweep in progress
        base_pos, base_amp = pos, amp
        free_idx = None
        active = None
        pending_considered = None

        for e in self.rec.events:
            k = e["kind"]
            if k == "init":
                bg = e["background"]
                continue

            if k == "seed":
                # the model in the event is the one the seeding actually saw
                self.frames.append(dict(
                    kind="seed", ev=e, d_e=d_e, model=e["model"], pos=pos, amp=amp,
                    patches=patches, active=None, bg=bg,
                ))
                # seeds are appended by the caller right after
                if len(e["seeds"]):
                    pos = np.vstack([pos, e["seeds"]]) if len(pos) else e["seeds"]
                continue

            if k == "sweep_start":
                base_pos, base_amp = e["positions"], e["amplitudes"]
                pos, amp, bg = base_pos, base_amp, e["background"]
                patches = e["patches"]
                self.frames.append(dict(
                    kind="sweep_start", ev=e, d_e=d_e,
                    model=calibrate.render_model(pos, amp, self.sigma, self.shape, bg),
                    pos=pos, amp=amp, patches=patches, active=None, bg=bg,
                ))
                continue

            if k == "search_start":
                free_idx = e.get("free_idx")
                active = e
                continue

            if k == "box_seed":
                self.frames.append(dict(
                    kind="box_seed", ev=e, d_e=d_e,
                    model=calibrate.render_model(pos, amp, self.sigma, self.shape, bg),
                    pos=pos, amp=amp, patches=patches, active=e, bg=bg,
                ))
                continue

            if k == "commit":
                pos, amp = e["positions"], e["amplitudes"]
                base_pos, base_amp = pos, amp
                self.frames.append(dict(
                    kind="commit", ev=e, d_e=d_e,
                    model=calibrate.render_model(pos, amp, self.sigma, self.shape, bg),
                    pos=pos, amp=amp, patches=patches, active=e, bg=bg,
                ))
                continue

            if k in ("move", "proposals"):
                if free_idx is None and "y0" not in e:
                    continue
                if k == "proposals" and e.get("accepted"):
                    # The `move` event right after says the same, except for the
                    # slate of alternatives -- carry that across so the accepted
                    # frame can show what the winning move was chosen AGAINST.
                    pending_considered = e.get("considered", [])
                    continue
                theta = e["theta_after"] if k == "move" else e["theta"]
                if free_idx is not None:
                    keep = np.setdiff1d(np.arange(len(base_amp)),
                                        np.asarray(free_idx, int))
                else:
                    # boxsolve: the free set is geometric, not a stored index
                    # list -- everything committed inside the fit region is a
                    # free variable of this box and is superseded by `theta`.
                    keep = np.nonzero(~_in_box(base_pos, e))[0]
                gp, ga = _theta_to_global(theta, e["y0"], e["x0"])
                live_pos = np.vstack([base_pos[keep], gp]) if len(keep) else gp
                live_amp = np.concatenate([base_amp[keep], ga]) if len(keep) else ga
                # the patch fits its own background; the rest of the frame keeps
                # the sweep's global one, which is what render_model was handed
                ev = e
                if k == "move" and pending_considered:
                    ev = dict(e, considered=pending_considered)
                    pending_considered = None
                self.frames.append(dict(
                    kind=k, ev=ev, d_e=d_e,
                    model=calibrate.render_model(live_pos, live_amp, self.sigma,
                                              self.shape, bg),
                    pos=live_pos, amp=live_amp, patches=patches, active=ev,
                    search=active, bg=bg,
                ))
                continue

            if k == "adopt":
                # Emitters recovered from a core seam after every box has run;
                # see boxsolve._adopt_orphans.
                pos, amp = e["positions"], e["amplitudes"]
                base_pos, base_amp = pos, amp
                self.frames.append(dict(
                    kind="adopt", ev=e, d_e=d_e,
                    model=calibrate.render_model(pos, amp, self.sigma, self.shape, bg),
                    pos=pos, amp=amp, patches=patches, active=None, bg=bg,
                ))
                continue

            if k == "sweep_end":
                pos, amp, bg = e["positions"], e["amplitudes"], e["background"]
                base_pos, base_amp = pos, amp
                free_idx = None
                self.frames.append(dict(
                    kind="sweep_end", ev=e, d_e=d_e,
                    model=calibrate.render_model(pos, amp, self.sigma, self.shape, bg),
                    pos=pos, amp=amp, patches=patches, active=None, bg=bg,
                ))
                continue

            if k == "eb_update":
                pos, amp, bg = e["positions"], e["amplitudes"], e["background"]
                base_pos, base_amp = pos, amp
                self.frames.append(dict(
                    kind="eb_update", ev=e, d_e=d_e, model=e["model"],
                    pos=pos, amp=amp, patches=[], active=None, bg=bg,
                ))
                # the gain update rescales the electron image for every frame
                # that follows; without this the movie would keep showing the
                # first pass's units after the loop has left them
                continue

            if k == "final":
                d_e = e["d_e"]
                pos, amp, bg = e["positions"], e["amplitudes"], e["background"]
                self.frames.append(dict(
                    kind="final", ev=e, d_e=d_e, model=e["model"],
                    pos=pos, amp=amp, patches=[], active=None, bg=bg,
                ))
                continue

    def __len__(self):
        return len(self.frames)


# --------------------------------------------------------------- drawing


def _ctx_label(e):
    o, r, s = e.get("outer", "?"), e.get("rnd", "?"), e.get("sweep", "?")
    rr = {-1: "seed", -2: "final"}.get(r, r)
    return f"pass {o} / round {rr} / sweep {s}"


def _nr(d_e, model):
    return (d_e - model) / np.sqrt(np.maximum(model, 1e-6))


def _draw_frame(fig, fr, sigma, vmax_data, z_clip=8.0):
    fig.clf()
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.22, wspace=0.12)
    ax = [fig.add_subplot(gs[0, i]) for i in range(4)]
    bx = [fig.add_subplot(gs[1, i]) for i in range(4)]

    d_e, model, pos = fr["d_e"], fr["model"], fr["pos"]
    nr = _nr(d_e, model)
    z = audit.score_map(d_e, model, sigma)

    ax[0].imshow(d_e, cmap="gray", vmin=0, vmax=vmax_data)
    ax[0].set_title("data (e-)", fontsize=9)
    if len(pos):
        ax[0].plot(pos[:, 1], pos[:, 0], "r+", ms=6, mew=1.0)
    for p in fr["patches"]:
        ax[0].add_patch(Rectangle((p["x0"] - .5, p["y0"] - .5),
                                  p["x1"] - p["x0"], p["y1"] - p["y0"],
                                  fill=False, ec="deepskyblue", lw=0.5, alpha=0.6))
    a = fr.get("active")
    if a is not None:
        ax[0].add_patch(Rectangle((a["x0"] - .5, a["y0"] - .5),
                                  a["x1"] - a["x0"], a["y1"] - a["y0"],
                                  fill=False, ec="yellow", lw=1.8))
        if "cy0" in a:   # the core: the only part of the box that gets committed
            ax[0].add_patch(Rectangle((a["cx0"] - .5, a["cy0"] - .5),
                                      a["cx1"] - a["cx0"], a["cy1"] - a["cy0"],
                                      fill=False, ec="lime", lw=1.6, ls="--"))

    ax[1].imshow(model, cmap="gray", vmin=0, vmax=vmax_data)
    ax[1].set_title(f"model  (N={len(pos)}, bg={fr['bg']:.2f})", fontsize=9)

    ax[2].imshow(nr, cmap="RdBu_r", vmin=-5, vmax=5)
    ax[2].set_title("normalized residual", fontsize=9)

    ax[3].imshow(z, cmap="RdBu_r", vmin=-z_clip, vmax=z_clip)
    p_, n_ = audit.residual_peaks(d_e, model, sigma, z_thresh=5.0)
    if len(p_):
        ax[3].scatter(p_[:, 1], p_[:, 0], s=90, facecolors="none",
                      edgecolors="yellow", lw=1.2)
    if len(n_):
        ax[3].scatter(n_[:, 1], n_[:, 0], s=90, facecolors="none",
                      edgecolors="cyan", lw=1.2)
    ax[3].set_title(f"score test z  ({len(p_)} missed / {len(n_)} piled)", fontsize=9)

    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])

    if fr["kind"] == "seed":
        _draw_seed_detail(bx, fr)
    elif fr["kind"] == "box_seed":
        _draw_box_seed_detail(bx, fr)
    elif fr["kind"] == "commit":
        _draw_commit_detail(bx, fr, vmax_data)
    elif fr["kind"] in ("move", "proposals"):
        _draw_patch_detail(bx, fr, sigma, vmax_data)
    else:
        _draw_plain_detail(bx, fr, sigma)
    for b_ in bx:
        b_.set_xticks([]); b_.set_yticks([])
    return fig


def _draw_seed_detail(bx, fr):
    """Every candidate, drawn at the stage it died."""
    e = fr["ev"]
    raw, on_frame, seeds = e["raw"], e["on_frame"], e["seeds"]
    log_f, nr = e["log_f"], e["nr"]

    bx[0].imshow(nr, cmap="magma", vmin=0, vmax=np.percentile(nr, 99.5))
    bx[0].set_title("normalized residual (seeding input)", fontsize=9)

    v = np.percentile(log_f, 99.5)
    bx[1].imshow(log_f, cmap="magma", vmin=0, vmax=max(v, e["threshold"]))
    bx[1].set_title(f"LoG(nr), threshold {e['threshold']}", fontsize=9)

    bx[2].imshow(log_f, cmap="gray", vmin=0, vmax=max(v, e["threshold"]))
    def _set(a, b):
        """rows of `a` not present in `b`"""
        if len(a) == 0:
            return a
        if len(b) == 0:
            return a
        keep = [i for i, r in enumerate(a)
                if not np.any(np.all(np.isclose(b, r), axis=1))]
        return a[keep] if keep else np.empty((0, 2))
    dropped_border = _set(raw, on_frame)
    dropped_near = _set(on_frame, seeds)
    for arr, c, lab in ((dropped_border, "orange", "dropped: border"),
                        (dropped_near, "red", "dropped: within sigma of an emitter"),
                        (seeds, "lime", "accepted seed")):
        if len(arr):
            bx[2].plot(arr[:, 1], arr[:, 0], "x", color=c, ms=7, mew=1.5, label=lab)
    ex = e.get("existing")
    if ex is not None and len(ex):
        bx[2].plot(np.asarray(ex)[:, 1], np.asarray(ex)[:, 0], "o", mfc="none",
                   mec="deepskyblue", ms=7, mew=1.0, label="existing emitter")
    bx[2].legend(fontsize=6, loc="upper right", framealpha=0.8)
    bx[2].set_title(f"candidates: {len(raw)} raw -> {len(seeds)} seeded", fontsize=9)

    bx[3].axis("off")
    bx[3].text(0.0, 1.0,
               f"{_ctx_label(e)}\n\n"
               f"raw local maxima      {len(raw)}\n"
               f"survived border cull  {len(on_frame)}\n"
               f"survived proximity    {len(seeds)}\n"
               f"existing emitters     {0 if ex is None else len(ex)}\n\n"
               f"local-max window      {2 * int(np.ceil(e['sigma'])) + 1} px\n"
               f"proximity cull radius {e['sigma']:.2f} px",
               va="top", ha="left", fontsize=8, family="monospace",
               transform=bx[3].transAxes)


def _draw_box_seed_detail(bx, fr):
    """What this box saw before its search: its own from-scratch residual and
    the candidates it starts from."""
    e = fr["ev"]
    bx[0].imshow(e["sub"], cmap="gray")
    bx[0].set_title(f"box @({e['y0']},{e['x0']}) data", fontsize=9)
    v = np.percentile(e["nr"], 99.5)
    bx[1].imshow(e["nr"], cmap="magma", vmin=0, vmax=max(v, 1.0))
    bx[1].set_title("box residual, from scratch", fontsize=9)
    lf = e["log_f"]
    bx[2].imshow(lf, cmap="gray", vmin=0, vmax=max(np.percentile(lf, 99.5),
                                                   e["threshold"]))
    ex = e.get("existing")
    if ex is not None and len(ex):
        bx[2].plot(ex[:, 1], ex[:, 0], "o", mfc="none", mec="deepskyblue", ms=8,
                   mew=1.2, label="committed, re-freed here")
    if len(e["seeds"]):
        bx[2].plot(e["seeds"][:, 1], e["seeds"][:, 0], "x", color="lime", ms=8,
                   mew=1.6, label="new candidate")
    if (ex is not None and len(ex)) or len(e["seeds"]):
        bx[2].legend(fontsize=6, loc="upper right", framealpha=0.8)
    bx[2].set_title(f"LoG(box residual): {len(e['seeds'])} new candidate(s)",
                    fontsize=9)
    bx[3].axis("off")
    bx[3].text(0.0, 1.0,
               f"{_ctx_label(e)}\n\nbox   y {e['y0']}:{e['y1']}  x {e['x0']}:{e['x1']}\n"
               f"core  y {e['cy0']}:{e['cy1']}  x {e['cx0']}:{e['cx1']}\n\n"
               f"committed emitters re-freed here: "
               f"{0 if ex is None else len(ex)}\n"
               f"new LoG candidates:               {len(e['seeds'])}",
               va="top", ha="left", fontsize=8, family="monospace",
               transform=bx[3].transAxes)


def _draw_commit_detail(bx, fr, vmax_data):
    """What the box decided to keep, and what it threw away."""
    e = fr["ev"]
    pad = 2
    d_z, (oy, ox) = _zoom(fr["d_e"], e["y0"], e["y1"], e["x0"], e["x1"], pad)
    bx[0].imshow(d_z, cmap="gray", vmin=0, vmax=vmax_data)
    if len(e["committed"]):
        bx[0].plot(e["committed"][:, 1] - ox, e["committed"][:, 0] - oy, "+",
                   color="lime", ms=11, mew=1.8, label="committed (in core)")
    if len(e["discarded"]):
        bx[0].plot(e["discarded"][:, 1] - ox, e["discarded"][:, 0] - oy, "x",
                   color="red", ms=9, mew=1.5, label="discarded (pad ring)")
    bx[0].add_patch(Rectangle((e["x0"] - ox - .5, e["y0"] - oy - .5),
                              e["x1"] - e["x0"], e["y1"] - e["y0"],
                              fill=False, ec="yellow", lw=1.2))
    bx[0].add_patch(Rectangle((e["cx0"] - ox - .5, e["cy0"] - oy - .5),
                              e["cx1"] - e["cx0"], e["cy1"] - e["cy0"],
                              fill=False, ec="lime", lw=1.4, ls="--"))
    if len(e["committed"]) or len(e["discarded"]):
        bx[0].legend(fontsize=6, loc="upper right", framealpha=0.8)
    bx[0].set_title("box result: core kept, ring discarded", fontsize=9)
    for b in bx[1:3]:
        b.axis("off")
    bx[3].axis("off")
    bx[3].text(0.0, 1.0,
               f"{_ctx_label(e)}\n\n"
               f"moves accepted in this box:\n  "
               + (", ".join(e["accepted"]) if e["accepted"] else "(none)")
               + f"\n\ncommitted (core):   {len(e['committed'])}\n"
                 f"discarded (ring):   {len(e['discarded'])}\n"
                 f"global N now:       {len(e['positions'])}",
               va="top", ha="left", fontsize=8, family="monospace",
               transform=bx[3].transAxes)


def _zoom(arr, y0, y1, x0, x1, pad):
    H, W = arr.shape
    a = max(0, y0 - pad); b = min(H, y1 + pad)
    c = max(0, x0 - pad); d = min(W, x1 + pad)
    return arr[a:b, c:d], (a, c)


def _draw_patch_detail(bx, fr, sigma, vmax_data):
    e = fr["ev"]
    pad = 3
    y0, y1, x0, x1 = e["y0"], e["y1"], e["x0"], e["x1"]
    d_z, (oy, ox) = _zoom(fr["d_e"], y0, y1, x0, x1, pad)
    m_z, _ = _zoom(fr["model"], y0, y1, x0, x1, pad)
    nr_z = (d_z - m_z) / np.sqrt(np.maximum(m_z, 1e-6))

    before = e["theta_before"] if fr["kind"] == "move" else e["theta"]
    after = e["theta_after"] if fr["kind"] == "move" else None
    pb, ab = _theta_to_global(before, y0, x0)
    pa, aa = _theta_to_global(after, y0, x0) if after is not None else (None, None)

    bx[0].imshow(d_z, cmap="gray", vmin=0, vmax=vmax_data)
    if len(pb):
        bx[0].plot(pb[:, 1] - ox, pb[:, 0] - oy, "o", mfc="none", mec="red",
                   ms=9, mew=1.2, label="before")
    if pa is not None and len(pa):
        bx[0].plot(pa[:, 1] - ox, pa[:, 0] - oy, "+", color="lime", ms=10,
                   mew=1.6, label="after")
    bx[0].add_patch(Rectangle((x0 - ox - .5, y0 - oy - .5), x1 - x0, y1 - y0,
                              fill=False, ec="yellow", lw=1.2))
    if len(pb) or (pa is not None and len(pa)):
        bx[0].legend(fontsize=6, loc="upper right", framealpha=0.8)
    bx[0].set_title(f"patch {e.get('patch')} @({y0},{x0})  data", fontsize=9)

    bx[1].imshow(m_z, cmap="gray", vmin=0, vmax=vmax_data)
    bx[1].set_title("patch model (live)", fontsize=9)
    bx[2].imshow(nr_z, cmap="RdBu_r", vmin=-5, vmax=5)
    bx[2].set_title("patch normalized residual", fontsize=9)

    bx[3].axis("off")
    lines = [_ctx_label(e), ""]
    if fr["kind"] == "move":
        lines += [f"ACCEPTED  {e['label']}   log BF = {e['log_bf']:+.2f}",
                  f"  I {e['I_before']:.2f} -> {e['I_after']:.2f}",
                  f"  K {(len(before)-1)//3} -> {(len(after)-1)//3}",
                  f"  fit: converged={e['converged']} stalled={e['stalled']} "
                  f"iters={e['n_iter']}", ""]
    else:
        lines += [f"STOP: best move {e.get('chosen')} "
                  f"log BF {e.get('log_bf', float('nan')):+.2f} <= 0"
                  if e.get("chosen") else
                  "STOP: every proposal was blocked by the guard", ""]
    lines.append("proposals considered:")
    for lab, bf, cond, blocked, _th in e.get("considered", [])[:12]:
        flag = f"  BLOCKED cond={cond:.1e}" if blocked else ""
        lines.append(f"  {lab:12s} {bf:+9.2f}{flag}")
    bx[3].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=7,
               family="monospace", transform=bx[3].transAxes)


def _draw_plain_detail(bx, fr, sigma):
    e = fr["ev"]
    for b in bx[:3]:
        b.axis("off")
    bx[3].axis("off")
    nr = _nr(fr["d_e"], fr["model"])
    a = audit.audit_result(fr["d_e"], fr["model"], sigma)
    txt = [f"{_ctx_label(e)}", "", f"event: {fr['kind']}",
           f"N = {len(fr['pos'])}   background = {fr['bg']:.3f}",
           f"normalized residual: median {np.median(nr):+.3f}, "
           f"robust sd {0.5*(np.percentile(nr,84.1)-np.percentile(nr,15.9)):.3f}",
           f"audit: {a['n_missed']} missed, {a['n_piled']} piled, "
           f"z sd {a['z_robust_std']:.2f}, z max {a['z_max']:+.1f}"]
    if fr["kind"] == "eb_update":
        txt += ["", f"lambda = {e['lam']:.4f}/px^2   A_s = {e['A_s']:.1f}",
                f"gain = {e['gain']:.3f}   gain_ratio = {e['gain_ratio']:.3f}"]
    if fr["kind"] == "sweep_end":
        txt += ["", f"patches searched: {e['n_patches']}   changed: {e['changed']}"]
    bx[1].axis("off")
    bx[1].text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=9,
               family="monospace", transform=bx[1].transAxes)


def select_frames(frames, focus=None, kinds=None, stride=1, limit=None):
    """Narrow the movie to what is worth watching.

    `focus=(y, x, r)` keeps only frames whose ACTIVE PATCH box comes within r
    pixels of (y, x) -- the whole workflow of "the audit says the model is
    wrong here, show me every decision that was made there" is this argument.
    Global checkpoints (seed, sweep boundaries, empirical-Bayes, final) are
    always kept, so the focused movie still shows the context each local
    decision was made in.
    """
    sel = frames
    if kinds:
        sel = [f for f in sel if f["kind"] in kinds]
    if focus is not None:
        fy, fx, fr = focus

        def near(f):
            a = f.get("active")
            if a is None:
                return f["kind"] in ("seed", "sweep_start", "sweep_end",
                                     "eb_update", "final")
            dy = max(a["y0"] - fy, fy - (a["y1"] - 1), 0)
            dx = max(a["x0"] - fx, fx - (a["x1"] - 1), 0)
            return np.hypot(dy, dx) <= fr

        sel = [f for f in sel if near(f)]
    sel = sel[::stride]
    return sel[:limit] if limit else sel


def render_frames(tl, outdir, dpi=90, stride=1, limit=None, focus=None, kinds=None):
    os.makedirs(outdir, exist_ok=True)
    vmax = float(np.percentile(tl.d_e, 99.8))
    fig = plt.figure(figsize=(16, 8))
    paths = []
    sel = select_frames(tl.frames, focus=focus, kinds=kinds,
                        stride=stride, limit=limit)
    for i, fr in enumerate(sel):
        _draw_frame(fig, fr, tl.sigma, vmax)
        fig.suptitle(f"[{i+1}/{len(sel)}] {fr['kind']}  --  {_ctx_label(fr['ev'])}",
                     fontsize=11)
        p = os.path.join(outdir, f"frame_{i:05d}.png")
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def write_movie(frame_dir, out, fps=4):
    """Assemble the frames with ffmpeg if it is available, else a GIF."""
    import shutil
    import subprocess
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
               "-i", os.path.join(frame_dir, "frame_%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out]
        subprocess.run(cmd, check=True)
        return out
    from PIL import Image
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    ims = [Image.open(os.path.join(frame_dir, f)) for f in files]
    gif = os.path.splitext(out)[0] + ".gif"
    ims[0].save(gif, save_all=True, append_images=ims[1:],
                duration=int(1000 / fps), loop=0)
    return gif


# ------------------------------------------------------------------- text


def write_log(rec, path):
    """A compact grep-able transcript: one line per decision.

    This is the fastest way into a trace. The movie shows what a step looked
    like; the log shows how many times it happened and where, which is what
    identifies a region worth cropping.
    """
    out = []
    for e in rec.events:
        k, c = e["kind"], _ctx_label(e)
        if k == "init":
            out.append(f"init  sigma={e['sigma']} gain={e['gain']:.3f} "
                       f"bg={e['background']:.3f} k_max={e['k_max']} "
                       f"border_margin={e['border_margin']}")
        elif k == "seed":
            out.append(f"{c}  SEED  raw={len(e['raw'])} on_frame={len(e['on_frame'])} "
                       f"accepted={len(e['seeds'])} existing="
                       f"{0 if e['existing'] is None else len(e['existing'])}")
            for y, x in np.atleast_2d(e["seeds"]) if len(e["seeds"]) else []:
                out.append(f"        + seed  y={y:5.1f} x={x:5.1f}")
        elif k == "sweep_start":
            sizes = [len(p["indices"]) for p in e["patches"]]
            out.append(f"{c}  SWEEP start  N={len(e['positions'])} "
                       f"patches={len(e['patches'])} K per patch="
                       f"{sorted(sizes, reverse=True)[:8]} bg={e['background']:.3f}")
        elif k == "search_start":
            out.append(f"{c}  patch {e.get('patch')} @({e['y0']},{e['x0']})-"
                       f"({e['y1']},{e['x1']})  K0={len(e['init_amp'])} "
                       f"frozen={len(e.get('frozen_idx', []))} I={e['I']:.2f}"
                       f"{'' if e['converged'] else '  NOT-CONVERGED'}"
                       f"{'  STALLED' if e['stalled'] else ''}")
        elif k == "move":
            out.append(f"{c}  patch {e.get('patch')} @({e['y0']},{e['x0']})  "
                       f"ACCEPT {e['label']:12s} logBF={e['log_bf']:+8.2f}  "
                       f"K {(len(e['theta_before'])-1)//3} -> "
                       f"{(len(e['theta_after'])-1)//3}"
                       f"{'' if e['converged'] else '  NOT-CONVERGED'}"
                       f"{'  STALLED' if e['stalled'] else ''}")
        elif k == "proposals" and not e.get("accepted"):
            blocked = [(l, b, cd) for l, b, cd, bl, _ in e.get("considered", []) if bl]
            tail = ""
            if blocked:
                tail = "  blocked: " + ", ".join(
                    f"{l}({b:+.1f},cond={cd:.0e})" for l, b, cd in blocked[:4])
            out.append(f"{c}  patch {e.get('patch')} @({e['y0']},{e['x0']})  STOP "
                       f"best={e.get('chosen')} logBF={e.get('log_bf', float('nan')):+.2f}"
                       f"{tail}")
        elif k == "adopt":
            pts = ", ".join(f"({y:.2f},{x:.2f})" for y, x in e["adopted"])
            out.append(f"{c}  ADOPT {len(e['adopted'])} from core seams: {pts}"
                       f"   N -> {len(e['positions'])}")
        elif k == "sweep_end":
            out.append(f"{c}  SWEEP end    N={len(e['positions'])} "
                       f"changed={e['changed']} bg={e['background']:.3f}")
        elif k == "eb_update":
            out.append(f"{c}  EB  N={len(e['positions'])} lam={e['lam']:.4f} "
                       f"A_s={e['A_s']:.1f} gain={e['gain']:.3f} "
                       f"gain_ratio={e['gain_ratio']:.3f}")
        elif k == "final":
            out.append(f"FINAL  N={len(e['positions'])} gain={e['gain']:.3f} "
                       f"bg={e['background']:.3f}")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return path


def storyboard(tl, path, sigma=None):
    """One row per checkpoint frame (seed / sweep_end / eb_update / final), so
    the whole run fits on a page and the pass where it went wrong is obvious
    without scrubbing the movie."""
    sigma = sigma or tl.sigma
    keys = [f for f in tl.frames
            if f["kind"] in ("seed", "eb_update", "final")]
    n = len(keys)
    if n == 0:
        return None
    fig, ax = plt.subplots(n, 4, figsize=(13, 3.1 * n), squeeze=False)
    vmax = float(np.percentile(tl.d_e, 99.8))
    for i, fr in enumerate(keys):
        nr = _nr(fr["d_e"], fr["model"])
        a = audit.audit_result(fr["d_e"], fr["model"], sigma)
        ax[i][0].imshow(fr["d_e"], cmap="gray", vmin=0, vmax=vmax)
        if len(fr["pos"]):
            ax[i][0].plot(fr["pos"][:, 1], fr["pos"][:, 0], "r+", ms=5, mew=0.9)
        ax[i][0].set_ylabel(f"{fr['kind']}\n{_ctx_label(fr['ev'])}", fontsize=7)
        ax[i][1].imshow(fr["model"], cmap="gray", vmin=0, vmax=vmax)
        ax[i][2].imshow(nr, cmap="RdBu_r", vmin=-5, vmax=5)
        z = audit.score_map(fr["d_e"], fr["model"], sigma)
        ax[i][3].imshow(z, cmap="RdBu_r", vmin=-8, vmax=8)
        if len(a["positive"]):
            ax[i][3].scatter(a["positive"][:, 1], a["positive"][:, 0], s=70,
                             facecolors="none", edgecolors="yellow", lw=1.0)
        if len(a["negative"]):
            ax[i][3].scatter(a["negative"][:, 1], a["negative"][:, 0], s=70,
                             facecolors="none", edgecolors="cyan", lw=1.0)
        ax[i][0].set_title(f"data  N={len(fr['pos'])}", fontsize=8)
        ax[i][1].set_title("model", fontsize=8)
        ax[i][2].set_title("norm. residual", fontsize=8)
        ax[i][3].set_title(f"score z: {a['n_missed']} missed / {a['n_piled']} piled",
                           fontsize=8)
        for a_ in ax[i]:
            a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ------------------------------------------------------------ diagnostics


def summarize(rec, max_sweeps=12):
    """The two tables that say whether the search is working at all.

    A sweep is supposed to be a step toward a fixed point: `sweep_patches`
    returns `changed=False` when no patch accepted anything, and the round
    ends. If no round ever reaches that -- if every one exhausts `max_sweeps`
    -- then the emitter count reported is not the search's answer, it is
    whichever half of an oscillation the loop happened to stop on. The first
    table shows N in and out of every sweep, so a two-cycle is visible at a
    glance; the second attributes every accepted move and every blocked
    proposal to the rule that caused it.
    """
    from collections import Counter
    ev = rec.events
    rows, cur = [], None
    for e in ev:
        if e["kind"] == "sweep_start":
            cur = {"N0": len(e["positions"]), "m": Counter()}
        elif e["kind"] == "move" and cur is not None:
            cur["m"][e["label"].split("[")[0]] += 1
        elif e["kind"] == "sweep_end" and cur is not None:
            rows.append((e.get("outer"), e.get("rnd"), e.get("sweep"),
                         cur["N0"], len(e["positions"]), dict(cur["m"])))
            cur = None

    out = ["sweep table -- N entering and leaving each sweep",
           f"{'pass':>4} {'rnd':>4} {'swp':>4} {'N in':>5} {'N out':>6}  moves accepted"]
    for r in rows:
        out.append(f"{r[0]:>4} {str(r[1]):>4} {r[2]:>4} {r[3]:>5} {r[4]:>6}  {r[5]}")

    byround = {}
    for r in rows:
        byround.setdefault((r[0], r[1]), []).append(r)
    n_conv = sum(1 for v in byround.values() if len(v) < max_sweeps)
    fixed = sum(1 for r in rows if not r[5])
    out += ["",
            f"rounds reaching a fixed point before max_sweeps={max_sweeps}: "
            f"{n_conv} of {len(byround)}",
            f"sweeps that accepted nothing (a real fixed point): {fixed} of {len(rows)}"]

    mv = Counter(e["label"].split("[")[0] for e in ev if e["kind"] == "move")
    forced = sum(1 for e in ev if e["kind"] == "move"
                 and e["label"].startswith("death") and np.isinf(e["log_bf"]))
    out += ["", "accepted moves: " + str(dict(mv)),
            f"  deaths: {forced} FORCED (+inf: amplitude unresolved, "
            f"evidence not computable) vs {mv['death'] - forced} on a finite "
            f"Bayes factor"]

    tot = blocked_resolved = blocked_cond = 0
    for e in ev:
        if e["kind"] == "proposals":
            for _lab, _bf, cond, bl, _th in e.get("considered", []):
                tot += 1
                if bl:
                    if np.isinf(cond):
                        blocked_resolved += 1
                    else:
                        blocked_cond += 1
    out += [f"proposals scored: {tot}; blocked {blocked_resolved + blocked_cond} "
            f"({blocked_resolved} by the A/SE >= RESOLVED_TAU guard, "
            f"{blocked_cond} by the numeric conditioning guard)"]
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    import tracer
    print(summarize(tracer.Recorder.load(sys.argv[1])))
