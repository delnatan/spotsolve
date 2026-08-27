"""Event recorder for the detection loop: a step-by-step trace of what the
search actually did, so a failure can be watched rather than inferred.

The pipeline is a nest of loops -- outer empirical-Bayes pass, re-seed round,
Jacobi sweep, per-patch model search, per-move fit -- and every summary
statistic it reports is taken at the top of that nest. When a dense region
comes out wrong, the summary cannot say WHICH loop failed: a bad initial peak,
a patch box that excluded the neighbour that would have explained it, a BIRTH
that was blocked by the conditioning guard, and a seed suppressed because it
fell within sigma of an existing emitter all look identical from outside.

This module records each of those decisions as it is made. It is OFF by
default and every hook is a single `if _REC is None: return`, so a
non-tracing run pays one attribute lookup per event and allocates nothing.

Usage
-----
    import tracer
    tracer.start(sigma=1.2)
    ...run detect()...
    rec = tracer.stop()
    rec.save("trace.npz")          # or iterate rec.events

Events carry the loop CONTEXT they were emitted under (pass / round / sweep /
patch), pushed by the hooks via `tracer.context(...)`, so a single flat event
list can be regrouped by any loop level afterwards without the emitters
having to know where they sit in the nest.
"""

import contextlib
import pickle

import numpy as np

__all__ = ["start", "stop", "active", "emit", "context", "set_context",
           "snap", "Recorder"]


class Recorder:
    """A flat, ordered list of events plus the meta of the run that made them."""

    def __init__(self, meta=None):
        self.meta = dict(meta or {})
        self.events = []
        self._ctx = {}

    def add(self, kind, payload):
        ev = {"seq": len(self.events), "kind": kind}
        ev.update(self._ctx)
        ev.update(payload)
        self.events.append(ev)
        return ev

    def of_kind(self, *kinds):
        return [e for e in self.events if e["kind"] in kinds]

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump({"meta": self.meta, "events": self.events}, fh, protocol=4)
        return path

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        r = Recorder(d["meta"])
        r.events = d["events"]
        return r

    def summary(self):
        from collections import Counter
        c = Counter(e["kind"] for e in self.events)
        return ", ".join(f"{k}={v}" for k, v in sorted(c.items()))


_REC = None


def start(**meta):
    global _REC
    _REC = Recorder(meta)
    return _REC


def stop():
    global _REC
    r, _REC = _REC, None
    return r


def active():
    return _REC is not None


def emit(kind, **payload):
    """Record one event. No-op (and no argument evaluation cost beyond the
    call itself) when tracing is off -- callers that must build an expensive
    payload should guard with `if tracer.active():` first."""
    if _REC is None:
        return None
    return _REC.add(kind, payload)


def set_context(**kw):
    """Set context keys for every event emitted from here on, until another
    call changes them. Unlike `context()` this does not pop -- it suits the
    strictly nested loop counters in detect(), where each level rewrites its
    own key on entry, and it costs no reindentation of the loop body."""
    if _REC is not None:
        _REC._ctx.update(kw)


@contextlib.contextmanager
def context(**kw):
    """Attach `kw` to every event emitted inside the block."""
    if _REC is None:
        yield
        return
    old = dict(_REC._ctx)
    _REC._ctx.update(kw)
    try:
        yield
    finally:
        _REC._ctx = old


def snap(a):
    """Copy an array into the trace. Events must not alias live buffers: the
    loop mutates `positions`/`amplitudes` in place between events, so a stored
    reference would show the FINAL state at every frame of the movie."""
    return None if a is None else np.array(a, copy=True)
