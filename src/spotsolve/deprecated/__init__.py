"""The Python reference implementation, kept until the Rust is hardened.

`box.localize_boxes` is the box search in Python -- the design record: every
constant's measurement lives in its notes, and `tests/test_localize.py`
holds the native detector (`spotsolve.localize`) to statistical parity with
it. The rest is what it is built from: `core` (FIND, the background surface,
the polish), `lmga` (the Python fitter every fixture is generated from),
`backend` (Python/Rust fitter dispatch), `patches`, `calibrate` and
`structs`.

Nothing in the production package imports from here. It is scheduled for
retirement once the Rust implementation is hardened; change the algorithm
here first only while it exists, measure, then port.
"""
