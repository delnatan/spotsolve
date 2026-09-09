//! `spotsolve` core: emitter detection and localization on one frame.
//!
//! A Rust port of the Python reference in this repository's root. `README.md`
//! describes what the algorithm does and why; `PORTING_NOTES.md` records the
//! implementation practices this port is built on, and its section numbers are
//! cited throughout as `[Pn]`.
//!
//! # What lives here, and what does not
//!
//! This crate owns the per-frame compute: every emitter fit, every evidence
//! evaluation, every window. It does **not** own `detect`'s round loop, the
//! `scipy.ndimage` filtering in `find_candidates`, or `background_map`'s
//! convolutions -- those stay in `core.py`. Every filter call in the pipeline
//! is per-*round* (~40 per frame, against ~700k fits) and profiles at 0.1-0.3%,
//! so reimplementing scipy's kernel construction and its three boundary modes
//! would buy 0.3% for the most fidelity-fragile code in the port.
//!
//! The entry points are therefore *passes*, not a `detect`: see `passes`.
//!
//! # Portability
//!
//! Pure Rust. `libm` is the only dependency, no `build.rs`, no C toolchain, no
//! `cfg(target_os)`. `libm::erf` is a port of musl's, so it is bit-identical on
//! macOS, Linux and Windows -- which is what makes the fingerprint test
//! meaningful on more than one machine. Do not enable fast-math, and do not
//! build with `-C target-cpu=native`: FMA contraction would change f64 results
//! between machines [P2].

pub mod evidence;
pub mod filters;
pub mod grid;
pub mod inference;
pub mod linalg;
pub mod lmcl;
pub mod moves;
pub mod passes;
pub mod patches;
pub mod psf;
pub mod render;
pub mod search;
pub mod sparse;
pub mod statistics;

pub mod affine;

pub mod geometry;

pub mod uncertainty;
