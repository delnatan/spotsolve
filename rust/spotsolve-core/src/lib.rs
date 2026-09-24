//! `spotsolve` core: emitter detection and localization.
//!
//! The detector is [`boxsearch`]: every change of count is decided by one
//! statistic, the efficient score for one more reference-width emitter,
//! against a threshold set by an expected false-positive rate. Background,
//! dispersion, seeds, per-seed windows and uncertainties all run here; the
//! one-pass result is then refined as one joint model of the frame by
//! [`joint`] (pinned by `tests/fixtures/11_joint.json`), and
//! the measurements behind its constants are recorded beside them. It was
//! prototyped in Python (`src/spotsolve/scoregate.py`, last present in
//! commit 7b7dfe7's successors on branch `scoregate-prototype`) and is pinned
//! to that prototype by `tests/fixtures/10_scoregate.json`.
//!
//! The rest are the layers it is built from: [`psf`] (the
//! pixel-integrated Gaussian and its Jacobian), [`lmcl`] (the bounded Poisson
//! fitter), [`linalg`] (its Cholesky), [`filters`] (`scipy.ndimage`'s filters,
//! matched exactly), [`grid`] and [`patches`] (spatial grouping), [`render`]
//! (images and masks) and [`statistics`]. The linker is [`track`], [`lap`]
//! and [`trackparams`].
//!
//! `PORTING_NOTES.md` records the implementation practices this port is
//! built on; its section numbers are cited throughout as `[Pn]`.
//!
//! # Portability
//!
//! Pure Rust. `libm` is the only dependency, no `build.rs`, no C toolchain, no
//! `cfg(target_os)`. `libm::erf` is a port of musl's, so it is bit-identical on
//! macOS, Linux and Windows. Do not enable fast-math, and do not build with
//! `-C target-cpu=native`: FMA contraction would change f64 results between
//! machines [P2].

pub mod aguet;
pub mod boxsearch;
mod frames;
pub mod filters;
pub mod grid;
pub mod joint;
pub mod lap;
pub mod linalg;
pub mod lmcl;
pub mod patches;
pub mod psf;
pub mod render;
pub mod statistics;
pub mod track;
pub mod trackparams;
