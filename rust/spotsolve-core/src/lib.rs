//! `spotsolve` core: emitter detection and localization.
//!
//! The detector is [`boxsearch`]: every change of count is proposed by one
//! statistic, the efficient score for one more reference-width emitter, and
//! decided by the likelihood ratio of the refit, against a threshold set by
//! an expected false-positive rate. Background, dispersion, seeds and
//! uncertainties run there; the seeds start one joint model of the frame,
//! [`joint`], which fits, removes and adds. The measurements behind the
//! constants are recorded beside them. The design was prototyped in Python
//! (`src/spotsolve/scoregate.py`, `scripts/jointfit_prototype.py`); since
//! 2026-09-24 the Rust is the reference, held by statistical tests on
//! simulated fields and pure noise (`tests/layer7_localize.rs`).
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
