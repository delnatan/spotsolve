//! `spotsolve` core: emitter detection and localization.
//!
//! The detector is [`detect`]: seeds are local maxima of the efficient
//! score for one more reference-width emitter, above a threshold set by an
//! expected false-positive rate; each starts as an emitter of one Poisson
//! model of the frame, [`model`], which fits, removes and adds by likelihood
//! ratios. `tests/layer7_localize.rs` holds it to recall, precision and
//! position error on simulated fields and to its false-positive rate on
//! pure noise.
//!
//! The rest are the layers it is built from: [`psf`] (the
//! pixel-integrated Gaussian and its derivatives), [`linalg`] (Cholesky,
//! banded solves), [`filters`] (`scipy.ndimage`'s filters, matched exactly),
//! [`grid`] (spatial grouping), [`render`] (model images) and
//! [`statistics`]. The linker is [`track`], [`lap`]
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
pub mod detect;
mod frames;
pub mod filters;
pub mod grid;
pub mod lap;
pub mod linalg;
pub mod model;
pub mod psf;
pub mod render;
pub mod statistics;
pub mod track;
pub mod trackparams;
