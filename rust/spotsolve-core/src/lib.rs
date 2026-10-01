//! `spotsolve` core: emitter detection and localization.
//!
//! The detector is [`detect`], after u-track's `pointSourceDetection`:
//! [`prefilter`] screens the frame by a Poisson score test and proposes
//! LoG seeds; each is fitted on its own window by [`fit`] and kept if its
//! likelihood ratio reaches a threshold set by an expected false-positive
//! rate. `tests/layer7_localize.rs` holds it to recall, precision and
//! position error on simulated fields and to its false-positive rate on
//! pure noise.
//!
//! The rest are the layers it is built from: [`psf`] (the
//! pixel-integrated Gaussian and its derivatives), [`linalg`] (Cholesky),
//! [`filters`] (`scipy.ndimage`'s filters) and [`render`] (model images).
//! The linker is [`track`], on the exact assignment in [`lap`].
//!
//! # Portability
//!
//! Pure Rust. `libm` is the only dependency, no `build.rs`, no C toolchain, no
//! `cfg(target_os)`. `libm::erf` is a port of musl's, so it is bit-identical on
//! macOS, Linux and Windows. Do not enable fast-math, and do not build with
//! `-C target-cpu=native`: FMA contraction would change f64 results between
//! machines.

pub mod detect;
mod frames;
pub mod filters;
pub mod fit;
pub mod lap;
pub mod linalg;
pub mod prefilter;
pub mod psf;
pub mod render;
pub mod track;
