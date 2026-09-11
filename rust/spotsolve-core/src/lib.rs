//! `spotsolve` core: emitter detection and localization.
//!
//! The detector is [`boxsearch`]: in each box, an emitter exists iff it
//! lowers the Poisson deviance by `ADD_NATS`, and a whole frame -- the gain,
//! FIND, the background surface, the search, the polish and the
//! classification -- runs here, and the measurements behind its constants
//! are recorded beside them. Its Python reference was retired on 2026-09-11;
//! it is last present in commit `ea6b17f`, under `src/spotsolve/deprecated/`.
//!
//! The rest are the layers it is built from: [`psf`] (the
//! pixel-integrated Gaussian and its Jacobian), [`lmcl`] (the bounded Poisson
//! fitter), [`linalg`] (its Cholesky), [`filters`] (`scipy.ndimage`'s filters,
//! matched exactly), [`grid`] and [`patches`] (spatial grouping), [`render`]
//! (images and masks) and [`statistics`].
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

pub mod boxsearch;
pub mod filters;
pub mod grid;
pub mod lap;
pub mod linalg;
pub mod lmcl;
pub mod patches;
pub mod psf;
pub mod render;
pub mod statistics;
pub mod track;
pub mod trackparams;
