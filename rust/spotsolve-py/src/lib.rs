//! PyO3 bindings for `spotsolve-core`, exposed to Python as `spotsolve_rs`.
//!
//! The boundary sits at the FRAME for detection: `detect_localize` and
//! `detect_localize_stack` (see `spotsolve_core::detect`) run whole frames natively with the
//! GIL released. For linking it sits at the MOVIE: `track_link`
//! (see `track`) takes a whole table of localizations.
//!
//! Arrays must be C-contiguous f64; `as_slice()` fails loudly otherwise rather
//! than silently transposing.

use pyo3::prelude::*;

mod aguet;
mod detect;
mod track;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn spotsolve_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    aguet::register(m)?;
    detect::register(m)?;
    track::register(m)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
