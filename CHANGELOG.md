# Changelog

## 0.1.0

First tagged release.

- Sparse and multi-emitter localization with fitted widths, fluxes and uncertainty diagnostics.
- Brownian-motion trajectory linking using frame-to-frame LAP assignment.
- Localization diagnostics retained for downstream filtering; width calibration and aggregate rejection removed.
- A single Python distribution bundles the Rust extension.
- Tested wheels for Linux x86-64 and ARM64, macOS Intel and Apple Silicon, and Windows x86-64.
- GitHub Actions builds release assets and tests each wheel on Python 3.10 and 3.14.
