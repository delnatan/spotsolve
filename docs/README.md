# Development documentation

The single active plan and current implementation direction are in
[Focused-emitter implementation](FOCUSED_EMITTER_PROPOSAL.md).

Priority: meaningful localization uncertainty; frame-to-frame statistical
consistency; accurate focused-source decisions around broad light and haze.
The calibrated local fitter and uncertainty calculation run in Rust through
PyO3/maturin. Python is the array/result interface; its old algorithm is an
explicit validation reference, never a fallback.

[Historical plans and experiment results](archive/README.md) are retained as
documentation, not executable code or active instructions. No old calibration
is automatically valid for the current solver/model.

The default dense native API is `spotsolve.inference`; the separate old Python
algorithm is `spotsolve.inference.reference`. `spotsolve.localize_sparse` is the
single-pass native reference for isolated emitters. The
[numerical contract for Rust](INFERENCE_CONTRACT.md) documents the selected
model, array layout, frozen fixture and efficiency priorities. The removed
`spotsolve.prototype` tree is not another port target.
