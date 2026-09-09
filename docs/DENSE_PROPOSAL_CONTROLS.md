# Dense initial-proposal controls

Run `python scripts/check_dense_proposals.py` after rebuilding the Rust extension.
The report is written to `reference/dense-proposals/results.json`, including
fixture/source hashes, budgets, proposal centers, likelihoods, stationarity,
position errors and uncertainty statuses. It uses the immutable synthetic
calibration fixture; no real-camera or repeated-draw calibration is implied.

The experimental `proposal_method="aguet"` option replaces the initial focused
moment/peak candidate list with Rust Aguet maxima at `proposal_alpha=0.05`.
The default remains `"moments"`. Both policies retain the same K=0 fit, zero-flux
nested starts, pair splits, broad residual restarts and optimization budgets.
Caller-supplied centers are added to either policy. Aguet maxima outside
`focus_bounds` are discarded, and its existing image-border exclusion applies.
An empty proposal list still runs every count hypothesis. Native fit records
include `proposal_centres` for inspection.

These are comparisons of fixed-count hypotheses. There is currently no
calibrated dense acceptance rule to hold fixed, so the report does not claim
false-positive rates, detection sensitivity or source-presence probabilities.
In particular, proposal proximity is not a localization or acceptance metric.

## Observed results (2026-09-09)

Six paired controls use identical pixels and budgets, with no supplied truth
centers: isolated focus, focus on broad light, an equal pair, an unequal pair,
broad-only light and a pair on noisy haze. The noiseless pairs are two pixels
apart, with fluxes 1500/1500 or 1500/450 photons. Broad light has 6000 photons;
the noisy-haze control uses one Poisson draw with seed 42.

All K=0/1/2 likelihoods agree between policies within 3e-14. Both recover the
noiseless focused positions within 1e-8 pixels at the true count. On the fixed
noisy-haze draw, both have maximum matched position error about 0.2494 pixels.
All hypotheses have constrained stationarity residuals below 1e-6.

Aguet produces one maximum in each control. For the equal pair this is the
midpoint; for the unequal pair it is the bright source. The common pair-split
search recovers the second source. Aguet also produces a maximum for broad-only
light, while the joint model's focused likelihood gains are effectively zero.
This is direct evidence against treating the Aguet significance decision as
final focused-source acceptance in dense regions.

Before the numerical-zero correction, uncertainty was less stable than position
in these controls. At essentially
identical equal-pair optima, one policy returns conditional covariance and the
other withholds it with `parameter_boundary`: the absent broad component can
land at exactly zero or at a tiny positive flux. The isolated control also
withheld covariance, and both noisy-haze fits reported
`background_pixel_boundary`. The native uncertainty routine now canonicalizes numerical broad zero using
the flux and per-pixel mean guards in [the contract](INFERENCE_CONTRACT.md),
then recomputes curvature and stationarity on that face. Both proposal policies
now return conditional covariance for the equal pair and isolated source.
`broad_conditioned_absent` makes the conditioning explicit. Weak positive light
outside the numerical guards still withholds covariance at a boundary; this
change does not solve uncertainty with a statistically weak broad component or
establish repeated-draw coverage.

The comparison also exposed a search robustness issue at a focused-coordinate
boundary: a screened affine fit can land just outside a box bound by roundoff,
and strict seed validation rejected its refinement. Internally recycled seeds
now receive the same box projection as raw starts. A boundary-localization
regression covers this case; direct public geometry validation stays strict.

Keep the existing default proposal policy. These controls provide no evidence
that changing it improves localization. The next scientific work is repeated-draw uncertainty validation, treatment of
statistically weak broad light and calibrated focused-count decisions,
followed by full-frame group ownership. A larger proposal campaign is not
justified by these results alone.
