# Experimental BIC count selection

`selection="bic"` asks whether background alone or fewer emitters explain a
fitting group's pixels. It is opt-in; the fixed 10-nat rule remains the default.

```python
locs = spotsolve.localize(frame, sigma=1.2, selection="bic", count_penalty=2)
movie = spotsolve.localize_stack(
    stack, sigma=1.2, selection="bic", count_penalty=2, n_threads=5,
)
```

The example penalty is not a universal setting. `count_penalty` must be finite
and non-negative; in fixed mode it adds to the existing 10-nat cost.

## Score and search

Minimize

```text
I_K / phi + K * (2*log(n_pixels) + count_penalty)
```

`I_K` is Poisson I-divergence, `phi` the measured group dispersion, and
`n_pixels` the fixed fitting rectangle's raw pixel count. Each emitter adds
four parameters (flux, y, x, width); the common background-parameter penalty
cancels. This is half the usual BIC plus a count cost, motivated by a prior
proportional to `exp(-count_penalty*K)`. No amplitude prior was added; positive
numerical amplitude bounds are unchanged.

This is a **BIC-inspired score**, not a normalized marginal likelihood.
Measured-dispersion Poisson fitting is approximate, and overlapping-emitter
models are singular. The score supplies no posterior, p-value or guaranteed
false-positive rate.

1. Freeze group pixels, background shape, neighboring light and dispersion.
   Fit the background level at K=0.
2. Add residual-LoG proposals and jointly refit each count, keeping the best
   score. A failed complexity test does not stop growth: higher counts may win.
3. Stop at `k_max`, no eligible peak, or failure to improve the finite
   likelihood. That stopping rule does not prove larger models are inferior.
4. From the largest fitted count, try every single-source deletion and refit
   survivors. Follow the best deletion through lower counts, retaining the
   best score across both paths and K=0. Ties favor fewer emitters.
5. After joint refinement, revisit original groups and pixels, updating
   neighboring light. Run one sequential downward-comparison pass, holding
   neighbors fixed within each comparison.
6. Refit retained sources, refresh uncertainties and apply the usual width band.

This bounded search depends on proposals, local minima, group boundaries and
processing order. It neither enumerates every configuration nor repeats
selection/refinement to convergence. Native `polish=False` also skips the
post-refinement comparison; public Python calls always refine.

`info` records `selection`, `count_penalty`, `search_fits` (initial count
comparisons), `selection_fits` (post-refinement comparisons), and `polish_fits`
(both refinement stages).

## Accuracy benchmark

```sh
maturin develop --release
python scripts/benchmark_count_selection.py --out /tmp/count-selection.json
```

The 2026-09-16 benchmark tested both modes at penalties `[0, 2, 4, 8, 16, 32]`,
threshold 2.75, with the original width band. Each condition used 12 calibration
frames (seeds 12000–12011) and 24 held-out frames (24000–24023): 64x64 pixels,
sigma 1.2, base background 20.

| Condition | Flux / density or other variation |
|---|---|
| Empty | Poisson background only |
| Faint sparse | Flux 150–300; density 0.003 per interior pixel |
| Faint dense | Flux 150–300; density 0.034 |
| Bright dense | Flux 900–1900; density 0.034 |
| Bright, variable width | Bright dense plus log-width SD 0.2 |
| Blur/haze | Flux 150–600; density 0.034; horizontal blur SD 0.8 and smooth added background 0–40 |

Dense frames contain 107 true emitters. One-to-one matching uses a one-pixel
radius. “Unmatched” includes invented, displaced and merged localizations;
recall includes all truth, even sources excluded by width reporting. No
tracking was run.

For each condition/mode, calibration selects the highest-recall penalty
meeting a mean unmatched-count budget of 1, 3 or 5 per frame. Evaluation
never selects the penalty. [Saved results](count_selection_benchmark.json)
include curves, selected settings and evaluation standard errors; the runner
also writes per-frame results for paired analysis.

### Five unmatched/frame calibration budget

| Condition | Mode | Penalty | Evaluation unmatched/frame | Recall | ms/frame |
|---|---|---:|---:|---:|---:|
| Bright dense | Fixed | 16 | 2.67 | 80.96% | 16.0 |
| Bright dense | BIC | 2 | 3.83 | 86.88% | 85.4 |
| Bright, variable width | Fixed | 16 | 4.13 | 76.60% | 20.7 |
| Bright, variable width | BIC | 4 | 3.54 | 81.11% | 226.0 |

These share a calibration budget, not an achieved error rate. BIC recalled
more matched-width sources with more unmatched detections; both quantities
improved for variable widths in this small sample. Serial runtime was about
5.4× and 10.9× the tuned fixed modes, respectively.

At a budget of three, BIC selected penalty 8 in both bright conditions:
2.67 unmatched/frame with 84.31% recall for matched widths, and 2.50 with
79.40% recall for variable widths. No tested fixed penalty met that
calibration budget.

### Limits

- Neither mode met the five-per-frame calibration budget on faint dense or
  blur/haze data. Raising penalties mainly lost recall; untested settings
  remain possible, and suppressing every detection would be uninformative.
- Faint sparse BIC at penalty 0 met a budget of one during calibration but
  produced 1.50 unmatched/frame during evaluation. Twelve-frame empirical
  means are not guarantees; zero empty-frame detections is not a zero bound.
- The comparison changes search/refinement as well as scoring, so it does
  not isolate the logarithmic penalty's benefit.
- These simulations establish neither real-movie false-positive rates nor
  improved downstream linking.

The score itself is cheap. Additional count fits and refitted deletions,
especially near overlapping/broad objects, dominate runtime. Keep BIC
experimental while evaluating its cost and faint/blurred failure cases.

## Speed

Both stack detectors use shared native frame workers with separate workspaces
and ordered results. Default worker count is the machine's cores.

For `hyp7gem_wt_01_crop_128x128.tif`, sigma 1.45, offset 100, BIC penalty 2:

| Frames | One worker | Five workers | Throughput gain |
|---|---:|---:|---:|
| First five | 2.810 s | 0.755 s | 3.72× |
| All 49 | 27.865 s | 6.691 s | 4.16× |

First-five times are medians of five warm runs; full-movie times are one run
per setting. Both include model/residual output, exclude loading/warm-up/
hashing, and were measured on the development Apple Silicon host. All output
fingerprints matched, including fits, uncertainties, rejects and images.
First-five counts were 242, 235, 258, 250, 230. This improves batch throughput;
individual-frame latency and search rules are unchanged.

Export the reviewed frames (`--frames` has an exclusive end):

```sh
python scripts/localize_movie.py \
  --image data/hyp7gem_wt_01_crop_128x128.tif --frames 0 5 \
  --selection bic --count-penalty 2 --threads 5 --out output/hyp7gem_bic_locs
```

The script records selection, penalty and threading with the output. Reproduce
timing and exact-output comparisons:

```sh
python scripts/benchmark_detector_speed.py \
  data/hyp7gem_wt_01_crop_128x128.tif --sigma 1.45 --offset 100 \
  --threads 1 --out /tmp/bic-serial.json
python scripts/benchmark_detector_speed.py \
  data/hyp7gem_wt_01_crop_128x128.tif --sigma 1.45 --offset 100 \
  --threads 5 --out /tmp/bic-parallel.json --compare /tmp/bic-serial.json
```

## Verification

The latest combined run passed 69 Python and 45 Rust tests. Count-selection
checks cover source deletion with neighbor refitting, K=0 despite strong
candidates, isolated sources, refreshed uncertainty, camera-scale invariance,
stack/thread/ROI consistency and invalid arguments.

Cleanup and scheduler sharing preserved all fields exactly across 134 cases:
49 real frames in each selection mode, plus three seeds in six simulated
conditions in both modes. [Historical commentary](archive/DETECTOR_DESIGN_NOTES.md)
was moved out of source; halo collection is shared and deletion selection
returns its selected fit explicitly.
