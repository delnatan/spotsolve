# Statistical Foundations of Sub-Pixel Spot Detection

Historical conceptual sketch, not an implementation specification. In
particular, its zero-amplitude t-test is not the amplitude-versus-residual-noise
test in `spotfitlm`. See the current [Aguet baseline](../AGUET_BASELINE.md)
for the implemented method, reference code and compatibility limits.

**Based on the method by Aguet et al.**

This document outlines the theoretical framework for detecting diffraction-limited spots in fluorescence microscopy images. It details why simple intensity thresholding fails and derives the rigorous statistical test (Student's t-test) used to identify "significant" signals.

## 1. The Image Formation Model

To detect objects statistically, we must first define what a "signal" looks like physically.

In fluorescence microscopy, a point source of light (e.g., a single vesicle or molecule) does not appear as a single pixel. It is diffracted by the microscope optics into a pattern called the **Point Spread Function (PSF)**. This pattern is accurately approximated by a 2D Gaussian function.

Therefore, the intensity $f[x]$ at any given pixel $x$ can be modeled as the sum of the signal, a local background, and random noise:

$$
f[x] = A \cdot g[x] + c + n[x]
$$

Where:

* $A$ **(Amplitude):** The intensity of the spot. This is the unknown signal we wish to quantify.

* $g[x]$ **(Geometry):** The Gaussian shape of the PSF. Since the microscope's numerical aperture and wavelength are known, the width ($\sigma$) of this Gaussian is **fixed and known**.

* $c$ **(Background):** A constant local background intensity.

* $n[x]$ **(Noise):** Additive white Gaussian noise with variance $\sigma_r^2$.

## 2. Motivation: Why Statistical Testing?

A naive approach to spot detection is **Intensity Thresholding**:

> *"Keep all pixels brighter than value X."*

This approach fails in biological imaging for two reasons:

1. **Variable Background:** A dim spot on a bright background may have a higher raw pixel value than a bright spot on a dark background.

2. **Variable Noise:** In noisy images, random fluctuations can create clusters of bright pixels that mimic spots.

### The Statistical Approach

Instead of asking "How bright is this pixel?", we ask:

> **"Given the local noise levels, what is the probability that this signal arose purely by chance?"**

This requires a **Hypothesis Test** at every candidate pixel:

* **Null Hypothesis (**$H_0$**):** $A = 0$. The pixel contains only background and noise.

* **Alternative Hypothesis (**$H_1$**):** $A > 0$. There is a significant structural signal present.

## 3. The Significance Test (Student's t-test)

To reject the Null Hypothesis, we need to compare our estimated signal against the uncertainty of that estimate. Since the true noise variance of the camera ($\sigma_r^2$) is usually unknown and must be estimated from the image itself, we use a **Student's t-test**.

The t-statistic is defined as:

$$
t = \frac{\hat{A}}{\sigma_{\hat{A}}}
$$

Where:

* $\hat{A}$ is the **Estimated Amplitude** (the "Mean").

* $\sigma_{\hat{A}}$ is the **Standard Error** of that estimate (the "Standard Deviation").

If $t$ is large enough (e.g., $t > \text{threshold}$ corresponding to a p-value of 0.05), we conclude the spot is real. To perform this test, we must derive estimators for these values.

## 4. Deriving the Estimators

Solving for the amplitude $A$ and background $c$ is typically a non-linear problem because the center position $(x_0, y_0)$ of the spot is unknown.

**The "Trick":** To enable rapid statistical testing, we temporarily assume the spot is **perfectly centered** on the candidate pixel. This fixes the geometry $g[x]$, turning the problem into a **Linear Least Squares** regression.

We minimize the sum of squared residuals ($v$) with respect to $A$ and $c$:

$$
v = \sum_{x \in W} (f[x] - A \cdot g[x] - c)^2
$$

By setting the partial derivatives $\frac{\partial v}{\partial A} = 0$ and $\frac{\partial v}{\partial c} = 0$, we obtain the following estimators.

### A. The Background Estimator ($\hat{c}$)

This derivation reveals a critical insight: **We cannot estimate the background independently of the signal.**

Standard approaches often estimate background by taking the median of the border pixels. However, the rigorous Least Squares solution couples the two:

$$
\hat{c} = \bar{f} - \hat{A} \cdot \bar{g}
$$

Where $\bar{f}$ is the average image intensity and $\bar{g}$ is the average Gaussian value in the window.

**Interpretation:** The estimator takes the average intensity of the window ($\bar{f}$) and **subtracts the estimated contribution of the spot** ($\hat{A} \cdot \bar{g}$). This prevents the bright spot itself from skewing the background calculation, a common error in simpler algorithms.

### B. The Amplitude Estimator ($\hat{A}$)

Solving the coupled system yields the estimator for the signal strength:

$$
\hat{A} = \frac{\sum (f[x] \cdot g[x]) - n \cdot \bar{f} \cdot \bar{g}}{\sum g[x]^2 - n \cdot \bar{g}^2}
$$

This formula acts as a "matched filter," effectively projecting the raw image data onto the expected Gaussian shape while accounting for the floating background.

### C. The Standard Error ($\sigma_{\hat{A}}$)

Finally, to perform the t-test, we need the uncertainty of our amplitude estimate. In linear regression, the variance of an estimator is the noise variance scaled by a geometric factor (related to the curvature of the Hessian):

$$
\text{Var}(\hat{A}) = \hat{\sigma}_r^2 \cdot \left( \frac{1}{\sum (g[x] - \bar{g})^2} \right)
$$

1. **The Noise (**$\hat{\sigma}_r^2$**):** We estimate the noise variance from the residuals (the difference between the fitted model and the raw image).

2. **The Design Factor:** The term in parentheses is constant for a given Gaussian width. It represents how "sharp" the Gaussian is; sharper PSFs yield lower uncertainty.

## 5. Summary

The power of the Aguet algorithm lies in this rigorous derivation. By calculating $\hat{A}$, $\hat{c}$, and $\sigma_{\hat{A}}$ via Least Squares, the method allows us to:

1. **Correct for local background** without contamination from the spot itself ($\hat{c} = \bar{f} - \hat{A}\bar{g}$).

2. **Normalize for noise**, allowing the algorithm to automatically adapt to noisy or clear regions of the image via the t-test.
