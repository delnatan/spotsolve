import numpy as np
import scipy.ndimage as ndi
import scipy.stats as stats


def estimate_significant_pixels(
    img: np.ndarray, sigma: float, truncate: int = 4, alpha: float = 0.05
):
    """Pixel-wise amplitude/background fit and significance test.

    Python re-implementation of the prefiltering stage of Francois Aguet's
    pointSourceDetection.m. At every pixel, fits a Gaussian of fixed width
    `sigma` plus a constant background by least squares over a
    (2*w+1)x(2*w+1) window, then tests whether the fitted amplitude is
    significantly above the local background noise (Student's t-test on
    a combined amplitude/noise variance, Welch-Satterthwaite dof).
    """
    # 1-D and 2-D (unnormalized) Gaussian kernel
    w = int(np.ceil(truncate * sigma))
    x = np.arange(-w, w + 1)
    g1 = np.exp(-(x**2) / (2 * sigma**2))
    g = np.outer(g1, g1)
    u = np.ones_like(g)
    n = g.size

    gsum = g.sum()
    g2sum = (g**2).sum()

    f = img.astype(float)
    fg = ndi.convolve(f, g, mode="reflect")
    fu = ndi.convolve(f, u, mode="reflect")
    fu2 = ndi.convolve(f * f, u, mode="reflect")

    # least-squares solution for amplitude (a) and background (c)
    a = (fg - gsum * fu / n) / (g2sum - gsum**2 / n)
    c = (fu - a * gsum) / n

    # residual sum of squares of f - (a*g + c) over the window
    fc = fu2 - 2 * c * fu + n * c**2
    rss = a**2 * g2sum - 2 * a * (fg - c * gsum) + fc
    rss[rss < 0] = 0.0  # roundoff can push RSS slightly negative

    # (1,1) entry of inv(J'J) for design matrix J = [g(:), ones(n,1)]
    j_aa = n / (n * g2sum - gsum**2)

    var_e = rss / (n - 3)  # residual (error) variance of the fit
    se_a = np.sqrt(var_e * j_aa)  # standard error of amplitude estimate a
    resid_std = np.sqrt(rss / (n - 1))  # standard deviation of the residuals

    # stage 2: test amplitude against background noise level
    k = stats.norm.ppf(1 - alpha / 2.0, 0, 1)
    se_c = resid_std / np.sqrt(2 * (n - 1)) * k

    df = (n - 1) * (se_a**2 + se_c**2) ** 2 / (se_a**4 + se_c**4)
    se_comb = np.sqrt((se_a**2 + se_c**2) / n)
    t = (a - resid_std * k) / se_comb
    pval = stats.t.sf(t, df)

    return a, c, pval
