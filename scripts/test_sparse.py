# %%
from dataclasses import fields

import matplotlib.pyplot as plt
import spotsolve
import tifffile

# %%
img = tifffile.imread("../data/beads_80pct-glycerol_crop.tif")

# %%
res = spotsolve.localize_sparse(
    img[0],
    sigma=1.2,
    offset=100.0,
    alpha=0.02,
    fit_sigma=True,
    sigma_bounds=(0.8, 1.8),
    fit_radius_sigma=4.0,
)

# %%
dres = spotsolve.localize(
    img[0],
    sigma=1.27,
    offset=100.0,
    gain=2.0,
)

# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img[0], interpolation="nearest")
ax.plot(res.positions[:, 1], res.positions[:, 0], "wx", ms=5)
ax.plot(dres.positions[:, 1], dres.positions[:, 0], "r+")
plt.show()
