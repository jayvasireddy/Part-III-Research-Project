"""
Example data realisation from the 2D Gaussian likelihood at theta_obs:
    x ~ N(theta_obs + l, sigma^2 I)
1D marginals on the diagonal, 2D joint off-diagonal (getdist).
"""

import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, plots

rng       = np.random.default_rng(0)
N         = 20_000
theta_obs = theta_observed.detach().cpu().numpy().reshape(2)
mu        = theta_obs + loc1
cov       = scale1 * np.eye(2)

draws = rng.multivariate_normal(mu, cov, size=N)

samples = MCSamples(
    samples=draws,
    names=["x1", "x2"],
    labels=[r"x_1", r"x_2"],
    label="Example data realisation",
)

g = plots.get_subplot_plotter(width_inch=5)
g.settings.alpha_filled_add = 0.6
g.settings.axes_fontsize    = 11
g.settings.lab_fontsize     = 13
g.settings.legend_fontsize  = 12
g.settings.title_limit_fontsize = 12

g.triangle_plot(
    samples,
    filled=True,
    contour_colors=["#4C72B0"],
    legend_loc="upper right",
    title_limit=1,
)

g.fig.suptitle(
    r"Example data realisation:  $\mathbf{x}\sim p(\mathbf{x}\mid\boldsymbol{\theta}_{\mathrm{obs}})"
    r" = \mathcal{N}(\boldsymbol{\theta}_{\mathrm{obs}}+\boldsymbol{\ell},\ \sigma^{2}\mathbf{I})$",
    fontsize=13, y=1.02,
)

plt.savefig("data_realisation_pairplot.pdf", bbox_inches="tight")
plt.show()
