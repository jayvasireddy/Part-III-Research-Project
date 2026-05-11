"""
Data realisations from the 2D Gaussian likelihood at theta_obs:
    null:        x ~ N(theta_obs + l,           sigma^2 I)
    systematic:  x ~ N(theta_obs + l + delta,   sigma^2 I)
1D marginals on the diagonal, 2D joint off-diagonal (getdist).
delta = 0.36 is the empirical sensitivity threshold from the KL sweep.
"""

import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, plots

rng       = np.random.default_rng(0)
N         = 20_000
delta     = 0.36                                          # sensitivity-threshold shift

theta_obs = theta_observed.detach().cpu().numpy().reshape(2)
mu_null   = theta_obs + loc1                              # null mean
mu_syst   = theta_obs + loc1 + delta                      # shift applied to both dims
cov       = scale1 * np.eye(2)                            # likelihood covariance

draws_null = rng.multivariate_normal(mu_null, cov, size=N)
draws_syst = rng.multivariate_normal(mu_syst, cov, size=N)

samples_null = MCSamples(
    samples=draws_null,
    names=["x1", "x2"],
    labels=[r"x_1", r"x_2"],
    label="Null",
)
samples_syst = MCSamples(
    samples=draws_syst,
    names=["x1", "x2"],
    labels=[r"x_1", r"x_2"],
    label=fr"Systematic ($\delta={delta}$)",
)

g = plots.get_subplot_plotter(width_inch=5)
g.settings.alpha_filled_add     = 0.6
g.settings.axes_fontsize        = 11
g.settings.lab_fontsize         = 13
g.settings.legend_fontsize      = 12
g.settings.title_limit_fontsize = 12

g.triangle_plot(
    [samples_null, samples_syst],
    filled=True,
    contour_colors=["#4C72B0", "#C44E52"],
    legend_loc="upper right",
)

g.fig.suptitle(
    r"Data realisations:  null vs.\ systematic shift $\delta=0.36$",
    fontsize=13, y=1.02,
)

plt.savefig("data_realisation_pairplot.pdf", bbox_inches="tight")
plt.show()
