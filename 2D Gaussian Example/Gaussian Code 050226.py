import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import eye, ones
from sbi.utils import BoxUniform
from torch.distributions import LogNormal, Independent
from torch.distributions import MultivariateNormal
from joblib import Parallel, delayed
from sbi.analysis import pairplot
from sbi.inference import NPE
from sbi.analysis import plot_summary

_ = torch.manual_seed(42)

_ = np.random.seed(0)


# Define likelihood function, ~N(\theta+loc, diag(scale)) in num_dims dimensions
def simulator(theta, loc, scale):
    """linear gaussian inspired by sbibm
    https://github.com/sbi-benchmark/sbibm/blob/15f068a08a938383116ffd92b92de50c580810a3/sbibm/tasks/gaussian_linear/task.py#L74
    """
    num_dim = theta.shape[-1]
    cov_ = scale * eye(num_dim)  # always positively semi-definite

    # using validate_args=False disables sanity checks on `covariance_matrix`
    # for the sake of speed
    value = MultivariateNormal(
        loc=(theta + loc), covariance_matrix=cov_, validate_args=False
    ).sample()
    return value


# Wrapper Function to choose prior distn
def choose_prior_and_generate_theta(num_dim, num_simulations):
    prior_mean = ones(num_dim)      #mean vector [1,1]
    prior_cov = 2 * eye(num_dim)       # cov of 2* identity matrix
    prior = MultivariateNormal(
    loc=prior_mean, covariance_matrix=prior_cov, validate_args=False
    )
    
    theta = prior.sample((num_simulations,))
    return prior, theta

# plot to check simulated data covers observed data
def plot_checker(x, x_obs):
        if getattr(x_obs, "ndim", None) == 2 and x_obs.shape[0] == 1:
            x_obs = x_obs[0]    
        _ = pairplot(
        samples=x,
        points=x_obs[None, :],  # `points` needs a batch dimension.
        figsize=(4, 4),
    )
        


# Generate net, stopping after max epochs, choosing or not choosing to 
def train_net_generate_samples(x,theta,x_obs, prior, verbose, max_epoch,true_val, loc, scale):
    
    inference = NPE(prior= prior, density_estimator="nsf")
    posterior_net = inference.append_simulations(theta, x).train(training_batch_size=200,max_num_epochs=max_epoch, stop_after_epochs=30)
    posterior_direct = inference.build_posterior(density_estimator=posterior_net,sample_with="direct")
    posterior_mcmc = inference.build_posterior(density_estimator=posterior_net,sample_with="mcmc")
    samples = posterior_direct.sample((10_000,), x=x_obs)
    predictive_samples = simulator(samples,loc,scale)

    if verbose == True:
        _ = plot_summary(
        inference,
        tags=["training_loss", "validation_loss"],
        figsize=(10, 2),
        )

        print(posterior_mcmc)
        print("Observation: ", x_obs)

        _ = pairplot(
            samples=samples,
            points=true_val,
            limits=list(zip(true_val.flatten() - 1.0, true_val.flatten() + 1.0, strict=False)),
            upper="kde",
            diag="kde",
            figsize=(5, 5),
            labels=[rf"$\theta_{d}$" for d in range(samples.shape[1])],
            )

        _ = pairplot(
            samples=predictive_samples,
            points=x_obs,
            limits=list(zip(x_obs.flatten() - 1.0, x_obs.flatten() + 1.0, strict=False)),
            upper="kde",
            diag="kde",
            figsize=(5, 5),
            labels=[rf"$x_{d}$" for d in range(3)], # this range is probbaly not future proof
            )
   
    return samples, posterior_mcmc, posterior_direct, inference


num_dim = 2 # 2d gaussian
num_simulations = 10_000    # no of training pairs we generate
max_epoch = 150

# these are our noise parameters
loc1 = 0.05 
scale1 = 0.03

loc2 = -0.02
scale2 = 0.1


prior, theta=choose_prior_and_generate_theta(num_dim, num_simulations)

x1 = simulator(theta,loc1, scale1) # noisy draws
x2 = simulator(theta,loc2, scale2) # noisy draws

theta_observed = prior.sample((1,))
x_obs1 = simulator(theta_observed,loc1, scale1) # noise added
x_obs2=simulator(theta_observed,loc2, scale2) # noise added

# Prior checks: does observation fall in prior support
plot_checker(x1,x_obs1)
plot_checker(x2,x_obs2)

samples1, posterior1_mcmc, posterior1_direct, inference1 = train_net_generate_samples(x1,theta,x_obs1, prior, verbose = True, max_epoch=max_epoch, true_val=theta_observed, loc=loc1, scale=scale1)
samples2, posterior2_mcmc, posterior2_direct, inference2 = train_net_generate_samples(x2,theta,x_obs2, prior, verbose = True, max_epoch=max_epoch, true_val=theta_observed, loc=loc2, scale=scale2)

results = {
        "samples1": samples1,          # posterior samples (1000, 2)
        "samples2": samples2,          # posterior samples (1000, 2)
        "posterior1_mcmc": posterior1_mcmc,   # sbi posterior object (if you want to reuse it)
        "posterior1_direct": posterior1_direct,   # sbi posterior object (if you want to reuse it)
        "posterior2_mcmc": posterior2_mcmc,   # sbi posterior object (if you want to reuse it)
        "posterior2_direct": posterior2_direct,   # sbi posterior object (if you want to reuse it)
        "inference1": inference1,      # NPE object
        "inference2": inference2,      # NPE object
        "prior": prior,
        "theta": theta,              # simulated θ used for training (10000, 4)
        "x1": x1,
        "x2": x2,                                   # 
        "x_obs1": x_obs1,              # 
        "x_obs2": x_obs2,
        "loc1": loc1,
        "loc2" : loc2,
        "scale1": scale1, 
        "scale2": scale2,
        "theta_observed": theta_observed                      
    }


fname = f"CALIBRATION TESTER_gaussian_loc1_loc2.pt"
torch.save(results, fname)


def load_results_file(file_name):
    res = torch.load(file_name)
    return res

def extract_results(res):
    """
    Given a loaded results dict, return a tuple in a fixed order.
    This is explicit and avoids surprises.
    """
    keys = [
        "samples1", "samples2",
        "posterior1_mcmc", "posterior1_direct",
        "posterior2_mcmc", "posterior2_direct",
        "inference1", "inference2",
        "prior",
        "theta",
        "x1", "x2",
        "x_obs1", "x_obs2",
        "loc1","loc2", 
        "scale1","scale2", "theta_observed"
        
    ]

    

    missing = [k for k in keys if k not in res]
    if missing:
        raise KeyError(f"Missing keys in results dict: {missing}")

    return tuple(res[k] for k in keys)


fname1 = "CALIBRATION TESTER_gaussian_loc1_loc2.pt"

res = load_results_file(fname1)
# update to include t_span when we retrain models
# update to include true_params if needed but think first
(samples1, samples2,
 posterior1_mcmc, posterior1_direct,
 posterior2_mcmc, posterior2_direct,
 inference1, inference2,
 prior,
 theta,
 x1, x2,
 x_obs1, x_obs2,
 loc1, loc2, scale1, scale2, theta_observed
) = extract_results(res)


#plot posteriors overlayed

_ = pairplot(
            samples=[samples1, samples2],
            points=theta_observed,
            limits=list(zip(theta_observed.flatten() - 1.0, theta_observed.flatten() + 1.0, strict=False)),
            upper="kde",
            diag="kde",
            figsize=(5, 5),
            labels=[rf"$\theta_{d}$" for d in range(samples1.shape[1])],
            )


# Overlay analytical Gaussian posterior on top of SBI posterior samples (2D only)
def analytic_posterior_gaussian(prior_mean, prior_cov, x_obs, loc, scale):
    """
    Closed-form posterior for:
      prior:  theta ~ N(mu0, Sigma0)
      likelihood: x | theta ~ N(theta + loc, scale * I)
    """
    mu0 = prior_mean.reshape(-1, 1)
    Sigma0 = prior_cov
    num_dim = Sigma0.shape[-1]
    Sigma_lik = scale * torch.eye(num_dim)

    Sigma0_inv = torch.linalg.inv(Sigma0)
    Sigma_lik_inv = torch.linalg.inv(Sigma_lik)

    x_centered = (x_obs.reshape(-1, 1) - loc)
    Sigma_post = torch.linalg.inv(Sigma0_inv + Sigma_lik_inv)
    mu_post = Sigma_post @ (Sigma0_inv @ mu0 + Sigma_lik_inv @ x_centered)
    return mu_post.flatten(), Sigma_post


def plot_posterior_with_gaussian_overlay(samples, x_obs, loc, scale, prior, theta_true, title):
    if samples.shape[1] != 2:
        raise ValueError("This overlay plot is only implemented for 2D.")

    prior_mean = prior.loc
    prior_cov = prior.covariance_matrix
    mu_post, Sigma_post = analytic_posterior_gaussian(prior_mean, prior_cov, x_obs, loc, scale)

    # Build grid around posterior mean
    mu_np = mu_post.detach().cpu().numpy()
    Sigma_np = Sigma_post.detach().cpu().numpy()
    stds = np.sqrt(np.diag(Sigma_np))
    grid_min = mu_np - 4 * stds
    grid_max = mu_np + 4 * stds

    gx = np.linspace(grid_min[0], grid_max[0], 200)
    gy = np.linspace(grid_min[1], grid_max[1], 200)
    XX, YY = np.meshgrid(gx, gy)
    grid = np.stack([XX, YY], axis=-1)

    mvn = MultivariateNormal(mu_post, Sigma_post, validate_args=False)
    grid_t = torch.from_numpy(grid).float().reshape(-1, 2)
    ZZ = torch.exp(mvn.log_prob(grid_t)).reshape(200, 200).detach().cpu().numpy()

    samples_np = samples.detach().cpu().numpy()
    theta_true_np = theta_true.detach().cpu().numpy().reshape(-1)

    plt.figure(figsize=(5, 5))
    plt.scatter(samples_np[:, 0], samples_np[:, 1], s=6, alpha=0.25, label="SBI posterior samples")
    plt.contour(XX, YY, ZZ, levels=6, colors="black", linewidths=1.0, alpha=0.8, label="Analytic posterior")
    plt.scatter(theta_true_np[0], theta_true_np[1], c="red", s=40, label="True theta")
    plt.title(title)
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.legend()
    plt.tight_layout()


# Example overlays for the two observation settings
plot_posterior_with_gaussian_overlay(
    samples1, x_obs1, loc1, scale1, prior, theta_observed,
    title="Posterior (loc1/scale1) with analytic Gaussian overlay",
)

plot_posterior_with_gaussian_overlay(
    samples2, x_obs2, loc2, scale2, prior, theta_observed,
    title="Posterior (loc2/scale2) with analytic Gaussian overlay",
)
