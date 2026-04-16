# Pipeline Methodology: `output_beam_miscalibration_same_sky.ipynb`

## Overview

This notebook implements a **Simulation-Based Inference (SBI) null test** for detecting beam miscalibration systematics in Cosmic Microwave Background (CMB) data. The core goal is to train a Neural Posterior Estimator (NPE) on simulated CMB power spectra, inject a subtle beam systematic into one data split, and then use the KL divergence between the resulting posteriors as a diagnostic test statistic to detect the systematic.

The "same sky" framing refers to a key design choice: both the null and perturbed observation splits are generated from **the same underlying CMB sky realisation** (i.e., the same `seed_cmb`), differing only in their beam FWHM. This ensures that any divergence between posteriors is attributable purely to the beam systematic, not to cosmic variance fluctuations.

---

## Scientific Motivation

In real CMB experiments, the telescope beam smooths the sky signal. If the beam is slightly miscalibrated — that is, if the instrument has a slightly different angular resolution than assumed — the inferred power spectrum will be biased. The angular power spectrum $C_\ell$ at multipole $\ell$ is suppressed by a factor $B_\ell^2$, the beam window function:

$$\tilde{C}_\ell = B_\ell^2 \cdot C_\ell + N_\ell$$

where $N_\ell$ is the noise power spectrum and $B_\ell = \exp\left(-\frac{\ell(\ell+1)\sigma_B^2}{2}\right)$, with $\sigma_B$ related to the beam FWHM by $\sigma_B = \theta_\text{FWHM} / \sqrt{8 \ln 2}$.

A miscalibrated beam means an incorrect $B_\ell$ is assumed during analysis, leading to biased cosmological parameter inference. This notebook tests whether the KL divergence between two NPE posteriors can detect this bias even for very small miscalibrations.

---

## Libraries and Global Settings

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from sbi.utils import BoxUniform
from torch.distributions import LogNormal, Independent, MultivariateNormal
from joblib import Parallel, delayed
from sbi.analysis import pairplot
from sbi.inference import NPE
import camb
import healpy as hp

_ = torch.manual_seed(42)
_ = np.random.seed(0)
```

Global random seeds are fixed at the start (`torch.manual_seed(42)`, `np.random.seed(0)`) to ensure reproducibility for prior sampling and training. Simulation-level randomness is controlled separately via explicit `seed_cmb` and `seed_noise` arguments in `blanket_simulator`.

---

## Parameters Inferred

The NPE is trained to infer five standard $\Lambda$CDM cosmological parameters:

| Symbol | Name | Fiducial Value |
|---|---|---|
| $H_0$ | Hubble constant (km/s/Mpc) | 67.5 |
| $\Omega_b h^2$ | Baryon density | 0.022 |
| $\Omega_c h^2$ | Cold dark matter density | 0.122 |
| $A_s$ | Scalar amplitude | $2 \times 10^{-9}$ |
| $n_s$ | Spectral index | 0.965 |

---

## Pipeline Stages

### Stage 1: Instrument and Compressor Configuration

```python
fiducial_params = {
    'H0': 67.5, 'ombh2': 0.022, 'omch2': 0.122,
    'As': 2e-9, 'ns': 0.965, 'lmax': 3000
}
nl_base = np.ones(3001) * (20 / 60 / 180 * np.pi)**2
beam_base       = 5.0           # arcmin — null beam
beam_systematic = beam_base * 1.001  # 0.1% broader beam

fiducial, cov, derivs, my_compressor = getCompression(
    param_dict=fiducial_params,
    derivatives=derivatives_frac,
    beam_fwhm=beam_base,
    noise_cl=nl_base
)
```

**Design choices:**
- The noise spectrum `nl_base` is a flat (white) noise spectrum corresponding to 20 arcmin noise level in $\mu\text{K}$-rad. It is held **constant across both splits** so that the only varying element is the beam.
- The compressor is **built using the null beam** (`beam_base = 5.0` arcmin). This is an intentional mismatch: when systematic data (with a slightly broader beam) is fed through this null-calibrated compressor, it will produce compressed statistics that differ subtly from what the null model expects, which is precisely what the KL divergence test is designed to detect.
- The beam miscalibration is set to `1.001×beam_base` (0.1% broader) — a deliberately small perturbation to test the pipeline's sensitivity threshold.

---

### Stage 2: `getSpectrum` — CAMB Spectrum Computation

This helper function calls the CAMB Boltzmann code to compute the theoretical TT CMB power spectrum $C_\ell$ for a given set of cosmological parameters.

```python
def getSpectrum(param_dict, lmax=3000):
    pars = camb.set_params(
        H0=param_dict['H0'], ombh2=param_dict['ombh2'], ...
    )
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)['total']
    return powers[:, 0]  # TT spectrum only
```

- Returns raw $C_\ell$ (not $\ell(\ell+1)C_\ell / 2\pi$) from $\ell=0$ to $\ell_\text{max}=3000$.
- Fixed nuisance parameters: `mnu=0.06`, `omk=0`, `tau=0.06`, `halofit_version='mead'`.

---

### Stage 3: `getCompression` — SCORE/MOPED Compression

Raw CMB power spectra contain ~3000 data points. Feeding this directly into the NPE would be impractical. The compressor reduces this to 5 summary statistics (one per inferred parameter) using the SCORE/MOPED algorithm.

```python
def getCompression(param_dict, derivatives, beam_fwhm, noise_cl):
    fiducial = getSpectrum(param_dict)     # C_ell at fiducial params
    beam = hp.gauss_beam(np.radians(beam_fwhm / 60), lmax=3000)
    fiducial *= beam[2:]**2               # Apply beam window function
    fiducial += noise_cl[2:]              # Add noise

    cov_mat = 2 / (2 * np.arange(2, 3001) + 1) * fiducial**2  # Gaussian covariance

    # Numerical derivatives via finite differences
    derivs = []
    for param_name in derivatives.keys():
        step = param_dict[param_name] * derivatives[param_name]
        up   = getSpectrum({...param + step...})
        down = getSpectrum({...param - step...})
        derivs.append((up - down) / (2 * step))
    derivs = np.array(derivs)

    def compressor(data):
        data = data[2:]  # Remove monopole (ell=0) and dipole (ell=1)
        return np.dot(derivs, (data - fiducial) / cov_mat)

    return fiducial, cov_mat, derivs, compressor
```

**Mathematical basis:** The MOPED/SCORE compression projects the data residual $(d - \mu)$ onto the Fisher score $\partial \ln L / \partial \theta_i \propto \mathbf{b}_i \cdot \Sigma^{-1} (d - \mu)$. For a Gaussian likelihood with diagonal covariance $\Sigma_{\ell\ell} = 2C_\ell^2 / (2\ell+1)$, this reduces to:

$$t_i = \sum_\ell \frac{\partial C_\ell / \partial \theta_i}{C_\ell^2 / (2\ell+1)} \cdot (d_\ell - C_\ell^\text{fid})$$

**Design choices:**
- The covariance $\Sigma$ uses the **Gaussian (Knox) formula**: $\sigma_\ell^2 = 2\tilde{C}_\ell^2 / (2\ell+1)$ where $\tilde{C}_\ell$ is the observed spectrum including beam and noise.
- Derivatives are computed numerically by finite differences with a fractional step size (e.g., 0.1% for $H_0$).
- The monopole and dipole ($\ell = 0, 1$) are excluded (`data[2:]`) as they are not cosmologically informative.
- The compressor is **frozen at the null instrument configuration** — this is the key to the test working. Passing a systematically biased observation through a null-calibrated compressor will give shifted compressed statistics, which the posterior will then try to interpret under the wrong model.

---

### Stage 4: `generateMock` — Simulating a CMB Sky Map

This function converts a theoretical $C_\ell$ into a noisy synthetic CMB observation:

```python
def generateMock(cl, nls=None, lmax=3000, beam_fwhm=5.0):
    cl = cl[:lmax+1]
    cmb_alm = hp.synalm(cl, lmax=lmax)            # Sample alm's from C_ell
    beam = hp.gauss_beam(np.radians(beam_fwhm/60), lmax=lmax)
    cmb_alm = hp.almxfl(cmb_alm, beam)            # Apply beam smoothing
    noise_alm = hp.synalm(nls[:lmax+1], lmax=lmax) # Sample noise alm's
    total_alm = cmb_alm + noise_alm               # Total observed signal
    return hp.alm2cl(total_alm)                   # Return observed C_ell
```

- `hp.synalm`: Draws $a_{\ell m}$ coefficients from a Gaussian distribution with variance $C_\ell$.
- `hp.almxfl`: Applies the beam window function $B_\ell$ multiplicatively in harmonic space (equivalent to convolution on the sphere).
- `hp.alm2cl`: Converts the total $a_{\ell m}$'s back to a pseudo-$C_\ell$ power spectrum.

This represents a single realisation of the observed sky: a true CMB signal blurred by a Gaussian beam and contaminated by noise.

---

### Stage 5: `blanket_simulator` — Full Simulation Pipeline

This is the main simulation workhorse, tying together CAMB, healpy, and compression:

```python
def blanket_simulator(theta, compressor, nl_split, lmax=3000,
                      beam_fwhm=5.0, seed_cmb=None, seed_noise=None):
    theta = np.asarray(theta, dtype=float)

    # 1. Compute theoretical spectrum from parameters
    cl_theory = get_camb_spectrum(theta, lmax=lmax)

    # 2. Generate CMB sky realisation with separate seeds for CMB and noise
    np.random.seed(seed_cmb)
    cmb_alm = hp.synalm(cl_theory[:lmax+1], lmax=lmax)
    beam = hp.gauss_beam(np.radians(beam_fwhm / 60), lmax=lmax)
    cmb_alm = hp.almxfl(cmb_alm, beam)

    np.random.seed(seed_noise)
    noise_alm = hp.synalm(nl_split[:lmax+1], lmax=lmax)

    total_cl = hp.alm2cl(cmb_alm + noise_alm)

    # 3. Compress the observed spectrum to 5 summary statistics
    return compressor(total_cl)
```

**Critical design: decoupled seeds**

The separation of `seed_cmb` and `seed_noise` is a fundamental statistical design decision. It allows:

- **Same sky, different noise**: `seed_cmb=i, seed_noise=i+1_000_000` vs `seed_cmb=i, seed_noise=i+2_000_000` — two observations of the same sky with independent noise realisations (the null split scenario).
- **Same sky, same noise, different beam**: `seed_cmb=TRUE_SEED_CMB, seed_noise=TRUE_SEED_NOISE` for both splits but with `beam_fwhm=beam_base` vs `beam_fwhm=beam_systematic` — isolating the beam systematic from all other sources of variation.

Without this decoupling, a shared seed would produce identical noise realisations across splits, artificially reducing or inflating the measured divergence.

---

### Stage 6: Prior Definition and Training Data Generation

```python
def define_normal_prior():
    # Normal prior centred on fiducial values with physically motivated widths
    return sbi.utils.posterior_nn(...)  # MultivariateNormal around fiducial

def generate_theta(prior, num_simulations):
    return prior.sample((num_simulations,))

theta_true = torch.tensor([67.5, 0.022, 0.122, 2e-9, 0.965])
num_simulations = 100_000
prior = define_normal_prior()
theta = generate_theta(prior, num_simulations)
unique_seeds = np.random.randint(0, 1_000_000, size=num_simulations)
```

- **100,000 simulations** are drawn from the prior. This large training set is required because the 5D compressed summary statistics have a non-trivial relationship with the parameters (mediated through the full CMB physics pipeline) and the NPE needs sufficient coverage to learn an accurate posterior.
- Each simulation receives a **unique random seed** to ensure fully independent CMB sky and noise realisations across the training set.
- A **Normal prior** centred on the fiducial ΛCDM values is used (rather than a BoxUniform). This reflects physical knowledge of approximate parameter values and improves training efficiency by concentrating samples where the likelihood has significant support.

---

### Stage 7: Locked-in True Observations

```python
TRUE_SEED_CMB   = 12345
TRUE_SEED_NOISE = 99999
```

Hardcoded seeds are used to generate the single "true" observed data vectors `x_obs_1` and `x_obs_2`. This is essential for the test:

- `x_obs_1`: Null split — observed with `beam_base`, using `(TRUE_SEED_CMB, TRUE_SEED_NOISE)`.
- `x_obs_2`: Systematic split — observed with `beam_systematic`, using the **same** `(TRUE_SEED_CMB, TRUE_SEED_NOISE)`.

The ONLY difference between them is the beam FWHM. Locking both seeds ensures that any shift in the posterior is due entirely to the beam miscalibration and not to a different CMB sky or noise draw.

---

### Stage 8: Parallel Training Data Simulation

```python
x_train_1 = parallel_simulate(theta, unique_seeds, my_compressor,
                               nl_base, beam_fwhm=beam_base)
x_train_2 = parallel_simulate(theta, unique_seeds, my_compressor,
                               nl_base, beam_fwhm=beam_systematic)
```

`parallel_simulate` uses `joblib.Parallel` with `delayed` to distribute the 100,000 simulations across CPU cores. Each worker calls `blanket_simulator` for one `(theta_i, seed_i)` pair.

**Design choice — shared `theta` and `unique_seeds`:** Both training sets use the **same parameter draws** `theta` and the **same CMB seeds** from `unique_seeds`. The only difference is `beam_fwhm`. This means the two NPEs (one per split) are trained on data that differs only in how the beam was applied, allowing a fair comparison of their posteriors.

---

### Stage 9: NPE Training

```python
inference1 = NPE(prior=prior)
inference1.append_simulations(theta, x_train_1)
posterior_net_1 = inference1.train()
posterior1_direct = inference1.build_posterior(
    density_estimator=posterior_net_1, sample_with="direct"
)
posterior1_mcmc = inference1.build_posterior(
    density_estimator=posterior_net_1, sample_with="mcmc"
)
```

The same procedure is applied to train `inference2` / `posterior2` on `x_train_2`.

- **NPE (Neural Posterior Estimation)** is used rather than NLE or NRE because NPE produces a fully normalised posterior density that can be both sampled and pointwise-evaluated (`log_prob`), which is required for computing the KL divergence.
- **Two separate NPEs** are trained: one on the null training data, one on the systematic training data. This allows the null posterior to be compared against a posterior that has "learned" to interpret the systematically biased data.
- Both `direct` and `mcmc` posteriors are built. The `direct` sampler (using normalising flows) is used for KL divergence computation due to its speed; the `mcmc` sampler provides a cross-check.
- Training runs until convergence (early stopping based on validation loss). In this run, convergence took ~385 epochs.

---

### Stage 10: Posterior Visualisation

```python
param_labels = [r"$H_0$", r"$\Omega_b h^2$", r"$\Omega_c h^2$", r"$A_s$", r"$n_s$"]
limits = [[60.0, 75.0], [0.020, 0.025], [0.10, 0.14], [1.5e-9, 2.5e-9], [0.90, 1.02]]

_ = pairplot(
    samples=[samples1, samples2],
    points=theta_true,
    limits=limits,
    upper="kde",
    diag="kde",
    figsize=(8, 8),
    labels=param_labels,
)
```

A pairplot comparing posterior samples from both NPEs is produced, with the true parameter values marked. Visual inspection shows whether the systematic split posterior is shifted or broadened relative to the null posterior.

---

### Stage 11: `calc_dkl` — KL Divergence Estimator

This is the core diagnostic function. It computes the Monte Carlo estimate of the KL divergence between two NPE posteriors conditioned on different data:

$$D_\text{KL}(p_1 \| p_2) = \mathbb{E}_{\theta \sim p_1(\theta|x_1)} \left[ \log p_1(\theta|x_1) - \log p_2(\theta|x_2) \right]$$

```python
def calc_dkl(posterior1, posterior2, x1, x2, n_theta):
    x1 = torch.as_tensor(x1, dtype=torch.float32).reshape(1, -1)
    x2 = torch.as_tensor(x2, dtype=torch.float32).reshape(1, -1)

    with torch.no_grad():
        theta_samples = posterior1.sample((n_theta,), x=x1, show_progress_bars=False)

        log_post1 = posterior1.log_prob(theta_samples, x=x1, norm_posterior=True)
        log_post2 = posterior2.log_prob(theta_samples, x=x2, norm_posterior=True)

        z = log_post1 - log_post2
        mean    = z.mean().item()
        std_dev = z.std(unbiased=True).item()
        error   = std_dev / np.sqrt(n_theta)

    return mean, std_dev, error
```

**Implementation details:**
- Samples $\{\theta_j\}$ are drawn from `posterior1` conditioned on `x1` (the null observation).
- `log_prob` evaluates the normalised log posterior density at each sample under both posteriors.
- The KL divergence is estimated as the empirical mean of $\log p_1 - \log p_2$ over the `n_theta` samples.
- `norm_posterior=True` applies the SBI library's leakage correction, renormalising the posterior mass to 1 within the prior support. This is important because NPE can occasionally place probability mass outside the prior, which would otherwise bias `log_prob` values.
- The standard deviation and standard error of the mean are returned alongside the point estimate to quantify Monte Carlo noise.
- `torch.no_grad()` is used to disable gradient tracking during inference, improving speed.

---

### Stage 12: Null Calibration — `calibrate_null_dkl_neat`

To interpret the observed $D_\text{KL}$ as a test statistic, a null distribution must be constructed empirically. This is done by repeatedly computing $D_\text{KL}$ under the null hypothesis (no systematic present):

```python
def calibrate_null_dkl_neat(null_posterior, prior, N, n_theta,
                             compressor, nl_null, beam_null, dkl_obs):
    dkls = np.zeros(N)
    for i in range(N):
        theta_i = np.asarray(prior.sample((1,)).squeeze(0), dtype=float)

        # Same CMB sky, independent noise draws
        x1_i = blanket_simulator(theta_i, compressor, nl_null,
                                 beam_fwhm=beam_null,
                                 seed_cmb=i, seed_noise=i + 1_000_000)
        x2_i = blanket_simulator(theta_i, compressor, nl_null,
                                 beam_fwhm=beam_null,
                                 seed_cmb=i, seed_noise=i + 2_000_000)

        dkls[i], _, _ = calc_dkl(null_posterior, null_posterior,
                                  x1_i, x2_i, n_theta=n_theta)

    crit_val_95 = float(np.quantile(dkls, 0.95))
    # [plotting code omitted]
    return dkls, crit_val_95
```

**Design choices:**
- `N` realisations are drawn, each from a freshly sampled $\theta_i$, so the null distribution marginalises over the prior.
- Both splits use `beam_null` — the same instrument — so any $D_\text{KL} > 0$ is due only to different noise realisations (the irreducible null variance).
- The **95th percentile** of this null distribution serves as the critical value. An observed $D_\text{KL}$ exceeding this threshold indicates rejection of the null hypothesis at the 5% significance level.
- Crucially, both splits use the **same posterior** (`null_posterior`), which means the test is asking: "given only the null model, how large can the divergence be due to noise alone?" This is the correct null against which to compare the systematic observation.

---

### Stage 13: Perturbed Calibration and Sensitivity Sweep — `calibrate_null_dkl_and_perturbed` / `calibrate_null_dkl_and_perturb_mean`

These functions extend the null calibration to also compute the $D_\text{KL}$ distribution under a perturbed beam, enabling sensitivity analysis:

```python
def calibrate_null_dkl_and_perturbed(null_posterior, prior, N, n_theta,
                                      compressor, nl_null, beam_null, beam_perturbed):
    dkls = np.zeros(N)
    dkls_perturbed = np.zeros(N)
    for i in range(N):
        theta_i = np.asarray(prior.sample((1,)).squeeze(0), dtype=float)

        x1_i = blanket_simulator(..., beam_fwhm=beam_null,
                                 seed_cmb=i, seed_noise=i + 1_000_000)
        x2_i = blanket_simulator(..., beam_fwhm=beam_null,
                                 seed_cmb=i, seed_noise=i + 2_000_000)

        # Perturbed: same sky and noise as x1_i, only beam is different
        x1_i_perturbed = blanket_simulator(..., beam_fwhm=beam_perturbed,
                                           seed_cmb=i, seed_noise=i + 1_000_000)

        dkls[i],           _, _ = calc_dkl(null_posterior, null_posterior,
                                            x1_i, x2_i, n_theta=n_theta)
        dkls_perturbed[i], _, _ = calc_dkl(null_posterior, null_posterior,
                                            x1_i_perturbed, x2_i, n_theta=n_theta)

    crit_val_95           = float(np.quantile(dkls, 0.95))
    crit_val_95_perturbed = float(np.quantile(dkls_perturbed, 0.95))
    ...
```

**Critical design — "same sky and noise, only beam changes":** For the perturbed split, `seed_cmb=i` and `seed_noise=i+1_000_000` are both kept the same as for `x1_i`. Only `beam_fwhm` changes. This means the measured $D_\text{KL}$ isolates the beam effect exactly, without conflating it with sampling variance from a different sky or noise realisation.

**Sensitivity sweep** (`calibrate_null_dkl_and_perturb_mean`): A version of this function loops over a range of beam multipliers:

```python
multipliers_to_test = [1.00025, 1.0005, 1.000625, 1.00075]

for multiplier_val in multipliers_to_test:
    results = calibrate_null_dkl_and_perturb_mean(
        posterior1_direct, prior, 200, 500,
        my_compressor, nl_base, beam_base, multiplier=multiplier_val
    )
    null_threshold   = results[1]  # 95th percentile of null DKL
    perturbed_median = results[4]  # Median of perturbed DKL

    if perturbed_median >= null_threshold:
        print(f"Sensitivity threshold reached at multiplier = {multiplier_val}!")
```

**Test statistic:** The sensitivity threshold is defined as the point where the **median** of the perturbed $D_\text{KL}$ distribution exceeds the **95th percentile** of the null distribution. This corresponds roughly to a 50% detection rate at the 5% false positive level — a reasonable definition of the minimum detectable systematic.

---

## Full Pipeline Flow Summary

```
Fiducial Parameters
       │
       ▼
  getSpectrum (CAMB)
       │
       ▼
  getCompression (SCORE/MOPED)
  ─── Compressor fixed at null beam ───────────────────────────┐
       │                                                         │
       ▼                                                         │
  define_normal_prior                                            │
       │                                                         │
       ▼                                                         │
  generate_theta (100,000 samples)                               │
       │                                                         │
       ├──────────────────────┬─────────────────────────────────┤
       │                      │                                  │
       ▼                      ▼                                  │
  parallel_simulate       parallel_simulate                      │
  (beam_base)             (beam_systematic)                      │
       │                      │                                  │
       ▼                      ▼                                  │
  x_train_1              x_train_2                              │
       │                      │                                  │
       ▼                      ▼                                  │
   NPE training 1         NPE training 2                         │
       │                      │                                  │
       ▼                      ▼                                  │
  posterior1_direct      posterior2_direct ◄────────────────────┘
       │                      │
       └──────────┬───────────┘
                  ▼
            calc_dkl(posterior1, posterior2, x_obs_1, x_obs_2)
                  │
                  ▼
            D_KL_observed
                  │
                  ▼
      calibrate_null_dkl(posterior1, prior, N=100, ...)
                  │
                  ▼
            Null Distribution + 95% Critical Value
                  │
                  ▼
         Compare D_KL_observed vs. crit_val_95
                  │
           ┌──────┴──────┐
           ▼             ▼
         PASS          REJECT
      (no systematic) (systematic detected)
```

---

## Key Design Decisions Summary

| Decision | Rationale |
|---|---|
| NPE over NLE/NRE | Requires normalised, pointwise-evaluable posterior for KL divergence calculation |
| SCORE/MOPED compression | Reduces 3000 $C_\ell$ values to 5, dramatically improving NPE training convergence |
| Compressor fixed at null beam | Creates intentional mismatch so systematic biases the compressed statistics |
| Normal prior over BoxUniform | Reflects physical knowledge; improves training efficiency |
| Decoupled `seed_cmb` / `seed_noise` | Enables isolating the beam systematic from noise variance |
| Same `theta` and `seed_cmb` for both training sets | Ensures NPE training difference is purely due to beam |
| `norm_posterior=True` in `log_prob` | Corrects for leakage outside prior support in the KL estimate |
| Null calibration via empirical distribution | Accounts for non-Gaussian null distribution; no parametric assumptions needed |
| Median of perturbed vs. 95th percentile of null | Defines sensitivity threshold as 50% detection rate at 5% false positive level |

---

## Output Artefacts

The notebook saves a `results` dictionary containing:

- `samples1`, `samples2` — posterior samples (N, 5) from null and systematic posteriors
- `posterior1_direct`, `posterior2_direct` — fast direct-sampling posteriors for KL computation
- `posterior1_mcmc`, `posterior2_mcmc` — MCMC posteriors for cross-checks
- `inference1`, `inference2` — trained NPE objects
- `theta`, `x_train_1`, `x_train_2` — training data
- `x_obs_1`, `x_obs_2` — compressed observed data vectors
- `nl_base`, `beam_base`, `beam_systematic` — instrument configuration
- `TRUE_SEED_CMB`, `TRUE_SEED_NOISE` — seeds used to generate observations
