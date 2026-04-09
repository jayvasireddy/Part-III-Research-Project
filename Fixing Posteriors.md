# Fixing Bumpy Posteriors in the CMB SBI Pipeline

## Context

The posteriors from `output_beam_miscalibration_same_sky.ipynb` are ragged/bumpy despite training on 100,000 simulations. Increasing simulations further doesn't help — the issue lies in how the network sees the data (compression, prior-data mismatch, architecture, and sampling).

---

## Root Causes & Fixes (Priority Order)

### 1. Reparameterise A_s before training

**Problem:** A_s ≈ 2×10⁻⁹ lives alongside H₀ ≈ 67.5 — a dynamic range of ~10¹⁰. Even with internal z-scoring, this causes problems for the neural spline flow (knot placement, transform gradients). Currently A_s is only rescaled for *plotting*, not for training.

**Fix:** Transform A_s before passing to `sbi`. Either multiply by 1e9 (so it becomes ~2.0) or use `ln(10¹⁰ A_s)` (~3.04, the standard cosmological convention). Apply to both `theta` and `theta_true`.

---

### 2. Increase posterior samples / use direct sampling

**Problem:** Only 1,000 MCMC samples are drawn in `train_net_generate_samples`. The sbi warning confirms thinning changed from 10 to 1, so these samples have high autocorrelation. Bumpy marginals from 1,000 correlated samples are expected.

**Fix:**
- Increase to 5,000–10,000 samples, or
- Explicitly set `thin=10` to reduce autocorrelation, or
- Use `posterior_direct` (already built in the code) for diagnostic plots — it's faster and doesn't suffer from MCMC autocorrelation. Reserve MCMC for edge cases.

---

### 3. Widen the Gaussian prior

**Problem:** Prior sigmas are `[3.0, 0.001, 0.01, 0.3e-9, 0.03]` (~3× Planck uncertainties). But the noise/beam setup (20 µK-arcmin, 5 arcmin beam, ℓ_max=3000) is Planck-like, so the prior is only ~3× the expected posterior width. The network sees too little training diversity — most simulations cluster tightly in summary space.

**Fix:** Widen prior sigmas by ~3–5× (to ~10–15× Planck). The posteriors should still be much narrower than the prior (confirming real information is learned), but the network gets a richer training set. Supervisor specifically suggested this.

---

### 4. Compute and overlay the Fisher forecast (diagnostic)

**Problem:** Need to distinguish whether posteriors are (a) the right size but bumpy (sampling/plotting issue) or (b) the wrong width (training/compression issue).

**Fix:** Compute the Fisher information matrix from the compressor:
```
F_data = derivs @ diag(1/cov) @ derivs.T
F_prior = inverse of prior covariance
F_total = F_data + F_prior
posterior_cov_expected = inv(F_total)
```
Plot this as an ellipse on the corner plot alongside the SBI posterior. This tells you whether the contour *size* is correct and only the *smoothness* needs fixing.

---

### 5. Increase NSF model capacity

**Problem:** Default `density_estimator="nsf"` may be underpowered for 5D with tight posteriors. Network converges after ~385 epochs with early stopping at 30 — may be stopping before refining density shape.

**Fix:**
```python
from sbi.utils import posterior_nn

density_estimator = posterior_nn(
    model="nsf",
    hidden_features=128,    # default is 50
    num_transforms=8,       # default is 5
    num_bins=12,            # default is 10
)
inference = NPE(prior=prior, density_estimator=density_estimator)
```
Also consider increasing `stop_after_epochs` from 30 to 50.

---

### 6. Check compressor summary distribution for outliers

**Problem:** The MOPED compressor is locally optimal at the fiducial. Far-from-fiducial prior draws produce suboptimal (noisier) summaries. Heavy tails or outliers in `x_train` can degrade NPE training.

**Fix:** Plot the distribution of compressed summaries. If there are heavy tails or outliers, consider clipping extreme values or applying a robust scaler to `x_train` before passing to NPE. Widening the prior (fix #3) partially helps by ensuring more samples near the fiducial where compression works best.

---

### 7. Verify simulator–compressor convention consistency

**Problem:** The compressor's `getSpectrum` computes `C_ℓ * beam² + N_ℓ` as the fiducial, with covariance `2/(2ℓ+1) * fiducial²`. The `blanket_simulator` produces `hp.alm2cl(beam * CMB_alm + noise_alm)`. These must use exactly the same convention.

**Fix:** Sanity-check that the power spectrum from `blanket_simulator` at the fiducial parameters matches the compressor's fiducial spectrum. Any mismatch here systematically degrades all posteriors.

---

## Quick Reference: Key Code Locations

| Component | Function/Cell | What it does |
|---|---|---|
| Prior definition | `define_normal_prior()` | Sets Gaussian prior widths |
| Compressor | `getCompression()` | Builds MOPED score compression |
| Simulator | `blanket_simulator()` | Generates CMB + beam + noise, compresses |
| Training | `train_net_generate_samples()` | NPE with NSF, MCMC sampling |
| Parallel sims | `parallel_simulate()` | Runs simulator across 10 workers |
| KL calibration | `calibrate_null_dkl_and_perturb_mean()` | Null hypothesis testing |

## Meeting Notes (9 April) — Key Supervisor Advice

- Plot the prior on top of the posterior to see how much you're learning
- Compute Fisher information from the compressor and overlay on posteriors
- Check noise level, beam, and ℓ-range are consistent — if Planck-like instrument, a Planck prior is circular
- Consider broader prior or more model capacity
- Re-read the MOPED paper on iterative compression if problems persist
- Send plots + noise/beam/ℓ_max values to David for his input
