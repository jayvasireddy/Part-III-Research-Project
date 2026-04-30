# Agent Brief: Classical Chi-Squared Analogue for Signal Perturbation Test

## Target File

`Cmb_Test_signal_perturbation.ipynb`
https://github.com/jayvasireddy/Part-III-Research-Project/blob/main/Cmb_Test_signal_perturbation.ipynb

Add new cells **at the end** of the notebook. Do not modify any existing cells.

---

## Objective

Add a classical difference-spectrum chi-squared null test and produce a direct comparison of its sensitivity threshold against the existing SBI (KL divergence) sensitivity threshold, for the same signal gain perturbation levels.

This is the "non-SBI analogue" discussed with supervisors. The goal is to show the SBI method recovers consistent sensitivity levels to the classical approach — the key credibility check before the paper is written up.

---

## Context: The SBI Test (Already Implemented)

The notebook should contain:

1. A single NPE posterior (`null_posterior`) trained on null CMB simulations.
2. A `blanket_simulator(theta, compressor, nl_split, lmax, beam_fwhm, seed_cmb, seed_noise)` function returning **compressed 5D summary statistics** (not raw $C_\ell$).
3. A signal gain perturbation — **read the notebook to confirm exactly how this is parameterised** before writing any new code. It is expected to be a multiplicative factor on the signal amplitude, but the exact variable name and whether it scales at field level or power level must be confirmed from the source.
4. A calibration/sensitivity sweep finding the perturbation level at which the **median** of the perturbed KL distribution exceeds the **95th percentile of the null KL distribution**.
5. Seed convention: `seed_cmb=i`, `seed_noise=i+1_000_000` for split 1, `seed_noise=i+2_000_000` for split 2. Perturbed split 1 reuses `seed_cmb=i` and `seed_noise=i+1_000_000`.

**Read the notebook first.** Confirm the actual variable names used for `nl_null`, `beam_null`/`beam_base`, `lmax`, `prior`, `null_posterior`, and the gain perturbation parameter before writing any new code.

---

## Classical Analogue: Procedure

### Step A — Form the difference spectrum

Under the null (same `seed_cmb`, independent noise seeds):
$$\Delta\tilde{C}_\ell = \tilde{C}_\ell^{(1)} - \tilde{C}_\ell^{(2)}$$
The CMB signal cancels exactly. Under a gain perturbation on split 1, the residual is $\approx \delta g \cdot C_\ell$ (to first order), which inflates the statistic.

### Step B — Noise-only covariance (diagonal)

For two independent noise realisations each with spectrum `nl_null`:
$$\text{Var}(\Delta\tilde{C}_\ell) = \frac{4}{2\ell+1} N_\ell^2$$
where $N_\ell$ = `nl_null[ell]`. **Use `nl_null` (pure noise spectrum) only in the denominator — do not use `fiducial`, which includes signal and beam.**

### Step C — Chi-squared statistic

$$\chi^2 = \sum_{\ell=2}^{\ell_\text{max}} \frac{(\Delta\tilde{C}_\ell)^2}{\text{Var}(\Delta\tilde{C}_\ell)}$$

Exclude $\ell = 0, 1$ (monopole/dipole), matching the compressor's `data[2:]` convention.

### Step D — Sensitivity threshold

Identical to the SBI test: smallest perturbation level at which the **median** of the perturbed $\chi^2$ distribution exceeds the **95th percentile of the null $\chi^2$ distribution**.

---

## Gain Perturbation Parameterisation

**Confirm from the notebook before coding.** The expected implementation is:

```python
cl_pure_perturbed = cl_pure * gain_multiplier**2
```

where `gain_multiplier = 1 + delta_g` (a field-level gain, squaring gives the power-level factor). If the notebook uses a different convention, match it exactly in `get_raw_cl` — the two tests must use identical perturbation definitions.

---

## New Functions to Add

### 1. `get_raw_cl`

Required because `blanket_simulator` returns compressed summaries; the classical test needs the full $C_\ell$ vector. This function must mirror `blanket_simulator`'s internal logic exactly.

```python
def get_raw_cl(theta, nl, beam_fwhm, lmax, seed_cmb, seed_noise, gain_multiplier=1.0):
    """
    Generate a raw (uncompressed) observed C_ell for use in the classical chi-squared test.
    Mirrors blanket_simulator's internal healpy logic, with an optional gain perturbation.

    Parameters
    ----------
    theta : array-like, shape (5,)
    nl : np.ndarray, shape (lmax+1,) — noise power spectrum
    beam_fwhm : float — beam FWHM in arcminutes
    lmax : int
    seed_cmb : int
    seed_noise : int
    gain_multiplier : float
        Applied as cl_pure *= gain_multiplier**2. Default 1.0 (no perturbation).

    Returns
    -------
    cl_obs : np.ndarray, shape (lmax+1,)
    """
    theta = np.asarray(theta, dtype=float)
    cl_pure = get_camb_spectrum(theta, lmax=lmax)[:lmax+1]
    cl_pure = cl_pure * gain_multiplier**2

    np.random.seed(int(seed_cmb))
    cmb_alm = hp.synalm(cl_pure, lmax=lmax)

    beam = hp.gauss_beam(beam_fwhm / 60 / 180 * np.pi, lmax=lmax)
    cmb_alm_beamed = hp.almxfl(cmb_alm, beam)

    np.random.seed(int(seed_noise))
    noise_alm = hp.synalm(np.asarray(nl)[:lmax+1], lmax=lmax)

    return hp.alm2cl(cmb_alm_beamed + noise_alm, lmax=lmax)
```

### 2. `compute_chi2_difference`

```python
def compute_chi2_difference(cl1, cl2, nl, lmax, ell_min=2):
    """
    Chi-squared on the difference spectrum using diagonal noise-only covariance.

    Parameters
    ----------
    cl1, cl2 : np.ndarray, shape (lmax+1,) — raw observed C_ell for each split
    nl : np.ndarray — noise power spectrum (NOT fiducial)
    lmax : int
    ell_min : int — default 2, excluding monopole and dipole

    Returns
    -------
    chi2 : float
    """
    ells = np.arange(ell_min, lmax + 1)
    delta_cl = cl1[ell_min:lmax+1] - cl2[ell_min:lmax+1]
    var_delta = (4.0 / (2.0 * ells + 1)) * nl[ell_min:lmax+1]**2
    return float(np.sum(delta_cl**2 / var_delta))
```

### 3. `calibrate_classical_chi2`

```python
def calibrate_classical_chi2(prior, N, nl_null, beam_null, lmax,
                               gain_multiplier=1.0, ell_min=2):
    """
    Build null and perturbed chi-squared distributions.
    Uses identical seed convention to the SBI calibration for direct comparability.

    Parameters
    ----------
    prior : MultivariateNormal
    N : int — number of Monte Carlo realisations
    nl_null : np.ndarray — null noise power spectrum
    beam_null : float — null beam FWHM in arcminutes
    lmax : int
    gain_multiplier : float — 1.0 = null; >1.0 = perturbed split 1
    ell_min : int

    Returns
    -------
    chi2s_null : np.ndarray, shape (N,)
    chi2s_perturbed : np.ndarray, shape (N,)
    crit_val_95 : float
    median_perturbed : float
    """
    chi2s_null = np.zeros(N)
    chi2s_perturbed = np.zeros(N)

    for i in range(N):
        if i % 100 == 0:
            print(f"{i}/{N}")

        theta_i = np.asarray(prior.sample((1,)).squeeze(0), dtype=float)

        # Null splits: same sky, independent noise, no perturbation
        cl1_null = get_raw_cl(theta_i, nl_null, beam_null, lmax,
                               seed_cmb=i, seed_noise=i + 1_000_000,
                               gain_multiplier=1.0)
        cl2_null = get_raw_cl(theta_i, nl_null, beam_null, lmax,
                               seed_cmb=i, seed_noise=i + 2_000_000,
                               gain_multiplier=1.0)

        # Perturbed split 1: same sky & noise seeds, only gain changes
        cl1_perturbed = get_raw_cl(theta_i, nl_null, beam_null, lmax,
                                    seed_cmb=i, seed_noise=i + 1_000_000,
                                    gain_multiplier=gain_multiplier)

        chi2s_null[i] = compute_chi2_difference(
            cl1_null, cl2_null, nl_null, lmax=lmax, ell_min=ell_min)
        chi2s_perturbed[i] = compute_chi2_difference(
            cl1_perturbed, cl2_null, nl_null, lmax=lmax, ell_min=ell_min)

    crit_val_95 = float(np.quantile(chi2s_null, 0.95))
    median_perturbed = float(np.median(chi2s_perturbed))
    return chi2s_null, chi2s_perturbed, crit_val_95, median_perturbed
```

---

## Sensitivity Sweep

Use the **same gain levels** as the existing SBI sweep — read these from the notebook. The comparison is only meaningful if both tests cover the same x-axis range.

```python
# Step 1: null calibration
chi2s_null, _, crit_val_95_classical, _ = calibrate_classical_chi2(
    prior, N=500, nl_null=nl_null, beam_null=beam_null, lmax=lmax, gain_multiplier=1.0
)
print(f"Classical null 95th percentile chi2: {crit_val_95_classical:.4f}")
# Sanity check: should be close to lmax - 1 + (a few sigma * sqrt(2*(lmax-1)))
# e.g. for lmax=3000: expect ~3000 +/- 150

# Step 2: sweep — replace gain_levels with the values used in the SBI sweep
gain_levels = [...]  # READ FROM NOTEBOOK

classical_medians = []
for g in gain_levels:
    _, _, _, median_pert = calibrate_classical_chi2(
        prior, N=500, nl_null=nl_null, beam_null=beam_null, lmax=lmax, gain_multiplier=g
    )
    classical_medians.append(median_pert)
    print(f"gain={g:.6f}, median chi2={median_pert:.2f}, "
          f"crossed: {median_pert >= crit_val_95_classical}")

# Step 3: threshold
classical_threshold = next(
    (g for g, m in zip(gain_levels, classical_medians) if m >= crit_val_95_classical),
    None
)
print(f"\nClassical sensitivity threshold: {classical_threshold}")
```

---

## Comparison Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# Left: SBI — use variable names from the existing notebook
ax = axes[0]
ax.plot(gain_levels, sbi_medians, color='tab:blue', marker='o', markersize=4,
        label=r'Median perturbed $D_\mathrm{KL}$')
ax.axhline(crit_val_95_sbi, color='red', linestyle='--', linewidth=1.5,
           label='95th percentile (null)')
if sbi_threshold is not None:
    ax.axvline(sbi_threshold, color='tab:blue', linestyle=':', linewidth=1.5,
               label=f'SBI threshold: {sbi_threshold}')
ax.set_xlabel('Signal gain multiplier')
ax.set_ylabel(r'$D_\mathrm{KL}$')
ax.set_title('SBI sensitivity (KL divergence)')
ax.legend(fontsize=8)
ax.grid(linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# Right: classical
ax = axes[1]
ax.plot(gain_levels, classical_medians, color='tab:orange', marker='s', markersize=4,
        label=r'Median perturbed $\chi^2$')
ax.axhline(crit_val_95_classical, color='red', linestyle='--', linewidth=1.5,
           label='95th percentile (null)')
if classical_threshold is not None:
    ax.axvline(classical_threshold, color='tab:orange', linestyle=':', linewidth=1.5,
               label=f'Classical threshold: {classical_threshold}')
ax.set_xlabel('Signal gain multiplier')
ax.set_ylabel(r'$\chi^2$')
ax.set_title(r'Classical sensitivity (difference spectrum $\chi^2$)')
ax.legend(fontsize=8)
ax.grid(linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

plt.suptitle('SBI vs Classical Sensitivity: Signal Gain Perturbation', fontsize=12)
plt.tight_layout()
plt.savefig('figures/classical_vs_sbi_sensitivity_gain.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSBI sensitivity threshold:       {sbi_threshold}")
print(f"Classical sensitivity threshold: {classical_threshold}")
```

---

## Sanity Checks — Run These Before Saving

**1. Null chi-squared mean.**
```python
print(f"Null chi2 mean:  {np.mean(chi2s_null):.1f}  (expected ~{lmax - 1})")
print(f"Null chi2 std:   {np.std(chi2s_null):.1f}   (expected ~{np.sqrt(2*(lmax-1)):.1f})")
```
If the mean is orders of magnitude off, the covariance is wrong — most likely `fiducial` was used instead of `nl_null`.

**2. Null histogram.**
Plot `chi2s_null` and verify it looks like a chi-squared distribution, roughly centred near `lmax - 1`. It should not be pathologically wide, narrow, or multi-modal.

**3. Seed consistency.**
For `i=0`, verify `get_raw_cl` with `gain_multiplier=1.0` produces a reasonable spectrum by plotting it against `fiducial`. It should match to within noise fluctuations.

**4. Threshold sanity.**
SBI and classical thresholds should agree to within a factor of ~2. If one is orders of magnitude more sensitive, flag for investigation — do not silently accept.

---

## Key Gotchas

- **`nl_null` only in the denominator, never `fiducial`.** `fiducial` includes signal + beam and would inflate the variance, making the test appear insensitive.
- **`blanket_simulator` returns compressed summaries** — never pass its output to `compute_chi2_difference`. Always use `get_raw_cl`.
- **`get_raw_cl` must mirror `blanket_simulator` exactly** — same seed application order, same beam convention (`beam_fwhm / 60 / 180 * np.pi`), same `lmax` truncation. Verify against the existing code.
- **Match the gain parameterisation.** If the SBI sweep uses `gain_multiplier` as a direct $C_\ell$ ratio (not a field-level gain), remove the `**2` in `get_raw_cl` accordingly.
- **Do not retrain the posterior.** Load from whatever checkpoint mechanism exists. The classical test needs neither the posterior nor the compressor.
