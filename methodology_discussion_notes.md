# Methodology Discussion Notes
**Project:** Searching for Systematics in the CMB with Simulation-Based Inference  
**Student:** Jay Vasireddy  
**Date:** 22 April 2026  
**Participants:** Jay Vasireddy, James Alvey, William Coulton, David Yallup

---

## Summary of Key Methodological Clarification

A significant clarification to the core methodology emerged from email correspondence with supervisors, resolving an ambiguity in the two-posterior setup that was reflected in both the pipeline implementation and the application text.

---

## The Original (Incorrect) Framing

The original implementation used **two separately trained posteriors**:

- **posterior1:** trained on null simulations (no systematic)
- **posterior2:** trained on simulations **with a systematic injected into the training data**

The observed KL divergence was then computed as KL(p1(θ|x_obs_1) || p2(θ|x_obs_2)).

**Problem:** This requires knowing what systematic you are testing for at training time, which defeats the purpose of a general null test for *unknown* systematics.

---

## James's Corrected Procedure

James clarified the correct procedure in email correspondence:

1. Have fixed simulation models for each data split: p1(x|z) and p2(x|z) — **no systematics in either**
2. Train a separate SBI model on each split (again, no systematics)
3. Sample data x1, x2, z ~ p1(x|z), p2(x|z), p(z) and compute KL(p1(z|x1) || p2(z|x2)) under the null — this gives the **null distribution**
4. Test on observed data x_1obs, x_2obs, which **may or may not contain a systematic**, by evaluating the trained null posteriors on the observed splits

The systematic is never in the training data — it manifests only through the observed data vectors at test time.

**David and Will confirmed this framing.** Will added that the key point is the method should detect effects not already known about in detail and not currently modelled — using known systematics in demonstrations is fine, but the inference model itself must remain agnostic.

---

## Why the Null Distribution is Unaffected

Under the null hypothesis, p1 and p2 are trained on statistically identical data (just different splits). Therefore:

> KL(p1(·|x_obs_1) || p2(·|x_obs_2)) = KL(p1(·|x_obs_1) || p1(·|x_obs_2))

up to Monte Carlo noise from finite training data. This means:
- The calibration procedure is unchanged — it always used only null simulations
- The sole change is in how the **observed KL** is computed: use p1 evaluated on both observed splits, not a separately trained p2

---

## On Data Splits with Different Information Content

**Question:** If splits have genuinely different information (e.g. high vs. low multipoles), should we expect identical posteriors under the null?

**Answer:** No — different splits have different constraining power, so posteriors will differ in width and shape even under the null. However, they should be *statistically consistent*, meaning the KL between them falls within the calibrated null distribution.

This is a strength of the method: the null distribution automatically accounts for differences in information content between splits, without requiring any analytical correction. Classical difference spectrum null tests would require more careful handling of this.

---

## On the Role of the Second Trained Posterior

**Question:** For identical splits (same sky, different noise realisations, e.g. 2D Gaussians), is there any significance to the second trained posterior?

**Answer:** No. A single well-trained posterior handles both observed data vectors. The only difference between the two observations is noise, which the null calibration already captures.

---

## On the Analytic Posterior Checks (2D Gaussian)

**Original role:** Validate that the two-posterior setup correctly recovers the analytic solution when a systematic is injected into training.

**New role:** Validate that the network converges correctly — a standard SBI diagnostic. These results show the NPE faithfully learns the posterior in a tractable setting.

**Key scientific point (Jay's observation, confirmed):** Standard SBI validation checks (e.g. NPC) are blind to the failure mode where a network converges correctly but on a biased model. A systematic that shifts the data but leaves the likelihood surface smooth will produce a perfectly converging network that confidently returns biased posteriors. The analytic posterior checks demonstrate this: *standard diagnostics pass, yet the posteriors are biased* — which is precisely the motivation for this framework.

This reframes the 2D Gaussian results from "no longer useful" to a **key part of the motivation** in the paper.

---

## On Taking a Draw from the Likelihood

Taking a draw from the likelihood p(x|z) is equivalent to taking an observation with the detector. In the pipeline, `blanket_simulator(theta, beam_fwhm=beam_base, seed_cmb=i, seed_noise=i+1_000_000)` is the null detector: beam and noise parameters define the instrument, seeds define the specific sky and noise realisation, and the output is the compressed data vector — what the detector handed you.

---

## Revised Description of the Method (Application Text)

The corrected two-paragraph description of the method for the IoA application is:

> The key aim is to identify unknown systematics by searching for statistically significant differences in inferred posteriors across various data splits. The new concept that my project aims to introduce is the use of the KL divergence as a test statistic for the presence of a systematic error. This involves taking physically motivated splits of mock CMB datasets and training an NPE posterior on each split, obtaining two posteriors. We then split our observed dataset in the same manner, where one of the observed splits may contain a systematic error. Conditioning our posteriors on the observed data splits, we compute the observed KL divergence between posteriors.
>
> This statistic is then used to perform a frequentist hypothesis test, comparing the observed KL value to a calibrated distribution of 'typical' KL values under the null hypothesis that both data splits are free of systematics and differ only due to noise. This null distribution is obtained via Monte Carlo simulations. For each simulation, we generate mock observations from the joint likelihood, split them as above, and compute the KL between each posterior conditioned on the corresponding split. This calibration process can be extended to find the distribution of KL values under a specific perturbation and compare these to the null threshold. This is achieved by adding a systematic to one of the mock observation splits during the Monte Carlo process, allowing us to quantify the sensitivity level of SBI analyses to specific systematics.

**Key changes from original:**
- Systematic is no longer injected into training simulations — only into observed/test data
- Observed KL uses the null-trained posteriors evaluated on observed splits
- Calibration description clarified to make clear both posteriors are null-trained
- Topic sentence added per James's suggestion

---

## Action Items

- [ ] Update pipeline code: remove second trained posterior; compute observed KL using p1 evaluated on both x_obs_1 and x_obs_2
- [ ] Update methodological diagram to reflect single-posterior training setup
- [ ] Update Plan.tex description of the method
- [ ] Reframe 2D Gaussian analytic checks as convergence validation + motivation for the framework
- [ ] Include NPC failure mode point in paper motivation section
- [ ] Copy David and Will on any further methodological clarifications
