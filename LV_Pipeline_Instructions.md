# Lotka-Volterra Code: Required Changes & Graph Guidance

## Context

These instructions are based on feedback from a supervision meeting (30 April). The Lotka-Volterra model is one of several toy models used to validate an SBI-based systematic detection method via KL divergence. The other models are 2D Gaussians and a CMB case. The narrative across all models should be coherent and pedagogically motivated.

---

## Required Changes to the Lotka-Volterra Code

### 1. Noise on Training Data Splits (Critical Fix)

**Current (incorrect) behaviour:** Different kinds of noise are added to each of the two training splits — e.g. one split has Gaussian noise and the other has a different noise model (e.g. Poissonian). This was meant to represent two different detectors observing the same population.

**Required change:** Both training splits must use the **same noise model**. Only the observed (test) data should differ — that's where the systematic is introduced. The training data should be consistent across both splits.

> "The main thing would be that the training data has to be the same kind. It's just the observation that changes."

---

### 2. Systematic Perturbations to Implement

Keep the existing systematic (a percentage increase in one of the prey populations, e.g. +5%) as the primary case. In addition, consider implementing the following:

**Option A — Percentage increase (keep as-is, fix noise as above):**
- Train both splits with identical Gaussian noise
- In the observed data, apply e.g. a 5% increase to one prey population
- This is the main systematic to report

**Option B — Discrete/different noise model as systematic:**
- Train both splits with identical Gaussian noise
- In the observed data, switch to a different noise model (e.g. Poissonian) to simulate a different detector
- This is an additional, qualitatively different test that "switches things up"
- Physical analogy: observing the same sky with a satellite vs. a ground instrument — both Gaussian but with different noise structures

> "I can do like, keep it just like Gaussian noise for both and then do the percentage increase and then I can try a discrete test as well and that might be a different way to switch up."

Try to implement both if feasible. Option B is secondary — only include it in the write-up if it relates clearly to analogous tests done in the CMB case.

---

### 3. Alignment with CMB Analogues

The Lotka-Volterra examples should mirror the kinds of systematics tested in the CMB case. The CMB beam miscalibration test looks at high/low multipoles and the same sky observed differently — analogous to comparing data splits and seeing how sensitivity changes.

- The percentage increase systematic has a clear CMB analogue (beam miscalibration/gain miscalibration)
- The different-detector (different noise) systematic has a weaker CMB analogue — include it only if it strengthens the overall narrative
- The goal is a coherent story across all three toy models, not just variety for its own sake

> "Whatever kind of makes more sense to you... it's about writing a story that's a narrative here that is interesting but demonstrates your key point."

---

## General Graph Guidance

### Reducing Repetition

There will be a large number of similar histogram plots across models (2D Gaussians: 2 systematics, Lotka-Volterra: 2 systematics, CMB: ~3 systematics). To avoid the write-up feeling repetitive:

- Consider **combining subplots** within a single figure where the systematics are sufficiently similar — e.g. two related Lotka-Volterra cases side by side in one figure
- You don't necessarily need to show every case in full detail — prioritise the CMB results and use the toy models to build intuition
- It may be worth **cutting one of the 2D Gaussian cases** to make space for more CMB material, since that is the primary application

> "You could combine some of them... maybe not the first one, but... you've got two, like, similar cases."

### General Presentation

- Keep graphs clean and clearly labelled — axes, legends, titles
- Where histograms are shown for the KL divergence comparison, make sure it is visually clear which distribution corresponds to the calibration set and which to the observed data
- Graphs should be self-contained enough that a reader can follow them without needing to re-read the surrounding prose
- Consistency in style across all toy model figures is important for readability

---

## Summary of Action Items

| Task | Priority |
|---|---|
| Fix training splits to use same noise model | High |
| Rerun Lotka-Volterra code with corrected noise | High |
| Implement % increase systematic (fix noise first) | High |
| Implement discrete/different noise systematic | Medium |
| Check CMB analogue alignment before including Option B | Medium |
| Reduce/combine figures to avoid repetition | Medium |
| Ensure graph style is consistent across all toy models | Low |