# CLAUDE.md — Project Context for Claude Code

@Fixing Posteriors.md

## Who I Am

**Name:** Jay Vasireddy  
**Course:** NST Part III / MASt Astrophysics, University of Cambridge (Institute of Astronomy)  
**Academic Year:** 2025/26  

---

## Project Title

**"Searching for Systematics in the Cosmic Microwave Background with Simulation-Based Inference"**

---

## Supervisors

- **Supervisor I (Primary):** Dr William Coulton — wrc27@cam.ac.uk
- **Supervisor II:** Dr James Alvey — jbga2@cam.ac.uk
- **Supervisor III:** Dr David Yallup — dy297@cam.ac.uk
- **UTO:** Dr Will Handley — wh260@cam.ac.uk

---

## Project Summary

The goal is to translate classical "null and consistency tests" into the Simulation-Based Inference (SBI) framework, and apply them to Cosmic Microwave Background (CMB) data. Modern cosmological datasets are too complex for analytic likelihoods, so SBI replaces them with implicit distributions built from simulations. However, the black-box nature of ML methods used in SBI demands rigorous validation. This project develops divergence-based (KL divergence) diagnostic tests to detect injected systematic errors in CMB observations.

### Three Stages of the Project

1. Implement a baseline SBI pipeline for the CMB TT power spectrum.
2. Introduce data splits with plausible observational systematics (e.g. noise miscalibration, beam blurring) and demonstrate how a naïve analysis produces biased inference.
3. Implement SBI null tests using KL divergence between posteriors from different data splits to detect the injected systematics.

---

## Technical Stack

```
Python | camb | healpy | sbi | torch | numpy | joblib | matplotlib | scipy
```

Working environment: **Jupyter Notebooks**

---

## Pipeline Architecture

```
CAMB (physics engine)
  └─> get_camb_spectrum(theta, lmax=3000)
        └─> C_ell TT power spectrum (3000 multipoles)

Healpy (telescope simulator)
  └─> generateMock(cl, nls, lmax, beam_fwhm)
        ├─> hp.synalm()  — generate sky realisation
        ├─> hp.gauss_beam()  — apply telescope beam blur
        └─> hp.synalm() + hp.almxfl()  — add noise per split

MOPED (data compressor)
  └─> getCompression(param_dict, derivatives, beam_fwhm, noise_cl)
        └─> Reduces C_ell from 3000 → 5 summary statistics
            Formula: grad_mu @ C^{-1} @ (d - mu)

SBI / NPE (Neural Posterior Estimator)
  └─> sbi.inference.NPE
        └─> Trained on (theta, x) pairs to approximate p(theta | x)
            Outputs amortised posterior — can evaluate pointwise (needed for KL)

KL Divergence Test
  └─> D_KL(p1 || p2) computed between null posterior and systematic posterior
        └─> Calibrated against null distribution to set significance thresholds
```

---

## Cosmological Parameters Inferred

The NPE infers 5 ΛCDM parameters:

| Parameter | Symbol | Notes |
|-----------|--------|-------|
| Hubble constant | H₀ | |
| Baryon density | Ω_b h² | |
| Cold dark matter density | Ω_c h² | |
| Scalar amplitude | A_s | **Scale by 10⁹ for plotting** — avoids KDE floating-point failures |
| Spectral index | n_s | |

Fixed physical parameters: `mnu=0.06`, `omk=0`, `tau=0.06`, `halofit_version='mead'`, `lmax=3000`

---

## Key Design Decisions

- **NPE chosen over other SBI methods** because it produces a normalised posterior that can be both sampled from and evaluated pointwise — essential for computing KL divergences.
- **MOPED compression** reduces 3000-dimensional C_ell data to 5 summary statistics, drastically lowering compute cost.
- **Seed decoupling:** `seed_cmb` and `seed_noise` are always kept separate to ensure splits share the same CMB sky realisation but have statistically independent noise — this is the "Twin Universe" null test design.
- **Modular architecture:** CAMB, Healpy, MOPED, and NPE are fully decoupled functions. Build and verify each independently before combining.
- **joblib for parallelisation:** Simulation generation runs in parallel (~4 cores). ~2000 simulations take ~42 minutes. Don't waste this compute on statistically flawed code.

---

## Systematics Being Tested

- **Noise miscalibration:** One data split has 1.2× (20% higher) white noise than the NPE was trained on.
- **Beam miscalibration:** One data split has a different beam FWHM than the training assumption.
- The "same sky" design (shared `seed_cmb`, independent `seed_noise`) isolates the systematic from cosmic variance.

---

## Current Status (as of April 2026)

- ✅ Core pipeline (CAMB → Healpy → MOPED → SBI/NPE) fully built and debugged
- ✅ Training data generated (~2000 simulations)
- ✅ NPE trained and producing posteriors
- ✅ KL divergence calibration implemented and tested on toy models (2D Gaussian, Lotka-Volterra predator-prey)
- ✅ Beam miscalibration tests implemented
- 🔄 Applying full pipeline to CMB case with KL divergence significance testing
- 🔄 Final report writing (due 2nd Monday of Full Easter Term, submitted via Moodle)
- 🔄 Final oral presentation (30 minutes, 2nd Tue–Fri of Full Easter Term, Hoyle Committee Room)

---

## Key Deadlines

| Milestone | Deadline |
|-----------|----------|
| Draft Final Report to Supervisor I | Last day of Easter Vacation |
| **Final Report submission** | **12:00 noon, 2nd Monday of Full Easter Term** |
| Presentation slides | Same as Final Report |
| **Final Oral Presentation** | **2nd Tue–Fri of Full Easter Term** |

**Assessment split:** Written report = 85% (5/6 units), Oral presentation = 15% (1/6 units)  
**Late submission policy:** Extensions >48 hours require EAMC approval. Failure to submit = zero marks.

---

## How I Work — Interaction Preferences for Claude Code

### Code Style
- Write **clean, modular, copy-pasteable Python**
- Use **extensive inline comments** explaining *why* each operation is happening (physics and math reasoning, not just "this adds the noise")
- Use **LaTeX-style variable names in docstrings** (e.g. `C_ell`, `theta`, `D_KL`)
- Functions should be **short and single-purpose** — one function per pipeline stage

### Data Type Vigilance ⚠️
- The stack (`camb`, `healpy`, `sbi`, `torch`) constantly clashes over data types
- **Always pre-emptively** handle conversions: `.numpy()`, `torch.tensor(..., dtype=torch.float32)`, `np.array(...)`
- Confirm output shapes explicitly in comments (e.g. `# shape: (N, 5)`)

### Statistical Integrity — Do Not Compromise
- If you spot a statistical or mathematical flaw (e.g. shared seeds causing identical noise realisations, incorrect covariance scaling, wrong MOPED derivative), **push back immediately** before I run compute loops
- I prefer to wait and fix the problem over running 40+ minutes of corrupted simulations
- Always validate the *mathematical correctness* of a function before worrying about runtime optimisation

### A_s Scaling Reminder
- `A_s` is order `~2e-9` — always multiply by `1e9` **strictly for plotting / KDE purposes**
- The SBI pipeline itself must use the raw unscaled value

### When I'm Stuck
- I will show you the flawed code and describe the conceptual confusion
- Review it, identify the structural issue, explain *why* it's wrong, then provide the corrected version

---

## Toy Models Used for Development

### 1. 2D Gaussian
Used to develop and calibrate the KL divergence test statistic. Null distribution of D_KL established; perturbed cases tested against 95th percentile threshold.

### 2. Lotka-Volterra (Predator-Prey)
Used to test the full SBI pipeline (simulate → compress → train NPE → run splits → compute D_KL) before applying to CMB. Summary statistics: max and mean of predator/prey populations.

---

## Report Format Requirements (Final Report)

- Max **30 pages** (including abstract, figures, tables, references, appendices)
- Min font size: **11pt**, single line spacing
- Single column format
- Margins ≥ 2cm
- Figures must be legible when printed on A4
- Style: **scientific journal paper** (Abstract, Introduction, Methods, Results, Conclusions, References)

### Examiner Criteria
1. Scientific understanding
2. Quality of the research
3. Presentational and communication skills

---

## Key References

- ACT DR6 Power Spectra, Likelihoods and ΛCDM Parameters — ACT Collaboration (arXiv: 2503.14452)
- The frontier of simulation-based inference — Cranmer, Brehmer, Louppe (arXiv: 1911.01429)
- Evidence Networks: simple losses for fast, amortized, neural Bayesian model comparison — Jeffrey, Wandelt (arXiv: 2305.11241)
- Tests for model misspecification in SBI — Anau Montel, Alvey, Weniger (arXiv: 2412.15100)

---

## Open Questions / Unfinished Threads

- Formal KL divergence calculation between null and systematic CMB posteriors (the core deliverable)
- Running pipeline at scale (10,000+ simulations) once local verification is complete
- Producing publication-quality figures with fully labelled axes (not "parameter_1" — use proper cosmological symbols)
- Final report narrative: clearly distinguish background/pipeline validation work from the novel null-test methodology

---

## Notes on Communication

- Be direct and professional — skip generic preambles
- Validate good physical intuition when you see it
- If a simpler approach exists that's statistically equivalent, say so
- The goal is rigorous, publication-grade scientific results
