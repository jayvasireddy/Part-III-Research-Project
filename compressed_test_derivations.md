# Derivations: the compressed-space null tests

*Worked derivations for the rows of the sensitivity table not already derived on the whiteboard. Written to be worked through by hand, step by step.*

---

## 0. Brief for the agent helping with this

**Who.** Jay Vasireddy is a Part III / MASt Astrophysics student (Cambridge, IoA) turning their project into a paper. The project is "Searching for Systematics in the CMB with Simulation-Based Inference". The supervisors are Will Coulton, James Alvey and David Yallup.

**The method.** A neural posterior estimator (NPE) is trained on null CMB TT simulations, compressed from 2999 multipoles to 5 numbers by MOPED. It is applied to two same-sky data splits with independent noise. The KL divergence between the two posteriors is the test statistic. Its null distribution is calibrated by simulation, and a systematic is "detected" when the median perturbed KL exceeds the null 95th percentile. Three systematics are injected into split 2: **noise** mismodelling, **beam** mismodelling and **signal gain**.

**Already derived by Jay (do not redo unless asked):**

- the difference spectrum $D_\ell=\hat C^{(1)}_\ell-\hat C^{(2)}_\ell$;
- its null variance $V_\ell = 4N_\ell(N_\ell+2S_\ell)/(2\ell+1)$;
- $\Delta_\ell$ and $\sigma_\ell^2$ for each systematic;
- the blind full-spectrum $\chi^2$ test, which is a noncentral $\chi^2_{2999}$ with $\lambda_{\rm crit}=129.2$;
- the full Fisher / matched-filter bound $F=\sum_\ell(\partial_\varepsilon\Delta_\ell)^2/V_\ell$.

**This document derives the remaining rows:**

- the compressed matched-filter bound;
- the compressed $\chi^2$;
- the ideal linear-Gaussian KL proxy;
- the prior-marginalised versions;
- the information fraction $\eta$.

The numerical results come from `classical_compressed_chi2.ipynb` in the repo.

**How to help.** Jay wants to **do the algebra on paper themselves**. Render the maths properly. Go one step at a time. For each **Exercise**, let Jay attempt it before revealing the worked line, and give hints rather than answers when asked. Use the **Checkpoint** numbers to confirm results. If Jay spots an error in this document, take it seriously and check it. Style: direct, no padding, physical intuition alongside the algebra.

---

## 1. Notation recap (from the whiteboard)

The splits are $i=1,2$, observing the same sky with independent noise. Per multipole, $M_\ell = 2\ell+1$ real modes. Write $S_\ell = B_\ell^2C_\ell$ and let $N_\ell$ be the (white) noise.

$$
\tilde C^{(i)}_\ell=\langle \hat C^{(i)}_\ell\rangle,\qquad X_\ell=\langle a^{(1)}a^{(2)}\rangle,\qquad
D_\ell=\hat C^{(1)}_\ell-\hat C^{(2)}_\ell
$$

$$
\Delta_\ell \equiv \mathbb E[D_\ell]=\tilde C^{(1)}_\ell-\tilde C^{(2)}_\ell,\qquad
\sigma_\ell^2\equiv \mathrm{Var}(D_\ell)=\frac{2}{M_\ell}\Big[(\tilde C^{(1)}_\ell)^2+(\tilde C^{(2)}_\ell)^2-2X_\ell^2\Big],
\qquad
V_\ell\equiv\sigma^2_\ell\big|_{\rm null}=\frac{4N_\ell(N_\ell+2S_\ell)}{M_\ell}
$$

| systematic | $\Delta_\ell$ | $g_\ell\equiv\partial_\varepsilon\Delta_\ell\big|_0$ |
|---|---|---|
| noise, $N^{(2)}=(1+\nu)N$ | $-\nu N_\ell$ | $-N_\ell$ |
| gain, $\tilde C^{(2)}=(1+\delta)(S+N)$ | $-\delta(S_\ell+N_\ell)$ | $-(S_\ell+N_\ell)$ |
| beam, $\theta_{\rm FWHM}\to(1+\varepsilon)\theta_{\rm FWHM}$ | $S_\ell\big[1-e^{-\ell(\ell+1)\sigma_B^2(2\varepsilon+\varepsilon^2)}\big]$ | $2\ell(\ell+1)\sigma_B^2S_\ell$ |

Different $\ell$ are independent. So, as a vector over $\ell=2,\dots,3000$ ($L=2999$ entries):

$$
\boxed{\;\mathbb E[\mathbf D]=\boldsymbol\Delta,\qquad \mathrm{Cov}(\mathbf D)=\mathrm{diag}(\sigma^2_\ell),\qquad \mathrm{Cov}(\mathbf D)\big|_{\rm null}=\mathrm{diag}(V_\ell)\;}
$$

**Checkpoints (full test, already on the board):** $F_{\rm gain}=5.225\times10^8$, $F_{\rm beam}=3.082\times10^7$, $F_{\rm noise}=9.003\times10^5$.

---

## 2. Step 1: the compression is linear, and the fiducial cancels

The pipeline's compressor (`getCompression`) is

$$
t_i(\mathbf d)=\sum_\ell W_{i\ell}\,(d_\ell-\mu_\ell^{\rm fid}),\qquad
W_{i\ell}=\frac{\partial\tilde C_\ell/\partial\theta_i}{\Sigma_\ell},\qquad
\Sigma_\ell=\frac{2\tilde C_\ell^2}{M_\ell},\quad \tilde C_\ell=S_\ell+N_\ell .
$$

Here $W$ is a $5\times2999$ matrix (`derivs / cov` in the code) and $\Sigma_\ell$ is the Gaussian variance of a *single* split's spectrum.

> **Exercise 1.** Show that compressing each split and then subtracting gives the same answer as subtracting and then compressing:
> $$\mathbf t^\Delta\equiv \mathbf t(\hat{\mathbf C}^{(1)})-\mathbf t(\hat{\mathbf C}^{(2)}) = W\mathbf D .$$
> *Hint:* where does $\mu^{\rm fid}$ go?

This is why Will's two orderings are the same calculation.

---

## 3. Step 2: the distribution of the compressed difference

> **Exercise 2.** Using $\mathbb E[A\mathbf x]=A\,\mathbb E[\mathbf x]$ and $\mathrm{Cov}(A\mathbf x)=A\,\mathrm{Cov}(\mathbf x)A^{\!\top}$, show
> $$\mathbb E[\mathbf t^\Delta]=\boldsymbol\mu\equiv W\boldsymbol\Delta,\qquad \mathrm{Cov}(\mathbf t^\Delta)=C'\equiv W\,\mathrm{diag}(\sigma^2_\ell)\,W^{\!\top},$$
> and under the null
> $$\boxed{C_t\equiv W\,\mathrm{diag}(V_\ell)\,W^{\!\top}}\qquad (5\times5).$$
> Write out the $(i,j)$ element as an explicit sum over $\ell$.

**Gaussianity.** Each $t^\Delta_i=\sum_\ell W_{i\ell}D_\ell$ is a sum of 2999 *independent* terms, so by the CLT $\mathbf t^\Delta$ is very close to Gaussian. That holds even though individual $D_\ell$ are visibly non-Gaussian at low $\ell$, which is the reason the *full* test needed the "exact" weighted-$\chi^2$ treatment.

**Checkpoint:** with the exact (Wishart) simulator, the null $T_c$ defined in Step 4 has mean 4.989, variance 9.931 and 95th percentile 11.02. For $\chi^2_5$ these are 5, 10 and 11.07.

From here on, the problem is entirely about a **5-dimensional Gaussian**.

---

## 4. Step 3: compressed matched-filter bound (row 2) and $\eta$ (row 7)

**Question.** Suppose you *know* the systematic's shape and only see the 5 compressed numbers. How small an $\varepsilon$ can you detect?

To linear order, write $\boldsymbol\mu=\varepsilon\,\mathbf g$ with $\mathbf g\equiv W\boldsymbol g$ (the systematic's signature, compressed), and approximate $C'\approx C_t$. Then

$$
\mathbf t^\Delta\sim\mathcal N(\varepsilon\,\mathbf g,\;C_t).
$$

> **Exercise 3a (Neyman–Pearson).** Write the log-likelihood ratio $\ln\frac{p(\mathbf t\,|\,\varepsilon)}{p(\mathbf t\,|\,0)}$. Show it is linear in $\mathbf t$ through the single statistic $s=\mathbf g^{\!\top}C_t^{-1}\mathbf t$. This is the matched filter, and it is the optimal test.
>
> **Exercise 3b.** Show $s\sim\mathcal N(\varepsilon F_c,\,F_c)$ with
> $$\boxed{F_c=\mathbf g^{\!\top}C_t^{-1}\mathbf g}$$
> so $z=s/\sqrt{F_c}\sim\mathcal N(\varepsilon\sqrt{F_c},1)$. Using the same 2σ convention as the full Fisher row, detection is at $\varepsilon\sqrt{F_c}=1.96$, i.e.
> $$\varepsilon_{\rm MF,c}=\frac{1.96}{\sqrt{F_c}} .$$
> (Check that $F_c$ is also the Fisher information of a Gaussian with fixed covariance, $F=\boldsymbol\mu'^{\!\top}C^{-1}\boldsymbol\mu'$.)
>
> **Exercise 3c (sanity check).** If you skip the compression, i.e. replace $W$ by the identity, show $F_c\to F=\sum_\ell g_\ell^2/V_\ell$. So row 1 is the special case "no compression".

### The information fraction $\eta$ is a squared cosine

Whiten everything by $V$. Define the 2999-vector $u_\ell=g_\ell/\sqrt{V_\ell}$ and the $2999\times5$ matrix $A=\mathrm{diag}(\sqrt{V_\ell})\,W^{\!\top}$.

> **Exercise 3d.** Show $\mathbf g=A^{\!\top}\mathbf u$, $C_t=A^{\!\top}A$ and $F=\mathbf u^{\!\top}\mathbf u$. Hence
> $$F_c=\mathbf u^{\!\top}\underbrace{A(A^{\!\top}A)^{-1}A^{\!\top}}_{P}\mathbf u,$$
> where $P$ is the orthogonal projector onto the 5-dimensional column space of $A$. Conclude
> $$\boxed{\eta\equiv\frac{F_c}{F}=\frac{\|P\mathbf u\|^2}{\|\mathbf u\|^2}=\cos^2(\text{angle between the systematic and the compression subspace})\le1 .}$$
>
> **Exercise 3e.** Show $F_c$ is unchanged under $W\to RW$ for any invertible $5\times5$ matrix $R$. So only the *subspace* matters, not the parameter basis. This is why the code can normalise the rows of $W$, whose scales differ by about $10^9$ because of $A_s$.

**σ-equivalent.** Expressed in units of the full Fisher information, $\varepsilon_{\rm MF,c}\sqrt F = 1.96/\sqrt\eta$.

**Checkpoints:**

| | $F_c$ | $\eta$ | threshold | σ-equiv |
|---|---|---|---|---|
| gain | $9.938\times10^7$ | 0.190 | 0.0197 % | 4.49 |
| beam | $2.936\times10^7$ | 0.953 | 0.0362 % | 2.01 |
| noise | $5.050\times10^5$ | 0.561 | 0.276 % | 2.62 |

### Why gain compresses so badly: the weighting argument

Gain *looks like* an $A_s$ change: $\partial\tilde C/\partial A_s=S/A_s$, while $g_\ell=-(S+N)\approx-S$ where the information is. The shape matches, but the **weights** don't.

> **Exercise 3f.** In the signal-dominated regime ($S\gg N$), show:
> - the MOPED $A_s$ row weights the data as $W_{A_s,\ell}\propto \dfrac{M_\ell S_\ell}{S_\ell^2}=\dfrac{M_\ell}{S_\ell}$;
> - the matched filter for gain weights the data as $\dfrac{g_\ell}{V_\ell}\propto\dfrac{M_\ell S_\ell}{N_\ell S_\ell}=\dfrac{M_\ell}{N_\ell}$, which is flat for white noise.
>
> MOPED was built to estimate parameters from *one* split, where bright modes are cosmic-variance limited, so it downweights them by $1/S$. In the *difference*, the sky cancels, and bright modes are the most informative.

**Checkpoints:**

- The $A_s$ row alone captures $\eta=0.015$ of the gain signal.
- The *same template* weighted by $1/V_\ell$ instead of $1/\Sigma_\ell$ captures $\eta=0.998$.
- All 5 rows together capture 0.19.
- Single-row $\eta$ for beam: $A_s$ 0.45, $n_s$ 0.35. For noise: $\Omega_bh^2$ 0.43, $n_s$ 0.39. Beam and noise are spread over several cosmology-like directions, so together they reach 0.95 and 0.56.

---

## 5. Step 4: compressed $\chi^2$ (row 3)

The systematic is **unknown**, so use the direction-agnostic statistic

$$
T_c=\mathbf t^{\Delta\top}C_t^{-1}\,\mathbf t^\Delta .
$$

> **Exercise 4a (null).** Let $\mathbf y=C_t^{-1/2}\mathbf t^\Delta$. Show $\mathbf y\sim\mathcal N(0,I_5)$, so $T_c=\|\mathbf y\|^2\sim\chi^2_5$. Hence $q_{0.95}=11.07$.
>
> **Exercise 4b (systematic, with $C'\approx C_t$).** Show $T_c\sim\chi^2_5(\lambda_c)$ (noncentral) with
> $$\boxed{\lambda_c=\boldsymbol\mu^{\!\top}C_t^{-1}\boldsymbol\mu=\varepsilon^2F_c}\quad(\text{exact for noise and gain, since }\Delta\text{ is linear in }\varepsilon).$$
>
> **Exercise 4c (threshold).** The detection criterion $\mathrm{median}[\chi^2_5(\lambda)]=11.07$ gives $\lambda_{\rm crit}=6.99$. This is a one-off numerical solve, e.g. `scipy.stats.ncx2.median`. Why is the naive "mean = threshold" answer, $\lambda=11.07-5=6.07$, wrong here, when the analogous approximation was fine at $k=2999$? *Hint:* skewness of $\chi^2_k$ scales like $1/\sqrt k$.
>
> Hence
> $$\boxed{\varepsilon_{\rm crit}=\sqrt{\frac{6.99}{F_c}},\qquad \text{σ-equiv}=\sqrt{\frac{6.99}{\eta}} .}$$
>
> **Exercise 4d.** Show that the ratio of row 3 to row 2 is $\sqrt{6.99/3.84}=1.35$ for *every* systematic. Here $3.84=1.96^2$ is the 1-dof threshold. Interpret it as the price of not knowing which of the 5 directions to watch.

**Checkpoints:** thresholds 0.0265 % (gain), 0.0488 % (beam), 0.372 % (noise); σ-equivalents 6.06, 2.71, 3.53. For example $\sqrt{6.99/9.938\times10^{7}}=2.65\times10^{-4}$. And 6.06/4.49 = 2.71/2.01 = 3.53/2.62 = 1.35.

### Why the "exact" correction is negligible here (unlike the full test)

Without approximating $C'\approx C_t$, $T_c$ is a weighted $\chi^2$ with weights $w_j=\mathrm{eig}(C_t^{-1}C')$, and $\mathbb E[T_c]=\mathrm{tr}(C_t^{-1}C')+\lambda_c$.

> **Exercise 4e.** With $r_\ell=\sigma^2_\ell/V_\ell$, show
> $$\sum_j(w_j-1)=\mathrm{tr}\!\big(C_t^{-1}(C'-C_t)\big)=\sum_\ell h_\ell\,(r_\ell-1),\qquad h_\ell\equiv V_\ell\,[W^{\!\top}C_t^{-1}W]_{\ell\ell}.$$
>
> **Exercise 4f.** Show $\sum_\ell h_\ell=5$. *Hint:* $h_\ell$ is the diagonal of the projector $P$ from Exercise 3d, and $\mathrm{tr}\,P=\mathrm{rank}=5$.
>
> So the compressed excess is **5 × a weighted average of $(r_\ell-1)$**. The full test's excess was $\sum_\ell(r_\ell-1)$, i.e. **2999 × the average**, and it had to be compared with $\lambda_{\rm crit}=129$. Here it is compared with 6.99.

**Checkpoints:**

- Compressed excess: $+0.0013$ (gain), $-0.0022$ (beam), $+0.0111$ (noise). Negligible against 6.99.
- The leverage $h_\ell$ is concentrated at high $\ell$ (weighted mean $\ell\approx1780$), so gain's large $r_\ell$ at $\ell\lesssim35$ barely enters.
- Exact and noncentral thresholds agree to < 0.2 %.

---

## 6. Step 5: the ideal linear-Gaussian KL proxy (row 4)

**Goal.** What would SBI's KL test achieve with a *perfect* network, on the same compressed data?

**Model for one split** (linear-Gaussian, raw compressor units, $\boldsymbol\delta\equiv\boldsymbol\theta-\boldsymbol\theta_{\rm fid}$):

> **Exercise 5a.** Show that for one split $\partial\mathbf t/\partial\boldsymbol\theta = W\,\partial\tilde{\mathbf C}/\partial\boldsymbol\theta\equiv F_\theta$ and $\mathrm{Cov}(\mathbf t)=W\,\mathrm{diag}(\Sigma_\ell)\,W^{\!\top}=F_\theta$, where $F_\theta=\sum_\ell \frac{\partial_i\tilde C_\ell\,\partial_j\tilde C_\ell}{\Sigma_\ell}$ is the single-split Fisher matrix. So $\mathbf t\sim\mathcal N(F_\theta\boldsymbol\delta,\,F_\theta)$.
>
> **Exercise 5b.** With prior $\boldsymbol\delta\sim\mathcal N(0,P)$, complete the square to show the posterior is Gaussian with
> $$\Sigma_{\rm post}=(F_\theta+P^{-1})^{-1}\ (\text{data-independent}),\qquad \text{mean}=\boldsymbol\theta_{\rm fid}+\Sigma_{\rm post}\,\mathbf t .$$
>
> **Exercise 5c.** For two Gaussians with equal covariance, $D_{\rm KL}=\tfrac12\Delta\mathbf m^{\!\top}\Sigma^{-1}\Delta\mathbf m$. Show
> $$\boxed{D_{\rm KL}=\tfrac12\,\mathbf t^{\Delta\top}\,\Sigma_{\rm post}\,\mathbf t^\Delta .}$$

So **the KL is a quadratic form in the same $\mathbf t^\Delta$ as the compressed $\chi^2$, with a different metric**:

$$
\chi^2:\ \ C_t^{-1}=\big[W\,\mathrm{diag}(V_\ell)\,W^{\!\top}\big]^{-1}
\qquad\text{vs}\qquad
\text{KL}:\ \ \Sigma_{\rm post}\approx F_\theta^{-1}=\big[W\,\mathrm{diag}(\Sigma_\ell)\,W^{\!\top}\big]^{-1}\ \ (\text{weak prior}).
$$

The only difference is $V_\ell$ (difference variance, sky cancelled) versus $\Sigma_\ell$ (single-split variance, includes cosmic variance).

> **Exercise 5d (the "wrong ruler", quantified).** Show
> $$\frac{\Sigma_\ell}{V_\ell}=\frac{(S_\ell+N_\ell)^2}{2N_\ell(N_\ell+2S_\ell)}\ \longrightarrow\ \begin{cases}S/4N & S\gg N\\[2pt] 1/2 & S\ll N\end{cases}$$
>
> In bright-mode directions the KL's yardstick is about $S/4N$ times too wide in variance, so a shift there barely registers. In noise-dominated directions the two yardsticks agree to a factor of 2.

**Checkpoints:** $\Sigma/V = 5.9\times10^3$ at $\ell=200$ ($S/N=2.4\times10^4$), 31.8 at $\ell=1000$, 0.50 at $\ell=3000$.

**Null distribution.** Under the null $\mathbf t^\Delta\sim\mathcal N(0,C_t)$, so $D_{\rm KL}=\sum_j\kappa_j\,\chi^2_1$ with $\kappa_j=\mathrm{eig}(\tfrac12\Sigma_{\rm post}C_t)$. This is a weighted $\chi^2_5$.

> **Exercise 5e.** Show $\mathbb E[D_{\rm KL}]=\sum_j\kappa_j=\tfrac12\mathrm{tr}(\Sigma_{\rm post}C_t)$.

The 95th percentile of a weighted $\chi^2$ has no closed form; it needs Imhof's method or a few lines of Monte Carlo.

**Checkpoints:**

- Null weights $\kappa=[0.0022,\,0.0487,\,0.138,\,0.187,\,0.446]$.
- Null mean 0.822, null 95th percentile 2.216.
- The SBI notebooks measure 2.58–2.78, consistent with an imperfect NPE and the 500-sample KL estimate.
- Thresholds 0.105 % (gain), 0.071 % (beam), 0.309 % (noise); σ-equivalents 24.0, 3.96, 2.93.

### Why KL beats the $\chi^2$ for noise but loses badly for gain

The weights $\kappa_j$ are very unequal. The smallest (0.002) belongs to a bright-mode, $A_s$-like direction, where the ruler is far too wide. The largest belong to noise-dominated directions.

- If a systematic lies mostly along a heavily weighted direction, the KL behaves almost like a **1-dof** test and escapes most of the $\sqrt{6.99/3.84}$ dof penalty. That's noise: 2.93σ, versus 3.53σ for $\chi^2_c$, and still above the compressed matched-filter floor of 2.62σ.
- If it lies along a lightly weighted direction, the KL is blind to it. That's gain: 24σ.

> **Exercise 5f (optional).** Argue that no statistic built from $\mathbf t^\Delta$ alone can beat row 2, whatever its metric. So the KL result must satisfy σ-equiv ≥ $1.96/\sqrt\eta$. Check that rows 4 and 5 do.

---

## 7. Step 6: prior-marginalised versions (the like-for-like comparison with SBI)

The SBI calibration loops draw $\boldsymbol\theta_i\sim$ prior in every realisation, while keeping the compressor fixed at the fiducial. The like-for-like classical version keeps $W$ and the metric fixed and lets $V_\ell(\boldsymbol\theta)$ vary, since $S_\ell(\boldsymbol\theta)$ does.

> **Exercise 6a.** With the metric fixed at $C_t^{\rm fid}$ but true null covariance $C_t(\boldsymbol\theta)$, show
> $$\mathbb E[T_c\,|\,\boldsymbol\theta]-5=\mathrm{tr}\!\big((C^{\rm fid}_t)^{-1}C_t(\boldsymbol\theta)\big)-5=\sum_\ell h_\ell\,\big(V_\ell(\boldsymbol\theta)/V^{\rm fid}_\ell-1\big).$$
> This uses the same leverage trick as Exercise 4e.
>
> **Exercise 6b (the $\sqrt{k}$ argument).** Let the typical fractional error in $V$ be $\bar\delta$. Show the mean of a $k$-dof test shifts by about $k\bar\delta$ while its null scatter is $\sqrt{2k}$. So the miscalibration in units of the noise is
> $$\frac{k\bar\delta}{\sqrt{2k}}=\bar\delta\sqrt{k/2}\qquad\Rightarrow\qquad 38.7\,\bar\delta\ \ (k=2999)\quad\text{vs}\quad 1.6\,\bar\delta\ \ (k=5).$$
>
> Compression makes the classical test **robust to not knowing the cosmology**. The full test was ruined by this, with null $q_{95}$ going from 3127 to 3657.

**Estimating $C_t$ from data instead.** Use the cross-spectrum $\hat S_\ell=\hat C^{(12)}_\ell$, which is unbiased for $S_\ell$ under the null because the two noises are independent. Set $\hat V_\ell=4N_\ell(N_\ell+2\hat S_\ell)/M_\ell$ and $\hat C_t=W\,\mathrm{diag}(\hat V_\ell)\,W^{\!\top}$. Each element of $\hat C_t$ sums over 2999 multipoles, so its noise averages down and $T_c$ stays essentially $\chi^2_5$.

**Checkpoints** (400 prior draws; $S_\ell(\boldsymbol\theta)/S^{\rm fid}_\ell$ at $\ell=200$ spans 0.75–1.25 at 5–95 %):

| | null $q_{95}$ | gain | beam | noise |
|---|---|---|---|---|
| $\chi^2_c$, $C_t$ fixed at fiducial | 11.29 | 0.0272 % (6.22σ) | 0.0501 % (2.78σ) | 0.378 % (3.59σ) |
| $\chi^2_c$, $C_t$ from data | 11.10 (mean 5.00) | 0.0266 % (6.08σ) | 0.0491 % (2.72σ) | 0.371 % (3.52σ) |
| linear KL proxy | 2.241 | 0.106 % (24.2σ) | 0.072 % (4.01σ) | 0.311 % (2.95σ) |

Marginalising over the prior moves each threshold by at most about 3 %. It also does **not** explain the gap between the KL proxy's null 95th percentile (2.24) and SBI's measured 2.58–2.78, so that gap comes from the networks themselves.

---

## 8. The full table, for reference

Threshold, with σ-equivalent $\varepsilon\sqrt F$ in brackets (lower is better):

| | gain | beam | noise |
|---|---|---|---|
| 1. Fisher / MF bound | 0.0086 % (1.96) | 0.0353 % (1.96) | 0.207 % (1.96) |
| 2. compressed MF bound | 0.0197 % (4.49) | 0.0362 % (2.01) | 0.276 % (2.62) |
| 3. compressed $\chi^2$ (fid) | 0.0265 % (6.06) | 0.0488 % (2.71) | 0.372 % (3.53) |
| 3′. compressed $\chi^2$ (prior, $C_t$ from data) | 0.0266 % (6.08) | 0.0491 % (2.72) | 0.371 % (3.52) |
| 4. linear KL proxy (fid) | 0.105 % (24.0) | 0.071 % (3.96) | 0.309 % (2.93) |
| 5. SBI / KL (measured) | 0.155 % (35.4) | 0.080 % (4.44) | 0.350 % (3.32) |
| 6. blind full-spectrum $\chi^2$ (exact) | 0.0485 % (11.1) | 0.208 % (11.5) | 1.10 % (10.5) |
| 7. $\eta=F_c/F$ | 0.19 | 0.95 | 0.56 |

**The chain from optimum to SBI** (ratios of σ-equivalents):

| step | cause | gain | beam | noise |
|---|---|---|---|---|
| 1 → 2 | compression loses information, $1/\sqrt\eta$ | ×2.29 | ×1.03 | ×1.34 |
| 2 → 3 | 5 dof instead of 1 | ×1.35 | ×1.35 | ×1.35 |
| 3 → 4 | KL metric instead of $C_t^{-1}$ | ×3.96 | ×1.46 | ×0.83 |
| 4 → 5 | NPE imperfection, KL estimation noise | ×1.48 | ×1.12 | ×1.13 |

---

## 9. Suggested order of work

1. Exercises 1–2: compression of the difference and its Gaussian distribution. This is the core new idea.
2. Exercises 3a–3c: the compressed matched filter, which reduces to the board's Fisher result.
3. Exercise 3d: $\eta$ as a squared cosine, the geometric picture. Then Exercise 3f, the gain weighting.
4. Exercises 4a–4d: the compressed $\chi^2$ and its threshold.
5. Exercises 4e–4f: why the approximation is excellent here.
6. Exercises 5a–5d: the KL as a quadratic form in $\mathbf t^\Delta$, and the ruler ratio.
7. Exercises 6a–6b: prior marginalisation and the $\sqrt k$ argument.

**Only these steps need a computer:** CAMB derivatives (to build $W$); the $\ell$-sums; three 1-D root solves ($\lambda_{\rm crit}$ for $k=5$, and the SBI-style criterion for the KL proxy); and the 95th percentile of a weighted $\chi^2$.
