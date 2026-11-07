# Method

## Question
A per-operation vibration baseline is set once, when a CNC process is qualified. On real production data, how fast does it go stale, what happens when one machine's baseline is copied to another, and does a multivariate or ML detector beat a plain RMS control chart? Stages 5 and 6 add a modern time-series model (MiniRocket) with a supervised and a label-free head, and ask whether it adds anything once the obvious shortcuts are removed: bad recordings in this dataset are trimmed excerpts, shorter than good ones and starting later in the program.

## Summary of the AI layer (stages 5–6)
A pre-declared gate decided the headline. The AI headline needed a model that, without seeing recording length, beat the best classical chart on like-for-like ranking (paired cluster-bootstrap CI above 0) and caught clearly more bad parts than the RMS chart after qualification at equal false alarms, in all 3 MiniRocket seeds. **Neither head passed, so the fallback headline ships.** An adversarial review afterwards found that the gate's duration-blind arm (a common start-anchored prefix) hid recording length but not where each recording starts: the bad files are trimmed excerpts that start a median 3.8 s later in the NC program, and that offset alone ranks bad parts with AUC 0.94. A program-aligned arm was added and the heads re-run on it (section 10). They do not pass there either.
- **Re-qualify, don't copy.** Every chart's 2019 limits go stale (16–80% false alarms on later good parts at a 5% design rate; up to 93% on aligned windows). Re-centring each chart on the first 20 confirmed-good parts after a change, and recalibrating its limit, brings every chart back to 1.5–5.5% pooled over 11 machine × periods (2.0–7.4% on aligned windows), against a nominal 4.76%. That comes from the conformal calibration, not from any model. It is an average: a single 20-part re-qualification delivered 0–29% here, and even for ideal exchangeable parts its rate is Beta(1, 20) distributed (95th percentile 13.9%).
- **Label-free ranking matches, but does not beat, the best classical chart.** On program-aligned windows, a MiniRocket nearest-neighbour memory bank that has never seen a bad part ranks bad parts with AUC 0.962–0.972 (3 seeds) against 0.960 for the best classical chart, +0.013 [−0.027, +0.052] in seed 0. On the start-anchored prefix it had reached 0.980–0.985; part of that was the start offset.
- **Recording length looks like a packaging artefact, not a cycle-time signal.** A "short cycle" rule ranks bad parts at 0.985 and the two-sided cycle-time chart catches 10 of 16 post-qualification bad parts at 5% false alarms. But no bad file has a length that is a multiple of the logger's 1,024-sample block, against 92% of good files, so these numbers most likely describe how the bad files were cut. They are not evidence that a machine's cycle time flags bad parts, and the cycle-time chart is not recommended on the strength of them.
- **The supervised head ranks best but did not hold up.** Its like-for-like AUC is 1.000 on the prefix and 0.992–1.000 on aligned windows, against 0.960–0.965; resampling whole machine × timeframe blocks, its edge is not significant (CI lower bound 0.000 or below in every seed and arm). After qualification, at equal 5% false alarms, it caught 3, 5 or 5 of 16 on the prefix and 2–3 of 16 on aligned windows (RMS: 3 of 16 aligned), so it is reported but not headlined.
- **On a new machine,** with charts copied from the other two machines and re-qualified on 20 of its parts, the label-free kNN on aligned windows caught 13/24 bad parts on M02 against 0/24 for the copied, re-centred RMS chart (exact McNemar p = 0.0002, which treats the 24 bad recordings, from 2 timeframes, as independent), but only 6/27 on M01 (p = 0.13). On the start-anchored prefix M01 had read 15/27.
- **Two grafts failed and are reported as negative results.** The forward-in-time drift label does not separate nuisance drift from defects. Only one of three autoencoder band fingerprints reproduces on aligned windows (M01 OP05 at 178–205 Hz, in 4 of 4 bad parts, where the RMS chart reads normal).

## Data
- **Bosch CNC Machining dataset**: https://github.com/boschresearch/CNC_Machining
- Tri-axial accelerometer (Bosch CISS) inside three brownfield CNC milling machines (M01–M03), 15 operations (OP00–OP14; M03 has OP00–OP13), sampled at 2 kHz.
- Six timeframes labelled Feb 2019, Aug 2019, Feb 2020, Aug 2020, Feb 2021, Aug 2021. Each recording is labelled good or bad by the dataset authors.
- **All 1,702 recordings are used** (70 bad, 4.1%). There are 44 machine × operation cells; 43 have enough good recordings to model and 1 (M03 OP03, 13 good) is skipped.
- **Licence:** data CC-BY-4.0, loader code BSD-3-Clause. Citation: Tnani, M.-A., Feil, M., Diepold, K. *Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring.* Procedia CIRP 107 (2022) 131–136. doi:10.1016/j.procir.2022.04.022
- No synthetic data in the analysis. The sine and noise signals in `tests/` are unit-test fixtures only.

## Pipeline
`pipeline/extract.py` → `pipeline/out/features.npz` → `pipeline/analyze.py` → `pipeline/diagnostics.py` → `pipeline/align.py` → `pipeline/windows.py` → `pipeline/embed.py` → `pipeline/ai_eval.py` → `public/data/results.json`. Stages 5–6 write the `ai` key; the stage 4 summary is in `public/data/diagnostics_summary.json`. Shared statistics (Wilson intervals, stratified AUC, split-conformal limits, bootstraps, McNemar) live in `pipeline/stats.py`, which both `diagnostics.py` and `ai_eval.py` import. Large caches (windows, embeddings, per-part scores) stay in the git-ignored `pipeline/out/`.

### 1. Features (`features.py`)
Each recording is linearly detrended, then per axis:
- RMS $= \sqrt{\tfrac1N\sum x_i^2}$, peak, crest factor = peak/RMS, excess kurtosis
- Welch PSD: Hann window, nperseg = 1024 (0.512 s, Δf ≈ 1.95 Hz), 50% overlap, one-sided density. Integrated into 8 log-spaced bands from 10 to 1000 Hz (Nyquist) by the trapezoid rule
- Spectral centroid $\sum f P(f) / \sum P(f)$
- Plus recording duration. That makes 40 extracted features in total.

**Monitored vector (p = 9).** For each axis: log10 RMS, log10 band power 10–100 Hz and log10 band power 100–1000 Hz. The full 40-feature vector was tried first. With n_train ≈ 16–25 recordings per cell, n < p, so every new recording has residual in the null space of the training covariance and SPE alarmed on 100% of parts. That made p < n a requirement.

### 2. Qualification baseline (static)
For each cell, the good recordings are taken from whole timeframes in time order until n ≥ 15 (in practice Feb + Aug 2019, n = 16–25). These recordings are standardised and PCA is fitted. Components are kept until they explain 90% of the variance, capped at ⌊n/3⌋ (k = 2–5 in practice).

**Hotelling T²** (Phase II, for a new observation):

$$T^2 = \sum_{j=1}^{k} t_j^2/\lambda_j,\qquad \mathrm{UCL} = \frac{k(n-1)(n+1)}{n(n-k)}\,F_{1-\alpha}(k,\,n-k)$$

**SPE / Q** (Jackson–Mudholkar 1979), with $\theta_i=\sum_{j>k}\lambda_j^i$ and $h_0 = 1-\frac{2\theta_1\theta_3}{3\theta_2^2}$:

$$Q_\alpha = \theta_1\left[\frac{z_\alpha\sqrt{2\theta_2h_0^2}}{\theta_1} + 1 + \frac{\theta_2h_0(h_0-1)}{\theta_1^2}\right]^{1/h_0}$$

α = 0.01 per chart. A recording alarms when $\max(T^2/\mathrm{UCL},\ Q/Q_\alpha) > 1$.

**Baselines on the same training set:**
- Shewhart chart on log10 of the vector-magnitude RMS, with limits at mean ± 3σ
- Isolation Forest (200 trees, seed 0) on the standardised vector, with its limit at the 99th percentile of training scores

### 3. Drift
Each static baseline is scored on the good recordings from every later timeframe; the false-alarm rate is the fraction above the limit. The **rolling re-baseline** refits the PCA monitor for each timeframe t on the good recordings of the most recent earlier timeframes (t−1, then t−2, and so on, until n ≥ 15).

### 4. Transfer
For each operation, the static monitor fitted on machine A is applied to machine B's non-training recordings. Distribution shift between the A and B training sets (both standardised by A) is measured with:
- 1-D Wasserstein distance per feature
- RBF-kernel MMD (median-heuristic bandwidth) with a 200-permutation test

### 5. Scorecard
Scores are pooled over all 43 cells. The evaluation set is every bad recording plus good recordings from timeframes after training (762 good, 70 bad, prevalence 8.4%). Each score is normalised by its own limit, so ROC-AUC, PR-AUC, false-alarm rate (score > 1 on good) and detection (score > 1 on bad) are comparable across cells.

### 6. Empirical limits (`diagnostics.py`)
The F and Jackson–Mudholkar limits assume a known covariance. With n = 16–25 they under-cover, so each limit is rescaled by a **split-conformal multiplier** taken from held-out data:

$$c = r_{(\lceil (n+1)(1-\alpha)\rceil)},\qquad r_i = \frac{T^2_{(-i)}(x_i)}{\mathrm{UCL}_{(-i)}}$$

Here $r_i$ is the leave-one-out ratio of qualification part $i$ to the limit of the model fitted without it, pooled over all cells. The limit becomes $c\cdot\mathrm{UCL}$. Under exchangeability, $P(r_{new} > c) \le \alpha$. One multiplier is fitted per statistic (T², SPE, RMS chart, Isolation Forest) with α = 0.01.

**Qualification-set screen.** Without screening, the 99th percentile of the T² ratios is 6.9× the UCL (95th percentile 0.71×, 98th 1.22×), which gives a multiplier of 12.2. The tail is not random noise. It comes from two production runs: part 000 on M02 in Feb 2019, more than 3× its UCL in 8 operations, and part 006 on M01 in Aug 2019, more than 3× in 4 operations. When one part is off across its whole program, that is more likely a machine-level event than process noise. The rule below was chosen after inspecting this tail, so it is post hoc: drop a run (machine, timeframe, part index) from every baseline when its leave-one-out T² is more than 3× the UCL in at least 3 operations. That drops 2 runs, 26 good recordings. (Runs are keyed machine × timeframe × part index; bad files are numbered separately and reuse those indices, so the key is applied to good files only. An earlier version also matched 10 bad files numbered 000, which removed them from the supervised head's training set; that was a bug and is fixed.) Screened recordings stay in every evaluation set and are scored out of sample.

**Validation of the multiplier.** The cells are split into random halves 200 times. The multiplier is fitted on one half and the leave-one-out false-alarm rate is measured on the other.

### 7. Matched-timeframe scorecard and uncertainty
The original scorecard compared all 70 bad parts (39 of them from Feb 2019, the training period) with good parts from 2020–2021. That mixes drift with detection. The **matched** scorecard keeps, for each cell, only the timeframes that contain bad parts, and compares bad parts with good parts from those same timeframes. Training-timeframe good parts get their leave-one-out score, so nothing is scored in sample. All scores are divided by the calibrated limits, then pooled.
- **Stratified AUC**: the probability that a bad part outscores a good part from the same cell and timeframe (ties count ½). Unlike the pooled AUC, it does not depend on offsets between cells.
- **Intervals**: Wilson 95% intervals on every rate, $\tilde p \pm \frac{z}{1+z^2/n}\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}$. For AUCs and paired AUC differences, a cluster bootstrap that resamples the 43 cells with replacement, 2,000 reps.

### 8. Root cause of the drift
- **Gain vs shape.** Per axis, gain $g_a = \log_{10}\mathrm{RMS}_a$. Shape $s_{a,b} = \log_{10}P_{a,b} - 2g_a$ for the 10–100 Hz and 100–1000 Hz bands. Band power scales with the square of a sensitivity change, so $s$ does not move with a sensor or mount gain change. It moves with real changes in spectral content (tool wear, runout, fixture compliance). T² monitors are fitted on gain only (p = 3), shape only (p = 6) and both, each with its own conformal multiplier.
- **Machine-level vs per-operation.** For each cell and later timeframe, the shift of every one of the 40 features is $z = (\bar x_{later} - \bar x_{train})/s_{train}$. For each machine × timeframe, the median across operations and the number of operations that move in the median's direction give a two-sided sign test (binomial, p = ½), Bonferroni-corrected over 40 features. A consistent shift across all operations at once points to one machine-level event. A scattered pattern points to per-operation wear.
- **Step, gradual or excursion.** Per machine, take the median |z| across operations and features for each later timeframe. **Step**: the first later timeframe already carries at least 70% of the largest shift. **Excursion**: the shift peaks, then falls back below 70% of the peak. **Gradual**: neither.
- **PSD overlay.** The mean log Welch PSD of training against the later timeframe, for the cell that moves most on the top-ranked band feature.

### 9. Ramp recipe
- **Copy cost without drift.** Machine A's monitor (calibrated T²) is scored on machine B's qualification-period good parts and compared with B's own leave-one-out scores on the same parts. Both sets come from the same timeframes, so no drift is mixed in.
- **Adaptation curve.** A's loadings $P$ and eigenvalues $\lambda$ are kept. The mean, and optionally the per-feature sd, is re-estimated from N good parts on B (N = 3, 5, 8, 10, 15), and the calibrated limit $c\cdot\mathrm{UCL}_A$ is applied. False alarms are measured on B's remaining qualification-period good parts. Detection is measured on B's bad parts from the same timeframes. Two ways of choosing the N parts are tested: the first N in recording order, and 500 random draws of N, which give the 95% band.
- **Re-baseline after a step.** Each cell's own static monitor is re-centred on the first N good parts of its first later timeframe, then scored on the rest of that timeframe and on later timeframes.
- **Operational units.** Alarms per 1,000 good parts = 1000 × false-alarm rate. Engineer-hours = alarms × triage minutes / 60, for 5, 15 and 30 minutes per alarm. These are a scenario on measured rates; the triage minutes are assumptions.

### 10. Raw windows and the duration control (`windows.py`)
Each recording is cut into K = 24 windows of L = 1,024 samples (0.512 s, 3 axes), with the mean removed per window and axis. There are two arms:
- **full.** Window starts are spread evenly over each recording's own length, the scheme all four prototypes used. A shorter recording therefore samples different parts of the toolpath.
- **crop (primary).** Each machine × operation cell gets a common prefix $T_c = \lfloor 0.45 \times \text{median length of the cell's qualification good parts} \rfloor$ (9.2–60.4 s). Every recording of the cell uses identical window starts, linspace$(0, T_c - L, 24)$. Recording length then carries no information. The shortest recording in the data is 0.494 × its cell's median, so every recording covers its prefix; this is asserted.
- **align (added after the first runs; `align.py`).** The crop hides length but assumes every recording starts at the same point of the program. It does not hold for the bad files. Each recording's envelope (log10 RMS of the mean-removed vector magnitude in 50 ms frames) is slid over its cell's template, the median envelope of the cell's qualification good parts, and the lag $\ell_i$ with the highest Pearson correlation over the recording's first $W$ frames is its start offset ($W$ = min(its length, half the template); lags from −2 s). Good parts sit at a median 0.0 s (5–95%: −0.4 to +2.0 s); bad parts at a median 3.8 s (5–95%: +1.0 to +16.4 s), with 66 of 70 more than 1 s late, many at 2.7–3.0 s. The aligned arm takes the program window every recording of the cell covers, $[A_c, A_c + T_c)$ with $A_c = \max(0, \max_i \ell_i)$ and $T_c = \min(\min_i(\ell_i + n_i) - A_c,\ T_c^{crop})$, and cuts 24 windows at identical program times. Lags are estimated without labels and the window placement is label-blind, but it does use every recording of the cell, bad ones included, to decide where the common window starts. Aligned lengths equal the crop lengths in 43 of 44 cells (M01 OP04: 27.0 s against 30.9 s). A unit test recovers known offsets exactly and checks that trimmed excerpts get identical program windows.

Per window the script also stores log10 band power on 20 attribution bands per axis. These are the 8 Welch bands of `features.py`, with each band above 100 Hz split into 4 log sub-bands, and every band holds at least 4 FFT bins. Per recording it stores the vector RMS and the repo's 40 features computed on the prefix; these feed each cut arm's RMS and T²+SPE charts. Rows are keyed by path relative to the data directory, because good and bad files share basenames: 61 of the 70 bad recordings have a good file with the same name.

### 11. MiniRocket embedding (`embed.py`)
MiniRocket (Dempster, Schmidt and Webb, KDD 2021; sktime `MiniRocketMultivariate`, 10,000 kernels giving 9,996 features) convolves a window with fixed length-9 kernels $w \in \{-1, 2\}^9$ at several dilations $d$ over random channel subsets. Each feature is the proportion of positive values (PPV) after subtracting a bias $b$:

$$\mathrm{PPV}(x; w, d, b) = \frac{1}{T}\sum_{t} \mathbb{1}\big[(w \ast_d x)_t - b > 0\big]$$

The biases are quantiles of the convolution outputs on 4,000 fitting windows.

**Cross-fit.** For each target machine M, the biases, a standard scaler and a 64-dimension PCA are fitted on windows of 2019 good parts from the other two machines only. No bad part and no part of M is used, and M's parts are only ever scored in fold M; a test asserts this on every cached fold. Head A uses the mean PPV over a recording's 24 windows. Head B uses the standardised PCA-64 vector of each window (PCA keeps 93–94% of the variance). Every arm and fold is run with three seeds (0, 1, 2).

### 12. Heads and baselines (`detectors.py`)
Every score is out of sample for the part it scores. A cell's reference set is its qualification good parts (the repo's rule, minus the 2 screened runs), and reference parts are scored leave-one-out. Five cells had too few 2019 good parts, so under the repo's n ≥ 15 rule their qualification runs into 2020–21: M01 OP06, OP11 and OP14, and M02 OP00 and OP04. "Later" parts are therefore defined per cell, as parts made after that cell's last reference timeframe. That gives 762 later good parts and 16 later bad parts.
- **RMS chart:** $|\log_{10}\mathrm{RMS} - \mu| / 3\sigma$.
- **Cycle-time chart:** the same chart on $\log_{10}$ recording length. It uses no vibration at all.
- **T²+SPE:** the repo's PCA monitor, $\max(T^2/c_T\mathrm{UCL},\ Q/c_Q Q_\alpha)$, with pooled leave-one-out conformal multipliers ($c_T$ = 1.39, $c_Q$ = 7.67, as in stage 4).
- **Head A, supervised dual ridge.** The target is $y \in \{-1, +1\}$, with class-balanced weights $w_i = n / (2 n_{y_i})$ and a linear kernel on the standardised mean-PPV embedding, $K = ZZ^\top/p + 1$:

$$a = (K_{SS} + \alpha W^{-1})^{-1} y_S, \qquad \text{score}(x) = K_{xS}\, a$$

  This equals the primal weighted ridge on $[Z/\sqrt p, 1]$; a unit test checks it. Standardisation and the penalty $\alpha \in \{10^{-3}, 10^{-2}, 10^{-1}, 1\}$ are chosen on training rows only, by leave-one-group-out CV over machine × timeframe groups. **Forward in time:** 2020–21 parts are scored by a model trained on 2019 labels only, and 2019 parts are scored out of fold by machine × timeframe.
- **Head B, label-free kNN memory bank (PatchCore-style).** The bank is the cell's qualification good parts. For each of a part's 24 windows, take the distance to the nearest window of any other bank part. The score is the mean of those distances, divided by the bank's median leave-one-out score:

$$s(q) = \frac{1}{K}\sum_{k=1}^{K} \min_{j \in B,\ j \ne q}\ \min_{v} \lVert e_{q,k} - e_{j,v} \rVert \Big/ \operatorname{median}_{j \in B}\, s_{\mathrm{LOO}}(j)$$

  It ignores where in the cycle a window sits and needs no bad parts. It re-qualifies by appending confirmed-good parts to the bank, with no retraining.
- **Fused rule** (declared in advance): $\max(u_{\text{head}}, u_{\text{cycle}})$ on limit-normalised scores $u = (s - \operatorname{median}_m)/(L_m - \operatorname{median}_m)$, where $u > 1$ exactly when the score exceeds machine m's limit. The fused score gets its own conformal limit.

### 13. Split-conformal limits (`stats.py`)
For each machine m, the limit is set on the reference parts' out-of-sample scores $s_1, \dots, s_n$:

$$k = \lceil (n+1)(1-\alpha) \rceil, \qquad L_m = \begin{cases} s_{(k)} & k \le n \\ +\infty & k > n \end{cases}, \qquad P(S_{new} > L_m) = 1 - \frac{k}{n+1} \le \alpha$$

The guarantee holds for continuous exchangeable scores. A finite limit needs n ≥ 19 at α = 0.05, or n ≥ 99 at α = 0.01. Below that, the tightest guaranteed rule is the **max-of-N** limit, with $P(S_{new} > \max) = 1/(N+1)$.

**What the guarantee does and does not promise.**
- It is marginal: an average over calibration sets. One calibration set's own false-alarm rate is $1 - F(s_{(k)}) \sim \mathrm{Beta}(n+1-k,\ k)$. For N = 20 that is Beta(1, 20): mean 4.76%, median 3.4%, 5–95% range 0.3–13.9%. For N = 10 (max-of-N) it is Beta(1, 10): mean 9.1%, 95th percentile 25.9%. So any one re-qualification can sit well above the design rate, and only the pooled rate over many re-qualifications should match it. `stats.conformal_far_quantiles` computes this and a simulation test checks it.
- It is exact only for split conformal, where the calibration scores come from a fixed model. The re-qualified charts score their N calibration parts leave-one-out (each part's score uses a centre or bank estimated from the other N − 1), which is not split conformal. A simulation of the hierarchical re-centring with exchangeable Gaussian parts gives about 4.9% (15,000 trials, ±0.05%) instead of 4.76%. That is close, but 4.76% is nominal for those charts, not exact. The kNN bank errs the other way (its test parts see one more bank part than its calibration parts).
- It needs the calibration parts to be exchangeable with the parts scored afterwards. The first 20 parts after a change are not a random sample of the period (section 15).

**Worked by hand, α = 0.05:**
- N = 20: k = ⌈21 × 0.95⌉ = ⌈19.95⌉ = 20. The limit is the largest of the 20 scores and the false-alarm rate is exactly 1/21 = 4.76%.
- N = 19: k = 19, giving 1/20 = 5%.
- N = 18: k = ⌈18.05⌉ = 19 > 18, so there is no finite conformal limit. Max-of-18 gives 1/19 = 5.3%.
- N = 10: max-of-N gives 1/11 = 9.1%.
- N = 5: max-of-N gives 1/6 = 16.7%.

The rank is computed in exact integer arithmetic. Static limits use α = 0.05, with α = 0.01 reported alongside.

### 14. The gate (`ai_eval.py`)
The gate was fixed in code before the first evaluation run. A head passes only if both conditions hold for all 3 seeds in the crop arm:
- **A.** Its like-for-like AUC (machine × operation × timeframe strata) minus the best classical chart has a paired cluster-bootstrap 95% CI above 0, from 2,000 resamples of whole strata. The best classical chart is the replicate-wise maximum of RMS and T²+SPE, each on the prefix and on the whole recording.
- **B.** On parts made after their cell's qualification, every chart is thresholded to flag 5% of the later good parts. The head's detection minus the RMS chart's (best of prefix and whole recording) must have a paired bootstrap 95% CI above 0.

The AI headline ships only if a head passes. Otherwise the label-free and re-qualification headline ships. Both arms are reported either way.

After the start-offset shortcut was found, the rule was tightened, not loosened: a head must now pass in the crop arm and in the aligned arm. The full arm stays reference only.

### 15. Re-qualification and new-machine tests
- **Re-qualification curve.** The test uses each machine × later timeframe with at least 25 good parts (55 for N = 50). From it, N ∈ {0, 5, 10, 19, 20, 50} confirmed-good parts are taken, either the first N in production order (part index, then operation) or in 50 random draws.
  - Location charts and the ridge are re-centred hierarchically. The offsets are $o_i = x_i - \mu_{c(i)}$, the machine shift is $\delta = \bar o$, and the per-operation shift is $s_c = \delta + w_c(\bar o_c - \delta)$ with $w_c = n_c/(n_c + 5)$. Operations without new parts use $\delta$.
  - T²+SPE does the same on its 9-feature centre. The kNN appends the parts to its bank.
  - The limit is then set on the N parts' leave-one-out scores: split conformal from N = 19, max-of-N below that.
  - As a fairness check, each classical chart is also re-qualified by recalibrating its limit only.
  - Detection is compared at each chart's own re-qualified limit, on the same bad parts, with a paired bootstrap over bad parts and an exact McNemar test.
- **New machine (leave one machine out).** Charts are copied from the other two machines: a pooled bank of the same operation, and a ridge trained on their labels. They are re-qualified on N ∈ {0, 5, 10, 20} good parts of the target machine as above, then scored on the target's qualification-period parts. This test uses seed 0.

### 16. Drift subspace and per-alarm label (`drift.py`)
- **Band vector.** Each part's band vector (log10 mean power over its windows, 60 values) is z-scored against its cell's reference with a shrunk SD, $s_c^2 = (\mathrm{df}_c s_c^2 + \nu s_{glob}^2)/(\mathrm{df}_c + \nu)$ with ν = 10, and clipped at ±8.
- **Drift subspace.** For period t, the subspace $U_t$ is the top 3 right-singular vectors of benign shifts seen **before t only**: each cell's mean good-part z in each earlier period (with at least 3 parts), plus the qualification-period machine-to-machine offsets. Only cells whose qualification ended before t contribute. Code asserts that every row used comes from an earlier period and is a good part.
- **Per-alarm label.** share $= \lVert U_t^\top z \rVert^2 / \lVert z \rVert^2$; a random direction gives 3/60 = 0.05. A share of at least 0.5 reads "nuisance drift: re-baseline", otherwise "out of window: investigate". The cut was declared in advance.

### 17. Attribution (no neural network)
Each alarm gets:
- the part's top 3 (axis, band) |z|, unclipped;
- for the kNN, the 3 windows that drove the score, with each one's time in the cycle and the bank part and window it matched;
- the band-power difference between each of those windows and its match, $10(\log_{10}P_q - \log_{10}P_{\text{match}})$ dB.

The autoencoder prototype reported three band fingerprints: M01 OP05 at 200–213 Hz on z/y, M02 OP12 at 701–720 Hz on z, and M01 OP01 at 720–760 Hz on x/y. They are re-checked here as hypotheses, at the coarser resolution of the attribution bands.

## Validation
| Check | Expected | Result |
|---|---|---|
| Sine of amplitude A: RMS = A/√2 | within 0.1% | pass |
| Parseval: ∫Welch PSD df = variance | within 2% | pass |
| T² UCL hand calculation, p = 3, n = 20, α = 0.01: 3·19·21/(20·17) · F₀.₉₉(3,17) = 3.5206 · 5.185 = **18.25** | 18.25 | code 18.25, pass |
| JM Q limit with one residual eigenvalue ≈ λ·χ²₀.₉₉(1) | within 10% | pass |
| T² empirical false-alarm rate on Gaussian data, n = 40 | ≈ 1% | < 3.5%, pass |
| Wilson interval, 5/100 (textbook) | [0.0215, 0.1118] | pass |
| Sign test, 13 of 15: 2·(C(15,13)+C(15,14)+1)/2¹⁵ = 242/32768 | 0.00739 | pass |
| Conformal rank on 1…99, α = 0.01: ⌈100·0.99⌉ = 99 | 99 | pass; coverage on heavy-tailed t² data within ±0.5% of 1% |
| Shape features invariant to a 3× gain; gain features shift by log₁₀3 | exact | pass |
| Re-centring removes a pure offset exactly | exact | pass |
| **Leave-one-out false alarms on real qualification parts, theoretical limits** (857 fits, unscreened) | T² 1% | **T² 3.3% [2.3, 4.7], SPE 10.4%, combined 11.6% [9.6, 13.9]** |
| Leave-one-out after screening + calibration (831 fits; in sample for the multiplier, so T² = 7 of 831 follows from the rank) | 1% | T² 0.8% [0.4, 1.7], combined 1.3% [0.7, 2.4] |
| **Cross-validated calibration** (fit on half the cells, test on the other half, 200 splits) | ≈ 1% | **T² 0.96% (95% of splits 0–2.5%)**, combined 1.5% |
| Conformal limit by hand: 20 scores, α = 0.05 → k = ⌈21·0.95⌉ = 20, false alarms 1/21; 50 scores → k = 49, 2/51 | exact | pass |
| No finite conformal limit below N = 19 (α = 0.05) or N = 99 (α = 0.01); the rank uses exact integer arithmetic | +∞ | pass |
| Conformal false-alarm rate by simulation (40,000 trials; Gaussian and t₃; N = 19, 20, 50, 120) | ≤ α + 3 SE and = 1 − k/(N+1) | pass |
| Max-of-N false-alarm rate by simulation (N = 5, 10) | 1/(N+1) | pass |
| One calibration set's false-alarm rate by simulation (N = 20, 40,000 sets): mean, median, 95th percentile | Beta(1, 20): 4.76%, 3.4%, 13.9% | pass |
| Re-centred, leave-one-out re-qualification (hierarchical shift on 20 parts over 10 operations, 3,000 trials × 40 test parts) | between 4.76% and 5% | about 4.9%, pass |
| Cluster bootstrap over blocks of strata: replicates only take block-mixture values; strata must nest in blocks | {0, 2/3, 1} | pass |
| Clustered detection bootstrap: point estimate unchanged; paired; draws with no bad cluster dropped | exact | pass |
| Clustered rate interval over groups | hand cases | pass |
| Stratified AUC: 2-stratum hand count (0.75); unchanged when a constant is added to each stratum; ties count ½ | exact | pass |
| Cluster bootstrap resamples whole strata: replicates only take stratum-mixture values | {0.5, 0.75, 1} | pass |
| Paired bootstrap of identical scores | exactly 0 | pass |
| Dual ridge against primal weighted ridge | 1e-8 | pass |
| Ridge: scores of test rows unchanged when other, very different test rows are added (scaler, penalty and fit use training rows only) | identical | pass |
| kNN: nearest-window distances against brute force; a part never matches itself; an exact duplicate is matched to its twin | exact | pass |
| Hierarchical re-centring removes a pure offset exactly, shrinks per-operation corrections and falls back to the machine offset | exact | pass |
| Duration crop: window starts depend only on the cell's prefix; truncating a recording to 85% leaves its prefix windows and kNN score unchanged, while the full arm changes | exact | pass |
| Drift subspace for period t uses good parts from earlier periods only; a random direction's share is K/p | exact; 0.050 | pass |
| Start offset: envelope alignment recovers known offsets of 0, 0.5, 3 and 11 s exactly; trimmed excerpts of one program get identical aligned windows | exact | pass |
| Real data: aligned window starts plus start offset are identical within every cell; no bad file is screened by a good run's key | exact | pass |
| Real data: RMS in the window cache against features.npz | 1e-6 | max relative error 8×10⁻⁸, pass |
| Real data: every cached embedding fold was fitted without its target machine, bad parts or post-2019 parts (27 folds: 3 arms × 3 seeds × 3 machines) | none | pass |
| Real data: reimplemented RMS and T²+SPE charts against the repo's like-for-like (stratified) AUCs | 0.884 / 0.960 | 0.8838 / 0.9596, pass |
| **Real data: split-conformal limits inside the qualification period** (random half splits per machine, every chart, every arm and seed) | 5% / 1% | **4.4–4.8% / 0.7–0.8%** (4.4–4.7% in the crop and aligned arms) |
| **Real data: textbook 3σ Shewhart limit**, leave-one-out on 831 qualification parts | 0.27% | **2.2% [1.4, 3.4]** (8× the nominal rate) |

## Results (measured; reproduced by `pipeline/analyze.py` + `pipeline/diagnostics.py`)
Multipliers after screening: T² 1.39, SPE 7.67, RMS chart 1.23, Isolation Forest 1.05. SPE needs a 7.7× correction, so its theoretical limit is not usable at this n.

**1. The baseline goes stale (calibrated limits).** With T² limits calibrated to a 1% target (0.96% on held-out cells, 0.8% on the parts used to fit the multiplier), the 2019 baseline flags **41.5% [38.0, 45.0] of the 762 later good parts**: Feb 2020 38.8%, Aug 2020 55.4%, Feb 2021 40.8%, Aug 2021 37.8%. The combined T²/SPE flags 58.1% [54.6, 61.6]. The earlier 76–97% headline used the uncalibrated combined limit and overstated the drift.

**2. Detector scorecard, like with like.** Matched timeframes: 70 bad, 344 good, prevalence 16.9%.

| Detector | ROC-AUC [95% cluster CI] | Stratified AUC | PR-AUC | False alarms at limit | Detection at limit |
|---|---|---|---|---|---|
| RMS Shewhart | 0.919 [0.841, 0.971] | 0.884 | 0.832 | 2.6% [1.4, 4.9] | 72.9% [61.5, 81.9] |
| PCA T² | 0.935 [0.893, 0.971] | 0.957 | 0.790 | 4.9% [3.1, 7.8] | 77.1% [66.0, 85.4] |
| PCA T² + SPE | **0.954** [0.927, 0.977] | **0.960** | 0.798 | 7.8% [5.5, 11.2] | **90.0%** [80.8, 95.1] |
| Isolation Forest | 0.857 [0.800, 0.907] | 0.894 | 0.607 | 1.7% [0.8, 3.8] | 28.6% [19.3, 40.1] |

RMS − T²/SPE AUC difference: −0.034 [−0.100, +0.007]. **The earlier claim that "plain SPC beats PCA" does not survive** a like-for-like comparison: the multivariate chart is at least as good, but the difference is not significant at 95%. On the drift-inclusive set (all bad parts against later good parts), RMS still wins (0.768 against 0.705; difference +0.063 [+0.007, +0.127]). The likely reason is that it reacts least to drift (it flags 22.2% of later good parts against 41.5% for T²), not that it separates bad parts better. The Isolation Forest is the weakest in both cases.

**3. Why it goes stale.**
- *Spectral shape, not sensor gain.* Later-good false alarms: gain-only monitor (3 log-RMS values) 24.3% [21.4, 27.4], shape-only 41.9% [38.4, 45.4], gain + shape 42.9%. More of the drift is in how energy is distributed across frequency than in overall level. A pure gain change cannot produce a shape shift, but a sensor reseat that changes the mount resonance could, so this does not rule out the sensor. The shape monitor also has 6 inputs to the gain monitor's 3, which gives it more directions in which to alarm, so the comparison is indicative. The gain-only monitor drifts least, at the cost of some detection (68.6% of all bad parts against 80.0%; matched AUC 0.900 against 0.929).
- *Machine-level events.* Several timeframes show (nearly) every operation on one machine moving the same way at once. The p-values below are nominal. Bonferroni over the 40 features multiplies them by 40 (0.0098 for 13 of 13, 0.0195 for 12 of 12). Across all 10 machine × timeframe groups (400 tests) no single result survives a full Bonferroni correction; with 13 operations the smallest possible p is 2.4×10⁻⁴. The z-shifts are measured in units of the training sd from 16–25 parts, so a narrow training spread can make them very large.
  - M03, Feb 2020: x-axis 100–178 Hz power rose by a median 5.2σ in 13 of 13 operations (sign test p = 2.4×10⁻⁴). M03 otherwise has the smallest overall shift of the three machines (median |z| 0.7).
  - M01, Feb 2021: z-axis 18–100 Hz bands shifted in 12 of 12 operations (p = 4.9×10⁻⁴), with 10 features significant after Bonferroni.
  - M02, Aug 2020: x-axis 178–316 Hz rose by a median 39σ in 12 of 13 operations (p = 3.4×10⁻³). This feature does not pass the 40-feature Bonferroni threshold (1.25×10⁻³), although 8 other features in the same machine × timeframe do.
  These look like one-machine events (maintenance, spindle or fixture work). If maintenance records confirm that, each calls for one re-baseline per machine rather than up to 15 separate investigations. The dataset has no maintenance log, so the cause is inferred, not confirmed.
- *Step or excursion.* M01 and M03 behave as steps: the first later timeframe already carries 76% and 81% of the largest shift. M02 is an excursion: its median shift rises from 2.6σ to 4.6σ in Aug 2020, then falls back to 2.0σ. The step and excursion labels are a fixed 70% rule applied to only 2–4 timeframes per machine, so they describe what was seen and are not a test.
- *Which band.* Ranked by median |z| over all cells and later timeframes, the top features are z-axis 18–32 Hz (4.7σ), 32–56 Hz (3.7σ) and 56–100 Hz (3.1σ, consistently down). In the most-shifted cell (M02 OP09, Feb 2021 against Aug 2019), mean z-axis power in 10–100 Hz is 0.92 decades lower (about 8× less), while 100–1000 Hz moves by only +0.08 decades.

**4. Ramp recipe (copying a baseline to a new machine, measured without drift).**

| Baseline on the new machine | False alarms on good parts | Detection, in-period bad |
|---|---|---|
| Copied unchanged (23%–86% by machine pair) | 68.9% [66.6, 71.1] | 91.7% |
| Copied loadings, re-centred on 10 parts | 21.9% [19.1, 25.0] | 72.6% |
| Copied loadings, re-centred + re-scaled on the first 10 parts in recording order | **7.6% [5.9, 9.7]** | 67.9% |
| Copied loadings, re-centred + re-scaled on 10 parts drawn at random from the qualification period (mean of 500 draws; brackets give the 2.5–97.5% range over draws) | **3.0% [1.7, 4.5]** | 60.5% [54.8, 67.9] |
| Full per-machine qualification (16–25 parts) | 0.8% [0.4, 1.3] | 69.0% [58.5, 77.9] |

The copied monitor's high detection is not a strength. It alarms on almost everything. Re-centring alone is not enough because each machine also has its own spread. Each cell's qualification set holds 1–5 Feb 2019 good parts and about 18 Aug 2019 good parts, so the first 10 in recording order are a few Feb 2019 parts plus the first Aug 2019 parts, while random draws sample the whole qualification period. The first 10 give 7.6% false alarms and random draws 3.0%. The random-draw figure assumes the reference parts represent the whole qualification period, which a real ramp cannot guarantee, so 7.6% is the more realistic estimate for a ramp. The curve is not monotonic in N: at N = 15 only 38 of 82 machine pairs have enough parts left to test, and the first-15 figure is 12.2%. The copied-baseline interval counts each new-machine part twice (once per lead machine), so it is narrower than it should be. After a step, re-centring the machine's own monitor on its first 5 parts cuts false alarms in that timeframe from 39.9% to 15.6%, but in later timeframes from only 23.4% to 21.3%. Each new step needs its own re-baseline.

Scenario at 15 min of triage per alarm: a copied baseline costs 689 alarms and about 172 engineer-hours per 1,000 good parts. Re-centring alone on 10 parts costs 219 alarms (55 h). Re-centring and re-scaling on the first 10 parts costs 76 alarms (19 h), and on 10 random qualification parts 30 alarms (7.4 h). Full qualification costs 8 alarms (1.9 h). The 15-minute triage time is an assumption; the false-alarm rates are measured.

## Results of the AI layer (measured; `pipeline/ai_eval.py`, about 2 minutes after the embeddings)
Unless a row says otherwise, numbers are from the crop arm, seed 0, with the aligned arm alongside where it changes the reading. The full arm and the other seeds are in `results.json → ai`. All numbers here are from the rerun after the screening-key fix (section 6), which changed only the supervised ridge and the fused ridge rule.

**5. The duration and start-offset giveaways.** Every bad recording is shorter than its cell's qualification median: the ratio runs from 0.49 to 0.98, with a median of 0.84. Good recordings sit between 0.86 and 1.10 (1st–99th percentile). A "shorter cycle" rule alone separates bad from good parts of the same machine, operation and timeframe with AUC 0.985; the two-sided cycle-time chart reaches 0.929. The bad files look like trimmed excerpts:
- none of the 70 has a length that is a multiple of the logger's 1,024-sample block, against 92% of good files (all 39 Feb 2019 bad lengths end in 1, all 17 bad lengths from 2021 are round thousands);
- they start a median 3.8 s later in the program than the qualification template (section 10), against 0.0 s for good parts, and that start offset alone gives a like-for-like AUC of 0.942.

So any model that sees whole recordings can win by reading length, and any model that sees a start-anchored prefix can still win by reading which toolpath phase the prefix covers. The crop arm removes the first cue and the aligned arm both. Head scores barely track either cue (|Spearman ρ| ≤ 0.23 with the duration ratio and ≤ 0.21 with the start offset, in every arm and seed), but the kNN's like-for-like AUC drops from 0.980 on the prefix to 0.972 aligned, and its new-machine detection on M01 from 15/27 to 6/27. The cycle-time chart's results below measure file cutting, not machine cycle time.

**6. Gate result: the AI headline does not ship.**

| Head | Arm | Seed | Like-for-like AUC | Δ vs best classical | A | Later bad parts caught at 5% false alarms (RMS) | B | Pass |
|---|---|---|---|---|---|---|---|---|
| Ridge (supervised) | crop | 0 | 1.000 | +0.035 [+0.002, +0.083] | ✓ | 3 of 16 (1), Δ CI [0.00, +0.50] | ✗ | ✗ |
| | | 1 | 1.000 | +0.035 [+0.002, +0.083] | ✓ | 5 of 16 (1), Δ CI [+0.06, +0.69] | ✓ | ✓ |
| | | 2 | 1.000 | +0.035 [+0.002, +0.083] | ✓ | 5 of 16 (1), Δ CI [0.00, +0.75] | ✗ | ✗ |
| | align | 0 | 1.000 | +0.040 [+0.006, +0.085] | ✓ | 3 of 16 (3), Δ CI [−0.25, +0.31] | ✗ | ✗ |
| | | 1 | 0.995 | +0.035 [0.000, +0.080] | ✗ | 2 of 16 (3), Δ CI [−0.25, +0.31] | ✗ | ✗ |
| | | 2 | 0.992 | +0.033 [+0.005, +0.068] | ✓ | 2 of 16 (3), Δ CI [−0.25, +0.44] | ✗ | ✗ |
| kNN (label-free) | crop | 0 | 0.980 | +0.015 [−0.010, +0.052] | ✗ | 6 of 16 (1), Δ CI [+0.06, +0.56] | ✓ | ✗ |
| | | 1 | 0.985 | +0.020 [−0.012, +0.062] | ✗ | 5 of 16 (1), Δ CI [+0.06, +0.50] | ✓ | ✗ |
| | | 2 | 0.982 | +0.018 [−0.006, +0.052] | ✗ | 5 of 16 (1), Δ CI [+0.06, +0.50] | ✓ | ✗ |
| | align | 0 | 0.972 | +0.013 [−0.027, +0.052] | ✗ | 4 of 16 (3), Δ CI [0.00, +0.25] | ✗ | ✗ |
| | | 1 | 0.962 | +0.003 [−0.060, +0.053] | ✗ | 4 of 16 (3), Δ CI [0.00, +0.19] | ✗ | ✗ |
| | | 2 | 0.967 | +0.008 [−0.043, +0.050] | ✗ | 5 of 16 (3), Δ CI [0.00, +0.31] | ✗ | ✗ |

The best classical chart is T²+SPE: 0.965 on the prefix in the crop arm, and 0.960 on the whole recording in the aligned arm (aligned-prefix T²+SPE reaches only 0.927). The ridge ranks better than every classical chart, but its post-qualification detection depends on the seed and, on aligned windows, is no better than RMS. The kNN's ranking edge is within noise in both arms, and on aligned windows its post-qualification edge over RMS disappears too. In the full arm (reference only), neither head passes either: the ridge's A interval touches 0, at [0.000, +0.084]. Before the screening-key fix the crop ridge caught 3, 4 and 6 of 16 and passed in seed 2 instead of seed 1; the decision is the same.

**Cluster sensitivity (statistics review, not part of the pre-declared rule).** The gate's intervals resample machine × operation × timeframe strata (condition A) and single parts (condition B). But the operations of one machine in one timeframe share a production episode: the 70 bad files sit in 11 machine × timeframe groups, numbered 000–002 across 1–10 operations each, and two groups (M01 and M02, Feb 2019) hold 39 of the 68 like-for-like bad parts. The 16 later bad parts come from 6 groups. Re-running both conditions with whole machine × timeframe blocks resampled (11 blocks; `results.json → ai.decision.sensitivity_machine_x_tf_clusters`):
- Ridge, condition A: +0.035 [0.000, +0.077] crop (all seeds); aligned +0.040 [0.000, +0.091], +0.035 [−0.006, +0.083], +0.033 [0.000, +0.076]. No longer above 0.
- kNN, condition B (crop): 6, 5, 5 of 16 against 1 of 16, Δ CI [0.00, +0.56], [0.00, +0.55], [0.00, +0.50]. No longer above 0.
- No head passes either condition in any arm or seed. The ridge's ranking edge and the kNN's crop-arm detection edge depend on treating strata or parts from one episode as independent.

**7. Like-for-like scorecard** (68 bad, 344 good, 46 machine × operation × timeframe strata; 95% cluster-bootstrap CIs):

| Detector | AUC | Δ vs best classical | PR-AUC | False alarms at the α = 0.05 limit | Detection at the limit |
|---|---|---|---|---|---|
| RMS chart, prefix | 0.861 [0.751, 0.951] | | 0.825 | 6.1% [4.0, 9.2] | 56/68 |
| RMS chart, whole recording | 0.884 [0.771, 0.966] | | 0.821 | 7.6% [5.2, 10.8] | 53/68 |
| T²+SPE, prefix | 0.965 [0.917, 0.998] | | 0.790 | 11.3% [8.4, 15.1] | 66/68 |
| T²+SPE, whole recording | 0.960 [0.911, 0.994] | | 0.795 | 12.2% [9.2, 16.1] | 63/68 |
| Cycle-time chart | 0.929 [0.849, 0.983] | −0.035 [−0.128, +0.039] | 0.851 | 7.0% [4.7, 10.2] | 59/68 |
| MiniRocket ridge (supervised) | **1.000** [1.000, 1.000] | **+0.035 [+0.002, +0.083]** | **0.983** | 3.2% [1.8, 5.6] | 67/68 |
| MiniRocket kNN (label-free) | 0.980 [0.955, 0.998] | +0.015 [−0.010, +0.052] | 0.912 | 14.2% [10.9, 18.3] | 66/68 |
| Fused ridge + cycle time | 0.992 [0.979, 1.000] | +0.028 [−0.008, +0.077] | 0.928 | 6.4% [4.3, 9.5] | 66/68 |
| Fused kNN + cycle time | 0.987 [0.972, 0.998] | +0.023 [−0.011, +0.069] | 0.913 | 10.8% [7.9, 14.5] | 68/68 |

Notes on the scorecard:
- The like-for-like set holds only 396 bad/good pairs, so AUC moves in steps of 0.0025, and most strata hold one bad part.
- The ridge's [1.000, 1.000] is a degenerate percentile interval: a scorer with no misordered pair keeps AUC 1 in every resample. It says nothing about uncertainty; with 68 bad parts in 11 machine × timeframe blocks, an AUC of 1.000 is compatible with a true AUC a few points lower.
- The part bootstrap gives a similar interval for the kNN's Δ: [−0.018, +0.061]. Over the 3 seeds the kNN's AUC runs from 0.980 to 0.985.
- The false-alarm column includes later-period good parts, which is why it sits above 5%.
- With wider machine × timeframe strata (70 bad, 1,255 good), the fused kNN + cycle-time rule does beat the best classical chart: 0.983 vs 0.936, Δ [+0.012, +0.082]. That was not the pre-declared comparison, and it leans on the cycle-time chart, which reads file cutting (section 5).
- **Aligned arm** (same parts and strata): RMS on the aligned window 0.838, T²+SPE aligned 0.927, whole-recording RMS 0.884 and T²+SPE 0.960, cycle time 0.929, ridge 1.000 (Δ vs best classical +0.040 [+0.006, +0.085]), kNN 0.972 (+0.013 [−0.027, +0.052]; part bootstrap [−0.033, +0.056]), fused kNN + cycle time 0.987. False alarms at the limit: ridge 2.3%, kNN 22.1%.

**8. Every chart's 2019 limit goes stale.** These are static per-machine conformal limits, which hold their level inside the qualification period (4.4–4.7% at α = 0.05, 0.7–0.8% at α = 0.01). The table gives false alarms on the 762 later good parts and detection of the 16 later bad parts:

| Detector | False alarms, α = 0.05 | Detection, α = 0.05 | False alarms, α = 0.01 | Detection, α = 0.01 |
|---|---|---|---|---|
| RMS chart, prefix | 30.3% [27.2, 33.7] | 13/16 | 16.7% [14.2, 19.5] | 9/16 |
| T²+SPE, prefix | 65.2% [61.8, 68.5] | 16/16 | 26.0% [23.0, 29.2] | 13/16 |
| Cycle-time chart | **16.4%** [13.9, 19.2] | 12/16 | **1.8%** [1.1, 3.1] | 6/16 |
| MiniRocket ridge | 23.6% [20.7, 26.8] | 15/16 | 7.7% [6.0, 9.9] | 13/16 |
| MiniRocket kNN | 80.2% [77.2, 82.9] | 16/16 | 37.7% [34.3, 41.2] | 13/16 |

The label-free kNN is the most drift-sensitive chart, because it tracks the process tightly; on aligned windows its static false-alarm rate is 93.3% (RMS 37.7%, T²+SPE 68.6%, ridge 23.4%). That is useful for asking "has the process moved?" and costly as a nuisance alarm.

At matched false-alarm rates on later parts (thresholds set on the later good parts themselves):
- At 5%, the cycle-time chart catches 10 of 16, the kNN 6, the ridge 3, T²+SPE 3 and RMS 1.
- At 10%, the ridge catches 13, the cycle-time chart 11, the kNN 8, T²+SPE 5 and RMS 3.
- On aligned windows: at 5%, cycle time 10, T²+SPE 4, kNN 4, RMS 3, ridge 3; at 10%, ridge 12, cycle time 11, kNN 6, T²+SPE 4, RMS 3.
- The cycle-time chart's lead is the file-cutting artefact of section 5, not evidence for the chart.

**9. Re-qualification: about 20 parts restore the limit for every chart.** The first N confirmed-good parts after the change are taken in each of 11 machine × later timeframes. False alarms and detection are measured on the rest of each period (542 good and 16 bad parts at N = 20):

| N | Limit rule (guarantee) | RMS chart | T²+SPE | Cycle-time chart | MiniRocket ridge | MiniRocket kNN |
|---|---|---|---|---|---|---|
| 0 | static 2019 limit | 30.3% / 13 | 65.2% / 16 | 16.4% / 12 | 23.6% / 15 | 80.2% / 16 |
| 5 | max-of-N (16.7%) | 17.5% / 14 | 16.3% / 12 | 16.0% / 14 | 9.1% / 14 | 18.2% / 15 |
| 10 | max-of-N (9.1%) | 5.1% / 12 | 5.5% / 10 | 4.9% / 11 | 2.8% / 14 | 2.6% / 14 |
| 20 | conformal (4.76%) | 4.4% / 11 | 4.6% / 10 | 3.0% / 10 | 1.5% / 13 | 5.4% / 14 |
| 0, aligned | static 2019 limit | 37.7% / 11 | 68.6% / 15 | 16.4% / 12 | 23.4% / 14 | 93.3% / 16 |
| 10, aligned | max-of-N (9.1%) | 16.4% / 10 | 8.3% / 10 | 4.9% / 11 | 3.1% / 13 | 4.4% / 11 |
| 20, aligned | conformal (4.76%) | 5.2% / 10 | 5.2% / 9 | 3.0% / 10 | 2.0% / 13 | 6.6% / 11 |

Each cell gives false alarms on later good parts / later bad parts caught, out of 16.

- **Intervals at N = 20.** Wilson (parts independent): RMS 4.4% [3.0, 6.5], T²+SPE 4.6% [3.1, 6.7], cycle time 3.0% [1.8, 4.7], ridge 1.5% [0.8, 2.9], kNN 5.4% [3.8, 7.6]. These are too narrow: every part of a machine × period shares one re-qualified limit. Resampling the 11 machine × periods instead gives RMS 4.4% [0.2, 11.0], T²+SPE 4.6% [1.0, 8.8], cycle time 3.0% [0.9, 6.6], ridge 1.5% [0.0, 4.5], kNN 5.4% [0.2, 13.0] (aligned: RMS 5.2% [2.3, 8.6], T²+SPE 5.2% [1.0, 9.2], kNN 6.6% [0.3, 13.7], fused kNN 7.4% [1.1, 14.4]). Per machine × period, the re-centred charts' rates run from 0% (the median group for RMS, T²+SPE, ridge and kNN) to 29% (fused ridge; RMS 19%, kNN 22%). That spread is what the Beta(1, 20) law of section 13 predicts for single re-qualifications, plus some non-exchangeability. Aligned kNN 6.6% and fused kNN 7.4% sit above the nominal 4.76%, but well inside their clustered intervals.
- **Random draws of 20 parts** (50 draws) give 3.0–4.8% false alarms, with mean detection of 81% for the ridge, 77% for the kNN, 71% for RMS, 56% for T²+SPE and 47% for the cycle-time chart.
- **No head beats the classical charts at any N.** At N = 20 the kNN catches 14 of 16 against 11 for RMS: 3 parts caught only by the kNN and none only by RMS, exact McNemar p = 0.25. On aligned windows it is 11 against 10 (3 vs 2 discordant, p = 1.0). No head beats the best classical chart significantly at any N.
- **The limit-only fairness check** (recalibrating without re-centring) gives more false alarms with the first 20 parts: RMS 8.1%, T²+SPE 7.4%. The first parts after a change are not exchangeable with the rest of the period, so re-centring matters. Pooled re-centring did not make the RMS chart worse here.
- **N = 50** leaves only 5 machine × timeframes and 3 bad parts.

The answer to "how many parts" is a property of conformal calibration, not of the model: 19–20 parts for a 5% limit on average over re-qualifications, and 10 parts for 9%. Neither is a promise about any single re-qualification (section 13); with 11 machine × periods the data can confirm the pooled level only roughly (clustered intervals above).

**10. A new machine (copied from the other two, re-qualified on 20 of its parts).**

| Target (bad parts) | Chart | Crop arm, copied: false alarms / caught | Crop arm, re-qualified on the first 20 | Aligned arm, copied | Aligned arm, re-qualified on the first 20 |
|---|---|---|---|---|---|
| M01 (27) | RMS chart | 24.2% / 11 | 0/257 / 2 | 23.5% / 7 | 1/257 / 2 |
| | T²+SPE | 99.6% / 27 | 12/257 / 1 | 99.6% / 27 | 0/257 / 2 |
| | Cycle-time chart (file length) | 21.7% / 26 | 43/257 (16.7%) / 26 | same | same |
| | MiniRocket kNN | 87.7% / 26 | 13/257 (5.1%) / 15 | 94.2% / 27 | 8/257 (3.1%) / **6** |
| | MiniRocket ridge | 43.0% / 27 | 8/257 (3.1%) / 26 | 15.5% / 26 | 11/257 (4.3%) / 25 |
| M02 (24) | RMS chart | 10.3% / 12 | 0/272 / 2 | 16.8% / 8 | 0/272 / 0 |
| | T²+SPE | 66.4% / 24 | 0/272 / 0 | 79.1% / 21 | 0/272 / 0 |
| | Cycle-time chart (file length) | 5.5% / 21 | 16/272 (5.9%) / 22 | same | same |
| | MiniRocket kNN | 52.7% / 24 | 7/272 (2.6%) / 14 | 62.0% / 24 | 9/272 (3.3%) / **13** |
| | MiniRocket ridge | 1.7% / 24 | 5/272 (1.8%) / 24 | 1.7% / 22 | 20/272 (7.3%) / 24 |

- **kNN against the copied RMS chart, aligned arm:** 4 vs 0 discordant parts on M01 (exact McNemar p = 0.13, not significant) and 13 vs 0 on M02 (p = 0.0002). The results export 144 McNemar tests for the new-machine tables (2 arms × 3 machines × 4 N × 6 pairs); 0.0002 × 144 = 0.035, so M02 survives a Bonferroni correction over all of them. McNemar treats bad recordings as independent, though, and M02's 24 come from 2 timeframes (20 from Feb 2019, numbered 000–002 across 10 operations), so the effective sample is closer to a handful of runs. On the start-anchored prefix M01 had read 13 vs 0 (p = 0.0002); the drop on aligned windows suggests most of that edge came from the start offset of section 5. With random draws of 20 parts (aligned), the kNN catches 35% and 52% at 4.6% and 3.8% false alarms; RMS catches 17% and 9%.
- **The cycle-time chart** catches 26/27 and 22/24, but it reads file length, which here records how the bad files were cut (section 5). It is listed for completeness and is not evidence for the chart.
- **The supervised ridge** catches 25/27 and 24/24 on aligned windows, but it needs labelled bad history from the other machines, and the bad files of every machine share the same trimming, so part of what it learns may transfer as an artefact rather than as process knowledge.
- **Caveats.** M03 has 3 bad parts and supports no conclusion. The copied classical charts are re-centred but not re-scaled, because 20 parts per machine (1–2 per operation) cannot re-estimate each operation's spread. The stage 4 ramp recipe re-scaled with 10 parts per operation and reached 60–68% detection, on a much larger budget of parts.

**11. The drift label does not work here.** The forward-in-time drift subspace is well defined: its top 3 directions carry 61–74% of benign-shift variance, and in the latest period direction 1 is x/z power at 237–274 Hz and 649–866 Hz. It does not separate nuisance drift from defects:
- The mean defect direction puts 16% (Feb 2020) and 51–63% (later periods) of its energy in the drift subspace, against 5% for a random direction. Drift and defects are not orthogonal in this representation.
- Later good parts have a median share of 0.21 and later bad parts 0.47, which is the opposite of what a triage label needs.
- Of the kNN's 611 static false alarms, only 92 (15%) are labelled drift, and 7 of its 16 true alarms are labelled drift.
- On aligned windows it is the same picture: defect-direction share 30% (Feb 2020) and 64–66% later; 85 of the kNN's 711 static false alarms labelled drift.

The label is exported for transparency but should not be used for triage. The machine-level sign test (section 3) remains the re-baseline trigger.

**12. Attribution: one of three fingerprints reproduces** (aligned arm, so a bad part's windows and the good windows they are matched to come from the same stretch of the program; the kNN match itself ignores exact position, and the 12 M01 OP05 window matches below sit 0–6.6 s apart in program time (median 3.1 s). On the start-anchored prefix the comparison crossed toolpath phases for most bad parts).
- **M01 OP05 reproduces.** All 4 bad parts (Feb 2019 × 2, Aug 2019, Feb 2021) peak in the 178–205 Hz band on all three axes, with part-level z of 9.9–10.3 on x, 8.8–9.3 on z and 7.8–8.4 on y. Their worst kNN windows carry +16 to +26 dB in that band against their closest-matching good windows. Meanwhile the RMS chart reads normal: u = 0.04 to 0.36, against a limit of 1. This matches the autoencoder's 200–213 Hz fingerprint at coarser resolution, and is strongest on x rather than z/y. The kNN flags three of the four and aligned T²+SPE one. These parts start 2.9–3.9 s late, so the same check on the prefix (where it also reproduced, at +15 to +29 dB) was not trustworthy on its own.
- **M02 OP12 (701–720 Hz z) does not reproduce** (0 of 2): its bad parts peak at z 237–274 Hz, z ≈ 15.6. **M01 OP01 (720–760 Hz x/y) barely does** (1 of 2, and only as the third band, y 649–750 Hz at z = 5.2); both parts peak at z/y 237–274 Hz (z 10–12).
- **Post hoc and descriptive:** 39 of the 70 bad parts have 237–274 Hz among their top 3 bands, 15 of them with z-axis 237–274 Hz as the top band, and all 70 top deviations are excess energy. Direction 1 of the drift subspace also loads on 237–274 Hz, which is consistent with the drift label failing. With a 1 kHz Nyquist and no spindle speed in the data, the band cannot be tied to a mechanism.

**13. Cost scenario** (assumed 15 minutes of triage per alarm, per 1,000 good parts made after a drift event):

| | RMS chart | T²+SPE | Cycle-time chart | MiniRocket ridge | MiniRocket kNN |
|---|---|---|---|---|---|
| Static 2019 limits | 303 alarms (76 h) | 652 (163 h) | 164 (41 h) | 236 (59 h) | 802 (200 h) |
| Re-qualified on the first 20 good parts | 44 alarms (11 h) | 46 (11.5 h) | 30 (7.4 h) | 15 (3.7 h) | 54 (13.4 h) |

**Recommendation.**
1. **Re-qualify on parts, not on the calendar.** After a machine-level event, take the first 20 confirmed-good parts and recalibrate each chart's limit by split conformal. That held every re-centred chart to 1.5–5.5% false alarms pooled here (2.0–7.4% on aligned windows), against 16–80% for the stale limits. Expect the rate of any one re-qualification to scatter around that (0–29% per machine × period here), and pool alarms over several re-qualifications before concluding that a limit is wrong. Ten parts give a nominal 9% (max-of-N), again on average; single 10-part re-qualifications scatter more (Beta(1, 10): 95th percentile 25.9%), and on aligned windows the RMS chart's pooled rate with 10 parts was 16.4%. Below 19 parts no split-conformal limit carries a 5% guarantee, so say so instead of quoting 5%.
2. **Bring-up:** copy the lead machine's PCA structure, then re-centre *and* re-scale on at least 10 good parts per operation, sampled as widely across the early runs as possible. Here the first 10 parts gave 7.6% false alarms and 10 random qualification parts 3.0%. Run a full 16–25 part qualification once the volume allows it. Screen the qualification set for runs that are off in several operations at once before setting limits. If only about 20 parts per machine are available, do not trust a copied RMS or T²+SPE chart that has only been re-centred: on the two testable new machines the re-centred copies caught 1–2 of 27 and 0–2 of 24 bad parts. A label-free nearest-neighbour bank of the new machine's own good parts is a candidate vibration chart (append confirmed-good parts, no retraining), but on aligned windows it caught 13 of 24 on one machine and only 6 of 27 on the other, so treat it as unproven. A cycle-time chart costs nothing and is worth logging, but this dataset cannot say whether it detects bad parts (section 5).
3. **Re-baseline trigger:** the machine-level sign test. When most operations on one machine shift the same way in the same band, re-baseline that machine once. The forward-in-time drift label tested in stage 6 does not separate drift from defects, so do not use it for triage.
4. **Chart:** T² + SPE with empirically calibrated limits (highest matched AUC among the classical charts, though not significantly above the RMS chart; its higher detection comes with about 3× the RMS chart's false alarms at the limits), backed by the RMS chart as a drift-robust check. The cycle-time chart drifts least, but its detection numbers here measure how the bad files were cut, not the machine. Do not use the theoretical SPE limit at n < 30.
5. **AI:** the label-free MiniRocket bank can run beside these charts, because it needs no bad history and points to the window and band that moved, but on program-aligned windows it adds no measurable detection over the classical charts here. Do not ship the supervised model as the detector until its post-qualification detection holds across seeds on more than 16 later bad parts.

## Assumptions and limitations
- Labels come from the dataset authors. A bad label means an anomalous process, not a measured dimensional defect. There are only 70 bad recordings (0–4 per cell); per-cell detection is not meaningful, so results are pooled with cluster-bootstrap intervals.
- The conformal multiplier assumes leave-one-out ratios are exchangeable across cells. The half-split check supports this (0.96% against 1%). The screening rule removes 26 of 857 qualification recordings. It was chosen after looking at the calibration tail (post hoc), and unscreened numbers are reported above.
- Wilson intervals treat parts as independent. Parts share a cell and timeframe, so these intervals are likely narrower than cluster-aware ones. Only the AUCs use a cluster bootstrap.
- Nyquist is 1 kHz and spindle speed and tool data are not in the dataset, so tooth-pass and spindle harmonics cannot be assigned. Band shifts are located but not attributed to a specific mechanism.
- Recording order within a timeframe is taken from the file index. Wall-clock timestamps are not available, so "first N parts" means first by index.
- Features are recording-level and ignore sub-operation segmentation. Timeframe sizes are uneven (Aug 2020 has 92 later good parts). The rolling re-baseline equals the static one in Feb 2020 by construction.
- **AI layer, sample size.** Only 16 bad parts come after qualification, from 6 machine × timeframe episodes, so every post-qualification detection comparison has wide intervals; one part is 6.25 points. The like-for-like set has 396 bad/good pairs in 46 strata, most with one bad part, but only 11 machine × timeframe blocks. AUCs near 1 have little resolution there. The gate's bootstrap over strata under-covers: resampling the 11 blocks removes every pass (section 6), and 11 blocks are themselves too few for a percentile bootstrap to be accurate.
- **AI layer, multiple comparisons.** The gate guards its own multiplicity: two heads each get a chance, so it requires both conditions in all 3 seeds and (after the review) in two arms; Bonferroni-over-2-heads intervals are exported as `ci_bonferroni2` and give the same decision. Everything else (two arms, five or more charts, six N values, three new machines, two operating points) is descriptive. Nominal p-values outside the gate are not corrected unless stated, and "not significant" is the reading for all of them except the M02 new-machine result.
- **AI layer, the duration and start-offset controls.** The crop removes recording length but not the start offset, which the aligned arm removes. Alignment is estimated from envelopes at 50 ms resolution; correlations are lower for bad parts (median r 0.47 against 0.62), some good parts in programs with repeated structure lock onto a later repeat (12% of good parts sit more than 1 s late), and the common window of a cell starts after its latest-starting recording, which is chosen without labels but with every recording of the cell. A bad file trimmed at both ends may still differ from good parts near the window edges. The ridge's near-perfect aligned AUC (0.992–1.000) may still rest on such residual cues; it is not tested further here.
- **AI layer, protocol.** The gate, the scorecards and the static drift tables use all 3 MiniRocket seeds. The re-qualification, new-machine, drift-subspace and attribution tables use seed 0. The ridge's 2019 scores come from models fitted without the scored machine × timeframe, while 2020–21 parts are scored by the model fitted on all of 2019; its conformal calibration treats these as exchangeable. Re-qualification calibration scores are leave-one-out (jackknife-style), so the guarantee is approximate. The observed rates sit at or below it in the random-draw scheme and within noise of it for the first 20 parts.
- **AI layer, bad labels.** The bad parts are gross process anomalies with different spectra, distributed as trimmed excerpts of their recordings, which makes near-perfect ranking easier than it would be for subtle drift toward a tolerance limit, and makes recording length and start unusable as evidence.

## Changes made after the first evaluation run of stages 5–6
The gate, the comparisons and the seeds were fixed in code before the first full run, and its outcome reproduced exactly on the rerun because the pipeline is deterministic. After that run:
1. Band attribution now ranks unclipped z. With z clipped at ±8, gross anomalies tied at the bound, and their top 3 bands fell back to band order. The clipped values are still used for the drift subspace.
2. Paired detection tests were added to the re-qualification and new-machine tables: a bootstrap over bad parts and an exact McNemar test.
3. The gate output now names the best classical chart.

Found by an adversarial leakage review, after the headline had been written (items 4–5), and by a statistics review (item 6):
4. **Start-offset shortcut.** The crop arm removes recording length but not where a recording starts; the bad files are trimmed excerpts starting a median 3.8 s late (start offset alone: AUC 0.94). `align.py` and the aligned arm were added, the gate now requires a pass in both cut arms, and attribution moved to the aligned arm. Neither head passes, so the decision is unchanged, but the headline's kNN AUC (0.98 → 0.96–0.97), its new-machine result on M01 (15/27 → 6/27), the reading of the cycle-time chart and the attribution dB values changed.
5. **Screening-key collision.** The qualification screen keyed runs by machine × timeframe × part index, and 10 bad files share index 000 with a screened good run, so they were silently dropped from the supervised head's training set (and counted as 36 screened recordings instead of 26). The key now matches good files only. This changed the ridge and fused-ridge numbers (for example static false alarms 17.7% → 23.6%, re-qualified 2.6% → 1.5%) and nothing else.
6. **Clustered uncertainty and what the conformal guarantee promises.** The gate's two conditions were re-run resampling whole machine × timeframe blocks (no head passes either condition; the decision is unchanged), re-qualification false alarms got intervals over the 11 machine × periods plus per-period ranges, the spread of a single re-qualification's rate (Beta law) was added, and the 4.76% is now called nominal for the leave-one-out re-centred charts, after a simulation put that procedure at about 4.9%. No existing number changed: the rerun reproduced every previous value in `results.json`.
7. **Claims review.** Every number in this file and the README was re-checked against `results.json`. Percentages are now rounded from the exact counts (k/n) rather than from the 4-decimal JSON values, which moved five table cells by 0.1 point (for example kNN 5.3% → 5.4% = 29/542). Wording was tightened: the attribution matches come from the same aligned stretch of the program, not the same instant; the bad files "look like" trimmed excerpts (the evidence is strong but indirect); the 10-part rate is nominal, not guaranteed; the in-period check covers 4.4–4.8% once the full arm is included. No result or decision changed.

A smoke run before the first full run found that five cells qualify into 2020–21. "Later" was then made per cell, and the drift subspace for period t now uses only cells qualified before t.

## Reproduce
```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
git clone --depth 1 https://github.com/boschresearch/CNC_Machining ../_downloads/bosch-cnc/repo
DATA=../_downloads/bosch-cnc/repo/data
.venv/bin/python -I pipeline/extract.py $DATA pipeline/out/features.npz
.venv/bin/python -I pipeline/analyze.py pipeline/out/features.npz public/data/results.json
.venv/bin/python -I pipeline/diagnostics.py pipeline/out/features.npz public/data/results.json
export OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
nice -n 10 .venv/bin/python -I pipeline/align.py $DATA pipeline/out        # start offsets, about 15 s
nice -n 10 .venv/bin/python -I pipeline/windows.py $DATA pipeline/out      # about 20 s; 1.5 GB of windows in pipeline/out
nice -n 10 .venv/bin/python -I pipeline/embed.py pipeline/out crop full align   # 3 arms x 3 seeds x 3 folds, about 23 min on CPU
nice -n 10 .venv/bin/python -I pipeline/ai_eval.py pipeline/out public/data/results.json   # about 2.5 min
.venv/bin/python -m pytest -q tests                                      # 72 tests
```
