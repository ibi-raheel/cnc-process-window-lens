# Method

## Question
A per-operation vibration baseline is set once, when a CNC process is qualified. On real production data, how fast does it go stale, what happens when one machine's baseline is copied to another, and does a multivariate or ML detector beat a plain RMS control chart? Stages 5 and 6 add a modern time-series model (MiniRocket) with a supervised and a label-free head, and ask whether it adds anything once the obvious shortcut is removed: bad recordings in this dataset are shorter than good ones.

## Summary of the AI layer (stages 5–6)
A pre-declared gate decided the headline. The AI headline needed a model that, without seeing recording length, beat the best classical chart on like-for-like ranking (paired cluster-bootstrap CI above 0) and caught clearly more bad parts than the RMS chart after qualification at equal false alarms, in all 3 MiniRocket seeds. **Neither head passed, so the fallback headline ships**:
- **Re-qualify, don't copy.** Every chart's 2019 limits go stale (16–80% false alarms on later good parts at a 5% design rate). Re-centring each chart on the first 20 confirmed-good parts after a change, and recalibrating its limit, brings every chart back to 2.6–5.5%, against an exact finite-sample guarantee of 4.76%. That comes from the split-conformal calibration, not from any model.
- **Label-free ranking matches the best classical chart.** A MiniRocket nearest-neighbour memory bank that sees only the first 45% of each cycle and has never seen a bad part ranks bad parts with AUC 0.980–0.985 (3 seeds) against 0.965 for the best classical chart. That difference is not significant: +0.015 [−0.010, +0.052].
- **A free cycle-time chart is the strongest simple baseline.** At equal 5% false alarms on post-qualification parts it caught 10 of 16 bad parts, against 6 for the best AI head and 1 for the RMS chart.
- **The supervised head ranks best but did not hold up.** Duration-blind, its like-for-like AUC is 1.000 against 0.965 (+0.035 [+0.002, +0.083]). On post-qualification parts it caught 3, 4 or 6 of 16 depending on the seed, so it is reported but not headlined.
- **On a new machine,** with charts copied from the other two machines and re-qualified on 20 of its parts, the label-free kNN caught 15/27 and 14/24 bad parts, against 2/27 and 2/24 for the copied, re-centred RMS chart (exact McNemar p ≤ 0.0005). The free cycle-time chart caught 26/27 and 22/24.
- **Two grafts failed and are reported as negative results.** The forward-in-time drift label does not separate nuisance drift from defects. Only one of three autoencoder band fingerprints reproduces (M01 OP05 at 178–205 Hz, in 4 of 4 bad parts, where the RMS chart reads normal).

## Data
- **Bosch CNC Machining dataset**: https://github.com/boschresearch/CNC_Machining
- Tri-axial accelerometer (Bosch CISS) inside three brownfield CNC milling machines (M01–M03), 15 operations (OP00–OP14; M03 has OP00–OP13), sampled at 2 kHz.
- Six timeframes labelled Feb 2019, Aug 2019, Feb 2020, Aug 2020, Feb 2021, Aug 2021. Each recording is labelled good or bad by the dataset authors.
- **All 1,702 recordings are used** (70 bad, 4.1%). There are 44 machine × operation cells; 43 have enough good recordings to model and 1 (M03 OP03, 13 good) is skipped.
- **Licence:** data CC-BY-4.0, loader code BSD-3-Clause. Citation: Tnani, M.-A., Feil, M., Diepold, K. *Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring.* Procedia CIRP 107 (2022) 131–136. doi:10.1016/j.procir.2022.04.022
- No synthetic data in the analysis. The sine and noise signals in `tests/` are unit-test fixtures only.

## Pipeline
`pipeline/extract.py` → `pipeline/out/features.npz` → `pipeline/analyze.py` → `pipeline/diagnostics.py` → `pipeline/windows.py` → `pipeline/embed.py` → `pipeline/ai_eval.py` → `public/data/results.json`. Stages 5–6 write the `ai` key; the stage 4 summary is in `public/data/diagnostics_summary.json`. Shared statistics (Wilson intervals, stratified AUC, split-conformal limits, bootstraps, McNemar) live in `pipeline/stats.py`, which both `diagnostics.py` and `ai_eval.py` import. Large caches (windows, embeddings, per-part scores) stay in the git-ignored `pipeline/out/`.

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

**Qualification-set screen.** Without screening, the 99th percentile of the T² ratios is 6.9× the UCL (95th percentile 0.71×, 98th 1.22×), which gives a multiplier of 12.2. The tail is not random noise. It comes from two production runs: part 000 on M02 in Feb 2019, more than 3× its UCL in 8 operations, and part 006 on M01 in Aug 2019, more than 3× in 4 operations. When one part is off across its whole program, that is more likely a machine-level event than process noise. The rule below was chosen after inspecting this tail, so it is post hoc: drop a run (machine, timeframe, part index) from every baseline when its leave-one-out T² is more than 3× the UCL in at least 3 operations. That drops 2 runs, 36 recordings. Screened recordings stay in every evaluation set and are scored out of sample.

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

Per window the script also stores log10 band power on 20 attribution bands per axis. These are the 8 Welch bands of `features.py`, with each band above 100 Hz split into 4 log sub-bands, and every band holds at least 4 FFT bins. Per recording it stores the vector RMS and the repo's 40 features computed on the prefix; these feed the crop arm's RMS and T²+SPE charts. Rows are keyed by path relative to the data directory, because good and bad files share basenames: 61 of the 70 bad recordings have a good file with the same name.

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
| Stratified AUC: 2-stratum hand count (0.75); unchanged when a constant is added to each stratum; ties count ½ | exact | pass |
| Cluster bootstrap resamples whole strata: replicates only take stratum-mixture values | {0.5, 0.75, 1} | pass |
| Paired bootstrap of identical scores | exactly 0 | pass |
| Dual ridge against primal weighted ridge | 1e-8 | pass |
| Ridge: scores of test rows unchanged when other, very different test rows are added (scaler, penalty and fit use training rows only) | identical | pass |
| kNN: nearest-window distances against brute force; a part never matches itself; an exact duplicate is matched to its twin | exact | pass |
| Hierarchical re-centring removes a pure offset exactly, shrinks per-operation corrections and falls back to the machine offset | exact | pass |
| Duration crop: window starts depend only on the cell's prefix; truncating a recording to 85% leaves its prefix windows and kNN score unchanged, while the full arm changes | exact | pass |
| Drift subspace for period t uses good parts from earlier periods only; a random direction's share is K/p | exact; 0.050 | pass |
| Real data: RMS in the window cache against features.npz | 1e-6 | max relative error 8×10⁻⁸, pass |
| Real data: every cached embedding fold was fitted without its target machine, bad parts or post-2019 parts (18 folds) | none | pass |
| Real data: reimplemented RMS and T²+SPE charts against the repo's like-for-like (stratified) AUCs | 0.884 / 0.960 | 0.8838 / 0.9596, pass |
| **Real data: split-conformal limits inside the qualification period** (random half splits per machine, every chart) | 5% / 1% | **4.4–4.8% / 0.7–0.8%** |
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
Unless a row says otherwise, numbers are from the crop arm, seed 0. The full arm and the other seeds are in `results.json → ai`.

**5. The duration giveaway.** Every bad recording is shorter than its cell's qualification median: the ratio runs from 0.49 to 0.98, with a median of 0.84. Good recordings sit between 0.86 and 1.10 (1st–99th percentile). A "shorter cycle" rule alone separates bad from good parts of the same machine, operation and timeframe with AUC 0.985; the two-sided cycle-time chart reaches 0.929. Any model that sees whole recordings can win by reading length, and the crop arm removes that cue. Head scores barely track recording length (|Spearman ρ| ≤ 0.23 with the duration ratio, in every arm and seed). The crop cannot rule out that some bad recordings start at a different point of the program: the kNN ignores window position, the ridge does not.

**6. Gate result: the AI headline does not ship.**

| Head (crop arm) | Seed | Like-for-like AUC | Δ vs best classical (0.965, T²+SPE on the prefix) | A | Later bad parts caught at 5% false alarms (RMS: 1 of 16) | B | Pass |
|---|---|---|---|---|---|---|---|
| Ridge (supervised) | 0 | 1.000 | +0.035 [+0.002, +0.083] | ✓ | 3 of 16, Δ CI [0.00, +0.56] | ✗ | ✗ |
| | 1 | 1.000 | +0.035 [+0.002, +0.083] | ✓ | 4 of 16, Δ CI [0.00, +0.69] | ✗ | ✗ |
| | 2 | 1.000 | +0.035 [+0.002, +0.083] | ✓ | 6 of 16, Δ CI [+0.06, +0.75] | ✓ | ✓ |
| kNN (label-free) | 0 | 0.980 | +0.015 [−0.010, +0.052] | ✗ | 6 of 16, Δ CI [+0.06, +0.56] | ✓ | ✗ |
| | 1 | 0.985 | +0.020 [−0.012, +0.062] | ✗ | 5 of 16, Δ CI [+0.06, +0.50] | ✓ | ✗ |
| | 2 | 0.982 | +0.018 [−0.006, +0.052] | ✗ | 5 of 16, Δ CI [+0.06, +0.50] | ✓ | ✗ |

The ridge ranks better than every classical chart, but its post-qualification detection depends on the MiniRocket seed. The kNN catches more later bad parts than RMS in every seed, but its ranking edge over T²+SPE is within noise. In the full arm (reference only), neither head passes either: the ridge's A interval touches 0, at [0.000, +0.084]. Neither head passes in all three seeds, so the label-free and re-qualification headline ships.

**7. Like-for-like scorecard** (68 bad, 344 good, 46 machine × operation × timeframe strata; 95% cluster-bootstrap CIs):

| Detector | AUC | Δ vs best classical | PR-AUC | False alarms at the α = 0.05 limit | Detection at the limit |
|---|---|---|---|---|---|
| RMS chart, prefix | 0.861 [0.751, 0.951] | | 0.825 | 6.1% [4.0, 9.2] | 56/68 |
| RMS chart, whole recording | 0.884 [0.771, 0.966] | | 0.821 | 7.6% [5.2, 10.8] | 53/68 |
| T²+SPE, prefix | 0.965 [0.917, 0.998] | | 0.790 | 11.3% [8.4, 15.1] | 66/68 |
| T²+SPE, whole recording | 0.960 [0.911, 0.994] | | 0.795 | 12.2% [9.2, 16.1] | 63/68 |
| Cycle-time chart | 0.929 [0.849, 0.983] | −0.035 [−0.128, +0.039] | 0.851 | 7.0% [4.7, 10.2] | 59/68 |
| MiniRocket ridge (supervised) | **1.000** [1.000, 1.000] | **+0.035 [+0.002, +0.083]** | **0.983** | 2.3% [1.2, 4.5] | 65/68 |
| MiniRocket kNN (label-free) | 0.980 [0.955, 0.998] | +0.015 [−0.010, +0.052] | 0.912 | 14.2% [10.9, 18.3] | 66/68 |
| Fused ridge + cycle time | 0.992 [0.979, 1.000] | +0.028 [−0.008, +0.077] | 0.927 | 5.8% [3.8, 8.8] | 66/68 |
| Fused kNN + cycle time | 0.987 [0.972, 0.998] | +0.023 [−0.011, +0.069] | 0.913 | 10.8% [7.9, 14.5] | 68/68 |

Notes on the scorecard:
- The like-for-like set holds only 396 bad/good pairs, so AUC moves in steps of 0.0025, and most strata hold one bad part.
- The part bootstrap gives a similar interval for the kNN's Δ: [−0.018, +0.061]. Over the 3 seeds the kNN's AUC runs from 0.980 to 0.985.
- The false-alarm column includes later-period good parts, which is why it sits above 5%.
- With wider machine × timeframe strata (70 bad, 1,255 good), the fused kNN + cycle-time rule does beat the best classical chart: 0.983 vs 0.936, Δ [+0.012, +0.082]. That was not the pre-declared comparison.

**8. Every chart's 2019 limit goes stale.** These are static per-machine conformal limits, which hold their level inside the qualification period (4.4–4.8% at α = 0.05, 0.7–0.8% at α = 0.01). The table gives false alarms on the 762 later good parts and detection of the 16 later bad parts:

| Detector | False alarms, α = 0.05 | Detection, α = 0.05 | False alarms, α = 0.01 | Detection, α = 0.01 |
|---|---|---|---|---|
| RMS chart, prefix | 30.3% [27.2, 33.7] | 13/16 | 16.7% [14.2, 19.5] | 9/16 |
| T²+SPE, prefix | 65.2% [61.8, 68.5] | 16/16 | 26.0% [23.0, 29.2] | 13/16 |
| Cycle-time chart | **16.4%** [13.9, 19.2] | 12/16 | **1.8%** [1.1, 3.1] | 6/16 |
| MiniRocket ridge | 17.7% [15.2, 20.6] | 13/16 | 8.0% [6.3, 10.2] | 13/16 |
| MiniRocket kNN | 80.2% [77.2, 82.9] | 16/16 | 37.7% [34.3, 41.2] | 13/16 |

The label-free kNN is the most drift-sensitive chart, because it tracks the process tightly. That is useful for asking "has the process moved?" and costly as a nuisance alarm.

At matched false-alarm rates on later parts (thresholds set on the later good parts themselves):
- At 5%, the cycle-time chart catches 10 of 16, the kNN 6, the ridge 3, T²+SPE 3 and RMS 1.
- At 10%, the ridge catches 13, the cycle-time chart 11, the kNN 8, T²+SPE 5 and RMS 3.

**9. Re-qualification: about 20 parts restore the limit for every chart.** The first N confirmed-good parts after the change are taken in each of 11 machine × later timeframes. False alarms and detection are measured on the rest of each period (542 good and 16 bad parts at N = 20):

| N | Limit rule (guarantee) | RMS chart | T²+SPE | Cycle-time chart | MiniRocket ridge | MiniRocket kNN |
|---|---|---|---|---|---|---|
| 0 | static 2019 limit | 30.3% / 13 | 65.2% / 16 | 16.4% / 12 | 17.7% / 13 | 80.2% / 16 |
| 5 | max-of-N (16.7%) | 17.5% / 14 | 16.3% / 12 | 16.0% / 14 | 8.1% / 14 | 18.2% / 15 |
| 10 | max-of-N (9.1%) | 5.1% / 12 | 5.5% / 10 | 4.9% / 11 | 3.4% / 13 | 2.6% / 14 |
| 20 | conformal (4.76%) | 4.4% / 11 | 4.6% / 10 | 2.9% / 10 | 2.6% / 13 | 5.3% / 14 |

Each cell gives false alarms on later good parts / later bad parts caught, out of 16.

- **Intervals at N = 20:** RMS 4.4% [3.0, 6.5], T²+SPE 4.6% [3.1, 6.7], cycle time 2.9% [1.8, 4.7], ridge 2.6% [1.5, 4.3], kNN 5.3% [3.8, 7.6].
- **Random draws of 20 parts** (50 draws) give 3.0–4.8% false alarms, with mean detection of 82% for the ridge, 77% for the kNN, 71% for RMS, 56% for T²+SPE and 47% for the cycle-time chart.
- **No head beats the classical charts at any N.** At N = 20 the kNN catches 14 of 16 against 11 for RMS: 3 parts caught only by the kNN and none only by RMS, exact McNemar p = 0.25. No head beats the best classical chart significantly at any N.
- **The limit-only fairness check** (recalibrating without re-centring) gives more false alarms with the first 20 parts: RMS 8.1%, T²+SPE 7.4%. The first parts after a change are not exchangeable with the rest of the period, so re-centring matters. Pooled re-centring did not make the RMS chart worse here.
- **N = 50** leaves only 5 machine × timeframes and 3 bad parts.

The answer to "how many parts" is a property of conformal calibration, not of the model: 19–20 parts for a guaranteed 5% limit, and 10 parts for a guaranteed 9%.

**10. A new machine (copied from the other two, re-qualified on 20 of its parts).**

| Target (bad parts) | Chart | Copied: false alarms / caught | Re-qualified on the first 20 parts: false alarms / caught |
|---|---|---|---|
| M01 (27) | RMS chart | 24.2% / 11 | 0/257 / 2 |
| | T²+SPE | 99.6% / 27 | 12/257 / 1 |
| | Cycle-time chart | 21.7% / 26 | 43/257 (16.7%) / 26 |
| | MiniRocket kNN | 87.7% / 26 | 13/257 (5.1%) / **15** |
| | MiniRocket ridge | 40.4% / 27 | 9/257 (3.5%) / 25 |
| M02 (24) | RMS chart | 10.3% / 12 | 0/272 / 2 |
| | T²+SPE | 66.4% / 24 | 0/272 / 0 |
| | Cycle-time chart | 5.5% / 21 | 16/272 (5.9%) / 22 |
| | MiniRocket kNN | 52.7% / 24 | 7/272 (2.6%) / **14** |
| | MiniRocket ridge | 1.7% / 24 | 5/272 (1.8%) / 24 |

- **kNN against the copied RMS chart:** 13 vs 0 and 12 vs 0 discordant parts (exact McNemar p = 0.0002 and 0.0005). With random draws of 20 parts, the kNN catches 52% and 55% at 4.7% and 3.5% false alarms; RMS catches 22% and 22%.
- **The free cycle-time chart catches more than the kNN:** 11 vs 0 discordant parts on M01 (p = 0.001) and 10 vs 2 on M02 (p = 0.04), though at 16.7% false alarms on M01.
- **The supervised ridge** catches 25/27 and 24/24, but it needs labelled bad history from the other machines.
- **Caveats.** M03 has 3 bad parts and supports no conclusion. The copied classical charts are re-centred but not re-scaled, because 20 parts per machine (1–2 per operation) cannot re-estimate each operation's spread. The stage 4 ramp recipe re-scaled with 10 parts per operation and reached 60–68% detection, on a much larger budget of parts.

**11. The drift label does not work here.** The forward-in-time drift subspace is well defined: its top 3 directions carry 61–74% of benign-shift variance, and in the latest period direction 1 is x/z power at 237–274 Hz and 649–866 Hz. It does not separate nuisance drift from defects:
- The mean defect direction puts 16% (Feb 2020) and 51–63% (later periods) of its energy in the drift subspace, against 5% for a random direction. Drift and defects are not orthogonal in this representation.
- Later good parts have a median share of 0.21 and later bad parts 0.47, which is the opposite of what a triage label needs.
- Of the kNN's 611 static false alarms, only 92 (15%) are labelled drift, and 7 of its 16 true alarms are labelled drift.

The label is exported for transparency but should not be used for triage. The machine-level sign test (section 3) remains the re-baseline trigger.

**12. Attribution: one of three fingerprints reproduces.**
- **M01 OP05 reproduces.** All 4 bad parts (Feb 2019 × 2, Aug 2019, Feb 2021) peak in the 178–205 Hz band on all three axes, with part-level z of 12–14 on x, 9–10 on z and 7–8 on y. Their worst kNN windows carry +15 to +29 dB in that band against the nearest good windows. Meanwhile the RMS chart reads normal: u = −0.39 to 0.31, against a limit of 1. This matches the autoencoder's 200–213 Hz fingerprint at coarser resolution, and is strongest on x rather than z/y. T²+SPE and the cycle-time chart flag all four; the kNN flags three.
- **M02 OP12 (701–720 Hz z) and M01 OP01 (720–760 Hz x/y) do not reproduce.** Their bad parts peak at 237–274 Hz, with z of 10–16.
- **Post hoc and descriptive:** 46 of the 70 bad parts have 237–274 Hz among their top 3 bands, 25 of them with z-axis 237–274 Hz as the top band, and 66 of 70 top deviations are excess energy. Direction 1 of the drift subspace also loads on 237–274 Hz, which is consistent with the drift label failing. With a 1 kHz Nyquist and no spindle speed in the data, the band cannot be tied to a mechanism.

**13. Cost scenario** (assumed 15 minutes of triage per alarm, per 1,000 good parts made after a drift event):

| | RMS chart | T²+SPE | Cycle-time chart | MiniRocket ridge | MiniRocket kNN |
|---|---|---|---|---|---|
| Static 2019 limits | 303 alarms (76 h) | 652 (163 h) | 164 (41 h) | 177 (44 h) | 802 (200 h) |
| Re-qualified on the first 20 good parts | 44 alarms (11 h) | 46 (11.5 h) | 30 (7.4 h) | 26 (6.5 h) | 54 (13.4 h) |

**Recommendation.**
1. **Re-qualify on parts, not on the calendar.** After a machine-level event, take the first 20 confirmed-good parts and recalibrate each chart's limit by split conformal. That held every re-centred chart to about 5% false alarms here, against 16–80% for the stale limits. Ten parts give a guaranteed 9% (max-of-N). Below 19 parts no split-conformal limit carries a 5% guarantee, so say so instead of quoting 5%.
2. **Bring-up:** copy the lead machine's PCA structure, then re-centre *and* re-scale on at least 10 good parts per operation, sampled as widely across the early runs as possible. Here the first 10 parts gave 7.6% false alarms and 10 random qualification parts 3.0%. Run a full 16–25 part qualification once the volume allows it. Screen the qualification set for runs that are off in several operations at once before setting limits. If only about 20 parts per machine are available, do not trust a copied RMS or T²+SPE chart that has only been re-centred: on the two testable new machines the re-centred copies caught 1–2 of 27 and 0–2 of 24 bad parts. Run the free cycle-time chart from day one, and add a label-free nearest-neighbour bank of the new machine's own good parts as the vibration chart (append confirmed-good parts, no retraining).
3. **Re-baseline trigger:** the machine-level sign test. When most operations on one machine shift the same way in the same band, re-baseline that machine once. The forward-in-time drift label tested in stage 6 does not separate drift from defects, so do not use it for triage.
4. **Chart:** T² + SPE with empirically calibrated limits (highest matched AUC among the classical charts, though not significantly above the RMS chart; its higher detection comes with about 3× the RMS chart's false alarms at the limits), backed by the RMS chart as a drift-robust check and the cycle-time chart, which costs nothing and drifts least. Do not use the theoretical SPE limit at n < 30.
5. **AI:** the label-free MiniRocket bank is worth running beside these charts, because it needs no bad history and points to the window and band that moved. Do not ship the supervised model as the detector until its post-qualification detection holds across seeds on more than 16 later bad parts.

## Assumptions and limitations
- Labels come from the dataset authors. A bad label means an anomalous process, not a measured dimensional defect. There are only 70 bad recordings (0–4 per cell); per-cell detection is not meaningful, so results are pooled with cluster-bootstrap intervals.
- The conformal multiplier assumes leave-one-out ratios are exchangeable across cells. The half-split check supports this (0.96% against 1%). The screening rule removes 36 of 857 qualification recordings. It was chosen after looking at the calibration tail (post hoc), and unscreened numbers are reported above.
- Wilson intervals treat parts as independent. Parts share a cell and timeframe, so these intervals are likely narrower than cluster-aware ones. Only the AUCs use a cluster bootstrap.
- Nyquist is 1 kHz and spindle speed and tool data are not in the dataset, so tooth-pass and spindle harmonics cannot be assigned. Band shifts are located but not attributed to a specific mechanism.
- Recording order within a timeframe is taken from the file index. Wall-clock timestamps are not available, so "first N parts" means first by index.
- Features are recording-level and ignore sub-operation segmentation. Timeframe sizes are uneven (Aug 2020 has 92 later good parts). The rolling re-baseline equals the static one in Feb 2020 by construction.
- **AI layer, sample size.** Only 16 bad parts come after qualification, so every post-qualification detection comparison has wide intervals; one part is 6.25 points. The like-for-like set has 396 bad/good pairs in 46 strata, most with one bad part. AUCs near 1 have little resolution there, and the cluster bootstrap over strata probably under-covers.
- **AI layer, the duration control.** The crop removes recording length, but it cannot rule out that some bad recordings start at a different point of the program. The kNN ignores where a window sits; the ridge's mean embedding does not.
- **AI layer, protocol.** The gate, the scorecards and the static drift tables use all 3 MiniRocket seeds. The re-qualification, new-machine, drift-subspace and attribution tables use seed 0. The ridge's 2019 scores come from models fitted without the scored machine × timeframe, while 2020–21 parts are scored by the model fitted on all of 2019; its conformal calibration treats these as exchangeable. Re-qualification calibration scores are leave-one-out (jackknife-style), so the guarantee is approximate. The observed rates sit at or below it in the random-draw scheme and within noise of it for the first 20 parts.
- **AI layer, bad labels.** The bad parts are gross process anomalies (shorter cycles, different spectra), which makes near-perfect ranking easier than it would be for subtle drift toward a tolerance limit.

## Changes made after the first evaluation run of stages 5–6
The gate, the comparisons and the seeds were fixed in code before the first full run, and its outcome reproduced exactly on the rerun because the pipeline is deterministic. After that run:
1. Band attribution now ranks unclipped z. With z clipped at ±8, gross anomalies tied at the bound, and their top 3 bands fell back to band order. The clipped values are still used for the drift subspace.
2. Paired detection tests were added to the re-qualification and new-machine tables: a bootstrap over bad parts and an exact McNemar test.
3. The gate output now names the best classical chart.

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
nice -n 10 .venv/bin/python -I pipeline/windows.py $DATA pipeline/out      # about 15 s; 1 GB of windows in pipeline/out
nice -n 10 .venv/bin/python -I pipeline/embed.py pipeline/out            # 2 arms x 3 seeds x 3 folds, about 15 min on CPU
nice -n 10 .venv/bin/python -I pipeline/ai_eval.py pipeline/out public/data/results.json   # about 2 min
.venv/bin/python -m pytest -q tests                                      # 65 tests
```
