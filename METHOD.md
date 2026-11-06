# Method

## Question
A per-operation vibration baseline is set once, when a CNC process is qualified. On real production data, how fast does it go stale, what happens when one machine's baseline is copied to another, and does a multivariate or ML detector beat a plain RMS control chart?

## Data
- **Bosch CNC Machining dataset**: https://github.com/boschresearch/CNC_Machining
- Tri-axial accelerometer (Bosch CISS) inside three brownfield CNC milling machines (M01–M03), 15 operations (OP00–OP14; M03 has OP00–OP13), sampled at 2 kHz.
- Six timeframes labelled Feb 2019, Aug 2019, Feb 2020, Aug 2020, Feb 2021, Aug 2021. Each recording is labelled good or bad by the dataset authors.
- **All 1,702 recordings are used** (70 bad, 4.1%). There are 44 machine × operation cells; 43 have enough good recordings to model and 1 (M03 OP03, 13 good) is skipped.
- **Licence:** data CC-BY-4.0, loader code BSD-3-Clause. Citation: Tnani, M.-A., Feil, M., Diepold, K. *Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring.* Procedia CIRP 107 (2022) 131–136. doi:10.1016/j.procir.2022.04.022
- No synthetic data in the analysis. The sine and noise signals in `tests/` are unit-test fixtures only.

## Pipeline
`pipeline/extract.py` → `pipeline/out/features.npz` → `pipeline/analyze.py` → `pipeline/diagnostics.py` → `public/data/results.json` (summary in `public/data/diagnostics_summary.json`)

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

**Qualification-set screen.** Without screening, the 99th percentile of the T² ratios is 6.9× the UCL (95th percentile 0.71×, 98th 1.22×), which gives a multiplier of 12.2. The tail is not random noise. It comes from two production runs: part 000 on M02 in Feb 2019, more than 3× its UCL in 8 operations, and part 006 on M01 in Aug 2019, more than 3× in 4 operations. When one part is off across its whole program, that is a machine-level event. The rule is fixed in advance: drop a run (machine, timeframe, part index) from every baseline when its leave-one-out T² is more than 3× the UCL in at least 3 operations. That drops 2 runs, 36 recordings. Screened recordings stay in every evaluation set and are scored out of sample.

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
| Leave-one-out after screening + calibration (831 fits) | 1% | T² 0.8% [0.4, 1.7], combined 1.3% [0.7, 2.4] |
| **Cross-validated calibration** (fit on half the cells, test on the other half, 200 splits) | ≈ 1% | **T² 0.96% (95% of splits 0–2.5%)**, combined 1.5% |

## Results (measured; reproduced by `pipeline/analyze.py` + `pipeline/diagnostics.py`)
Multipliers after screening: T² 1.39, SPE 7.67, RMS chart 1.23, Isolation Forest 1.05. SPE needs a 7.7× correction, so its theoretical limit is not usable at this n.

**1. The baseline goes stale (calibrated limits).** With T² limits calibrated to 0.8% leave-one-out false alarms inside qualification, the 2019 baseline flags **41.5% [38.0, 45.0] of the 762 later good parts**: Feb 2020 38.8%, Aug 2020 55.4%, Feb 2021 40.8%, Aug 2021 37.8%. The combined T²/SPE flags 58.1% [54.6, 61.6]. The earlier 76–97% headline used the uncalibrated combined limit and overstated the drift.

**2. Detector scorecard, like with like.** Matched timeframes: 70 bad, 344 good, prevalence 16.9%.

| Detector | ROC-AUC [95% cluster CI] | Stratified AUC | PR-AUC | False alarms at limit | Detection at limit |
|---|---|---|---|---|---|
| RMS Shewhart | 0.919 [0.841, 0.971] | 0.884 | 0.832 | 2.6% [1.4, 4.9] | 72.9% [61.5, 81.9] |
| PCA T² | 0.935 [0.893, 0.971] | 0.957 | 0.790 | 4.9% [3.1, 7.8] | 77.1% [66.0, 85.4] |
| PCA T² + SPE | **0.954** [0.927, 0.977] | **0.960** | 0.798 | 7.8% [5.5, 11.2] | **90.0%** [80.8, 95.1] |
| Isolation Forest | 0.857 [0.800, 0.907] | 0.894 | 0.607 | 1.7% [0.8, 3.8] | 28.6% [19.3, 40.1] |

RMS − T²/SPE AUC difference: −0.034 [−0.100, +0.007]. **The earlier claim that "plain SPC beats PCA" does not survive** a like-for-like comparison: the multivariate chart is at least as good, but the difference is not significant at 95%. On the drift-inclusive set (all bad parts against later good parts), RMS still wins (0.768 against 0.705; difference +0.063 [+0.007, +0.127]). It wins only because it reacts least to drift, not because it separates bad parts better. The Isolation Forest is the weakest in both cases.

**3. Why it goes stale.**
- *Spectral shape, not sensor gain.* Later-good false alarms: gain-only monitor (3 log-RMS values) 24.3% [21.4, 27.4], shape-only 41.9% [38.4, 45.4], gain + shape 42.9%. Most of the drift is in how energy is distributed across frequency, which a sensor reseat or gain change cannot produce. The gain-only monitor drifts least, at the cost of some detection (68.6% of all bad parts against 80.0%; matched AUC 0.900 against 0.929).
- *Machine-level events.* Several timeframes show every operation on one machine moving the same way at once:
  - M03, Feb 2020: x-axis 100–178 Hz power rose by a median 5.2σ in 13 of 13 operations (sign test p = 2.4×10⁻⁴).
  - M01, Feb 2021: z-axis 18–100 Hz bands shifted in 12 of 12 operations (p = 4.9×10⁻⁴), with 10 features significant after Bonferroni.
  - M02, Aug 2020: x-axis 178–316 Hz rose by a median 39σ in 12 of 13 operations (p = 3.4×10⁻³).
  These are one-machine events (maintenance, spindle or fixture work), so they call for one re-baseline per machine, not 15 separate investigations.
- *Step or excursion.* M01 and M03 behave as steps: the first later timeframe already carries 76% and 81% of the largest shift. M02 is an excursion: its median shift rises from 2.6σ to 4.6σ in Aug 2020, then falls back to 2.0σ. The step and excursion labels are a fixed 70% rule applied to only 2–4 timeframes per machine, so they describe what was seen and are not a test.
- *Which band.* Ranked by median |z| over all cells and later timeframes, the top features are z-axis 18–32 Hz (4.7σ), 32–56 Hz (3.7σ) and 56–100 Hz (3.1σ, consistently down). In the most-shifted cell (M02 OP09, Feb 2021 against Aug 2019), mean z-axis power in 10–100 Hz is 0.92 decades lower (about 8× less), while 100–1000 Hz moves by only +0.08 decades.

**4. Ramp recipe (copying a baseline to a new machine, measured without drift).**

| Baseline on the new machine | False alarms on good parts | Detection, in-period bad |
|---|---|---|
| Copied unchanged | 68.9% [66.6, 71.1] | 91.7% |
| Copied loadings, re-centred on 10 parts | 21.9% [19.1, 25.0] | 72.6% |
| Copied loadings, re-centred + re-scaled on 10 consecutive parts | 7.6% [5.9, 9.7] | 67.9% |
| Copied loadings, re-centred + re-scaled on 10 parts spread across the run (500 draws) | **3.0% [1.7, 4.5]** | 60.5% [54.8, 67.9] |
| Full per-machine qualification (16–25 parts) | 0.8% [0.4, 1.3] | 69.0% [58.5, 77.9] |

The copied monitor's high detection is not a strength. It alarms on almost everything. Re-centring alone is not enough because each machine also has its own spread. Consecutive parts under-represent that spread, so the same 10 parts give 7.6% false alarms when consecutive and 3.0% when spread out. After a step, re-centring the machine's own monitor on its first 5 parts cuts false alarms in that timeframe from 39.9% to 15.6%, but in later timeframes from only 23.4% to 21.3%. Each new step needs its own re-baseline.

Scenario at 15 min of triage per alarm: a copied baseline costs 689 alarms and about 172 engineer-hours per 1,000 good parts. Re-centring alone on 10 parts costs 219 alarms (55 h). Re-centring and re-scaling on 10 spread parts costs 30 alarms (7.5 h). Full qualification costs 8 alarms (2 h). The 15-minute triage time is an assumption; the false-alarm rates are measured.

**Recommendation.**
1. **Bring-up:** copy the lead machine's PCA structure, then re-centre *and* re-scale on 10 good parts spread across the first runs, not 10 consecutive ones. Run a full 16–25 part qualification once the volume allows it. Screen the qualification set for runs that are off in several operations at once before setting limits.
2. **Re-baseline trigger:** the machine-level sign test. When most operations on one machine shift the same way in the same band, re-baseline that machine once.
3. **Chart:** T² + SPE with empirically calibrated limits (best matched AUC and detection), backed by the RMS chart as a drift-robust check. Do not use the theoretical SPE limit at n < 30.

## Assumptions and limitations
- Labels come from the dataset authors. A bad label means an anomalous process, not a measured dimensional defect. There are only 70 bad recordings (0–4 per cell); per-cell detection is not meaningful, so results are pooled with cluster-bootstrap intervals.
- The conformal multiplier assumes leave-one-out ratios are exchangeable across cells. The half-split check supports this (0.96% against 1%). The screening rule removes 36 of 857 qualification recordings; it was fixed before the scorecard was run, and unscreened numbers are reported above.
- Nyquist is 1 kHz and spindle speed and tool data are not in the dataset, so tooth-pass and spindle harmonics cannot be assigned. Band shifts are located but not attributed to a specific mechanism.
- Recording order within a timeframe is taken from the file index. Wall-clock timestamps are not available, so "first N parts" means first by index.
- Features are recording-level and ignore sub-operation segmentation. Timeframe sizes are uneven (Aug 2020 has 92 later good parts). The rolling re-baseline equals the static one in Feb 2020 by construction.

## Reproduce
```bash
uv venv .venv && uv pip install --python .venv/bin/python h5py numpy scipy scikit-learn pytest
git clone --depth 1 https://github.com/boschresearch/CNC_Machining ../downloads/bosch-cnc
.venv/bin/python -I pipeline/extract.py ../downloads/bosch-cnc/data pipeline/out/features.npz
.venv/bin/python -I pipeline/analyze.py pipeline/out/features.npz public/data/results.json
.venv/bin/python -I pipeline/diagnostics.py pipeline/out/features.npz public/data/results.json
.venv/bin/python -m pytest -q tests
```
