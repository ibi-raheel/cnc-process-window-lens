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
`pipeline/extract.py` → `pipeline/out/features.npz` → `pipeline/analyze.py` → `public/data/results.json`

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

## Validation
| Check | Expected | Result |
|---|---|---|
| Sine of amplitude A: RMS = A/√2 | within 0.1% | pass (`test_sine_rms_is_amplitude_over_root2`) |
| Parseval: ∫Welch PSD df = variance | within 2% | pass (`test_parseval_welch_integral_equals_variance`) |
| T² UCL hand calculation, p = 3, n = 20, α = 0.01: 3·19·21/(20·17) · F₀.₉₉(3,17) = 3.5206 · 5.185 = **18.25** | 18.25 | code 18.25, pass |
| JM Q limit with one residual eigenvalue ≈ λ·χ²₀.₉₉(1) | within 10% | pass |
| T² empirical false-alarm rate on Gaussian data, n = 40 | ≈ 1% | < 3.5%, pass |
| **Leave-one-out false-alarm rate on real training good parts** (857 held-out fits) | T² 1%, combined ≈ 2% | **T² 3.3%, SPE 10.4%, combined 11.6%** |
| ML detector compared only against the RMS chart and T²/SPE on the same held-out set | | see results |

The leave-one-out row is the honest calibration check. T² is close to nominal (3.3% against 1%). The SPE limit is anti-conservative at this sample size: Jackson–Mudholkar assumes the covariance is known, and with n = 16–25 the residual eigenvalues are underestimated. Part of the later false-alarm rate is therefore calibration, not drift. Even the T²-only static false-alarm rate on later good parts is 44.2%, against 3.3% within the qualification period.

## Results (measured; reproduced by `pipeline/analyze.py`)
- **A baseline goes stale fast.** Static-baseline false-alarm rate on good parts: Feb 2020 75.9% → Aug 2020 83.7% → Feb 2021 96.8% → Aug 2021 90.1%. In-qualification leave-one-out: 11.6%.
- **Re-baselining helps but does not fix it.** Rolling re-baseline gives 75.9%, 59.8%, 91.7% and 62.2% for the same timeframes.
- **Machines differ.** The MMD permutation test finds a significant shift (p < 0.05) in 41 of 41 operation pairs between machines. Copying a baseline to another machine raises the false-alarm rate to 96.9–100%, against a same-machine 67.1% (M03), 94.4% (M01) and 100% (M02).
- **Plain SPC wins on ranking.** ROC-AUC: RMS Shewhart 0.754, PCA T²/SPE 0.689, Isolation Forest 0.666. PR-AUC: 0.194, 0.141 and 0.201 (prevalence 0.084). At its fixed limit, PCA detects 98.6% of bad parts only because it alarms on 87.5% of good ones.

## Assumptions and limitations
- Labels come from the dataset authors; a bad label means an anomalous process, not a measured dimensional defect. There are only 70 bad recordings, 0–4 per cell, so per-cell detection numbers are not meaningful and only pooled numbers are reported.
- The qualification set is small (16–25 recordings per cell), so limits from F and Jackson–Mudholkar theory under-cover, as the leave-one-out check shows. A production version would set limits empirically (bootstrap or conformal) or pool across operations.
- The features are recording-level and ignore sub-operation segmentation and the tool-path phase.
- The timeframe sizes are uneven (Aug 2019 has 738 recordings; Aug 2020 has 99).

## Reproduce
```bash
uv venv .venv && uv pip install --python .venv/bin/python h5py numpy scipy scikit-learn pytest
git clone --depth 1 https://github.com/boschresearch/CNC_Machining ../downloads/bosch-cnc
.venv/bin/python -I pipeline/extract.py ../downloads/bosch-cnc/data pipeline/out/features.npz
.venv/bin/python -I pipeline/analyze.py pipeline/out/features.npz public/data/results.json
.venv/bin/python -m pytest -q tests
```
