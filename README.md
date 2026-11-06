# CNC process-window lens

How fast does a CNC operation's vibration baseline go stale, why, and how many parts does a new machine need before its limits can be trusted? Measured on two and a half years of real production data (Feb 2019 to Aug 2021) from three milling machines.

**Live demo:** DEMO_URL_PLACEHOLDER

## The problem
A machining process is qualified once, and its monitoring limits are set from that qualification run. Tools wear, fixtures get reworked and spindles age, and new machines get the lead machine's limits copied over during ramp-up. The limits then either miss real drift or bury the line in false alarms. Someone has to decide which alarms are real, for each operation and each machine, month after month.

## What it measures
On 1,702 real recordings (3 machines, 15 operations, Feb 2019 to Aug 2021). Every rate below has a 95% interval. Wilson intervals treat parts as independent, so they are likely narrower than an interval that accounts for clustering by cell. The limits are calibrated empirically to a 1% target: 0.96% false alarms on held-out cells (0.8% on the parts the multiplier was fitted on, after screening out 2 off-nominal qualification runs).
- **Drift:** a T² baseline set at 2019 qualification flags **41.5% [38.0–45.0] of later good parts**, against about 1% inside qualification.
- **Why:** more of the drift is in spectral shape than in overall vibration level. A gain-only monitor (3 inputs) flags 24% of later good parts and a shape-only monitor (6 inputs) 42%; the input counts differ, so this is indicative rather than a clean split. Some shifts appear on a whole machine at once. On one mill, x-axis 100–178 Hz power rose a median 5.2σ in 13 of 13 operations (sign test nominal p = 2.4×10⁻⁴, about 0.01 after correcting for the 40 features tested). That pattern points to one machine-level cause and one re-baseline rather than 13 separate investigations; maintenance records would be needed to confirm it.
- **Copying a baseline to another machine** (measured in the qualification period, so drift is not mixed in): false alarms rise from about 1% to **68.9%** (23% to 86% depending on the machine pair). Keeping the lead machine's PCA structure and re-centring and re-scaling it on 10 good parts from the new machine brings that down to **7.6%** when the parts are its first 10 in recording order, and to **3.0%** (range 1.7–4.5% over 500 draws) when they are drawn at random from its whole qualification period, with 60% detection against 69% for a full qualification. The random-draw figure assumes the 10 parts represent the whole qualification period, which a real ramp cannot guarantee.
- **Detector scorecard, like with like** (bad and good parts from the same timeframes, cluster-bootstrap intervals): PCA T²+SPE ROC-AUC 0.954 [0.927–0.977], RMS Shewhart 0.919 [0.841–0.971], Isolation Forest 0.857. The RMS minus T²+SPE difference is −0.034 [−0.100, +0.007], so neither chart is significantly better. The earlier "plain RMS chart wins" result does not hold on this like-for-like comparison. RMS only comes out ahead when good parts from later timeframes are included, most likely because it reacts least to drift.
- **Cost scenario** at 15 min of triage per alarm, per 1,000 good parts: copied baseline 172 engineer-hours, re-centred and re-scaled on 10 random qualification parts 7.4 h (19 h on the first 10 parts), full qualification 1.9 h. The triage time is an assumption.

The current page shows interim results tables computed with the original theoretical limits, which overstate drift; the calibrated numbers above are in METHOD.md. The guided visual view is in progress.

## How it works
- `pipeline/features.py`: detrend, then per-axis RMS, peak, crest, kurtosis, Welch PSD band powers and spectral centroid
- `pipeline/spc.py`: PCA Hotelling T² (F-distribution UCL), SPE/Q (Jackson–Mudholkar limit) and a Shewhart RMS chart
- `pipeline/analyze.py`: qualification baselines, drift with static and rolling baselines, cross-machine transfer (Wasserstein, MMD), detector scorecard, written to static JSON
- `pipeline/diagnostics.py`: split-conformal limit calibration, qualification-set screen, matched-timeframe scorecard with Wilson and cluster-bootstrap intervals, gain/shape drift decomposition, machine-level sign test, PSD overlay, and the ramp re-centring curve
- `tests/`: hand-calculation and signal-processing checks (pytest)
- `METHOD.md`: equations, assumptions, validation and limitations

## Run locally
```bash
npm install
npm run dev
```
To rerun the pipeline, see the Reproduce section in METHOD.md.

## Data
Bosch CNC Machining dataset, https://github.com/boschresearch/CNC_Machining. The data is licensed CC-BY-4.0. Please cite: Tnani, Feil, Diepold, *Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring*, Procedia CIRP 107 (2022) 131–136, doi:10.1016/j.procir.2022.04.022. No simulated data is used in the analysis.

---
Built by Ibi Raheel, mechanical engineer building AI tools. More work at [ibiraheel.com](https://ibiraheel.com).
