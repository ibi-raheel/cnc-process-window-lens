# CNC process-window lens

How fast does a CNC operation's vibration baseline go stale, why, and how many parts does a new machine need before its limits can be trusted? Measured on three years of real production data from three milling machines.

**Live demo:** DEMO_URL_PLACEHOLDER

## The problem
A machining process is qualified once, and its monitoring limits are set from that qualification run. Tools wear, fixtures get reworked and spindles age, and new machines get the lead machine's limits copied over during ramp-up. The limits then either miss real drift or bury the line in false alarms. Someone has to decide which alarms are real, for each operation and each machine, month after month.

## What it measures
On 1,702 real recordings (3 machines, 15 operations, Feb 2019 to Aug 2021). Every rate below has a 95% interval, and the limits are calibrated empirically: leave-one-out false alarms inside qualification are 0.8%, and 0.96% when cross-validated across cells.
- **Drift:** a T² baseline set at 2019 qualification flags **41.5% [38.0–45.0] of later good parts**, against 0.8% inside qualification.
- **Why:** most of the drift is in spectral shape, not sensor gain. A gain-only monitor drifts 24% and a shape-only monitor 42%. Several drift events hit a whole machine at once. On one mill, x-axis 100–178 Hz power rose 5.2σ in 13 of 13 operations (sign test p = 2×10⁻⁴), which calls for one re-baseline, not 13 investigations.
- **Copying a baseline to another machine** (measured in the qualification period, so drift is not mixed in): false alarms rise from 0.8% to **68.9%**. Keeping the lead machine's PCA structure and re-centring and re-scaling it on **10 good parts spread across the first runs** brings that down to **3.0% [1.7–4.5]**, with 60% detection against 69% for a full qualification. Taking the same 10 parts consecutively gives 7.6%.
- **Detector scorecard, like with like** (bad and good parts from the same timeframes, cluster-bootstrap intervals): PCA T²+SPE ROC-AUC 0.954 [0.927–0.977], RMS Shewhart 0.919 [0.841–0.971], Isolation Forest 0.857. The earlier "plain RMS chart wins" result came from drift confounding. RMS only comes out ahead when good parts from later timeframes are included, because it reacts least to drift.
- **Cost scenario** at 15 min of triage per alarm, per 1,000 good parts: copied baseline 172 engineer-hours, 10-part re-centre and re-scale 7.5 h, full qualification 2 h.

The current page shows interim results tables. The guided visual view is in progress.

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
