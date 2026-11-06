# CNC process-window lens

How fast does a CNC operation's vibration baseline go stale, and what does it cost to copy one machine's baseline to another? Measured on three years of real production data from three milling machines.

**Live demo:** DEMO_URL_PLACEHOLDER

## The problem
A machining process is qualified once, and its monitoring limits are set from that qualification run. Tools wear, fixtures get reworked and spindles age, and new machines get the lead machine's limits copied over during ramp-up. The limits then either miss real drift or bury the line in false alarms. Someone has to decide which alarms are real, for each operation and each machine, month after month.

## What it measures
On 1,702 real recordings (3 machines, 15 operations, Feb 2019 to Aug 2021):
- **Drift:** a baseline fitted on the 2019 good parts flags 75.9% of good parts in the next timeframe and 96.8% two years on. Re-baselining every six months gets this to 59.8–91.7%.
- **Transfer:** machine-to-machine distribution shift is significant in 41 of 41 operation pairs (MMD permutation test), and a copied baseline alarms on 96.9–100% of good parts.
- **Detector scorecard:** a plain RMS Shewhart chart ranks bad parts best (ROC-AUC 0.754), ahead of PCA Hotelling T²/SPE (0.689) and an Isolation Forest (0.666).

The current page shows interim results tables. The guided visual view is in progress.

## How it works
- `pipeline/features.py`: detrend, then per-axis RMS, peak, crest, kurtosis, Welch PSD band powers and spectral centroid
- `pipeline/spc.py`: PCA Hotelling T² (F-distribution UCL), SPE/Q (Jackson–Mudholkar limit) and a Shewhart RMS chart
- `pipeline/analyze.py`: qualification baselines, drift with static and rolling baselines, cross-machine transfer (Wasserstein, MMD), detector scorecard, written to static JSON
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
