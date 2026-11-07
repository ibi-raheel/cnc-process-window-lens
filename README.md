# CNC process-window lens

How fast does a CNC operation's vibration baseline go stale, why, how many parts does a new or drifted machine need before its limits can be trusted, and does an AI model add anything once the easy shortcut is removed? Measured on two and a half years of real production data (Feb 2019 to Aug 2021) from three milling machines.

**Live demo:** DEMO_URL_PLACEHOLDER

## The problem
A machining process is qualified once, and its monitoring limits are set from that qualification run. Tools wear, fixtures get reworked and spindles age, and new machines get the lead machine's limits copied over during ramp-up. The limits then either miss real drift or bury the line in false alarms. Someone has to decide which alarms are real, for each operation and each machine, month after month.

## Headline
**Re-qualify, don't copy.**
- **Stale limits.** On 1,702 real recordings, every chart's 2019 limits went stale: 16–80% false alarms on later good parts, against a 5% design rate.
- **Twenty parts fix it.** Re-centring each chart on the first 20 confirmed-good parts after a change, and recalibrating its limit, brought every chart back to 2.6–5.5%. The exact finite-sample guarantee is 4.76%, and it comes from the split-conformal calibration, not from any model.
- **AI without labels.** A label-free MiniRocket nearest-neighbour model ranks bad parts as well as the best classical chart: AUC 0.98 against 0.96, not significantly better. It sees only the first 45% of each cycle and has never seen a bad part.
- **A free chart wins on later parts.** At equal 5% false alarms, a cycle-time chart caught the most post-qualification bad parts: 10 of 16, against 6 for the best AI head.

A gate written before the first evaluation run decided this headline. The AI model had to beat the best classical chart on like-for-like ranking without seeing cycle length, and catch clearly more bad parts than the RMS chart after qualification at equal false alarms, in all 3 seeds. Neither the supervised nor the label-free model passed both conditions. The supervised model did rank better (AUC 1.000 vs 0.965, Δ +0.035 [+0.002, +0.083]), but its post-qualification detection depended on the seed. Both are reported.

## What it measures
Every rate below has a 95% interval; details, equations and every table are in METHOD.md.
- **The shortcut.** Every bad recording is shorter than its operation's normal cycle (median 0.84×). A "short cycle" rule alone ranks bad parts with AUC 0.985, so the AI models are tested on a common start-anchored prefix of each cycle, where recording length carries no information.
- **Limits go stale.** With static per-machine conformal limits (4.4–4.8% false alarms inside qualification at a 5% design rate), false alarms on the 762 later good parts are:
  - 80% for the label-free nearest-neighbour model, the most drift-sensitive;
  - 65% for PCA T²+SPE;
  - 30% for the RMS chart;
  - 18% for the supervised model;
  - 16% for the cycle-time chart.

  The textbook 3σ limit is miscalibrated even inside qualification: 2.2% leave-one-out false alarms against 0.27% nominal.
- **How many parts to re-qualify.** Split-conformal limits need at least 19 parts for a guaranteed 5% (10 parts guarantee 9%).
  - With the first 20 good parts after a change, every re-centred chart lands at 2.6–5.5% false alarms.
  - Detection of the 16 later bad parts is then 14 for the label-free model, 13 for the supervised model, 11 for the RMS chart, and 10 each for T²+SPE and the cycle-time chart.
  - None of these differences is significant (label-free vs RMS: exact McNemar p = 0.25).
- **A new machine.** Charts are copied from the other two machines and re-qualified on 20 of the new machine's parts.
  - A re-centred copy of the RMS chart catches 2 of 27 and 2 of 24 bad parts.
  - The label-free model catches 15 and 14 (p ≤ 0.0005).
  - The free cycle-time chart catches 26 and 22.
  - The supervised model, trained on the other machines' labelled history, catches 25 and 24.
- **Why baselines drift.** More of the drift is in spectral shape than in overall level: a shape-only monitor flags 42% of later good parts and a gain-only monitor 24%. Some shifts hit a whole machine at once: on one mill, x-axis 100–178 Hz power rose a median 5.2σ in 13 of 13 operations (sign test nominal p = 2.4×10⁻⁴). That points to one re-baseline rather than 13 investigations; maintenance records would be needed to confirm it.
- **What each alarm points to.** Each alarm gets the 0.5 s windows of the cycle that drove it, the known-good window each one matched, and the band whose power moved.
  - On one operation, all 4 bad parts carry +15 to +29 dB at 178–205 Hz on all three axes while the RMS chart reads normal. This reproduces one of three band fingerprints found by an earlier autoencoder prototype; the other two do not reproduce.
- **What did not work.** A forward-in-time "nuisance drift vs out of window" label: drift and defect directions overlap (51–63% in later periods), and it labels only 15% of false alarms as drift. It is reported, not recommended.
- **Copying a classical baseline, measured without drift.** False alarms rise from about 1% to 68.9%. Re-centring and re-scaling the lead machine's PCA structure on 10 good parts per operation brings that to 7.6% with the first 10 parts, or 3.0% with 10 random parts.
- **Cost scenario** at an assumed 15 min of triage per alarm, per 1,000 good parts after a drift event: stale limits cost 41–200 engineer-hours depending on the chart; re-qualified on 20 parts, 6.5–13.4 h.

The current web page shows interim results tables; every number above is exported to `public/data/results.json` for the guided view.

## How it works
- `pipeline/features.py`: detrend, then per-axis RMS, peak, crest, kurtosis, Welch PSD band powers and spectral centroid
- `pipeline/spc.py`: PCA Hotelling T² (F-distribution UCL), SPE/Q (Jackson–Mudholkar limit) and a Shewhart RMS chart
- `pipeline/analyze.py` and `pipeline/diagnostics.py`: qualification baselines, drift, cross-machine transfer, conformal limit calibration, the matched scorecard, the gain/shape drift decomposition, the machine-level sign test and the ramp recipe
- `pipeline/windows.py`: 24 raw 0.5 s windows per recording in two arms, the whole recording and a common start-anchored prefix per operation (the duration control), plus per-window band power
- `pipeline/embed.py`: MiniRocket embeddings (sktime, 9,996 features), with biases fitted on 2019 good windows of the other machines only (leave-one-machine-out), for 3 seeds
- `pipeline/detectors.py`: the supervised dual ridge, the label-free nearest-window memory bank, the RMS, T²+SPE and cycle-time charts, and hierarchical re-centring
- `pipeline/stats.py`: split-conformal limits with the exact finite-sample bound and the max-of-N fallback, stratified AUC, cluster and paired bootstraps, Wilson intervals and the McNemar test (shared by diagnostics and the AI evaluation)
- `pipeline/drift.py`: the forward-in-time drift subspace and per-(axis, band) attribution
- `pipeline/ai_eval.py`: the gate, scorecards, drift, the re-qualification curve, the new-machine test and the per-alarm drill-down, written to static JSON
- `tests/`: 65 pytest checks, including the conformal guarantee by simulation, hand-checked limits, leakage guards, the duration-crop logic and regression against the repo's own numbers
- `METHOD.md`: equations, assumptions, validation, results and limitations

No API keys or GPU: the page reads precomputed JSON.

## Run locally
```bash
npm install
npm run dev
```
To rerun the pipeline, see the Reproduce section in METHOD.md (Python 3.12, `requirements.txt`).

## Data
Bosch CNC Machining dataset, https://github.com/boschresearch/CNC_Machining. The data is licensed CC-BY-4.0. Please cite: Tnani, Feil, Diepold, *Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring*, Procedia CIRP 107 (2022) 131–136, doi:10.1016/j.procir.2022.04.022. No simulated data is used in the analysis.

---
Built by Ibi Raheel, mechanical engineer building AI tools. More work at [ibiraheel.com](https://ibiraheel.com).
