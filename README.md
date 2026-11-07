# CNC process-window lens

How fast does a CNC operation's vibration baseline go stale, why, how many parts does a new or drifted machine need before its limits can be trusted, and does an AI model add anything once the easy shortcuts are removed? Measured on two and a half years of real production data (Feb 2019 to Aug 2021) from three milling machines.

**Live demo:** DEMO_URL_PLACEHOLDER

## The problem
A machining process is qualified once, and its monitoring limits are set from that qualification run. Tools wear, fixtures get reworked and spindles age, and new machines get the lead machine's limits copied over during ramp-up. The limits then either miss real drift or bury the line in false alarms. Someone has to decide which alarms are real, for each operation and each machine, month after month.

## Headline
**Re-qualify, don't copy.**
- **Stale limits.** On 1,702 real recordings, every chart's 2019 limits went stale: 16–80% false alarms on later good parts, against a 5% design rate.
- **Twenty parts fix it, on average.** Re-centring each chart on the first 20 confirmed-good parts after a change, and recalibrating its limit, brought every chart back to 1.5–5.5% false alarms, pooled over 11 machine × periods. The 4.76% design rate comes from the conformal calibration, not from any model, and it is an average over re-qualifications: one 20-part re-qualification landed anywhere from 0% to 29% here, and even with ideal data 1 in 20 would exceed 13.9%.
- **AI without labels, no better than the best chart.** With every recording cut to the same window of its operation's NC program, a label-free MiniRocket nearest-neighbour model ranks bad parts about as well as the best classical chart: AUC 0.96–0.97 over 3 seeds against 0.96, not significantly better. It has never seen a bad part.
- **Recording length and start are not signals here.** Every bad file in this dataset is a trimmed excerpt. It is shorter than a normal cycle and starts a median 3.8 s later in the program. Length alone ranks bad parts at AUC 0.985, start offset alone at 0.94. So a "cycle-time chart" that seems to catch the most bad parts is reading how the files were cut, not the machine.

A gate written before the first evaluation run decided this headline. The AI model had to beat the best classical chart on like-for-like ranking without seeing cycle length, and catch clearly more bad parts than the RMS chart after qualification at equal false alarms, in all 3 seeds. Neither the supervised nor the label-free model passed. An adversarial review then found that the gate's duration-blind arm was blind to length but not to where a recording starts, so the models were re-run on program-aligned windows. They still do not pass there. The supervised model ranks bad parts almost perfectly on aligned windows (AUC 0.99–1.00), but after qualification, at equal false alarms, it catches 2–3 of 16 bad parts, the same as the RMS chart (3 of 16). Both arms are reported. The samples are small: the 16 bad parts made after qualification come from 6 machine × timeframe episodes, and the 68 like-for-like bad parts from 11. When the bootstrap resamples those whole episodes instead of single strata or parts, neither model passes either condition in any arm or seed.

## What it measures
Every rate below has a 95% interval; details, equations and every table are in METHOD.md. Plain (Wilson) intervals treat parts as independent. Parts of one machine and period share a limit and often a production episode, so those intervals are too narrow, and METHOD.md gives clustered intervals where it matters.
- **The shortcut.** The dataset's bad files are excerpts cut from longer recordings.
  - Every bad recording is shorter than its operation's normal cycle (median 0.84×). A "short cycle" rule alone ranks bad parts with AUC 0.985.
  - None of the 70 bad files has a length that is a multiple of the logger's 1,024-sample block; 92% of good files do.
  - Aligning each recording's vibration envelope to its operation's qualification template shows 66 of 70 bad files start more than 1 s late (median 3.8 s), against 12% of good files. Start offset alone ranks bad parts with AUC 0.94.
  - So the AI models are tested on windows at the same point of the program for every recording. A common start-anchored prefix, the first control tried, removes length but not the start offset.
- **Limits go stale.** With static per-machine conformal limits (4.4–4.7% false alarms inside qualification at a 5% design rate), false alarms on the 762 later good parts are:
  - 80% for the label-free nearest-neighbour model, the most drift-sensitive (93% on aligned windows);
  - 65% for PCA T²+SPE;
  - 30% for the RMS chart;
  - 24% for the supervised model;
  - 16% for the cycle-time chart.

  The textbook 3σ limit is miscalibrated even inside qualification: 2.2% leave-one-out false alarms against 0.27% nominal.
- **How many parts to re-qualify.** Split-conformal limits need at least 19 parts for a 5% false-alarm rate on average over re-qualifications (10 parts give 9%); a single re-qualification can land well above that.
  - With the first 20 good parts after a change, every re-centred chart lands at 1.5–5.5% false alarms pooled (2.0–7.4% on aligned windows, where the nearest-neighbour model sits at 6.6%). Per machine × period the rate ranges from 0% to 29%. Resampling the 11 machine × periods gives, for example, RMS 4.4% [0.2, 11.0] and the label-free model 5.3% [0.2, 13.0].
  - Detection of the 16 later bad parts on aligned windows is then 13 for the supervised model, 11 for the label-free model, 10 for the RMS chart and 9 for T²+SPE.
  - None of these differences is significant (label-free vs RMS: exact McNemar p = 1.0).
- **A new machine.** Charts are copied from the other two machines and re-qualified on 20 of the new machine's parts (aligned windows).
  - A re-centred copy of the RMS chart catches 2 of 27 and 0 of 24 bad parts.
  - The label-free model catches 6 of 27 (p = 0.13, not significant) and 13 of 24 (p = 0.0002). On the start-anchored prefix it had caught 15 of 27; most of that edge on the first machine was the start offset. The 0.0002 survives a Bonferroni correction over all 144 tests in the new-machine tables, but it treats the 24 bad recordings as independent, and they come from only 2 timeframes.
  - The supervised model, trained on the other machines' labelled history, catches 25 and 24.
- **Why baselines drift.** More of the drift is in spectral shape than in overall level: a shape-only monitor flags 42% of later good parts and a gain-only monitor 24%. Some shifts hit a whole machine at once: on one mill, x-axis 100–178 Hz power rose a median 5.2σ in 13 of 13 operations (sign test nominal p = 2.4×10⁻⁴). That points to one re-baseline rather than 13 investigations; maintenance records would be needed to confirm it.
- **What each alarm points to.** Each alarm gets the 0.5 s windows of the cycle that drove it, the known-good window each one matched, and the band whose power moved.
  - On one operation, all 4 bad parts carry +16 to +26 dB at 178–205 Hz against the nearest good windows at the same point of the program, while the RMS chart reads normal. This reproduces one of three band fingerprints found by an earlier autoencoder prototype; the other two do not.
- **What did not work.** A forward-in-time "nuisance drift vs out of window" label: drift and defect directions overlap (51–63% in later periods), and it labels only 15% of false alarms as drift. It is reported, not recommended.
- **Copying a classical baseline, measured without drift.** False alarms rise from about 1% to 68.9%. Re-centring and re-scaling the lead machine's PCA structure on 10 good parts per operation brings that to 7.6% with the first 10 parts, or 3.0% with 10 random parts.
- **Cost scenario** at an assumed 15 min of triage per alarm, per 1,000 good parts after a drift event: stale limits cost 41–200 engineer-hours depending on the chart; re-qualified on 20 parts, 3.7–13.4 h.

The current web page shows interim results tables; every number above is exported to `public/data/results.json` for the guided view.

## How it works
- `pipeline/features.py`: detrend, then per-axis RMS, peak, crest, kurtosis, Welch PSD band powers and spectral centroid
- `pipeline/spc.py`: PCA Hotelling T² (F-distribution UCL), SPE/Q (Jackson–Mudholkar limit) and a Shewhart RMS chart
- `pipeline/analyze.py` and `pipeline/diagnostics.py`: qualification baselines, drift, cross-machine transfer, conformal limit calibration, the matched scorecard, the gain/shape drift decomposition, the machine-level sign test and the ramp recipe
- `pipeline/align.py`: where in the NC program each recording starts, from its vibration envelope against the operation's qualification template (no labels)
- `pipeline/windows.py`: 24 raw 0.5 s windows per recording in three arms: the whole recording, a common start-anchored prefix per operation, and a common program-aligned window per operation (the duration and start-offset control), plus per-window band power
- `pipeline/embed.py`: MiniRocket embeddings (sktime, 9,996 features), with biases fitted on 2019 good windows of the other machines only (leave-one-machine-out), for 3 seeds
- `pipeline/detectors.py`: the supervised dual ridge, the label-free nearest-window memory bank, the RMS, T²+SPE and cycle-time charts, and hierarchical re-centring
- `pipeline/stats.py`: split-conformal limits with their finite-sample bound, the spread of a single calibration's false-alarm rate and the max-of-N fallback, stratified AUC, cluster bootstraps over strata or machine × timeframe blocks, paired bootstraps, Wilson intervals and the McNemar test (shared by diagnostics and the AI evaluation)
- `pipeline/drift.py`: the forward-in-time drift subspace and per-(axis, band) attribution
- `pipeline/ai_eval.py`: the gate, scorecards, drift, the re-qualification curve, the new-machine test and the per-alarm drill-down, written to static JSON
- `tests/`: 72 pytest checks, including the conformal guarantee by simulation (also for the re-centred, leave-one-out re-qualification), hand-checked limits, leakage guards, the duration-crop and alignment logic and regression against the repo's own numbers
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
