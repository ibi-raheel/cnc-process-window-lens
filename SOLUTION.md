# The solution: a 20-part re-qualification recipe

**Question.** A CNC operation moves to a new machine or a second supplier, or comes back from a tool, fixture or spindle change. How many good parts do you need before its control limits can be trusted again, and how do you tell harmless drift from a real process change?

**Answer.** About 20. Copy the chart, never the limits. Re-fit on the machine's own first good parts after the change and let the calibration set the limit. Every number below is measured on 1,702 real production recordings from three CNC mills (2019–2021) and is exported to `public/data/results.json`. METHOD.md has the equations, intervals and caveats.

## What the data says

| Situation | False alarms on good parts | Design rate |
|---|---|---|
| Qualified limits copied unchanged to a sister machine (no drift involved) | **68.9%** [66.6, 71.1], 1,089 of 1,581 | 1% |
| The same machine's own qualification | 0.76%, 12 of 1,581 | 1% |
| Copied chart re-centred only, on 10 parts per operation | 21.9% | 1% |
| Copied chart re-centred and re-scaled on the first 10 parts per operation | **7.6%** (3.0% with 10 random parts) | 1% |
| 2019 limits on the 762 good parts made later | **16–80%**, by chart (93% on program-aligned windows) | 5% |
| Re-centred on the first 20 good parts after a change, split-conformal limit | **1.5–5.4%** pooled over 11 machine × periods (2.0–6.6% on aligned windows) | 4.76% |

Pooled over re-qualifications the 20-part limit holds, but one re-qualification scatters: 0% to 29% per machine × period here, and even for ideal data the 95th percentile of a single 20-part limit is 13.9%.

## The recipe

**Trigger:** a new machine or second supplier on an operation, or a machine-level event: tool, fixture or spindle change, preventive maintenance, relocation.

1. **Copy the chart, never the limits.** Carry over the lead machine's chart structure (features, PCA loadings) and throw away its centre, spread and limit. Copied limits flagged 69% of good parts.
2. **Bring-up of a new machine or supplier:** re-centre *and* re-scale the copied chart on at least 10 good parts per operation, sampled as widely across the early runs as you can. The first 10 parts gave 7.6%; re-centring alone gave 21.9%. Run a full qualification once volume allows.
3. **After a change on a running machine:** take the first 20 confirmed-good parts off the machine (across its operations, so 1–2 per operation is enough), re-centre each operation (one machine-wide shift plus a shrunk per-operation correction) and set one split-conformal limit at 5%. Every chart came back to 1.5–5.4% pooled.
4. **Only 10 parts? Quote 9%, not 5%.** Below 19 parts no split-conformal limit at 5% exists; the max-of-N limit on 10 parts buys 1/11 = 9.09%.
5. **Pool before you judge a limit.** One re-qualification scatters (95th percentile 13.9% at 20 parts). Pool alarms over several re-qualifications before calling a limit wrong.
6. **Triage at machine level first.** If the same band shifts the same way on most operations of one machine (sign test), do one re-baseline and check the maintenance log. If only one operation moved, investigate it, starting from the band that moved.
7. **Check the data before trusting a signal.** Recording length and start offset can encode how files were logged rather than the process. In the public data they ranked bad parts at AUC 0.985 and 0.94, so the free cycle-time chart is logged as an honest baseline, not as evidence.
8. **Drop textbook 3σ.** Leave-one-out it flagged 2.2% of qualification parts against 0.27% nominal. Calibrate limits from the data.

### Decision rules and thresholds

| Decision | Rule | Where the threshold comes from |
|---|---|---|
| Parts for a 5% limit | N ≥ 19 (use 20) | split-conformal rank ⌈(N+1)(1−α)⌉ ≤ N; at N = 20 the limit is the largest score and buys 1/21 = 4.76% |
| Fewer parts | quote 1/(N+1): 10 parts → 9.09% | max-of-N limit |
| Is one re-qualification off? | only if its rate is beyond ~14% at N = 20; otherwise pool | Beta(1, 20) 95th percentile 13.9% |
| Machine-wide shift | a band moves the same way on ≥ 80% of operations, sign test p < 0.05 / 24 bands | on one mill x-axis 100–178 Hz rose a median 5.2σ in 13 of 13 operations (p = 2.4×10⁻⁴): one re-baseline, not 13 investigations |
| Isolated alarm | u = score ÷ limit > 1 on an operation that is not part of a machine-wide shift | investigate, starting from the band with the largest |z| |

### What the AI adds, honestly
A label-free MiniRocket nearest-neighbour model of good windows needs no defect history: append confirmed-good parts, no retraining. On program-aligned windows it ranks bad parts at AUC 0.962–0.972 over 3 seeds, against 0.960 for the best classical chart. That matches the best chart; it does not beat it, and the gate declared before the first run was not passed. On a new machine it caught 13 of 24 bad parts where the copied RMS chart caught 0 (p = 0.0002), but only 6 of 27 against 2 on the other machine (p = 0.13). Treat it as a candidate chart for a new machine, not a proven one.

### What it costs
At an assumed 15 minutes of triage per alarm, per 1,000 good parts: a copied baseline costs about 172 engineer-hours; re-centring and re-scaling on the first 10 parts costs 19 h. The triage time is an assumption; the false-alarm rates are measured.

## The tool: `requalify`

A thin command-line wrapper over the tested pipeline functions (`pipeline/requalify.py`). It adds no new statistics.

```bash
.venv/bin/python -I pipeline/requalify.py \
  --lead "../_downloads/bosch-cnc/repo/data/M01/*/good/M01_*_2019_*.h5" \
  --new  "../_downloads/bosch-cnc/repo/data/M01/*/*/M01_Feb_2021_*.h5" \
  --n 20 --alpha 0.05 --chart rms,t2spe --out requal.json --md requal.md
```

- `--lead`: qualification recordings of the lead machine, or of this machine before the change (a folder or a glob).
- `--new`: recordings from the new or changed machine. They are put in production order (period, part index, operation); the first N that are not under a `bad/` folder are the confirmed-good reference parts and everything else is scored.
- Input format: the dataset's own `M0x_<Mon>_<YYYY>_OPxx_<n>.h5` files with a `vibration_data` (samples × 3) array at 2 kHz. The limit arithmetic works on any per-part score.

It returns, as `requal.json` and a one-page `requal.md`:
- per chart: the limit, the rule (conformal or max-of-N), the false-alarm rate N actually buys and the 5–95% range of a single re-qualification, with a plain warning when N is below 19;
- per operation: lead, reference and scored parts, and its alarms;
- a ranked alarm queue (u = score ÷ limit), each alarm with the band that moved and by how many dB;
- a machine-level call: "machine-wide shift: re-baseline once" or "isolated: investigate".

Charts: the re-centred RMS chart and PCA T²+SPE, because the 20-part result does not depend on the chart. Out of scope, on purpose: a per-alarm "drift vs defect" label (it labelled only 15% of false alarms as drift and failed validation), and a cycle-time chart (its detection here measures how the files were cut). The CLI scores whole recordings; the published tables also report the start-anchored and program-aligned windows.

Tests: `tests/test_requalify.py` checks the 20-part and 10-part limits, the machine-wide call (a shift on every operation triggers it, a shift on one does not), and replays the published 20-part RMS row through the wrapper: 24 false alarms on 542 later good parts, exactly as in `results.json`.

## Week 1 on a real line: one operation, one machine, one recipe

- **Day 1.** Pick one CNC operation on a machine that recently came up, or went through a tool, fixture or spindle change, with a few months of per-part history: spindle load or vibration, touch-probe readings, or CMM dimensions. The limit arithmetic works on any per-part score.
- **Day 2.** Data hygiene. Check whether record length, start time or batch boundaries give away the label (the trap found in the public data). Map the files to the tool's input and mark the change events from the maintenance log.
- **Day 3.** Replay history. For each past change event, measure the false-alarm rate the existing limits actually gave on later good parts, against a 20-part re-qualification, both with intervals.
- **Day 4.** Run the machine-level check across all operations on that machine: one re-baseline, or real investigations? Reconcile it with the maintenance log and walk it through with the engineer who owns the process.
- **Day 5.** Hand over the recipe filled in with the line's own numbers: the limits file for that machine, the part count each upcoming bring-up needs (and what 10 parts versus 20 actually buys), and a short list of alarms worth investigating.

---
Built on the public Bosch CNC Machining dataset (CC BY 4.0). Ibi Raheel · [ibiraheel.com](https://ibiraheel.com)
