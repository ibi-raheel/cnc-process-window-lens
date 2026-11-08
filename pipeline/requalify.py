"""requalify: the 20-part re-qualification recipe as a command-line tool.

A thin wrapper over the tested pipeline functions; it adds no new statistics.

    python -I pipeline/requalify.py --lead <lead qualification .h5: dir or glob> \
        --new <new or changed machine .h5: dir or glob> --n 20 [--alpha 0.05] \
        [--chart rms,t2spe] [--out requal.json] [--md requal.md]

--lead  qualification recordings of the lead machine (or of this machine before the change).
--new   recordings from the new or changed machine. They are put in production order
        (timeframe, part index, operation) and the first N that are not under a `bad/` folder are
        the confirmed-good reference parts. Everything else is scored.

Recipe, as measured in METHOD.md section 9 (first 20 good parts of a machine x period):
1. Lead structure per operation: Shewhart mean and spread of log RMS (detectors.shewhart_oos form)
   and a PCA T^2 + SPE monitor on the 9 monitored features (spc.fit_pca), with split-conformal
   multipliers on the lead's leave-one-out ratios (detectors.pca_oos, stats.conformal_multiplier).
2. Re-centre on the N reference parts: one machine-wide offset plus a per-operation correction
   shrunk by n_c / (n_c + kappa) (detectors.hier_shift). Calibration scores are leave-one-out
   (detectors.hier_shift_loo).
3. Limit: split conformal at alpha, or max-of-N below stats.min_n_for_alpha (stats.limit_with_fallback).
   The 5-95% range of what one re-qualification delivers comes from stats.conformal_far_quantiles.
4. Alarms (u = score / limit > 1) are ranked by u, each with the band that moved (drift.top_bands).
5. Machine-level call: for each (axis, band), the sign test across operations of the post-change
   mean shift (diagnostics.sign_test_p, as in METHOD.md's machine-level analysis).

Not included, on purpose: a per-alarm "drift vs defect" label (drift.label_alarm failed validation)
and a cycle-time chart (in the public dataset its detection reflects how the files were cut).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import MIN_TRAIN, model_features  # noqa: E402
from detectors import hier_shift, hier_shift_loo, pca_oos, shift_for, t2spe_shifted  # noqa: E402
from diagnostics import sign_test_p  # noqa: E402
from drift import band_names, top_bands  # noqa: E402
from extract import one  # noqa: E402
from features import BAND_EDGES, feature_names  # noqa: E402
from stats import conformal_far_quantiles, conformal_multiplier, limit_with_fallback, min_n_for_alpha  # noqa: E402

CHARTS = ("rms", "t2spe")
BANDS = [f"{a}_band{i}_log" for a in "xyz" for i in range(len(BAND_EDGES) - 1)]
BAND_LABELS = band_names(BAND_EDGES)
MACHINE_SHARE = 0.8      # share of operations that must move the same way
MIN_OPS, MIN_PARTS = 5, 3


# ---------------------------------------------------------------- core (array in, array out)
def lead_stats(x: np.ndarray, ops: np.ndarray) -> dict:
    """op -> (mean, sample sd) of a per-part value on the lead's qualification parts."""
    return {o: (float(x[ops == o].mean()), float(max(x[ops == o].std(ddof=1), 1e-12))) for o in np.unique(ops)}


def rms_requal(stats_: dict, x_new, op_new, x_eval, op_eval):
    """Re-centred Shewhart RMS chart (the same arithmetic as ai_eval.Requal.scores).
    Returns (scores of the parts to evaluate, leave-one-out calibration scores of the N parts)."""
    mu = lambda ops: np.array([stats_[o][0] for o in ops])  # noqa: E731
    sd = lambda ops: np.array([stats_[o][1] for o in ops])  # noqa: E731
    off = np.asarray(x_new, float) - mu(op_new)
    d, per = hier_shift(off, op_new)
    test = np.abs(np.asarray(x_eval, float) - mu(op_eval) - shift_for(op_eval, d, per)) / (3 * sd(op_eval))
    calib = np.abs(off - hier_shift_loo(off, op_new)) / (3 * sd(op_new))
    return test, calib


def t2spe_requal(mods: dict, c_t2: float, c_spe: float, X_new, op_new, X_eval, op_eval):
    """Re-centred PCA T^2 + SPE with the lead's loadings (ai_eval.Requal.scores, t2spe branch)."""
    off = np.stack([X_new[j] - mods[o].mu for j, o in enumerate(op_new)])
    d, per = hier_shift(off, op_new)
    test = np.empty(len(op_eval))
    for o in np.unique(op_eval):
        m = op_eval == o
        test[m] = t2spe_shifted(mods[o], X_eval[m], per.get(o, d), c_t2, c_spe)
    loo = hier_shift_loo(off, op_new)
    calib = np.array([t2spe_shifted(mods[o], X_new[j][None, :], loo[j], c_t2, c_spe)[0] for j, o in enumerate(op_new)])
    return test, calib


def limit_report(calib, alpha: float) -> dict:
    lim, info = limit_with_fallback(calib, alpha)
    n = len(calib)
    spread = conformal_far_quantiles(n, alpha) if n else None
    note = None
    if n < min_n_for_alpha(alpha):
        note = f"no {alpha:.0%} limit exists with {n} parts; this limit buys {1 / (n + 1):.1%} (max-of-N)"
    return {"limit": lim, "rule": info["rule"], "N": n, "promised_far": info["guarantee"],
            "far_5_95": [spread["0.05"], spread["0.95"]] if spread else None, "note": note}


def machine_calls(z_by_op: dict, names=BAND_LABELS, alpha_family: float = 0.05) -> list:
    """z_by_op: op -> per-band mean shift of its post-change parts, in lead sd units.
    A band is a machine-wide shift when most operations move the same way (sign test,
    Bonferroni over bands, and at least MACHINE_SHARE of operations agreeing)."""
    if len(z_by_op) < MIN_OPS:
        return []
    V = np.array(list(z_by_op.values()))
    out = []
    for j, name in enumerate(names):
        col = V[:, j][np.isfinite(V[:, j]) & (V[:, j] != 0)]
        if not len(col):
            continue
        med = float(np.median(col))
        k = int((np.sign(col) == np.sign(med)).sum())
        p = sign_test_p(k, len(col))
        wide = p < alpha_family / len(names) and k / len(col) >= MACHINE_SHARE
        out.append({"band": name, "median_z": round(med, 2), "agree": k, "n_ops": int(len(col)), "p": p,
                    "call": "machine-wide shift: re-baseline once, confirm with the maintenance log" if wide else None})
    out.sort(key=lambda r: r["p"])
    return out


# ---------------------------------------------------------------- I/O
def files_of(spec: str) -> list[str]:
    p = Path(spec)
    return sorted(str(f) for f in p.rglob("*.h5")) if p.is_dir() else sorted(glob.glob(spec, recursive=True))


def read(files: list[str]) -> dict:
    metas, X40 = zip(*(one(f) for f in files))
    X40 = np.vstack(X40)
    names = feature_names()
    return {"path": np.array(files), "op": np.array([m["op"] for m in metas]),
            "order": np.array([(m["tf_index"], m["idx"]) for m in metas]),
            "known_bad": np.array([m["label"] for m in metas]), "X9": model_features(X40)[0],
            "lrms": np.log10([m["rms_vec"] for m in metas]), "B": X40[:, [names.index(b) for b in BANDS]]}


def run(lead_files, new_files, n: int, alpha: float, charts=CHARTS) -> dict:
    L, W = read(lead_files), read(new_files)
    keep_ops = [o for o in np.unique(L["op"]) if (L["op"] == o).sum() >= MIN_TRAIN]
    lead = {k: v[np.isin(L["op"], keep_ops)] for k, v in L.items()}
    W = {k: v[np.isin(W["op"], keep_ops)] for k, v in W.items()}
    order = np.lexsort((W["op"], W["order"][:, 1], W["order"][:, 0]))      # production order
    good = order[W["known_bad"][order] == 0]
    ref, rest = good[:n], np.setdiff1d(np.arange(len(W["op"])), good[:n])
    op_r, op_e = W["op"][ref], W["op"][rest]

    scores, limits = {}, {}
    if "rms" in charts:
        scores["rms"], cal = rms_requal(lead_stats(lead["lrms"], lead["op"]), W["lrms"][ref], op_r, W["lrms"][rest], op_e)
        limits["rms"] = limit_report(cal, alpha)
    if "t2spe" in charts:
        idx = np.arange(len(lead["op"]))
        mods, t2, spe = {}, [], []
        for o in keep_ops:
            ii = idx[lead["op"] == o]
            a, b, mods[o] = pca_oos(lead["X9"], ii, ii)
            t2.append(a); spe.append(b)
        c_t2, c_spe = conformal_multiplier(np.concatenate(t2), 0.01), conformal_multiplier(np.concatenate(spe), 0.01)
        scores["t2spe"], cal = t2spe_requal(mods, c_t2, c_spe, W["X9"][ref], op_r, W["X9"][rest], op_e)
        limits["t2spe"] = limit_report(cal, alpha)

    bstat = {o: (lead["B"][lead["op"] == o].mean(0), np.maximum(lead["B"][lead["op"] == o].std(0, ddof=1), 1e-9)) for o in keep_ops}
    bmu = lambda ops: np.array([bstat[o][0] for o in ops])  # noqa: E731
    bsd = lambda ops: np.array([bstat[o][1] for o in ops])  # noqa: E731
    d, per = hier_shift(W["B"][ref] - bmu(op_r), op_r)
    expected = bmu(op_e) + shift_for(op_e, d, per)
    z_part = (W["B"][rest] - expected) / bsd(op_e)

    alarms = []
    for c in scores:
        u = scores[c] / limits[c]["limit"]
        for j in np.where(u > 1)[0]:
            tb = top_bands(z_part[j], BAND_LABELS, 1)[0]
            b = BAND_LABELS.index(tb["band"])
            alarms.append({"file": Path(W["path"][rest][j]).name, "op": str(op_e[j]), "chart": c, "u": round(float(u[j]), 2),
                           "top_band": tb["band"], "z": tb["z"], "dB": round(float(10 * (W["B"][rest][j, b] - expected[j, b])), 1)})
    alarms.sort(key=lambda a: -a["u"])

    allW = np.arange(len(W["op"]))
    z_by_op = {o: ((W["B"][allW[W["op"] == o]] - bstat[o][0]) / bstat[o][1]).mean(0)
               for o in keep_ops if (W["op"] == o).sum() >= MIN_PARTS}
    calls = machine_calls(z_by_op)
    wide = [r for r in calls if r["call"]]
    ops = []
    for o in keep_ops:
        ops.append({"op": str(o), "lead_parts": int((lead["op"] == o).sum()), "reference_parts": int((op_r == o).sum()),
                    "scored_parts": int((op_e == o).sum()),
                    "alarms": {c: int(sum(1 for a in alarms if a["op"] == o and a["chart"] == c)) for c in scores},
                    "call": (None if not any(a["op"] == o for a in alarms)
                             else "machine-wide: re-baseline" if wide else "isolated: investigate")})
    out = {"N": int(len(ref)), "alpha": alpha, "charts": limits, "operations": ops, "alarms": alarms,
           "machine_calls": calls[:5], "machine_wide": bool(wide),
           "verdict": wide[0]["call"] + f" ({wide[0]['band']}, {wide[0]['agree']} of {wide[0]['n_ops']} operations)" if wide
           else "no machine-wide shift: treat each alarm as isolated and investigate it"}
    if W["known_bad"][rest].any():   # only when the file layout already marks known-bad parts (public dataset)
        y = W["known_bad"][rest]
        out["check_against_known_labels"] = {c: {"false_alarms": [int((scores[c][y == 0] > limits[c]["limit"]).sum()), int((y == 0).sum())],
                                                 "caught": [int((scores[c][y == 1] > limits[c]["limit"]).sum()), int((y == 1).sum())]}
                                             for c in scores}
    return out


def markdown(r: dict) -> str:
    L = [f"# Re-qualification on {r['N']} parts", "", f"**Machine-level call:** {r['verdict']}.", "",
         "| Chart | Rule | Limit (score) | False-alarm rate this N buys | One re-qualification, 5–95% |", "|---|---|---|---|---|"]
    for c, x in r["charts"].items():
        rng = f"{x['far_5_95'][0]:.1%}–{x['far_5_95'][1]:.1%}" if x["far_5_95"] else "n/a"
        L.append(f"| {c} | {x['rule']} | {x['limit']:.3f} | {x['promised_far']:.2%} | {rng} |")
    notes = [x["note"] for x in r["charts"].values() if x["note"]]
    L += ([""] + [f"> {n}" for n in dict.fromkeys(notes)]) if notes else []
    L += ["", "| Operation | Lead parts | Reference parts | Scored | Alarms | Call |", "|---|---|---|---|---|---|"]
    for o in r["operations"]:
        L.append(f"| {o['op']} | {o['lead_parts']} | {o['reference_parts']} | {o['scored_parts']} | "
                 f"{', '.join(f'{c} {k}' for c, k in o['alarms'].items())} | {o['call'] or ''} |")
    L += ["", "## Alarm queue (highest first)", "", "| File | Op | Chart | u | Band that moved | dB |", "|---|---|---|---|---|---|"]
    L += [f"| {a['file']} | {a['op']} | {a['chart']} | {a['u']} | {a['top_band']} | {a['dB']:+} |" for a in r["alarms"][:25]]
    return "\n".join(L) + "\n"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lead", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--chart", default="rms,t2spe")
    ap.add_argument("--out", default="requal.json")
    ap.add_argument("--md", default="requal.md")
    a = ap.parse_args(argv)
    charts = tuple(c for c in a.chart.split(",") if c in CHARTS)
    r = run(files_of(a.lead), files_of(a.new), a.n, a.alpha, charts)
    Path(a.out).write_text(json.dumps(r, indent=1, default=float))
    Path(a.md).write_text(markdown(r))
    print(markdown(r))
    return r


if __name__ == "__main__":
    main()
