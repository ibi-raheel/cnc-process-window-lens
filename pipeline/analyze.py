"""Stage 3: baselines, drift, transfer and detector scorecard -> static JSON for the UI.
Usage: python -I pipeline/analyze.py pipeline/out/features.npz public/data/results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FS, TIMEFRAMES, feature_names, load, welch_psd  # noqa: E402
from spc import ALPHA, fit_pca, fit_shewhart  # noqa: E402

MIN_TRAIN = 15


def model_features(X40: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Compact monitored vector (p = 9) so that n_train > p: per axis log RMS,
    log band power 10-100 Hz (bands 0-3) and 100-1000 Hz (bands 4-7)."""
    names40 = feature_names()
    cols, names = [], []
    for a in ("x", "y", "z"):
        cols.append(X40[:, names40.index(f"{a}_log_rms")]); names.append(f"{a}_log_rms")
        for lab, rng in (("lo", range(0, 4)), ("hi", range(4, 8))):
            v = np.stack([X40[:, names40.index(f"{a}_band{i}_log")] for i in rng], axis=1)
            cols.append(np.log10(np.sum(10.0 ** v, axis=1))); names.append(f"{a}_log_band_{lab}")
    return np.stack(cols, axis=1), names
MACHINES = ["M01", "M02", "M03"]
OPS = [f"OP{i:02d}" for i in range(15)]
RNG = np.random.default_rng(0)


def pick_train(tf: np.ndarray, label: np.ndarray, allowed_tfs) -> np.ndarray:
    """Good recordings from the given timeframes, taken whole, in order, until n >= MIN_TRAIN."""
    sel = np.zeros(len(tf), bool)
    for t in allowed_tfs:
        sel |= (tf == t) & (label == 0)
        if sel.sum() >= MIN_TRAIN:
            break
    return sel


class Models:
    def __init__(self, X, rms):
        self.pca = fit_pca(X)
        self.shew = fit_shewhart(np.log10(rms))
        Z = (X - self.pca.mu) / self.pca.sd
        self.iso = IsolationForest(n_estimators=200, random_state=0).fit(Z)
        s = -self.iso.score_samples(Z)
        self.iso_lim = float(np.quantile(s, 1 - ALPHA))

    def ratios(self, X, rms):
        Z = (X - self.pca.mu) / self.pca.sd
        return {"pca": self.pca.ratio(X),
                "rms": self.shew.ratio(np.log10(rms)),
                "iso": -self.iso.score_samples(Z) / self.iso_lim}


def mmd_rbf(A: np.ndarray, B: np.ndarray, n_perm: int = 200):
    Xall = np.vstack([A, B])
    d2 = np.sum((Xall[:, None, :] - Xall[None, :, :]) ** 2, axis=-1)
    gamma = 1.0 / np.median(d2[d2 > 0])
    K = np.exp(-gamma * d2)
    na = len(A)

    def stat(idx):
        a, b = idx[:na], idx[na:]
        return K[np.ix_(a, a)].mean() + K[np.ix_(b, b)].mean() - 2 * K[np.ix_(a, b)].mean()

    base = np.arange(len(Xall))
    obs = stat(base)
    perms = [stat(RNG.permutation(base)) for _ in range(n_perm)]
    p = (1 + sum(v >= obs for v in perms)) / (n_perm + 1)
    return float(obs), float(p)


def main(npz: str, out: str):
    D = np.load(npz, allow_pickle=False)
    X, machine, op, tf, label, rms, path = (D[k] for k in
                                             ("X", "machine", "op", "tf", "label", "rms", "path"))
    X, names = model_features(X)
    cells, chart, loo = [], {}, {"t2": [], "spe": [], "comb": []}
    pooled = {k: {"score": [], "y": []} for k in ("pca", "rms", "iso")}
    far_tf = {"static": [[0, 0] for _ in TIMEFRAMES], "rolling": [[0, 0] for _ in TIMEFRAMES]}
    det_static = [0, 0]
    static_models = {}

    for m in MACHINES:
        for o in OPS:
            idx = np.where((machine == m) & (op == o))[0]
            if len(idx) == 0:
                continue
            t, y = tf[idx], label[idx]
            tr = pick_train(t, y, range(len(TIMEFRAMES)))
            if tr.sum() < MIN_TRAIN:
                cells.append({"machine": m, "op": o, "n": int(len(idx)), "skipped": True})
                continue
            train_tfs = sorted(set(t[tr].tolist()))
            last_train = max(train_tfs)
            mod = Models(X[idx[tr]], rms[idx[tr]])
            static_models[(m, o)] = (mod, idx[tr])
            r = mod.ratios(X[idx], rms[idx])

            # leave-one-out empirical false-alarm rate inside the training set
            tri = idx[tr]
            for j in range(len(tri)):
                keep = np.delete(tri, j)
                pm = fit_pca(X[keep])
                t2, spe = pm.scores(X[tri[j]][None, :])
                loo["t2"].append(bool(t2[0] > pm.t2_lim))
                loo["spe"].append(bool(spe[0] > pm.spe_lim))
                loo["comb"].append(bool(t2[0] > pm.t2_lim or spe[0] > pm.spe_lim))

            evalm = ~tr & ((t > last_train) | (y == 1))
            for k in pooled:
                pooled[k]["score"] += r[k][evalm].tolist()
                pooled[k]["y"] += y[evalm].tolist()

            # rolling re-baseline: train on the most recent earlier timeframes
            roll = np.full(len(idx), np.nan)
            for tt in sorted(set(t.tolist())):
                if tt <= last_train:
                    continue
                prev = [p for p in range(tt - 1, -1, -1)]
                rtr = pick_train(t, y, prev)
                if rtr.sum() < MIN_TRAIN:
                    continue
                pm = fit_pca(X[idx[rtr]])
                sel = t == tt
                roll[sel] = pm.ratio(X[idx[sel]])

            good_post = (y == 0) & (t > last_train)
            for tt in range(len(TIMEFRAMES)):
                g = good_post & (t == tt)
                far_tf["static"][tt][0] += int((r["pca"][g] > 1).sum())
                far_tf["static"][tt][1] += int(g.sum())
                gr = g & ~np.isnan(roll)
                far_tf["rolling"][tt][0] += int((roll[gr] > 1).sum())
                far_tf["rolling"][tt][1] += int(gr.sum())
            bad = y == 1
            det_static[0] += int((r["pca"][bad] > 1).sum())
            det_static[1] += int(bad.sum())

            per_tf = []
            for tt in range(len(TIMEFRAMES)):
                g = good_post & (t == tt)
                gr = g & ~np.isnan(roll)
                per_tf.append({
                    "n_good": int(g.sum()),
                    "far_static": None if g.sum() == 0 else float((r["pca"][g] > 1).mean()),
                    "far_rolling": None if gr.sum() == 0 else float((roll[gr] > 1).mean()),
                })
            cells.append({
                "machine": m, "op": o, "n": int(len(idx)), "n_bad": int(bad.sum()),
                "n_train": int(tr.sum()), "train_tfs": train_tfs, "k": mod.pca.k,
                "t2_ucl": mod.pca.t2_lim, "far_t2": float((mod.pca.scores(X[idx][good_post])[0] > mod.pca.t2_lim).mean()) if good_post.any() else None, "spe_lim": mod.pca.spe_lim,
                "far_static": float((r["pca"][good_post] > 1).mean()) if good_post.any() else None,
                "far_rolling": float((roll[good_post & ~np.isnan(roll)] > 1).mean())
                if (good_post & ~np.isnan(roll)).any() else None,
                "n_good_post": int(good_post.sum()),
                "det_static": float((r["pca"][bad] > 1).mean()) if bad.any() else None,
                "per_tf": per_tf, "skipped": False,
            })
            chart[f"{m}-{o}"] = [
                {"tf": int(t[j]), "y": int(y[j]), "train": bool(tr[j]),
                 "s": round(float(r["pca"][j]), 4),
                 "r": None if np.isnan(roll[j]) else round(float(roll[j]), 4)}
                for j in np.argsort(t, kind="stable")
            ]

    # ---- scorecard
    score = {}
    for k, v in pooled.items():
        s, yy = np.array(v["score"]), np.array(v["y"])
        score[k] = {"roc_auc": float(roc_auc_score(yy, s)), "pr_auc": float(average_precision_score(yy, s)),
                    "far": float((s[yy == 0] > 1).mean()), "det": float((s[yy == 1] > 1).mean()),
                    "n_good": int((yy == 0).sum()), "n_bad": int((yy == 1).sum()),
                    "prevalence": float(yy.mean())}

    # ---- transfer: baseline from machine A applied to machine B, same operation
    transfer = {a: {b: {"fa": [0, 0], "det": [0, 0]} for b in MACHINES} for a in MACHINES}
    shift = []
    for o in OPS:
        for b in MACHINES:
            if (b, o) not in static_models:
                continue
            idx_b = np.where((machine == b) & (op == o))[0]
            _, trb = static_models[(b, o)]
            test = np.setdiff1d(idx_b, trb)
            for a in MACHINES:
                if (a, o) not in static_models:
                    continue
                mod_a, tra = static_models[(a, o)]
                rr = mod_a.pca.ratio(X[test])
                yy = label[test]
                transfer[a][b]["fa"][0] += int((rr[yy == 0] > 1).sum())
                transfer[a][b]["fa"][1] += int((yy == 0).sum())
                transfer[a][b]["det"][0] += int((rr[yy == 1] > 1).sum())
                transfer[a][b]["det"][1] += int((yy == 1).sum())
                if a < b:
                    A = (X[tra] - mod_a.pca.mu) / mod_a.pca.sd
                    B = (X[trb] - mod_a.pca.mu) / mod_a.pca.sd
                    wd = [stats.wasserstein_distance(A[:, f], B[:, f]) for f in range(X.shape[1])]
                    mmd, p = mmd_rbf(A, B)
                    top = int(np.argmax(wd))
                    shift.append({"op": o, "a": a, "b": b, "w_mean": float(np.mean(wd)),
                                  "w_top_feature": names[top], "w_top": float(wd[top]),
                                  "mmd": mmd, "p": p})
    tmat = {a: {b: {"far": v["fa"][0] / max(v["fa"][1], 1), "n_good": v["fa"][1],
                    "det": v["det"][0] / max(v["det"][1], 1), "n_bad": v["det"][1]}
                for b, v in row.items()} for a, row in transfer.items()}

    # ---- example recording pair for the story's first two steps
    ex = max((c for c in cells if not c["skipped"]), key=lambda c: (c["n_bad"], c["n"]))
    eidx = np.where((machine == ex["machine"]) & (op == ex["op"]))[0]
    bad_i = eidx[label[eidx] == 1][0]
    goods = eidx[(label[eidx] == 0) & (tf[eidx] == tf[bad_i])]
    good_i = goods[0] if len(goods) else eidx[label[eidx] == 0][-1]
    example = {"machine": ex["machine"], "op": ex["op"], "recordings": []}
    for i in (good_i, bad_i):
        d = load(path[i]).astype(float)
        d = d - d.mean(axis=0)
        nb = 600
        edges = np.linspace(0, len(d), nb + 1).astype(int)
        env = [[[float(d[s:e, k].min()), float(d[s:e, k].max())] for s, e in zip(edges[:-1], edges[1:])]
               for k in range(3)]
        psd = []
        for k in range(3):
            f, p = welch_psd(d[:, k])
            psd.append([round(float(v), 8) for v in p])
        example["recordings"].append({"label": int(label[i]), "tf": TIMEFRAMES[int(tf[i])],
                                      "duration_s": len(d) / FS, "env": env,
                                      "psd": psd, "freq": [float(v) for v in f]})

    far_curve = {k: [None if n == 0 else {"rate": a / n, "n": n} for a, n in v] for k, v in far_tf.items()}
    shifted = sum(1 for s in shift if s["p"] < 0.05)
    out_obj = {
        "dataset": {"n_recordings": int(len(X)), "n_bad": int(label.sum()), "n_features": int(X.shape[1]), "n_extracted": 40,
                    "feature_names": names, "timeframes": TIMEFRAMES, "alpha": ALPHA,
                    "min_train": MIN_TRAIN},
        "loo_far": {k: {"rate": float(np.mean(v)), "n": len(v)} for k, v in loo.items()},
        "cells": cells, "chart": chart, "scorecard": score, "transfer": tmat,
        "shift": shift, "shift_summary": {"pairs": len(shift), "p_lt_05": shifted},
        "far_curve": far_curve, "det_static": {"rate": det_static[0] / max(det_static[1], 1), "n": det_static[1]},
        "example": example,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(out_obj, separators=(",", ":")))
    print(json.dumps({k: out_obj[k] for k in ("loo_far", "scorecard", "transfer", "shift_summary",
                                               "far_curve", "det_static")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
