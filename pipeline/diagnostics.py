"""Stage 4: calibrated limits, matched-timeframe scorecard with intervals, drift root cause,
and a ramp (re-centring) recipe. Adds keys to the results JSON written by analyze.py.

Usage: python -I pipeline/diagnostics.py pipeline/out/features.npz public/data/results.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import MACHINES, MIN_TRAIN, OPS, model_features, pick_train  # noqa: E402
from features import TIMEFRAMES, feature_names, load, parse_name, welch_psd  # noqa: E402
from spc import ALPHA, fit_pca, fit_shewhart  # noqa: E402

N_BOOT = 2000
N_BOOT_RAMP = 500
RAMP_N = [3, 5, 8, 10, 15]
TRIAGE_MIN = [5, 15, 30]
SCREEN_RATIO = 3.0   # LOO T2 more than 3x its theoretical UCL ...
SCREEN_MIN_OPS = 3   # ... in at least 3 operations of the same machine run


# ---------------------------------------------------------------- small, tested helpers
def wilson(k: int, n: int, z: float = 1.959964) -> dict:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return {"rate": None, "lo": None, "hi": None, "k": 0, "n": 0}
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h), "k": int(k), "n": int(n)}


def conformal_multiplier(ratios: np.ndarray, alpha: float = ALPHA) -> float:
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest held-out ratio.
    Scaling a limit by this value gives P(new ratio > limit) <= alpha under exchangeability."""
    r = np.sort(np.asarray(ratios, float))
    rank = math.ceil((len(r) + 1) * (1 - alpha))
    return float(r[min(rank, len(r)) - 1])


def stratified_auc(score: np.ndarray, y: np.ndarray, strata: np.ndarray) -> float:
    """Probability a bad part outscores a good part from the same stratum (ties count 1/2)."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        b, g = score[m & (y == 1)], score[m & (y == 0)]
        if len(b) == 0 or len(g) == 0:
            continue
        d = b[:, None] - g[None, :]
        num += (d > 0).sum() + 0.5 * (d == 0).sum()
        den += d.size
    return float(num / den) if den else float("nan")


def sign_test_p(k: int, n: int) -> float:
    return float(stats.binomtest(k, n, 0.5).pvalue) if n else 1.0


def gain_shape(X40: np.ndarray) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Gain = per-axis log10 RMS. Shape = log10 band power minus 2 log10 RMS (band power scales
    with gain squared, so this is invariant to a sensor or mount sensitivity change)."""
    X9, n9 = model_features(X40)
    g_cols = [n9.index(f"{a}_log_rms") for a in "xyz"]
    gain = X9[:, g_cols]
    shape, s_names = [], []
    for a in "xyz":
        r = X9[:, n9.index(f"{a}_log_rms")]
        for lab in ("lo", "hi"):
            shape.append(X9[:, n9.index(f"{a}_log_band_{lab}")] - 2 * r)
            s_names.append(f"{a}_shape_{lab}")
    shape = np.stack(shape, axis=1)
    return {"gain": (gain, [n9[i] for i in g_cols]),
            "shape": (shape, s_names),
            "gain+shape": (np.hstack([gain, shape]), [n9[i] for i in g_cols] + s_names),
            "monitored9": (X9, n9)}


def classify_trajectory(v: list[float], frac: float = 0.7) -> str:
    """step: the first later timeframe already carries >= frac of the largest shift;
    excursion: the shift peaks and then falls back below frac of the peak; gradual: otherwise."""
    mx = max(v)
    if v[0] >= frac * mx:
        return "step"
    if int(np.argmax(v)) < len(v) - 1 and v[-1] < frac * mx:
        return "excursion"
    return "gradual"


def recentre_t2(mod, Xref: np.ndarray, Xeval: np.ndarray, rescale: bool) -> np.ndarray:
    """T2 with a lead monitor's loadings and eigenvalues, re-centred (and optionally re-scaled)
    on reference parts from the new machine."""
    mu = Xref.mean(axis=0)
    sd = mod.sd
    if rescale:
        sd = Xref.std(axis=0, ddof=1)
        sd = np.where(sd < 1e-9, 1.0, sd)
    T = ((Xeval - mu) / sd) @ mod.P
    return np.sum(T ** 2 / mod.lam, axis=1)


def screen_runs(X9, machine, op, tf, label, seq, ratio=SCREEN_RATIO, min_ops=SCREEN_MIN_OPS):
    """Qualification-set screen. A run (machine, timeframe, part index) is dropped from every
    baseline when its leave-one-out T2 exceeds ratio x UCL in >= min_ops operations: one part
    whose whole program is off is a machine-level event, not process noise.
    Returns (set of flagged runs, list of per-part LOO rows, unscreened LOO T2 and SPE ratios)."""
    hits, rows, lt2, lspe = {}, [], [], []
    for m in MACHINES:
        for o in OPS:
            idx = np.where((machine == m) & (op == o))[0]
            if len(idx) == 0:
                continue
            tr = pick_train(tf[idx], label[idx], range(len(TIMEFRAMES)))
            if tr.sum() < MIN_TRAIN:
                continue
            tri = idx[tr]
            for j in range(len(tri)):
                pm = fit_pca(np.delete(X9[tri], j, axis=0))
                a, b = pm.scores(X9[tri[j]][None, :])
                r = float(a[0] / pm.t2_lim)
                lt2.append(r); lspe.append(float(b[0] / pm.spe_lim))
                key = (m, int(tf[tri[j]]), int(seq[tri[j]]))
                if r > ratio:
                    hits.setdefault(key, []).append((o, r))
    flagged = {k for k, v in hits.items() if len(v) >= min_ops}
    for k in sorted(hits):
        rows.append({"machine": k[0], "timeframe": TIMEFRAMES[k[1]], "part": k[2],
                     "ops_over": len(hits[k]), "max_ratio": max(r for _, r in hits[k]),
                     "dropped": k in flagged})
    return flagged, rows, np.array(lt2), np.array(lspe)


def auc_pair(score, y):
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


# ---------------------------------------------------------------- main analysis
def main(npz: str, out: str):
    rng = np.random.default_rng(0)
    D = np.load(npz, allow_pickle=False)
    X40, machine, op, tf, label, rms, path = (D[k] for k in
                                               ("X", "machine", "op", "tf", "label", "rms", "path"))
    seq = np.array([parse_name(p)["idx"] for p in path])
    FS_ = gain_shape(X40)
    X9 = FS_["monitored9"][0]
    lrms = np.log10(rms)
    DETS = ["rms", "t2", "comb", "iso"]
    flagged, screen_rows, u_t2, u_spe = screen_runs(X9, machine, op, tf, label, seq)
    dropped = np.array([(machine[i], int(tf[i]), int(seq[i])) in flagged for i in range(len(seq))])
    print(f"screen: {len(flagged)} runs flagged, {int(dropped.sum())} recordings", flush=True)

    cells = []  # per-cell bookkeeping
    for m in MACHINES:
        for o in OPS:
            idx = np.where((machine == m) & (op == o))[0]
            if len(idx) == 0:
                continue
            idx = idx[np.lexsort((seq[idx], tf[idx]))]  # time order
            t, y = tf[idx], label[idx]
            tr = pick_train(t, y, range(len(TIMEFRAMES)))
            if tr.sum() < MIN_TRAIN:
                continue
            tr &= ~dropped[idx]          # screened runs leave the baseline, stay in evaluation
            tri = idx[tr]
            c = {"m": m, "o": o, "idx": idx, "tr": tr, "tri": tri, "last": int(t[tr].max()),
                 "train_tfs": sorted(set(t[tr].tolist())), "fs": {}}
            for name, (Xf, _) in FS_.items():
                mod = fit_pca(Xf[tri])
                t2, spe = mod.scores(Xf[idx])
                lt2, lspe = np.empty(len(tri)), np.empty(len(tri))
                for j in range(len(tri)):
                    pm = fit_pca(np.delete(Xf[tri], j, axis=0))
                    a, b = pm.scores(Xf[tri[j]][None, :])
                    lt2[j], lspe[j] = a[0] / pm.t2_lim, b[0] / pm.spe_lim
                c["fs"][name] = {"mod": mod, "t2": t2 / mod.t2_lim, "spe": spe / mod.spe_lim,
                                 "loo_t2": lt2, "loo_spe": lspe}
            # RMS Shewhart and Isolation Forest on the monitored vector, full fit and LOO
            sh = fit_shewhart(lrms[tri])
            c["rms_full"] = sh.ratio(lrms[idx])
            c["rms_loo"] = np.array([fit_shewhart(np.delete(lrms[tri], j)).ratio(lrms[tri[j]:tri[j] + 1])[0]
                                     for j in range(len(tri))])

            def iso_fit(Xtr):
                mod9 = fit_pca(Xtr)
                Z = (Xtr - mod9.mu) / mod9.sd
                f = IsolationForest(n_estimators=200, random_state=0).fit(Z)
                lim = float(np.quantile(-f.score_samples(Z), 1 - ALPHA))
                return lambda Xe: -f.score_samples((Xe - mod9.mu) / mod9.sd) / lim
            c["iso_full"] = iso_fit(X9[tri])(X9[idx])
            c["iso_loo"] = np.array([iso_fit(np.delete(X9[tri], j, axis=0))(X9[tri[j]][None, :])[0]
                                     for j in range(len(tri))])
            cells.append(c)
    print(f"{len(cells)} cells fitted", flush=True)

    # ---------------- upgrade 1a: empirical (split-conformal) limit multipliers
    def pooled(key, fs="monitored9"):
        return np.concatenate([c["fs"][fs][key] for c in cells])
    raw_loo = {"t2": pooled("loo_t2"), "spe": pooled("loo_spe"),
               "rms": np.concatenate([c["rms_loo"] for c in cells]),
               "iso": np.concatenate([c["iso_loo"] for c in cells])}
    mult = {k: conformal_multiplier(v) for k, v in raw_loo.items()}
    comb_loo_before = np.maximum(raw_loo["t2"], raw_loo["spe"])
    comb_loo_after = np.maximum(raw_loo["t2"] / mult["t2"], raw_loo["spe"] / mult["spe"])
    n_loo = len(comb_loo_before)

    # cross-validated check: calibrate on a random half of the cells, test on the other half
    cell_loo = [(c["fs"]["monitored9"]["loo_t2"], c["fs"]["monitored9"]["loo_spe"]) for c in cells]
    cv_t2, cv_comb = [], []
    for _ in range(200):
        perm = rng.permutation(len(cells))
        A, B = perm[: len(cells) // 2], perm[len(cells) // 2:]
        mt = conformal_multiplier(np.concatenate([cell_loo[i][0] for i in A]))
        ms = conformal_multiplier(np.concatenate([cell_loo[i][1] for i in A]))
        bt = np.concatenate([cell_loo[i][0] for i in B])
        bs = np.concatenate([cell_loo[i][1] for i in B])
        cv_t2.append(float((bt > mt).mean()))
        cv_comb.append(float((np.maximum(bt / mt, bs / ms) > 1).mean()))
    mu_t2, mu_spe = conformal_multiplier(u_t2), conformal_multiplier(u_spe)
    nu = len(u_t2)
    calibration = {
        "unscreened": {"n_loo": int(nu), "multiplier": {"t2": mu_t2, "spe": mu_spe},
                       "loo_far_theoretical": {"t2": wilson(int((u_t2 > 1).sum()), nu),
                                               "spe": wilson(int((u_spe > 1).sum()), nu),
                                               "comb": wilson(int((np.maximum(u_t2, u_spe) > 1).sum()), nu)},
                       "quantiles_t2": {q: float(np.quantile(u_t2, float(q))) for q in ("0.95", "0.98", "0.99")}},
        "screen": {"rule": f"drop a run whose LOO T2 > {SCREEN_RATIO} x UCL in >= {SCREEN_MIN_OPS} operations",
                   "runs_flagged": int(len(flagged)), "recordings_dropped": int(dropped.sum()),
                   "candidates": screen_rows},
        "method": "split-conformal multiplier on pooled leave-one-out ratios, target 1% per statistic",
        "n_loo": int(n_loo), "multiplier": mult,
        "loo_far_before": {"t2": wilson(int((raw_loo["t2"] > 1).sum()), n_loo),
                           "spe": wilson(int((raw_loo["spe"] > 1).sum()), n_loo),
                           "comb": wilson(int((comb_loo_before > 1).sum()), n_loo),
                           "rms": wilson(int((raw_loo["rms"] > 1).sum()), n_loo),
                           "iso": wilson(int((raw_loo["iso"] > 1).sum()), n_loo)},
        "loo_far_after": {"t2": wilson(int((raw_loo["t2"] > mult["t2"]).sum()), n_loo),
                          "spe": wilson(int((raw_loo["spe"] > mult["spe"]).sum()), n_loo),
                          "comb": wilson(int((comb_loo_after > 1).sum()), n_loo)},
        "cross_validated_half_split": {"reps": 200,
                                       "t2_mean": float(np.mean(cv_t2)), "t2_ci": ci(cv_t2),
                                       "comb_mean": float(np.mean(cv_comb)), "comb_ci": ci(cv_comb)},
    }

    # calibrated scores per cell: full-model scores and LOO scores for the training parts
    for c in cells:
        f = c["fs"]["monitored9"]
        c["full"] = {"rms": c["rms_full"] / mult["rms"], "t2": f["t2"] / mult["t2"],
                     "comb": np.maximum(f["t2"] / mult["t2"], f["spe"] / mult["spe"]),
                     "iso": c["iso_full"] / mult["iso"]}
        c["loo"] = {"rms": c["rms_loo"] / mult["rms"], "t2": f["loo_t2"] / mult["t2"],
                    "comb": np.maximum(f["loo_t2"] / mult["t2"], f["loo_spe"] / mult["spe"]),
                    "iso": c["iso_loo"] / mult["iso"]}
        # out-of-sample score for every recording in the cell
        c["oos"] = {}
        for d in DETS:
            s = c["full"][d].copy()
            s[c["tr"]] = c["loo"][d]
            c["oos"][d] = s

    # ---------------- upgrade 1b: matched-timeframe scorecard vs drift-inclusive scorecard
    def eval_set(c, kind):
        t, y = tf[c["idx"]], label[c["idx"]]
        if kind == "matched":
            bad_tfs = set(t[y == 1].tolist())
            return np.isin(t, list(bad_tfs))   # bad parts plus goods from the same timeframes
        return (y == 1) | ((~c["tr"]) & (t > c["last"]))   # original: all bad + later goods

    def assemble(kind, ci_ids=None):
        ids = range(len(cells)) if ci_ids is None else ci_ids
        S = {d: [] for d in DETS}
        Y, ST = [], []
        for r, i in enumerate(ids):
            c = cells[i]
            m = eval_set(c, kind)
            for d in DETS:
                S[d].append(c["oos"][d][m])
            Y.append(label[c["idx"]][m])
            ST.append(r * 10 + tf[c["idx"]][m])
        return {d: np.concatenate(v) for d, v in S.items()}, np.concatenate(Y), np.concatenate(ST)

    per_cell = {kind: [assemble(kind, [i]) for i in range(len(cells))] for kind in ("matched", "drift")}

    def metrics(kind, ids):
        parts = [per_cell[kind][i] for i in ids]
        S = {d: np.concatenate([p[0][d] for p in parts]) for d in DETS}
        Y = np.concatenate([p[1] for p in parts])
        ST = np.concatenate([np.full(len(p[1]), r * 10) + (p[2] % 10) for r, p in enumerate(parts)])
        return S, Y, ST

    scorecards = {}
    for kind in ("matched", "drift"):
        S, Y, ST = metrics(kind, range(len(cells)))
        sc = {"n_good": int((Y == 0).sum()), "n_bad": int((Y == 1).sum()),
              "prevalence": float(Y.mean()), "detectors": {}}
        for d in DETS:
            s = S[d]
            sc["detectors"][d] = {"roc_auc": auc_pair(s, Y), "pr_auc": float(average_precision_score(Y, s)),
                                  "strat_auc": stratified_auc(s, Y, ST),
                                  "far": wilson(int((s[Y == 0] > 1).sum()), int((Y == 0).sum())),
                                  "det": wilson(int((s[Y == 1] > 1).sum()), int((Y == 1).sum()))}
        # cluster bootstrap over cells
        boots = {d: [] for d in DETS}
        sboots = {d: [] for d in DETS}
        for _ in range(N_BOOT):
            ids = rng.integers(0, len(cells), len(cells))
            Sb, Yb, STb = metrics(kind, ids)
            for d in DETS:
                boots[d].append(auc_pair(Sb[d], Yb))
                sboots[d].append(stratified_auc(Sb[d], Yb, STb))
        for d in DETS:
            sc["detectors"][d]["roc_auc_ci"] = ci(boots[d])
            sc["detectors"][d]["strat_auc_ci"] = ci(sboots[d])
        diffs = {}
        for other in ("t2", "comb", "iso"):
            dv = np.array(boots["rms"]) - np.array(boots[other])
            ds = np.array(sboots["rms"]) - np.array(sboots[other])
            ok = ~np.isnan(dv)
            diffs[f"rms_minus_{other}"] = {
                "roc_auc_diff": sc["detectors"]["rms"]["roc_auc"] - sc["detectors"][other]["roc_auc"],
                "ci": ci(dv), "p_rms_not_better": float((dv[ok] <= 0).mean()),
                "strat_diff": sc["detectors"]["rms"]["strat_auc"] - sc["detectors"][other]["strat_auc"],
                "strat_ci": ci(ds), "strat_p_rms_not_better": float((ds[~np.isnan(ds)] <= 0).mean())}
        sc["paired"] = diffs
        scorecards[kind] = sc
    print("scorecards done", flush=True)

    # ---------------- calibrated drift numbers (static, later goods), Wilson intervals
    drift = {"t2": [], "comb": []}
    for d in ("t2", "comb"):
        for tt in range(len(TIMEFRAMES)):
            k = n = 0
            for c in cells:
                t, y = tf[c["idx"]], label[c["idx"]]
                g = (y == 0) & (t > c["last"]) & (t == tt) & ~c["tr"]
                k += int((c["full"][d][g] > 1).sum()); n += int(g.sum())
            drift[d].append(wilson(k, n) if n else None)
    later = {}
    for d in ("t2", "comb", "rms", "iso"):
        k = n = 0
        for c in cells:
            t, y = tf[c["idx"]], label[c["idx"]]
            g = (y == 0) & (t > c["last"])
            k += int((c["full"][d][g] > 1).sum()); n += int(g.sum())
        later[d] = wilson(k, n)
    in_period = {d: wilson(int(sum((c["loo"][d] > 1).sum() for c in cells)), n_loo) for d in DETS}

    # ---------------- upgrade 2a: gain vs shape monitors (T2, each calibrated by its own LOO)
    gs = {}
    for name in FS_:
        mt = conformal_multiplier(np.concatenate([c["fs"][name]["loo_t2"] for c in cells]))
        kl = nl = kb = nb = 0
        S, Y, ST = [], [], []
        for r, c in enumerate(cells):
            f = c["fs"][name]
            t, y = tf[c["idx"]], label[c["idx"]]
            s_full = f["t2"] / mt
            g = (y == 0) & (t > c["last"])
            kl += int((s_full[g] > 1).sum()); nl += int(g.sum())
            kb += int((s_full[y == 1] > 1).sum()); nb += int((y == 1).sum())
            s = s_full.copy(); s[c["tr"]] = f["loo_t2"] / mt
            m = eval_set(c, "matched")
            S.append(s[m]); Y.append(y[m]); ST.append(r * 10 + t[m])
        S, Y, ST = np.concatenate(S), np.concatenate(Y), np.concatenate(ST)
        gs[name] = {"p": int(FS_[name][0].shape[1]), "features": FS_[name][1], "multiplier": mt,
                    "later_far": wilson(kl, nl), "det_all_bad": wilson(kb, nb),
                    "matched_roc_auc": auc_pair(S, Y), "matched_strat_auc": stratified_auc(S, Y, ST)}

    # ---------------- upgrade 2b: machine-level step vs per-operation drift
    names40 = feature_names()
    shifts = []  # (cell, tf, 40-vector of z-shifts)
    for c in cells:
        t, y = tf[c["idx"]], label[c["idx"]]
        Xtr = X40[c["tri"]]
        mu, sd = Xtr.mean(0), Xtr.std(0, ddof=1)
        sd = np.where(sd < 1e-9, np.inf, sd)
        for tt in range(c["last"] + 1, len(TIMEFRAMES)):
            g = (y == 0) & (t == tt)
            if g.sum() >= 3:
                shifts.append((c, tt, (X40[c["idx"][g]].mean(0) - mu) / sd))
    machine_tf = []
    for m in MACHINES:
        for tt in range(len(TIMEFRAMES)):
            V = np.array([v for c, t_, v in shifts if c["m"] == m and t_ == tt])
            if len(V) < 5:
                continue
            med = np.median(V, axis=0)
            feats = []
            for fi, fname in enumerate(names40):
                col = V[:, fi]
                col = col[np.isfinite(col) & (col != 0)]
                k = int((np.sign(col) == np.sign(med[fi])).sum())
                feats.append({"feature": fname, "median_z": float(med[fi]), "agree": k, "n": int(len(col)),
                              "p": sign_test_p(k, len(col))})
            feats.sort(key=lambda r: -abs(r["median_z"]))
            n_sig = sum(1 for r in feats if r["p"] < 0.05 / len(names40))
            # cosine agreement of op shift vectors on the 9 monitored-style features (log rms + bands)
            sub = [names40.index(f"{a}_log_rms") for a in "xyz"] + \
                  [names40.index(f"{a}_band{i}_log") for a in "xyz" for i in range(8)]
            U = V[:, sub]
            U = U / np.linalg.norm(U, axis=1, keepdims=True)
            cos = U @ U.T
            mean_cos = float((cos.sum() - len(U)) / (len(U) * (len(U) - 1)))
            machine_tf.append({"machine": m, "tf": tt, "timeframe": TIMEFRAMES[tt], "n_ops": int(len(V)),
                               "top": feats[:5], "n_features_sig_bonferroni": n_sig,
                               "mean_pairwise_cosine": mean_cos,
                               "median_abs_shift": float(np.median(np.abs(V[:, sub])))})
    # step vs gradual: per machine, median |shift| over later timeframes, and the ratio of the
    # first later timeframe to the maximum
    step = {}
    for m in MACHINES:
        rows = [r for r in machine_tf if r["machine"] == m]
        if not rows:
            continue
        traj = [(r["timeframe"], r["median_abs_shift"]) for r in rows]
        first, mx = traj[0][1], max(v for _, v in traj)
        step[m] = {"trajectory": traj, "first_over_max": first / mx,
                   "pattern": classify_trajectory([v for _, v in traj])}
    # feature ranking over all cell x later-timeframe pairs
    V_all = np.array([v for _, _, v in shifts])
    med_abs = np.median(np.abs(np.where(np.isfinite(V_all), V_all, np.nan)), axis=0)
    med_sgn = np.nanmedian(np.where(np.isfinite(V_all), V_all, np.nan), axis=0)
    order = np.argsort(-np.nan_to_num(med_abs))
    ranking = [{"feature": names40[i], "median_abs_z": float(med_abs[i]), "median_z": float(med_sgn[i])}
               for i in order[:12]]

    # PSD overlay for the cell that moves most on the top-ranked band feature
    top_feat = next(r["feature"] for r in ranking if "band" in r["feature"])
    fi = names40.index(top_feat)
    axis = "xyz".index(top_feat[0])
    best = max((s for s in shifts if np.sum((label[s[0]["idx"]] == 0) & (tf[s[0]["idx"]] == s[1])) >= 8),
               key=lambda s: abs(s[2][fi]))
    c_b, tt_b = best[0], best[1]

    def mean_psd(ii):
        P = []
        for i in ii:
            d = load(path[i]).astype(float)
            f, p = welch_psd(d[:, axis] - d[:, axis].mean())
            P.append(p)
        return f, np.mean(np.log10(np.array(P) + 1e-15), axis=0)
    gi = c_b["idx"][(label[c_b["idx"]] == 0) & (tf[c_b["idx"]] == tt_b)]
    try:
        f, p_tr = mean_psd(c_b["tri"])
        _, p_late = mean_psd(gi)
        psd = {"machine": c_b["m"], "op": c_b["o"], "axis": "xyz"[axis], "feature": top_feat,
               "train_timeframes": [TIMEFRAMES[i] for i in c_b["train_tfs"]], "later_timeframe": TIMEFRAMES[tt_b],
               "z_shift": float(best[2][fi]), "n_train": int(len(c_b["tri"])), "n_later": int(len(gi)),
               "freq": [round(float(v), 2) for v in f],
               "log10_psd_train": [round(float(v), 4) for v in p_tr],
               "log10_psd_later": [round(float(v), 4) for v in p_late]}
        lo = (f >= 10) & (f < 100); hi = (f >= 100)
        psd["mean_log_diff_10_100"] = float(np.mean(p_late[lo] - p_tr[lo]))
        psd["mean_log_diff_100_1000"] = float(np.mean(p_late[hi] - p_tr[hi]))
    except OSError as e:  # raw data not present
        psd = {"error": str(e)}
    root_cause = {"gain_shape": gs, "machine_tf": machine_tf, "step": step, "ranking": ranking, "psd": psd}
    print("root cause done", flush=True)

    # ---------------- upgrade 3: ramp recipe
    ct = mult["t2"]
    by = {(c["m"], c["o"]): c for c in cells}
    copy_rows, own_rows = [], []
    pairs = []
    for o in OPS:
        for a in MACHINES:
            for b in MACHINES:
                if a == b or (a, o) not in by or (b, o) not in by:
                    continue
                A, B = by[(a, o)], by[(b, o)]
                pairs.append((A, B))
                ma = A["fs"]["monitored9"]["mod"]
                t2c = ma.scores(X9[B["tri"]])[0] / ma.t2_lim / ct
                copy_rows.append((a, b, int((t2c > 1).sum()), len(t2c)))
                own = B["loo"]["t2"]
                own_rows.append((a, b, int((own > 1).sum()), len(own)))
    k_c = sum(r[2] for r in copy_rows); n_c = sum(r[3] for r in copy_rows)
    k_o = sum(r[2] for r in own_rows); n_o = sum(r[3] for r in own_rows)
    copy_mat = {a: {b: wilson(sum(r[2] for r in copy_rows if r[0] == a and r[1] == b),
                              sum(r[3] for r in copy_rows if r[0] == a and r[1] == b))
                    for b in MACHINES if b != a} for a in MACHINES}

    def bad_in_period(B):
        t, y = tf[B["idx"]], label[B["idx"]]
        return B["idx"][(y == 1) & np.isin(t, B["train_tfs"])]

    def adapt(A, B, ref_pos, rescale):
        ma = A["fs"]["monitored9"]["mod"]
        ref = B["tri"][ref_pos]
        rest = np.delete(B["tri"], ref_pos)
        lim = ma.t2_lim * ct
        fa = recentre_t2(ma, X9[ref], X9[rest], rescale) > lim
        bi = bad_in_period(B)
        de = recentre_t2(ma, X9[ref], X9[bi], rescale) > lim if len(bi) else np.array([], bool)
        return int(fa.sum()), len(fa), int(de.sum()), len(de)

    ramp = {"variants": {}}
    for variant, rescale in (("recentre", False), ("recentre_rescale", True)):
        rows = []
        for N in RAMP_N:
            if rescale and N < 5:
                continue
            use = [(A, B) for A, B in pairs if len(B["tri"]) - N >= 5]
            tot = np.array([adapt(A, B, np.arange(N), rescale) for A, B in use]).sum(0)
            bfar, bdet = [], []
            for _ in range(N_BOOT_RAMP):
                s = np.array([adapt(A, B, rng.choice(len(B["tri"]), N, replace=False), rescale)
                              for A, B in use]).sum(0)
                bfar.append(s[0] / s[1]); bdet.append(s[2] / s[3])
            rows.append({"N": N, "pairs": len(use), "far": wilson(int(tot[0]), int(tot[1])),
                         "det_in_period": wilson(int(tot[2]), int(tot[3])),
                         "far_random_subset_mean": float(np.mean(bfar)), "far_random_subset_ci": ci(bfar),
                         "det_random_subset_mean": float(np.mean(bdet)), "det_random_subset_ci": ci(bdet)})
        ramp["variants"][variant] = rows
    # detection references on the same in-period bad parts
    k_cd = n_cd = k_od = n_od = 0
    for A, B in pairs:
        bi = bad_in_period(B)
        if not len(bi):
            continue
        ma = A["fs"]["monitored9"]["mod"]
        k_cd += int((ma.scores(X9[bi])[0] / ma.t2_lim / ct > 1).sum()); n_cd += len(bi)
        pos = np.searchsorted(B["idx"], bi) if False else [int(np.where(B["idx"] == i)[0][0]) for i in bi]
        k_od += int((B["full"]["t2"][pos] > 1).sum()); n_od += len(bi)
    ramp["copy"] = {"far": wilson(k_c, n_c), "det_in_period": wilson(k_cd, n_cd), "matrix": copy_mat}
    ramp["own_qualification"] = {"far": wilson(k_o, n_o), "det_in_period": wilson(k_od, n_od)}

    # re-baseline after a step: own static monitor re-centred on the first N goods of the first
    # later timeframe; evaluated on the rest of that timeframe and on all later timeframes
    reb = []
    for N in (5, 10):
        acc = {"static_same": [0, 0], "recentred_same": [0, 0], "static_after": [0, 0],
               "recentred_after": [0, 0], "rerun_det": [0, 0], "static_det": [0, 0]}
        for c in cells:
            t, y = tf[c["idx"]], label[c["idx"]]
            mod = c["fs"]["monitored9"]["mod"]
            lim = mod.t2_lim * ct
            later_tfs = [tt for tt in range(c["last"] + 1, len(TIMEFRAMES)) if ((y == 0) & (t == tt)).sum() >= N + 5]
            if not later_tfs:
                continue
            s_tf = later_tfs[0]
            g = c["idx"][(y == 0) & (t == s_tf)]
            ref, rest = g[:N], g[N:]
            aft = c["idx"][(y == 0) & (t > s_tf)]
            badl = c["idx"][(y == 1) & (t >= s_tf)]
            for key, Xe in (("same", rest), ("after", aft)):
                if not len(Xe):
                    continue
                st = mod.scores(X9[Xe])[0] > lim
                rc = recentre_t2(mod, X9[ref], X9[Xe], False) > lim
                acc[f"static_{key}"][0] += int(st.sum()); acc[f"static_{key}"][1] += len(Xe)
                acc[f"recentred_{key}"][0] += int(rc.sum()); acc[f"recentred_{key}"][1] += len(Xe)
            if len(badl):
                acc["static_det"][0] += int((mod.scores(X9[badl])[0] > lim).sum()); acc["static_det"][1] += len(badl)
                acc["rerun_det"][0] += int((recentre_t2(mod, X9[ref], X9[badl], False) > lim).sum())
                acc["rerun_det"][1] += len(badl)
        reb.append({"N": N, **{k: wilson(*v) for k, v in acc.items()}})
    ramp["rebaseline_after_step"] = reb

    # operational units, scenario on measured rates
    def ops_row(label_, w):
        r = w["rate"]
        return {"scenario": label_, "far": r, "alarms_per_1000_good": 1000 * r,
                "alarms_per_1000_ci": [1000 * w["lo"], 1000 * w["hi"]],
                "engineer_hours_per_1000_good": {str(mn): 1000 * r * mn / 60 for mn in TRIAGE_MIN}}
    rc = {r["N"]: r for r in ramp["variants"]["recentre"]}
    rs10 = next(r for r in ramp["variants"]["recentre_rescale"] if r["N"] == 10)
    ramp["ops_units"] = {
        "assumptions": {"triage_minutes_per_alarm": TRIAGE_MIN,
                        "note": "Scenario on measured false-alarm rates. Triage minutes are editable assumptions, not measurements."},
        "rows": [ops_row("Copied baseline, unchanged", ramp["copy"]["far"]),
                 ops_row("Copied loadings, re-centred on 5 parts", rc[5]["far"]),
                 ops_row("Copied loadings, re-centred on 10 parts", rc[10]["far"]),
                 ops_row("Copied loadings, re-centred + re-scaled on 10 spread parts",
                         {"rate": rs10["far_random_subset_mean"], "lo": rs10["far_random_subset_ci"][0],
                          "hi": rs10["far_random_subset_ci"][1]}),
                 ops_row("Full per-machine qualification (16-25 parts)", ramp["own_qualification"]["far"]),
                 ops_row("Stale 2019 baseline on later good parts", later["t2"])]}
    print("ramp done", flush=True)

    res = json.loads(Path(out).read_text())
    res["calibration"] = calibration
    res["scorecard_v2"] = scorecards
    res["drift_calibrated"] = {"by_timeframe": drift, "later_goods": later, "in_period_loo": in_period}
    res["root_cause"] = root_cause
    res["ramp"] = ramp
    Path(out).write_text(json.dumps(res, separators=(",", ":")))
    slim = {"calibration": calibration, "scorecard_v2": scorecards,
            "drift_calibrated": res["drift_calibrated"],
            "root_cause": {k: v for k, v in root_cause.items() if k != "psd"},
            "psd": {k: v for k, v in psd.items() if not k.startswith(("freq", "log10"))},
            "ramp": ramp}
    Path(out).with_name("diagnostics_summary.json").write_text(json.dumps(slim, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
