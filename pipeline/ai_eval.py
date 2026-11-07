"""Stage 6: the MiniRocket heads against the classical charts, duration gate first.
Adds results["ai"] to the results JSON.

Usage: python -I pipeline/ai_eval.py pipeline/out public/data/results.json
Inputs: out/windows_meta.npz (windows.py), out/features.npz (extract.py) and
        out/emb_<arm>_s<seed>_<M>.npz (embed.py).

Pre-declared before any result was computed:
  arms       crop (primary, duration-blind: every recording of a cell cut to a common prefix of
             0.45 x the cell's qualification median length, identical window starts) and full
             (windows spread over each whole recording, as in the prototypes)
  heads      A = dual ridge on mean PPV, trained on 2019 labels only and scored forward in time
             (2019 parts cross-fitted by machine x timeframe); B = kNN memory bank of good windows
  baselines  rms, t2spe, cycle (Shewhart on log10 cycle time) and the fused rule
             max(head, cycle) on limit-normalised scores, on every table
  limits     split-conformal per machine on the reference parts' out-of-sample scores, alpha 0.05
             (+inf below N = 19; the max-of-N limit with FAR <= 1/(N+1) as the fallback)
  gate       a head passes if, for all 3 seeds, in the crop arm
             (A) its like-for-like AUC (machine x op x timeframe strata) minus the best classical
                 chart (replicate-wise max of RMS and T^2+SPE, each on the prefix and on the whole
                 recording) has a paired cluster-bootstrap 95% CI above 0, and
             (B) on parts made after their cell's qualification (2020-21 for 38 of 43 cells; five
                 cells with few 2019 good parts qualify into 2020-21 under the repo's n >= 15
                 rule), with every chart thresholded to flag 5% of those later good parts, its
                 detection minus the RMS chart's (best of prefix and whole-recording RMS) has a
                 paired bootstrap 95% CI above 0.
             The AI headline ships only if a head passes; otherwise the label-free and
             re-qualification headline ships. Both arms are reported either way.
  drift label  share of the alarm's band z-vector energy in the forward-in-time drift subspace;
             >= 0.5 reads "nuisance drift: re-baseline", else "out of window: investigate"
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import model_features  # noqa: E402
from detectors import (hier_shift, hier_shift_loo, knn_score, pca_oos, reference_sets, ridge_head,  # noqa: E402
                       shewhart_oos, shift_for, t2spe_shifted)
from diagnostics import screen_runs  # noqa: E402
from drift import (CUT, K_DRIFT, band_names, band_vectors, cell_z, drift_share, label_alarm,  # noqa: E402
                   shift_vectors, subspace, top_bands)
from features import FS, TIMEFRAMES  # noqa: E402
from stats import (boot_delta, ci, cluster_bootstrap_auc, conformal_multiplier, conformal_threshold,  # noqa: E402
                   detection_at_far, eligible, exact_far, limit_with_fallback, max_of_n_far, mcnemar_exact,
                   min_n_for_alpha, part_bootstrap_auc, pooled_ap, strata_ids, stratified_auc, wilson)

SEEDS = (0, 1, 2)
ARMS = ("crop", "full")
MACHINES = ("M01", "M02", "M03")
N_BOOT = 2000
ALPHA = 0.05
ALPHA_STRICT = 0.01
GATE_FAR = 0.05
REQ_N = (0, 5, 10, 19, 20, 50)
REQ_DRAWS = 50
LOMO_N = (0, 5, 10, 20)
LOMO_DRAWS = 50
HEADS = ("ridge", "knn")
DETS = ("rms", "t2spe", "cycle", "ridge", "knn", "fused_ridge", "fused_knn")
# Re-qualification fairness check: the pooled machine-wide offset can hurt a univariate chart whose
# shifts are not shared across operations, so each classical chart is also re-qualified by
# recalibrating its limit only (no re-centring). Comparisons quote the better classical variant.
LIMIT_ONLY = ("rms_lo", "t2spe_lo", "cycle_lo")
REQ_DETS = DETS + LIMIT_ONLY


def base_det(k: str) -> str:
    return k[:-3] if k.endswith("_lo") else k
HYPOTHESES = [  # band fingerprints reported by the autoencoder prototype, re-checked here
    {"cell": "M01-OP05", "lo": 200, "hi": 213, "axes": "zy"},
    {"cell": "M02-OP12", "lo": 701, "hi": 720, "axes": "z"},
    {"cell": "M01-OP01", "lo": 720, "hi": 760, "axes": "xy"},
]


def rnd(x, d=4):
    if isinstance(x, dict):
        return {str(k): rnd(v, d) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, d) for v in x]
    if isinstance(x, np.ndarray):
        return [rnd(v, d) for v in x.tolist()]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), d)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, out: Path):
        M = np.load(out / "windows_meta.npz")
        F = np.load(out / "features.npz")
        self.rel = M["relpath"]
        key = np.array(["/".join(p.split("/")[-4:]) for p in F["path"]])
        pos = {k: i for i, k in enumerate(key)}
        j = np.array([pos[r] for r in self.rel])      # matched by relative path, never by basename
        assert (F["label"][j] == M["label"]).all() and (F["tf"][j] == M["tf"]).all()
        self.machine, self.op, self.tf, self.idx, self.label = (M[k] for k in ("machine", "op", "tf", "idx", "label"))
        self.n_samples, self.crop_len = M["n_samples"], M["crop_len"]
        self.starts = {"full": M["starts_full"], "crop": M["starts_crop"]}
        self.band = {"full": M["band_full"], "crop": M["band_crop"]}
        self.edges = M["attr_edges"]
        self.K, self.L = int(M["K"]), int(M["L"])
        self.crop_frac = float(M["crop_frac"])
        X40 = F["X"][j]
        self.X9 = {"full": model_features(X40)[0], "crop": model_features(M["feat40_prefix"])[0]}
        self.lrms = {"full": np.log10(F["rms"][j]), "crop": np.log10(M["rms_prefix"])}
        self.ldur = np.log10(self.n_samples / FS)
        self.n = len(self.label)
        self.cell = np.char.add(np.char.add(self.machine, "-"), self.op)
        flagged, self.screen_rows, _, _ = screen_runs(self.X9["full"], self.machine, self.op, self.tf,
                                                      self.label, self.idx)
        self.dropped = np.array([(self.machine[i], int(self.tf[i]), int(self.idx[i])) in flagged
                                 for i in range(self.n)])
        refs = reference_sets(self.machine, self.op, self.tf, self.label, self.idx, self.dropped)
        self.refs = {f"{m}-{o}": r for (m, o), r in refs.items()}
        self.cells = sorted(self.refs)
        self.cell_idx = {c: np.where(self.cell == c)[0] for c in self.cells}
        self.in_scope = np.isin(self.cell, self.cells)
        self.refmask = np.zeros(self.n, bool)
        for r in self.refs.values():
            self.refmask[r] = True
        # The repo's rule takes whole timeframes in order until n >= 15, so a cell with few 2019 good
        # parts qualifies into 2020-21. "Later" is therefore per cell: after its last reference period.
        self.last = {c: int(self.tf[r].max()) for c, r in self.refs.items()}
        last_row = np.array([self.last.get(c, 99) for c in self.cell])
        self.later = self.in_scope & (self.tf > last_row)
        self.in_ref_period = self.in_scope & (self.tf <= last_row)
        self.late_qualified = {c: sorted(set(self.tf[r].tolist())) for c, r in self.refs.items() if self.last[c] > 1}
        assert not (self.refmask & self.later).any()
        self.qmed = {c: float(np.median(self.n_samples[r])) for c, r in self.refs.items()}
        self.dur_ratio = np.array([self.n_samples[i] / self.qmed[c] if c in self.qmed else np.nan
                                   for i, c in enumerate(self.cell)])
        self.mot = strata_ids(self.machine, self.op, self.tf)
        self.mt = strata_ids(self.machine, self.tf)
        self.cell_of = {int(i): self.cell[i] for i in np.where(self.in_scope)[0]}


def load_emb(out: Path, arm: str, seed: int, m: str):
    Z = np.load(out / f"emb_{arm}_s{seed}_{m}.npz")
    return Z["E_mean"], Z["E_win"]


# ---------------------------------------------------------------- scorers
def classical(D: Data, arm: str):
    X9, lr = D.X9[arm], D.lrms[arm]
    rms, cyc, t2, spe = (np.full(D.n, np.nan) for _ in range(4))
    models = {}
    for c in D.cells:
        ii, ref = D.cell_idx[c], D.refs[c]
        rms[ii] = shewhart_oos(lr, ref, ii)
        cyc[ii] = shewhart_oos(D.ldur, ref, ii)
        t2[ii], spe[ii], models[c] = pca_oos(X9, ref, ii)
    c_t2 = conformal_multiplier(t2[D.refmask], 0.01)
    c_spe = conformal_multiplier(spe[D.refmask], 0.01)
    return ({"rms": rms, "t2spe": np.maximum(t2 / c_t2, spe / c_spe), "cycle": cyc},
            {"c_t2": c_t2, "c_spe": c_spe, "models": models})


def heads(D: Data, out: Path, arm: str, seed: int):
    knn, ridge = np.full(D.n, np.nan), np.full(D.n, np.nan)
    ex = {"dist": np.full((D.n, D.K), np.nan, np.float32), "part": np.full((D.n, D.K), -1),
          "win": np.full((D.n, D.K), -1), "scale": {}}
    alphas = {}
    pool = D.in_scope & (D.tf <= 1) & ~D.dropped          # labelled 2019 history
    for m in MACHINES:
        Em, Ew = load_emb(out, arm, seed, m)               # fold m: never fitted on machine m
        for c in D.cells:
            if not c.startswith(m):
                continue
            ii, ref = D.cell_idx[c], D.refs[c]
            raw, dist, part, win = knn_score(Ew, ii, ref)
            scale = float(np.median(raw[np.isin(ii, ref)]))
            knn[ii] = raw / scale
            ex["dist"][ii], ex["part"][ii], ex["win"][ii], ex["scale"][c] = dist / scale, part, win, scale
        rows = np.where(D.in_scope & (D.machine == m))[0]
        for t in (0, 1):                                    # 2019: cross-fitted by machine x timeframe
            test = rows[D.tf[rows] == t]
            train = np.where(pool & ~((D.machine == m) & (D.tf == t)))[0]
            ridge[test], alphas[f"{m}/tf{t}"] = ridge_head(Em, D.label, D.mt, D.mt, train, test)
        test = rows[D.tf[rows] >= 2]                        # 2020-21: trained on 2019 only
        train = np.where(pool)[0]
        assert D.tf[train].max() <= 1
        ridge[test], alphas[f"{m}/later"] = ridge_head(Em, D.label, D.mt, D.mt, train, test)
    return {"ridge": ridge, "knn": knn}, ex, alphas


def machine_limits(D: Data, s: np.ndarray, alpha: float = ALPHA) -> dict:
    return {m: conformal_threshold(s[D.refmask & (D.machine == m)], alpha) for m in MACHINES}


def normalise(D: Data, s: np.ndarray, alpha: float = ALPHA):
    """u = (s - median_m) / (limit_m - median_m) per machine: u > 1 exactly when s exceeds the
    machine's split-conformal limit."""
    lim = machine_limits(D, s, alpha)
    med = {m: float(np.median(s[D.refmask & (D.machine == m)])) for m in MACHINES}
    Lr = np.array([lim[m] for m in D.machine])
    Mr = np.array([med[m] for m in D.machine])
    return (s - Mr) / (Lr - Mr), lim, med


def add_fused(D: Data, S: dict):
    u = {k: normalise(D, S[k])[0] for k in ("ridge", "knn", "cycle")}
    S["fused_ridge"] = np.maximum(u["ridge"], u["cycle"])
    S["fused_knn"] = np.maximum(u["knn"], u["cycle"])


# ---------------------------------------------------------------- tables
def ap_cluster_ci(U: dict, y, strata, n_boot=1000, seed=0) -> dict:
    keys = np.unique(strata)
    members = [np.where(strata == k)[0] for k in keys]
    rng = np.random.default_rng(seed + 7)
    boots = {k: [] for k in U}
    for _ in range(n_boot):
        ii = np.concatenate([members[p] for p in rng.integers(0, len(keys), len(keys))])
        if y[ii].min() == y[ii].max():
            continue
        for k, s in U.items():
            boots[k].append(pooled_ap(s[ii], y[ii]))
    return {k: ci(v) for k, v in boots.items()}


def scorecard(D: Data, S: dict, U: dict, mask, strata, best, seed=0, part=False, ap=True) -> dict:
    sub = np.where(mask)[0]
    el = np.zeros(D.n, bool)
    el[sub[eligible(strata[sub], D.label[sub])]] = True
    y, st = D.label[el], strata[el]
    res = cluster_bootstrap_auc({k: S[k][el] for k in S}, y, st, N_BOOT, seed)
    apci = ap_cluster_ci({k: U[k][el] for k in U}, y, st, 1000, seed) if ap else {}
    out = {"n_bad": int(y.sum()), "n_good": int((y == 0).sum()), "n_strata": res["_design"]["n_strata"],
           "detectors": {}}
    for k in S:
        r = {"auc": res[k]["point"], "ci": ci(res[k]["boots"])}
        if ap:
            r.update(pr_auc=pooled_ap(U[k][el], y), pr_auc_ci=apci[k])
        al = U[k][el] > 1.0   # u > 1: above the machine's split-conformal limit (alpha 0.05)
        r.update(far_at_limit=wilson(int(al[y == 0].sum()), int((y == 0).sum())),
                 det_at_limit=wilson(int(al[y == 1].sum()), int((y == 1).sum())))
        for tag, bases in (("rms", ["rms"]), ("t2spe", ["t2spe"]), ("cycle", ["cycle"]), ("best_classical", best)):
            if k not in bases:
                r[f"delta_vs_{tag}"] = boot_delta(res[k]["point"], res[k]["boots"], [res[b]["point"] for b in bases],
                                                  [res[b]["boots"] for b in bases])
        out["detectors"][k] = r
    if part:
        pres = part_bootstrap_auc({k: S[k][el] for k in (*HEADS, *best)}, y, st, N_BOOT, seed)
        for k in HEADS:
            out["detectors"][k]["part_bootstrap_delta_vs_best_classical"] = boot_delta(
                pres[k]["point"], pres[k]["boots"], [pres[b]["point"] for b in best], [pres[b]["boots"] for b in best])
    return out


def gate(D: Data, S: dict, U: dict, arm: str) -> dict:
    best = ["rms", "t2spe", "rms_full", "t2spe_full"] if arm == "crop" else ["rms", "t2spe"]
    rms_set = ["rms", "rms_full"] if arm == "crop" else ["rms"]
    sc = scorecard(D, S, U, D.in_scope, D.mot, best, ap=False)
    lat = D.later
    names = list(dict.fromkeys([*HEADS, *rms_set, "t2spe", "cycle", "fused_ridge", "fused_knn"]))
    pt, bt = detection_at_far({k: U[k][lat] for k in names}, D.label[lat], GATE_FAR, N_BOOT, 0)
    out = {"comparator_auc": best, "comparator_detection": rms_set, "heads": {}}
    for h in HEADS:
        A = sc["detectors"][h]["delta_vs_best_classical"]
        B = boot_delta(pt[h]["det"], bt[h], [pt[b]["det"] for b in rms_set], [bt[b] for b in rms_set])
        out["heads"][h] = {"auc": sc["detectors"][h]["auc"],
                           "best_classical_auc": max(sc["detectors"][b]["auc"] for b in best),
                           "best_classical_name": max(best, key=lambda b: sc["detectors"][b]["auc"]),
                           "A_delta_auc": A, "A_pass": bool(A["ci"][0] > 0),
                           "det_at_far": pt[h]["det"], "rms_det_at_far": max(pt[b]["det"] for b in rms_set),
                           "B_delta_det": B, "B_pass": bool(B["ci"][0] > 0)}
        out["heads"][h]["pass"] = out["heads"][h]["A_pass"] and out["heads"][h]["B_pass"]
    out["detection_at_far"] = pt
    return out


def drift_static(D: Data, S: dict, alpha: float) -> dict:
    out = {}
    g_all, b_all = D.later & (D.label == 0), D.later & (D.label == 1)
    rng = np.random.default_rng(0)
    for k, s in S.items():
        lim = machine_limits(D, s, alpha)
        al = s > np.array([lim[m] for m in D.machine])
        split = []
        for m in MACHINES:
            cal = np.where(D.refmask & (D.machine == m))[0]
            for _ in range(200):
                p = rng.permutation(cal)
                h = len(p) // 2
                split.append(np.mean(s[p[h:]] > conformal_threshold(s[p[:h]], alpha)))
        out[k] = {"far_later": wilson(int(al[g_all].sum()), int(g_all.sum())),
                  "det_later": wilson(int(al[b_all].sum()), int(b_all.sum())),
                  "far_by_tf": {TIMEFRAMES[t]: wilson(int(al[g_all & (D.tf == t)].sum()), int((g_all & (D.tf == t)).sum()))
                                for t in range(2, 6)},
                  "in_period_far_split_half": float(np.mean(split)), "limits": lim}
    return out


def matched_far(D: Data, U: dict, far: float, base_sets: dict) -> dict:
    lat = D.later
    pt, bt = detection_at_far({k: U[k][lat] for k in U}, D.label[lat], far, N_BOOT, 0)
    for k in U:
        for tag, bases in base_sets.items():
            if k not in bases:
                pt[k][f"det_delta_vs_{tag}"] = boot_delta(pt[k]["det"], bt[k], [pt[b]["det"] for b in bases],
                                                          [bt[b] for b in bases])
    return pt


# ---------------------------------------------------------------- re-qualification
class Requal:
    """Re-qualify every chart on N confirmed-good parts of one machine.
    Location charts and the ridge: hierarchical re-centring (machine-wide offset + shrunk per-op
    correction). T^2+SPE: the same on the 9-feature centre. kNN: append the parts to the bank.
    Fused: re-qualified components on the static normalisation, one limit on the max.
    The limit always comes from the N parts' leave-one-out scores: split-conformal, or max-of-N
    below the minimum N."""

    def __init__(self, D: Data, S: dict, arm: str, cls: dict, Ew: dict, scale: dict):
        self.D, self.S, self.arm, self.cls, self.Ew, self.scale = D, S, arm, cls, Ew, scale
        self.x = {"rms": D.lrms[arm], "cycle": D.ldur}
        self.cen = {c: {k: (self.x[k][r].mean(), self.x[k][r].std(ddof=1)) for k in self.x} | {"ridge": S["ridge"][r].mean()}
                    for c, r in D.refs.items()}
        self.norm = {k: normalise(D, S[k])[1:] for k in ("ridge", "knn", "cycle")}

    def _u(self, k, v, parts):
        lim, med = self.norm[k]
        Lr = np.array([lim[self.D.machine[i]] for i in parts])
        Mr = np.array([med[self.D.machine[i]] for i in parts])
        return (v - Mr) / (Lr - Mr)

    def scores(self, m: str, rows: np.ndarray, new: np.ndarray) -> dict:
        D, cn, cr = self.D, self.D.cell[new], self.D.cell[rows]
        res = {}
        for k, x in self.x.items():
            off = x[new] - np.array([self.cen[c][k][0] for c in cn])
            d, per = hier_shift(off, cn)
            mur = np.array([self.cen[c][k][0] for c in cr])
            sdr = np.array([self.cen[c][k][1] for c in cr])
            sdn = np.array([self.cen[c][k][1] for c in cn])
            res[k] = (np.abs(x[rows] - mur - shift_for(cr, d, per)) / (3 * sdr),
                      np.abs(off - hier_shift_loo(off, cn)) / (3 * sdn))
        s = self.S["ridge"]
        off = s[new] - np.array([self.cen[c]["ridge"] for c in cn])
        d, per = hier_shift(off, cn)
        res["ridge"] = (s[rows] - shift_for(cr, d, per), s[new] - hier_shift_loo(off, cn))
        mods, X9 = self.cls["models"], D.X9[self.arm]
        off = np.stack([X9[i] - mods[c].mu for i, c in zip(new, cn)])
        d, per = hier_shift(off, cn)
        test = np.empty(len(rows))
        for c in np.unique(cr):
            mm = cr == c
            test[mm] = t2spe_shifted(mods[c], X9[rows[mm]], per.get(c, d), self.cls["c_t2"], self.cls["c_spe"])
        loo = hier_shift_loo(off, cn)
        calib = np.array([t2spe_shifted(mods[c], X9[i][None, :], loo[j], self.cls["c_t2"], self.cls["c_spe"])[0]
                          for j, (i, c) in enumerate(zip(new, cn))])
        res["t2spe"] = (test, calib)
        E = self.Ew[m]
        test, calib = np.empty(len(rows)), np.empty(len(new))
        for c in np.unique(np.concatenate([cr, cn])):
            bank = np.concatenate([D.refs[c], new[cn == c]])
            if (cr == c).any():
                test[cr == c] = knn_score(E, rows[cr == c], bank)[0] / self.scale[c]
            if (cn == c).any():
                calib[cn == c] = knn_score(E, new[cn == c], bank)[0] / self.scale[c]
        res["knn"] = (test, calib)
        for f, h in (("fused_ridge", "ridge"), ("fused_knn", "knn")):
            res[f] = (np.maximum(self._u(h, res[h][0], rows), self._u("cycle", res["cycle"][0], rows)),
                      np.maximum(self._u(h, res[h][1], new), self._u("cycle", res["cycle"][1], new)))
        for k in LIMIT_ONLY:   # static 2019 scores; only the limit is recalibrated on the N parts
            b = base_det(k)
            res[k] = (self.S[b][rows], self.S[b][new])
        return res


CLASSICAL_REQ = ("rms", "t2spe", "cycle", "rms_lo", "t2spe_lo", "cycle_lo")


def paired_detection(H: dict, Y: np.ndarray, seed: int, tag: int) -> dict:
    """Each chart at its own re-qualified limit (same false-alarm guarantee): paired difference in
    detection on the SAME bad parts. Bootstrap over bad parts (CI), and an exact McNemar test on the
    discordant parts. 'Best classical' is the replicate-wise max over every classical variant."""
    b = np.where(Y == 1)[0]
    rng = np.random.default_rng([seed, 4242, tag])
    idx = rng.integers(0, len(b), (N_BOOT, len(b)))
    point = {k: float(H[k][b].mean()) for k in H}
    boots = {k: H[k][b][idx].mean(axis=1) for k in H}
    best = max(CLASSICAL_REQ, key=lambda k: point[k])
    out = {"n_bad": int(len(b)), "best_classical": best}
    for h in ("ridge", "knn", "fused_ridge", "fused_knn"):
        out[h] = {"vs_rms": {**boot_delta(point[h], boots[h], [point["rms"]], [boots["rms"]]),
                             "mcnemar": mcnemar_exact(H[h][b], H["rms"][b])},
                  "vs_best_classical": {**boot_delta(point[h], boots[h], [point[k] for k in CLASSICAL_REQ],
                                                     [boots[k] for k in CLASSICAL_REQ]),
                                        "mcnemar": mcnemar_exact(H[h][b], H[best][b])}}
    return out


def requal_curve(D: Data, S: dict, arm: str, cls: dict, out: Path, seed: int, ex: dict) -> dict:
    Ew = {m: load_emb(out, arm, seed, m)[1] for m in MACHINES}
    RQ = Requal(D, S, arm, cls, Ew, ex["scale"])
    static = {k: machine_limits(D, S[base_det(k)]) for k in REQ_DETS}
    groups = []
    for m in MACHINES:
        for t in range(2, 6):
            sel = D.later & (D.machine == m) & (D.tf == t)
            g = np.where(sel & (D.label == 0))[0]
            g = g[np.lexsort((D.op[g], D.idx[g]))]          # production order: part index, then op
            groups.append((m, t, g, np.where(sel & (D.label == 1))[0]))
    res = {"first": {}, "random": {}}
    for scheme in res:
        for N in REQ_N:
            need = 55 if N == 50 else 25
            use = [G for G in groups if len(G[2]) >= need]
            draws = REQ_DRAWS if (scheme == "random" and N > 0) else 1
            acc = {k: {"far": [], "det": [], "auc": [], "fa": 0, "ng": 0, "da": 0, "nb": 0} for k in REQ_DETS}
            for r in range(draws):
                rng = np.random.default_rng([seed, N, r])
                tally = {k: [0, 0, 0, 0] for k in REQ_DETS}
                parts = {k: [] for k in REQ_DETS}
                hits = {k: [] for k in REQ_DETS}
                Y, T = [], []
                for m, t, g, b in use:
                    if N == 0:
                        new, rest = np.array([], int), g
                    elif scheme == "first":
                        new, rest = g[:N], g[N:]
                    else:
                        new = rng.choice(g, N, replace=False)
                        rest = np.setdiff1d(g, new)
                    rows = np.concatenate([rest, b])
                    yy = D.label[rows]
                    if N == 0:
                        sc = {k: S[base_det(k)][rows] for k in REQ_DETS}
                        lim = {k: static[k][m] for k in REQ_DETS}
                    else:
                        q = RQ.scores(m, rows, new)
                        sc = {k: q[k][0] for k in REQ_DETS}
                        lim = {k: limit_with_fallback(q[k][1], ALPHA)[0] for k in REQ_DETS}
                    for k in REQ_DETS:
                        al = sc[k] > lim[k]
                        tally[k][0] += int(al[yy == 0].sum()); tally[k][1] += int((yy == 0).sum())
                        tally[k][2] += int(al[yy == 1].sum()); tally[k][3] += int((yy == 1).sum())
                        parts[k].append(sc[k])
                        hits[k].append(al)
                    Y.append(yy); T.append(D.mot[rows])
                Y, T = np.concatenate(Y), np.concatenate(T)
                el = eligible(T, Y)
                for k in REQ_DETS:
                    a, tk = acc[k], tally[k]
                    a["far"].append(tk[0] / tk[1]); a["det"].append(tk[2] / tk[3] if tk[3] else np.nan)
                    a["fa"] += tk[0]; a["ng"] += tk[1]; a["da"] += tk[2]; a["nb"] += tk[3]
                    a["auc"].append(stratified_auc(np.concatenate(parts[k])[el], Y[el], T[el]))
            row = {"N": N, "n_groups": len(use), "draws": draws,
                   "rule": "static 2019 limit" if N == 0 else ("conformal" if N >= min_n_for_alpha(ALPHA) else "max-of-N"),
                   "guarantee": None if N == 0 else (exact_far(N, ALPHA) if N >= min_n_for_alpha(ALPHA) else max_of_n_far(N)),
                   "n_good_eval": int(acc["rms"]["ng"] / draws), "n_bad_eval": int(acc["rms"]["nb"] / draws), "detectors": {}}
            if draws == 1:   # one fixed set of parts: paired tests of detection at each chart's own limit
                row["paired_detection"] = paired_detection({k: np.concatenate(v) for k, v in hits.items()}, Y, seed, N)
            for k in REQ_DETS:
                a = acc[k]
                if draws == 1:
                    row["detectors"][k] = {"far": wilson(a["fa"], a["ng"]), "det": wilson(a["da"], a["nb"]), "auc": a["auc"][0]}
                else:
                    row["detectors"][k] = {"far": {"rate": float(np.mean(a["far"])), "range": ci(a["far"])},
                                           "det": {"rate": float(np.nanmean(a["det"])), "range": ci(a["det"])},
                                           "auc": float(np.nanmean(a["auc"]))}
            res[scheme][f"N={N}"] = row
    return res


def cost_table(rq: dict, triage_min: float = 15.0) -> dict:
    """Scenario on measured rates: alarms and engineer-hours per 1,000 good parts after a drift event,
    for the stale 2019 limit (N = 0) and after re-qualifying on the first 20 good parts. The triage
    time is the repo's existing 15-minute assumption, not a measurement."""
    rows = []
    for N in (0, 20):
        r = rq["first"][f"N={N}"]["detectors"]
        for k, v in r.items():
            far = v["far"]["rate"]
            rows.append({"N": N, "detector": k, "far": far, "det": v["det"]["rate"],
                         "alarms_per_1000_good": 1000 * far,
                         "engineer_hours_per_1000_good": 1000 * far * triage_min / 60})
    return {"triage_minutes_assumed": triage_min, "scheme": "first N good parts after the drift event", "rows": rows}


# ---------------------------------------------------------------- new machine (leave one machine out)
def lomo(D: Data, S: dict, arm: str, out: Path, seed: int) -> dict:
    X9, lr = D.X9[arm], D.lrms[arm]
    res = {}
    for m in MACHINES:
        Em, Ew = load_emb(out, arm, seed, m)
        src_train = np.where(D.in_scope & (D.machine != m) & ~D.dropped)[0]
        src_ref = np.where(D.refmask & (D.machine != m))[0]
        tq = np.where(D.in_ref_period & (D.machine == m))[0]       # the target's own qualification periods
        tref = np.where(D.refmask & (D.machine == m))[0]
        tbad = tq[D.label[tq] == 1]
        ev = np.concatenate([tref, tbad])
        sc = {k: np.full(D.n, np.nan) for k in ("rms", "cycle", "t2spe", "ridge", "knn")}
        cal = {k: [] for k in sc}
        sc["ridge"][ev], alpha_m = ridge_head(Em, D.label, D.mt, D.mt, src_train, ev)
        oof = np.full(D.n, np.nan)
        for g in np.unique(D.mt[src_ref]):
            hold = src_ref[D.mt[src_ref] == g]
            oof[hold], _ = ridge_head(Em, D.label, D.mt, D.mt, src_train[D.mt[src_train] != g], hold)
        cal["ridge"] = list(oof[src_ref])
        copy = {}
        t2l, spel, banks = [], [], {}
        for c in D.cells:
            if not c.startswith(m):
                continue
            o = c.split("-")[1]
            srcs = [f"{mm}-{o}" for mm in MACHINES if mm != m and f"{mm}-{o}" in D.refs]
            if not srcs:
                continue
            bank = np.concatenate([D.refs[s] for s in srcs])
            ii = ev[D.cell[ev] == c]
            banks[c] = bank
            rb = knn_score(Ew, bank, bank)[0]
            scale = float(np.median(rb))
            sc["knn"][ii] = knn_score(Ew, ii, bank)[0] / scale
            cal["knn"] += list(rb / scale)
            for k, x in (("rms", lr), ("cycle", D.ldur)):
                sc[k][ii] = shewhart_oos(x, bank, ii)
                cal[k] += list(shewhart_oos(x, bank, bank))
            t2b, speb, mod = pca_oos(X9, bank, bank)
            t2i, spei = mod.scores(X9[ii])
            t2l.append((ii, t2i / mod.t2_lim, spei / mod.spe_lim))
            spel.append((t2b, speb))
            copy[c] = {"bank": bank, "scale": scale, "mod": mod, "mu": {k: x[bank].mean() for k, x in (("rms", lr), ("cycle", D.ldur))},
                       "sd": {k: x[bank].std(ddof=1) for k, x in (("rms", lr), ("cycle", D.ldur))},
                       "ridge_mu": float(np.mean(oof[bank]))}
        c_t2 = conformal_multiplier(np.concatenate([a for a, _ in spel]), 0.01)
        c_spe = conformal_multiplier(np.concatenate([b for _, b in spel]), 0.01)
        for ii, a, b in t2l:
            sc["t2spe"][ii] = np.maximum(a / c_t2, b / c_spe)
        cal["t2spe"] = list(np.concatenate([np.maximum(a / c_t2, b / c_spe) for a, b in spel]))
        lim_copy = {k: conformal_threshold(np.array(v), ALPHA) for k, v in cal.items()}
        order = tref[np.lexsort((D.op[tref], D.idx[tref], D.tf[tref]))]
        rows_out = {}
        for N in LOMO_N:
            for scheme in (("copy",) if N == 0 else ("first", "random")):
                draws = LOMO_DRAWS if scheme == "random" else 1
                acc = {k: {"far": [], "det": [], "auc": [], "n": [0, 0, 0, 0]} for k in sc}
                hits = {}
                for r in range(draws):
                    rng = np.random.default_rng([seed, 99, N, r])
                    new = np.array([], int) if N == 0 else (order[:N] if scheme == "first" else rng.choice(tref, N, replace=False))
                    rest = np.setdiff1d(tref, new)
                    rows = np.concatenate([rest, tbad])
                    cr, cn = D.cell[rows], D.cell[new]
                    if N == 0:
                        cur = {k: (sc[k][rows], lim_copy[k]) for k in sc}
                    else:
                        cur = {}
                        for k, x in (("rms", lr), ("cycle", D.ldur)):
                            off = x[new] - np.array([copy[c]["mu"][k] for c in cn])
                            d, per = hier_shift(off, cn)
                            mur = np.array([copy[c]["mu"][k] for c in cr]); sdr = np.array([copy[c]["sd"][k] for c in cr])
                            sdn = np.array([copy[c]["sd"][k] for c in cn])
                            te = np.abs(x[rows] - mur - shift_for(cr, d, per)) / (3 * sdr)
                            ca = np.abs(off - hier_shift_loo(off, cn)) / (3 * sdn)
                            cur[k] = (te, limit_with_fallback(ca, ALPHA)[0])
                        s = sc["ridge"]
                        off = s[new] - np.array([copy[c]["ridge_mu"] for c in cn])
                        d, per = hier_shift(off, cn)
                        cur["ridge"] = (s[rows] - shift_for(cr, d, per),
                                        limit_with_fallback(s[new] - hier_shift_loo(off, cn), ALPHA)[0])
                        off = np.stack([X9[i] - copy[c]["mod"].mu for i, c in zip(new, cn)])
                        d, per = hier_shift(off, cn)
                        te = np.empty(len(rows))
                        for c in np.unique(cr):
                            te[cr == c] = t2spe_shifted(copy[c]["mod"], X9[rows[cr == c]], per.get(c, d), c_t2, c_spe)
                        loo = hier_shift_loo(off, cn)
                        ca = np.array([t2spe_shifted(copy[c]["mod"], X9[i][None, :], loo[j], c_t2, c_spe)[0]
                                       for j, (i, c) in enumerate(zip(new, cn))])
                        cur["t2spe"] = (te, limit_with_fallback(ca, ALPHA)[0])
                        te, ca = np.empty(len(rows)), np.empty(len(new))
                        for c in np.unique(np.concatenate([cr, cn])):
                            bank = np.concatenate([copy[c]["bank"], new[cn == c]])
                            if (cr == c).any():
                                te[cr == c] = knn_score(Ew, rows[cr == c], bank)[0] / copy[c]["scale"]
                            if (cn == c).any():
                                ca[cn == c] = knn_score(Ew, new[cn == c], bank)[0] / copy[c]["scale"]
                        cur["knn"] = (te, limit_with_fallback(ca, ALPHA)[0])
                    yy = D.label[rows]
                    el = eligible(D.mot[rows], yy)
                    for k, (v, lim) in cur.items():
                        acc[k]["far"].append(float(np.mean(v[yy == 0] > lim)))
                        acc[k]["det"].append(float(np.mean(v[yy == 1] > lim)) if (yy == 1).any() else np.nan)
                        acc[k]["auc"].append(stratified_auc(v[el], yy[el], D.mot[rows][el]))
                        acc[k]["n"] = [acc[k]["n"][0] + int((v[yy == 0] > lim).sum()), acc[k]["n"][1] + int((yy == 0).sum()),
                                       acc[k]["n"][2] + int((v[yy == 1] > lim).sum()), acc[k]["n"][3] + int((yy == 1).sum())]
                        hits[k] = v[yy == 1] > lim
                out_k = {}
                for k, a in acc.items():
                    r = {"far": float(np.mean(a["far"])), "det": float(np.nanmean(a["det"])),
                         "auc": float(np.nanmean(a["auc"]))}
                    if draws > 1:
                        r.update(far_range=ci(a["far"]), det_range=ci(a["det"]))
                    else:   # one fixed set of parts: Wilson intervals over parts
                        r.update(far_ci=wilson(a["n"][0], a["n"][1]), det_ci=wilson(a["n"][2], a["n"][3]))
                    out_k[k] = r
                if draws == 1:   # same target bad parts for every chart: exact paired test
                    out_k["paired_mcnemar"] = {f"{h}_vs_{c}": mcnemar_exact(hits[h], hits[c])
                                               for h in HEADS for c in ("rms", "cycle", "t2spe")}
                rows_out[f"N={N}/{scheme}"] = out_k
        res[m] = {"n_good": int(len(tref)), "n_bad": int(len(tbad)), "ridge_alpha": alpha_m, "rows": rows_out}
    return res


# ---------------------------------------------------------------- drift subspace, labels, attribution
def drift_section(D: Data, arm: str, S: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    B = band_vectors(D.band[arm])
    g19 = [c for c in D.cells if D.last[c] <= 1]
    Z, st = cell_z(B, D.refs, D.cell_of, np.where(D.in_scope)[0], glob_from=g19)       # clipped: subspace
    Zraw, _ = cell_z(B, D.refs, D.cell_of, np.where(D.in_scope)[0], glob_from=g19, clip=False)  # attribution
    keep = D.in_scope & ~D.dropped
    cells_ops = {}
    for c in D.cells:
        cells_ops.setdefault(c.split("-")[1], []).append(c)
    U, var, nvec, maxtf = {}, {}, {}, {}
    for t in range(2, 6):
        refs_t = {c: r for c, r in D.refs.items() if D.last[c] < t}   # cells qualified before period t
        V, used = shift_vectors(Z, D.cell, D.tf, D.label, keep, refs_t, st, t, cells_ops)
        # forward in time: the subspace for period t is built from rows of earlier periods only
        assert used.size > 0 and int(D.tf[used].max()) < t, f"drift subspace for tf {t} saw tf >= {t}"
        assert (D.label[used] == 0).all(), "drift subspace saw a bad part"
        U[t], var[t] = subspace(V)
        nvec[t], maxtf[t] = len(V), int(D.tf[used].max())
    share = np.full(D.n, np.nan)
    for t in range(2, 6):
        ii = np.where(D.later & (D.tf == t))[0]
        share[ii] = drift_share(Z[ii], U[t])
    names = band_names(D.edges)
    val = {}
    for k, s in S.items():
        lim = machine_limits(D, s)
        al = (s > np.array([lim[m] for m in D.machine])) & D.later
        fa, ta = al & (D.label == 0), al & (D.label == 1)
        val[k] = {"false_alarms": int(fa.sum()), "false_alarms_labelled_drift": int((share[fa] >= CUT).sum()),
                  "true_alarms": int(ta.sum()), "true_alarms_labelled_out_of_window": int((share[ta] < CUT).sum())}
    g, b = D.later & (D.label == 0), D.later & (D.label == 1)
    dvec = np.nanmean(Z[D.in_scope & (D.label == 1)], axis=0)
    p = Z.shape[1]
    out = {"k": K_DRIFT, "p": int(p), "cut": CUT, "random_share": K_DRIFT / p,
           "n_shift_vectors": {TIMEFRAMES[t]: nvec[t] for t in U},
           "latest_period_used": {TIMEFRAMES[t]: TIMEFRAMES[maxtf[t]] for t in U},
           "variance_share_top_k": {TIMEFRAMES[t]: var[t] for t in U},
           "loadings_latest": [{"band": names[j], "loading": float(U[5][j, 0])}
                               for j in np.argsort(-np.abs(U[5][:, 0]))[:6]],
           "defect_direction_share": {TIMEFRAMES[t]: float(drift_share(dvec, U[t])[0]) for t in U},
           "later_good_share_median": float(np.nanmedian(share[g])), "later_bad_share_median": float(np.nanmedian(share[b])),
           "later_good_labelled_drift": float(np.mean(share[g] >= CUT)),
           "later_bad_labelled_out_of_window": float(np.mean(share[b] < CUT)),
           "alarm_labels_at_alpha_0.05": val}
    return out, Z, Zraw, share


def drilldown(D: Data, arm: str, S: dict, U: dict, ex: dict, Z: np.ndarray, share: np.ndarray, primary: str) -> list:
    names = band_names(D.edges)
    lim = {k: machine_limits(D, S[k]) for k in DETS}
    alarm = {k: S[k] > np.array([lim[k][m] for m in D.machine]) for k in DETS}
    bad = np.where(D.in_scope & (D.label == 1))[0]
    fa = np.where(D.later & (D.label == 0) & alarm[primary])[0]
    fa = fa[np.argsort(-U[primary][fa])][:40]
    items = []
    st_all, band = D.starts[arm], D.band[arm]
    for i in np.concatenate([bad, fa]):
        c = D.cell[i]
        d, p, w = ex["dist"][i], ex["part"][i], ex["win"][i]
        worst = np.argsort(-d)[:3]
        diff = 10.0 * np.mean([band[i, v] - band[p[v], w[v]] for v in worst], axis=0).reshape(-1)
        items.append({
            "row": int(i), "file": str(D.rel[i]), "machine": str(D.machine[i]), "op": str(D.op[i]),
            "timeframe": TIMEFRAMES[int(D.tf[i])], "label": int(D.label[i]), "part": int(D.idx[i]),
            "later": bool(D.later[i]), "duration_ratio": float(D.dur_ratio[i]),
            "scores": {k: float(U[k][i]) for k in DETS},
            "alarms": [k for k in DETS if alarm[k][i]],
            "drift_share": None if not np.isfinite(share[i]) else float(share[i]),
            "drift_label": label_alarm(share[i]) if np.isfinite(share[i]) else None,
            "band_z_top": top_bands(Z[i], names),
            "band_z": [round(float(v), 2) for v in Z[i]],
            "windows": {"t_s": [round(float(s) / FS, 2) for s in st_all[i]],
                        "dist": [round(float(v), 3) for v in d],
                        "worst": [{"t_s": round(float(st_all[i][v]) / FS, 2),
                                   "cycle_frac": round(float(st_all[i][v]) / D.qmed[c], 3),
                                   "dist": round(float(d[v]), 3), "matched_file": str(D.rel[p[v]]),
                                   "matched_t_s": round(float(st_all[p[v]][w[v]]) / FS, 2)} for v in worst],
                        "band_diff_db_top": [{"band": t["band"], "db": t["z"]} for t in top_bands(diff, names)]}})
    return items


def fingerprints(D: Data, Z: np.ndarray) -> list:
    nb = len(D.edges) - 1
    out = []
    for h in HYPOTHESES:
        bads = np.where((D.cell == h["cell"]) & (D.label == 1) & D.in_scope)[0]
        rows, hits = [], 0
        for i in bads:
            top = np.argsort(-np.abs(Z[i]), kind="stable")[:3]
            match = any("xyz"[j // nb] in h["axes"] and D.edges[j % nb] < h["hi"] and D.edges[j % nb + 1] > h["lo"]
                        for j in top)
            hits += int(match)
            rows.append({"timeframe": TIMEFRAMES[int(D.tf[i])], "part": int(D.idx[i]), "match": bool(match),
                         "top": [{"band": f"{'xyz'[j // nb]} {D.edges[j % nb]:.0f}-{D.edges[j % nb + 1]:.0f} Hz",
                                  "z": round(float(Z[i][j]), 2)} for j in top]})
        out.append({**h, "n_bad": int(len(bads)), "n_match_top3": hits, "parts": rows})
    return out


def parts_table(D: Data, U: dict) -> dict:
    ii = np.where(D.in_scope)[0]
    cells = sorted(set(D.cell[ii].tolist()))
    cpos = {c: j for j, c in enumerate(cells)}
    return {"cells": cells, "row": ii.tolist(), "cell": [cpos[c] for c in D.cell[ii]], "tf": D.tf[ii].tolist(),
            "label": D.label[ii].tolist(), "part": D.idx[ii].tolist(),
            "duration_ratio": [round(float(v), 3) for v in D.dur_ratio[ii]],
            "u": {k: [None if not np.isfinite(v) else round(float(v), 3) for v in U[k][ii]]
                  for k in (*DETS, "rms_full", "t2spe_full") if k in U},
            "note": "u = (score - machine median) / (machine alpha=0.05 limit - median); u > 1 is an alarm. "
                    "Crop arm: rms and t2spe use the common prefix, rms_full and t2spe_full the whole recording."}


# ---------------------------------------------------------------- main
def main(out_dir: str, results_path: str):
    T0 = time.time()
    out = Path(out_dir)
    D = Data(out)
    timing = {}
    cls_all, Scl_all = {}, {}
    for arm in ("full", "crop"):
        Scl_all[arm], cls_all[arm] = classical(D, arm)
    timing["classical_s"] = time.time() - T0
    R = {"config": {"seeds": list(SEEDS), "arms": list(ARMS), "primary_arm": "crop", "K": D.K, "L": D.L,
                    "crop_frac": D.crop_frac, "alpha": ALPHA, "alpha_strict": ALPHA_STRICT, "gate_far": GATE_FAR,
                    "n_boot": N_BOOT, "requal_N": list(REQ_N), "requal_draws": REQ_DRAWS, "lomo_N": list(LOMO_N),
                    "drift_k": K_DRIFT, "drift_cut": CUT, "min_n_alpha_0.05": min_n_for_alpha(ALPHA),
                    "min_n_alpha_0.01": min_n_for_alpha(ALPHA_STRICT)},
         "data": {"n_recordings": int(D.n), "n_bad": int(D.label.sum()), "n_cells": len(D.cells),
                  "n_in_scope": int(D.in_scope.sum()), "n_bad_in_scope": int(D.label[D.in_scope].sum()),
                  "n_later_good": int((D.later & (D.label == 0)).sum()), "n_later_bad": int((D.later & (D.label == 1)).sum()),
                  "screened_recordings": int(D.dropped.sum()),
                  "cells_qualified_beyond_2019": {c: [TIMEFRAMES[t] for t in v] for c, v in D.late_qualified.items()},
                  "crop_seconds_range": [float(D.crop_len.min() / FS), float(D.crop_len.max() / FS)],
                  "min_length_over_qual_median": float(np.nanmin(D.dur_ratio[D.in_scope])),
                  "bad_with_same_index_good_twin": int(sum(1 for i in np.where(D.label == 1)[0]
                                                           if ((D.machine == D.machine[i]) & (D.op == D.op[i]) & (D.tf == D.tf[i])
                                                               & (D.idx == D.idx[i]) & (D.label == 0)).any()))}}
    try:
        import numba
        import sklearn
        import sktime
        R["config"]["versions"] = {"numpy": np.__version__, "sklearn": sklearn.__version__,
                                   "sktime": sktime.__version__, "numba": numba.__version__}
    except ImportError:
        pass
    # regression against the repo's own numbers (whole-recording classical charts)
    el = np.zeros(D.n, bool)
    sub = np.where(D.in_scope)[0]
    el[sub[eligible(D.mot[sub], D.label[sub])]] = True
    R["regression"] = {"rms_like_for_like": stratified_auc(Scl_all["full"]["rms"][el], D.label[el], D.mot[el]),
                       "t2spe_like_for_like": stratified_auc(Scl_all["full"]["t2spe"][el], D.label[el], D.mot[el]),
                       "multiplier_t2": cls_all["full"]["c_t2"], "multiplier_spe": cls_all["full"]["c_spe"],
                       "duration_only_auc": max(stratified_auc(D.ldur[el], D.label[el], D.mot[el]),
                                                stratified_auc(-D.ldur[el], D.label[el], D.mot[el])),
                       "short_cycle_auc": stratified_auc(-D.ldur[el], D.label[el], D.mot[el]),
                       "bad_duration_ratio_quantiles": {q: float(np.nanquantile(D.dur_ratio[D.in_scope & (D.label == 1)], float(q)))
                                                        for q in ("0.0", "0.1", "0.5", "0.9", "1.0")},
                       "good_duration_ratio_quantiles": {q: float(np.nanquantile(D.dur_ratio[D.in_scope & (D.label == 0)], float(q)))
                                                         for q in ("0.01", "0.5", "0.99")}}
    # calibration panel: the textbook 3-sigma limit on held-out (leave-one-out) qualification parts
    ref = D.refmask
    R["calibration"] = {k: {"loo_far_3sigma": wilson(int((Scl_all["full"][k][ref] > 1).sum()), int(ref.sum())),
                            "nominal": 0.0027} for k in ("rms", "cycle")}
    print(f"classical charts ready ({time.time() - T0:.0f}s); regression {rnd(R['regression'], 4)}", flush=True)
    arms = {}
    for arm in ARMS:
        A = {"seeds": {}}
        for seed in SEEDS:
            ts = time.time()
            H, ex, alphas = heads(D, out, arm, seed)
            S = {**Scl_all[arm], **H}
            add_fused(D, S)
            if arm == "crop":
                S["rms_full"], S["t2spe_full"] = Scl_all["full"]["rms"], Scl_all["full"]["t2spe"]
            U = {k: normalise(D, S[k])[0] for k in S}
            np.savez(out / f"scores_{arm}_s{seed}.npz", rel=D.rel, **{f"S_{k}": v for k, v in S.items()},
                     **{f"U_{k}": v for k, v in U.items()})
            best = ["rms", "t2spe", "rms_full", "t2spe_full"] if arm == "crop" else ["rms", "t2spe"]
            g = gate(D, S, U, arm)
            sc = scorecard(D, S, U, D.in_scope, D.mot, best, part=(seed == 0))
            dr = drift_static(D, S, ALPHA)
            sp = {h: {"good": float(spearmanr(S[h][D.in_scope & (D.label == 0)], D.dur_ratio[D.in_scope & (D.label == 0)])[0]),
                      "bad": float(spearmanr(S[h][D.in_scope & (D.label == 1)], D.dur_ratio[D.in_scope & (D.label == 1)])[0])}
                  for h in HEADS}
            A["seeds"][seed] = {"gate": g, "scorecard": sc, "drift_alpha_0.05": dr, "ridge_alpha": alphas,
                                "spearman_score_vs_duration_ratio": sp}
            print(f"{arm} seed {seed}: " + ", ".join(f"{h} AUC {g['heads'][h]['auc']:.3f} vs best classical "
                                                      f"{g['heads'][h]['best_classical_auc']:.3f} "
                                                      f"dCI {np.round(g['heads'][h]['A_delta_auc']['ci'], 3).tolist()} "
                                                      f"det {g['heads'][h]['det_at_far']:.2f} vs {g['heads'][h]['rms_det_at_far']:.2f} "
                                                      f"pass {g['heads'][h]['pass']}" for h in HEADS)
                  + f"  ({time.time() - ts:.0f}s)", flush=True)
            if seed == 0:
                ts = time.time()
                A["scorecard_machine_x_tf"] = scorecard(D, S, U, D.in_scope, D.mt, best)
                A["scorecard_later_only"] = scorecard(D, S, U, D.later, D.mot, best)
                A["drift_alpha_0.01"] = drift_static(D, S, ALPHA_STRICT)
                bsets = {"rms": ["rms"], "best_rms": ["rms", "rms_full"] if arm == "crop" else ["rms"]}
                A["matched_far"] = {f"{f:.2f}": matched_far(D, U, f, bsets) for f in (0.05, 0.10)}
                print(f"{arm} seed 0 tables done ({time.time() - ts:.0f}s)", flush=True)
                ts = time.time()
                A["requal"] = requal_curve(D, S, arm, cls_all[arm], out, seed, ex)
                A["cost_scenario"] = cost_table(A["requal"])
                timing[f"{arm}_requal_s"] = time.time() - ts
                print(f"{arm} seed 0 requal done ({time.time() - ts:.0f}s)", flush=True)
                ts = time.time()
                A["lomo"] = lomo(D, S, arm, out, seed)
                timing[f"{arm}_lomo_s"] = time.time() - ts
                print(f"{arm} seed 0 lomo done ({time.time() - ts:.0f}s)", flush=True)
                A["drift_subspace"], Z, Zraw, share = drift_section(D, arm, S)
                if arm == "crop":
                    A["fingerprints"] = fingerprints(D, Zraw)
                    keep = (S, U, ex, Zraw, share)
        names = list(A["seeds"][0]["scorecard"]["detectors"])
        A["seed_summary"] = {k: {"auc_by_seed": [A["seeds"][s]["scorecard"]["detectors"][k]["auc"] for s in SEEDS]}
                             for k in names}
        for k in names:
            v = A["seed_summary"][k]["auc_by_seed"]
            A["seed_summary"][k]["range"] = [min(v), max(v)]
        arms[arm] = A
        timing[f"{arm}_s"] = time.time() - T0
    # ---- decision
    dec = {}
    for arm in ARMS:
        dec[arm] = {h: all(arms[arm]["seeds"][s]["gate"]["heads"][h]["pass"] for s in SEEDS) for h in HEADS}
    passed = [h for h in HEADS if dec["crop"][h]]
    R["decision"] = {"gate_crop": dec["crop"], "gate_full_for_reference": dec["full"], "passed_heads": passed,
                     "headline_mode": "ai" if passed else "fallback_label_free_requalification",
                     "rule": "a head passes when conditions A and B hold for all 3 seeds in the crop arm",
                     "by_seed": {arm: {h: [{k: arms[arm]["seeds"][s]["gate"]["heads"][h][k]
                                            for k in ("auc", "best_classical_auc", "best_classical_name", "A_delta_auc",
                                                      "A_pass", "det_at_far",
                                                      "rms_det_at_far", "B_delta_det", "B_pass", "pass")}
                                           for s in SEEDS] for h in HEADS} for arm in ARMS}}
    primary = passed[0] if passed else "knn"
    S, U, ex, Z, share = keep
    R["primary_head"] = primary
    R["alarms"] = drilldown(D, "crop", S, U, ex, Z, share, primary)
    R["parts"] = parts_table(D, U)
    R["arms"] = arms
    timing["total_s"] = time.time() - T0
    R["runtime"] = timing
    res = json.loads(Path(results_path).read_text())
    res["ai"] = rnd(R)
    Path(results_path).write_text(json.dumps(res, separators=(",", ":")))
    print(json.dumps(rnd({"decision": R["decision"], "regression": R["regression"], "runtime": timing}, 4), indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
