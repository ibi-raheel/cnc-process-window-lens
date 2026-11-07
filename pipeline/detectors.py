"""Stage 5c: every scorer on identical reference sets. Each score is out of sample for the part it
scores.

Reference set of a machine x operation cell: its qualification good parts (whole timeframes in time
order until n >= 15, i.e. Feb + Aug 2019), minus the two runs screened out in diagnostics.py.
Reference parts are scored leave-one-out.

Classical charts
  rms    Shewhart |log10 RMS - mu| / (3 sigma), vector-magnitude RMS
  cycle  the same chart on log10 cycle time (recording length): uses no vibration at all
  t2spe  PCA T^2 + SPE on the 9 monitored features, max(T^2/UCL/c_T2, SPE/Q/c_SPE) with
         split-conformal multipliers fitted on the pooled leave-one-out ratios (alpha = 0.01)
Head A, supervised: class-weighted ridge on the standardised mean-PPV MiniRocket embedding,
  solved in the dual. Standardisation and the ridge penalty are chosen on training rows only.
Head B, label-free: PatchCore-style memory bank of good windows. Score = mean over a part's K
  windows of the distance to the nearest window of any bank part other than itself, divided by
  the median leave-one-out score of the cell's bank.
Re-qualification helpers: hierarchical re-centring (one machine-wide offset plus a per-operation
  correction shrunk by n_c / (n_c + kappa)).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import MIN_TRAIN, pick_train  # noqa: E402
from spc import fit_pca  # noqa: E402
from stats import stratified_auc  # noqa: E402

RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0)
KAPPA = 5.0


# ---------------------------------------------------------------- reference sets
def reference_sets(machine, op, tf, label, idx, dropped) -> dict:
    """(machine, op) -> reference indices in time order; cells below MIN_TRAIN are skipped."""
    refs = {}
    for m in sorted(set(machine.tolist())):
        for o in sorted(set(op.tolist())):
            ii = np.where((machine == m) & (op == o))[0]
            if len(ii) == 0:
                continue
            ii = ii[np.lexsort((idx[ii], tf[ii]))]
            tr = pick_train(tf[ii], label[ii], range(6))
            if tr.sum() < MIN_TRAIN:
                continue
            tr &= ~dropped[ii]
            refs[(m, o)] = ii[tr]
    return refs


# ---------------------------------------------------------------- classical charts
def shewhart_oos(x: np.ndarray, ref: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """|x - mu| / (3 sigma) from the reference set; reference members are scored leave-one-out."""
    xr = x[ref]
    out = np.abs(x[idx] - xr.mean()) / (3 * max(xr.std(ddof=1), 1e-12))
    pos = {int(r): j for j, r in enumerate(ref)}
    for k, i in enumerate(idx):
        j = pos.get(int(i))
        if j is not None:
            r = np.delete(xr, j)
            out[k] = abs(x[i] - r.mean()) / (3 * max(r.std(ddof=1), 1e-12))
    return out


def pca_oos(X: np.ndarray, ref: np.ndarray, idx: np.ndarray):
    """T^2/UCL and SPE/Q ratios; reference members scored by the model fitted without them."""
    mod = fit_pca(X[ref])
    t2, spe = mod.scores(X[idx])
    t2, spe = t2 / mod.t2_lim, spe / mod.spe_lim
    pos = {int(r): j for j, r in enumerate(ref)}
    for k, i in enumerate(idx):
        j = pos.get(int(i))
        if j is not None:
            pm = fit_pca(np.delete(X[ref], j, axis=0))
            a, b = pm.scores(X[i][None, :])
            t2[k], spe[k] = a[0] / pm.t2_lim, b[0] / pm.spe_lim
    return t2, spe, mod


def t2spe_shifted(mod, X: np.ndarray, shift: np.ndarray, c_t2: float, c_spe: float) -> np.ndarray:
    """Combined calibrated ratio with the cell's static PCA model, its centre moved by `shift`."""
    Z = (X - mod.mu - shift) / mod.sd
    T = Z @ mod.P
    t2 = np.sum(T ** 2 / mod.lam, axis=1)
    R = Z - T @ mod.P.T
    spe = np.sum(R ** 2, axis=1)
    return np.maximum(t2 / mod.t2_lim / c_t2, spe / mod.spe_lim / c_spe)


# ---------------------------------------------------------------- head B: kNN memory bank
def nearest_windows(Q: np.ndarray, B: np.ndarray):
    """Q (nq, K, d), B (nb, K, d) -> D (nq, K, nb): distance from each query window to the nearest
    window of each bank part, and V (nq, K, nb): which bank window that was."""
    nq, k, d = Q.shape
    nb, kb, _ = B.shape
    q = Q.reshape(-1, d).astype(np.float64)
    b = B.reshape(-1, d).astype(np.float64)
    D2 = np.sum(q * q, 1)[:, None] + np.sum(b * b, 1)[None, :] - 2.0 * q @ b.T
    D = np.sqrt(np.maximum(D2, 0.0)).reshape(nq, k, nb, kb)
    return D.min(axis=3), D.argmin(axis=3)


def knn_score(E: np.ndarray, idx: np.ndarray, bank: np.ndarray):
    """Raw kNN score of each part in idx against bank (a part never matches itself).
    Returns raw (n,), per-window nearest distance (n, K), matched bank part (n, K) and matched
    bank window (n, K)."""
    D, V = nearest_windows(E[idx], E[bank])
    self_ = idx[:, None] == bank[None, :]
    D = np.where(self_[:, None, :], np.inf, D)
    j = D.argmin(axis=2)                                   # (n, K) index into bank
    dist = np.take_along_axis(D, j[..., None], 2)[..., 0]
    win = np.take_along_axis(V, j[..., None], 2)[..., 0]
    return dist.mean(axis=1), dist, bank[j], win


# ---------------------------------------------------------------- head A: dual ridge
def standardise(E: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    mu = E[fit_rows].mean(axis=0)
    sd = E[fit_rows].std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((E - mu) / sd).astype(np.float32)


def ridge_dual(Kmat: np.ndarray, tr: np.ndarray, te: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Weighted ridge on y in {-1, +1} with kernel K = Z Z^T / p + 1, class-balanced weights w:
    solve (K_SS + alpha W^-1) a = y_S; score(x) = K_xS a. Equals the primal weighted ridge on
    [Z / sqrt(p), 1]."""
    yt = np.where(y[tr] == 1, 1.0, -1.0)
    n1, n0 = max((yt > 0).sum(), 1), max((yt < 0).sum(), 1)
    w = np.where(yt > 0, len(yt) / (2 * n1), len(yt) / (2 * n0))
    a = np.linalg.solve(Kmat[np.ix_(tr, tr)] + np.diag(alpha / w), yt)
    return Kmat[np.ix_(te, tr)] @ a


def ridge_head(E: np.ndarray, y: np.ndarray, groups: np.ndarray, strata: np.ndarray,
               train: np.ndarray, test: np.ndarray):
    """Standardise on `train`, pick alpha by leave-one-group-out CV inside `train`, fit on all of
    `train`, score `test`. Nothing from `test` touches the scaler or the penalty."""
    rows = np.union1d(train, test)
    Z = standardise(E[rows], np.searchsorted(rows, train)).astype(np.float64)
    Kmat = Z @ Z.T / Z.shape[1] + 1.0
    tr, te = np.searchsorted(rows, train), np.searchsorted(rows, test)
    yl, gl, sl = y[rows], groups[rows], strata[rows]
    best, best_a = -1.0, RIDGE_ALPHAS[0]
    for a in RIDGE_ALPHAS:
        s = np.full(len(rows), np.nan)
        for g in np.unique(gl[tr]):
            hold = tr[gl[tr] == g]
            fit = tr[gl[tr] != g]
            if yl[fit].max() == 0:
                continue
            s[hold] = ridge_dual(Kmat, fit, hold, yl, a)
        ok = tr[np.isfinite(s[tr])]
        v = stratified_auc(s[ok], yl[ok], sl[ok])
        if np.isfinite(v) and v > best:
            best, best_a = v, a
    return ridge_dual(Kmat, tr, te, yl, best_a), best_a


# ---------------------------------------------------------------- re-qualification
def hier_shift(off: np.ndarray, cell: np.ndarray, kappa: float = KAPPA):
    """Machine-wide offset delta = mean of the N reference offsets, plus a per-cell correction
    w_c (mean_c - delta) with w_c = n_c / (n_c + kappa). Returns (delta, {cell: shift})."""
    off = np.asarray(off, float)
    delta = off.mean(axis=0)
    per = {}
    for c in np.unique(cell):
        v = off[cell == c]
        w = len(v) / (len(v) + kappa)
        per[c] = delta + w * (v.mean(axis=0) - delta)
    return delta, per


def shift_for(cells, delta, per) -> np.ndarray:
    return np.array([per.get(c, delta) for c in cells])


def hier_shift_loo(off: np.ndarray, cell: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    """Shift applied to each reference part when it is left out of the estimate (for calibration)."""
    out = []
    for i in range(len(off)):
        keep = np.arange(len(off)) != i
        d, per = hier_shift(off[keep], cell[keep], kappa)
        out.append(per.get(cell[i], d))
    return np.array(out)
