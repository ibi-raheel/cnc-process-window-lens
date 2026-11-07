"""Stage 5d: forward-in-time drift subspace, per-alarm drift/defect label and per-(axis, band)
attribution. No neural network.

Band vector of a part: log10 of the mean band power over its K windows, on the 20 attribution
bands per axis (60 values). Each part is z-scored against its cell's reference parts:
  z = (b - mu_c) / s_c,   s_c^2 = (df_c s_c^2 + NU s_glob^2) / (df_c + NU)
(per-feature SD shrunk towards the median SD over cells with NU = 10 pseudo-parts; z clipped
at +-8).

Drift subspace for period t (forward in time only): the top K directions (uncentred SVD) of
benign shifts measured on GOOD parts from periods before t: the mean z of each cell's good parts
in each earlier period with at least 3 of them (including within-qualification variation),
plus qualification-period machine-to-machine offsets of the same operation. Only cells whose
qualification ended before t contribute (a few cells qualify into 2020-21 because they had too
few 2019 good parts), and the SD shrinkage target uses 2019-qualified cells only. A part in
period t is never explained with directions or scales estimated from period t or later.

Per-alarm label: share = ||U_t^T z||^2 / ||z||^2. share >= CUT (pre-declared 0.5) reads
"nuisance drift: re-baseline the machine"; otherwise "out of window: investigate". A random
K-dimensional direction has expected share K/p (= 0.05 here).
"""
from __future__ import annotations

import numpy as np

NU = 10.0
Z_CLIP = 8.0
K_DRIFT = 3
CUT = 0.5


def band_vectors(band: np.ndarray) -> np.ndarray:
    """(N, K, 3, nb) per-window log10 band power -> (N, 3 * nb) log10 mean power over windows."""
    p = np.mean(10.0 ** band.astype(np.float64), axis=1)
    return np.log10(p).reshape(len(band), -1)


def cell_z(B: np.ndarray, refs: dict, cell_of: dict, rows: np.ndarray,
           glob_from: list | None = None, clip: bool = True) -> tuple[np.ndarray, dict]:
    """z-score every part in `rows` against its cell's reference parts (shrunk SDs). The shrinkage
    target is the median reference variance over the cells in `glob_from` (default: all cells).
    clip=True bounds z at +-Z_CLIP (used for the drift subspace, so one extreme band cannot set a
    direction); attribution ranks unclipped z, since clipped values tie at the bound."""
    var, df, mu = {}, {}, {}
    for c, ref in refs.items():
        mu[c] = B[ref].mean(axis=0)
        var[c] = B[ref].var(axis=0, ddof=1)
        df[c] = len(ref) - 1
    glob = np.median(np.stack([var[c] for c in (glob_from or list(refs))]), axis=0)
    sd = {c: np.sqrt((df[c] * var[c] + NU * glob) / (df[c] + NU)) for c in refs}
    Z = np.full(B.shape, np.nan)
    for i in rows:
        c = cell_of[int(i)]
        z = (B[i] - mu[c]) / sd[c]
        Z[i] = np.clip(z, -Z_CLIP, Z_CLIP) if clip else z
    return Z, {"mu": mu, "sd": sd}


def shift_vectors(Z, cell, tf, label, keep, refs, stats, before_t: int, cells_ops):
    """Benign shifts from good parts in periods < before_t, plus qualification machine offsets.
    Returns (V, used): the shift vectors and every row index that went into them, so callers can
    assert that nothing from period before_t or later was used."""
    V, used = [], []
    for c in refs:
        for t in range(before_t):
            ii = np.where((cell == c) & (tf == t) & (label == 0) & keep)[0]
            if len(ii) >= 3:
                V.append(Z[ii].mean(axis=0))
                used.append(ii)
    for o, cs in cells_ops.items():
        cs = [c for c in cs if c in refs]
        if len(cs) > 1:
            om = np.mean([stats["mu"][c] for c in cs], axis=0)
            for c in cs:
                V.append(np.clip((stats["mu"][c] - om) / stats["sd"][c], -Z_CLIP, Z_CLIP))
                used.append(np.asarray(refs[c]))
    used = np.unique(np.concatenate(used)) if used else np.array([], int)
    return np.asarray(V), used


def subspace(V: np.ndarray, k: int = K_DRIFT):
    _, s, Vt = np.linalg.svd(V, full_matrices=False)
    return Vt[:k].T, (s ** 2 / np.sum(s ** 2))[:k]


def drift_share(z: np.ndarray, U: np.ndarray) -> np.ndarray:
    z = np.atleast_2d(z)
    num = np.sum((z @ U) ** 2, axis=1)
    den = np.sum(z ** 2, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def label_alarm(share: float, cut: float = CUT) -> str:
    return "nuisance drift: re-baseline" if share >= cut else "out of window: investigate"


def top_bands(v: np.ndarray, names: list[str], n: int = 3) -> list[dict]:
    """The n bands with the largest |value|, reported with their sign (a defect can remove energy
    from a band as well as add it)."""
    order = np.argsort(-np.abs(v), kind="stable")[:n]
    return [{"band": names[j], "z": round(float(v[j]), 2)} for j in order]


def band_names(edges: np.ndarray) -> list[str]:
    return [f"{a} {edges[j]:.0f}-{edges[j + 1]:.0f} Hz" for a in "xyz" for j in range(len(edges) - 1)]
