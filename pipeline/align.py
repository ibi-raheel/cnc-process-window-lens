"""Stage 5a': program-phase alignment, the check that the crop arm is really duration-blind.

The crop arm cuts every recording of a cell to a common start-anchored prefix. That removes
recording LENGTH, but it only removes the truncation shortcut if every recording starts at the
same point of the NC program. This stage measures where each recording starts.

Envelope: log10 RMS of the mean-removed vector magnitude in frames of F = 100 samples (50 ms).
Template of a cell: the median envelope of its qualification good parts (the same parts that set
the crop length in windows.py; no bad part, no later part), cut to the shortest of them.
Start offset of a recording: the lag (in frames, -2 s .. as far as the template allows) that
maximises the Pearson correlation between the first W frames of its envelope and the template,
W = min(its length, half the template). A positive lag means the recording starts that much later
in the program than the qualification parts. Labels are never used to estimate a lag.

Usage: python -I pipeline/align.py <data_dir> pipeline/out   (run before windows.py)
Writes out/align.npz (rows in windows.py's order: sorted relative paths):
  lag (N,) samples, corr (N,), W (N,) frames, F, and the per-cell template lengths.
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import pick_train  # noqa: E402
from features import FS, TIMEFRAMES, load, parse_name  # noqa: E402

F = 100                      # samples per envelope frame (50 ms)
MIN_LAG_S = 2.0              # search from -2 s


def envelope(d: np.ndarray, frame: int = F) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    v = np.linalg.norm(d - d.mean(axis=0), axis=1)
    m = len(v) // frame
    return np.log10(np.sqrt(np.mean(v[:m * frame].reshape(m, frame) ** 2, axis=1)) + 1e-9)


def best_lag(e: np.ndarray, tmpl: np.ndarray, min_lag: int) -> tuple[int, float, int]:
    """Lag (frames) of e's first W frames inside tmpl that maximises Pearson r; returns (lag, r, W)."""
    W = int(min(len(e), len(tmpl) // 2))
    x = e[:W]
    x = (x - x.mean()) / (x.std() + 1e-12)
    best_r, best_l = -np.inf, 0
    for lag in range(-min_lag, len(tmpl) - W + 1):
        if lag < 0:
            a, b = x[-lag:], tmpl[:W + lag]
        else:
            a, b = x, tmpl[lag:lag + W]
        if len(a) < W // 2:
            continue
        b = (b - b.mean()) / (b.std() + 1e-12)
        r = float(np.mean(a * b))
        if r > best_r:
            best_r, best_l = r, lag
    return best_l, best_r, W


def _env(path: str) -> np.ndarray:
    return envelope(load(path))


def main(data_dir: str, out_dir: str):
    t0 = time.time()
    from windows import label_of, relpath
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(data_dir).rglob("*.h5"), key=lambda p: relpath(p, data_dir))
    rel = np.array([relpath(p, data_dir) for p in files])
    meta = [parse_name(r) for r in rel]
    machine = np.array([m["machine"] for m in meta]); op = np.array([m["op"] for m in meta])
    tf = np.array([m["tf_index"] for m in meta]); label = np.array([label_of(r) for r in rel])
    with Pool(4) as pool:
        env = pool.map(_env, [str(Path(data_dir) / r) for r in rel], chunksize=8)
    cell = np.char.add(np.char.add(machine, "-"), op)
    N = len(rel)
    lag, corr, Wf = np.zeros(N, int), np.zeros(N), np.zeros(N, int)
    tlen = {}
    min_lag = int(MIN_LAG_S * FS / F)
    for c in np.unique(cell):
        ii = np.where(cell == c)[0]
        q = pick_train(tf[ii], label[ii], range(len(TIMEFRAMES)))
        ref = ii[q] if q.sum() else ii[label[ii] == 0]
        n = min(len(env[i]) for i in ref)
        tmpl = np.median(np.stack([env[i][:n] for i in ref]), axis=0)
        tlen[c] = n
        for i in ii:
            l, r, w = best_lag(env[i], tmpl, min_lag)
            lag[i], corr[i], Wf[i] = l * F, r, w
    np.savez(out / "align.npz", relpath=rel, lag=lag, corr=corr, W=Wf, F=F,
             cells=np.array(list(tlen)), template_frames=np.array(list(tlen.values())))
    s = lag / FS
    for name, m in (("good", label == 0), ("bad", label == 1)):
        print(f"{name}: lag s quantiles 5/50/95% {np.round(np.quantile(s[m], [0.05, 0.5, 0.95]), 2).tolist()}, "
              f"share > 1 s {np.mean(s[m] > 1):.3f}, median r {np.median(corr[m]):.2f}")
    print(f"{N} recordings aligned in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
