"""Stage 5a: raw vibration windows for the MiniRocket heads, in two arms.

full : K windows of L samples evenly spaced over each recording's own length (the scheme the
       four prototypes used). A truncated recording therefore samples different toolpath phases.
crop : the duration control. Per machine x operation cell, a common prefix of
       T_c = floor(0.45 x median length of the cell's qualification good parts) samples, and K
       windows at identical start indices linspace(0, T_c - L, K) for every recording of the cell.
       Recording length then carries no information. Every recording must cover T_c (asserted;
       on this data the shortest recording is 0.494 x its cell's qualification median).

Per recording it also stores the vector-magnitude RMS of the whole recording and of the prefix
(the RMS chart in each arm), the repo's 40 features on the prefix (T^2+SPE in the crop arm),
per-window log10 band power on ATTR_EDGES (the repo's 8 Welch bands with each band above 100 Hz
split into 4 log sub-bands: 20 bands per axis, at least 4 FFT bins each), window start indices,
length, and the path relative to the data directory. Good and bad files share basenames, so
rows are keyed by relative path.

Usage: python -I pipeline/windows.py <data_dir> pipeline/out
Writes out/windows_full.npy and out/windows_crop.npy (N x K x 3 x L float32, about 0.5 GB each)
and out/windows_meta.npz.
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import MIN_TRAIN, pick_train  # noqa: E402
from features import BAND_EDGES, FS, TIMEFRAMES, features, load, parse_name, rms  # noqa: E402

L = 1024          # 0.512 s per window
K = 24            # windows per recording
CROP_FRAC = 0.45  # common prefix as a fraction of the cell's qualification median length
_hi = np.concatenate([np.geomspace(BAND_EDGES[i], BAND_EDGES[i + 1], 5)[:-1] for i in range(4, 8)])
ATTR_EDGES = np.concatenate([BAND_EDGES[:4], _hi, [BAND_EDGES[-1]]])   # 21 edges -> 20 bands


def relpath(path: str | Path, root: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def label_of(rel: str) -> int:
    parts = rel.split("/")
    if "bad" in parts:
        return 1
    if "good" in parts:
        return 0
    raise ValueError(f"no good/bad folder in {rel}")


def full_starts(n: int, k: int = K, length: int = L) -> np.ndarray:
    return np.linspace(0, n - length, k).astype(int)


def prefix_starts(T: int, k: int = K, length: int = L) -> np.ndarray:
    """Start-anchored, fixed-stride windows inside [0, T): identical for every recording of a cell."""
    if T < length:
        raise ValueError("prefix shorter than one window")
    return np.linspace(0, T - length, k).astype(int)


def crop_lengths(n: np.ndarray, cell: np.ndarray, tf: np.ndarray, label: np.ndarray,
                 frac: float = CROP_FRAC) -> dict:
    """Per cell: T_c = floor(frac x median length of its qualification good parts), capped at the
    shortest recording of the cell so that every recording covers the prefix."""
    out = {}
    for c in np.unique(cell):
        ii = np.where(cell == c)[0]
        q = pick_train(tf[ii], label[ii], range(len(TIMEFRAMES)))
        ref = ii[q] if q.sum() else ii[label[ii] == 0]
        T = int(np.floor(frac * np.median(n[ref])))
        out[c] = {"T": min(T, int(n[ii].min())), "T_uncapped": T, "n_ref": int(len(ref)),
                  "qual_median": float(np.median(n[ref])), "min_len": int(n[ii].min()),
                  "n_qual_ge_min_train": bool(q.sum() >= MIN_TRAIN)}
    return out


def cut(d: np.ndarray, starts: np.ndarray, length: int = L) -> np.ndarray:
    """(n, 3) signal -> (k, 3, L) float32 windows, mean removed per window and axis."""
    w = np.stack([d[s:s + length] for s in starts])
    w = w - w.mean(axis=1, keepdims=True)
    return np.transpose(w, (0, 2, 1)).astype(np.float32)


def band_log_power(w: np.ndarray, edges: np.ndarray = ATTR_EDGES) -> np.ndarray:
    """(..., L) windows -> (..., n_bands) log10 band power. Welch with nperseg = L (one Hann
    segment, the repo's settings), integrated by the trapezoid rule as in features.band_powers."""
    f, p = signal.welch(w, fs=FS, window="hann", nperseg=w.shape[-1], noverlap=w.shape[-1] // 2,
                        detrend="constant", scaling="density", axis=-1)
    out = []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (f >= lo) & ((f < hi) if j < len(edges) - 2 else (f <= hi))
        out.append(np.trapezoid(p[..., sel], f[sel], axis=-1))
    return np.log10(np.stack(out, axis=-1) + 1e-12).astype(np.float32)


def vec_rms(d: np.ndarray) -> float:
    """The repo's RMS-chart statistic (extract.py): RMS of the mean-removed vector magnitude."""
    dt = d - d.mean(axis=0)
    return rms(np.linalg.norm(dt, axis=1))


def one(args):
    path, T = args
    d = np.asarray(load(path), dtype=np.float64)
    n = len(d)
    sf, sc = full_starts(n), prefix_starts(T)
    assert n >= T and sc[-1] + L <= T
    wf, wc = cut(d, sf), cut(d, sc)
    return (wf, wc, band_log_power(wf), band_log_power(wc), sf, sc, n,
            vec_rms(d), vec_rms(d[:T]), features(d[:T]))


def main(data_dir: str, out_dir: str):
    t0 = time.time()
    import h5py
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(data_dir).rglob("*.h5"), key=lambda p: relpath(p, data_dir))
    rel = [relpath(p, data_dir) for p in files]
    meta = [parse_name(r) for r in rel]
    machine = np.array([m["machine"] for m in meta]); op = np.array([m["op"] for m in meta])
    tf = np.array([m["tf_index"] for m in meta]); idx = np.array([m["idx"] for m in meta])
    label = np.array([label_of(r) for r in rel])
    n_len = np.empty(len(files), int)
    for i, p in enumerate(files):
        with h5py.File(p, "r") as fh:
            n_len[i] = fh["vibration_data"].shape[0]
    cell = np.char.add(np.char.add(machine, "-"), op)
    crops = crop_lengths(n_len, cell, tf, label)
    T_row = np.array([crops[c]["T"] for c in cell])
    assert (n_len >= T_row).all(), "a recording is shorter than its cell's prefix"
    N = len(files)
    Wf = np.lib.format.open_memmap(out / "windows_full.npy", "w+", np.float32, (N, K, 3, L))
    Wc = np.lib.format.open_memmap(out / "windows_crop.npy", "w+", np.float32, (N, K, 3, L))
    nb = len(ATTR_EDGES) - 1
    band_full = np.empty((N, K, 3, nb), np.float32); band_crop = np.empty((N, K, 3, nb), np.float32)
    st_full = np.empty((N, K), int); st_crop = np.empty((N, K), int)
    rms_full = np.empty(N); rms_pre = np.empty(N); f40 = np.empty((N, 40))
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap(one, [(str(p), int(T)) for p, T in zip(files, T_row)], chunksize=8)):
            wf, wc, bf, bc, sf, sc, n, rf, rp, ff = r
            assert n == n_len[i]
            Wf[i], Wc[i], band_full[i], band_crop[i] = wf, wc, bf, bc
            st_full[i], st_crop[i], rms_full[i], rms_pre[i], f40[i] = sf, sc, rf, rp, ff
    Wf.flush(); Wc.flush()
    np.savez_compressed(out / "windows_meta.npz", relpath=np.array(rel), machine=machine, op=op, tf=tf,
                        idx=idx, label=label, n_samples=n_len, crop_len=T_row, starts_full=st_full,
                        starts_crop=st_crop, band_full=band_full, band_crop=band_crop, rms_full=rms_full,
                        rms_prefix=rms_pre, feat40_prefix=f40, attr_edges=ATTR_EDGES, fs=FS, L=L, K=K,
                        crop_frac=CROP_FRAC)
    lens = np.array([v["T"] / v["qual_median"] for v in crops.values()])
    print(f"{N} recordings, {len(crops)} cells, K={K} x L={L}; crop T/qual-median "
          f"{lens.min():.3f}-{lens.max():.3f}; capped cells "
          f"{sum(v['T'] < v['T_uncapped'] for v in crops.values())}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
