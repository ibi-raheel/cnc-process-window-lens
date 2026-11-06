"""Stage 1: per-recording vibration features.

Each recording is a (samples, 3) array of X/Y/Z acceleration at FS = 2 kHz.
For every axis we compute time-domain statistics and Welch band powers.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import signal, stats

FS = 2000.0  # Hz, from the dataset README
AXES = ("x", "y", "z")
# log-spaced band edges, 10 Hz .. 1 kHz Nyquist
BAND_EDGES = np.geomspace(10.0, 1000.0, 9)
TIMEFRAMES = ["Feb_2019", "Aug_2019", "Feb_2020", "Aug_2020", "Feb_2021", "Aug_2021"]
NAME_RE = re.compile(r"(M0\d)_([A-Za-z]{3}_\d{4})_(OP\d\d)_(\d+)\.h5$")


def parse_name(path: str | Path) -> dict:
    m = NAME_RE.search(str(path))
    if not m:
        raise ValueError(f"unexpected file name {path}")
    machine, tf, op, idx = m.groups()
    return {"machine": machine, "timeframe": tf, "tf_index": TIMEFRAMES.index(tf),
            "op": op, "idx": int(idx)}


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def welch_psd(x: np.ndarray, fs: float = FS, nperseg: int = 1024):
    """One-sided Welch PSD, Hann window, 50% overlap, mean removed."""
    nper = min(nperseg, len(x))
    return signal.welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                        detrend="constant", scaling="density")


def band_powers(f: np.ndarray, pxx: np.ndarray, edges=BAND_EDGES) -> np.ndarray:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi) if hi < edges[-1] else (f >= lo) & (f <= hi)
        out.append(np.trapezoid(pxx[sel], f[sel]) if sel.sum() > 1 else 0.0)
    return np.asarray(out)


def feature_names() -> list[str]:
    names = []
    for a in AXES:
        names += [f"{a}_log_rms", f"{a}_log_peak", f"{a}_crest", f"{a}_kurtosis"]
        names += [f"{a}_band{i}_log" for i in range(len(BAND_EDGES) - 1)]
        names += [f"{a}_centroid_hz"]
    names += ["duration_s"]
    return names


def features(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    data = signal.detrend(data, axis=0, type="linear")
    feats = []
    for k in range(3):
        x = data[:, k]
        r = rms(x)
        pk = float(np.max(np.abs(x)))
        f, pxx = welch_psd(x)
        bp = band_powers(f, pxx)
        centroid = float(np.sum(f * pxx) / np.sum(pxx))
        feats += [np.log10(r + 1e-9), np.log10(pk + 1e-9), pk / (r + 1e-12),
                  float(stats.kurtosis(x, fisher=True))]
        feats += list(np.log10(bp + 1e-12))
        feats += [centroid]
    feats.append(len(data) / FS)
    return np.asarray(feats)


def load(path: str | Path) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as fh:
        return fh["vibration_data"][:]
