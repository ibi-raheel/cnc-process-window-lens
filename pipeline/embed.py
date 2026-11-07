"""Stage 5b: one MiniRocket embedding per (arm, seed, fold), cross-fitted leave-one-machine-out.

MiniRocket (Dempster, Schmidt and Webb, KDD 2021; sktime MiniRocketMultivariate) turns a
3-axis window into 9,996 features. Each feature is the proportion of positive values (PPV) of a
fixed length-9 convolution (weights in {-1, 2}, several dilations, random channel subsets) after
subtracting a bias taken from quantiles of the fitting windows.

Fold M (target machine M): the biases, a standard scaler and a 64-d PCA are fitted on windows of
2019 (qualification-period) GOOD parts of the other two machines only. No bad part and no part
of machine M is used for fitting. Every recording is embedded with that fold's transform, and
machine M's parts are only ever scored in fold M's space, so the embedding never saw the machine
it monitors.

Per (arm, seed, fold) it writes out/emb_<arm>_s<seed>_<M>.npz with
  E_mean (N, 9996) float32: mean PPV over a recording's K windows   -> head A (ridge)
  E_win  (N, K, 64) float32: standardised, PCA-64 PPV of each window -> head B (kNN memory bank)

Usage: python -I pipeline/embed.py pipeline/out [arm ...]   (arms: full crop; resumable)
Set OMP_NUM_THREADS / NUMBA_NUM_THREADS to 4 and run under nice.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
MACHINES = ("M01", "M02", "M03")
N_KERNELS = 10_000
N_FIT_WINDOWS = 4_000
PCA_DIM = 64
CHUNK = 100   # recordings per transform call
N_JOBS = 4


def fit_rows(machine: np.ndarray, tf: np.ndarray, label: np.ndarray, target: str) -> np.ndarray:
    """Recordings whose windows may fit fold `target`: 2019 good parts of the other machines."""
    return np.where((machine != target) & (tf <= 1) & (label == 0))[0]


def fit_windows(rows: np.ndarray, k: int, seed: int, fold: int, n: int = N_FIT_WINDOWS):
    rng = np.random.default_rng([seed, fold])
    return rng.choice(rows, n), rng.integers(0, k, n)


def embed_fold(W: np.ndarray, rows: np.ndarray, seed: int, fold: int):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    N, k, c, length = W.shape
    pr, pw = fit_windows(rows, k, seed, fold)
    Xfit = np.ascontiguousarray(W[pr, pw])
    mr = MiniRocketMultivariate(num_kernels=N_KERNELS, random_state=seed, n_jobs=N_JOBS)
    mr.fit(Xfit)
    tr = lambda A: np.asarray(mr.transform(A), dtype=np.float32)  # noqa: E731
    Ffit = tr(Xfit)
    sc = StandardScaler().fit(Ffit)
    pca = PCA(n_components=PCA_DIM, random_state=seed).fit(sc.transform(Ffit))
    E_mean = np.empty((N, Ffit.shape[1]), np.float32)
    E_win = np.empty((N, k, PCA_DIM), np.float32)
    for s in range(0, N, CHUNK):
        A = np.ascontiguousarray(W[s:s + CHUNK]).reshape(-1, c, length)
        F = tr(A)
        nn = F.shape[0] // k
        E_mean[s:s + nn] = F.reshape(nn, k, -1).mean(axis=1)
        E_win[s:s + nn] = pca.transform(sc.transform(F)).astype(np.float32).reshape(nn, k, PCA_DIM)
    return E_mean, E_win, pr, pw, float(pca.explained_variance_ratio_.sum())


def main(out_dir: str, arms: list[str]):
    out = Path(out_dir)
    M = np.load(out / "windows_meta.npz")
    machine, tf, label = M["machine"], M["tf"], M["label"]
    for arm in arms or ["crop", "full"]:
        W = np.load(out / f"windows_{arm}.npy", mmap_mode="r")
        for seed in SEEDS:
            for fi, tgt in enumerate(MACHINES):
                path = out / f"emb_{arm}_s{seed}_{tgt}.npz"
                if path.exists():
                    continue
                t0 = time.time()
                rows = fit_rows(machine, tf, label, tgt)
                assert not np.isin(machine[rows], [tgt]).any() and label[rows].max() == 0 and tf[rows].max() <= 1
                E_mean, E_win, pr, pw, ev = embed_fold(W, rows, seed, fi)
                np.savez(path, E_mean=E_mean, E_win=E_win, fit_rec=pr, fit_win=pw, pca_var=ev,
                         seconds=time.time() - t0, n_kernels=N_KERNELS, n_fit_windows=N_FIT_WINDOWS)
                print(f"{arm} seed {seed} fold {tgt}: E_mean {E_mean.shape}, PCA-{PCA_DIM} keeps "
                      f"{ev:.2f} of variance, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
