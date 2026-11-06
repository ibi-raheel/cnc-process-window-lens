"""Stage 2: multivariate SPC (PCA Hotelling T^2 + SPE/Q) and the RMS Shewhart baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

ALPHA = 0.01


def t2_ucl(p: int, n: int, alpha: float = ALPHA) -> float:
    """Phase II Hotelling T^2 limit for a new observation.

    UCL = p (n-1)(n+1) / (n (n-p)) * F_{1-alpha}(p, n-p)
    """
    if n <= p:
        raise ValueError("need n > p")
    return p * (n - 1) * (n + 1) / (n * (n - p)) * stats.f.ppf(1 - alpha, p, n - p)


def spe_limit(resid_eigs: np.ndarray, alpha: float = ALPHA) -> float:
    """Jackson-Mudholkar (1979) limit for the squared prediction error Q."""
    lam = np.asarray(resid_eigs, dtype=float)
    lam = lam[lam > 1e-12]
    if lam.size == 0:
        return 1e-9
    th1, th2, th3 = (np.sum(lam ** i) for i in (1, 2, 3))
    h0 = 1.0 - 2.0 * th1 * th3 / (3.0 * th2 ** 2)
    z = stats.norm.ppf(1 - alpha)
    term = z * np.sqrt(2.0 * th2 * h0 ** 2) / th1 + 1.0 + th2 * h0 * (h0 - 1.0) / th1 ** 2
    return float(th1 * term ** (1.0 / h0))


@dataclass
class PcaMonitor:
    mu: np.ndarray
    sd: np.ndarray
    P: np.ndarray        # loadings (features x k)
    lam: np.ndarray      # retained eigenvalues
    t2_lim: float
    spe_lim: float
    k: int
    n: int

    def scores(self, X: np.ndarray):
        Z = (X - self.mu) / self.sd
        T = Z @ self.P
        t2 = np.sum(T ** 2 / self.lam, axis=1)
        R = Z - T @ self.P.T
        spe = np.sum(R ** 2, axis=1)
        return t2, spe

    def ratio(self, X: np.ndarray) -> np.ndarray:
        """Combined statistic: max(T2/UCL, SPE/Q). Alarm when > 1."""
        t2, spe = self.scores(X)
        return np.maximum(t2 / self.t2_lim, spe / self.spe_lim)


def fit_pca(X: np.ndarray, var_frac: float = 0.90, k_max: int | None = None,
            alpha: float = ALPHA) -> PcaMonitor:
    X = np.asarray(X, dtype=float)
    n, m = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (X - mu) / sd
    # eigen-decomposition of the sample covariance via SVD
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    eig = s ** 2 / (n - 1)
    cum = np.cumsum(eig) / np.sum(eig)
    k = int(np.searchsorted(cum, var_frac) + 1)
    cap = max(1, n // 3) if k_max is None else k_max
    k = max(1, min(k, cap, n - 2))
    P = Vt[:k].T
    lam = eig[:k]
    return PcaMonitor(mu, sd, P, lam, t2_ucl(k, n, alpha), spe_limit(eig[k:], alpha), k, n)


@dataclass
class ShewhartRms:
    mu: float
    sigma: float

    def ratio(self, rms: np.ndarray) -> np.ndarray:
        """|x - mu| / 3 sigma. Alarm when > 1."""
        return np.abs(np.asarray(rms) - self.mu) / (3.0 * self.sigma)


def fit_shewhart(rms: np.ndarray) -> ShewhartRms:
    rms = np.asarray(rms, dtype=float)
    return ShewhartRms(float(rms.mean()), float(max(rms.std(ddof=1), 1e-12)))
