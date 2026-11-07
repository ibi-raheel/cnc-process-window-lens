"""Shared statistics for the classical charts (diagnostics.py) and the MiniRocket heads
(ai_eval.py): binomial intervals, like-for-like AUC, split-conformal limits with their exact
finite-sample guarantee, and paired bootstraps. numpy / scikit-learn only.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from sklearn.metrics import average_precision_score

Z95 = 1.959964


# ---------------------------------------------------------------- rates and intervals
def wilson(k: int, n: int, z: float = Z95) -> dict:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return {"rate": None, "lo": None, "hi": None, "k": 0, "n": 0}
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h), "k": int(k), "n": int(n)}


def ci(v, q: tuple[float, float] = (0.025, 0.975)) -> list[float]:
    """Percentile interval of bootstrap replicates (NaN replicates dropped)."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(v, q[0])), float(np.quantile(v, q[1]))]


# ---------------------------------------------------------------- split-conformal limits
def conformal_multiplier(ratios: np.ndarray, alpha: float = 0.01) -> float:
    """Legacy multiplier used by diagnostics.py: the ceil((n+1)(1-alpha))-th smallest held-out
    ratio, with the rank clipped to n. Prefer conformal_threshold, which returns +inf when no
    finite limit carries the guarantee."""
    r = np.sort(np.asarray(ratios, float))
    rank = math.ceil((len(r) + 1) * (1 - alpha))
    return float(r[min(rank, len(r)) - 1])


def conformal_rank(n: int, alpha: float) -> int:
    """k = ceil((n+1)(1-alpha)), computed exactly with alpha as a fraction (no float round-off)."""
    a = Fraction(alpha).limit_denominator(1_000_000)
    num = (n + 1) * (a.denominator - a.numerator)
    return -(-num // a.denominator)


def conformal_threshold(cal, alpha: float) -> float:
    """Split-conformal upper limit: the k-th smallest of n calibration scores, k = ceil((n+1)(1-alpha)).
    For a new score exchangeable with the calibration scores, P(new > limit) <= alpha.
    If k > n no finite limit carries that guarantee, so the limit is +inf."""
    s = np.sort(np.asarray(cal, float))
    k = conformal_rank(len(s), alpha)
    return float("inf") if k > len(s) else float(s[k - 1])


def min_n_for_alpha(alpha: float) -> int:
    """Smallest calibration size with a finite split-conformal limit (19 at 0.05, 99 at 0.01)."""
    n = 1
    while conformal_rank(n, alpha) > n:
        n += 1
    return n


def exact_far(n: int, alpha: float) -> float:
    """Exact marginal false-alarm rate of the conformal limit for continuous exchangeable scores:
    P(new > s_(k)) = 1 - k/(n+1) <= alpha. NaN when the limit is +inf."""
    k = conformal_rank(n, alpha)
    return float("nan") if k > n else 1.0 - k / (n + 1)


def max_of_n_far(n: int) -> float:
    """P(new > max of n exchangeable continuous scores) = 1/(n+1)."""
    return 1.0 / (n + 1)


def limit_with_fallback(cal, alpha: float) -> tuple[float, dict]:
    """Conformal limit when it is finite, otherwise the max-of-N limit. Returns the limit and
    the false-alarm rate it guarantees (alpha-level bound for conformal, 1/(N+1) for max-of-N)."""
    s = np.sort(np.asarray(cal, float))
    n = len(s)
    if n == 0:
        return float("inf"), {"rule": "none", "n": 0, "guarantee": 0.0}
    k = conformal_rank(n, alpha)
    if k <= n:
        return float(s[k - 1]), {"rule": "conformal", "n": n, "rank": k, "guarantee": 1.0 - k / (n + 1)}
    return float(s[-1]), {"rule": "max_of_n", "n": n, "rank": n, "guarantee": max_of_n_far(n)}


# ---------------------------------------------------------------- like-for-like AUC
def strata_ids(*keys) -> np.ndarray:
    tup = list(zip(*keys))
    uniq = {k: i for i, k in enumerate(sorted(set(tup)))}
    return np.array([uniq[k] for k in tup])


def eligible(strata: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Parts whose stratum holds at least one bad AND one good part."""
    ok = np.zeros(len(y), bool)
    for s in np.unique(strata):
        m = strata == s
        if y[m].min() == 0 and y[m].max() == 1:
            ok |= m
    return ok


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


def pooled_ap(score, y) -> float:
    return float(average_precision_score(y, score))


def _pair_matrices(score, y, strata, keys):
    return [((score[(strata == s) & (y == 1)][:, None] > score[(strata == s) & (y == 0)][None, :])
             + 0.5 * (score[(strata == s) & (y == 1)][:, None] == score[(strata == s) & (y == 0)][None, :]))
            for s in keys]


def cluster_bootstrap_auc(scores: dict, y, strata, n_boot: int = 2000, seed: int = 0, clusters=None) -> dict:
    """Within-stratum AUC per scorer plus replicates from resampling whole strata with replacement.
    The same resamples are used for every scorer, so replicate-wise differences are paired.
    Only strata holding both classes are used (pass eligible parts).
    With `clusters` (one id per part, constant within a stratum), whole clusters of strata are
    resampled instead, e.g. machine x timeframe blocks, whose operations share one production episode."""
    y = np.asarray(y)
    strata = np.asarray(strata)
    keys = [s for s in np.unique(strata) if (y[strata == s] == 1).any() and (y[strata == s] == 0).any()]
    rng = np.random.default_rng(seed)
    S = len(keys)
    if clusters is None:
        W = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(n_boot)]).astype(float)
        n_units, unit = S, "whole strata"
    else:
        clusters = np.asarray(clusters)
        of = []
        for k in keys:
            c = np.unique(clusters[strata == k])
            assert len(c) == 1, "strata must be nested in clusters"
            of.append(c[0])
        ucl, cidx = np.unique(np.array(of), return_inverse=True)
        C = len(ucl)
        Wc = np.stack([np.bincount(rng.integers(0, C, C), minlength=C) for _ in range(n_boot)]).astype(float)
        W = Wc[:, cidx]
        n_units, unit = C, "whole clusters of strata"
    out = {}
    for name, s in scores.items():
        s = np.asarray(s, float)
        mats = _pair_matrices(s, y, strata, keys)
        num = np.array([m.sum() for m in mats])
        den = np.array([m.size for m in mats], float)
        out[name] = {"point": float(num.sum() / den.sum()), "boots": (W @ num) / (W @ den)}
    out["_design"] = {"n_strata": S, "n_units": n_units, "n_boot": n_boot, "resample": unit}
    return out


def part_bootstrap_auc(scores: dict, y, strata, n_boot: int = 2000, seed: int = 0) -> dict:
    """Secondary interval: bad parts resampled across strata (each keeps its own stratum) and good
    parts resampled within their stratum. Paired across scorers."""
    y = np.asarray(y)
    strata = np.asarray(strata)
    keys = [s for s in np.unique(strata) if (y[strata == s] == 1).any() and (y[strata == s] == 0).any()]
    nb = [int(((strata == s) & (y == 1)).sum()) for s in keys]
    ng = [int(((strata == s) & (y == 0)).sum()) for s in keys]
    off = np.cumsum([0] + nb)
    rng = np.random.default_rng(seed)
    Wb, Wg = [], []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, off[-1], off[-1]), minlength=off[-1]).astype(float)
        Wb.append([cnt[off[k]:off[k + 1]] for k in range(len(keys))])
        Wg.append([np.bincount(rng.integers(0, g, g), minlength=g).astype(float) for g in ng])
    out = {}
    for name, s in scores.items():
        mats = _pair_matrices(np.asarray(s, float), y, strata, keys)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            num = sum(Wb[b][k] @ mats[k] @ Wg[b][k] for k in range(len(keys)))
            den = sum(Wb[b][k].sum() * Wg[b][k].sum() for k in range(len(keys)))
            boots[b] = num / den if den else np.nan
        tot = sum(m.sum() for m in mats) / sum(m.size for m in mats)
        out[name] = {"point": float(tot), "boots": boots}
    out["_design"] = {"n_strata": len(keys), "n_boot": n_boot,
                      "resample": "bad parts across strata, good parts within stratum"}
    return out


def boot_delta(point: float, boots: np.ndarray, base_points: list[float], base_boots: list[np.ndarray]) -> dict:
    """Paired difference against the best of one or more baselines. With several baselines the
    comparison is with their replicate-wise maximum, so picking the best baseline after the fact
    is paid for inside the interval."""
    bb = np.max(np.stack(base_boots), axis=0)
    d = np.asarray(boots, float) - bb
    ok = ~np.isnan(d)
    return {"est": float(point - max(base_points)), "ci": ci(d), "ci_bonferroni2": ci(d, (0.0125, 0.9875)),
            "p_le_0": float(np.mean(d[ok] <= 0))}


def summarise_boot(res: dict, base: str | list[str] | None = None) -> dict:
    """Point + 95% CI per scorer; paired delta vs base (a list = the best of those scorers)."""
    out = {}
    for k, v in res.items():
        if k.startswith("_"):
            continue
        out[k] = {"auc": v["point"], "ci": ci(v["boots"])}
    if base is not None:
        bases = [base] if isinstance(base, str) else list(base)
        tag = "+".join(bases)
        for k in out:
            if k not in bases:
                out[k][f"delta_vs_{tag}"] = boot_delta(res[k]["point"], res[k]["boots"],
                                                       [res[b]["point"] for b in bases],
                                                       [res[b]["boots"] for b in bases])
    return out


def mcnemar_exact(a, b) -> dict:
    """Exact two-sided McNemar test for paired binary outcomes, e.g. two charts' alarms on the same
    bad parts. Only discordant parts carry information: p = 2-sided Binomial(a_only; a_only + b_only, 1/2)."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return {"a_only": n10, "b_only": n01, "both": int((a & b).sum()), "neither": int((~a & ~b).sum()), "p": p}


# ---------------------------------------------------------------- clustered rates
def cluster_rate_ci(k, n, n_boot: int = 4000, seed: int = 0) -> dict:
    """Pooled rate sum(k)/sum(n) over groups, with a percentile CI from resampling whole groups.
    Use when every part of a group shares one limit (e.g. one re-qualification per machine x period),
    so parts are not independent and a Wilson interval on the pooled count is too narrow."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(k), (n_boot, len(k)))
    den = n[idx].sum(axis=1)
    boots = np.where(den > 0, k[idx].sum(axis=1) / np.where(den > 0, den, 1), np.nan)
    r = k / np.where(n > 0, n, np.nan)
    return {"rate": float(k.sum() / n.sum()) if n.sum() else None, "ci": ci(boots), "n_groups": int(len(k)),
            "group_min": float(np.nanmin(r)), "group_median": float(np.nanmedian(r)), "group_max": float(np.nanmax(r))}


def conformal_far_quantiles(n: int, alpha: float, q=(0.05, 0.5, 0.95)) -> dict | None:
    """Spread of the false-alarm rate that ONE calibration set of n exchangeable continuous scores
    actually delivers. With limit s_(k), the rate is 1 - F(s_(k)) ~ Beta(n + 1 - k, k): its mean is the
    marginal guarantee 1 - k/(n+1), but a single re-qualification can land well above it.
    Below the minimum n the max-of-N limit (k = n) is used."""
    from scipy.stats import beta
    k = min(conformal_rank(n, alpha), n)
    return {"mean": 1.0 - k / (n + 1), **{str(p): float(beta.ppf(p, n + 1 - k, k)) for p in q}}


# ---------------------------------------------------------------- operating points
def detection_at_far(scores: dict, y, far: float, n_boot: int = 2000, seed: int = 0,
                     clusters=None) -> tuple[dict, dict]:
    """Each scorer's threshold is set so it flags the same fraction `far` of the good parts;
    detection is measured on the bad parts. Bootstrap: good and bad parts resampled independently,
    thresholds re-estimated in every replicate, identical resamples for every scorer (paired).
    With `clusters`, whole clusters (good and bad parts together) are resampled instead; replicates
    that draw no bad part are dropped. Returns (point estimates, detection replicates)."""
    y = np.asarray(y)
    g, b = np.where(y == 0)[0], np.where(y == 1)[0]
    rng = np.random.default_rng(seed)
    point, boots = {}, {k: np.empty(n_boot) for k in scores}
    for k, s in scores.items():
        thr = np.quantile(s[g], 1 - far)
        point[k] = {"far_target": far, "far": float(np.mean(s[g] > thr)), "det": float(np.mean(s[b] > thr)),
                    "k_det": int(np.sum(s[b] > thr)), "n_bad": int(len(b)), "n_good": int(len(g))}
    if clusters is not None:
        clusters = np.asarray(clusters)
        ucl = np.unique(clusters)
        members = [np.where(clusters == c)[0] for c in ucl]
    for i in range(n_boot):
        if clusters is None:
            gi, bi = rng.choice(g, len(g)), rng.choice(b, len(b))
        else:
            ii = np.concatenate([members[j] for j in rng.integers(0, len(ucl), len(ucl))])
            gi, bi = ii[y[ii] == 0], ii[y[ii] == 1]
            if len(bi) == 0 or len(gi) == 0:
                for k in scores:
                    boots[k][i] = np.nan
                continue
        for k, s in scores.items():
            boots[k][i] = np.mean(s[bi] > np.quantile(s[gi], 1 - far))
    for k in scores:
        point[k]["det_ci"] = ci(boots[k])
    return point, boots
