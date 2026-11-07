"""Checks for the MiniRocket heads, the shared statistics, the duration crop, re-qualification and
the drift subspace. Arrays here are synthetic test fixtures, not data; the tests at the bottom read
the real pipeline outputs and skip when they are absent."""
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from detectors import (hier_shift, hier_shift_loo, knn_score, nearest_windows, ridge_dual,  # noqa: E402
                       ridge_head, shewhart_oos, shift_for)
from drift import drift_share, label_alarm, shift_vectors, subspace, top_bands  # noqa: E402
from embed import fit_rows  # noqa: E402
from stats import (boot_delta, cluster_bootstrap_auc, conformal_rank, conformal_threshold,  # noqa: E402
                   detection_at_far, eligible, exact_far, limit_with_fallback, max_of_n_far,
                   min_n_for_alpha, stratified_auc)
from windows import (ATTR_EDGES, L, band_log_power, crop_lengths, cut, full_starts,  # noqa: E402
                     label_of, prefix_starts, relpath, vec_rms)

OUT = ROOT / "pipeline" / "out"
RESULTS = ROOT / "public" / "data" / "results.json"


# ---------------------------------------------------------------- split-conformal limits
def test_conformal_limit_hand_calculation():
    # n = 20 scores 1..20, alpha = 0.05: k = ceil(21 * 0.95) = ceil(19.95) = 20 -> limit = 20,
    # exact false-alarm rate 1 - 20/21 = 1/21 = 0.0476
    s = np.arange(1, 21, dtype=float)
    assert conformal_threshold(s, 0.05) == 20.0
    assert exact_far(20, 0.05) == pytest.approx(1 / 21)
    # n = 19: k = ceil(20 * 0.95) = 19 -> the maximum, FAR exactly 1/20 = alpha
    assert conformal_threshold(np.arange(1, 20, dtype=float), 0.05) == 19.0
    assert exact_far(19, 0.05) == pytest.approx(0.05)
    # n = 50: k = ceil(51 * 0.95) = ceil(48.45) = 49 -> second largest, FAR 2/51
    assert conformal_threshold(np.arange(1, 51, dtype=float), 0.05) == 49.0
    assert exact_far(50, 0.05) == pytest.approx(2 / 51)
    # order of the input does not matter
    rng = np.random.default_rng(0)
    assert conformal_threshold(rng.permutation(s), 0.05) == 20.0


def test_conformal_limit_is_infinite_below_minimum_n():
    assert min_n_for_alpha(0.05) == 19
    assert min_n_for_alpha(0.01) == 99
    for n in range(1, 19):
        assert math.isinf(conformal_threshold(np.arange(n, dtype=float), 0.05))
    assert math.isfinite(conformal_threshold(np.arange(19, dtype=float), 0.05))
    assert math.isinf(conformal_threshold(np.arange(98, dtype=float), 0.01))
    assert math.isfinite(conformal_threshold(np.arange(99, dtype=float), 0.01))
    assert math.isnan(exact_far(18, 0.05))


def test_conformal_rank_is_exact_integer_arithmetic():
    for alpha in (0.05, 0.01, 0.1, 0.2):
        a = Fraction(alpha).limit_denominator(1000)
        for n in range(1, 400):
            assert conformal_rank(n, alpha) == math.ceil((n + 1) * (1 - a))


@pytest.mark.parametrize("dist", ["gauss", "t3"])
@pytest.mark.parametrize("n", [19, 20, 50, 120])
def test_conformal_false_alarm_guarantee_by_simulation(dist, n):
    """P(new > limit) = 1 - k/(n+1) <= alpha for exchangeable continuous scores."""
    rng = np.random.default_rng(n)
    trials, alpha = 40_000, 0.05
    X = rng.standard_normal((trials, n + 1)) if dist == "gauss" else rng.standard_t(3, (trials, n + 1))
    k = conformal_rank(n, alpha)
    lim = np.sort(X[:, :n], axis=1)[:, k - 1]
    far = float(np.mean(X[:, n] > lim))
    se = math.sqrt(alpha * (1 - alpha) / trials)
    assert far <= alpha + 3 * se
    assert far == pytest.approx(exact_far(n, alpha), abs=4 * se)


@pytest.mark.parametrize("n", [5, 10])
def test_max_of_n_limit_false_alarm_is_one_over_n_plus_one(n):
    rng = np.random.default_rng(10 + n)
    trials = 40_000
    X = rng.standard_t(3, (trials, n + 1))
    far = float(np.mean(X[:, n] > X[:, :n].max(axis=1)))
    se = math.sqrt(far * (1 - far) / trials)
    assert far == pytest.approx(max_of_n_far(n), abs=4 * se)
    lim, info = limit_with_fallback(X[0, :n], 0.05)
    assert info["rule"] == "max_of_n" and lim == X[0, :n].max() and info["guarantee"] == pytest.approx(1 / (n + 1))


def test_limit_with_fallback_switches_to_conformal_at_19():
    lim, info = limit_with_fallback(np.arange(1, 20, dtype=float), 0.05)
    assert info["rule"] == "conformal" and lim == 19.0 and info["guarantee"] == pytest.approx(0.05)
    lim, info = limit_with_fallback(np.arange(1, 19, dtype=float), 0.05)
    assert info["rule"] == "max_of_n" and lim == 18.0 and info["guarantee"] == pytest.approx(1 / 19)


# ---------------------------------------------------------------- like-for-like AUC and bootstraps
def test_stratified_auc_hand_count_two_strata():
    # stratum 0: bad 1 vs goods {0, 2} -> 1 win, 1 loss; stratum 1: bad 5 vs goods {0, 1} -> 2 wins.
    # 3 wins of 4 within-stratum pairs = 0.75
    s = np.array([0, 2, 1, 0, 1, 5.0])
    y = np.array([0, 0, 1, 0, 0, 1])
    st = np.array([0, 0, 0, 1, 1, 1])
    assert stratified_auc(s, y, st) == 0.75


def test_stratified_auc_ignores_offsets_between_strata():
    rng = np.random.default_rng(1)
    st = np.repeat(np.arange(6), 20)
    y = (rng.random(120) < 0.25).astype(int)
    s = rng.normal(size=120) + y
    base = stratified_auc(s, y, st)
    shifted = s + 100.0 * st                     # a different constant added to every stratum
    assert stratified_auc(shifted, y, st) == pytest.approx(base, abs=1e-12)
    assert roc_auc_score(y, shifted) != pytest.approx(base, abs=1e-3)   # the pooled AUC does move


def test_stratified_auc_ties_count_half():
    assert stratified_auc(np.array([1.0, 1.0]), np.array([0, 1]), np.array([0, 0])) == 0.5


def test_eligible_keeps_only_strata_with_both_classes():
    st = np.array([0, 0, 1, 1, 2])
    y = np.array([0, 1, 0, 0, 1])
    assert eligible(st, y).tolist() == [True, True, False, False, False]


def test_cluster_bootstrap_resamples_strata_not_parts():
    # stratum 0 has AUC 1/2 over 2 pairs, stratum 1 has AUC 1 over 2 pairs. Resampling whole strata
    # can only give 0.5 (0,0), 0.75 (0,1) or 1.0 (1,1); a part bootstrap would give other values.
    s = np.array([0, 2, 1, 0, 1, 5.0])
    y = np.array([0, 0, 1, 0, 0, 1])
    st = np.array([0, 0, 0, 1, 1, 1])
    res = cluster_bootstrap_auc({"a": s}, y, st, n_boot=2000, seed=0)
    vals = set(np.round(res["a"]["boots"], 9).tolist())
    assert vals == {0.5, 0.75, 1.0}
    assert res["a"]["point"] == 0.75 and res["_design"]["resample"] == "whole strata"


def test_paired_bootstrap_of_identical_scores_is_exactly_zero():
    rng = np.random.default_rng(2)
    st = np.repeat(np.arange(8), 10)
    y = (rng.random(80) < 0.3).astype(int)
    s = rng.normal(size=80) + y
    res = cluster_bootstrap_auc({"a": s, "b": s.copy()}, y, st, n_boot=500, seed=1)
    d = boot_delta(res["a"]["point"], res["a"]["boots"], [res["b"]["point"]], [res["b"]["boots"]])
    assert d["est"] == 0.0 and d["ci"] == [0.0, 0.0]
    pt, bt = detection_at_far({"a": s, "b": s.copy()}, y, 0.1, n_boot=200, seed=0)
    assert np.array_equal(bt["a"], bt["b"]) and pt["a"]["det"] == pt["b"]["det"]


def test_delta_against_several_baselines_uses_their_replicatewise_max():
    a = np.array([0.9, 0.8, 0.7])
    b1, b2 = np.array([0.5, 0.85, 0.1]), np.array([0.6, 0.2, 0.65])
    d = boot_delta(0.8, a, [0.55, 0.6], [b1, b2])
    assert d["est"] == pytest.approx(0.8 - 0.6)
    assert np.allclose(sorted([0.9 - 0.6, 0.8 - 0.85, 0.7 - 0.65]), sorted([0.3, -0.05, 0.05]))
    assert d["ci"][0] == pytest.approx(np.quantile([0.3, -0.05, 0.05], 0.025))


# ---------------------------------------------------------------- head A: dual ridge
def test_dual_ridge_equals_primal_weighted_ridge():
    rng = np.random.default_rng(3)
    n, p = 40, 15
    Z = rng.normal(size=(n, p))
    y = (rng.random(n) < 0.3).astype(int)
    tr, te = np.arange(30), np.arange(30, 40)
    K = Z @ Z.T / p + 1.0
    alpha = 0.1
    dual = ridge_dual(K, tr, te, y, alpha)
    X = np.hstack([Z / np.sqrt(p), np.ones((n, 1))])
    yt = np.where(y[tr] == 1, 1.0, -1.0)
    n1, n0 = (yt > 0).sum(), (yt < 0).sum()
    w = np.where(yt > 0, len(yt) / (2 * n1), len(yt) / (2 * n0))
    beta = np.linalg.solve(X[tr].T @ (w[:, None] * X[tr]) + alpha * np.eye(p + 1), X[tr].T @ (w * yt))
    assert np.allclose(dual, X[te] @ beta, atol=1e-8)


def test_ridge_head_test_rows_never_touch_the_fit():
    """Scores of test set A do not change when unrelated test rows B are scored with it:
    standardisation, penalty choice and fit use the training rows only."""
    rng = np.random.default_rng(4)
    n, p = 90, 30
    E = rng.normal(size=(n, p)).astype(np.float32)
    y = (rng.random(n) < 0.25).astype(int)
    E[y == 1] += 0.8
    groups = np.repeat(np.arange(6), 15)
    train = np.where(groups < 4)[0]
    A, B = np.where(groups == 4)[0], np.where(groups == 5)[0]
    sA, aA = ridge_head(E, y, groups, groups, train, A)
    E2 = E.copy()
    E2[B] *= 50.0                                         # wildly different rows B
    sAB, aAB = ridge_head(E2, y, groups, groups, train, np.concatenate([A, B]))
    assert aA == aAB
    assert np.allclose(sA, sAB[:len(A)], atol=1e-6)


# ---------------------------------------------------------------- head B: kNN memory bank
def test_nearest_windows_matches_brute_force():
    rng = np.random.default_rng(5)
    Q, Bk = rng.normal(size=(3, 4, 5)), rng.normal(size=(4, 4, 5))
    D, V = nearest_windows(Q, Bk)
    for q in range(3):
        for w in range(4):
            for b in range(4):
                d = [np.linalg.norm(Q[q, w] - Bk[b, v]) for v in range(4)]
                assert D[q, w, b] == pytest.approx(min(d), abs=1e-9)
                assert V[q, w, b] == int(np.argmin(d))


def test_knn_leave_one_out_never_matches_itself():
    rng = np.random.default_rng(6)
    E = rng.normal(size=(6, 4, 3))
    bank = np.arange(6)
    raw, dist, part, _ = knn_score(E, bank, bank)
    assert (raw > 0).all()                                # no part vouches for itself
    assert not (part == bank[:, None]).any()
    E[5] = E[2]                                           # an exact duplicate in the bank ...
    raw, _, part, _ = knn_score(E, bank, bank)
    assert raw[2] == pytest.approx(0.0) and raw[5] == pytest.approx(0.0)   # ... is matched (not itself)
    assert (part[2] == 5).all() and (part[5] == 2).all()


def test_knn_appending_good_parts_lowers_scores_of_similar_parts():
    rng = np.random.default_rng(7)
    old = rng.normal(size=(10, 4, 3))
    new = rng.normal(loc=3.0, size=(6, 4, 3))            # drifted process
    E = np.concatenate([old, new])
    bank, q = np.arange(10), np.arange(13, 16)
    before = knn_score(E, q, bank)[0]
    after = knn_score(E, q, np.concatenate([bank, [10, 11, 12]]))[0]
    assert (after < before).all()


# ---------------------------------------------------------------- re-qualification
def test_hierarchical_recentring_removes_a_pure_offset_exactly():
    rng = np.random.default_rng(8)
    cells = np.array(["a", "a", "b", "c", "c", "c"])
    shift = np.array([0.7, -0.2, 1.5])
    off = np.tile(shift, (6, 1))                          # every new part sits exactly +shift away
    delta, per = hier_shift(off, cells)
    assert np.allclose(delta, shift) and all(np.allclose(v, shift) for v in per.values())
    x = rng.normal(size=(4, 3)) + shift
    assert np.allclose(x - shift_for(np.array(["a", "b", "c", "d"]), delta, per), x - shift)


def test_hierarchical_recentring_shrinks_per_op_and_falls_back_to_machine_offset():
    off = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    cells = np.array(["a", "a", "a", "a", "b"])
    delta, per = hier_shift(off, cells, kappa=5.0)
    assert delta == pytest.approx(1.8)
    assert per["b"] == pytest.approx(1.8 + (1 / 6) * (5.0 - 1.8))          # w = 1 / (1 + 5)
    assert per["a"] == pytest.approx(1.8 + (4 / 9) * (1.0 - 1.8))          # w = 4 / (4 + 5)
    assert shift_for(np.array(["z"]), delta, per)[0] == pytest.approx(1.8)  # no parts on op z
    loo = hier_shift_loo(off, cells, kappa=5.0)
    assert loo[4] == pytest.approx(1.0)                   # without part 4, op b has no parts left


def test_shewhart_reference_parts_are_scored_leave_one_out():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    ref = np.array([0, 1, 2])
    s = shewhart_oos(x, ref, np.array([0, 3]))
    assert s[0] == pytest.approx(abs(1.0 - 2.5) / (3 * np.std([2.0, 3.0], ddof=1)))
    assert s[1] == pytest.approx(abs(10.0 - 2.0) / 3.0)


# ---------------------------------------------------------------- drift subspace and attribution
def test_drift_share_inside_and_orthogonal_to_the_subspace():
    rng = np.random.default_rng(9)
    U, _ = subspace(rng.normal(size=(12, 20)), k=3)
    assert np.allclose(U.T @ U, np.eye(3), atol=1e-10)
    z_in = U @ np.array([1.0, -2.0, 0.5])
    assert drift_share(z_in, U)[0] == pytest.approx(1.0)
    z_out = rng.normal(size=20)
    z_out -= U @ (U.T @ z_out)
    assert drift_share(z_out, U)[0] == pytest.approx(0.0, abs=1e-12)
    assert label_alarm(0.9) == "nuisance drift: re-baseline" and label_alarm(0.1) == "out of window: investigate"


def test_random_direction_share_is_k_over_p():
    rng = np.random.default_rng(10)
    U, _ = subspace(rng.normal(size=(30, 60)), k=3)
    share = drift_share(rng.normal(size=(20_000, 60)), U)
    assert share.mean() == pytest.approx(3 / 60, abs=0.002)


def test_drift_subspace_for_period_t_uses_earlier_periods_only():
    rng = np.random.default_rng(11)
    n = 120
    tf = np.repeat(np.arange(6), 20)
    cell = np.tile(np.array(["M01-OP00", "M02-OP00"]), 60)
    label = (rng.random(n) < 0.1).astype(int)
    Z = rng.normal(size=(n, 8))
    refs = {c: np.where((cell == c) & (tf <= 1) & (label == 0))[0] for c in ("M01-OP00", "M02-OP00")}
    st = {"mu": {c: Z[r].mean(0) for c, r in refs.items()}, "sd": {c: Z[r].std(0) + 1 for c, r in refs.items()}}
    for t in range(2, 6):
        V, used = shift_vectors(Z, cell, tf, label, np.ones(n, bool), refs, st, t, {"OP00": list(refs)})
        assert len(V) > 0 and tf[used].max() < t and (label[used] == 0).all()


def test_top_bands_rank_by_magnitude_and_keep_sign():
    v = np.array([0.5, -4.0, 3.0, -0.1])
    top = top_bands(v, ["a", "b", "c", "d"], n=2)
    assert [t["band"] for t in top] == ["b", "c"] and top[0]["z"] == -4.0


# ---------------------------------------------------------------- duration crop (windows.py)
def test_prefix_windows_are_identical_for_recordings_of_any_length():
    T = 10_000
    s = prefix_starts(T, k=24)
    assert s[0] == 0 and s[-1] + L <= T and len(s) == 24
    assert np.array_equal(s, prefix_starts(T, k=24))      # depends on the cell's T only, never on n
    assert not np.array_equal(full_starts(40_000), full_starts(30_000))   # the full arm does depend on n
    with pytest.raises(ValueError):
        prefix_starts(L - 1)


def test_crop_length_is_fraction_of_qualification_median_capped_at_shortest():
    tf = np.array([0] * 16 + [1] * 4)
    label = np.array([0] * 16 + [1] * 4)
    n_good = np.arange(1000, 1016) * 10                   # median 10075
    cell = np.array(["c"] * 20)
    n = np.concatenate([n_good, [9000, 8000, 7000, 6000]])
    out = crop_lengths(n, cell, tf, label, frac=0.45)
    assert out["c"]["T"] == int(np.floor(0.45 * np.median(n_good))) and out["c"]["T"] <= n.min()
    n2 = np.concatenate([n_good, [9000, 8000, 7000, 4000]])  # a recording shorter than 0.45 x median
    out2 = crop_lengths(n2, cell, tf, label, frac=0.45)
    assert out2["c"]["T"] == 4000 and out2["c"]["T_uncapped"] > 4000


def test_truncation_does_not_change_prefix_windows_or_their_knn_score():
    """The crop arm makes recording length uninformative: cutting a recording to 85% of its length
    leaves its prefix windows, and therefore its kNN score, unchanged; the full arm does change."""
    rng = np.random.default_rng(12)
    n = 30_000
    t = np.arange(n) / 2000.0
    d = np.stack([np.sin(2 * np.pi * f * t) + 0.2 * rng.normal(size=n) for f in (60, 210, 480)], axis=1)
    short = d[: int(0.85 * n)]
    T = int(0.45 * n)
    sp = prefix_starts(T)
    w_full_len, w_short = cut(d, sp), cut(short, sp)
    assert np.array_equal(w_full_len, w_short)
    assert not np.array_equal(cut(d, full_starts(len(d))), cut(short, full_starts(len(short))))
    bank_parts = [np.stack([np.sin(2 * np.pi * f * t[:T]) + 0.2 * rng.normal(size=T) for f in (60, 210, 480)], 1)
                  for _ in range(5)]
    emb = lambda x: band_log_power(cut(x, sp)).reshape(len(sp), -1)   # noqa: E731
    E = np.stack([emb(b) for b in bank_parts] + [emb(d), emb(short)])
    s = knn_score(E, np.array([5, 6]), np.arange(5))[0]
    assert s[0] == s[1]


def test_cut_removes_the_mean_per_window_and_axis():
    rng = np.random.default_rng(13)
    d = rng.normal(loc=[1.0, -2.0, 5.0], size=(5000, 3))
    w = cut(d, np.array([0, 1000, 3000]))
    assert w.shape == (3, 3, L) and np.allclose(w.mean(axis=2), 0.0, atol=1e-5)


def test_attribution_bands_each_hold_four_fft_bins_and_locate_a_tone():
    f = np.fft.rfftfreq(L, 1 / 2000.0)                    # the Welch bins for nperseg = L
    n_last = len(ATTR_EDGES) - 2
    counts = [int(((f >= lo) & ((f < hi) if j < n_last else (f <= hi))).sum())
              for j, (lo, hi) in enumerate(zip(ATTR_EDGES[:-1], ATTR_EDGES[1:]))]
    assert min(counts) >= 4
    t = np.arange(L) / 2000.0
    w = np.sin(2 * np.pi * 210.0 * t)[None, :]
    j = int(np.searchsorted(ATTR_EDGES, 210.0) - 1)
    assert int(np.argmax(band_log_power(w)[0])) == j


def test_vector_rms_matches_the_repo_definition():
    rng = np.random.default_rng(14)
    d = rng.normal(loc=[0.3, 0.0, -1.0], size=(4000, 3))
    dt = d - d.mean(axis=0)
    assert vec_rms(d) == pytest.approx(np.sqrt(np.mean(np.linalg.norm(dt, axis=1) ** 2)))


def test_good_and_bad_files_with_the_same_basename_stay_distinct(tmp_path):
    a = tmp_path / "data" / "M01" / "OP00" / "good" / "M01_Feb_2019_OP00_000.h5"
    b = tmp_path / "data" / "M01" / "OP00" / "bad" / "M01_Feb_2019_OP00_000.h5"
    ra, rb = relpath(a, tmp_path / "data"), relpath(b, tmp_path / "data")
    assert ra != rb and label_of(ra) == 0 and label_of(rb) == 1


def test_embedding_fit_rows_exclude_the_target_machine_bad_parts_and_later_periods():
    machine = np.array(["M01", "M02", "M03"] * 6)
    tf = np.repeat(np.arange(6), 3)
    label = np.array([0, 1] * 9)
    for tgt in ("M01", "M02", "M03"):
        r = fit_rows(machine, tf, label, tgt)
        assert len(r) and not (machine[r] == tgt).any() and (label[r] == 0).all() and (tf[r] <= 1).all()


# ---------------------------------------------------------------- real pipeline outputs
@pytest.mark.skipif(not (OUT / "windows_meta.npz").exists() or not (OUT / "features.npz").exists(),
                    reason="pipeline caches not present")
def test_real_windows_meta_matches_features_and_crop_is_consistent():
    M = np.load(OUT / "windows_meta.npz")
    F = np.load(OUT / "features.npz")
    assert len(set(M["relpath"].tolist())) == len(M["relpath"])
    key = ["/".join(p.split("/")[-4:]) for p in F["path"]]
    pos = {k: i for i, k in enumerate(key)}
    j = np.array([pos[r] for r in M["relpath"]])
    assert np.allclose(M["rms_full"], F["rms"][j], rtol=1e-6, atol=0)
    assert (M["label"] == F["label"][j]).all()
    cell = np.char.add(np.char.add(M["machine"], "-"), M["op"])
    for c in np.unique(cell):
        sc = M["starts_crop"][cell == c]
        assert (sc == sc[0]).all()
    assert (M["starts_crop"][:, -1] + int(M["L"]) <= M["crop_len"]).all()
    assert (M["crop_len"] <= M["n_samples"]).all()


@pytest.mark.skipif(not list(OUT.glob("emb_*_s*_M0*.npz")) or not (OUT / "windows_meta.npz").exists(),
                    reason="embedding caches not present")
def test_real_embedding_folds_never_fitted_on_their_target_machine():
    M = np.load(OUT / "windows_meta.npz")
    for p in sorted(OUT.glob("emb_*_s*_M0*.npz")):
        tgt = p.stem.split("_")[-1]
        r = np.load(p)["fit_rec"]
        assert not (M["machine"][r] == tgt).any(), p.name
        assert (M["label"][r] == 0).all() and (M["tf"][r] <= 1).all(), p.name


def _ai():
    d = json.loads(RESULTS.read_text())
    return d, d.get("ai")


@pytest.mark.skipif(not RESULTS.exists() or "\"ai\"" not in RESULTS.read_text()[:2_000_000], reason="ai_eval not run")
def test_real_ai_results_reproduce_the_repo_and_respect_the_gate():
    d, ai = _ai()
    m = d["scorecard_v2"]["matched"]["detectors"]
    # the reimplemented charts reproduce the repo's like-for-like (stratified) AUCs
    assert ai["regression"]["rms_like_for_like"] == pytest.approx(m["rms"]["strat_auc"], abs=1e-4)
    assert ai["regression"]["t2spe_like_for_like"] == pytest.approx(m["comb"]["strat_auc"], abs=1e-4)
    assert ai["data"]["n_later_good"] == d["scorecard_v2"]["drift"]["n_good"]
    dec = ai["decision"]
    assert (dec["headline_mode"] == "ai") == bool(dec["passed_heads"])
    for h in dec["passed_heads"]:
        assert all(s["pass"] for s in dec["by_seed"]["crop"][h])
    # split-conformal limits hold their level inside the qualification period (random half splits)
    for arm in ("crop", "full"):
        for k, v in ai["arms"][arm]["seeds"]["0"]["drift_alpha_0.05"].items():
            assert v["in_period_far_split_half"] <= 0.055, (arm, k)
    # re-qualification rows report the guarantee they rely on
    rows = ai["arms"]["crop"]["requal"]["first"]
    assert rows["N=5"]["rule"] == "max-of-N" and rows["N=5"]["guarantee"] == pytest.approx(1 / 6, abs=1e-4)
    assert rows["N=20"]["rule"] == "conformal" and rows["N=20"]["guarantee"] <= 0.05
