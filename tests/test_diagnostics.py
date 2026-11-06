"""Checks for the calibration, scorecard, root-cause and ramp helpers.
Signals and matrices here are synthetic test fixtures, not data."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from diagnostics import (classify_trajectory, conformal_multiplier, gain_shape,  # noqa: E402
                         recentre_t2, sign_test_p, stratified_auc, wilson)
from features import features  # noqa: E402
from spc import fit_pca  # noqa: E402


def test_wilson_matches_textbook_interval():
    # 5/100: centre (0.05 + 1.96^2/200) / (1 + 1.96^2/100) = 0.0666,
    # half-width 1.96 sqrt(0.05*0.95/100 + 1.96^2/40000) / 1.0384 = 0.0451 -> [0.0215, 0.1118]
    w = wilson(5, 100)
    assert w["lo"] == pytest.approx(0.0215, abs=5e-4)
    assert w["hi"] == pytest.approx(0.1118, abs=5e-4)


def test_wilson_zero_successes_has_positive_upper_bound():
    w = wilson(0, 50)
    assert w["lo"] == 0.0 and 0.05 < w["hi"] < 0.09   # rule of three gives ~0.06


def test_conformal_multiplier_rank():
    # n = 99 ratios 1..99, alpha = 0.01: rank ceil(100 * 0.99) = 99 -> 99
    assert conformal_multiplier(np.arange(1, 100), 0.01) == 99
    # n = 199: rank ceil(200 * 0.99) = 198
    assert conformal_multiplier(np.arange(1, 200), 0.01) == 198


def test_conformal_multiplier_coverage_on_heavy_tails():
    rng = np.random.default_rng(3)
    cal = rng.standard_t(2, 2000) ** 2
    new = rng.standard_t(2, 200000) ** 2
    q = conformal_multiplier(cal, 0.01)
    assert (new > q).mean() == pytest.approx(0.01, abs=0.005)


def test_stratified_auc_single_stratum_equals_roc_auc():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(40), np.ones(10)].astype(int)
    s = rng.normal(size=50) + y
    assert stratified_auc(s, y, np.zeros(50)) == pytest.approx(roc_auc_score(y, s))


def test_stratified_auc_ignores_between_stratum_offset():
    # stratum A: goods 0,1, bad 2; stratum B (all shifted by +10): goods 10,11, bad 12.
    # Within each stratum the bad part ranks top -> 1.0, while the pooled AUC is 2/4 + ... < 1.
    s = np.array([0, 1, 2, 10, 11, 12.0])
    y = np.array([0, 0, 1, 0, 0, 1])
    st = np.array([0, 0, 0, 1, 1, 1])
    assert stratified_auc(s, y, st) == 1.0
    assert roc_auc_score(y, s) < 1.0


def test_sign_test_hand_calculation():
    # 13 of 15 same sign: p = 2 * (C(15,13) + C(15,14) + C(15,15)) / 2^15 = 242 / 32768
    assert sign_test_p(13, 15) == pytest.approx(242 / 32768, rel=1e-9)


def test_shape_features_are_gain_invariant():
    rng = np.random.default_rng(5)
    t = np.arange(8000) / 2000.0
    sig = np.stack([np.sin(2 * np.pi * f * t) + 0.3 * rng.normal(size=t.size) for f in (40, 150, 400)], 1)
    X = np.vstack([features(sig), features(3.0 * sig)])
    fs = gain_shape(X)
    shape, gain = fs["shape"][0], fs["gain"][0]
    assert np.allclose(shape[0], shape[1], atol=1e-6)
    assert np.allclose(gain[1] - gain[0], np.log10(3.0), atol=1e-6)


def test_recentre_removes_a_pure_offset():
    rng = np.random.default_rng(2)
    A = rng.multivariate_normal(np.zeros(4), np.diag([3, 2, 1, 0.5]), 60)
    mod = fit_pca(A, k_max=3)
    offset = np.array([5.0, -4.0, 2.0, 1.0])
    B = A + offset
    copied = mod.scores(B)[0]
    recentred = recentre_t2(mod, B, B, rescale=False)
    assert copied.mean() > 3 * recentred.mean()
    assert np.allclose(recentred, mod.scores(A)[0], atol=1e-8)


def test_trajectory_classes():
    assert classify_trajectory([1.0, 1.1, 1.2]) == "step"
    assert classify_trajectory([0.5, 1.0, 0.4]) == "excursion"
    assert classify_trajectory([0.2, 0.5, 1.0]) == "gradual"


RESULTS = ROOT / "public" / "data" / "results.json"


@pytest.mark.skipif(not RESULTS.exists(), reason="pipeline not run")
def test_calibrated_results_are_consistent():
    d = json.loads(RESULTS.read_text())
    c = d["calibration"]
    # after calibration the pooled leave-one-out T2 false-alarm rate is at or below 1%
    assert c["loo_far_after"]["t2"]["rate"] <= 0.0101
    # and it holds on cells it was not calibrated on (half-split cross-validation)
    assert c["cross_validated_half_split"]["t2_mean"] < 0.02
    m = d["scorecard_v2"]["matched"]
    assert m["n_bad"] == 70
    for v in m["detectors"].values():
        lo, hi = v["roc_auc_ci"]
        assert lo <= v["roc_auc"] <= hi
    r = d["ramp"]
    assert r["own_qualification"]["far"]["rate"] < r["copy"]["far"]["rate"]
