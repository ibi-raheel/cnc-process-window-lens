"""Validation tests. Signals here are synthetic test fixtures, not data."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from features import FS, band_powers, feature_names, features, parse_name, rms, welch_psd  # noqa: E402
from spc import fit_pca, fit_shewhart, spe_limit, t2_ucl  # noqa: E402


def test_sine_rms_is_amplitude_over_root2():
    t = np.arange(int(10 * FS)) / FS
    A = 3.7
    x = A * np.sin(2 * np.pi * 50.0 * t)
    assert rms(x) == pytest.approx(A / np.sqrt(2), rel=1e-3)


def test_parseval_welch_integral_equals_variance():
    rng = np.random.default_rng(1)
    t = np.arange(int(20 * FS)) / FS
    x = 2.0 * np.sin(2 * np.pi * 120 * t) + rng.normal(0, 0.5, t.size)
    f, p = welch_psd(x)
    assert np.trapezoid(p, f) == pytest.approx(np.var(x), rel=0.02)


def test_band_powers_put_tone_in_its_band():
    t = np.arange(int(10 * FS)) / FS
    f, p = welch_psd(np.sin(2 * np.pi * 200 * t))
    bp = band_powers(f, p)
    # 200 Hz lies in band 5 of the edges geomspace(10, 1000, 9)
    assert int(np.argmax(bp)) == 5


def test_t2_ucl_matches_hand_calculation():
    # p = 3, n = 20, alpha = 0.01; F_0.99(3, 17) = 5.185 (table)
    # UCL = 3*19*21 / (20*17) * 5.185 = 3.5206 * 5.185 = 18.25
    assert t2_ucl(3, 20, 0.01) == pytest.approx(18.25, abs=0.02)


def test_spe_limit_single_eigen_reduces_to_scaled_chi2_shape():
    # one residual eigenvalue lam: h0 = 1 - 2/3 = 1/3; the JM limit should be close to
    # lam * chi2_{0.99}(1) = lam * 6.635 (Box approximation, within 10%)
    lam = 2.0
    assert spe_limit(np.array([lam]), 0.01) == pytest.approx(lam * stats.chi2.ppf(0.99, 1), rel=0.10)


def test_t2_empirical_false_alarm_near_alpha_on_gaussian_data():
    rng = np.random.default_rng(7)
    cov = np.diag([5.0, 3.0, 1.0, 0.2, 0.1])
    trials, alarms = 400, 0
    for _ in range(trials):
        tr = rng.multivariate_normal(np.zeros(5), cov, size=40)
        m = fit_pca(tr, k_max=3)
        t2, _ = m.scores(rng.multivariate_normal(np.zeros(5), cov, size=1))
        alarms += int(t2[0] > m.t2_lim)
    assert alarms / trials < 0.035  # nominal 0.01


def test_shewhart_three_sigma():
    s = fit_shewhart(np.array([1.0, 2.0, 3.0]))
    assert s.ratio(np.array([2.0 + 3.0]))[0] == pytest.approx(1.0)


def test_feature_vector_length_and_names():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 3))
    assert features(x).shape == (len(feature_names()),)


def test_parse_name():
    m = parse_name("data/M02/OP03/good/M02_Aug_2019_OP03_000.h5")
    assert (m["machine"], m["op"], m["timeframe"], m["tf_index"]) == ("M02", "OP03", "Aug_2019", 1)


RESULTS = ROOT / "public" / "data" / "results.json"


@pytest.mark.skipif(not RESULTS.exists(), reason="pipeline not run")
def test_results_counts_and_ranges():
    d = json.loads(RESULTS.read_text())
    assert d["dataset"]["n_recordings"] == 1702
    assert d["dataset"]["n_bad"] == 70
    for v in d["scorecard"].values():
        assert 0.0 <= v["roc_auc"] <= 1.0
    assert d["loo_far"]["t2"]["n"] > 500
