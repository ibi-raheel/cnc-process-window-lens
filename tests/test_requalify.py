"""Checks for the requalify wrapper. The synthetic arrays are test fixtures, not data; the last test
replays a real slice of the pipeline output and skips when pipeline/out is absent."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from requalify import lead_stats, limit_report, machine_calls, rms_requal  # noqa: E402

OUT = ROOT / "pipeline" / "out"
RESULTS = ROOT / "public" / "data" / "results.json"


def test_twenty_parts_buy_the_conformal_rate():
    r = limit_report(np.arange(20.0), 0.05)
    assert r["rule"] == "conformal" and r["limit"] == 19.0          # k = ceil(21 * 0.95) = 20: the largest
    assert r["promised_far"] == pytest.approx(1 / 21) and r["note"] is None


def test_ten_parts_fall_back_to_max_of_n():
    r = limit_report(np.arange(10.0), 0.05)
    assert r["rule"] == "max_of_n" and r["limit"] == 9.0
    assert r["promised_far"] == pytest.approx(1 / 11)
    assert "no 5% limit exists with 10 parts" in r["note"]


def test_machine_call_needs_most_operations_to_move():
    rng = np.random.default_rng(0)
    base = {f"OP{i:02d}": rng.normal(0, 0.3, 24) for i in range(12)}
    every = {o: v + np.eye(24)[9] * 5 for o, v in base.items()}          # band 9 rises on every operation
    one = dict(base); one["OP03"] = base["OP03"] + np.eye(24)[9] * 5      # ... or on one operation only
    assert machine_calls(every)[0]["call"] and machine_calls(every)[0]["band"].startswith("y ")
    assert not any(r["call"] for r in machine_calls(one))


def test_rms_wrapper_matches_hand_arithmetic():
    st = lead_stats(np.array([0.0, 0.2, 0.4, 1.0, 1.2, 1.4]), np.array(["A"] * 3 + ["B"] * 3))
    assert st["A"] == pytest.approx((0.2, 0.2)) and st["B"] == pytest.approx((1.2, 0.2))
    # every new part sits +0.5 above its lead mean: the shift is learnt and the evaluated part scores 0
    test, cal = rms_requal(st, np.array([0.7, 1.7]), np.array(["A", "B"]), np.array([0.7]), np.array(["A"]))
    assert test == pytest.approx([0.0]) and cal == pytest.approx([0.0, 0.0])


@pytest.mark.skipif(not (OUT / "windows_meta.npz").exists(), reason="pipeline outputs not present")
def test_replays_the_published_twenty_part_rms_row():
    """The README's N = 20 RMS row (start-anchored prefix): 24 false alarms on 542 later good parts."""
    from ai_eval import Data
    from stats import limit_with_fallback
    D = Data(OUT)
    x = D.lrms["crop"]
    want = json.loads(RESULTS.read_text())["ai"]["arms"]["crop"]["requal"]["first"]["N=20"]["detectors"]["rms"]["far"]
    fa = ng = 0
    for m in ("M01", "M02", "M03"):
        for t in range(2, 6):
            sel = D.later & (D.machine == m) & (D.tf == t)
            g = np.where(sel & (D.label == 0))[0]
            g = g[np.lexsort((D.op[g], D.idx[g]))]
            if len(g) < 25:
                continue
            new, rest = g[:20], g[20:]
            ref = np.concatenate([D.refs[c] for c in np.unique(D.cell[np.concatenate([new, rest])])])
            st = lead_stats(x[ref], D.cell[ref])
            test, cal = rms_requal(st, x[new], D.cell[new], x[rest], D.cell[rest])
            lim = limit_with_fallback(cal, 0.05)[0]
            fa += int((test > lim).sum()); ng += len(rest)
    assert (fa, ng) == (want["k"], want["n"]) == (24, 542)
