"""Tests for the ratio-correction features.

The load-bearing test is anti-leakage: a feature for day ``i`` must depend only on
days strictly before ``i``. We prove it by perturbing one day and checking that no
earlier feature row moves. The second test proves the recursive ``ratio_row`` builds
exactly the same features as the vectorised ``build_ratio_frame``, so the multi-step
forecast and the training features can never silently diverge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wellforecast.features import (
    RATIO_FEATURES,
    build_ratio_frame,
    decline_features,
    ratio_row,
)


def test_features_never_use_the_present_or_future():
    rng = np.random.default_rng(0)
    t = np.arange(0, 120, 1.0)
    rho = 1.0 + rng.normal(0, 0.1, len(t))
    base = build_ratio_frame(t, rho)[RATIO_FEATURES]

    k = 60
    perturbed_rho = rho.copy()
    perturbed_rho[k] += 5.0  # a big change on day k
    perturbed = build_ratio_frame(t, perturbed_rho)[RATIO_FEATURES]

    # Every feature row up to and including day k must be unchanged: they are built
    # only from days before themselves, so none of them can see day k.
    before = base.iloc[: k + 1].to_numpy()
    before_p = perturbed.iloc[: k + 1].to_numpy()
    assert np.array_equal(np.nan_to_num(before), np.nan_to_num(before_p))
    # And at least one later row *does* change, proving the perturbation was real.
    after = base.iloc[k + 1 :].to_numpy()
    after_p = perturbed.iloc[k + 1 :].to_numpy()
    assert not np.array_equal(np.nan_to_num(after), np.nan_to_num(after_p))


def test_ratio_row_matches_the_vectorised_frame():
    rng = np.random.default_rng(1)
    t = np.arange(0, 80, 1.0)
    rho = 1.0 + rng.normal(0, 0.1, len(t))
    frame = build_ratio_frame(t, rho)

    i = 50  # a fully-populated row
    row = ratio_row(list(rho), t[i], i)
    for feat in RATIO_FEATURES:
        # approx, not ==: pandas rolling means and np.mean can differ in the last bit.
        assert row[feat] == pytest.approx(frame[feat].iloc[i]), feat


def test_decline_features_are_shifted_and_present():
    df = pd.DataFrame(
        {
            "rate": np.linspace(1000, 100, 60),
            "oil_vol": np.linspace(1000, 100, 60),
            "gas_vol": np.linspace(100000, 10000, 60),
            "wat_vol": np.linspace(10, 500, 60),
            "avg_downhole_pressure": np.linspace(300, 200, 60),
            "gor": np.linspace(100, 90, 60),
            "wor": np.linspace(0.01, 5, 60),
        }
    )
    out = decline_features(df)
    for col in ("cum_oil", "rate_roll_7", "rate_roll_30", "decline_30", "pressure"):
        assert col in out.columns
    # cum_oil is shifted: the first day has produced nothing yet.
    assert out["cum_oil"].iloc[0] == 0.0
    # The 7-day trailing mean on the first day is unknown (no history to average).
    assert np.isnan(out["rate_roll_7"].iloc[0])
