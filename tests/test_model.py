"""Tests for the metrics and the ML decline forecaster."""

from __future__ import annotations

import numpy as np
import pytest

from wellforecast.arps import fit_arps
from wellforecast.model import DeclineForecaster, mae, mape, metrics, rmse
from wellforecast.real_benchmark import temporal_split


def test_metrics_are_correct():
    y = np.array([100.0, 200.0, 300.0])
    p = np.array([110.0, 180.0, 330.0])
    assert mae(y, p) == pytest.approx((10 + 20 + 30) / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt((100 + 400 + 900) / 3))
    assert mape(y, p) == pytest.approx((0.1 + 0.1 + 0.1) / 3)


def test_mape_ignores_zero_actuals():
    y = np.array([0.0, 100.0])
    p = np.array([50.0, 90.0])
    assert mape(y, p) == pytest.approx(0.1)  # the zero row is skipped
    assert mape(np.array([0.0]), np.array([1.0])) == 0.0


def test_perfect_prediction_scores_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics(y, y) == {"mae": 0.0, "rmse": 0.0, "mape": 0.0}


def test_forecaster_requires_fitting():
    f = DeclineForecaster()
    with pytest.raises(RuntimeError):
        f.forecast([1.0, 2.0])


def test_forecast_is_a_genuine_decline(f14):
    train, test = temporal_split(f14)
    t_tr, q_tr = train["t"].to_numpy(float), train["rate"].to_numpy(float)
    t_te = test["t"].to_numpy(float)
    arps = fit_arps(t_tr, q_tr)
    f = DeclineForecaster().fit(t_tr, q_tr, arps)
    forecast = f.forecast(t_te)
    assert len(forecast) == len(t_te)
    assert np.all(forecast > 0)
    # The forecast tracks a declining well: its mean is below the training mean.
    assert forecast.mean() < q_tr.mean()


def test_ml_correction_beats_raw_arps_on_f14(f14):
    train, test = temporal_split(f14)
    t_tr, q_tr = train["t"].to_numpy(float), train["rate"].to_numpy(float)
    t_te, q_te = test["t"].to_numpy(float), test["rate"].to_numpy(float)
    arps = fit_arps(t_tr, q_tr)
    f = DeclineForecaster().fit(t_tr, q_tr, arps)
    arps_mape = mape(q_te, arps.predict(t_te))
    ml_mape = mape(q_te, f.forecast(t_te))
    assert ml_mape < arps_mape  # the correction improves on the bare decline curve


def test_onestep_uses_the_actual_past_and_covers_most_rows(f14):
    t = f14["t"].to_numpy(float)
    q = f14["rate"].to_numpy(float)
    arps = fit_arps(t, q)
    f = DeclineForecaster().fit(t, q, arps)
    y_true, y_pred = f.predict_onestep(t, q)
    # Only the first rows (no lag/rolling history) are dropped.
    assert len(y_true) >= len(q) - 31
    assert len(y_true) == len(y_pred)
    # One-step with the real past is close: a tight MAPE, unlike the multi-step tail.
    assert mape(y_true, y_pred) < 0.2
