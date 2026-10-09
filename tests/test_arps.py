"""Tests for the Arps decline-curve model.

The decisive test is parameter recovery: given a rate history generated from a known
Arps curve, the fitter must recover ``q_i``, ``D_i`` and ``b`` almost exactly. That
proves the equation and the fit are implemented correctly, independently of any real
data. The rest pin down the exponential/harmonic limits, the cumulative, the ``b``
bound, and a sane fit on a real well.
"""

from __future__ import annotations

import numpy as np
import pytest

from wellforecast.arps import arps_cumulative, arps_rate, fit_arps


def test_recovers_known_hyperbolic_parameters(synthetic_decline):
    t, q, (qi, di, b) = synthetic_decline
    model = fit_arps(t, q)
    assert model.qi == pytest.approx(qi, rel=1e-3)
    assert model.di == pytest.approx(di, rel=1e-3)
    assert model.b == pytest.approx(b, rel=1e-2)
    assert model.regime == "hyperbolic"
    assert model.r2 > 0.999


def test_recovers_exponential_limit():
    # A pure exponential decline must be recovered with b -> 0.
    t = np.arange(0, 800, 1.0)
    q = 2000.0 * np.exp(-0.005 * t)
    model = fit_arps(t, q)
    assert model.b < 0.02
    assert model.regime == "exponential"
    assert model.di == pytest.approx(0.005, rel=1e-2)
    # The exponential closed form the model falls back to must match the curve.
    assert np.allclose(model.predict(t), q, rtol=1e-3)


def test_recovers_harmonic_limit():
    t = np.arange(0, 800, 1.0)
    q = arps_rate(t, 1500.0, 0.01, 1.0)
    model = fit_arps(t, q)
    assert model.b > 0.98
    assert model.regime == "harmonic"


def test_recovery_survives_noise(synthetic_decline):
    t, q, (qi, di, b) = synthetic_decline
    rng = np.random.default_rng(0)
    noisy = q * (1 + rng.normal(0, 0.05, len(t)))
    model = fit_arps(t, noisy)
    # 5% multiplicative noise must not move the parameters more than a few percent.
    assert model.qi == pytest.approx(qi, rel=0.05)
    assert model.di == pytest.approx(di, rel=0.1)
    assert model.b == pytest.approx(b, rel=0.1)


def test_b_is_bounded_to_unit_interval():
    # A rate that *rises* (impossible for decline) must not push b out of [0, 1];
    # the bounded fit stays physical instead of chasing a nonsensical exponent.
    t = np.arange(0, 400, 1.0)
    rising = 500.0 + 2.0 * t
    model = fit_arps(rising[:200], rising[:200])
    assert 0.0 <= model.b <= 1.0


def test_cumulative_is_monotonic_and_matches_exponential_limit():
    t = np.arange(0, 500, 1.0)
    # Exponential: Q(t) = qi/Di * (1 - e^{-Di t}); check against the closed form.
    qi, di = 1000.0, 0.01
    cum = arps_cumulative(t, qi, di, 1e-7)
    expected = qi / di * (1 - np.exp(-di * t))
    assert np.allclose(cum, expected, rtol=1e-4)
    assert np.all(np.diff(cum) >= 0)  # cumulative production never decreases


def test_predict_is_a_decreasing_forecast(synthetic_decline):
    t, q, _ = synthetic_decline
    model = fit_arps(t, q)
    future = np.arange(t[-1], t[-1] + 365, 1.0)
    forecast = model.predict(future)
    assert forecast[0] > forecast[-1]  # a decline curve keeps declining
    assert np.all(forecast > 0)


def test_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_arps([0.0, 1.0], [100.0, 90.0])


def test_absolute_loss_also_recovers_parameters(synthetic_decline):
    t, q, (qi, di, b) = synthetic_decline
    model = fit_arps(t, q, loss="absolute")
    assert model.qi == pytest.approx(qi, rel=1e-2)
    assert model.b == pytest.approx(b, rel=5e-2)
    assert model.loss == "absolute"


def test_unknown_loss_rejected():
    with pytest.raises(ValueError):
        fit_arps([0.0, 1.0, 2.0], [100.0, 90.0, 80.0], loss="huber")


def test_harmonic_cumulative_matches_closed_form():
    t = np.arange(0, 400, 1.0)
    qi, di = 1200.0, 0.02
    cum = arps_cumulative(t, qi, di, 1.0)  # harmonic: Q = qi/Di * ln(1 + Di t)
    expected = qi / di * np.log1p(di * t)
    assert np.allclose(cum, expected, rtol=1e-6)


def test_sane_fit_on_a_real_well(f14):
    model = fit_arps(f14["t"].to_numpy(float), f14["rate"].to_numpy(float))
    assert 0.0 <= model.b <= 1.0
    assert model.di > 0  # a real producer declines
    assert model.qi > f14["rate"].to_numpy(float)[-1]  # starts above the depleted tail
    assert model.n_obs == len(f14)
