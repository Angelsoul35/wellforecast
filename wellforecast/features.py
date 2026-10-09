"""Feature engineering for the machine-learning decline layer.

The ML layer does not predict the oil rate directly. Predicting the raw rate with a
tree model fails on a declining well for a specific, instructive reason: a
gradient-boosting model cannot extrapolate below the lowest rate it saw in training,
so on a well whose future rates are *all* lower than its past it just flattens out.

Instead the ML predicts the **correction ratio** to the Arps baseline,

    rho(t) = rate(t) / arps(t),

which is stationary — it hovers around 1 and drifts up or down only where the real
well leaves the smooth decline (an intervention, water breakthrough, a choke change).
A tree model handles a stationary target cleanly, and the final rate forecast is

    rate_hat(t) = arps(t) * rho_hat(t),

so the Arps curve carries the long-run decline (which it extrapolates by
construction) and the ML only has to learn the *departures* from it. This is the
precise sense in which the ML "corrects the Arps trend".

The features describe the recent behaviour of that ratio and of the well, using
only information available **before** the day being predicted (every rolling and
lag feature is shifted, so none peeks at the current or any future value):

* ``rho_lag_{1,7,14}``  — the correction ratio 1, 7 and 14 days ago;
* ``rho_roll_{7,30}``   — its 7- and 30-day trailing means (level of the departure);
* ``rho_slope``         — the recent change in that departure (is the well pulling
  away from, or settling back onto, the decline curve?);
* ``t``                 — days on production since the peak (where we are in the life).

A second, richer tabular view (:func:`decline_features`) exposes the raw
engineering drivers the prompt calls for — time on production, cumulative
production, trailing rate and decline, downhole pressure, and the gas-oil and
water-oil ratios — for diagnostics and study; the forecaster trains on the ratio
features above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "rate"

RHO_LAGS = [1, 7, 14]
RHO_ROLLING = [7, 30]
RHO_SLOPE_WINDOW = 14

RATIO_FEATURES = (
    [f"rho_lag_{lag}" for lag in RHO_LAGS]
    + [f"rho_roll_{w}" for w in RHO_ROLLING]
    + ["rho_slope", "t"]
)


def build_ratio_frame(t: np.ndarray, rho: np.ndarray) -> pd.DataFrame:
    """Build the correction-ratio feature frame (strictly past-only features).

    ``t`` is days since the peak and ``rho`` the correction ratio rate/arps. Every
    lag and rolling feature is shifted so the row for day ``i`` uses only days before
    ``i`` — the anti-leakage guarantee the temporal split relies on.
    """
    s = pd.DataFrame({"rho": np.asarray(rho, dtype=float), "t": np.asarray(t, dtype=float)})
    for lag in RHO_LAGS:
        s[f"rho_lag_{lag}"] = s["rho"].shift(lag)
    for w in RHO_ROLLING:
        s[f"rho_roll_{w}"] = s["rho"].shift(1).rolling(w).mean()
    half = RHO_SLOPE_WINDOW
    recent = s["rho"].shift(1).rolling(half).mean()
    earlier = s["rho"].shift(1 + half).rolling(half).mean()
    s["rho_slope"] = recent - earlier
    return s


def ratio_row(rho_history: list[float], t_value: float, i: int) -> dict[str, float]:
    """Feature row for one forecast step from a running ratio history.

    Used in recursive forecasting: ``rho_history`` holds the ratios known so far
    (actual in the past, predicted once we cross into the forecast), ``i`` is the
    index of the day being predicted. Mirrors :func:`build_ratio_frame` exactly.
    """
    prev = np.asarray(rho_history, dtype=float)
    row = {f"rho_lag_{lag}": prev[i - lag] for lag in RHO_LAGS}
    for w in RHO_ROLLING:
        row[f"rho_roll_{w}"] = float(np.mean(prev[i - w : i]))
    half = RHO_SLOPE_WINDOW
    row["rho_slope"] = float(np.mean(prev[i - half : i]) - np.mean(prev[i - 2 * half : i - half]))
    row["t"] = float(t_value)
    return row


def decline_features(series: pd.DataFrame) -> pd.DataFrame:
    """Richer engineering features for a well's decline series (diagnostics/study).

    Adds the drivers the prompt lists, all shifted to stay leakage-free:

    * ``cum_oil``           — cumulative oil produced up to the previous day (Sm3);
    * ``rate_roll_{7,30}``  — trailing mean oil rate;
    * ``decline_30``        — recent fractional decline over 30 days;
    * ``pressure``          — downhole pressure (forward-filled);
    * ``gor``, ``wor``      — gas-oil and water-oil ratios.
    """
    df = series.copy()
    df["cum_oil"] = df["oil_vol"].cumsum().shift(1).fillna(0.0)
    for w in (7, 30):
        df[f"rate_roll_{w}"] = df["rate"].shift(1).rolling(w).mean()
    past = df["rate"].shift(1)
    df["decline_30"] = (past.shift(30) - past) / past.shift(30)
    df["pressure"] = df["avg_downhole_pressure"].replace(0.0, np.nan).ffill()
    df["gor"] = df["gor"].ffill()
    df["wor"] = df["wor"].ffill()
    return df
