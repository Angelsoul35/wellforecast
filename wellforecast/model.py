"""The machine-learning decline forecaster and the error metrics.

:class:`DeclineForecaster` wraps a gradient-boosting regressor that predicts the
correction ratio ``rho = rate / arps`` (see :mod:`wellforecast.features` for why the
ratio and not the raw rate). It fits on a well's early history and forecasts the
rate forward in two ways:

* :meth:`forecast` — **multi-step** from a fixed origin. It walks day by day,
  feeding each predicted ratio back in to build the next day's lag and rolling
  features (a recursive forecast), and multiplies by the Arps baseline. This is the
  real forecasting task: no actual rate from the forecast window is ever used.
* :meth:`predict_onestep` — **one-step-ahead** over a stretch of history, using the
  actual past at every step. This is only a diagnostic: comparing the one-step error
  on train vs held-out isolates over/underfitting from the harder multi-step horizon.

The metrics are the three standard ones. MAE and RMSE are in Sm3/day (RMSE punishes
large misses harder); MAPE is the scale-free percentage error, which lets wells of
very different rates be pooled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .arps import ArpsModel
from .features import RATIO_FEATURES, build_ratio_frame, ratio_row


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true, y_pred) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 4),
    }


# Deliberately light regularisation. On a noisy daily rate a heavier model shaves a
# little off the headline error but opens a wide train/held-out gap (it memorises
# day-to-day noise); these shallow, leaf-constrained trees generalise cleanly — a
# small gap — which matters more than squeezing the last point off the test number.
_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 2,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 20,
    "random_state": 42,
}


@dataclass
class DeclineForecaster:
    """Gradient-boosting correction on top of an Arps baseline."""

    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.params = {**_DEFAULT_PARAMS, **self.params}
        self.model: GradientBoostingRegressor | None = None
        self.arps: ArpsModel | None = None
        self._rho_history: list[float] = []
        self._t_history: list[float] = []

    def fit(self, t, q, arps: ArpsModel) -> DeclineForecaster:
        """Fit the correction model on a training history.

        ``t``/``q`` are days-since-peak and oil rate on the *training* days only;
        ``arps`` is the decline curve already fitted on that same training history.
        """
        t = np.asarray(t, dtype=float)
        q = np.asarray(q, dtype=float)
        self.arps = arps
        baseline = arps.predict(t)
        rho = q / baseline
        frame = build_ratio_frame(t, rho)
        train = frame.dropna()
        if train.empty:
            raise ValueError("not enough history to build correction features")
        self.model = GradientBoostingRegressor(**self.params)
        self.model.fit(train[RATIO_FEATURES], train["rho"])
        self._rho_history = list(rho)
        self._t_history = list(t)
        return self

    def forecast(self, t_future) -> np.ndarray:
        """Recursive multi-step rate forecast for ``t_future`` (days since peak).

        Each step predicts the correction ratio from the ratios known so far, appends
        it, and uses it to build the next step's features. No actual future rate is
        used, so this is a genuine from-origin forecast.
        """
        if self.model is None or self.arps is None:
            raise RuntimeError("forecaster is not fitted")
        t_future = np.asarray(t_future, dtype=float)
        rho_hist = list(self._rho_history)
        start = len(rho_hist)
        preds = []
        for step, t_val in enumerate(t_future):
            i = start + step
            row = ratio_row(rho_hist, t_val, i)
            rho_hat = float(self.model.predict(pd.DataFrame([row])[RATIO_FEATURES])[0])
            rho_hist.append(rho_hat)
            preds.append(rho_hat)
        return self.arps.predict(t_future) * np.asarray(preds, dtype=float)

    def predict_onestep(self, t, q) -> tuple[np.ndarray, np.ndarray]:
        """One-step-ahead rate predictions over ``t``/``q`` using the actual past.

        Returns ``(y_true, y_pred)`` over the rows that have full feature history.
        Diagnostic only — it never feeds predictions back, so it does not measure
        the multi-step forecast, only how well the fitted model reproduces one step.
        """
        if self.model is None or self.arps is None:
            raise RuntimeError("forecaster is not fitted")
        t = np.asarray(t, dtype=float)
        q = np.asarray(q, dtype=float)
        baseline = self.arps.predict(t)
        rho = q / baseline
        frame = build_ratio_frame(t, rho)
        valid = frame.dropna().index.to_numpy()
        rho_hat = self.model.predict(frame.loc[valid, RATIO_FEATURES])
        y_pred = baseline[valid] * rho_hat
        return q[valid], y_pred
