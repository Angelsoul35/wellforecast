"""Arps decline-curve analysis — the petroleum-engineering baseline.

An oil well's production rate falls over time as the reservoir depletes. In 1945
J. J. Arps described that fall with a one-line empirical model that is still the
industry standard for reserves and production forecasting:

    q(t) = q_i / (1 + b * D_i * t) ** (1 / b)

where

* ``q(t)``  is the production rate at time ``t`` (here: oil rate in Sm3/day),
* ``q_i``   is the initial rate at the start of the decline (``t = 0``),
* ``D_i``   is the initial nominal decline rate (per day), ``D_i > 0``,
* ``b``     is the decline exponent, bounded to ``0 <= b <= 1``.

The exponent ``b`` selects the decline *shape*:

* ``b = 0``       → **exponential** decline, ``q(t) = q_i * exp(-D_i * t)`` (the
  ``b → 0`` limit of the formula; a constant *fractional* loss per unit time);
* ``0 < b < 1``   → **hyperbolic** decline (the common case; the decline rate
  itself slows down as pressure support and relative permeability change);
* ``b = 1``       → **harmonic** decline, ``q(t) = q_i / (1 + D_i * t)``.

Cumulative production (used for estimated ultimate recovery, EUR) is the integral
of the rate. For hyperbolic ``b`` (with ``b != 0, 1``):

    Q(t) = q_i / (D_i * (1 - b)) * (1 - (1 + b * D_i * t) ** (1 - 1 / b))

with the exponential (``b → 0``) and harmonic (``b = 1``) limits handled
separately.

This module fits the model to a well's rate history with non-linear least squares
(:func:`scipy.optimize.curve_fit`), bounding ``b`` into ``[0, 1]`` so the fit stays
physically meaningful, and forecasts the rate and the cumulative forward. The fit
minimises **relative** error by default (it regresses ``log q``), which is the
right objective for a quantity that spans an order of magnitude over a well's life:
a 50 Sm3/day miss matters far more at the depleted tail than near the peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

# Bounds that keep the fit physical: a positive initial rate, a positive decline,
# and a decline exponent in the classic [0, 1] band. ``b`` is clamped just above 0
# so the hyperbolic form stays numerically well defined (its b -> 0 limit is the
# exponential, which the formula reproduces in the limit).
_B_FLOOR = 1e-6
_PARAM_LO = (1e-6, 1e-9, _B_FLOOR)
_PARAM_HI_B = 1.0


def arps_rate(t, qi: float, di: float, b: float):
    """Arps rate ``q(t) = q_i / (1 + b*D_i*t)^(1/b)`` (exponential as ``b -> 0``)."""
    t = np.asarray(t, dtype=float)
    b = float(np.clip(b, _B_FLOOR, _PARAM_HI_B))
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def arps_cumulative(t, qi: float, di: float, b: float):
    """Cumulative production ``Q(t)`` under the Arps model (the EUR integrand).

    Uses the hyperbolic closed form, with the exponential (``b -> 0``) and harmonic
    (``b = 1``) limits handled explicitly to stay numerically stable.
    """
    t = np.asarray(t, dtype=float)
    b = float(np.clip(b, _B_FLOOR, _PARAM_HI_B))
    if b < 1e-4:  # exponential limit: Q = q_i/D_i * (1 - exp(-D_i t))
        return qi / di * (1.0 - np.exp(-di * t))
    if abs(b - 1.0) < 1e-6:  # harmonic limit: Q = q_i/D_i * ln(1 + D_i t)
        return qi / di * np.log1p(di * t)
    return qi / (di * (1.0 - b)) * (1.0 - np.power(1.0 + b * di * t, 1.0 - 1.0 / b))


@dataclass
class ArpsModel:
    """A fitted Arps decline curve and its forecast methods.

    ``qi`` is the initial rate (Sm3/day), ``di`` the nominal decline (per day), and
    ``b`` the decline exponent. ``r2`` is the coefficient of determination of the fit
    on the data it was trained on (on ``log q`` when ``loss='relative'``).
    """

    qi: float
    di: float
    b: float
    n_obs: int = 0
    r2: float = float("nan")
    loss: str = "relative"

    @property
    def regime(self) -> str:
        if self.b < 0.05:
            return "exponential"
        if self.b > 0.95:
            return "harmonic"
        return "hyperbolic"

    def predict(self, t) -> np.ndarray:
        """Forecast the rate at day(s) ``t`` since the start of the decline."""
        return np.asarray(arps_rate(t, self.qi, self.di, self.b), dtype=float)

    def cumulative(self, t) -> np.ndarray:
        """Cumulative production up to day(s) ``t`` (for EUR-style estimates)."""
        return np.asarray(arps_cumulative(t, self.qi, self.di, self.b), dtype=float)

    def as_dict(self) -> dict:
        return {
            "qi": round(self.qi, 4),
            "di": round(self.di, 8),
            "b": round(self.b, 4),
            "regime": self.regime,
            "n_obs": self.n_obs,
            "r2": round(self.r2, 4),
        }


def fit_arps(t, q, loss: str = "relative") -> ArpsModel:
    """Fit the Arps decline curve to a rate history by non-linear least squares.

    Parameters
    ----------
    t, q:
        Time (days since the start of the decline) and the matching oil rate.
        Only strictly positive rates are used (the model is multiplicative).
    loss:
        ``"relative"`` (default) regresses ``log q`` so every point weighs by its
        *fractional* error — the right objective across a well's wide rate range.
        ``"absolute"`` regresses the raw rate (ordinary least squares).

    Returns an :class:`ArpsModel` with ``b`` bounded to ``[0, 1]``.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = (q > 0) & np.isfinite(q) & np.isfinite(t)
    t, q = t[mask], q[mask]
    if len(t) < 3:
        raise ValueError("need at least 3 positive-rate points to fit a decline curve")

    qi0 = float(np.mean(q[: min(30, len(q))]))
    p0 = [qi0, 1e-3, 0.5]
    hi = [qi0 * 3.0 + 1.0, _PARAM_HI_B, _PARAM_HI_B]

    if loss == "relative":

        def model(tt, qi, di, b):
            b = np.clip(b, _B_FLOOR, _PARAM_HI_B)
            return np.log(qi) - (1.0 / b) * np.log1p(b * di * tt)

        target = np.log(q)
    elif loss == "absolute":

        def model(tt, qi, di, b):
            return arps_rate(tt, qi, di, b)

        target = q
    else:
        raise ValueError(f"unknown loss {loss!r}; use 'relative' or 'absolute'")

    popt, _ = curve_fit(
        model, t, target, p0=p0, bounds=(_PARAM_LO, hi), maxfev=40000
    )
    qi, di, b = (float(popt[0]), float(popt[1]), float(popt[2]))

    pred = model(t, qi, di, b)
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return ArpsModel(qi=qi, di=di, b=b, n_obs=int(len(t)), r2=r2, loss=loss)
