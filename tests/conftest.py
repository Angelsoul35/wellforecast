from __future__ import annotations

import numpy as np
import pytest

from wellforecast.arps import arps_rate
from wellforecast.real_benchmark import run_benchmark
from wellforecast.real_data import load_production, producing_wells, well_series


@pytest.fixture(scope="session")
def production():
    return load_production()


@pytest.fixture(scope="session")
def wells(production):
    return producing_wells()


@pytest.fixture(scope="session")
def f14(production):
    """The longest-history producer, F-14 H: a textbook hyperbolic decline."""
    return well_series("F-14 H", production)


@pytest.fixture(scope="session")
def report():
    """The full real-data benchmark (fitted once; the slow fixture)."""
    return run_benchmark()


@pytest.fixture
def synthetic_decline():
    """A clean synthetic Arps curve with known parameters, for recovery tests."""
    qi, di, b = 1000.0, 0.01, 0.5
    t = np.arange(0, 900, 1.0)
    q = arps_rate(t, qi, di, b)
    return t, q, (qi, di, b)
