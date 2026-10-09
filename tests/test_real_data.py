"""Tests for loading the real Volve data and the fetch/vendor path.

They prove the vendored data is the real producing fleet with genuinely declining
series, that the rate is built and the decline segment cut correctly, and that the
download-and-tidy path keeps only producers and reshapes the columns — all without
touching the network (the download is monkeypatched).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import wellforecast.real_data as rd
from wellforecast.real_data import (
    load_production,
    well_codes,
    well_series,
)


def test_vendored_data_is_the_real_producer_fleet(production):
    # The Volve producers (the well codes Equinor published), nothing injected.
    assert {"F-12 H", "F-14 H", "F-11 H", "F-15 D", "F-1 C"}.issubset(set(well_codes()))
    assert production["date"].min().year == 2008
    assert production["date"].max().year == 2016
    # Realistic North Sea daily oil volumes (Sm3), never negative.
    assert production["oil_vol"].max() > 2000
    assert production["oil_vol"].min() >= 0


def test_producing_wells_have_long_declining_histories(wells, production):
    assert len(wells) >= 4
    for code in wells:
        s = well_series(code, production)
        assert len(s) >= 300
        # The rate is a real producing rate, strictly positive after cleaning.
        assert (s["rate"] > 0).all()
        # From the peak onward the well declines: the last-quarter mean sits well
        # below the first-quarter mean.
        q = s["rate"].to_numpy(float)
        first_quarter = q[: len(q) // 4].mean()
        last_quarter = q[-len(q) // 4 :].mean()
        assert last_quarter < 0.8 * first_quarter


def test_rate_is_uptime_normalised(production):
    s = well_series("F-14 H", production)
    # rate == oil_vol / on_stream_hrs * 24, within rounding of the vendored volumes.
    recomputed = s["oil_vol"] / s["on_stream_hrs"] * 24.0
    assert np.allclose(s["rate"].to_numpy(float), recomputed.to_numpy(float), rtol=1e-6)
    # Only days with real uptime survive the filter.
    assert (s["on_stream_hrs"] >= rd.MIN_ON_STREAM_HRS).all()


def test_time_axis_starts_at_peak(f14):
    assert f14["t"].iloc[0] == 0.0
    assert f14["t"].is_monotonic_increasing
    # t is days since the (smoothed) peak; the series should open near its maximum.
    assert f14["rate"].iloc[:60].max() >= 0.9 * f14["rate"].max()


def test_unknown_well_raises(production):
    with pytest.raises(KeyError):
        well_series("NO SUCH WELL", production)


def test_missing_dataset_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(rd, "_DATASET_PATH", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        load_production()


def _fake_workbook(path):
    """A tiny raw-shaped Volve workbook: one producer, one injector, one partial day."""
    rows = [
        # producer F-99 X: two clean days + one partial (low-uptime) day
        ("2010-01-01", "NO 15/9-F-99 X", "OP", 24.0, 300.0, 500.0, 50000.0, 10.0),
        ("2010-01-02", "NO 15/9-F-99 X", "OP", 24.0, 298.0, 480.0, 48000.0, 12.0),
        ("2010-01-03", "NO 15/9-F-99 X", "OP", 2.0, 0.0, 30.0, 3000.0, 1.0),
        # water injector WI: must be dropped entirely
        ("2010-01-01", "NO 15/9-F-98 W", "WI", 24.0, 0.0, 0.0, 0.0, 9000.0),
    ]
    cols = [
        "DATEPRD", "WELL_BORE_CODE", "WELL_TYPE", "ON_STREAM_HRS",
        "AVG_DOWNHOLE_PRESSURE", "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL",
    ]
    pd.DataFrame(rows, columns=cols).to_excel(path, index=False, engine="openpyxl")


def test_fetch_keeps_only_producers_and_tidies(monkeypatch, tmp_path):
    workbook = tmp_path / "raw.xlsx"
    _fake_workbook(workbook)

    def fake_download(url, dest):
        dest.write_bytes(workbook.read_bytes())

    monkeypatch.setattr(rd, "_download", fake_download)
    dest = tmp_path / "volve.csv"
    n = rd.fetch_volve(url="https://example/ignored.xlsx", dest=dest)

    assert n == 3  # three producer rows kept; the injector row dropped
    out = pd.read_csv(dest)
    assert list(out.columns) == [
        "date", "well", "on_stream_hrs", "avg_downhole_pressure",
        "oil_vol", "gas_vol", "wat_vol",
    ]
    assert set(out["well"]) == {"F-99 X"}  # injector gone, code simplified
    # The raw xlsx must have been cleaned up after reading.
    assert not (dest.parent / "_volve_raw.xlsx").exists()


def test_fetch_then_load_builds_a_usable_series(monkeypatch, tmp_path):
    workbook = tmp_path / "raw.xlsx"
    _fake_workbook(workbook)
    monkeypatch.setattr(rd, "_download", lambda url, dest: dest.write_bytes(workbook.read_bytes()))
    dest = tmp_path / "volve.csv"
    rd.fetch_volve(dest=dest)
    monkeypatch.setattr(rd, "_DATASET_PATH", dest)

    df = load_production()
    assert set(df["well"]) == {"F-99 X"}
    s = well_series("F-99 X", df)
    # The 2-hour partial day is below MIN_ON_STREAM_HRS and must be filtered out.
    assert (s["on_stream_hrs"] >= rd.MIN_ON_STREAM_HRS).all()
    assert len(s) == 2
