"""Real Volve field production data — the series the models are measured on.

Everything in WellForecast runs on **real** daily production from the Volve field,
a North Sea oil field Equinor operated until 2016 and then released as open data.
No synthetic production is used for the headline numbers; a small synthetic Arps
curve appears only in the tests, to prove the decline-curve fitter recovers known
parameters.

Source
------
Equinor's Volve daily production report, mirrored as an Excel workbook at
``github.com/ser-arthur/volve-field-data-analysis`` (15,634 rows, 26 columns of
daily volumes, pressures and well metadata). ``fetch_volve`` downloads that
workbook, keeps the oil **producers** (``WELL_TYPE == "OP"``), tidies the columns,
and vendors the subset under ``datasets/volve_production.csv`` (see
``datasets/README.md`` for provenance and the licence note). ``load_production``
reads that vendored CSV.

The production rate
-------------------
The target is the **oil rate** in Sm3/day. The raw ``BORE_OIL_VOL`` is the oil
*volume* produced on a calendar day, which mixes the reservoir's delivery rate with
how many hours the well was actually open. A day that flowed for 6 hours reports a
small volume that is not a small *rate*. So the rate is the uptime-normalised value

    rate = BORE_OIL_VOL / ON_STREAM_HRS * 24      (Sm3/day)

computed only on days with enough uptime (``ON_STREAM_HRS >= MIN_ON_STREAM_HRS``)
and positive oil. This is the producing rate the decline curve is meant to model;
``ON_STREAM_HRS`` is kept as a feature and as this filter.

The decline segment
-------------------
Arps decline describes the fall *after* a well reaches its peak rate; the early
ramp-up is not decline. :func:`well_series` therefore locates the peak on a 30-day
smoothed rate and returns the series from the peak onward, with ``t`` = days since
the peak.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPO_ROOT / "datasets" / "volve_production.csv"

VOLVE_URL = (
    "https://raw.githubusercontent.com/ser-arthur/volve-field-data-analysis/"
    "main/oilwell_production_data.xlsx"
)

# Columns kept from the raw workbook, renamed to tidy snake_case.
_RAW_COLUMNS = {
    "DATEPRD": "date",
    "WELL_BORE_CODE": "well",
    "ON_STREAM_HRS": "on_stream_hrs",
    "AVG_DOWNHOLE_PRESSURE": "avg_downhole_pressure",
    "BORE_OIL_VOL": "oil_vol",
    "BORE_GAS_VOL": "gas_vol",
    "BORE_WAT_VOL": "wat_vol",
}

MIN_ON_STREAM_HRS = 12.0  # a day must flow at least this long to give a clean rate
PEAK_SMOOTH_DAYS = 30  # window for the smoothed rate used to locate the peak


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``. Patched in tests to avoid real network I/O."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https URL
        dest.write_bytes(resp.read())


def fetch_volve(url: str = VOLVE_URL, dest: str | Path | None = None) -> int:
    """Download the Volve workbook and vendor the tidy producer subset as CSV.

    Reads the workbook with pandas + openpyxl, keeps the oil producers
    (``WELL_TYPE == "OP"``), renames the columns used downstream, simplifies the
    well code (``"NO 15/9-F-14 H"`` → ``"F-14 H"``), and writes the vendored CSV.
    Returns the number of rows written.
    """
    target = Path(dest) if dest is not None else _DATASET_PATH
    raw_path = target.parent / "_volve_raw.xlsx"
    _download(url, raw_path)
    try:
        raw = pd.read_excel(raw_path, engine="openpyxl")
    finally:
        raw_path.unlink(missing_ok=True)

    producers = raw[raw["WELL_TYPE"] == "OP"].copy()
    tidy = producers[list(_RAW_COLUMNS)].rename(columns=_RAW_COLUMNS)
    tidy["well"] = tidy["well"].str.replace("NO 15/9-", "", regex=False).str.strip()
    tidy["date"] = pd.to_datetime(tidy["date"]).dt.strftime("%Y-%m-%d")
    for col in ("on_stream_hrs", "avg_downhole_pressure", "oil_vol", "gas_vol", "wat_vol"):
        tidy[col] = pd.to_numeric(tidy[col], errors="coerce").round(3)
    tidy = tidy.sort_values(["well", "date"]).reset_index(drop=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(target, index=False)
    return len(tidy)


def load_production() -> pd.DataFrame:
    """Load the vendored tidy Volve producer records (all producing and idle days)."""
    if not _DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Vendored Volve data not found at {_DATASET_PATH}. Run "
            "`python -c 'from wellforecast.real_data import fetch_volve; fetch_volve()'`."
        )
    df = pd.read_csv(_DATASET_PATH, parse_dates=["date"])
    return df


def well_codes() -> list[str]:
    """The producer well codes present in the vendored data, longest history first."""
    df = load_production()
    counts = df.groupby("well")["date"].count().sort_values(ascending=False)
    return list(counts.index)


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rate"] = df["oil_vol"] / df["on_stream_hrs"] * 24.0
    # Gas-oil and water-oil ratios: standard diagnostic features. Guarded against
    # divide-by-zero (oil can be ~0 on a partial day that slipped the uptime filter).
    df["gor"] = np.where(df["oil_vol"] > 0, df["gas_vol"] / df["oil_vol"], np.nan)
    df["wor"] = np.where(df["oil_vol"] > 0, df["wat_vol"] / df["oil_vol"], np.nan)
    return df


def well_series(code: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one well's clean decline series, from its peak rate onward.

    Steps: keep producing days with enough uptime, build the uptime-normalised
    ``rate``, find the peak on a smoothed rate, and return the rows from the peak
    on, with ``t`` = days since the peak. Columns: ``date, t, rate, oil_vol,
    gas_vol, wat_vol, on_stream_hrs, avg_downhole_pressure, gor, wor``.
    """
    df = df if df is not None else load_production()
    g = df[df["well"] == code].sort_values("date").reset_index(drop=True)
    if g.empty:
        raise KeyError(f"no such well {code!r}; known wells: {well_codes()}")
    g = _add_derived(g)
    g = g[(g["oil_vol"] > 0) & (g["on_stream_hrs"] >= MIN_ON_STREAM_HRS)].copy()
    g = g.reset_index(drop=True)

    smoothed = g["rate"].rolling(PEAK_SMOOTH_DAYS, min_periods=5, center=True).median()
    # On a very short series the smoothed rate can be all-NaN; fall back to the raw
    # peak so the function still returns a usable decline segment.
    peak_idx = int(smoothed.idxmax()) if smoothed.notna().any() else int(g["rate"].idxmax())
    decline = g.iloc[peak_idx:].reset_index(drop=True)
    decline["t"] = (decline["date"] - decline["date"].iloc[0]).dt.days.astype(float)
    cols = [
        "date", "t", "rate", "oil_vol", "gas_vol", "wat_vol",
        "on_stream_hrs", "avg_downhole_pressure", "gor", "wor",
    ]
    return decline[cols]


def producing_wells(min_decline_days: int = 300) -> list[str]:
    """Producers with at least ``min_decline_days`` of post-peak producing history.

    These are the wells the benchmark forecasts; very short strings (e.g. a well
    that came on stream only months before the field closed) are excluded because a
    decline curve cannot be fit and tested on them.
    """
    df = load_production()
    out = []
    for code in well_codes():
        if len(well_series(code, df)) >= min_decline_days:
            out.append(code)
    return out
