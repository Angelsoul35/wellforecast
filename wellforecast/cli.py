"""WellForecast command-line interface."""

from __future__ import annotations

import argparse

from .arps import fit_arps
from .model import DeclineForecaster, metrics
from .real_benchmark import format_report, run_benchmark, temporal_split
from .real_data import producing_wells, well_series


def cmd_demo(args) -> None:
    """Fit and forecast one well end to end, then print the fleet benchmark."""
    code = args.well or producing_wells()[0]
    series = well_series(code)
    train, test = temporal_split(series)
    t_tr, q_tr = train["t"].to_numpy(float), train["rate"].to_numpy(float)
    t_te, q_te = test["t"].to_numpy(float), test["rate"].to_numpy(float)

    print(f"1) Well {code}: {len(series)} post-peak producing days "
          f"({len(train)} train / {len(test)} held out)")

    arps = fit_arps(t_tr, q_tr)
    print(f"2) Arps fit (train)  qi={arps.qi:.0f} Sm3/d  Di={arps.di:.5f}/day  "
          f"b={arps.b:.3f} ({arps.regime})")
    arps_m = metrics(q_te, arps.predict(t_te))
    print(f"   Arps held-out       {arps_m}")

    forecaster = DeclineForecaster().fit(t_tr, q_tr, arps)
    ml_m = metrics(q_te, forecaster.forecast(t_te))
    print(f"3) ML correction held-out {ml_m}")

    eur = arps.cumulative(float(series['t'].iloc[-1]) + 365 * 5)
    print(f"4) Arps 5-year-ahead cumulative (EUR-style): {eur / 1000:.0f} kSm3")

    print("\n5) Fleet benchmark:")
    print(format_report(run_benchmark()))


def main() -> None:
    parser = argparse.ArgumentParser(description="WellForecast CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="fit + forecast one well, then the fleet benchmark")
    p.add_argument("--well", type=str, default=None, help="well code, e.g. 'F-14 H'")
    p.set_defaults(func=cmd_demo)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
