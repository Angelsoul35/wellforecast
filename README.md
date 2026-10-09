# WellForecast

**Oil-well production forecasting: Arps decline curves + machine learning, on the real Volve field.**

WellForecast forecasts the oil production rate of declining wells. It combines the
petroleum engineer's classic tool — **Arps decline-curve analysis** — with a
gradient-boosting model that corrects the decline curve using recent production
dynamics, and it measures both, rigorously, against a naive baseline on a
**leakage-free temporal split**.

> Everything runs on **real** data: Equinor's open **Volve** field (North Sea),
> daily production for six oil producers, 2008–2016. No synthetic production is used
> for the headline numbers — a synthetic Arps curve appears only in the tests, to
> prove the decline-curve fitter recovers known parameters. See
> [`datasets/README.md`](datasets/README.md) for provenance and licence.

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why it exists

Forecasting how much a well will produce next year drives reserves booking,
investment, and abandonment decisions. Two cultures answer it: petroleum engineers
fit **decline curves** (interpretable, physical, extrapolate by construction), and
data scientists fit **ML models** (flexible, data-hungry, poor at extrapolating a
declining trend). WellForecast builds both, shows exactly where each wins, and
combines them — the decline curve carries the long-run trend, the ML corrects its
departures. It also does the parts that separate a real forecast from a demo:
a temporal split that never lets the future leak into the past, comparison against a
naive baseline, an over/underfit check, and a shuffle control.

## How it works

```
 Volve daily volumes          Arps decline curve            ML correction
 oil / on-stream hours   -->   q(t)=q_i/(1+b·D_i·t)^(1/b) -->  rate = arps(t) · ρ̂(t)
   = producing rate            (scipy, b∈[0,1])            gradient boosting on ρ
        |                             |                     (recent ρ lags/rolls, t)
        |                             v                            |
   peak detection              long-run decline shape,       learns the departures
   t = days since peak         EUR via cumulative            from the smooth curve
```

The ML never predicts the raw rate — a tree model cannot extrapolate below the
lowest rate it saw in training, so on a declining well it would just flatten. It
predicts the **correction ratio** `ρ(t) = rate(t) / arps(t)`, which is stationary
(it hovers around 1), and the forecast is `arps(t) · ρ̂(t)`: the Arps curve supplies
the decline, the ML supplies the correction.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m wellforecast.real_benchmark    # the full fleet benchmark (below)
python -m wellforecast.cli demo          # fit + forecast one well, then the fleet
make test                                # the test suite with coverage
```

## The two layers

### 1. Arps decline curve (`arps.py`) — the domain baseline

The industry-standard empirical decline, fit per well with non-linear least squares
(`scipy.optimize.curve_fit`), with the exponent bounded to its physical range:

```
q(t) = q_i / (1 + b · D_i · t) ** (1 / b)          0 ≤ b ≤ 1
```

* `b = 0` → **exponential** decline (`q = q_i · e^(−D_i·t)`, the b→0 limit);
* `0 < b < 1` → **hyperbolic** (the common case);
* `b = 1` → **harmonic** (`q = q_i / (1 + D_i·t)`).

The fit minimises *relative* error (it regresses `log q`), the right objective for a
rate that spans an order of magnitude over a well's life. The model also integrates
to a cumulative (EUR-style) forecast.

### 2. ML correction (`model.py`) — gradient boosting on the ratio

A `GradientBoostingRegressor` predicts the correction ratio from features that use
only the past (every lag and rolling feature is shifted): recent ratio lags
(1/7/14 days), its 7- and 30-day trailing means, its recent slope, and time on
production. The multi-step forecast is recursive — each predicted ratio feeds the
next step's features — so no future rate is ever used. The trees are deliberately
shallow and leaf-constrained: a heavier model shaves a little off the headline error
but opens a wide train/validation gap (it memorises day-to-day noise), which is the
wrong trade.

## Results on real Volve data

Run `python -m wellforecast.real_benchmark`. Each well is split **in time** — the
early history trains, the later tail is held out — and three forecasts of the
held-out tail are compared: the Arps extrapolation, the ML correction, and
**persistence** (the last training rate held flat, the naive "last value" baseline).

```
    well  train  test     b      regime  Arps MAPE  ML MAPE  naive MAPE   ML MAE
  F-14 H   1807   775  0.00 exponential      0.869    0.640       1.348     112
  F-12 H   1701   730  0.01 exponential      0.594    0.618       0.385     274
  F-11 H    347   149  0.00 exponential      0.333    0.143       0.188      67
  F-15 D    522   224  1.00    harmonic      0.316    0.198       0.222      30
   F-1 C    289   124  1.00    harmonic      0.488    0.606       0.521     106
---------------------------------------------------------------------------------
   FLEET (pooled held-out days)              0.643    0.543       0.733     158

fleet MAE (Sm3/day) : Arps 183  ML 158  naive 175
fleet RMSE (Sm3/day): Arps 236  ML 228  naive 235
ML beats persistence: True  (3/5 wells)    ML beats raw Arps: True
one-step MAPE in-sample/validation (within training): 0.0676 / 0.1258  (gap 0.0582)
shuffle control (train rates shuffled in time): ML 6.879 vs real persistence 0.733
verdict: well-fit
```

**What this says.** Over a demanding multi-year held-out tail, the ML correction
cuts the pooled MAPE by about a quarter versus naive persistence (0.543 vs 0.733)
and beats the raw Arps extrapolation (0.643), while staying **well-fit** — the
one-step train-vs-validation gap, measured *inside* the training era so it isn't
confused with the decline itself, is only 0.058.

These are **real, mid-single-digit-to-tens-of-percent** errors, not a fake near-
zero: real wells are noisy, go through interventions, and a two-year-ahead daily
forecast is genuinely hard. Per well, the ML wins where the tail keeps declining
(F-14, F-11, F-15) and loses where it plateaus — persistence's best case — on F-12
and the short, noisy F-1 C. That is reported, not hidden.

**The two layers are complementary, not rivals.** Arps is the interpretable
reference: it needs no recent data, gives the decline shape and an EUR, and is what a
reservoir engineer reads. The ML sits on top and corrects where the real well leaves
the smooth curve. Together they beat either alone.

**The signal is real.** The shuffle control retrains with each well's training rates
scrambled in time: the ML's held-out error explodes to 6.9 MAPE — far above plain
persistence — so its edge came from genuine temporal structure, not from fitting
noise. `tests/test_real_benchmark.py` asserts all of this.

## Anti-leakage, by construction

* **Temporal split.** `temporal_split` is a positional split in chronological order,
  so every held-out day comes strictly after every training day. A test asserts
  `train.date.max() < test.date.min()`.
* **Shifted features.** Every lag and rolling feature uses `shift`, so the feature
  row for day *t* depends only on days before *t*. A test perturbs one day and checks
  no earlier feature row moves.
* **Train-only fits.** The Arps curve and the ML are fit on the training region
  alone; the held-out tail is never seen during fitting.
* **Leakage-free over/underfit check.** The gap is measured on a sub-split *within* the
  training era (same regime) so it is not inflated by the decline's natural
  non-stationarity.

## Project layout

```
wellforecast/
  real_data.py      load/fetch the real Volve producers; build the per-well rate series
  arps.py           the Arps decline-curve model: fit (scipy) + forecast + cumulative
  features.py       ratio-correction features (leakage-free) + engineering diagnostics
  model.py          the gradient-boosting correction + MAE/RMSE/MAPE metrics
  real_benchmark.py temporal held-out: Arps + ML vs persistence, gap, shuffle, verdict
  cli.py            a `demo` command
```

## Testing and quality

* **44 tests** cover the Arps model (parameter recovery on a synthetic curve,
  exponential/harmonic limits, the cumulative, the `b` bound, a sane fit on a real
  well), the real-data loader and the monkeypatched fetch/tidy path, the leakage-free
  features, the metrics and the forecaster, and the benchmark (the temporal split
  can't leak, ML beats persistence and Arps on the fleet, the well-fit verdict, and
  the shuffle collapse). CI requires at least 85% coverage (currently ~98%).
* `ruff` for lint. CI runs lint, tests, and a benchmark smoke test on every push.

See the Spanish study guide in [`docs/estudio/`](docs/estudio/) for the physics of
decline curves, why temporal splits, the metrics, and a worked example with these
real numbers.

## Notes and next steps

A single Arps curve cannot follow every well intervention, which is exactly why the
ML correction helps and why long-horizon extrapolation is hard; a production system
would add segmented decline fits, a direct multi-horizon model for long lead times,
and prediction intervals. The rate here is uptime-normalised daily oil
(`oil / on-stream hours · 24`); pressure, GOR and WOR are exposed as features for
extending the correction model.

## License

[MIT](LICENSE) © Angelo David Padilla Romero
