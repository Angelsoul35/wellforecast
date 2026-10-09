# Datasets

## `volve_production.csv` — real Volve field daily production (Equinor open data)

Daily production records for the oil **producers** of the Volve field, a North Sea
oil field in the Norwegian sector that Equinor (then Statoil) operated from 2008 to
2016 and released in full as open data in 2018. WellForecast forecasts these wells'
oil rate; the file is the real series every headline number is computed on.

**Source.** Equinor's Volve daily production report, mirrored as an Excel workbook at
[`ser-arthur/volve-field-data-analysis`](https://github.com/ser-arthur/volve-field-data-analysis)
(`oilwell_production_data.xlsx`, 15,634 rows × 26 columns of daily volumes, pressures
and well metadata). `wellforecast.real_data.fetch_volve` downloads that workbook,
keeps the rows with `WELL_TYPE == "OP"` (the oil producers, dropping the water
injectors), renames the handful of columns used downstream, simplifies the well code
(`"NO 15/9-F-14 H"` → `"F-14 H"`), and writes this CSV.

**Columns.**

| Column                  | Unit     | Meaning                                                 |
| ----------------------- | -------- | ------------------------------------------------------- |
| `date`                  | —        | production day (`DATEPRD`)                              |
| `well`                  | —        | well-bore code, e.g. `F-14 H`                           |
| `on_stream_hrs`         | hours    | hours the well was open that day (`ON_STREAM_HRS`)      |
| `avg_downhole_pressure` | bar      | average downhole pressure (`AVG_DOWNHOLE_PRESSURE`)     |
| `oil_vol`               | Sm³      | daily oil volume (`BORE_OIL_VOL`) — the production      |
| `gas_vol`               | Sm³      | daily gas volume (`BORE_GAS_VOL`)                       |
| `wat_vol`               | Sm³      | daily water volume (`BORE_WAT_VOL`)                     |

9,143 daily rows span 2008-02-12 to 2016-09-17 across 6 producers (`F-12 H`,
`F-14 H`, `F-11 H`, `F-15 D`, `F-1 C`, `F-5 AH`), including idle days (zero oil).

**How the rate is built.** The loader derives the oil **rate** on the fly as
`oil_vol / on_stream_hrs * 24` (Sm³/day), keeping only producing days with
`on_stream_hrs >= 12`, and cuts each well's decline segment from its smoothed peak.
That logic lives in `wellforecast.real_data.well_series`, so this file stays a
faithful, minimally-processed subset of the source. `gor` and `wor` (gas-oil and
water-oil ratios) are derived in `wellforecast.features`, not stored here.

### Regenerating

```bash
python -c "from wellforecast.real_data import fetch_volve; print(fetch_volve())"
```

Needs outbound access to `raw.githubusercontent.com`. The Volve data is published by
Equinor under its open-data terms (Equinor Open Data Licence / CC BY 4.0-style
attribution to Equinor and the Volve licence partners); this producer subset is
vendored for reproducible benchmarking only, with attribution to Equinor as the
data owner.
