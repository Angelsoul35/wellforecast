"""Tests for the real-data benchmark.

They prove the temporal split can never leak the future into the past, that the ML
correction beats the persistence baseline (and the raw Arps extrapolation) on the
pooled fleet, that the model is well-fit (a small train-internal gap), and that the
shuffle control collapses the edge — the evidence the gain is real temporal
structure, not memorised noise.
"""

from __future__ import annotations

import numpy as np

from wellforecast.real_benchmark import (
    GAP_THRESHOLD,
    format_report,
    run_benchmark,
    temporal_split,
)


def test_temporal_split_never_leaks_future_into_past(f14):
    train, test = temporal_split(f14, test_fraction=0.3)
    # The whole training region lies strictly before the whole held-out region:
    # no held-out day can precede a training day, so the future cannot leak back.
    assert train["t"].max() < test["t"].min()
    assert train["date"].max() < test["date"].min()
    # The split is a clean, exhaustive partition — nothing dropped or duplicated.
    assert len(train) + len(test) == len(f14)
    assert len(train) == int(len(f14) * 0.7)


def test_split_fraction_controls_held_out_size(f14):
    for frac in (0.2, 0.4):
        train, test = temporal_split(f14, test_fraction=frac)
        assert len(train) == int(len(f14) * (1 - frac))
        assert len(test) == len(f14) - len(train)


def test_ml_beats_persistence_on_the_fleet(report):
    assert report.ml_beats_naive
    assert report.fleet_ml["mape"] < report.fleet_naive["mape"]
    # The improvement is material, not a rounding artefact.
    assert report.fleet_ml["mape"] < 0.9 * report.fleet_naive["mape"]


def test_ml_beats_raw_arps_on_the_fleet(report):
    assert report.ml_beats_arps
    assert report.fleet_ml["mape"] < report.fleet_arps["mape"]


def test_model_is_well_fit(report):
    # Small train-internal generalisation gap -> not overfit.
    assert report.gap <= GAP_THRESHOLD, report.gap
    assert report.onestep_train_mape < report.onestep_test_mape  # some generalisation cost
    assert report.verdict.startswith("well-fit")


def test_shuffle_control_collapses_the_edge(report):
    # Destroying the temporal order of the training rates pushes the ML's held-out
    # error far above real persistence: the edge was real temporal structure.
    assert report.shuffle_collapses
    assert report.shuffle_ml_mape > report.fleet_naive["mape"]
    assert report.shuffle_ml_mape > 2 * report.fleet_ml["mape"]


def test_every_well_reports_a_bounded_b_and_three_metrics(report):
    assert len(report.wells) >= 4
    for w in report.wells:
        assert 0.0 <= w.arps["b"] <= 1.0
        for m in (w.arps_metrics, w.ml_metrics, w.naive_metrics):
            assert set(m) == {"mae", "rmse", "mape"}
            assert all(v >= 0 for v in m.values())


def test_benchmark_can_target_a_subset(production):
    report = run_benchmark(codes=["F-14 H", "F-11 H"])
    assert [w.code for w in report.wells] == ["F-14 H", "F-11 H"]


def test_format_report_renders_the_story(report):
    text = format_report(report)
    assert "Volve" in text
    assert "verdict:" in text
    assert "shuffle control" in text
    assert f"{report.fleet_ml['mape']:.3f}" in text
    assert "persistence" in text


def test_main_prints_a_report(capsys):
    from wellforecast.real_benchmark import main

    main()
    out = capsys.readouterr().out
    assert "WellForecast" in out
    assert "verdict:" in out


def test_naive_is_the_last_training_value_held_flat(f14):
    # Pin the definition of the persistence baseline the whole comparison rests on.
    train, test = temporal_split(f14)
    naive_value = train["rate"].to_numpy(float)[-1]
    report = run_benchmark(codes=["F-14 H"])
    # Reconstruct the naive MAE and match the reported one.
    q_te = test["rate"].to_numpy(float)
    expected_mae = float(np.mean(np.abs(q_te - naive_value)))
    assert report.wells[0].naive_metrics["mae"] == round(expected_mae, 4)
