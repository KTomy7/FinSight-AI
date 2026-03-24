from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from finsight.infrastructure.features import TimeSplitPolicy


def _to_iso_dates(series: pd.Series) -> list[str]:
    return series.apply(lambda value: value.date().isoformat()).tolist()


def test_basic_split() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "target_ret_1d": [0.1, 0.2, 0.3, 0.4],
        }
    )

    policy = TimeSplitPolicy(cutoff_date="2024-01-03")
    train_df, test_df = policy.split_frame(df)

    assert _to_iso_dates(train_df["date"]) == ["2024-01-01", "2024-01-02"]
    assert _to_iso_dates(test_df["date"]) == ["2024-01-03", "2024-01-04"]
    assert set(train_df.columns) == set(df.columns)
    assert set(test_df.columns) == set(df.columns)


def test_cutoff_date_normalizes_to_date_for_supported_input_types() -> None:
    from_str = TimeSplitPolicy(cutoff_date="2024-01-03")
    from_date = TimeSplitPolicy(cutoff_date=date(2024, 1, 3))
    from_datetime = TimeSplitPolicy(cutoff_date=datetime(2024, 1, 3, 15, 30, 0))

    assert from_str.cutoff_date == date(2024, 1, 3)
    assert from_date.cutoff_date == date(2024, 1, 3)
    assert from_datetime.cutoff_date == date(2024, 1, 3)


def test_boundary_goes_to_test() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "ticker": ["AAA", "AAA", "AAA"],
            "x": [1.0, 2.0, 3.0],
            "target_ret_1d": [0.1, 0.2, 0.3],
        }
    )

    policy = TimeSplitPolicy(cutoff_date="2024-01-02")
    train_df, test_df = policy.split_frame(df)

    assert _to_iso_dates(train_df["date"]) == ["2024-01-01"]
    assert _to_iso_dates(test_df["date"]) == ["2024-01-02", "2024-01-03"]


def test_boundary_goes_to_train_when_inclusive_test_false() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "ticker": ["AAA", "AAA", "AAA"],
            "x": [1.0, 2.0, 3.0],
            "target_ret_1d": [0.1, 0.2, 0.3],
        }
    )

    policy = TimeSplitPolicy(cutoff_date="2024-01-02", inclusive_test=False)
    train_df, test_df = policy.split_frame(df)

    assert _to_iso_dates(train_df["date"]) == ["2024-01-01", "2024-01-02"]
    assert _to_iso_dates(test_df["date"]) == ["2024-01-03"]
    assert "2024-01-02" in _to_iso_dates(train_df["date"])
    assert "2024-01-02" not in _to_iso_dates(test_df["date"])


def test_missing_date_column_raises() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "x": [1.0, 2.0],
            "target_ret_1d": [0.1, 0.2],
        }
    )

    policy = TimeSplitPolicy(cutoff_date="2024-01-02")

    with pytest.raises(ValueError, match="Missing required date column"):
        policy.split_frame(df)


def test_invalid_date_values_raise() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "not-a-date", "2024-01-03"],
            "ticker": ["AAA", "AAA", "AAA"],
            "x": [1.0, 2.0, 3.0],
            "target_ret_1d": [0.1, 0.2, 0.3],
        }
    )

    policy = TimeSplitPolicy(cutoff_date="2024-01-02")

    with pytest.raises(ValueError, match="invalid date value"):
        policy.split_frame(df)


def test_empty_train_or_test_raises() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-10", "2024-01-11", "2024-01-12"],
            "ticker": ["AAA", "AAA", "AAA"],
            "x": [1.0, 2.0, 3.0],
            "target_ret_1d": [0.1, 0.2, 0.3],
        }
    )

    with pytest.raises(ValueError, match="Time split produced an empty partition"):
        TimeSplitPolicy(cutoff_date="2024-01-01").split_frame(df)

    with pytest.raises(ValueError, match="Time split produced an empty partition"):
        TimeSplitPolicy(cutoff_date="2024-02-01").split_frame(df)


def test_split_is_deterministically_sorted_from_unsorted_input() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04", "2024-01-01", "2024-01-03"],
            "ticker": ["BBB", "AAA", "AAA", "BBB", "AAA", "BBB"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target_ret_1d": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    policy = TimeSplitPolicy(cutoff_date="2024-01-03")
    train_df, test_df = policy.split_frame(df)

    train_pairs = list(zip(train_df["ticker"].tolist(), _to_iso_dates(train_df["date"])))
    test_pairs = list(zip(test_df["ticker"].tolist(), _to_iso_dates(test_df["date"])))

    assert train_pairs == [
        ("AAA", "2024-01-01"),
        ("AAA", "2024-01-02"),
        ("BBB", "2024-01-01"),
    ]
    assert test_pairs == [
        ("AAA", "2024-01-03"),
        ("BBB", "2024-01-03"),
        ("BBB", "2024-01-04"),
    ]

