"""Unit tests for finsight.domain.value_objects."""
from __future__ import annotations

import datetime

import pytest

from finsight.domain.value_objects import DateRange, Interval, Period, Ticker


# ---------------------------------------------------------------------------
# Ticker
# ---------------------------------------------------------------------------


class TestTicker:
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Ticker cannot be empty."):
            Ticker("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Ticker cannot be empty."):
            Ticker("  ")

    def test_normalises_to_uppercase(self) -> None:
        t = Ticker(" aapl ")
        assert t.value == "AAPL"

    def test_str_returns_value(self) -> None:
        assert str(Ticker("MSFT")) == "MSFT"

    def test_already_uppercase_stored_as_is(self) -> None:
        assert Ticker("GOOG").value == "GOOG"


# ---------------------------------------------------------------------------
# Period
# ---------------------------------------------------------------------------


class TestPeriod:
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Period cannot be empty."):
            Period("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Period cannot be empty."):
            Period("   ")

    def test_str_returns_value(self) -> None:
        assert str(Period("1y")) == "1y"

    def test_default_value(self) -> None:
        assert Period().value == "1y"


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------


class TestInterval:
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Interval cannot be empty."):
            Interval("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Interval cannot be empty."):
            Interval("   ")

    def test_str_returns_value(self) -> None:
        assert str(Interval("1d")) == "1d"

    def test_default_value(self) -> None:
        assert Interval().value == "1d"


# ---------------------------------------------------------------------------
# DateRange
# ---------------------------------------------------------------------------


class TestDateRange:
    def test_valid_iso_strings_accepted(self) -> None:
        dr = DateRange(start="2024-01-01", end="2024-03-31")
        assert dr.start == datetime.date(2024, 1, 1)
        assert dr.end == datetime.date(2024, 3, 31)

    def test_invalid_iso_start_raises(self) -> None:
        with pytest.raises(ValueError, match="DateRange.start is not a valid ISO date"):
            DateRange(start="not-a-date", end="2024-01-01")

    def test_invalid_iso_end_raises(self) -> None:
        with pytest.raises(ValueError, match="DateRange.end is not a valid ISO date"):
            DateRange(start="2024-01-01", end="bad-end")

    def test_datetime_start_normalised_to_date(self) -> None:
        dt_start = datetime.datetime(2024, 6, 15, 12, 30, 0)
        dr = DateRange(start=dt_start, end=datetime.date(2024, 6, 20))
        assert dr.start == datetime.date(2024, 6, 15)
        assert isinstance(dr.start, datetime.date)
        assert not isinstance(dr.start, datetime.datetime)

    def test_datetime_end_normalised_to_date(self) -> None:
        dt_end = datetime.datetime(2024, 6, 20, 23, 59, 59)
        dr = DateRange(start=datetime.date(2024, 6, 15), end=dt_end)
        assert dr.end == datetime.date(2024, 6, 20)
        assert not isinstance(dr.end, datetime.datetime)

    def test_start_greater_than_end_raises(self) -> None:
        with pytest.raises(ValueError, match="start.*<=.*end"):
            DateRange(start="2024-03-01", end="2024-01-01")

    def test_start_equals_end_accepted(self) -> None:
        dr = DateRange(start="2024-06-01", end="2024-06-01")
        assert dr.start == dr.end

    def test_end_exclusive_is_end_plus_one_day(self) -> None:
        dr = DateRange(start="2024-01-01", end="2024-01-31")
        assert dr.end_exclusive == datetime.date(2024, 2, 1)

    def test_str_format(self) -> None:
        dr = DateRange(start="2024-01-01", end="2024-12-31")
        assert str(dr) == "2024-01-01 to 2024-12-31"
