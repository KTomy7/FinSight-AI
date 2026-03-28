"""Unit tests for helper functions in finsight.application.use_cases.train_model."""
from __future__ import annotations

import pytest

from finsight.application.use_cases.train_model import (
    _get_training_tickers,
    _parse_iso_date,
    _validate_model_types,
)


# ---------------------------------------------------------------------------
# _validate_model_types
# ---------------------------------------------------------------------------


class TestValidateModelTypes:
    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="model_types must contain at least one model type."):
            _validate_model_types([])

    def test_valid_single_type_returns_list(self) -> None:
        result = _validate_model_types(["naive_zero"])
        assert result == ["naive_zero"]

    def test_valid_both_types_returns_list(self) -> None:
        result = _validate_model_types(["naive_zero", "naive_mean"])
        assert result == ["naive_zero", "naive_mean"]

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported model type"):
            _validate_model_types(["naive_last"])

    def test_duplicate_types_raises(self) -> None:
        with pytest.raises(ValueError, match="model_types must be unique"):
            _validate_model_types(["naive_zero", "naive_zero"])


# ---------------------------------------------------------------------------
# _parse_iso_date
# ---------------------------------------------------------------------------


class TestParseIsoDate:
    def test_valid_iso_string_returns_date(self) -> None:
        import datetime

        result = _parse_iso_date("2024-06-01")
        assert result == datetime.date(2024, 6, 1)

    def test_invalid_string_raises_with_message(self) -> None:
        with pytest.raises(ValueError, match="Invalid ISO 8601 date for 'end'"):
            _parse_iso_date("not-a-date")

    def test_partially_valid_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid ISO 8601 date for 'end'"):
            _parse_iso_date("2024-13-01")


# ---------------------------------------------------------------------------
# _get_training_tickers
# ---------------------------------------------------------------------------


class TestGetTrainingTickers:
    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="must contain at least one symbol"):
            _get_training_tickers([])

    def test_duplicate_tickers_raises(self) -> None:
        with pytest.raises(ValueError, match="must not contain duplicates"):
            _get_training_tickers(["AAPL", "AAPL"])

    def test_case_normalisation(self) -> None:
        result = _get_training_tickers(["aapl", "msft"])
        assert result == ["AAPL", "MSFT"]

    def test_whitespace_stripped(self) -> None:
        result = _get_training_tickers([" GOOG "])
        assert result == ["GOOG"]

    def test_valid_tickers_returned(self) -> None:
        result = _get_training_tickers(["AAPL", "MSFT"])
        assert result == ["AAPL", "MSFT"]

    def test_duplicates_detected_after_normalisation(self) -> None:
        # "aapl" and "AAPL" normalise to the same ticker → duplicate
        with pytest.raises(ValueError, match="must not contain duplicates"):
            _get_training_tickers(["aapl", "AAPL"])


# ---------------------------------------------------------------------------
# TrainModel.execute — years <= 0
# ---------------------------------------------------------------------------


class TestTrainModelExecuteValidation:
    def _make_use_case(self) -> "TrainModel":
        """Build a TrainModel instance with a minimal stub for FetchMarketData."""
        from unittest.mock import MagicMock
        from finsight.application.use_cases.train_model import TrainModel
        from finsight.application.use_cases.fetch_market_data import FetchMarketData

        mock_fetch = MagicMock(spec=FetchMarketData)
        return TrainModel(
            fetch_market_data=mock_fetch,
            training_tickers=["AAPL"],
        )

    def test_years_zero_raises(self) -> None:
        from finsight.application.use_cases.train_model import TrainModelRequest

        uc = self._make_use_case()
        with pytest.raises(ValueError, match="years must be a positive integer"):
            uc.execute(TrainModelRequest(cutoff_date="2024-01-01", years=0))

    def test_years_negative_raises(self) -> None:
        from finsight.application.use_cases.train_model import TrainModelRequest

        uc = self._make_use_case()
        with pytest.raises(ValueError, match="years must be a positive integer"):
            uc.execute(TrainModelRequest(cutoff_date="2024-01-01", years=-1))
