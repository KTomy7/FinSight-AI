from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import finsight.adapters.web_streamlit.views.market_data as market_data_view
from finsight.domain.entities import OHLCVSeries, StockSummary
from finsight.domain.value_objects import DateRange, Interval, Ticker


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture
def market_data_settings():
    return SimpleNamespace(
        ticker_catalog=SimpleNamespace(
            entries=(
                SimpleNamespace(symbol="AAPL", company_name="Apple Inc."),
                SimpleNamespace(symbol="JPM", company_name="JPMorgan Chase & Co."),
            )
        )
    )


@pytest.fixture
def market_data_result(mock_stock_price_df: pd.DataFrame):
    history = OHLCVSeries(
        ticker=Ticker("AAPL"),
        date_range=DateRange(start="2024-01-01", end="2024-01-10"),
        interval=Interval("1d"),
        df=mock_stock_price_df,
    )
    summary = StockSummary(
        ticker=Ticker("AAPL"),
        data={
            "name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "current_price": 109.5,
            "previous_close": 108.0,
            "market_cap": 2_400_000_000_000,
            "pe_ratio": 29.1,
            "fifty_two_week_high": 199.9,
            "fifty_two_week_low": 124.17,
            "volume": 1_090_000,
            "avg_volume": 1_045_000,
            "dividend_yield": 0.0045,
        },
    )
    return SimpleNamespace(history=history, summary=summary)


@pytest.fixture
def market_data_result_no_summary(mock_stock_price_df: pd.DataFrame):
    history = OHLCVSeries(
        ticker=Ticker("AAPL"),
        date_range=DateRange(start="2024-01-01", end="2024-01-10"),
        interval=Interval("1d"),
        df=mock_stock_price_df,
    )
    return SimpleNamespace(history=history, summary=None)


def _patch_common_ui(monkeypatch, events: list[tuple[str, object]]) -> None:
    monkeypatch.setattr(market_data_view.st, "title", lambda _msg: None)
    monkeypatch.setattr(market_data_view.st, "markdown", lambda _msg: None)
    monkeypatch.setattr(market_data_view.st, "subheader", lambda msg: events.append(("subheader", msg)))
    monkeypatch.setattr(market_data_view.st, "info", lambda msg: events.append(("info", msg)))
    monkeypatch.setattr(market_data_view.st, "warning", lambda msg: events.append(("warning", msg)))
    monkeypatch.setattr(market_data_view.st, "success", lambda msg: events.append(("success", msg)))
    monkeypatch.setattr(market_data_view.st, "error", lambda msg: events.append(("error", msg)))
    monkeypatch.setattr(market_data_view.st, "caption", lambda msg: events.append(("caption", msg)))
    monkeypatch.setattr(market_data_view.st, "write", lambda msg: events.append(("write", msg)))
    monkeypatch.setattr(market_data_view.st, "metric", lambda label, value, **kwargs: events.append(("metric", (label, value, kwargs))))
    monkeypatch.setattr(market_data_view.st, "dataframe", lambda frame, **kwargs: events.append(("dataframe", frame.copy(), kwargs)))
    monkeypatch.setattr(market_data_view.st, "line_chart", lambda frame, **kwargs: events.append(("line_chart", frame.copy(), kwargs)))
    monkeypatch.setattr(market_data_view.st, "form", lambda _name: _Ctx())
    monkeypatch.setattr(market_data_view.st, "columns", lambda spec: tuple(_Ctx() for _ in range(len(spec) if isinstance(spec, list) else spec)))
    monkeypatch.setattr(market_data_view.st, "tabs", lambda labels: tuple(_Ctx() for _ in labels))
    monkeypatch.setattr(market_data_view.st, "spinner", lambda _msg: _Ctx())
    monkeypatch.setattr(market_data_view.st, "session_state", _SessionState(), raising=False)


def test_render_stops_before_fetch_when_form_is_not_submitted(monkeypatch, market_data_settings) -> None:
    events: list[tuple[str, object]] = []
    container_called = []

    monkeypatch.setattr(market_data_view, "_SETTINGS", market_data_settings)
    monkeypatch.setattr(market_data_view, "build_ticker_select_items", lambda _entries: [("AAPL", "AAPL - Apple Inc."), ("JPM", "JPM - JPMorgan Chase & Co.")])
    _patch_common_ui(monkeypatch, events)
    monkeypatch.setattr(market_data_view.st, "selectbox", lambda _label, options, **_kwargs: options[0])
    monkeypatch.setattr(market_data_view.st, "form_submit_button", lambda _label, **_kwargs: False)
    monkeypatch.setattr(market_data_view, "build_container", lambda: container_called.append(True) or SimpleNamespace(fetch_market_data=SimpleNamespace(execute=lambda _req: None)))

    market_data_view.render()

    assert ("info", "👉 Select a ticker and click 'Load Data' to begin your analysis.") in events
    assert container_called == []
    assert not any(kind in {"line_chart", "metric", "dataframe", "success"} for kind, *_ in events)


def test_render_renders_summary_metrics_and_ohlc_lines(monkeypatch, market_data_settings, market_data_result) -> None:
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(market_data_view, "_SETTINGS", market_data_settings)
    monkeypatch.setattr(market_data_view, "build_ticker_select_items", lambda _entries: [("AAPL", "AAPL - Apple Inc."), ("JPM", "JPM - JPMorgan Chase & Co.")])
    _patch_common_ui(monkeypatch, events)
    monkeypatch.setattr(market_data_view.st, "selectbox", lambda _label, options, **_kwargs: options[0])
    monkeypatch.setattr(market_data_view.st, "form_submit_button", lambda _label, **_kwargs: True)

    monkeypatch.setattr(
        market_data_view,
        "build_container",
        lambda: SimpleNamespace(fetch_market_data=SimpleNamespace(execute=lambda _req: market_data_result)),
    )

    market_data_view.render()

    assert ("success", "✅ Data loaded for AAPL") in events
    assert ("subheader", "💰 Price & Valuation Metrics") in events
    assert ("subheader", "📉 Historical Price Chart") in events
    assert ("subheader", "📊 Historical Data Table") in events
    assert any(kind == "metric" and payload[0] == "Current Price" for kind, payload, *_ in events if kind == "metric")
    assert any(kind == "metric" and payload[0] == "Market Cap" for kind, payload, *_ in events if kind == "metric")
    line_chart_events = [payload for kind, payload, *_ in events if kind == "line_chart"]
    assert line_chart_events
    chart_frame = line_chart_events[0]
    assert list(chart_frame.columns) == ["Open", "High", "Low", "Close"]
    assert any(kind == "dataframe" for kind, *_ in events)
    assert any(kind == "caption" and "trading days" in str(payload) for kind, payload, *_ in events if kind == "caption")


def test_render_uses_lowercase_ohlc_fallback_for_line_chart(monkeypatch, market_data_settings) -> None:
    events: list[tuple[str, object]] = []
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-02-01", periods=3, freq="D"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000_000, 1_100_000, 1_200_000],
        }
    )
    result = SimpleNamespace(
        history=SimpleNamespace(
            ticker=Ticker("AAPL"),
            date_range=SimpleNamespace(start=pd.Timestamp("2024-02-01").date(), end=pd.Timestamp("2024-02-03").date()),
            interval=Interval("1d"),
            df=df,
        ),
        summary=None,
    )

    monkeypatch.setattr(market_data_view, "_SETTINGS", market_data_settings)
    monkeypatch.setattr(market_data_view, "build_ticker_select_items", lambda _entries: [("AAPL", "AAPL - Apple Inc.")])
    _patch_common_ui(monkeypatch, events)
    monkeypatch.setattr(market_data_view.st, "selectbox", lambda _label, options, **_kwargs: options[0])
    monkeypatch.setattr(market_data_view.st, "form_submit_button", lambda _label, **_kwargs: True)
    monkeypatch.setattr(
        market_data_view,
        "build_container",
        lambda: SimpleNamespace(fetch_market_data=SimpleNamespace(execute=lambda _req: result)),
    )

    market_data_view.render()

    line_chart_events = [payload for kind, payload, *_ in events if kind == "line_chart"]
    assert line_chart_events
    chart_frame = line_chart_events[0]
    assert list(chart_frame.columns) == ["open", "high", "low", "close"]


def test_render_warns_when_use_case_raises(monkeypatch, market_data_settings) -> None:
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(market_data_view, "_SETTINGS", market_data_settings)
    monkeypatch.setattr(market_data_view, "build_ticker_select_items", lambda _entries: [("AAPL", "AAPL - Apple Inc.")])
    _patch_common_ui(monkeypatch, events)
    monkeypatch.setattr(market_data_view.st, "selectbox", lambda _label, options, **_kwargs: options[0])
    monkeypatch.setattr(market_data_view.st, "form_submit_button", lambda _label, **_kwargs: True)
    monkeypatch.setattr(
        market_data_view,
        "build_container",
        lambda: SimpleNamespace(fetch_market_data=SimpleNamespace(execute=lambda _req: (_ for _ in ()).throw(RuntimeError("boom")))),
    )

    market_data_view.render()

    assert any(kind == "error" and "Failed to load market data" in str(payload) for kind, payload in events)
    assert any(kind == "info" and "Tip: Make sure the ticker is valid" in str(payload) for kind, payload in events)


