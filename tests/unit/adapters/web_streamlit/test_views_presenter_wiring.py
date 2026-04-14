from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import finsight.adapters.web_streamlit.views.compare as compare_view
import finsight.adapters.web_streamlit.views.predict as predict_view
from finsight.application.dto import CompareModelsResult, ForecastResult, ModelComparisonRow


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ButtonCol(_Ctx):
    def __init__(self, returns: dict[str, bool] | None = None) -> None:
        self._returns = returns or {}

    def button(self, label: str, **_kwargs) -> bool:
        return self._returns.get(label, False)


def test_compare_render_shows_info_when_form_not_submitted(monkeypatch) -> None:
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(compare_view, "_SETTINGS", SimpleNamespace(model_defaults=SimpleNamespace(
        training_model_ids=lambda: ("ridge",),
        id_to_label=lambda: {"ridge": "Ridge"},
    )))
    monkeypatch.setattr(compare_view.st, "title", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "markdown", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "info", lambda msg: events.append(("info", msg)))
    monkeypatch.setattr(compare_view.st, "warning", lambda msg: events.append(("warning", msg)))
    monkeypatch.setattr(compare_view.st, "form", lambda _name: _Ctx())
    monkeypatch.setattr(compare_view.st, "multiselect", lambda _label, options, **_kwargs: list(options))
    monkeypatch.setattr(compare_view.st, "form_submit_button", lambda _label: False)

    compare_view.render()

    assert ("info", "Choose models and ranking metrics, then build the leaderboard.") in events


def test_compare_render_happy_path_renders_dataframe_and_caption(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(compare_view, "_SETTINGS", SimpleNamespace(model_defaults=SimpleNamespace(
        training_model_ids=lambda: ("ridge",),
        id_to_label=lambda: {"ridge": "Ridge"},
    )))
    monkeypatch.setattr(compare_view.st, "title", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "markdown", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "info", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "warning", lambda msg: events.append(("warning", msg)))
    monkeypatch.setattr(compare_view.st, "error", lambda msg: events.append(("error", msg)))
    monkeypatch.setattr(compare_view.st, "subheader", lambda msg: events.append(("subheader", msg)))
    monkeypatch.setattr(compare_view.st, "dataframe", lambda frame, **_kwargs: events.append(("dataframe", frame)))
    monkeypatch.setattr(compare_view.st, "caption", lambda msg: events.append(("caption", msg)))
    monkeypatch.setattr(compare_view.st, "form", lambda _name: _Ctx())

    selections = {
        "Models to compare": ["ridge"],
        "Ranking metrics": ["mae", "rmse"],
    }
    monkeypatch.setattr(compare_view.st, "multiselect", lambda label, _options, **_kwargs: selections[label])
    monkeypatch.setattr(compare_view.st, "form_submit_button", lambda _label: True)

    result = CompareModelsResult(
        rows=[
            ModelComparisonRow(
                rank=1,
                model_id="ridge",
                run_id="run_1",
                metrics={"mae": 0.1, "rmse": 0.2},
                sort_key=(0.1, 0.2, "ridge", "run_1"),
            )
        ],
        rank_by=["mae", "rmse"],
        metric_directions={"mae": "asc", "rmse": "asc"},
    )

    uc = SimpleNamespace(execute=lambda _req: result)
    monkeypatch.setattr(compare_view, "_compare_models_uc", lambda: uc)
    monkeypatch.setattr(
        compare_view.ComparisonPresenter,
        "format_leaderboard_frame",
        staticmethod(lambda _result, *, label_lookup: pd.DataFrame([{"rank": 1, "model": label_lookup["ridge"]}])),
    )

    compare_view.render()

    assert ("subheader", "Leaderboard") in events
    assert any(event[0] == "dataframe" for event in events)
    assert any(event[0] == "caption" and "Ranking priority:" in str(event[1]) for event in events)
    assert not [event for event in events if event[0] in {"warning", "error"}]


def test_compare_render_shows_error_when_use_case_raises_value_error(monkeypatch) -> None:
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(compare_view, "_SETTINGS", SimpleNamespace(model_defaults=SimpleNamespace(
        training_model_ids=lambda: ("ridge",),
        id_to_label=lambda: {"ridge": "Ridge"},
    )))
    monkeypatch.setattr(compare_view.st, "title", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "markdown", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "info", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "warning", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "error", lambda msg: events.append(("error", msg)))
    monkeypatch.setattr(compare_view.st, "form", lambda _name: _Ctx())
    monkeypatch.setattr(compare_view.st, "multiselect", lambda _label, options, **_kwargs: list(options))
    monkeypatch.setattr(compare_view.st, "form_submit_button", lambda _label: True)

    uc = SimpleNamespace(execute=lambda _req: (_ for _ in ()).throw(ValueError("bad request")))
    monkeypatch.setattr(compare_view, "_compare_models_uc", lambda: uc)

    compare_view.render()

    assert any("Unable to build leaderboard" in msg for kind, msg in events if kind == "error")


def test_compare_render_shows_warning_when_no_training_models(monkeypatch) -> None:
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(compare_view, "_SETTINGS", SimpleNamespace(model_defaults=SimpleNamespace(
        training_model_ids=lambda: (),
        id_to_label=lambda: {},
    )))
    monkeypatch.setattr(compare_view.st, "title", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "markdown", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "info", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "warning", lambda msg: events.append(("warning", msg)))

    compare_view.render()

    assert ("warning", "No training-enabled models are configured.") in events


def test_compare_render_warns_when_presenter_returns_empty_frame(monkeypatch) -> None:
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(compare_view, "_SETTINGS", SimpleNamespace(model_defaults=SimpleNamespace(
        training_model_ids=lambda: ("ridge",),
        id_to_label=lambda: {"ridge": "Ridge"},
    )))
    monkeypatch.setattr(compare_view.st, "title", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "markdown", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "info", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "error", lambda _msg: None)
    monkeypatch.setattr(compare_view.st, "warning", lambda msg: events.append(("warning", msg)))
    monkeypatch.setattr(compare_view.st, "form", lambda _name: _Ctx())
    monkeypatch.setattr(compare_view.st, "multiselect", lambda _label, options, **_kwargs: list(options))
    monkeypatch.setattr(compare_view.st, "form_submit_button", lambda _label: True)

    result = CompareModelsResult(rows=[], rank_by=["mae"], metric_directions={"mae": "asc"})
    monkeypatch.setattr(compare_view, "_compare_models_uc", lambda: SimpleNamespace(execute=lambda _req: result))
    monkeypatch.setattr(
        compare_view.ComparisonPresenter,
        "format_leaderboard_frame",
        staticmethod(lambda _result, *, label_lookup: pd.DataFrame()),
    )

    compare_view.render()

    assert ("warning", "No comparison rows were returned.") in events


def test_predict_render_forecast_warns_when_predictions_are_empty(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    result = ForecastResult(model_id="ridge", ticker="AAPL", horizon_days=3, predictions=[])

    monkeypatch.setattr(
        predict_view.ForecastPresenter,
        "format_predictions_table",
        staticmethod(lambda _result: pd.DataFrame()),
    )
    monkeypatch.setattr(
        predict_view.ForecastPresenter,
        "format_price_chart_data",
        staticmethod(lambda _result: None),
    )
    monkeypatch.setattr(predict_view.st, "warning", lambda msg: events.append(("warning", msg)))
    monkeypatch.setattr(predict_view.st, "subheader", lambda msg: events.append(("subheader", msg)))
    monkeypatch.setattr(predict_view.st, "dataframe", lambda *_args, **_kwargs: events.append(("dataframe", None)))
    monkeypatch.setattr(predict_view.st, "line_chart", lambda *_args, **_kwargs: events.append(("line_chart", None)))

    predict_view._render_forecast(result)

    assert ("warning", "No forecast rows were returned.") in events
    assert not any(kind == "dataframe" for kind, _ in events)
    assert not any(kind == "line_chart" for kind, _ in events)


def test_predict_render_forecast_renders_table_and_chart(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    result = ForecastResult(model_id="ridge", ticker="AAPL", horizon_days=3, predictions=[])

    table_df = pd.DataFrame([{"date": "2026-04-15", "pred_close": 101.0}])
    chart_df = pd.DataFrame({"pred_close": [101.0]}, index=pd.to_datetime(["2026-04-15"]))

    monkeypatch.setattr(
        predict_view.ForecastPresenter,
        "format_predictions_table",
        staticmethod(lambda _result: table_df),
    )
    monkeypatch.setattr(
        predict_view.ForecastPresenter,
        "format_price_chart_data",
        staticmethod(lambda _result: chart_df),
    )
    monkeypatch.setattr(predict_view.st, "warning", lambda msg: events.append(("warning", msg)))
    monkeypatch.setattr(predict_view.st, "subheader", lambda msg: events.append(("subheader", msg)))
    monkeypatch.setattr(predict_view.st, "dataframe", lambda frame, **_kwargs: events.append(("dataframe", frame)))
    monkeypatch.setattr(predict_view.st, "line_chart", lambda series, **_kwargs: events.append(("line_chart", series)))

    predict_view._render_forecast(result)

    assert ("subheader", "Forecast Results") in events
    assert ("subheader", "Predicted Close Price") in events
    assert any(kind == "dataframe" for kind, _ in events)
    assert any(kind == "line_chart" for kind, _ in events)
    assert not any(kind == "warning" for kind, _ in events)


def test_predict_render_warns_when_no_prediction_models(monkeypatch) -> None:
    events: list[tuple[str, str]] = []

    model_defaults = SimpleNamespace(
        id_to_label=lambda: {},
        prediction_model_ids=lambda: (),
        default_model_id="ridge",
        horizon_min=1,
        horizon_max=30,
        default_horizon=7,
    )
    monkeypatch.setattr(
        predict_view,
        "_SETTINGS",
        SimpleNamespace(
            ticker_catalog=SimpleNamespace(entries=()),
            model_defaults=model_defaults,
        ),
    )

    monkeypatch.setattr(predict_view, "build_ticker_select_items", lambda _entries: [("AAPL", "AAPL")])
    monkeypatch.setattr(predict_view.st, "title", lambda _msg: None)
    monkeypatch.setattr(predict_view.st, "markdown", lambda _msg: None)
    monkeypatch.setattr(predict_view.st, "subheader", lambda _msg: None)
    monkeypatch.setattr(predict_view.st, "warning", lambda msg: events.append(("warning", msg)))
    monkeypatch.setattr(predict_view.st, "selectbox", lambda _label, options, **_kwargs: options[0])
    monkeypatch.setattr(predict_view.st, "slider", lambda _label, **_kwargs: 7)

    first_cols = (_ButtonCol(), _ButtonCol())
    second_cols = (_ButtonCol({"Fetch Historical Data": False}), _ButtonCol({"Run Prediction": False}))
    calls = {"n": 0}

    def _columns(_n: int):
        calls["n"] += 1
        return first_cols if calls["n"] == 1 else second_cols

    monkeypatch.setattr(predict_view.st, "columns", _columns)

    predict_view.render()

    assert ("warning", "No prediction-enabled models are configured.") in events


def test_predict_render_executes_forecast_use_case_and_renders_result(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    model_defaults = SimpleNamespace(
        id_to_label=lambda: {"ridge": "Ridge"},
        prediction_model_ids=lambda: ("ridge",),
        default_model_id="ridge",
        horizon_min=1,
        horizon_max=30,
        default_horizon=7,
    )
    monkeypatch.setattr(
        predict_view,
        "_SETTINGS",
        SimpleNamespace(
            ticker_catalog=SimpleNamespace(entries=()),
            model_defaults=model_defaults,
        ),
    )

    monkeypatch.setattr(predict_view, "build_ticker_select_items", lambda _entries: [("AAPL", "AAPL")])
    monkeypatch.setattr(predict_view.st, "title", lambda _msg: None)
    monkeypatch.setattr(predict_view.st, "markdown", lambda _msg: None)
    monkeypatch.setattr(predict_view.st, "subheader", lambda _msg: None)
    monkeypatch.setattr(predict_view.st, "warning", lambda _msg: None)
    monkeypatch.setattr(predict_view.st, "error", lambda msg: events.append(("error", msg)))
    monkeypatch.setattr(predict_view.st, "selectbox", lambda _label, options, **_kwargs: options[0])
    monkeypatch.setattr(predict_view.st, "slider", lambda _label, **_kwargs: 5)

    first_cols = (_ButtonCol(), _ButtonCol())
    second_cols = (_ButtonCol({"Fetch Historical Data": False}), _ButtonCol({"Run Prediction": True}))
    calls = {"n": 0}

    def _columns(_n: int):
        calls["n"] += 1
        return first_cols if calls["n"] == 1 else second_cols

    monkeypatch.setattr(predict_view.st, "columns", _columns)

    forecast_result = ForecastResult(model_id="ridge", ticker="AAPL", horizon_days=5, predictions=[])
    monkeypatch.setattr(
        predict_view,
        "_forecast_uc",
        lambda: SimpleNamespace(execute=lambda _req: forecast_result),
    )
    monkeypatch.setattr(
        predict_view,
        "_render_forecast",
        lambda result: events.append(("render_forecast", result)),
    )

    predict_view.render()

    assert any(kind == "render_forecast" for kind, _ in events)
    assert not any(kind == "error" for kind, _ in events)


