import streamlit as st

from finsight.application.dto import ForecastRequest, ForecastResult
from finsight.application.use_cases.forecast import Forecast
from finsight.adapters.web_streamlit.presenters import ForecastPresenter
from finsight.adapters.web_streamlit.ticker_options import build_ticker_select_items
from finsight.bootstrap.container import build_container
from finsight.config.settings import get_settings


_SETTINGS = get_settings()


@st.cache_resource(ttl=_SETTINGS.cache.resource_ttl_seconds)
def _forecast_uc() -> Forecast:
    return build_container().forecast


def _render_forecast(result: ForecastResult) -> None:
    """Render forecast results using presenter formatting."""
    predictions_frame = ForecastPresenter.format_predictions_table(result)
    if predictions_frame.empty:
        st.warning("No forecast rows were returned.")
        return

    st.subheader("Forecast Results")
    st.dataframe(predictions_frame, width='stretch')

    chart_df = ForecastPresenter.format_price_chart_data(result)
    if chart_df is not None:
        st.subheader("Predicted Close Price")
        st.line_chart(chart_df["pred_close"])


def render():
    st.title("Price Prediction")
    st.markdown(
        "Select a ticker and a trained model, then generate a forward price forecast. "
        "The latest market data is loaded automatically behind the scenes."
    )

    model_defaults = _SETTINGS.model_defaults
    id_to_label = model_defaults.id_to_label()
    prediction_model_ids = list(model_defaults.prediction_model_ids())
    has_prediction_models = bool(prediction_model_ids)

    if model_defaults.default_model_id in prediction_model_ids:
        default_model_id = model_defaults.default_model_id
    elif prediction_model_ids:
        default_model_id = prediction_model_ids[0]
    else:
        default_model_id = None

    selected_model_id: str | None = None

    if not has_prediction_models:
        st.warning("No prediction-enabled models are configured.")

    ticker_items = build_ticker_select_items(_SETTINGS.ticker_catalog.entries)
    ticker_symbols = [symbol for symbol, _label in ticker_items]
    ticker_label_lookup = {symbol: label for symbol, label in ticker_items}

    with st.form("price_prediction_form"):
        st.subheader("Forecast inputs")
        ticker = st.selectbox(
            "Choose a stock ticker",
            ticker_symbols,
            format_func=lambda symbol: ticker_label_lookup.get(symbol, symbol),
        )

        if has_prediction_models:
            default_selection = default_model_id if default_model_id is not None else prediction_model_ids[0]
            selected_index = prediction_model_ids.index(default_selection)
            selected_model_id = st.selectbox(
                "Choose a prediction model",
                prediction_model_ids,
                index=selected_index,
                format_func=lambda model_id: id_to_label.get(model_id, model_id),
            )
        else:
            selected_model_id = None
            st.selectbox(
                "Choose a prediction model",
                ["No prediction-enabled models configured"],
                index=0,
                disabled=True,
            )

        horizon = st.slider(
            "Prediction horizon (in days)",
            min_value=model_defaults.horizon_min,
            max_value=model_defaults.horizon_max,
            value=model_defaults.default_horizon,
            help="How far into the future you want the model to forecast stock prices.",
        )

        predict_button = st.form_submit_button(
            "Run Forecast",
            disabled=not has_prediction_models,
        )

    if not predict_button:
        st.info("Choose a ticker and model, then run the forecast.")
        return

    if selected_model_id is None:
        st.warning("Select a prediction model.")
        return

    try:
        forecast_result = _forecast_uc().execute(
            ForecastRequest(
                ticker=ticker,
                model_id=selected_model_id,
                horizon_days=horizon,
            )
        )
        _render_forecast(forecast_result)
    except FileNotFoundError as error:
        st.error(f"No trained run artifacts were found: {error}")
    except (ValueError, TypeError) as error:
        st.error(f"Unable to run forecast: {error}")
    except Exception as error:  # pragma: no cover - defensive fallback for UI resilience
        st.error(f"Forecast failed unexpectedly: {error}")
