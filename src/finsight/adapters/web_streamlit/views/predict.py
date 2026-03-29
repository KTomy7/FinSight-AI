import matplotlib.pyplot as plt
import streamlit as st
from typing import TYPE_CHECKING

from finsight.application.use_cases.fetch_market_data import (
    FetchMarketData,
    FetchMarketDataRequest,
)
from finsight.config.settings import get_settings
from finsight.bootstrap.container import build_container

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


_SETTINGS = get_settings()


@st.cache_resource(ttl=_SETTINGS.cache.resource_ttl_seconds)
def _fetch_market_data_uc() -> FetchMarketData:
    # Use the composition root so adapters do not wire concrete infra directly.
    return build_container().fetch_market_data


@st.cache_data(ttl=_SETTINGS.cache.data_ttl_seconds)
def _get_market_snapshot(ticker: str) -> tuple["pd.DataFrame", dict[str, object]]:
    result = _fetch_market_data_uc().execute(
        FetchMarketDataRequest(ticker=ticker, include_summary=True)
    )
    return result.history.df, dict(result.summary_dict or {})


def _render_market_data(ticker: str) -> None:
    df, summary = _get_market_snapshot(ticker)

    st.subheader("Stock Information")
    st.write(summary)

    st.subheader("Raw Historical Data")
    st.dataframe(df.tail(10), use_container_width=True)

    st.subheader("Closing Price Over Time")
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price ($)")
    ax.set_title(f"{ticker} Closing Price")
    st.pyplot(fig)


def render():
    st.title("Stock Price Predictor")
    st.markdown(
        "Use artificial intelligence models - including machine learning and deep learning - "
        "to forecast future stock prices based on historical market data."
    )

    st.subheader("Select Your Inputs")

    col1, col2 = st.columns(2)

    with col1:
        ticker = st.selectbox("Choose a stock ticker", list(_SETTINGS.training.training_tickers))

    model_defaults = _SETTINGS.model_defaults
    id_to_label = model_defaults.id_to_label()
    label_to_id = model_defaults.label_to_id()
    model_labels = list(model_defaults.labels())
    default_model_label = id_to_label[model_defaults.default_model_id]

    # TODO: selected_model_id will be passed to the forecast use case.
    with col2:
        selected_label = st.selectbox(
            "Choose a prediction model",
            model_labels,
            index=model_labels.index(default_model_label),
        )
    selected_model_id = label_to_id[selected_label]

    # TODO: horizon will be used by the forecast use case.
    horizon = st.slider(
        "Prediction horizon (in days)",
        min_value=model_defaults.horizon_min,
        max_value=model_defaults.horizon_max,
        value=model_defaults.default_horizon,
        help="How far into the future you want the model to forecast stock prices.",
    )

    fetch_col, predict_col = st.columns(2)
    fetch_data_button = fetch_col.button("Fetch Historical Data", use_container_width=True)
    predict_button = predict_col.button("Run Prediction", use_container_width=True)

    if fetch_data_button:
        try:
            _render_market_data(ticker)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")

    if predict_button:
        st.info(
            f"Prediction flow is not implemented yet (selected model_id='{selected_model_id}')."
        )
