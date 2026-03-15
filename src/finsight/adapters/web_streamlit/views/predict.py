import matplotlib.pyplot as plt
import streamlit as st

from finsight.application.use_cases.fetch_market_data import (
    FetchMarketData,
    FetchMarketDataRequest,
)
from finsight.config.settings import get_settings
from finsight.bootstrap.container import build_container


_SETTINGS = get_settings()


@st.cache_resource(ttl=_SETTINGS.cache.resource_ttl_seconds)
def _fetch_market_data_uc() -> FetchMarketData:
    # Use the composition root so adapters do not wire concrete infra directly.
    return build_container().fetch_market_data


def render():
    st.title("Stock Price Predictor")
    st.markdown(
        "Use artificial intelligence models - including machine learning and deep learning - "
        "to forecast future stock prices based on historical market data."
    )

    st.subheader("Select Your Inputs")

    col1, col2 = st.columns(2)

    with col1:
        ticker = st.selectbox("Choose a stock ticker", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NFLX"])

    model_defaults = _SETTINGS.model_defaults

    # TODO: model_choice is not used
    with col2:
        model_choice = st.selectbox(
            "Choose a prediction model",
            list(model_defaults.options),
            index=list(model_defaults.options).index(model_defaults.default_model),
        )

    # TODO: horizon is not used
    horizon = st.slider(
        "Prediction horizon (in days)",
        min_value=model_defaults.horizon_min,
        max_value=model_defaults.horizon_max,
        value=model_defaults.default_horizon,
        help="How far into the future you want the model to forecast stock prices.",
    )

    predict_button = st.button("Run Prediction")

    if predict_button:
        try:
            st.write(
                f"Fetching data for ticker: **{ticker}** using model: **{model_choice}** for the next {horizon} days..."
            )

            result = _fetch_market_data_uc().execute(
                FetchMarketDataRequest(ticker=ticker, include_summary=True)
            )
            df = result.history.df
            summary = result.summary_dict or {}

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

        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
