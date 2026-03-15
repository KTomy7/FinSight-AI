import matplotlib.pyplot as plt
import streamlit as st

from finsight.application.use_cases.fetch_market_data import (
    FetchMarketData,
    FetchMarketDataRequest,
)
from finsight.infrastructure.market_data.yfinance_provider import YFinanceMarketDataProvider


@st.cache_resource
def _fetch_market_data_uc() -> FetchMarketData:
    # Cache provider + use-case across Streamlit reruns.
    return FetchMarketData(
        YFinanceMarketDataProvider(),
        default_period="5y",
        default_interval="1d",
    )


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

    with col2:
        model_choice = st.selectbox(
            "Choose a prediction model",
            ["Linear Regression", "Random Forest", "LSTM (Neural Network)"],
        )

    horizon = st.slider(
        "Prediction horizon (in days)",
        min_value=7,
        max_value=90,
        value=30,
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

