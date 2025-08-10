import streamlit as st
from finsight.core.stock_service import StockDataService
import matplotlib.pyplot as plt

def render():
    st.title("Stock Price Predictor")
    st.markdown("Use artificial intelligence models—including machine learning and deep learning—to forecast future stock prices based on historical market data.")

    st.subheader("🔍 Select Your Inputs")

    col1, col2 = st.columns(2)

    with col1:
        ticker = st.selectbox("Choose a stock ticker", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NFLX"])

    with col2:
        model_choice = st.selectbox(
            "Choose a prediction model",
            ["Linear Regression", "Random Forest", "LSTM (Neural Network)"]
        )

    horizon = st.slider(
        "Prediction horizon (in days)",
        min_value=7,
        max_value=90,
        value=30,
        help="How far into the future you want the model to forecast stock prices."
    )

    predict_button = st.button("🚀 Run Prediction")

    if predict_button:
        try:
            st.write(f"Fetching data for ticker: **{ticker}** using model: **{model_choice}** for the next {horizon} days...")
            data_service = StockDataService(ticker=ticker, period="5y", interval="1d")
            df = data_service.get_stock_history()
            summary = data_service.get_summary_info()

            # Display summary
            st.subheader("📊 Stock Information")
            st.write(summary)

            # Show raw data
            st.subheader("🧾 Raw Historical Data")
            st.dataframe(df.tail(10), use_container_width=True)

            # Plot closing prices
            st.subheader("📉 Closing Price Over Time")
            fig, ax = plt.subplots()
            ax.plot(df['Date'], df['Close'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price ($)")
            ax.set_title(f"{ticker} Closing Price")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
