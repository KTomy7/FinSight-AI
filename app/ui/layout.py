import streamlit as st
from datetime import date

def render_sidebar():
    """Create the sidebar layout for the app."""

    st.sidebar.header("📂 Stock Selection")

    # Stock ticker input
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")

    # Date range input
    start_date = st.sidebar.date_input("Start Date", value=date(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=date.today())

    # Predict button
    predict_button = st.sidebar.button("Run Prediction")

    return {
        "ticker": ticker.upper(),
        "start_date": start_date,
        "end_date": end_date,
        "predict_button": predict_button
    }

def render_main_ui(user_input):
    """Render the main UI components of the app."""

    # App title
    st.title("📈 FinSight AI – Stock Market Prediction using AI")
    st.markdown("""
    Welcome to FinSight AI, your personal stock analysis dashboard powered by Streamlit.
    Use the sidebar to enter a stock symbol and select a date range.
    Once ready, click **Run Prediction** to analyze historical data and predict future trends.
    """)

    # Display selected inputs
    st.subheader("🔍 Stock Summary")
    st.write(f"**Ticker**: `{user_input['ticker']}`")
    st.write(f"**Date Range**: `{user_input['start_date']}` to `{user_input['end_date']}`")

    # Placeholder for data preview and charts
    st.subheader("📊 Stock Data Preview")
    st.info("Stock data will be shown here after you implement the fetching logic.")

    if user_input["predict_button"]:
        st.info(
            f"Fetching data for `{user_input['ticker']}` from {user_input['start_date']} to {user_input['end_date']}...")
        # You’ll call data fetching + prediction functions here later
