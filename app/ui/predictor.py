import streamlit as st

def render():
    st.title("Stock Price Predictor")
    st.markdown("Use artificial intelligence models—including machine learning and deep learning—to forecast future stock prices based on historical market data.")

    st.subheader("🔍 Select Your Inputs")

    col1, col2 = st.columns(2)

    # Step 1: Select a stock ticker
    with col1:
        ticker = st.selectbox("Choose a stock ticker", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NFLX"])

    # Step 2: Select prediction model
    with col2:
        model_choice = st.selectbox(
            "Choose a prediction model",
            ["Linear Regression", "Random Forest", "LSTM (Neural Network)"]
        )

    # Step 3: Select prediction horizon
    horizon = st.slider(
        "Prediction horizon (in days)",
        min_value=7,
        max_value=90,
        value=30,
        help="How far into the future you want the model to forecast stock prices."
    )

    # Step 4: Prediction trigger
    predict_button = st.button("🚀 Run Prediction")

    # Placeholder for results
    if predict_button:
        st.info("Prediction is running... (logic to be added)")
        # In the future: call run_prediction(ticker, model_choice, horizon)
