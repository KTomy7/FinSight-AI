from pathlib import Path

import streamlit as st

# .../finsight/adapters/web_streamlit/views/home.py -> parents[3] == .../finsight
ASSETS_PATH = Path(__file__).resolve().parents[3] / "assets"


def render():
    st.title("FinSight AI - Stock Market Prediction using AI")

    st.markdown("## Welcome")
    st.markdown(
        """
        **FinSight AI** is a smart, AI-powered stock market analysis tool built to help you gain insight into market movements,
        evaluate model performance, and manage your personal stock portfolio - all from a clean, intuitive interface.
        """
    )

    banner_path = ASSETS_PATH / "banner.jpg"
    if banner_path.exists():
        st.image(str(banner_path), use_container_width=True)
    else:
        st.warning(f"Banner image not found. Please ensure 'banner.jpg' is present at: {banner_path}")

    st.markdown("---")
    st.markdown("### What can you do with FinSight AI?")
    st.markdown(
        """
        - **Stock Price Prediction**  
          Use multiple machine learning models to forecast future stock prices based on historical trends.

        - **Model Comparison**  
          Compare different AI model predictions with real market data to identify which model performs best over time.

        - **Interactive Visualizations**  
          Analyze market patterns and predictions using intuitive charts.

        - **Real-Time Data Integration**  
          Fetch up-to-date stock data using Yahoo Finance and visualize trends with just a few clicks.

        - **Clean Web Interface**  
          Navigate between tools using a modern sidebar and responsive layout.

        - **User Portfolio Tracking**  
          Track a sample portfolio with visual feedback on gains/losses and projected returns.
        """
    )

    st.markdown("---")
    st.markdown("### Technologies Behind the Scenes")
    st.markdown(
        """
        - **Python** with **Streamlit**
        - **Machine Learning**: scikit-learn, pandas, yfinance
        - **Data Visualization**: matplotlib, seaborn, Altair
        - **APIs**: Yahoo Finance
        """
    )

    st.markdown("---")
    st.info("Tip: Head over to the *Predict* tab to start analyzing stock performance!")

