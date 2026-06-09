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
        evaluate model performance, and predict - all from a clean, intuitive interface.
        """
    )

    #banner_path = ASSETS_PATH / "banner.jpg"
    #if banner_path.exists():
    #    st.image(str(banner_path), use_container_width=True)
    #else:
    #    st.warning(f"Banner image not found. Please ensure 'banner.jpg' is present at: {banner_path}")

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

        """
    )

    st.markdown("---")
    st.markdown("### Technologies Behind the Scenes")
    st.markdown(
        """
        - **App Framework**: Python with **Streamlit**
        - **Data Processing**: `pandas`, `numpy`
        - **Machine Learning**: `scikit-learn`, `xgboost`
        - **Model Persistence**: `joblib`
        - **Configuration**: `PyYAML`
        - **Market Data**: Yahoo Finance via `yfinance`
        """
    )

    st.markdown("---")
