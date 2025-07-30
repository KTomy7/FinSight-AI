import streamlit as st

def render():
    # App title
    st.title("📈 FinSight AI – Stock Market Prediction using AI")
    st.markdown("""
        Welcome to FinSight AI, your personal stock analysis dashboard powered by Streamlit.
        Use the sidebar to enter a stock symbol and select a date range.
        Once ready, click **Run Prediction** to analyze historical data and predict future trends.
        """)
