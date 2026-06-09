import streamlit as st
from streamlit_option_menu import option_menu


def render_sidebar():
    with st.sidebar:
        selected = option_menu(
            None,
            ["Home", "Market Data", "Predict", "Backtest", "Train Model", "Compare Models"],
            icons=["house", "bar-chart", "graph-up", "calendar3", "activity", "clipboard-data"],
            menu_icon="cast",
            default_index=0,
        )
    return selected

