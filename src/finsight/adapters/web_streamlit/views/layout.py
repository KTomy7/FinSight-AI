import streamlit as st
from streamlit_option_menu import option_menu


def render_sidebar():
    with st.sidebar:
        selected = option_menu(
            None,
            ["Home", "Predict", "Backtest", "Train Model", "Compare Models"],
            icons=["house", "graph-up", "calendar3", "activity", "bar-chart"],
            menu_icon="cast",
            default_index=0,
        )
    return selected

