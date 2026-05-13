import streamlit as st
from streamlit_option_menu import option_menu


def render_sidebar():
    with st.sidebar:
        selected = option_menu(
            None,
            ["Home", "Predict", "Compare Models", "Train & Backtest"],
            icons=["house", "graph-up", "bar-chart", "activity"],
            menu_icon="cast",
            default_index=0,
        )
    return selected

