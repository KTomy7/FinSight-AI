import streamlit as st
from ui import *

def main():
    st.set_page_config(
        page_title="FinSight AI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    selected = render_layout()
    if selected == "Home":
        render_landing()
    elif selected == "Predict":
        render_predictor()
    elif selected == "Compare Models":
        render_comparison()

if __name__ == "__main__":
    main()
