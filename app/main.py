import streamlit as st
from ui import landing, predictor, compare_models, layout

def main():
    st.set_page_config(
        page_title="FinSight AI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    selected = layout.render_sidebar()
    if selected == "Home":
        landing.render()
    elif selected == "Predict":
        predictor.render()
    elif selected == "Compare Models":
        compare_models.render()

if __name__ == "__main__":
    main()
