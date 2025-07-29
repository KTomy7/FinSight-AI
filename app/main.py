import streamlit as st
from ui.layout import render_sidebar, render_main_ui

def main():
    st.set_page_config(
        page_title="FinSight AI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Render sidebar and get user input
    user_input = render_sidebar()
    # Render main content area with user input
    render_main_ui(user_input)

if __name__ == "__main__":
    main()
