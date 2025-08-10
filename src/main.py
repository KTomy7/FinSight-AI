import streamlit as st
from finsight.ui import render_layout, PAGE_HANDLERS

def configure_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="FinSight AI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def main():
    """Main application entry point."""
    configure_page()

    try:
        selected = render_layout()

        handler = PAGE_HANDLERS.get(selected)
        if handler:
            handler()
        else:
            st.error(f"Page '{selected}' not found.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
if __name__ == "__main__":
    main()
