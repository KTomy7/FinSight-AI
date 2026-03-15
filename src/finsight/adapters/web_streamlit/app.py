from __future__ import annotations
import sys
import streamlit as st
from pathlib import Path

# Allow `streamlit run src/finsight/adapters/web_streamlit/app.py` without needing
# to manually set PYTHONPATH=src.
SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finsight.adapters.web_streamlit.views import render_layout, PAGE_HANDLERS

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
