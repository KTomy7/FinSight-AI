"""
Market Data visualization view.

Displays current market data and summary information for selected ticker(s).
Fetches real-time data from yfinance via the FetchMarketData use case.
"""
import streamlit as st

from finsight.application.dto import FetchMarketDataRequest
from finsight.adapters.web_streamlit.presenters import MarketDataPresenter
from finsight.adapters.web_streamlit.ticker_options import build_ticker_select_items
from finsight.bootstrap.container import build_container
from finsight.config.settings import get_settings


_SETTINGS = get_settings()


def render() -> None:
    st.title("📈 Market Data Analysis")
    st.markdown(
        "Dive deep into market data for any ticker. Select a stock to view current metrics, "
        "historical trends, and detailed financial visualization powered by live yfinance data."
    )

    # Get ticker catalog and build display options
    ticker_items = build_ticker_select_items(_SETTINGS.ticker_catalog.entries)
    ticker_symbols = [symbol for symbol, _label in ticker_items]
    ticker_label_lookup = {symbol: label for symbol, label in ticker_items}

    # Form for ticker selection and fetch trigger
    with st.form("market_data_form"):
        st.subheader("Select Ticker")

        selected_ticker = st.selectbox(
            "Choose a stock ticker",
            ticker_symbols,
            format_func=lambda symbol: ticker_label_lookup.get(symbol, symbol),
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            fetch_button = st.form_submit_button("📊 Load Data", use_container_width=True)

    if not fetch_button:
        st.info("👉 Select a ticker and click 'Load Data' to begin your analysis.")
        return

    # Fetch and display market data on-demand
    try:
        container = build_container()

        # Fetch market data with summary
        with st.spinner(f"📡 Loading market data for {selected_ticker}..."):
            result = container.fetch_market_data.execute(
                FetchMarketDataRequest(
                    ticker=selected_ticker,
                    include_summary=True,
                )
            )

        # Display ticker name and sector
        st.success(f"✅ Data loaded for {selected_ticker}")

        # === SUMMARY METRICS ===
        if result.summary:
            metrics = MarketDataPresenter.format_summary_metrics(result.summary)

            st.subheader("💰 Price & Valuation Metrics")

            # Display key metrics in columns for better layout
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Current Price",
                    f"${metrics.get('current_price', 'N/A'):.2f}" if isinstance(metrics.get('current_price'), (int, float)) else metrics.get('current_price', 'N/A'),
                    delta="Real-time",
                )

            with col2:
                prev_close = metrics.get('previous_close', 'N/A')
                if isinstance(prev_close, (int, float)) and isinstance(metrics.get('current_price'), (int, float)):
                    change = metrics.get('current_price') - prev_close
                    st.metric(
                        "Previous Close",
                        f"${prev_close:.2f}",
                        delta=f"${change:+.2f}" if not isinstance(change, str) else None,
                    )
                else:
                    st.metric("Previous Close", f"${prev_close:.2f}" if isinstance(prev_close, (int, float)) else prev_close)

            with col3:
                st.metric(
                    "P/E Ratio",
                    f"{metrics.get('pe_ratio', 'N/A'):.2f}" if isinstance(metrics.get('pe_ratio'), (int, float)) else metrics.get('pe_ratio', 'N/A'),
                )

            with col4:
                st.metric(
                    "Dividend Yield",
                    f"{metrics.get('dividend_yield', 'N/A'):.2%}" if isinstance(metrics.get('dividend_yield'), (int, float)) else metrics.get('dividend_yield', 'N/A'),
                )

            # 52-Week Range and Volume
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("52W High", f"${metrics.get('fifty_two_week_high', 'N/A'):.2f}" if isinstance(metrics.get('fifty_two_week_high'), (int, float)) else metrics.get('fifty_two_week_high', 'N/A'))
                st.metric("52W Low", f"${metrics.get('fifty_two_week_low', 'N/A'):.2f}" if isinstance(metrics.get('fifty_two_week_low'), (int, float)) else metrics.get('fifty_two_week_low', 'N/A'))

            with col2:
                st.metric("Market Cap", metrics.get('market_cap', 'N/A'))
                st.metric("Volume", metrics.get('volume', 'N/A'))

            with col3:
                st.metric("Avg Volume", metrics.get('avg_volume', 'N/A'))

            # === COMPANY INFORMATION ===
            st.subheader("🏢 Company Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Company Name:** {metrics.get('name', 'N/A')}")
                st.write(f"**Sector:** {metrics.get('sector', 'N/A')}")
            with col2:
                st.write(f"**Industry:** {metrics.get('industry', 'N/A')}")
        else:
            st.warning("⚠️ No summary data available for this ticker.")

        # === FINANCIAL CHART ===
        st.subheader("📉 Historical Price Chart")

        if result.history and result.history.df is not None and not result.history.df.empty:
            # Prepare data for chart
            chart_df = MarketDataPresenter.format_chart_data(result.history)

            if chart_df is not None and not chart_df.empty:
                # Show OHLC as separate lines for a financial-style trend view
                line_columns = []
                for candidate in ("Open", "High", "Low", "Close"):
                    if candidate in chart_df.columns:
                        line_columns.append(candidate)

                # Handle lowercase fallback if provider/output changes
                if not line_columns:
                    for candidate in ("open", "high", "low", "close"):
                        if candidate in chart_df.columns:
                            line_columns.append(candidate)

                if line_columns:
                    st.line_chart(chart_df[line_columns], use_container_width=True)
                else:
                    st.warning("⚠️ No OHLC columns available for line chart rendering.")

                # Display chart info
                st.caption(
                    f"📊 Historical data from {result.history.date_range.start.isoformat()} "
                    f"to {result.history.date_range.end.isoformat()} "
                    f"({len(chart_df)} trading days)"
                )
            else:
                st.warning("⚠️ No historical price data available for charting.")
        else:
            st.warning("⚠️ No historical price data available for this ticker.")

        # === DETAILED STATISTICS ===
        st.subheader("📊 Historical Data Table")
        if result.history and result.history.df is not None and not result.history.df.empty:
            display_df = result.history.df.copy()

            # Show tabs for different views
            tab1, tab2, tab3 = st.tabs(["Latest 30 Days", "Latest 60 Days", "Full Dataset"])

            with tab1:
                st.dataframe(display_df.tail(30), use_container_width=True, height=400)
                st.caption(f"Showing last 30 rows of {len(display_df)} rows")

            with tab2:
                st.dataframe(display_df.tail(60), use_container_width=True, height=400)
                st.caption(f"Showing last 60 rows of {len(display_df)} rows")

            with tab3:
                st.dataframe(display_df, use_container_width=True, height=400)
                st.caption(f"Full dataset: {len(display_df)} rows total")

            # Summary statistics
            st.subheader("📈 Basic Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                if "Close" in display_df.columns:
                    close_col = "Close"
                else:
                    close_col = next((col for col in display_df.columns if col.lower() == "close"), None)

                if close_col:
                    st.metric("Avg Close Price", f"${display_df[close_col].mean():.2f}")
                    st.metric("Min Close Price", f"${display_df[close_col].min():.2f}")

            with col2:
                if close_col:
                    st.metric("Max Close Price", f"${display_df[close_col].max():.2f}")
                    st.metric("Std Deviation", f"${display_df[close_col].std():.2f}")

            with col3:
                if "Volume" in display_df.columns:
                    st.metric("Avg Volume", f"{display_df['Volume'].mean():,.0f}")
                    st.metric("Total Volume", f"{display_df['Volume'].sum():,.0f}")
        else:
            st.warning("⚠️ No historical data available.")

    except Exception as error:  # pragma: no cover - defensive fallback for UI resilience
        st.error(f"❌ Failed to load market data: {str(error)}")
        st.info("💡 Tip: Make sure the ticker is valid and try again.")


