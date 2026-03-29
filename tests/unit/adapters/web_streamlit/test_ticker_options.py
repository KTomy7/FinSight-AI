from finsight.adapters.web_streamlit.ticker_options import build_ticker_select_items
from finsight.config.settings import TickerCatalogEntry


def test_build_ticker_select_items_formats_label_and_keeps_symbol() -> None:
    entries = (
        TickerCatalogEntry(symbol="AAPL", company_name="Apple Inc."),
        TickerCatalogEntry(symbol="JPM", company_name="JPMorgan Chase & Co."),
    )

    select_items = build_ticker_select_items(entries)

    assert select_items == [
        ("AAPL", "AAPL - Apple Inc."),
        ("JPM", "JPM - JPMorgan Chase & Co."),
    ]


def test_build_ticker_select_items_preserves_catalog_order() -> None:
    entries = (
        TickerCatalogEntry(symbol="TSLA", company_name="Tesla, Inc."),
        TickerCatalogEntry(symbol="KO", company_name="The Coca-Cola Company"),
    )

    select_items = build_ticker_select_items(entries)

    assert [symbol for symbol, _label in select_items] == ["TSLA", "KO"]

