from __future__ import annotations

from finsight.config.settings import TickerCatalogEntry


def build_ticker_select_items(entries: tuple[TickerCatalogEntry, ...]) -> list[tuple[str, str]]:
    """Build ordered (symbol, label) pairs for ticker select controls."""
    return [(entry.symbol, f"{entry.symbol} - {entry.company_name}") for entry in entries]

