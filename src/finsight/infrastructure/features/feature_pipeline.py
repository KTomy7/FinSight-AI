from __future__ import annotations

import pandas as pd

from finsight.domain.entities import OHLCVSeries

REQUIRED_PANEL_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
FEATURE_COLUMNS = [
    "ret_1d",
    "ret_2d",
    "ret_5d",
    "ret_10d",
    "mom_20d",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "sma_5",
    "sma_20",
    "sma_50",
    "close_sma20_ratio",
    "close_sma50_ratio",
    "vol_chg_1d",
    "volume_z20",
]
TARGET_COLUMNS = ["target_ret_1d"]


def _normalize_ohlcv_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    working = df.copy()

    # yfinance can return Date as an index for daily data.
    if "Date" not in working.columns and str(working.index.name).lower() == "date":
        working = working.reset_index()

    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    working = working.rename(columns=rename_map)

    missing = [col for col in REQUIRED_PANEL_COLUMNS if col != "ticker" and col not in working.columns]
    if missing:
        raise ValueError(f"OHLCV frame for {ticker} is missing columns: {missing}")

    working["ticker"] = str(ticker)
    normalized = working[REQUIRED_PANEL_COLUMNS].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    if normalized["date"].isna().any():
        raise ValueError(f"OHLCV frame for {ticker} contains invalid date values.")
    return normalized


def to_panel_df(series_list: list[OHLCVSeries]) -> pd.DataFrame:
    if not series_list:
        return pd.DataFrame(columns=REQUIRED_PANEL_COLUMNS)

    frames = [_normalize_ohlcv_frame(series.df, str(series.ticker)) for series in series_list]
    panel = pd.concat(frames, ignore_index=True)

    panel = panel.drop_duplicates(subset=["ticker", "date"], keep="last")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std()


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    by_ticker_close = df.groupby("ticker", sort=False)["close"]
    by_ticker_volume = df.groupby("ticker", sort=False)["volume"]

    df["ret_1d"] = by_ticker_close.pct_change(1)
    df["ret_2d"] = by_ticker_close.pct_change(2)
    df["ret_5d"] = by_ticker_close.pct_change(5)
    df["ret_10d"] = by_ticker_close.pct_change(10)
    df["mom_20d"] = by_ticker_close.pct_change(20)

    df["vol_5d"] = df.groupby("ticker", sort=False)["ret_1d"].transform(_rolling_std, window=5)
    df["vol_10d"] = df.groupby("ticker", sort=False)["ret_1d"].transform(_rolling_std, window=10)
    df["vol_20d"] = df.groupby("ticker", sort=False)["ret_1d"].transform(_rolling_std, window=20)

    df["sma_5"] = by_ticker_close.transform(_rolling_mean, window=5)
    df["sma_20"] = by_ticker_close.transform(_rolling_mean, window=20)
    df["sma_50"] = by_ticker_close.transform(_rolling_mean, window=50)

    df["close_sma20_ratio"] = (df["close"] / df["sma_20"]) - 1.0
    df["close_sma50_ratio"] = (df["close"] / df["sma_50"]) - 1.0

    df["vol_chg_1d"] = by_ticker_volume.pct_change(1)

    vol_mean_20 = by_ticker_volume.transform(_rolling_mean, window=20)
    vol_std_20 = by_ticker_volume.transform(_rolling_std, window=20)
    df["volume_z20"] = (df["volume"] - vol_mean_20) / vol_std_20

    return df


def add_target(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    next_close = df.groupby("ticker", sort=False)["close"].shift(-1)
    df["target_ret_1d"] = (next_close / df["close"]) - 1.0
    return df


def finalize_feature_frame(df: pd.DataFrame, *, dropna: bool = True) -> pd.DataFrame:
    ordered_columns = ["date", "ticker", *FEATURE_COLUMNS, *TARGET_COLUMNS]
    missing = [col for col in ordered_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Feature frame is missing expected columns: {missing}")

    final_df = df[ordered_columns].copy()
    if dropna:
        final_df = final_df.dropna(subset=[*FEATURE_COLUMNS, *TARGET_COLUMNS])

    final_df = final_df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    )
    return final_df.reset_index(drop=True)


def build_feature_dataset(series_list: list[OHLCVSeries]) -> pd.DataFrame:
    panel = to_panel_df(series_list)
    panel = add_features(panel)
    panel = add_target(panel)
    return finalize_feature_frame(panel)

