from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd
import numpy as np

from finsight.application.use_cases.fetch_market_data import FetchMarketData, FetchMarketDataRequest
from finsight.infrastructure.market_data.yfinance_provider import YFinanceMarketDataProvider
from finsight.infrastructure.features.feature_pipeline import (
    FEATURE_COLUMNS,
    REQUIRED_PANEL_COLUMNS,
    TARGET_COLUMNS,
    add_features,
    add_target,
    build_feature_dataset,
    to_panel_df,
)


def _pct_dropped(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return 100.0 * (before - after) / before


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test: fetch real OHLCV and build features.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT"],
        help="Tickers to fetch (default: AAPL MSFT).",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Lookback window in years (default: 2).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help='Optional end date ISO "YYYY-MM-DD" (default: today).',
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Interval (default: 1d).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot a couple of feature columns using matplotlib.",
    )
    args = parser.parse_args()

    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    start = end - dt.timedelta(days=round(args.years * 365.25))

    provider = YFinanceMarketDataProvider()
    fetch = FetchMarketData(market_data=provider, default_interval=args.interval)

    series_list = []
    for t in args.tickers:
        res = fetch.execute(
            FetchMarketDataRequest(
                ticker=t,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                interval=args.interval,
                include_summary=False,
            )
        )
        series_list.append(res.history)

    # 1) Build raw panel (normalized OHLCV)
    panel = to_panel_df(series_list)

    # Basic schema checks
    missing_panel_cols = [c for c in REQUIRED_PANEL_COLUMNS if c not in panel.columns]
    if missing_panel_cols:
        raise RuntimeError(f"Panel is missing required columns: {missing_panel_cols}")

    print("\n=== Raw panel ===")
    print("Rows:", len(panel))
    print("Tickers:", sorted(panel["ticker"].unique().tolist()))
    print("Date range:", panel["date"].min(), "→", panel["date"].max())
    print("\nHead:")
    print(panel.head(5))
    print("\nTail:")
    print(panel.tail(5))

    # 2) Build features (full pipeline)
    features = build_feature_dataset(series_list)

    print("\n=== Feature dataset ===")
    print("Rows:", len(features))
    print("Columns:", len(features.columns))
    print("Dropped % (panel -> features):", f"{_pct_dropped(len(panel), len(features)):.2f}%")

    missing_feature_cols = [c for c in ["date", "ticker", *FEATURE_COLUMNS, *TARGET_COLUMNS] if c not in features.columns]
    if missing_feature_cols:
        raise RuntimeError(f"Feature dataset missing expected columns: {missing_feature_cols}")

    # No NaNs expected after finalize(dropna=True)
    nan_counts = features[[*FEATURE_COLUMNS, *TARGET_COLUMNS]].isna().sum().sum()
    print("NaN count in features/target:", int(nan_counts))
    if nan_counts != 0:
        raise RuntimeError("Found NaNs in finalized feature dataset (should be 0).")

    # 1) Uniqueness: (ticker, date) should be unique
    dup_count = int(features.duplicated(subset=["ticker", "date"]).sum())
    print("\n=== Extra checks ===")
    print("Duplicate (ticker,date) rows:", dup_count)
    if dup_count != 0:
        raise RuntimeError(f"Found {dup_count} duplicate (ticker,date) rows in feature dataset.")

    # 2) Monotonic increasing dates per ticker (after sorting)
    # (If you always sort in finalize_feature_frame, this should pass.)
    non_mono = []
    for t, g in features.groupby("ticker", sort=False):
        # ensure strictly increasing (no equals) since we require unique dates
        if not g["date"].is_monotonic_increasing:
            non_mono.append(t)

    print("Tickers with non-monotonic dates:", non_mono)
    if non_mono:
        raise RuntimeError(f"Non-monotonic date order for tickers: {non_mono}")

    # 3) Finite values: no inf/-inf in features/target
    num_cols = [*FEATURE_COLUMNS, *TARGET_COLUMNS]
    finite_mask = np.isfinite(features[num_cols].to_numpy(dtype=float))
    all_finite = bool(finite_mask.all())
    print("All finite (no inf/-inf) in features/target:", all_finite)
    if not all_finite:
        # Identify offending columns for easier debugging
        bad_cols = []
        for c in num_cols:
            col_ok = np.isfinite(features[c].to_numpy(dtype=float)).all()
            if not col_ok:
                bad_cols.append(c)
        raise RuntimeError(f"Found non-finite values (inf/-inf) in columns: {bad_cols}")

    # Sanity ranges (soft checks; print warnings rather than failing hard)
    print("\n=== Sanity stats (overall) ===")
    print("ret_1d min/max:", float(features["ret_1d"].min()), float(features["ret_1d"].max()))
    print("vol_20d min/max:", float(features["vol_20d"].min()), float(features["vol_20d"].max()))
    print("volume_z20 mean/std:", float(features["volume_z20"].mean()), float(features["volume_z20"].std()))

    # Per-ticker quick view
    print("\n=== Per-ticker row counts ===")
    print(features.groupby("ticker")["date"].count())

    print("\nHead:")
    print(features.head(5))
    print("\nTail:")
    print(features.tail(5))

    # 3) Verify last input day is not present as a labeled row per ticker (because target needs t+1)
    # Do this by comparing each ticker's max date in raw panel vs max date in finalized dataset.
    print("\n=== Target alignment check (last day dropped) ===")
    raw_max = panel.groupby("ticker")["date"].max()
    feat_max = features.groupby("ticker")["date"].max()
    for t in sorted(panel["ticker"].unique().tolist()):
        print(f"{t}: raw max={raw_max[t].date()}  features max={feat_max[t].date()}")

    # 4) Optional plotting
    if args.plot:
        import matplotlib.pyplot as plt

        # Plot ret_1d and target_ret_1d for the first ticker
        first_ticker = sorted(features["ticker"].unique().tolist())[0]
        sub = features[features["ticker"] == first_ticker].copy()

        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(sub["date"], sub["ret_1d"], label="ret_1d")
        ax[0].set_title(f"{first_ticker} ret_1d")
        ax[0].grid(True)

        ax[1].plot(sub["date"], sub["target_ret_1d"], label="target_ret_1d", color="orange")
        ax[1].set_title(f"{first_ticker} target_ret_1d")
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Helps with wide DataFrame printing
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    main()
