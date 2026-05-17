from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

from finsight.adapters.web_streamlit.presenters import TrainPresenter
from finsight.application.dto import TrainModelRequest, FetchMarketDataRequest
from finsight.bootstrap.container import build_container
from finsight.config.settings import get_settings
from finsight.adapters.web_streamlit.ticker_options import build_ticker_select_items


_SETTINGS = get_settings()
_ARTIFACTS_DIR = "artifacts/runs"
_DEFAULT_YEARS = 2
_DEFAULT_CUTOFF_DAYS_BACK = 365  # Use a date N days back as the default cutoff


def render() -> None:
    st.title("Train Model (Single-Split Evaluation)")
    st.markdown(
        """
        Train selected models on the fixed ticker basket using a single train/test split.
        This page persists trained model artifacts to disk for later use in predictions.
        
        **For robust multi-fold walk-forward evaluation, use the dedicated Backtest page instead.**

        This page uses the configured training tickers and lets you select model types plus which ticker results to display.
        After the run completes the page will display per-model metrics and per-ticker actual vs predicted results.
        """
    )

    model_defaults = _SETTINGS.model_defaults
    training_model_ids = list(model_defaults.training_model_ids())
    model_id_to_label = model_defaults.id_to_label()
    all_tickers = _SETTINGS.ticker_catalog.symbols()

    # Compute default cutoff date (N days back from today)
    default_cutoff = (date.today() - timedelta(days=_DEFAULT_CUTOFF_DAYS_BACK)).isoformat()

    # Create container once (it's cached by lru_cache, so no performance cost)
    container = build_container()

    # Ticker selection (multiselect to filter result display) — outside form for immediate responsiveness
    ticker_items = build_ticker_select_items(_SETTINGS.ticker_catalog.entries)
    ticker_symbols = [symbol for symbol, _label in ticker_items]
    ticker_label_lookup = {symbol: label for symbol, label in ticker_items}
    selected_tickers = st.multiselect(
        "Ticker results to display (select which ticker results to view)",
        ticker_symbols,
        default=ticker_symbols,
        format_func=lambda symbol: ticker_label_lookup.get(symbol, symbol),
    )

    with st.form("train_model_form"):
        st.info(f"Training configuration (fixed):\n- Cutoff date: {default_cutoff}\n- Lookback window: {_DEFAULT_YEARS} years\n- Training tickers: {', '.join(all_tickers)}")

        # Model selection
        selected_models = st.multiselect(
            "Models to train",
            training_model_ids,
            default=training_model_ids,
            format_func=lambda model_id: model_id_to_label.get(model_id, model_id),
        )

        submit = st.form_submit_button("Run training")

    # If form submitted, run training and cache results in session_state
    if submit:
        if not selected_models:
            st.warning("Select at least one model to train.")
            return

        if not selected_tickers:
            st.warning("Select at least one ticker to display results for.")
            return

        request = TrainModelRequest(
            cutoff_date=default_cutoff,
            years=_DEFAULT_YEARS,
            end=None,
            interval=None,
            model_types=list(selected_models),
            artifacts_dir=_ARTIFACTS_DIR,
        )

        try:
            with st.spinner("Running training (this may take a while)..."):
                result = container.train_model.execute(request)
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return

        # Cache results in session_state and clear any derived caches
        st.session_state.train_result = result
        st.session_state.label_lookup = model_id_to_label
        # Clear derived model_data so reruns don't show stale predictions/manifests
        st.session_state.pop("model_data", None)

        st.success("Training complete")

    # If no cached results yet, show info and return
    if "train_result" not in st.session_state:
        st.info("Configure training options and submit to run training.")
        return

    # Use cached results; ticker selection is always live from the widget above
    result = st.session_state.train_result
    label_lookup = st.session_state.label_lookup

    # Display aggregate metrics
    metrics_frame = TrainPresenter.format_metrics_frame(result, label_lookup=label_lookup)
    if metrics_frame.empty:
        st.warning("No metrics were produced by the training run.")
    else:
        st.subheader("Per-model metrics summary")
        st.dataframe(metrics_frame, use_container_width=True)

    # Load all model predictions and manifests upfront (cache in session_state)
    if "model_data" not in st.session_state:
        model_data: dict[str, dict[str, Any]] = {}
        for model_id, run_dir in result.run_dirs.items():
            pred_df = TrainPresenter.load_predictions_csv(run_dir)
            manifest_path = Path(run_dir) / "manifest.json"
            manifest: dict[str, Any] | None = None
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    manifest = None

            if pred_df is not None and not pred_df.empty:
                model_data[model_id] = {
                    "predictions": pred_df,
                    "manifest": manifest,
                }

        st.session_state.model_data = model_data
    else:
        model_data = st.session_state.model_data

    if not model_data:
        st.warning("No predictions data available for any model.")
        st.stop()

    # Build combined backtest data per ticker across all models
    st.subheader("Evaluation results by ticker")

    # Model filter: allow user to toggle models on/off
    st.markdown("**Select models to display:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    model_filters = {}
    cols = [col1, col2, col3, col4, col5]
    for idx, model_id in enumerate(sorted(model_data.keys())):
        col = cols[idx % len(cols)]
        with col:
            model_filters[model_id] = st.checkbox(
                label_lookup.get(model_id, model_id),
                value=True,
            )

    # Per-ticker consolidated view
    for ticker in selected_tickers:
        st.markdown(f"#### {ticker}")

        # Collect evaluation data for all selected models for this ticker
        combined_backtest_data: dict[str, pd.DataFrame] = {}
        market_history_df = pd.DataFrame()

        for model_id, data in model_data.items():
            if not model_filters[model_id]:
                continue

            pred_df = data["predictions"]
            manifest = data["manifest"]

            if "ticker" not in pred_df.columns:
                continue

            ticker_mask = pred_df["ticker"] == ticker
            if not ticker_mask.any():
                continue

            ticker_preds = pred_df.loc[ticker_mask].copy()

            # Determine market history date range from manifest if available
            start_date = None
            end_date = None
            if isinstance(manifest, dict):
                dates = manifest.get("dates", {})
                start_date = dates.get("requested_start")
                end_date = dates.get("requested_end")

            # Fallback: use prediction date window
            if start_date is None:
                try:
                    start_date = pd.to_datetime(ticker_preds["date"]).min().date().isoformat()
                except Exception:
                    start_date = None
            if end_date is None:
                try:
                    end_date = pd.to_datetime(ticker_preds["date"]).max().date().isoformat()
                except Exception:
                    end_date = None

            # Fetch market history once per ticker (reuse for all models)
            if market_history_df.empty and start_date and end_date:
                try:
                    interval = manifest.get("params", {}).get("interval", "1d") if manifest else "1d"
                    market_data = container.fetch_market_data.execute(
                        FetchMarketDataRequest(ticker=ticker, start_date=start_date, end_date=end_date, interval=interval, include_summary=False)
                    )
                    market_history_df = market_data.history.df.copy()
                except Exception:
                    market_history_df = pd.DataFrame()

            try:
                backtest_frame = cast(
                    pd.DataFrame,
                    TrainPresenter.assemble_backtest_for_ticker(ticker_preds, market_history_df, ticker),
                )
                if not backtest_frame.empty:
                    combined_backtest_data[model_id] = backtest_frame
            except Exception as exc:
                st.warning(f"Could not assemble results for {ticker} with {label_lookup.get(model_id, model_id)}: {exc}")
                continue

        if not combined_backtest_data:
            st.info(f"No evaluation data for {ticker}.")
            continue

        # Build combined chart data: align all models on next_date
        all_dates = set()
        for bt_df in combined_backtest_data.values():
            all_dates.update(bt_df["next_date"].unique())

        chart_data_dict: dict[str, list[object]] = {model_id: [] for model_id in combined_backtest_data.keys()}
        chart_data_dict["next_date"] = []
        chart_data_dict["actual_next_close"] = []

        sorted_dates = sorted(all_dates)
        for date_val in sorted_dates:
            chart_data_dict["next_date"].append(date_val)

            # Get actual close for this date (use first model that has it)
            actual_close = None
            for bt_df in combined_backtest_data.values():
                row = bt_df[bt_df["next_date"] == date_val]
                if not row.empty:
                    ac = row.iloc[0].get("actual_next_close")
                    if ac is not None and pd.notna(ac):
                        actual_close = float(ac)
                        break
            chart_data_dict["actual_next_close"].append(actual_close if actual_close is not None else float("nan"))

            # Get predicted close for each model
            for model_id, bt_df in combined_backtest_data.items():
                row = bt_df[bt_df["next_date"] == date_val]
                if not row.empty:
                    pc = row.iloc[0].get("pred_next_close")
                    chart_data_dict[model_id].append(float(pc) if pc is not None else float("nan"))
                else:
                    chart_data_dict[model_id].append(float("nan"))

        chart_df = pd.DataFrame(chart_data_dict)
        chart_df["next_date"] = pd.to_datetime(chart_df["next_date"])
        chart_df = chart_df.sort_values("next_date").set_index("next_date")

        # Plot all models on one chart
        if not chart_df.empty:
            st.line_chart(chart_df)
        else:
            st.info("No chart data available.")

        # Show metrics for selected models together
        st.markdown("**Model comparison for this ticker:**")
        comparison_rows = []
        for model_id in sorted(combined_backtest_data.keys()):
            bt_df = combined_backtest_data[model_id]
            # Calculate simple metrics: mean error, count
            bt_df_with_error = bt_df.dropna(subset=["y_true", "y_pred"]).copy()
            if not bt_df_with_error.empty:
                mean_error = (bt_df_with_error["y_pred"] - bt_df_with_error["y_true"]).mean()
                rmse = ((bt_df_with_error["y_pred"] - bt_df_with_error["y_true"]) ** 2).mean() ** 0.5
                comparison_rows.append({
                    "Model": label_lookup.get(model_id, model_id),
                    "Predictions": len(bt_df),
                    "Mean Error": f"{mean_error:.4f}",
                    "RMSE": f"{rmse:.4f}",
                })

        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("---")

