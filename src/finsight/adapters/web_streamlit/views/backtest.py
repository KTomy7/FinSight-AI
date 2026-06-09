from __future__ import annotations

import streamlit as st

from finsight.adapters.web_streamlit.presenters import BacktestPresenter
from finsight.application.dto import BacktestRequest
from finsight.application.use_cases.backtest import Backtest
from finsight.bootstrap.container import build_container
from finsight.config.settings import get_settings


_SETTINGS = get_settings()


@st.cache_resource(ttl=_SETTINGS.cache.resource_ttl_seconds)
def _backtest_uc() -> Backtest:
    return build_container().backtest


def render() -> None:
    st.title("Walk-Forward Backtesting")
    st.markdown(
        "Run walk-forward evaluation across one or more models using the configured training ticker basket. "
        "This page is dedicated to robust model comparison and does not persist model artifacts."
    )

    model_defaults = _SETTINGS.model_defaults
    model_ids = list(model_defaults.training_model_ids())
    id_to_label = model_defaults.id_to_label()

    if not model_ids:
        st.warning("No training-enabled models are configured.")
        return

    with st.form("walk_forward_backtest_form"):
        selected_model_ids = st.multiselect(
            "Models to backtest",
            model_ids,
            default=model_ids,
            format_func=lambda model_id: id_to_label.get(model_id, model_id),
        )
        years = int(st.slider("History window (years)", min_value=1, max_value=5, value=2))
        min_train_days = int(st.number_input("Minimum train window (days)", min_value=30, value=252, step=1))
        test_window_days = int(st.number_input("Test window per fold (days)", min_value=1, value=21, step=1))
        step_days = int(st.number_input("Step size between folds (days)", min_value=1, value=21, step=1))
        max_folds = int(st.number_input("Maximum folds", min_value=1, value=6, step=1))
        submit = st.form_submit_button("Run walk-forward backtest")

    if not submit:
        st.info("Choose models and walk-forward settings, then run backtest.")
        return

    if not selected_model_ids:
        st.warning("Select at least one model to backtest.")
        return

    try:
        report = _backtest_uc().execute(
            BacktestRequest(
                model_ids=list(selected_model_ids),
                years=years,
                min_train_days=min_train_days,
                test_window_days=test_window_days,
                step_days=step_days,
                max_folds=max_folds,
            )
        )
    except (ValueError, TypeError) as error:
        st.error(f"Unable to run backtest: {error}")
        return
    except Exception as error:  # pragma: no cover - defensive fallback for UI resilience
        st.error(f"Backtest failed unexpectedly: {error}")
        return

    summary_frame = BacktestPresenter.format_model_metrics_frame(report, label_lookup=id_to_label)
    if summary_frame.empty:
        st.warning("No backtest rows were returned.")
        return

    st.subheader("Backtest summary by model")
    st.dataframe(summary_frame, width='stretch', hide_index=True)

    fold_count = report.split_spec.get("fold_count")
    if fold_count is not None:
        st.caption(f"Walk-forward folds evaluated: {fold_count}")

    st.subheader("Fold-level details")
    for result in report.results:
        model_label = id_to_label.get(result.model_id, result.model_id)
        st.markdown(f"**{model_label}**")
        fold_frame = BacktestPresenter.format_fold_frame(result)
        if fold_frame.empty:
            st.info(f"No fold rows available for {model_label}.")
            continue
        st.dataframe(fold_frame, width='stretch', hide_index=True)

