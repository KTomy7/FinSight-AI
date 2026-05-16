from __future__ import annotations

import streamlit as st
from pathlib import Path

from finsight.application.dto import CompareModelsRequest
from finsight.adapters.web_streamlit.presenters import ComparisonPresenter
from finsight.bootstrap.container import build_container
from finsight.config.settings import get_settings
from finsight.domain.metrics import METRIC_DIRECTION_ACCURACY, METRIC_MAE, METRIC_RMSE


_SETTINGS = get_settings()
_METRIC_LABELS = {
    METRIC_MAE: "Mean Absolute Error (lower is better)",
    METRIC_RMSE: "Root Mean Squared Error (lower is better)",
    METRIC_DIRECTION_ACCURACY: "Direction Accuracy (higher is better)",
}


@st.cache_resource(ttl=_SETTINGS.cache.resource_ttl_seconds)
def _compare_models_uc():
    return build_container().compare_models



def render():
    st.title("Compare Models")
    st.markdown(
        "Build a deterministic leaderboard from the latest saved model runs. "
        "Ranking follows a fixed metric priority among the metrics you select, then model ID, then run ID.")
    st.info(
        "Metric direction defaults: MAE/RMSE are ranked lower-is-better, while Direction Accuracy is ranked higher-is-better. "
        "Use Ranking metrics to include or exclude metrics. Priority remains MAE -> RMSE -> Direction Accuracy."
    )

    model_defaults = _SETTINGS.model_defaults
    model_ids = list(model_defaults.training_model_ids())
    id_to_label = model_defaults.id_to_label()

    if not model_ids:
        st.warning("No training-enabled models are configured.")
        return

    default_rank_by = [METRIC_MAE, METRIC_RMSE, METRIC_DIRECTION_ACCURACY]

    with st.form("compare_models_form"):
        selected_model_ids = st.multiselect(
            "Models to compare",
            model_ids,
            default=model_ids,
            format_func=lambda model_id: id_to_label.get(model_id, model_id),
        )
        selected_rank_by = st.multiselect(
            "Ranking metrics",
            default_rank_by,
            default=default_rank_by,
            format_func=lambda metric_name: _METRIC_LABELS.get(metric_name, metric_name),
        )
        submit = st.form_submit_button("Build leaderboard")

    if not submit:
        st.info("Choose models and ranking metrics, then build the leaderboard.")
        return

    if not selected_model_ids:
        st.warning("Select at least one model to compare.")
        return

    if not selected_rank_by:
        st.warning("Select at least one ranking metric.")
        return

    try:
        result = _compare_models_uc().execute(
            CompareModelsRequest(
                model_ids=list(selected_model_ids),
                rank_by=list(selected_rank_by),
            )
        )
    except FileNotFoundError as error:
        st.error(f"No trained run artifacts were found: {error}")
        return
    except (ValueError, TypeError) as error:
        st.error(f"Unable to build leaderboard: {error}")
        return
    except Exception as error:  # pragma: no cover - defensive fallback for UI resilience
        st.error(f"Leaderboard generation failed unexpectedly: {error}")
        return

    # Load registry to enrich with best-run metadata
    registry_snapshot = None
    try:
        # Use standard artifacts directory (consistent with CLI default)
        artifacts_dir = "artifacts/runs"
        from finsight.infrastructure.ml.run_registry import LocalFileRunRegistry
        registry = LocalFileRunRegistry(
            supported_model_ids=_SETTINGS.model_defaults.training_model_ids()
        )
        registry_snapshot = registry.load_registry(artifact_root=artifacts_dir)
    except Exception:
        pass  # Graceful fallback: registry is optional for display

    # Format leaderboard with registry metadata if available
    if registry_snapshot:
        frame = ComparisonPresenter.format_leaderboard_with_best_runs(
            result,
            label_lookup=id_to_label,
            registry_snapshot=registry_snapshot,
        )
    else:
        frame = ComparisonPresenter.format_leaderboard_frame(result, label_lookup=id_to_label)

    if frame.empty:
        st.warning("No comparison rows were returned.")
        return

    st.subheader("Leaderboard")

    # Display leaderboard with conditional column hiding based on registry availability
    if registry_snapshot and "is_best_run" in frame.columns:
        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
            column_config={
                "is_best_run": st.column_config.CheckboxColumn(
                    "⭐ Best",
                    help="Marked as the best performing run for this model in the registry",
                ),
                "best_run_since": st.column_config.TextColumn(
                    "Best Since",
                    help="When this run became the best for its model",
                ),
            },
        )
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)

    st.caption(
        "Ranking priority: "
        + " → ".join(_METRIC_LABELS.get(metric_name, metric_name) for metric_name in result.rank_by)
        + "."
    )
