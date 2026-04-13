from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from finsight.application.dto import CompareModelsRequest, CompareModelsResult
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


def _build_comparison_frame(result: CompareModelsResult, *, label_lookup: Mapping[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in result.rows:
        record: dict[str, object] = {
            "rank": row.rank,
            "model": label_lookup.get(row.model_id, row.model_id),
            "model_id": row.model_id,
            "run_id": row.run_id,
        }
        record.update(row.metrics)
        rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    base_columns = ["rank", "model", "model_id", "run_id"]
    metric_columns = [column for column in result.rank_by if column in frame.columns]
    remaining_columns = [
        column for column in frame.columns if column not in base_columns and column not in metric_columns
    ]
    return frame[base_columns + metric_columns + sorted(remaining_columns)]


def render():
    st.title("Compare Models")
    st.markdown(
        "Build a deterministic leaderboard from the latest saved model runs. "
        "Ranking follows the selected metric order, then model ID, then run ID.")

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

    frame = _build_comparison_frame(result, label_lookup=id_to_label)
    if frame.empty:
        st.warning("No comparison rows were returned.")
        return

    st.subheader("Leaderboard")
    st.dataframe(frame, use_container_width=True, hide_index=True)

    st.caption(
        "Ranking priority: "
        + " → ".join(_METRIC_LABELS.get(metric_name, metric_name) for metric_name in result.rank_by)
        + "."
    )
