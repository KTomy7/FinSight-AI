from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "run_id",
    "model_id",
    "feature_columns",
    "target",
    "split_policy",
    "dates",
    "params",
    "artifact_paths",
    "created_at",
)


def build_run_manifest(
    *,
    run_id: str,
    model_id: str,
    feature_columns: Sequence[str],
    target: str,
    split_policy: Mapping[str, Any],
    dates: Mapping[str, Any],
    params: Mapping[str, Any],
    artifact_paths: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "model_id": model_id,
        "feature_columns": list(feature_columns),
        "target": target,
        "split_policy": dict(split_policy),
        "dates": dict(dates),
        "params": dict(params),
        "artifact_paths": dict(artifact_paths),
        "created_at": created_at,
    }
    return validate_run_manifest(manifest)


def validate_run_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)

    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Manifest is missing required key(s): {missing}.")

    _require_non_empty_str(payload["run_id"], key="run_id")
    _require_non_empty_str(payload["model_id"], key="model_id")
    _require_non_empty_str(payload["target"], key="target")
    _require_iso_datetime_z(payload["created_at"], key="created_at")

    feature_columns = payload["feature_columns"]
    if isinstance(feature_columns, (str, bytes)) or not isinstance(feature_columns, Sequence):
        raise TypeError("Manifest key 'feature_columns' must be a non-empty sequence of strings.")
    if not feature_columns:
        raise TypeError("Manifest key 'feature_columns' must be a non-empty sequence of strings.")
    if any(not isinstance(col, str) or not col.strip() for col in feature_columns):
        raise TypeError("Manifest key 'feature_columns' must contain non-empty strings.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Manifest key 'feature_columns' must not contain duplicates.")
    payload["feature_columns"] = list(feature_columns)

    split_policy = _require_mapping(payload["split_policy"], key="split_policy")
    for key in ("name", "cutoff_date", "date_col", "inclusive_test"):
        if key not in split_policy:
            raise ValueError(f"Manifest key 'split_policy' is missing '{key}'.")
    _require_non_empty_str(split_policy["name"], key="split_policy.name")
    _require_non_empty_str(split_policy["date_col"], key="split_policy.date_col")
    _require_iso_date(split_policy["cutoff_date"], key="split_policy.cutoff_date")
    if not isinstance(split_policy["inclusive_test"], bool):
        raise TypeError("Manifest key 'split_policy.inclusive_test' must be a boolean.")

    dates = _require_mapping(payload["dates"], key="dates")
    for key in (
        "requested_start",
        "requested_end",
        "train_min",
        "train_max",
        "test_min",
        "test_max",
    ):
        if key not in dates:
            raise ValueError(f"Manifest key 'dates' is missing '{key}'.")
        _require_iso_date(dates[key], key=f"dates.{key}")

    _require_mapping(payload["params"], key="params")

    artifact_paths = _require_mapping(payload["artifact_paths"], key="artifact_paths")
    for key in ("run_dir", "metrics", "manifest", "predictions"):
        if key not in artifact_paths:
            raise ValueError(f"Manifest key 'artifact_paths' is missing '{key}'.")
        _require_non_empty_str(artifact_paths[key], key=f"artifact_paths.{key}")

    return payload


def _require_mapping(value: Any, *, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Manifest key '{key}' must be a mapping.")
    return value


def _require_non_empty_str(value: Any, *, key: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Manifest key '{key}' must be a non-empty string.")


def _require_iso_date(value: Any, *, key: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"Manifest key '{key}' must be an ISO date string.")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Manifest key '{key}' must be an ISO date string (YYYY-MM-DD).") from exc


def _require_iso_datetime_z(value: Any, *, key: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise TypeError(f"Manifest key '{key}' must be a UTC ISO-8601 timestamp ending in 'Z'.")

    normalized = value[:-1] + "+00:00"
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Manifest key '{key}' must be a valid UTC ISO-8601 timestamp (e.g. 2026-03-29T12:00:00Z)."
        ) from exc
