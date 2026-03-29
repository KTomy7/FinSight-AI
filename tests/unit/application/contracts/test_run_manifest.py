import pytest
from types import MappingProxyType

from finsight.application.contracts import REQUIRED_MANIFEST_KEYS, build_run_manifest, validate_run_manifest


def _valid_manifest() -> dict[str, object]:
    return {
        "run_id": "2026-03-29T120000Z__naive_zero",
        "model_id": "naive_zero",
        "feature_columns": ["ret_1d", "mom_20d"],
        "target": "target_ret_1d",
        "split_policy": {
            "name": "time_split",
            "cutoff_date": "2025-06-01",
            "date_col": "date",
            "inclusive_test": True,
        },
        "dates": {
            "requested_start": "2024-03-18",
            "requested_end": "2026-03-17",
            "train_min": "2024-03-18",
            "train_max": "2025-05-31",
            "test_min": "2025-06-01",
            "test_max": "2026-03-17",
        },
        "params": {"tickers": ["AAPL", "JPM"], "interval": "1d", "years": 2, "horizon": "1d"},
        "artifact_paths": {
            "run_dir": "artifacts/runs/2026-03-29T120000Z__naive_zero",
            "metrics": "artifacts/runs/2026-03-29T120000Z__naive_zero/metrics.json",
            "manifest": "artifacts/runs/2026-03-29T120000Z__naive_zero/manifest.json",
            "predictions": "artifacts/runs/2026-03-29T120000Z__naive_zero/predictions.csv",
        },
        "created_at": "2026-03-29T12:00:00Z",
    }


def test_validate_run_manifest_accepts_valid_payload() -> None:
    manifest = _valid_manifest()
    validated = validate_run_manifest(manifest)
    assert set(REQUIRED_MANIFEST_KEYS) == set(validated.keys())


def test_validate_run_manifest_rejects_missing_required_key() -> None:
    manifest = _valid_manifest()
    manifest.pop("artifact_paths")

    with pytest.raises(ValueError, match="missing required key"):
        validate_run_manifest(manifest)


def test_validate_run_manifest_rejects_bad_created_at_format() -> None:
    manifest = _valid_manifest()
    manifest["created_at"] = "2026-03-29 12:00:00"

    with pytest.raises(TypeError, match="created_at"):
        validate_run_manifest(manifest)


def test_validate_run_manifest_rejects_created_at_with_space_separator() -> None:
    manifest = _valid_manifest()
    manifest["created_at"] = "2026-03-29 12:00:00Z"

    with pytest.raises(ValueError, match="YYYY-MM-DDTHH:MM:SSZ"):
        validate_run_manifest(manifest)


def test_validate_run_manifest_rejects_created_at_with_fractional_seconds() -> None:
    manifest = _valid_manifest()
    manifest["created_at"] = "2026-03-29T12:00:00.123Z"

    with pytest.raises(ValueError, match="YYYY-MM-DDTHH:MM:SSZ"):
        validate_run_manifest(manifest)


def test_validate_run_manifest_rejects_bad_dates_format() -> None:
    manifest = _valid_manifest()
    manifest["dates"] = {**manifest["dates"], "test_max": "03-29-2026"}

    with pytest.raises(ValueError, match="dates.test_max"):
        validate_run_manifest(manifest)


def test_validate_run_manifest_accepts_feature_columns_tuple_and_normalizes_to_list() -> None:
    manifest = _valid_manifest()
    manifest["feature_columns"] = ("ret_1d", "mom_20d")

    validated = validate_run_manifest(manifest)

    assert validated["feature_columns"] == ["ret_1d", "mom_20d"]
    assert isinstance(validated["feature_columns"], list)


def test_validate_run_manifest_rejects_feature_columns_string() -> None:
    manifest = _valid_manifest()
    manifest["feature_columns"] = "ret_1d"

    with pytest.raises(TypeError, match="feature_columns"):
        validate_run_manifest(manifest)


def test_validate_run_manifest_normalizes_nested_mappings_to_dict() -> None:
    manifest = _valid_manifest()
    manifest["split_policy"] = MappingProxyType(dict(manifest["split_policy"]))
    manifest["dates"] = MappingProxyType(dict(manifest["dates"]))
    manifest["params"] = MappingProxyType(dict(manifest["params"]))
    manifest["artifact_paths"] = MappingProxyType(dict(manifest["artifact_paths"]))

    validated = validate_run_manifest(manifest)

    assert isinstance(validated["split_policy"], dict)
    assert isinstance(validated["dates"], dict)
    assert isinstance(validated["params"], dict)
    assert isinstance(validated["artifact_paths"], dict)


def test_build_run_manifest_validates_on_create() -> None:
    payload = _valid_manifest()
    payload["feature_columns"] = ["ret_1d", "ret_1d"]

    with pytest.raises(ValueError, match="must not contain duplicates"):
        build_run_manifest(
            run_id=payload["run_id"],
            model_id=payload["model_id"],
            feature_columns=payload["feature_columns"],
            target=payload["target"],
            split_policy=payload["split_policy"],
            dates=payload["dates"],
            params=payload["params"],
            artifact_paths=payload["artifact_paths"],
            created_at=payload["created_at"],
        )
