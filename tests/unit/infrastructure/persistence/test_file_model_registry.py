import csv
from pathlib import Path

import pandas as pd
import pytest

from finsight.infrastructure.ml.registry import LocalModelRegistry


def test_create_run_dir_increments_suffix_until_available(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    artifact_root = tmp_path / "runs"
    model_run_id = "2026-03-28T101010Z__naive_zero"

    (artifact_root / model_run_id).mkdir(parents=True)
    (artifact_root / f"{model_run_id}_1").mkdir(parents=True)

    run_dir = registry.create_run_dir(artifact_root=str(artifact_root), model_run_id=model_run_id)

    assert Path(run_dir).name == f"{model_run_id}_2"
    assert Path(run_dir).exists()


def test_save_predictions_writes_csv_from_row_mappings(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    registry.save_predictions(
        run_dir=str(run_dir),
        predictions=[
            {"date": "2024-01-01", "ticker": "AAPL", "y_true": 0.1, "y_pred": 0.0},
            {"date": "2024-01-02", "ticker": "AAPL", "y_true": -0.1, "y_pred": 0.05},
        ],
    )

    with (run_dir / "predictions.csv").open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert len(rows) == 2
    assert rows[0]["ticker"] == "AAPL"
    assert rows[1]["y_pred"] == "0.05"


def test_save_predictions_writes_csv_from_dataframe(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    predictions_df = pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "AAPL", "y_true": 0.1, "y_pred": 0.0},
            {"date": "2024-01-02", "ticker": "MSFT", "y_true": -0.1, "y_pred": 0.05},
        ]
    )

    registry.save_predictions(run_dir=str(run_dir), predictions=predictions_df)

    loaded = pd.read_csv(run_dir / "predictions.csv")
    assert list(loaded["ticker"]) == ["AAPL", "MSFT"]


def test_save_predictions_rejects_object_with_non_callable_to_csv(tmp_path: Path) -> None:
    class _FakePredictions:
        to_csv = "not-callable"

    registry = LocalModelRegistry()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(TypeError, match="must be a pandas DataFrame or list"):
        registry.save_predictions(run_dir=str(run_dir), predictions=_FakePredictions())


def test_save_predictions_rejects_non_list_payload(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(TypeError, match="must be a pandas DataFrame or list"):
        registry.save_predictions(run_dir=str(run_dir), predictions={"ticker": "AAPL"})


def test_save_predictions_rejects_non_mapping_rows(tmp_path: Path) -> None:
    registry = LocalModelRegistry()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(TypeError, match="Each prediction row must be a mapping"):
        registry.save_predictions(run_dir=str(run_dir), predictions=["bad-row"])

