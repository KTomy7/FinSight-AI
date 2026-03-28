"""Unit tests for LocalFileModelRegistry in finsight.infrastructure.persistence.file_model_registry."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from finsight.infrastructure.persistence.file_model_registry import LocalFileModelRegistry


@pytest.fixture
def registry() -> LocalFileModelRegistry:
    return LocalFileModelRegistry()


# ---------------------------------------------------------------------------
# create_run_dir
# ---------------------------------------------------------------------------


class TestCreateRunDir:
    def test_creates_directory_on_first_call(self, tmp_path: Path, registry: LocalFileModelRegistry) -> None:
        run_dir = registry.create_run_dir(artifact_root=str(tmp_path), run_id="run_001")
        assert Path(run_dir).is_dir()
        assert Path(run_dir).name == "run_001"

    def test_collision_creates_suffix_1(self, tmp_path: Path, registry: LocalFileModelRegistry) -> None:
        registry.create_run_dir(artifact_root=str(tmp_path), run_id="run_001")
        run_dir_2 = registry.create_run_dir(artifact_root=str(tmp_path), run_id="run_001")
        assert Path(run_dir_2).name == "run_001_1"

    def test_double_collision_creates_suffix_2(self, tmp_path: Path, registry: LocalFileModelRegistry) -> None:
        # Occupy both run_001 and run_001_1 so the next call must use run_001_2.
        registry.create_run_dir(artifact_root=str(tmp_path), run_id="run_001")
        registry.create_run_dir(artifact_root=str(tmp_path), run_id="run_001")  # → run_001_1
        run_dir_3 = registry.create_run_dir(artifact_root=str(tmp_path), run_id="run_001")
        assert Path(run_dir_3).name == "run_001_2"


# ---------------------------------------------------------------------------
# save_predictions
# ---------------------------------------------------------------------------


class TestSavePredictions:
    def test_dataframe_path_writes_csv(self, tmp_path: Path, registry: LocalFileModelRegistry) -> None:
        run_dir = str(tmp_path)
        df = pd.DataFrame({"y_true": [0.01, -0.02], "y_pred": [0.0, 0.0]})
        registry.save_predictions(run_dir=run_dir, predictions=df)
        output = tmp_path / "predictions.csv"
        assert output.exists()
        loaded = pd.read_csv(output)
        assert list(loaded.columns) == ["y_true", "y_pred"]
        assert len(loaded) == 2

    def test_list_of_dicts_writes_csv(self, tmp_path: Path, registry: LocalFileModelRegistry) -> None:
        run_dir = str(tmp_path)
        rows = [{"y_true": 0.01, "y_pred": 0.0}, {"y_true": -0.02, "y_pred": 0.0}]
        registry.save_predictions(run_dir=run_dir, predictions=rows)
        output = tmp_path / "predictions.csv"
        assert output.exists()
        with output.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            loaded = list(reader)
        assert len(loaded) == 2
        assert loaded[0]["y_true"] == "0.01"

    def test_empty_list_writes_empty_file(self, tmp_path: Path, registry: LocalFileModelRegistry) -> None:
        run_dir = str(tmp_path)
        registry.save_predictions(run_dir=run_dir, predictions=[])
        output = tmp_path / "predictions.csv"
        assert output.exists()
        assert output.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# _normalize_rows
# ---------------------------------------------------------------------------


class TestNormalizeRows:
    def test_non_list_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be a pandas DataFrame or list of row mappings"):
            LocalFileModelRegistry._normalize_rows("not-a-list")  # type: ignore[arg-type]

    def test_list_with_non_mapping_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Each prediction row must be a mapping"):
            LocalFileModelRegistry._normalize_rows([{"a": 1}, "bad-row"])  # type: ignore[list-item]

    def test_valid_list_returns_list_of_dicts(self) -> None:
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = LocalFileModelRegistry._normalize_rows(rows)
        assert result == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
