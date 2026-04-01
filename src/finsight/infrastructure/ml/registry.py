from __future__ import annotations

import csv
import json
import pickle
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from finsight.domain.ports import ModelRegistryPort


MODEL_FILE_NAME = "model.pkl"
MANIFEST_FILE_NAME = "manifest.json"
METRICS_FILE_NAME = "metrics.json"
PREDICTIONS_FILE_NAME = "predictions.csv"


class LocalModelRegistry(ModelRegistryPort):
    """Local filesystem-backed model registry rooted under artifacts/runs."""

    # Legacy API retained for compatibility with the current TrainModel flow.
    def create_run_dir(self, *, artifact_root: str, run_id: str) -> str:
        root = Path(artifact_root)
        root.mkdir(parents=True, exist_ok=True)

        base_path = root / run_id

        try:
            base_path.mkdir(parents=True, exist_ok=False)
            return str(base_path)
        except FileExistsError:
            pass

        suffix = 1
        while True:
            candidate = Path(f"{base_path}_{suffix}")
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return str(candidate)
            except FileExistsError:
                suffix += 1

    def save_metrics(self, *, run_dir: str, metrics: Mapping[str, float | int | str]) -> None:
        self._write_json(Path(run_dir) / METRICS_FILE_NAME, metrics)

    def save_manifest(self, *, run_dir: str, manifest: Mapping[str, Any]) -> None:
        self._write_json(Path(run_dir) / MANIFEST_FILE_NAME, manifest)

    def save_predictions(self, *, run_dir: str, predictions: object) -> None:
        output_path = Path(run_dir) / PREDICTIONS_FILE_NAME

        if isinstance(predictions, pd.DataFrame):
            predictions.to_csv(output_path, index=False)
            return

        rows = self._normalize_rows(predictions)
        with output_path.open("w", newline="", encoding="utf-8") as file_obj:
            if not rows:
                file_obj.write("")
                return

            fieldnames = list(rows[0].keys())
            for row in rows[1:]:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save_run(
        self,
        *,
        artifact_root: str,
        model_run_id: str,
        model_artifact: object,
        manifest: Mapping[str, Any],
        metrics: Mapping[str, float | int | str],
        predictions: object | None = None,
    ) -> str:
        run_dir = self._resolve_run_dir(artifact_root=artifact_root, model_run_id=model_run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._write_pickle(run_dir / MODEL_FILE_NAME, model_artifact)
        self._write_json(run_dir / MANIFEST_FILE_NAME, manifest)
        self._write_json(run_dir / METRICS_FILE_NAME, metrics)

        if predictions is not None:
            self.save_predictions(run_dir=str(run_dir), predictions=predictions)

        return str(run_dir)

    def load_run(self, *, artifact_root: str, model_run_id: str) -> Mapping[str, Any]:
        run_dir = self._resolve_run_dir(artifact_root=artifact_root, model_run_id=model_run_id)
        model_path = run_dir / MODEL_FILE_NAME
        manifest_path = run_dir / MANIFEST_FILE_NAME
        metrics_path = run_dir / METRICS_FILE_NAME
        predictions_path = run_dir / PREDICTIONS_FILE_NAME

        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist for model_run_id='{model_run_id}': {run_dir}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact for model_run_id='{model_run_id}': {model_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest for model_run_id='{model_run_id}': {manifest_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics for model_run_id='{model_run_id}': {metrics_path}")

        payload: dict[str, Any] = {
            "model_run_id": model_run_id,
            "run_dir": str(run_dir),
            "model_artifact": self._read_pickle(model_path),
            "manifest": self._read_json(manifest_path),
            "metrics": self._read_json(metrics_path),
        }

        if predictions_path.exists():
            payload["predictions"] = pd.read_csv(predictions_path)

        return payload

    @staticmethod
    def _resolve_run_dir(*, artifact_root: str, model_run_id: str) -> Path:
        return Path(artifact_root) / model_run_id

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file_obj:
            loaded = json.load(file_obj)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected JSON object at {path}.")
        return loaded

    @staticmethod
    def _write_pickle(path: Path, payload: object) -> None:
        with path.open("wb") as file_obj:
            pickle.dump(payload, file_obj)

    @staticmethod
    def _read_pickle(path: Path) -> object:
        with path.open("rb") as file_obj:
            return pickle.load(file_obj)

    @staticmethod
    def _normalize_rows(predictions: object) -> list[dict[str, Any]]:
        if not isinstance(predictions, list):
            raise TypeError("predictions must be a pandas DataFrame or list of row mappings.")

        normalized: list[dict[str, Any]] = []
        for row in predictions:
            if not isinstance(row, MappingABC):
                raise TypeError("Each prediction row must be a mapping.")
            normalized.append(dict(row))
        return normalized


