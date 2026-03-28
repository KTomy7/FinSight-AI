from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from finsight.domain.ports import ModelRegistryPort


class LocalFileModelRegistry(ModelRegistryPort):
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
        self._write_json(Path(run_dir) / "metrics.json", metrics)

    def save_metadata(self, *, run_dir: str, metadata: Mapping[str, Any]) -> None:
        self._write_json(Path(run_dir) / "metadata.json", metadata)

    def save_predictions(self, *, run_dir: str, predictions: object) -> None:
        output_path = Path(run_dir) / "predictions.csv"

        if isinstance(predictions, pd.DataFrame):
            predictions.to_csv(output_path, index=False)
            return

        rows = self._normalize_rows(predictions)
        with output_path.open("w", newline="", encoding="utf-8") as file_obj:
            if not rows:
                file_obj.write("")
                return

            # Compute fieldnames as the ordered union of keys across all rows
            fieldnames = list(rows[0].keys())
            for row in rows[1:]:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _normalize_rows(predictions: object) -> list[dict[str, Any]]:
        if not isinstance(predictions, list):
            raise TypeError("predictions must be a pandas DataFrame or list of row mappings.")

        normalized: list[dict[str, Any]] = []
        for row in predictions:
            if not isinstance(row, Mapping):
                raise TypeError("Each prediction row must be a mapping.")
            normalized.append(dict(row))
        return normalized




