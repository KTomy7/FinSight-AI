from __future__ import annotations

from typing import Sequence

from finsight.domain.ports import ModelPort


class SklearnModelRouter(ModelPort):
    """Routes model evaluation requests to the adapter that owns a model id."""

    def __init__(self, adapters: Sequence[ModelPort]) -> None:
        if not adapters:
            raise ValueError("SklearnModelRouter requires at least one model adapter.")

        adapter_map: dict[str, ModelPort] = {}
        ordered_model_ids: list[str] = []

        for adapter in adapters:
            for model_id in tuple(adapter.supported_model_types()):
                if model_id in adapter_map:
                    raise ValueError(f"Duplicate model id '{model_id}' detected across model adapters.")
                adapter_map[model_id] = adapter
                ordered_model_ids.append(model_id)

        if not ordered_model_ids:
            raise ValueError("SklearnModelRouter adapters must expose at least one supported model id.")

        self._adapter_by_model_id = adapter_map
        self._supported_model_ids = tuple(ordered_model_ids)

    def evaluate(
        self,
        *,
        train_dataset: object,
        test_dataset: object,
        model_type: str,
        target_column: str,
        id_columns: Sequence[str] = ("date", "ticker"),
    ) -> tuple[dict[str, float], object]:
        adapter = self._adapter_by_model_id.get(model_type)
        if adapter is None:
            raise ValueError(
                f"Unsupported model type '{model_type}'. Supported model types: {self._supported_model_ids}."
            )

        return adapter.evaluate(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_type=model_type,
            target_column=target_column,
            id_columns=id_columns,
        )

    def supported_model_types(self) -> tuple[str, ...]:
        return self._supported_model_ids

