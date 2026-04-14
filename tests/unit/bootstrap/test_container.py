from __future__ import annotations

from finsight.bootstrap import container as container_module
from finsight.config.settings import Settings
from finsight.application.use_cases.compare_models import CompareModels


def test_build_container_exposes_compare_models_use_case(monkeypatch) -> None:
    container_module.build_container.cache_clear()
    monkeypatch.setattr(container_module, "get_settings", lambda: Settings())

    app_container = container_module.build_container()

    assert isinstance(app_container.compare_models, CompareModels)
    assert app_container.compare_models is app_container.compare_models

