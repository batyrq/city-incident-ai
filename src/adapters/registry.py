from __future__ import annotations

from src.adapters.base import DatasetAdapter
from src.adapters.custom_yolo import CustomYoloAdapter
from src.adapters.dfire import DFireAdapter


class GenericAdapter(DatasetAdapter):
    pass


def get_adapter(config: dict) -> DatasetAdapter:
    adapter_name = config.get("adapter", "")
    if adapter_name == "dfire":
        return DFireAdapter.from_config(config)
    if adapter_name == "custom_yolo":
        return CustomYoloAdapter.from_config(config)
    return GenericAdapter.from_config(config)
