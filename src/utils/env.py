from __future__ import annotations

import importlib.util
from typing import Iterable


def has_package(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def package_summary(packages: Iterable[str]) -> dict[str, bool]:
    return {name: has_package(name) for name in packages}
