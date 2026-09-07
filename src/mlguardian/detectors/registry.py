"""Detector registry."""

from __future__ import annotations

from collections.abc import Callable

from mlguardian.detectors.base import Detector

REGISTRY: dict[str, Callable[[], Detector]] = {}


def register_detector(name: str) -> Callable[[Callable[[], Detector]], Callable[[], Detector]]:
    def decorator(factory: Callable[[], Detector]) -> Callable[[], Detector]:
        REGISTRY[name] = factory
        return factory

    return decorator


def build_detectors() -> list[Detector]:
    return [factory() for _, factory in sorted(REGISTRY.items())]
