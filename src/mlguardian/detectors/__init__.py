"""Detector package and registry bootstrap."""

from mlguardian.detectors.base import AuditContext, Detector
from mlguardian.detectors.implementations import summarize_dataset
from mlguardian.detectors.registry import build_detectors

__all__ = ["AuditContext", "Detector", "build_detectors", "summarize_dataset"]
