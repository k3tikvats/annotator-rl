"""
Annotation QA Environment — A real-world OpenEnv for ML annotation quality assurance.

This environment exposes an AI agent to intentionally-flawed annotations on
synthetic scenes, challenging it to detect and correct errors.
"""

from .client import AnnotationQAEnv
from .models import AnnotationQAAction, AnnotationQAObservation, AnnotationQAState

__all__ = [
    "AnnotationQAEnv",
    "AnnotationQAAction",
    "AnnotationQAObservation",
    "AnnotationQAState",
]
