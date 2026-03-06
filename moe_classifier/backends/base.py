"""
Abstract base class for classifier backends.

All deployment modes (local, remote, distributed) implement this interface
so that MOEClassifier can delegate to any backend transparently.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from moe_classifier.types import ClassificationResult


class ClassifierBackend(ABC):
    """
    Interface that every deployment backend must implement.

    Lifecycle:
        1. ``__init__()`` — store config (no heavy work)
        2. ``initialize()`` — load models / connect to services
        3. ``classify()`` / ``get_stats()`` — use the backend
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Prepare the backend for use.

        For local: loads ML models into GPU.
        For remote: authenticates and verifies connectivity.
        For distributed: loads gating models and reads worker mapping.
        """
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True after ``initialize()`` has completed successfully."""
        ...

    @abstractmethod
    def classify(
        self,
        text: str,
        description: str = "",
        *,
        return_domain_probabilities: bool = False,
        return_raw_response: bool = False,
    ) -> ClassificationResult:
        """Classify a single piece of text and return a ClassificationResult."""
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return system capability / status information."""
        ...
