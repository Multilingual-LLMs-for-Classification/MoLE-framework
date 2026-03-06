"""
Local backend — full in-process pipeline on a single GPU.

This is the original ``MOEClassifier`` behavior: loads all gating models
and the LLM expert pool into GPU memory, runs everything in-process.
"""

import time
from typing import Any, Dict

from moe_classifier.types import ClassificationResult
from .base import ClassifierBackend


class LocalBackend(ClassifierBackend):
    """
    Monolithic single-GPU backend.

    Wraps ``PromptRoutingSystem(coordinator_only=False)`` and calls
    ``route_prompt()`` for each classification request.
    """

    def __init__(self) -> None:
        self._system = None
        self._initialized = False

    def initialize(self) -> None:
        try:
            from moe_router.gating.components.routing_system import PromptRoutingSystem
        except ImportError as exc:
            raise RuntimeError(
                "moe_router package not found.  Make sure you are running "
                "from the MoLE-framework directory and the package is "
                "installed (pip install -e .)."
            ) from exc

        try:
            self._system = PromptRoutingSystem(
                training_mode=False,
                coordinator_only=False,
            )
            self._initialized = True
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize local routing system: {exc}"
            ) from exc

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._system is not None

    def classify(
        self,
        text: str,
        description: str = "",
        *,
        return_domain_probabilities: bool = False,
        return_raw_response: bool = False,
    ) -> ClassificationResult:
        prompt = f"{description}\n\n{text}".strip() if description else text

        t0 = time.perf_counter()
        raw = self._system.route_prompt(
            prompt=prompt,
            input_data={"text": text},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ClassificationResult(
            language=raw.get("language", "unknown"),
            domain=raw.get("domain", "unknown"),
            task=raw.get("task", "unknown"),
            result=str(raw.get("result", "")),
            routing_path=raw.get("routing_path", ""),
            confidence=raw.get("expert_confidence"),
            domain_probabilities=(
                raw.get("domain_probabilities") if return_domain_probabilities else None
            ),
            raw_response=(
                raw.get("raw_response") if return_raw_response else None
            ),
            processing_time_ms=elapsed_ms,
        )

    def get_stats(self) -> Dict[str, Any]:
        return self._system.get_system_stats()
