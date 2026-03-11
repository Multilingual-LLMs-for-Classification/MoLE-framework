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

    def __init__(self, pipeline_config=None) -> None:
        self._config = pipeline_config
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

        # Build keyword overrides from PipelineConfig (if supplied)
        kwargs = {}
        if self._config:
            if self._config.language_model:
                kwargs["language_model"] = self._config.language_model
            if self._config.expert_registry:
                kwargs["expert_registry_override"] = self._config.expert_registry
            if self._config.domain_model_dir:
                kwargs["domain_model_dir"] = self._config.domain_model_dir
            if self._config.domain_model_name != "xlm-roberta-base":
                kwargs["domain_model_name"] = self._config.domain_model_name
            if self._config.task_router_dir:
                kwargs["task_router_dir"] = self._config.task_router_dir
            if self._config.task_encoder_name != "xlm-roberta-base":
                kwargs["task_encoder_name"] = self._config.task_encoder_name

        try:
            self._system = PromptRoutingSystem(
                training_mode=False,
                coordinator_only=False,
                **kwargs,
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
