"""
Distributed backend — gating locally + HTTP dispatch to expert workers.

Embeds the coordinator logic (language detection, domain classification,
Q-learning task routing, expert selection) without the FastAPI layer,
then dispatches LLM inference to remote expert workers via HTTP.

This is the programmatic equivalent of ``SERVICE_MODE=coordinator``.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from moe_classifier.types import ClassificationResult
from .base import ClassifierBackend


class DistributedBackend(ClassifierBackend):
    """
    Coordinator-mode backend: gating in-process, inference via HTTP workers.

    Parameters
    ----------
    expert_mapping : str, optional
        Path to ``expert_machine_mapping.json``.  Defaults to
        ``config/expert_machine_mapping.json`` relative to the project root.
    timeout : float, optional
        HTTP timeout for worker requests in seconds (default: 600).
    """

    def __init__(
        self,
        expert_mapping: Optional[str] = None,
        timeout: float = 600.0,
        pipeline_config=None,
    ) -> None:
        self._mapping_path = expert_mapping
        self._timeout = timeout
        self._config = pipeline_config
        self._system = None
        self._model_to_worker: Dict[str, str] = {}
        self._workers: Dict[str, Dict] = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        # 1. Load the gating pipeline (no LLM pool)
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
                coordinator_only=True,
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize gating pipeline: {exc}"
            ) from exc

        # 2. Load the expert worker mapping
        mapping_path = self._resolve_mapping_path()
        self._load_mapping(mapping_path)

        self._initialized = True
        print(
            f"[DistributedBackend] Ready — gating loaded, "
            f"{len(self._workers)} worker(s) registered."
        )

    def _resolve_mapping_path(self) -> Path:
        """Resolve the expert_machine_mapping.json path."""
        if self._mapping_path:
            p = Path(self._mapping_path)
            if p.exists():
                return p
            raise FileNotFoundError(
                f"Expert mapping file not found: {self._mapping_path}"
            )

        # Default: look relative to the moe_router package root
        try:
            import moe_router
            package_root = Path(moe_router.__file__).parent.parent
        except (ImportError, AttributeError):
            package_root = Path.cwd()

        default = package_root / "config" / "expert_machine_mapping.json"
        if default.exists():
            return default

        raise FileNotFoundError(
            "expert_machine_mapping.json not found.  Pass the path "
            "explicitly via expert_mapping= or ensure the file exists "
            f"at {default}"
        )

    def _load_mapping(self, path: Path) -> None:
        """Load worker URL mapping from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self._model_to_worker = mapping["model_to_worker"]
        self._workers = mapping["workers"]
        print(
            f"[DistributedBackend] Loaded mapping from {path}: "
            f"{len(self._workers)} worker(s)"
        )

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._system is not None

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        description: str = "",
        *,
        return_domain_probabilities: bool = False,
        return_raw_response: bool = False,
    ) -> ClassificationResult:
        t0 = time.perf_counter()

        # Phase 1: lightweight gating (runs locally, no LLM)
        prompt = f"{description}\n\n{text}".strip() if description else text
        gating = self._system.run_gating(prompt)

        # Phase 2: dispatch to the correct expert worker via HTTP
        expert_result = self._dispatch_to_worker(
            base_model_key=gating.base_model_key,
            payload={
                "task_key": f"{gating.domain}/{gating.task}",
                "language": gating.language,
                "text": text,
                "description": description,
                "adapter_name": gating.adapter_name,
                "request_id": f"sdk-{int(time.time()*1000)}",
            },
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        routing_path = f"{gating.routing_path} -> gateway:{gating.base_model_key}"

        return ClassificationResult(
            language=gating.language,
            domain=gating.domain,
            task=gating.task,
            result=str(expert_result.get("result", "")),
            routing_path=routing_path,
            confidence=expert_result.get("confidence"),
            domain_probabilities=None,  # gating doesn't return probs in this path
            raw_response=(
                expert_result.get("raw_response") if return_raw_response else None
            ),
            processing_time_ms=elapsed_ms,
        )

    def _dispatch_to_worker(
        self,
        base_model_key: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send an inference request to the worker that hosts the given model."""
        worker_id = self._model_to_worker.get(base_model_key)
        if worker_id is None:
            raise KeyError(
                f"No worker registered for base_model_key='{base_model_key}'. "
                f"Check expert_machine_mapping.json."
            )

        worker_url = self._workers[worker_id]["url"]
        endpoint = f"{worker_url}/api/v1/expert/classify"

        timeout = httpx.Timeout(
            connect=10.0, read=self._timeout, write=30.0, pool=5.0
        )
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        stats = self._system.get_system_stats()
        stats["deployment_mode"] = "distributed"
        stats["registered_workers"] = len(self._workers)
        return stats
