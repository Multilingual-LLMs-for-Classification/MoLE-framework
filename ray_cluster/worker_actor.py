"""
ExpertWorkerActor — Ray Actor wrapping a SingleModelPool for GPU-resident LLM inference.

Replaces expert_worker/main.py + expert_worker/router.py.

Each instance holds exactly one base LLM permanently in GPU memory (via SingleModelPool).
Ray handles:
  - GPU assignment and scheduling across cluster nodes
  - Actor lifecycle, health monitoring, and automatic restart on crash
  - Cross-node serialization and async communication
  - Resource tracking (num_gpus=1 reserves one full GPU per actor)

The ML logic (SingleModelPool, TaskExpert, LoRA adapters) is unchanged.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import ray


# ---------------------------------------------------------------------------
# Helpers (module-level so they run inside the actor process)
# ---------------------------------------------------------------------------

def _resolve_registry_path(registry_path: Optional[str] = None) -> Path:
    if registry_path:
        return Path(registry_path)
    return (
        Path(__file__).parents[1]
        / "moe_router" / "experts" / "config" / "experts_registry.json"
    )


def _build_experts(pool, registry_path: Path) -> Dict:
    """
    Instantiate TaskExpert objects for tasks whose base model matches this worker.
    Identical logic to expert_worker/main.py::_build_experts().
    """
    from moe_router.experts.llms.task_expert import TaskExpert, TaskExpertConfig

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    experts: Dict = {}
    for task_key, tcfg in registry.get("tasks", {}).items():
        model_keys_for_task = {tcfg.get("base_model_key")}
        for lang_cfg in tcfg.get("language_mapping", {}).values():
            if isinstance(lang_cfg, dict) and "base_model_key" in lang_cfg:
                model_keys_for_task.add(lang_cfg["base_model_key"])

        if pool.assigned_model_key not in model_keys_for_task:
            continue

        domain, task = task_key.split("/", 1)
        experts.setdefault(domain, {})
        experts[domain][task] = TaskExpert(
            TaskExpertConfig(
                domain=domain,
                task=task,
                registry_path=str(registry_path),
                generation=None,
            ),
            pool=pool,
        )
        print(f"[ExpertWorkerActor] Registered expert: {task_key}")

    return experts


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=1)
class ExpertWorkerActor:
    """
    Ray Actor that permanently holds one LLM in GPU memory.

    Lifecycle
    ---------
    1. __init__: loads the base LLM via SingleModelPool.preload()
    2. classify(): called by the coordinator; runs inference synchronously
       (Ray executes each actor method call in the actor's thread; no need for
       asyncio inside the actor — the caller side awaits the ObjectRef).
    3. On crash: Ray automatically restarts the actor (model reloads on init).

    Resource declaration
    --------------------
    ``@ray.remote(num_gpus=1)`` tells the Ray scheduler that this actor
    requires one full GPU.  Ray places it on a node that has a free GPU and
    sets CUDA_VISIBLE_DEVICES automatically — no manual GPU assignment needed.
    """

    def __init__(
        self,
        model_key: str,
        worker_id: str,
        registry_path: Optional[str] = None,
    ):
        from expert_worker.single_model_pool import SingleModelPool

        self.worker_id = worker_id
        self.model_key = model_key
        registry = _resolve_registry_path(registry_path)

        print(f"[ExpertWorkerActor:{worker_id}] Loading model '{model_key}' ...")
        self.pool = SingleModelPool(model_key=model_key, registry_path=registry)
        self.pool.preload()

        self.experts = _build_experts(self.pool, registry)
        expert_count = sum(len(v) for v in self.experts.values())
        print(
            f"[ExpertWorkerActor:{worker_id}] Ready — "
            f"{expert_count} expert(s) loaded on GPU."
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(
        self,
        task_key: str,
        language: str,
        text: str,
        description: str,
        adapter_name: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Run LLM inference for the given task + language.

        Called by the coordinator via ``actor.classify.remote(...)``.
        Ray serializes arguments and return value across nodes automatically.
        The GPU semaphore from the old FastAPI worker is no longer needed —
        Ray serializes calls to the same actor automatically (each actor runs
        in a single thread by default).
        """
        if not self.pool.is_ready():
            raise RuntimeError(f"Worker {self.worker_id}: model not ready")

        domain, task = task_key.split("/", 1)
        if domain not in self.experts or task not in self.experts[domain]:
            raise ValueError(
                f"Task '{task_key}' not handled by this worker "
                f"(model: {self.pool.assigned_model_key})"
            )

        start = time.perf_counter()
        prompt = f"{description}\n\n{text}"
        input_data = {"text": text}

        expert = self.experts[domain][task]
        prediction = expert.predict(input_data, prompt, language)

        if isinstance(prediction, tuple) and len(prediction) >= 3:
            result, confidence, raw_response = (
                prediction[0], prediction[1], prediction[2]
            )
        else:
            raise RuntimeError("Unexpected prediction format from expert")

        processing_time_ms = (time.perf_counter() - start) * 1000

        return {
            "result": str(result),
            "confidence": float(confidence) if confidence else 0.0,
            "processing_time_ms": processing_time_ms,
            "worker_id": self.worker_id,
            "model_key": self.model_key,
            "request_id": request_id,
            "raw_response": str(raw_response) if raw_response else None,
        }

    # ------------------------------------------------------------------
    # Admin / health
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self.pool.is_ready()

    def get_info(self) -> Dict[str, Any]:
        """Return worker metadata (used by admin/health endpoints)."""
        return {
            "worker_id": self.worker_id,
            "model_key": self.model_key,
            "is_ready": self.pool.is_ready(),
            "tasks": [
                f"{domain}/{task}"
                for domain, tasks in self.experts.items()
                for task in tasks
            ],
        }
