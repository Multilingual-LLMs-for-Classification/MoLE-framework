"""
RayGatewayService — replaces app/services/gateway_service.py.

Instead of dispatching inference requests over HTTP to separate FastAPI
processes, this service calls Ray Actor methods directly.

What Ray replaces compared to the old GatewayService
-----------------------------------------------------
Old (HTTP-based):
  coordinator → httpx.AsyncClient.post(worker_url/expert/classify) → worker FastAPI

New (Ray-based):
  coordinator → actor.classify.remote(...) → Ray cluster routes to actor's GPU node

Benefits:
  - No HTTP overhead, no serialization to JSON over the wire
  - No manual worker URL registry (expert_machine_mapping.json no longer needed
    for host/port — only for model→actor_name mapping)
  - No Redis queue / drain loops / job polling
  - Ray handles fault tolerance: if a worker node dies, Ray restarts the actor
  - Actor discovery by name works across all nodes in the cluster
"""

import asyncio
from typing import Dict, Any

import ray


class RayGatewayService:
    """
    Routes expert inference requests to named Ray Actors.

    Usage
    -----
    gateway = RayGatewayService({"llama-2-7b-hf": "expert-worker-worker-0"})
    result  = await gateway.dispatch("llama-2-7b-hf", payload)
    """

    def __init__(self, model_to_actor: Dict[str, str]):
        """
        Parameters
        ----------
        model_to_actor : dict
            Mapping of base_model_key → Ray actor name.
            Built by spawn_workers.py at startup from expert_machine_mapping.json.
        """
        self._model_to_actor: Dict[str, str] = model_to_actor
        # Cache actor handles so we don't call ray.get_actor() on every request
        self._actor_cache: Dict[str, Any] = {}
        print(
            f"[RayGatewayService] Initialized with "
            f"{len(model_to_actor)} actor(s): {list(model_to_actor.values())}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_actor(self, actor_name: str):
        """Return a cached Ray actor handle, looking it up by name if needed."""
        if actor_name not in self._actor_cache:
            self._actor_cache[actor_name] = ray.get_actor(actor_name)
        return self._actor_cache[actor_name]

    # ------------------------------------------------------------------
    # Public API — same interface as the old GatewayService
    # ------------------------------------------------------------------

    async def dispatch(
        self, base_model_key: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Forward an inference request to the Ray actor that owns base_model_key.

        The actor's classify() method runs synchronously inside the actor's
        dedicated thread on whichever GPU node Ray scheduled it on.
        ray.get() blocks until the result is ready; we run it in a thread pool
        (asyncio.to_thread) so the coordinator's async event loop is not blocked.

        Parameters
        ----------
        base_model_key : str
            Key from experts_registry.json (e.g. "llama-2-7b-hf").
        payload : dict
            Must contain: task_key, language, text, description,
                          adapter_name, request_id.

        Returns
        -------
        dict
            Result dict from ExpertWorkerActor.classify().
        """
        actor_name = self._model_to_actor.get(base_model_key)
        if actor_name is None:
            raise KeyError(
                f"No Ray actor registered for base_model_key='{base_model_key}'. "
                "Check that spawn_workers completed successfully."
            )

        actor = self._get_actor(actor_name)

        # actor.classify.remote() returns a Ray ObjectRef (future)
        result_ref = actor.classify.remote(
            task_key=payload["task_key"],
            language=payload["language"],
            text=payload["text"],
            description=payload["description"],
            adapter_name=payload["adapter_name"],
            request_id=payload["request_id"],
        )

        # ray.get() is blocking — run in thread pool to keep event loop free
        return await asyncio.to_thread(ray.get, result_ref)

    async def health_check_all(self) -> Dict[str, Any]:
        """
        Check health of all registered actors by calling get_info().
        Returns dict mapping actor_name → health status.
        """
        results: Dict[str, Any] = {}
        for model_key, actor_name in self._model_to_actor.items():
            try:
                actor = self._get_actor(actor_name)
                info_ref = actor.get_info.remote()
                info = await asyncio.to_thread(ray.get, info_ref)
                results[actor_name] = {"status": "ok", "info": info}
            except Exception as exc:
                # Invalidate cached handle so next call re-resolves the actor
                self._actor_cache.pop(actor_name, None)
                results[actor_name] = {"status": "unreachable", "error": str(exc)}
        return results

    def get_worker_info(self) -> Dict[str, Any]:
        """
        Return the worker registry in the same format as the old GatewayService,
        so that existing admin/health endpoints work without changes.
        """
        workers = {
            actor_name: {
                "actor_name": actor_name,
                "base_model_key": model_key,
                "type": "ray_actor",
            }
            for model_key, actor_name in self._model_to_actor.items()
        }
        return {
            "workers": workers,
            "model_to_worker": self._model_to_actor,
        }
