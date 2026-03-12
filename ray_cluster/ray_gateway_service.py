"""
RayGatewayService — replaces app/services/gateway_service.py.

Instead of dispatching inference requests over HTTP to separate FastAPI
processes, this service calls Ray Actor methods directly.

What Ray replaces compared to the old GatewayService
-----------------------------------------------------
Old (HTTP-based):
  coordinator → httpx.AsyncClient.post(worker_url/expert/classify) → worker FastAPI

New (Ray-based):
  coordinator → await actor.classify.remote(...) → Ray cluster routes to actor's GPU node

Improvements over the first version
-------------------------------------
1. ``await result_ref`` instead of ``asyncio.to_thread(ray.get, ref)``
   ExpertWorkerActor.classify() is now an async method, so Ray exposes an
   awaitable ObjectRef.  The coordinator can ``await`` it directly inside
   a FastAPI async handler — no thread-pool hop needed.

2. Round-robin ActorPool per model
   spawn_workers.py now returns Dict[str, List[str]] (model_key → list of
   actor names).  When num_replicas > 1 in ray_worker_registry.json, multiple
   actor instances back the same model.  RayGatewayService distributes requests
   across replicas in round-robin order so no single GPU is a bottleneck.

   Example config (ray_worker_registry.json):
     "worker-0": {"base_model_key": "llama-2-7b-hf", "num_replicas": 2, ...}
   This spawns expert-worker-worker-0-r0 and expert-worker-worker-0-r1.
   Consecutive requests alternate between the two.
"""

from typing import Dict, Any, List

import ray


class RayGatewayService:
    """
    Routes expert inference requests to named Ray Actors with round-robin
    load balancing across replicas.

    Usage
    -----
    gateway = RayGatewayService({"llama-2-7b-hf": ["expert-worker-worker-0-r0",
                                                    "expert-worker-worker-0-r1"]})
    result  = await gateway.dispatch("llama-2-7b-hf", payload)
    """

    def __init__(self, model_to_actors: Dict[str, List[str]]):
        """
        Parameters
        ----------
        model_to_actors : dict
            Mapping of base_model_key → list of Ray actor names.
            Built by spawn_workers.py at startup from ray_worker_registry.json.
            A single-replica model has a list of length 1.
        """
        self._model_to_actors: Dict[str, List[str]] = model_to_actors
        # Cache actor handles so we don't call ray.get_actor() on every request
        self._actor_cache: Dict[str, Any] = {}
        # Per-model round-robin counters
        self._rr_index: Dict[str, int] = {k: 0 for k in model_to_actors}

        total_actors = sum(len(v) for v in model_to_actors.values())
        print(
            f"[RayGatewayService] Initialized with "
            f"{len(model_to_actors)} model(s), {total_actors} actor replica(s): "
            f"{model_to_actors}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_actor(self, actor_name: str) -> Any:
        """Return a cached Ray actor handle, looking it up by name if needed."""
        if actor_name not in self._actor_cache:
            self._actor_cache[actor_name] = ray.get_actor(actor_name)
        return self._actor_cache[actor_name]

    def _pick_actor(self, base_model_key: str) -> Any:
        """
        Pick the next actor for base_model_key using round-robin.
        Invalidates the cache entry if the actor is unreachable.
        """
        actor_names = self._model_to_actors[base_model_key]
        idx = self._rr_index[base_model_key] % len(actor_names)
        self._rr_index[base_model_key] = idx + 1
        return self._get_actor(actor_names[idx])

    # ------------------------------------------------------------------
    # Public API — same interface as the old GatewayService
    # ------------------------------------------------------------------

    async def dispatch(
        self, base_model_key: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Forward an inference request to a Ray actor that owns base_model_key.

        ExpertWorkerActor.classify() is async, so actor.classify.remote()
        returns an awaitable ObjectRef.  ``await result_ref`` suspends the
        FastAPI coroutine until Ray delivers the result — no thread pool needed.

        Round-robin across replicas: if the selected actor is unreachable its
        handle is evicted from the cache so the next call re-resolves it.

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
        if base_model_key not in self._model_to_actors:
            raise KeyError(
                f"No Ray actor registered for base_model_key='{base_model_key}'. "
                "Check that spawn_workers completed successfully."
            )

        actor = self._pick_actor(base_model_key)

        result_ref = actor.classify.remote(
            task_key=payload["task_key"],
            language=payload["language"],
            text=payload["text"],
            description=payload["description"],
            adapter_name=payload["adapter_name"],
            request_id=payload["request_id"],
        )

        # ExpertWorkerActor.classify is async → ObjectRef is directly awaitable.
        # This is equivalent to asyncio.to_thread(ray.get, ref) but avoids the
        # thread-pool overhead.
        return await result_ref

    async def health_check_all(self) -> Dict[str, Any]:
        """
        Check health of all registered actor replicas by calling get_info().
        Returns dict mapping actor_name → health status.
        """
        results: Dict[str, Any] = {}
        for actor_names in self._model_to_actors.values():
            for actor_name in actor_names:
                try:
                    actor = self._get_actor(actor_name)
                    info = await actor.get_info.remote()
                    results[actor_name] = {"status": "ok", "info": info}
                except (ValueError, RuntimeError, ray.exceptions.RayError) as exc:
                    # ValueError  — actor not found by name (not yet scheduled)
                    # RayError    — actor unreachable / crashed / restarting
                    # RuntimeError — actor returned an error result
                    self._actor_cache.pop(actor_name, None)
                    results[actor_name] = {"status": "unreachable", "error": str(exc)}
        return results

    def get_worker_info(self) -> Dict[str, Any]:
        """
        Return the worker registry in the same format as the old GatewayService,
        so that existing admin/health endpoints work without changes.
        """
        workers = {}
        for model_key, actor_names in self._model_to_actors.items():
            for actor_name in actor_names:
                workers[actor_name] = {
                    "actor_name": actor_name,
                    "base_model_key": model_key,
                    "num_replicas": len(actor_names),
                    "type": "ray_actor",
                }
        return {
            "workers": workers,
            "model_to_actors": self._model_to_actors,
        }
