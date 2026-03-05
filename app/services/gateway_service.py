"""
Gateway service — coordinator-side HTTP dispatcher.

After the lightweight gating pipeline resolves the required base_model_key,
the gateway looks up which expert worker hosts that model and forwards the
inference request there via HTTP.

The mapping is read once at startup from ``config/expert_machine_mapping.json``
and is static for the lifetime of the process.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx


class GatewayService:
    """
    Routes expert inference requests to the correct worker machine.

    Usage
    -----
    gateway = GatewayService("/app/config/expert_machine_mapping.json")
    result  = await gateway.dispatch("llama-2-7b-hf", payload)
    """

    def __init__(self, mapping_path: str):
        path = Path(mapping_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Expert machine mapping not found: {mapping_path}\n"
                "Create config/expert_machine_mapping.json or set EXPERT_MAPPING_PATH."
            )
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        self._model_to_worker: Dict[str, str] = mapping["model_to_worker"]
        self._workers: Dict[str, Dict] = mapping["workers"]
        self._drain_tasks: List[asyncio.Task] = []
        print(
            f"[GatewayService] Loaded mapping: {len(self._workers)} worker(s) registered."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_worker_id(self, base_model_key: str) -> str:
        """Return the worker_id responsible for the given base model key."""
        worker_id = self._model_to_worker.get(base_model_key)
        if worker_id is None:
            raise KeyError(
                f"No worker registered for base_model_key='{base_model_key}'. "
                f"Check config/expert_machine_mapping.json."
            )
        return worker_id

    def start_drain_loops(self, job_store) -> None:
        """
        Launch one asyncio background Task per registered worker.

        Each task does a blocking BRPOP on that worker's Redis queue, forwards
        the job to the worker HTTP endpoint, and stores the result back in Redis.
        Must be called from inside a running event loop (i.e. from app lifespan).
        """
        for worker_id in self._workers:
            task = asyncio.create_task(
                self._drain_worker(worker_id, job_store),
                name=f"drain-{worker_id}",
            )
            self._drain_tasks.append(task)
        print(f"[GatewayService] Started {len(self._drain_tasks)} drain loop(s).")

    def stop_drain_loops(self) -> None:
        """Cancel all drain-loop tasks (called on app shutdown)."""
        for task in self._drain_tasks:
            task.cancel()
        self._drain_tasks.clear()

    async def _drain_worker(self, worker_id: str, job_store) -> None:
        """
        Continuously drain the job queue for one worker.

        BRPOP blocks for up to 5 s then loops — this keeps the coroutine
        responsive to CancelledError without holding Redis connections open
        indefinitely.
        """
        queue_key = f"job_queue:{worker_id}"
        endpoint = f"{self._workers[worker_id]['url']}/api/v1/expert/classify"
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=5.0)
        print(f"[GatewayService] Drain loop running for {worker_id}")

        while True:
            job_id = None
            try:
                item = await job_store._redis.brpop(queue_key, timeout=5)
                if item is None:
                    continue

                _, raw = item
                entry = json.loads(raw)
                job_id = entry.pop("job_id")

                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(endpoint, json=entry)
                    resp.raise_for_status()
                    worker_result = resp.json()

                await job_store.set_result(job_id, {
                    "result": worker_result.get("result"),
                    "confidence": worker_result.get("confidence"),
                    "processing_time_ms": worker_result.get("processing_time_ms"),
                    "worker_id": worker_result.get("worker_id"),
                    "model_key": worker_result.get("model_key"),
                })

            except asyncio.CancelledError:
                print(f"[GatewayService] Drain loop for {worker_id} stopped.")
                break
            except Exception as exc:
                print(f"[GatewayService] Drain error for {worker_id}: {exc}")
                if job_id:
                    try:
                        await job_store.set_error(job_id, str(exc))
                    except Exception:
                        pass

    async def dispatch(self, base_model_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward an expert inference request to the worker that owns ``base_model_key``.

        Parameters
        ----------
        base_model_key : str
            Key from experts_registry.json (e.g. ``"llama-2-7b-hf"``).
        payload : dict
            JSON body to POST to the worker's /api/v1/expert/classify endpoint.

        Returns
        -------
        dict
            Parsed JSON response from the worker.

        Raises
        ------
        KeyError
            If no worker is registered for the given model key.
        httpx.HTTPStatusError
            If the worker returns a non-2xx response.
        """
        worker_id = self._model_to_worker.get(base_model_key)
        if worker_id is None:
            raise KeyError(
                f"No worker registered for base_model_key='{base_model_key}'. "
                f"Check config/expert_machine_mapping.json."
            )

        worker_url = self._workers[worker_id]["url"]
        endpoint = f"{worker_url}/api/v1/expert/classify"

        timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()

    async def health_check_all(self) -> Dict[str, Any]:
        """
        Ping the /api/v1/health/ready endpoint of every registered worker.
        Returns a dict mapping worker_id → health status.
        """
        results: Dict[str, Any] = {}
        timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for worker_id, worker in self._workers.items():
                url = f"{worker['url']}/api/v1/health/ready"
                try:
                    r = await client.get(url)
                    results[worker_id] = {"status": "ok", "http_status": r.status_code}
                except Exception as exc:
                    results[worker_id] = {"status": "unreachable", "error": str(exc)}
        return results

    def get_worker_info(self) -> Dict[str, Any]:
        """Return the full worker registry (for admin/status endpoints)."""
        return {
            "workers": self._workers,
            "model_to_worker": self._model_to_worker,
        }


# Global singleton — initialized by app/main.py during startup
gateway_service: GatewayService = None  # type: ignore[assignment]
