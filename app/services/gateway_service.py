"""
Gateway service — coordinator-side HTTP dispatcher.

After the lightweight gating pipeline resolves the required base_model_key,
the gateway looks up which expert worker hosts that model and forwards the
inference request there via HTTP.

The mapping is read once at startup from ``config/expert_machine_mapping.json``
and is static for the lifetime of the process.
"""

import json
from pathlib import Path
from typing import Dict, Any

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
        print(
            f"[GatewayService] Loaded mapping: {len(self._workers)} worker(s) registered."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
