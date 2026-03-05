"""
Redis-backed job store for async inference requests.

Jobs flow through three states:
  queued  — coordinator enqueued the job, drain-loop hasn't picked it up yet
  done    — worker returned a result successfully
  error   — worker or network error occurred

Results are stored with a TTL (default 1 hour) so Redis doesn't grow unbounded.
"""

import json
from uuid import uuid4

import redis.asyncio as aioredis

from app.config import settings


class JobStore:
    """
    Thin wrapper around Redis that manages job lifecycle.

    connect() / close() are called from app lifespan (app/main.py).
    enqueue() is called from the classify router after gating.
    set_result() / set_error() are called from GatewayService drain loops.
    get_status() is called from the jobs polling router.
    """

    def __init__(self):
        self._redis: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self):
        try:
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            await self._redis.ping()
            print(f"[JobStore] Connected to Redis at {settings.redis_url}")
        except Exception as exc:
            print(f"[JobStore] WARNING: Could not connect to Redis: {exc}")
            self._redis = None

    async def close(self):
        if self._redis:
            await self._redis.aclose()

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def new_job_id() -> str:
        return str(uuid4())

    async def enqueue(
        self,
        worker_id: str,
        job_id: str,
        worker_payload: dict,
        job_metadata: dict,
    ) -> None:
        """
        Push a job onto the per-worker Redis List and store the initial record.

        worker_payload  — sent as-is to the expert worker HTTP endpoint
        job_metadata    — gating results stored in the job record for the final response
                          (language, domain, task, routing_path, request_id)
        """
        queue_entry = {"job_id": job_id, **worker_payload}
        await self._redis.lpush(f"job_queue:{worker_id}", json.dumps(queue_entry))
        await self._redis.setex(
            f"job:{job_id}",
            settings.job_ttl_seconds,
            json.dumps({"status": "queued", **job_metadata}),
        )

    async def set_result(self, job_id: str, worker_result: dict) -> None:
        """Merge worker result into the job record and mark it done."""
        raw = await self._redis.get(f"job:{job_id}")
        record = json.loads(raw) if raw else {}
        record.update({"status": "done", **worker_result})
        await self._redis.setex(
            f"job:{job_id}",
            settings.job_ttl_seconds,
            json.dumps(record),
        )

    async def set_error(self, job_id: str, detail: str) -> None:
        raw = await self._redis.get(f"job:{job_id}")
        record = json.loads(raw) if raw else {}
        record.update({"status": "error", "detail": detail})
        await self._redis.setex(
            f"job:{job_id}",
            settings.job_ttl_seconds,
            json.dumps(record),
        )

    async def get_status(self, job_id: str) -> dict | None:
        raw = await self._redis.get(f"job:{job_id}")
        return json.loads(raw) if raw else None


# Global singleton — connected/closed by app lifespan (app/main.py)
job_store = JobStore()
