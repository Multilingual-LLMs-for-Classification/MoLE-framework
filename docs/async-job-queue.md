# Async Job Queue — Coordinator→Worker Dispatch

## Problem

The coordinator→worker communication is currently fully synchronous:

```
Client → POST /api/v1/classify
           └─ gating pipeline (CPU, ~50ms)
           └─ httpx.post(worker) ← blocks here, waiting for LLM inference
                └─ worker GPU semaphore serializes requests
                └─ inference runs (~5–30s for 7B model)
           └─ return ClassifyResponse to client
```

Under load, multiple requests to the same worker all hold open HTTP connections to the
coordinator — one per in-flight request — waiting for the GPU to finish. This means:

- Client connections are held open for minutes
- Coordinator memory scales with number of waiting requests
- A single slow request blocks all subsequent ones behind the semaphore
- Long-lived HTTP connections are fragile across proxies and load balancers

---

## Broker Decision

| Option | Decision | Reason |
|--------|----------|--------|
| **Kafka** | No | Overkill. Needs ZooKeeper/KRaft, schema registry, multi-broker setup. Designed for millions of msg/sec with replay semantics — none of which are needed here. Operating cost far exceeds benefit for 7 workers. |
| **RabbitMQ** | No | More infra than needed. AMQP overhead, management UI, separate container. Good choice at larger scale but unnecessary here. |
| **asyncio.Queue (in-process)** | No | Cannot bridge machines. Workers run on separate hosts — an in-process queue on the coordinator can't reach them. |
| **Redis List (LPUSH / BRPOP)** | **Yes** | Single `redis:7-alpine` container (~20MB RAM). Python `redis[asyncio]` client is excellent. Supports per-worker queues, atomic pop, result storage with TTL, zero config. Already standard in ML serving stacks. |

---

## Architecture After This Change

```
Client
  │
  │  POST /api/v1/classify
  ▼
Coordinator
  ├─ gating pipeline (FastText → XLM-R → Q-learning)   ← unchanged
  ├─ resolves worker_id from base_model_key              ← unchanged
  ├─ pushes job JSON to Redis List                       ← NEW
  │    key:   "job_queue:{worker_id}"
  │    value: {job_id, task_key, language, text, ...}
  ├─ stores initial status in Redis                      ← NEW
  │    key: "job:{job_id}" → {"status": "queued"}
  └─ returns 202 {"job_id": "...", "status": "queued", "poll_url": "..."}
         ▲
         │ Client polls: GET /api/v1/classify/{job_id}   ← NEW endpoint

Coordinator — drain-loop coroutine (one per worker, runs in background)   ← NEW
  └─ BRPOP "job_queue:{worker_id}"   (blocks up to 5s, then loops)
  └─ httpx.post(worker /api/v1/expert/classify, payload)   ← same call as before
  └─ store result: "job:{job_id}" → {"status": "done", "result": ..., ...}
  └─ on error: "job:{job_id}" → {"status": "error", "detail": ...}

Worker                                                    ← unchanged
  └─ gpu_semaphore(1) serializes GPU inference
  └─ POST /api/v1/expert/classify endpoint unchanged
```

---

## What Changes

### New files

| File | Purpose |
|------|---------|
| `app/services/job_store.py` | Redis-backed job state: enqueue, get_status, set_result, set_error |
| `app/routers/jobs.py` | `GET /api/v1/classify/{job_id}` — polling endpoint, requires auth |

### Modified files

| File | Change |
|------|--------|
| `app/config.py` | Add `REDIS_URL = "redis://localhost:6379"` and `JOB_TTL_SECONDS = 3600` |
| `app/services/gateway_service.py` | Add `start_drain_loops(job_store)` — launches one asyncio.Task per worker |
| `app/routers/classify.py` | Return `202 + job_id` immediately instead of blocking for result |
| `app/main.py` | Start drain loops in lifespan after gateway init; cancel on shutdown |
| `app/schemas/responses.py` | Add `JobAcceptedResponse` and `JobStatusResponse` schemas |
| `docker/docker-compose-distributed.yml` | Add `redis:7-alpine` service with `network_mode: host` |
| `requirements.txt` | Add `redis[asyncio]>=5.0.0` |

### What does NOT change

- `expert_worker/router.py` — GPU semaphore and inference path unchanged
- `expert_worker/main.py` — worker startup unchanged
- Gating pipeline — fully unchanged
- Worker HTTP API — drain loop calls the same `/api/v1/expert/classify` endpoint

---

## Key Implementation Details

### `app/services/job_store.py`

```python
async def enqueue(self, worker_id: str, job_id: str, payload: dict):
    payload["job_id"] = job_id
    await self._redis.lpush(f"job_queue:{worker_id}", json.dumps(payload))
    await self._redis.setex(f"job:{job_id}", settings.job_ttl_seconds,
                            json.dumps({"status": "queued"}))

async def get_status(self, job_id: str) -> dict | None:
    raw = await self._redis.get(f"job:{job_id}")
    return json.loads(raw) if raw else None
```

### Drain loop in `GatewayService`

```python
async def _drain_worker(self, worker_id: str, job_store):
    queue_key = f"job_queue:{worker_id}"
    endpoint = f"{self._workers[worker_id]['url']}/api/v1/expert/classify"
    while True:
        try:
            item = await job_store._redis.brpop(queue_key, timeout=5)
            if item is None:
                continue
            _, raw = item
            payload = json.loads(raw)
            job_id = payload.pop("job_id")

            async with httpx.AsyncClient(timeout=...) as client:
                resp = await client.post(endpoint, json=payload)
                resp.raise_for_status()
                await job_store.set_result(job_id, resp.json())
        except asyncio.CancelledError:
            break
        except Exception as exc:
            await job_store.set_error(job_id, str(exc))
```

`BRPOP timeout=5` means the loop wakes up every 5 seconds at minimum, which allows
clean shutdown via `CancelledError` without hanging.

### Classify endpoint

```
Before:  POST /api/v1/classify  →  200 ClassifyResponse  (after full inference)
After:   POST /api/v1/classify  →  202 {"job_id": "...", "status": "queued"}
         GET  /api/v1/classify/{job_id}  →  {"status": "queued"|"done"|"error", ...}
```

---

## Redis Key Schema

| Key | TTL | Value |
|-----|-----|-------|
| `job_queue:{worker_id}` | none (list) | JSON job payloads (FIFO via LPUSH/BRPOP) |
| `job:{job_id}` | 1 hour | `{"status": "queued"}` / `{"status": "done", "result": ..., "confidence": ...}` / `{"status": "error", "detail": ...}` |

---

## Verification Steps

1. Start the stack including Redis:
   ```bash
   cd docker
   docker-compose -f docker-compose-distributed.yml up -d
   ```

2. Register + login and get a token (see README Step 7)

3. Submit a request — should return immediately:
   ```bash
   curl -X POST http://localhost:8000/api/v1/classify \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"text": "Revenue up 20%.", "description": "Classify sentiment 1-5."}'
   # Expected: 202 {"job_id": "abc-123", "status": "queued", "poll_url": "/api/v1/classify/abc-123"}
   ```

4. Poll for result:
   ```bash
   curl http://localhost:8000/api/v1/classify/abc-123 \
     -H "Authorization: Bearer $TOKEN"
   # Initially: {"status": "queued"}
   # After inference: {"status": "done", "result": "4", "confidence": 0.91, ...}
   ```

5. Concurrent load test — send 5 requests simultaneously:
   ```bash
   for i in {1..5}; do
     curl -s -X POST http://localhost:8000/api/v1/classify \
       -H "Authorization: Bearer $TOKEN" \
       -H "Content-Type: application/json" \
       -d '{"text": "Test text.", "description": "Classify."}' &
   done
   wait
   # All 5 should return job_ids instantly
   # Results arrive sequentially (GPU semaphore still serializes inference)
   ```

6. Inspect Redis directly:
   ```bash
   docker exec -it $(docker ps -qf name=redis) redis-cli keys "job:*"
   docker exec -it $(docker ps -qf name=redis) redis-cli get "job:abc-123"
   ```
