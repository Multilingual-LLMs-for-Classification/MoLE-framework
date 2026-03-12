# Ray Integration Improvements — Changelog

**Date:** 2026-03-12
**Scope:** `ray_cluster/`, `app/config.py`, `app/services/routing_service.py`, `config/`

---

## Summary

Seven targeted improvements made to the initial Ray integration to reduce overhead,
improve observability, enable horizontal scaling, and decouple Ray-specific config
from the HTTP-mode setup.

---

## Changes

### 1. `await result_ref` — remove thread-pool hop in gateway dispatch

**File:** `ray_cluster/ray_gateway_service.py`

`ExpertWorkerActor.classify()` was made async (see change 3). Async actor methods
return an awaitable `ObjectRef`, so the coordinator can `await` it directly.

```python
# Before
return await asyncio.to_thread(ray.get, result_ref)

# After
return await result_ref
```

The old pattern used `asyncio.to_thread` to run blocking `ray.get()` in a thread
pool so the FastAPI event loop was not blocked. With an async actor the `ObjectRef`
integrates with the event loop natively — no thread needed.

---

### 2. Ray metrics on every worker actor

**File:** `ray_cluster/worker_actor.py`

Two metrics are now published per actor using `ray.util.metrics`. They appear in
the Ray dashboard **Metrics** tab and are scraped by any Prometheus endpoint attached
to the cluster.

| Metric name | Type | Tags |
|---|---|---|
| `mole_expert_requests_total` | Counter | `worker_id`, `model_key`, `status` (`ok` / `error`) |
| `mole_expert_latency_ms` | Histogram | `worker_id`, `model_key` |

Boundaries for the latency histogram: 50 ms, 100 ms, 200 ms, 500 ms, 1 s, 2 s, 5 s, 10 s.

No external metric infrastructure required — Ray ships the Prometheus endpoint with
`ray[default]`.

---

### 3. Explicit `max_concurrency=1` + async `classify()`

**File:** `ray_cluster/worker_actor.py`

```python
# Before
@ray.remote(num_gpus=1)
class ExpertWorkerActor:
    def classify(self, ...) -> dict: ...

# After
@ray.remote(num_gpus=1, max_concurrency=1)
class ExpertWorkerActor:
    async def classify(self, ...) -> dict: ...
        # sync GPU inference dispatched to thread executor
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._sync_classify, ...
        )
```

`max_concurrency=1` was always the effective default but is now explicit so the
GPU serialisation guarantee is visible without a comment.

Making `classify` async lets the actor's event loop remain responsive to health
check calls (`is_ready`, `get_info`) while inference runs in the executor — they
no longer queue behind the GPU call.

---

### 4. Dedicated Ray config file (`ray_worker_registry.json`)

**Files:** `config/ray_worker_registry.json` *(new)*, `ray_cluster/spawn_workers.py`,
`app/config.py`

`expert_machine_mapping.json` had `url`, `machine`, and `note` fields that are
meaningless in Ray mode. A new file `config/ray_worker_registry.json` is used
exclusively when `USE_RAY=true`.

Fields per worker entry:

| Field | Required | Description |
|---|---|---|
| `base_model_key` | yes | Must match a key in `experts_registry.json` |
| `num_replicas` | no (default 1) | How many actor replicas to spawn |
| `num_gpus` | no (default 1) | GPU slots per replica |
| `placement_group_strategy` | no | Ray placement group strategy (see change 6) |

`expert_machine_mapping.json` is unchanged and still consumed by `GatewayService`
in HTTP mode.

New env var: `RAY_WORKER_REGISTRY_PATH` (default: `config/ray_worker_registry.json`).

---

### 5. Round-robin replica pool

**Files:** `config/ray_worker_registry.json`, `ray_cluster/spawn_workers.py`,
`ray_cluster/ray_gateway_service.py`

`spawn_workers` now supports `num_replicas > 1` per worker entry. Each replica
is a separate actor with its own GPU slot, named `expert-worker-<id>-r<N>`.

`RayGatewayService` was updated to accept `Dict[str, List[str]]`
(model_key → list of actor names) and distributes requests across replicas in
round-robin order using a per-model counter.

Example — two replicas of the highest-traffic model:

```json
"worker-0": {
  "base_model_key": "llama-2-7b-hf",
  "num_replicas": 2,
  "num_gpus": 1
}
```

Spawns `expert-worker-worker-0-r0` and `expert-worker-worker-0-r1` on two
separate GPU slots. Requests alternate between them.

`spawn_workers` return type changed from `Dict[str, str]` to `Dict[str, List[str]]`.

---

### 6. Placement group support

**File:** `ray_cluster/spawn_workers.py`

An optional `placement_group_strategy` field in `ray_worker_registry.json` pins
an actor's replicas to nodes using a Ray placement group:

```json
"worker-0": {
  "base_model_key": "llama-2-7b-hf",
  "num_replicas": 1,
  "num_gpus": 1,
  "placement_group_strategy": "STRICT_PACK"
}
```

Supported values: `STRICT_PACK`, `PACK`, `SPREAD`, `STRICT_SPREAD`. Omit the field
to let Ray schedule freely (recommended for most deployments).

Use cases:
- `STRICT_PACK` — force all replicas of a model onto the same physical node
- `SPREAD` — force replicas onto different nodes for fault isolation

---

### 7. GatingActor — optional fractional GPU for the gating pipeline

**Files:** `ray_cluster/gating_actor.py` *(new)*, `app/services/routing_service.py`,
`app/config.py`

The gating pipeline (FastText → XLM-RoBERTa → Q-learning) normally runs in the
coordinator process on CPU. When `USE_GATING_ACTOR=true`, it is moved to a
dedicated Ray Actor:

- Named `mole-gating-actor`, `lifetime="detached"`, `max_restarts=-1`
- `run_gating(prompt)` is `async` — runs sync transformer inference in an executor
- Spawned with `.options(num_gpus=GATING_ACTOR_NUM_GPUS)` for fractional GPU

GPU fraction options:

| `GATING_ACTOR_NUM_GPUS` | Effect |
|---|---|
| `0.0` | CPU only (default — safe on any machine, no GPU required) |
| `0.1` | XLM-RoBERTa runs on GPU; 10 gating actors share one GPU slice |

`routing_service._classify_coordinator()` branches on whether the gating actor
is initialised:

```python
if self._gating_actor is not None:
    gating_dict = await self._gating_actor.run_gating.remote(prompt)
    gating = GatingResult(**gating_dict)
else:
    gating = await loop.run_in_executor(None, self._routing_system.run_gating, prompt)
```

When the actor is not configured the code path is identical to before — no
regression for the default setup.

---

## New environment variables

| Variable | Default | Description |
|---|---|---|
| `RAY_WORKER_REGISTRY_PATH` | `config/ray_worker_registry.json` | Path to Ray-native worker config |
| `USE_GATING_ACTOR` | `false` | Move gating pipeline to a Ray Actor |
| `GATING_ACTOR_NUM_GPUS` | `0.0` | GPU fraction for the gating actor |

All existing variables (`USE_RAY`, `RAY_ADDRESS`, `EXPERT_REGISTRY_PATH`, etc.) are
unchanged.

---

## Files changed

| File | Type | Change |
|---|---|---|
| `config/ray_worker_registry.json` | new | Ray-only worker config with `num_replicas` and placement group fields |
| `ray_cluster/gating_actor.py` | new | `GatingActor` Ray Actor + `spawn_gating_actor()` helper |
| `ray_cluster/worker_actor.py` | modified | `max_concurrency=1`, async `classify()`, Ray metrics |
| `ray_cluster/ray_gateway_service.py` | modified | `await result_ref`, round-robin over `List[str]` replicas, specific exception types |
| `ray_cluster/spawn_workers.py` | modified | Reads `ray_worker_registry.json`, `num_replicas`, placement groups, argparse CLI, returns `Dict[str, List[str]]` |
| `app/config.py` | modified | Added `ray_worker_registry_path`, `use_gating_actor`, `gating_actor_num_gpus` |
| `app/services/routing_service.py` | modified | Uses `ray_worker_registry_path`, wires `GatingActor`, gating phase branches on actor presence |

---

## What was NOT changed

- All ML core logic (`moe_router/gating/`, `moe_router/experts/`, `expert_worker/`)
- All API endpoints and schemas (`app/routers/`, `app/schemas/`)
- `config/expert_machine_mapping.json` — still used as-is for HTTP mode
- `docker/docker-compose-ray.yml` and `docker/docker-compose-ray-worker.yml`
- Legacy HTTP mode (`USE_RAY=false`) — completely unaffected
