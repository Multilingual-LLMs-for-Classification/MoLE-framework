# Ray Migration: Replacing Custom Distributed Infrastructure with Ray

## Overview

The original MoLE distributed setup was built from scratch: expert workers ran as
separate FastAPI Docker containers, the coordinator dispatched to them over HTTP via
`GatewayService`, and a Redis queue handled async job tracking. This worked but
required maintaining a large amount of distributed-systems boilerplate that Ray already
provides as a battle-tested framework.

This document describes what was replaced, why Ray was chosen, the new architecture,
and how to operate the system.

---

## What Was Built From Scratch vs What Ray Provides

| Custom code (before) | Ray equivalent (after) |
|---|---|
| `app/services/gateway_service.py` — HTTP POST to worker URLs | `ray.remote()` actor method calls |
| Redis job queue + drain loops | Ray object store + `ObjectRef` futures |
| `job_store.py` — job polling + TTL expiry | `asyncio.to_thread(ray.get, ref)` |
| `config/expert_machine_mapping.json` host:port entries | Ray resource scheduling (`num_gpus=1`) |
| Per-worker FastAPI apps (`expert_worker/main.py`) | `ExpertWorkerActor` class (`@ray.remote`) |
| Manual health check endpoints on every worker | `ray.get_actor()` + actor method call |
| Docker container per worker | Named, detached Ray Actor per model |
| Manual service discovery (static JSON) | Ray GCS (Global Control Store) |
| Worker restart on crash: manual | `max_restarts=-1` on actor declaration |

---

## Why Ray

Several distributed frameworks were considered:

| Framework | Verdict |
|---|---|
| **Ray** | Best fit. GPU-aware actor model maps directly to one-model-per-actor pattern. Multi-node out of the box. Works inside Docker. |
| Celery + Redis | Task queue only — does not handle GPU affinity, actor state, or model residency. |
| KServe / Triton | Kubernetes-only. Overkill for this setup; hard to integrate custom gating pipeline. |
| BentoML | Good for single-model serving; not designed for a coordinator-routed multi-model setup. |
| Ray Serve | Considered for the coordinator too, but the existing FastAPI coordinator was kept unchanged to avoid regressions in auth, admin, and analytics routes. |

Key Ray capabilities that directly solve MoLE's problems:

- **`num_gpus=1` resource declaration**: Ray's scheduler places the actor on a node with a
  free GPU and sets `CUDA_VISIBLE_DEVICES` automatically — no manual host:port config.
- **Named actors (`lifetime="detached"`)**: Actors survive coordinator restarts and are
  discoverable by name across the entire cluster with `ray.get_actor(name)`.
- **Automatic serialization**: Arguments and return values are serialized by Ray over the
  cluster network — no manual JSON encoding or HTTP framing.
- **Fault tolerance**: `max_restarts=-1` tells Ray to restart a crashed actor indefinitely,
  reloading the LLM into GPU memory automatically.

---

## New Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  COORDINATOR MACHINE                                                 │
│                                                                      │
│  ┌──────────────────────┐    ┌─────────────────────────────────────┐ │
│  │  Ray Head Node       │    │  Coordinator (FastAPI, port 8000)   │ │
│  │  port 6379 (GCS)     │◄───│  - Gating pipeline (CPU)            │ │
│  │  port 8265 (dashboard│    │  - RayGatewayService                │ │
│  │  port 10001 (client) │    │  - JWT auth, admin, analytics       │ │
│  └──────────┬───────────┘    └──────────────┬──────────────────────┘ │
│             │                               │ actor.classify.remote() │
└─────────────┼───────────────────────────────┼────────────────────────┘
              │ Ray cluster protocol           │
    ┌─────────┼───────────────────────────────┼──────────────────────┐
    │ GPU MACHINE 1                           │                      │
    │  ┌──────┴──────────┐    ┌───────────────▼───────────────────┐  │
    │  │ Ray Worker Node │    │ ExpertWorkerActor (num_gpus=1)    │  │
    │  │ joined cluster  │    │ model: llama-2-7b-hf              │  │
    │  └─────────────────┘    │ GPU: assigned by Ray scheduler    │  │
    │                         └───────────────────────────────────┘  │
    └────────────────────────────────────────────────────────────────┘
              │
    ┌─────────┼──────────────────────────────────────────────────────┐
    │ GPU MACHINE 2           │                                      │
    │  ┌──────┴──────────┐    ┌───────────────────────────────────┐  │
    │  │ Ray Worker Node │    │ ExpertWorkerActor (num_gpus=1)    │  │
    │  │ joined cluster  │    │ model: mistral-7b                 │  │
    │  └─────────────────┘    └───────────────────────────────────┘  │
    └────────────────────────────────────────────────────────────────┘
```

### Request flow (Ray mode)

```
POST /api/v1/classify
        │
        ▼
Coordinator: run_gating(text)
   → language: "english"
   → domain:   "finance"
   → task:     "rating"
   → model:    "llama-2-7b-hf"
        │
        ▼
RayGatewayService.dispatch("llama-2-7b-hf", payload)
   → ray.get_actor("expert-worker-worker-0")
   → actor.classify.remote(task_key, language, text, ...)
   → await asyncio.to_thread(ray.get, result_ref)
        │
        ▼ (Ray routes call to the actor's GPU node)
ExpertWorkerActor.classify()
   → expert.predict(input_data, prompt, language)
   → return {"result": "4", "confidence": 0.94, ...}
        │
        ▼
HTTP 200 ClassifyResponse
```

No Redis. No polling. No HTTP between coordinator and workers.

---

## New Files

### `ray_cluster/worker_actor.py`

`ExpertWorkerActor` — a `@ray.remote(num_gpus=1)` class. On construction it calls
`SingleModelPool.preload()` (identical to the old worker startup). The `classify()`
method mirrors the logic in `expert_worker/router.py` but runs as a plain synchronous
method — Ray handles the async dispatch from the coordinator side.

Key difference from the old FastAPI worker: there is no GPU semaphore. Ray serializes
calls to the same actor automatically because each actor runs in a single dedicated
thread by default.

### `ray_cluster/ray_gateway_service.py`

`RayGatewayService` — drop-in replacement for `GatewayService`. Exposes the same
`dispatch()`, `health_check_all()`, and `get_worker_info()` methods so no changes
were needed in the routers or admin endpoints.

Internally, `dispatch()` calls `actor.classify.remote()` and awaits the result via
`asyncio.to_thread(ray.get, ref)`, which runs the blocking `ray.get()` in a thread
pool so the FastAPI event loop is not blocked.

### `ray_cluster/spawn_workers.py`

Called once during coordinator startup (from `routing_service.initialize()`).
Reads `config/expert_machine_mapping.json` and creates one named, detached
`ExpertWorkerActor` per worker entry. If an actor with the same name already exists
(e.g. coordinator restarted), it is reused — the LLM does not reload.

Can also be run standalone:
```bash
python -m ray_cluster.spawn_workers config/expert_machine_mapping.json
```

---

## Modified Files

### `app/config.py`

Three new settings (all set via environment variables):

| Variable | Default | Description |
|---|---|---|
| `USE_RAY` | `false` | Switch between Ray mode and legacy HTTP mode |
| `RAY_ADDRESS` | `auto` | Ray cluster address for `ray.init()` |
| `EXPERT_REGISTRY_PATH` | _(in-repo default)_ | Path to `experts_registry.json` passed to worker actors |

### `app/main.py`

Calls `ray.init(address=settings.ray_address)` before `routing_service.initialize()`
when `USE_RAY=true`. Calls `ray.shutdown()` on application shutdown.

### `app/services/routing_service.py`

In `initialize()`, when `service_mode=coordinator` and `use_ray=true`, creates a
`RayGatewayService` instead of the old `GatewayService`. The rest of the routing
logic (gating pipeline, `_classify_coordinator`, response building) is unchanged.

### `docker/Dockerfile`

- Added `psmisc` and `procps` system packages (required by Ray's process manager).
- Added `COPY ray_cluster/ ./ray_cluster/`.
- Added Ray ports to `EXPOSE` (6379, 8265, 10001).
- `ray[default]>=2.10.0` is installed via `requirements.txt`.

---

## Docker Compose Files

### `docker/docker-compose-ray.yml` — coordinator machine

Starts two services:

1. **`ray-head`** — Ray GCS + scheduler + dashboard. Runs on the coordinator machine
   with `--num-gpus=0` (CPU only; GPU reserved for worker actors on GPU nodes).
2. **`coordinator`** — the existing FastAPI app with `USE_RAY=true` and
   `RAY_ADDRESS=auto`. Connects to `ray-head` at startup, then `spawn_workers.py`
   creates actor handles for each entry in `expert_machine_mapping.json`.

### `docker/docker-compose-ray-worker.yml` — GPU worker machines

Starts one service:

- **`ray-worker`** — joins the Ray cluster at `RAY_HEAD_ADDRESS`. Declares `--num-gpus=N`.
  Once joined, the coordinator's scheduler will place `ExpertWorkerActor` instances
  here automatically.

---

## Deployment

### Single machine (coordinator + GPU on same host)

```bash
cd /home/cse/Desktop/MoLE-framework
docker-compose -f docker/docker-compose-ray.yml up --build
```

Ray will schedule all worker actors on the local GPU since the head node is the only
node with GPUs available.

### Multi-machine (coordinator + N remote GPU nodes)

**Step 1 — Start coordinator machine:**
```bash
docker-compose -f docker/docker-compose-ray.yml up --build
```

**Step 2 — On each GPU worker machine, join the cluster:**
```bash
RAY_HEAD_ADDRESS=<coordinator-ip>:6379 \
docker-compose -f docker/docker-compose-ray-worker.yml up --build
```

**Step 3 — Verify the cluster:**
```bash
# From any node in the cluster
ray status --address <coordinator-ip>:6379
```

**Step 4 — Check the Ray dashboard:**

Open `http://<coordinator-ip>:8265` in a browser. All nodes, actors, GPU utilization,
and task history are visible there.

### Backward compatibility (legacy HTTP mode)

The original `docker-compose-distributed.yml` still works. The `USE_RAY` variable
defaults to `false`, so no code paths changed for the legacy setup.

---

## Ray Dashboard

Once the cluster is running, the Ray dashboard at `http://<coordinator-ip>:8265`
provides:

- Live actor list with state (ALIVE / RESTARTING / DEAD)
- GPU utilization per node
- Task throughput and latency histograms
- Cluster resource overview (CPUs, GPUs, memory)
- Actor logs streamed in the browser

This replaces the need for custom health check endpoints and log aggregation.

---

## What Was NOT Changed

All ML core logic is intact and untouched:

- `moe_router/gating/` — FastText, XLM-RoBERTa, Q-learning pipeline
- `moe_router/experts/llms/expert_pool.py` — LLMAdapterPool
- `moe_router/experts/llms/task_expert.py` — TaskExpert
- `expert_worker/single_model_pool.py` — SingleModelPool (reused inside `ExpertWorkerActor`)
- `app/routers/` — all API endpoints (auth, admin, analytics, classify, health)
- `app/schemas/` — request/response models
- `app/middleware/` — error handling
- `config/experts_registry.json` — model/adapter registry
- `config/expert_machine_mapping.json` — still used by `spawn_workers.py` to know
  which models to spawn (host/port entries are ignored in Ray mode)
