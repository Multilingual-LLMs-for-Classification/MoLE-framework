# MoLE + Ray: Communication Architecture Report

**Date:** 2026-03-23
**Scope:** How all components in the Ray-mode MoLE system communicate, what
transport mechanisms are used at each boundary, retry behaviour, and how the
system behaves under network partition.

---

## 1. System Boundaries

The deployed system has four distinct communication boundaries. Each uses a
different transport:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  YOUR LOCAL MACHINE                                                     │
│  Browser / curl / Python client                                         │
└────────────────────┬────────────────────────────────────────────────────┘
                     │  [A] HTTPS / HTTP  (SSH tunnel → port 8000)
┌────────────────────▼────────────────────────────────────────────────────┐
│  csetuf07  (coordinator machine)                                        │
│                                                                         │
│  ┌──────────────────┐   [B] Ray Client   ┌───────────────────────────┐ │
│  │  mole-coordinator│ ◄──────────────── ►│  mole-ray-head            │ │
│  │  FastAPI :8000   │   TCP :10001       │  GCS :6379  Dashboard :8265│ │
│  └────────┬─────────┘                   └─────────────┬──────────────┘ │
│           │                                           │                 │
│           │  [C] Ray Actor RPC                        │ [D] Ray cluster │
│           │  (via Ray Client → GCS → raylet)          │  internal proto │
│           │                                           │                 │
│  ┌────────▼─────────────────────────────────────────▼──────────────┐  │
│  │  mole-ray-worker-local                                           │  │
│  │  Ray worker node  (GPU 0)                                        │  │
│  │  ExpertWorkerActor — llama-2-7b-hf  (permanently in VRAM)       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                     │  [D] Ray cluster internal protocol
          ┌──────────▼──────────────────────────────────┐
          │  Remote GPU machines  (10.8.100.X)          │
          │  mole-ray-worker (joined via docker-compose) │
          │  One ExpertWorkerActor per node              │
          └─────────────────────────────────────────────┘
```

---

## 2. Boundary A — Client ↔ Coordinator (HTTP)

### Transport

Plain HTTP/1.1 over TCP. When accessed via SSH, this travels through an SSH
tunnel (`ssh -L 8000:localhost:8000 cse@csetuf07`) which provides an
additional TLS layer end-to-end between the client machine and `csetuf07`.

### Protocol details

| Property | Value |
|---|---|
| Protocol | HTTP/1.1 |
| Port | 8000 |
| Framework | FastAPI (Starlette/uvicorn) |
| Auth | JWT Bearer token (`Authorization: Bearer <token>`) |
| Content type | `application/json` |
| Async | Yes — uvicorn event loop; single worker process |

### Request flow

```
POST /api/v1/classify
  Headers: Authorization: Bearer <jwt>
  Body:    {"text": "...", "description": "..."}

← 200 OK
  Body:    {"result": "4", "confidence": 0.94,
             "language": "english", "domain": "finance",
             "routing_path": "english → finance → rating → gateway:llama-2-7b-hf",
             "processing_time_ms": 312.4}
```

### Retry behaviour (client side)

The API is synchronous from the client's perspective — one request, one
response. There is no built-in client-side retry. If the connection drops,
the client receives a TCP reset or a 502/504 and must retry at the application
level.

**Timeout:** `REQUEST_TIMEOUT_SECONDS=300` (configurable via env var). If the
full pipeline (gating + LLM inference) does not complete within 300 seconds,
FastAPI raises `asyncio.TimeoutError` and the client receives HTTP 504.

---

## 3. Boundary B — Coordinator ↔ Ray Head (Ray Client / TCP)

### Why Ray Client and not direct GCS

Ray has two connection modes:

| Mode | Address format | Transport | Requirement |
|---|---|---|---|
| Direct GCS | `localhost:6379` | TCP for scheduling + **Unix socket** for raylet IPC | Shared `/tmp/ray` filesystem |
| Ray Client | `ray://localhost:10001` | **Pure TCP** | None |

Because `mole-coordinator` and `mole-ray-head` are separate Docker containers
(separate filesystems even on `network_mode: host`), the direct GCS mode fails
with `Failed to connect to socket at /tmp/ray/.../sockets/raylet`. Ray Client
uses only TCP, making it compatible with the container topology.

### Transport

```
mole-coordinator (port: ephemeral)
        │
        │  TCP connection to :10001  (ray-client-server-port)
        │
mole-ray-head
        │
        │  GCS protocol (TCP :6379) — scheduling, actor registry, cluster state
        │
Ray cluster (all nodes)
```

### What travels over Ray Client

- Actor method calls: `actor.classify.remote(...)` serialises arguments with
  **msgpack** and sends them to the Ray head. Ray then routes the call to
  whichever node holds the actor.
- Return values: serialised with msgpack, returned through the same TCP channel.
- `ray.get_actor(name)`: looks up the actor's location in the GCS (in-memory
  key-value store in `mole-ray-head`) and returns a handle.

### Ray Client reconnection

Ray Client maintains a gRPC-based persistent channel to the Ray Client server.
If the TCP connection drops (e.g. transient network glitch), the Ray Client
library **automatically reconnects** with exponential back-off. In-flight
`ObjectRef`s that were awaiting results will raise `RaySystemError` if
reconnection fails; they complete normally if reconnection succeeds before the
call returns.

---

## 4. Boundary C — Coordinator → ExpertWorkerActor (Ray Actor RPC)

This is the most important boundary for inference latency.

### Transport

```
coordinator process
    │
    │  await actor.classify.remote(task_key, language, text, ...)
    │
    │  → Ray Client sends msgpack-serialised call to ray-head
    │  → ray-head GCS resolves actor location (which node, which process)
    │  → ray-head routes call to the raylet on the worker node (TCP)
    │  → raylet delivers call to the actor's asyncio event loop (Unix socket —
    │    but this socket is WITHIN the worker node container, not cross-container)
    │
ExpertWorkerActor.classify()  (running on mole-ray-worker-local or remote GPU node)
    │
    │  result returned via the same path in reverse (msgpack over TCP)
    │
coordinator awaits ObjectRef  →  result dict
```

### Serialisation

Arguments and return values are serialised with **msgpack** (for small objects)
or stored in the **Ray object store** (shared memory, Apache Plasma) for large
objects. For MoLE:

- Inference inputs (`text`, `description`) — small strings → msgpack over TCP
- Return dicts (`result`, `confidence`, `processing_time_ms`, ...) — small
  dicts → msgpack over TCP
- The LLM itself is **never serialised** — it lives permanently in the actor's
  GPU memory and is never transferred across the network

### Actor method call path in code

```python
# ray_cluster/ray_gateway_service.py
result_ref = actor.classify.remote(
    task_key=payload["task_key"],
    language=payload["language"],
    text=payload["text"],
    description=payload["description"],
    adapter_name=payload["adapter_name"],
    request_id=payload["request_id"],
)
return await result_ref   # directly awaitable — no thread pool
```

`ExpertWorkerActor.classify()` is declared `async`, so Ray registers it on the
actor's asyncio event loop. The actual GPU inference is dispatched via
`asyncio.get_event_loop().run_in_executor(None, self._sync_classify, ...)` so
the actor event loop stays responsive to health check calls while inference
runs.

### max_concurrency=1

```python
@ray.remote(num_gpus=1, max_concurrency=1)
class ExpertWorkerActor:
```

`max_concurrency=1` tells Ray to queue incoming calls to this actor and execute
them one at a time. This replaces the `asyncio.Semaphore(1)` that was in the
old FastAPI worker. A second request arriving while inference is in progress
waits in Ray's internal actor mailbox — no custom queue, no Redis, no drain
loop.

### Round-robin across replicas

```python
# ray_cluster/ray_gateway_service.py
def _pick_actor(self, base_model_key: str) -> Any:
    actor_names = self._model_to_actors[base_model_key]
    idx = self._rr_index[base_model_key] % len(actor_names)
    self._rr_index[base_model_key] = idx + 1
    return self._get_actor(actor_names[idx])
```

When `num_replicas > 1` in `ray_worker_registry.json`, `RayGatewayService`
maintains a per-model integer counter and cycles through actor names in order.
Each replica is a separate actor on its own GPU; Ray routes each call to the
selected replica's node.

---

## 5. Boundary D — Ray Cluster Internal Protocol

### Between ray-head and ray-worker nodes

Ray uses its own binary protocol over TCP for cluster-internal communication:

| Channel | Purpose | Port |
|---|---|---|
| GCS server (`ray-head`) | Actor registry, cluster state, object directory | 6379 |
| Raylet ↔ Raylet | Task scheduling, object transfer | random ephemeral |
| Object store (Plasma) | Large object transfer between nodes | Shared memory on same node; object transfer protocol on different nodes |
| Dashboard | Web UI | 8265 |

### How remote GPU machines join

On each remote GPU machine:
```bash
RAY_HEAD_ADDRESS=<csetuf07-ip>:6379 \
docker-compose -f docker/docker-compose-ray-worker.yml up -d
```

The `ray start --address=<head>:6379` command in the container:
1. Connects to the GCS on `csetuf07:6379` over TCP
2. Registers the node's resources (CPUs, GPUs, memory) with the GCS
3. Starts a local raylet process inside the container
4. Starts a local object store (Plasma) process

From this point the head's scheduler can place actors on this node. Actor
method calls from the coordinator travel:
`coordinator → ray-head (TCP :10001) → target raylet (TCP) → actor process`

---

## 6. Retry Mechanisms

### 6.1 Actor restart on crash (`max_restarts=-1`)

```python
# ray_cluster/spawn_workers.py
actor_options = {
    "name": actor_name,
    "lifetime": "detached",
    "max_restarts": -1,   # restart indefinitely
    "num_gpus": num_gpus,
}
```

If an `ExpertWorkerActor` process crashes (OOM, segfault, uncaught exception):

1. Ray detects the crash via the raylet heartbeat (default 1-second interval)
2. Ray marks the actor as `RESTARTING`
3. Ray spawns a new actor process on the same (or a different, if that node
   died) GPU node
4. The new process runs `__init__` again — `SingleModelPool.preload()` reloads
   the LLM into GPU memory
5. Any `ObjectRef`s that were awaiting results from the crashed actor receive
   `RayActorError` — the caller must handle this

In `RayGatewayService.health_check_all()`, `RayActorError` is caught and the
actor handle is evicted from the cache:

```python
except (ValueError, RuntimeError, ray.exceptions.RayError) as exc:
    self._actor_cache.pop(actor_name, None)
    results[actor_name] = {"status": "unreachable", "error": str(exc)}
```

The next `dispatch()` call will call `ray.get_actor(actor_name)` again. If the
actor has restarted by then, the call succeeds. If still restarting, the call
raises `ValueError("Actor not found")` which propagates as HTTP 503 to the
client.

### 6.2 Actor reuse across coordinator restarts

```python
# ray_cluster/spawn_workers.py
try:
    ray.get_actor(actor_name)
    print(f"[spawn_workers] Actor '{actor_name}' already running — reusing.")
    actor_names.append(actor_name)
    continue
except ValueError:
    pass  # Does not exist yet — create below.
```

Because actors have `lifetime="detached"`, they survive coordinator container
restarts. When the coordinator restarts and calls `spawn_workers()`, it finds
existing actors and reuses their handles — the LLM does not reload. This makes
coordinator restarts very fast (seconds) compared to worker restarts (minutes,
due to model loading).

### 6.3 Ray Client reconnection

The Ray Client gRPC channel reconnects automatically on transient TCP drops.
The reconnection window is ~30 seconds by default. Calls in-flight during a
reconnection receive `RaySystemError` if the window expires.

### 6.4 No application-level retry on inference calls

There is currently **no automatic retry** at the `RayGatewayService.dispatch()`
level. If `await result_ref` raises (actor crashed, timeout, network error),
the exception propagates to the FastAPI classify endpoint which returns HTTP
500. The client is responsible for retrying the request.

This is intentional for LLM inference: retrying a failed inference automatically
could produce duplicate side effects in stateful tasks and would double latency
for the common case where the actor restarted and the new instance needs time to
load. A retry with back-off at the client level is the recommended pattern.

---

## 7. Network Partition Behaviour

A network partition means some subset of nodes can no longer communicate with
the rest. Below are the four realistic partition scenarios for this deployment.

### Scenario 1: SSH tunnel drops (client ↔ csetuf07)

| What happens | HTTP connection to port 8000 is severed |
|---|---|
| In-flight request | Client receives TCP RST or connection timeout |
| Coordinator | Unaffected — continues running, actors continue running |
| Recovery | Client reconnects and retries the request |
| Data loss | None — the LLM inference may have completed on the server side; result is lost only because the TCP connection died before the response was sent |

### Scenario 2: Ray Client TCP drops (coordinator ↔ ray-head, port 10001)

| What happens | The gRPC channel from coordinator to Ray Client server breaks |
|---|---|
| In-flight `await result_ref` | Raises `RaySystemError` after reconnection window (~30s) |
| Coordinator | Ray Client attempts automatic reconnection with back-off |
| Ray head | Unaffected — GCS continues running, actors continue running |
| Recovery | If reconnection succeeds within ~30s, queued requests complete normally. If not, coordinator logs `RaySystemError` and returns HTTP 500 for affected requests |
| HTTP requests during reconnect | FastAPI continues accepting new requests but `dispatch()` calls will fail until the channel recovers |

### Scenario 3: Partition between ray-head and ray-worker-local (same machine)

This would require the host's loopback interface to fail — essentially
impossible in normal operation. Both containers use `network_mode: host` and
communicate via `localhost`, so they share the host network stack.

If for some reason the worker container exits unexpectedly:

1. The raylet heartbeat from `mole-ray-worker-local` stops reaching the head
2. After the dead-node timeout (~30s by default), the GCS marks the node as dead
3. All actors on that node are marked as `DEAD`
4. With `max_restarts=-1`, Ray attempts to reschedule them on another available
   GPU node
5. If no other GPU node is available, the actors remain in `PENDING` state until
   a GPU node rejoins

### Scenario 4: Partition between csetuf07 and a remote GPU worker machine

This is the most realistic multi-machine failure scenario.

**Timeline:**

| Time | Event |
|---|---|
| T=0 | Network between csetuf07 and 10.8.100.X is severed |
| T=0 to T≈30s | Raylet heartbeats from the remote node stop reaching the head. Requests to actors on that node are queued in the actor mailbox. `await result_ref` is waiting. |
| T≈30s | GCS marks the remote node as dead. All actors on that node (`ExpertWorkerActor` for the model assigned to that machine) receive `RayActorError`. |
| T≈30s | In-flight `await result_ref` calls raise `RayActorError` → HTTP 500 |
| T≈30s | With `max_restarts=-1`, Ray tries to reschedule the actor on another GPU node. If a spare GPU slot exists elsewhere in the cluster, the actor spawns there and reloads its model. If no GPU slot is available, the actor remains `PENDING`. |
| T=recovery | When the network is restored, the remote node reconnects to the GCS. Ray detects the node is alive again. If the actor was rescheduled elsewhere, the old instance on the recovered node is ignored (it is superseded). New requests succeed. |

**Impact on models assigned exclusively to the partitioned node:**

- Requests for that model's `base_model_key` fail with `RayActorError` (→ HTTP 500)
  until the actor is rescheduled elsewhere or the partition heals
- Requests for models on other nodes are completely unaffected
- The coordinator and gating pipeline continue operating normally throughout

**Mitigation:**

Setting `num_replicas: 2` in `ray_worker_registry.json` for critical models
spreads replicas across two different physical nodes. During a partition that
kills one node, `RayGatewayService._pick_actor()` round-robins to the surviving
replica. The failed actor's handle is evicted from the cache by `health_check_all()`.
Note that without proactive health checking, the round-robin counter may still
route ~50% of requests to the dead actor before the error is detected. An
improvement would be to remove dead actor names from the round-robin pool on
`RayActorError`.

---

## 8. Communication Summary Table

| Boundary | Protocol | Transport | Port | Serialisation | Auth |
|---|---|---|---|---|---|
| Client → Coordinator | HTTP/1.1 | TCP (SSH tunnel) | 8000 | JSON | JWT Bearer |
| Coordinator → Ray head | Ray Client (gRPC) | TCP | 10001 | msgpack / protobuf | None (internal) |
| Coordinator → ExpertWorkerActor | Ray Actor RPC | TCP via Ray Client | 10001→routed | msgpack | None (internal) |
| Ray head → Ray workers | Ray cluster protocol | TCP | 6379 + ephemeral | Ray binary | None (internal) |
| Ray head Dashboard | HTTP | TCP | 8265 | JSON/HTML | None |
| ExpertWorkerActor GPU inference | in-process | — | — | PyTorch tensors in VRAM | — |

---

## 9. What Is NOT Used

| Mechanism | Status |
|---|---|
| Unix domain sockets (cross-container) | Not used. The `raylet` Unix socket at `/tmp/ray/.../sockets/raylet` is only used within a single container (raylet ↔ worker processes on the same node). Cross-container communication is all TCP. |
| Redis | Not used in Ray mode. Redis was part of the original HTTP mode (job queue + result store). Ray's GCS and object store replace it entirely. |
| HTTP between coordinator and workers | Not used in Ray mode. `GatewayService` (httpx-based) is replaced by `RayGatewayService`. |
| Shared filesystem | Not used. Models are loaded independently by each actor from the HuggingFace cache volume mount. No NFS or shared storage is needed for inference. |

---

## 10. Port Reference

| Port | Service | Direction | Purpose |
|---|---|---|---|
| 8000 | mole-coordinator | inbound | FastAPI REST API |
| 6379 | mole-ray-head | inbound from workers | Ray GCS (cluster join, scheduling) |
| 8265 | mole-ray-head | inbound from browser | Ray dashboard |
| 10001 | mole-ray-head | inbound from coordinator | Ray Client server (actor RPC) |
