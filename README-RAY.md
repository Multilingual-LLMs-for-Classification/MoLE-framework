# MoLE — Ray Distributed Deployment Guide

This guide covers everything needed to run the MoLE Classification Service using
[Ray](https://ray.io) as the distributed backend. Ray replaces the custom HTTP
worker setup with a proper distributed framework that handles GPU scheduling,
cross-node RPC, fault tolerance, and observability automatically.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Project Structure](#3-project-structure)
4. [Configuration Files](#4-configuration-files)
5. [Environment Variables](#5-environment-variables)
6. [Scenario A — Single Machine](#6-scenario-a--single-machine)
7. [Scenario B — Multi-Machine Cluster](#7-scenario-b--multi-machine-cluster)
8. [Verifying the Cluster](#8-verifying-the-cluster)
9. [Ray Dashboard](#9-ray-dashboard)
10. [Making API Requests](#10-making-api-requests)
11. [Adding More GPU Workers](#11-adding-more-gpu-workers)
12. [Stopping the System](#12-stopping-the-system)
13. [Troubleshooting](#13-troubleshooting)
14. [Advanced Options](#14-advanced-options)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  COORDINATOR MACHINE  (e.g. csetuf07)                           │
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │  mole-ray-head  │    │  mole-coordinator  (port 8000)   │   │
│  │  port 6379 GCS  │◄───│  FastAPI + Gating pipeline       │   │
│  │  port 8265 dash │    │  FastText + XLM-RoBERTa + QL     │   │
│  │  port 10001 RPC │    │  RayGatewayService               │   │
│  └────────┬────────┘    └─────────────────┬────────────────┘   │
│           │                               │ actor.classify      │
│  ┌────────┴────────────────────────────── ┼ ───────────────┐   │
│  │  mole-ray-worker-local  (GPU 0)        │                │   │
│  │  Ray worker node — advertises GPU      │                │   │
│  │  ExpertWorkerActor lands here ◄────────┘                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │ Ray cluster protocol (TCP port 6379)
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  REMOTE GPU MACHINE 2  (optional)                               │
│  mole-ray-worker  joins cluster, advertises its GPU(s)         │
│  ExpertWorkerActors for other models are scheduled here         │
└─────────────────────────────────────────────────────────────────┘
```

### What each container does

| Container | Role | GPU | Port |
|---|---|---|---|
| `mole-ray-head` | Ray cluster manager — GCS, scheduler, dashboard | No | 6379, 8265, 10001 |
| `mole-coordinator` | FastAPI app — gating pipeline + Ray gateway | No | 8000 |
| `mole-ray-worker-local` | Registers local GPU with the cluster | GPU 0 | — |
| `mole-ray-worker` _(remote)_ | Registers remote machine GPU(s) | GPU(s) | — |

### Request flow

```
POST /api/v1/classify
        ↓
Coordinator: gating pipeline (CPU)
  language → domain → task → base_model_key
        ↓
RayGatewayService.dispatch()
  ray.get_actor("expert-worker-worker-0-r0")
  actor.classify.remote(...)
        ↓  (Ray routes to GPU node)
ExpertWorkerActor.classify()
  LLM inference on GPU
        ↓
HTTP 200  ClassifyResponse
```

---

## 2. Prerequisites

### On the coordinator machine

- Docker ≥ 24 with Docker Compose plugin
- NVIDIA driver ≥ 520
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- `nvidia-persistenced` running (required for NVIDIA CDI socket)
- At least 20 GB RAM
- Port 6379, 8000, 8265, 10001 reachable from worker machines

### On each remote GPU worker machine

- Docker ≥ 24 with Docker Compose plugin
- NVIDIA driver ≥ 520
- NVIDIA Container Toolkit
- `nvidia-persistenced` running
- HuggingFace model weights present at `/home/cse/.cache/huggingface`
- LoRA adapter weights present in `docker/volumes/adapter_weights/`
- Network access to coordinator machine on port 6379

### Check nvidia-persistenced

```bash
# Must show "active (running)"
systemctl status nvidia-persistenced

# If not running:
sudo systemctl start nvidia-persistenced
```

---

## 3. Project Structure

```
MoLE-framework/
├── app/                              # FastAPI coordinator service
│   ├── config.py                    # All settings (Ray address, modes, paths)
│   ├── main.py                      # ray.init() / ray.shutdown() lifecycle
│   └── services/
│       └── routing_service.py       # Wires gating → RayGatewayService
├── ray_cluster/
│   ├── worker_actor.py              # ExpertWorkerActor (@ray.remote, num_gpus=1)
│   ├── ray_gateway_service.py       # Replaces HTTP GatewayService
│   ├── spawn_workers.py             # Creates named actors from ray_worker_registry.json
│   └── gating_actor.py             # Optional: gating pipeline as a Ray Actor
├── config/
│   ├── ray_worker_registry.json     # Ray-mode worker config (models, replicas, GPUs)
│   └── expert_machine_mapping.json  # HTTP-mode only (not used in Ray mode)
├── docker/
│   ├── docker-compose-ray.yml       # Coordinator machine (ray-head + coordinator + local GPU worker)
│   └── docker-compose-ray-worker.yml# Remote GPU machine (joins the cluster)
└── moe_router/                      # Gating + expert ML code (unchanged)
```

---

## 4. Configuration Files

### `config/ray_worker_registry.json`

Declares which LLM actors to spawn and how. Only used when `USE_RAY=true`.

```json
{
  "workers": {
    "worker-0": {
      "base_model_key": "llama-2-7b-hf",
      "num_replicas": 1,
      "num_gpus": 1
    },
    "worker-1": {
      "base_model_key": "qwen2.5-7b-instruct",
      "num_replicas": 1,
      "num_gpus": 1
    }
  }
}
```

| Field | Default | Description |
|---|---|---|
| `base_model_key` | required | Must match a key in `experts_registry.json` |
| `num_replicas` | `1` | Actor replicas for this model. >1 = round-robin load balancing (each replica needs its own GPU) |
| `num_gpus` | `1` | GPU slots per replica. Use `0.5` to share a GPU between two models |
| `placement_group_strategy` | _(none)_ | Pin replicas to nodes: `STRICT_PACK`, `PACK`, `SPREAD`, `STRICT_SPREAD` |

Only include workers whose models are downloaded on the available GPU nodes.
The coordinator will wait for actors to become ready before accepting traffic.

---

## 5. Environment Variables

All variables are set in `docker/docker-compose-ray.yml`. Override via `.env` file or shell export.

### Core Ray variables

| Variable | Default | Description |
|---|---|---|
| `USE_RAY` | `false` | **Set to `true` to enable Ray mode** |
| `RAY_ADDRESS` | `ray://localhost:10001` | Ray cluster address. Use `ray://` (TCP) when coordinator and ray-head are separate containers |
| `RAY_WORKER_REGISTRY_PATH` | `config/ray_worker_registry.json` | Path to Ray worker config |
| `RAY_SCHEDULER_EVENTS` | `0` | Suppress autoscaler event noise in logs |

### Optional gating actor variables

| Variable | Default | Description |
|---|---|---|
| `USE_GATING_ACTOR` | `false` | Move gating pipeline to a Ray Actor |
| `GATING_ACTOR_NUM_GPUS` | `0.0` | GPU fraction for gating actor (`0.0` = CPU, `0.1` = 10% of GPU) |

### Other coordinator variables

| Variable | Description |
|---|---|
| `SERVICE_MODE` | `coordinator` (gating only) or `monolithic` (full pipeline in-process) |
| `JWT_SECRET_KEY` | Change this in production |
| `DATABASE_URL` | SQLite path for user accounts |
| `REQUEST_TIMEOUT_SECONDS` | Per-request timeout (default 300s) |

---

## 6. Scenario A — Single Machine

Use this when the coordinator and GPU are on the **same machine** (e.g. `csetuf07`).
The `docker-compose-ray.yml` handles everything — ray-head, coordinator, and a local
GPU worker node all in one command.

### Step 1 — Ensure nvidia-persistenced is running

```bash
sudo systemctl start nvidia-persistenced
systemctl status nvidia-persistenced   # must be active
```

### Step 2 — Edit ray_worker_registry.json

Only keep workers whose models are downloaded on this machine:

```bash
nano config/ray_worker_registry.json
# Remove worker entries whose model weights are not in /home/cse/.cache/huggingface
```

### Step 3 — Start the stack

```bash
cd /home/cse/Desktop/MoLE-framework

docker-compose -f docker/docker-compose-ray.yml up --build -d
```

Startup order enforced by healthchecks:
1. `mole-ray-head` starts and becomes healthy (~30s)
2. `mole-ray-worker-local` joins the cluster and becomes healthy (~15s)
3. `mole-coordinator` connects to Ray, loads gating models, spawns LLM actors (~2–5 min)

### Step 4 — Watch the coordinator load

```bash
docker logs -f mole-coordinator
```

Look for these lines in order:

```
[Ray] Connected — cluster resources: {'GPU': 1.0, 'CPU': 8.0, ...}
Initializing PromptRoutingSystem (mode=coordinator) ...
[spawn_workers] Spawning actor 'expert-worker-worker-0-r0' (llama-2-7b-hf, 1 GPU(s)) ...
[spawn_workers] Actor 'expert-worker-worker-0-r0' scheduled.
```

### Step 5 — Verify

```bash
# All three containers healthy
docker ps

# Ray cluster shows GPU registered
docker exec mole-ray-head ray status

# API responding
curl http://localhost:8000/api/v1/health
```

---

## 7. Scenario B — Multi-Machine Cluster

Use this when GPU worker machines are separate from the coordinator machine.

### Step 1 — Start the coordinator machine (same as Scenario A steps 1–3)

```bash
# On csetuf07 (coordinator machine)
sudo systemctl start nvidia-persistenced
cd /home/cse/Desktop/MoLE-framework
docker-compose -f docker/docker-compose-ray.yml up --build -d
```

Note the coordinator machine's IP address:
```bash
hostname -I | awk '{print $1}'
# e.g. 10.8.100.21
```

### Step 2 — Prepare each remote GPU worker machine

On each remote machine, ensure:
- Docker + NVIDIA Container Toolkit installed
- `nvidia-persistenced` running
- Model weights present at `/home/cse/.cache/huggingface`
- Adapter weights present at the same path as on the coordinator

Copy the project to the remote machine:
```bash
# From coordinator machine
rsync -av --exclude='.git' \
  /home/cse/Desktop/MoLE-framework/ \
  cse@<worker-machine-ip>:/home/cse/Desktop/MoLE-framework/
```

### Step 3 — Join the remote worker to the cluster

```bash
# On the remote GPU machine
sudo systemctl start nvidia-persistenced
cd /home/cse/Desktop/MoLE-framework

RAY_HEAD_ADDRESS=10.8.100.21:6379 \
docker-compose -f docker/docker-compose-ray-worker.yml up --build -d
```

For a machine with 2 GPUs:
```bash
RAY_HEAD_ADDRESS=10.8.100.21:6379 NUM_GPUS=2 \
docker-compose -f docker/docker-compose-ray-worker.yml up --build -d
```

### Step 4 — Verify the worker joined

```bash
# On coordinator machine — should show new node with GPU
docker exec mole-ray-head ray status
```

Ray will automatically schedule pending `ExpertWorkerActor` instances (those waiting
for a GPU slot) onto the newly joined node.

### Step 5 — Update ray_worker_registry.json

Add entries for models on the new worker machine. Coordinator spawns actors on
any node that has a free GPU — no IP/port configuration needed.

---

## 8. Verifying the Cluster

```bash
# Full cluster status (nodes, resources, actors)
docker exec mole-ray-head ray status

# List all running actors
docker exec mole-coordinator python3 -c "
import ray; ray.init('ray://localhost:10001')
print(ray.util.state.list_actors())
"

# Check coordinator health
curl http://localhost:8000/api/v1/health

# Check all worker actors are alive
curl http://localhost:8000/api/v1/health/workers
```

Expected `ray status` output when cluster is healthy:

```
======== Autoscaler status ========
Node status
---------------------------------------------------------------
Active:
 1 node(s) with resources: {'GPU': 1.0, 'CPU': 4.0, ...}  # ray-worker-local

Resources
---------------------------------------------------------------
Usage:
 1.0/1.0 GPU       ← 1 GPU slot in use by ExpertWorkerActor
 3.0/8.0 CPU
 ...
```

---

## 9. Ray Dashboard

The Ray dashboard provides live visibility into all actors, GPU utilization,
task throughput, and logs — no additional tooling needed.

### Access from your local machine (SSH tunnel)

```bash
# Open a tunnel on your local machine
ssh -L 8265:localhost:8265 cse@csetuf07

# Or as a background tunnel (no interactive shell)
ssh -N -L 8265:localhost:8265 cse@csetuf07 &
```

Then open **http://localhost:8265** in your browser.

### Dashboard tabs

| Tab | What you see |
|---|---|
| **Overview** | Cluster resource usage, node count, actor count |
| **Cluster** | Per-node CPU, GPU, memory breakdown |
| **Actors** | All `ExpertWorkerActor` instances — state (ALIVE/RESTARTING/DEAD), GPU usage |
| **Metrics** | `mole_expert_requests_total`, `mole_expert_latency_ms` histograms |
| **Logs** | Per-actor stdout/stderr streamed in browser |
| **Jobs** | Ray job history |

---

## 10. Making API Requests

### Access the API from your local machine

```bash
ssh -L 8000:localhost:8000 cse@csetuf07
```

Then use `http://localhost:8000`.

### Get an access token

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin"
```

### Classify text

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quarterly earnings report exceeded analyst expectations significantly.",
    "task_hint": "finance"
  }'
```

### Interactive API docs

Open **http://localhost:8000/docs** (Swagger UI) for all endpoints.

---

## 11. Adding More GPU Workers

To expand the cluster with additional GPU machines at any time (cluster is already running):

```bash
# On the new GPU machine
sudo systemctl start nvidia-persistenced
cd /home/cse/Desktop/MoLE-framework

RAY_HEAD_ADDRESS=<coordinator-ip>:6379 \
docker-compose -f docker/docker-compose-ray-worker.yml up --build -d
```

Ray detects the new node automatically. If there are any actors in PENDING state
(waiting for GPU slots), they will be scheduled on the new node immediately.
No coordinator restart needed.

---

## 12. Stopping the System

### Stop coordinator machine

```bash
cd /home/cse/Desktop/MoLE-framework
docker-compose -f docker/docker-compose-ray.yml down
```

Named Ray actors use `lifetime="detached"` so they persist through coordinator
restarts but are cleaned up when the ray-head stops.

### Stop a remote worker node

```bash
# On the remote worker machine
docker-compose -f docker/docker-compose-ray-worker.yml down
```

Ray will detect the node failure and mark its actors as DEAD. If `max_restarts=-1`
is set on the actors, Ray will attempt to reschedule them on other available nodes.

### Full teardown (all machines)

```bash
# On coordinator machine
docker-compose -f docker/docker-compose-ray.yml down

# On each worker machine
docker-compose -f docker/docker-compose-ray-worker.yml down
```

---

## 13. Troubleshooting

### `Failed to connect to socket at address: /tmp/ray/session_.../sockets/raylet`

**Cause:** `RAY_ADDRESS` is set to `localhost:6379` (GCS address) instead of
`ray://localhost:10001` (Ray Client). GCS connection requires a shared Unix socket
between containers, which doesn't work with separate Docker containers.

**Fix:** Set `RAY_ADDRESS=ray://localhost:10001` in `docker-compose-ray.yml`.
The Ray Client protocol uses TCP only — no shared filesystem needed.

---

### `No available node types can fulfill resource request {'GPU': 1.0}`

**Cause:** No GPU node has joined the Ray cluster yet. The ray-head starts with
`--num-gpus=0` (CPU only).

**Fix:** Ensure `mole-ray-worker-local` is running and healthy:

```bash
docker ps | grep ray-worker
docker logs mole-ray-worker-local
docker exec mole-ray-head ray status   # should show GPU: 1.0
```

---

### `open /run/nvidia-persistenced/socket: no such file or directory`

**Cause:** `nvidia-persistenced` is not running. The NVIDIA CDI spec references
this socket and Docker fails to start the container.

**Fix:**
```bash
sudo systemctl start nvidia-persistenced
# Then restart the stack
docker-compose -f docker/docker-compose-ray.yml up -d
```

---

### `(raylet) file_system_monitor.cc: ... is over 95% full`

**Cause:** The filesystem where `/tmp/ray/` resides is nearly full (typically the
root partition). This is a warning, not an error — Ray inference workloads rarely
need to spill objects to disk.

**To suppress the autoscaler noise** (already set in `docker-compose-ray.yml`):
```yaml
- RAY_SCHEDULER_EVENTS=0
```

The raylet monitor warning cannot be suppressed by env var — it is a C++ level log.
Free up disk space to resolve it permanently:
```bash
df -h                               # identify which partition
du -sh /home/cse/.cache/huggingface/*   # check model cache size
```

---

### Coordinator starts but actors stay PENDING

**Cause:** Not enough GPU slots in the cluster for all workers defined in
`ray_worker_registry.json`.

**Fix:** Either add more GPU worker nodes, or comment out worker entries whose
models are not yet available:

```bash
nano config/ray_worker_registry.json
# Remove or comment entries for unavailable models
docker-compose -f docker/docker-compose-ray.yml restart coordinator
```

---

### Actor crashes / restarts in a loop

```bash
# Check actor logs in the Ray dashboard → Actors → select actor → Logs
# Or from CLI:
docker exec mole-coordinator python3 -c "
import ray; ray.init('ray://localhost:10001')
actors = ray.util.state.list_actors(filters=[('state','=','DEAD')])
for a in actors: print(a)
"
```

Common cause: GPU OOM. Check if model fits in VRAM:
```bash
nvidia-smi
```

---

## 14. Advanced Options

### Scale up a hot model with multiple replicas

Edit `config/ray_worker_registry.json`:

```json
"worker-0": {
  "base_model_key": "llama-2-7b-hf",
  "num_replicas": 2,
  "num_gpus": 1
}
```

This spawns two actors (`-r0`, `-r1`), each on its own GPU. Requests are distributed
round-robin. Requires two available GPU slots in the cluster.

Restart the coordinator to respawn:
```bash
docker-compose -f docker/docker-compose-ray.yml restart coordinator
```

---

### Move the gating pipeline to GPU

The gating pipeline (FastText + XLM-RoBERTa + Q-learning) runs on CPU by default.
To move it to a fractional GPU slice:

```yaml
# In docker-compose-ray.yml, coordinator environment:
- USE_GATING_ACTOR=true
- GATING_ACTOR_NUM_GPUS=0.1
```

`0.1` means 10% of one GPU — up to 10 gating actors can share a GPU alongside
LLM workers.

---

### Pin a model to a specific node

Use placement groups in `ray_worker_registry.json`:

```json
"worker-0": {
  "base_model_key": "llama-2-7b-hf",
  "num_replicas": 1,
  "num_gpus": 1,
  "placement_group_strategy": "STRICT_PACK"
}
```

| Strategy | Behaviour |
|---|---|
| `STRICT_PACK` | All replicas on the same node |
| `SPREAD` | Each replica on a different node |
| `PACK` | Try to pack, fall back to spread |
| `STRICT_SPREAD` | Force separate nodes, fail if not possible |

---

### Use the legacy HTTP mode (no Ray)

Set `USE_RAY=false` and use the original compose file:

```bash
docker-compose -f docker/docker-compose-distributed.yml up -d
```

No Ray components are started. `GatewayService` dispatches to worker URLs defined
in `config/expert_machine_mapping.json`.
