# Docker in the MoLE Framework — What It Does and Why

## What Is Docker (in Plain Terms)?

Docker lets you package an application — its code, Python version, libraries,
and configuration — into a self-contained unit called a **container**. A container
runs in isolation from the host system, so it behaves the same whether it's on your
laptop, a cloud VM, or a GPU server.

Think of it like this:

```
Without Docker:               With Docker:
  Machine A runs Python 3.9     Every machine runs the same
  Machine B runs Python 3.11    container image built from
  Dependencies differ           Dockerfile — identical every time
  "Works on my machine" bugs
```

A **Docker image** is the blueprint (built once from a Dockerfile).
A **container** is a running instance of that image.

---

## Why MoLE Uses Docker

MoLE is a **distributed** system — different LLMs run on different machines.
Docker solves three specific problems this creates:

| Problem | How Docker Solves It |
|---|---|
| Each machine needs the same Python env | One `Dockerfile` defines it — build once, run anywhere |
| Coordinator and workers must be isolated | Each service is a separate container with its own process space |
| LLM model weights (multi-GB) must be accessible | Docker **volumes** mount weight files into containers without copying them |
| Services must communicate across machines | Docker **networks** + real IPs in `expert_machine_mapping.json` |

---

## The Three Compose Files

```
docker/
├── docker-compose-distributed.yml   ← coordinator + worker-0 (same machine)
├── docker-compose-worker.yml        ← template for remote workers (machine 1–N)
└── docker-compose.yml               ← monolithic mode (single GPU, all-in-one)
```

---

## What Happens When You Run the Command

```bash
docker-compose -f docker-compose-distributed.yml up --build -d
```

### Flag Breakdown

| Flag | Meaning |
|---|---|
| `-f docker-compose-distributed.yml` | Use this specific compose file (not the default `docker-compose.yml`) |
| `up` | Create and start the services defined in the file |
| `--build` | Rebuild Docker images from the `Dockerfile` before starting (picks up any code changes) |
| `-d` | Detached mode — run in the background, return the terminal immediately |

---

### Step-by-Step Execution Order

#### Step 1 — Read the Compose File

Docker Compose reads `docker-compose-distributed.yml` and finds:
- 2 services to start: `coordinator` and `expert-worker-0`
- 1 internal network to create: `mole-network`
- 5 named volumes to create (if they don't already exist):
  `gating_models`, `language_models`, `adapter_weights`, `huggingface_cache`, `fasttext_cache`

#### Step 2 — Create the Network

```
mole-network (bridge driver)
```

A private virtual network is created on this machine. Both containers get
an IP address on this network and can reach each other by **service name**
(e.g., the coordinator can call `http://expert-worker-0:8001` and Docker
resolves it automatically — no need to hard-code IPs between containers on
the same machine).

#### Step 3 — Build the Docker Image (`--build`)

Both services use the **same image**, built from:

```
Build context: MoLE-framework/          ← the whole project root
Dockerfile:    docker/Dockerfile
```

What the Dockerfile does, layer by layer:

```
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
  └─ Base image with CUDA libraries pre-installed (needed for GPU access)

RUN apt-get install python3.11 ...
  └─ Install Python 3.11 and pip

COPY requirements.txt .
RUN pip install -r requirements.txt
  └─ Install all Python packages (FastAPI, torch, transformers, peft, etc.)
     Cached as a layer — only re-runs if requirements.txt changes

COPY app/ ./app/
COPY moe_router/ ./moe_router/
COPY expert_worker/ ./expert_worker/
  └─ Copy the application source code into the image

ENV PYTHONPATH="/app"
  └─ Makes all packages importable from /app

EXPOSE 8000
  └─ Documents the default port (actual port binding is in compose)
```

One image is built, then reused for both containers — they differ only in the
`command:` and `environment:` settings the compose file injects.

#### Step 4 — Start `expert-worker-0` First

The coordinator has:
```yaml
depends_on:
  expert-worker-0:
    condition: service_healthy
```

So Docker starts `expert-worker-0` first and **waits** until it passes its
health check before starting the coordinator.

What happens inside the `expert-worker-0` container on startup:

```
command: uvicorn expert_worker.main:app --host 0.0.0.0 --port 8001 --workers 1
```

1. `uvicorn` starts the FastAPI app defined in `expert_worker/main.py`
2. The `lifespan` function runs:
   - Reads `WORKER_MODEL_KEY=llama-2-7b-hf` from the environment
   - Creates a `SingleModelPool` and calls `pool.preload()`
   - This downloads (or loads from HuggingFace cache) **llama-2-7b-hf** into GPU memory
   - Builds `TaskExpert` objects for every task that uses this model
3. The worker is now listening on port `8001`

**Environment variables** injected by the compose file:

| Variable | Value | Purpose |
|---|---|---|
| `WORKER_MODEL_KEY` | `llama-2-7b-hf` | Which LLM this worker loads |
| `WORKER_ID` | `worker-0` | Identifier for logging |
| `CUDA_VISIBLE_DEVICES` | `0` | Use GPU 0 |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Expose all GPUs to the NVIDIA runtime |
| `MAX_CONCURRENT_GPU_REQUESTS` | `1` | Serialize inference (one at a time) |

**Volumes** mounted into the container:

| Host Volume | Container Path | Contents |
|---|---|---|
| `adapter_weights` | `/app/moe_router/experts/llms/adapters` | LoRA adapter weights |
| `huggingface_cache` | `/root/.cache/huggingface` | Downloaded model weights cache |
| `../config` (bind mount) | `/app/config` | `expert_machine_mapping.json`, `experts_registry.json` |

#### Step 5 — Health Check Loop for `expert-worker-0`

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8001/api/v1/health/ready"]
  interval: 30s
  timeout: 10s
  start_period: 300s   # 5-minute grace period for model loading
  retries: 10
```

Docker calls this URL every 30 seconds. The `/health/ready` endpoint returns
`200 OK` only after the LLM is fully loaded into GPU memory. With a 5-minute
grace period, the worker has time to download/load a 7B model before Docker
considers it unhealthy.

#### Step 6 — Start `coordinator` (after worker is healthy)

```
command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

What happens inside:

1. `uvicorn` starts the FastAPI app defined in `app/main.py`
2. The `lifespan` function runs with `SERVICE_MODE=coordinator`:
   - Initializes `routing_service` (the gating pipeline)
   - Loads lightweight models: FastText (language detection), XLM-RoBERTa (domain classification), Q-learning NN (task routing)
   - Reads `expert_machine_mapping.json` to know where each expert worker is
3. The coordinator is now ready to accept classification requests on port `8000`

**Volumes** mounted into the coordinator:

| Host Volume | Container Path | Contents |
|---|---|---|
| `gating_models` | `/app/moe_router/gating/models` | XLM-RoBERTa + Q-learning weights |
| `language_models` | `/app/moe_router/models` | `lid.176.bin` (FastText language model) |
| `fasttext_cache` | `/root/.cache/fasttext` | FastText runtime cache |
| `../config` (bind mount) | `/app/config` | Routing config files |

---

## The Full Picture: How a Classification Request Flows

```
Client (HTTP)
     │
     ▼  POST /api/v1/classify
┌────────────────────────┐
│  mole-coordinator      │  port 8000 (public)
│  container             │
│                        │
│  1. FastText           │ → detect language (e.g., "en")
│  2. XLM-RoBERTa        │ → classify domain (e.g., "finance")
│  3. Q-learning NN      │ → select task (e.g., "sentiment_analysis")
│  4. expert_mapping     │ → resolve worker URL
└────────────┬───────────┘
             │  HTTP POST /api/v1/expert/classify
             │  (inside mole-network)
             ▼
┌────────────────────────┐        ┌──────────────────────────┐
│  mole-expert-worker-0  │        │  worker-4 (10.8.100.28)  │
│  llama-2-7b-hf         │   OR   │  mistral unsloth 4-bit   │
│  port 8001 (internal)  │        │  port 8005 (remote IP)   │
└────────────────────────┘        └──────────────────────────┘
             │
             ▼
     LLM inference result
             │
             ▼
        coordinator
             │
             ▼
        Client response
```

Workers on the **same machine** (worker-0) are reached via the Docker internal
network (`http://expert-worker-0:8001`).

Workers on **remote machines** (worker-4) are reached via real IP
(`http://10.8.100.28:8005`) — these containers are not in the same compose
network, so Docker DNS doesn't apply.

---

## Named Volumes vs Bind Mounts

The compose file uses both:

**Named volumes** (managed by Docker, persist across container restarts):
```yaml
volumes:
  gating_models:    # → /var/lib/docker/volumes/docker_gating_models/_data
  adapter_weights:  # → /var/lib/docker/volumes/docker_adapter_weights/_data
  huggingface_cache:
  ...
```
Pre-populated by copying weight files into them before first run
(see `docker/volumes/` in the repo — those files are bind-mounted in the old approach
and were copied to named volumes during setup).

**Bind mounts** (link directly to a host directory):
```yaml
- ../config:/app/config:ro    # live-editable — change mapping.json without rebuild
```

---

## GPU Access: How Docker Talks to the NVIDIA Driver

```
docker-compose (runtime: nvidia)
        │
        ▼
nvidia-container-runtime
        │  reads CDI spec at /etc/cdi/nvidia.yaml
        ▼
NVIDIA driver on host (580.126.09)
        │
        ▼
RTX 4070 Ti SUPER (GPU 0)
        │
        ▼ exposed inside container as /dev/nvidia0
Container sees GPU via CUDA
```

`runtime: nvidia` (set in the compose file) tells Docker to use the NVIDIA
Container Runtime instead of the default `runc`. The runtime injects GPU device
files and CUDA libraries into the container so PyTorch/CUDA code works as if
running directly on the host.

---

## Useful Commands After Starting

```bash
# Check which containers are running and their health status
docker ps

# Watch live logs from the worker (model loading progress)
docker logs -f mole-expert-worker-0

# Watch live logs from the coordinator
docker logs -f mole-coordinator

# Confirm GPU is visible inside the worker container
docker exec mole-expert-worker-0 nvidia-smi

# Stop everything (keeps volumes/data)
docker-compose -f docker-compose-distributed.yml down

# Stop and wipe all volumes (model cache will be re-downloaded next start)
docker-compose -f docker-compose-distributed.yml down -v
```
