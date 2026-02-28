# MoLE Framework — Distributed Mixture-of-Language-Experts

A distributed multi-GPU inference framework for multilingual text classification.
Each GPU machine permanently pre-loads one expert LLM, eliminating the model
load/unload latency of the original monolithic architecture.

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │              MACHINE 0  (GPU 0)                  │
                    │                                                  │
  Client ─────────► │  ┌──────────────────────────────────────────┐   │
  HTTP :8000        │  │           COORDINATOR  (app/)             │   │
                    │  │  Language Detector  (FastText)            │   │
                    │  │  Domain Classifier  (XLM-RoBERTa)        │   │
                    │  │  Q-Learning Router  (task detection)      │   │
                    │  │  Expert Selector    (registry lookup)     │   │
                    │  │  Gateway            (HTTP dispatcher)     │   │
                    │  └──────────────────┬───────────────────────┘   │
                    │                     │                            │
                    │  ┌──────────────────▼───────────────────────┐   │
                    │  │     EXPERT WORKER 0  (:8001)              │   │
                    │  │     LLM: llama-2-7b-hf  (permanent)      │   │
                    │  └──────────────────────────────────────────┘   │
                    └──────────────────────┬──────────────────────────┘
                                           │
          ┌────────────────────────────────┼───────────────────────────┐
          │                                │                           │
          ▼                                ▼                           ▼
  ┌────────────────┐             ┌────────────────┐          ┌────────────────┐
  │  MACHINE 1     │             │  MACHINE 2     │          │  MACHINE N     │
  │  GPU 1         │             │  GPU 2         │          │  GPU N         │
  │                │             │                │          │                │
  │  Worker 1:8002 │             │  Worker 2:8003 │          │  Worker N:800N │
  │  qwen2.5-7b    │             │  bloomz-7b1    │          │  ...           │
  │  (permanent)   │             │  (permanent)   │          │  (permanent)   │
  └────────────────┘             └────────────────┘          └────────────────┘
```

**Request flow:**
1. Client → Coordinator (port 8000)
2. Coordinator runs lightweight gating: FastText → XLM-RoBERTa → Q-Learning → registry lookup
3. Gateway resolves `base_model_key` → worker URL from `config/expert_machine_mapping.json`
4. Gateway dispatches HTTP POST to the correct expert worker
5. Worker runs inference — LLM is already in GPU memory, zero load latency
6. Result returned to client

---

## Expert ↔ Worker Mapping

| Worker | Port | LLM | Handles | Machine | Status |
|--------|------|-----|---------|---------|--------|
| worker-0 | 8001 | `llama-2-7b-hf` | Sentiment: EN, DE, FR | `10.8.100.21` | **active** |
| worker-1 | 8002 | `qwen2.5-7b-instruct` | Sentiment: ES, JA · ESCI: ES | — | pending |
| worker-2 | 8003 | `bloomz-7b1` | Sentiment: ZH | — | pending |
| worker-3 | 8004 | `mistral-7B-Instruct-v0.3` | PII: all languages · ESCI: ES | — | pending |
| worker-4 | 8005 | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` | News: EN, DA, ES, PL | `10.8.100.28` | **active** |
| worker-5 | 8006 | `facebook/xglm-7.5B` | News: TR | — | pending |
| worker-6 | 8007 | `llama-3.1-8b-instruct` | ESCI: EN, JA | — | pending |

Each worker runs on a **dedicated GPU machine**. The coordinator runs separately and dispatches to workers via HTTP.

---

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ on each machine
- Docker + NVIDIA Container Toolkit (`nvidia-container-toolkit`) on each machine
- Trained gating model weights (domain classifier + Q-learning task routers)
- LoRA adapter weights for each expert
- HuggingFace access token (for Llama/Mistral gated models)

---

## File Structure

```
MoLE-framework/
├── app/                              # Coordinator FastAPI service
│   ├── main.py                       # Entry point (mode-aware startup)
│   ├── config.py                     # Settings (SERVICE_MODE, EXPERT_MAPPING_PATH …)
│   ├── dependencies.py               # DI: routing, gateway services
│   ├── routers/                      # API endpoints (auth, classify, health, admin)
│   ├── schemas/                      # Pydantic request/response models
│   └── services/
│       ├── routing_service.py        # Coordinator vs monolithic dispatch
│       ├── gateway_service.py        # HTTP dispatcher to expert workers  ← NEW
│       ├── auth_service.py
│       ├── analytics_service.py
│       └── config_service.py
├── expert_worker/                    # Expert worker FastAPI service  ← NEW
│   ├── main.py                       # Startup: pre-loads one LLM
│   ├── single_model_pool.py          # LLMAdapterPool with no LRU eviction
│   ├── router.py                     # POST /api/v1/expert/classify
│   └── health.py                     # GET /api/v1/health/ready
├── moe_router/                       # Core MoE routing package
│   ├── gating/components/
│   │   ├── routing_system.py         # Added: coordinator_only, run_gating(), GatingResult
│   │   ├── language_detector.py
│   │   ├── domain_classifier.py
│   │   └── q_learning_router.py
│   └── experts/
│       ├── config/experts_registry.json
│       └── llms/
│           ├── expert_pool.py        # LRU-based pool (used in monolithic mode)
│           └── task_expert.py
├── config/
│   └── expert_machine_mapping.json   # model_key → worker URL  ← NEW
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml            # Original monolithic deployment
│   ├── docker-compose-distributed.yml  # Coordinator + Worker 0 on Machine 0  ← NEW
│   └── docker-compose-worker.yml       # Template for remote workers  ← NEW
├── .env.example
└── requirements.txt
```

---

## Setup: Step-by-Step

### Step 1 — Prepare all machines

Do this on **every machine** (Machine 0 through Machine N):

```bash
# 1. Clone the repo
git clone <your-repo-url> MoLE-framework
cd MoLE-framework

# 2. Install NVIDIA Container Toolkit (if not already installed)
# Add the NVIDIA Container Toolkit repo (modern method)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

### Step 2 — Place model weights

The gating models go on **Machine 0 only**.
Adapter weights go on **every machine** that runs a worker.

```
docker/volumes/
├── gating_models/          # domain classifier + Q-learning task routers
│   ├── domain_xlmr/        # XLM-RoBERTa domain classifier weights
│   └── task_routers_qlearning/  # per-domain Q-learning router weights
├── language_models/        # FastText language identification model
│   └── lid.176.ftz
└── adapter_weights/        # LoRA adapters (mirror this on every worker machine)
    └── finance/
        ├── sentiment_analysis/
        │   ├── llama-2-7b-hf/
        │   ├── qwen2.5/
        │   └── bloomz-7b1/
        ├── pii/
        │   └── Mistral-7B-Instruct-v0.3/
        ├── news_classification/
        │   ├── mistral-7b/
        │   └── xglm-7.5B/
        └── esci/
            ├── LLama-3-8.1B/
            └── Mistral-7B/
```

---

### Step 3 — Configure the Coordinator machine

```bash
# On the coordinator machine
cd MoLE-framework
cp .env.example .env
```

Edit `.env`:

```dotenv
SERVICE_MODE=coordinator
EXPERT_MAPPING_PATH=/app/config/expert_machine_mapping.json
JWT_SECRET_KEY=<your-secret-key-change-this>
REQUEST_TIMEOUT_SECONDS=300
CUDA_VISIBLE_DEVICES=0
```

Open `config/expert_machine_mapping.json` and set the real IPs for active workers.
Leave inactive workers pointing to placeholder hostnames for now:

```json
"worker-0": { "url": "http://10.8.100.21:8001", ... },
"worker-4": { "url": "http://10.8.100.28:8005", ... }
```

---

### Step 4 — Start the Coordinator

```bash
# On the coordinator machine
cd MoLE-framework/docker

docker-compose -f docker-compose-distributed.yml up --build -d

# Watch startup logs
docker-compose -f docker-compose-distributed.yml logs -f
```

**What happens:**
- Coordinator loads gating models (FastText + XLM-RoBERTa + Q-learning, ~1–2 min)
- Workers are remote and started separately (Step 5)

**Verify the coordinator is ready:**

```bash
curl http://localhost:8000/api/v1/health/ready
```

---

### Step 5 — Start remote expert workers

Repeat for each active worker machine. Example for **worker-0** on `10.8.100.21`:

```bash
# On 10.8.100.21
cd MoLE-framework

export WORKER_MODEL_KEY=llama-2-7b-hf
export WORKER_ID=worker-0
export WORKER_PORT=8001
export ADAPTER_PATH=$(pwd)/docker/volumes/adapter_weights
export HF_CACHE_PATH=/root/.cache/huggingface
export CONFIG_PATH=$(pwd)/config

docker-compose -f docker/docker-compose-worker.yml up --build -d
docker-compose -f docker/docker-compose-worker.yml logs -f
```

**Verify the worker is ready:**

```bash
curl http://localhost:8001/api/v1/health/ready
# Expected: {"status":"ready","model_loaded":true,"worker_id":"worker-0",...}
```

**Current active workers:**

| IP | WORKER_MODEL_KEY | WORKER_ID | WORKER_PORT | Status |
|----|-----------------|-----------|-------------|--------|
| `10.8.100.21` | `llama-2-7b-hf` | `worker-0` | `8001` | **active** |
| `10.8.100.28` | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` | `worker-4` | `8005` | **active** |

**Future workers (add as more GPUs become available):**

| WORKER_MODEL_KEY | WORKER_ID | WORKER_PORT |
|-----------------|-----------|-------------|
| `qwen2.5-7b-instruct` | `worker-1` | `8002` |
| `bloomz-7b1` | `worker-2` | `8003` |
| `mistral-7B-Instruct-v0.3` | `worker-3` | `8004` |
| `facebook/xglm-7.5B` | `worker-5` | `8006` |
| `llama-3.1-8b-instruct` | `worker-6` | `8007` |

---

### Step 6 — Update the expert machine mapping with real IPs

Update `config/expert_machine_mapping.json` on the coordinator machine with the real IPs
of active workers. Leave inactive workers pointing to placeholder hostnames:

```json
{
  "workers": {
    "worker-0": { "url": "http://10.8.100.21:8001",      ... },
    "worker-1": { "url": "http://expert-worker-1:8002",  ... },
    "worker-2": { "url": "http://expert-worker-2:8003",  ... },
    "worker-3": { "url": "http://expert-worker-3:8004",  ... },
    "worker-4": { "url": "http://10.8.100.28:8005",      ... },
    "worker-5": { "url": "http://expert-worker-5:8006",  ... },
    "worker-6": { "url": "http://expert-worker-6:8007",  ... }
  },
  ...
}
```

Then restart the coordinator to pick up the new mapping:

```bash
# On Machine 0
docker-compose -f docker/docker-compose-distributed.yml restart coordinator
```

---

### Step 7 — Verify end-to-end

```bash
# 1. Register a user (on Machine 0)
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "changeme123"}'

# 2. Get a JWT token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=changeme123" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 3. Send a classification request
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Revenue increased by 20% year over year, exceeding analyst expectations.",
    "description": "Classify the sentiment of this financial statement on a scale of 1-5."
  }'
```

**Expected response:**

```json
{
  "request_id": "...",
  "language": "english",
  "domain": "finance",
  "task": "rating",
  "result": "4",
  "confidence": 0.91,
  "routing_path": "english -> finance -> rating -> gateway:llama-2-7b-hf",
  "processing_time_ms": 350.2
}
```

The `routing_path` field shows the full distributed path taken.

---

### Step 8 — Verify GPU utilization

On each machine, confirm the LLM is resident in GPU memory and stays stable:

```bash
# On any machine — watch GPU memory usage
watch -n 2 nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

# Expected on a worker machine (7B model in 4-bit):
#   0, NVIDIA ..., ~3500 MiB used, stable (no spikes between requests)

# Expected on Machine 0 (coordinator + worker-0):
#   0, NVIDIA ..., ~4000 MiB used (gating models ~500MB + LLM ~3500MB), stable
```

---

## Running Without Docker (Development)

If you want to run services directly without Docker:

### Coordinator (Machine 0)

```bash
cd MoLE-framework
pip install -r requirements.txt

export SERVICE_MODE=coordinator
export EXPERT_MAPPING_PATH=$(pwd)/config/expert_machine_mapping.json
export CUDA_VISIBLE_DEVICES=0

uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Expert Worker (any machine)

```bash
cd MoLE-framework
pip install -r requirements.txt

export WORKER_MODEL_KEY=llama-2-7b-hf     # change per machine
export WORKER_ID=worker-0
export WORKER_PORT=8001
export CUDA_VISIBLE_DEVICES=0

uvicorn expert_worker.main:app --host 0.0.0.0 --port 8001 --workers 1
```

### Monolithic mode (original single-GPU, no workers needed)

```bash
export SERVICE_MODE=monolithic
export CUDA_VISIBLE_DEVICES=0

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## API Reference

### Authentication

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/auth/register` | No | Register a new user |
| POST | `/api/v1/auth/token` | No | Get JWT access token |

### Classification

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/classify` | Bearer | Classify a single text |
| POST | `/api/v1/classify/batch` | Bearer | Classify multiple texts |
| GET | `/api/v1/classify/stats` | Bearer | System statistics |

**Request body (`POST /api/v1/classify`):**

```json
{
  "text": "The stock rose 15% after earnings.",
  "description": "Classify sentiment of this financial news (1-5 scale).",
  "options": {
    "return_probabilities": false,
    "return_raw_response": false
  }
}
```

### Health

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/v1/health` | No | Basic alive check |
| GET | `/api/v1/health/ready` | No | Ready check (models loaded) |
| GET | `/api/v1/health/live` | No | Liveness probe |

**Worker-specific health:**

```bash
# Check any worker directly
curl http://<worker-ip>:<worker-port>/api/v1/health/ready
```

---

## Configuration Reference

### Coordinator (`app/`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_MODE` | `coordinator` | `coordinator` or `monolithic` |
| `EXPERT_MAPPING_PATH` | `config/expert_machine_mapping.json` | Path to worker URL mapping |
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Listen port |
| `JWT_SECRET_KEY` | (random) | JWT signing secret — **change in production** |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU for gating models |
| `REQUEST_TIMEOUT_SECONDS` | `600` | Total request timeout (gating + worker inference) |

### Expert Worker (`expert_worker/`)

| Variable | Required | Description |
|----------|----------|-------------|
| `WORKER_MODEL_KEY` | Yes | Base model key from `experts_registry.json` |
| `WORKER_ID` | Yes | Human-readable identifier (e.g. `worker-1`) |
| `WORKER_PORT` | Yes | Port this worker listens on |
| `CUDA_VISIBLE_DEVICES` | Yes | GPU index (always `0` per machine) |
| `EXPERT_REGISTRY_PATH` | No | Path to `experts_registry.json` (auto-resolved) |

---

## Troubleshooting

### Coordinator cannot reach a worker

```bash
# On Machine 0, check if the worker URL is reachable
curl http://<worker-ip>:<port>/api/v1/health

# Common causes:
# - Firewall blocking the worker port
# - Wrong IP in config/expert_machine_mapping.json
# - Worker container not started yet
```

Open the worker port on the remote machine:
```bash
sudo ufw allow <worker-port>/tcp
# or with iptables:
sudo iptables -A INPUT -p tcp --dport <worker-port> -j ACCEPT
```

### Worker takes a long time to start

The first startup downloads the base model from HuggingFace (~14GB per model at 4-bit).
Subsequent starts load from the local cache and take ~30–90 seconds.

```bash
# Mount a persistent HuggingFace cache to avoid re-downloading
# In docker-compose-worker.yml the HF_CACHE_PATH volume handles this
```

### Out of GPU memory on Machine 0

Machine 0 runs both the coordinator (gating models ~500MB) and Worker 0 (LLM ~3.5GB).
Total requirement is ~4GB VRAM. If your GPU has less than 6GB:

- Move Worker 0 to a separate machine
- Edit `config/expert_machine_mapping.json`: change `worker-0.url` to the new machine's IP
- Remove `expert-worker-0` from `docker-compose-distributed.yml`

### Checking the routing path

Every classification response includes a `routing_path` field that shows exactly which
worker handled the request:

```
"routing_path": "japanese -> finance -> rating -> gateway:qwen2.5-7b-instruct"
```

This confirms the request went through gating on the coordinator and was dispatched to
the correct expert worker (worker-1 in this case, which hosts qwen2.5-7b-instruct).

### Running in monolithic mode (fallback)

If distributed setup is not possible, set `SERVICE_MODE=monolithic` to run the original
single-GPU architecture. All processing happens in-process with LRU model eviction
(higher latency but no additional machines required):

```bash
export SERVICE_MODE=monolithic
docker-compose -f docker/docker-compose.yml up --build
```

---

## Supported Tasks

| Domain | Task | Languages | Model |
|--------|------|-----------|-------|
| finance | `rating` (sentiment 1–5) | English, German, French | llama-2-7b-hf |
| finance | `rating` (sentiment 1–5) | Spanish, Japanese | qwen2.5-7b-instruct |
| finance | `rating` (sentiment 1–5) | Chinese | bloomz-7b1 |
| finance | `pii` (entity extraction) | EN, NL, FR, DE, IT, ES, SV | mistral-7B-Instruct-v0.3 |
| finance | `news` (category) | EN, DA, ES, PL | mistral unsloth 4-bit |
| finance | `news` (category) | Turkish | facebook/xglm-7.5B |
| finance | `esci` (product relevance E/S/C/I) | EN, JA | llama-3.1-8b-instruct |
| finance | `esci` (product relevance E/S/C/I) | Spanish | mistral-7B-Instruct-v0.3 |

---

## License

MIT
