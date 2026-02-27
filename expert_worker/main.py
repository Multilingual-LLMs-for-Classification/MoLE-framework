"""
Expert Worker — standalone FastAPI application.

Each instance of this service is responsible for exactly one base LLM, which
is pre-loaded into GPU memory at startup and never evicted.  The coordinator
service routes inference requests here after completing the lightweight gating
pipeline (language detection → domain classification → Q-learning task routing
→ expert selection).

Environment variables
---------------------
WORKER_MODEL_KEY        Base model key as it appears in experts_registry.json
                        e.g. "llama-2-7b-hf"
WORKER_ID               Human-readable worker identifier, e.g. "worker-0"
EXPERT_REGISTRY_PATH    Absolute path to experts_registry.json
                        Default: moe_router/experts/config/experts_registry.json
CUDA_VISIBLE_DEVICES    GPU to use (default "0")
"""

import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from expert_worker.single_model_pool import SingleModelPool
from expert_worker import router as classify_router
from expert_worker import health as health_router


def _resolve_registry_path() -> Path:
    env_path = os.environ.get("EXPERT_REGISTRY_PATH", "")
    if env_path:
        return Path(env_path)
    # Default: relative to the MoLE-framework project root
    return Path(__file__).parents[1] / "moe_router" / "experts" / "config" / "experts_registry.json"


def _build_experts(pool: SingleModelPool):
    """
    Instantiate TaskExpert objects for every task whose resolved base model
    matches this worker's assigned model key.  Other tasks are skipped.
    """
    import json
    from moe_router.experts.llms.task_expert import TaskExpert, TaskExpertConfig

    registry_path = pool.registry_path
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    experts: dict = {}
    for task_key, tcfg in registry.get("tasks", {}).items():
        # Collect all base_model_keys referenced by this task (default + language mappings)
        model_keys_for_task = {tcfg.get("base_model_key")}
        for lang_cfg in tcfg.get("language_mapping", {}).values():
            if isinstance(lang_cfg, dict) and "base_model_key" in lang_cfg:
                model_keys_for_task.add(lang_cfg["base_model_key"])

        if pool.assigned_model_key not in model_keys_for_task:
            continue  # This task is handled by a different worker

        domain, task = task_key.split("/", 1)
        experts.setdefault(domain, {})
        experts[domain][task] = TaskExpert(
            TaskExpertConfig(
                domain=domain,
                task=task,
                registry_path=str(registry_path),
                generation=None,
            ),
            pool=pool,
        )
        print(f"[ExpertWorker] Registered expert: {task_key}")

    return experts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; keep it resident for the process lifetime."""
    model_key = os.environ.get("WORKER_MODEL_KEY", "")
    if not model_key:
        raise RuntimeError("WORKER_MODEL_KEY environment variable must be set")

    worker_id = os.environ.get("WORKER_ID", "worker-unknown")
    registry_path = _resolve_registry_path()

    print("=" * 60)
    print(f"Expert Worker starting — model: {model_key}  id: {worker_id}")
    print("=" * 60)

    pool = SingleModelPool(model_key=model_key, registry_path=registry_path)

    # Pre-load the model in a thread so the event loop stays responsive
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, pool.preload)

    experts = _build_experts(pool)
    print(f"[ExpertWorker] {sum(len(v) for v in experts.values())} expert(s) ready.")

    app.state.worker_pool = pool
    app.state.worker_id = worker_id
    app.state.experts = experts

    yield  # Application is running

    print(f"[ExpertWorker] Shutting down worker '{worker_id}' ...")
    pool.unload()


app = FastAPI(
    title="MoLE Expert Worker",
    version="1.0.0",
    description="Single-LLM expert worker node for the distributed MoLE framework",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify_router.router, prefix="/api/v1")
app.include_router(health_router.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "MoLE Expert Worker",
        "worker_id": app.state.worker_id if hasattr(app.state, "worker_id") else "starting",
        "health": "/api/v1/health",
        "classify": "/api/v1/expert/classify",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("WORKER_PORT", "8001"))
    uvicorn.run("expert_worker.main:app", host="0.0.0.0", port=port, reload=False)
