"""
Application configuration using Pydantic Settings.
Loads from environment variables or .env file.
"""

import secrets
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root directory (where this repo is cloned)
_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Settings
    api_title: str = "MOE-Router Classification Service"
    api_version: str = "1.0.0"
    api_description: str = "Multilingual text classification using Mixture of Experts routing"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # JWT Settings
    jwt_secret_key: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # GPU Settings
    cuda_visible_devices: str = "0"
    gpu_memory_fraction: Optional[float] = None

    # Request Settings
    request_timeout_seconds: int = 600
    max_concurrent_gpu_requests: int = 1

    # Distributed mode settings
    # "coordinator" — gating pipeline only + gateway dispatch (default)
    # "monolithic"  — full pipeline including LLM in-process (original behaviour)
    service_mode: str = "coordinator"

    # Path to config/expert_machine_mapping.json (used in coordinator mode)
    expert_mapping_path: str = str(_PROJECT_ROOT / "config" / "expert_machine_mapping.json")

    # Database URL for user storage
    database_url: str = "sqlite:////app/data/users.db"

    # ---------------------------------------------------------------------------
    # Ray distributed framework settings
    # ---------------------------------------------------------------------------
    # Set use_ray=true to replace the HTTP-based GatewayService with Ray actors.
    # When false (default), the original HTTP + Docker worker setup is used.
    use_ray: bool = False

    # Ray cluster address for ray.init().
    #   "ray://localhost:10001"  — Ray Client (TCP). Use when coordinator and ray-head
    #                              are separate Docker containers. No shared filesystem
    #                              required. Port 10001 = ray-client-server-port.
    #   "localhost:6379"         — Direct GCS attach. Only works when coordinator runs
    #                              IN the same container/process as ray start --head
    #                              (needs shared /tmp/ray Unix socket).
    #   "local"                  — Start a new single-node Ray instance in-process
    #                              (testing only, no persistent cluster).
    ray_address: str = "ray://localhost:10001"

    # Path to experts_registry.json passed to each worker actor so it can build
    # its TaskExpert instances.
    expert_registry_path: str = str(
        _PROJECT_ROOT / "moe_router" / "experts" / "config" / "experts_registry.json"
    )

    # Path to config/ray_worker_registry.json (Ray-mode only).
    # This file declares which models to spawn as Ray Actors, how many replicas,
    # and optional placement group strategies.  It has no url/port fields.
    # The old expert_machine_mapping.json is still used for HTTP mode.
    ray_worker_registry_path: str = str(
        _PROJECT_ROOT / "config" / "ray_worker_registry.json"
    )

    # ---------------------------------------------------------------------------
    # Gating actor settings (Ray mode only)
    # ---------------------------------------------------------------------------
    # When use_gating_actor=true, the gating pipeline (FastText + XLM-RoBERTa +
    # Q-learning) runs inside a named Ray Actor instead of the coordinator process.
    # Benefits:
    #   - XLM-RoBERTa inference offloaded from coordinator CPU
    #   - Can run on a fractional GPU slice (gating_actor_num_gpus=0.1)
    #   - Actor is detached — survives coordinator restarts without reloading models
    # Requires use_ray=true.
    use_gating_actor: bool = False

    # GPU fraction allocated to the gating actor.
    #   0.0 — CPU only (safe default; works on any machine)
    #   0.1 — 10% of one GPU (10 gating actors share a GPU alongside LLM workers)
    # Only used when use_gating_actor=true.
    gating_actor_num_gpus: float = 0.0


# Global settings instance
settings = Settings()
