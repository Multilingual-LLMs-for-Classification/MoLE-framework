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
    #   "auto"  — connect to an existing Ray cluster (standard for production)
    #   "local" — start a local single-machine Ray instance (useful for testing)
    ray_address: str = "auto"

    # Path to experts_registry.json passed to each worker actor so it can build
    # its TaskExpert instances.
    expert_registry_path: str = str(
        _PROJECT_ROOT / "moe_router" / "experts" / "config" / "experts_registry.json"
    )


# Global settings instance
settings = Settings()
