"""
Service for reading and writing MOE system configuration files.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any

_PROJECT_ROOT = Path(__file__).parent.parent.parent  # moe-classification-service/

# Task key -> relative path to template.json
_TEMPLATE_PATHS = {
    "finance/rating": "moe_router/experts/llms/adapters/finance/sentiment_analysis/template.json",
    "finance/pii": "moe_router/experts/llms/adapters/finance/pii/template.json",
    "finance/news": "moe_router/experts/llms/adapters/finance/news_classification/template.json",
    "finance/esci": "moe_router/experts/llms/adapters/finance/esci/template.json",
}


class ConfigService:
    """Reads and writes JSON config files from the moe_router package."""

    def __init__(self):
        self._registry_path = _PROJECT_ROOT / "moe_router" / "experts" / "config" / "experts_registry.json"
        self._router_config_path = _PROJECT_ROOT / "moe_router" / "gating" / "router_config.json"

    # ── Read methods ──

    def get_experts_registry(self) -> Dict[str, Any]:
        return self._read_json(self._registry_path)

    def get_task_config(self, task_key: str) -> Dict[str, Any]:
        registry = self.get_experts_registry()
        tasks = registry.get("tasks", {})
        if task_key not in tasks:
            raise KeyError(f"Task '{task_key}' not found")
        return tasks[task_key]

    def get_task_template(self, task_key: str) -> Dict[str, str]:
        if task_key not in _TEMPLATE_PATHS:
            raise KeyError(f"No template path for task '{task_key}'")
        path = _PROJECT_ROOT / _TEMPLATE_PATHS[task_key]
        return self._read_json(path)

    def get_router_config(self) -> Dict[str, Any]:
        return self._read_json(self._router_config_path)

    # ── Write methods ──

    def update_task_config(self, task_key: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge partial updates into a task's config in experts_registry.json."""
        registry = self.get_experts_registry()
        tasks = registry.get("tasks", {})
        if task_key not in tasks:
            raise KeyError(f"Task '{task_key}' not found")
        tasks[task_key].update(updates)
        self._write_json(self._registry_path, registry)
        return tasks[task_key]

    def update_language_mapping(self, task_key: str, lang: str, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Add or update a language mapping for a task."""
        registry = self.get_experts_registry()
        tasks = registry.get("tasks", {})
        if task_key not in tasks:
            raise KeyError(f"Task '{task_key}' not found")
        task = tasks[task_key]
        task["language_mapping"][lang] = mapping
        if lang not in task.get("supported_languages", []):
            task["supported_languages"].append(lang)
        self._write_json(self._registry_path, registry)
        return task["language_mapping"][lang]

    def delete_language_mapping(self, task_key: str, lang: str) -> None:
        """Remove a language mapping from a task."""
        registry = self.get_experts_registry()
        tasks = registry.get("tasks", {})
        if task_key not in tasks:
            raise KeyError(f"Task '{task_key}' not found")
        task = tasks[task_key]
        if lang not in task.get("language_mapping", {}):
            raise KeyError(f"Language '{lang}' not found in task '{task_key}'")
        del task["language_mapping"][lang]
        if lang in task.get("supported_languages", []):
            task["supported_languages"].remove(lang)
        self._write_json(self._registry_path, registry)

    def update_default_generation(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update default_generation params in experts_registry.json."""
        registry = self.get_experts_registry()
        registry["default_generation"].update(updates)
        self._write_json(self._registry_path, registry)
        return registry["default_generation"]

    def update_task_template(self, task_key: str, templates: Dict[str, str]) -> Dict[str, str]:
        """Overwrite a task's template.json."""
        if task_key not in _TEMPLATE_PATHS:
            raise KeyError(f"No template path for task '{task_key}'")
        path = _PROJECT_ROOT / _TEMPLATE_PATHS[task_key]
        self._write_json(path, templates)
        return templates

    def reload_routing_system(self) -> bool:
        """Reset and re-initialize the routing system to pick up config changes."""
        from app.services.routing_service import routing_service
        routing_service._routing_system = None
        routing_service._initialized = False
        return routing_service.initialize()

    # ── Internal helpers ──

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: Any) -> None:
        """Atomic write: write to .tmp, backup to .bak, then replace."""
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        bak_path = path.with_suffix(path.suffix + ".bak")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        if path.exists():
            shutil.copy2(path, bak_path)
        os.replace(tmp_path, path)


config_service = ConfigService()
