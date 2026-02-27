"""
Admin configuration router for viewing and editing system config.
"""

from fastapi import APIRouter, HTTPException

from app.dependencies import CurrentUser
from app.services.config_service import config_service
from app.schemas.admin import (
    ConfigOverviewResponse,
    ExpertsRegistryResponse,
    TaskConfig,
    TaskTemplateResponse,
    RouterConfigResponse,
    TaskSummary,
    TaskConfigUpdate,
    LanguageMappingUpdate,
    DefaultGenerationUpdate,
    TemplateUpdate,
    ReloadResponse,
)

router = APIRouter(prefix="/api/v1/admin", tags=["Admin Configuration"])


@router.get("/config/overview", response_model=ConfigOverviewResponse)
async def get_config_overview(current_user: CurrentUser):
    """High-level summary of the current configuration."""
    registry = config_service.get_experts_registry()
    tasks_raw = registry.get("tasks", {})
    all_langs = set()
    task_summaries = []
    for key, t in tasks_raw.items():
        langs = t.get("supported_languages", [])
        all_langs.update(langs)
        task_summaries.append(TaskSummary(
            task_key=key,
            base_model_key=t["base_model_key"],
            adapter_name=t["adapter_name"],
            label_count=len(t.get("label_set", [])),
            language_count=len(langs),
            supported_languages=langs,
        ))
    return ConfigOverviewResponse(
        total_base_models=len(registry.get("base_models", {})),
        total_tasks=len(tasks_raw),
        total_languages=len(all_langs),
        default_generation=registry["default_generation"],
        tasks=task_summaries,
    )


@router.get("/config/experts-registry", response_model=ExpertsRegistryResponse)
async def get_experts_registry(current_user: CurrentUser):
    """Full experts registry configuration."""
    return config_service.get_experts_registry()


@router.get("/config/tasks/{task_key:path}", response_model=TaskConfig)
async def get_task_config(task_key: str, current_user: CurrentUser):
    """Config for a specific task (e.g. finance/rating)."""
    try:
        return config_service.get_task_config(task_key)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task '{task_key}' not found")


@router.get("/config/templates/{task_key:path}", response_model=TaskTemplateResponse)
async def get_task_template(task_key: str, current_user: CurrentUser):
    """Prompt templates for a specific task."""
    try:
        templates = config_service.get_task_template(task_key)
        return TaskTemplateResponse(task_key=task_key, templates=templates)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/config/router", response_model=RouterConfigResponse)
async def get_router_config(current_user: CurrentUser):
    """Router/gating configuration (domain classifier, Q-learning params)."""
    return config_service.get_router_config()


# ── Write endpoints ──


@router.patch("/config/tasks/{task_key:path}", response_model=TaskConfig)
async def update_task_config(task_key: str, body: TaskConfigUpdate, current_user: CurrentUser):
    """Partial update of a task's configuration."""
    try:
        updates = body.model_dump(exclude_none=True)
        if body.generation is not None:
            updates["generation"] = body.generation.model_dump()
        updated = config_service.update_task_config(task_key, updates)
        return updated
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task '{task_key}' not found")


@router.put("/config/tasks/{task_key:path}/languages/{lang}")
async def update_language_mapping(task_key: str, lang: str, body: LanguageMappingUpdate, current_user: CurrentUser):
    """Add or update a language mapping for a task."""
    try:
        mapping = body.model_dump(exclude_none=True)
        updated = config_service.update_language_mapping(task_key, lang, mapping)
        return updated
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/config/tasks/{task_key:path}/languages/{lang}")
async def delete_language_mapping(task_key: str, lang: str, current_user: CurrentUser):
    """Remove a language mapping from a task."""
    try:
        config_service.delete_language_mapping(task_key, lang)
        return {"detail": f"Language mapping '{lang}' removed from task '{task_key}'"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/config/default-generation")
async def update_default_generation(body: DefaultGenerationUpdate, current_user: CurrentUser):
    """Update default generation parameters."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updated = config_service.update_default_generation(updates)
    return updated


@router.put("/config/templates/{task_key:path}", response_model=TaskTemplateResponse)
async def update_task_template(task_key: str, body: TemplateUpdate, current_user: CurrentUser):
    """Replace prompt templates for a task."""
    try:
        updated = config_service.update_task_template(task_key, body.templates)
        return TaskTemplateResponse(task_key=task_key, templates=updated)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/config/reload", response_model=ReloadResponse)
async def reload_system(current_user: CurrentUser):
    """Reload the routing system to pick up config changes."""
    try:
        success = config_service.reload_routing_system()
        if success:
            return ReloadResponse(success=True, message="Routing system reloaded successfully")
        return ReloadResponse(success=False, message="Failed to initialize routing system")
    except Exception as e:
        return ReloadResponse(success=False, message=str(e))
