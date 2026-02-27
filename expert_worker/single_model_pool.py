"""
SingleModelPool — an LLMAdapterPool that permanently holds exactly one base model
in GPU memory with no LRU eviction.

Inherits the full load / adapter / generation logic from LLMAdapterPool and
overrides only the memory-management parts so the model is never evicted.
"""

from pathlib import Path

from moe_router.experts.llms.expert_pool import LLMAdapterPool


class SingleModelPool(LLMAdapterPool):
    """
    LLMAdapterPool variant for expert worker nodes.

    At startup, ``preload()`` eagerly loads the assigned base model into GPU
    memory.  Afterwards, the model stays resident for the lifetime of the
    process — no LRU eviction, no CPU↔GPU transfers during inference.

    Multiple LoRA adapters can still be loaded and hot-swapped on top of the
    single resident base model (adapter activation is cheap, model reload is not).
    """

    def __init__(self, model_key: str, registry_path: Path):
        super().__init__(registry_path)
        self.assigned_model_key = model_key
        # Effectively disable eviction by setting a very high limit
        self.max_loaded_models = 999
        self._is_ready = False

    # ------------------------------------------------------------------
    # Override: disable LRU eviction entirely on this worker
    # ------------------------------------------------------------------
    def _ensure_memory_available(self, base_key_to_load: str):
        """No-op on expert worker nodes — we never evict the resident model."""
        pass

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def preload(self):
        """
        Eagerly load the assigned base model into GPU memory.
        Called once during application startup; subsequent requests hit the
        already-loaded model with zero load latency.
        """
        print(f"[SingleModelPool] Pre-loading model '{self.assigned_model_key}' ...")
        self._load_base_if_needed(self.assigned_model_key)
        self._is_ready = True
        print(f"[SingleModelPool] Model '{self.assigned_model_key}' is ready.")

    def is_ready(self) -> bool:
        return self._is_ready

    def unload(self):
        """Gracefully unload the resident model on shutdown."""
        if self.assigned_model_key in self.base_models:
            self._unload_base_model(self.assigned_model_key)
            print(f"[SingleModelPool] Model '{self.assigned_model_key}' unloaded.")
