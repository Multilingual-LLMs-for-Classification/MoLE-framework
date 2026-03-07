"""
Pipeline configuration for overriding default component paths.

Users can provide custom model paths or registry files to swap
individual pipeline components without modifying framework code.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    """
    Override paths for individual pipeline components.

    All fields default to ``None``, meaning the framework's built-in
    defaults are used.  Set only the fields you want to override.

    Example::

        from moe_classifier import MOEClassifier, PipelineConfig

        config = PipelineConfig(
            domain_model_dir="models/my_legal_domain_classifier/",
            expert_registry="config/my_experts_registry.json",
        )
        clf = MOEClassifier(pipeline_config=config)
        clf.initialize()

    Attributes
    ----------
    language_model : str, optional
        Path to a custom FastText language-identification model (``.bin``).
        Default: the framework downloads ``lid.176.bin`` automatically.
    expert_registry : str, optional
        Path to a custom ``experts_registry.json`` that defines
        domains, tasks, adapters, and supported languages.
        Used by the language detector, expert selector, and LLM pool.
    domain_model_dir : str, optional
        Directory containing a trained domain classifier checkpoint
        (``domain_head.pt``, ``domain2id.json``, ``prototypes.pt``).
    domain_model_name : str
        HuggingFace encoder name for the domain classifier.
        Default: ``"xlm-roberta-base"``.
    task_router_dir : str, optional
        Directory containing trained Q-learning task router checkpoints
        (per-domain ``.pt`` files and ``task2id_*.json``).
    task_encoder_name : str
        HuggingFace encoder name for the task router.
        Default: ``"xlm-roberta-base"``.
    """

    # Language detector
    language_model: Optional[str] = None
    expert_registry: Optional[str] = None

    # Domain classifier
    domain_model_dir: Optional[str] = None
    domain_model_name: str = "xlm-roberta-base"

    # Task router
    task_router_dir: Optional[str] = None
    task_encoder_name: str = "xlm-roberta-base"
