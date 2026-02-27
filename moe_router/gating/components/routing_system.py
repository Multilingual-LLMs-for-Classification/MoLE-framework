"""
Main routing system orchestrator.

Coordinates language detection, domain classification, task routing,
and expert execution. This is the top-level component that ties everything together.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch

from moe_router.experts.util.domain_task_loader import DomainTaskLoader
from moe_router.experts.util.model_loader import ModelLoader
from moe_router.experts.llms.task_expert import TaskExpert, TaskExpertConfig
from moe_router.experts.llms.expert_pool import LLMAdapterPool

from .language_detector import LanguageDetector
from .domain_classifier import DomainClassifier
from .q_learning_router import QLearningTaskClassifier

# Get DEVICE constant
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Resolve project root (moe_router package root)
_PACKAGE_ROOT = Path(__file__).parents[2]  # moe_router/


@dataclass
class GatingResult:
    """Result from the lightweight gating pipeline (no LLM involved)."""
    language: str
    domain: str
    task: str
    base_model_key: str
    adapter_name: str
    routing_path: str


class PromptRoutingSystem:
    def __init__(self, training_mode: bool = False, coordinator_only: bool = False):
        config_path = _PACKAGE_ROOT / "experts" / "config"
        self.expert_registry_path = config_path / "experts_registry.json"
        self.coordinator_only = coordinator_only

        # Load registry JSON for model resolution (needed in both modes)
        with open(self.expert_registry_path, "r", encoding="utf-8") as f:
            self._registry = json.load(f)

        # Initialize language detector with registry path
        self.language_detector = LanguageDetector(registry_path=self.expert_registry_path)
        self.domain_classifier = DomainClassifier(
            model_name="xlm-roberta-base",
            model_dir=_PACKAGE_ROOT / "gating" / "models" / "domain_xlmr",
            max_len=128,
            alpha_proto=0.30,
            proto_temp=10.0
        )

        # ModelLoader is optional (legacy component for external model downloads)
        # Not needed when using LLMAdapterPool for model management
        try:
            self.model_loader = ModelLoader(config_path / "model_config.json")
        except FileNotFoundError:
            print("model_config.json not found - skipping legacy ModelLoader (not needed for LLMAdapterPool)")
            self.model_loader = None

        self.domain_tasks_obj = DomainTaskLoader(config_path / "experts_registry.json")
        self.domain_tasks = self.domain_tasks_obj.domain_tasks if hasattr(self.domain_tasks_obj, "domain_tasks") else self.domain_tasks_obj

        # Q-learning task classifier
        self.task_classifier = QLearningTaskClassifier(
            self.domain_tasks,
            model_dir=_PACKAGE_ROOT / "gating" / "models" / "task_routers_qlearning",
            encoder_name="xlm-roberta-base",
            max_len=128,
            batch_size=16,
            lr=1e-5,
            epochs=1,
            eps_start=0.2,
            eps_end=0.01,
            eps_decay_steps=10000
        )
        # Only load existing models when not in training mode
        if not training_mode:
            self.domain_classifier.load_model()
            self.task_classifier.load_models()

        # Download any external models if needed (optional legacy feature)
        if self.model_loader:
            print("Checking and downloading models if needed...")
            self.model_loader.download_all_models()

        if coordinator_only:
            # Coordinator mode: gating pipeline only, no LLM pool or experts
            self.expert_pool = None
            self.experts = {}
            print("Coordinator mode: expert pool not loaded (gating pipeline only)")
        else:
            # Monolithic / worker mode: load full expert pool
            self.expert_pool = LLMAdapterPool(self.expert_registry_path)

            # Instantiate experts per domain/task using the registry
            self.experts = {}
            for domain, tasks in self.domain_tasks.items():
                self.experts[domain] = {}
                for task in tasks.keys():
                    self.experts[domain][task] = TaskExpert(
                        TaskExpertConfig(
                            domain=domain,
                            task=task,
                            registry_path=str(self.expert_registry_path),
                            generation=None  # or per-task overrides dict
                        ),
                        pool=self.expert_pool
                    )
            expert_count = sum(len(tasks) for tasks in self.experts.values())
            print(f"Initialized {expert_count} task experts across {len(self.experts)} domains")

    def save_all_models(self):
        print("Saving all models...")
        self.domain_classifier.save_model()
        self.task_classifier.save_models()
        print("All models saved successfully!")

    def train_domain_classifier(self, training_data: List[Dict], **kwargs):
        """
        Train the transformer domain classifier on labeled prompts.
        kwargs are passed to fit_from_labeled_prompts (epochs, batch_size, lr, freeze_encoder, ...).
        """
        print("Training Domain Classifier (Transformer)...")
        self.domain_classifier.fit_from_labeled_prompts(training_data, **kwargs)
        self.domain_classifier.save_model()

    def train_q_routers(self, training_data: List[Dict]):
        """Train per-domain QRouters on labeled (domain, task, prompt) items."""
        self.task_classifier.train(training_data, val_split=0.1)
        self.task_classifier.save_models()

    def _resolve_model_for_task(self, task_key: str, language: str) -> Tuple[str, str]:
        """
        Returns (base_model_key, adapter_name) for the given task + detected language.
        Mirrors the logic in LLMAdapterPool._resolve_base_model_for_language().
        """
        tcfg = self._registry["tasks"].get(task_key)
        if not tcfg:
            raise ValueError(f"Task '{task_key}' not found in registry")

        default_base = tcfg.get("base_model_key")
        default_adapter = tcfg.get("adapter_name")
        lang_mapping = tcfg.get("language_mapping")

        if not lang_mapping or not language:
            return default_base, default_adapter

        lang_lower = language.lower()

        # Priority 1: direct per-language entry
        if lang_lower in lang_mapping:
            cfg = lang_mapping[lang_lower]
            if "languages" not in cfg:
                return (
                    cfg.get("base_model_key", default_base),
                    cfg.get("adapter_name", default_adapter)
                )

        # Priority 2: group membership
        for group_cfg in lang_mapping.values():
            if lang_lower in group_cfg.get("languages", []):
                return (
                    group_cfg.get("base_model_key", default_base),
                    group_cfg.get("adapter_name", default_adapter)
                )

        # Priority 3: fallback to task default
        return default_base, default_adapter

    def run_gating(self, prompt: str) -> GatingResult:
        """
        Lightweight gating pipeline — language detection, domain classification,
        task routing, and expert selection via registry lookup.
        No LLM involved. Runs on the coordinator.

        Returns:
            GatingResult with language, domain, task, base_model_key, adapter_name
        """
        language = self.language_detector.detect_language(prompt)
        domain = self.domain_classifier.classify_domain(prompt)
        task = self.task_classifier.classify_task(prompt, domain)
        task_key = f"{domain}/{task}"
        base_model_key, adapter_name = self._resolve_model_for_task(task_key, language)
        return GatingResult(
            language=language,
            domain=domain,
            task=task,
            base_model_key=base_model_key,
            adapter_name=adapter_name,
            routing_path=f"{language} -> {domain} -> {task}"
        )

    def route_prompt(self, prompt: str, classification_text: str = None,
                    review_title: str = None, input_data: Dict[str, str] = None) -> Dict:
        """
        Generic routing method supporting multiple task types.

        Args:
            prompt: Full prompt with instructions
            classification_text: (Legacy) Text to classify (use input_data instead)
            review_title: (Legacy) Title text (use input_data instead)
            input_data: Task-specific data fields (e.g., {"text": "...", "title": "..."})

        Returns:
            Dict with routing results including language, domain, task, result

        Usage:
            # New way (preferred):
            system.route_prompt(prompt, input_data={"text": "...", "title": "..."})

            # Old way (backward compatible):
            system.route_prompt(prompt, classification_text, review_title)
        """
        # Handle backward compatibility
        if input_data is None:
            # Legacy mode: construct input_data from positional arguments
            input_data = {
                "classification_text": classification_text or "",
                "review_title": review_title or ""
            }

        language = self.language_detector.detect_language(prompt)
        domain = self.domain_classifier.classify_domain(prompt)
        domain_probs = self.domain_classifier.get_domain_probabilities(prompt)
        task = self.task_classifier.classify_task(prompt, domain)
        expert = self.experts[domain][task]

        # Pass input_data directly to expert - it handles field extraction.
        # Support both legacy 3-value and current 5-value expert return formats.
        prediction = expert.predict(input_data, prompt, language)
        base_model_key = None
        prompt_sent = None

        if isinstance(prediction, tuple):
            if len(prediction) >= 5:
                result, expert_confidence, raw_response, base_model_key, prompt_sent = prediction[:5]
            elif len(prediction) >= 3:
                result, expert_confidence, raw_response = prediction[:3]
            else:
                raise ValueError(
                    f"Unexpected expert.predict tuple length: {len(prediction)} for {domain}/{task}"
                )
        else:
            raise TypeError(
                f"expert.predict must return a tuple, got {type(prediction).__name__} for {domain}/{task}"
            )
        output = {
            'input': prompt,
            'input_data': input_data,  # Include for downstream use
            'language': language,
            'domain': domain,
            'domain_probabilities': domain_probs,
            'task': task,
            'result': result,
            'expert_confidence': expert_confidence,
            'routing_path': f"{language} -> {domain} -> {task}",
            'raw_response': raw_response,
            'base_model_key': base_model_key,
            'prompt_sent': prompt_sent
        }
        return output

    def get_system_stats(self):
        total_tasks = sum(len(tasks) for tasks in self.domain_tasks.items())
        all_languages = self.language_detector.all_supported_languages
        return {
            'total_domains': len(self.domain_tasks),
            'total_tasks': total_tasks,
            'supported_languages': len(all_languages),
            'all_languages': sorted(all_languages),
            'languages_by_task': self.language_detector.supported_languages_by_task,
            'domains': list(self.domain_tasks.keys())
        }
