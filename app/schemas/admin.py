"""
Admin configuration response schemas.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class GenerationConfig(BaseModel):
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: Optional[float] = None


class BaseModelConfig(BaseModel):
    hf_name: str
    load_in_4bit: bool
    device_map: str


class LanguageMappingEntry(BaseModel):
    base_model_key: str
    adapter_name: str
    adapter_path: str
    template_path: str


class TaskConfig(BaseModel):
    base_model_key: str
    adapter_name: str
    adapter_path: str
    expert_path: str
    template_path: str
    label_set: List[str]
    strict_label_decoding: bool
    constrained_single_token: Optional[bool] = None
    supported_languages: List[str]
    generation: Optional[GenerationConfig] = None
    language_mapping: Dict[str, LanguageMappingEntry]


class ExpertsRegistryResponse(BaseModel):
    default_generation: GenerationConfig
    base_models: Dict[str, BaseModelConfig]
    tasks: Dict[str, TaskConfig]


class TaskTemplateResponse(BaseModel):
    task_key: str
    templates: Dict[str, str]


class DomainClassifierConfig(BaseModel):
    model_name: str
    model_dir: Optional[str] = None
    max_len: int
    alpha_proto: float
    proto_temp: float
    epochs: int
    batch_size: int
    lr: float
    freeze_encoder: bool
    class_weighting: bool


class QLearningConfig(BaseModel):
    encoder_name: str
    model_dir: Optional[str] = None
    max_len: int
    batch_size: int
    lr: float
    epochs: int
    eps_start: float
    eps_end: float
    eps_decay_steps: int
    val_split: float


class LanguageRouterConfig(BaseModel):
    registry_path: str
    fasttext_model_path: Optional[str] = None
    chunk_size: int


class TrainingConfig(BaseModel):
    data_path: str
    enable_training: bool


class EvaluationConfig(BaseModel):
    test_data_path: str
    test_n: Optional[int] = None
    output_path: str
    enable_evaluation: bool


class RouterConfigResponse(BaseModel):
    expert_registry_path: str
    domain_tasks_path: str
    model_config_path: str
    language_config: LanguageRouterConfig
    domain_config: DomainClassifierConfig
    qlearning_config: QLearningConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


class TaskSummary(BaseModel):
    task_key: str
    base_model_key: str
    adapter_name: str
    label_count: int
    language_count: int
    supported_languages: List[str]


class ConfigOverviewResponse(BaseModel):
    total_base_models: int
    total_tasks: int
    total_languages: int
    default_generation: GenerationConfig
    tasks: List[TaskSummary]


# ── Request models for config updates ──

class TaskConfigUpdate(BaseModel):
    base_model_key: Optional[str] = None
    adapter_name: Optional[str] = None
    adapter_path: Optional[str] = None
    expert_path: Optional[str] = None
    template_path: Optional[str] = None
    label_set: Optional[List[str]] = None
    strict_label_decoding: Optional[bool] = None
    constrained_single_token: Optional[bool] = None
    generation: Optional[GenerationConfig] = None


class LanguageMappingUpdate(BaseModel):
    base_model_key: str
    adapter_name: str
    adapter_path: str
    template_path: Optional[str] = None


class DefaultGenerationUpdate(BaseModel):
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None


class TemplateUpdate(BaseModel):
    templates: Dict[str, str]


class ReloadResponse(BaseModel):
    success: bool
    message: str
