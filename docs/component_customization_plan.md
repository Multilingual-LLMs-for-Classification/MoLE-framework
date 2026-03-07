# MoLE Library — Component Customization

Allow SDK users to override pipeline components — swap models, provide custom model paths, or train the default architecture on their own data.

## Key Insight

The underlying classes **already support customization** — they just aren't exposed:

| Component | Class | Existing params (not exposed) |
|-----------|-------|------|
| Language detection | `LanguageDetector` | `registry_path` |
| Domain classification | `DomainClassifier` | `model_name`, `model_dir` + `fit_from_labeled_prompts()`, `save_model()`, `load_model()` |
| Task routing | `QLearningTaskClassifier` | `model_dir`, `encoder_name` + `train()`, `save_models()`, `load_models()` |
| Expert execution | `LLMAdapterPool` | `registry_path` (= `experts_registry.json`) |

**The work is mainly plumbing** — threading these params from `MOEClassifier` → `LocalBackend` → `PromptRoutingSystem` → components.

## User Review Required

> [!IMPORTANT]
> **Scope decision**: This plan adds **Level 1** (custom model paths) and **Level 2** (train-on-your-data API) to the SDK. We are NOT adding pluggable class-based component injection (e.g., `language_detector=MyDetector()`) at this stage — that would require defining abstract interfaces for every component. Is that acceptable?

> [!IMPORTANT]
> **Training API surface**: Training methods will live on a separate `MOETrainer` class for clean separation of concerns.

---

## Proposed Changes

### Phase 1: Config plumbing (custom model paths)

---

#### [NEW] [pipeline_config.py](file:///C:/FYP/MoLE-framework/moe_classifier/pipeline_config.py)

A `PipelineConfig` dataclass to bundle all component-level overrides:

```python
@dataclass
class PipelineConfig:
    """Override paths for individual pipeline components."""

    # Language detector
    language_model: str | None = None              # path to custom FastText .bin
    expert_registry: str | None = None             # path to custom experts_registry.json

    # Domain classifier
    domain_model_dir: str | None = None            # path to custom domain classifier weights
    domain_model_name: str = "xlm-roberta-base"    # base encoder name

    # Task router
    task_router_dir: str | None = None             # path to custom Q-learning weights
    task_encoder_name: str = "xlm-roberta-base"    # encoder for task routing

    # Expert pool
    expert_config: str | None = None               # path to custom experts_registry.json
                                                    # (for adapter paths, label sets, etc.)
```

Users use it like:

```python
from moe_classifier import MOEClassifier, PipelineConfig

# Override just what you need  —  everything else uses defaults
config = PipelineConfig(
    domain_model_dir="models/my_legal_domain_classifier/",
    expert_registry="config/my_experts_registry.json",
)
clf = MOEClassifier(deployment="local", pipeline_config=config)
clf.initialize()
```

---

#### [MODIFY] [classifier.py](file:///C:/FYP/MoLE-framework/moe_classifier/classifier.py)

Add `pipeline_config: PipelineConfig = None` parameter to `MOEClassifier.__init__()` and pass it to the backend.

---

#### [MODIFY] [backends/local.py](file:///C:/FYP/MoLE-framework/moe_classifier/backends/local.py)

`LocalBackend.__init__()` accepts `PipelineConfig` and passes individual fields to `PromptRoutingSystem`.

---

#### [MODIFY] [backends/distributed.py](file:///C:/FYP/MoLE-framework/moe_classifier/backends/distributed.py)

Same as local — passes `PipelineConfig` to `PromptRoutingSystem` for the gating components.

---

#### [MODIFY] [routing_system.py](file:///C:/FYP/MoLE-framework/moe_router/gating/components/routing_system.py)

Extend `PromptRoutingSystem.__init__()` to accept optional override paths:

```diff
 def __init__(
     self,
     training_mode: bool = False,
     coordinator_only: bool = False,
+    language_model: str | None = None,
+    expert_registry_path: str | None = None,
+    domain_model_dir: str | None = None,
+    domain_model_name: str = "xlm-roberta-base",
+    task_router_dir: str | None = None,
+    task_encoder_name: str = "xlm-roberta-base",
 ):
```

Each component uses the override if provided, else falls back to defaults.

---

### Phase 2: Training API

---

#### [NEW] [trainer.py](file:///C:/FYP/MoLE-framework/moe_classifier/trainer.py)

A separate `MOETrainer` class for training pipeline components:

```python
class MOETrainer:
    """Train MoLE pipeline components on custom data."""

    def __init__(self, pipeline_config: PipelineConfig = None):
        """Initialize trainer with optional config overrides."""

    def train_domain_classifier(
        self, training_data, *, epochs=3, lr=2e-5,
        output_dir=None, **kwargs
    ):
        """Train the domain classifier on labeled data.
        
        Args:
            training_data: List[dict] with keys {"prompt", "domain"}
            output_dir: Where to save the trained model
        """

    def train_task_routers(self, training_data, *, output_dir=None):
        """Train Q-learning task routers on labeled data.
        
        Args:
            training_data: List[dict] with keys {"prompt", "domain", "task"}
        """
```

Internally creates a `PromptRoutingSystem(training_mode=True)` and delegates to its existing `train_domain_classifier()` and `train_q_routers()` methods.

---

### Phase 3: Exports & docs

---

#### [MODIFY] [\_\_init\_\_.py](file:///C:/FYP/MoLE-framework/moe_classifier/__init__.py)

Add `PipelineConfig` to `__all__`.

#### [MODIFY] [types.py](file:///C:/FYP/MoLE-framework/moe_classifier/types.py) or [pipeline_config.py](file:///C:/FYP/MoLE-framework/moe_classifier/pipeline_config.py)

Export `PipelineConfig`.

#### [MODIFY] [basic_usage.py](file:///C:/FYP/MoLE-framework/examples/basic_usage.py)

Add examples for custom config and training.

---

## Example: End-to-End Custom Domain

A user adding a "legal" domain to the system:

```python
from moe_classifier import MOEClassifier, MOETrainer, PipelineConfig

# Step 1: Train domain classifier on your data
trainer = MOETrainer()
training_data = [
    {"prompt": "The defendant filed a motion to dismiss.", "domain": "legal"},
    {"prompt": "Q3 revenue exceeded expectations.", "domain": "finance"},
    # ... more labeled examples
]
trainer.train_domain_classifier(training_data, epochs=5, output_dir="models/my_domain/")

# Step 2: Use your trained model
config = PipelineConfig(
    domain_model_dir="models/my_domain/",
    expert_registry="config/my_experts_registry.json",  # with legal adapters
)
clf = MOEClassifier(pipeline_config=config)
clf.initialize()
result = clf.classify(text="Motion to suppress evidence denied.")
```

---

## Verification Plan

### Automated Tests

Extend `tests/test_classifier_modes.py` with:

| Test | What it verifies |
|------|-----------------|
| `test_pipeline_config_defaults` | `PipelineConfig()` with no args uses all `None` defaults |
| `test_pipeline_config_passed_to_backend` | Config propagates from `MOEClassifier` → backend |
| `test_custom_domain_model_dir` | Backend creates `DomainClassifier(model_dir=custom_path)` |
| `test_custom_expert_registry` | Custom registry path is used for `LanguageDetector` and `LLMAdapterPool` |
| `test_trainer_domain_classifier_delegates` | `MOETrainer().train_domain_classifier()` calls through to `PromptRoutingSystem` |

### Manual Verification

Test with actual model weights on GPU machines to verify custom paths load correctly.
