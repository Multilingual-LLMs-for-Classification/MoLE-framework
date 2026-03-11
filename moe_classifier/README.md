# moe-classifier

Python SDK for the **Multilingual Mixture-of-Experts (MOE) Classification Pipeline**.

Classify text across multiple domains and tasks — sentiment rating, PII detection, news categorization, and e-commerce relevance — in 176 languages, with automatic language detection, domain routing, and task selection handled entirely by the pipeline.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Supported Tasks](#supported-tasks)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [MOEClassifier](#moeclassifier)
  - [PipelineConfig](#pipelineconfig)
  - [MOETrainer](#moetrainer)
  - [ClassificationResult](#classificationresult)
  - [BatchResult](#batchresult)
  - [BatchItem](#batchitem)
- [Component Customization](#component-customization)
- [The `description` Parameter](#the-description-parameter)
- [Batch Classification](#batch-classification)
- [Error Handling](#error-handling)
- [Optional Response Fields](#optional-response-fields)
- [Performance Notes](#performance-notes)

---

## How It Works

Every classification request passes through a four-stage hierarchical pipeline:

```
Input text
    │
    ▼
┌─────────────────────┐
│  Language Detector  │  FastText (176 languages)
│                     │  → 'english', 'japanese', 'spanish', ...
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Domain Classifier  │  XLM-RoBERTa + prototype ensembling
│                     │  → 'finance', 'general', ...
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Task Router       │  Q-Learning (per-domain neural router)
│                     │  → 'rating', 'pii', 'news', 'esci', ...
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Expert LLM        │  LoRA-adapted 7B LLM (language-specific)
│                     │  → final classification result
└─────────────────────┘
```

The pipeline selects the right base model and LoRA adapter automatically based on the detected language and routed task, so callers only need to provide text.

---

## Supported Tasks

| Task key | Domain | Description | Output |
|---|---|---|---|
| `finance/rating` | Finance | Sentiment rating of a product/service review | `1` – `5` (star rating) |
| `finance/pii` | Finance | PII entity detection in financial documents | Comma-separated entity types |
| `finance/news` | Finance | News article topic classification | Category name |
| `finance/esci` | Finance | E-commerce search relevance judgement | `E` / `S` / `C` / `I` |

### Output details

**`finance/rating`**
Rates a review on a 1–5 scale: `"1"` (very negative) to `"5"` (very positive).

**`finance/pii`**
Returns one or more of: `person_name`, `date`, `location`, `organization`,
`contact_info`, `government_id`, `financial_account`, `payment_card`,
`user_identifier`, `secret`, `ip_address`.

**`finance/news`**
Categories: `Finance`, `Tax & Accounting`, `Government & Controls`,
`Technology`, `Industry`, `Business & Management`.

**`finance/esci`**
- `E` — Exact match (product directly satisfies the query)
- `S` — Substitute (related but not exact)
- `C` — Complement (goes well with the query item)
- `I` — Irrelevant

---

## Installation

Install in editable mode from the `moe-classification-service/` root directory:

```bash
cd /path/to/moe-classification-service
pip install -e .
```

This installs both `moe_classifier` (the SDK) and `moe_router` (the underlying engine).

### Dependencies

| Package | Purpose |
|---|---|
| `torch >= 2.0` | Model inference |
| `transformers >= 4.30` | XLM-RoBERTa encoder & LLM tokenization |
| `peft >= 0.7` | LoRA adapter loading |
| `fasttext-wheel >= 0.9` | Language detection |
| `bitsandbytes >= 0.41` | 4-bit quantization (reduces VRAM usage) |
| `accelerate >= 0.24` | Multi-device model placement |
| `scikit-learn >= 1.3` | Domain classifier utilities |
| `numpy >= 1.24` | Numerical operations |

---

## Quick Start

```python
from moe_classifier import MOEClassifier

# 1. Create and initialize (loads all models — do this once)
clf = MOEClassifier()
clf.initialize()

# 2. Classify a single text
result = clf.classify(
    text="This product exceeded my expectations! Great quality and fast shipping.",
    description="Rate this product review from 1 to 5 stars.",
)

print(result.result)        # "5"
print(result.language)      # "english"
print(result.domain)        # "finance"
print(result.task)          # "rating"
print(result.routing_path)  # "english -> finance -> rating"
print(result.confidence)    # 0.94
```

---

## API Reference

### MOEClassifier

```python
class MOEClassifier
```

Main entry point for the SDK. The underlying models are large, so initialization is explicit — create the instance cheaply, then call `initialize()` once before any classification.

---

#### `MOEClassifier(deployment="local", *, pipeline_config=None, ...)`

```python
clf = MOEClassifier()                          # all defaults
clf = MOEClassifier(pipeline_config=config)     # with custom components
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `deployment` | `str` or `DeploymentMode` | `"local"` | Deployment mode: `"local"`, `"remote"`, or `"distributed"`. |
| `pipeline_config` | `PipelineConfig` | `None` | Optional component overrides — custom model paths, encoder names, or expert registries. See [PipelineConfig](#pipelineconfig). |
| `coordinator_url` | `str` | `None` | URL for remote mode. |
| `credentials` | `dict` | `None` | `{"username": ..., "password": ...}` for remote mode. |
| `token` | `str` | `None` | Pre-existing JWT token for remote mode. |
| `expert_mapping` | `str` | `None` | Path to `expert_machine_mapping.json` for distributed mode. |

No models are loaded at construction time — call `initialize()` to load them.

---

#### `clf.initialize() -> None`

Loads all pipeline models into memory (language detector, domain classifier, task router, and LLM expert pool). This is a blocking call that may take **30–120 seconds** on first run while models are downloaded and loaded onto the GPU.

Call this once and reuse the same `clf` instance for all subsequent requests.

```python
clf.initialize()
print(clf.is_ready)  # True
```

**Raises:**
- `RuntimeError` — if `moe_router` is not found (not installed) or if model loading fails.

---

#### `clf.is_ready -> bool`

`True` after `initialize()` has completed successfully, `False` otherwise.

```python
if not clf.is_ready:
    clf.initialize()
```

---

#### `clf.classify(text, description="", *, return_domain_probabilities=False, return_raw_response=False) -> ClassificationResult`

Classify a single piece of text.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | *(required)* | The text to classify. |
| `description` | `str` | `""` | Free-text task hint prepended to the routing prompt. Helps the router pick the right expert when the text alone is ambiguous. See [The description parameter](#the-description-parameter). |
| `return_domain_probabilities` | `bool` | `False` | Include the full domain probability distribution in the result. |
| `return_raw_response` | `bool` | `False` | Include the unprocessed LLM output in the result. |

**Returns:** [`ClassificationResult`](#classificationresult)

**Raises:**
- `RuntimeError` — if `initialize()` has not been called.
- `ValueError` — if `text` is empty.

```python
result = clf.classify(
    text="John Smith's credit card number is 4111-1111-1111-1111.",
    description="Detect any PII entities in this text.",
    return_domain_probabilities=True,
)
print(result.result)               # "person_name, payment_card"
print(result.domain_probabilities) # {"finance": 0.91, "general": 0.09}
```

---

#### `clf.classify_batch(items, *, return_domain_probabilities=False, return_raw_response=False, skip_errors=True) -> BatchResult`

Classify a list of texts sequentially.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `items` | `list[dict]` | *(required)* | List of dicts with `"text"` (required) and optional `"description"`. |
| `return_domain_probabilities` | `bool` | `False` | Forwarded to each `classify()` call. |
| `return_raw_response` | `bool` | `False` | Forwarded to each `classify()` call. |
| `skip_errors` | `bool` | `True` | If `True`, failed items are captured in `BatchItem.error` and the batch continues. If `False`, the first error is re-raised immediately. |

**Returns:** [`BatchResult`](#batchresult)

**Raises:**
- `RuntimeError` — if `initialize()` has not been called.
- `ValueError` — if `items` is empty.
- Any exception from a failed item when `skip_errors=False`.

```python
batch = clf.classify_batch([
    {"text": "Excellent product!", "description": "Rate this review 1-5."},
    {"text": "Terrible quality.",  "description": "Rate this review 1-5."},
    {"text": "Query: laptop  Product: 13-inch MacBook Pro", "description": "Judge e-commerce relevance."},
])

print(batch.successful)  # 3
print(batch.failed)      # 0

for item in batch.items:
    if item.success:
        print(f"[{item.index}] {item.result.result}")
    else:
        print(f"[{item.index}] ERROR: {item.error}")
```

---

#### `clf.get_stats() -> dict`

Returns a summary of what the pipeline supports.

```python
stats = clf.get_stats()
```

**Return value keys:**

| Key | Type | Description |
|---|---|---|
| `total_domains` | `int` | Number of supported domains. |
| `total_tasks` | `int` | Total tasks across all domains. |
| `supported_languages` | `int` | Number of detectable languages. |
| `all_languages` | `list[str]` | Alphabetically sorted list of language names. |
| `languages_by_task` | `dict` | Per-task list of supported language codes. |
| `domains` | `list[str]` | Domain names (e.g. `["finance"]`). |

**Raises:**
- `RuntimeError` — if `initialize()` has not been called.

---

### PipelineConfig

```python
from moe_classifier import PipelineConfig
```

A dataclass that bundles override paths for individual pipeline components. All fields default to `None`, meaning the framework's built-in defaults are used. Set only the fields you want to override.

```python
@dataclass
class PipelineConfig
```

| Field | Type | Default | Description |
|---|---|---|---|
| `language_model` | `str \| None` | `None` | Path to a custom FastText language-identification model (`.bin`). |
| `expert_registry` | `str \| None` | `None` | Path to a custom `experts_registry.json` defining domains, tasks, adapters, and supported languages. |
| `domain_model_dir` | `str \| None` | `None` | Directory containing a trained domain classifier checkpoint (`domain_head.pt`, `domain2id.json`, `prototypes.pt`). |
| `domain_model_name` | `str` | `"xlm-roberta-base"` | HuggingFace encoder name for the domain classifier. |
| `task_router_dir` | `str \| None` | `None` | Directory containing trained Q-learning task router checkpoints (per-domain `.pt` files and `task2id_*.json`). |
| `task_encoder_name` | `str` | `"xlm-roberta-base"` | HuggingFace encoder name for the task router. |

```python
config = PipelineConfig(
    domain_model_dir="models/my_domain_classifier/",
    expert_registry="config/my_experts_registry.json",
)
clf = MOEClassifier(pipeline_config=config)
clf.initialize()
```

---

### MOETrainer

```python
from moe_classifier import MOETrainer
```

Train MoLE pipeline components (domain classifier, task routers) on your own labeled data. Creates a `PromptRoutingSystem` in training mode internally and delegates to its built-in training methods.

```python
class MOETrainer
```

---

#### `MOETrainer(pipeline_config=None)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pipeline_config` | `PipelineConfig` | `None` | Override component paths (custom registry, encoder name, etc.). If `None`, uses all framework defaults. |

---

#### `trainer.train_domain_classifier(training_data, *, epochs=3, lr=2e-5, batch_size=32, val_split=0.1, freeze_encoder=True, output_dir=None) -> dict`

Train the XLM-RoBERTa domain classifier on labeled prompts.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `training_data` | `list[dict]` | *(required)* | Each dict must have `"prompt"` and `"domain"` keys. |
| `epochs` | `int` | `3` | Number of training epochs. |
| `lr` | `float` | `2e-5` | Learning rate. |
| `batch_size` | `int` | `32` | Batch size. |
| `val_split` | `float` | `0.1` | Fraction of data for validation. |
| `freeze_encoder` | `bool` | `True` | If `True`, freeze the XLM-R encoder and only train the classification head (faster). |
| `output_dir` | `str` | `None` | Directory to save the trained model. If `None`, saves to the default framework directory. |

**Returns:** `dict` — training summary with loss and accuracy information.

```python
result = trainer.train_domain_classifier(
    training_data=[
        {"prompt": "Revenue missed expectations.", "domain": "finance"},
        {"prompt": "Patient showed improvement.", "domain": "health"},
    ],
    epochs=5,
    output_dir="models/my_domain_classifier/",
)
```

---

#### `trainer.train_task_routers(training_data, *, val_split=0.1, output_dir=None) -> None`

Train Q-learning task routers on labeled prompts.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `training_data` | `list[dict]` | *(required)* | Each dict must have `"prompt"`, `"domain"`, and `"task"` keys. |
| `val_split` | `float` | `0.1` | Fraction of data for validation. |
| `output_dir` | `str` | `None` | Directory to save trained routers. If `None`, saves to the default framework directory. |

```python
trainer.train_task_routers(
    training_data=[
        {"prompt": "Rate this product.", "domain": "finance", "task": "rating"},
        {"prompt": "Classify this article.", "domain": "finance", "task": "news"},
    ],
    output_dir="models/my_task_routers/",
)
```

---

### ClassificationResult

```python
@dataclass
class ClassificationResult
```

Returned by `clf.classify()`. All fields are populated unless noted as optional.

| Field | Type | Description |
|---|---|---|
| `language` | `str` | Detected language name. E.g. `"english"`, `"japanese"`, `"spanish"`. |
| `domain` | `str` | Classified domain. E.g. `"finance"`. |
| `task` | `str` | Routed task within the domain. E.g. `"rating"`, `"pii"`, `"news"`, `"esci"`. |
| `result` | `str` | Expert output. See [Supported Tasks](#supported-tasks) for per-task values. |
| `routing_path` | `str` | Human-readable routing trace. E.g. `"english -> finance -> rating"`. |
| `confidence` | `float \| None` | Expert model confidence score in range `[0, 1]`. May be `None` for some tasks. |
| `domain_probabilities` | `dict[str, float] \| None` | Full domain probability distribution. Only populated when `return_domain_probabilities=True`. |
| `raw_response` | `str \| None` | Unprocessed LLM output before post-processing. Only populated when `return_raw_response=True`. |
| `processing_time_ms` | `float` | End-to-end time in milliseconds for this request. |

```python
result = clf.classify(text="Great product!", description="Rate 1-5.")

print(result.result)            # "4"
print(result.confidence)        # 0.87
print(result.language)          # "english"
print(result.domain)            # "finance"
print(result.task)              # "rating"
print(result.routing_path)      # "english -> finance -> rating"
print(result.processing_time_ms)  # 1240.5
```

---

### BatchResult

```python
@dataclass
class BatchResult
```

Returned by `clf.classify_batch()`.

| Field | Type | Description |
|---|---|---|
| `items` | `list[BatchItem]` | Per-item results in the same order as the input list. |
| `total_processing_time_ms` | `float` | Total wall-clock time for the entire batch. |
| `successful` | `int` | Number of items classified without error. |
| `failed` | `int` | Number of items that raised an error. |
| `results` | `list[ClassificationResult \| None]` | Convenience property — shorthand for `[item.result for item in batch.items]`. |

```python
batch = clf.classify_batch(items)

print(f"{batch.successful}/{len(batch.items)} succeeded")
print(f"Total time: {batch.total_processing_time_ms:.0f} ms")

# Access all results directly
for result in batch.results:
    if result:
        print(result.result)
```

---

### BatchItem

```python
@dataclass
class BatchItem
```

Represents one item within a `BatchResult`.

| Field | Type | Description |
|---|---|---|
| `index` | `int` | 0-based index of this item in the original input list. |
| `result` | `ClassificationResult \| None` | Classification result. `None` if the item failed. |
| `error` | `str \| None` | Error message if classification failed, otherwise `None`. |
| `success` | `bool` | Property — `True` when `result` is not `None`. |

```python
for item in batch.items:
    if item.success:
        print(f"[{item.index}] → {item.result.result}")
    else:
        print(f"[{item.index}] FAILED: {item.error}")
```

---

## The `description` Parameter

The `description` is prepended to `text` to form the routing prompt. It is the primary signal the domain classifier and task router use to select the right expert.

```
routing_prompt = description + "\n\n" + text
```

**When to include it:**

- Always include it when your use case is clear — it significantly improves routing accuracy.
- Omit it only when the text is self-evidently domain-specific and you want fully automatic routing.

**Examples by task:**

```python
# finance/rating — sentiment star rating
clf.classify(
    text="Fast delivery and the item looks exactly as described.",
    description="Rate this product review from 1 to 5 stars based on sentiment.",
)

# finance/pii — PII detection
clf.classify(
    text="Please transfer $5,000 to John Doe, account 98765432.",
    description="Identify all personally identifiable information in this text.",
)

# finance/news — news topic
clf.classify(
    text="The Federal Reserve raised interest rates by 25 basis points today.",
    description="Classify this news article into a financial news category.",
)

# finance/esci — e-commerce relevance
clf.classify(
    text="Query: wireless headphones  Product: Sony WH-1000XM5",
    description="Judge whether this product is relevant to the search query.",
)
```

---

## Batch Classification

`classify_batch()` processes each item sequentially through the full pipeline.

### Input format

Each item in the list is a `dict` with:
- `"text"` *(required)* — the text to classify.
- `"description"` *(optional)* — task hint for the router.

```python
items = [
    {
        "text": "Absolutely loved this product, 10/10 would buy again.",
        "description": "Rate this review from 1 to 5 stars.",
    },
    {
        "text": "Broken on arrival. Very disappointed.",
        "description": "Rate this review from 1 to 5 stars.",
    },
    {
        # description is optional — router will infer from text alone
        "text": "Contact us at support@example.com or call +1-800-555-0199.",
    },
]

batch = clf.classify_batch(items)
```

### Error handling in batches

By default (`skip_errors=True`), a failure on one item does not stop the rest of the batch. The failed item gets `BatchItem.error` set to the error message and `BatchItem.result` is `None`.

```python
batch = clf.classify_batch(items, skip_errors=True)  # default

for item in batch.items:
    if item.success:
        print(item.result.result)
    else:
        print(f"Item {item.index} failed: {item.error}")
```

To raise immediately on the first failure instead:

```python
batch = clf.classify_batch(items, skip_errors=False)
```

---

## Error Handling

```python
from moe_classifier import MOEClassifier

clf = MOEClassifier()

# Calling classify before initialize() raises RuntimeError
try:
    clf.classify(text="hello")
except RuntimeError as e:
    print(e)  # "MOEClassifier is not initialized. Call classifier.initialize() first."

clf.initialize()

# Empty text raises ValueError
try:
    clf.classify(text="   ")
except ValueError as e:
    print(e)  # "'text' must be a non-empty string."

# initialize() raises RuntimeError if models cannot be loaded
try:
    clf.initialize()
except RuntimeError as e:
    print(f"Model loading failed: {e}")
```

---

## Optional Response Fields

Two optional fields on `ClassificationResult` are off by default to avoid unnecessary computation. Enable them per call:

### `return_domain_probabilities=True`

Returns the full probability distribution across all domains from the XLM-RoBERTa domain classifier. Useful for debugging routing decisions or when you need confidence in the domain assignment.

```python
result = clf.classify(
    text="Apple reported record Q4 earnings driven by iPhone sales.",
    description="Classify this news article.",
    return_domain_probabilities=True,
)

for domain, prob in sorted(result.domain_probabilities.items(), key=lambda x: -x[1]):
    print(f"  {domain}: {prob:.1%}")
# finance: 88.3%
# general: 11.7%
```

### `return_raw_response=True`

Returns the raw, unprocessed LLM output string before the expert's `clean_output()` post-processing step. Useful for debugging unexpected results.

```python
result = clf.classify(
    text="Mediocre experience overall.",
    description="Rate this review 1-5.",
    return_raw_response=True,
)

print(result.result)       # "3"  (after post-processing)
print(result.raw_response) # "The rating for this review is 3 out of 5."  (raw)
```

Both flags can be combined:

```python
result = clf.classify(
    text="...",
    return_domain_probabilities=True,
    return_raw_response=True,
)
```

---

## Component Customization

MoLE lets you override any pipeline component — swap models, provide custom checkpoints, or train the default architecture on your own data — without modifying framework code.

### Using custom models

Use `PipelineConfig` to point the pipeline at your own model files:

```python
from moe_classifier import MOEClassifier, PipelineConfig

# Override just what you need — everything else uses defaults
config = PipelineConfig(
    domain_model_dir="models/my_legal_domain_classifier/",
    expert_registry="config/my_experts_registry.json",
)

clf = MOEClassifier(pipeline_config=config)
clf.initialize()
result = clf.classify(text="Motion to suppress evidence denied.")
```

### Training on your own data

Use `MOETrainer` to train the domain classifier or task routers on your labeled data, then use the checkpoints with `PipelineConfig`:

```python
from moe_classifier import MOETrainer, PipelineConfig, MOEClassifier

# Step 1: Train domain classifier
trainer = MOETrainer()
trainer.train_domain_classifier(
    training_data=[
        {"prompt": "The defendant filed a motion to dismiss.", "domain": "legal"},
        {"prompt": "Q3 revenue exceeded expectations.", "domain": "finance"},
        {"prompt": "Patient showed improvement after treatment.", "domain": "health"},
        # ... more labeled examples
    ],
    epochs=5,
    output_dir="models/my_domain_classifier/",
)

# Step 2: Train task routers
trainer.train_task_routers(
    training_data=[
        {"prompt": "Rate this product 1-5.", "domain": "finance", "task": "rating"},
        {"prompt": "Classify this article.", "domain": "finance", "task": "news"},
        # ... more labeled examples
    ],
    output_dir="models/my_task_routers/",
)

# Step 3: Use your trained models
config = PipelineConfig(
    domain_model_dir="models/my_domain_classifier/",
    task_router_dir="models/my_task_routers/",
)
clf = MOEClassifier(pipeline_config=config)
clf.initialize()
result = clf.classify(text="Motion to suppress evidence denied.")
```

### What you can override

| Component | PipelineConfig field | What to provide |
|---|---|---|
| Language detector | `language_model` | Path to a FastText `.bin` model |
| Expert registry | `expert_registry` | Path to a custom `experts_registry.json` |
| Domain classifier | `domain_model_dir` | Directory with `domain_head.pt`, `domain2id.json`, `prototypes.pt` |
| Domain encoder | `domain_model_name` | Any HuggingFace encoder (default: `xlm-roberta-base`) |
| Task routers | `task_router_dir` | Directory with per-domain `.pt` files and `task2id_*.json` |
| Task encoder | `task_encoder_name` | Any HuggingFace encoder (default: `xlm-roberta-base`) |

---

## Performance Notes

- **Model loading** (`initialize()`) is the slow step — 30–120 seconds depending on hardware and whether models are cached. Do it once, then reuse the instance.
- **GPU is required** for practical throughput. The pipeline defaults to `device_map="auto"` and will use all available CUDA devices.
- **4-bit quantization** is enabled by default for base LLM models via `bitsandbytes`, significantly reducing VRAM requirements with minimal accuracy loss.
- **LRU eviction** — only one base LLM model is kept loaded on the GPU at a time (`max_loaded_models=1`). Switching tasks with different base models incurs a reload penalty.
- **Batch processing** is sequential (one item at a time through the GPU). For high-throughput workloads, use the web service's batch endpoint which is backed by an async semaphore for better concurrency control.
