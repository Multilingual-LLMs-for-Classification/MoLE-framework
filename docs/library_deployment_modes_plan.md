# MoLE Library — Deployment Mode Support

Add a `deployment` parameter to `MOEClassifier` so the Python library supports all three deployment scenarios of the MoLE framework: **local** (single GPU, monolithic), **remote** (HTTP client to a running coordinator), and **distributed** (gating locally + HTTP dispatch to workers).

## User Review Required

> [!IMPORTANT]
> **Backward compatibility**: `MOEClassifier()` with no arguments will still default to `deployment="local"`, preserving current behavior exactly. Existing code won't break.

> [!IMPORTANT]
> **New dependency**: The `remote` and `distributed` backends will use `httpx` (already in `requirements.txt` for the FastAPI app). This will be added to `setup.py` core dependencies. Is that acceptable, or should it be an optional extra like `pip install moe-classifier[remote]`?

---

## Proposed Changes

### moe_classifier types & enum

#### [MODIFY] [types.py](file:///C:/FYP/MoLE-framework/moe_classifier/types.py)

Add a `DeploymentMode` enum with three values:

```python
class DeploymentMode(str, Enum):
    LOCAL = "local"            # Full in-process pipeline (current behavior)
    REMOTE = "remote"          # HTTP client to a running coordinator/monolithic service
    DISTRIBUTED = "distributed"  # Gating locally + dispatch to workers via HTTP
```

Using `str, Enum` so users can pass `"local"` as a string or `DeploymentMode.LOCAL`.

---

### Backend strategy pattern

#### [NEW] [backends/\_\_init\_\_.py](file:///C:/FYP/MoLE-framework/moe_classifier/backends/__init__.py)

Empty init file for the backends subpackage.

#### [NEW] [backends/base.py](file:///C:/FYP/MoLE-framework/moe_classifier/backends/base.py)

Abstract base class defining the backend interface:

```python
class ClassifierBackend(ABC):
    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def classify(self, text, description, **kwargs) -> ClassificationResult: ...

    @abstractmethod
    def get_stats(self) -> dict: ...

    @property
    @abstractmethod
    def is_ready(self) -> bool: ...
```

#### [NEW] [backends/local.py](file:///C:/FYP/MoLE-framework/moe_classifier/backends/local.py)

Wraps the existing `PromptRoutingSystem(coordinator_only=False)` logic — essentially what `MOEClassifier.classify()` does today. Moved here so `classifier.py` stays clean.

#### [NEW] [backends/remote.py](file:///C:/FYP/MoLE-framework/moe_classifier/backends/remote.py)

Thin HTTP client that calls a running MoLE service (coordinator or monolithic):

- Constructor takes `coordinator_url` (e.g. `"http://10.8.100.21:8000"`)
- Optional `api_key` / `credentials` for JWT auth (`username`/`password` auto-login)
- `classify()` → `POST /api/v1/classify` with the existing API schema
- `get_stats()` → `GET /api/v1/classify/stats`
- Uses `httpx.Client` (sync) for simplicity in SDK usage

#### [NEW] [backends/distributed.py](file:///C:/FYP/MoLE-framework/moe_classifier/backends/distributed.py)

Embeds the coordinator logic without the FastAPI layer:

- Constructor takes `expert_mapping` path (defaults to `config/expert_machine_mapping.json`)
- `initialize()` → creates `PromptRoutingSystem(coordinator_only=True)` + reads the mapping JSON
- `classify()` → calls `run_gating()` then dispatches via `httpx.Client` to the resolved worker URL at `/api/v1/expert/classify` (same payload structure as `GatewayService.dispatch()`)
- `get_stats()` → delegates to `PromptRoutingSystem.get_system_stats()`

---

### Classifier refactor

#### [MODIFY] [classifier.py](file:///C:/FYP/MoLE-framework/moe_classifier/classifier.py)

Refactor `MOEClassifier` to delegate to a backend:

```python
class MOEClassifier:
    def __init__(
        self,
        deployment: Union[str, DeploymentMode] = "local",
        *,
        coordinator_url: str = None,          # for "remote"
        credentials: dict = None,             # for "remote" ({"username": ..., "password": ...})
        expert_mapping: str = None,           # for "distributed"
    ) -> None:
```

- **`__init__`**: Validates arguments, stores config, does NOT load models yet.
- **`initialize()`**: Creates the appropriate backend and calls `backend.initialize()`.
- **`classify()`** / **`classify_batch()`** / **`get_stats()`**: Delegate to `self._backend`.

Changes are minimal — the current classify/batch logic moves into `LocalBackend` and the classifier becomes a thin dispatcher.

---

### Exports & packaging

#### [MODIFY] [\_\_init\_\_.py](file:///C:/FYP/MoLE-framework/moe_classifier/__init__.py)

Add `DeploymentMode` to `__all__` exports.

#### [MODIFY] [setup.py](file:///C:/FYP/MoLE-framework/setup.py)

Add `httpx>=0.24.0` to `install_requires` (or to a new `extras_require["remote"]` if preferred).

---

### Documentation & examples

#### [MODIFY] [basic_usage.py](file:///C:/FYP/MoLE-framework/examples/basic_usage.py)

Add examples showing all three modes:

```python
# Local (unchanged)
clf = MOEClassifier()

# Remote
clf = MOEClassifier(deployment="remote", coordinator_url="http://localhost:8000",
                    credentials={"username": "alice", "password": "secret"})

# Distributed
clf = MOEClassifier(deployment="distributed", expert_mapping="config/expert_machine_mapping.json")
```

---

## Verification Plan

### Automated Tests

The existing tests in `tests/` are HTTP-level integration tests (`test_api.py`, `test_auth.py`, `test_health.py`) that require a running service. They are not pytest unit tests.

We will create a new pytest test file:

#### [NEW] `tests/test_classifier_modes.py`

| Test | What it verifies |
|------|-----------------|
| `test_default_deployment_is_local` | `MOEClassifier()` defaults to `DeploymentMode.LOCAL` |
| `test_deployment_string_coercion` | `MOEClassifier(deployment="remote")` correctly converts to enum |
| `test_invalid_deployment_raises` | `MOEClassifier(deployment="invalid")` raises `ValueError` |
| `test_remote_requires_url` | `MOEClassifier(deployment="remote").initialize()` raises if no `coordinator_url` |
| `test_distributed_requires_mapping` | Validates the mapping path config |
| `test_remote_backend_classify` | Mocks `httpx.Client.post` and verifies the remote backend sends the correct HTTP request and parses the response |
| `test_distributed_backend_gating` | Mocks `PromptRoutingSystem.run_gating` and `httpx.Client.post`, verifies the distributed backend chains gating → HTTP dispatch correctly |
| `test_backward_compat_classify` | Verifies that `MOEClassifier()` (no args) produces the same API surface as the current v1.0.0 |

**Run command:**
```bash
cd C:\FYP\MoLE-framework
python -m pytest tests/test_classifier_modes.py -v
```

### Manual Verification

Since the local and distributed modes require GPU + model weights to actually run inference, automated tests will mock the heavy components. For end-to-end validation:

1. **Remote mode** (easiest to test without GPU): If you have a MoLE service running (even on the lab machines), you can test:
   ```python
   clf = MOEClassifier(deployment="remote", coordinator_url="http://<your-coordinator-ip>:8000",
                       credentials={"username": "testuser", "password": "testpassword123"})
   clf.initialize()
   result = clf.classify(text="Great product!", description="Rate 1-5.")
   print(result)
   ```

2. **Backward compatibility**: Run the existing `examples/basic_usage.py` unchanged — it should work identically.

> [!NOTE]
> I'd like your guidance on whether `httpx` should be a core dependency or an optional extra (`pip install moe-classifier[remote]`). Core is simpler; optional keeps the base install lighter for users who only use local mode.
