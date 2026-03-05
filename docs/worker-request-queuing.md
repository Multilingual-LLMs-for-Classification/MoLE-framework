# Worker Request Queuing — GPU Inference Serialization

## Problem

Each expert worker hosts exactly one LLM permanently resident in GPU memory.
The worker runs as a FastAPI service, which accepts multiple concurrent HTTP
connections by default. Without any coordination, concurrent requests cause two
distinct failure modes:

### 1. Adapter race condition

`SingleModelPool` activates a LoRA adapter with `set_adapter()` and then calls
`model.generate()`. These two operations are **not atomic**. Under concurrency:

```
Request A: set_adapter("finance_en")  ─┐
Request B: set_adapter("news_pl")    ──┼─ B's adapter is now active on the shared model
Request A: model.generate()          ──┘  ← runs with B's adapter — wrong output
```

### 2. Blocking event loop

`expert.predict()` is a synchronous, CPU/GPU-bound call. The router handler is
`async def`, but calling a sync function directly inside it **blocks the entire
asyncio event loop** for the full duration of inference (potentially 10–30 s for
a 7B model). While blocked, no other coroutines can run — including health
checks, incoming request acceptance, and timeout handling.

### 3. GPU OOM from concurrent inference

If two `model.generate()` calls run simultaneously on the same device, their
combined KV-cache and activation tensors can exceed available VRAM, causing an
out-of-memory crash.

---

## Solution

Two changes were made, both in `expert_worker/`:

### 1. `asyncio.Semaphore(1)` — GPU serialization

A semaphore with a count of `1` ensures only one inference call runs on the GPU
at a time. It is created once during application startup (in the `lifespan`
context manager) and stored on `app.state` so all request handlers share the
same instance.

**`expert_worker/main.py`**

```python
app.state.gpu_semaphore = asyncio.Semaphore(1)
```

The semaphore is initialized after the model is preloaded, so it is guaranteed
to exist before any request is handled.

### 2. `loop.run_in_executor()` — non-blocking inference

`expert.predict()` is wrapped in `run_in_executor(None, ...)`, which offloads it
to the default `ThreadPoolExecutor`. This yields control back to the event loop
while inference runs in a background thread, keeping the worker responsive to
health checks and new incoming connections.

`functools.partial` is used to pass arguments to the synchronous callable:

```python
functools.partial(expert.predict, input_data, prompt, req.language)
```

**`expert_worker/router.py`**

```python
gpu_semaphore = request.app.state.gpu_semaphore
loop = asyncio.get_event_loop()
async with gpu_semaphore:
    prediction = await loop.run_in_executor(
        None, functools.partial(expert.predict, input_data, prompt, req.language)
    )
```

The `async with gpu_semaphore` block wraps the entire executor call, so the
semaphore is held for the full duration of inference and released atomically
when `predict()` returns (or raises).

---

## Behaviour after the fix

| Scenario | Before | After |
|----------|--------|-------|
| Two requests arrive simultaneously | Race on `set_adapter` + potential OOM | Second request waits in semaphore queue |
| Inference running | Event loop blocked, health checks time out | Event loop free, `/health/ready` responds normally |
| Inference crashes (OOM, exception) | Semaphore never created, state corrupted | `async with` releases semaphore on exception; next request proceeds cleanly |

---

## Files changed

| File | Change |
|------|--------|
| `expert_worker/main.py` | Added `app.state.gpu_semaphore = asyncio.Semaphore(1)` in `lifespan` |
| `expert_worker/router.py` | Added `import asyncio, functools`; wrapped `predict()` call with semaphore + `run_in_executor` |

---

## Notes

- The semaphore count is `1` (strict serial). If a future worker machine has
  multiple GPUs assigned to the same worker process, increase the count to match
  the number of GPUs (requires corresponding changes to `SingleModelPool` to
  target a specific device per inference call).
- The coordinator's gating pipeline (FastText + XLM-RoBERTa + Q-learning) was
  not changed. Gating is fast (~50 ms, mostly CPU) and its models are read-only,
  so concurrent access does not cause the same race conditions.
- This is a **per-worker** semaphore. Workers on different machines each have
  their own independent semaphore — there is no distributed locking between
  workers.
