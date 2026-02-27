# Model Weights

This directory stores trained model weights required for inference.
These files are **not tracked by git** due to their large size.

## Required Models

### Gating Models (placed in `moe_router/gating/models/`)

| File | Description | Size |
|------|-------------|------|
| `domain_xlmr/domain_cls.pt` | XLM-R domain classifier checkpoint | ~5MB |
| `task_routers_qlearning/encoder.pth` | Shared XLM-R encoder for task routing | ~1.1GB |
| `task_routers_qlearning/router_finance.pth` | Q-learning router for finance domain | ~5MB |
| `task_routers_qlearning/qrouter_config.json` | Router configuration | <1KB |

### Language Model (`moe_router/models/`)

| File | Description | Size |
|------|-------------|------|
| `lid.176.bin` | FastText language identification model | ~130MB |

### LoRA Adapter Weights (placed in `moe_router/experts/llms/adapters/finance/`)

These are task-specific LoRA adapter weights for the base LLMs:

- `sentiment_analysis/llama-2-7b-hf/` - Sentiment (EN, DE, FR)
- `sentiment_analysis/qwen2.5/` - Sentiment (ES, JA)
- `sentiment_analysis/bloomz-7b1/` - Sentiment (ZH)
- `pii/Mistral-7B-Instruct-v0.3/` - PII extraction (7 languages)
- `news_classification/mistral-7b/` - News classification (EN, DA, ES, PL)
- `news_classification/xglm-7.5B/` - News classification (TR)
- `esci/LLama-3-8.1B/` - ESCI classification (EN, JA)
- `esci/Mistral-7B/` - ESCI classification (ES)

## How to Get the Weights

### Option 1: Copy from training machine

```bash
# Copy gating models
cp -r /path/to/moe-router/src/gating/models/* moe_router/gating/models/

# Copy language model
cp /path/to/moe-router/models/lid.176.bin moe_router/models/

# Copy adapter weights (safetensors files)
# These are inside the adapter directories under experts/llms/adapters/finance/
```

### Option 2: Download script

```bash
./scripts/download_models.sh
```

### Option 3: Docker volume mount

If using Docker, you can mount pre-existing model directories as volumes
(see `docker/docker-compose.yml` for configuration).
