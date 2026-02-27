#!/bin/bash
# Download required model weights for the classification service.
#
# Usage:
#   ./scripts/download_models.sh [--source PATH]
#
# Options:
#   --source PATH   Copy models from an existing moe-router installation
#                   (e.g., --source /path/to/moe-router)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "MOE Classification Service - Model Downloader"
echo "=============================================="

# Parse arguments
SOURCE_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --- FastText Language Model ---
FASTTEXT_DIR="${PROJECT_ROOT}/moe_router/models"
FASTTEXT_MODEL="${FASTTEXT_DIR}/lid.176.bin"

if [ ! -f "$FASTTEXT_MODEL" ]; then
    mkdir -p "$FASTTEXT_DIR"
    if [ -n "$SOURCE_PATH" ] && [ -f "${SOURCE_PATH}/models/lid.176.bin" ]; then
        echo "Copying FastText model from source..."
        cp "${SOURCE_PATH}/models/lid.176.bin" "$FASTTEXT_MODEL"
    else
        echo "Downloading FastText language identification model..."
        curl -L -o "$FASTTEXT_MODEL" \
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    fi
    echo "FastText model ready: $FASTTEXT_MODEL"
else
    echo "FastText model already exists: $FASTTEXT_MODEL"
fi

# --- Gating Models ---
GATING_MODELS_DIR="${PROJECT_ROOT}/moe_router/gating/models"

if [ -n "$SOURCE_PATH" ]; then
    SRC_GATING="${SOURCE_PATH}/src/gating/models"
    if [ -d "$SRC_GATING" ]; then
        echo "Copying gating models from source..."

        # Domain classifier
        mkdir -p "${GATING_MODELS_DIR}/domain_xlmr"
        if [ -f "${SRC_GATING}/domain_xlmr/domain_cls.pt" ]; then
            cp "${SRC_GATING}/domain_xlmr/domain_cls.pt" "${GATING_MODELS_DIR}/domain_xlmr/"
            echo "  Copied domain_cls.pt"
        fi

        # Q-learning task routers
        mkdir -p "${GATING_MODELS_DIR}/task_routers_qlearning"
        for f in encoder.pth router_finance.pth qrouter_config.json; do
            if [ -f "${SRC_GATING}/task_routers_qlearning/$f" ]; then
                cp "${SRC_GATING}/task_routers_qlearning/$f" "${GATING_MODELS_DIR}/task_routers_qlearning/"
                echo "  Copied $f"
            fi
        done

        echo "Gating models copied successfully"
    else
        echo "WARNING: Gating models not found at ${SRC_GATING}"
    fi
else
    echo ""
    echo "NOTE: Gating models (domain_cls.pt, encoder.pth, router_finance.pth)"
    echo "must be copied manually from your training environment."
    echo "Use: ./scripts/download_models.sh --source /path/to/moe-router"
fi

# --- LoRA Adapter Weights ---
if [ -n "$SOURCE_PATH" ]; then
    SRC_ADAPTERS="${SOURCE_PATH}/src/experts/llms/adapters/finance"
    DST_ADAPTERS="${PROJECT_ROOT}/moe_router/experts/llms/adapters/finance"

    if [ -d "$SRC_ADAPTERS" ]; then
        echo "Copying LoRA adapter weights from source..."

        # Copy only model files (safetensors, bin, json configs)
        for task_dir in "$SRC_ADAPTERS"/*/; do
            task_name=$(basename "$task_dir")
            for model_dir in "$task_dir"/*/; do
                if [ -d "$model_dir" ]; then
                    model_name=$(basename "$model_dir")
                    dst_dir="${DST_ADAPTERS}/${task_name}/${model_name}"
                    mkdir -p "$dst_dir"

                    # Copy safetensors and config files
                    for ext in safetensors bin json; do
                        for f in "$model_dir"/*."$ext"; do
                            if [ -f "$f" ]; then
                                cp "$f" "$dst_dir/"
                            fi
                        done
                    done
                    echo "  Copied adapters for ${task_name}/${model_name}"
                fi
            done
        done
        echo "LoRA adapter weights copied successfully"
    else
        echo "WARNING: Adapter weights not found at ${SRC_ADAPTERS}"
    fi
else
    echo ""
    echo "NOTE: LoRA adapter weights must be copied manually."
    echo "Use: ./scripts/download_models.sh --source /path/to/moe-router"
fi

echo ""
echo "=============================================="
echo "Model setup complete!"
echo "=============================================="
