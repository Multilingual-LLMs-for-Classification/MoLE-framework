#!/bin/bash
# Start MoLE worker-1 (qwen2.5-7b-instruct) on port 8002

cd /home/cse_g4/MoLE-framework

export WORKER_MODEL_KEY=qwen2.5-7b-instruct
export WORKER_ID=worker-1
export WORKER_PORT=8002
export CUDA_VISIBLE_DEVICES=0
export EXPERT_REGISTRY_PATH=/home/cse_g4/MoLE-framework/moe_router/experts/config/experts_registry.json
export HF_HOME=/home/cse_g4/.cache/huggingface
export TRANSFORMERS_CACHE=/home/cse_g4/.cache/huggingface
export MAX_CONCURRENT_GPU_REQUESTS=1

UVICORN=/home/cse_g4/miniconda3/envs/cfenv/bin/uvicorn
LOG=/home/cse_g4/MoLE-framework/worker-1.log

nohup "$UVICORN" expert_worker.main:app \
    --host 0.0.0.0 \
    --port 8002 \
    --workers 1 \
    >> "$LOG" 2>&1 &

echo $! > /home/cse_g4/MoLE-framework/worker-1.pid
echo "Worker-1 started (PID: $!). Logs: $LOG"
