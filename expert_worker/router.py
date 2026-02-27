"""
Classification endpoint for the expert worker service.

The coordinator calls this endpoint after completing the gating pipeline.
The worker's LLM is already in GPU memory — no load latency.
"""

import time
import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class ExpertRequest(BaseModel):
    task_key: str          # e.g. "finance/rating"
    language: str          # e.g. "english"
    text: str              # raw input text for the expert
    description: str       # task description (combined with text to form the prompt)
    adapter_name: str      # pre-resolved adapter name from the coordinator
    request_id: str        # pass-through for logging / tracing


class ExpertResponse(BaseModel):
    result: str
    confidence: float
    processing_time_ms: float
    worker_id: str
    model_key: str
    request_id: str
    raw_response: Optional[str] = None


@router.post("/expert/classify", response_model=ExpertResponse)
async def expert_classify(req: ExpertRequest, request: Request):
    """
    Run LLM inference for the given task + language.

    The base model is already loaded in GPU memory (via SingleModelPool.preload()).
    Only adapter activation is required before generation, which is fast (~ms).
    """
    pool = request.app.state.worker_pool
    experts = request.app.state.experts

    if not pool.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")

    domain, task = req.task_key.split("/", 1)
    if domain not in experts or task not in experts[domain]:
        raise HTTPException(
            status_code=400,
            detail=f"Task '{req.task_key}' not handled by this worker (model: {pool.assigned_model_key})"
        )

    start = time.perf_counter()

    prompt = f"{req.description}\n\n{req.text}"
    input_data = {"text": req.text}

    expert = experts[domain][task]
    prediction = expert.predict(input_data, prompt, req.language)

    # prediction is a 5-tuple: (cleaned_output, conf, raw_output, base_model_key, prompt_sent)
    if isinstance(prediction, tuple) and len(prediction) >= 3:
        result, confidence, raw_response = prediction[0], prediction[1], prediction[2]
    else:
        raise HTTPException(status_code=500, detail="Unexpected prediction format from expert")

    processing_time_ms = (time.perf_counter() - start) * 1000

    return ExpertResponse(
        result=str(result),
        confidence=float(confidence) if confidence else 0.0,
        processing_time_ms=processing_time_ms,
        worker_id=request.app.state.worker_id,
        model_key=pool.assigned_model_key,
        request_id=req.request_id,
        raw_response=str(raw_response) if raw_response else None,
    )
