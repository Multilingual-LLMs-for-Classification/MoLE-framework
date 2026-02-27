"""
Routing service that wraps the MOE-Router PromptRoutingSystem.

Supports two operating modes controlled by SERVICE_MODE env var:

  coordinator  (default)
      Runs the lightweight gating pipeline (language detection, domain
      classification, Q-learning task routing, expert selection) then
      forwards inference to the appropriate expert worker via GatewayService.
      No LLM is loaded in this process.

  monolithic
      Original single-GPU mode.  Runs the full pipeline including LLM
      inference in-process (backward-compatible with the original service).
"""

import time
import asyncio
from asyncio import Semaphore
from typing import Dict, Any, Optional
from uuid import uuid4

from app.config import settings
from app.schemas.requests import ClassifyRequest
from app.schemas.responses import ClassifyResponse


class RoutingService:
    """
    Service wrapper around PromptRoutingSystem.

    Handles:
    - Lazy initialization of the routing system
    - Mode-aware classification (coordinator vs monolithic)
    - Request/response transformation
    - Timeout handling
    """

    def __init__(self):
        self._routing_system = None
        self._gateway = None
        self._initialized = False
        # Semaphore is only used in monolithic mode (GPU concurrency control)
        self._gpu_semaphore = Semaphore(settings.max_concurrent_gpu_requests)

    def initialize(self) -> bool:
        """
        Initialize the routing system.

        In coordinator mode: loads only the gating models + gateway config.
        In monolithic mode: loads gating models + full LLM expert pool.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True

        try:
            from moe_router.gating.components.routing_system import PromptRoutingSystem

            coordinator_only = (settings.service_mode == "coordinator")
            mode_label = "coordinator" if coordinator_only else "monolithic"
            print(f"Initializing PromptRoutingSystem (mode={mode_label}) ...")

            self._routing_system = PromptRoutingSystem(
                training_mode=False,
                coordinator_only=coordinator_only
            )

            if coordinator_only:
                from app.services.gateway_service import GatewayService
                self._gateway = GatewayService(settings.expert_mapping_path)
                print("Gateway service initialized.")

            self._initialized = True
            print(f"PromptRoutingSystem initialized successfully (mode={mode_label})")
            return True

        except Exception as e:
            print(f"Failed to initialize routing system: {e}")
            import traceback
            traceback.print_exc()
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized and self._routing_system is not None

    def get_system_stats(self) -> Dict[str, Any]:
        if not self.is_initialized:
            return {"error": "Routing system not initialized"}
        return self._routing_system.get_system_stats()

    async def classify(
        self,
        request: ClassifyRequest,
        timeout_seconds: Optional[int] = None
    ) -> ClassifyResponse:
        """
        Classify text using the routing system.

        In coordinator mode:
          1. Run lightweight gating in a thread pool (no GPU for gating models)
          2. Dispatch inference to expert worker via GatewayService (async HTTP)

        In monolithic mode:
          Full pipeline (gating + LLM inference) runs in a thread pool behind
          a GPU semaphore (original behaviour, unchanged).

        Raises:
            RuntimeError: If routing system not initialized.
            TimeoutError: If classification times out.
        """
        if not self.is_initialized:
            raise RuntimeError("Routing system not initialized")

        timeout = timeout_seconds or settings.request_timeout_seconds
        request_id = str(uuid4())

        if settings.service_mode == "coordinator":
            return await self._classify_coordinator(request, request_id, timeout)
        else:
            return await self._classify_monolithic(request, request_id, timeout)

    # ------------------------------------------------------------------
    # Coordinator mode
    # ------------------------------------------------------------------
    async def _classify_coordinator(
        self,
        request: ClassifyRequest,
        request_id: str,
        timeout: int
    ) -> ClassifyResponse:
        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()

        # Phase 1: lightweight gating (CPU/small-GPU; runs concurrently)
        prompt = f"{request.description}\n\n{request.text}"
        gating = await asyncio.wait_for(
            loop.run_in_executor(None, self._routing_system.run_gating, prompt),
            timeout=timeout
        )

        # Phase 2: async HTTP dispatch to expert worker
        payload = {
            "task_key": f"{gating.domain}/{gating.task}",
            "language": gating.language,
            "text": request.text,
            "description": request.description,
            "adapter_name": gating.adapter_name,
            "request_id": request_id,
        }
        expert_result = await asyncio.wait_for(
            self._gateway.dispatch(gating.base_model_key, payload),
            timeout=timeout
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        routing_path = f"{gating.routing_path} -> gateway:{gating.base_model_key}"

        response = ClassifyResponse(
            request_id=request_id,
            language=gating.language,
            domain=gating.domain,
            task=gating.task,
            result=str(expert_result.get("result", "")),
            confidence=expert_result.get("confidence"),
            routing_path=routing_path,
            processing_time_ms=processing_time_ms,
        )

        if request.options.return_raw_response:
            response.raw_response = expert_result.get("raw_response")

        return response

    # ------------------------------------------------------------------
    # Monolithic mode (original behaviour, unchanged)
    # ------------------------------------------------------------------
    async def _classify_monolithic(
        self,
        request: ClassifyRequest,
        request_id: str,
        timeout: int
    ) -> ClassifyResponse:
        async with self._gpu_semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        self._sync_classify,
                        request,
                        request_id
                    ),
                    timeout=timeout
                )
                return result
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Classification timed out after {timeout} seconds"
                )

    def _sync_classify(
        self,
        request: ClassifyRequest,
        request_id: str
    ) -> ClassifyResponse:
        """Synchronous classification for monolithic mode (runs in thread pool)."""
        start_time = time.perf_counter()

        prompt = f"{request.description}\n\n{request.text}"
        input_data = {"text": request.text}

        result = self._routing_system.route_prompt(
            prompt=prompt,
            input_data=input_data
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        response = ClassifyResponse(
            request_id=request_id,
            language=result.get("language", "unknown"),
            domain=result.get("domain", "unknown"),
            task=result.get("task", "unknown"),
            result=str(result.get("result", "")),
            confidence=result.get("expert_confidence"),
            routing_path=result.get("routing_path", ""),
            processing_time_ms=processing_time_ms
        )

        if request.options.return_probabilities:
            response.domain_probabilities = result.get("domain_probabilities")

        if request.options.return_raw_response:
            response.raw_response = result.get("raw_response")

        return response


# Global service instance
routing_service = RoutingService()
