"""
MOEClassifier — main entry point for the moe-classifier SDK.

Usage::

    from moe_classifier import MOEClassifier

    clf = MOEClassifier()
    clf.initialize()                          # load models once

    result = clf.classify(
        text="Great product, loved it!",
        description="Rate this review 1–5 stars.",
    )
    print(result.result, result.confidence)
"""

import time
from typing import Any, Dict, List, Optional

from .types import BatchItem, BatchResult, ClassificationResult


class MOEClassifier:
    """
    Thin wrapper around PromptRoutingSystem providing a clean SDK API.

    The underlying ML models are heavy (language detector, XLM-RoBERTa
    domain/task classifiers, and LoRA-adapted LLM experts), so
    initialization is explicit via :meth:`initialize` rather than
    happening in ``__init__``.  Call ``initialize()`` once, then
    reuse the same instance for all subsequent calls.
    """

    def __init__(self) -> None:
        self._system = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Load all models and prepare the routing system.

        This must be called before :meth:`classify` or
        :meth:`classify_batch`.  It may take tens of seconds on first
        run (model downloads + GPU loading).

        Raises:
            RuntimeError: If the underlying routing system fails to load.
        """
        try:
            from moe_router.gating.components.routing_system import PromptRoutingSystem
        except ImportError as exc:
            raise RuntimeError(
                "moe_router package not found.  Make sure you are running "
                "from the moe-classification-service directory and the "
                "package is installed (pip install -e .)."
            ) from exc

        try:
            self._system = PromptRoutingSystem(training_mode=False)
            self._initialized = True
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize routing system: {exc}") from exc

    @property
    def is_ready(self) -> bool:
        """True after :meth:`initialize` has completed successfully."""
        return self._initialized and self._system is not None

    def _require_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError(
                "MOEClassifier is not initialized.  Call classifier.initialize() first."
            )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        description: str = "",
        *,
        return_domain_probabilities: bool = False,
        return_raw_response: bool = False,
    ) -> ClassificationResult:
        """
        Classify a single piece of text.

        Args:
            text:
                The text to classify (e.g. a product review, news article,
                or document containing PII).
            description:
                Optional free-text description of the task.  This is
                prepended to *text* to form the routing prompt and helps
                the domain/task router select the right expert, especially
                when the text alone is ambiguous.
                Example: ``"Rate this product review from 1 to 5 stars."``
            return_domain_probabilities:
                If True, populate :attr:`ClassificationResult.domain_probabilities`
                with the full domain probability distribution.
            return_raw_response:
                If True, populate :attr:`ClassificationResult.raw_response`
                with the unprocessed LLM output before post-processing.

        Returns:
            :class:`ClassificationResult` with the classification output.

        Raises:
            RuntimeError: If the classifier has not been initialized.
            ValueError: If *text* is empty.
        """
        self._require_ready()

        if not text or not text.strip():
            raise ValueError("'text' must be a non-empty string.")

        # Combine description and text into the routing prompt (mirrors the
        # FastAPI service's _sync_classify logic).
        prompt = f"{description}\n\n{text}".strip() if description else text

        t0 = time.perf_counter()
        raw = self._system.route_prompt(
            prompt=prompt,
            input_data={"text": text},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ClassificationResult(
            language=raw.get("language", "unknown"),
            domain=raw.get("domain", "unknown"),
            task=raw.get("task", "unknown"),
            result=str(raw.get("result", "")),
            routing_path=raw.get("routing_path", ""),
            confidence=raw.get("expert_confidence"),
            domain_probabilities=(
                raw.get("domain_probabilities") if return_domain_probabilities else None
            ),
            raw_response=(
                raw.get("raw_response") if return_raw_response else None
            ),
            processing_time_ms=elapsed_ms,
        )

    def classify_batch(
        self,
        items: List[Dict[str, str]],
        *,
        return_domain_probabilities: bool = False,
        return_raw_response: bool = False,
        skip_errors: bool = True,
    ) -> BatchResult:
        """
        Classify a list of texts.

        Args:
            items:
                List of dicts, each with:

                * ``"text"`` *(required)* — the text to classify.
                * ``"description"`` *(optional)* — task description hint.

                Example::

                    [
                        {"text": "Great product!", "description": "Rate 1-5."},
                        {"text": "Terrible quality."},
                    ]

            return_domain_probabilities:
                Forwarded to each :meth:`classify` call.
            return_raw_response:
                Forwarded to each :meth:`classify` call.
            skip_errors:
                If True (default), failed items are recorded as
                :attr:`BatchItem.error` and processing continues.
                If False, the first error is re-raised immediately.

        Returns:
            :class:`BatchResult` with per-item results and summary stats.

        Raises:
            RuntimeError: If the classifier has not been initialized.
            ValueError: If *items* is empty.
        """
        self._require_ready()

        if not items:
            raise ValueError("'items' must be a non-empty list.")

        batch_items: List[BatchItem] = []
        successful = 0
        failed = 0
        t0 = time.perf_counter()

        for idx, item in enumerate(items):
            text = item.get("text", "")
            description = item.get("description", "")
            try:
                result = self.classify(
                    text=text,
                    description=description,
                    return_domain_probabilities=return_domain_probabilities,
                    return_raw_response=return_raw_response,
                )
                batch_items.append(BatchItem(index=idx, result=result))
                successful += 1
            except Exception as exc:
                if not skip_errors:
                    raise
                batch_items.append(BatchItem(index=idx, error=str(exc)))
                failed += 1

        total_ms = (time.perf_counter() - t0) * 1000

        return BatchResult(
            items=batch_items,
            total_processing_time_ms=total_ms,
            successful=successful,
            failed=failed,
        )

    # ------------------------------------------------------------------
    # System information
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return system capability information.

        Returns a dict with keys:

        * ``total_domains`` — number of supported domains
        * ``total_tasks`` — total tasks across all domains
        * ``supported_languages`` — number of detectable languages
        * ``all_languages`` — sorted list of language names
        * ``languages_by_task`` — per-task supported languages
        * ``domains`` — list of domain names

        Raises:
            RuntimeError: If the classifier has not been initialized.
        """
        self._require_ready()
        return self._system.get_system_stats()

    def __repr__(self) -> str:
        state = "ready" if self.is_ready else "not initialized"
        return f"MOEClassifier({state})"
