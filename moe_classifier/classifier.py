"""
MOEClassifier — main entry point for the moe-classifier SDK.

Supports three deployment modes:

**Local** (default) — full in-process pipeline on a single GPU::

    from moe_classifier import MOEClassifier

    clf = MOEClassifier()
    clf.initialize()
    result = clf.classify(text="Great product!", description="Rate 1–5.")

**Remote** — HTTP client to a running MoLE service::

    clf = MOEClassifier(
        deployment="remote",
        coordinator_url="http://10.8.100.21:8000",
        credentials={"username": "alice", "password": "secret"},
    )
    clf.initialize()
    result = clf.classify(text="Great product!", description="Rate 1–5.")

**Distributed** — gating locally + dispatch to remote workers::

    clf = MOEClassifier(
        deployment="distributed",
        expert_mapping="config/expert_machine_mapping.json",
    )
    clf.initialize()
    result = clf.classify(text="Great product!", description="Rate 1–5.")
"""

import time
from typing import Any, Dict, List, Optional, Union

from .types import (
    BatchItem,
    BatchResult,
    ClassificationResult,
    DeploymentMode,
)
from .pipeline_config import PipelineConfig


class MOEClassifier:
    """
    Multilingual Mixture-of-Experts text classifier.

    The underlying ML models are heavy (language detector, XLM-RoBERTa
    domain/task classifiers, and LoRA-adapted LLM experts), so
    initialization is explicit via :meth:`initialize` rather than
    happening in ``__init__``.  Call ``initialize()`` once, then
    reuse the same instance for all subsequent calls.

    Parameters
    ----------
    deployment : str or DeploymentMode
        ``"local"`` (default), ``"remote"``, or ``"distributed"``.
    coordinator_url : str, optional
        Base URL of a running MoLE service.  **Required** for ``"remote"`` mode.
    credentials : dict, optional
        ``{"username": "...", "password": "..."}`` for automatic JWT
        authentication.  Used in ``"remote"`` mode.
    token : str, optional
        Pre-existing JWT access token.  If provided, ``credentials`` is
        ignored.  Used in ``"remote"`` mode.
    expert_mapping : str, optional
        Path to ``expert_machine_mapping.json``.  Used in ``"distributed"``
        mode.  Defaults to ``config/expert_machine_mapping.json``.
    """

    def __init__(
        self,
        deployment: Union[str, DeploymentMode] = "local",
        *,
        coordinator_url: Optional[str] = None,
        credentials: Optional[Dict[str, str]] = None,
        token: Optional[str] = None,
        expert_mapping: Optional[str] = None,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> None:
        # Coerce string to enum
        try:
            self._deployment = DeploymentMode(deployment)
        except ValueError:
            valid = ", ".join(f'"{m.value}"' for m in DeploymentMode)
            raise ValueError(
                f"Invalid deployment mode: {deployment!r}.  "
                f"Valid modes: {valid}"
            )

        self._coordinator_url = coordinator_url
        self._credentials = credentials
        self._token = token
        self._expert_mapping = expert_mapping
        self._pipeline_config = pipeline_config
        self._backend = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Load models / connect to services based on the deployment mode.

        This must be called before :meth:`classify` or
        :meth:`classify_batch`.  Depending on the mode it may take
        a moment (local: model loading) or be near-instant (remote).

        Raises:
            RuntimeError: If initialization fails.
        """
        self._backend = self._create_backend()
        self._backend.initialize()
        self._initialized = True

    def _create_backend(self):
        """Instantiate the correct backend for the chosen deployment mode."""
        mode = self._deployment

        if mode == DeploymentMode.LOCAL:
            from .backends.local import LocalBackend
            return LocalBackend(pipeline_config=self._pipeline_config)

        elif mode == DeploymentMode.REMOTE:
            if not self._coordinator_url:
                raise ValueError(
                    "deployment='remote' requires coordinator_url=... "
                    "(e.g. 'http://localhost:8000')"
                )
            from .backends.remote import RemoteBackend
            return RemoteBackend(
                coordinator_url=self._coordinator_url,
                credentials=self._credentials,
                token=self._token,
            )

        elif mode == DeploymentMode.DISTRIBUTED:
            from .backends.distributed import DistributedBackend
            return DistributedBackend(
                expert_mapping=self._expert_mapping,
                pipeline_config=self._pipeline_config,
            )

        else:
            raise ValueError(f"Unknown deployment mode: {mode}")

    @property
    def is_ready(self) -> bool:
        """True after :meth:`initialize` has completed successfully."""
        return self._initialized and self._backend is not None and self._backend.is_ready

    @property
    def deployment_mode(self) -> DeploymentMode:
        """The deployment mode this classifier was configured with."""
        return self._deployment

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

        return self._backend.classify(
            text=text,
            description=description,
            return_domain_probabilities=return_domain_probabilities,
            return_raw_response=return_raw_response,
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
        return self._backend.get_stats()

    def __repr__(self) -> str:
        state = "ready" if self.is_ready else "not initialized"
        return f"MOEClassifier(deployment={self._deployment.value!r}, {state})"
