"""
Result types for the MOE Classifier SDK.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClassificationResult:
    """Result of a single classification request."""

    language: str
    """Detected input language (e.g. 'english', 'japanese')."""

    domain: str
    """Classified domain (e.g. 'finance')."""

    task: str
    """Routed task within the domain (e.g. 'rating', 'pii')."""

    result: str
    """Expert output (e.g. '4' for a 4-star rating, 'E' for ESCI Exact)."""

    routing_path: str
    """Human-readable routing trace, e.g. 'english -> finance -> rating'."""

    confidence: Optional[float] = None
    """Expert model confidence score (0–1), if available."""

    domain_probabilities: Optional[Dict[str, float]] = None
    """Probability distribution across all domains."""

    raw_response: Optional[str] = None
    """Raw LLM output before post-processing."""

    processing_time_ms: float = 0.0
    """End-to-end processing time in milliseconds."""

    def __repr__(self) -> str:
        conf = f"{self.confidence:.2%}" if self.confidence is not None else "N/A"
        return (
            f"ClassificationResult("
            f"result={self.result!r}, "
            f"task={self.task!r}, "
            f"language={self.language!r}, "
            f"confidence={conf}, "
            f"time={self.processing_time_ms:.1f}ms)"
        )


@dataclass
class BatchItem:
    """A single item in a batch classification result."""

    index: int
    """0-based index of this item in the original input list."""

    result: Optional[ClassificationResult] = None
    """Classification result, or None if this item failed."""

    error: Optional[str] = None
    """Error message if classification failed, otherwise None."""

    @property
    def success(self) -> bool:
        return self.result is not None


@dataclass
class BatchResult:
    """Result of a batch classification request."""

    items: List[BatchItem] = field(default_factory=list)
    """Per-item results, in the same order as the input list."""

    total_processing_time_ms: float = 0.0
    """Total wall-clock time for the entire batch."""

    successful: int = 0
    """Number of successfully classified items."""

    failed: int = 0
    """Number of items that raised an error."""

    @property
    def results(self) -> List[Optional[ClassificationResult]]:
        """Convenience accessor — list of ClassificationResult (None on failure)."""
        return [item.result for item in self.items]

    def __repr__(self) -> str:
        return (
            f"BatchResult("
            f"total={len(self.items)}, "
            f"successful={self.successful}, "
            f"failed={self.failed}, "
            f"time={self.total_processing_time_ms:.1f}ms)"
        )
