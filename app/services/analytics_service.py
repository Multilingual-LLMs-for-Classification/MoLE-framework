"""
In-memory analytics collector for classification requests.
"""

from collections import Counter, deque
from datetime import datetime
from threading import Lock
from typing import Dict, Any, Optional


class AnalyticsService:
    """Thread-safe in-memory analytics collector."""

    def __init__(self, max_recent: int = 50):
        self._lock = Lock()
        self._total_requests = 0
        self._total_errors = 0

        self._language_counter: Counter = Counter()
        self._domain_counter: Counter = Counter()
        self._task_counter: Counter = Counter()

        self._task_confidence_sum: Dict[str, float] = {}
        self._task_confidence_count: Dict[str, int] = {}

        self._processing_time_sum = 0.0
        self._processing_time_count = 0

        self._recent_requests: deque = deque(maxlen=max_recent)

    def record_classification(self, response_data: Dict[str, Any]) -> None:
        """Record a successful classification result."""
        with self._lock:
            self._total_requests += 1

            lang = response_data.get("language", "unknown")
            domain = response_data.get("domain", "unknown")
            task = response_data.get("task", "unknown")

            self._language_counter[lang] += 1
            self._domain_counter[domain] += 1
            self._task_counter[task] += 1

            confidence = response_data.get("confidence")
            if confidence is not None:
                self._task_confidence_sum[task] = self._task_confidence_sum.get(task, 0.0) + confidence
                self._task_confidence_count[task] = self._task_confidence_count.get(task, 0) + 1

            proc_time = response_data.get("processing_time_ms", 0.0)
            self._processing_time_sum += proc_time
            self._processing_time_count += 1

            self._recent_requests.append({
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": response_data.get("request_id", ""),
                "language": lang,
                "domain": domain,
                "task": task,
                "confidence": confidence,
                "processing_time_ms": proc_time,
            })

    def record_error(self) -> None:
        """Record a failed classification."""
        with self._lock:
            self._total_errors += 1

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregated analytics summary."""
        with self._lock:
            avg_processing_time: Optional[float] = None
            if self._processing_time_count > 0:
                avg_processing_time = self._processing_time_sum / self._processing_time_count

            task_avg_confidence: Dict[str, float] = {}
            for task, total in self._task_confidence_sum.items():
                count = self._task_confidence_count.get(task, 1)
                task_avg_confidence[task] = total / count

            avg_confidence: Optional[float] = None
            if task_avg_confidence:
                avg_confidence = sum(task_avg_confidence.values()) / len(task_avg_confidence)

            total = self._total_requests + self._total_errors
            error_rate = (self._total_errors / total) if total > 0 else 0.0

            return {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "error_rate": error_rate,
                "avg_confidence": avg_confidence,
                "avg_processing_time_ms": avg_processing_time,
                "language_distribution": dict(self._language_counter),
                "domain_distribution": dict(self._domain_counter),
                "task_distribution": dict(self._task_counter),
                "task_avg_confidence": task_avg_confidence,
                "recent_requests": list(self._recent_requests),
            }


analytics_service = AnalyticsService()
