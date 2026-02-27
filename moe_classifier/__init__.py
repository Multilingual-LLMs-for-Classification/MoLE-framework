"""
moe-classifier — Python SDK for the Multilingual Mixture-of-Experts
classification pipeline.

Quick start::

    from moe_classifier import MOEClassifier

    clf = MOEClassifier()
    clf.initialize()          # load models (once)

    result = clf.classify(
        text="Great product, highly recommend!",
        description="Rate this review from 1 to 5 stars.",
    )
    print(result.result)      # e.g. "4"
    print(result.routing_path)  # "english -> finance -> rating"
"""

from .classifier import MOEClassifier
from .types import BatchItem, BatchResult, ClassificationResult

__version__ = "1.0.0"

__all__ = [
    "MOEClassifier",
    "ClassificationResult",
    "BatchResult",
    "BatchItem",
]
