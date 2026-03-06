"""
moe-classifier — Python SDK for the Multilingual Mixture-of-Experts
classification pipeline.

Supports three deployment modes:

**Local** (single GPU, in-process)::

    from moe_classifier import MOEClassifier

    clf = MOEClassifier()
    clf.initialize()
    result = clf.classify(
        text="Great product, highly recommend!",
        description="Rate this review from 1 to 5 stars.",
    )
    print(result.result)        # e.g. "4"
    print(result.routing_path)  # "english -> finance -> rating"

**Remote** (HTTP client to a running MoLE service)::

    clf = MOEClassifier(
        deployment="remote",
        coordinator_url="http://10.8.100.21:8000",
        credentials={"username": "alice", "password": "secret"},
    )
    clf.initialize()
    result = clf.classify(text="Great product!", description="Rate 1-5.")

**Distributed** (gating locally + HTTP dispatch to workers)::

    clf = MOEClassifier(
        deployment="distributed",
        expert_mapping="config/expert_machine_mapping.json",
    )
    clf.initialize()
    result = clf.classify(text="Great product!", description="Rate 1-5.")
"""

from .classifier import MOEClassifier
from .types import (
    BatchItem,
    BatchResult,
    ClassificationResult,
    DeploymentMode,
)

__version__ = "2.0.0"

__all__ = [
    "MOEClassifier",
    "ClassificationResult",
    "BatchResult",
    "BatchItem",
    "DeploymentMode",
]
