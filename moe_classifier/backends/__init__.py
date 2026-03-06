"""
Backend implementations for different deployment modes.

- LocalBackend:       Full in-process pipeline (single GPU, monolithic)
- RemoteBackend:      HTTP client to a running MoLE coordinator/service
- DistributedBackend: Gating locally + HTTP dispatch to expert workers
"""

from .base import ClassifierBackend
from .local import LocalBackend
from .remote import RemoteBackend
from .distributed import DistributedBackend

__all__ = [
    "ClassifierBackend",
    "LocalBackend",
    "RemoteBackend",
    "DistributedBackend",
]
