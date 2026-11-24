"""
Contrail detection models and inference pipelines.

This module provides implementations for detecting contrails in satellite imagery
using various approaches including API-based retrieval and local inference.

Note: LocalContrailModel requires torch and is imported lazily to avoid
requiring heavy dependencies when not needed.
"""

from contrail_detect.models.contrail_model_base import ContrailModelBase
from contrail_detect.models.contrails_api import ContrailsAPIContrailModel
from contrail_detect.models.google_api import GoogleAPIContrailModel


# Lazy import for LocalContrailModel to avoid torch dependency
def __getattr__(name):
    if name == "LocalContrailModel":
        from contrail_detect.models.local_model import LocalContrailModel

        return LocalContrailModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContrailModelBase",
    "ContrailsAPIContrailModel",
    "GoogleAPIContrailModel",
    "LocalContrailModel",
]
