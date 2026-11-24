"""
Core utilities and configuration.

This module provides common functionality used across the contrail-detect package,
including color schemes, coordinate transformations, configuration management,
and shared utility functions.
"""

from contrail_detect.core.common import (
    Colors,
    Transforms,
    extract_date_time_components,
)
from contrail_detect.core.config import (
    PACKAGE_DIR,
    TEST_DIR,
)

__all__ = [
    "Colors",
    "Transforms",
    "extract_date_time_components",
    "PACKAGE_DIR",
    "TEST_DIR",
]
