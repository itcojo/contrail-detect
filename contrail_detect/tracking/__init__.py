"""
Contrail tracking and temporal evolution analysis.

This module provides algorithms for tracking contrails across multiple satellite
images over time, including region refinement and utilities for analyzing
contrail persistence and evolution.
"""

from contrail_detect.tracking.tracking import TrackingManager
from contrail_detect.tracking.tracking_viz import generate_tracking_GIF, plot_tracks

__all__ = ["TrackingManager", "generate_tracking_GIF", "plot_tracks"]
