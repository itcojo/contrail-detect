"""
Geostationary satellite data processing and contrail detection.

This module provides implementations for different geostationary satellite
platforms including GOES, Himawari, and Meteosat, with support for contrail
detection and tracking from ABI and FCI instruments.
"""

from contrail_detect.geostationary.abc import Geostationary
from contrail_detect.geostationary.abi_base import ABIDetector
from contrail_detect.geostationary.fci_base import FCIDetector
from contrail_detect.geostationary.goes import GOESDetector
from contrail_detect.geostationary.himawari import HimawariDetector
from contrail_detect.geostationary.meteosat import MeteosatDetector

__all__ = [
    "Geostationary",
    "ABIDetector",
    "FCIDetector",
    "GOESDetector",
    "HimawariDetector",
    "MeteosatDetector",
]
