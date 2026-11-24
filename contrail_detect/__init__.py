"""Contrail detection, tracking, and matching to flights from geostationary satellite imagery."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("contrail-detect")
except PackageNotFoundError:
    # Running from a source tree that has not been installed
    __version__ = "unknown"
