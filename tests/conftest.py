"""Shared test helpers."""

import pytest


@pytest.fixture
def bare():
    """Build an instance without running ``__init__``, setting only the given attributes.

    Lets tests reach a method without credentials, downloads or satellite data.
    """

    def _bare(cls: type, **attrs):
        obj = object.__new__(cls)
        for name, value in attrs.items():
            setattr(obj, name, value)
        return obj

    return _bare


# ── Geostationary detector stubs ──────────────────────────────────────────────


@pytest.fixture
def geo_detector():
    """Build a detector stub using the real get_zarr_path, get_npy_path and zarr reader.

    Call it with the output folder, the timestamps, the detection subdirectory and the
    fixed-grid coordinates in metres.
    """
    from types import MethodType, SimpleNamespace

    import numpy as np
    import pandas as pd

    from contrail_detect.geostationary.abc import Geostationary

    def _geo_detector(detect_path, timestamps, subdirectory, geo_x, geo_y, satellite="GOES-19"):
        det = SimpleNamespace()
        for name in [
            "get_zarr_path",
            "load_detections_from_zarr",
            "_reproject_cropped_to_full_grid",
            "get_npy_path",
            "scan_start_offsets_s",
        ]:
            setattr(det, name, MethodType(getattr(Geostationary, name), det))
        det.image_geo_coords = (geo_x, geo_y)
        det.lats = np.zeros((len(geo_y), len(geo_x)))
        det.satellite, det.region = satellite, "F"
        det.detection_subdirectory, det.LONG_NAME, det.DETECT_PATH = subdirectory, "T", detect_path
        det.timestamp_sequence = [np.datetime64(t) for t in timestamps]
        det.timestamp_strings = [pd.Timestamp(t).strftime("%Y-%m-%dT%H%M") for t in timestamps]
        det.tile_extent = (0.0, 1.0, 0.0, 1.0)
        det.min_longitude, det.max_longitude = -1, 2
        det.min_latitude, det.max_latitude = -1, 2
        det.storage_threshold = 0.01
        det.save_static_geometry = lambda zarr_path: None
        det.has_shifting_domain = False
        return det

    return _geo_detector


@pytest.fixture
def stub_abi():
    """ABIDetector subclass with the abstract parts filled in and no downloads."""
    from contrail_detect.geostationary.abi_base import ABIDetector

    class _StubABI(ABIDetector):
        def _frame_extent(self, da):
            pass

        def _frame_projection(self, da):
            pass

        def _gen_scan_time_offset(self):
            pass

        def _gen_tiles(self):
            pass

        def _get_frame(self, time):
            raise AssertionError("no downloads in tests")

        def _set_central_lon(self):
            pass

        @property
        def image_full_extent(self):
            return None

        @property
        def image_geo_coords(self):
            return None

        @property
        def sat_pyproj_crs(self):
            return None

    return _StubABI
