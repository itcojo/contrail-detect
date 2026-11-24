"""The hand-written __repr__ methods run and name the class.

Instances are built with ``object.__new__`` and only the attributes ``__repr__`` reads,
so no credentials, downloads or satellite data are needed.
"""

import numpy as np
import pytest

from contrail_detect.flights.contrails_api import ContrailsAPIFlightData
from contrail_detect.flights.flights_base import FlightData
from contrail_detect.flights.generic import GenericFlightData
from contrail_detect.geostationary.goes import GOESDetector
from contrail_detect.geostationary.himawari import HimawariDetector
from contrail_detect.geostationary.meteosat import MeteosatDetector
from contrail_detect.matching.matching import Matcher
from contrail_detect.matching.scoring import Scorer


@pytest.mark.parametrize("cls", [ContrailsAPIFlightData, GenericFlightData])
def test_flight_data_repr(cls: type) -> None:
    flights = object.__new__(cls)
    FlightData.__init__(
        flights, req_timestamps=["2025-06-01T12:00", "2025-06-01T13:00"], req_bbox=[0, 1, 0, 1]
    )
    flights.target_addresses = None
    text = repr(flights)
    assert text.startswith(f"{cls.__name__}(")
    assert text.endswith(")")
    assert "max_time=2025-06-01T13:00" in text


@pytest.mark.parametrize("cls", [GOESDetector, HimawariDetector, MeteosatDetector])
def test_detector_repr(cls: type) -> None:
    detector = object.__new__(cls)
    detector.satellite, detector.region, detector.color_scheme = "SAT", "F", "ash"
    detector.timestamp_sequence = [np.datetime64("2025-06-01T12:00")]
    text = repr(detector)
    assert text.startswith(f"{cls.__name__}(satellite=SAT, region=F, start_time=")
    assert text.endswith(")")


@pytest.mark.parametrize("obj", [Matcher(), Scorer()])
def test_matching_repr(obj: object) -> None:
    assert repr(obj).startswith(f"{type(obj).__name__}(")
