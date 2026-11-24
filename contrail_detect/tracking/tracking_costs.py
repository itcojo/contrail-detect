"""Cost components summed into the detection-to-track assignment cost."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np

from contrail_detect.tracking.tracking_gates import GateContext
from contrail_detect.tracking.tracking_utils import (
    haversine,
    perpendicular_distance_to_gc,
    projected_overlap_fraction,
)


@dataclass(frozen=True)
class CostComponent(ABC):
    """Abstract base class for individual cost components used in assignment.

    Attributes:
        weight: Scalar weight applied to this component's cost value.
    """

    weight: float = 1.0

    @abstractmethod
    def compute(self: Self, context: GateContext) -> float:
        """Compute the unnormalized cost for the given context.

        Args:
            context: The gate context containing contrail and track.

        Returns:
            A non-negative cost value (typically in [0, 1] before weighting).
        """
        pass


@dataclass(frozen=True)
class CentroidCost(CostComponent):
    """Cost component based on great-circle distance between centroids.

    Attributes:
        max_dist: Distance in meters used to normalize the cost to [0, 1].
    """

    max_dist: float = 20000.0  # meters, adjust as appropriate

    def compute(self: Self, context: GateContext) -> float:
        """Compute normalized centroid distance cost.

        Args:
            context: The gate context containing contrail and track.

        Returns:
            Cost in [0, 1] clamped at 1.0 for distances beyond max_dist.
        """
        predicted_lon, predicted_lat, _, _ = context.potential_tracked_contrail.kalman_filter.state
        dist = haversine(
            context.contrail.centroid_lon,
            context.contrail.centroid_lat,
            predicted_lon,
            predicted_lat,
            input_unit="degrees",
        )
        cost = dist / self.max_dist
        return min(cost, 1.0)


@dataclass(frozen=True)
class LengthCost(CostComponent):
    """Cost component penalizing relative length change between contrails."""

    def compute(self: Self, context: GateContext) -> float:
        """Compute the relative length change cost.

        Args:
            context: The gate context containing contrail and track.

        Returns:
            Absolute length difference divided by the larger of the two lengths,
            in [0, 1].
        """
        last_length = context.potential_tracked_contrail.last_contrail.length
        current_length = context.contrail.length
        length_diff = abs(current_length - last_length)

        # Cost between 0 and 1, less sensitive to small length differences,
        # more sensitive to large length differences
        return length_diff / max(last_length, current_length)


@dataclass(frozen=True)
class OrientationCost(CostComponent):
    """Cost component penalizing azimuth divergence between contrails.

    Attributes:
        normalizing_angle: Angle in degrees used to normalize the cost to [0, 1].
    """

    normalizing_angle: float = 30.0  # degrees, adjust as appropriate

    def compute(self: Self, context: GateContext) -> float:
        """Compute normalized orientation change cost.

        Args:
            context: The gate context containing contrail and track.

        Returns:
            Cost in [0, 1] clamped at 1.0, treating 0 and 180 degrees as equivalent.
        """
        last_azimuth = context.potential_tracked_contrail.last_contrail.azimuth
        current_azimuth = context.contrail.azimuth
        # Minimal difference, treating 0 and 180 as equivalent
        angle_diff = abs(current_azimuth - last_azimuth) % 180
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        cost = angle_diff / self.normalizing_angle
        return min(cost, 1.0)


@dataclass(frozen=True)
class ParallelOverlapCost(CostComponent):
    """Cost component measuring the fraction of contrails that do not overlap.

    Computes the symmetric average projected overlap fraction between the
    current and predicted contrail, then returns the complement.
    """

    def compute(self: Self, context: GateContext) -> float:
        """Compute the non-overlap cost.

        Args:
            context: The gate context containing contrail and track.

        Returns:
            One minus the average projected overlap fraction, in [0, 1].
        """
        contrail = context.contrail
        predicted = context.potential_tracked_contrail.predicted_contrail
        # Get the endpoints of the predicted contrail
        p1_lon, p1_lat = np.radians(predicted.endpoint1_lon), np.radians(predicted.endpoint1_lat)
        p2_lon, p2_lat = np.radians(predicted.endpoint2_lon), np.radians(predicted.endpoint2_lat)

        c1_lon, c1_lat = np.radians(contrail.endpoint1_lon), np.radians(contrail.endpoint1_lat)
        c2_lon, c2_lat = np.radians(contrail.endpoint2_lon), np.radians(contrail.endpoint2_lat)
        overlap_fraction_1 = projected_overlap_fraction(
            c1_lat, c1_lon, c2_lat, c2_lon, p1_lat, p1_lon, p2_lat, p2_lon
        )
        overlap_fraction_2 = projected_overlap_fraction(
            p1_lat, p1_lon, p2_lat, p2_lon, c1_lat, c1_lon, c2_lat, c2_lon
        )
        return 1 - (overlap_fraction_1 + overlap_fraction_2) / 2


@dataclass(frozen=True)
class PerpendicularCost(CostComponent):
    """Cost component based on symmetric average perpendicular distance.

    Attributes:
        normalizing_distance: Distance in meters used to normalize the raw cost.
        scaling: Either 'linear' or 'exponential' normalization strategy.
    """

    normalizing_distance: float = 50000.0  # meters
    scaling: Literal["linear", "exponential"] = "linear"

    def compute(self: Self, context: GateContext) -> float:
        """Compute the perpendicular distance cost.

        Args:
            context: The gate context containing contrail and track.

        Returns:
            Normalised perpendicular cost value.
        """
        contrail = context.contrail
        predicted = context.potential_tracked_contrail.predicted_contrail

        # Get the endpoints of the predicted contrail
        p1_lon, p1_lat = np.radians(predicted.endpoint1_lon), np.radians(predicted.endpoint1_lat)
        p2_lon, p2_lat = np.radians(predicted.endpoint2_lon), np.radians(predicted.endpoint2_lat)

        c1_lon, c1_lat = np.radians(contrail.endpoint1_lon), np.radians(contrail.endpoint1_lat)
        c2_lon, c2_lat = np.radians(contrail.endpoint2_lon), np.radians(contrail.endpoint2_lat)

        # Get perpendicular geodesic distance from predicted points to infinite
        # (actually circular) line, take average
        c_to_p = (
            np.sum(
                perpendicular_distance_to_gc(
                    np.array([c1_lat, c2_lat]),
                    np.array([c1_lon, c2_lon]),
                    p1_lat,
                    p1_lon,
                    p2_lat,
                    p2_lon,
                )
            )
            / 2
        )
        p_to_c = (
            np.sum(
                perpendicular_distance_to_gc(
                    np.array([p1_lat, p2_lat]),
                    np.array([p1_lon, p2_lon]),
                    c1_lat,
                    c1_lon,
                    c2_lat,
                    c2_lon,
                )
            )
            / 2
        )
        raw_cost = (c_to_p + p_to_c) / 2
        if self.scaling == "linear":
            raw_cost = raw_cost / self.normalizing_distance
        elif self.scaling == "exponential":
            raw_cost = np.exp(raw_cost / self.normalizing_distance) - 1
        else:
            raise ValueError(f"Invalid scaling option: {self.scaling}")
        return raw_cost
