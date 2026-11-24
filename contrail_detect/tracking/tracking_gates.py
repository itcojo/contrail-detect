"""Gates that rule out implausible detection-to-track assignments before costing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

if TYPE_CHECKING:
    from contrail_detect.tracking.tracking import (
        Contrail,
        TrackedContrail,
        TrackingDebugInfo,
    )

from contrail_detect.tracking.tracking_utils import (
    azimuth,
    haversine,
    perpendicular_distance_to_gc,
)


@dataclass(frozen=True)
class GateContext:
    """Immutable context bundle passed to gates and cost components.

    Attributes:
        contrail: The candidate Contrail from the current frame.
        potential_tracked_contrail: The existing track being evaluated.
    """

    contrail: Contrail
    potential_tracked_contrail: TrackedContrail


class TrackingGate(ABC):
    """Abstract base class for tracking gates.

    A gate is a binary filter that accepts or rejects a (contrail, track) pair
    before cost evaluation. Cheaper gates should be applied first.
    """

    @abstractmethod
    def allow(
        self: Self,
        context: GateContext,
        debug_info: TrackingDebugInfo | None = None,
    ) -> bool:
        """Return True if this gate accepts the candidate pair.

        Args:
            context: The gate context containing contrail and track.
            debug_info: Optional debug container for recording gate decisions.

        Returns:
            True if the pair passes the gate, False otherwise.
        """
        pass


@dataclass(frozen=True)
class CentroidTravelDirectionGate(TrackingGate):
    """Gate that rejects pairs whose travel direction change exceeds a threshold.

    Attributes:
        max_direction_change: Maximum allowed direction change in degrees.
    """

    max_direction_change: float = 30.0  # degrees

    def allow(
        self: Self,
        context: GateContext,
        debug_info: TrackingDebugInfo | None = None,
    ) -> bool:
        """Return True if the travel direction change is within max_direction_change.

        Args:
            context: The gate context containing contrail and track.
            debug_info: Optional debug container for recording gate decisions.

        Returns:
            True if the direction change passes the limit, or if no prior
            direction of travel is available.
        """
        new_direction = azimuth(
            context.potential_tracked_contrail.last_contrail.centroid_lat,
            context.potential_tracked_contrail.last_contrail.centroid_lon,
            context.contrail.centroid_lat,
            context.contrail.centroid_lon,
        )
        old_direction = context.potential_tracked_contrail.direction_of_travel
        if old_direction is None:
            return True  # if we don't have a direction of travel yet, allow the match
        angle_diff = abs((new_direction - old_direction + 180) % 360 - 180)
        return angle_diff <= self.max_direction_change


@dataclass(frozen=True)
class HemisphereShiftGate(TrackingGate):
    """Gate that enforces consistency of lateral hemisphere shifts across a track."""

    def allow(
        self: Self,
        context: GateContext,
        debug_info: TrackingDebugInfo | None = None,
    ) -> bool:
        """Return True if the hemisphere shift is consistent with the track history.

        Calculates the hemisphere shift for the new contrail relative to the last
        contrail in the track, then rejects the pair if the shift contradicts the
        most recent non-zero shift already recorded.

        Args:
            context: The gate context containing contrail and track.
            debug_info: Optional debug container for recording gate decisions.

        Returns:
            True if the shift is consistent with track history or is zero.
        """
        new_contrail = context.contrail
        last_contrail = context.potential_tracked_contrail.last_contrail
        shift = context.potential_tracked_contrail._hemisphere_shift(last_contrail, new_contrail)
        last_nonzero_shift = context.potential_tracked_contrail.last_nonzero_shift
        allowed = True
        reason = ""
        if shift == 0 or last_nonzero_shift == 0:
            allowed = True
            reason = "zero shift"
        elif shift != last_nonzero_shift:
            allowed = False
            reason = "shift mismatch"
            logging.debug(
                f"Hemisphere shift gate: new shift={shift}, "
                f"last_nonzero_shift={last_nonzero_shift}, rejecting match"
            )
        if debug_info is not None:
            debug_info.gates.append(
                {
                    "gate": "HemisphereShiftGate",
                    "allowed": allowed,
                    "shift": shift,
                    "last_nonzero_shift": last_nonzero_shift,
                    "last_endpoint1_lon": last_contrail.endpoint1_lon,
                    "last_endpoint1_lat": last_contrail.endpoint1_lat,
                    "last_endpoint2_lon": last_contrail.endpoint2_lon,
                    "last_endpoint2_lat": last_contrail.endpoint2_lat,
                    "new_endpoint1_lon": new_contrail.endpoint1_lon,
                    "new_endpoint1_lat": new_contrail.endpoint1_lat,
                    "new_endpoint2_lon": new_contrail.endpoint2_lon,
                    "new_endpoint2_lat": new_contrail.endpoint2_lat,
                    "track_id": getattr(context.potential_tracked_contrail, "track_id", None),
                    "contrail_id": getattr(context.contrail, "contrail_id", None),
                    "reason": reason,
                }
            )
        return allowed


@dataclass(frozen=True)
class OrientationChangeGate(TrackingGate):
    """Gate that rejects pairs with excessive change in contrail azimuth.

    Attributes:
        max_orientation_change: Maximum allowed azimuth change in degrees,
            accounting for 180-degree orientation ambiguity.
    """

    max_orientation_change: float = 30.0  # degrees

    def allow(
        self: Self,
        context: GateContext,
        debug_info: TrackingDebugInfo | None = None,
    ) -> bool:
        """Return True if the azimuth change is within max_orientation_change.

        Args:
            context: The gate context containing contrail and track.
            debug_info: Optional debug container for recording gate decisions.

        Returns:
            True if the orientation change passes the limit.
        """
        last_azimuth = context.potential_tracked_contrail.last_contrail.azimuth
        current_azimuth = context.contrail.azimuth
        angle_diff = abs(current_azimuth - last_azimuth) % 180
        if angle_diff > 90:
            angle_diff = (
                180 - angle_diff
            )  # account for 180 degree ambiguity in contrail orientation

        allow = angle_diff <= self.max_orientation_change

        if debug_info is not None:
            debug_info.gates.append(
                {
                    "gate": "OrientationChangeGate",
                    "allowed": allow,
                    "last_azimuth": last_azimuth,
                    "current_azimuth": current_azimuth,
                    "angle_diff": angle_diff,
                    "max_orientation_change": self.max_orientation_change,
                    "track_id": getattr(context.potential_tracked_contrail, "track_id", None),
                    "contrail_id": getattr(context.contrail, "contrail_id", None),
                }
            )
        return allow


@dataclass(frozen=True)
class PerpendicularDistanceGate(TrackingGate):
    """Gate that rejects pairs whose average perpendicular distance exceeds a threshold.

    Attributes:
        max_perpendicular_distance: Maximum allowed symmetric average perpendicular
            distance in meters.
    """

    max_perpendicular_distance: float = 50000.0  # meters

    def allow(
        self: Self,
        context: GateContext,
        debug_info: TrackingDebugInfo | None = None,
    ) -> bool:
        """Return True if the average perpendicular distance is within the limit.

        Computes the symmetric average of perpendicular distances from each
        contrail's endpoints to the great-circle line of the other.

        Args:
            context: The gate context containing contrail and track.
            debug_info: Optional debug container for recording gate decisions.

        Returns:
            True if the average perpendicular distance passes the limit.
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
        c_to_p = np.round(
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
        p_to_c = np.round(
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
        avg_perpendicular_distance = (c_to_p + p_to_c) / 2
        allow = avg_perpendicular_distance <= self.max_perpendicular_distance

        if debug_info is not None:
            debug_info.gates.append(
                {
                    "gate": "PerpendicularDistanceGate",
                    "allowed": allow,
                    "c_to_p": c_to_p,
                    "p_to_c": p_to_c,
                    "avg_perpendicular_distance": avg_perpendicular_distance,
                    "max_perpendicular_distance": self.max_perpendicular_distance,
                    "track_id": getattr(context.potential_tracked_contrail, "track_id", None),
                    "contrail_id": getattr(context.contrail, "contrail_id", None),
                }
            )
            logging.debug(
                f"PerpendicularDistanceGate: c_to_p={c_to_p:.2f}m, p_to_c={p_to_c:.2f}m, "
                f"avg={avg_perpendicular_distance:.2f}m, "
                f"contrail_id={getattr(context.contrail, 'contrail_id', None)}, "
                f"track_id={getattr(context.potential_tracked_contrail, 'track_id', None)}, "
                f"allowed={allow}"
            )
        return allow


@dataclass(frozen=True)
class ProportionalPositionChangeGate(TrackingGate):
    """Gate that rejects pairs with disproportionate centroid displacement.

    Attributes:
        max_position_change: Maximum fractional change in centroid displacement
            compared to the previous frame's displacement.
    """

    max_position_change: float = (
        0.5  # fractional change in position compared to last position change
    )

    def allow(
        self: Self,
        context: GateContext,
        debug_info: TrackingDebugInfo | None = None,
    ) -> bool:
        """Return True if the proportional position change is within the limit.

        Args:
            context: The gate context containing contrail and track.
            debug_info: Optional debug container for recording gate decisions.

        Returns:
            True if the fractional displacement change passes the limit, or if
            no prior position change is available.
        """
        last_position_change = context.potential_tracked_contrail.last_position_change
        if last_position_change is None or last_position_change < 1e-6:
            return True  # if we don't have a position change yet, allow the match
        new_position_change = haversine(
            context.potential_tracked_contrail.last_contrail.centroid_lon,
            context.potential_tracked_contrail.last_contrail.centroid_lat,
            context.contrail.centroid_lon,
            context.contrail.centroid_lat,
            input_unit="degrees",
        )

        # Allow increase or decrease in position change, but not more than
        # max_position_change in either direction
        fractional_change = abs(new_position_change - last_position_change) / last_position_change
        allow = fractional_change <= self.max_position_change
        if not allow:
            logging.debug(
                f"ProportionalPositionChangeGate: last={last_position_change:.2f}m, "
                f"new={new_position_change:.2f}m, "
                f"fractional change={fractional_change:.2%} "
                f"exceeds max {self.max_position_change:.2%}, rejecting match"
            )
        return allow


@dataclass(frozen=True)
class VicinityGate(TrackingGate):
    """Gate that rejects pairs whose centroids are too far apart in degrees.

    This gate is computationally cheap and should be applied first to filter
    out obviously bad matches.

    Attributes:
        max_match_deg: Maximum allowed separation in degrees (longitude and
            latitude independently).
    """

    max_match_deg: float = 2.5  # degrees

    def allow(
        self: Self, context: GateContext, debug_info: TrackingDebugInfo | None = None
    ) -> bool:
        """Return True if centroid separation is within max_match_deg.

        Args:
            context: The gate context containing contrail and track.
            debug_info: Optional debug container (unused by this gate).

        Returns:
            True if both longitude and latitude separations are within the limit.
        """
        predicted_lon, predicted_lat, _, _ = context.potential_tracked_contrail.kalman_filter.state
        dlon = abs(context.contrail.centroid_lon - predicted_lon)
        if dlon > 180.0:  # either side of the date line
            dlon = 360.0 - dlon
        dlat = abs(context.contrail.centroid_lat - predicted_lat)
        allow = dlon <= self.max_match_deg and dlat <= self.max_match_deg
        return allow
