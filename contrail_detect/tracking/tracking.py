"""
Multi-object contrail tracker using Kalman filtering and Hungarian assignment.

Provides the config dataclasses, tracker implementations (forward, backward,
bidirectional) and the top-level TrackingManager interface. Gates live in
``tracking_gates`` and cost components in ``tracking_costs``.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pycontrails.physics import units
from scipy.optimize import linear_sum_assignment

from contrail_detect.core.common import Transforms
from contrail_detect.tracking.tracking_costs import (
    CostComponent,
    OrientationCost,
    PerpendicularCost,
)
from contrail_detect.tracking.tracking_gates import (
    GateContext,
    OrientationChangeGate,
    PerpendicularDistanceGate,
    TrackingGate,
    VicinityGate,
)
from contrail_detect.tracking.tracking_utils import (
    azimuth,
    dataclass_to_dict,
    haversine,
    hemisphere,
)

transforms = Transforms()


def _wrap_lon(lon: float) -> float:
    """Longitude, or longitude difference, wrapped into [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


@dataclass
class TrackingDebugInfo:
    """Mutable container for per-frame debug information collected during tracking."""

    costs: list[dict] = field(default_factory=list)
    gates: list[dict] = field(default_factory=list)
    wind_estimates: list[dict] = field(default_factory=list)
    kalman_filter_states: list[dict] = field(default_factory=list)
    predicted_contrails: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Contrail:
    """Immutable representation of a detected contrail line segment.

    Attributes:
        contrail_id: Unique identifier string, or None for synthetic contrails.
        endpoint1_lon: Longitude of the first endpoint in degrees.
        endpoint1_lat: Latitude of the first endpoint in degrees.
        endpoint2_lon: Longitude of the second endpoint in degrees.
        endpoint2_lat: Latitude of the second endpoint in degrees.
        u_init: Optional initial zonal wind estimate in m/s.
        v_init: Optional initial meridional wind estimate in m/s.
        debug_info: Optional mutable debug container attached to this contrail.
    """

    contrail_id: str | None
    endpoint1_lon: float
    endpoint1_lat: float
    endpoint2_lon: float
    endpoint2_lat: float
    u_init: float | None = None  # m/s
    v_init: float | None = None  # m/s
    debug_info: TrackingDebugInfo | None = None

    @property
    def azimuth(self: Self) -> float:
        """Forward azimuth of the contrail in degrees."""
        return azimuth(
            self.endpoint1_lat,
            self.endpoint1_lon,
            self.endpoint2_lat,
            self.endpoint2_lon,
        )

    @property
    def centroid_lat(self: Self) -> float:
        """Centroid latitude in degrees."""
        return (self.endpoint1_lat + self.endpoint2_lat) / 2

    @property
    def centroid_lon(self: Self) -> float:
        """Centroid longitude in degrees."""
        delta = self.endpoint2_lon - self.endpoint1_lon
        if abs(delta) <= 180.0:
            return (self.endpoint1_lon + self.endpoint2_lon) / 2
        # Endpoints either side of the date line
        return _wrap_lon(self.endpoint1_lon + _wrap_lon(delta) / 2)

    @property
    def length(self: Self) -> float:
        """Great-circle length of the contrail in meters."""
        return haversine(
            self.endpoint1_lon,
            self.endpoint1_lat,
            self.endpoint2_lon,
            self.endpoint2_lat,
            input_unit="degrees",
        )

    @classmethod
    def from_centroid_azimuth_length(
        cls,
        contrail_id: str | None,
        centroid_lon: float,
        centroid_lat: float,
        azimuth_deg: float,
        length_m: float,
        u_init: float | None = None,
        v_init: float | None = None,
        debug_info: TrackingDebugInfo | None = None,
    ) -> "Contrail":
        """Create a Contrail from centroid, azimuth, and length.

        Args:
            contrail_id: Unique identifier string, or None for synthetic contrails.
            centroid_lon: Centroid longitude in degrees.
            centroid_lat: Centroid latitude in degrees.
            azimuth_deg: Forward azimuth in degrees.
            length_m: Total contrail length in meters.
            u_init: Optional initial zonal wind estimate in m/s.
            v_init: Optional initial meridional wind estimate in m/s.
            debug_info: Optional mutable debug container.

        Returns:
            A new Contrail instance constructed from the given geometric parameters.
        """
        half_length = length_m / 2
        fwd_az_rad = np.radians(azimuth_deg)
        dlon_half = units.m_to_longitude_distance(half_length * np.sin(fwd_az_rad), centroid_lat)
        dlat_half = units.m_to_latitude_distance(half_length * np.cos(fwd_az_rad))
        endpoint1_lon = centroid_lon - dlon_half
        endpoint1_lat = centroid_lat - dlat_half
        endpoint2_lon = centroid_lon + dlon_half
        endpoint2_lat = centroid_lat + dlat_half
        if debug_info is not None:
            debug_info.predicted_contrails.append(
                {
                    "contrail_id": contrail_id,
                    "centroid_lon": centroid_lon,
                    "centroid_lat": centroid_lat,
                    "azimuth_deg": azimuth_deg,
                    "length_m": length_m,
                    "endpoint1_lon": endpoint1_lon,
                    "endpoint1_lat": endpoint1_lat,
                    "endpoint2_lon": endpoint2_lon,
                    "endpoint2_lat": endpoint2_lat,
                    "u_init": u_init,
                    "v_init": v_init,
                }
            )
        return cls(
            contrail_id=contrail_id,
            endpoint1_lon=endpoint1_lon,
            endpoint1_lat=endpoint1_lat,
            endpoint2_lon=endpoint2_lon,
            endpoint2_lat=endpoint2_lat,
            u_init=u_init,
            v_init=v_init,
            debug_info=debug_info,
        )

    def reversed_wind(self: Self) -> "Contrail":
        """Return a new Contrail with u_init and v_init reversed in sign.

        Returns:
            A copy of this Contrail with negated wind components.
        """
        return Contrail(
            contrail_id=self.contrail_id,
            endpoint1_lon=self.endpoint1_lon,
            endpoint1_lat=self.endpoint1_lat,
            endpoint2_lon=self.endpoint2_lon,
            endpoint2_lat=self.endpoint2_lat,
            u_init=-self.u_init if self.u_init is not None else None,
            v_init=-self.v_init if self.v_init is not None else None,
        )


@dataclass(frozen=True)
class KalmanFilterConfig:
    """Configuration dataclass for KalmanFilter2D noise parameters.

    Attributes:
        position_measurement_standard_deviation: Measurement noise for position
            in degrees. Keep small to trust measurements.
        position_process_standard_deviation: Process noise for position in
            degrees. Increase to forget old predictions faster.
        velocity_process_standard_deviation: Process noise for velocity in m/s.
            Increase to allow velocity to adapt quickly.
        initial_position_standard_deviation: Initial position uncertainty in
            degrees. Importance decreases with more frames.
        initial_velocity_standard_deviation: Initial velocity uncertainty in
            m/s. Importance decreases with more frames.
    """

    position_measurement_standard_deviation: float = 0.001
    position_process_standard_deviation: float = 0.005
    velocity_process_standard_deviation: float = 0.5
    initial_position_standard_deviation: float = 0.005
    initial_velocity_standard_deviation: float = 0.5


@dataclass(frozen=True)
class TrackingConfig:
    """Configuration dataclass controlling all aspects of the tracking pipeline.

    Attributes:
        name: Human-readable name for this configuration.
        cost_components: Tuple of weighted CostComponent instances.
        gates: Tuple of TrackingGate instances applied in order (cheap first).
        kalman_filter_config: Noise parameters for the Kalman filter.
        max_cost: Maximum allowable total cost for a valid match.
        max_frames_to_skip: Maximum consecutive unmatched frames before a track
            is closed.
        method: Tracking direction — 'forward', 'backward', or 'bidirectional'.
        use_wind_initialization: Whether to use meteorological windspeed data for
            initial contrail track velocity estimates.
        debug: Whether to collect debug info during tracking.
    """

    # Defaults are the GOES-East operating point chosen from a sweep over skipped frames,
    # vicinity and perpendicular-gate settings
    name: str = "default"

    # Default cost components for track assignment
    cost_components: tuple[CostComponent, ...] = (
        PerpendicularCost(weight=0.7, normalizing_distance=10000, scaling="linear"),
        OrientationCost(weight=0.3, normalizing_angle=30),
    )

    # Default gates for restricting track assignment search, order matters:
    #  - cheaper gates should be first to filter out bad matches
    #  - expensive gates applied afterwards for refinement
    gates: tuple[TrackingGate, ...] = (
        VicinityGate(max_match_deg=0.8),
        OrientationChangeGate(max_orientation_change=30.0),
        PerpendicularDistanceGate(max_perpendicular_distance=10000.0),
    )

    kalman_filter_config: KalmanFilterConfig = KalmanFilterConfig()

    # Doesn't really matter if cost and gates are set well, hence kept high here
    max_cost: float = 10

    # Maximum number of consecutive frames a track can be unmatched before it is closed
    # (1 bridges a contrail briefly missed by the detector; higher values over-merge)
    max_frames_to_skip: int = 1

    # Whether to track forward in time, backward in time, or create links consistent
    # with both directions
    method: Literal["forward", "backward", "bidirectional"] = "forward"

    # Whether to use meteorological windspeed data for initialization
    use_wind_initialization: bool = True

    # Whether to collect debug info during tracking
    debug: bool = False

    # --- --- ---

    def to_dict(self: Self) -> dict:
        """Return a dictionary representation of the config including component attributes.

        Returns:
            A dict containing name, serialized cost components and gates,
            Kalman filter config, and scalar tracking parameters.
        """
        return {
            "name": self.name,
            "cost_components": [
                {"type": type(c).__name__, **dataclass_to_dict(c)} for c in self.cost_components
            ],
            "gates": [{"type": type(g).__name__, **dataclass_to_dict(g)} for g in self.gates],
            "kalman_filter_config": dataclass_to_dict(self.kalman_filter_config),
            "max_cost": self.max_cost,
            "max_frames_to_skip": self.max_frames_to_skip,
            "method": self.method,
            "use_wind_initialization": self.use_wind_initialization,
            "debug": self.debug,
        }


class KalmanFilter2D:
    """Two-dimensional Kalman filter tracking contrail centroid position and velocity.

    Maintains a four-element state vector [lon, lat, u, v] where lon/lat are
    in degrees and u/v are wind velocity components in m/s.

    Attributes:
        config: KalmanFilterConfig holding noise parameters.
        state: Current state vector [lon, lat, u, v].
        P: State covariance matrix (4x4).
        Q: Process noise covariance matrix (4x4).
        H: Measurement matrix (2x4), extracts position from state.
        R: Measurement noise covariance matrix (2x2).
        F: State transition matrix (4x4), updated each predict step.
    """

    def __init__(
        self: Self,
        lon: float,
        lat: float,
        u: float,
        v: float,
        config: KalmanFilterConfig,
    ) -> None:
        """Initialize filter at a given position and velocity.

        Args:
            lon: Initial longitude in degrees.
            lat: Initial latitude in degrees.
            u: Initial zonal velocity in m/s.
            v: Initial meridional velocity in m/s.
            config: Noise parameter configuration.
        """
        self.config = config
        self.state: NDArray = np.array([lon, lat, u, v])  # State vector [lon, lat, u, v]
        self.P: NDArray = np.diag(
            [
                config.initial_position_standard_deviation**2,
                config.initial_position_standard_deviation**2,
                config.initial_velocity_standard_deviation**2,
                config.initial_velocity_standard_deviation**2,
            ]
        )  # State covariance
        self.Q: NDArray = np.diag(
            [
                config.position_process_standard_deviation**2,
                config.position_process_standard_deviation**2,
                config.velocity_process_standard_deviation**2,
                config.velocity_process_standard_deviation**2,
            ]
        )  # Process noise covariance
        self.H: NDArray = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )  # Measurement matrix
        self.R: NDArray = np.diag(
            [
                config.position_measurement_standard_deviation**2,
                config.position_measurement_standard_deviation**2,
            ]
        )  # Measurement noise covariance

        self.F = np.eye(4)  # State transition matrix, will be updated based on delta_t

    def __repr__(self: Self) -> str:
        """Return a concise string representation of the filter state."""
        lon, lat, u, v = self.state
        return f"KalmanFilter2D(lon={lon:.4f}, lat={lat:.4f}, u={u:.2f}, v={v:.2f})"

    def predict(self: Self, delta_t: float) -> None:
        """Predict the next state based on the current state and time interval.

        Args:
            delta_t: Time interval in seconds since the last update.
        """
        lon, lat, u, v = self.state
        dlon_deg = units.m_to_longitude_distance(u * delta_t, lat)
        dlat_deg = units.m_to_latitude_distance(v * delta_t)
        self.F[0, 2] = delta_t * units.m_to_longitude_distance(1.0, lat)  # unit movement
        self.F[1, 3] = delta_t * units.m_to_latitude_distance(1.0)  # unit movement
        new_lon = lon + dlon_deg
        if not -180.0 <= new_lon < 180.0:  # moved across the date line
            new_lon = _wrap_lon(new_lon)
        self.state = [new_lon, lat + dlat_deg, u, v]
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self: Self, measurement_lon: float, measurement_lat: float) -> None:
        """Update the state based on a new position measurement.

        Args:
            measurement_lon: Measured longitude in degrees.
            measurement_lat: Measured latitude in degrees.
        """
        z = np.array([measurement_lon, measurement_lat])
        y = z - self.H @ self.state  # Measurement residual
        if abs(y[0]) > 180.0:  # measurement and state either side of the date line
            y[0] = _wrap_lon(y[0])
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state = self.state + K @ y  # Updated state estimate
        if not -180.0 <= self.state[0] < 180.0:
            self.state[0] = _wrap_lon(self.state[0])
        eye = np.eye(4)
        self.P = (eye - K @ self.H) @ self.P  # Updated estimate covariance


class TrackedContrail:
    """A multi-frame contrail track maintained by a Kalman filter.

    Accumulates successive Contrail observations, predicts centroid movement
    via KalmanFilter2D, and records hemisphere-shift history for orientation
    consistency checks.

    Attributes:
        kalman_filter: The underlying KalmanFilter2D for this track.
        contrails: Ordered list of all Contrail observations in this track.
        frames_since_update: Number of consecutive frames without a match.
        first_contrail: The initial Contrail that started this track.
        hemisphere_shifts: List of hemisphere-shift values recorded at each update.
        last_nonzero_shift: The most recent non-zero hemisphere shift value.
        reference_azimuth: Azimuth of the first contrail, used for orientation.
        debug_info: Optional mutable debug container.
    """

    def __init__(
        self: Self,
        initial_contrail: Contrail,
        config: KalmanFilterConfig,
        debug_info: TrackingDebugInfo | None = None,
    ) -> None:
        """Initialize a new track from a single contrail observation.

        Args:
            initial_contrail: The first Contrail observation for this track.
            config: Kalman filter noise configuration.
            debug_info: Optional mutable debug container.
        """
        self.kalman_filter = KalmanFilter2D(
            lon=initial_contrail.centroid_lon,
            lat=initial_contrail.centroid_lat,
            u=initial_contrail.u_init or 0.0,
            v=initial_contrail.v_init or 0.0,
            config=config,
        )
        self.contrails: list[Contrail] = [initial_contrail]
        self.frames_since_update: int = 0
        self.first_contrail = initial_contrail
        self.hemisphere_shifts: list[int] = []
        self.last_nonzero_shift: int = 0

        self.reference_azimuth = initial_contrail.azimuth
        self.debug_info = debug_info
        logging.debug(f"Created new track for contrail {initial_contrail.contrail_id}.")

    @property
    def contrail_ids(self: Self) -> list[str]:
        """List of contrail_id values for all observations in this track."""
        return [c.contrail_id for c in self.contrails]

    @property
    def direction_of_travel(self: Self) -> float | None:
        """Azimuth of centroid movement between the last two observations, or None."""
        if len(self.contrails) < 2:
            return None
        last_contrail = self.contrails[-1]
        prev_contrail = self.contrails[-2]
        return azimuth(
            prev_contrail.centroid_lat,
            prev_contrail.centroid_lon,
            last_contrail.centroid_lat,
            last_contrail.centroid_lon,
        )

    @property
    def last_contrail(self: Self) -> Contrail:
        """The most recently observed Contrail in this track."""
        return self.contrails[-1]

    @property
    def last_position_change(self: Self) -> float | None:
        """Great-circle distance between the last two centroids in meters, or None."""
        if len(self.contrails) < 2:
            return None
        last_contrail = self.contrails[-1]
        prev_contrail = self.contrails[-2]
        return haversine(
            prev_contrail.centroid_lon,
            prev_contrail.centroid_lat,
            last_contrail.centroid_lon,
            last_contrail.centroid_lat,
            input_unit="degrees",
        )

    @property
    def predicted_contrail(self: Self) -> Contrail:
        """Synthetic Contrail at the Kalman-predicted centroid position."""
        predicted_lon, predicted_lat, _, _ = self.kalman_filter.state
        return Contrail.from_centroid_azimuth_length(
            contrail_id=None,
            centroid_lon=predicted_lon,
            centroid_lat=predicted_lat,
            azimuth_deg=self.last_contrail.azimuth,
            length_m=self.last_contrail.length,
            debug_info=self.debug_info,
        )

    @property
    def track_id(self: Self) -> str:
        """ID of the contrail that started this track.

        In a backward pass this is the latest contrail; ``_process_ordered`` names
        finished tracks by the earliest.
        """
        return self.contrails[0].contrail_id

    def predict(
        self: Self,
        delta_t: float,
        debug_info: TrackingDebugInfo | None = None,
    ) -> None:
        """Advance the Kalman filter by one time step.

        Args:
            delta_t: Time interval in seconds.
            debug_info: Optional debug container for recording filter states.
        """
        if debug_info is not None:
            base_lon, base_lat, base_u, base_v = self.kalman_filter.state

        self.kalman_filter.predict(delta_t)
        if debug_info is not None:
            pred_lon, pred_lat, pred_u, pred_v = self.kalman_filter.state
            debug_info.kalman_filter_states.append(
                {
                    "track_id": getattr(self, "track_id", None),
                    "base_lon": base_lon,
                    "base_lat": base_lat,
                    "base_u": base_u,
                    "base_v": base_v,
                    "pred_lon": pred_lon,
                    "pred_lat": pred_lat,
                    "pred_u": pred_u,
                    "pred_v": pred_v,
                }
            )
        self.frames_since_update += 1

    def update(
        self: Self,
        new_contrail: Contrail,
        debug_info: TrackingDebugInfo | None = None,
        timestamp: pd.Timestamp | None = None,
        debug_direction: str | None = None,
    ) -> None:
        """Incorporate a new contrail observation into the track.

        Args:
            new_contrail: The newly matched Contrail observation.
            debug_info: Optional debug container for recording wind estimates.
            timestamp: Frame timestamp associated with this observation.
            debug_direction: Tracking direction label for debug records.
        """
        self.kalman_filter.update(new_contrail.centroid_lon, new_contrail.centroid_lat)
        prev_contrail = self.contrails[-1]
        self.contrails.append(new_contrail)
        shift = self._hemisphere_shift(prev_contrail, new_contrail)
        self.hemisphere_shifts.append(shift)
        if shift != 0:
            self.last_nonzero_shift = shift
        self.frames_since_update = 0
        logging.debug(f"Updated track with contrail {new_contrail.contrail_id}.")
        if debug_info is not None:
            _, _, u, v = self.kalman_filter.state
            debug_info.wind_estimates.append(
                {
                    "track_id": getattr(self, "track_id", None),
                    "debug_direction": debug_direction,
                    "u": u,
                    "v": v,
                    "contrail_id": getattr(new_contrail, "contrail_id", None),
                    "timestamp": timestamp,
                }
            )

    def _hemisphere_shift(
        self: Self,
        prev_contrail: Contrail,
        new_contrail: Contrail,
    ) -> int:
        """Compute hemisphere shift relative to previous contrail, oriented to reference azimuth.

        Args:
            prev_contrail: The previous Contrail observation in this track.
            new_contrail: The new candidate Contrail observation.

        Returns:
            Integer hemisphere shift: -1, 0, or +1.
        """
        prev_p1, prev_p2 = self._orient_like_reference(prev_contrail)
        new_p1, new_p2 = self._orient_like_reference(new_contrail)
        shift1 = hemisphere(
            np.radians(new_p1[0]),
            np.radians(new_p1[1]),
            np.radians(prev_p1[0]),
            np.radians(prev_p1[1]),
            np.radians(prev_p2[0]),
            np.radians(prev_p2[1]),
        )
        shift2 = hemisphere(
            np.radians(new_p2[0]),
            np.radians(new_p2[1]),
            np.radians(prev_p1[0]),
            np.radians(prev_p1[1]),
            np.radians(prev_p2[0]),
            np.radians(prev_p2[1]),
        )
        if shift1 == -1 and shift2 == 1:
            return 0
        elif shift1 == 1 and shift2 == -1:
            return 0
        else:
            return shift1

    def _orient_like_reference(
        self: Self, contrail: Contrail
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Orient contrail endpoints to be within ±90° of reference azimuth (first contrail).

        Args:
            contrail: The Contrail whose endpoints are to be oriented.

        Returns:
            A pair of (lat, lon) tuples for the oriented (p1, p2) endpoints.
        """
        p1 = (contrail.endpoint1_lat, contrail.endpoint1_lon)
        p2 = (contrail.endpoint2_lat, contrail.endpoint2_lon)
        seg_heading = azimuth(p1[0], p1[1], p2[0], p2[1])
        diff = (seg_heading - self.reference_azimuth + 180) % 360 - 180
        # Always orient so heading is within ±90° of reference
        if abs(diff) > 90:
            return p2, p1
        else:
            return p1, p2


class BaseTracker(ABC):
    """Abstract base class for directional contrail trackers.

    Manages gate evaluation, cost matrix construction, Hungarian assignment,
    and track lifecycle (creation, update, closure).

    Attributes:
        gates: Sequence of TrackingGate instances applied before cost evaluation.
        cost_components: Sequence of weighted CostComponent instances.
        kalman_filter_config: Noise parameters forwarded to new TrackedContrail
            instances.
        max_cost: Maximum total cost for a valid assignment.
        max_frames_to_skip: Maximum consecutive unmatched frames before closure.
        debug_info: Optional shared debug container.
        direction: Direction label set by subclasses (``'forward'`` or ``'backward'``).
    """

    def __init__(
        self: Self,
        config: TrackingConfig,
        debug_info: TrackingDebugInfo | None = None,
    ) -> None:
        """Initialize the tracker from a TrackingConfig.

        Args:
            config: Full tracking configuration.
            debug_info: Optional shared debug container.
        """
        self.gates = config.gates
        self.cost_components = config.cost_components
        self.kalman_filter_config = config.kalman_filter_config
        self.max_cost = config.max_cost
        self.max_frames_to_skip = config.max_frames_to_skip
        self.debug_info = debug_info
        self.direction = None  # will be set in subclasses

    @abstractmethod
    def process(
        self: Self,
        frames: dict[pd.Timestamp, tuple[Contrail, ...]],
        frame_order: list[pd.Timestamp],
        frame_intervals: NDArray,
    ) -> dict[str, list[str]]:
        """Run tracking over all frames and return a track-to-contrail mapping.

        Args:
            frames: Mapping from timestamp to tuple of Contrail observations.
            frame_order: Chronologically sorted list of timestamps.
            frame_intervals: Array of inter-frame durations in seconds.

        Returns:
            A dict mapping track_id to an ordered list of contrail_id strings.
        """
        pass

    def _assign_matches(
        self: Self,
        frame_contrails: tuple[Contrail, ...],
        active_tracks: list[TrackedContrail],
        cost_matrix: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Assign contrails to tracks using the Hungarian algorithm.

        Args:
            frame_contrails: Contrails detected in the current frame.
            active_tracks: Currently active tracked contrails.
            cost_matrix: Shape (n_contrails, n_tracks), np.inf for gated-out pairs.

        Returns:
            matched: List of (contrail_idx, track_idx) pairs.
            unmatched_contrails: Contrail indices with no match.
            unmatched_tracks: Track indices with no match.
        """
        n_contrails = len(frame_contrails)
        n_tracks = len(active_tracks)

        # Find all valid (below max_cost) pairs
        valid_mask = cost_matrix <= self.max_cost
        valid_rows, valid_cols = np.where(valid_mask)

        if len(valid_rows) == 0:
            return [], list(range(n_contrails)), list(range(n_tracks))

        # Build a reduced cost matrix with only the rows/cols that have at least one valid pair
        unique_rows = np.unique(valid_rows)
        unique_cols = np.unique(valid_cols)
        reduced = cost_matrix[np.ix_(unique_rows, unique_cols)]

        # Entries that are still invalid in the reduced matrix get a large but finite value
        reject_value = 1e10
        reduced = np.where(reduced <= self.max_cost, reduced, reject_value)

        row_ind, col_ind = linear_sum_assignment(reduced)

        # Map back to original indices and filter
        matched = []
        for r, c in zip(row_ind, col_ind):
            orig_r = unique_rows[r]
            orig_c = unique_cols[c]
            if cost_matrix[orig_r, orig_c] <= self.max_cost:
                matched.append((int(orig_r), int(orig_c)))
                if self.debug_info is not None:
                    for entry in self.debug_info.costs:
                        if entry.get("track_idx") == int(orig_c) and entry.get(
                            "contrail_idx"
                        ) == int(orig_r):
                            entry["accepted"] = True
                            break

        matched_contrails = {m[0] for m in matched}
        matched_tracks = {m[1] for m in matched}
        unmatched_contrails = [i for i in range(n_contrails) if i not in matched_contrails]
        unmatched_tracks = [j for j in range(n_tracks) if j not in matched_tracks]

        return matched, unmatched_contrails, unmatched_tracks

    def _build_cost_matrix(
        self: Self,
        frame_contrails: tuple[Contrail, ...],
        active_tracks: list[TrackedContrail],
    ) -> np.ndarray:
        """Build a cost matrix for all contrail-track pairs.

        Pairs that fail any gate are assigned np.inf. Passing pairs receive a
        weighted sum of cost component values.

        Args:
            frame_contrails: Contrails detected in the current frame.
            active_tracks: Currently active tracked contrails.

        Returns:
            A float array of shape (n_contrails, n_tracks) with np.inf for
            gated-out pairs.
        """
        cost_matrix = np.full((len(frame_contrails), len(active_tracks)), np.inf)

        for i, contrail in enumerate(frame_contrails):
            for j, track in enumerate(active_tracks):
                context = GateContext(contrail=contrail, potential_tracked_contrail=track)
                if all(gate.allow(context, debug_info=self.debug_info) for gate in self.gates):
                    cost_matrix[i, j] = self._evaluate_cost(context)

                    if self.debug_info is not None:
                        component_costs = {}
                        for component in self.cost_components:
                            component_costs[type(component).__name__] = component.compute(context)
                        self.debug_info.costs.append(
                            {
                                "accepted": False,  # Will update to True for assigned matches
                                "components": component_costs,
                                "track_idx": j,
                                "contrail_idx": i,
                                "track_id": getattr(track, "track_id", None),
                                "contrail_id": getattr(contrail, "contrail_id", None),
                                "direction": self.direction,
                            }
                        )

        return cost_matrix

    def _evaluate_cost(self: Self, context: GateContext) -> float:
        """Compute the weighted total cost for a contrail-track pair.

        Args:
            context: The gate context containing contrail and track.

        Returns:
            The weighted sum of all cost component values.
        """
        total_cost = 0.0
        for component in self.cost_components:
            component_cost = component.compute(context)
            total_cost += component.weight * component_cost
            logging.debug(
                f"Cost component {type(component).__name__}: "
                f"{component_cost:.4f} (weight {component.weight})"
            )
        logging.debug(
            f"Total cost for track "
            f"{getattr(context.potential_tracked_contrail, 'track_id', None)} "
            f"and contrail {getattr(context.contrail, 'contrail_id', None)}: {total_cost:.4f}"
        )
        return total_cost

    def _process_ordered(
        self: Self,
        frames: dict[pd.Timestamp, tuple[Contrail, ...]],
        frame_order: list[pd.Timestamp],
        frame_intervals: NDArray,
        reverse: bool,
    ) -> dict[str, list[str]]:
        """Core frame-by-frame tracking loop.

        Iterates over frames in the specified order, predicts track positions,
        builds the cost matrix, assigns matches via the Hungarian algorithm, and
        maintains track lifecycle.

        Args:
            frames: Mapping from timestamp to tuple of Contrail observations.
            frame_order: Chronologically sorted list of timestamps.
            frame_intervals: Array of inter-frame durations in seconds.
            reverse: If True, iterate frames from last to first.

        Returns:
            A dict mapping track_id to an ordered list of contrail_id strings.
        """
        active_tracks: list[TrackedContrail] = []
        closed_tracks: list[TrackedContrail] = []

        ordered = list(reversed(frame_order)) if reverse else frame_order
        intervals = frame_intervals[::-1] if reverse else frame_intervals

        for idx, timestamp in enumerate(ordered):
            frame_contrails = frames[timestamp]

            if idx > 0:
                delta_t = intervals[idx - 1]
                for track in active_tracks:
                    track.predict(delta_t, debug_info=self.debug_info)

            cost_matrix = self._build_cost_matrix(frame_contrails, active_tracks)
            matched, unmatched_contrails, unmatched_tracks = self._assign_matches(
                frame_contrails, active_tracks, cost_matrix
            )

            logging.info(
                f"Frame {idx + 1}/{len(ordered)}: {len(frame_contrails)} contrails, "
                f"{len(active_tracks)} active tracks, {len(matched)} matches, "
                f"{len(unmatched_contrails)} new, {len(unmatched_tracks)} unmatched."
            )

            new_active_tracks = []
            # Matched contrails are used to update existing tracks
            for contrail_idx, track_idx in matched:
                updated_track = active_tracks[track_idx]
                updated_track.update(
                    frame_contrails[contrail_idx],
                    debug_info=self.debug_info,
                    timestamp=timestamp,
                    debug_direction=self.direction,
                )
                new_active_tracks.append(updated_track)
            # Unmatched tracks are kept active if they haven't been unmatched for too many frames
            for track_idx in unmatched_tracks:
                track = active_tracks[track_idx]
                if track.frames_since_update <= self.max_frames_to_skip:
                    new_active_tracks.append(track)
                else:
                    closed_tracks.append(track)
                    logging.debug(
                        f"Closed track with contrails: {[c.contrail_id for c in track.contrails]}"
                    )
            # Unmatched contrails start new tracks
            for contrail_idx in unmatched_contrails:
                initial_contrail = frame_contrails[contrail_idx]
                if reverse:
                    initial_contrail = initial_contrail.reversed_wind()
                new_track = TrackedContrail(
                    initial_contrail=initial_contrail,
                    config=self.kalman_filter_config,
                    debug_info=self.debug_info,
                )
                new_active_tracks.append(new_track)

            active_tracks = new_active_tracks

        # At the end, all remaining active tracks are closed
        closed_tracks.extend(active_tracks)

        # track_id -> contrail_ids, each track named by its earliest contrail
        return {
            (track.contrail_ids[-1] if reverse else track.contrail_ids[0]): track.contrail_ids
            for track in closed_tracks
        }


class BackwardTracker(BaseTracker):
    """Tracker that processes frames in reverse chronological order."""

    def __init__(
        self: Self,
        config: TrackingConfig,
        debug_info: TrackingDebugInfo | None = None,
    ) -> None:
        """Initialize a backward tracker.

        Args:
            config: Full tracking configuration.
            debug_info: Optional shared debug container.
        """
        super().__init__(config, debug_info=debug_info)
        self.direction = "backward"

    def process(
        self: Self,
        frames: dict[pd.Timestamp, tuple[Contrail, ...]],
        frame_order: list[pd.Timestamp],
        frame_intervals: NDArray,
    ) -> dict[str, list[str]]:
        """Run backward tracking over all frames.

        Args:
            frames: Mapping from timestamp to tuple of Contrail observations.
            frame_order: Chronologically sorted list of timestamps.
            frame_intervals: Array of inter-frame durations in seconds.

        Returns:
            A dict mapping track_id to an ordered list of contrail_id strings.
        """
        return self._process_ordered(frames, frame_order, frame_intervals, reverse=True)


class ForwardTracker(BaseTracker):
    """Tracker that processes frames in chronological order."""

    def __init__(
        self: Self,
        config: TrackingConfig,
        debug_info: TrackingDebugInfo | None = None,
    ) -> None:
        """Initialize a forward tracker.

        Args:
            config: Full tracking configuration.
            debug_info: Optional shared debug container.
        """
        super().__init__(config, debug_info=debug_info)
        self.direction = "forward"

    def process(
        self: Self,
        frames: dict[pd.Timestamp, tuple[Contrail, ...]],
        frame_order: list[pd.Timestamp],
        frame_intervals: NDArray,
    ) -> dict[str, list[str]]:
        """Run forward tracking over all frames.

        Args:
            frames: Mapping from timestamp to tuple of Contrail observations.
            frame_order: Chronologically sorted list of timestamps.
            frame_intervals: Array of inter-frame durations in seconds.

        Returns:
            A dict mapping track_id to an ordered list of contrail_id strings.
        """
        return self._process_ordered(frames, frame_order, frame_intervals, reverse=False)


class BidirectionalTracker(BaseTracker):
    """Tracker that retains only links consistent with both forward and backward passes."""

    def process(
        self: Self,
        frames: dict[pd.Timestamp, tuple[Contrail, ...]],
        frame_order: list[pd.Timestamp],
        frame_intervals: NDArray,
    ) -> dict[str, list[str]]:
        """Run bidirectional tracking and return only reciprocally consistent links.

        Args:
            frames: Mapping from timestamp to tuple of Contrail observations.
            frame_order: Chronologically sorted list of timestamps.
            frame_intervals: Array of inter-frame durations in seconds.

        Returns:
            A dict mapping track_id to an ordered list of contrail_id strings,
            containing only links that appear in both directions.
        """
        forward_tracks = self._process_ordered(frames, frame_order, frame_intervals, reverse=False)
        forward_links = self._build_links_from_tracking_dict(forward_tracks)
        backward_tracks = self._process_ordered(frames, frame_order, frame_intervals, reverse=True)
        backward_links = self._build_links_from_tracking_dict(backward_tracks)
        backward_links_forward = {(b, a) for (a, b) in backward_links}

        reciprocal_links = forward_links & backward_links_forward
        return self._links_to_tracking_dict(reciprocal_links)

    def _build_links_from_tracking_dict(
        self: Self,
        tracking_dict: dict[str, list[str]],
    ) -> set[tuple[str, str]]:
        """Create a set of consecutive contrail_id pairs from a tracking dict.

        Args:
            tracking_dict: Mapping from track_id to ordered list of contrail_ids.

        Returns:
            A set of (contrail_id_a, contrail_id_b) pairs for each consecutive
            observation within each track.
        """
        links: set[tuple[str, str]] = set()
        for contrail_ids in tracking_dict.values():
            for i in range(len(contrail_ids) - 1):
                links.add((contrail_ids[i], contrail_ids[i + 1]))
        return links

    def _links_to_tracking_dict(
        self: Self,
        links: set[tuple[str, str]],
    ) -> dict[str, list[str]]:
        """Build a tracking dict from a set of consecutive contrail_id pairs.

        Args:
            links: Set of (contrail_id_a, contrail_id_b) consecutive pairs.

        Returns:
            A dict mapping track_id to an ordered list of contrail_id strings.
        """
        # Build adjacency mapping
        next_map: dict[str, str] = {}
        prev_map: dict[str, str] = {}
        for a, b in links:
            next_map[a] = b
            prev_map[b] = a

        # Find track starts (contrails with no predecessor)
        starts = [a for a in next_map if a not in prev_map]

        tracking_dict: dict[str, list[str]] = {}
        for start in starts:
            chain = [start]
            curr = start
            while curr in next_map:
                curr = next_map[curr]
                chain.append(curr)
            # Name the track by its earliest contrail
            tracking_dict[chain[0]] = chain
        return tracking_dict


class TrackingManager:
    """Top-level interface for running multi-object contrail tracking.

    Converts a contrail summary DataFrame into per-frame Contrail objects,
    selects the appropriate tracker based on config, runs tracking, and
    returns a contrail_id-to-track_id mapping.

    Attributes:
        config: The TrackingConfig controlling tracker behavior.
        debug_info: Optional debug container populated during tracking.
        frame_interval: Inferred or supplied inter-frame Timedelta.
        timeline: Full date_range from first to last timestamp.
        frames: Mapping from timestamp to tuple of Contrail observations.
        frame_order: Sorted list of frame timestamps.
        frame_intervals: Array of inter-frame durations in seconds.
    """

    def __init__(
        self: Self,
        contrail_summary: pd.DataFrame,
        config: TrackingConfig = TrackingConfig(),
        frame_interval: pd.Timedelta | None = None,
        enforce_uniform_intervals: bool = True,
        use_wind_initialization: bool | None = None,
    ) -> None:
        """Initialize the manager from a contrail summary DataFrame.

        Args:
            contrail_summary: DataFrame with columns including ``'timestamp'``,
                ``'contrail_id'``, ``'endpoint1_lon'``, ``'endpoint1_lat'``,
                ``'endpoint2_lon'``, ``'endpoint2_lat'``, and optionally ``'u_init'``
                and ``'v_init'``.
            config: Full tracking configuration.
            frame_interval: Optional fixed inter-frame interval. If None, it
                is inferred as the minimum gap between timestamps.
            enforce_uniform_intervals: If True, raise ValueError when a timestamp is
                not a whole number of ``frame_interval`` after the first. If False,
                such frames are added to the timeline.
            use_wind_initialization: If not None, overrides
                config.use_wind_initialization.
        """
        self.config = config

        # Optionally override wind initialization boolean
        if use_wind_initialization is not None:
            self.config = replace(self.config, use_wind_initialization=use_wind_initialization)

        # Tracking debug populating
        self.debug_info = TrackingDebugInfo() if self.config.debug else None

        # Set time attributes
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(contrail_summary["timestamp"].sort_values().unique())
        )
        if frame_interval is None:
            if len(timestamps) < 2:
                raise ValueError("Cannot infer frame_interval from fewer than 2 timestamps.")
            diffs = (timestamps[1:] - timestamps[:-1]).to_numpy()
            min_diff = min(diffs)
            frame_interval = pd.Timedelta(min_diff)
        self.frame_interval = frame_interval

        # Create timeline attribute first
        min_time = timestamps.min()
        max_time = timestamps.max()
        self.timeline = pd.date_range(
            start=min_time,
            end=max_time,
            freq=frame_interval,
        )

        # Frames not on the frame_interval grid
        off_grid = timestamps[~timestamps.isin(self.timeline)]
        if len(off_grid) > 0:
            if enforce_uniform_intervals:
                shown = ", ".join(str(ts) for ts in off_grid[:5])
                raise ValueError(
                    f"{len(off_grid)} frame time(s) not a whole number of {frame_interval} "
                    f"after the first frame ({min_time}): {shown}"
                    f"{', ...' if len(off_grid) > 5 else ''}. Pass the right frame_interval, "
                    "or enforce_uniform_intervals=False to track them at their own times."
                )
            self.timeline = self.timeline.union(off_grid)

        # Create dictionary of frame data for each frame from contrail summary
        self.frames: dict[pd.Timestamp, tuple[Contrail, ...]] = self._create_frames(
            contrail_summary
        )
        self.frame_order: list[pd.Timestamp] = sorted(self.frames.keys())
        self.frame_intervals: NDArray = self._compute_frame_intervals()

        if enforce_uniform_intervals:
            self._validate_uniform_frame_intervals()

    def run_tracking(self: Self) -> dict[str, str]:
        """Execute tracking and return a contrail_id-to-track_id mapping.

        Returns:
            A dict mapping each contrail_id that belongs to a multi-frame track
            to its track_id string.
        """
        method = self.config.method
        logging.info(f"Starting tracking with method: {method}")
        trackers: dict[str, BaseTracker] = {
            "forward": ForwardTracker(self.config, debug_info=self.debug_info),
            "backward": BackwardTracker(self.config, debug_info=self.debug_info),
            "bidirectional": BidirectionalTracker(self.config, debug_info=self.debug_info),
        }
        if method not in trackers:
            logging.error(f"Unknown tracking method: {method}")
            raise ValueError(f"Unknown tracking method: {method}")

        tracker = trackers[method]
        result = tracker.process(
            frames=self.frames,
            frame_order=self.frame_order,
            frame_intervals=self.frame_intervals,
        )

        mapping = self._convert_tracking_dict_to_mapping(result)

        logging.info("Tracking complete.")

        return mapping

    def _compute_frame_intervals(self: Self) -> NDArray:
        """Compute inter-frame durations in seconds from the ordered frame list.

        Returns:
            A float array of length (n_frames - 1) with durations in seconds,
            or an empty array when fewer than two frames are present.
        """
        if len(self.frame_order) < 2:
            return np.array([])
        return np.array(
            [
                (self.frame_order[idx] - self.frame_order[idx - 1]).total_seconds()
                for idx in range(1, len(self.frame_order))
            ]
        )

    def _convert_tracking_dict_to_mapping(
        self: Self,
        tracking_dict: dict[str, list[str]],
    ) -> dict[str, str]:
        """Convert track_id -> list of contrail_ids to contrail_id -> track_id.

        Only includes tracks that contain more than one contrail observation.

        Args:
            tracking_dict: Mapping from track_id to ordered list of contrail_ids.

        Returns:
            A dict mapping contrail_id to track_id for all multi-frame tracks.
        """
        mapping: dict[str, str] = {}
        multiframe_tracks = 0
        for track_id, contrail_ids in tracking_dict.items():
            if len(contrail_ids) > 1:
                for contrail_id in contrail_ids:
                    mapping[contrail_id] = track_id
                multiframe_tracks += 1

        logging.info(f"Total multiframe tracks: {multiframe_tracks}")
        return mapping

    def _create_frames(
        self: Self, contrail_summary: pd.DataFrame
    ) -> dict[pd.Timestamp, tuple[Contrail, ...]]:
        """Build the per-timestamp frame dictionary from the summary DataFrame.

        Args:
            contrail_summary: Source DataFrame with contrail observations.

        Returns:
            A dict mapping each timeline timestamp to a tuple of Contrail objects,
            with empty tuples for timestamps with no observations.
        """
        grouped = {
            pd.Timestamp(ts): tuple(self._row_to_contrail(row) for _, row in group.iterrows())
            for ts, group in contrail_summary.groupby("timestamp", sort=False)
        }
        # Fill in all timeline slots, empty if no data
        frames = {ts: grouped.get(ts, tuple()) for ts in self.timeline}
        return frames

    def _row_to_contrail(self: Self, row: pd.Series) -> Contrail:
        """Convert a single DataFrame row to a Contrail instance.

        Args:
            row: A pandas Series with contrail observation fields.

        Returns:
            A Contrail instance populated from the row values.
        """

        u_init: float | None = None
        v_init: float | None = None

        if self.config.use_wind_initialization:
            u_init = row.get("u_init")
            v_init = row.get("v_init")
            if u_init is None or v_init is None or np.isnan(u_init) or np.isnan(v_init):
                raise ValueError(
                    "Wind initialization is enabled but 'u_init' or 'v_init' is NaN or "
                    "missing in the DataFrame. Please provide these columns or disable "
                    "wind initialization in the config."
                )

        return Contrail(
            contrail_id=row["contrail_id"],
            endpoint1_lon=row["endpoint1_lon"],
            endpoint1_lat=row["endpoint1_lat"],
            endpoint2_lon=row["endpoint2_lon"],
            endpoint2_lat=row["endpoint2_lat"],
            u_init=u_init,
            v_init=v_init,
        )

    def _validate_uniform_frame_intervals(self: Self) -> None:
        """Raise ValueError if frame intervals are non-uniform or absent."""
        unique_intervals = np.unique(self.frame_intervals)
        if len(unique_intervals) > 1:
            raise ValueError(f"Non-uniform frame intervals detected: {unique_intervals}")
        elif len(unique_intervals) == 0:
            raise ValueError("Not enough frames to compute intervals.")
        else:
            logging.info(f"Uniform frame interval: {unique_intervals[0]}s")
