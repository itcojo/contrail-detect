"""Classes for computing contrail-flight matching scores.

Provides _GCResult for bundling great-circle geometry results, and the Scorer
class which computes geometric scores based on perpendicular distances from
advected flight tracks to detected contrail lines.
"""

import logging
from collections.abc import Callable
from typing import Any, ClassVar, NamedTuple, Self

import numpy as np
import pandas as pd
from pycontrails.physics import constants

from contrail_detect.geostationary.geo_utils import convert_latlon_cartesian

# Scoring-stage row statuses, in pipeline order. Only "scored" rows carry scores.
SCORE_STATUSES = ("no_advected_flights", "no_flights_in_window", "too_few_points", "scored")


class _GCResult(NamedTuple):
    """Return type of Scorer._calculate_gc_perpendicular_distances.

    Attributes:
        perp_distances_km: (N,) perpendicular distances, NOT pre-filtered.
        within_span_mask: (N,) bool — inside loose span tolerance.
        within_tight_span_mask: (N,) bool — inside tight span tolerance.
        position: (N,) arc-length along contrail (m), signed; <0 before ep1,
            >seg_len after ep2.
        points_cart: (N, 3) unit Cartesian flight points.
        n_contrail: (3,) unit pole of contrail GC.
        ep1_cart: (3,) unit Cartesian of endpoint 1.
        ep2_cart: (3,) unit Cartesian of endpoint 2.
        seg_len: Contrail segment arc length (m).
    """

    perp_distances_km: np.ndarray
    within_span_mask: np.ndarray
    within_tight_span_mask: np.ndarray
    position: np.ndarray
    points_cart: np.ndarray
    n_contrail: np.ndarray
    ep1_cart: np.ndarray
    ep2_cart: np.ndarray
    seg_len: float


class Scorer:
    """Scoring system for contrail-flight matching.

    Computes geometric scores based on perpendicular distances from advected
    flight tracks to detected contrail lines.

    Each scoring method is an independent function (np.ndarray) -> float operating
    on the array of perpendicular distances for a single flight. Multiple scorers
    can be requested; the first is the primary scorer used for ranking (output column
    ``'score'``). Additional scorers produce supplementary columns named after the scorer.

    Built-in scoring methods: ``'chevallier'``, ``'mean_distance'``, ``'max_distance'``,
    ``'min_distance'``, ``'std_distance'``, ``'median_distance'``, ``'rms_distance'``.

    When use_span_intersections is True, the perpendicular distance array for each
    flight is augmented with up to 2 extra distances — the perpendicular distances
    from the span-gate intersection points to the contrail GC. Span gates are the
    great circles perpendicular to the contrail at each endpoint; their intersection
    with the flight's great circle captures the flight offset at the contrail boundaries.

    Attributes:
        scoring_methods: Ordered list of scoring methods (names or callables).
        use_span_intersections: Whether to augment distances with span-gate
            intersection points.

    Example:
        scorer = Scorer(scoring_methods=["rms_distance", "mean_distance"])
        # or with a custom function:
        scorer = Scorer(scoring_methods=["rms_distance", my_fn])
        # with span-gate intersection augmentation:
        scorer = Scorer(scoring_methods=["rms_distance"], use_span_intersections=True)
    """

    _SCORERS: ClassVar[dict[str, Callable[[np.ndarray], float]]] = {}

    def __init__(
        self: Self,
        scoring_methods: list[str | Callable[[np.ndarray], float]] | None = None,
        use_span_intersections: bool = False,
    ) -> None:
        """Initialize the Scorer.

        Args:
            scoring_methods: Ordered list of scoring methods. Each entry is either a
                registered name (str) or a callable (np.ndarray) -> float. The first
                entry is the primary scorer used for flight ranking. Its column is named
                after the scorer (registered name or fn.__name__). Defaults to
                ['rms_distance'].
            use_span_intersections: If True, augment each flight's perpendicular distance
                array with up to 2 extra distances: the perpendicular distances from the
                span-gate intersection points to the contrail GC. Span gates are great
                circles perpendicular to the contrail at each endpoint. Default: False.
        """
        if scoring_methods is None:
            scoring_methods = ["rms_distance"]
        self.scoring_methods = scoring_methods
        self.use_span_intersections = use_span_intersections
        self._validate_scoring_methods()

    def __repr__(self: Self) -> str:
        """Return string representation of the Scorer object."""
        return (
            f"Scorer(scoring_methods={self.scoring_methods!r}, "
            f"use_span_intersections={self.use_span_intersections!r})"
        )

    @property
    def primary_score_col(self: Self) -> str:
        """Column name produced by the primary (first) scorer."""
        method = self.scoring_methods[0]
        return method if isinstance(method, str) else method.__name__

    def compute_scores_per_flight(
        self: Self,
        advected_points: pd.DataFrame,
        endpoints: tuple[tuple[float, float], tuple[float, float]],
        min_points: int = 5,
        top_n: int | None = 10,
        buffer_deg: float = 0.5,
    ) -> dict[str, Any]:
        """Compute flight-contrail matching scores for each flight_id and return top N.

        Args:
            advected_points: DataFrame with advected flight points.
            endpoints: Contrail endpoints as [(lon1, lat1), (lon2, lat2)].
            min_points: Minimum number of points required for analysis.
            top_n: Number of top flights to return, ranked by the primary scorer.
                None returns every flight passing min_points.
            buffer_deg: Buffer in degrees around contrail bounding box.

        Returns:
            Dict keyed by flight_id. Scored flights come first (status ``'scored'``: one
            entry per scorer, flight metrics), best top_n by the primary scorer. Flights
            with fewer than min_points points follow (status ``'too_few_points'``: point
            counts only, no scores). Empty when no flight point falls in the window.
        """
        # Filter points to bounding box
        advected_points_filtered, scoring_area_km2 = self._filter_points_by_bounding_box(
            advected_points, endpoints, buffer_deg
        )

        if len(advected_points_filtered) == 0:
            return {}

        n_candidates_scored = advected_points_filtered["flight_id"].nunique()
        logging.debug(
            f"Scoring area: {scoring_area_km2:.0f} km², {n_candidates_scored} candidate flights"
        )

        # Resolve scorer functions and their output column names (done once per call)
        resolved_scorers = self._resolve_scorers()

        # Batch-compute GC distances for all filtered points in a single vectorized call
        gc_result = self._calculate_gc_perpendicular_distances(advected_points_filtered, endpoints)

        flight_ids_arr = advected_points_filtered["flight_id"].values
        unique_flight_ids = advected_points_filtered["flight_id"].unique()
        results = {}
        too_few = {}

        for flight_id in unique_flight_ids:
            if pd.isna(flight_id):
                continue

            flight_mask = flight_ids_arr == flight_id
            flight_within_span = gc_result.within_span_mask[flight_mask]
            flight_within_tight_span = gc_result.within_tight_span_mask[flight_mask]
            perp_distances_km = gc_result.perp_distances_km[flight_mask][flight_within_span]

            if self.use_span_intersections:
                intersection_dists = self._compute_span_gate_intersection_distances(
                    gc_result.points_cart[flight_mask],
                    gc_result.position[flight_mask],
                    gc_result.n_contrail,
                    gc_result.ep1_cart,
                    gc_result.ep2_cart,
                )
                if len(intersection_dists) > 0:
                    perp_distances_km = np.append(perp_distances_km, intersection_dists)

            if len(perp_distances_km) < min_points:
                too_few[flight_id] = {
                    "num_points": int(flight_mask.sum()),
                    "num_within_span": int(np.sum(flight_within_span)),
                    "n_candidates_scored": n_candidates_scored,
                    "status": "too_few_points",
                }
                continue

            # Apply all requested scorers
            score_metrics = {col: fn(perp_distances_km) for col, fn in resolved_scorers}

            # Extract additional flight metrics
            flight_points_df = advected_points_filtered[flight_mask]
            flight_metrics = self._extract_flight_metrics(
                flight_points_df, flight_within_span, flight_within_tight_span
            )

            results[flight_id] = {
                **score_metrics,
                **flight_metrics,
                "n_candidates_scored": n_candidates_scored,
                "scoring_area_km2": round(scoring_area_km2, 1),
                "status": "scored",
            }

        # Sort by primary score and keep top N, then the flights with too few points
        primary_col = resolved_scorers[0][0]
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1][primary_col])[:top_n])
        sorted_results.update(too_few)

        return sorted_results

    def score_all_contrails(
        self: Self,
        contrails_df: pd.DataFrame,
        advected_flights_df: pd.DataFrame,
        min_points: int = 5,
        top_n: int | None = 10,
        buffer_deg: float = 0.5,
    ) -> pd.DataFrame:
        """Score all contrails and return a DataFrame with results.

        Args:
            contrails_df: DataFrame with contrail geometries.
            advected_flights_df: DataFrame with advected flight points.
            min_points: Minimum number of points required for analysis.
            top_n: Number of top flights to return per contrail. None returns every
                flight passing min_points.
            buffer_deg: Buffer in degrees around contrail bounding box.

        Returns:
            DataFrame with one row per (contrail, flight in the search window), scored
            rows first, sorted by (contrail_id, primary score). ``status`` is one of
            SCORE_STATUSES: ``'no_advected_flights'`` and ``'no_flights_in_window'`` rows
            stand for the whole contrail (no flight_id), ``'too_few_points'`` rows have
            no scores, and ``'scored'`` rows carry one column per scorer and flight metric.
        """
        results = []

        # Group on parsed times without modifying the caller's DataFrames
        contrails_by_timestamp = contrails_df.groupby(
            pd.to_datetime(contrails_df["timestamp"]).rename("timestamp")
        )
        advected_by_frame = advected_flights_df.groupby(
            pd.to_datetime(advected_flights_df["frame_time"]).rename("frame_time")
        )

        total_timestamps = len(contrails_by_timestamp)
        total_contrails = len(contrails_df)
        matched_contrails = 0

        logging.info(
            f"Starting scoring for {total_contrails} contrails across {total_timestamps} timestamps"
        )
        logging.info(
            f"frame timestamps in advected flights: {list(advected_by_frame.groups.keys())}"
        )
        logging.info(f"timestamp in contrails: {list(contrails_by_timestamp.groups.keys())}")

        for timestamp_idx, (timestamp, contrails_group) in enumerate(contrails_by_timestamp, 1):
            logging.info(
                f"[{timestamp_idx}/{total_timestamps}] Processing "
                f"{len(contrails_group)} contrails at timestamp {timestamp}"
            )

            if timestamp not in advected_by_frame.groups:
                logging.warning(f"No advected flights found for timestamp {timestamp}")
                for _, contrail_row in contrails_group.iterrows():
                    results.append(
                        {
                            "contrail_id": contrail_row["contrail_id"],
                            "timestamp": timestamp,
                            "flight_id": np.nan,
                            "status": "no_advected_flights",
                        }
                    )
                continue

            df_adv_frame = advected_by_frame.get_group(timestamp)

            for _, (_, contrail_row) in enumerate(contrails_group.iterrows(), 1):
                contrail_id = contrail_row["contrail_id"]

                start_point = (
                    contrail_row["endpoint1_lon"],
                    contrail_row["endpoint1_lat"],
                )
                end_point = (
                    contrail_row["endpoint2_lon"],
                    contrail_row["endpoint2_lat"],
                )

                scoring_results = self.compute_scores_per_flight(
                    advected_points=df_adv_frame,
                    endpoints=(start_point, end_point),
                    min_points=min_points,
                    top_n=top_n,
                    buffer_deg=buffer_deg,
                )

                if not scoring_results:
                    results.append(
                        {
                            "contrail_id": contrail_id,
                            "timestamp": timestamp,
                            "flight_id": np.nan,
                            "status": "no_flights_in_window",
                        }
                    )
                    continue

                matched_contrails += 1

                for flight_id, metrics in scoring_results.items():
                    result_entry = {
                        "contrail_id": contrail_id,
                        "timestamp": timestamp,
                        "flight_id": flight_id,
                    }
                    result_entry.update(metrics)
                    results.append(result_entry)

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            # Sort only rows with actual candidates; sentinels have no score to sort on
            scored_mask = results_df["status"] == "scored"
            if scored_mask.any() and self.primary_score_col in results_df.columns:
                scored_sorted = results_df[scored_mask].sort_values(
                    by=["contrail_id", self.primary_score_col], ascending=True
                )
                results_df = pd.concat([scored_sorted, results_df[~scored_mask]], ignore_index=True)
        return results_df

    def _calculate_gc_perpendicular_distances(
        self: Self,
        flight_points_df: pd.DataFrame,
        endpoints: tuple[tuple[float, float], tuple[float, float]],
        tolerance_m: float = 20_000.0,
        tight_tolerance_m: float = 1_000.0,
    ) -> _GCResult:
        """Return a _GCResult for all input points using great-circle geometry.

        Endpoints are in degrees. All GC math is done entirely in Cartesian space
        to avoid lat/lon to Cartesian roundtrips. Arc distances use dot products
        (arccos) instead of haversine.

        Note: perp_distances_km is NOT pre-filtered by within_span_mask — callers must
        apply the mask themselves (e.g. perp_distances_km[within_span_mask]).

        Args:
            flight_points_df: DataFrame with ``'lon_pc'`` and ``'lat_pc'`` columns.
            endpoints: Contrail endpoints as [(lon1, lat1), (lon2, lat2)].
            tolerance_m: Loose span tolerance in meters.
            tight_tolerance_m: Tight span tolerance in meters.

        Returns:
            _GCResult namedtuple with distances, masks, and geometry arrays.
        """
        R = constants.radius_earth

        # 1. Endpoints to unit Cartesian vectors (scalar, computed once)
        ep1_cart = np.array(
            convert_latlon_cartesian(np.radians(endpoints[0][1]), np.radians(endpoints[0][0]))
        )  # (3,)
        ep2_cart = np.array(
            convert_latlon_cartesian(np.radians(endpoints[1][1]), np.radians(endpoints[1][0]))
        )  # (3,)

        # 2. GC pole n = normalize(ep1 × ep2) — computed once
        n = np.cross(ep1_cart, ep2_cart)
        n /= np.linalg.norm(n)

        # 3. All flight points to unit Cartesian (N, 3) — single trig pass
        lons_rad = np.radians(flight_points_df["lon_pc"].values)
        lats_rad = np.radians(flight_points_df["lat_pc"].values)
        x, y, z = convert_latlon_cartesian(lats_rad, lons_rad)
        points = np.stack([x, y, z], axis=1)  # (N, 3)

        # 4. Perpendicular distance: arcsin(|n · p|) × R  (avoids full projection)
        dot_n_p = points @ n  # (N,)
        perp_distances_m = np.arcsin(np.clip(np.abs(dot_n_p), 0.0, 1.0)) * R
        perp_distances_km = perp_distances_m / 1000.0

        # 5. Project each point onto the GC (Cartesian, no lat/lon roundtrip):
        #    pole_cross = normalize(p × n)  →  proj = n × pole_cross
        #    proj is already unit because n ⊥ pole_cross (both unit vectors)
        pole_cross = np.cross(points, n)  # (N, 3) — n broadcasts
        pole_cross_norms = np.sqrt(np.einsum("ij,ij->i", pole_cross, pole_cross))
        unit_pole_cross = pole_cross / pole_cross_norms[:, None]  # (N, 3)
        proj_cart = np.cross(n, unit_pole_cross)  # (N, 3) — unit vectors, n broadcasts

        # 6. Arc distances via dot product: arccos(a · b) × R for unit vectors
        seg_len = np.arccos(np.clip(ep1_cart @ ep2_cart, -1.0, 1.0)) * R  # scalar
        d_from_ep1 = np.arccos(np.clip(proj_cart @ ep1_cart, -1.0, 1.0)) * R  # (N,)
        d_from_ep2 = np.arccos(np.clip(proj_cart @ ep2_cart, -1.0, 1.0)) * R  # (N,)

        # 7. Arc-length position along segment (signed)
        sum_dist = d_from_ep1 + d_from_ep2
        # Points "between" ep1 and ep2 satisfy d_from_ep1 + d_from_ep2 ≈ seg_len
        pos_tol = 1000.0  # meters
        between = np.abs(sum_dist - seg_len) < pos_tol
        position = np.where(
            between,
            d_from_ep1,
            np.where(d_from_ep1 < d_from_ep2, -d_from_ep1, seg_len + d_from_ep2),
        )

        # 8. Span masks
        within_span_mask = (position >= -tolerance_m) & (position <= seg_len + tolerance_m)
        within_tight_span_mask = (position >= -tight_tolerance_m) & (
            position <= seg_len + tight_tolerance_m
        )

        return _GCResult(
            perp_distances_km=perp_distances_km,
            within_span_mask=within_span_mask,
            within_tight_span_mask=within_tight_span_mask,
            position=position,
            points_cart=points,
            n_contrail=n,
            ep1_cart=ep1_cart,
            ep2_cart=ep2_cart,
            seg_len=seg_len,
        )

    def _compute_span_gate_intersection_distances(
        self: Self,
        flight_points_cart: np.ndarray,
        position: np.ndarray,
        n_contrail: np.ndarray,
        ep1_cart: np.ndarray,
        ep2_cart: np.ndarray,
    ) -> np.ndarray:
        """Compute perpendicular distances from span-gate intersections to the contrail GC.

        Span gates are great circles perpendicular to the contrail at each endpoint.
        Their pole at endpoint ep_i is: n_gate_i = normalize(n_contrail x ep_i).
        The flight's great circle is defined by the two extremal flight points (by
        arc-length position). The intersection of the flight GC and each gate GC gives
        a point whose perpendicular distance to the contrail GC is returned.

        Args:
            flight_points_cart: (N, 3) unit Cartesian flight points.
            position: (N,) arc-length position along contrail (m).
            n_contrail: (3,) unit pole of contrail great circle.
            ep1_cart: (3,) unit Cartesian of endpoint 1.
            ep2_cart: (3,) unit Cartesian of endpoint 2.

        Returns:
            Array of 0, 1, or 2 distances (km). Returns empty array if the geometry
            is degenerate (fewer than 2 flight points, or near-collinear extremes).
        """
        R = constants.radius_earth

        if len(flight_points_cart) < 2:
            return np.array([])

        # Flight GC defined by the extremal points (min/max arc-length position)
        i_min = int(np.argmin(position))
        i_max = int(np.argmax(position))
        p_min = flight_points_cart[i_min]
        p_max = flight_points_cart[i_max]

        n_f = np.cross(p_min, p_max)
        norm_nf = np.linalg.norm(n_f)
        if norm_nf < 1e-10:
            # Points are (nearly) coincident or antipodal — degenerate flight GC
            return np.array([])
        n_f /= norm_nf

        distances_km = []
        for ep_cart in (ep1_cart, ep2_cart):
            # Span-gate pole: perpendicular to contrail GC at this endpoint
            n_gate = np.cross(n_contrail, ep_cart)
            norm_ng = np.linalg.norm(n_gate)
            if norm_ng < 1e-10:
                continue
            n_gate /= norm_ng

            # Intersection of flight GC and gate GC: ±normalize(n_f × n_gate)
            q = np.cross(n_f, n_gate)
            norm_q = np.linalg.norm(q)
            if norm_q < 1e-10:
                # Flight GC and gate GC share the same pole — degenerate
                continue
            q /= norm_q

            # Choose the sign whose hemisphere matches the endpoint
            if np.dot(q, ep_cart) < 0:
                q = -q

            # Perpendicular distance of intersection point q to the contrail GC
            d_km = np.arcsin(np.clip(np.abs(np.dot(n_contrail, q)), 0.0, 1.0)) * R / 1000.0
            distances_km.append(d_km)

        return np.array(distances_km)

    @staticmethod
    def _formation_ages(
        flight_points_df: pd.DataFrame, within_tight_span_mask: np.ndarray
    ) -> dict[str, float]:
        """Ages of the formation points inside the tight span, at the pixel scan time.

        A waypoint's age is its ``total_advected_time``: from the time the flight passed it
        (``original_time``) to the scan time of the frame it is advected to. Over the tight
        span (the points that also set ``first_formation_point_time``):

            first_formation_point_age_seconds  age of the oldest point, i.e. of the end of
                                               the contrail the flight passed first
            mean_formation_point_age_seconds   mean age of the points

        On the scan clock, unlike ``timestamp - first_formation_point_time``, which uses the
        frame's nominal time. NaN when no point is in the tight span.

        Args:
            flight_points_df: One flight's points, with ``total_advected_time``.
            within_tight_span_mask: Boolean mask for points within the tight span.

        Returns:
            Dict with the two ages in seconds.
        """
        if "total_advected_time" not in flight_points_df.columns or not np.any(
            within_tight_span_mask
        ):
            return {
                "first_formation_point_age_seconds": np.nan,
                "mean_formation_point_age_seconds": np.nan,
            }
        ages = flight_points_df["total_advected_time"].values[within_tight_span_mask]
        ages_s = ages / np.timedelta64(1, "s")
        return {
            "first_formation_point_age_seconds": float(np.nanmax(ages_s)),
            "mean_formation_point_age_seconds": float(np.nanmean(ages_s)),
        }

    def formation_ages_for_flight(
        self: Self,
        advected_points: pd.DataFrame,
        endpoints: tuple[tuple[float, float], tuple[float, float]],
        flight_id: str,
        buffer_deg: float = 0.5,
    ) -> dict[str, float]:
        """Formation-point ages of one flight for one contrail, as ``compute_scores_per_flight``.

        Same bounding box, great-circle geometry and tight span as the scorer, restricted to
        ``flight_id``, so the result equals the scorer's columns for that pair without scoring
        every candidate. For recomputing the ages on existing scores.

        Args:
            advected_points: Advected flight points for the contrail's frame (``lon_pc``,
                ``lat_pc``, ``flight_id``, ``total_advected_time``).
            endpoints: Contrail endpoints as [(lon1, lat1), (lon2, lat2)].
            flight_id: The flight to evaluate.
            buffer_deg: Bounding-box buffer, as ``compute_scores_per_flight``.

        Returns:
            Dict as ``_formation_ages``; NaN ages when the flight has no point in the box.
        """
        points = advected_points[advected_points["flight_id"] == flight_id]
        points, _ = self._filter_points_by_bounding_box(points, endpoints, buffer_deg)
        if len(points) == 0:
            return self._formation_ages(points, np.zeros(0, dtype=bool))
        gc_result = self._calculate_gc_perpendicular_distances(points, endpoints)
        return self._formation_ages(points, gc_result.within_tight_span_mask)

    def _extract_flight_metrics(
        self: Self,
        flight_points_df: pd.DataFrame,
        within_span_mask: np.ndarray,
        within_tight_span_mask: np.ndarray,
    ) -> dict[str, Any]:
        """Extract flight metrics like age, altitude, and formation times.

        Args:
            flight_points_df: DataFrame with flight points.
            within_span_mask: Boolean mask for points within the loose span.
            within_tight_span_mask: Boolean mask for points within the tight span.

        Returns:
            Dictionary with flight metrics.
        """
        metrics = {}

        # Calculate implied contrail age
        if "total_advected_time" in flight_points_df.columns:
            advected_times = flight_points_df["total_advected_time"].values[within_span_mask]
            mean_implied_age_nanosec = np.mean(advected_times)
            min_implied_age_nanosec = np.min(advected_times)
            max_implied_age_nanosec = np.max(advected_times)
            metrics["mean_implied_age_seconds"] = mean_implied_age_nanosec / np.timedelta64(1, "s")
            metrics["min_implied_age_seconds"] = min_implied_age_nanosec / np.timedelta64(1, "s")
            metrics["max_implied_age_seconds"] = max_implied_age_nanosec / np.timedelta64(1, "s")
        else:
            metrics["mean_implied_age_seconds"] = np.nan
            metrics["min_implied_age_seconds"] = np.nan
            metrics["max_implied_age_seconds"] = np.nan

        # Calculate mean altitude
        if "altitude_ft" in flight_points_df.columns:
            mean_altitude_ft = np.mean(flight_points_df["altitude_ft"].values[within_span_mask])
            metrics["implied_altitude_ft"] = np.round(mean_altitude_ft)
        else:
            metrics["implied_altitude_ft"] = np.nan

        # Extract formation point times
        if "original_time" in flight_points_df.columns and np.sum(within_tight_span_mask) > 0:
            times = flight_points_df["original_time"].values[within_tight_span_mask]
            metrics["first_formation_point_time"] = np.nanmin(times)
            metrics["last_formation_point_time"] = np.nanmax(times)
        else:
            metrics["first_formation_point_time"] = pd.NaT
            metrics["last_formation_point_time"] = pd.NaT

        metrics.update(self._formation_ages(flight_points_df, within_tight_span_mask))

        metrics["num_points"] = len(flight_points_df)
        metrics["num_within_span"] = np.sum(within_span_mask)

        return metrics

    def _filter_points_by_bounding_box(
        self: Self,
        advected_points: pd.DataFrame,
        endpoints: tuple[np.ndarray, np.ndarray],
        buffer_deg: float = 0.5,
    ) -> tuple[pd.DataFrame, float]:
        """Filter advected points to bounding box around contrail endpoints.

        Longitudes are compared modulo 360, so the box spans the shorter way between the
        endpoints, and endpoints and points may use either [-180, 180) or [0, 360).

        Args:
            advected_points: DataFrame with advected flight points.
            endpoints: Contrail endpoints as [(lon1, lat1), (lon2, lat2)].
            buffer_deg: Buffer in degrees around contrail bounding box.

        Returns:
            Tuple of (filtered DataFrame, bounding box area in km²).
        """
        point_A, point_B = endpoints

        # Put endpoint B within 180 degrees of endpoint A
        lon_a, lon_b = point_A[0], point_B[0]
        if abs(lon_b - lon_a) > 180.0:
            lon_b += 360.0 if lon_b < lon_a else -360.0
        lon_min = min(lon_a, lon_b) - buffer_deg
        lon_max = max(lon_a, lon_b) + buffer_deg
        lat_min = min(point_A[1], point_B[1]) - buffer_deg
        lat_max = max(point_A[1], point_B[1]) + buffer_deg

        lons = advected_points["lon_pc"]
        if -180.0 <= lon_min and lon_max <= 180.0:
            # Box clear of the date line, compared directly to stay exact at the edges
            lon_mask = (lons >= lon_min) & (lons <= lon_max)
        else:
            # Eastward offset of each point from the box's western edge, in [0, 360)
            lon_mask = (lons - lon_min) % 360.0 <= lon_max - lon_min
        bbox_mask = (
            lon_mask
            & (advected_points["lat_pc"] >= lat_min)
            & (advected_points["lat_pc"] <= lat_max)
        )

        area_m2 = (
            constants.radius_earth**2
            * abs(np.sin(np.radians(lat_max)) - np.sin(np.radians(lat_min)))
            * np.radians(lon_max - lon_min)
        )

        return advected_points[bbox_mask], round(area_m2 / 1e6, 1)

    def _resolve_scorers(
        self: Self,
    ) -> list[tuple[str, Callable[[np.ndarray], float]]]:
        """Resolve self.scoring_methods to (column_name, callable) pairs.

        Each scorer uses its registered name (str) or fn.__name__ (callable).
        The first entry is the primary scorer used for flight ranking.

        Returns:
            List of (column_name, fn) pairs in the order of scoring_methods.
        """
        resolved = []
        for method in self.scoring_methods:
            if callable(method):
                col_name = method.__name__
                fn = method
            else:
                col_name = method
                fn = self._SCORERS[method]
            resolved.append((col_name, fn))
        return resolved

    def _validate_scoring_methods(self: Self) -> None:
        """Validate that all entries in scoring_methods are registered names or callables."""
        for method in self.scoring_methods:
            if callable(method):
                continue
            if method not in self._SCORERS:
                raise ValueError(
                    f"Unknown scoring method: '{method}'. "
                    f"Registered methods: {list(self._SCORERS.keys())}"
                )

    @staticmethod
    def _chevallier_score(distances: np.ndarray) -> float:
        """Chevallier distance = max(distances)² / mean(distances)."""
        mean_dist = np.mean(distances)
        return float((np.max(distances) ** 2) / mean_dist) if mean_dist > 0 else 0.0

    @staticmethod
    def _max_distance_score(distances: np.ndarray) -> float:
        """Max perpendicular distance (Hausdorff)."""
        return float(np.max(distances))

    @staticmethod
    def _mean_distance_score(distances: np.ndarray) -> float:
        """Mean perpendicular distance."""
        return float(np.mean(distances))

    @staticmethod
    def _median_distance_score(distances: np.ndarray) -> float:
        """Median perpendicular distance."""
        return float(np.median(distances))

    @staticmethod
    def _min_distance_score(distances: np.ndarray) -> float:
        """Min perpendicular distance."""
        return float(np.min(distances))

    @staticmethod
    def _rms_distance_score(distances: np.ndarray) -> float:
        """Root-mean-square perpendicular distance."""
        return float(np.sqrt(np.mean(distances**2)))

    @staticmethod
    def _std_distance_score(distances: np.ndarray) -> float:
        """Standard deviation of perpendicular distances."""
        return float(np.std(distances))


# Populate the scorer registry after class definition
Scorer._SCORERS = {
    "chevallier": Scorer._chevallier_score,
    "mean_distance": Scorer._mean_distance_score,
    "max_distance": Scorer._max_distance_score,
    "min_distance": Scorer._min_distance_score,
    "std_distance": Scorer._std_distance_score,
    "median_distance": Scorer._median_distance_score,
    "rms_distance": Scorer._rms_distance_score,
}
