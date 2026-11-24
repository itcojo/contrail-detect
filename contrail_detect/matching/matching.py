"""
Classes for performing contrail-flight matching.

Provides MatchingConfig for bundling all matching parameters, and the Matcher
class which scores contrail-flight pairs and assigns ranked matches.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd

from contrail_detect.flights.flights_base import FlightData
from contrail_detect.geostationary.abc import Geostationary
from contrail_detect.matching.scoring import Scorer

# Structural / identity columns — everything else in a scores DataFrame is a metric.
_NON_METRIC_COLS = {"contrail_id", "flight_id", "timestamp", "rank", "status"}

# Scorer columns describing one frame, which a track reads at its first scored frame.
_FORMATION_AGE_COLS = ("first_formation_point_age_seconds", "mean_formation_point_age_seconds")

# Scores columns that Matcher.load_scores re-types after reading a CSV
_CSV_ID_COLS = ("contrail_id", "flight_id")
_CSV_TIME_COLS = ("timestamp", "first_formation_point_time", "last_formation_point_time")

# Matching-stage candidate statuses (label_candidates), in pipeline order.
CANDIDATE_STATUSES = (
    "outside_filter_bounds",
    "single_frame",
    "track_too_short",
    "missing_frame",
    "eligible",
)

# Track statuses ({prefix}_status in the matching results): the first stage a track failed.
TRACK_STATUSES = (
    "single_frame",
    "track_too_short",
    "no_flights_in_window",
    "no_flight_with_enough_points",
    "all_outside_filter_bounds",
    "no_complete_flight",
    "ranked",
)

# track_id prefix given to contrails the tracker did not link
_SINGLE_FRAME_PREFIX = "single_frame_"


@dataclass(frozen=True)
class MatchingConfig:
    """Configuration for the Matcher class.

    All parameters that control matching behavior are bundled here so that a single
    config object can be passed to ``run_matching`` / ``assign_matches_to_contrails`` and stored
    alongside results for reproducibility.

    Attributes:
        name: Human-readable label for this configuration.
        match_mode: ``'single'`` for per-frame matching, ``'tracked'`` for
            track-based matching. Only tracked mode is currently supported.
        scoring_methods: List/tuple of scoring method names (str) or callables
            ``(np.ndarray) -> float``. ``None`` delegates to the Scorer default
            (``['rms_distance']``). The first entry becomes the ranking column.
        use_span_intersections: Whether the Scorer adds the flight's crossings of the
            contrail's end gates to each flight's distances (default True, as in the
            match presets' calibration). See ``Scorer``.
        top_n: Number of ranked matches to attach per contrail / track (default 3).
            In tracked mode every flight is scored, and top_n applies only to the
            track ranking.
        filters: Optional dict of ``{column: (min_val, max_val)}`` bounds applied to
            the scores DataFrame before ranking. Either bound may be ``None`` to skip
            that direction, e.g.::

                filters={"min_implied_age_seconds": (300, 2400)}

        track_scoring_depth: Number of leading contrails per track to score and
            aggregate over (default 3). Tracks shorter than this value use all
            available contrails.
        track_score_aggregation: How to collapse multiple contrail-level scores into a
            single track-level score. One of
            ``'best'``, ``'first'``, ``'mean'``, ``'last'``, ``'weighted_mean'``,
            or a callable ``(group_df) -> scalar``.
            ``'weighted_mean'`` applies exponential decay weights (earlier frames
            weighted more heavily, since advection error grows over time).
        min_points: Minimum projected flight points required for a contrail to be
            scored (default 5).
        buffer_deg: Degrees of padding added around each contrail bounding box when
            searching for candidate flights (default 0.5).
        column_prefix: Output column prefix, e.g. ``'s'`` or ``'t'``. If ``None``,
            defaults to the first letter of ``match_mode``.
        weighted_mean_decay: Exponential decay rate per frame used by the
            ``'weighted_mean'`` aggregation. Weight for frame at position ``i``
            within the track is ``exp(-weighted_mean_decay * i)``, so position 0
            (earliest frame) always has weight 1.0. Higher values downweight later
            frames more aggressively. Default 0.5. Ignored for all other
            aggregation modes.
        min_track_frames: Minimum number of contrails a track needs to receive a
            track score and a match (default 3). Shorter tracks are still scored
            frame by frame but are left unmatched. A candidate flight receives a
            track score only if it has a valid score on every frame the aggregation
            uses: the first scored frame for ``'first'``, the last for ``'last'``,
            and all scored frames otherwise.
    """

    name: str = "default"
    match_mode: Literal["single", "tracked"] = "tracked"
    scoring_methods: tuple | None = None
    use_span_intersections: bool = True
    top_n: int = 3
    filters: dict[str, tuple[float | None, float | None]] | None = None
    track_scoring_depth: int = 3
    track_score_aggregation: str = "weighted_mean"
    min_points: int = 5
    buffer_deg: float = 0.5
    column_prefix: str | None = None
    weighted_mean_decay: float = 0.5
    min_track_frames: int = 3

    def __post_init__(self: Self) -> None:
        # Restrictions and warnings for invalid inputs
        if self.match_mode not in ("single", "tracked"):
            raise ValueError(f"match_mode must be 'single' or 'tracked', got {self.match_mode!r}")
        if self.track_score_aggregation not in ("best", "first", "mean", "last", "weighted_mean"):
            raise ValueError(
                "track_score_aggregation must be one of "
                "'best', 'first', 'mean', 'last', or 'weighted_mean'"
            )
        if self.min_track_frames < 1:
            raise ValueError(f"min_track_frames must be at least 1, got {self.min_track_frames}")

        # Restrictions on currently unimplemented features
        if self.match_mode == "single":
            raise NotImplementedError(
                "match_mode 'single' is not currently supported; please use "
                "tracked mode only for now"
            )

    def to_dict(self: Self) -> dict:
        """Return a plain-dict representation suitable for logging and serialization."""

        def _method_name(m):
            return m if isinstance(m, str) else getattr(m, "__name__", repr(m))

        return {
            "name": self.name,
            "match_mode": self.match_mode,
            "scoring_methods": (
                [_method_name(m) for m in self.scoring_methods]
                if self.scoring_methods is not None
                else None
            ),
            "use_span_intersections": self.use_span_intersections,
            "top_n": self.top_n,
            "filters": self.filters,
            "track_scoring_depth": self.track_scoring_depth,
            "track_score_aggregation": (
                self.track_score_aggregation
                if isinstance(self.track_score_aggregation, str)
                else getattr(
                    self.track_score_aggregation, "__name__", repr(self.track_score_aggregation)
                )
            ),
            "min_points": self.min_points,
            "buffer_deg": self.buffer_deg,
            "column_prefix": self.column_prefix,
            "weighted_mean_decay": self.weighted_mean_decay,
            "min_track_frames": self.min_track_frames,
        }


class Matcher:
    """Matches detected contrail segments to advected ADS-B flight tracks.

    Coordinates the full scoring and assignment pipeline: parallax correction,
    per-contrail scoring via a Scorer, and ranked match assignment. Supports
    single-frame and track-based matching modes via MatchingConfig.

    Attributes:
        detector: Geostationary satellite detector providing contrail data.
        flights: Flight data source providing advected ADS-B tracks.
        matching_results: Output DataFrame from the most recent ``run_matching`` call.
        scores: Raw scores DataFrame from the most recent scoring step.
        candidates: Labelled candidate rows (``label_candidates``) from the most recent
            tracked assignment.
    """

    def __init__(
        self: Self,
        detector: Geostationary = None,
        flights: FlightData = None,
    ) -> None:
        """Initialize the Matcher class.

        Args:
            detector: Optional geostationary satellite object.
            flights: Optional flight data object.
        """
        # Store inputs (optional for functional use)
        self.detector: Geostationary = detector
        self.flights: FlightData = flights

        self.matching_results: pd.DataFrame = None
        self.scores: pd.DataFrame = None
        self.candidates: pd.DataFrame = None

        # Initialize coordinate transformers if detector is provided
        if detector is not None:
            coords = self.detector.image_geo_coords
            self.geo_x, self.geo_y = coords

    def __repr__(self: Self) -> str:
        """Return string representation of the Matcher object."""
        detector = type(self.detector).__name__ if self.detector is not None else None
        flights = type(self.flights).__name__ if self.flights is not None else None
        return f"Matcher(detector={detector}, flights={flights})"

    @staticmethod
    def aggregate_track_scores(
        merged_df: pd.DataFrame,
        primary_score_col: str,
        track_score_aggregation: str | Callable,
        metric_cols: list[str],
        weighted_mean_decay: float = 0.5,
    ) -> pd.DataFrame:
        """Collapse multiple contrail-level score rows per (track_id, flight_id) into one.

        Registered aggregation modes:

        - ``'best'``: row with best (min) primary score per (track_id, flight_id)
        - ``'first'``: first row per group (earliest contrail in track)
        - ``'mean'``: first row for all columns, but primary score replaced with group mean
        - ``'last'``: last row per group
        - ``'weighted_mean'``: exponentially decayed mean — earlier frames (lower
          position index within the track) receive higher weight (settable decay rate per
          frame), reflecting that advection error accumulates over time
        - callable: fn(group_df) -> scalar used as primary score; other cols from first row

        Pass only ``'eligible'`` rows from ``label_candidates``, so every flight is
        aggregated over the same frames.

        Args:
            merged_df: Scores merged with track_id, sorted by (track_id, timestamp).
            primary_score_col: Column name used for ranking.
            track_score_aggregation: Aggregation strategy.
            metric_cols: List of metric column names (reserved for future use).
            weighted_mean_decay: Exponential decay rate per frame for ``'weighted_mean'``.

        Returns:
            One row per (track_id, flight_id).

        Raises:
            ValueError: If track_score_aggregation is not a recognized string or callable.
        """
        if track_score_aggregation == "best":
            idx = merged_df.groupby(["track_id", "flight_id"])[primary_score_col].idxmin()
            return merged_df.loc[idx].reset_index(drop=True)

        elif track_score_aggregation == "first":
            return merged_df.groupby(["track_id", "flight_id"]).first().reset_index()

        elif track_score_aggregation == "mean":
            first_rows = merged_df.groupby(["track_id", "flight_id"]).first()
            first_rows[primary_score_col] = (
                merged_df.groupby(["track_id", "flight_id"])[primary_score_col].mean().values
            )
            return first_rows.reset_index()

        elif track_score_aggregation == "last":
            return merged_df.groupby(["track_id", "flight_id"]).last().reset_index()

        elif track_score_aggregation == "weighted_mean":
            # Assign exponential decay weights by frame position within each track.
            # Position 0 (earliest frame) has weight 1.0; each subsequent frame is
            # multiplied by exp(-weighted_mean_decay), since advection error grows over time.
            # The position is the frame's rank by timestamp, not the row's: each frame
            # holds one row per candidate flight, so counting rows would decay by the
            # number of candidates per frame rather than by one per frame.
            frame_pos = merged_df.groupby("track_id")["timestamp"].rank(method="dense") - 1
            weight = np.exp(-weighted_mean_decay * frame_pos.to_numpy())
            score = merged_df[primary_score_col].to_numpy()
            valid = ~np.isnan(score)

            keys = [merged_df["track_id"], merged_df["flight_id"]]
            weighted_sum = pd.Series(np.where(valid, weight * score, 0.0), index=merged_df.index)
            weight_sum = pd.Series(np.where(valid, weight, 0.0), index=merged_df.index)
            first_rows = merged_df.groupby(["track_id", "flight_id"]).first()
            first_rows[primary_score_col] = (
                weighted_sum.groupby(keys).sum() / weight_sum.groupby(keys).sum()
            ).reindex(first_rows.index)

            return first_rows.reset_index()

        elif callable(track_score_aggregation):
            first_rows = merged_df.groupby(["track_id", "flight_id"]).first()
            agg_scores = merged_df.groupby(["track_id", "flight_id"]).apply(track_score_aggregation)
            first_rows[primary_score_col] = agg_scores.values
            return first_rows.reset_index()

        else:
            raise ValueError(
                f"Unknown track_score_aggregation '{track_score_aggregation}'. "
                "Use 'best', 'first', 'mean', 'last', 'weighted_mean', or a callable."
            )

    def assign_matches_to_contrails(
        self: Self,
        contrails_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        config: MatchingConfig | None = None,
        ranking_score: str | None = None,
    ) -> pd.DataFrame:
        """Assign flight matches to contrails based on scores.

        Args:
            contrails_df: DataFrame with contrail information (must have ``'contrail_id'``,
                ``'timestamp'``). For tracked mode, must also have ``'track_id'``.
            scores_df: DataFrame with flight scores (from ``score_all_contrails``).
            config: MatchingConfig controlling mode, filters, ranking, etc.
                Defaults to ``MatchingConfig()`` if None.
            ranking_score: Name of the score column used for ranking. If None, inferred
                as the first non-structural column in scores_df (with a warning).
                Explicitly provided by ``run_matching`` via ``scorer.primary_score_col``.

        Returns:
            DataFrame with contrails and assigned matches. In tracked mode it also
            carries ``{prefix}_status``, the first stage each track failed
            (TRACK_STATUSES), and the labelled candidates are kept in
            ``self.candidates``. If no eligible candidates remain, returns the contrails
            unmatched (with a warning logged) instead of raising.

        Raises:
            ValueError: If match_mode is not ``'single'`` or ``'tracked'``, or if
                ranking_score cannot be inferred from scores_df columns.
        """
        if config is None:
            config = MatchingConfig()

        if config.match_mode not in ["single", "tracked"]:
            raise ValueError("match_mode must be 'single' or 'tracked'")

        result_df = contrails_df.copy()

        # Resolve column prefix
        prefix = config.column_prefix or ("s" if config.match_mode == "single" else "t")

        # Infer ranking_score with warning if not provided
        if ranking_score is None:
            ranking_score = next((c for c in scores_df.columns if c not in _NON_METRIC_COLS), None)
            if ranking_score is None:
                raise ValueError("Cannot infer ranking_score from scores_df columns")
            logging.warning(
                f"ranking_score not specified; inferred as '{ranking_score}'. "
                "Pass ranking_score explicitly to suppress this warning."
            )

        # Label candidates and keep the eligible ones (tracked), or apply filters (single)
        if config.match_mode == "tracked":
            result_df = self._validate_and_prepare_tracking_metadata(
                self._fill_single_frame_tracks(result_df), config.track_scoring_depth
            )
            self.candidates = self.label_candidates(scores_df, result_df, ranking_score, config)
            track_status = self.track_status(scores_df, result_df, self.candidates, config)
            result_df[f"{prefix}_status"] = result_df["track_id"].map(track_status)
            eligible = self.candidates["status"] == "eligible"
            scores_filtered = self.candidates[eligible].drop(columns="track_id")
        else:
            scored = scores_df[scores_df["status"] == "scored"]
            scores_filtered = scored[self._filter_mask(scored, config.filters).to_numpy()]
        scores_filtered = scores_filtered.copy()

        if scores_filtered.empty or ranking_score not in scores_filtered.columns:
            logging.warning(
                f"No eligible flight candidates (ranking column '{ranking_score}'). "
                "Skipping rank assignment."
            )
            return result_df

        scores_filtered["rank"] = scores_filtered.groupby("contrail_id")[ranking_score].rank(
            method="first", ascending=True
        )

        # Assign matches for each rank
        for rank in range(1, config.top_n + 1):
            result_df = self._assign_rank_matches(
                result_df,
                scores_filtered,
                rank,
                prefix,
                config.match_mode,
                ranking_score,
                config.track_score_aggregation,
                config.weighted_mean_decay,
            )

        return result_df

    def label_candidates(
        self: Self,
        scores_df: pd.DataFrame,
        contrails_df: pd.DataFrame,
        primary_score_col: str,
        config: MatchingConfig | None = None,
    ) -> pd.DataFrame:
        """Label each scored candidate with the first matching rule it fails.

        The candidates are the ``'scored'`` rows on a scored frame of their track (its
        first ``track_scoring_depth`` contrails). The rules, in order:

        1. ``'outside_filter_bounds'``: outside a ``config.filters`` bound.
        2. ``'single_frame'`` / ``'track_too_short'``: the track has fewer than
           ``min_track_frames`` contrails; ``'single_frame'`` if the tracker never
           linked the contrail.
        3. ``'missing_frame'``: the flight has no valid row on a frame the aggregation
           uses (the first scored frame for ``'first'``, the last for ``'last'``, every
           scored frame otherwise), whatever removed that row.
        4. ``'eligible'``: enters the track ranking.

        Args:
            scores_df: Scorer output, with statuses from SCORE_STATUSES.
            contrails_df: Contrails with ``contrail_id``, ``timestamp`` and ``track_id``
                (NaN for contrails the tracker did not link).
            primary_score_col: Column name used for ranking.
            config: MatchingConfig; defaults to ``MatchingConfig()``.

        Returns:
            The candidate rows with ``track_id`` added and ``status`` replaced by one of
            CANDIDATE_STATUSES.
        """
        if config is None:
            config = MatchingConfig()
        contrails = self._validate_and_prepare_tracking_metadata(
            self._fill_single_frame_tracks(contrails_df), config.track_scoring_depth
        )
        on_scored_frames = contrails.loc[contrails["scored_in_track"], "contrail_id"]

        out = scores_df[
            (scores_df["status"] == "scored") & scores_df["contrail_id"].isin(on_scored_frames)
        ].copy()
        out["track_id"] = out["contrail_id"].map(contrails.set_index("contrail_id")["track_id"])
        status = pd.Series("eligible", index=out.index, dtype=object)

        # 1. Filter bounds
        status[~self._filter_mask(out, config.filters)] = "outside_filter_bounds"

        # 2. Track length
        n_frames = out["track_id"].map(contrails.groupby("track_id").size())
        short = (status == "eligible") & (n_frames < config.min_track_frames)
        untracked = out["track_id"].astype(str).str.startswith(_SINGLE_FRAME_PREFIX)
        status[short & untracked] = "single_frame"
        status[short & ~untracked] = "track_too_short"

        # 3. Frame coverage: a valid row on every frame the aggregation uses
        pos = contrails.groupby("track_id").cumcount()
        n_scored = contrails.groupby("track_id")["scored_in_track"].transform("sum")
        if config.track_score_aggregation == "first":
            required = pos == 0
        elif config.track_score_aggregation == "last":
            required = pos == n_scored - 1
        else:
            required = contrails["scored_in_track"]
        n_required = required.groupby(contrails["track_id"]).sum()

        score = out.get(primary_score_col, pd.Series(np.nan, index=out.index))
        valid = (
            (status == "eligible")
            & score.notna()
            & out["contrail_id"].isin(contrails.loc[required, "contrail_id"])
        )
        n_valid = (
            out[valid]
            .groupby(["track_id", "flight_id"])["contrail_id"]
            .nunique()
            .rename("_n_valid")
        )
        pairs = out[["track_id", "flight_id"]].join(n_valid, on=["track_id", "flight_id"])
        complete = pairs["_n_valid"] == out["track_id"].map(n_required)
        status[(status == "eligible") & ~complete] = "missing_frame"

        out["status"] = status
        logging.info(f"Candidate statuses: {out['status'].value_counts().to_dict()}")
        return out

    def load_scores(
        self: Self,
        path: str | Path,
        required_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load scores from disk into self.scores.

        Format is inferred from file extension: '.csv' -> CSV, anything else -> Parquet.
        From CSV, ID columns are read as strings and time columns as datetimes.

        Args:
            path: File path to read from.
            required_columns: Optional list of column names that must be present.

        Returns:
            Loaded DataFrame (also stored in self.scores).

        Raises:
            ValueError: If required_columns are missing from the loaded file.
        """
        path = Path(path)
        if path.suffix == ".csv":
            # Read IDs as strings, or all-digit IDs lose their leading zeros
            df = pd.read_csv(path, dtype={c: str for c in _CSV_ID_COLS})
            for col in _CSV_TIME_COLS:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        else:
            df = pd.read_parquet(path)

        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                raise ValueError(f"Loaded scores missing required columns: {missing}")

        self.scores = df
        logging.info(f"Loaded {len(df)} score rows from {path}")
        return df

    def run_matching(
        self: Self,
        config: MatchingConfig | None = None,
        skip_scoring: bool = False,
    ) -> None:
        """Run the complete contrail-flight matching process.

        Performs parallax correction on flight tracks, computes scores, and assigns
        matches to contrails using the provided config. Updates self.scores and
        self.matching_results with results.

        Args:
            config: MatchingConfig controlling all matching behavior. Defaults to
                ``MatchingConfig()`` (tracked matching with the rms_distance scorer,
                no filters).
            skip_scoring: If True, skip score computation and use self.scores directly.
                Useful for re-running the assignment step with different filters or
                aggregation without re-scoring. Call ``load_scores()`` first if no scores
                are pre-loaded.

        Raises:
            ValueError: If required data (contrails, flights, detector) is not set,
                or if skip_scoring=True but self.scores is None / missing columns.

        Example::

            # Basic single-frame matching
            matcher.run_matching()

            # Tracked matching with age filter and 2 contrails per track
            config = MatchingConfig(
                match_mode="tracked",
                scoring_methods=("rms_distance",),
                filters={"min_implied_age_seconds": (300, 2400)},
                track_scoring_depth=2,
                track_score_aggregation="best",
            )
            matcher.run_matching(config)

            # Save scores and reload later
            matcher.save_scores("scores.parquet")
            matcher.load_scores("scores.parquet")
            matcher.run_matching(config, skip_scoring=True)
        """
        if config is None:
            config = MatchingConfig()

        if self.flights is None:
            raise ValueError("FlightData must be set before running matching")

        if self.detector is None:
            raise ValueError("Geostationary detector must be set before running matching")

        if config.match_mode not in ["single", "tracked"]:
            raise ValueError(
                f"Invalid match_mode: '{config.match_mode}'. Must be 'single' or 'tracked'"
            )

        if config.match_mode == "tracked":
            if "track_id" not in self.detector.contrail_summary.columns:
                raise ValueError(
                    "track_contrail_id_mapping is empty. Run tracking first for tracked matching."
                )

        logging.info(f"Starting matching pipeline: {config.to_dict()}")

        # Parallax-correct flight tracks
        logging.info("Applying parallax correction to flight tracks...")
        self.flights.parallax_correct_advected_tracks(
            detector=self.detector,
            save_output=self.flights.save_path is not None,
        )

        # Contrails
        contrails_df = self.detector.contrail_summary

        # Assign synthetic single-frame tracks before any downstream code sees track_id,
        # so that _validate_and_prepare_tracking_metadata sets tracked=True for these rows
        if config.match_mode == "tracked":
            contrails_df = self._fill_single_frame_tracks(contrails_df)

        # Scorer (needed for primary_score_col even when skip_scoring)
        scorer = Scorer(
            scoring_methods=list(config.scoring_methods) if config.scoring_methods else None,
            use_span_intersections=config.use_span_intersections,
        )
        logging.info(f"Scorer methods: {scorer.scoring_methods}")

        # Compute or reuse scores
        if skip_scoring:
            if self.scores is None:
                raise ValueError("No pre-computed scores. Call load_scores() first.")
            required_str = [m for m in scorer.scoring_methods if isinstance(m, str)]
            missing = [c for c in required_str if c not in self.scores.columns]
            if missing:
                raise ValueError(f"Pre-computed scores missing columns needed by scorer: {missing}")
            scores_df = self.scores
            logging.info(f"Using {len(scores_df)} pre-computed score rows (skip_scoring=True)")
        else:
            if config.match_mode == "tracked":
                # Score first track_scoring_depth contrails per track
                tracked_only = contrails_df[contrails_df["track_id"].notna()].copy()
                sorted_df = tracked_only.sort_values(["track_id", "timestamp"])
                sorted_df["_pos"] = sorted_df.groupby("track_id").cumcount()
                track_max = sorted_df.groupby("track_id")["_pos"].transform("max")
                # clip ensures short tracks (length < track_scoring_depth) are included
                effective_max = track_max.clip(upper=config.track_scoring_depth - 1)
                contrails_to_score = sorted_df[sorted_df["_pos"] <= effective_max].drop(
                    columns=["_pos"]
                )
                logging.info(
                    f"Scoring {len(contrails_to_score)} contrails "
                    f"(track positions 1..{config.track_scoring_depth})..."
                )
                # Keep every flight: top_n applies to the track ranking, not to each frame
                scores_df = scorer.score_all_contrails(
                    contrails_df=contrails_to_score,
                    advected_flights_df=self.flights.adv_flight_data,
                    min_points=config.min_points,
                    top_n=None,
                    buffer_deg=config.buffer_deg,
                )
            else:
                logging.info(
                    f"Scoring all {len(contrails_df)} contrails for single-frame matching..."
                )
                scores_df = scorer.score_all_contrails(
                    contrails_df=contrails_df,
                    advected_flights_df=self.flights.adv_flight_data,
                    min_points=config.min_points,
                    top_n=config.top_n,
                    buffer_deg=config.buffer_deg,
                )

            self.scorer = scorer
            self.scores = scores_df
            logging.info(f"Scored {len(scores_df)} contrail-flight pairs")

        # Assign matches
        prefix = config.column_prefix or config.match_mode[0]
        logging.info(f"Assigning {config.match_mode} matches (prefix: '{prefix}')...")
        self.matching_results = self.assign_matches_to_contrails(
            contrails_df=contrails_df,
            scores_df=scores_df,
            config=config,
            ranking_score=scorer.primary_score_col,
        )

        logging.info("Matching complete.")

    def save_scores(
        self: Self,
        path: str | Path,
        format: str = "parquet",
    ) -> None:
        """Persist self.scores to disk.

        Args:
            path: File path to write to.
            format: ``'parquet'`` (default) or ``'csv'``.

        Raises:
            ValueError: If no scores have been computed yet.
        """
        if self.scores is None:
            raise ValueError("No scores to save. Run run_matching() first.")
        path = Path(path)
        if format == "csv":
            self.scores.to_csv(path, index=False)
        else:
            self.scores.to_parquet(path, index=False)
        logging.info(f"Saved {len(self.scores)} score rows to {path}")

    def summarize_match(self: Self, id: str) -> None:
        """Print a formatted summary table for a track or contrail match.

        Looks up ``id`` in ``matching_results`` as either a ``track_id`` or a
        ``contrail_id``. For a ``track_id``, the scored contrail row
        (``scored_in_track == True``) is shown. For a ``contrail_id``, the
        matching row is shown directly.

        Args:
            id: A ``track_id`` or ``contrail_id`` string.

        Raises:
            ValueError: If no matching results are available or ``id`` is not found.
        """
        if self.matching_results is None:
            raise ValueError("No matching results available. Run run_matching() first.")

        df = self.matching_results
        if id in df["track_id"].values:
            rows = df[df["track_id"] == id]
            row = rows[rows["scored_in_track"]]
            if row.empty:
                row = rows.iloc[[0]]
            row = row.iloc[0]
        elif id in df["contrail_id"].values:
            row = df[df["contrail_id"] == id].iloc[0]
        else:
            raise ValueError(f"ID '{id}' not found in matching_results as track_id or contrail_id.")

        # Infer the rank-1 prefix from whichever t1_*/s1_* match column exists
        if "t1_match" in df.columns:
            p = "t1"
        elif "s1_match" in df.columns:
            p = "s1"
        else:
            p = "t1"

        # Detect scorer columns: rank-1 columns that are not fixed metadata fields
        _fixed = {
            f"{p}_match",
            f"{p}_mean_implied_age_seconds",
            f"{p}_min_implied_age_seconds",
            f"{p}_max_implied_age_seconds",
            f"{p}_implied_altitude_ft",
            f"{p}_implied_formation_timestamp",
            f"{p}_first_formation_point_time",
            f"{p}_last_formation_point_time",
            f"{p}_num_points",
            f"{p}_num_within_span",
        }
        scorer_cols = [
            c
            for c in df.columns
            if c.startswith(f"{p}_") and c not in _fixed and pd.api.types.is_float_dtype(df[c])
        ]

        age = (
            f"{row[f'{p}_mean_implied_age_seconds'] / 60:.1f} min"
            if pd.notna(row.get(f"{p}_mean_implied_age_seconds"))
            else "—"
        )
        alt = (
            f"{row[f'{p}_implied_altitude_ft']:.0f} ft"
            if pd.notna(row.get(f"{p}_implied_altitude_ft"))
            else "—"
        )
        pts = (
            f"{row[f'{p}_num_within_span']:.0f} / {row[f'{p}_num_points']:.0f}"
            if pd.notna(row.get(f"{p}_num_within_span"))
            else "—"
        )

        _score_labels = {
            "chevallier": "Chevallier score (km²/km)",
            "mean_distance": "Mean distance (km)",
            "max_distance": "Max distance (km)",
            "min_distance": "Min distance (km)",
            "median_distance": "Median distance (km)",
            "rms_distance": "RMS distance (km)",
            "std_distance": "Std distance (km)",
        }
        score_entries = [
            (
                _score_labels.get(col.removeprefix(f"{p}_"), col.removeprefix(f"{p}_")),
                f"{row[col]:.4g}" if pd.notna(row.get(col)) else "—",
            )
            for col in scorer_cols
        ]

        entries = [
            ("Track ID", row.get("track_id", "—")),
            ("Contrail ID", row.get("contrail_id", "—")),
            ("Timestamp", str(row.get("timestamp", "—"))),
            ("──────────────────", "──────────────────────────────────────"),
            ("Rank-1 flight", str(row.get(f"{p}_match", "—"))),
            *score_entries,
            ("Mean implied age", age),
            ("Implied altitude", alt),
            ("Formation timestamp", str(row.get(f"{p}_implied_formation_timestamp", "—"))),
            ("First formation pt", str(row.get(f"{p}_first_formation_point_time", "—"))),
            ("Last formation pt", str(row.get(f"{p}_last_formation_point_time", "—"))),
            ("Points within span", pts),
        ]

        col_width = max(len(label) for label, _ in entries if "───" not in label)
        border = f"  {'─' * (col_width + 2 + 38)}"
        print(border)
        for label, value in entries:
            if "───" in label:
                print(f"  {'─' * col_width}  {value}")
            else:
                print(f"  {label:<{col_width}}  {value}")
        print(border)

    def track_status(
        self: Self,
        scores_df: pd.DataFrame,
        contrails_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        config: MatchingConfig | None = None,
    ) -> pd.Series:
        """First stage each track failed, as one of TRACK_STATUSES.

        Args:
            scores_df: Scorer output, with statuses from SCORE_STATUSES.
            contrails_df: Contrails with ``contrail_id``, ``timestamp`` and ``track_id``
                (NaN for contrails the tracker did not link).
            candidates_df: Output of ``label_candidates`` for the same inputs and config.
            config: MatchingConfig; defaults to ``MatchingConfig()``.

        Returns:
            Series indexed by track_id (``single_frame_<contrail_id>`` for contrails the
            tracker did not link).
        """
        if config is None:
            config = MatchingConfig()
        contrails = self._validate_and_prepare_tracking_metadata(
            self._fill_single_frame_tracks(contrails_df), config.track_scoring_depth
        )
        on_frames = scores_df.merge(
            contrails.loc[contrails["scored_in_track"], ["contrail_id", "track_id"]],
            on="contrail_id",
        )
        has_flight = on_frames.loc[
            on_frames["status"].isin(["scored", "too_few_points"]), "track_id"
        ]
        has_scored = on_frames.loc[on_frames["status"] == "scored", "track_id"]
        has_unfiltered = candidates_df.loc[
            candidates_df["status"] != "outside_filter_bounds", "track_id"
        ]
        has_eligible = candidates_df.loc[candidates_df["status"] == "eligible", "track_id"]

        n_frames = contrails.groupby("track_id").size()
        tracks = n_frames.index.to_series()
        short = n_frames < config.min_track_frames
        conditions = [
            short & tracks.astype(str).str.startswith(_SINGLE_FRAME_PREFIX),
            short,
            ~tracks.isin(has_flight),
            ~tracks.isin(has_scored),
            ~tracks.isin(has_unfiltered),
            ~tracks.isin(has_eligible),
        ]
        status = np.select(conditions, TRACK_STATUSES[:-1], default=TRACK_STATUSES[-1])
        return pd.Series(status, index=n_frames.index)

    def _assign_rank_matches(
        self: Self,
        contrails_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        rank: int,
        prefix: str,
        match_mode: str,
        primary_score_col: str,
        track_score_aggregation: str | Callable = "best",
        weighted_mean_decay: float = 0.5,
    ) -> pd.DataFrame:
        """Assign matches for a specific rank.

        Args:
            contrails_df: DataFrame with contrail information.
            scores_df: DataFrame with flight scores.
            rank: Rank to assign (1, 2, 3, etc.).
            prefix: Column prefix (e.g., ``'s'`` for single-frame, ``'t'`` for tracked).
            match_mode: ``'single'`` or ``'tracked'``.
            primary_score_col: Name of the primary score column.
            track_score_aggregation: Aggregation strategy for track-level scoring.
            weighted_mean_decay: Exponential decay rate per frame for ``'weighted_mean'``.

        Returns:
            DataFrame with rank matches assigned.
        """
        result_df = contrails_df.copy()

        if match_mode == "tracked":
            result_df = self._assign_tracked_rank(
                result_df,
                scores_df,
                rank,
                prefix,
                primary_score_col,
                track_score_aggregation,
                weighted_mean_decay,
            )
        elif match_mode == "single":
            result_df = self._assign_single_frame_rank(result_df, scores_df, rank, prefix)
        else:
            raise ValueError(f"Invalid match_mode: '{match_mode}'. Must be 'single' or 'tracked'")

        return result_df

    def _assign_single_frame_rank(
        self: Self,
        contrails_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        rank: int,
        prefix: str,
    ) -> pd.DataFrame:
        """Assign single-frame matches for a specific rank.

        Args:
            contrails_df: DataFrame with contrail information.
            scores_df: DataFrame with flight scores.
            rank: Rank to assign.
            prefix: Column prefix.

        Returns:
            DataFrame with rank matches assigned.
        """
        result_df = contrails_df.copy()

        # Discover metric columns dynamically
        metric_cols = [c for c in scores_df.columns if c not in _NON_METRIC_COLS]

        # Filter to this rank
        rank_scores = scores_df[scores_df["rank"] == rank].copy()

        if len(rank_scores) == 0:
            result_df[f"{prefix}{rank}_match"] = np.nan
            for metric in metric_cols:
                result_df[f"{prefix}{rank}_{metric}"] = np.nan
            result_df[f"{prefix}{rank}_implied_formation_timestamp"] = pd.NaT
            return result_df

        # Map flight ID
        flight_id_map = rank_scores.set_index("contrail_id")["flight_id"].to_dict()
        result_df[f"{prefix}{rank}_match"] = result_df["contrail_id"].map(flight_id_map)

        # Map all metrics (includes all scorers and flight metrics)
        for metric in metric_cols:
            if metric in rank_scores.columns:
                m_map = rank_scores.set_index("contrail_id")[metric].to_dict()
                result_df[f"{prefix}{rank}_{metric}"] = result_df["contrail_id"].map(m_map)
            else:
                result_df[f"{prefix}{rank}_{metric}"] = np.nan

        # Calculate implied formation timestamp
        age_col = f"{prefix}{rank}_mean_implied_age_seconds"
        if age_col in result_df.columns:
            result_df[f"{prefix}{rank}_implied_formation_timestamp"] = (
                pd.to_datetime(result_df["timestamp"])
                - pd.to_timedelta(result_df[age_col], unit="s")
            ).dt.round("s")
        else:
            result_df[f"{prefix}{rank}_implied_formation_timestamp"] = pd.NaT

        return result_df

    def _assign_tracked_rank(
        self: Self,
        contrails_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        rank: int,
        prefix: str,
        primary_score_col: str,
        track_score_aggregation: str | Callable = "best",
        weighted_mean_decay: float = 0.5,
    ) -> pd.DataFrame:
        """Assign tracked matches for a specific rank.

        Works on ALL rows in scores_df (already restricted to first N contrails per
        track by run_matching), aggregates per (track_id, flight_id) using
        track_score_aggregation, then re-ranks at the track level.

        Args:
            contrails_df: DataFrame with contrail information (must have scored_in_track).
            scores_df: DataFrame with flight scores (all scored contrails).
            rank: Track-level rank to assign (1 = best match, 2 = second best, etc.).
            prefix: Column prefix.
            primary_score_col: Name of the primary score column.
            track_score_aggregation: How to aggregate scores across contrails per track.
            weighted_mean_decay: Exponential decay rate per frame for ``'weighted_mean'``.

        Returns:
            DataFrame with rank matches assigned.
        """
        result_df = contrails_df.copy()

        # Discover metric columns dynamically
        metric_cols = [c for c in scores_df.columns if c not in _NON_METRIC_COLS]

        # Merge ALL scored contrail scores with track_id
        merged = scores_df.merge(
            contrails_df[["contrail_id", "track_id"]],
            on="contrail_id",
            how="left",
        )
        merged = merged.sort_values(by=["track_id", "timestamp"])

        # Aggregate across contrails per (track_id, flight_id)
        agg_df = self.aggregate_track_scores(
            merged,
            primary_score_col,
            track_score_aggregation,
            metric_cols,
            weighted_mean_decay,
        )

        if len(agg_df) == 0:
            result_df[f"{prefix}{rank}_match"] = np.nan
            for metric in metric_cols:
                result_df[f"{prefix}{rank}_{metric}"] = np.nan
            result_df[f"{prefix}{rank}_implied_formation_timestamp"] = pd.NaT
            return result_df

        agg_df = self._first_frame_ages(merged, agg_df)

        # Rank at track level (not contrail level)
        agg_df["_track_rank"] = agg_df.groupby("track_id")[primary_score_col].rank(
            method="first", ascending=True
        )

        # Pick requested rank
        rank_data = agg_df[agg_df["_track_rank"] == rank].drop(columns=["_track_rank"])

        # Build track-level lookup maps (keyed by track_id)
        flight_id_map = rank_data.set_index("track_id")["flight_id"].to_dict()
        metric_maps = {}
        for metric in metric_cols:
            if metric in rank_data.columns:
                metric_maps[metric] = rank_data.set_index("track_id")[metric].to_dict()

        # Map to all contrails in track
        result_df[f"{prefix}{rank}_match"] = result_df["track_id"].map(flight_id_map)

        for metric in metric_cols:
            if metric in metric_maps:
                result_df[f"{prefix}{rank}_{metric}"] = result_df["track_id"].map(
                    metric_maps[metric]
                )
            else:
                result_df[f"{prefix}{rank}_{metric}"] = np.nan

        # Zero out per-frame metrics for contrails not included in scoring
        first_only_metrics = [
            m
            for m in metric_cols
            if any(kw in m for kw in ("distance", "age", "altitude", "formation", "num_"))
        ]
        scored_mask = result_df.get("scored_in_track", pd.Series(True, index=result_df.index))
        for metric in first_only_metrics:
            col = f"{prefix}{rank}_{metric}"
            if col in result_df.columns:
                result_df.loc[~scored_mask, col] = np.nan

        # Zero out primary score for non-scored contrails
        primary_col = f"{prefix}{rank}_{primary_score_col}"
        if primary_col in result_df.columns:
            result_df.loc[~scored_mask, primary_col] = np.nan

        # Calculate formation timestamp
        return self._calculate_formation_timestamp(result_df, rank, prefix)

    def _calculate_formation_timestamp(
        self: Self,
        contrails_df: pd.DataFrame,
        rank: int,
        prefix: str,
    ) -> pd.DataFrame:
        """Set one formation timestamp per track and each frame's ages from it.

        The formation timestamp is the first frame's time minus its mean implied age.
        Each age column is the first frame's value plus the time since that frame.

        Args:
            contrails_df: DataFrame with contrail information.
            rank: Rank being processed.
            prefix: Column prefix.

        Returns:
            DataFrame with formation timestamps and per-frame ages.
        """
        result_df = contrails_df.copy()

        age_col = f"{prefix}{rank}_mean_implied_age_seconds"
        ts_col = f"{prefix}{rank}_implied_formation_timestamp"

        if age_col not in result_df.columns:
            result_df[ts_col] = pd.NaT
            return result_df

        ts = pd.to_datetime(result_df["timestamp"])
        tracks = result_df["track_id"]
        first_ts = ts.groupby(tracks).transform("min")
        on_first_frame = ts == first_ts
        since_first_s = (ts - first_ts).dt.total_seconds()

        def first_frame_value(col: str) -> pd.Series:
            return result_df[col].where(on_first_frame).groupby(tracks).transform("first")

        # One formation timestamp per track, from its first frame
        result_df[ts_col] = (
            first_ts - pd.to_timedelta(first_frame_value(age_col), unit="s")
        ).dt.round("s")
        result_df[age_col] = (ts - result_df[ts_col]).dt.total_seconds()

        # Other ages advance from their first-frame values in the same way
        other_age_cols = [
            f"{prefix}{rank}_{c}"
            for c in (
                "min_implied_age_seconds",
                "max_implied_age_seconds",
                *_FORMATION_AGE_COLS,
            )
        ]
        for col in other_age_cols:
            if col in result_df.columns:
                result_df[col] = first_frame_value(col) + since_first_s

        return result_df

    def _validate_and_prepare_tracking_metadata(
        self: Self,
        contrails_df: pd.DataFrame,
        track_scoring_depth: int = 1,
    ) -> pd.DataFrame:
        """Validate track_id exists and add tracking metadata columns.

        Assumes tracking has already been performed and track_id column exists.
        Adds convenience columns: tracked, first_in_track, scored_in_track.

        Args:
            contrails_df: DataFrame with contrail information (must include track_id).
            track_scoring_depth: Number of leading contrails per track included in
                scoring. scored_in_track is True for contrails at positions
                0..track_scoring_depth-1.

        Returns:
            DataFrame with tracking metadata added.

        Raises:
            ValueError: If track_id column is missing.
        """
        if "track_id" not in contrails_df.columns:
            raise ValueError(
                "track_id column must be present in contrails_df for tracked matching. "
                "Run tracking first or use match_mode='single'."
            )

        result_df = contrails_df.copy()

        # Add tracking metadata
        result_df["tracked"] = result_df["track_id"].notna()
        result_df = result_df.sort_values(by=["track_id", "timestamp"])

        # Identify first contrail in each track (kept for downstream compatibility)
        result_df["first_in_track"] = result_df["track_id"].shift(1) != result_df["track_id"]
        result_df["first_in_track"] = result_df["first_in_track"].fillna(True)

        # Track length (number of contrails per track)
        result_df["track_length"] = result_df.groupby("track_id")["contrail_id"].transform("count")

        # Identify contrails included in scoring (positions 0..track_scoring_depth-1)
        result_df["scored_in_track"] = False
        if result_df["tracked"].any():
            tracked_mask = result_df["tracked"]
            track_pos = pd.Series(0, index=result_df.index, dtype=int)
            track_pos[tracked_mask] = result_df[tracked_mask].groupby("track_id").cumcount().values
            result_df.loc[tracked_mask, "scored_in_track"] = (
                track_pos[tracked_mask] < track_scoring_depth
            )

        return result_df

    @staticmethod
    def _fill_single_frame_tracks(contrails_df: pd.DataFrame) -> pd.DataFrame:
        """Give each contrail the tracker did not link its own single-frame track_id."""
        if "track_id" not in contrails_df.columns:
            return contrails_df
        untracked = contrails_df["track_id"].isna()
        if not untracked.any():
            return contrails_df
        logging.info(f"Assigning single-frame track_ids to {int(untracked.sum())} contrails")
        out = contrails_df.copy()
        out["track_id"] = out["track_id"].astype(object)
        out.loc[untracked, "track_id"] = _SINGLE_FRAME_PREFIX + out.loc[
            untracked, "contrail_id"
        ].astype(str)
        return out

    @staticmethod
    def _filter_mask(
        scores_df: pd.DataFrame,
        filters: dict[str, tuple[float | None, float | None]] | None,
    ) -> pd.Series:
        """True where a row lies inside every (min, max) bound in filters.

        Either bound may be None to skip that direction. Unknown columns are warned
        and skipped.

        Args:
            scores_df: DataFrame of scores.
            filters: Dict of {column: (min_val, max_val)}. None passes every row.

        Returns:
            Boolean Series aligned to scores_df.
        """
        mask = pd.Series(True, index=scores_df.index)
        for col, (min_val, max_val) in (filters or {}).items():
            if col not in scores_df.columns:
                logging.warning(f"Filter column '{col}' not found in scores, skipping")
                continue
            if min_val is not None:
                mask &= scores_df[col] >= min_val
            if max_val is not None:
                mask &= scores_df[col] <= max_val
        return mask

    @staticmethod
    def _first_frame_ages(merged_df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
        """Read the formation ages at each track's first frame.

        ``aggregate_track_scores`` carries non-score columns through ``groupby.first``, which
        falls through to a later frame's value where the first frame has none. An age belongs
        to its frame, so such a pair stays NaN here.

        Args:
            merged_df: Scores merged with track_id, one row per frame and flight.
            agg_df: The result of ``aggregate_track_scores`` on ``merged_df``.

        Returns:
            ``agg_df`` with its formation-age columns, where present, taken from the first frame.
        """
        age_cols = [c for c in _FORMATION_AGE_COLS if c in agg_df.columns]
        if not age_cols:
            return agg_df
        first_ts = merged_df.groupby("track_id")["timestamp"].transform("min")
        first_frame = merged_df[merged_df["timestamp"] == first_ts]
        ages = first_frame.groupby(["track_id", "flight_id"])[age_cols].first()
        pairs = pd.MultiIndex.from_frame(agg_df[["track_id", "flight_id"]])
        out = agg_df.copy()
        out[age_cols] = ages.reindex(pairs).to_numpy()
        return out
