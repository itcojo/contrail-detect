"""Unit tests for contrail_detect.matching: track aggregation, candidate rules, presets."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPoint

from contrail_detect.matching.matching import Matcher, MatchingConfig
from contrail_detect.matching.matching_viz import display_match, plot_match
from contrail_detect.matching.presets import (
    MATCH_PRESETS,
    match_tracks,
    matched_mask,
    track_match_table,
)

T0 = pd.Timestamp("2025-01-01")


def _track_scores(
    n_flights: int, flight0_scores: list[float], other_score: float = 5.0
) -> pd.DataFrame:
    """One track, one row per (frame, candidate flight), sorted as Matcher sorts them."""
    rows = []
    for pos, s0 in enumerate(flight0_scores):
        for f in range(n_flights):
            rows.append(
                {
                    "track_id": "t0",
                    "flight_id": f"f{f}",
                    "contrail_id": f"c{pos}",
                    "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=10 * pos),
                    "rms_distance": s0 if f == 0 else other_score,
                }
            )
    return pd.DataFrame(rows).sort_values(["track_id", "timestamp"])


def _weighted_mean(df: pd.DataFrame, flight_id: str = "f0", decay: float = 0.5) -> float:
    agg = Matcher.aggregate_track_scores(
        df, "rms_distance", "weighted_mean", ["rms_distance"], decay
    )
    return float(agg.loc[agg["flight_id"] == flight_id, "rms_distance"].iloc[0])


class TestWeightedMeanAggregation:
    def test_weights_decay_per_frame(self):
        w = np.exp(-0.5 * np.arange(3))
        expected = (w * np.array([0.0, 10.0, 10.0])).sum() / w.sum()
        assert np.isclose(_weighted_mean(_track_scores(1, [0.0, 10.0, 10.0])), expected)

    def test_independent_of_candidate_count(self):
        one = _weighted_mean(_track_scores(1, [0.0, 10.0, 10.0]))
        five = _weighted_mean(_track_scores(5, [0.0, 10.0, 10.0]))
        assert np.isclose(one, five)

    def test_missing_or_nan_frame_uses_remaining_frames(self):
        w = np.exp(-0.5 * np.array([0, 2]))
        expected = (w * np.array([2.0, 6.0])).sum() / w.sum()
        df = _track_scores(3, [2.0, 4.0, 6.0])
        df = df[~((df["flight_id"] == "f0") & (df["contrail_id"] == "c1"))]
        assert np.isclose(_weighted_mean(df), expected)
        assert np.isclose(_weighted_mean(_track_scores(2, [2.0, np.nan, 6.0])), expected)

    def test_one_row_per_track_flight(self):
        agg = Matcher.aggregate_track_scores(
            _track_scores(4, [1.0, 2.0, 3.0]),
            "rms_distance",
            "weighted_mean",
            ["rms_distance"],
            0.5,
        )
        assert len(agg) == 4
        assert {"contrail_id", "timestamp"} <= set(agg.columns)


def _contrails(track_lengths: dict[str, int]) -> pd.DataFrame:
    """One contrail per frame, 10 min apart, for each track."""
    rows = [
        {
            "contrail_id": f"{t}_c{i}",
            "track_id": t,
            "timestamp": T0 + pd.Timedelta(minutes=10 * i),
        }
        for t, n in track_lengths.items()
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def _scores(
    contrails: pd.DataFrame, flight_scores: dict[str, list], status: str = "scored"
) -> pd.DataFrame:
    """Score rows per flight by frame position; None means no row for that frame."""
    rows = []
    for c in contrails.itertuples():
        pos = int(c.contrail_id.rsplit("_c", 1)[1])
        for flight, scores in flight_scores.items():
            if pos < len(scores) and scores[pos] is not None:
                rows.append(
                    {
                        "contrail_id": c.contrail_id,
                        "flight_id": flight,
                        "timestamp": c.timestamp,
                        "rms_distance": scores[pos],
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def _best_flight(contrails: pd.DataFrame, scores: pd.DataFrame, **config) -> object:
    out = Matcher().assign_matches_to_contrails(
        contrails, scores, MatchingConfig(**config), ranking_score="rms_distance"
    )
    if "t1_match" not in out.columns:  # no candidate survived; returned unmatched
        return np.nan
    return out.groupby("track_id")["t1_match"].first().iloc[0]


class TestMatchableCandidates:
    def test_short_track_not_matched(self):
        contrails = _contrails({"t0": 2})
        scores = _scores(contrails, {"f0": [1.0, 1.0]})
        assert pd.isna(_best_flight(contrails, scores))
        assert _best_flight(contrails, scores, min_track_frames=2) == "f0"
        assert pd.isna(_best_flight(contrails, scores, track_score_aggregation="first"))

    def test_incomplete_candidate_not_matched(self):
        contrails = _contrails({"t0": 3})
        missing = _scores(contrails, {"f0": [1.0, None, 1.0]})
        assert pd.isna(_best_flight(contrails, missing))
        assert pd.isna(_best_flight(contrails, _scores(contrails, {"f0": [1.0, np.nan, 1.0]})))
        assert pd.isna(_best_flight(contrails, missing, track_score_aggregation="best"))

    def test_complete_candidate_beats_incomplete_one(self):
        contrails = _contrails({"t0": 3})
        scores = _scores(contrails, {"f0": [1.0, None, 1.0], "f1": [2.0, 2.0, 2.0]})
        assert _best_flight(contrails, scores) == "f1"

    def test_long_track_needs_only_leading_frames(self):
        contrails = _contrails({"t0": 5})
        scores = _scores(contrails, {"f0": [1.0, 1.0, 1.0]})
        assert _best_flight(contrails, scores) == "f0"

    def test_first_mode_uses_first_frame_only(self):
        contrails = _contrails({"t0": 3})
        scores = _scores(contrails, {"f0": [1.0, None, None]})
        assert _best_flight(contrails, scores, track_score_aggregation="first") == "f0"
        scores = _scores(contrails, {"f0": [None, 1.0, 1.0], "f1": [2.0, 2.0, 2.0]})
        assert _best_flight(contrails, scores, track_score_aggregation="first") == "f1"

    def test_last_needs_only_last_scored_frame(self):
        contrails = _contrails({"t0": 5})
        scores = _scores(contrails, {"f0": [None, None, 1.0]})
        assert _best_flight(contrails, scores, track_score_aggregation="last") == "f0"

    def test_old_success_status_is_not_a_candidate(self):
        contrails = _contrails({"t0": 3})
        scores = _scores(contrails, {"f0": [1.0, 1.0, 1.0]}, status="success")
        assert pd.isna(_best_flight(contrails, scores))


class TestMatchingConfig:
    def test_matching_config(self):
        with pytest.raises(ValueError):
            MatchingConfig(min_track_frames=0)
        assert MatchingConfig().to_dict()["min_track_frames"] == 3
        assert MatchingConfig().use_span_intersections is True
        assert MatchingConfig().to_dict()["use_span_intersections"] is True


def _run(contrails: pd.DataFrame, scores: pd.DataFrame, **config) -> tuple:
    """Candidates and per-track status from one tracked assignment."""
    matcher = Matcher()
    out = matcher.assign_matches_to_contrails(
        contrails, scores, MatchingConfig(**config), ranking_score="rms_distance"
    )
    return matcher.candidates, out.groupby("track_id")["t_status"].first()


class TestStatuses:
    def test_candidate_statuses_in_pipeline_order(self):
        contrails = _contrails({"t0": 3, "t1": 2})
        scores = _scores(
            contrails,
            {"f0": [1.0, 1.0, 1.0], "f1": [1.0, None, 1.0], "f2": [9.0, 1.0, 1.0]},
        )
        candidates, _ = _run(contrails, scores, filters={"rms_distance": (None, 5.0)})
        status = candidates.set_index(["flight_id", "contrail_id"])["status"]
        assert status[("f0", "t0_c0")] == "eligible"
        assert status[("f1", "t0_c0")] == "missing_frame"
        assert status[("f2", "t0_c0")] == "outside_filter_bounds"
        assert status[("f2", "t0_c1")] == "missing_frame"
        assert status[("f0", "t1_c0")] == "track_too_short"

    def test_untracked_contrail_is_single_frame(self):
        contrails = pd.DataFrame([{"contrail_id": "u_c0", "track_id": np.nan, "timestamp": T0}])
        scores = _scores(contrails, {"f0": [1.0]})
        candidates, track_status = _run(contrails, scores)
        assert set(candidates["status"]) == {"single_frame"}
        assert track_status.iloc[0] == "single_frame"

    def test_public_helpers_match_assignment(self):
        contrails = _contrails({"ok": 3, "short": 2, "inc": 3})
        scores = _scores(contrails, {"f0": [1.0, 1.0, 1.0]})
        scores = scores[scores["contrail_id"] != "inc_c1"]
        matcher = Matcher()
        candidates = matcher.label_candidates(scores, contrails, "rms_distance")
        direct = matcher.track_status(scores, contrails, candidates)
        _, via_assignment = _run(contrails, scores)
        assert direct.to_dict() == via_assignment.to_dict()

    def test_track_statuses(self):
        contrails = _contrails({"ok": 3, "short": 2, "none": 3, "few": 3, "filt": 3, "inc": 3})

        def rows(track: str, flight_scores: dict, status: str = "scored") -> pd.DataFrame:
            return _scores(contrails[contrails["track_id"] == track], flight_scores, status)

        scores = pd.concat(
            [
                rows("ok", {"f0": [1.0, 1.0, 1.0]}),
                rows("short", {"f0": [1.0, 1.0]}),
                rows("few", {"f0": [1.0, 1.0, 1.0]}, status="too_few_points"),
                rows("filt", {"f0": [9.0, 9.0, 9.0]}),
                rows("inc", {"f0": [1.0, None, 1.0]}),
            ],
            ignore_index=True,
        )
        _, track_status = _run(contrails, scores, filters={"rms_distance": (None, 5.0)})
        assert track_status.to_dict() == {
            "ok": "ranked",
            "short": "track_too_short",
            "none": "no_flights_in_window",
            "few": "no_flight_with_enough_points",
            "filt": "all_outside_filter_bounds",
            "inc": "no_complete_flight",
        }


# ── Presets ───────────────────────────────────────────────────────────────────


def _matching_results() -> pd.DataFrame:
    """Two frames of track A (clear match) and one scored frame of track B (ambiguous)."""
    return pd.DataFrame(
        {
            "track_id": ["A", "A", "B"],
            "contrail_id": ["a0", "a1", "b0"],
            "timestamp": pd.to_datetime(
                ["2025-06-01T12:00", "2025-06-01T12:10", "2025-06-01T12:00"]
            ),
            "t1_rms_distance": [3.0, np.nan, 3.0],
            "t2_rms_distance": [9.0, np.nan, 4.0],
            "t1_first_formation_point_time": pd.to_datetime(
                ["2025-06-01T11:45", pd.NaT, "2025-06-01T11:50"]
            ),
            "t1_mean_formation_point_age_seconds": [720.0, np.nan, 480.0],
        }
    )


def test_track_match_table_collapses_per_track() -> None:
    track = track_match_table(_matching_results())
    assert track.loc["A", "t1"] == 3.0
    assert track.loc["A", "t2"] == 9.0
    assert track.loc["A", "ttfd_min"] == 15.0
    assert track.loc["B", "ttfd_min"] == 10.0
    assert track.loc["A", "mean_age_min"] == 12.0
    assert track.loc["B", "mean_age_min"] == 8.0


def test_matched_mask_applies_gap_and_age() -> None:
    track = track_match_table(_matching_results())
    mask = matched_mask(track, "default")
    assert mask["A"]
    assert not mask["B"]  # gap of 1 km is below the default 4.26 km
    matched = match_tracks(_matching_results(), preset="default")
    assert list(matched.index) == ["A"]


def test_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="preset must be one of"):
        match_tracks(_matching_results(), preset="loose")


def test_matched_mask_age_window() -> None:
    """The window bounds the mean age, so an older oldest end does not reject a track."""
    track = pd.DataFrame(
        {
            "t1": 1.0,
            "t2": np.nan,
            "mean_age_min": [5.0, 5.5, 20.0, 40.0],
            "ttfd_min": 90.0,
        }
    )
    assert matched_mask(track, "default").tolist() == [False, False, True, False]


def test_first_frame_ages_do_not_fall_through_to_later_frames() -> None:
    merged = pd.DataFrame(
        {
            "track_id": ["t", "t", "t", "t"],
            "flight_id": ["F", "G", "F", "G"],
            "timestamp": [T0, T0, T0 + pd.Timedelta(minutes=10), T0 + pd.Timedelta(minutes=10)],
            "mean_formation_point_age_seconds": [np.nan, 60.0, 300.0, 120.0],
        }
    )
    agg = merged.groupby(["track_id", "flight_id"]).first().reset_index()
    assert agg["mean_formation_point_age_seconds"].tolist() == [300.0, 60.0]
    ages = Matcher._first_frame_ages(merged, agg).set_index("flight_id")
    assert np.isnan(ages.loc["F", "mean_formation_point_age_seconds"])
    assert ages.loc["G", "mean_formation_point_age_seconds"] == 60.0


def test_presets_follow_ladder() -> None:
    """Score and TTFD cuts loosen from strict to relaxed; the gap cut tightens."""
    strict, default, relaxed = (MATCH_PRESETS[n] for n in ("strict", "default", "relaxed"))
    assert strict["score_cut"] < default["score_cut"] < relaxed["score_cut"]
    assert strict["ttfd_cut"] < default["ttfd_cut"] < relaxed["ttfd_cut"]
    assert strict["gap_cut"] > default["gap_cut"] > relaxed["gap_cut"]


# ── Static match figure ───────────────────────────────────────────────────────


def test_plot_match_draws_one_panel_per_frame(tmp_path) -> None:
    times = [T0 + pd.Timedelta(minutes=10 * i) for i in range(5)]
    results = pd.DataFrame(
        {
            "track_id": "t0",
            "contrail_id": [f"c{i}" for i in range(5)],
            "timestamp": times,
            "endpoint1_lon": -80.0,
            "endpoint1_lat": 30.0,
            "endpoint2_lon": -79.0,
            "endpoint2_lat": 30.5,
            "centroid_lon": -79.5,
            "centroid_lat": 30.25,
            "t1_match": "f0",
            "t1_rms_distance": [1.2, 1.2, 1.2, np.nan, np.nan],
        }
    )
    adv = pd.DataFrame(
        {
            "flight_id": "f0",
            "frame_time": np.repeat(times, 2),
            "original_time": T0,
            "lon_pc": [-80.0, -79.0] * 5,
            "lat_pc": [30.0, 30.5] * 5,
        }
    )
    clouds = {f"c{i}": MultiPoint([(-79.5, 30.25), (-79.4, 30.3)]) for i in range(5)}

    fig = plot_match("t0", "f0", results, adv, clouds, ncols=4, fig_output_dir=tmp_path)

    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titles) == 5
    assert sum("(scored)" in t for t in titles) == 3
    # The sixth slot of the 2 x 3 grid holds the legend
    assert len(fig.axes) == 6 and fig.axes[-1].get_legend() is not None
    assert "track RMS 1.20 km" in fig._suptitle.get_text()
    assert (tmp_path / "match-t0-f0.png").exists()


def test_unknown_track_raises() -> None:
    results = pd.DataFrame({"track_id": ["t0"], "timestamp": [T0]})
    with pytest.raises(ValueError, match="not in matching_results"):
        plot_match("missing", "f0", results, pd.DataFrame(), {})
    with pytest.raises(ValueError, match="not in matching_results"):
        display_match("missing", "f0", results, pd.DataFrame(), pd.DataFrame(), {})


def _wrap(lon):
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def _match_inputs(lon0: float):
    """One two-frame track near ``lon0`` with its flight, longitudes wrapped to [-180, 180)."""
    times = [T0, T0 + pd.Timedelta(minutes=10)]
    results = pd.DataFrame(
        {
            "track_id": "t0",
            "contrail_id": ["c0", "c1"],
            "timestamp": times,
            "endpoint1_lon": _wrap([lon0 - 0.2, lon0 - 0.15]),
            "endpoint1_lat": [40.0, 40.05],
            "centroid_lon": _wrap([lon0, lon0 + 0.05]),
            "centroid_lat": [40.1, 40.15],
            "endpoint2_lon": _wrap([lon0 + 0.2, lon0 + 0.25]),
            "endpoint2_lat": [40.2, 40.25],
            "t1_match": "f0",
            "t1_rms_distance": [3.2, np.nan],
        }
    )
    adv = pd.DataFrame(
        {
            "flight_id": "f0",
            "frame_time": np.repeat(times, 2),
            "original_time": [T0, T0 + pd.Timedelta(minutes=1)] * 2,
            "lon_pc": _wrap([lon0 - 0.3, lon0 + 0.3, lon0 - 0.25, lon0 + 0.35]),
            "lat_pc": [40.0, 40.2, 40.05, 40.25],
        }
    )
    flights = pd.DataFrame(
        {
            "flight_id": "f0",
            "time": [T0 - pd.Timedelta(minutes=20), T0 - pd.Timedelta(minutes=5)],
            "longitude": _wrap([lon0 - 0.4, lon0 + 0.4]),
            "latitude": [40.0, 40.3],
        }
    )
    clouds = {"c0": MultiPoint([(_wrap(lon0), 40.1)]), "c1": MultiPoint([(_wrap(lon0), 40.15)])}
    return results, adv, flights, clouds


def _map_xs(ax) -> list[np.ndarray]:
    """X coordinates of each drawn line in the map's own projection."""
    xs = []
    for line in ax.lines:
        path = line.get_transform().transform_path(line.get_path())
        xs.append(ax.transData.inverted().transform(path.vertices)[:, 0])
    return xs


def test_plot_match_across_date_line_stays_local() -> None:
    results, adv, flights, clouds = _match_inputs(179.95)
    fig = plot_match("t0", "f0", results, adv, clouds, flight_data=flights)
    ax = fig.axes[1]
    lon_min, lon_max, _, _ = ax.get_extent(ax.projection)
    assert lon_max - lon_min < 2.0  # not the whole globe
    # centre lines (earlier and current), the flight path and the advected flight
    assert len(ax.lines) == 4
    assert all(np.ptp(x) < 1.0 for x in _map_xs(ax))


def test_plot_match_away_from_date_line() -> None:
    results, adv, flights, clouds = _match_inputs(-75.0)
    fig = plot_match("t0", "f0", results, adv, clouds, flight_data=flights)
    ax = fig.axes[1]
    earlier, current = results.iloc[0], results.iloc[1]
    ends = ["endpoint1_lon", "centroid_lon", "endpoint2_lon"]
    drawn = [line.get_xdata() for line in ax.lines]
    np.testing.assert_array_equal(drawn[0], earlier[ends].to_numpy(dtype=float))
    np.testing.assert_array_equal(drawn[1], current[ends].to_numpy(dtype=float))
    np.testing.assert_array_equal(drawn[2], flights["longitude"])
    np.testing.assert_array_equal(drawn[3], adv["lon_pc"].iloc[2:])
    lon_min, lon_max, _, _ = ax.get_extent(ax.projection)
    assert (lon_min + lon_max) / 2 == pytest.approx(-74.975)


def test_plot_match_score_column_naming() -> None:
    results, adv, _, clouds = _match_inputs(-75.0)
    other = results.rename(columns={"t1_rms_distance": "t1_mean_distance"})
    fig = plot_match("t0", "f0", other, adv, clouds, score_col="mean_distance")
    assert fig.axes[0].get_title() == "00:00 UTC (scored)"
    assert fig._suptitle.get_text().endswith("track mean_distance 3.20")
    with pytest.raises(ValueError, match="score_col"):
        plot_match("t0", "f0", other, adv, clouds)
    single = results.rename(columns={"t1_match": "s1_match", "t1_rms_distance": "s1_rms_distance"})
    fig = plot_match("t0", "f0", single, adv, clouds)
    assert "track RMS 3.20 km" in fig._suptitle.get_text()


def _aged_scores(contrails: pd.DataFrame, formed: pd.Timestamp) -> pd.DataFrame:
    """Scores for two flights on the first three frames, with ages that follow from ``formed``."""
    rows = []
    for c in contrails.itertuples():
        if int(c.contrail_id.rsplit("_c", 1)[1]) >= 3:
            continue
        age = (c.timestamp - formed).total_seconds()
        for flight, score in (("f0", 1.0), ("f1", 9.0)):
            rows.append(
                {
                    "contrail_id": c.contrail_id,
                    "flight_id": flight,
                    "timestamp": c.timestamp,
                    "rms_distance": score,
                    "first_formation_point_time": formed,
                    "mean_implied_age_seconds": age,
                    "min_implied_age_seconds": age - 60,
                    "max_implied_age_seconds": age + 60,
                    "first_formation_point_age_seconds": age + 30,
                    "mean_formation_point_age_seconds": age,
                    "status": "scored",
                }
            )
    return pd.DataFrame(rows)


class TestFormationTimeAndAges:
    FORMED = T0 - pd.Timedelta(minutes=10)

    def _result(self) -> pd.DataFrame:
        contrails = _contrails({"t0": 5})
        out = Matcher().assign_matches_to_contrails(
            contrails,
            _aged_scores(contrails, self.FORMED),
            MatchingConfig(),
            ranking_score="rms_distance",
        )
        return out.sort_values("timestamp").reset_index(drop=True)

    def test_ages_grow_with_frame_time(self):
        out = self._result()
        assert (out["t1_implied_formation_timestamp"] == self.FORMED).all()
        true_age = (out["timestamp"] - self.FORMED).dt.total_seconds()
        np.testing.assert_allclose(out["t1_mean_implied_age_seconds"], true_age)
        np.testing.assert_allclose(out["t1_min_implied_age_seconds"], true_age - 60)
        np.testing.assert_allclose(out["t1_max_implied_age_seconds"], true_age + 60)
        np.testing.assert_allclose(out["t1_first_formation_point_age_seconds"], true_age + 30)
        np.testing.assert_allclose(out["t1_mean_formation_point_age_seconds"], true_age)

    def test_preset_age_from_first_frame(self):
        track = track_match_table(self._result())
        assert np.isclose(track["mean_age_min"].iloc[0], 10.0)


class TestScoresCsvRoundTrip:
    def test_ids_and_times_keep_their_types(self, tmp_path):
        scores = pd.DataFrame(
            {
                "contrail_id": ["0012", "0013"],
                "flight_id": ["001234", np.nan],
                "timestamp": [T0, T0],
                "first_formation_point_time": [T0 - pd.Timedelta(minutes=5), pd.NaT],
                "rms_distance": [1.0, np.nan],
                "status": ["scored", "no_flights_in_window"],
            }
        )
        matcher = Matcher()
        matcher.scores = scores
        matcher.save_scores(tmp_path / "s.csv", format="csv")
        back = matcher.load_scores(tmp_path / "s.csv")
        assert back["contrail_id"].tolist() == ["0012", "0013"]
        assert back["flight_id"].iloc[0] == "001234" and pd.isna(back["flight_id"].iloc[1])
        assert pd.api.types.is_datetime64_any_dtype(back["timestamp"])
        assert pd.api.types.is_datetime64_any_dtype(back["first_formation_point_time"])
        assert back["first_formation_point_time"].iloc[0] == T0 - pd.Timedelta(minutes=5)

    def test_matching_identical_after_round_trip(self, tmp_path):
        contrails = _contrails({"t0": 4})
        scores = _aged_scores(contrails, T0 - pd.Timedelta(minutes=10))
        matcher = Matcher()
        before = matcher.assign_matches_to_contrails(
            contrails, scores, MatchingConfig(), ranking_score="rms_distance"
        )
        matcher.scores = scores
        matcher.save_scores(tmp_path / "s.csv", format="csv")
        after = matcher.assign_matches_to_contrails(
            contrails,
            matcher.load_scores(tmp_path / "s.csv"),
            MatchingConfig(),
            ranking_score="rms_distance",
        )
        pd.testing.assert_frame_equal(before, after, check_dtype=False)
