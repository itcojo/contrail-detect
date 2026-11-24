import numpy as np
import pandas as pd
import pytest

from contrail_detect.matching.scoring import Scorer


class TestScorerInit:
    def test_scoring_methods_and_primary_column(self):
        def my_scorer(x: np.ndarray) -> float:
            return float(np.sum(x))

        assert Scorer().scoring_methods == ["rms_distance"]
        assert len(Scorer(["mean_distance", "max_distance"]).scoring_methods) == 2
        assert Scorer([my_scorer]).scoring_methods == [my_scorer]
        assert len(Scorer(["chevallier", my_scorer]).scoring_methods) == 2
        assert Scorer(["mean_distance"]).primary_score_col == "mean_distance"
        assert Scorer([my_scorer]).primary_score_col == "my_scorer"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown scoring method"):
            Scorer(["not_a_real_method"])


class TestScoringStaticMethods:
    def test_score_functions(self):
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert Scorer._chevallier_score(d) == pytest.approx((5.0**2) / np.mean(d))
        assert Scorer._chevallier_score(np.array([0.0, 0.0])) == 0.0
        assert Scorer._mean_distance_score(d) == pytest.approx(3.0)
        assert Scorer._max_distance_score(d) == pytest.approx(5.0)
        assert Scorer._min_distance_score(d) == pytest.approx(1.0)
        assert Scorer._median_distance_score(d) == pytest.approx(3.0)
        assert Scorer._rms_distance_score(d) == pytest.approx(float(np.sqrt(np.mean(d**2))))
        assert Scorer._std_distance_score(d) == pytest.approx(float(np.std(d)))


# ── Scoring geometry ──────────────────────────────────────────────────────────

# A contrail along the equator from 0°E to 1°E. Flights parallel to it sit at a known
# perpendicular distance: 0.1° of latitude is 11.12 km.
CONTRAIL = ((0.0, 0.0), (1.0, 0.0))
KM_PER_DEG = np.radians(1.0) * 6371.229
T0 = pd.Timestamp("2026-01-09T15:00")


def _flight(flight_id: str, lat: float, lons, **extra) -> pd.DataFrame:
    lons = np.asarray(lons, dtype=float)
    return pd.DataFrame(
        {"flight_id": flight_id, "lon_pc": lons, "lat_pc": lat, "frame_time": T0, **extra}
    )


def _flights() -> pd.DataFrame:
    return pd.concat(
        [
            _flight("near", 0.1, np.linspace(0.0, 1.0, 11)),
            _flight("far", 0.3, np.linspace(0.0, 1.0, 11)),
            _flight("sparse", 0.2, [0.4, 0.5, 0.6]),
            _flight("away", 5.0, np.linspace(0.0, 1.0, 11)),  # outside the search window
        ],
        ignore_index=True,
    )


class TestComputeScoresPerFlight:
    def test_parallel_flight_scores_its_offset(self):
        scores = Scorer(["rms_distance", "max_distance"]).compute_scores_per_flight(
            _flights(), CONTRAIL
        )
        assert scores["near"]["rms_distance"] == pytest.approx(0.1 * KM_PER_DEG, rel=1e-3)
        assert scores["far"]["max_distance"] == pytest.approx(0.3 * KM_PER_DEG, rel=1e-3)

    def test_ranked_by_primary_score_then_too_few_points(self):
        scores = Scorer().compute_scores_per_flight(_flights(), CONTRAIL)
        assert list(scores) == ["near", "far", "sparse"]
        assert [s["status"] for s in scores.values()] == ["scored", "scored", "too_few_points"]
        assert "rms_distance" not in scores["sparse"]
        assert scores["near"]["n_candidates_scored"] == 3
        top = Scorer().compute_scores_per_flight(_flights(), CONTRAIL, top_n=1)
        assert list(top) == ["near", "sparse"]

    def test_points_beyond_the_span_are_not_scored(self):
        # Two extra points 0.4° past the contrail's end, far off the line
        flights = pd.concat(
            [_flight("near", 0.1, np.linspace(0, 1, 11)), _flight("near", 0.5, [1.4, 1.4])]
        )
        scores = Scorer().compute_scores_per_flight(flights, CONTRAIL)
        assert scores["near"]["num_points"] == 13
        assert scores["near"]["num_within_span"] == 11
        assert scores["near"]["rms_distance"] == pytest.approx(0.1 * KM_PER_DEG, rel=1e-3)

    def test_no_points_in_window_returns_empty(self):
        assert Scorer().compute_scores_per_flight(_flight("away", 5.0, [0.5] * 10), CONTRAIL) == {}

    def test_span_intersections_add_gate_distances(self):
        flight = _flight("short", 0.1, [0.3, 0.4, 0.5, 0.6])  # 4 points, below min_points
        assert Scorer().compute_scores_per_flight(flight, CONTRAIL)["short"]["status"] == (
            "too_few_points"
        )
        scores = Scorer(use_span_intersections=True).compute_scores_per_flight(flight, CONTRAIL)
        # The gate crossings of a parallel flight lie at the same offset
        assert scores["short"]["status"] == "scored"
        assert scores["short"]["rms_distance"] == pytest.approx(0.1 * KM_PER_DEG, rel=1e-3)

    def test_flight_metrics(self):
        lons = np.linspace(0.0, 1.0, 11)
        flight = _flight(
            "near",
            0.1,
            lons,
            altitude_ft=np.linspace(34000, 36000, 11),
            total_advected_time=pd.to_timedelta(np.arange(11) * 60, unit="s"),
            original_time=T0 - pd.to_timedelta(np.arange(11) * 60, unit="s"),
        )
        metrics = Scorer().compute_scores_per_flight(flight, CONTRAIL)["near"]
        assert metrics["implied_altitude_ft"] == 35000
        assert metrics["mean_implied_age_seconds"] == pytest.approx(300.0)
        assert (metrics["min_implied_age_seconds"], metrics["max_implied_age_seconds"]) == (0, 600)
        assert metrics["first_formation_point_time"] == T0 - pd.Timedelta(minutes=10)
        assert metrics["last_formation_point_time"] == T0
        assert metrics["first_formation_point_age_seconds"] == pytest.approx(600.0)
        assert metrics["mean_formation_point_age_seconds"] == pytest.approx(300.0)

    def test_formation_ages_exclude_points_beyond_the_tight_span(self):
        # Two old points ~11 and ~17 km past the contrail's end: inside the loose span (20 km),
        # so they enter mean_implied_age_seconds, but outside the tight span (1 km).
        lons = np.concatenate([np.linspace(0.0, 1.0, 11), [1.1, 1.15]])
        ages = pd.to_timedelta(np.concatenate([np.arange(11) * 60, [3600, 3600]]), unit="s")
        flight = _flight("near", 0.1, lons, total_advected_time=ages)
        metrics = Scorer().compute_scores_per_flight(flight, CONTRAIL)["near"]
        assert metrics["max_implied_age_seconds"] == pytest.approx(3600.0)
        assert metrics["first_formation_point_age_seconds"] == pytest.approx(600.0)
        assert metrics["mean_formation_point_age_seconds"] == pytest.approx(300.0)

    def test_formation_ages_for_flight_matches_scorer(self):
        ages = pd.to_timedelta(np.arange(11) * 45, unit="s")
        flights = pd.concat(
            [
                _flight("near", 0.1, np.linspace(0.0, 1.0, 11), total_advected_time=ages),
                _flight("other", 0.2, np.linspace(-0.1, 0.9, 11), total_advected_time=ages * 2),
            ]
        )
        scorer = Scorer(use_span_intersections=True)
        scores = scorer.compute_scores_per_flight(flights, CONTRAIL)
        for fid in ("near", "other"):
            ages_one = scorer.formation_ages_for_flight(flights, CONTRAIL, fid)
            for key, value in ages_one.items():
                assert value == pytest.approx(scores[fid][key])
        ages = Scorer().formation_ages_for_flight(_flights(), CONTRAIL, "no_such_flight")
        assert np.isnan(ages["first_formation_point_age_seconds"])
        assert np.isnan(ages["mean_formation_point_age_seconds"])


class TestScoreAllContrails:
    def test_statuses_and_ordering(self):
        contrails = pd.DataFrame(
            {
                "contrail_id": ["c_far_away", "c_main", "c_no_frame"],
                "timestamp": [T0, T0, T0 + pd.Timedelta(minutes=10)],
                "endpoint1_lon": [50.0, 0.0, 0.0],
                "endpoint1_lat": [50.0, 0.0, 0.0],
                "endpoint2_lon": [51.0, 1.0, 1.0],
                "endpoint2_lat": [50.0, 0.0, 0.0],
            }
        )
        out = Scorer().score_all_contrails(contrails, _flights())

        by_contrail = out.groupby("contrail_id")["status"].apply(list).to_dict()
        assert by_contrail == {
            "c_far_away": ["no_flights_in_window"],
            "c_main": ["scored", "scored", "too_few_points"],
            "c_no_frame": ["no_advected_flights"],
        }
        assert list(out["flight_id"][:2]) == ["near", "far"]  # scored rows first, best first

    def test_inputs_not_modified(self):
        contrails = pd.DataFrame(
            {
                "contrail_id": ["c_main"],
                "timestamp": [T0.isoformat()],  # text, as from a CSV
                "endpoint1_lon": [0.0],
                "endpoint1_lat": [0.0],
                "endpoint2_lon": [1.0],
                "endpoint2_lat": [0.0],
            }
        )
        flights = _flights().assign(frame_time=T0.isoformat())
        contrails_before, flights_before = contrails.copy(), flights.copy()
        out = Scorer().score_all_contrails(contrails, flights)
        assert (out["status"] == "scored").sum() == 2
        assert out["timestamp"].iloc[0] == T0
        pd.testing.assert_frame_equal(contrails, contrails_before)
        pd.testing.assert_frame_equal(flights, flights_before)


class TestBoundingBoxDateLine:
    POINTS = pd.DataFrame({"lon_pc": [179.9, -179.9, -170.0, 0.0, 170.0], "lat_pc": [40.0] * 5})

    def _kept(self, endpoints) -> list:
        kept, _ = Scorer()._filter_points_by_bounding_box(self.POINTS, endpoints, 0.5)
        return kept["lon_pc"].tolist()

    def test_box_across_date_line(self):
        assert self._kept(((179.8, 40.0), (-179.8, 40.0))) == [179.9, -179.9]
        assert self._kept(((179.5, 40.0), (179.7, 40.0))) == [179.9, -179.9]
        assert self._kept(((179.8, 40.0), (180.2, 40.0))) == [179.9, -179.9]

    def test_area_of_box_across_date_line(self):
        _, area = Scorer()._filter_points_by_bounding_box(
            self.POINTS, ((179.8, 40.0), (-179.8, 40.0)), 0.5
        )
        _, area_ref = Scorer()._filter_points_by_bounding_box(
            self.POINTS, ((9.8, 40.0), (10.2, 40.0)), 0.5
        )
        assert area == pytest.approx(area_ref)

    def test_scores_flight_across_date_line(self):
        lons = np.linspace(179.6, 180.4, 9)
        flight = _flight("across", 40.1, np.where(lons >= 180, lons - 360, lons))
        scores = Scorer().compute_scores_per_flight(flight, ((179.7, 40.0), (-179.7, 40.0)))
        assert scores["across"]["status"] == "scored"
        assert scores["across"]["rms_distance"] == pytest.approx(0.1 * KM_PER_DEG, rel=2e-2)
