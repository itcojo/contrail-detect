"""Unit tests for the shared Geostationary base class (contrail_detect.geostationary.abc).

Covers archive and processed-output paths, which must stay stable so existing caches keep
being found, the handling of M1/M2 sectors whose domain shifts between epochs, and the
display downsampling used by the Full Disk detection maps.
"""

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.collections import QuadMesh

from contrail_detect.core._plotting import add_globe_inset, lon_span, max_pool, unwrap_lons
from contrail_detect.flights import plot_flight_coverage
from contrail_detect.geostationary.abc import Geostationary
from contrail_detect.geostationary.goes import GOESDetector
from contrail_detect.tracking import plot_tracks

# ── Concrete stub (satisfies the ABC without any real implementation) ─────────


class _StubGeo(Geostationary):
    def _gen_scan_time_offset(self):
        pass

    def _gen_tiles(self):
        pass

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


def _bare_geo(satellite: str = "GOES-16", region: str = "F") -> _StubGeo:
    """Minimal Geostationary instance with only the attributes the path builders need."""
    obj = object.__new__(_StubGeo)
    obj.satellite = satellite
    obj.region = region
    obj.LONG_NAME = None
    obj.detection_subdirectory = None
    return obj


_EPOCH = np.datetime64("2024-05-01T12:00")


def _map_x(ax, line) -> np.ndarray:
    """X coordinates of a drawn line in the map's projection."""
    path = line.get_transform().transform_path(line.get_path())
    return ax.transData.inverted().transform(path.vertices)[:, 0]


# ── get_zarr_path ─────────────────────────────────────────────────────────────


class TestGetZarrPath:
    @pytest.mark.parametrize("region", ["F", "C"])
    def test_fixed_region_ignores_epoch(self, region, tmp_path):
        path = _bare_geo(region=region).get_zarr_path(zarr_path=tmp_path, epoch_start=_EPOCH)
        assert path == tmp_path / "GOES-16" / region / "detection_archive.zarr"

    @pytest.mark.parametrize("region", ["M1", "M2"])
    def test_mesoscale_archive_is_namespaced_by_epoch(self, region, tmp_path):
        obj = _bare_geo(satellite="GOES-18", region=region)
        obj.LONG_NAME, obj.detection_subdirectory = "contrails_api", "local"
        path = obj.get_zarr_path(zarr_path=tmp_path, epoch_start=_EPOCH)
        assert path == (
            tmp_path
            / "GOES-18"
            / region
            / "local"
            / "contrails_api"
            / "2024-05-01T1200"
            / "detection_archive.zarr"
        )


# ── Processed-output paths ────────────────────────────────────────────────────


@pytest.fixture
def detector(bare) -> GOESDetector:
    return bare(
        GOESDetector,
        satellite="G16",
        region="F",
        detection_subdirectory="google_api",
        LONG_NAME="LONG",
    )


BASE = Path("G16/F/google_api/LONG")


def _settings_dir(detector, stitching: bool, threshold: float) -> str:
    return f"settings_{detector._decomposition_fingerprint(stitching, threshold)}"


@pytest.mark.parametrize(
    "method, stitching, threshold, subdir",
    [
        (None, True, 0.25, ""),
        ("google_api", True, 0.25, "google_api"),
        ("decomposer", True, 0.25, "decomposer/stitched/threshold_025"),
        ("decomposer", False, 0.5, "decomposer/unstitched/threshold_050"),
    ],
)
def test_segment_and_point_paths(detector, tmp_path, method, stitching, threshold, subdir):
    expected_dir = tmp_path / BASE / subdir
    if method == "decomposer":
        expected_dir = expected_dir / _settings_dir(detector, stitching, threshold)
    kwargs = dict(method=method, contrail_stitching=stitching, prediction_threshold=threshold)
    assert detector._get_line_segments_path(tmp_path, **kwargs) == (
        expected_dir / "contrail_summary.parquet"
    )
    assert expected_dir.is_dir()
    assert detector._get_point_clouds_path(tmp_path, **kwargs) == (
        expected_dir / "contrail_points.parquet"
    )


@pytest.mark.parametrize(
    "source, expected",
    [("local", 0.30), ("contrails_api", 0.30), ("google_api", 0.40), (None, 0.30)],
)
def test_default_prediction_threshold_follows_detection_source(bare, source, expected):
    det = bare(GOESDetector, detection_subdirectory=source)
    assert det._resolve_prediction_threshold(None) == expected
    assert det._resolve_prediction_threshold(0.5) == 0.5


@pytest.mark.parametrize(
    ("method", "kwargs", "folder"),
    [
        (None, {}, ""),
        ("decomposer", {}, "decomposer/stitched/threshold_040"),
        ("decomposer", {"prediction_threshold": 0.3}, "decomposer/stitched/threshold_030"),
        ("decomposer", {"contrail_stitching": False}, "decomposer/unstitched/threshold_040"),
    ],
)
def test_geo_point_paths(detector, tmp_path, method, kwargs, folder):
    """Point clouds sit next to the line segments of the run that made them."""
    path = detector._get_geo_point_clouds_path(tmp_path, method=method, **kwargs)
    folder_path = tmp_path / BASE / folder
    if method == "decomposer":
        folder_path = folder_path / _settings_dir(
            detector,
            kwargs.get("contrail_stitching", True),
            kwargs.get("prediction_threshold", 0.40),
        )
    assert path == folder_path / "contrail_geopoints.geojson"
    if method == "decomposer":
        segments = detector._get_line_segments_path(tmp_path, method=method, **kwargs)
        assert path.parent == segments.parent


# ── has_shifting_domain ───────────────────────────────────────────────────────


class TestHasShiftingDomain:
    """Which regions are treated as repositioning between epochs."""

    def test_only_mesoscale_regions_shift(self):
        """The property must not drift from SHIFTING_DOMAIN_REGIONS."""
        for region in ("M1", "M2"):
            assert _bare_geo(region=region).has_shifting_domain is True
        for region in ("C", "F"):
            assert _bare_geo(region=region).has_shifting_domain is False
        for region in Geostationary.SHIFTING_DOMAIN_REGIONS:
            assert _bare_geo(region=region).has_shifting_domain is True


# ── save_static_geometry ──────────────────────────────────────────────────────


class TestSaveStaticGeometry:
    """A shifting domain has no shared grid, so no shared store may be written."""

    def test_shifting_domain_writes_nothing(self, tmp_path):
        for region in ("M1", "M2"):
            assert _bare_geo(region=region).save_static_geometry(zarr_path=tmp_path) is None
            assert list(tmp_path.iterdir()) == [], region

    def test_fixed_region_builds_the_store(self, tmp_path):
        """Gets past the guard and starts reading the grid, which the stub lacks."""
        with pytest.raises(AttributeError, match="lons"):
            _bare_geo(region="F").save_static_geometry(zarr_path=tmp_path)


class TestStaticScanTimes:
    def test_base_class_keeps_grid_and_zero_starts(self):
        obj = _bare_geo()
        obj.lats = np.array([[0.0, np.nan]])
        obj.__dict__["scan_time_offset"] = np.array([[5000, 6000]], dtype="timedelta64[ms]")
        np.testing.assert_array_equal(obj.static_scan_time_offset_s(), [[5.0, np.nan]])
        np.testing.assert_array_equal(obj.scan_start_offsets_s(np.zeros(3)), [0.0, 0.0, 0.0])

    def test_old_goes_store_is_refused(self, tmp_path):
        obj = _bare_geo(region="F")
        obj.SCANTIME_FROM_SCAN_START = True
        path = obj.get_static_geometry_path(tmp_path)
        xr.Dataset({"scantime_offset": (("y", "x"), np.zeros((2, 2)))}).to_zarr(path)
        for overwrite in (False, True):
            with pytest.raises(ValueError, match="migrate_static_geometry"):
                obj.save_static_geometry(zarr_path=tmp_path, overwrite=overwrite)

    def test_old_store_is_kept_where_meaning_is_unchanged(self, tmp_path):
        """Himawari and MTG grids were always measured from the nominal time."""
        obj = _bare_geo(satellite="HIMAWARI-9", region="F")
        path = obj.get_static_geometry_path(tmp_path)
        xr.Dataset({"scantime_offset": (("y", "x"), np.zeros((2, 2)))}).to_zarr(path)
        assert obj.save_static_geometry(zarr_path=tmp_path) == path
        assert obj.migrate_static_geometry(tmp_path) is None
        assert path.exists()

    def test_new_store_is_reused(self, tmp_path):
        obj = _bare_geo(region="F")
        path = obj.get_static_geometry_path(tmp_path)
        xr.Dataset(attrs={"contrail_detect_version": "1.13.0"}).to_zarr(path)
        assert obj.save_static_geometry(zarr_path=tmp_path) == path


class TestMigrateStaticGeometry:
    def _tree(self, tmp_path):
        obj = _bare_geo(region="F")
        obj.SCANTIME_FROM_SCAN_START = True
        store = obj.get_static_geometry_path(tmp_path)
        xr.Dataset({"scantime_offset": (("y", "x"), np.ones((2, 2)))}).to_zarr(store)
        root = store.parent / "google_api"
        old = root / "2025-01-01" / "detection_archive.zarr"
        new = root / "2026-01-01" / "detection_archive.zarr"
        xr.Dataset(attrs={"static_geometry": str(store)}).to_zarr(old)
        xr.Dataset(
            attrs={"static_geometry": str(store), "contrail_detect_version": "1.13.0"}
        ).to_zarr(new)
        return obj, store, old, new

    def test_moves_store_and_repoints_only_old_archives(self, tmp_path):
        obj, store, old, new = self._tree(tmp_path)
        moved = obj.migrate_static_geometry(tmp_path)
        assert moved == store.with_name("static_geometry_pre1.13.zarr")
        assert not store.exists() and moved.exists()
        assert xr.open_zarr(old).attrs["static_geometry"] == str(moved)
        assert xr.open_zarr(new).attrs["static_geometry"] == str(store)
        np.testing.assert_array_equal(xr.open_zarr(moved)["scantime_offset"].values, 1.0)

    def test_dry_run_changes_nothing(self, tmp_path):
        obj, store, old, _ = self._tree(tmp_path)
        obj.migrate_static_geometry(tmp_path, dry_run=True)
        assert store.exists()
        assert xr.open_zarr(old).attrs["static_geometry"] == str(store)

    def test_rerun_is_harmless(self, tmp_path):
        obj, store, old, _ = self._tree(tmp_path)
        moved = obj.migrate_static_geometry(tmp_path)
        assert obj.migrate_static_geometry(tmp_path) == moved
        assert xr.open_zarr(old).attrs["static_geometry"] == str(moved)

    def test_new_store_is_left_alone(self, tmp_path):
        obj = _bare_geo(region="F")
        obj.SCANTIME_FROM_SCAN_START = True
        store = obj.get_static_geometry_path(tmp_path)
        xr.Dataset(attrs={"contrail_detect_version": "1.13.0"}).to_zarr(store)
        assert obj.migrate_static_geometry(tmp_path) is None
        assert store.exists()


class TestMigrateEdgeCases:
    def test_repoints_archives_written_with_another_root(self, tmp_path):
        obj = _bare_geo(region="F")
        obj.SCANTIME_FROM_SCAN_START = True
        store = obj.get_static_geometry_path(tmp_path)
        xr.Dataset({"scantime_offset": (("y", "x"), np.ones((2, 2)))}).to_zarr(store)
        archive = store.parent / "google_api" / "detection_archive.zarr"
        xr.Dataset(
            attrs={"static_geometry": "/rds/other/root/GOES-16/F/static_geometry.zarr"}
        ).to_zarr(archive)
        moved = obj.migrate_static_geometry(tmp_path)
        assert xr.open_zarr(archive).attrs["static_geometry"] == str(moved)

    def test_rerun_after_new_store_finishes_the_repointing(self, tmp_path):
        obj = _bare_geo(region="F")
        obj.SCANTIME_FROM_SCAN_START = True
        store = obj.get_static_geometry_path(tmp_path)
        old = store.with_name("static_geometry_pre1.13.zarr")
        xr.Dataset({"scantime_offset": (("y", "x"), np.ones((2, 2)))}).to_zarr(old)
        xr.Dataset(attrs={"contrail_detect_version": "1.13.0"}).to_zarr(store)
        archive = store.parent / "google_api" / "detection_archive.zarr"
        xr.Dataset(attrs={"static_geometry": str(store)}).to_zarr(archive)
        assert obj.migrate_static_geometry(tmp_path) == old
        assert xr.open_zarr(archive).attrs["static_geometry"] == str(old)
        assert store.exists()


# ── plot_decomposition ────────────────────────────────────────────────────────


class _MapStubGeo(_StubGeo):
    """Stub on a plain lon/lat grid, so geostationary coordinates are degrees."""

    @property
    def image_geo_coords(self):
        return self.lons[0], self.lats[:, 0]

    @property
    def image_transforms(self):
        return ccrs.PlateCarree(), None


def _map_geo() -> _MapStubGeo:
    obj = _bare_geo()
    obj.__class__ = _MapStubGeo
    lon = np.linspace(-92.0, -88.0, 41)
    lat = np.linspace(33.0, 29.0, 41)
    obj.lons, obj.lats = np.meshgrid(lon, lat)
    detection = np.zeros(obj.lons.shape)
    detection[20, 5:35] = 0.8  # east-west contrail
    detection[5:35, 20] = 0.6  # north-south contrail, crossing it
    obj.detections = {_EPOCH: detection}
    obj.contrail_summary = pd.DataFrame(
        {
            "contrail_id": ["ew", "ns"],
            "timestamp": _EPOCH,
            "endpoint1_lon": [lon[5], lon[20]],
            "endpoint1_lat": [lat[20], lat[5]],
            "endpoint2_lon": [lon[34], lon[20]],
            "endpoint2_lat": [lat[20], lat[34]],
            "centroid_lon": [lon[20], lon[20]],
            "centroid_lat": [lat[20], lat[20]],
        }
    )
    obj.contrail_point_clouds = {
        "ew": np.array([[20, c] for c in range(5, 35)]),
        "ns": np.array([[r, 20] for r in range(5, 35)]),
    }
    return obj


class _DateLineStubGeo(_MapStubGeo):
    """Stub whose image spans 170E to 170W, with pixel lons in [-180, 180)."""

    @property
    def image_geo_coords(self):
        return np.linspace(170.0, 190.0, 21), self.lats[:, 0]


def _date_line_geo() -> _DateLineStubGeo:
    obj = _map_geo()
    obj.__class__ = _DateLineStubGeo
    lon = (np.linspace(170.0, 190.0, 21) + 180.0) % 360.0 - 180.0
    obj.lons, obj.lats = np.meshgrid(lon, np.linspace(45.0, 35.0, 11))
    obj.detections = {_EPOCH: np.full(obj.lons.shape, 0.9)}
    obj.contrail_summary = pd.DataFrame(
        {
            "contrail_id": ["c1"],
            "timestamp": [_EPOCH],
            "endpoint1_lon": [178.0],
            "endpoint1_lat": [40.0],
            "centroid_lon": [-179.5],
            "centroid_lat": [40.0],
            "endpoint2_lon": [-177.0],
            "endpoint2_lat": [40.0],
        }
    )
    obj.contrail_point_clouds = {"c1": np.array([[5, c] for c in range(8, 14)])}
    return obj


class TestPlotDecomposition:
    BBOX = (-91.0, -89.0, 30.0, 32.0)

    def test_draws_mask_and_contrails(self, tmp_path):
        geo = _map_geo()
        fig = geo.plot_decomposition(_EPOCH, bbox=self.BBOX, fig_output_dir=tmp_path)
        assert fig.axes[0].get_title(loc="left").startswith("GOES-16 F, 2024-05-01 12:00")
        assert len(fig.axes[1].lines) == 2  # one centre line per contrail
        np.testing.assert_allclose(fig.axes[0].get_extent(ccrs.PlateCarree()), self.BBOX)
        ew = geo.contrail_summary.iloc[0]
        np.testing.assert_array_equal(
            fig.axes[1].lines[0].get_xdata(),
            [ew["endpoint1_lon"], ew["centroid_lon"], ew["endpoint2_lon"]],
        )
        assert (tmp_path / "GOES-16-F-decomposition-2024-05-01T1200.png").exists()

    def test_passed_contrails_override_the_loaded_ones(self):
        geo = _map_geo()
        one = geo.contrail_summary.iloc[:1]
        fig = geo.plot_decomposition(
            _EPOCH, bbox=self.BBOX, contrail_summary=one, point_clouds=geo.contrail_point_clouds
        )
        assert len(fig.axes[1].lines) == 1

        fig = geo.plot_decomposition(_EPOCH, bbox=self.BBOX, contrail_summary=geo.contrail_summary)
        assert len(fig.axes[1].lines) == 2

    def test_requires_decomposed_contrails(self):
        geo = _map_geo()
        geo.contrail_summary, geo.contrail_point_clouds = pd.DataFrame(), {}
        with pytest.raises(ValueError, match="get_decomposed_contrails"):
            geo.plot_decomposition(_EPOCH, bbox=self.BBOX)

    def test_region_outside_the_image_raises(self):
        with pytest.raises(ValueError, match="No image pixels"):
            _map_geo().plot_decomposition(_EPOCH, bbox=(10.0, 11.0, 10.0, 11.0))

    @pytest.mark.parametrize("bbox", [(170.0, -170.0, 35.0, 45.0), (170.0, 190.0, 35.0, 45.0)])
    def test_region_across_date_line(self, bbox):
        fig = _date_line_geo().plot_decomposition(_EPOCH, bbox=bbox)
        (mesh,) = [c for c in fig.axes[0].collections if isinstance(c, QuadMesh)]
        assert mesh.get_array().shape == (11, 21)  # every column kept
        centred = ccrs.PlateCarree(central_longitude=180.0)
        np.testing.assert_allclose(fig.axes[0].get_extent(centred), (-10.0, 10.0, 35.0, 45.0))
        line_x = _map_x(fig.axes[1], fig.axes[1].lines[0])
        assert np.ptp(line_x) == pytest.approx(5.0)  # 178 to -177 is 5 degrees, not 355
        assert len(line_x) == 3  # not split at the date line


class TestDiskMaps:
    """plot_tracks and plot_flight_coverage, which draw on the detector's disk map."""

    def _tracked_geo(self):
        geo = _map_geo()
        geo.central_lon = -75.0
        geo.contrail_summary = pd.concat(
            [geo.contrail_summary, geo.contrail_summary.assign(timestamp=_EPOCH + 600)]
        ).assign(track_id=["a", None, "a", None])
        return geo

    def test_plot_tracks_draws_one_path_per_track(self, tmp_path):
        geo = self._tracked_geo()
        fig = plot_tracks(geo, fig_output_dir=tmp_path)
        assert len(fig.axes[0].lines) == 1
        assert fig.axes[0].get_title(loc="left") == "GOES-16 F, 1 contrail track"
        assert (tmp_path / "GOES-16-F-tracks.png").exists()
        track = geo.contrail_summary.dropna(subset=["track_id"]).sort_values("timestamp")
        np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), track["centroid_lon"])

    def test_plot_tracks_across_date_line(self):
        geo = _map_geo()
        geo.central_lon = 140.7  # Himawari
        geo.contrail_summary = pd.DataFrame(
            {
                "track_id": "a",
                "timestamp": [_EPOCH, _EPOCH + 600, _EPOCH + 1200],
                "centroid_lon": [179.8, -179.9, -179.6],
                "centroid_lat": [40.0, 40.1, 40.2],
            }
        )
        ax = plot_tracks(geo).axes[0]
        assert np.ptp(_map_x(ax, ax.lines[0])) == pytest.approx(0.6)

    def test_plot_tracks_requires_tracking(self):
        geo = _map_geo()
        geo.central_lon = -75.0
        with pytest.raises(ValueError, match="TrackingManager"):
            plot_tracks(geo)

    def test_plot_flight_coverage(self, tmp_path):
        geo = _map_geo()
        geo.central_lon = -75.0
        flights = pd.DataFrame({"longitude": [-90.0, -89.5, -80.0], "latitude": [30.0, 31.0, 40.0]})
        fig = plot_flight_coverage(geo, flights, fig_output_dir=tmp_path)
        assert len(fig.axes[0].child_axes) == 1  # colour bar
        assert (tmp_path / "GOES-16-F-flight-coverage.png").exists()

    def test_plot_flight_coverage_without_flights_raises(self):
        geo = _map_geo()
        geo.central_lon = -75.0
        empty = pd.DataFrame({"longitude": [], "latitude": []})
        with pytest.raises(ValueError, match="no waypoints"):
            plot_flight_coverage(geo, empty)

    def test_plot_ash_rgb_leaves_rcparams_unchanged(self):
        geo = _map_geo()
        geo.central_lon = -75.0
        with plt.rc_context({"font.size": 11.0}):
            fig = geo.plot_ash_rgb(
                np.zeros((4, 4, 3)), (-80.0, -70.0, 0.0, 10.0), ccrs.PlateCarree()
            )
            assert plt.rcParams["font.size"] == 11.0
        assert fig.axes[0].texts[0].get_fontsize() == 8
        plt.close(fig)

    def test_plot_ash_rgb_centres_on_satellite(self, tmp_path):
        geo = _map_geo()
        geo.central_lon = 140.7  # Himawari, across the antimeridian from GOES
        rgb = np.zeros((4, 4, 3))
        fig = geo.plot_ash_rgb(
            rgb, (130.0, 150.0, -10.0, 10.0), ccrs.PlateCarree(), fig_output_dir=tmp_path
        )
        lon_min, lon_max, _, _ = fig.axes[0].get_extent(
            crs=ccrs.PlateCarree(central_longitude=140.7)
        )
        assert lon_min == pytest.approx(-90.0) and lon_max == pytest.approx(90.0)
        assert (tmp_path / "GOES-16-F-ash-rgb.png").exists()


class TestFrameFigures:
    """plot_detections keeps figures open when not saving, and saves and closes them otherwise."""

    @staticmethod
    def _plot_frame(timestamp, fig_output_dir):
        if timestamp == _EPOCH + 600:
            return None  # a frame that could not be plotted
        fig = plt.figure()
        _map_geo()._save_frame_figure(fig, fig_output_dir, timestamp)
        return fig

    def test_returns_open_figures_when_not_saving(self):
        figures = _map_geo()._plot_frames(
            self._plot_frame, [_EPOCH, _EPOCH + 600], None, desc="frames"
        )
        assert len(figures) == 1
        assert plt.fignum_exists(figures[0].number)
        plt.close("all")

    def test_saves_and_closes_when_given_a_directory(self, tmp_path):
        figures = _map_geo()._plot_frames(self._plot_frame, [_EPOCH], tmp_path, desc="frames")
        assert figures == []
        assert not plt.get_fignums()
        assert (tmp_path / "2024-05-01" / "GOES-16-F-2024-05-01T1200.png").exists()


class TestMaxPool:
    def test_keeps_thin_feature_and_ignores_nan(self):
        scores = np.full((6, 6), np.nan)
        scores[:, 4:] = 0.0
        scores[2, 4] = 0.9  # a one-pixel contrail
        x = np.arange(6.0) * 2.0
        y = np.arange(6.0)
        px, py, pooled = max_pool(scores, x, y, 3)
        assert pooled.shape == (2, 2)
        assert pooled[0, 1] == 0.9
        assert np.isnan(pooled[0, 0])  # all-NaN block stays NaN
        np.testing.assert_allclose(px, [2.0, 8.0])
        np.testing.assert_allclose(py, [1.0, 4.0])

    def test_pads_uneven_grid(self):
        scores = np.arange(25.0).reshape(5, 5)
        px, py, pooled = max_pool(scores, np.arange(5.0), np.arange(5.0), 2)
        assert pooled.shape == (3, 3)
        assert pooled[-1, -1] == 24.0
        # The last, partial block is centred on the coordinates it has
        np.testing.assert_allclose(px, [0.5, 2.5, 4.0])


class TestLongitudeHelpers:
    def test_unwrap_lons(self):
        np.testing.assert_allclose(unwrap_lons([179.8, -179.9, -179.6]), [179.8, 180.1, 180.4])
        np.testing.assert_allclose(unwrap_lons([-179.9, 179.9]), [-179.9, -180.1])

        lons = np.array([-75.3, -74.1, 0.2, 120.7, 179.9])
        np.testing.assert_array_equal(unwrap_lons(lons), lons)

        out = unwrap_lons([179.8, np.nan, -179.9])
        assert np.isnan(out[1])
        np.testing.assert_allclose(out[[0, 2]], [179.8, 180.1])

    def test_lon_span(self):
        assert lon_span([-75.0, -74.0, -76.0]) == (-76.0, -74.0)
        assert lon_span([179.9, -179.9, 179.5, np.nan]) == pytest.approx((179.5, 180.1))

    @pytest.mark.parametrize(
        ("extent", "centre"),
        [([10.0, 20.0, 30.0, 40.0], 15.0), ([175.0, -175.0, 30.0, 40.0], -180.0)],
    )
    def test_globe_inset_centre(self, extent, centre):
        fig = plt.figure()
        add_globe_inset(fig, extent)
        assert fig.axes[0].projection.proj4_params["lon_0"] == centre
        plt.close(fig)


# ── create_timestamp_sequence ─────────────────────────────────────────────────


def _sequence(interval, start, end, satellite="GOES-19"):
    obj = _bare_geo(satellite=satellite)
    obj.timestamp_strings = ["left over from an earlier call"]
    type(obj).image_interval = property(lambda self: interval)
    obj.create_timestamp_sequence(np.datetime64(start), np.datetime64(end))
    return obj


class TestCreateTimestampSequence:
    def test_stops_at_end_time(self):
        """18:00 to 18:15 at 10 minutes is 18:00 and 18:10, not 18:20."""
        obj = _sequence(10, "2026-01-09T18:00", "2026-01-09T18:15")
        assert obj.timestamp_sequence == [
            np.datetime64("2026-01-09T18:00"),
            np.datetime64("2026-01-09T18:10"),
        ]
        obj = _sequence(10, "2026-01-09T18:00", "2026-01-09T18:20")
        assert obj.timestamp_sequence[-1] == np.datetime64("2026-01-09T18:20")

    @pytest.mark.parametrize("start", ["2026-01-09T18:05", "2026-01-09T18:00:30"])
    def test_start_off_the_image_schedule_raises(self, start):
        with pytest.raises(ValueError, match="no image at") as err:
            _sequence(10, start, "2026-01-09T18:30")
        assert "2026-01-09T18:00:00 or 2026-01-09T18:10:00" in str(err.value)

    def test_five_minute_schedule(self):
        obj = _sequence(5, "2026-01-09T18:05", "2026-01-09T18:10")
        assert obj.timestamp_strings == ["2026-01-09T1805", "2026-01-09T1810"]

    def test_strings_keep_seconds_when_the_interval_needs_them(self):
        """A 2.5-minute schedule must not label 12:02:30 as 12:02."""
        obj = _sequence(2.5, "2026-01-09T12:00", "2026-01-09T12:05", satellite="MTG")
        assert obj.timestamp_strings == [
            "2026-01-09T120000",
            "2026-01-09T120230",
            "2026-01-09T120500",
        ]
        parsed = [np.datetime64(pd.to_datetime(s)) for s in obj.timestamp_strings]
        assert parsed == list(obj.timestamp_sequence)

    def test_strings_replace_those_of_an_earlier_call(self):
        obj = _sequence(10, "2026-01-09T18:00", "2026-01-09T18:10")
        obj.create_timestamp_sequence(
            np.datetime64("2026-01-10T18:00"), np.datetime64("2026-01-10T18:00")
        )
        assert obj.timestamp_strings == ["2026-01-10T1800"]


# ── GOES operational dates ────────────────────────────────────────────────────


class TestGoesOperationalDates:
    """Only periods when each GOES was in service at its operational position and rate."""

    @pytest.mark.parametrize(
        ("satellite", "start"),
        [
            ("GOES-16", "2019-04-02T15:50"),  # Full Disk every 15 min until 16:00 UTC
            ("GOES-16", "2017-06-01T00:00"),  # post-launch testing at 89.5°W
            ("GOES-18", "2022-06-01T00:00"),  # post-launch testing
            ("GOES-19", "2025-03-01T00:00"),  # post-launch testing at 89.5°W
        ],
    )
    def test_before_operations_raises(self, satellite, start):
        with pytest.raises(ValueError, match="before the valid first available"):
            _sequence(10, start, start, satellite=satellite)

    def test_goes16_after_handover_raises(self):
        with pytest.raises(ValueError, match="after the valid last available"):
            _sequence(10, "2025-04-07T00:10", "2025-04-07T00:10", satellite="GOES-16")

    @pytest.mark.parametrize(
        ("satellite", "start"),
        [
            ("GOES-16", "2019-04-02T16:00"),
            ("GOES-16", "2025-04-06T23:50"),
            ("GOES-18", "2023-01-04T18:00"),
            ("GOES-19", "2025-04-07T00:00"),
        ],
    )
    def test_operational_period_accepted(self, satellite, start):
        _sequence(10, start, start, satellite=satellite)


def test_goes_position_away_from_operational_raises():
    """A satellite in test position can't be fixed by swapping the longitude."""
    obj = object.__new__(GOESDetector)
    obj.satellite = "GOES-19"
    obj.da_goes = type(
        "DA", (), {"attrs": {"goes_imager_projection": {"longitude_of_projection_origin": -89.5}}}
    )()
    with pytest.raises(ValueError, match="-89.5°, but its operational position is -75.0°"):
        GOESDetector._get_projection_attrs.func(obj)


# ── Reading back what was saved ───────────────────────────────────────────────


def test_mesoscale_archive_readers_find_what_the_writer_wrote(tmp_path):
    """The writer namespaces M1/M2 archives by epoch; readers default to the same epoch."""
    obj = _bare_geo(region="M1")
    obj.timestamp_sequence = [_EPOCH, _EPOCH + np.timedelta64(1, "m")]
    assert obj.get_zarr_path(tmp_path) == obj.get_zarr_path(tmp_path, epoch_start=_EPOCH)


def test_load_needs_a_detection_source():
    from contrail_detect.geostationary.abi_base import ABIDetector

    obj = object.__new__(GOESDetector)
    obj.detection_subdirectory = None
    with pytest.raises(ValueError, match="which source's saved detections"):
        ABIDetector.get_detections(obj, method="load")


def test_cropped_array_off_the_grid_raises():
    obj = _bare_geo()
    obj.lats = np.zeros((3, 4))
    type(obj).image_geo_coords = property(lambda self: (np.arange(4.0), np.arange(3.0)))
    with pytest.raises(ValueError, match="do not line up"):
        obj._reproject_cropped_to_full_grid(
            np.ones((2, 2)), np.array([10.0, 11.0]), np.array([10.0, 11.0])
        )


# ── Positions outside a partial-disk image ────────────────────────────────────


def test_position_outside_the_image_is_not_imaged():
    """Beyond the image's edge is 'not imaged' (-1), not the edge pixel."""
    obj = _bare_geo(region="C")
    x, y = np.arange(0.0, 10.0), np.arange(9.0, -1.0, -1.0)  # 10 x 10, y descending
    type(obj).latlon_to_sat_transform = property(
        lambda self: type("T", (), {"transform": staticmethod(lambda lon, lat: (lon, lat))})()
    )
    i, j = obj.get_grid_index_from_position(
        np.array([5.0, 9.4, 9.6, -0.6, 5.0]), np.array([5.0, 5.0, 5.0, 5.0, 12.0]), x, y
    )
    assert list(j) == [5, 9, -1, -1, -1]
    assert list(i[:2]) == [4, 4]
    assert i[4] == -1


@pytest.mark.parametrize(
    ("y", "row"),
    [(4.4, 0), (3.6, 0), (3.4, 1), (3.0, 1), (2.6, 1), (2.4, 2)],
)
def test_position_gets_the_row_whose_edges_contain_it(y, row):
    """Rows centred at 4, 3, 2 (north to south) span half a row either side of the centre."""
    obj = _bare_geo(region="C")
    type(obj).latlon_to_sat_transform = property(
        lambda self: type("T", (), {"transform": staticmethod(lambda lon, lat: (lon, lat))})()
    )
    i, j = obj.get_grid_index_from_position(1.0, y, np.arange(3.0), np.array([4.0, 3.0, 2.0]))
    assert (i, j) == (row, 1)


def test_detectors_take_satellite_then_region():
    """Every detector takes satellite first and region second, so positional calls agree."""
    import inspect

    from contrail_detect.geostationary.himawari import HimawariDetector
    from contrail_detect.geostationary.meteosat import MeteosatDetector

    for cls in (GOESDetector, HimawariDetector, MeteosatDetector):
        params = list(inspect.signature(cls.__init__).parameters)[1:3]
        assert params[0].endswith("_satellite") and params[1].endswith("_region"), cls.__name__
