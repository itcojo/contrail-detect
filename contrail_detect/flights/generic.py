"""Flight data handler for your own data already in the standard schema."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Self

import pandas as pd

from contrail_detect.core.config import DATA_DIR
from contrail_detect.flights.flights_base import FlightData


class GenericFlightData(FlightData):
    """Generic flight data handler for your own flight data.

    Loads flight data already in the standard schema (``flight_id``, ``time``,
    ``latitude``, ``longitude``, ``altitude_ft``) with ``load_from_file`` or
    ``load_from_dataframe``. It has no download methods.

    ``altitude_ft`` should be geometric (GNSS) altitude in feet, used for parallax. An
    optional ``pressure_altitude_ft`` column (barometric, feet) picks the ERA5 wind level;
    without it, ``altitude_ft`` is used for both.
    """

    _READERS: dict[str, Callable] = {
        "parquet": pd.read_parquet,
        "csv": pd.read_csv,
        "json": pd.read_json,
    }

    _WRITERS: dict[str, str] = {
        "parquet": "to_parquet",
        "csv": "to_csv",
        "json": "to_json",
    }

    def __init__(self: Self, save_path: str | Path | None = None) -> None:
        """Initialize the generic flight data handler.

        Args:
            save_path: Directory used by ``save_to_file`` when no file path is given.
                Defaults to the package data directory.
        """
        super().__init__()
        if save_path is None:
            self.save_path = DATA_DIR / "flight_data" / "generic"
        else:
            self.save_path = Path(save_path)

    def load_from_dataframe(self: Self, df: pd.DataFrame) -> None:
        """Load flight data from a DataFrame in the standard schema.

        Args:
            df: Flight data. It is copied, so later changes to ``df`` have no effect.
        """
        self.flight_data = df.copy()
        self._postprocess()
        logging.info(f"Loaded {len(self.flight_data)} flight records from DataFrame")

    def load_from_file(
        self: Self,
        file_path: str | Path,
        file_format: str | None = None,
    ) -> None:
        """Load flight data from a Parquet, CSV or JSON file in the standard schema.

        Args:
            file_path: Path to the file.
            file_format: 'parquet', 'csv' or 'json'. If None, inferred from the suffix.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            ValueError: If the format is unsupported or cannot be inferred.
        """
        path, fmt = self._resolve_path(file_path, file_format)

        if not path.exists():
            raise FileNotFoundError(f"Flight data file not found: {path}")

        reader = self._READERS[fmt]
        self.flight_data = reader(path)

        self._postprocess()

        logging.info(f"Loaded {len(self.flight_data)} records from {path}")

    def save_to_file(
        self: Self,
        file_path: str | Path | None = None,
        file_format: str | None = None,
    ) -> None:
        """Save the loaded flight data to a Parquet, CSV or JSON file.

        Args:
            file_path: Output path. If None, writes ``generic_<start>_<end>`` under
                ``save_path``, named from the data's time range.
            file_format: 'parquet', 'csv' or 'json'. If None, inferred from the suffix,
                or Parquet when ``file_path`` is None.

        Raises:
            ValueError: If the format is unsupported or cannot be inferred.
        """
        if self.flight_data.empty:
            logging.warning("No flight data to save")
            return

        path, fmt = self._resolve_path(file_path, file_format)
        path.parent.mkdir(parents=True, exist_ok=True)

        writer_method = getattr(self.flight_data, self._WRITERS[fmt])

        if fmt == "json":
            writer_method(path, orient="records", date_format="iso")
        else:
            writer_method(path, index=False)

        logging.info(f"Saved {len(self.flight_data)} records to {path}")

    def _normalize_columns(self: Self) -> None:
        """GenericFlightData assumes data is already in the standard schema."""
        pass

    def _postprocess(self: Self) -> None:
        """Common cleanup after loading data."""
        if "time" in self.flight_data.columns:
            self.flight_data["time"] = pd.to_datetime(self.flight_data["time"])

    def _resolve_path(
        self: Self,
        file_path: str | Path | None,
        file_format: str | None,
    ) -> tuple[Path, str]:
        """Resolve file path and format consistently."""
        if file_path is None:
            if file_format is None:
                file_format = "parquet"
            times = pd.to_datetime(self.flight_data["time"])
            save_name = f"generic_{times.min():%Y%m%dT%H%M}_{times.max():%Y%m%dT%H%M}"
            path = self.save_path / f"{save_name}.{file_format}"
        else:
            path = Path(file_path)
            if file_format is None:
                if path.suffix:
                    file_format = path.suffix.lstrip(".")
                else:
                    raise ValueError("File format could not be inferred from file path")

        file_format = file_format.lower()

        if file_format not in self._READERS:
            raise ValueError(f"Unsupported file format: {file_format}")

        return path, file_format
