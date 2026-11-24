"""
Interaction with and manipulation of ADS-B flight data from the Contrails.org API.

"""

import io
import logging
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from contrail_detect.core.config import (
    CONTRAILS_API_HEADERS,
    CONTRAILS_API_URL_TELEMETRY,
    DATA_DIR,
)
from contrail_detect.flights.flights_base import FlightData
from contrail_detect.models.contrail_model_base import get_with_retries

# Per-hour telemetry request: attempts and per-attempt timeout
_MAX_RETRIES = 4
_TIMEOUT_S = 120.0


class ContrailsAPIFlightData(FlightData):
    """Flight data handler for ADS-B data sourced from the Contrails.org API.

    Provides methods for downloading and processing Contrails.org API ADS-B flight data
    based on specified dates, bounding boxes, and target addresses. Supports advection
    of flight paths using meteorological data.

    The API only serves data more than 48 hours old. Requests for more recent hours are
    refused, and ``download_from_api`` stops with an error naming them.

    Attributes:
        target_addresses: ICAO 24-bit addresses to filter flights by, or None for all.
    """

    def __init__(
        self: Self,
        req_timestamps: list[str | np.datetime64],
        adv_max: int | np.timedelta64,
        req_bbox: list[float] | None = None,
        add_bbox_buffer: float | None = None,
        target_addresses: list[str] | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        """Initialize the Contrails API flight data handler.

        Args:
            req_timestamps: Times flight data is needed for, normally the detector's frame
                times (``detector.timestamp_sequence``); strings are parsed as
                ``numpy.datetime64``. Every hour from the earliest time minus ``adv_max`` to
                the latest time is downloaded, so pass times, not dates: ``'2026-01-09'``
                means midnight, not the whole day.
            adv_max: Maximum time for flight path advection. If integer, interpreted as hours.
            req_bbox: Bounding box [min_lon, max_lon, min_lat, max_lat] for flight data request.
            add_bbox_buffer: Buffer in degrees to add around bounding box.
            target_addresses: List of ICAO 24-bit address strings for filtering flights.
            save_path: Path where downloaded data will be saved. Defaults to package data directory.
        """
        super().__init__(
            req_timestamps=req_timestamps,
            adv_max=adv_max,
            req_bbox=req_bbox,
            add_bbox_buffer=add_bbox_buffer,
        )

        # List of target_address ICAO 24-bit strings for flight data request
        self.target_addresses = target_addresses

        # Path to where downloaded data will be saved
        if save_path is None:
            self.save_path = DATA_DIR / "flight_data" / "contrails_api"
        else:
            self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Descriptive name for saved files
        self.save_name = (
            f"api_"
            f"{np.datetime64(self.req_timestamps[0], 's').item().strftime('%Y%m%dT%H%M')}_"
            f"{np.datetime64(self.req_timestamps[-1], 's').item().strftime('%Y%m%dT%H%M')}"
        )
        if self.req_bbox is not None:
            self.save_name += (
                f"_BBOX_"
                f"{int(self.req_bbox[0])}_"
                f"{int(self.req_bbox[1])}_"
                f"{int(self.req_bbox[2])}_"
                f"{int(self.req_bbox[3])}"
            )

    def _repr_fields(self: Self) -> dict:
        return {**super()._repr_fields(), "target_addresses": self.target_addresses}

    def download_from_api(
        self: Self,
        save_data: bool = True,
        allow_missing_hours: bool = False,
    ) -> None:
        """Download ADS-B telemetry data from the Contrails.org API.

        Downloads one hour at a time, retrying timeouts, dropped connections and server
        errors. Data is only available once it is more than 48 hours old. Calling this
        again replaces the flight data rather than adding to it.

        Args:
            save_data: Whether to cache each downloaded hour in ``save_path / "raw"``.
            allow_missing_hours: If False (the default), stop with an error when any hour
                cannot be downloaded, since a gap leaves its contrails unmatched or matched
                to the wrong flight. If True, log the missing hours and continue.

        Raises:
            RuntimeError: If any hour could not be downloaded and ``allow_missing_hours``
                is False.
        """
        first_date = np.datetime64(np.min(self.req_timestamps), "h")
        last_date = np.datetime64(np.max(self.req_timestamps), "h")
        logging.info(f"Assessing data from {first_date} to {last_date}...")

        # Hourly requests needed based on the API description
        hours_to_request = pd.date_range(
            start=pd.Timestamp(first_date.item()), end=pd.Timestamp(last_date.item()), freq="h"
        )

        dataframes = []
        failed_hours = []
        raw_adsb_path = self.save_path / "raw"
        if save_data:
            raw_adsb_path.mkdir(parents=True, exist_ok=True)

        for hr in tqdm(hours_to_request, desc="Downloading Contrails API telemetry"):
            date_str = hr.strftime("%Y-%m-%dT%H")
            local_file = raw_adsb_path / f"{date_str}.pq"

            if local_file.exists():
                logging.info(f"Loading cached file: {local_file}")
                df = pd.read_parquet(local_file)
                dataframes.append(df)
            else:
                try:
                    r = get_with_retries(
                        CONTRAILS_API_URL_TELEMETRY,
                        max_retries=_MAX_RETRIES,
                        timeout=_TIMEOUT_S,
                        headers=CONTRAILS_API_HEADERS,
                        params={"date": date_str},
                    )
                except requests.RequestException as e:
                    detail = e.response.text if getattr(e, "response", None) is not None else ""
                    logging.warning(f"Failed to fetch {date_str}: {e} {detail}".strip())
                    failed_hours.append(date_str)
                    continue
                df = pd.read_parquet(io.BytesIO(r.content))
                if save_data:
                    df.to_parquet(local_file, index=False)
                dataframes.append(df)

        if failed_hours:
            message = (
                f"Could not download {len(failed_hours)} of {len(hours_to_request)} hours of "
                f"flight data: {', '.join(failed_hours)}. The API serves only data more than "
                "48 hours old; for older hours, try again later."
            )
            if not allow_missing_hours:
                raise RuntimeError(
                    message + " To continue without them, pass allow_missing_hours=True."
                )
            logging.warning(message + " Continuing without them.")

        if not dataframes:
            logging.warning("No data downloaded or loaded from API.")
            self.flight_data = pd.DataFrame()
            return

        full_df = pd.concat(dataframes, ignore_index=True)

        # Apply target_addresses filter if specified
        if self.target_addresses:
            full_df = full_df[full_df["icao_address"].isin(self.target_addresses)]

        # Apply bounding box filter if specified
        if self.req_bbox:
            min_lon, max_lon, min_lat, max_lat = self.req_bbox
            full_df = full_df[
                (full_df["longitude"] >= min_lon)
                & (full_df["longitude"] <= max_lon)
                & (full_df["latitude"] >= min_lat)
                & (full_df["latitude"] <= max_lat)
            ]

        self.flight_data = full_df.reset_index(drop=True)
        self._normalize_columns()

    def _normalize_columns(self: Self) -> None:
        """Normalize Contrails API column names to the standard schema."""
        if self.flight_data.empty:
            return

        # Two altitudes, each filled from the other where missing:
        # - altitude_ft, geometric (GNSS): the aircraft's height, used for parallax
        # - pressure_altitude_ft, barometric: picks the ERA5 wind level, since ERA5 levels
        #   are pressure altitudes. At cruise the two differ by up to about 1,000 ft.
        columns = self.flight_data.columns
        gnss = self.flight_data["altitude_gnss"] if "altitude_gnss" in columns else None
        baro = self.flight_data["altitude_baro"] if "altitude_baro" in columns else None
        if gnss is not None or baro is not None:
            gnss = gnss if gnss is not None else baro
            baro = baro if baro is not None else gnss
            self.flight_data = self.flight_data.drop(
                columns=[c for c in ("altitude_gnss", "altitude_baro") if c in columns]
            ).assign(altitude_ft=gnss.fillna(baro), pressure_altitude_ft=baro.fillna(gnss))

        # Rename other columns to match standard schema. Airline and airport codes keep
        # their API names, which say which code system they use.
        rename = {"timestamp": "time"}
        if "flight_id" not in self.flight_data.columns:
            rename["icao_address"] = "flight_id"
        self.flight_data = self.flight_data.rename(columns=rename)
