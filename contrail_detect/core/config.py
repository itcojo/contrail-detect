"""
Configuration settings and global variables for the contrail-detection package.

This module defines constants, file paths, and parameters used throughout the project.

"""

import logging
import os
from pathlib import Path

import dotenv

# logging.info() at import would configure the root logger and make a later
# logging.basicConfig() do nothing
logger = logging.getLogger(__name__)

# Define package head directory
PACKAGE_DIR = Path(__file__).resolve().parent.parent
KAGGLE_DIR = PACKAGE_DIR / "models" / "kaggle_contrails_solution"

# Load environment variables from ~/.env if it exists. This must run before anything below
# reads the environment, so that CONTRAILS_DATA_DIR set there is honoured. An empty
# CONTRAILS_DATA_DIR is treated as unset.
if os.environ.get("CONTRAILS_DATA_DIR") == "":
    del os.environ["CONTRAILS_DATA_DIR"]
_ = dotenv.load_dotenv(dotenv_path=Path.home() / ".env")

# --- --- ---
# Data directory (downloads, caches and outputs)
# --- --- ---
# An editable install from a clone keeps its data inside the repository. If the clone is
# not writable, or the package was installed without -e (into site-packages), it uses a
# folder in the home directory instead. CONTRAILS_DATA_DIR overrides both.

DEFAULT_USER_DATA_DIR = Path.home() / ".contrail-detect"


def _is_source_checkout(package_dir: Path) -> bool:
    """Whether the package is running from a writable source checkout."""
    installed = any(part in ("site-packages", "dist-packages") for part in package_dir.parts)
    return not installed and os.access(package_dir / "data", os.W_OK)


def _resolve_data_dir(package_dir: Path) -> Path:
    """Return the data directory: CONTRAILS_DATA_DIR, the checkout, or the user folder.

    Environment variables and ``~`` in CONTRAILS_DATA_DIR are expanded, and a relative
    path is made absolute against the current working directory.
    """
    if os.getenv("CONTRAILS_DATA_DIR"):
        expanded = os.path.expanduser(os.path.expandvars(os.environ["CONTRAILS_DATA_DIR"]))
        return Path(expanded).absolute()
    if _is_source_checkout(package_dir):
        return package_dir / "data"
    return DEFAULT_USER_DATA_DIR


DATA_DIR = _resolve_data_dir(PACKAGE_DIR)
_IN_CHECKOUT = DATA_DIR == PACKAGE_DIR / "data"

# Not created here: importing the package should not touch the filesystem. Every writer
# creates the subfolder it needs (with parents) on first use.
if not DATA_DIR.exists():
    logger.info(
        f"contrail-detect will store downloads, caches and outputs in {DATA_DIR}. "
        "Set the environment variable CONTRAILS_DATA_DIR to use another location."
    )

RESULT_DIR = DATA_DIR / "inference_data"
# Locations predating DATA_DIR are kept for source checkouts so existing data stays put.
TEST_DIR = KAGGLE_DIR / "data" / "test" if _IN_CHECKOUT else DATA_DIR / "tmp" / "test"
WEIGHTS_DIR = KAGGLE_DIR / "input" / "weights" if _IN_CHECKOUT else DATA_DIR / "weights"
FIGURE_DIR = PACKAGE_DIR.parent / "figures" if _IN_CHECKOUT else DATA_DIR / "figures"

# --- --- ---
# Contrails.org API configuration
# --- --- ---

# Build headers for the Contrails API
_api_key = os.getenv("CONTRAILS_API_KEY")
if _api_key is None:
    logger.warning(
        "Environment variable CONTRAILS_API_KEY is not set. "
        "Requests will be made without an API key header."
    )
CONTRAILS_API_HEADERS = {k: v for k, v in {"x-api-key": _api_key}.items() if v is not None}

# URL endpoints for the Contrails API
CONTRAILS_API_URL_DETECTIONS = (
    "https://api.contrails.org/v1/observations/contrails-org/goes/L1/detections"
)
CONTRAILS_API_URL_TELEMETRY = "https://api.contrails.org/v1/adsb/telemetry"

# --- --- ---
# Google Contrails API configuration
# --- --- ---

_google_api_key = os.getenv("GOOGLE_CONTRAILS_API_KEY")
if _google_api_key is None:
    logger.warning(
        "Environment variable GOOGLE_CONTRAILS_API_KEY is not set. "
        "Requests to the Google Contrails API will fail authentication."
    )
GOOGLE_CONTRAILS_API_KEY = _google_api_key
