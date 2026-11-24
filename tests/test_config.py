"""Tests for contrail_detect.core.config: data-directory resolution and API URLs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from contrail_detect.core import config


def _make_package(root: Path) -> Path:
    """Create a fake package directory with a data folder and return it."""
    package_dir = root / "contrail_detect"
    (package_dir / "data").mkdir(parents=True)
    return package_dir


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTRAILS_DATA_DIR", raising=False)


def test_source_checkout_uses_package_data(tmp_path: Path) -> None:
    package_dir = _make_package(tmp_path / "repo")
    assert config._resolve_data_dir(package_dir) == package_dir / "data"


def test_falls_back_to_user_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for site in ("site-packages", "dist-packages"):
        package_dir = _make_package(tmp_path / "lib" / "python3.11" / site)
        assert config._resolve_data_dir(package_dir) == config.DEFAULT_USER_DATA_DIR

    package_dir = _make_package(tmp_path / "repo")
    monkeypatch.setattr(config.os, "access", lambda path, mode: False)
    assert config._resolve_data_dir(package_dir) == config.DEFAULT_USER_DATA_DIR


def test_env_var_is_normalised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = _make_package(tmp_path / "repo")

    monkeypatch.setenv("CONTRAILS_DATA_DIR", str(tmp_path / "elsewhere"))
    assert config._resolve_data_dir(package_dir) == tmp_path / "elsewhere"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONTRAILS_DATA_DIR", "relative/data")
    assert config._resolve_data_dir(package_dir) == tmp_path / "relative" / "data"

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for value in ("$HOME/x", "${HOME}/x", "~/x"):
        monkeypatch.setenv("CONTRAILS_DATA_DIR", value)
        assert config._resolve_data_dir(package_dir) == tmp_path / "home" / "x", value


def _clean_env(home: Path, **extra: str) -> dict[str, str]:
    """Return an environment with a temporary HOME and no package variables set."""
    drop = ("CONTRAILS_DATA_DIR", "CONTRAILS_API_KEY", "GOOGLE_CONTRAILS_API_KEY")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["HOME"] = str(home)
    env.update(extra)
    return env


def _run(code: str, env: dict[str, str], cwd: Path | None = None) -> str:
    """Run ``code`` in a fresh interpreter and return its stripped stdout."""
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=cwd, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def test_import_does_not_configure_root_logger(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    env = _clean_env(home, CONTRAILS_DATA_DIR=str(tmp_path / "missing"))
    code = (
        "import logging, contrail_detect.core.config\n"
        "print(len(logging.getLogger().handlers))\n"
        "logging.basicConfig(level=logging.INFO)\n"
        "print(logging.getLogger().level == logging.INFO)\n"
    )
    assert _run(code, env).split() == ["0", "True"]


def test_home_dotenv(tmp_path: Path) -> None:
    """CONTRAILS_DATA_DIR set only in ~/.env, as the README suggests, must take effect."""
    code = "from contrail_detect.core.config import DATA_DIR; print(DATA_DIR)"

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(f"CONTRAILS_DATA_DIR={tmp_path / 'from_dotenv'}\n")
    assert _run(code, _clean_env(home)) == str(tmp_path / "from_dotenv")
    assert _run(code, _clean_env(home, CONTRAILS_DATA_DIR="")) == str(tmp_path / "from_dotenv")

    (home / ".env").write_text("CONTRAILS_DATA_DIR=$HOME/from_dotenv\n")
    assert _run(code, _clean_env(home)) == str(home / "from_dotenv")
