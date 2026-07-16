"""Env-driven configuration for the MLflow observability exporter.

Every knob is an environment variable so the same package runs bare-metal on a
dev workstation (defaults below) and as a docker sidecar in production
(docker-compose.mlflow.yml sets the URIs/paths explicitly). No Django, no
engine config import — this module must stay importable anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = parent of this package (…/veda-platform).
PLATFORM_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent

# Local (no docker) data home for the sqlite store + artifacts + checkpoint.
# Git-ignored via this package's .gitignore.
DATA_HOME = Path(os.environ.get("VEDA_MLFLOW_DATA_HOME", PACKAGE_ROOT / "mlflow_data"))


def _default_trace_log() -> Path:
    """The engine writes logs/explain_trace.jsonl relative to its own cwd.
    Bare metal that cwd is veda_core/; in docker it is /app/veda_core (the repo
    is bind-mounted at /app), so the same repo-relative path holds everywhere.
    """
    return PLATFORM_ROOT / "veda_core" / "logs" / "explain_trace.jsonl"


def _default_tracking_uri() -> str:
    # Prefer the standard MLflow env var if the operator already set one
    # (e.g. a remote tracking server in production).
    std = os.environ.get("MLFLOW_TRACKING_URI")
    if std:
        return std
    db = DATA_HOME / "mlflow.db"
    return "sqlite:///" + db.as_posix()


@dataclass(frozen=True)
class Settings:
    trace_log: Path = field(
        default_factory=lambda: Path(os.environ.get("VEDA_TRACE_LOG", _default_trace_log())))
    tracking_uri: str = field(
        default_factory=lambda: os.environ.get("VEDA_MLFLOW_TRACKING_URI", _default_tracking_uri()))
    experiment: str = field(
        default_factory=lambda: os.environ.get("VEDA_MLFLOW_EXPERIMENT", "VEDA-Query-Observability"))
    # Only used for local file/sqlite stores; remote servers manage artifacts.
    artifact_location: str = field(
        default_factory=lambda: os.environ.get(
            "VEDA_MLFLOW_ARTIFACTS", (DATA_HOME / "artifacts").as_uri()
            if not os.environ.get("VEDA_MLFLOW_TRACKING_URI", os.environ.get("MLFLOW_TRACKING_URI", "")).startswith("http")
            else ""))
    checkpoint_path: Path = field(
        default_factory=lambda: Path(os.environ.get(
            "VEDA_MLFLOW_CHECKPOINT", DATA_HOME / "exporter_checkpoint.json")))
    poll_seconds: float = field(
        default_factory=lambda: float(os.environ.get("VEDA_MLFLOW_POLL_SECS", "5")))
    # Tagged onto every run so local vs production runs are separable in the UI.
    environment: str = field(
        default_factory=lambda: os.environ.get("VEDA_ENVIRONMENT", "local"))
    # Cap on how many chars of a param value we send (MLflow hard limit ~6000).
    param_value_max: int = 500


def load() -> Settings:
    s = Settings()
    if not str(s.tracking_uri).startswith(("http://", "https://")):
        DATA_HOME.mkdir(parents=True, exist_ok=True)
    s.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return s
