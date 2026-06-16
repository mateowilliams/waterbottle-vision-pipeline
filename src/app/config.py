"""Central app configuration, read from environment variables.

No secrets are hardcoded: everything comes from the environment (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

# Project root: .../proyecto2/
ROOT = Path(__file__).resolve().parents[2]


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/bottles")

MODEL_ARTIFACT = _resolve(os.getenv("MODEL_ARTIFACT", "artifacts/waterbottle_resnet18_v1.pt"))
REFERENCE_STATS = _resolve(os.getenv("REFERENCE_STATS", "artifacts/reference_stats.json"))

# Threshold below which a prediction is flagged as low-confidence in monitoring.
LOW_CONF_THRESHOLD = float(os.getenv("LOW_CONF_THRESHOLD", "0.60"))

WEB_DIR = ROOT / "web"
UPLOAD_DIR = ROOT / "data" / "uploads"
