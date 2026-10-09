"""Shared pytest fixtures for the TinyMalariaNet suite.

Adds the repository root to ``sys.path`` so ``import src.*`` / ``import app.*``
work regardless of the directory pytest was started from, and caches the
expensive fixtures (trained artifacts, segmentation runs) for the session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def sample_frame(repo_root: Path) -> Path:
    frames = sorted((repo_root / "data" / "phone_test").glob("*.jpg"))
    if not frames:
        pytest.skip("no sample frames in data/phone_test")
    return frames[0]


@pytest.fixture(scope="session")
def sample_records(repo_root: Path):
    """Records for the synthetic mock crops shipped with the repo."""
    from src.make_sample_data import ensure_sample_dataset

    return ensure_sample_dataset(root=repo_root / "data")


@pytest.fixture(scope="session")
def onnx_model(repo_root: Path) -> Path | None:
    path = repo_root / "models" / "tinymalaria_2.1mb_int8.onnx"
    return path if path.exists() else None


@pytest.fixture(scope="session")
def stage2_checkpoint(repo_root: Path) -> Path | None:
    path = repo_root / "checkpoints" / "stage2" / "best.pt"
    return path if path.exists() else None
