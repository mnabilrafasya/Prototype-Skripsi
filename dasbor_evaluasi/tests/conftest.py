"""Fixture bersama untuk tes: artefak dan data uji asli (hanya dibaca)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.artifacts import Artefak, muat_artefak
from core.data import DataUji, muat_data_uji

AKAR_PROYEK = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def artefak() -> Artefak:
    return muat_artefak(AKAR_PROYEK / "artefak")


@pytest.fixture(scope="session")
def data_uji(artefak: Artefak) -> DataUji:
    return muat_data_uji(AKAR_PROYEK / "data_uji", artefak.manifest)
