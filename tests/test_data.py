# Тесты сборки и загрузки датасетов MAST (требуют файлов в data/).

import os

import pandas as pd
import pytest

from topostress import data
from topostress.config import PROCESSED_DATA_DIR

REQUIRED_FILES = [
    "mast_wide.csv",
    "mast_long_cortisol.csv",
    "mast_long_psych.csv",
]


def _files_present():
    return all(os.path.exists(os.path.join(PROCESSED_DATA_DIR, f))
               for f in REQUIRED_FILES)


pytestmark = pytest.mark.skipif(
    not _files_present(),
    reason="производные датасеты не собраны: запустите scripts/run_data.py",
)


def test_wide_shape_and_no_nan():
    w = data.load_wide()
    assert len(w) == 371
    assert w.isna().sum().sum() == 0
    assert "SubID" in w.columns
    assert "Cortisol_peak" in w.columns


def test_subid_unique():
    w = data.load_wide()
    assert w["SubID"].nunique() == 371


def test_long_cortisol_shape():
    long_c = data.load_long_cortisol()
    assert len(long_c) == 371 * 5
    assert set(long_c.columns) == {
        "Subject", "SubID", "Exp", "Group", "Sex", "BMI", "Age",
        "Trait_Anxiety", "Time", "Cortisol", "LogCortisol"}


def test_build_is_idempotent():
    import numpy as np

    w = data.load_wide()
    rebuilt = data.build_wide(w)
    assert len(rebuilt) == 371
    assert np.allclose(rebuilt["Cortisol_peak"], w["Cortisol_peak"])
    assert np.allclose(rebuilt["LogAUCg"], w["LogAUCg"])


def test_17d_var_count():
    assert len(data.FULL_17D_VARS) == 17
    w = data.load_wide()
    missing = [v for v in data.FULL_17D_VARS if v not in w.columns]
    assert missing == []
