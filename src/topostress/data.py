# ===== Сборка и загрузка датасетов MAST =====
# Источник: MAST_371_2025111002_Raw_Recode.csv (371 испытуемый, чистая версия).
# Экспериментальный дизайн: острая лабораторная стресс-проба (TSST-подобная).
#   Кортизол: 5 сэмплов (01=базал, 02..05 — реакция/восстановление), плюс log-версии.
#   Психометрика: State anxiety, Positive/Negative affect в 4 точках (02..05).
#   Группы: Stress (240) vs Control (131); эксперименты HS01, HS02, FLH01, FLH02, LLQ00, YR00.
#
# Примечание о времени сэмплов: точные минуты в файле отсутствуют,
# поэтому AUC вычисляется в предположении равномерной дискретизации
# (единичный интервал между соседними сэмплами) — формула Прусснера et al. 2003.
#
# ВАЖНО: идентификатор испытуемого — колонка SubID (371 уникальных значений).
# Колонка Subject повторяется между экспериментами (FLH01/FLH02) и относится
# к РАЗНЫМ людям, поэтому в качестве группировки не используется.

import os

import numpy as np
import pandas as pd

from topostress.config import (
    RAW_MAST,
    PROCESSED_DATA_DIR,
    PROC_WIDE,
    PROC_LONG_CORT,
    PROC_LONG_PSYCH,
)

CORTISOL_COLS = [f"Cortisol_{i:02d}" for i in range(1, 6)]
LOG_CORTISOL_COLS = [f"LogCortisol_{i:02d}" for i in range(1, 6)]
ANX_COLS = [f"State_anxiety_{i:02d}" for i in range(2, 6)]
POS_COLS = [f"Positive_{i:02d}" for i in range(2, 6)]
NEG_COLS = [f"Negative_{i:02d}" for i in range(2, 6)]

# Наборы переменных для TDA: 5D (кортизол) и 17D (кортизол + психометрика)
FULL_17D_VARS = LOG_CORTISOL_COLS + ANX_COLS + NEG_COLS + POS_COLS


def pruessner_auc(m: np.ndarray, dt: float = 1.0):
    """AUCg и AUCi по формуле Pruessner et al. 2003, равномерные интервалы dt."""
    aucg = np.trapezoid(m, dx=dt)
    auci = aucg - m[0] * (len(m) - 1) * dt
    return aucg, auci


def ols_slope(y: np.ndarray):
    """Наклон линейной регрессии y ~ x (x = 0..len-1)."""
    x = np.arange(len(y), dtype=float)
    xm, ym = x.mean(), y.mean()
    return np.dot(x - xm, y - ym) / np.dot(x - xm, x - xm)


def zscore(df: pd.DataFrame, cols):
    """Стандартизация выбранных колонок (внтури таблицы, колонки *_z)."""
    for c in cols:
        df[f"{c}_z"] = (df[c] - df[c].mean()) / df[c].std()
    return df


def build_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Сборка широкого датасета: производные кортизоловые и психометрические признаки."""
    w = df.copy()

    w["Cortisol_peak"] = w[CORTISOL_COLS].max(axis=1)
    w["Cortisol_peak_time"] = w[CORTISOL_COLS].values.argmax(axis=1)
    w["Cortisol_increase"] = w["Cortisol_peak"] - w["Cortisol_01"]

    aucg, auci, log_aucg, log_auci = [], [], [], []
    slope_raw, slope_log = [], []
    for _, row in w.iterrows():
        m = row[CORTISOL_COLS].values.astype(float)
        lm = row[LOG_CORTISOL_COLS].values.astype(float)
        g, i = pruessner_auc(m)
        lg, li = pruessner_auc(lm)
        aucg.append(g)
        auci.append(i)
        log_aucg.append(lg)
        log_auci.append(li)
        slope_raw.append(ols_slope(m))
        slope_log.append(ols_slope(lm))

    w["AUCg"] = aucg
    w["AUCi"] = auci
    w["LogAUCg"] = log_aucg
    w["LogAUCi"] = log_auci
    w["Slope_cortisol"] = slope_raw
    w["Slope_logcortisol"] = slope_log

    w["Anxiety_peak"] = w[ANX_COLS].max(axis=1)
    w["Anxiety_change"] = w["State_anxiety_05"] - w["State_anxiety_02"]
    w["Negative_peak"] = w[NEG_COLS].max(axis=1)
    w["Positive_peak"] = w[POS_COLS].max(axis=1)

    return w


def build_long_cortisol(w: pd.DataFrame) -> pd.DataFrame:
    """Длинный датасет кортизола: 5 строк на испытуемого (Time = 0..4)."""
    rows = []
    for _, r in w.iterrows():
        for t in range(5):
            rows.append({
                "Subject": r["Subject"],
                "SubID": r["SubID"],
                "Exp": r["Exp"],
                "Group": r["Group"],
                "Sex": r["Gender"],
                "BMI": r["BMI"],
                "Age": r["Age"],
                "Trait_Anxiety": r["Trait_Anxiety"],
                "Time": t,
                "Cortisol": r[f"Cortisol_{t + 1:02d}"],
                "LogCortisol": r[f"LogCortisol_{t + 1:02d}"],
            })
    return pd.DataFrame(rows)


def build_long_psych(w: pd.DataFrame) -> pd.DataFrame:
    """Длинный датасет психометрики: 4 строки на испытуемого (Time = 0..3)."""
    rows = []
    for _, r in w.iterrows():
        for t in range(4):
            rows.append({
                "Subject": r["Subject"],
                "SubID": r["SubID"],
                "Exp": r["Exp"],
                "Group": r["Group"],
                "Sex": r["Gender"],
                "BMI": r["BMI"],
                "Age": r["Age"],
                "Trait_Anxiety": r["Trait_Anxiety"],
                "Time": t,
                "State_anxiety": r[f"State_anxiety_{t + 2:02d}"],
                "Positive": r[f"Positive_{t + 2:02d}"],
                "Negative": r[f"Negative_{t + 2:02d}"],
            })
    return pd.DataFrame(rows)


def build_datasets(src: str = RAW_MAST, out_dir: str = PROCESSED_DATA_DIR):
    """Полный пересчёт производных датасетов. Возвращает словарь DataFrame."""
    df = pd.read_csv(src)
    wide = build_wide(df)
    assert len(wide) == 371
    assert wide.isna().sum().sum() == 0

    long_cort = build_long_cortisol(wide)
    long_psych = build_long_psych(wide)

    os.makedirs(out_dir, exist_ok=True)
    wide.to_csv(os.path.join(out_dir, "mast_wide.csv"), index=False)
    long_cort.to_csv(os.path.join(out_dir, "mast_long_cortisol.csv"), index=False)
    long_psych.to_csv(os.path.join(out_dir, "mast_long_psych.csv"), index=False)

    # Удобные наборы переменных для TDA (значения без стандартизации)
    wide[FULL_17D_VARS + ["Group"]].to_csv(
        os.path.join(out_dir, "mast_17d.csv"), index=False)
    wide[CORTISOL_COLS + ["Group"]].to_csv(
        os.path.join(out_dir, "mast_cortisol_5d.csv"), index=False)

    return {"wide": wide, "long_cortisol": long_cort, "long_psych": long_psych}


def load_wide(path: str = PROC_WIDE) -> pd.DataFrame:
    return pd.read_csv(path)


def load_long_cortisol(path: str = PROC_LONG_CORT) -> pd.DataFrame:
    return pd.read_csv(path)
