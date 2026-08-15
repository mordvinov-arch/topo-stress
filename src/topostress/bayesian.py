# Байесовские иерархические модели (Bambi / PyMC).
# Спецификации моделей M1/M2/M3, LOO/BMA-веса и предзарегистрированное правило
# выбора основной модели:
#   * расходимостей в M3 нет  -> ковариаты остаются, основная M3;
#   * иначе                    -> ковариаты исключаются, основная M2.

import numpy as np

from topostress.config import DRAWS, TUNE, TARGET_ACCEPT, NONCENTERED

MODEL_SPECS = {
    "M1_Time_x_Group": "LogCortisol ~ Time * Group01",
    "M2_Hier_Subj_Exp": "LogCortisol ~ Time * Group01 + (Time|SubID) + (1|Exp)",
    "M3_Plus_covariates": ("LogCortisol ~ Time * Group01 + Trait_Anxiety_z + "
                           "BMI_z + Age_z + Sex + (Time|SubID) + (1|Exp)"),
}

EXCLUDED_STATEMENT = (
    "Из-за проблем со сходимостью при добавлении ковариат, "
    "мы сообщаем результаты M2 как основные; ковариаты не "
    "улучшили предсказание и будут изучены в дальнейшем."
)


def model_weight(comparison, name):
    """BMA-вес модели по LOO (stacking) либо по exp(elpd/2)."""
    if "weight" in comparison.columns:
        return float(comparison.loc[name, "weight"])
    elpd = comparison["elpd"].values
    w = np.exp(0.5 * (elpd - elpd.max()))
    return float(w / w.sum())


def decide_model(summaries):
    """Правило выбора основной модели по числу расходимостей M3.

    Возвращает (decision, primary, statement).
    """
    div_m3 = summaries["M3_Plus_covariates"]["divergences"]
    if div_m3 == 0:
        return "M3_kept", "M3_Plus_covariates", None
    return "M2_primary_covariates_excluded", "M2_Hier_Subj_Exp", EXCLUDED_STATEMENT
