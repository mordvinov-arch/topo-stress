# Сборка производных датасетов MAST.
# Источник: data/raw/MAST_371_2025111002_Raw_Recode.csv -> data/processed/
#   mast_wide.csv, mast_long_cortisol.csv, mast_long_psych.csv,
#   mast_17d.csv, mast_cortisol_5d.csv

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import RAW_MAST, PROCESSED_DATA_DIR
from topostress.data import build_datasets


def main():
    d = build_datasets(RAW_MAST, PROCESSED_DATA_DIR)
    wide = d["wide"]
    print(f"Исходных строк: {len(wide)}, колонок: {len(wide.columns)} (0 пропусков)")
    print(f"mast_wide.csv:          {len(wide)} x {len(wide.columns)}")
    print(f"mast_long_cortisol.csv: {len(d['long_cortisol'])} x {len(d['long_cortisol'].columns)}")
    print(f"mast_long_psych.csv:    {len(d['long_psych'])} x {len(d['long_psych'].columns)}")

    print("\nСводка по группам (wide):")
    g = wide.groupby("Group")[["Cortisol_peak", "AUCg", "AUCi",
                               "Slope_logcortisol", "Anxiety_peak", "Negative_peak"]].mean().round(3)
    print(g.to_string())
    print("\nРаспределение по экспериментам и группам:")
    print(wide.groupby(["Exp", "Group"]).size().to_string())
    print("\nГотово. Файлы в", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
