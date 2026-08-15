# ===== Конфигурация проекта =====
# Все пути вычисляются относительно корня репозитория, поэтому скрипты можно
# запускать из любой директории. Параметры анализа переопределяются переменными
# окружения (MAST_PERM, MAST_DRAWS, MAST_TUNE, MAST_BOOT, FORCE).

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
ARTICLE_DIR = os.path.join(PROJECT_ROOT, "article")

RAW_MAST = os.path.join(RAW_DATA_DIR, "MAST_371_2025111002_Raw_Recode.csv")
PROC_WIDE = os.path.join(PROCESSED_DATA_DIR, "mast_wide.csv")
PROC_LONG_CORT = os.path.join(PROCESSED_DATA_DIR, "mast_long_cortisol.csv")
PROC_LONG_PSYCH = os.path.join(PROCESSED_DATA_DIR, "mast_long_psych.csv")
PROC_MMASH_LONG = os.path.join(PROCESSED_DATA_DIR, "MMASH_long.csv")

# Байесовский анализ (env-переопределяемые)
DRAWS = int(os.environ.get("MAST_DRAWS", 4000))
TUNE = int(os.environ.get("MAST_TUNE", 4000))
TARGET_ACCEPT = 0.99
NONCENTERED = True

# Пермутационные тесты и бутстреп (env-переопределяемые)
N_PERM = int(os.environ.get("MAST_PERM", 999))
N_EPS = int(os.environ.get("MAST_EPS", 200))
N_BOOT = int(os.environ.get("MAST_BOOT", 500))

FORCE = os.environ.get("FORCE", "0") == "1"
