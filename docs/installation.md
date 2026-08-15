# Установка

## Требования

- Python **3.10+** (разработка и проверка — на Python 3.14 / Windows)
- pip

## Шаги

```bash
# 1. Клонирование
git clone <repo-url> topo-stress
cd topo-stress

# 2. (рекомендуется) виртуальное окружение
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Зависимости
pip install -r requirements.txt

# 4. (опционально) установить как пакет — тогда `import topostress`
#    работает из любой директории
pip install -e .
```

## Проверка установки

```bash
python -c "import topostress; print(topostress.__version__)"
python scripts/run_data.py          # пересоберёт данные в data/processed
python -m pytest tests/             # если установлен pytest
```

## Зависимости (requirements.txt)

| Пакет | Назначение |
|---|---|
| numpy, scipy, pandas | базовые вычисления и таблицы |
| scikit-learn | стандартизация, MDS, модели для конформного предсказания |
| matplotlib | все рисунки |
| bambi / pymc / arviz | байесовские иерархические модели, LOO/BMA |
| ripser, persim | персистентная гомология, bottleneck-дистанция |
| networkx | корреляционная сеть (EVT) |
| python-docx | сборка статьи в docx |

> Установка `bambi`/`pymc` может занять несколько минут (тянут numba,
> aeon, pydantic и др.).

## Примечания для Windows

- Для корректного вывода кириллицы используйте `$env:PYTHONIOENCODING='utf-8'`.
- При импорте `ripser` может печатать предупреждение
  `g++ not available...` — это безвредно (используется numba-JIT).
