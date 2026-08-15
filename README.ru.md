# topo-stress

**English** · [Deutsch](README.de.md) · [Русский](README.ru.md)

> Русская версия описания находится здесь; английская — в [README.md](README.md).

Мультиметодный байесовско-топологический анализ острой лабораторной стресс-пробы **MAST** (n = 371).

Проект объединяет **девять аналитических методов** на одних данных и в одном воспроизводимом конвейере: топологический анализ данных (TDA), теория экстремальных значений (EVT), функциональный анализ данных (FDA), байесовские иерархические модели, случайные матрицы (RMT), HSIC-критерий нелинейных зависимостей, конформное предсказание, информационная геометрия и новые топологические функционалы.

## Ключевые результаты (MAST)

| Метод | Результат |
|---|---|
| TDA (d̄_topo) | Stress vs Control значимо разделяются, p < 0.002 (500 пермутаций); полный 17D-набор: d̄_topo = 0.102 |
| Байесовская модель M3 | Time×Group = +0.054 [0.029, 0.079], 0 расходимостей, R̂ = 1.00; BMA-вес 0.87 против неиерархической M1 (−1300) |
| EVT | Frechet-хвосты; пермутационный тест RL99: p < 0.001 |
| FDA | maxT-пермутация: p = 0.002 |
| HSIC | Anxiety_peak p = 0.036 (Pearson p = 0.123) — нелинейная связь, недоступная линейным методам |
| Инф. геометрия | 3 физиотипа, включая «супер-реактивных» (22 испытуемых, 100% Stress, LogAUCg 6.41 vs 3.08), χ² = 14.36, p = 0.0008 |

Полное описание — в `article/mast_article.md` (и `article/mast_article.docx`).

## Структура

```
topo-stress/
├── data/
│   ├── raw/             # MAST_371_2025111002_Raw_Recode.csv (371 испытуемый)
│   └── processed/       # mast_wide.csv, mast_long_cortisol.csv, mast_long_psych.csv, ...
├── src/topostress/      # Пакет: topology, bayesian, evt, fda, rmt, hsic, conformal,
│                        #        info_geometry, data, utils, config
├── scripts/             # Запускаемые скрипты (конвейер): run_all.py, run_tda.py, ...
├── figures/             # Все рисунки для статьи
├── results/             # JSON/CSV результаты
├── article/             # Статья (markdown + docx)
├── notebooks/           # Jupyter для разведки
├── tests/               # Unit-тесты
└── docs/                # Документация
```

## Установка

```bash
# Python 3.10+
pip install -r requirements.txt
pip install -e .            # опционально: установить как пакет
```

Подробнее: [docs/installation.md](docs/installation.md).

## Запуск

```bash
python scripts/run_all.py          # весь конвейер (с кэшем)
FORCE=1 python scripts/run_all.py  # пересчёт всего
python scripts/run_tda.py          # только TDA
python scripts/run_bayesian.py     # только байесовский анализ
```

Скрипты пропускаются, если их результат уже есть в `results/`. Параметры
анализа переопределяются переменными окружения: `MAST_PERM`, `MAST_DRAWS`,
`MAST_TUNE`, `MAST_BOOT`, `FORCE`. Подробнее: [docs/usage.md](docs/usage.md).

## Данные

- **MAST**: `MAST_371_2025111002_Raw_Recode.csv` — острая лабораторная
  стресс-проба, 371 испытуемый (240 Stress / 131 Control), кортизол (5 сэмплов),
  психометрика, 6 экспериментальных протоколов.
- **MMASH** (используется только для регрессионной проверки переносимости
  метрик): `data/processed/MMASH_long.csv` (производный файл; сырые данные и
  лицензия — см. [data/README.md](data/README.md)).

> ВАЖНО: идентификатор испытуемого в MAST — колонка `SubID` (371 уровень).
> Колонка `Subject` повторяется между экспериментами и относится к разным людям,
> поэтому как группировка она не используется.

## Лицензия

MIT — см. [LICENSE](LICENSE). Данные MAST/MMASH распространяются по их
собственным лицензиям (см. [data/README.md](data/README.md)).
