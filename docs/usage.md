# Использование

Все команды выполняются из корня репозитория.

## Полный конвейер

```bash
python scripts/run_all.py
```

Конвейер по шагам:

| Шаг | Скрипт | Результат |
|---|---|---|
| Данные | `run_data.py` | `data/processed/*.csv` |
| Регрессионный тест MMASH | `run_repro_mmash.py` | `results/repro_mmash_done.flag` |
| Байесовские модели M1/M2/M3 | `run_bayesian.py` | `results/bayesian_mast.json`, `figures/bayesian_mast.png` |
| EVT (GEV/GPD, RL99) | `run_evt.py` | `results/evt_mast.json`, `figures/evt_mast.png` |
| FDA | `run_fda.py` | `results/fda_mast.json`, `figures/fda_mast.png` |
| RMT (Марченко–Пастур) | `run_rmt.py` | `results/rmt_mast.json`, `figures/rmt_mast.png` |
| HSIC | `run_hsic.py` | `results/hsic_mast.json`, `figures/hsic_mast.png` |
| Конформное предсказание | `run_conformal.py` | `results/conformal_mast.json`, `figures/conformal_mast.png` |
| Информационная геометрия | `run_info_geometry.py` | `results/info_geometry_mast.json`, `figures/info_geometry_mast.png` |
| TDA | `run_tda.py` | `results/tda_mast.json`, `figures/tda_mast.png` |
| Новые функционалы | `run_novelty.py` | `results/novelty_mast.json`, `figures/novelty_mast.png` |
| Сводные CSV | `export_summaries.py` | `results/bayesian_summary.csv`, `hsic_results.csv`, `physiotypes.csv` |
| Статья docx | `build_article_docx.py` | `article/mast_article.docx` |

## Кэширование и перезапуск

Каждый шаг пропускается, если его артефакт уже существует в `results/` (или
`data/processed/`). Чтобы пересчитать всё принудительно:

```bash
FORCE=1 python scripts/run_all.py     # Windows PowerShell
FORCE=1 python scripts/run_all.py     # bash тоже (переменная окружения)
```

Чтобы пересчитать только один анализ:

```bash
python scripts/run_tda.py
```

> ВАЖНО: повторный запуск перезаписывает `results/*.json` и `figures/*.png`.
> Настоящие результаты статьи воспроизводятся параметрами по умолчанию
> (`MAST_PERM=999`, `MAST_DRAWS=4000`, `MAST_TUNE=4000`, `MAST_BOOT=500`).

## Параметры (переменные окружения)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `MAST_PERM` | 999 | число пермутаций (EVT, FDA, HSIC, TDA, novelty) |
| `MAST_EPS` | 200 | размер сетки ε для d̄_topo |
| `MAST_DRAWS` | 4000 | пост-выборки NUTS (после прогрева) |
| `MAST_TUNE` | 4000 | прогрев NUTS |
| `MAST_BOOT` | 500 | бутстреп-повторов в информационной геометрии |
| `FORCE` | 0 | 1 = пересчитать всё независимо от кэша |

Пример быстрой проверки (малый объём):

```bash
MAST_PERM=50 MAST_BOOT=20 python scripts/run_tda.py
```

> Примечание: при `MAST_DRAWS < ~50` psis-LOO может не вычислить ELPD —
> это ограничение малого объёма выборки, а не ошибка.

## Воспроизводимость

- Фиксированные seed для всех ГПСЧ (`random_seed=42` в NUTS, `np.random.default_rng`
  с фиксированными seed в пермутациях/бутстрепе).
- `target_accept = 0.99`, нецентрированная параметризация случайных эффектов,
  группировка по `SubID` (371 уровень).
- Правило выбора основной модели задокументировано в `src/topostress/bayesian.py`:
  если в M3 нет расходимостей — ковариаты остаются (основная M3); иначе —
  ковариаты исключаются (основная M2).
- Изменение кода анализов не меняет уже сохранённые результаты, пока вы не
  запустите скрипты с `FORCE=1`.

## Работа с пакетом в своём коде

```python
import sys
sys.path.insert(0, "src")          # если не установлен pip install -e .

from topostress import data, topology, hsic, info_geometry

wide = data.load_wide()
X = wide[["LogCortisol_01", "LogCortisol_02"]].values
d, t, b1, b2 = topology.d_topo_normalized(X[:50], X[50:100])
hs, p = hsic.hsic_test_median(X[:100, 0], X[100:200, 0], n_perm=100)
```

Подробная справка по функциям — в [docs/api_reference.md](api_reference.md).
