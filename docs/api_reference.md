# API Reference

Пакет `topostress` (в `src/topostress/`). Импорт после установки
(`pip install -e .`) либо через `sys.path.insert(0, "src")`.

## topostress.config

Константы путей и параметров анализа.

- `PROJECT_ROOT`, `DATA_DIR`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`,
  `RESULTS_DIR`, `FIGURES_DIR`, `ARTICLE_DIR` — абсолютные пути.
- `RAW_MAST`, `PROC_WIDE`, `PROC_LONG_CORT`, `PROC_LONG_PSYCH`, `PROC_MMASH_LONG` —
  пути к конкретным файлам.
- `DRAWS = 4000`, `TUNE = 4000`, `TARGET_ACCEPT = 0.99`, `NONCENTERED = True` —
  параметры NUTS (переопределяются env `MAST_DRAWS`, `MAST_TUNE`).
- `N_PERM = 999`, `N_EPS = 200`, `N_BOOT = 500` — пермутации/сетка/бутстреп
  (переопределяются env `MAST_PERM`, `MAST_EPS`, `MAST_BOOT`).
- `FORCE` — флаг принудительного пересчёта.

## topostress.data

Сборка и загрузка датасетов.

- `pruessner_auc(m, dt=1.0)` → `(aucg, auci)` — AUC по Pruessner et al. 2003.
- `ols_slope(y)` — наклон линейной регрессии на равномерную сетку.
- `zscore(df, cols)` — стандартизация колонок `cols` (добавляет `*_z`).
- `build_wide(df)` → DataFrame — производные кортизоловые и психометрические признаки.
- `build_long_cortisol(w)` / `build_long_psych(w)` — длинные датасеты.
- `build_datasets(src=RAW_MAST, out_dir=PROCESSED_DATA_DIR)` → dict — пересчёт всех
  производных файлов.
- `load_wide(path=PROC_WIDE)`, `load_long_cortisol(path=PROC_LONG_CORT)` — загрузка.
- Константы: `CORTISOL_COLS`, `LOG_CORTISOL_COLS`, `FULL_17D_VARS`.

## topostress.topology

Топологический анализ (Vietoris–Rips, β0/β1).

- `betti_0_curve(X, epsilons)` → β0(ε) (union-find, O(n² log n)).
- `normalized_betti_curve(X, D, nbar, t_grid)` — β̄0(t) = β0(t·D)/n̄.
- `d_topo_normalized(X1, X2, n_eps=200)` → `(d, t_grid, β̄1, β̄2)` — нормированная
  топологическая дивергенция.
- `d_topo_raw(X1, X2, n_eps=100)` — сырая дивергенция по общей шкале ε.
- `d_combined(X1, X2, lam1=0.5, lam2=0.5, n_eps=200)` → `(d_comb, d_topo, d_mean)`.
- `persistence_bottleneck(X1, X2, maxdim=0, seed=42)` — bottleneck-дистанция
  (требует `persim`, `ripser`).
- `beta1_curve(X, epsilons)` — β1(ε) через ripser.

## topostress.bayesian

Байесовские иерархические модели (Bambi/PyMC).

- `MODEL_SPECS` — формулы M1/M2/M3.
- `EXCLUDED_STATEMENT` — текст о сходимости для случая исключения ковариат.
- `model_weight(comparison, name)` — BMA-вес по LOO (stacking) либо `exp(elpd/2)`.
- `decide_model(summaries)` → `(decision, primary, statement)` — правило выбора
  основной модели по числу расходимостей M3.

## topostress.evt

Теория экстремальных значений.

- `gev_return_level(xi, mu, sigma, p)` — уровень возврата GEV.
- `gpd_return_level(scale, xi, u, n, p)` — уровень возврата GPD (POT).
- `tail_type(xi)` — классификация хвоста (Weibull/Gumbel/Frechet).

## topostress.fda

Функциональный анализ данных.

- `N_GRID = 100`, `T_RAW = [0..4]`.
- `interp_curves(mat)` → `(grid, out)` — интерполяция строк на общую сетку.
- `pointwise_stats(y, x)` → `(r, p)` — поточечные point-biserial корреляции.
- `max_t_stat(y, x)` — maxT-статистика (макс. `-log10 p`).

## topostress.rmt

Случайные матрицы.

- `marchenko_pastur_bound(p, n, sigma=1.0)` — λ+.
- `correlation_spectrum(X)` — спектр корреляционной матрицы.

## topostress.hsic

HSIC и нелинейные зависимости.

- `rbf_kernel(X, sigma=1.0)` — RBF-ядро.
- `median_bandwidth(x)` — медианное правило ширины ядра.
- `center_kernel(K)` — центрирование `H K H`.
- `hsic_centered(Kc, Lc)` — эмпирический HSIC по центрированным ядрам.
- `hsic(X, Y, sigma_x=1.0, sigma_y=1.0)` — HSIC с фиксированной шириной.
- `hsic_test(X, Y, n_perm=3000, seed=42)` → `(stat, p)` — тест с перестановками.
- `hsic_test_median(x, y, n_perm=2000, seed=42)` → `(stat, p)` — с медианной шириной.

## topostress.conformal

Конформное предсказание.

- `conformal_intervals(X, y, alpha=0.1, model=None, seed=42)` → `(lower, upper, q_hat, coverage)` —
  Leave-One-Out вариант.
- `split_conformal(X, y, alpha=0.1, seed=42, frac_cal=0.35)` → `(lower, upper, q_hat, cal_idx, model)`.

## topostress.info_geometry

Информационная геометрия.

- `wasserstein_matrix(samples)` — попарные Вассерштейны.
- `mds(D, n_components=2, seed=42)` → `(embedding, model)`.
- `ward_clusters(D, n_clusters=None)` → метки (или дендрограмма `Z`).

## topostress.utils

Вспомогательные статистики.

- `permutation_test(statistic_fn, X1, X2, n_perm=5000, seed=42)` → `(observed, p, perms)`.
- `fisher_combine(p_values)` — объединение p-значений методом Фишера.
- `pearson_test(X, Y)` → `(r, p)`.
