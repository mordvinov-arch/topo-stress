# Функциональный анализ данных (FDA).
# Интерполяция кривых на общую сетку, поточечная функциональная регрессия
# на бинарный предиктор, maxT-пермутационная статистика.

import numpy as np
from scipy.interpolate import interp1d
from scipy import stats

N_GRID = 100
T_RAW = np.arange(5, dtype=float)


def interp_curves(mat):
    """Линейная интерполяция строк mat (n×5) на сетку из N_GRID точек."""
    grid = np.linspace(0, 4, N_GRID)
    out = np.zeros((mat.shape[0], N_GRID))
    for i in range(mat.shape[0]):
        f = interp1d(T_RAW, mat[i], kind="linear", bounds_error=False,
                     fill_value="extrapolate")
        out[i] = f(grid)
    return grid, out


def pointwise_stats(y, x):
    """Поточечные point-biserial корреляции r(t) и p(t) кривых y с бинарным x."""
    r = np.zeros(N_GRID)
    p = np.zeros(N_GRID)
    for j in range(N_GRID):
        r[j], p[j] = stats.pointbiserialr(x, y[:, j])
    return r, p


def max_t_stat(y, x):
    """maxT-статистика: максимум -log10 p по сетке."""
    p = np.zeros(N_GRID)
    for j in range(N_GRID):
        _, p[j] = stats.pointbiserialr(x, y[:, j])
    return np.max(-np.log10(p + 1e-300))
