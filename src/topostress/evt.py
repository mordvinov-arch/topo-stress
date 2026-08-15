# ===== Теория экстремальных значений (EVT) =====
# GEV- и GPD-уровни возврата, классификация хвостов (Weibull/Gumbel/Frechet).

import numpy as np


def gev_return_level(xi, mu, sigma, p):
    """Уровень возврата GEV: значение, превышаемое с вероятностью p."""
    if abs(xi) < 1e-9:
        return mu - sigma * np.log(-np.log(1 - p))
    return mu - sigma / xi * (1 - (-np.log(1 - p)) ** (-xi))


def gpd_return_level(scale, xi, u, n, p):
    """Уровень возврата GPD: превышение порога u при n наблюдениях, вероятность p."""
    if abs(xi) < 1e-9:
        return u + scale * np.log(n * p)
    return u + scale / xi * ((n * p) ** xi - 1)


def tail_type(xi):
    """Интерпретация параметра формы xi."""
    if xi < -0.2:
        return "Weibull (лёгкий хвост, ограниченный сверху)"
    if xi < 0.2:
        return "Gumbel (экспоненциальный хвост)"
    return "Frechet (тяжёлый хвост, склонность к экстремальным пикам)"
