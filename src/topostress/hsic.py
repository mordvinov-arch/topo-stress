# HSIC — нелинейные зависимости.
# Гильберт-Шмидтов критерий независимости (Gretton et al. 2005) с RBF-ядрами,
# перестановочным тестом и медианным правилом для ширины ядра.

import numpy as np


def rbf_kernel(X, sigma=1.0):
    """RBF-ядро exp(-‖x-x'‖²/(2σ²)) для матрицы наблюдений X."""
    d = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(X ** 2, axis=1) - 2 * X @ X.T
    return np.exp(-d / (2 * sigma ** 2))


def median_bandwidth(x):
    """Медианное правило для ширины ядра (устойчиво к выбросам)."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    d = np.abs(x[:, None] - x[None, :])
    return float(np.median(d[d > 0])) if (d > 0).any() else 1.0


def center_kernel(K):
    """Центрирование ядерной матрицы H K H."""
    return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()


def hsic_centered(Kc, Lc):
    """Эмпирический HSIC по центрированным ядрам: tr(Kc Lc) / n²."""
    return float(np.sum(Kc * Lc) / Kc.shape[0] ** 2)


def hsic(X, Y, sigma_x=1.0, sigma_y=1.0):
    """Эмпирический HSIC с фиксированной шириной ядра."""
    n = len(X)
    K = rbf_kernel(X.reshape(-1, 1), sigma_x)
    L = rbf_kernel(Y.reshape(-1, 1), sigma_y)
    H = np.eye(n) - np.ones((n, n)) / n
    return np.trace(K @ H @ L @ H) / n ** 2


def hsic_test(X, Y, n_perm=3000, seed=42):
    """HSIC с перестановочным тестом (фиксированная ширина ядра). Возвращает (stat, p)."""
    rng = np.random.default_rng(seed)
    observed = hsic(X, Y)
    perm = []
    for _ in range(n_perm):
        Yp = Y[rng.permutation(len(Y))]
        perm.append(hsic(X, Yp))
    p = np.mean(np.array(perm) >= observed)
    return observed, p


def hsic_test_median(x, y, n_perm=2000, seed=42):
    """HSIC-тест с медианной шириной ядра. Возвращает (stat, p)."""
    rng = np.random.default_rng(seed)
    Kx = center_kernel(rbf_kernel(x.reshape(-1, 1), median_bandwidth(x)))
    Ly = center_kernel(rbf_kernel(y.reshape(-1, 1), median_bandwidth(y)))
    observed = hsic_centered(Kx, Ly)
    n = len(y)
    perm = np.zeros(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(n)
        perm[i] = hsic_centered(Kx, Ly[np.ix_(idx, idx)])
    return observed, float(np.mean(perm >= observed))
