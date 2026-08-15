# ===== Случайные матрицы (RMT) =====
# Граница Марченко–Пастура и спектр корреляционной матрицы.

import numpy as np


def marchenko_pastur_bound(p, n, sigma=1.0):
    """Верхняя граница Марченко–Пастура λ+ = σ²(1 + √(p/n))²."""
    q = p / n
    return sigma ** 2 * (1 + np.sqrt(q)) ** 2


def correlation_spectrum(X):
    """Отсортированный по убыванию спектр корреляционной матрицы переменных."""
    C = np.corrcoef(X.T)
    return np.sort(np.linalg.eigvalsh(C))[::-1]
