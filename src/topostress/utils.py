# ===== Вспомогательные статистические инструменты =====
# Общий пермутационный тест, объединение p-значений Фишером, линейная корреляция.

import numpy as np


def permutation_test(statistic_fn, X1, X2, n_perm=5000, seed=42, n_eps_perm=50):
    """Общий пермутационный тест: H0 = группы не различаются.

    Возвращает (observed, p, perms).
    """
    rng = np.random.default_rng(seed)
    observed = statistic_fn(X1, X2)
    X = np.vstack([X1, X2])
    n1 = len(X1)
    perms = np.zeros(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(X))
        perms[i] = statistic_fn(X[idx[:n1]], X[idx[n1:]])
    p = np.mean(perms >= observed)
    return observed, p, perms


def fisher_combine(p_values):
    """Мета-анализ: объединение p-значений методом Фишера."""
    import scipy.stats as st

    stat = -2 * np.sum(np.log(np.clip(p_values, 1e-12, 1)))
    df = 2 * len(p_values)
    return 1 - st.chi2.cdf(stat, df)


def pearson_test(X, Y):
    """Корреляция Пирсона: (r, p)."""
    from scipy.stats import pearsonr

    r, p = pearsonr(X, Y)
    return r, p
