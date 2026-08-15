# Конформное предсказание.
# Два варианта конформных интервалов: Leave-One-Out (общий) и split-conformal.

import numpy as np


def conformal_intervals(X, y, alpha=0.1, model=None, seed=42):
    """Конформное предсказание (Leave-One-Out). Возвращает интервалы и покрытие."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut

    if model is None:
        model = LinearRegression
    n = len(y)
    scores = np.zeros(n)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        m = model().fit(X[train_idx], y[train_idx])
        scores[test_idx[0]] = abs(y[test_idx[0]] - m.predict(X[test_idx])[0])
    q_hat = np.quantile(scores, 1 - alpha)
    final = model().fit(X, y)
    preds = final.predict(X)
    lower = preds - q_hat
    upper = preds + q_hat
    coverage = np.mean((y >= lower) & (y <= upper))
    return lower, upper, q_hat, coverage


def split_conformal(X, y, alpha=0.1, seed=42, frac_cal=0.35):
    """Split-conformal (Vovk et al.): калибровочный сплит, квантиль абс. ошибок.

    Возвращает (lower, upper, q_hat, cal_idx, model).
    """
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_cal = int(n * frac_cal)
    cal_idx, tr_idx = idx[:n_cal], idx[n_cal:]
    m = LinearRegression().fit(X[tr_idx], y[tr_idx])
    pred_cal = m.predict(X[cal_idx])
    scores = np.abs(y[cal_idx] - pred_cal)
    q_hat = np.quantile(scores, 1 - alpha, method="higher")
    pred_all = m.predict(X)
    lower = pred_all - q_hat
    upper = pred_all + q_hat
    return lower, upper, q_hat, cal_idx, m
