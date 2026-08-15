# Тесты вспомогательных статистик и построения признаков.

import numpy as np
import pandas as pd
import pytest

from topostress import utils, data
from topostress import rmt, hsic


def test_pruessner_auc_linear_ramp():
    m = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    aucg, auci = data.pruessner_auc(m)
    assert np.isclose(aucg, 8.0)            # трапеции: 0.5+1.5+2.5+3.5
    assert np.isclose(auci, 8.0)            # m0 * (n-1) = 0


def test_pruessner_auc_flat():
    m = np.full(5, 2.0)
    aucg, auci = data.pruessner_auc(m)
    assert np.isclose(aucg, 8.0)
    assert np.isclose(auci, 0.0)


def test_ols_slope():
    y = 2.0 * np.arange(5, dtype=float)
    assert np.isclose(data.ols_slope(y), 2.0)


def test_zscore():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})
    data.zscore(df, ["v"])
    z = df["v_z"]
    assert np.isclose(z.mean(), 0.0, atol=1e-12)
    assert np.isclose(z.std(ddof=1), 1.0, atol=1e-12)  # стандартизация по выборке


def test_marchenko_pastur_bound():
    # q = 0.1: lambda+ = (1 + sqrt(0.1))^2
    lam = rmt.marchenko_pastur_bound(10, 100)
    assert np.isclose(lam, (1 + np.sqrt(0.1)) ** 2, atol=1e-12)


def test_correlation_spectrum_shape_and_order():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 6))
    spec = rmt.correlation_spectrum(X)
    assert spec.shape == (6,)
    assert np.all(np.diff(spec) <= 1e-12)  # по убыванию


def test_hsic_test_same_vs_independent():
    rng = np.random.default_rng(1)
    n = 60
    same_x = np.arange(n, dtype=float)
    hs_same, p_same = hsic.hsic_test(same_x, same_x, n_perm=100, seed=1)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    hs_ind, p_ind = hsic.hsic_test(x, y, n_perm=100, seed=1)
    assert p_same <= p_ind


def test_permutation_test_constant_statistic():
    def stat(X1, X2):
        return 1.0

    X1 = np.zeros((10, 2))
    X2 = np.zeros((10, 2))
    obs, p, perms = utils.permutation_test(stat, X1, X2, n_perm=20, seed=0)
    assert obs == 1.0
    assert p == 1.0


def test_fisher_combine_bounds_and_scale():
    p_small = utils.fisher_combine([0.001, 0.01])
    p_big = utils.fisher_combine([0.5, 0.6])
    assert 0.0 <= p_small <= 1.0
    assert 0.0 <= p_big <= 1.0
    assert p_small < p_big


def test_pearson_test():
    x = np.arange(20, dtype=float)
    r, p = utils.pearson_test(x, x)
    assert np.isclose(r, 1.0)
