# Тесты топологических функций.

import numpy as np
import pytest

from topostress import topology


def make_cloud(n, rng):
    return rng.standard_normal((n, 3))


def test_d_topo_identical_clouds_is_zero():
    rng = np.random.default_rng(0)
    X = make_cloud(30, rng)
    d, t, b1, b2 = topology.d_topo_normalized(X, X.copy(), n_eps=100)
    assert d < 1e-12
    assert np.allclose(b1, b2)


def test_d_topo_symmetric():
    rng = np.random.default_rng(1)
    X1, X2 = make_cloud(25, rng), make_cloud(25, rng)
    d12, *_ = topology.d_topo_normalized(X1, X2, n_eps=50)
    d21, *_ = topology.d_topo_normalized(X2, X1, n_eps=50)
    assert np.isclose(d12, d21, atol=1e-12)


def test_d_topo_translation_invariant():
    rng = np.random.default_rng(2)
    X = make_cloud(25, rng)
    far = X + 10.0
    d_same, *_ = topology.d_topo_normalized(X, X, n_eps=50)
    d_far, *_ = topology.d_topo_normalized(X, far, n_eps=50)
    assert d_same == 0.0
    assert np.isclose(d_far, 0.0, atol=1e-9)  # сдвиг не меняет d~topo


def test_d_topo_structural_difference_positive():
    rng = np.random.default_rng(7)
    cluster = rng.standard_normal((25, 3)) * 0.3
    spread = rng.uniform(-2, 2, (25, 3))
    d, *_ = topology.d_topo_normalized(cluster, spread, n_eps=50)
    assert d > 0.05  # разная структура -> ненулевая дивергенция


def test_betti_0_curve_two_clusters():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((10, 2)) * 0.1
    B = rng.standard_normal((10, 2)) * 0.1 + np.array([10.0, 0.0])
    X = np.vstack([A, B])
    eps_small = np.array([0.01])
    eps_mid = np.array([1.0])
    eps_large = np.array([100.0])
    assert topology.betti_0_curve(X, eps_small)[0] == 20  # все изолированы
    assert topology.betti_0_curve(X, eps_mid)[0] == 2     # два кластера
    assert topology.betti_0_curve(X, eps_large)[0] == 1   # всё связано


def test_d_combined_decomposition():
    rng = np.random.default_rng(4)
    X1, X2 = make_cloud(20, rng), make_cloud(20, rng)
    d_comb, d_topo, d_mean = topology.d_combined(X1, X2, lam1=0.3, lam2=0.7, n_eps=50)
    assert np.isclose(d_comb, 0.3 * d_topo + 0.7 * d_mean, atol=1e-10)


def test_d_topo_raw_nonnegative():
    rng = np.random.default_rng(5)
    X1, X2 = make_cloud(15, rng), make_cloud(15, rng)
    d, eps, b1, b2 = topology.d_topo_raw(X1, X2, n_eps=40)
    assert d >= 0
    assert eps[0] == 0


def test_normalized_betti_bounds():
    rng = np.random.default_rng(6)
    X = make_cloud(10, rng)
    D = float(np.max(np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)))
    t = np.linspace(0, 1, 20)
    b = topology.normalized_betti_curve(X, D, 10.0, t)
    assert b.min() >= 0
    assert np.isclose(b[0], 1.0, atol=1e-12)  # t=0: все компоненты, /nbar


@pytest.mark.skipif(True, reason="ripser/persim не обязательны для unit-тестов")
def test_persistence_bottleneck_installed():
    import persim  # noqa: F401
    import ripser  # noqa: F401
