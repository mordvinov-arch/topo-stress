# ===== Топологический анализ данных =====
# Реализация нормированной топологической дивергенции d~topo (Втор_анализ_для_ГХ):
#   d~topo(X, Y) = ∫_0^1 | β̄0(t; X) − β̄0(t; Y) | dt,
#   β̄0(t; X) = β0(t·D; X) / n̄,  D = diam(X ∪ Y), n̄ = (n_X + n_Y)/2.
# А также: сырая дивергенция, кривые Бетти, bottleneck и β1-гомологии (ripser).

import numpy as np
from scipy.spatial.distance import pdist, squareform


def betti_0_curve(X, epsilons):
    """Числа Бетти нулевого порядка β0(ε) для облака X (компоненты связности).
    Union-find с одним проходом по отсортированным рёбрам: O(n² log n)."""
    n = len(X)
    if n == 0:
        return np.zeros(len(epsilons))
    dist = squareform(pdist(X))
    iu = np.triu_indices(n, 1)
    edges = np.column_stack([dist[iu[0], iu[1]], iu[0], iu[1]])
    edges = edges[np.argsort(edges[:, 0])]
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    comps = np.zeros(len(epsilons))
    ncomp = n
    e = 0
    for k, eps in enumerate(epsilons):
        while e < len(edges) and edges[e, 0] <= eps:
            u = int(edges[e, 1])
            v = int(edges[e, 2])
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                ncomp -= 1
            e += 1
        comps[k] = ncomp
    return comps


def normalized_betti_curve(X, D, nbar, t_grid):
    """β̄0(t; X) = β0(t·D; X)/n̄ на сетке t ∈ [0,1]."""
    eps = t_grid * D
    beta = betti_0_curve(X, eps)
    return beta / nbar


def d_topo_normalized(X1, X2, n_eps=200):
    """Нормированная топологическая дивергенция (статья). Возвращает d, t_grid, β̄1, β̄2."""
    nX, nY = len(X1), len(X2)
    nbar = (nX + nY) / 2.0
    D = np.max(squareform(pdist(np.vstack([X1, X2]))))
    t_grid = np.linspace(0.0, 1.0, n_eps)
    b1 = normalized_betti_curve(X1, D, nbar, t_grid)
    b2 = normalized_betti_curve(X2, D, nbar, t_grid)
    d = np.trapezoid(np.abs(b1 - b2), t_grid)
    return d, t_grid, b1, b2


def d_topo_raw(X1, X2, n_eps=100):
    """Сырая дивергенция (площадь между β0-кривыми, общий диапазон ε)."""
    X = np.vstack([X1, X2])
    max_dist = np.max(squareform(pdist(X)))
    epsilons = np.linspace(0, max_dist * 1.2, n_eps)
    b1 = betti_0_curve(X1, epsilons)
    b2 = betti_0_curve(X2, epsilons)
    d = np.trapezoid(np.abs(b1 - b2), epsilons)
    return d, epsilons, b1, b2


def d_combined(X1, X2, lam1=0.5, lam2=0.5, n_eps=200):
    """Комбинированная метрика d_comb = λ1·d~topo + λ2·‖μ1 − μ2‖."""
    d_topo, _, _, _ = d_topo_normalized(X1, X2, n_eps)
    m1, m2 = X1.mean(axis=0), X2.mean(axis=0)
    d_mean = np.linalg.norm(m1 - m2)
    return lam1 * d_topo + lam2 * d_mean, d_topo, d_mean


def persistence_bottleneck(X1, X2, maxdim=0, seed=42):
    """Bottleneck-дистанция между персистентными диаграммами (persim)."""
    from persim import bottleneck
    from ripser import ripser

    dgm1 = ripser(X1, maxdim=maxdim)["dgms"][maxdim]
    dgm2 = ripser(X2, maxdim=maxdim)["dgms"][maxdim]
    return bottleneck(dgm1, dgm2, matching=False)


def beta1_curve(X, epsilons):
    """Числа Бетти первого порядка β1(ε) через ripser."""
    from ripser import ripser

    dgm = ripser(X, maxdim=1)["dgms"][1]
    betti = np.zeros(len(epsilons))
    for i, eps in enumerate(epsilons):
        # рождены при b <= eps, умерли при d > eps (или навсегда)
        n_alive = np.sum((dgm[:, 0] <= eps) & ((dgm[:, 1] > eps) | np.isinf(dgm[:, 1])))
        betti[i] = n_alive
    return betti
