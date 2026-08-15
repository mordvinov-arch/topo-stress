# ===== Информационная геометрия =====
# Матрица расстояний Вассерштейна между распределениями, MDS, Уорд-кластеризация.

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import squareform


def wasserstein_matrix(samples):
    """Попарные L1-Вассерштейны между наборами выборок.
    samples: список (n_k,) массивов. Возвращает симметричную матрицу."""
    k = len(samples)
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = wasserstein_distance(samples[i], samples[j])
            D[i, j] = d
            D[j, i] = d
    return D


def mds(D, n_components=2, seed=42):
    """Метрическое MDS по матрице расстояний."""
    from sklearn.manifold import MDS

    mds = MDS(n_components=n_components, dissimilarity="precomputed",
              random_state=seed, n_init=20, normalized_stress="auto")
    return mds.fit_transform(D), mds


def ward_clusters(D, n_clusters=None):
    """Иерархическая кластеризация Уорда по матрице расстояний."""
    from scipy.cluster.hierarchy import linkage, fcluster

    condensed = squareform(D)
    Z = linkage(condensed, method="ward")
    if n_clusters is None:
        return Z
    return fcluster(Z, t=n_clusters, criterion="maxclust"), Z
