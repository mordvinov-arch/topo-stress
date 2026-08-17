# GDC2: стабильность кластеризации физиотипов по числу кластеров (k=2..5)
# и числу генов (100/200/300/500).
#
# Для каждой пары (k, n_genes): Wasserstein+Ward на топ-n_genes HVG (simplex).
# Метрики: silhouette (все образцы); для k=3 — ARI с эталонными физиотипами
# (полные данные, 200 генов); устойчивость между соседними наборами генов
# внутри k (ARI попарно); воспроизводимость подвыборок (50% бутстреп, 10 повторов).
#
# Выход: results/gdc2_cluster_stability.json, figures/gdc2_cluster_stability.png.

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from topostress.info_geometry import ward_clusters  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
LABELS = os.path.join(PROCESSED_DATA_DIR, "gdc2_labels.csv")
GENE_COUNTS = [100, 200, 300, 500]
KS = [2, 3, 4, 5]
N_BOOT = 10


def pairwise_wasserstein_fast(X):
    """L1-Вассерштейн между строками X (одинаковые размерности):
    scipy.wasserstein_distance(u,v) = mean|sort(u)-sort(v)|; попарно через
    sum|a-b| = sum(a)+sum(b)-2*sum(min(a,b)) на отсортированных строках."""
    S = np.sort(X, axis=1)
    n, m = S.shape
    sx = S.sum(axis=1)
    D = np.empty((n, n))
    for i in range(n):
        D[i] = (sx + sx[i] - 2.0 * np.minimum(S[i], S).sum(axis=1)) / m
    return D


def cluster(X, k, sub):
    d = X[:, :sub] - X[:, :sub].min(axis=1, keepdims=True)
    d = d / (d.sum(axis=1, keepdims=True) + 1e-12)
    D = pairwise_wasserstein_fast(d)
    lab, _ = ward_clusters(D, k)
    return lab


def main():
    wide = pd.read_csv(WIDE, index_col=0)
    labels = pd.read_csv(LABELS)
    ref = dict(zip(labels["sample"], labels["physiotype"]))
    samples = list(wide.index)
    y_ref = np.array([ref[s] for s in samples])
    X = wide.values
    rng = np.random.default_rng(42)
    print("n=%d" % len(samples), flush=True)

    res = {"method": "Cluster stability: k x n_genes (Wasserstein+Ward)",
           "k_values": KS, "gene_counts": GENE_COUNTS, "n_boot": N_BOOT}

    all_labels = {}
    for k in KS:
        all_labels[k] = {}
        for ng in GENE_COUNTS:
            lab = cluster(X, k, ng)
            all_labels[k][ng] = lab

            row = {"n_genes": ng, "k": k}
            row["silhouette"] = round(float(silhouette_score(X, lab)), 4)
            if k == 3:
                row["ari_vs_reference"] = round(float(adjusted_rand_score(y_ref, lab)), 4)
            if ng == 500:
                row["ari_vs_canonical_200"] = round(
                    float(adjusted_rand_score(all_labels[k][200], lab)), 4)

            # бутстреп-воспроизводимость (50% выборка)
            aris = []
            for _ in range(N_BOOT):
                sub_idx = rng.choice(len(samples), size=int(0.5 * len(samples)), replace=False)
                lab_sub = cluster(X[sub_idx], k, ng)
                full_sub = lab[sub_idx]
                aris.append(adjusted_rand_score(full_sub, lab_sub))
            row["bootstrap_ari_mean_sd"] = [round(float(np.mean(aris)), 4),
                                            round(float(np.std(aris)), 4)]
            res.setdefault("results", {})["%d_genes_k%d" % (ng, k)] = row
            print("k=%d genes=%d sil=%.3f ari_ref=%s ari_canon=%s boot=%.3f±%.3f"
                  % (k, ng, row["silhouette"],
                     row.get("ari_vs_reference"), row.get("ari_vs_canonical_200"),
                     row["bootstrap_ari_mean_sd"][0], row["bootstrap_ari_mean_sd"][1]), flush=True)

    # устойчивость между соседними наборами генов (внутри k)
    res["adjacent_gene_set_ari"] = {}
    for k in KS:
        for a, b in zip(GENE_COUNTS, GENE_COUNTS[1:]):
            res["adjacent_gene_set_ari"]["k%d: %d->%d" % (k, a, b)] = round(
                float(adjusted_rand_score(all_labels[k][a], all_labels[k][b])), 4)

    with open(os.path.join(RESULTS_DIR, "gdc2_cluster_stability.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ===== рисунок =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sil = np.array([[res["results"]["%d_genes_k%d" % (ng, k)]["silhouette"] for ng in GENE_COUNTS]
                    for k in KS])
    im = axes[0].imshow(sil, aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(GENE_COUNTS))); axes[0].set_xticklabels(GENE_COUNTS)
    axes[0].set_yticks(range(len(KS))); axes[0].set_yticklabels(["k=%d" % k for k in KS])
    axes[0].set_title("silhouette (k x n_genes)")
    fig.colorbar(im, ax=axes[0])
    for i in range(len(KS)):
        for j in range(len(GENE_COUNTS)):
            axes[0].text(j, i, "%.3f" % sil[i, j], ha="center", va="center", fontsize=8,
                         color="white" if sil[i, j] > sil.mean() else "black")

    for k in [2, 3, 4, 5]:
        boot = [res["results"]["%d_genes_k%d" % (ng, k)]["bootstrap_ari_mean_sd"][0] for ng in GENE_COUNTS]
        axes[1].plot(GENE_COUNTS, boot, "o-", label="k=%d" % k)
    axes[1].set_xlabel("n genes"); axes[1].set_ylabel("bootstrap ARI (50%% resample)")
    axes[1].set_title("cluster reproducibility"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_cluster_stability.png"), dpi=150)
    plt.close(fig)
    print("saved gdc2_cluster_stability.json + figure", flush=True)


if __name__ == "__main__":
    main()
