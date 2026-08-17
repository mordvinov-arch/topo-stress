# GDC2: чувствительность результатов к числу генов.
#
# 1) d_topo (PC1+ vs PC1-, весь набор 601) при n_genes = 100/200/300/400/500
#    (логарифмированные TPM, топ-n по дисперсии, как в каноническом пайплайне).
# 2) Физиотипы (Wasserstein+Ward, k=3) при n_genes = 100/200/300/400/500:
#    ARI с эталонными физиотипами (200 генов) и silhouette.
#
# Выход: results/gdc2_sensitivity.json, figures/gdc2_sensitivity.png.

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

from topostress import topology, utils  # noqa: E402
from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from topostress.info_geometry import ward_clusters  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
LABELS = os.path.join(PROCESSED_DATA_DIR, "gdc2_labels.csv")
N_EPS = int(os.environ.get("MAST_EPS", 100))
GENE_COUNTS = [100, 200, 300, 400, 500]


def pairwise_wasserstein_fast(X):
    S = np.sort(X, axis=1)
    n, m = S.shape
    sx = S.sum(axis=1)
    D = np.empty((n, n))
    for i in range(n):
        D[i] = (sx + sx[i] - 2.0 * np.minimum(S[i], S).sum(axis=1)) / m
    return D


def main():
    wide = pd.read_csv(WIDE, index_col=0)
    labels = pd.read_csv(LABELS)
    ref = dict(zip(labels["sample"], labels["physiotype"]))
    samples = list(wide.index)
    y_ref = np.array([ref[s] for s in samples])
    X = wide.values
    print("n=%d" % len(samples), flush=True)

    # канонический PC1± сплит на 500 генов (как run_gdc.py)
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    pc1 = Z @ vt[0]
    g1, g2 = pc1 > np.median(pc1), pc1 <= np.median(pc1)

    res = {"method": "Sensitivity to number of genes: d_topo (PC1+/-) and physiotypes (k=3)",
           "n": int(len(samples)), "n_eps": N_EPS}
    res["d_topo_by_ngenes"] = {}
    res["physiotype_by_ngenes"] = {}
    ref_lab = None
    for ng in GENE_COUNTS:
        Xs = X[:, :ng]
        d, t, b1, b2 = topology.d_topo_normalized(Xs[g1], Xs[g2], n_eps=N_EPS)
        res["d_topo_by_ngenes"][str(ng)] = {"d_topo": round(float(d), 4)}
        print("d_topo n_genes=%d: %.4f" % (ng, d), flush=True)

        d0 = Xs - Xs.min(axis=1, keepdims=True)
        d0 = d0 / (d0.sum(axis=1, keepdims=True) + 1e-12)
        D = pairwise_wasserstein_fast(d0)
        lab, _ = ward_clusters(D, 3)
        if ng == 200:
            ref_lab = lab
        sil = silhouette_score(Xs, lab)
        row = {"n_genes": ng, "silhouette": round(float(sil), 4),
               "cluster_sizes": {str(k): int((lab == k).sum()) for k in sorted(set(lab))}}
        if ref_lab is not None:
            row["ari_vs_reference_200"] = round(float(adjusted_rand_score(ref_lab, lab)), 4)
        res["physiotype_by_ngenes"][str(ng)] = row
        print("physiotypes n_genes=%d: sil=%.3f sizes=%s ari200=%s"
              % (ng, sil, row["cluster_sizes"], row.get("ari_vs_reference_200")), flush=True)

    # дополнительно: полная пермутационная p для d_topo на 500 генов
    Xs = X[:, :500]
    _, p, _ = utils.permutation_test(
        lambda A, B: topology.d_topo_normalized(A, B, n_eps=N_EPS)[0],
        Xs[g1], Xs[g2], n_perm=int(os.environ.get("GDC_SENS_PERM", 199)), seed=42)
    res["d_topo_500_permutation_p"] = round(float(p), 4)
    print("d_topo(500) permutation p=%.4f" % p, flush=True)

    with open(os.path.join(RESULTS_DIR, "gdc2_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ===== рисунок =====
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ng = [int(k) for k in res["d_topo_by_ngenes"]]
    ds = [res["d_topo_by_ngenes"][str(k)]["d_topo"] for k in ng]
    axes[0].plot(ng, ds, "o-", color="darkorchid")
    axes[0].set_xlabel("n genes"); axes[0].set_ylabel("d_topo (PC1+ vs PC1-)")
    axes[0].set_title("d_topo sensitivity"); axes[0].grid(alpha=0.3)
    axes[1].plot(ng, [res["physiotype_by_ngenes"][str(k)]["silhouette"] for k in ng],
                 "o-", color="steelblue", label="silhouette")
    ari = [res["physiotype_by_ngenes"][str(k)].get("ari_vs_reference_200", np.nan) for k in ng]
    axes[1].plot(ng, ari, "s--", color="crimson", label="ARI vs 200-gene ref")
    axes[1].set_xlabel("n genes"); axes[1].set_ylabel("score")
    axes[1].set_title("Physiotype sensitivity (k=3)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_sensitivity.png"), dpi=150)
    plt.close(fig)
    print("saved gdc2_sensitivity.json + figure", flush=True)


if __name__ == "__main__":
    main()
