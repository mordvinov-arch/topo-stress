# GDC: UMAP + k-means, сравнение с физиотипами (Wasserstein+Ward), топ-50 генов.
# Сохраняет data/processed/gdc2_labels.csv (sample, tissue, umap, kmeans, physiotype).

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from topostress.info_geometry import wasserstein_matrix, ward_clusters  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
META = os.path.join(PROCESSED_DATA_DIR, "gdc_metadata.csv")
LABELS_OUT = os.path.join(PROCESSED_DATA_DIR, "gdc2_labels.csv")


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def main():
    wide = pd.read_csv(WIDE, index_col=0)
    meta = pd.read_csv(META)
    fn2tissue = dict(zip(meta["file_name"], meta["tissue_type"]))
    samples = list(wide.index)
    tissue = np.array([fn2tissue[s] for s in samples])
    X = wide.values
    print("matrix:", X.shape, flush=True)

    # физиотипы (как в run_gdc_realgroups)
    sub = min(200, X.shape[1])
    dists = X[:, :sub] - X[:, :sub].min(axis=1, keepdims=True)
    dists = dists / (dists.sum(axis=1, keepdims=True) + 1e-12)
    D = wasserstein_matrix([dists[i] for i in range(len(dists))])
    physio, Z = ward_clusters(D, 3)

    # UMAP
    import umap
    Zz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(Zz)
    print("UMAP done:", emb.shape, flush=True)

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(emb)
    kmean_labels = km.labels_
    sil = silhouette_score(emb, kmean_labels)

    res = {
        "n": len(samples),
        "umap_kmeans_k": 3,
        "silhouette": round(float(sil), 4),
        "ari_umap_kmeans_vs_tissue": round(ari(kmean_labels, tissue), 4),
        "ari_umap_kmeans_vs_physiotype": round(ari(kmean_labels, physio), 4),
        "ari_physiotype_vs_tissue": round(ari(physio, tissue), 4),
        "physiotype_x_tissue": pd.crosstab(
            pd.Series(physio, name="physiotype"), pd.Series(tissue, name="tissue")
        ).to_dict(),
    }
    print(json.dumps(res, indent=2), flush=True)
    with open(os.path.join(RESULTS_DIR, "gdc2_umap.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    cmap_km = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for cl in range(3):
        m = kmean_labels == cl
        axes[0].scatter(emb[m, 0], emb[m, 1], s=14, c=cmap_km[cl], label="km%d" % cl, alpha=0.7)
    axes[0].legend(); axes[0].set_title("UMAP: k-means (k=3)")
    for cl in sorted(set(physio)):
        m = physio == cl
        axes[1].scatter(emb[m, 0], emb[m, 1], s=14, c=cmap_km[cl % 3], label="ph%d" % cl, alpha=0.7)
    axes[1].legend(); axes[1].set_title("UMAP: physiotypes")
    for tname, color in [("Tumor", "crimson"), ("Normal", "steelblue")]:
        m = tissue == tname
        axes[2].scatter(emb[m, 0], emb[m, 1], s=14, c=color, alpha=0.6, label=tname)
    axes[2].legend(); axes[2].set_title("UMAP: tissue")
    for ax in axes:
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_umap.png"), dpi=150)
    plt.close(fig)

    # метки для последующих анализов
    labels = pd.DataFrame({
        "sample": samples, "tissue": tissue,
        "umap1": emb[:, 0], "umap2": emb[:, 1],
        "kmeans": kmean_labels, "physiotype": physio,
        "plate": [meta.set_index("file_name")["plate_id"][s] for s in samples],
        "case_id": [meta.set_index("file_name")["case_id"][s] for s in samples],
    })
    labels.to_csv(LABELS_OUT, index=False)
    print("saved:", LABELS_OUT, flush=True)

    # C5: топ-50 генов каждого физиотипа (эффект: z-нормированная разница средних)
    out_rows = []
    for cl in sorted(set(physio)):
        m = physio == cl
        mean_cl = X[m].mean(axis=0)
        mean_all = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-12
        effect = (mean_cl - mean_all) / sd
        top = np.argsort(effect)[::-1][:50]
        for i in top:
            out_rows.append({"physiotype": cl, "gene": wide.columns[i],
                             "effect_z": float(effect[i]), "mean_log1p": float(mean_cl[i])})
    top_genes = pd.DataFrame(out_rows)
    top_genes.to_csv(os.path.join(RESULTS_DIR, "gdc2_physiotype_top50_genes.csv"), index=False)
    print("top-50 genes per physiotype saved", flush=True)
    for cl in sorted(set(physio)):
        print("  ph%d top5:" % cl,
              list(top_genes[top_genes["physiotype"] == cl]["gene"][:5]), flush=True)


if __name__ == "__main__":
    main()
