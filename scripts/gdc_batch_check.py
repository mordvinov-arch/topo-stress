# GDC: проверка батч-эффектов через PCA.
# PC1..PC5 лог-экспрессии (500 HVG) сопоставляются с tissue_type, plate_id,
# sequencing_center: R^2 и пермутационный p ассоциации.

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import DATA_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
META = os.path.join(PROCESSED_DATA_DIR, "gdc_metadata.csv")
RNG = np.random.default_rng(42)


def r2_and_p(x, labels):
    g = labels.astype("category").cat.codes.values
    grand = x.mean()
    sst = ((x - grand) ** 2).sum()
    ssb = sum(((x[g == k].mean() - grand) ** 2) * (g == k).sum() for k in np.unique(g))
    r2 = ssb / sst if sst > 0 else 0.0
    obs = r2
    perms = 199
    cnt = 0
    for _ in range(perms):
        gg = RNG.permutation(g)
        ssb_p = sum(((x[gg == k].mean() - grand) ** 2) * (gg == k).sum() for k in np.unique(g))
        if (ssb_p / sst if sst > 0 else 0.0) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (perms + 1)


def main():
    wide = pd.read_csv(WIDE, index_col=0)
    meta = pd.read_csv(META)
    fn2tissue = dict(zip(meta["file_name"], meta["tissue_type"]))
    fn2plate = dict(zip(meta["file_name"], meta["plate_id"]))
    fn2center = dict(zip(meta["file_name"], meta["sequencing_center"]))
    cols = list(wide.index)
    tissue = pd.Series([fn2tissue[c] for c in cols], index=cols).astype("category")
    plate = pd.Series([fn2plate[c] for c in cols], index=cols).astype("category")
    center = pd.Series([fn2center[c] for c in cols], index=cols).astype("category")
    print("tissue:", tissue.value_counts().to_dict(), flush=True)
    print("plates:", plate.nunique(), "| centers:", center.nunique(), flush=True)

    X = wide.values
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    P = Z @ Vt.T
    var = S ** 2 / (len(cols) - 1)
    pve = var / var.sum() * 100
    print("PVE top5: %.2f %.2f %.2f %.2f %.2f" % tuple(pve[:5]), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    import matplotlib.patches as mpatches
    colors = ["crimson" if v == "Tumor" else "steelblue" for v in tissue]
    axes[0].scatter(P[:, 0], P[:, 1], s=14, c=colors, alpha=0.6)
    axes[0].legend(handles=[mpatches.Patch(color="crimson", label="Tumor"),
                            mpatches.Patch(color="steelblue", label="Normal")])
    axes[0].set_xlabel("PC1 (%.1f%%)" % pve[0]); axes[0].set_ylabel("PC2 (%.1f%%)" % pve[1])
    axes[0].set_title("PCA: by tissue")
    sc = axes[1].scatter(P[:, 0], P[:, 1], s=14, c=plate.cat.codes, cmap="viridis", alpha=0.6)
    axes[1].set_xlabel("PC1 (%.1f%%)" % pve[0]); axes[1].set_ylabel("PC2 (%.1f%%)" % pve[1])
    axes[1].set_title("PCA: by plate (26)")
    fig.colorbar(sc, ax=axes[1], label="plate code")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc_batch_pca.png"), dpi=150)
    plt.close(fig)

    res = {"pve": [float(pve[i]) for i in range(5)]}
    for name, lab in [("tissue", tissue), ("plate", plate), ("center", center)]:
        for k in range(5):
            r2, p = r2_and_p(P[:, k], lab)
            res[f"PC{k+1}_{name}_R2"] = round(float(r2), 4)
            res[f"PC{k+1}_{name}_p"] = round(float(p), 4)
    with open(os.path.join(RESULTS_DIR, "gdc_batch_check.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
