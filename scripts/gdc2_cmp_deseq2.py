# GDC2: сравнение DESeq2 с HVG и PC1 loadings.
# PC1 loadings считаются по всем генам через randomized SVD,
# сопоставляются с log2FC из DESeq2; пересечения top-500.

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

DESEQ = os.path.join(RESULTS_DIR, "gdc_deseq2.csv")
COUNTS_NPZ = os.path.join(PROCESSED_DATA_DIR, "gdc_counts.npz")


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    d = np.load(COUNTS_NPZ, allow_pickle=True)
    counts = d["counts"].astype(float)
    genes = list(d["genes"])
    samples = list(d["samples"])
    print("counts:", counts.shape, flush=True)

    meta = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "gdc_metadata.csv"))
    fn2tissue = dict(zip(meta["file_name"], meta["tissue_type"]))
    tissue = np.array([fn2tissue[s] for s in samples])

    expr = np.log1p(counts.T)  # samples x genes
    Z = (expr - expr.mean(axis=0)) / (expr.std(axis=0) + 1e-12)

    from scipy.sparse.linalg import svds
    U, S, Vt = svds(Z, k=5)
    order = np.argsort(S)[::-1]
    S, Vt = S[order], Vt[order]
    pve = S ** 2 / (len(samples) - 1) / np.var(Z, axis=0).sum() * 100
    pc1_loading = Vt[0]
    print("PC1 PVE: %.2f%%" % pve[0], flush=True)

    deseq = pd.read_csv(DESEQ, index_col=0)
    deseq = deseq[~deseq.index.duplicated()]
    l2fc = deseq["log2FoldChange"].dropna()
    padj = deseq["padj"].dropna()
    sig = deseq.dropna(subset=["padj", "log2FoldChange"])
    sig = sig[sig["padj"] < 0.05]

    common = list(set(genes) & set(l2fc.index) & set(pc1_loading is not None and genes))
    ldf = pd.DataFrame({"log2FC": l2fc.reindex(common),
                        "loading": pd.Series(pc1_loading, index=genes).reindex(common)}).dropna()
    r = ldf["log2FC"].corr(ldf["loading"])
    r_abs = abs(r)  # знак PC1 произволен (ось определена с точностью до знака)
    print("n common:", len(ldf), " corr(log2FC, PC1 loading) = %.3f (|r| = %.3f)" % (r, r_abs), flush=True)

    sig_up = set(l2fc[l2fc > 1].index)
    sig_dn = set(l2fc[l2fc < -1].index)
    top_padj = set(padj.sort_values().index[:500])
    hvg = list(pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv"), index_col=0).columns)

    hv = pd.DataFrame({"loading": pd.Series(pc1_loading, index=genes).abs()})
    top_pc = set(hv.sort_values("loading", ascending=False).index[:500])

    res = {
        "pc1_pve_pct": float(pve[0]),
        "corr_log2FC_pc1loading_abs": round(float(r_abs), 4),
        "jaccard_HVG_top500_deseq_padj": round(jaccard(hvg, top_padj), 4),
        "jaccard_HVG_top500_pc1load": round(jaccard(hvg, top_pc), 4),
        "jaccard_deseq_padj_topPC1": round(jaccard(top_padj, top_pc), 4),
        "n_sig_up": int((sig["log2FoldChange"] > 1).sum()),
        "n_sig_down": int((sig["log2FoldChange"] < -1).sum()),
    }
    with open(os.path.join(RESULTS_DIR, "gdc2_cmp_deseq2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, indent=2), flush=True)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(ldf["loading"], ldf["log2FC"], s=3, alpha=0.4, c="steelblue")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("PC1 loading (all genes)"); ax.set_ylabel("DESeq2 log2FC (Tumor/Normal)")
    ax.set_title("PC1 loadings vs DESeq2 log2FC (|r|=%.3f)" % r_abs)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_cmp_deseq2.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
