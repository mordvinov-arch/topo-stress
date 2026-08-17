# GDC2: деконволюция микроокружения и иммунные сигнатуры для TCGA-LUAD.
#
# 1) MCP-counter-подобные скоры (Becht et al. 2016): среднее log1p-CPM маркёрных
#    генов на клеточный тип (B, T, CD8 T, цитотоксические лимфоциты, NK,
#    моноциты/макрофаги, дендритные, нейтрофилы, эндотелий, фибробласты).
# 2) Иммунные сигнатуры: цитолитический скор (GZMA+PRF1, Rooney et al.),
#    T-cell-inflamed GEP (Ayers et al. 2017, 18 генов), B-клеточная сигнатура.
# 3) Функциональные скоры: пролиферация, строма, гипоксия, EMT.
# 4) Корреляции IGKC-сигнатуры с сигнатурами; сравнение физиотипов (Kruskal-Wallis).
#
# Выход: data/processed/gdc2_immune_scores.csv (образцы), results/gdc2_immune_scores.json,
# figures/gdc2_immune_boxplots.png, figures/gdc2_igkc_immune_corr.png.

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
from topostress.utils import pearson_test  # noqa: E402

COUNTS = os.path.join(PROCESSED_DATA_DIR, "gdc_counts.npz")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
LMAX = os.path.join(RESULTS_DIR, "gdc2_lmax.json")

# Маркёрные наборы (по MCP-counter, Becht et al. 2016, Human)
MCP_MARKERS = {
    "B_cells": ["CD79B", "CD79A", "MS4A1", "BLK", "FCRL2", "CD19", "TNFRSF13C",
                "BACH2", "FCRLA", "SPIB", "TCL1A", "BANK1", "CD22", "FCRL1", "CR2"],
    "T_cells": ["CD3D", "CD3E", "CD8A", "CD6", "CD3G", "SH2D1A", "CD2", "TRAT1",
                "CD7", "SLAMF6", "ICOS", "CTLA4", "CD247", "LEF1", "BCL11B", "CCL5"],
    "CD8_T_cells": ["CD8A", "CD8B", "PRF1", "GZMA", "GZMB", "GZMK", "GZMH",
                    "CST7", "CCL5", "XCL2", "XCL1", "KLRK1", "APOBEC3G", "TIGIT"],
    "Cytotoxic": ["GZMA", "GZMB", "GZMH", "NKG7", "KLRD1", "KLRB1", "CST7",
                  "CTSW", "PRF1", "CCL4", "CCL3", "CTSC", "CLIC3", "GZMM",
                  "CD2", "GNLY", "KLRF1", "SRGN", "CX3CR1"],
    "NK_cells": ["NKG7", "KLRD1", "KLRF1", "KLRB1", "KLRG1", "GNLY", "CTSW",
                 "IL21R", "GZMH", "S1PR5", "NCR1", "ARL4C", "XCL2", "GZMM", "SPON2"],
    "Monocytic": ["CD14", "LYZ", "S100A8", "S100A9", "FCN1", "NCF2", "NCF4",
                  "THBS1", "CSF1R", "CD33", "SIGLEC1", "MS4A7", "PSAP", "TREM1",
                  "IL1B", "ANPEP", "ASGR1"],
    "Myeloid_DC": ["CD1C", "CD1D", "CLEC10A", "CD1E", "FCER1A", "CD1B", "CLEC9A",
                   "C1QA", "ITGAX", "SLCO5A1", "C1QB", "DHRS9", "C1QC", "GAS6"],
    "Neutrophils": ["FCGR3B", "CSF3R", "S100A12", "GBP2", "CD177", "VNN1", "VNN3",
                    "AQP9", "PLAC8", "FPR1", "CLEC5A", "NCF1C", "DYSF", "CRISPLD2"],
    "Endothelial": ["MMRN1", "PECAM1", "CDH5", "VWF", "MCAM", "PTPRB", "ECSCR",
                    "CLEC14A", "AQP1", "ROBO4", "ESM1", "FLT1", "CD34", "FLT4", "EDNRA"],
    "Fibroblasts": ["COL1A2", "COL1A1", "COL5A1", "COL3A1", "DCN", "FBLN2", "LUM",
                    "CALD1", "S100A4", "MRC2", "COL6A1", "PDGFRB", "COL6A3", "OGN",
                    "FN1", "ITGA5", "PXDN", "COL6A2", "CDH11"],
}

IMMUNE_SIGS = {
    "cytolytic": ["GZMA", "PRF1"],
    "Tcell_inflamed_GEP18": ["CD8A", "GZMA", "GZMB", "IFNG", "CXCL9", "CXCL10",
                             "CXCL11", "CCL5", "IDO1", "STAT1", "ICOS", "CD27",
                             "CXCR6", "HLA-DQA1", "HLA-DRB1", "HLA-DPB1",
                             "HLA-DPA1", "CD274"],
    "B_cell_sig": ["CD19", "MS4A1", "CD79A", "CD79B", "IGHM", "TNFRSF13C", "BLK", "FCRL5"],
}

FUNC_SIGS = {
    "proliferation": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7"],
    "stroma": ["COL1A1", "FN1", "VIM", "ACTA2"],
    "hypoxia": ["HIF1A", "VEGFA", "CA9", "SLC2A1"],
    "EMT": ["CDH1", "CDH2", "VIM", "SNAI1", "TWIST1"],
}


def zscore(a):
    return (a - a.mean()) / (a.std(ddof=0) + 1e-12)


def main():
    d = np.load(COUNTS, allow_pickle=True)
    counts = d["counts"].astype(float)  # genes x samples
    genes = list(d["genes"])
    samples = list(d["samples"])

    clin = pd.read_csv(MERGED)
    s2ph = dict(zip(clin["sample"], clin["physiotype"]))
    s2tissue = dict(zip(clin["sample"], clin["tissue"]))
    physio = np.array([s2ph[s] for s in samples])
    tissue = np.array([s2tissue[s] for s in samples])

    cpm = counts * 1e6 / counts.sum(axis=0)
    expr = np.log1p(cpm)  # genes x samples

    def mean_score(gene_list):
        g = [x for x in gene_list if x in genes]
        idx = [genes.index(x) for x in g]
        return expr[idx].mean(axis=0), g

    res = {"method": "MCP-counter-like deconvolution + immune/functional signatures",
           "n_samples": int(len(samples)),
           "marker_sets_used": {}, "scores_missing_genes": {}}
    score_cols = {}
    all_sets = {}
    all_sets.update(MCP_MARKERS)
    all_sets.update(IMMUNE_SIGS)
    all_sets.update(FUNC_SIGS)
    for name, gl in all_sets.items():
        sc, present = mean_score(gl)
        missing = [g for g in gl if g not in genes]
        score_cols[name] = sc
        res["marker_sets_used"][name] = present
        res["scores_missing_genes"][name] = missing
        print("%-22s markers %d/%d" % (name, len(present), len(gl)), flush=True)

    sc_df = pd.DataFrame(score_cols, index=samples)
    sc_df["sample"] = samples
    sc_df["tissue"] = tissue
    sc_df["physiotype"] = physio
    sc_df = sc_df.merge(clin[["sample", "case_id"]], on="sample", how="left")
    sc_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "gdc2_immune_scores.csv"), index=False)

    # IGKC-сигнатура = средний z-скор 20 генов λ_max-модуля
    lmax = json.load(open(LMAX, encoding="utf-8"))
    ig_genes = [g for g, _ in lmax["Tumor"]["top20_genes"] if g in genes]
    ig_idx = [genes.index(g) for g in ig_genes]
    igkc = zscore(expr[ig_idx].mean(axis=0))
    res["igkc_signature_genes"] = ig_genes
    res["igkc_vs_scores_pearson"] = {}
    for name, gl in all_sets.items():
        r, p = pearson_test(igkc, score_cols[name])
        res["igkc_vs_scores_pearson"][name] = {"r": round(float(r), 3), "p": float(p)}

    # Сравнение физиотипов (tumor-only): Kruskal-Wallis
    from scipy.stats import kruskal
    tum = tissue == "Tumor"
    res["physiotype_kruskal_tumor"] = {}
    for name in all_sets:
        groups = [score_cols[name][tum & (physio == k)] for k in sorted(set(physio[tum]))]
        if all(len(g) > 0 for g in groups):
            h, p = kruskal(*groups)
            res["physiotype_kruskal_tumor"][name] = {
                "H": round(float(h), 3), "p": float(p),
                "mean_by_physio": {str(k): round(float(score_cols[name][tum & (physio == k)].mean()), 3)
                                   for k in sorted(set(physio[tum]))},
                "n_by_physio": {str(k): int((tum & (physio == k)).sum())
                                for k in sorted(set(physio[tum]))}}

    # ===== рисунки =====
    cols_bx = ["B_cells", "Cytotoxic", "T_cells", "NK_cells", "Monocytic",
               "proliferation", "stroma", "hypoxia", "EMT"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    colors = {1: "seagreen", 2: "darkorange", 3: "slateblue"}
    for ax, name in zip(axes.flat, cols_bx):
        data = [score_cols[name][tum & (physio == k)] for k in [1, 2, 3]]
        bp = ax.boxplot(data, labels=["PH1", "PH2", "PH3"], patch_artist=True)
        for patch, k in zip(bp["boxes"], [1, 2, 3]):
            patch.set_facecolor(colors[k]); patch.set_alpha(0.6)
        p = res["physiotype_kruskal_tumor"][name]["p"]
        ax.set_title("%s (KW p=%.3g)" % (name, p), fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("TCGA-LUAD tumor: microenvironment / functional scores by physiotype", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_immune_boxplots.png"), dpi=150)
    plt.close(fig)

    # корреляции IGKC-сигнатуры с сигнатурами
    order = [n for n in all_sets]
    rs = [res["igkc_vs_scores_pearson"][n]["r"] for n in order]
    ps = [res["igkc_vs_scores_pearson"][n]["p"] for n in order]
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_colors = ["crimson" if p < 0.05 else "steelblue" for p in ps]
    ax.barh(order, rs, color=bar_colors, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Pearson r with IGKC signature"); ax.set_title("IGKC signature vs immune/functional scores")
    ax.axvline(0, color="k", lw=0.8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_igkc_immune_corr.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "gdc2_immune_scores.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved results/gdc2_immune_scores.json + figures", flush=True)


if __name__ == "__main__":
    main()
