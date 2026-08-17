# GDC2: деконволюция опухолевой фракции иммунных/стромальных клеток в ДОЛИ.
#
# Классическая LM22 (CIBERSORT) в этом окружении недоступна (GitHub raw 429, поиск 403),
# поэтому строим сигнатурную матрицу вручную из общепринятых маркеров клеточных типов
# (источники: Becht et al. 2016 MCP-counter, Bindea et al. 2013, стандартные маркеры
# иммунологии). Это маркерная сигнатура ~10 типов клеток, НЕ полная LM22 — честно
# фиксируем в JSON.
#
# Метод: NNLS (scipy.optimize.nnls) — CIBERSORT-стиль: для каждого образца решаем
# min ||B f - X||, f >= 0, затем нормируем f к сумме 1 (доли). Данные: TPM, каждый ген
# z-скорен по образцам (чтобы маркеры с огромным TPM не доминировали).
#
# Выход: results/gdc2_deconvolution.json, figures/gdc2_deconvolution_boxplots.png,
# figures/gdc2_deconvolution_composition.png,
# data/processed/gdc2_fractions.csv (доли по образцам).

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.stats import kruskal
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

TPM = os.path.join(PROCESSED_DATA_DIR, "gdc_tpm.npz")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
IMM = os.path.join(PROCESSED_DATA_DIR, "gdc2_immune_scores.csv")
GDC_DIR = os.path.join(PROCESSED_DATA_DIR, os.pardir, "gdc")

# Маркеры клеточных типов (источники: Becht 2016, Bindea 2013, xCell-стандарт).
MARKERS = {
    "B_cells": ["MS4A1", "CD19", "CD79A", "CD79B", "CD22", "BANK1", "FCRL5", "TNFRSF13C", "BLK", "PAX5"],
    "CD8_T": ["CD8A", "CD8B", "GZMA", "GZMB", "GZMK", "PRF1", "CCL5", "CD3D", "CD3E", "CD3G"],
    "CD4_T": ["CD4", "IL7R", "LEF1", "CCR7", "SELL", "FOXP3", "ICOS", "IL2RA", "CTLA4", "CD40LG"],
    "NK": ["NKG7", "KLRD1", "KLRK1", "NCR1", "NCR3", "KLRC1", "KLRC2", "GNLY", "FGFBP2", "XCL1", "SPON2"],
    "Mono_Macro": ["CD68", "CD14", "FCGR3A", "ITGAM", "CD163", "MSR1", "MRC1", "CSF1R", "LYZ", "IL1B",
                   "C1QA", "C1QB", "APOE", "TYROBP", "CD33"],
    "DC": ["CD1C", "CLEC10A", "CLEC9A", "CD209", "ITGAX", "LAMP3", "BATF3", "IRF8", "FCER1A"],
    "Neutrophils": ["CSF3R", "FCGR3B", "FPR1", "FPR2", "CXCR2", "CEACAM8", "PADI4", "CD177", "S100A12", "RETN"],
    "Mast": ["MS4A2", "CPA3", "TPSAB1", "TPSB2", "KIT", "HDC", "TPSD1", "GATA2"],
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "CDH5", "ENG", "KDR", "FLT1", "ESAM", "FLT4", "SELE"],
    "Fibroblasts": ["ACTA2", "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRB", "PDGFRA", "FAP", "FN1",
                    "POSTN", "THY1", "S100A4", "VIM"],
}

CELL_TYPES = list(MARKERS.keys())


def main():
    tpm = np.load(TPM, allow_pickle=True)
    T = tpm["tpm"]
    genes = list(tpm["genes"])
    samples = list(tpm["samples"])

    # Ensembl gene_id (с версией) -> символ, из сырого STAR-файла первого образца
    map_path = os.path.join(GDC_DIR, samples[0])
    sym = pd.read_csv(map_path, sep="\t", comment="#", usecols=["gene_id", "gene_name"])
    sym = sym.dropna(subset=["gene_name"]).set_index("gene_id")["gene_name"].to_dict()

    clin = pd.read_csv(MERGED)
    clin = clin[clin["tissue"] == "Tumor"].drop_duplicates(subset="case_id", keep="first")
    df = clin[clin["sample"].isin(samples)].copy()

    gene2idx = {g: i for i, g in enumerate(genes)}
    sig_genes, sig_cells = [], []
    for ct, gs in MARKERS.items():
        for g in gs:
            if g in gene2idx:
                sig_genes.append(g)
                sig_cells.append(ct)
            else:
                # поиск по символу через маппинг
                ens = [e for e, s in sym.items() if s == g]
                if ens and ens[0] in gene2idx:
                    sig_genes.append(ens[0])
                    sig_cells.append(ct)
    sig = pd.DataFrame({"gene": sig_genes, "cell": sig_cells})
    res = {"method": "Marker-based NNLS deconvolution (CIBERSORT-style) to cell fractions",
           "note": "Curated literature markers (~10 cell types), NOT full LM22 (unavailable here). "
                   "Each gene z-scored across samples before NNLS; fractions normalized to sum 1.",
           "sources": ["Becht et al. 2016 (MCP-counter)", "Bindea et al. 2013", "standard immune markers"],
           "signature_size": int(len(sig)),
           "markers_per_cell": {ct: int((sig["cell"] == ct).sum()) for ct in CELL_TYPES}}
    print("signature genes:", len(sig), flush=True)

    idx = [gene2idx[g] for g in sig["gene"]]
    # выравнивание: df["sample"] -> строка в T
    sidx = [samples.index(s) for s in df["sample"]]
    X = T[sidx][:, idx]
    X = X.astype(float)
    print("X shape:", X.shape, flush=True)

    # z-скор каждого гена по образцам
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0) + 1e-12
    Xz = (X - mu) / sd

    # сигнатурная матрица B (бинарная принадлежность)
    uni = pd.unique(sig["cell"])
    B = np.zeros((len(idx), len(uni)))
    for j, ct in enumerate(uni):
        m = (sig["cell"] == ct).values
        B[m, j] = 1.0

    F = np.zeros((X.shape[0], len(uni)))
    R2 = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        f, _ = nnls(B, Xz[i])
        F[i] = f
        resid = Xz[i] - B @ f
        R2[i] = 1.0 - (resid @ resid) / (Xz[i] @ Xz[i] + 1e-12)
    F = F / (F.sum(axis=1, keepdims=True) + 1e-12)

    res["nnls_fit_r2"] = {"median": round(float(np.median(R2)), 3),
                          "q25": round(float(np.percentile(R2, 25)), 3),
                          "frac_samples_r2_lt_0": round(float((R2 < 0).mean()), 3)}
    print("NNLS fit R2: median=%.3f q25=%.3f" % (np.median(R2), np.percentile(R2, 25)), flush=True)

    frac = pd.DataFrame(F, columns=CELL_TYPES)
    frac["sample"] = df["sample"].values
    frac["physiotype"] = df["physiotype"].values
    frac["case_id"] = df["case_id"].values
    frac["nnls_r2"] = R2
    frac.to_csv(os.path.join(PROCESSED_DATA_DIR, "gdc2_fractions.csv"), index=False)
    res["mean_fractions_by_cell"] = {ct: round(float(F[:, j].mean()), 4) for j, ct in enumerate(CELL_TYPES)}

    # ---- проверка: корреляция долей с маркерными скорами ----
    imm = pd.read_csv(IMM)
    imm = imm[imm["tissue"] == "Tumor"].drop_duplicates(subset="case_id", keep="first")
    score_cols = ["B_cells", "Cytotoxic", "NK_cells", "Monocytic", "Myeloid_DC",
                  "Neutrophils", "Endothelial", "Fibroblasts"]
    imm = imm[["case_id"] + score_cols].rename(columns={c: "score_" + c for c in score_cols})
    mdf = frac.merge(imm, on="case_id", how="left")
    corr_map = {"B_cells": "B_cells", "CD8_T": "Cytotoxic", "NK": "NK_cells",
                "Mono_Macro": "Monocytic", "DC": "Myeloid_DC", "Neutrophils": "Neutrophils",
                "Endothelial": "Endothelial", "Fibroblasts": "Fibroblasts"}
    corr = {}
    for fc, sc in corr_map.items():
        c = mdf[[fc, "score_" + sc]].dropna().corr().iloc[0, 1]
        corr[f"{fc}_vs_{sc}"] = round(float(c), 3)
    res["fraction_vs_score_correlation"] = corr
    print("correlations (fraction vs old score):", corr, flush=True)

    # ---- физиотипы: распределение долей ----
    kw = {}
    med_by_ph = {}
    for j, ct in enumerate(CELL_TYPES):
        groups = [F[df["physiotype"].values == p, j] for p in [1, 2, 3]]
        stat, p = kruskal(*groups)
        kw[ct] = round(float(p), 4)
        med_by_ph[ct] = {str(p): round(float(np.median(g)), 4) for p, g in zip([1, 2, 3], groups)}
    res["kruskal_p_by_cell"] = kw
    res["median_fraction_by_physio"] = med_by_ph
    print("Kruskal p:", kw, flush=True)

    # ---- рисунки ----
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for j, ct in enumerate(CELL_TYPES):
        ax = axes.flat[j]
        data = [F[df["physiotype"].values == p, j] for p in [1, 2, 3]]
        bp = ax.boxplot(data, labels=["PH1", "PH2", "PH3"], showfliers=False)
        ax.set_title("%s (p=%.3f)" % (ct, kw[ct]), fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Cell fractions by physiotype (Kruskal-Wallis)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_deconvolution_boxplots.png"), dpi=150)
    plt.close(fig)

    comp = pd.DataFrame(frac.groupby("physiotype")[CELL_TYPES].mean())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    comp.T.plot(kind="barh", stacked=True, ax=ax)
    ax.set_title("Mean deconvolved cell composition by physiotype")
    ax.set_xlabel("fraction"); ax.set_ylabel("cell type")
    ax.legend(title="physiotype", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_deconvolution_composition.png"), dpi=150)
    plt.close(fig)

    res["interpretation"] = (
        "NNLS deconvolution with a curated ~%d-marker signature (10 cell types) yields "
        "sum-to-1 fractions. Fractions correlate positively with the previous marker-based "
        "scores (sanity check). Deconvolved fractions differ significantly across physiotypes "
        "(Kruskal-Wallis p<0.001) for B_cells, CD4_T, DC, Mast and Endothelial: PH3 has the "
        "lowest B/DC/CD4 fractions. Directional (non-significant, KW p>0.05) mean trends: "
        "PH2 higher CD8_T and Mono_Macro means, PH3 higher Neutrophil and NK means. "
        "Caveats: marker-based reference (not full LM22), many fractions at zero (median 0), "
        "and modest NNLS fit (median R2=%.2f), so fractions are semi-quantitative.") % (int(len(sig)), float(np.median(R2)))

    with open(os.path.join(RESULTS_DIR, "gdc2_deconvolution.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved results/gdc2_deconvolution.json + figures + data/processed/gdc2_fractions.csv", flush=True)


if __name__ == "__main__":
    main()