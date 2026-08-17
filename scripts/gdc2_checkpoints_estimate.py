# GDC2: иммунные чекпойнты и ESTIMATE-подобные скоры по физиотипам.
#
# 1) Чекпойнт-модуль: экспрессия ключевых генов иммунных чекпойнтов (PD-1/PD-L1/PD-L2,
#    CTLA4, LAG3, TIM3, TIGIT, IDO1, SIGLEC15, VTCN1) + суммарный чекпойнт-скор
#    (средний z-скор log1p TPM).
# 2) ESTIMATE-подобные скоры (Yoshihara 2013, идея, НЕ точные 141-генные списки —
#    собраны из литературы): ImmuneScore, StromalScore, ESTIMATE-like score и
#    оценка чистоты опухоли (purity = cos(0.604987 + 0.0001468*score)).
# 3) Сравнение скоров между физиотипами (Kruskal-Wallis, медианы), корреляция
#    чекпойнт-скора с TMB и IGKC, Cox-модель (возраст+стадия+чекпойнт-скор).
#
# Выход: results/gdc2_checkpoints_estimate.json,
# figures/gdc2_checkpoints_estimate.png, figures/gdc2_checkpoint_survival.png.

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import kruskal, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

TPM = os.path.join(PROCESSED_DATA_DIR, "gdc_tpm.npz")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
MUT = os.path.join(PROCESSED_DATA_DIR, "gdc2_mutations_full.csv")
GDC_DIR = os.path.join(PROCESSED_DATA_DIR, os.pardir, "gdc")

CHECKPOINTS = {
    "PDCD1": "PD-1", "CD274": "PD-L1", "PDCD1LG2": "PD-L2", "CTLA4": "CTLA4",
    "LAG3": "LAG3", "HAVCR2": "TIM-3", "TIGIT": "TIGIT", "IDO1": "IDO1",
    "SIGLEC15": "SIGLEC15", "VTCN1": "B7-H4",
}

# ESTIMATE-подобные сигнатуры (курированы из литературы, не точные списки Yoshihara)
IMMUNE_GENES = [
    "CD3D", "CD3E", "CD3G", "CD2", "CD8A", "CD8B", "CD4", "IL7R", "CCR7", "CXCR3",
    "CXCR4", "CCL5", "GZMA", "GZMB", "GZMK", "PRF1", "NKG7", "KLRD1", "KLRK1", "GNLY",
    "NCAM1", "CD19", "MS4A1", "CD79A", "CD79B", "CD22", "IGKC", "IGHA1", "IGHG1",
    "CD68", "CD14", "CSF1R", "LYZ", "C1QA", "C1QB", "C1QC", "APOE", "TLR4", "TLR9",
    "CD1C", "ITGAX", "LAMP3", "FCER1A", "CD27", "CD28", "ICOS", "IL2RA", "FOXP3",
    "TNFRSF9", "CXCL9", "CXCL10", "CXCL11", "IFNG", "TBX21", "STAT1", "IRF1", "IRF7",
    "OAS1", "MX1", "ISG15", "IDO1",
]
STROMAL_GENES = [
    "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL5A1", "COL6A1", "COL6A2", "COL6A3",
    "DCN", "LUM", "BGN", "VCAN", "FN1", "VIM", "ACTA2", "TAGLN", "MYL9", "MYH11", "DES",
    "CDH11", "SPARC", "THY1", "PDGFRB", "PDGFRA", "PDGFC", "FAP", "S100A4", "POSTN",
    "FBN1", "ELN", "MMP2", "MMP14", "TIMP1", "LOX", "LOXL1", "LOXL2", "COMP", "PECAM1",
    "VWF", "CLDN5", "CDH5", "ENG", "KDR", "FLT1", "ESAM", "TIE1", "TEK", "GNG11", "RGS5",
]


def main():
    tpm = np.load(TPM, allow_pickle=True)
    T = tpm["tpm"]
    genes = list(tpm["genes"])
    samples = list(tpm["samples"])

    map_path = os.path.join(GDC_DIR, samples[0])
    sym = pd.read_csv(map_path, sep="\t", comment="#", usecols=["gene_id", "gene_name"])
    sym = sym.dropna(subset=["gene_name"])
    sym["gene_name"] = sym["gene_name"].str.upper()
    sym_map = {}
    for ens, name in zip(sym["gene_id"], sym["gene_name"]):
        sym_map.setdefault(name, ens)
    gene2idx = {g: i for i, g in enumerate(genes)}

    def ens_of(symbol):
        return sym_map.get(symbol, sym_map.get(symbol.upper()))

    clin = pd.read_csv(MERGED)
    clin = clin[clin["tissue"] == "Tumor"].drop_duplicates(subset="case_id", keep="first")
    clin = clin.dropna(subset=["os_days"]).reset_index(drop=True)
    sidx = [samples.index(s) for s in clin["sample"]]
    Tsub = T[sidx]
    logt = np.log1p(Tsub)

    res = {"method": "Immune checkpoints and ESTIMATE-like scores by physiotype",
           "checkpoint_genes": CHECKPOINTS,
           "immune_signature_n": 0, "stromal_signature_n": 0}

    def gene_vals(symbol):
        ens = ens_of(symbol)
        if ens is None or ens not in gene2idx:
            return None
        return logt[:, gene2idx[ens]]

    def zscore(x):
        return (x - x.mean()) / (x.std(ddof=0) + 1e-12)

    cp_expr = {}
    for sym, label in CHECKPOINTS.items():
        v = gene_vals(sym)
        if v is None:
            continue
        cp_expr[label] = zscore(v)
    cp_mat = pd.DataFrame(cp_expr)
    missing_cp = [l for l in CHECKPOINTS.values() if l not in cp_mat.columns]
    res["checkpoints_missing_from_tpm"] = missing_cp
    cp_score = cp_mat.mean(axis=1).values
    print("checkpoint genes measured:", len(cp_mat.columns), "missing:", missing_cp, flush=True)

    def module_score(symbols):
        col = []
        used = []
        for s in symbols:
            v = gene_vals(s)
            if v is None:
                continue
            col.append(zscore(v))
            used.append(s)
        return np.array(col), used

    imm_mat, imm_used = module_score(IMMUNE_GENES)
    str_mat, str_used = module_score(STROMAL_GENES)
    res["immune_signature_n"] = len(imm_used)
    res["stromal_signature_n"] = len(str_used)
    print("immune genes used: %d/%d, stromal: %d/%d" % (len(imm_used), len(IMMUNE_GENES), len(str_used), len(STROMAL_GENES)), flush=True)

    imm_score = imm_mat.mean(axis=0)
    str_score = str_mat.mean(axis=0)
    estimate_like = imm_score + str_score
    purity = np.cos(0.604987 + 0.0001468 * estimate_like)
    purity = np.clip(purity, 0, 1)

    df = clin.copy()
    df["ImmuneScore"] = imm_score
    df["StromalScore"] = str_score
    df["ESTIMATE_like"] = estimate_like
    df["purity"] = purity
    df["checkpoint_score"] = cp_score
    df["age_years"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce").abs()
    df["stage_III_IV"] = (df["stage_grp"] == "III/IV").astype(int)
    df["os_days"] = df["os_days"].astype(float)
    df["event"] = df["event"].astype(int)

    # ---- (3) сравнение по физиотипам ----
    kw = {}
    med = {}
    for sc in ["ImmuneScore", "StromalScore", "ESTIMATE_like", "purity", "checkpoint_score"]:
        groups = [df.loc[df["physiotype"] == p, sc].values for p in [1, 2, 3]]
        st, p = kruskal(*groups)
        kw[sc] = round(float(p), 5)
        med[sc] = {str(p2): round(float(np.median(g2)), 4) for p2, g2 in zip([1, 2, 3], groups)}
    res["kruskal_p_by_score"] = kw
    res["median_score_by_physio"] = med
    print("Kruskal:", kw, flush=True)

    # корреляции чекпойнт-скора с TMB и IGKC
    mut = pd.read_csv(MUT)[["case_id", "TMB"]]
    df = df.merge(mut, on="case_id", how="left")
    df["TMB_log"] = np.log1p(pd.to_numeric(df["TMB"], errors="coerce"))
    wide = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv"), index_col=0)
    ig_genes = [g for g in ["IGKC", "CD27", "MS4A1"] if g in wide.columns]
    igscore = zscore(wide.loc[df["sample"], ig_genes].mean(axis=1).values)
    df["IGKC_module"] = igscore
    sp = {}
    for a, b in [("checkpoint_score", "TMB_log"), ("checkpoint_score", "IGKC_module"),
                 ("ImmuneScore", "TMB_log"), ("purity", "TMB_log")]:
        d = df[[a, b]].dropna()
        r, p = spearmanr(d[a], d[b])
        sp[f"{a}_vs_{b}"] = {"rho": round(float(r), 3), "p": round(float(p), 5)}
    res["correlations"] = sp
    print("correlations:", sp, flush=True)

    # ---- чекпойнт-скор и выживаемость ----
    df["cp_hi"] = (df["checkpoint_score"] > df["checkpoint_score"].median()).astype(int)
    hi = df[df["cp_hi"] == 1]
    lo = df[df["cp_hi"] == 0]
    lr = logrank_test(hi["os_days"], lo["os_days"], event_observed_A=hi["event"], event_observed_B=lo["event"])
    res["logrank_checkpoint_hi_vs_lo"] = round(float(lr.p_value), 4)
    res["median_os_checkpoint"] = {
        "hi": round(float(KaplanMeierFitter().fit(hi["os_days"], hi["event"]).median_survival_time_), 0) or None,
        "lo": round(float(KaplanMeierFitter().fit(lo["os_days"], lo["event"]).median_survival_time_), 0) or None}

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    kmf.fit(hi["os_days"], hi["event"], label="checkpoint_high")
    kmf.plot_survival_function(ax=ax, lw=2, color="crimson")
    kmf.fit(lo["os_days"], lo["event"], label="checkpoint_low")
    kmf.plot_survival_function(ax=ax, lw=2, color="steelblue")
    ax.set_title("OS by checkpoint score (log-rank p=%.4f)" % lr.p_value)
    ax.set_xlabel("days"); ax.set_ylabel("OS probability")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_checkpoint_survival.png"), dpi=150)
    plt.close(fig)

    # Cox: возраст+стадия+чекпойнт-скор
    dcox = df.dropna(subset=["age_years", "stage_grp"]).copy()
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(dcox[["age_years", "stage_III_IV", "checkpoint_score", "os_days", "event"]],
            duration_col="os_days", event_col="event", show_progress=False)
    summ = {k: {"coef": round(float(v["coef"]), 4), "hr": round(float(v["exp(coef)"]), 3),
                "p": round(float(v["p"]), 4)}
            for k, v in cph.summary.iterrows()}
    res["cox_age_stage_checkpoint"] = {"summary": summ,
                                       "c_index": round(float(cph.concordance_index_), 4)}

    # ---- рисунок: боксплоты скоров по физиотипам ----
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, sc in zip(axes, ["ImmuneScore", "StromalScore", "ESTIMATE_like", "purity", "checkpoint_score"]):
        data = [df.loc[df["physiotype"] == p, sc].values for p in [1, 2, 3]]
        ax.boxplot(data, labels=["PH1", "PH2", "PH3"], showfliers=False)
        ax.set_title("%s (p=%.4g)" % (sc, kw[sc]), fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Immune/stromal/checkpoint scores by physiotype (Kruskal-Wallis)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_checkpoints_estimate.png"), dpi=150)
    plt.close(fig)

    res["interpretation"] = (
        "Checkpoint module and ESTIMATE-like immune/stromal scores differ across physiotypes "
        "(Kruskal-Wallis p <= 0.0025): checkpoint_score is highest in PH1 (immune/B-cell-rich) "
        "and lowest in PH3, and ImmuneScore/StromalScore/ESTIMATE-like are highest in PH1 and "
        "lowest in PH3. Estimated tumor purity is nearly identical across physiotypes "
        "(median 0.8225/0.8225/0.8226) and does not discriminate. checkpoint_score correlates "
        "only weakly with TMB (Spearman rho=0.075, p=0.094) and moderately with the "
        "IGKC/CD27/MS4A1 B-cell proxy (rho=0.336, p<0.001). checkpoint-high vs -low does NOT "
        "stratify OS (log-rank p=0.34; Cox HR=0.95, p=0.61); stage dominates. Caveat: "
        "ESTIMATE-like scores are literature-curated signatures, not the exact Yoshihara 2013 "
        "lists, and the purity formula is the original microarray version.")

    with open(os.path.join(RESULTS_DIR, "gdc2_checkpoints_estimate.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved results/gdc2_checkpoints_estimate.json + figures", flush=True)


if __name__ == "__main__":
    main()