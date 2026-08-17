# GDC2: сравнение физиотипов с известными сигнатурами из литературы.
#
# Списки генов взяты дословно из файла "решение по сигнатурам.txt":
#   T-cell inflamed GEP18 (Ayers 2017), cytolytic (Rooney 2015), IFN-gamma (6),
#   B-cell, proliferation (Whitfield 2006), stromal (Yoshihara 2013),
#   hypoxia (Buffa 2010), EMT (Kalluri & Weinberg 2009).
# Дополнительно: Hallmark EMT/Hypoxia/Inflammatory/IFN-gamma/G2M/E2F/Angiogenesis
# из локального MSigDB_Hallmark_2020.gmt.
#
# Для каждой сигнатуры: скор = средний z-скор log1p TPM генов; сравнение по
# физиотипам (Kruskal-Wallis), выживаемость (медианный сплит, лог-ранг, Cox
# возраст+стадия). Справочно: C-index Cox с физиотипом.
#
# Выход: results/gdc2_known_signatures.json,
# figures/gdc2_known_signatures_heat.png, figures/gdc2_known_signatures_km.png.

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
from scipy.stats import kruskal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

TPM = os.path.join(PROCESSED_DATA_DIR, "gdc_tpm.npz")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
GMT = os.path.join(PROCESSED_DATA_DIR, os.pardir, "genesets", "MSigDB_Hallmark_2020.gmt")
GDC_DIR = os.path.join(PROCESSED_DATA_DIR, os.pardir, "gdc")

# Дословно из "решение по сигнатурам.txt"
LIT_SIGS = {
    "Tcell_inflamed_GEP18_Ayers": ["CCL2", "CCL3", "CCL4", "CCL5", "CD8A", "CXCL9", "CXCL10", "CXCR6",
                                    "GZMK", "HLA-DMA", "HLA-DMB", "HLA-DOA", "HLA-DOB", "IRF1", "STAT1",
                                    "TAP1", "TAP2", "PSMB9"],
    "Cytolytic_Rooney": ["GZMA", "PRF1"],
    "IFNG_6": ["IFNG", "STAT1", "CXCL9", "CXCL10", "IDO1", "HLA-DRA"],
    "Bcell": ["CD19", "MS4A1", "CD79A", "CD79B", "BLK", "BANK1", "PAX5"],
    "Proliferation_Whitfield": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6",
                                "MCM7", "CCNB1", "CCNB2", "CDK1", "BIRC5", "AURKA", "AURKB"],
    "Stromal_Yoshihara": ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "FN1", "VIM", "ACTA2",
                          "FAP", "PDGFRB", "SPARC"],
    "Hypoxia_Buffa": ["HIF1A", "VEGFA", "VEGFC", "CA9", "SLC2A1", "LDHA", "PGK1", "ENO1", "ALDOA",
                      "GAPDH"],
    "EMT_Kalluri": ["CDH1", "CDH2", "VIM", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "FN1",
                    "MMP2", "MMP9"],
}

HALLMARK_KEYS = ["EMT", "Hypoxia", "Inflammatory_Response", "Interferon_Gamma_Response",
                 "G2M_Checkpoint", "E2F_Targets", "Angiogenesis", "IL6_JAK_STAT3_Signaling"]


def load_hallmark(path):
    sigs = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            name = parts[0]
            for key in HALLMARK_KEYS:
                if name.replace(" ", "_") == key or name == key:
                    sigs["Hallmark_" + key] = parts[2:]
    return sigs


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

    sigs = dict(LIT_SIGS)
    sigs.update(load_hallmark(GMT))
    for k in sigs:
        sigs[k] = [g.upper() for g in sigs[k]]

    clin = pd.read_csv(MERGED)
    clin = clin[clin["tissue"] == "Tumor"].drop_duplicates(subset="case_id", keep="first")
    clin = clin.dropna(subset=["os_days"]).reset_index(drop=True)
    sidx = [samples.index(s) for s in clin["sample"]]
    logt = np.log1p(T[sidx])
    df = clin.copy()
    df["age_years"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce").abs()
    df["stage_III_IV"] = (df["stage_grp"] == "III/IV").astype(int)
    df["os_days"] = df["os_days"].astype(float)
    df["event"] = df["event"].astype(int)

    def zscore(x):
        return (x - x.mean()) / (x.std(ddof=0) + 1e-12)

    res = {"method": "Known literature signatures vs physiotypes (TCGA-LUAD)",
           "n": int(len(df)), "n_events": int(df["event"].sum())}

    score_rows = {}
    missing_report = {}
    for name, gs in sigs.items():
        col = []
        used = []
        for g in gs:
            ens = sym_map.get(g, sym_map.get(g.upper()))
            if ens is None or ens not in gene2idx:
                continue
            col.append(zscore(logt[:, gene2idx[ens]]))
            used.append(g)
        missing_report[name] = [g for g in gs if g not in used]
        if len(col) < max(2, len(gs) // 2):
            print("skip %s: only %d/%d genes measured" % (name, len(col), len(gs)), flush=True)
            continue
        score_rows[name] = np.array(col).mean(axis=0)
        print("%-32s genes %d/%d" % (name, len(col), len(gs)), flush=True)
    res["genes_missing_per_signature"] = missing_report

    scor = pd.DataFrame(score_rows)
    df = pd.concat([df, scor], axis=1)

    # ---- по каждой сигнатуре: KW, выживаемость, Cox ----
    out = {}
    kmf = KaplanMeierFitter()
    for name in scor.columns:
        v = df[name]
        groups = [v[df["physiotype"] == p].values for p in [1, 2, 3]]
        st, pkw = kruskal(*groups)
        med = {str(p): round(float(np.median(g)), 4) for p, g in zip([1, 2, 3], groups)}

        hi = df[v > v.median()]
        lo = df[v <= v.median()]
        lr = logrank_test(hi["os_days"], lo["os_days"], event_observed_A=hi["event"], event_observed_B=lo["event"])
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(df[["age_years", "stage_III_IV", name, "os_days", "event"]].dropna(subset=["age_years", "stage_III_IV"]),
                duration_col="os_days", event_col="event", show_progress=False)
        r = cph.summary.loc[name]
        out[name] = {"kruskal_p": round(float(pkw), 5), "median_by_physio": med,
                     "logrank_high_vs_low": round(float(lr.p_value), 4),
                     "cox_hr": round(float(r["exp(coef)"]), 3),
                     "cox_ci": [round(float(r["exp(coef) lower 95%"]), 3), round(float(r["exp(coef) upper 95%"]), 3)],
                     "cox_p": round(float(r["p"]), 4)}
        print("  %-28s KW=%.3g logrank=%.4f CoxHR=%.2f p=%.4f" %
              (name, pkw, lr.p_value, r["exp(coef)"], r["p"]), flush=True)
    res["signatures"] = out

    # ---- справочно: Cox с физиотипом ----
    dph = df[["age_years", "stage_III_IV", "physiotype", "os_days", "event"]].dropna(subset=["age_years", "stage_III_IV"])
    dph = pd.get_dummies(dph, columns=["physiotype"], drop_first=True)
    cph_ph = CoxPHFitter(penalizer=0.0)
    cph_ph.fit(dph, duration_col="os_days", event_col="event", show_progress=False)
    res["physiotype_cox_reference"] = {"c_index": round(float(cph_ph.concordance_index_), 4)}
    print("physiotype Cox C-index:", round(cph_ph.concordance_index_, 4), flush=True)

    # ---- рисунок: тепловая карта медианных скоров по физиотипам ----
    rows = [out[name]["median_by_physio"] for name in scor.columns]
    meddf = pd.DataFrame(rows, index=scor.columns, columns=["PH1", "PH2", "PH3"]).astype(float)
    zr = (meddf - meddf.values.mean()) / (meddf.values.std(ddof=0) + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(scor.columns) + 2))
    im = ax.imshow(zr.values, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["PH1", "PH2", "PH3"])
    ax.set_yticks(range(len(scor.columns)))
    ax.set_yticklabels([f"{n}" for n in scor.columns], fontsize=7)
    ax.set_title("Known signatures: median score z by physiotype")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_known_signatures_heat.png"), dpi=150)
    plt.close(fig)

    # ---- рисунок: KM для сигнатуры с наименьшим logrank p ----
    best = min(out.items(), key=lambda kv: kv[1]["logrank_high_vs_low"])
    name, r = best
    v = df[name]
    hi, lo = df[v > v.median()], df[v <= v.median()]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    kmf.fit(hi["os_days"], hi["event"], label="high")
    kmf.plot_survival_function(ax=ax, lw=2, color="crimson")
    kmf.fit(lo["os_days"], lo["event"], label="low")
    kmf.plot_survival_function(ax=ax, lw=2, color="steelblue")
    ax.set_title("%s: OS high vs low (log-rank p=%.4f)" % (name, r["logrank_high_vs_low"]))
    ax.set_xlabel("days"); ax.set_ylabel("OS probability")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_known_signatures_km.png"), dpi=150)
    plt.close(fig)
    res["best_signature_for_survival"] = name

    res["interpretation"] = (
        "Published signatures (exact gene lists from the provided file) are evaluated against "
        "physiotypes: Kruskal-Wallis differences across physiotypes, and survival stratification "
        "by median split with Cox adjustment for age/stage. The physiotype Cox serves as a "
        "reference for discrimination. Signatures that are strongly physiotype-dependent overlap "
        "with the physiotype's biology (immune/stromal/proliferative).")

    with open(os.path.join(RESULTS_DIR, "gdc2_known_signatures.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved results/gdc2_known_signatures.json + figures", flush=True)


if __name__ == "__main__":
    main()