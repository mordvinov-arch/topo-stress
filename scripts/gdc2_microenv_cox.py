# GDC2: мультивариантная Cox — физиотип + микроокружение (MCP-скоры) + мутации
# + стадия. Tumor-only пациенты. Оценка: коэффициенты/HR/p + C-index.
#
# Вход: gdc2_clinical_merged.csv, gdc2_immune_scores.csv (образцы),
# gdc2_mutations_full.csv (пациенты, 11 генов + TMB).
#
# Выход: results/gdc2_microenv_cox.json, figures/gdc2_microenv_cox.png.

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
IMMUNE = os.path.join(PROCESSED_DATA_DIR, "gdc2_immune_scores.csv")
MUTFULL = os.path.join(PROCESSED_DATA_DIR, "gdc2_mutations_full.csv")


def main():
    clin = pd.read_csv(MERGED)
    imm = pd.read_csv(IMMUNE)
    mut = pd.read_csv(MUTFULL)

    df = clin[clin["tissue"] == "Tumor"].copy()
    df = df.drop_duplicates(subset="case_id", keep="first")
    df = df.merge(imm[["sample", "B_cells", "T_cells", "CD8_T_cells", "Cytotoxic",
                       "NK_cells", "Monocytic", "Myeloid_DC", "Neutrophils",
                       "Endothelial", "Fibroblasts", "proliferation", "stroma",
                       "hypoxia", "EMT"]], on="sample", how="left")
    mut_cols = ["STK11", "KEAP1", "MET", "BRAF", "ROS1", "ALK", "NF1", "RBM10", "TMB"]
    df = df.drop(columns=[c for c in mut_cols if c in df.columns], errors="ignore")
    df = df.merge(mut[["case_id"] + mut_cols], on="case_id", how="left")
    df["os_days"] = pd.to_numeric(df["os_days"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df["age_years"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce").abs()
    df["physio"] = df["physiotype"].astype(int)
    df["stage_grp"] = df["stage_grp"].fillna("NA")
    for g in ["STK11", "KEAP1", "MET", "BRAF", "ROS1", "ALK", "NF1", "RBM10"]:
        df[g] = df[g].fillna(0).astype(int)
    df["TMB_log"] = np.log1p(pd.to_numeric(df["TMB"], errors="coerce"))

    # стандартизация скоров микроокружения для сопоставимости HR
    zcols = ["B_cells", "T_cells", "CD8_T_cells", "Cytotoxic", "NK_cells",
             "Monocytic", "Myeloid_DC", "Neutrophils", "Endothelial", "Fibroblasts",
             "proliferation", "stroma", "hypoxia", "EMT"]
    for c in zcols:
        df[c + "_z"] = (df[c] - df[c].mean()) / (df[c].std(ddof=0) + 1e-12)

    formula = ("physio + age_years + stage_grp + T_cells_z + CD8_T_cells_z + "
               "Cytotoxic_z + B_cells_z + Monocytic_z + Fibroblasts_z + "
               "stroma_z + EMT_z + STK11 + KEAP1 + MET + TMB_log")
    d = df[["os_days", "event", "physio", "age_years", "stage_grp",
            "T_cells_z", "CD8_T_cells_z", "Cytotoxic_z", "B_cells_z", "Monocytic_z",
            "Fibroblasts_z", "stroma_z", "EMT_z",
            "STK11", "KEAP1", "MET", "TMB_log"]].dropna(subset=["os_days", "event",
                                                                  "age_years"]).dropna(how="any").copy()
    print("model n:", len(d), "events:", int(d["event"].sum()), flush=True)

    cph = CoxPHFitter()
    cph.fit(d, duration_col="os_days", event_col="event", formula=formula)
    cidx = concordance_index(d["os_days"], -cph.predict_partial_hazard(d), d["event"])
    summary = {k: {"coef": round(float(v), 3), "hr": round(float(np.exp(v)), 3),
                   "ci95_low": round(float(cph.summary.loc[k, "exp(coef) lower 95%"]), 3),
                   "ci95_up": round(float(cph.summary.loc[k, "exp(coef) upper 95%"]), 3),
                   "p": round(float(cph.summary.loc[k, "p"]), 4)}
               for k, v in cph.summary["coef"].items()}
    res = {"method": "Multivariate Cox: physiotype + microenvironment (MCP-like) + mutations + stage",
           "formula": formula, "n": int(cph._n_examples), "events": int(d["event"].sum()),
           "c_index": round(float(cidx), 4), "summary": summary}
    print("C-index=%.3f" % cidx, flush=True)
    for k, s in summary.items():
        print("  %-14s HR=%.2f [%.2f, %.2f] p=%.4f" % (k, s["hr"], s["ci95_low"], s["ci95_up"], s["p"]), flush=True)

    with open(os.path.join(RESULTS_DIR, "gdc2_microenv_cox.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ===== рисунок: forest plot HR =====
    items = list(summary.items())
    items = [it for it in items if not it[0].startswith("stage_grp")]
    names = [k for k, _ in items]
    hrs = [v["hr"] for _, v in items]
    lo = [v["ci95_low"] for _, v in items]
    hi = [v["ci95_up"] for _, v in items]
    sig = [v["p"] < 0.05 for _, v in items]
    order = np.argsort(hrs)
    fig, ax = plt.subplots(figsize=(7, max(6, 0.45 * len(names) + 2)))
    ypos = np.arange(len(names))
    ax.axvline(1, color="k", lw=0.8)
    for i, o in enumerate(order):
        ax.errorbar(hrs[o], i, xerr=[[hrs[o] - lo[o]], [hi[o] - hrs[o]]],
                    fmt="o", color="crimson" if sig[o] else "steelblue", capsize=3)
        ax.text(1.02, i, "%.2f (p=%.3g)" % (hrs[o], summary[names[o]]["p"]), fontsize=8, va="center")
    ax.set_yticks(ypos); ax.set_yticklabels([names[o] for o in order])
    ax.set_xlabel("hazard ratio (95% CI)")
    ax.set_title("Multivariate Cox (C-index=%.3f)" % cidx)
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_microenv_cox.png"), dpi=150)
    plt.close(fig)
    print("saved gdc2_microenv_cox.json + figure", flush=True)


if __name__ == "__main__":
    main()
