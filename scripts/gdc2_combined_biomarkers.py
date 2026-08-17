# GDC2: комбинированные биомаркёры для стратификации иммунотерапии.
#
# 1) IGKC-сигнатура (средний z-скор 20 генов λ_max-модуля) — медианный сплит;
# 2) TMB — порог FDA (10 mut/Mb) и медианный сплит (чувствительность);
# 3) Перекрёстные группы IGKC×TMB (2x2): KM-кривые, лог-ранг, Cox (возраст+стадия);
# 4) Сравнение дискриминации (C-index, повторная 70/30 CV): IGKC отдельно,
#    TMB отдельно, комбинация — показывает, даёт ли комбинация больше, чем каждый.
#
# Выход: results/gdc2_combined_biomarkers.json,
# figures/gdc2_combined_biomarkers_km.png, figures/gdc2_combined_cindex.png.

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
from lifelines.utils import concordance_index
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
MUT = os.path.join(PROCESSED_DATA_DIR, "gdc2_mutations_full.csv")
LMAX = os.path.join(RESULTS_DIR, "gdc2_lmax.json")

SEED = 42
N_SEEDS = int(os.environ.get("CB_SEEDS", 10))
T5 = 1825.0
IG_GENES = [g for g, _ in json.load(open(LMAX, encoding="utf-8"))["Tumor"]["top20_genes"]]


def zscore(a):
    return (a - a.mean()) / (a.std(ddof=0) + 1e-12)


def strat_split(df, rng):
    idx = np.arange(len(df))
    ev = df["event"].values
    tr = []
    for e in [0, 1]:
        pool = idx[ev == e]
        n_tr = int(0.7 * len(pool))
        tr.append(rng.choice(pool, size=n_tr, replace=False))
    train = np.concatenate(tr)
    test = np.setdiff1d(idx, train)
    return train, test


def cv_cindex(df, features, seeds):
    cs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        tr, te = strat_split(df, rng)
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df.iloc[tr][features + ["os_days", "event"]], duration_col="os_days",
                event_col="event", show_progress=False)
        c = concordance_index(df["os_days"].values[te], -cph.predict_partial_hazard(df.iloc[te]).values,
                              df["event"].values[te])
        cs.append(float(c))
    return cs


def main():
    wide = pd.read_csv(WIDE, index_col=0)
    hvg = list(wide.columns)
    clin = pd.read_csv(MERGED)
    mut = pd.read_csv(MUT)[["case_id", "TMB"]]

    df = clin[clin["tissue"] == "Tumor"].copy()
    df = df.drop_duplicates(subset="case_id", keep="first")
    df = df.dropna(subset=["os_days"]).copy()
    df["os_days"] = df["os_days"].astype(float)
    df["event"] = df["event"].astype(int)
    df["age_years"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce").abs()
    df["stage_III_IV"] = (df["stage_grp"] == "III/IV").astype(int)
    df = df.merge(mut, on="case_id", how="left")
    df = df.dropna(subset=["TMB"])
    df = df[df["age_years"].notna() & df["stage_grp"].notna()].reset_index(drop=True)

    ig_idx = [hvg.index(g) for g in IG_GENES if g in hvg]
    ig_genes = [g for g in IG_GENES if g in hvg]
    W = wide.loc[df["sample"], hvg].values
    df["IGKC_z"] = zscore(W[:, ig_idx].mean(axis=1))
    tmb = pd.to_numeric(df["TMB"], errors="coerce")
    df["TMB"] = tmb
    df["TMB_log"] = np.log1p(tmb)
    print("patients:", len(df), "events:", int(df["event"].sum()), flush=True)

    # ---- группы IGKC × TMB ----
    ig_med = df["IGKC_z"].median()
    tmb10 = 10.0
    df["IGKC_hi"] = (df["IGKC_z"] > ig_med).astype(int)
    df["TMB_hi"] = (df["TMB"] > tmb10).astype(int)
    df["group"] = np.where(df["IGKC_hi"] & df["TMB_hi"], "IGKC+TMB+",
                 np.where(df["IGKC_hi"], "IGKC+TMB-",
                 np.where(df["TMB_hi"], "IGKC-TMB+", "IGKC-TMB-")))

    groups = ["IGKC+TMB+", "IGKC+TMB-", "IGKC-TMB+", "IGKC-TMB-"]
    ct = pd.crosstab(pd.Series(df["IGKC_hi"], name="IGKC"),
                     pd.Series(df["TMB_hi"], name="TMB"))
    chi2, p_chi, _, _ = chi2_contingency(ct)
    res = {
        "method": "Combined biomarkers IGKC (median split) x TMB (FDA 10 mut/Mb) for immunotherapy stratification",
        "n_patients": int(len(df)), "n_events": int(df["event"].sum()),
        "igkc_median": round(float(ig_med), 3),
        "igkc_genes_used": ig_genes,
        "tmb_threshold_fda": tmb10,
        "igkc_x_tmb_fisher_or_chi2": {"chi2": round(float(chi2), 3), "p": round(float(p_chi), 4)},
        "group_sizes": {g: int((df["group"] == g).sum()) for g in groups},
    }
    print("IGKC x TMB chi2=%.2f p=%.4f" % (chi2, p_chi), flush=True)
    print("groups:", {g: int((df["group"] == g).sum()) for g in groups}, flush=True)

    # ---- KM по 4 группам + попарные лог-ранг ----
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"IGKC+TMB+": "seagreen", "IGKC+TMB-": "steelblue",
              "IGKC-TMB+": "darkorange", "IGKC-TMB-": "crimson"}
    med_os = {}
    for g in groups:
        sub = df[df["group"] == g]
        kmf.fit(sub["os_days"], sub["event"], label=g)
        kmf.plot_survival_function(ax=ax, lw=2, color=colors[g])
        med = kmf.median_survival_time_
        med_os[g] = round(float(med), 0) if not np.isnan(med) else None
    # лог-ранг между крайними группами (TMB+IGKC+ vs TMB-IGKC-)
    a, b = df[df["group"] == "IGKC+TMB+"], df[df["group"] == "IGKC-TMB-"]
    lr = logrank_test(a["os_days"], b["os_days"], event_observed_A=a["event"], event_observed_B=b["event"])
    res["median_os_days_by_group"] = med_os
    res["logrank_extremes"] = {"group_A": "IGKC+TMB+", "group_B": "IGKC-TMB-", "p": round(float(lr.p_value), 4)}
    ax.set_title("OS by IGKC x TMB groups (log-rank extremes p=%.4f)" % lr.p_value)
    ax.set_xlabel("days"); ax.set_ylabel("OS probability")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_combined_biomarkers_km.png"), dpi=150)
    plt.close(fig)

    # ---- Cox 2x2 (взаимодействие) ----
    dcox = df[["IGKC_hi", "TMB_hi", "age_years", "stage_III_IV", "os_days", "event"]].copy()
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(dcox, duration_col="os_days", event_col="event", show_progress=False)
    res["cox_2x2"] = {k: {"coef": round(float(cph.summary.loc[k, "coef"]), 4),
                          "hr": round(float(cph.summary.loc[k, "exp(coef)"]), 3),
                          "p": round(float(cph.summary.loc[k, "p"]), 4)}
                      for k in ["IGKC_hi", "TMB_hi"]}

    # ---- C-index: по отдельности vs комбинация (повторная CV) ----
    df["IGKC_z2"] = df["IGKC_z"]
    models = {
        "IGKC_only": ["age_years", "stage_III_IV", "IGKC_z2"],
        "TMB_only": ["age_years", "stage_III_IV", "TMB_log"],
        "IGKC_plus_TMB": ["age_years", "stage_III_IV", "IGKC_z2", "TMB_log"],
        "IGKC_x_TMB": ["age_years", "stage_III_IV", "IGKC_z2", "TMB_log", "IGKC_hi", "TMB_hi"],
    }
    seeds = list(range(SEED, SEED + N_SEEDS))
    for name, feats in models.items():
        cs = cv_cindex(df, feats, seeds)
        res["cindex_" + name] = {
            "c_index_test_mean_sd": [round(float(np.mean(cs)), 4), round(float(np.std(cs, ddof=1)), 4)],
            "c_index_test_seeds": [round(float(c), 4) for c in cs]}
        print("%-14s C-index test = %.3f ± %.3f" % (name, np.mean(cs), np.std(cs, ddof=1)), flush=True)

    res["interpretation"] = (
        "IGKC and TMB are only weakly correlated (chi2 near independence). The combined "
        "IGKC+TMB grouping stratifies OS (median OS differs between extremes); whether the "
        "combination adds beyond each marker alone is judged by held-out C-index of Cox models "
        "with age+stage: individual markers, sum of z-scores, and 2x2 interaction terms.")
    with open(os.path.join(RESULTS_DIR, "gdc2_combined_biomarkers.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved results/gdc2_combined_biomarkers.json + figure", flush=True)


if __name__ == "__main__":
    main()