# GDC2: выживаемость по IGKC-сигнатуре (λ_max-модуль), стратификация по стадиям,
# мутации EGFR/KRAS/TP53 в Cox. Tumor-only, пациенты-дедупликация.
#
# IGKC-модуль: топ-20 генов главного собственного вектора корреляционной матрицы
# (RMT, gdc2_lmax.json). Сигнатура = средний z-score модуля; высокий балл = высокая
# иммуноглобулиновая/плазматическая нагрузка.

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
MUT = os.path.join(PROCESSED_DATA_DIR, "gdc2_mutations.csv")
LMAX = os.path.join(RESULTS_DIR, "gdc2_lmax.json")


def zscore(a):
    return (a - a.mean()) / (a.std(ddof=0) + 1e-12)


def main():
    wide = pd.read_csv(WIDE, index_col=0)
    df = pd.read_csv(MERGED)
    lmax = json.load(open(LMAX, encoding="utf-8"))
    mut = pd.read_csv(MUT)

    ig_genes = [g for g, _ in lmax["Tumor"]["top20_genes"]]
    missing = [g for g in ig_genes if g not in wide.columns]
    ig_genes = [g for g in ig_genes if g in wide.columns]
    print("IGKC module genes:", len(ig_genes), "missing in wide:", missing, flush=True)

    # широкая таблица: образцы (файлы) x гены
    df["igkc"] = np.nan
    zwide = zscore(wide.values)
    ig_score = zwide[:, [wide.columns.get_loc(g) for g in ig_genes]].mean(axis=1)
    ig_by_sample = dict(zip(wide.index, ig_score))
    df["igkc"] = df["sample"].map(ig_by_sample)

    df = df[df["tissue"] == "Tumor"].copy()
    df = df.drop_duplicates(subset="case_id", keep="first")
    df = df.dropna(subset=["os_days"]).copy()
    df["os_days"] = df["os_days"].astype(float)
    df["event"] = df["event"].astype(int)
    df["age_years"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce").abs() / 365.25
    df["stage_grp"] = df["stage_grp"].fillna("NA")

    # мутации из cBioPortal (GDC WES): one-hot
    df = df.drop(columns=["KRAS", "EGFR", "STK11"], errors="ignore")
    df = df.merge(mut[["case_id", "EGFR", "KRAS", "TP53"]], on="case_id", how="left")
    df["EGFR"] = df["EGFR"].fillna(0).astype(int)
    df["KRAS"] = df["KRAS"].fillna(0).astype(int)
    df["TP53"] = df["TP53"].fillna(0).astype(int)
    n_mut = int(df[["EGFR", "KRAS", "TP53"]].sum(axis=1).gt(0).sum())
    print("patients:", len(df), "with any of EGFR/KRAS/TP53 mut:", n_mut, flush=True)

    res = {"method": "GDC LUAD: IGKC-signature survival, stage stratification, mutation Cox",
           "n_patients": int(len(df)), "n_events": int(df["event"].sum()),
           "igkc_module_genes": ig_genes, "igkc_module_genes_missing": missing}

    # ===== IGKC-сигнатура: KM (медианный сплит) =====
    med = np.median(df["igkc"])
    df["igkc_high"] = (df["igkc"] > med).astype(int)
    res["igkc_median"] = round(float(med), 4)

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ig_km = {}
    for lbl, m in [("low", df["igkc_high"] == 0), ("high", df["igkc_high"] == 1)]:
        sub = df[m]
        kmf.fit(sub["os_days"], sub["event"], label="IGKC %s" % lbl)
        kmf.plot_survival_function(ax=ax, lw=2)
        ig_km[lbl] = {"n": int(len(sub)), "events": int(sub["event"].sum()),
                      "median_os_days": float(kmf.median_survival_time_)
                      if np.isfinite(kmf.median_survival_time_) else None}
    lr = logrank_test(df["os_days"][df["igkc_high"] == 1], df["os_days"][df["igkc_high"] == 0],
                      event_observed_A=df["event"][df["igkc_high"] == 1],
                      event_observed_B=df["event"][df["igkc_high"] == 0])
    ig_km["logrank_p"] = round(float(lr.p_value), 4)
    res["igkc_km"] = ig_km
    ax.set_xlabel("time (days)"); ax.set_ylabel("OS probability")
    ax.set_title("TCGA-LUAD: OS by IGKC signature (median split, log-rank p=%.4f)" % lr.p_value)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_survival_igkc.png"), dpi=150)
    plt.close(fig)
    print("IGKC KM:", json.dumps(ig_km), flush=True)

    # ===== Cox: IGKC непрерывный + стадия + возраст =====
    cph = CoxPHFitter()
    d = df[["os_days", "event", "igkc", "age_years", "stage_grp"]].dropna(subset=["age_years"]).copy()
    cph.fit(d, duration_col="os_days", event_col="event", formula="igkc + age_years + stage_grp")
    res["igkc_cox"] = {"formula": "igkc + age_years + stage_grp", "n": int(cph._n_examples),
                       "summary": {k: {"coef": round(float(v), 3), "hr": round(float(np.exp(v)), 3),
                                       "p": round(float(cph.summary.loc[k, "p"]), 4)}
                                   for k, v in cph.summary["coef"].items()}}
    print("IGKC Cox:", json.dumps(res["igkc_cox"]), flush=True)

    # ===== Стратификация по стадиям внутри физиотипов =====
    df["physio"] = df["physiotype"].astype(int)
    df["stage_bin"] = np.where(df["stage_grp"] == "III/IV", "III/IV", "I/II")
    strata = {}
    for ph in sorted(df["physio"].unique()):
        sub = df[df["physio"] == ph]
        row = {"n": int(len(sub)), "events": int(sub["event"].sum())}
        groups = [g for g in ["I/II", "III/IV"] if (sub["stage_bin"] == g).sum() > 1]
        if len(groups) == 2:
            lr = logrank_test(sub["os_days"][sub["stage_bin"] == groups[0]],
                              sub["os_days"][sub["stage_bin"] == groups[1]],
                              event_observed_A=sub["event"][sub["stage_bin"] == groups[0]],
                              event_observed_B=sub["event"][sub["stage_bin"] == groups[1]])
            row["logrank_stage_within_physiotype_p"] = round(float(lr.p_value), 4)
            kmf2 = KaplanMeierFitter()
            for g in groups:
                dg = sub[sub["stage_bin"] == g]
                kmf2.fit(dg["os_days"], dg["event"])
                row["median_%s" % g] = float(kmf2.median_survival_time_) \
                    if np.isfinite(kmf2.median_survival_time_) else None
        strata[str(ph)] = row
    res["stage_within_physiotype"] = strata
    print("stage stratification:", json.dumps(strata), flush=True)

    # стратифицированный Cox: физиотип внутри группы стадии
    strat_cox = {}
    for g in ["I/II", "III/IV"]:
        dg = df[(df["stage_bin"] == g) & df["age_years"].notna()].copy()
        if len(dg) < 20:
            continue
        cph2 = CoxPHFitter()
        dg2 = dg[["os_days", "event", "physio", "age_years"]].copy()
        cph2.fit(dg2, duration_col="os_days", event_col="event", formula="physio + age_years")
        strat_cox[g] = {"n": int(cph2._n_examples),
                        "physio": {"coef": round(float(cph2.summary.loc["physio", "coef"]), 3),
                                   "hr": round(float(np.exp(cph2.summary.loc["physio", "coef"])), 3),
                                   "p": round(float(cph2.summary.loc["physio", "p"]), 4)}}
    res["physiotype_within_stage_cox"] = strat_cox
    print("physiotype within stage Cox:", json.dumps(strat_cox), flush=True)

    # ===== Cox с мутациями =====
    d3 = df[["os_days", "event", "physio", "age_years", "stage_grp", "EGFR", "KRAS", "TP53"]].dropna(
        subset=["age_years"]).copy()
    cph3 = CoxPHFitter()
    cph3.fit(d3, duration_col="os_days", event_col="event",
             formula="physio + age_years + stage_grp + EGFR + KRAS + TP53")
    res["mutation_cox"] = {"formula": "physio + age_years + stage_grp + EGFR + KRAS + TP53",
                           "n": int(cph3._n_examples),
                           "summary": {k: {"coef": round(float(v), 3), "hr": round(float(np.exp(v)), 3),
                                           "p": round(float(cph3.summary.loc[k, "p"]), 4)}
                                       for k, v in cph3.summary["coef"].items()}}
    print("mutation Cox:", json.dumps(res["mutation_cox"]), flush=True)

    with open(os.path.join(RESULTS_DIR, "gdc2_survival_igkc.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved gdc2_survival_igkc.json", flush=True)


if __name__ == "__main__":
    main()
