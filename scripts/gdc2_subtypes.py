# GDC2: сравнение физиотипов с известными подтипами LUAD.
# Подтипы TRU / prox.-inflam / prox.-prolif. (TCGA LUAD, Nature 2014, n=230)
# извлечены из Supplementary Table 7 (n13385).
# Сопоставление по case_id, анализ перекрытия (chi2, ARI, доля согласия),
# выживаемость по подтипам как независимая проверка.

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

MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
SUBTYPES = os.path.join(PROCESSED_DATA_DIR, "tcga2014_subtypes.csv")


def main():
    df = pd.read_csv(MERGED)
    sub = pd.read_csv(SUBTYPES)
    sub.columns = ["case_id", "subtype"]
    sub["case_id"] = sub["case_id"].str.strip()
    sub["subtype"] = sub["subtype"].str.strip().replace(
        {"prox.-inflam": "PI", "prox.-prolif.": "PP", "TRU": "TRU"})

    df = df[df["tissue"] == "Tumor"].copy()
    df = df.drop_duplicates(subset="case_id", keep="first")
    df = df.merge(sub, on="case_id", how="left")
    df = df.dropna(subset=["os_days"]).copy()
    df["os_days"] = df["os_days"].astype(float)
    df["event"] = df["event"].astype(int)
    df["age_years"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce").abs()
    df["stage_grp"] = df["stage_grp"].fillna("NA")
    df["physio"] = df["physiotype"].astype(int)

    n_match = int(df["subtype"].notna().sum())
    print("patients:", len(df), "with TRU/PI/PP subtype:", n_match, flush=True)

    res = {"method": "Physiotypes vs TCGA 2014 expression subtypes (TRU/PI/PP)",
           "n_patients": int(len(df)), "n_with_subtype": n_match}

    d = df.dropna(subset=["subtype"]).copy()
    ct = pd.crosstab(pd.Series(d["physio"], name="physiotype"),
                     pd.Series(d["subtype"], name="subtype"))
    ct_rowpct = ct.div(ct.sum(axis=1), axis=0).round(3)
    res["contingency"] = ct.to_dict()
    res["row_percent"] = ct_rowpct.to_dict()

    from scipy.stats import chi2_contingency
    chi2, p_chi, dof, _ = chi2_contingency(ct)
    res["chi2"] = round(float(chi2), 2)
    res["p_chi2"] = float(p_chi)
    res["dof"] = int(dof)

    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(d["physio"].astype(str), d["subtype"])
    res["ari"] = round(float(ari), 3)
    print("crosstab:\n", ct, flush=True)
    print("chi2=%.2f p=%.3g ari=%.3f" % (chi2, p_chi, ari), flush=True)

    # доля согласия: мажоритарный подтип каждого физиотипа
    dom = {}
    for ph in sorted(d["physio"].unique()):
        vc = d.loc[d["physio"] == ph, "subtype"].value_counts()
        dom[int(ph)] = {"dominant_subtype": str(vc.index[0]),
                        "dominant_share": round(float(vc.iloc[0] / vc.sum()), 3),
                        "n": int(vc.sum())}
    res["dominant_subtype_by_physiotype"] = dom
    print("dominant:", dom, flush=True)

    # выживаемость по подтипам (независимая проверка)
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    km_data = {}
    colors = {"TRU": "seagreen", "PI": "darkorange", "PP": "slateblue"}
    for st in ["TRU", "PI", "PP"]:
        sub_d = d[d["subtype"] == st]
        if len(sub_d) == 0:
            continue
        kmf.fit(sub_d["os_days"], sub_d["event"], label=st)
        kmf.plot_survival_function(ax=ax, color=colors[st], lw=2)
        km_data[st] = {"n": int(len(sub_d)), "events": int(sub_d["event"].sum()),
                       "median_os_days": float(kmf.median_survival_time_)
                       if np.isfinite(kmf.median_survival_time_) else None}
    lr = logrank_test(d["os_days"][d["subtype"] == "TRU"], d["os_days"][d["subtype"] == "PI"],
                      event_observed_A=d["event"][d["subtype"] == "TRU"],
                      event_observed_B=d["event"][d["subtype"] == "PI"])
    res["survival_by_subtype"] = km_data
    res["logrank_TRU_vs_PI_p"] = round(float(lr.p_value), 4)
    ax.set_xlabel("time (days)"); ax.set_ylabel("OS probability")
    ax.set_title("TCGA-LUAD: OS by expression subtype (2014)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_subtypes_survival.png"), dpi=150)
    plt.close(fig)
    print("survival by subtype:", json.dumps(km_data), flush=True)

    # выживаемость по подтипам Cox
    d2 = d[["os_days", "event", "subtype", "age_years", "stage_grp"]].dropna(subset=["age_years"]).copy()
    cph = CoxPHFitter()
    d2["subtype"] = d2["subtype"].astype("category")
    cph.fit(d2, duration_col="os_days", event_col="event", formula="subtype + age_years + stage_grp")
    res["subtype_cox"] = {"formula": "subtype + age_years + stage_grp", "n": int(cph._n_examples),
                          "summary": {k: {"coef": round(float(v), 3), "hr": round(float(np.exp(v)), 3),
                                          "p": round(float(cph.summary.loc[k, "p"]), 4)}
                                      for k, v in cph.summary["coef"].items()}}
    print("subtype Cox:", json.dumps(res["subtype_cox"]), flush=True)

    # физиотип внутри подтипа (Cox, контроль подтипа)
    d3 = d[["os_days", "event", "physio", "subtype", "age_years", "stage_grp"]].dropna(
        subset=["age_years"]).copy()
    cph2 = CoxPHFitter()
    cph2.fit(d3, duration_col="os_days", event_col="event",
             formula="physio + subtype + age_years + stage_grp")
    res["physio_controlled_for_subtype_cox"] = {
        "formula": "physio + subtype + age_years + stage_grp", "n": int(cph2._n_examples),
        "summary": {k: {"coef": round(float(v), 3), "hr": round(float(np.exp(v)), 3),
                        "p": round(float(cph2.summary.loc[k, "p"]), 4)}
                    for k, v in cph2.summary["coef"].items()}}
    print("physio controlling subtype Cox:", json.dumps(res["physio_controlled_for_subtype_cox"]), flush=True)

    with open(os.path.join(RESULTS_DIR, "gdc2_subtypes.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved gdc2_subtypes.json", flush=True)


if __name__ == "__main__":
    main()
