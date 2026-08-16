# GDC2: выживаемость по физиотипам (KM + Cox), tumor-only, пациенты-дедупликация.

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


def main():
    df = pd.read_csv(MERGED)
    df = df[df["tissue"] == "Tumor"].copy()
    df = df.drop_duplicates(subset="case_id", keep="first")
    df = df.dropna(subset=["os_days"]).copy()
    df["os_days"] = df["os_days"].astype(float)
    df["event"] = df["event"].astype(int)

    age = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce")
    age = age.abs() / 365.25
    df["age_years"] = age
    df["stage_grp"] = df["stage_grp"].fillna("NA")

    res = {"method": "KM + CoxPH per physiotype (tumor, patient-level)",
           "n_patients": int(len(df)), "n_events": int(df["event"].sum())}

    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = {1: "seagreen", 2: "darkorange", 3: "slateblue"}
    km_data = {}
    for ph in sorted(df["physiotype"].unique()):
        sub = df[df["physiotype"] == ph]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_days"], sub["event"], label="physiotype %d" % ph)
        kmf.plot_survival_function(ax=ax, color=colors[int(ph)])
        km_data[str(int(ph))] = {
            "n": int(len(sub)), "events": int(sub["event"].sum()),
            "median_os_days": float(kmf.median_survival_time_) if np.isfinite(kmf.median_survival_time_) else None,
            "median_ci": [float(x) if np.isfinite(x) else None for x in kmf.confidence_interval_["physiotype %d" % ph]
                          .median().tolist()] if False else None,
        }
    res["km"] = km_data
    lr = logrank_test(df["os_days"], df["physiotype"].astype(int), df["event"])
    res["logrank_p"] = round(float(lr.p_value), 4)
    print("logrank p = %.4f" % lr.p_value, flush=True)

    cph = CoxPHFitter()
    d = df[["os_days", "event", "physiotype", "age_years", "stage_grp"]].copy()
    d["physiotype"] = d["physiotype"].astype(int)
    d = d[d["age_years"].notna() & d["stage_grp"].notna()]
    cph.fit(d, duration_col="os_days", event_col="event",
            formula="physiotype + age_years + stage_grp")
    res["cox"] = {"formula": "physiotype + age_years + stage_grp",
                  "n": int(cph._n_examples),
                  "summary": {k: {"coef": round(float(v), 3), "hr": round(float(np.exp(v)), 3),
                                  "p": round(float(cph.summary.loc[k, "p"]), 4)}
                              for k, v in cph.summary["coef"].items()}}
    print("COX:", json.dumps(res["cox"], indent=1), flush=True)

    ax.set_xlabel("time (days)"); ax.set_ylabel("overall survival probability")
    ax.set_title("TCGA-LUAD: OS by physiotype (log-rank p=%.4f)" % lr.p_value)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_survival.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "gdc2_survival.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved gdc2_survival.json + gdc2_survival.png", flush=True)
    print("km:", json.dumps(km_data, indent=1), flush=True)


if __name__ == "__main__":
    main()
