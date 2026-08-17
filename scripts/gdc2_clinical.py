# GDC2: клинические стадии (Xena TCGA-LUAD) против физиотипов и PC1-оси.
# Проверяет, коррелирует ли транскрипционная структура (физиотипы, PC1-градиент) со стадией,
# и выдаёт данные для выживаемости (vital_status, дни) по физиотипам.

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

CLIN = os.path.join(PROCESSED_DATA_DIR, "gdc_clinical_xena.tsv")
LABELS = os.path.join(PROCESSED_DATA_DIR, "gdc2_labels.csv")
WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")


def main():
    clin = pd.read_csv(CLIN, sep="\t", dtype=str)
    clin = clin.rename(columns={"sampleID": "sample"})
    clin["patient_id"] = clin["_PATIENT"].str.strip().str[:12]
    keep = ["patient_id", "pathologic_stage", "pathologic_T", "pathologic_N",
            "pathologic_M", "vital_status", "days_to_death", "days_to_last_followup",
            "age_at_initial_pathologic_diagnosis", "gender", "histological_type",
            "Expression_Subtype", "KRAS", "EGFR", "STK11"]
    clin = clin[[c for c in keep if c in clin.columns]]
    clin = clin.drop_duplicates(subset="patient_id", keep="first")

    import re
    def parse_stage(s):
        if not isinstance(s, str):
            return np.nan
        m = re.search(r"STAGE\s*([IV]+)", s.upper())
        if not m:
            return np.nan
        roman = m.group(1)
        return {"I": "I", "II": "II", "III": "III", "IV": "IV"}.get(roman, np.nan)

    clin["stage"] = clin["pathologic_stage"].map(parse_stage)
    clin["stage_grp"] = clin["stage"].map(
        lambda s: "I/II" if s in ("I", "II") else ("III/IV" if s in ("III", "IV") else np.nan))

    lab = pd.read_csv(LABELS)
    merged = lab.merge(clin, left_on="case_id", right_on="patient_id", how="left")
    print("merged rows:", len(merged), "with stage:", merged["stage"].notna().sum(), flush=True)

    res = {}
    tumor = merged[merged["tissue"] == "Tumor"]

    ct = pd.crosstab(pd.Series(tumor["physiotype"], name="physiotype"),
                     pd.Series(tumor["stage"].fillna("NA"), name="stage"))
    chi2, p, _, _ = chi2_contingency(ct)
    res["physiotype_x_stage_all"] = {"table": ct.to_dict(), "chi2": round(float(chi2), 3),
                                     "p": round(float(p), 4), "n": int(ct.values.sum())}
    print("physiotype x stage (all):\n", ct, "\nchi2=%.2f p=%.4f" % (chi2, p), flush=True)

    c2 = pd.crosstab(pd.Series(tumor["physiotype"], name="physiotype"),
                     pd.Series(tumor["stage_grp"].fillna("NA"), name="stage(I/II vs III/IV)"))
    c2 = c2[["I/II", "III/IV"]] if set(["I/II", "III/IV"]) <= set(c2.columns) else c2
    chi2, p, _, _ = chi2_contingency(c2)
    res["physiotype_x_stage_binary"] = {"table": c2.to_dict(), "chi2": round(float(chi2), 3),
                                        "p": round(float(p), 4), "n": int(c2.values.sum())}
    print("physiotype x stage I/II vs III/IV:\n", c2, "\nchi2=%.2f p=%.4f" % (chi2, p), flush=True)

    wide = pd.read_csv(WIDE, index_col=0)
    Z = wide.values
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-12)
    pc1 = Z @ np.linalg.svd(Z, full_matrices=False)[2][0]
    pc1sign = np.where(pc1 > np.median(pc1), "PC1+", "PC1-")
    merged["pc1sign"] = pd.Series(pc1sign, index=wide.index).reindex(merged["sample"]).values
    t2 = merged[(merged["tissue"] == "Tumor") & (merged["stage_grp"].isin(["I/II", "III/IV"]))]
    c3 = pd.crosstab(pd.Series(t2["pc1sign"], name="PC1"), pd.Series(t2["stage_grp"], name="stage"))
    chi2, p, _, _ = chi2_contingency(c3)
    res["PC1_x_stage_binary_tumor"] = {"table": c3.to_dict(), "chi2": round(float(chi2), 3),
                                       "p": round(float(p), 4), "n": int(c3.values.sum())}
    print("PC1 x stage (tumor):\n", c3, "\nchi2=%.2f p=%.4f" % (chi2, p), flush=True)

    merged["os_days"] = pd.to_numeric(merged["days_to_death"], errors="coerce").fillna(
        pd.to_numeric(merged["days_to_last_followup"], errors="coerce"))
    merged["event"] = (merged["vital_status"].astype(str).str.strip().str.lower()
                       .isin(["dead", "deceased"])).astype(int)
    surv = merged[merged["tissue"] == "Tumor"].drop_duplicates(subset="case_id", keep="first")
    surv = surv.dropna(subset=["os_days"]).reset_index(drop=True)
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    med = {}
    for k, g in surv.groupby("physiotype"):
        kmf.fit(g["os_days"].astype(float), g["event"].astype(int))
        m = kmf.median_survival_time_
        med[str(int(k))] = round(float(m), 0) if not np.isnan(m) else None
    res["survival_summary"] = {
        "patients_with_os": int(len(surv)),
        "events": int(surv["event"].sum()),
        "per_physiotype": {
            str(int(k)): {"n": int(len(g)), "events": int(g["event"].sum()),
                          "median_os_days": med[str(int(k))]}
            for k, g in surv.groupby("physiotype")
        },
    }
    print("survival summary:", json.dumps(res["survival_summary"], indent=2), flush=True)

    res["n_patients"] = int(merged["patient_id"].nunique())
    res["n_with_stage"] = int(merged["stage"].notna().sum())
    with open(os.path.join(RESULTS_DIR, "gdc2_clinical.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved gdc2_clinical.json", flush=True)

    out = merged[["sample", "tissue", "physiotype", "case_id", "stage", "stage_grp",
                  "vital_status", "os_days", "event", "age_at_initial_pathologic_diagnosis",
                  "gender", "KRAS", "EGFR", "STK11"]]
    out.to_csv(os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv"), index=False)
    print("saved gdc2_clinical_merged.csv", flush=True)


if __name__ == "__main__":
    main()
