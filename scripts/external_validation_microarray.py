# Внешняя валидация IGKC-сигнатуры и физиотипов на микрочиповых когортах (План B).
#
# Когорты:
#   GSE68465 (n=462, GPL96/HG-U133A, LUAD, OS месяцы, vital_status)
#   GSE50081 (n=181, GPL570/HG-U133 Plus 2.0, ранние NSCLC, OS годы, status)
#   GSE72094 (n=442, GPL15048/HuRSTA (merck-*), LUAD, OS дни, vital_status)
# Платформы не содержат прямого измерения IGKC: используется суррогат —
# средний per-gene z-скор log1p экспрессии по B-плазматическому набору генов,
# валидированный против IGKC на TCGA (полный TPM). Набор генов фиксирован как
# пересечение генов, доступных на ВСЕХ трёх платформах:
#   MZB1, TNFRSF17, DERL3, CD79A, POU2AF1, JCHAIN, CD19, SDC1, XBP1, IRF4,
#   FKBP11, GPR183 (+ IGHG3, IGLL5 где есть).
#
# Протокол:
#   1) series matrix -> клиника + экспрессия probe->gene (среднее по пробам);
#   2) per-gene z внутри когорты (log1p) -> суррогатный скор;
#   3) медианный сплит -> KM + log-rank + Cox (скор + возраст + стадия);
#   4) на TCGA: корреляция суррогата с IGKC и Cox с суррогатом vs IGKC;
#   5) pooled: стратифицированный Cox (strata=cohort) по микрочиповым когортам
#      и по всем 4 когортам (TCGA-суррогат на тех же генах).
#
# Выход: results/external_validation_microarray.json,
#   figures/external_validation_microarray_km_<cohort>.png,
#   figures/external_validation_microarray_km_pooled.png.

import gzip
import json
import os
import re
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

TEMP = r"C:\Users\46AD~1\AppData\Local\Temp\opencode"
MERK = os.path.join(PROCESSED_DATA_DIR, "probe_map_gpl15048.json")
MAP96 = os.path.join(PROCESSED_DATA_DIR, "probe_map_gpl96.json")
MAP570 = os.path.join(PROCESSED_DATA_DIR, "probe_map_gpl570.json")
TPM = os.path.join(PROCESSED_DATA_DIR, "gdc_tpm.npz")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")

# Суррогатный набор: пересечение доступных на всех трёх платформах.
SURROGATE = ["MZB1", "TNFRSF17", "DERL3", "CD79A", "POU2AF1", "JCHAIN", "CD19",
             "SDC1", "XBP1", "IRF4", "FKBP11", "GPR183"]

COHORTS = {
    "GSE68465": {
        "matrix": os.path.join(TEMP, "GSE68465_series_matrix.txt.gz"),
        "probe_map": MAP96,
        "stage_parser": "tnm",
        "time_scale": "months",
    },
    "GSE50081": {
        "matrix": os.path.join(TEMP, "GSE50081_series_matrix.txt.gz"),
        "probe_map": MAP570,
        "stage_parser": "abc",
        "time_scale": "years",
    },
    "GSE72094": {
        "matrix": os.path.join(TEMP, "GSE72094_series_matrix.txt.gz"),
        "probe_map": MERK,
        "stage_parser": "abc",
        "time_scale": "days",
    },
}


def zscore_rows(df):
    return df.apply(lambda c: (c - c.mean()) / (c.std(ddof=0) + 1e-12), axis=0)


def parse_stage_tnm(raw):
    # 'pN0pT1' / 'pN1pT2' -> стадия (I/II/III/IV)
    if not isinstance(raw, str):
        return np.nan
    t = re.search(r"pT(\d+[ab]?)", raw)
    n = re.search(r"pN(\d+)", raw)
    t = int(t.group(1)) if t else None
    n = int(n.group(1)) if n else None
    if t is None or n is None:
        return np.nan
    if n == 0:
        if t == 1:
            return 1
        if t == 2:
            return 1
        if t in (3, 4):
            return 2
        return 2
    if n == 1:
        if t == 1:
            return 2
        if t in (2, 3):
            return 2
        return 3
    if n == 2:
        return 3
    if n == 3:
        return 3
    return 4


def parse_stage_abc(raw):
    # '1A' '1B' '2A' '2B' '3' '4' '1' -> 1..4
    if not isinstance(raw, str):
        return np.nan
    m = re.match(r"(\d+)", str(raw))
    if not m:
        return np.nan
    s = int(m.group(1))
    if s >= 1:
        return s
    return np.nan


def read_series_matrix(path):
    """Возвращает (expr_df: probe x sample, clin_df: sample x field)."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        lines = [l.rstrip("\n") for l in fh]

    sample_ids = None
    expr_rows = []
    chars = {}
    in_table = False
    for line in lines:
        if line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        if line.startswith("!series_matrix_table_end"):
            in_table = False
            continue
        if line.startswith('"ID_REF"'):
            cols = [c.strip('"') for c in line.split("\t")]
            sample_ids = cols[1:]
            continue
        if line.startswith("!Sample_characteristics"):
            vals = line.split("\t")[1:]
            key = None
            parsed = []
            for v in vals:
                v = v.strip('"')
                if ":" in v:
                    k, val = v.split(":", 1)
                    if key is None:
                        key = k
                    parsed.append(val.strip())
                else:
                    parsed.append("")
            if key is not None:
                chars[key] = parsed
            continue
        if in_table and line.startswith('"'):
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            pid = parts[0].strip('"')
            vals = [float(x) if x else np.nan for x in parts[1:]]
            expr_rows.append((pid, vals))

    if sample_ids is None:
        raise ValueError("no ID_REF header found")
    expr = pd.DataFrame({p: vals for p, vals in expr_rows}).T
    expr.columns = sample_ids
    clin = pd.DataFrame(chars, index=sample_ids)
    return expr, clin


def ensg_symbol_map():
    f = os.path.join("data", "gdc")
    from glob import glob
    src = glob(os.path.join(f, "*.rna_seq.augmented_star_gene_counts.tsv"))[0]
    out = {}
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#") or line.startswith("gene_id\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            out[parts[0].split(".")[0]] = parts[1]
    return out


def load_tcga():
    tpm = np.load(TPM, allow_pickle=True)
    T = tpm["tpm"]
    genes = list(tpm["genes"])
    samples = list(tpm["samples"])
    ensg = ensg_symbol_map()
    sym = [ensg.get(g.split(".")[0]) for g in genes]
    df = pd.DataFrame(T, index=samples, columns=sym)
    return df


def tcga_surrogate_scores(exp_tcga, genes):
    """На TCGA: скор суррогата (средний per-gene z log1p TPM) по данным генам и корреляция с IGKC."""
    present = [g for g in genes if g in exp_tcga.columns]
    if not present:
        return None, None, None
    sub = exp_tcga[present].astype(float)
    z = zscore_rows(np.log1p(sub)).mean(axis=1)
    igkc = np.log1p(exp_tcga["IGKC"]) if "IGKC" in exp_tcga.columns else None
    if igkc is None:
        return z, None, present
    r = float(np.corrcoef(z.values, igkc.values)[0, 1])
    return z, r, present


def make_clinical(expr, clin, cohort_cfg):
    c = clin.copy()
    if cohort_cfg["time_scale"] == "months":
        c["os_days"] = pd.to_numeric(c.get("months_to_last_contact_or_death"), errors="coerce") * 30.4375
        c["event"] = (c.get("vital_status").astype(str).str.lower().str.strip() == "dead").astype(int)
        c["age"] = pd.to_numeric(c.get("age"), errors="coerce")
        c["stage_num"] = c.get("disease_stage").map(parse_stage_tnm)
    elif cohort_cfg["time_scale"] == "years":
        c["os_days"] = pd.to_numeric(c.get("survival time"), errors="coerce") * 365.25
        c["event"] = (c.get("status").astype(str).str.lower().str.strip() == "dead").astype(int)
        c["age"] = pd.to_numeric(c.get("age"), errors="coerce")
        c["stage_num"] = c.get("Stage").map(parse_stage_abc)
    elif cohort_cfg["time_scale"] == "days":
        c["os_days"] = pd.to_numeric(c.get("survival_time_in_days"), errors="coerce")
        c["event"] = (c.get("vital_status").astype(str).str.lower().str.strip() == "dead").astype(int)
        c["age"] = pd.to_numeric(c.get("age_at_diagnosis"), errors="coerce")
        c["stage_num"] = c.get("Stage").map(parse_stage_abc)
    keep = ["os_days", "event", "age", "stage_num"]
    clin2 = pd.DataFrame({k: c[k] for k in keep}, index=c.index)
    clin2 = clin2[clin2["os_days"] > 0]
    return clin2


def build_expression(expr, probe_map_path):
    probe_map = json.load(open(probe_map_path, encoding="utf-8"))
    probe_map = {k: (v[0] if isinstance(v, list) else v) for k, v in probe_map.items()}
    syms = pd.Series(probe_map, dtype=object)
    # probe -> symbol
    ok = expr.index.intersection(syms.index)
    e = expr.loc[ok]
    e.index = syms.loc[ok].values
    e = e.groupby(level=0).mean()
    e = e.astype(float)
    return e


def survival_block(df, group_col, title, fname):
    kmf = KaplanMeierFitter()
    out = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = sorted(df[group_col].unique())
    for g in groups:
        d = df[df[group_col] == g]
        kmf.fit(d["os_days"], d["event"], label="%s=%d (n=%d, ev=%d)" % (group_col, g, len(d), int(d["event"].sum())))
        kmf.plot_survival_function(ax=ax, lw=2)
        med = kmf.median_survival_time_
        out[str(g)] = {"n": int(len(d)), "events": int(d["event"].sum()),
                       "median_os_days": round(float(med), 0) if not np.isnan(med) else None}
    if len(groups) >= 2:
        lr = logrank_test(df[df[group_col] == groups[0]]["os_days"],
                          df[df[group_col] == groups[-1]]["os_days"],
                          event_observed_A=df[df[group_col] == groups[0]]["event"],
                          event_observed_B=df[df[group_col] == groups[-1]]["event"])
        out["logrank_p"] = round(float(lr.p_value), 4)
    ax.set_xlabel("days"); ax.set_ylabel("OS probability")
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    plt.close(fig)
    return out, fname


def cox_surrogate(df):
    d = df.dropna(subset=["os_days", "event", "age", "stage_num", "score"]).copy()
    d = d[d["os_days"] > 0]
    d["os_days"] = d["os_days"].astype(float)
    d["event"] = d["event"].astype(int)
    cph = CoxPHFitter()
    cph.fit(d[["os_days", "event", "score", "age", "stage_num"]].copy(),
            duration_col="os_days", event_col="event", formula="score + age + stage_num")
    summ = cph.summary
    return {"n": int(len(d)), "events": int(d["event"].sum()),
            "formula": "score + age + stage_num",
            "c_index": round(float(cph.concordance_index_), 4),
            "score": {"hr": round(float(summ.loc["score", "exp(coef)"]), 4),
                      "p": round(float(summ.loc["score", "p"]), 4)},
            "age": {"hr": round(float(summ.loc["age", "exp(coef)"]), 4),
                    "p": round(float(summ.loc["age", "p"]), 4)},
            "stage_num": {"hr": round(float(summ.loc["stage_num", "exp(coef)"]), 4),
                          "p": round(float(summ.loc["stage_num", "p"]), 4)}}


def main():
    exp_tcga = load_tcga()
    print("TCGA TPM genes:", exp_tcga.shape[1], "samples:", exp_tcga.shape[0], flush=True)

    res = {
        "method": ("External microarray validation (Plan B): GSE68465 (GPL96, LUAD, n=462), "
                   "GSE50081 (GPL570, early NSCLC, n=181), GSE72094 (GPL15048/HuRSTA merck, LUAD, n=442). "
                   "IGKC is not directly measured on these platforms; a surrogate is used = mean of "
                   "per-gene z-scores (log1p) over a fixed plasma-B gene set present on ALL three "
                   "platforms: " + ", ".join(SURROGATE) + ". Surrogate validated against IGKC on TCGA "
                   "(full TPM) and then applied per cohort (median split for KM, continuous for Cox, "
                   "adjusted for age and stage)."),
        "surrogate_genes": SURROGATE,
    }

    # TCGA surrogate validation
    z_tcga, r, present = tcga_surrogate_scores(exp_tcga, SURROGATE)
    res["tcga_surrogate"] = {
        "genes_used": present,
        "corr_with_igkc": round(float(r), 4) if r is not None else None,
        "genes_missing": [g for g in SURROGATE if g not in present],
    }
    print("TCGA surrogate corr with IGKC:", res["tcga_surrogate"]["corr_with_igkc"], flush=True)

    # Cox on TCGA: surrogate vs real IGKC
    tcga_clin = pd.read_csv(MERGED)
    tcga_clin = tcga_clin[tcga_clin["tissue"] == "Tumor"].drop_duplicates(subset="case_id", keep="first")
    tcga_surv = tcga_clin[["sample", "os_days", "event", "age_at_initial_pathologic_diagnosis", "stage_grp"]].copy()
    tcga_surv = tcga_surv.dropna(subset=["os_days", "event"])
    tcga_surv["age"] = pd.to_numeric(tcga_surv["age_at_initial_pathologic_diagnosis"], errors="coerce")
    tcga_surv["stage_num"] = np.where(tcga_surv["stage_grp"] == "III/IV", 2, 1)
    tcga_surv = tcga_surv[tcga_surv["os_days"] > 0]
    tcga_surv["score_sur"] = z_tcga.reindex(tcga_surv["sample"]).values
    tcga_surv["igkc"] = np.log1p(exp_tcga["IGKC"]).reindex(tcga_surv["sample"]).values

    c_sur = cox_surrogate(tcga_surv.rename(columns={"score_sur": "score"}))
    d = tcga_surv.dropna(subset=["os_days", "event", "age", "stage_num", "igkc"]).copy()
    cph = CoxPHFitter()
    cph.fit(d[["os_days", "event", "igkc", "age", "stage_num"]].copy(),
            duration_col="os_days", event_col="event", formula="igkc + age + stage_num")
    summ = cph.summary
    c_ig = {"n": int(len(d)), "events": int(d["event"].sum()),
            "igkc": {"hr": round(float(summ.loc["igkc", "exp(coef)"]), 4),
                     "p": round(float(summ.loc["igkc", "p"]), 4)}}
    res["tcga_cox_surrogate"] = c_sur
    res["tcga_cox_igkc_true"] = c_ig
    print("TCGA Cox surrogate:", c_sur["score"], "| IGKC true:", c_ig["igkc"], flush=True)

    # Per-cohort
    cohort_tables = []
    for name, cfg in COHORTS.items():
        expr, clin = read_series_matrix(cfg["matrix"])
        print("[%s] raw probes x samples: %s" % (name, expr.shape), flush=True)
        e = build_expression(expr, cfg["probe_map"])
        cl = make_clinical(expr.columns, clin, cfg)
        # align
        common = cl.index.intersection(e.columns)
        e = e[common]
        cl = cl.loc[common]
        genes = [g for g in SURROGATE if g in e.index]
        print("[%s] surrogate genes available: %d/%d -> %s" % (name, len(genes), len(SURROGATE), genes), flush=True)
        sub = e.loc[genes].T
        z = zscore_rows(np.log1p(sub)).mean(axis=1)
        cl["score"] = z.reindex(cl.index).values
        cl = cl.dropna(subset=["os_days", "event", "score"])
        cl["event"] = cl["event"].astype(int)
        cl["high"] = (cl["score"] > cl["score"].median()).astype(int)
        cl["cohort"] = name

        km, fname = survival_block(cl, "high",
                                   "%s: OS by IGKC surrogate (median split)" % name,
                                   "external_validation_microarray_km_%s.png" % name)
        cox = cox_surrogate(cl)
        blk = {
            "n": int(len(cl)), "events": int(cl["event"].sum()),
            "surrogate_genes_used": genes,
            "surrogate_genes_missing": [g for g in SURROGATE if g not in genes],
            "survival_by_high": km,
            "km_figure": fname,
            "cox": cox,
        }
        res[name] = blk
        cohort_tables.append(cl.copy())
        print("[%s] n=%d events=%d Cox score HR %.3f p=%.3f"
              % (name, int(len(cl)), int(cl["event"].sum()),
                 cox["score"]["hr"], cox["score"]["p"]), flush=True)

    # Pooled stratified Cox across 3 microarrays
    pooled = pd.concat(cohort_tables, ignore_index=True)
    pooled["score_z"] = pooled.groupby("cohort")["score"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))
    pooled["stage_z"] = pooled.groupby("cohort")["stage_num"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))
    pooled = pooled.dropna(subset=["os_days", "event", "score_z", "age", "stage_z"])
    scph = CoxPHFitter()
    scph.fit(pooled[["os_days", "event", "score_z", "age", "stage_z"]].copy().assign(cohort=pooled["cohort"].values),
             duration_col="os_days", event_col="event", strata=["cohort"], formula="score_z + age + stage_z")
    summ = scph.summary
    res["pooled_microarray"] = {
        "n": int(len(pooled)), "events": int(pooled["event"].sum()),
        "n_by_cohort": {c: int((pooled["cohort"] == c).sum()) for c in pooled["cohort"].unique()},
        "score_z": {"hr": round(float(summ.loc["score_z", "exp(coef)"]), 4),
                    "p": round(float(summ.loc["score_z", "p"]), 4)},
        "age": {"hr": round(float(summ.loc["age", "exp(coef)"]), 4),
                "p": round(float(summ.loc["age", "p"]), 4)},
        "stage_z": {"hr": round(float(summ.loc["stage_z", "exp(coef)"]), 4),
                    "p": round(float(summ.loc["stage_z", "p"]), 4)},
    }
    print("pooled microarray:", res["pooled_microarray"], flush=True)

    # KM pooled
    pooled["high"] = (pooled["score_z"] > pooled["score_z"].median()).astype(int)
    km_p, fname_p = survival_block(pooled, "high",
                                   "Pooled microarrays (GSE68465+GSE50081+GSE72094): OS by IGKC surrogate",
                                   "external_validation_microarray_km_pooled.png")
    res["pooled_microarray"]["survival_by_high"] = km_p
    res["pooled_microarray"]["km_figure"] = fname_p

    # All 4 cohorts: TCGA-surrogate + 3 microarrays
    tcga_surv2 = tcga_surv[["os_days", "event", "score_sur", "age", "stage_num"]].copy()
    tcga_surv2 = tcga_surv2.rename(columns={"score_sur": "score"})
    tcga_surv2["cohort"] = "TCGA"
    all4 = pd.concat([tcga_surv2, pooled[["os_days", "event", "score", "age", "stage_num", "cohort"]].copy()],
                     ignore_index=True)
    all4 = all4.dropna(subset=["os_days", "event", "score", "age", "stage_num"])
    all4["score_z"] = all4.groupby("cohort")["score"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))
    all4["stage_z"] = all4.groupby("cohort")["stage_num"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))
    all4 = all4.dropna(subset=["os_days", "event", "score_z", "age", "stage_z"])
    scph4 = CoxPHFitter()
    scph4.fit(all4[["os_days", "event", "score_z", "age", "stage_z"]].copy().assign(cohort=all4["cohort"].values),
              duration_col="os_days", event_col="event", strata=["cohort"], formula="score_z + age + stage_z")
    summ4 = scph4.summary
    res["pooled_all4_surrogate"] = {
        "n": int(len(all4)), "events": int(all4["event"].sum()),
        "n_by_cohort": {c: int((all4["cohort"] == c).sum()) for c in all4["cohort"].unique()},
        "score_z": {"hr": round(float(summ4.loc["score_z", "exp(coef)"]), 4),
                    "p": round(float(summ4.loc["score_z", "p"]), 4)},
    }
    print("pooled all4:", res["pooled_all4_surrogate"], flush=True)

    with open(os.path.join(RESULTS_DIR, "external_validation_microarray.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print("saved results/external_validation_microarray.json", flush=True)


if __name__ == "__main__":
    main()