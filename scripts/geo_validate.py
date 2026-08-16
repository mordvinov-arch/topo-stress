# GEO-валидация (GSE31210): воспроизводимость физиотипов и выживаемости.
#
# 1) Физиотипы: точно как в GDC-конвейере (run_gdc_realgroups.py): первые 200
#    HVG-генов, сдвиг на минимум, нормировка на распределение, Вассерштейн+Уорд, k=3.
# 2) Модули GDC (топ-50 каждого физиотипа) -> z-обогащение кластеров GSE31210,
#    сопоставление кластеров с физиотипами GDC.
# 3) Выживаемость: KM+log-rank и Cox по физиотипам; IGKC-сигнатура (медианный сплит).
# 4) Стабильность кластеров (bootstrap ARI).
# 5) RMT: главный собственный вектор корреляционной матрицы опухолей ->
#    воспроизводится ли IGKC-модуль (λ_max) на независимой когорте.
# 6) TDA: d_topo(IGKC-high vs IGKC-low) и d_topo(опухоль vs норма), пермутационный тест.
#
# Выходы: results/geo_validate.json, figures/geo_validate_*.png.

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress import rmt, topology, utils  # noqa: E402
from topostress.info_geometry import wasserstein_matrix, ward_clusters  # noqa: E402
from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

EXPR = os.path.join(PROCESSED_DATA_DIR, "gse31210_expression.csv")
CLIN = os.path.join(PROCESSED_DATA_DIR, "gse31210_clinical.csv")
WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
TOP50 = os.path.join(RESULTS_DIR, "gdc2_physiotype_top50_genes.csv")
LMAX = os.path.join(RESULTS_DIR, "gdc2_lmax.json")

N_CLUSTERS = 3
N_PERM = int(os.environ.get("MAST_PERM", 199))
N_EPS = 100
SEED = 42
N_BOOT = 50


def zscore(a):
    return (a - a.mean()) / (a.std(ddof=0) + 1e-12)


def physiotype_labels(X):
    sub = min(200, X.shape[1])
    d = X[:, :sub] - X[:, :sub].min(axis=1, keepdims=True)
    d = d / (d.sum(axis=1, keepdims=True) + 1e-12)
    D = wasserstein_matrix([d[i] for i in range(len(d))])
    labels, _ = ward_clusters(D, N_CLUSTERS)
    return labels, D


def best_permutation_ari(a, b, k):
    from itertools import permutations
    from sklearn.metrics import adjusted_rand_score
    best, best_ari = None, -1
    for perm in permutations(range(k)):
        mapped = np.array([perm[x - 1] + 1 for x in a])
        v = adjusted_rand_score(mapped, b)
        if v > best_ari:
            best_ari, best = v, perm
    return best_ari


def main():
    expr = pd.read_csv(EXPR, index_col=0)
    clin = pd.read_csv(CLIN)
    top50 = pd.read_csv(TOP50)
    lmax = json.load(open(LMAX, encoding="utf-8"))

    Xraw = expr.values.T  # 246 x 313
    gsm = list(expr.columns)
    X = np.log2(Xraw + 1.0)
    genes = list(expr.index)

    clin = clin.set_index("gsm").loc[gsm].reset_index()
    tissue = clin["tissue"].values
    tumor_mask = tissue == "Tumor"
    print("samples:", X.shape, "tumors:", int(tumor_mask.sum()),
          "normals:", int((~tumor_mask).sum()), flush=True)

    # GDC-модули (топ-50 физиотипов GDC) в терминах зондов GSE31210
    module_genes = {}
    for ph in sorted(top50["physiotype"].unique()):
        gs = top50[top50["physiotype"] == ph]["gene"].tolist()
        module_genes[ph] = [g for g in gs if g in genes]
    ig_core = [g for g, _ in lmax["Tumor"]["top20_genes"] if g in genes]  # IGKC, IGHG1
    ig_signature = list(dict.fromkeys(ig_core + module_genes[1]))
    print("IG signature genes present on GPL570:", ig_signature, flush=True)

    def module_score(mat, genes_sub):
        gidx = [genes.index(g) for g in genes_sub]
        return mat[:, gidx].mean(axis=1)

    res = {"method": "GEO validation GSE31210 (GPL570)",
           "samples": int(len(gsm)), "tumors": int(tumor_mask.sum()),
           "normals": int((~tumor_mask).sum()),
           "genes": int(len(genes)),
           "ig_signature_present": ig_signature,
           "module_genes_present": {str(k): v for k, v in module_genes.items()}}

    # ---- 1) физиотипы на опухолях ----
    Xt = X[tumor_mask]
    labels, Dt = physiotype_labels(Xt)
    res["physiotypes"] = {
        "n_clusters": N_CLUSTERS, "n_genes_sub": 200,
        "cluster_sizes": pd.Series(labels).value_counts().sort_index().to_dict()}

    # ---- 2) сопоставление с физиотипами GDC через z-обогащение модулей ----
    Zm = np.column_stack([zscore(module_score(Xt, module_genes[ph])) for ph in [1, 2, 3]])
    enrichment = pd.DataFrame(Zm, columns=["PH1", "PH2", "PH3"])
    enrichment["cl"] = labels
    mean_enr = enrichment.groupby("cl").mean()
    mapping = {}
    used = set()
    for cl in sorted(set(labels)):
        best = mean_enr.loc[cl].sort_values(ascending=False)
        for ph in best.index:
            if ph not in used:
                mapping[int(cl)] = int(ph[2:])
                used.add(ph)
                break
    print("cluster->GDC physiotype mapping:", mapping, flush=True)
    res["cluster_mapping_to_gdc_physiotype"] = {str(k): v for k, v in mapping.items()}
    res["module_enrichment_z_by_cluster"] = mean_enr.round(3).to_dict("index")

    mapped = np.array([mapping[l] for l in labels])
    res["gdc_physiotype_sizes_in_geo"] = pd.Series(mapped).value_counts().sort_index().to_dict()

    # ---- 3) выживаемость ----
    surv = clin[tumor_mask].copy()
    surv["physio"] = mapped
    ig_all = zscore(module_score(Xt, ig_signature))
    surv["ig_score"] = ig_all
    surv["ig_high"] = (ig_all > np.median(ig_all)).astype(int)
    # стадия в упорядоченные числа
    stage_num = {"IA": 1, "IB": 2, "II": 3, "III": 4}.get
    surv["stage_num"] = surv["stage"].map(stage_num)

    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test

    def km_block(df, group_col, title, time_col="os_days", event_col="event"):
        kmf = KaplanMeierFitter()
        out = {}
        fig, ax = plt.subplots(figsize=(7, 5))
        for g in sorted(df[group_col].unique()):
            m = df[group_col] == g
            d = df[m]
            kmf.fit(d[time_col], event_observed=d[event_col], label=str(g))
            kmf.plot_survival_function(ax=ax, ci_show=True, lw=2)
            med = kmf.median_survival_time_
            out[str(g)] = {"n": int(m.sum()), "events": int(d[event_col].sum()),
                           "median_os_days": float(med) if not np.isnan(med) else None}
        if len(df[group_col].unique()) == 2:
            g0, g1 = sorted(df[group_col].unique())
            lr = logrank_test(df.loc[df[group_col] == g0, time_col],
                              df.loc[df[group_col] == g1, time_col],
                              event_observed_A=df.loc[df[group_col] == g0, event_col],
                              event_observed_B=df.loc[df[group_col] == g1, event_col])
            out["logrank_p"] = float(lr.p_value)
        else:
            groups = sorted(df[group_col].unique())
            out["logrank_p"] = float(logrank_test(
                df.loc[df[group_col] == groups[0], time_col],
                df.loc[df[group_col] == groups[1], time_col],
                event_observed_A=df.loc[df[group_col] == groups[0], event_col],
                event_observed_B=df.loc[df[group_col] == groups[1], event_col]).p_value)
        ax.set_xlabel("days"); ax.set_ylabel("survival probability")
        ax.set_title(title)
        ax.legend(title=group_col)
        fig.tight_layout()
        fname = "geo_validate_" + group_col + ".png"
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
        plt.close(fig)
        return out, fname

    def cox_block(df, extra_formula=""):
        sub = df.dropna(subset=["os_days", "stage_num", "age_years"])
        sub = sub[sub["os_days"] > 0]
        cols = ["os_days", "event", "stage_num", "age_years"] + \
            ([extra_formula] if extra_formula else [])
        cdf = sub[["os_days", "event", "stage_num", "age_years"]].copy()
        cdf["physio"] = sub["physio"].astype("category")
        cdf["physio_num"] = sub["physio"].astype(int)
        cph = CoxPHFitter()
        cph.fit(cdf, duration_col="os_days", event_col="event", formula="physio + stage_num + age_years")
        summ = cph.summary
        out = {"n": int(len(cdf)), "formula": "physio + stage_num + age_years"}
        for name in ["physio", "stage_num", "age_years"]:
            if name in summ.index:
                out[name] = {"coef": float(summ.loc[name, "coef"]),
                             "hr": float(summ.loc[name, "exp(coef)"]),
                             "p": float(summ.loc[name, "p"])}
        return out

    # пациенты, исключённые авторами (неполная резекция/адъювантная терапия)
    for subset_name, mask in [("all", np.ones(len(surv), bool)),
                              ("authors_excluded", surv["exclude"].astype(bool).values),
                              ("prognosis", (~surv["exclude"].astype(bool)).values)]:
        df = surv[mask]
        if len(df) < 10:
            continue
        km, fname = km_block(df, "physio", "GSE31210: OS by physiotype (%s)" % subset_name)
        res["survival_by_physiotype_" + subset_name] = km
        res["survival_figure_" + subset_name] = fname
        print("OS by physiotype [%s]:" % subset_name, json.dumps(km, default=str), flush=True)

        # RFS (рецидив) — основной эндпоинт этой когорты
        dfr = df.dropna(subset=["rfs_days"])
        dfr = dfr[dfr["rfs_days"] > 0]
        if len(dfr) >= 10:
            kmr, fname_r = km_block(dfr, "physio", "GSE31210: RFS by physiotype (%s)" % subset_name,
                                    time_col="rfs_days", event_col="relapse_event")
            res["rfs_by_physiotype_" + subset_name] = kmr
            res["rfs_figure_" + subset_name] = fname_r
            print("RFS by physiotype [%s]:" % subset_name, json.dumps(kmr, default=str), flush=True)

    # IGKC-сигнатура: медианный сплит
    ig_sub = surv[(surv["exclude"] != 1) & (surv["os_days"] > 0)]
    km_ig, fname_ig = km_block(ig_sub, "ig_high", "GSE31210: OS by IGKC signature (median split)")
    res["survival_by_ig_signature"] = km_ig
    res["survival_ig_figure"] = fname_ig
    print("OS by IGKC signature:", json.dumps(km_ig, default=str), flush=True)
    ig_r = ig_sub.dropna(subset=["rfs_days"])
    ig_r = ig_r[ig_r["rfs_days"] > 0]
    if len(ig_r) >= 10:
        km_ig_r, fname_ig_r = km_block(ig_r, "ig_high",
                                       "GSE31210: RFS by IGKC signature (median split)",
                                       time_col="rfs_days", event_col="relapse_event")
        res["rfs_by_ig_signature"] = km_ig_r
        res["rfs_ig_figure"] = fname_ig_r
        print("RFS by IGKC signature:", json.dumps(km_ig_r, default=str), flush=True)
    # корреляция IG-сигнатуры с временем выживаемости/событием
    r, p = utils.pearson_test(ig_sub["ig_score"], np.log1p(ig_sub["os_days"]))
    res["ig_score_os_days_pearson"] = {"r": round(float(r), 3), "p": float(p)}
    r_ev, p_ev = utils.pearson_test(ig_sub["ig_score"], ig_sub["event"])
    res["ig_score_event_pearson"] = {"r": round(float(r_ev), 3), "p": float(p_ev)}

    # IG-сигнатура в норме vs опухоль
    ig_norm = zscore(module_score(X[~tumor_mask], ig_signature))
    r_ig, p_ig = utils.pearson_test(np.concatenate([ig_all, ig_norm]),
                                    np.concatenate([np.zeros(len(ig_all)), np.ones(len(ig_norm))]))
    res["ig_signature_tumor_vs_normal_ttest_p"] = float(p_ig)
    print("IG high n=%d events=%d | IG low n=%d events=%d"
          % (int(ig_sub["ig_high"].sum()), int(ig_sub[ig_sub["ig_high"] == 1]["event"].sum()),
             int((ig_sub["ig_high"] == 0).sum()), int(ig_sub[ig_sub["ig_high"] == 0]["event"].sum())), flush=True)

    # ---- 4) стабильность кластеров (bootstrap) ----
    rng = np.random.default_rng(SEED)
    ari_full = np.zeros(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.choice(len(Xt), size=int(0.8 * len(Xt)), replace=False)
        bl, _ = physiotype_labels(Xt[idx])
        bl = np.array([mapping[l] for l in bl])
        ari_full[i] = best_permutation_ari(mapped[idx], bl, N_CLUSTERS)
    res["cluster_stability_bootstrap"] = {
        "n_boot": N_BOOT, "frac": 0.8, "ari_mean": round(float(ari_full.mean()), 3),
        "ari_sd": round(float(ari_full.std(ddof=1)), 3),
        "ari_min": round(float(ari_full.min()), 3)}
    print("bootstrap ARI: %.3f +/- %.3f" % (ari_full.mean(), ari_full.std(ddof=1)), flush=True)

    # ---- 5) RMT: IGKC-модуль в главном собственном векторе? ----
    spec = rmt.correlation_spectrum(Xt)
    lam_plus = rmt.marchenko_pastur_bound(Xt.shape[1], Xt.shape[0])
    Zt = zscore(Xt)
    _, _, vt = np.linalg.svd(Zt, full_matrices=False)
    top_vec = vt[0]
    top_idx = np.argsort(np.abs(top_vec))[::-1][:20]
    top_genes = [(genes[i], round(float(top_vec[i]), 4)) for i in top_idx]
    ig_positions = [i for i, g in enumerate(genes) if g in ig_signature]
    ig_loading = float(np.mean(np.abs(top_vec[ig_positions])))
    all_loading = float(np.mean(np.abs(top_vec)))
    res["rmt"] = {"lambda_max": float(spec[0]), "lambda_plus": float(lam_plus),
                  "fraction_above_bound": float((spec > lam_plus).mean()),
                  "top20_abs_loadings": top_genes,
                  "mean_abs_loading_ig_module": ig_loading,
                  "mean_abs_loading_all_genes": all_loading,
                  "ig_module_loading_ratio": round(ig_loading / all_loading, 2)}
    print("RMT: lambda_max=%.2f (bound %.2f), IG ratio=%.2f"
          % (spec[0], lam_plus, ig_loading / all_loading), flush=True)
    print("RMT top genes:", top_genes[:10], flush=True)

    # ---- 6) TDA ----
    ig_hi = ig_all > np.median(ig_all)
    d_obs, p_d, _ = utils.permutation_test(
        lambda A, B: topology.d_topo_normalized(A, B, n_eps=N_EPS)[0],
        Xt[ig_hi], Xt[~ig_hi], n_perm=N_PERM, seed=SEED)
    res["tda_ig_high_vs_low"] = {"d_topo": round(float(d_obs), 4),
                                 "p_permutation": float(p_d), "n_perm": N_PERM,
                                 "n_high": int(ig_hi.sum()), "n_low": int((~ig_hi).sum())}
    print("TDA IG-high vs IG-low: d_topo=%.4f p=%.3f" % (d_obs, p_d), flush=True)

    d_tn, p_tn, _ = utils.permutation_test(
        lambda A, B: topology.d_topo_normalized(A, B, n_eps=N_EPS)[0],
        Xt, X[~tumor_mask], n_perm=N_PERM, seed=SEED)
    res["tda_tumor_vs_normal"] = {"d_topo": round(float(d_tn), 4),
                                  "p_permutation": float(p_tn), "n_perm": N_PERM}
    print("TDA tumor vs normal: d_topo=%.4f p=%.3f" % (d_tn, p_tn), flush=True)

    with open(os.path.join(RESULTS_DIR, "geo_validate.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print("saved results/geo_validate.json", flush=True)


if __name__ == "__main__":
    main()
