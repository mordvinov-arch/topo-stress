# GDC2: компактная панель генов (20-50) для предсказания физиотипа.
#
# Идея: вместо 500 HVG предлагаем практическую панель из 20-50 генов, которую можно
# измерить дешёвой панелью (Nanostring / qPCR) и которая воспроизводит физиотипы.
#
# Протокол (без утечки):
# 1) Вложенная CV: в каждом фолде гены отбираются по ANOVA F ТОЛЬКО на обучающей части,
#    RF обучается на них, точность считается на тестовой части. Перебор K = 20/30/40/50.
# 2) Сравнение лучшей панели с полным набором 500 HVG (тот же протокол).
# 3) Кросс-платформенная панель: то же самое, но пространство генов ограничено
#    измеряемыми на обеих платформах (TCGA RNA-seq и GSE31210 микромассив, 313 генов).
#    Так отбор и валидация используют одно и то же пространство — честно.
# 4) Внешняя валидация на GSE31210: z-скор каждого гена внутри датасета, предсказание
#    физиотипов и проверка, что они стратифицируют OS.
#
# Выход: results/gdc2_gene_panel.json, figures/gdc2_gene_panel_cv.png,
# figures/gdc2_gene_panel_gse_km.png.

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
GSE_EXPR = os.path.join(PROCESSED_DATA_DIR, "gse31210_expression.csv")
GSE_CLIN = os.path.join(PROCESSED_DATA_DIR, "gse31210_clinical.csv")

SEED = 42
N_REPEATS = 5
N_FOLDS = 5
K_SIZES = [20, 30, 40, 50]


def anova_rank(X, y):
    genes = []
    for i in range(X.shape[1]):
        groups = [X[y == c, i] for c in np.unique(y)]
        if min(len(g) for g in groups) < 2:
            genes.append(0.0)
        else:
            genes.append(f_oneway(*groups).statistic)
    return np.argsort(genes)[::-1]


def fit_eval_panel(X, y, k, rng):
    accs, aris = [], []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rng)
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        top = anova_rank(Xtr, ytr)[:k]
        m = RandomForestClassifier(n_estimators=300, random_state=0)
        m.fit(Xtr[:, top], ytr)
        pred = m.predict(Xte[:, top])
        accs.append(accuracy_score(yte, pred))
        aris.append(adjusted_rand_score(yte, pred))
    return np.mean(accs), np.mean(aris)


def zscore_cols(df):
    return df.apply(lambda c: (c - c.mean()) / (c.std(ddof=0) + 1e-12), axis=0)


def cv_panels(X, y, names):
    out = {}
    for name, k in names.items():
        accs, aris = [], []
        for rep in range(N_REPEATS):
            a, ar = fit_eval_panel(X, y, k, SEED + rep)
            accs.append(a)
            aris.append(ar)
        out[name] = {"accuracy_mean": round(float(np.mean(accs)), 4),
                     "accuracy_sd": round(float(np.std(accs, ddof=1)), 4),
                     "ari_mean": round(float(np.mean(aris)), 4)}
        print("panel %s: acc=%.3f±%.3f ari=%.3f" % (name, np.mean(accs), np.std(accs, ddof=1), np.mean(aris)), flush=True)
    return out


def main():
    wide = pd.read_csv(WIDE, index_col=0)
    hvg = list(wide.columns)
    clin = pd.read_csv(MERGED)
    clin = clin[clin["tissue"] == "Tumor"].drop_duplicates(subset="case_id", keep="first")
    clin = clin.dropna(subset=["os_days"])
    df = clin.merge(wide, left_on="sample", right_index=True, how="inner")
    df = df.dropna(subset=["physiotype"])
    y = df["physiotype"].astype(int).values
    X = df[hvg].values.astype(float)
    print("TCGA n=%d, features=%d, classes=%s" % (len(df), X.shape[1], np.bincount(y).tolist()), flush=True)

    res = {"method": "Compact gene panel (20-50 genes) predicting physiotype",
           "n_train_tcga": int(len(df))}

    # ---- (1) панель из полных 500 HVG ----
    res["cv_500hvg"] = cv_panels(X, y, {str(k): k for k in K_SIZES})
    res["cv_full_500hvg"] = cv_panels(X, y, {"full_500": X.shape[1]})["full_500"]
    best_k = max(K_SIZES, key=lambda k: res["cv_500hvg"][str(k)]["accuracy_mean"])
    top = anova_rank(X, y)[:best_k]
    panel_genes = [hvg[i] for i in top]
    m = RandomForestClassifier(n_estimators=500, random_state=0)
    m.fit(X[:, top], y)
    res["best_k_500hvg"] = int(best_k)
    res["panel_genes_500hvg"] = panel_genes
    res["train_accuracy_500hvg_full_data"] = round(float(accuracy_score(y, m.predict(X[:, top]))), 4)
    imp = sorted(zip(panel_genes, m.feature_importances_), key=lambda t: -t[1])
    res["panel_gene_importances_500hvg"] = {g: round(float(v), 4) for g, v in imp}

    # ---- (2) кросс-платформенная панель (гены, измеряемые и в TCGA, и в GSE31210) ----
    gex = pd.read_csv(GSE_EXPR, index_col=0)
    gex_genes = [g for g in hvg if g in gex.index]
    idx313 = [hvg.index(g) for g in gex_genes]
    print("cross-platform genes available (TCGA x GSE31210): %d" % len(gex_genes), flush=True)
    res["crossplatform_genes_n"] = int(len(gex_genes))
    res["crossplatform_gene_list"] = gex_genes

    res["cv_crossplatform"] = cv_panels(X[:, idx313], y, {str(k): k for k in K_SIZES})
    best_k2 = max(K_SIZES, key=lambda k: res["cv_crossplatform"][str(k)]["accuracy_mean"])
    top2 = anova_rank(X[:, idx313], y)[:best_k2]
    panel_cp = [gex_genes[i] for i in top2]
    m_cp = RandomForestClassifier(n_estimators=500, random_state=0)
    m_cp.fit(X[:, [hvg.index(g) for g in panel_cp]], y)
    res["best_k_crossplatform"] = int(best_k2)
    res["panel_genes_crossplatform"] = panel_cp
    res["train_accuracy_crossplatform_full_data"] = round(
        float(accuracy_score(y, m_cp.predict(X[:, [hvg.index(g) for g in panel_cp]]))), 4)

    # ---- (3) внешняя валидация GSE31210 (кросс-платформенной панелью) ----
    # Прямой перенос обученного RF на z-скор GSE31210 вырождается: все образцы
    # классифицируются как PH3 из-за batch/платформенного
    # сдвига (TCGA RNA-seq vs Affymetrix). Поэтому используем более устойчивый метод:
    # назначение по максимальной корреляции Пирсона профиля образца с центроидом
    # физиотипа (паттерн-подход, инвариантный к масштабу).
    gcl = pd.read_csv(GSE_CLIN)
    gcl = gcl[gcl["exclude"] == 0]
    tumor_gsm = gcl[gcl["tissue"] == "Tumor"]["gsm"].tolist()
    gex_t = gex[tumor_gsm]

    Xp = X[:, [hvg.index(g) for g in panel_cp]]
    Xz = np.array([(Xp[:, i] - Xp[:, i].mean()) / (Xp[:, i].std(ddof=0) + 1e-12)
                   for i in range(Xp.shape[1])]).T
    centroids = np.array([Xz[y == c].mean(axis=0) for c in [1, 2, 3]])
    gz = zscore_cols(gex_t.loc[panel_cp].T).values
    gz_n = (gz - gz.mean(axis=1, keepdims=True)) / (gz.std(axis=1, keepdims=True) + 1e-12)
    cen_n = (centroids - centroids.mean(axis=1, keepdims=True)) / (centroids.std(axis=1, keepdims=True) + 1e-12)
    corr = gz_n @ cen_n.T / len(panel_cp)
    gse_pred = corr.argmax(axis=1) + 1

    # для прозрачности: что дал бы прямой перенос RF (вырождается -> все в PH3)
    gz_rf = zscore_cols(gex_t.loc[panel_cp].T).values
    rf_pred = m_cp.predict(gz_rf)
    rf_direct_counts = {int(c): int((rf_pred == c).sum()) for c in [1, 2, 3]}
    res["gse31210_direct_rf_assignments"] = {
        "note": "direct application of the TCGA-trained RF to z-scored GSE31210 collapses "
                "to a single class (platform/batch shift)", "counts": rf_direct_counts}

    gcl = gcl[gcl["tissue"] == "Tumor"].copy()
    gcl = gcl.merge(pd.DataFrame({"gsm": gex_t.loc[panel_cp].T.index, "physiotype_pred": gse_pred}), on="gsm", how="inner")
    gcl["os_days"] = gcl["os_days"].astype(float)
    gcl["event"] = gcl["event"].astype(int)
    gcl = gcl.dropna(subset=["os_days", "event"]).reset_index(drop=True)
    print("GSE31210 after filter+merge: n=%d events=%d" % (len(gcl), int(gcl["event"].sum())), flush=True)

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    med_os = {}
    pred_counts = {}
    for c in sorted(np.unique(gse_pred)):
        sub = gcl[gcl["physiotype_pred"] == c]
        pred_counts[int(c)] = int(len(sub))
        if len(sub) < 5:
            continue
        kmf.fit(sub["os_days"], sub["event"], label="PH%d" % c)
        kmf.plot_survival_function(ax=ax, lw=2)
        med = kmf.median_survival_time_
        med_os[int(c)] = round(float(med), 0) if not np.isnan(med) else None
    mt = multivariate_logrank_test(gcl["os_days"], gcl["physiotype_pred"], gcl["event"])
    lr13 = None
    sub1 = gcl[gcl["physiotype_pred"] == 1]
    sub3 = gcl[gcl["physiotype_pred"] == 3]
    if len(sub1) >= 5 and len(sub3) >= 5:
        lr = logrank_test(sub1["os_days"], sub3["os_days"],
                          event_observed_A=sub1["event"], event_observed_B=sub3["event"])
        lr13 = round(float(lr.p_value), 4)
    res["gse31210_validation"] = {
        "assignment_method": "max Pearson correlation of sample profile with physiotype centroid (robust to batch shift)",
        "n": int(len(gcl)), "n_events": int(gcl["event"].sum()),
        "predicted_physiotype_counts": pred_counts,
        "median_os_days": med_os,
        "multivariate_logrank_p": (round(float(mt.p_value), 4) if not np.isnan(mt.p_value) else None),
        "logrank_ph1_vs_ph3_p": lr13,
        "mean_corr_to_centroids": {str(c): round(float(corr[:, c - 1].mean()), 3) for c in [1, 2, 3]}}
    print("GSE31210 multivariate log-rank p=%s; PH1 vs PH3 p=%s" % (mt.p_value, lr13), flush=True)
    ax.set_title("GSE31210 OS by panel-predicted physiotype (log-rank p=%.3g)" % mt.p_value)
    ax.set_xlabel("days"); ax.set_ylabel("OS probability")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_gene_panel_gse_km.png"), dpi=150)
    plt.close(fig)

    # ---- рисунок: CV-кривые ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = K_SIZES
    ax.plot(ks, [res["cv_500hvg"][str(k)]["accuracy_mean"] for k in ks], "o-", label="panel from 500 HVG")
    ax.plot(ks, [res["cv_crossplatform"][str(k)]["accuracy_mean"] for k in ks], "s--", label="cross-platform panel")
    ax.set_xlabel("panel size K"); ax.set_ylabel("CV accuracy")
    ax.set_title("Panel size vs accuracy (nested CV)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_gene_panel_cv.png"), dpi=150)
    plt.close(fig)

    res["interpretation"] = (
        "Internal: a %d-gene panel reproduced physiotypes at CV accuracy %.3f vs %.3f for the "
        "full 500-HVG set, and a panel selected only from the 313 genes measurable on both TCGA "
        "and GSE31210 reaches %.3f. External validation on GSE31210 (independent microarray "
        "cohort) is underpowered (n=%d, only %d OS events): direct transfer of the trained RF "
        "collapses to a single class (platform shift), while the robust correlation-to-centroid "
        "assignment yields a non-degenerate distribution (PH1/PH2/PH3 = %s) but no significant "
        "OS stratification (multivariate log-rank p=%s). Conclusion: the panel reproduces "
        "physiotypes within a platform, but cross-platform external validation needs a better "
        "validation cohort (more events) and a batch-correction step." % (
            best_k, res["cv_500hvg"][str(best_k)]["accuracy_mean"],
            res["cv_full_500hvg"]["accuracy_mean"],
            res["cv_crossplatform"][str(best_k2)]["accuracy_mean"],
            res["gse31210_validation"]["n"], res["gse31210_validation"]["n_events"],
            res["gse31210_validation"]["predicted_physiotype_counts"],
            res["gse31210_validation"]["multivariate_logrank_p"]))

    with open(os.path.join(RESULTS_DIR, "gdc2_gene_panel.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved results/gdc2_gene_panel.json + figures", flush=True)


if __name__ == "__main__":
    main()