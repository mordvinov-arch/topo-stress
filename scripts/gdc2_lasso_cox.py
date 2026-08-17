# GDC2: LASSO-Cox (Coxnet) с C-index — физиотип + 500 HVG + возраст + стадия.
#
# Дизайн: tumor-only пациенты (дедупликация по case_id). Признаки: PH2/PH3
# (дамми физиотипа), age_years, стадия III/IV, 500 HVG (лог1п-TPM, среднее по
# опухолевому образцу пациента). LASSO-Cox через scikit-survival
# (CoxnetSurvivalAnalysis); alpha выбирается внутренней 5-fold CV на train,
# финальная оценка — concordance index на test (70/30, стратифицированный сплит).
# Если C-index на test > 0.7 — строится номограмма.
#
# Выход: results/gdc2_lasso_cox.json, figures/gdc2_lasso_cindex.png,
# figures/gdc2_nomogram.png, figures/gdc2_lasso_risk_km.png.

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
SEED = 42


def main():
    wide = pd.read_csv(WIDE, index_col=0)
    hvg = list(wide.columns)
    clin = pd.read_csv(MERGED)

    df = clin[clin["tissue"] == "Tumor"].copy()
    df = df.drop_duplicates(subset="case_id", keep="first")
    df = df.dropna(subset=["os_days"]).copy()
    df["os_days"] = df["os_days"].astype(float)
    df["event"] = df["event"].astype(int)
    df["age_years"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce").abs()
    df["stage_III_IV"] = (df["stage_grp"] == "III/IV").astype(int)
    df["PH2"] = (df["physiotype"].astype(int) == 2).astype(int)
    df["PH3"] = (df["physiotype"].astype(int) == 3).astype(int)

    expr = wide.loc[df["sample"], hvg].reset_index(drop=True)
    X = pd.concat([df[["PH2", "PH3", "age_years", "stage_III_IV"]].reset_index(drop=True), expr],
                  axis=1)
    y = Surv.from_arrays(df["event"].values, df["os_days"].values)
    keep = X["age_years"].notna()
    X = X[keep].reset_index(drop=True)
    y = y[keep.values]
    print("patients:", int(keep.sum()), " features:", X.shape[1], flush=True)

    rng = np.random.default_rng(SEED)
    # стратифицированный 70/30 сплит по событию
    idx_all = np.arange(len(y))
    ev = np.array([e for e, _ in y])
    tr = []
    for e in [0, 1]:
        pool = idx_all[ev == e]
        n_tr = int(0.7 * len(pool))
        tr.append(rng.choice(pool, size=n_tr, replace=False))
    train_idx = np.concatenate(tr)
    test_idx = np.setdiff1d(idx_all, train_idx)
    print("train:", len(train_idx), " test:", len(test_idx), flush=True)

    alphas = np.logspace(np.log10(0.001), np.log10(0.2), 40)
    best_alpha, best_c = None, -1
    rng_cv = np.random.default_rng(SEED)
    # внутренняя 5-fold CV на train для выбора alpha
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    Xtr, ytr = X.iloc[train_idx], y[train_idx]
    for a in alphas:
        cs = []
        for tr2, va in kf.split(Xtr):
            model = CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=1.0, max_iter=2000)
            model.fit(Xtr.iloc[tr2], ytr[tr2])
            risk = model.predict(Xtr.iloc[va])
            ev_va = np.array([e for e, _ in ytr[va]])
            tm_va = np.array([t for _, t in ytr[va]])
            c, _, _, _, _ = concordance_index_censored(ev_va, tm_va, risk)
            cs.append(c)
        m = float(np.mean(cs))
        if m > best_c:
            best_c, best_alpha = m, a
    print("best alpha=%.4f  inner-CV C-index=%.3f" % (best_alpha, best_c), flush=True)

    model = CoxnetSurvivalAnalysis(alphas=[best_alpha], l1_ratio=1.0, max_iter=2000)
    model.fit(Xtr, ytr)
    coef = pd.Series(model.coef_[:, 0], index=X.columns)
    n_nonzero = int((coef.abs() > 1e-8).sum())
    print("non-zero coefficients:", n_nonzero, flush=True)

    def cindex(idx):
        risk = model.predict(X.iloc[idx])
        ev_ = np.array([e for e, _ in y[idx]])
        tm_ = np.array([t for _, t in y[idx]])
        c, _, _, _, _ = concordance_index_censored(ev_, tm_, risk)
        return c, risk

    c_train, risk_tr = cindex(train_idx)
    c_test, risk_te = cindex(test_idx)
    print("C-index train=%.3f test=%.3f" % (c_train, c_test), flush=True)

    top_genes = coef[coef.abs() > 1e-8].abs().sort_values(ascending=False)
    res = {
        "method": "LASSO-Cox (Coxnet): physiotype + 500 HVG + age + stage",
        "n_patients": int(len(y)), "n_features": int(X.shape[1]),
        "train_n": int(len(train_idx)), "test_n": int(len(test_idx)),
        "alpha_grid": [round(float(a), 5) for a in alphas],
        "best_alpha_cv5": float(best_alpha), "inner_cv_cindex": round(best_c, 4),
        "n_nonzero_coef": n_nonzero,
        "c_index_train": round(c_train, 4), "c_index_test": round(c_test, 4),
        "top_features": {str(k): round(float(v), 4) for k, v in top_genes.head(25).items()},
    }

    # ===== рисунки: C-index по alpha (CV), риск-группы KM на test, номограмма =====
    fig, ax = plt.subplots(figsize=(6, 4))
    cv = []
    for a in alphas:
        cs = []
        for tr2, va in kf.split(Xtr):
            m = CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=1.0, max_iter=2000)
            m.fit(Xtr.iloc[tr2], ytr[tr2])
            ev_va = np.array([e for e, _ in ytr[va]])
            tm_va = np.array([t for _, t in ytr[va]])
            c, *_ = concordance_index_censored(ev_va, tm_va, m.predict(Xtr.iloc[va]))
            cs.append(c)
        cv.append(float(np.mean(cs)))
    ax.plot(alphas, cv, "o-", color="steelblue")
    ax.axvline(best_alpha, color="crimson", ls="--", label="best alpha")
    ax.set_xscale("log"); ax.set_xlabel("alpha (L1)"); ax.set_ylabel("C-index (5-fold CV, train)")
    ax.set_title("LASSO-Cox alpha selection")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_lasso_cindex.png"), dpi=150)
    plt.close(fig)

    # KM по риск-группам на test
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    med = np.median(risk_te)
    hi = risk_te > med
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    t_te = np.array([t for _, t in y[test_idx]])
    e_te = np.array([e for e, _ in y[test_idx]])
    kmf.fit(t_te[~hi], e_te[~hi], label="low risk"); kmf.plot_survival_function(ax=ax, lw=2)
    kmf.fit(t_te[hi], e_te[hi], label="high risk"); kmf.plot_survival_function(ax=ax, lw=2)
    lr = logrank_test(t_te[hi], t_te[~hi], event_observed_A=e_te[hi], event_observed_B=e_te[~hi])
    ax.set_title("Test set: OS by LASSO-Cox risk (log-rank p=%.4f)" % lr.p_value)
    ax.set_xlabel("days"); ax.set_ylabel("OS probability"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_lasso_risk_km.png"), dpi=150)
    plt.close(fig)
    res["test_risk_median_split"] = {"logrank_p": round(float(lr.p_value), 4),
                                     "n_high": int(hi.sum()), "n_low": int((~hi).sum())}

    # номограмма, если C-index > 0.7
    if c_test > 0.7:
        from lifelines import CoxPHFitter
        sel = coef[coef.abs() > 1e-8]
        keep_cols = list(sel.index)[:12]
        dcox = X[keep_cols].copy()
        dcox["os_days"] = np.array([t for _, t in y])
        dcox["event"] = np.array([e for e, _ in y])
        cph = CoxPHFitter()
        cph.fit(dcox, duration_col="os_days", event_col="event", show_progress=False)
        summ = cph.summary["coef"].to_dict()
        fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(keep_cols) + 2)))
        ax.axis("off")
        ax.set_title("LASSO-Cox nomogram (test C-index = %.3f)" % c_test, fontsize=12)
        spans = []
        for col in keep_cols:
            if dcox[col].nunique() <= 2:
                spans.append(max(1e-6, abs(summ[col])))
            else:
                spans.append(abs(summ[col]) * (dcox[col].max() - dcox[col].min()))
        scale = 100.0 / max(spans)
        ypos = np.arange(len(keep_cols))[::-1]
        for i, col in enumerate(keep_cols):
            pts = abs(summ[col]) * scale
            if dcox[col].nunique() <= 2:
                lbl = "%s  (0/1 -> %.0f pts)" % (col, pts)
            else:
                rng_pts = pts * (dcox[col].max() - dcox[col].min()) / max(1e-9, abs(summ[col]))
                lbl = "%s  (range %.0f pts)" % (col, rng_pts)
            ax.text(0.02, ypos[i], lbl, fontsize=9, va="center")
        ax.text(0.02, len(keep_cols) + 0.4, "Predictors (points toward total score)", fontsize=10, va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "gdc2_nomogram.png"), dpi=150)
        plt.close(fig)
        res["nomogram"] = {"built": True, "predictors": keep_cols,
                           "coefficients": {k: round(float(v), 4) for k, v in summ.items()}}
        print("nomogram built (C-index > 0.7)", flush=True)
    else:
        res["nomogram"] = {"built": False, "reason": "test C-index <= 0.7"}

    with open(os.path.join(RESULTS_DIR, "gdc2_lasso_cox.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved results/gdc2_lasso_cox.json", flush=True)


if __name__ == "__main__":
    main()
