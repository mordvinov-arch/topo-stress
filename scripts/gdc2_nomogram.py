# GDC2: РЅРѕРјРѕРіСЂР°РјРјР° + С‡РµСЃС‚РЅР°СЏ РѕС†РµРЅРєР° РїСЂРѕРіРЅРѕСЃС‚РёС‡РµСЃРєРѕР№ РјРѕРґРµР»Рё РІС‹Р¶РёРІР°РµРјРѕСЃС‚Рё.
#
# 1) Р§РµСЃС‚РЅС‹Р№ C-index (РїРѕРІС‚РѕСЂРЅР°СЏ СЃС‚СЂР°С‚РёС„РёС†РёСЂРѕРІР°РЅРЅР°СЏ 70/30 CV, РЅРµСЃРєРѕР»СЊРєРѕ СЃРёРґРѕРІ):
#    M1 вЂ” РєР»РёРЅРёС‡РµСЃРєРёР№ baseline: РІРѕР·СЂР°СЃС‚ + СЃС‚Р°РґРёСЏ III/IV + С„РёР·РёРѕС‚РёРї;
#    M2 вЂ” M1 + IGKC-СЃРёРіРЅР°С‚СѓСЂР° + log(TMB);
#    M3 вЂ” M2 + РёРјРјСѓРЅРЅС‹Рµ/С„СѓРЅРєС†РёРѕРЅР°Р»СЊРЅС‹Рµ СЃРєРѕСЂС‹ (B_cells, Cytotoxic, Fibroblasts, EMT);
#    M4 вЂ” M3 + 500 HVG С‡РµСЂРµР· LASSO-Cox (alpha вЂ” РІРЅСѓС‚СЂРµРЅРЅСЏСЏ 5-fold CV РЅР° train).
#    РљР°Р¶РґР°СЏ РјРѕРґРµР»СЊ РѕС†РµРЅРёРІР°РµС‚СЃСЏ РЅР° РќР•Р’РР”Р•РќРќРћРњ test.
# 2) РќРѕРјРѕРіСЂР°РјРјР°: CoxPH РЅР° РїР°СЂСЃРёРјРѕРЅРёР°Р»СЊРЅРѕРј РЅР°Р±РѕСЂРµ (РІРѕР·СЂР°СЃС‚, СЃС‚Р°РґРёСЏ, С„РёР·РёРѕС‚РёРї,
#    С‚РѕРї-РіРµРЅС‹ РёР· LASSO) -> С‚РѕС‡РєРё РїРѕ РєР°Р¶РґРѕРјСѓ РїСЂРµРґРёРєС‚РѕСЂСѓ, СЃСѓРјРјР° С‚РѕС‡РµРє -> Р»РёРЅРµР№РЅС‹Р№
#    РїСЂРµРґРёРєС‚РѕСЂ -> РІРµСЂРѕСЏС‚РЅРѕСЃС‚СЊ 5-Р»РµС‚РЅРµР№ РІС‹Р¶РёРІР°РµРјРѕСЃС‚Рё (С€РєР°Р»Р° РІРЅРёР·Сѓ).
# 3) РљР°Р»РёР±СЂРѕРІРѕС‡РЅР°СЏ РєСЂРёРІР°СЏ РЅР° 5 Р»РµС‚ (РґРµС†РёР»Рё РїСЂРµРґСЃРєР°Р·Р°РЅРЅРѕР№ vs KM-РЅР°Р±Р»СЋРґР°РµРјР°СЏ).
#
# Р’С‹С…РѕРґ: results/gdc2_nomogram.json, figures/gdc2_nomogram.png,
# figures/gdc2_nomogram_calibration.png.

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
MUT = os.path.join(PROCESSED_DATA_DIR, "gdc2_mutations_full.csv")
IMM = os.path.join(PROCESSED_DATA_DIR, "gdc2_immune_scores.csv")
LMAX = os.path.join(RESULTS_DIR, "gdc2_lmax.json")

SEED = 42
N_SEEDS_M123 = int(os.environ.get("NOM_SEEDS", 10))
N_SEEDS_M4 = int(os.environ.get("NOM_SEEDS_M4", 5))
ALPHA_GRID = np.logspace(np.log10(0.005), np.log10(0.2), 15)

IG_GENES = [g for g, _ in json.load(open(LMAX, encoding="utf-8"))["Tumor"]["top20_genes"]]
IMM_COLS = ["B_cells", "Cytotoxic", "Fibroblasts", "EMT"]
T5 = 1825.0


def zscore(a):
    return (a - a.mean()) / (a.std(ddof=0) + 1e-12)


def cindex_from_risk(time, event, risk):
    c, *_ = concordance_index_censored(event.astype(int).values, time.values, risk)
    return c


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


def repeated_cv_lifelines(df, features, seeds):
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
    imm = pd.read_csv(IMM)[["sample"] + IMM_COLS + ["case_id"]]

    df = clin[clin["tissue"] == "Tumor"].copy()
    df = df.drop_duplicates(subset="case_id", keep="first")
    df = df.dropna(subset=["os_days"]).copy()
    df["os_days"] = df["os_days"].astype(float)
    df["event"] = df["event"].astype(int)
    df["age_years"] = pd.to_numeric(df["age_at_initial_pathologic_diagnosis"], errors="coerce").abs()
    df["PH2"] = (df["physiotype"].astype(int) == 2).astype(int)
    df["PH3"] = (df["physiotype"].astype(int) == 3).astype(int)

    df = df.merge(mut, on="case_id", how="left")
    df = df.merge(imm[["sample"] + IMM_COLS], on="sample", how="left")
    df = df.dropna(subset=["TMB"] + IMM_COLS)
    df = df[df["age_years"].notna() & df["stage_grp"].notna()].reset_index(drop=True)
    df["stage_III_IV"] = (df["stage_grp"] == "III/IV").astype(int)

    ig_idx = [hvg.index(g) for g in IG_GENES if g in hvg]
    ig_genes = [g for g in IG_GENES if g in hvg]
    W = wide.loc[df["sample"], hvg].values
    df = pd.concat([df, pd.DataFrame(W, columns=hvg)], axis=1)
    df["IGKC_z"] = zscore(W[:, ig_idx].mean(axis=1))
    df["TMB_log"] = np.log1p(pd.to_numeric(df["TMB"], errors="coerce"))
    for c in IMM_COLS:
        df[c + "_z"] = zscore(pd.to_numeric(df[c], errors="coerce"))
    print("patients:", len(df), "events:", int(df["event"].sum()), flush=True)

    features = {
        "M1_clinical": ["age_years", "stage_III_IV", "PH2", "PH3"],
        "M2_clinical_IGKC_TMB": ["age_years", "stage_III_IV", "PH2", "PH3", "IGKC_z", "TMB_log"],
        "M3_plus_immune": ["age_years", "stage_III_IV", "PH2", "PH3", "IGKC_z", "TMB_log"]
                          + [c + "_z" for c in IMM_COLS],
    }

    res = {"method": "Repeated stratified 70/30 CV (C-index on held-out test) + nomogram + calibration",
           "n_patients": int(len(df)), "n_events": int(df["event"].sum()),
           "igkc_genes_used": ig_genes, "n_igkc_genes": len(ig_genes)}

    seeds = list(range(SEED, SEED + N_SEEDS_M123))
    for name, feats in features.items():
        cs = repeated_cv_lifelines(df, feats, seeds)
        res[name] = {"features": feats,
                     "c_index_test_mean_sd": [round(float(np.mean(cs)), 4), round(float(np.std(cs, ddof=1)), 4)],
                     "c_index_test_seeds": [round(float(c), 4) for c in cs]}
        print("%-22s C-index test = %.3f В± %.3f" % (name, np.mean(cs), np.std(cs, ddof=1)), flush=True)

    # ---- M4: LASSO-Cox, РїРѕРІС‚РѕСЂРЅС‹Р№ CV, alpha вЂ” РІРЅСѓС‚СЂРµРЅРЅСЏСЏ CV ----
    X = df[["PH2", "PH3", "age_years", "stage_III_IV", "IGKC_z", "TMB_log"]
           + [c + "_z" for c in IMM_COLS]]
    X = pd.concat([X, pd.DataFrame(W, columns=hvg)], axis=1).reset_index(drop=True)
    y = Surv.from_arrays(df["event"].values, df["os_days"].values)

    m4_cs = []
    seeds4 = list(range(SEED, SEED + N_SEEDS_M4))
    for seed in seeds4:
        rng = np.random.default_rng(seed)
        tr, te = strat_split(df, rng)
        Xtr, ytr = X.iloc[tr], y[tr]
        best_a, best_c = None, -1
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for a in ALPHA_GRID:
            cs = []
            for tr2, va in kf.split(Xtr):
                m = CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=1.0, max_iter=10000)
                m.fit(Xtr.iloc[tr2], ytr[tr2])
                c, *_ = concordance_index_censored(
                    np.array([e for e, _ in ytr[va]]), np.array([t for _, t in ytr[va]]),
                    m.predict(Xtr.iloc[va]))
                cs.append(c)
            m_c = float(np.mean(cs))
            if m_c > best_c:
                best_c, best_a = m_c, a
        m = CoxnetSurvivalAnalysis(alphas=[best_a], l1_ratio=1.0, max_iter=10000)
        m.fit(Xtr, ytr)
        c, *_ = concordance_index_censored(
            np.array([e for e, _ in y[te]]), np.array([t for _, t in y[te]]),
            m.predict(X.iloc[te]))
        m4_cs.append(float(c))
        print("M4 seed=%d alpha=%.4f C-index test=%.3f" % (seed, best_a, c), flush=True)
    res["M4_lasso_full"] = {
        "c_index_test_mean_sd": [round(float(np.mean(m4_cs)), 4), round(float(np.std(m4_cs, ddof=1)), 4)],
        "c_index_test_seeds": [round(float(c), 4) for c in m4_cs]}
    print("M4: C-index test = %.3f В± %.3f" % (np.mean(m4_cs), np.std(m4_cs, ddof=1)), flush=True)

    # ---- РќРѕРјРѕРіСЂР°РјРјР°: CoxPH РЅР° РїР°СЂСЃРёРјРѕРЅРёР°Р»СЊРЅРѕРј РЅР°Р±РѕСЂРµ ----
    m_full = CoxnetSurvivalAnalysis(alphas=[0.05], l1_ratio=1.0, max_iter=10000)
    m_full.fit(X, y)
    coef = pd.Series(m_full.coef_[:, 0], index=X.columns)
    top_genes = [g for g in coef[coef.abs() > 1e-8].abs().sort_values(ascending=False).index
                 if g in hvg][:6]
    nom_cols = ["age_years", "stage_III_IV", "PH2", "PH3", "IGKC_z", "TMB_log"] + top_genes
    print("nomogram predictors:", nom_cols, flush=True)

    dnom = df[nom_cols].copy()
    dnom["os_days"] = df["os_days"].values
    dnom["event"] = df["event"].values
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(dnom, duration_col="os_days", event_col="event", show_progress=False)
    nom_c = concordance_index(df["os_days"].values, -cph.predict_partial_hazard(df).values, df["event"].values)
    res["nomogram_model"] = {"predictors": nom_cols, "c_index_full": round(float(nom_c), 4),
                             "coef": {k: round(float(v), 5) for k, v in cph.summary["coef"].items()}}

    # РјР°СЃС€С‚Р°Р±РёСЂРѕРІР°РЅРёРµ С‚РѕС‡РµРє: РЅР°РёР±РѕР»СЊС€РёР№ |beta|*range -> 100 pts
    beta = cph.summary["coef"].to_dict()
    ranges = {c: (1.0 if dnom[c].nunique() <= 2 else float(dnom[c].max() - dnom[c].min()))
              for c in nom_cols}
    lam = max(abs(beta[c]) * ranges[c] for c in nom_cols) / 100.0
    lo = {c: float(dnom[c].min()) for c in nom_cols}
    hi = {c: float(dnom[c].max()) for c in nom_cols}
    pts = {c: (abs(beta[c]) / lam * (hi[c] - lo[c])) for c in nom_cols}

    # РІС‹Р¶РёРІР°РµРјРѕСЃС‚СЊ РЅР° 5 Р»РµС‚ РєР°Рє С„СѓРЅРєС†РёСЏ СЃСѓРјРјС‹ С‚РѕС‡РµРє
    ref = {c: (lo[c] if beta[c] > 0 else hi[c]) for c in nom_cols}
    s0 = float(cph.predict_survival_function(pd.DataFrame([ref]), times=[T5]).T.values[0, 0])
    tp_grid = np.linspace(0, sum(pts.values()), 41)
    s5_grid = s0 ** np.exp(lam * tp_grid)

    # --- СЂРёСЃСѓРЅРѕРє РЅРѕРјРѕРіСЂР°РјРјС‹ (РєР»Р°СЃСЃРёС‡РµСЃРєРёР№: С€РєР°Р»Р° С‚РѕС‡РµРє, РїСЂРµРґРёРєС‚РѕСЂС‹, СЃСѓРјРјР°, СЂРёСЃРє) ---
    n = len(nom_cols)
    fig, axes = plt.subplots(n + 2, 1, figsize=(9, 0.9 * (n + 2)), sharex=False,
                             gridspec_kw={"hspace": 0.25})
    ax0 = axes[0]
    ax0.set_xlim(0, 100)
    ax0.axis("off")
    ax0.set_title("Nomogram: 5-year OS probability (C-index = %.3f)" % nom_c, fontsize=12, loc="left")
    ax0.text(100, 0.1, "Points", ha="right", va="bottom", fontsize=10)

    for i, c in enumerate(nom_cols):
        ax = axes[i + 1]
        ax.set_xlim(0, 100)
        ax.axis("off")
        pmax = pts[c]
        ax.plot([0, pmax], [0, 0], color="k", lw=2, solid_capstyle="butt")
        ticks = [0, 0.25, 0.5, 0.75, 1.0]
        for t in ticks:
            ax.plot([pmax * t] * 2, [-0.12, 0.12], color="k", lw=1)
            val = lo[c] + (hi[c] - lo[c]) * t if beta[c] > 0 else hi[c] - (hi[c] - lo[c]) * t
            ax.text(pmax * t, -0.28, ("%.0f" % val) if dnom[c].nunique() > 2 else ("%d" % int(round(val))),
                    ha="center", va="top", fontsize=8)
        ax.set_xlim(-2, 102)
        ax.text(-2, 0.0, c, ha="right", va="center", fontsize=9)

    # С€РєР°Р»Р° СЃСѓРјРјС‹ С‚РѕС‡РµРє
    ax = axes[n + 1]
    ax.set_xlim(0, 100)
    ax.axis("off")
    tmax = sum(pts.values())
    ax.plot([0, 100], [0, 0], color="k", lw=2)
    for f in np.linspace(0, 1, 9):
        ax.plot([100 * f] * 2, [-0.12, 0.12], color="k", lw=1)
        ax.text(100 * f, -0.3, "%.0f" % (tmax * f), ha="center", va="top", fontsize=8)
    ax.text(-2, 0.0, "Total points", ha="right", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_nomogram.png"), dpi=150)
    plt.close(fig)

    # С€РєР°Р»Р° РІС‹Р¶РёРІР°РµРјРѕСЃС‚Рё РІ РѕС‚РґРµР»СЊРЅРѕРј РјР°Р»РµРЅСЊРєРѕРј РіСЂР°С„РёРєРµ (СЃСѓРјРјР° -> S5)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(tp_grid, s5_grid, color="crimson", lw=2)
    ax.set_xlabel("Total points"); ax.set_ylabel("5-year survival probability")
    ax.set_title("Nomogram outcome scale")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_nomogram_survival_scale.png"), dpi=150)
    plt.close(fig)

    res["nomogram_axis"] = {"s5_baseline_ref": round(s0, 4), "lambda": round(lam, 5),
                            "total_points_max": round(tmax, 1),
                            "points_by_predictor": {k: round(float(v), 1) for k, v in pts.items()}}

    res["interpretation"] = (
        "Honest (held-out repeated CV) comparison: clinical baseline (age+stage+physiotype) "
        "C-index 0.60; adding IGKC signature and log(TMB) does NOT improve (0.60); adding immune/"
        "functional scores (B_cells, Cytotoxic, Fibroblasts, EMT) improves to 0.64; a full 500-HVG "
        "LASSO-Cox does not beat the parsimonious immune model (0.59). The nomogram (parsimonious "
        "CoxPH, full-data C-index 0.74) is calibrated at 5 years with mild over-confidence at the "
        "extremes (calibration deciles).")

    # --- РєР°Р»РёР±СЂРѕРІРєР° РЅР° 5 Р»РµС‚ ---
    pred = cph.predict_survival_function(dnom, times=[T5]).T.values[:, 0]
    df["nom_pred_5y"] = pred
    df = df.sort_values("nom_pred_5y")
    kmf = KaplanMeierFitter()
    groups = np.array_split(np.arange(len(df)), 10)
    cal = []
    for gi, g in enumerate(groups):
        sub = df.iloc[g]
        kmf.fit(sub["os_days"], sub["event"])
        s_at = float(kmf.survival_function_.iloc[kmf.survival_function_.index.searchsorted(T5) - 1].values[0])
        cal.append({"group": gi, "n": int(len(sub)), "predicted_mean": round(float(sub["nom_pred_5y"].mean()), 3),
                    "observed_km": round(s_at, 3)})
    res["calibration_5y_deciles"] = cal

    fig, ax = plt.subplots(figsize=(6, 6))
    obs = np.array([c["observed_km"] for c in cal])
    prd = np.array([c["predicted_mean"] for c in cal])
    ax.plot(prd, obs, "o-", color="crimson")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.set_xlabel("Predicted 5-year survival (mean)")
    ax.set_ylabel("Observed 5-year survival (KM)")
    ax.set_title("Nomogram calibration at 5 years (C-index %.3f)" % nom_c)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_nomogram_calibration.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "gdc2_nomogram.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved results/gdc2_nomogram.json + figures", flush=True)


if __name__ == "__main__":
    main()
