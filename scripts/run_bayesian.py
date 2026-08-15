# ===== MAST: Байесовская иерархическая регрессия =====
# LogCortisol(t) ~ Time * Group + ковариаты + (Time | SubID) + (1 | Exp)
# Сравнение моделей LOO + BMA-веса. Сохраняет результаты в results/bayesian_mast.json
# и рисунок figures/bayesian_mast.png.
#
# ВАЖНО: группировка по SubID (371 уровень), а не по Subject.
# Колонка Subject в Recode-файле повторяется между экспериментами (FLH01/FLH02),
# но коды относятся к РАЗНЫМ людям (разные Age/BMI/кортизол), поэтому она не
# является идентификатором испытуемого.

import json
import os
import sys

import arviz as az
import bambi as bmb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.bayesian import MODEL_SPECS, model_weight, decide_model
from topostress.config import (
    DRAWS, TUNE, TARGET_ACCEPT, NONCENTERED,
    PROC_LONG_CORT, RESULTS_DIR, FIGURES_DIR,
)
from topostress.data import zscore

LONG = PROC_LONG_CORT


def main():
    df = pd.read_csv(LONG)
    df = df.dropna(subset=["LogCortisol"])
    df["Group01"] = (df["Group"] == "Stress").astype(int)
    df = zscore(df, ["Trait_Anxiety", "BMI", "Age"])
    print(f"Строк: {len(df)}, испытуемых (SubID): {df['SubID'].nunique()}, "
          f"экспериментов: {df['Exp'].nunique()}")

    specs = MODEL_SPECS

    idatas = {}
    summaries = {}
    for name, formula in specs.items():
        print(f"\n=== {name} ===")
        model = bmb.Model(formula, data=df, family="gaussian",
                          noncentered=NONCENTERED)
        idata = model.fit(draws=DRAWS, tune=TUNE, chains=2, cores=1,
                          target_accept=TARGET_ACCEPT, random_seed=42,
                          idata_kwargs={"log_likelihood": True})
        idatas[name] = idata
        div = int(idata.sample_stats.diverging.sum())
        summary = az.summary(idata, ci_prob=0.94)
        print(f"divergences = {div}")
        print(summary[["mean", "sd", "eti94_lb", "eti94_ub", "r_hat"]].to_string())
        summaries[name] = {
            "divergences": div,
            "params": {k: {kk: float(vv) for kk, vv in v.items()}
                       for k, v in summary.to_dict("index").items()},
        }

    comparison = az.compare(idatas)
    print("\n=== LOO (stacking) ===")
    print(comparison[["elpd", "se", "p", "elpd_diff", "weight"]].to_string() if
          "weight" in comparison.columns else comparison.to_string())

    weights = {name: model_weight(comparison, name) for name in MODEL_SPECS}
    print("BMA веса:", {k: round(v, 3) for k, v in weights.items()})

    decision, primary, statement = decide_model(summaries)
    if statement is None:
        print("\nРасходимостей в M3 нет: ковариаты остаются, основная модель M3.")
    else:
        print(f"\nM3: расходимости — ковариаты исключаются из отчёта, "
              f"основная модель M2.")

    results = {
        "draws": DRAWS,
        "tune": TUNE,
        "target_accept": TARGET_ACCEPT,
        "noncentered": NONCENTERED,
        "n_rows": int(len(df)),
        "n_subjects": int(df["SubID"].nunique()),
        "summaries": summaries,
        "loo": comparison.reset_index().to_dict("records"),
        "bma_weights": weights,
        "decision": decision,
        "primary_model": primary,
        "excluded_statement": statement,
    }
    with open(os.path.join(RESULTS_DIR, "bayesian_mast.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nСохранено: results/bayesian_mast.json")

    plot_from_results(results)


def plot_from_results(results):
    summaries = results["summaries"]
    weights = results["bma_weights"]
    names = list(summaries.keys())
    colors = {"Time": "#e67e22", "Group01": "#e74c3c", "Time:Group01": "#8e44ad"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Байесовские иерархические модели (NUTS, 2x4000 warmup + 2x4000, "
                 f"target_accept=0.99, нецентр. параметризация; decision={results['decision']})",
                 fontsize=11)

    # ---- (0,0) фиксированные эффекты ----
    x = np.arange(len(names))
    for p, c in colors.items():
        means, lbs, ubs = [], [], []
        for name in names:
            pm = summaries[name]["params"].get(p)
            if pm is None:
                means.append(np.nan); lbs.append(np.nan); ubs.append(np.nan)
                continue
            means.append(float(pm["mean"]))
            lbs.append(float(pm["eti94_lb"]))
            ubs.append(float(pm["eti94_ub"]))
        axes[0, 0].errorbar(x, means, fmt="o", color=c, capsize=4, lw=1.5,
                            label=p,
                            yerr=[np.abs(np.array(means) - np.array(lbs)),
                                  np.abs(np.array(ubs) - np.array(means))])
    axes[0, 0].axhline(0, color="gray", ls="--", lw=1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([n[:2] for n in names])
    axes[0, 0].set_title("Фиксированные эффекты (94% ETI)")
    axes[0, 0].set_ylabel("β")
    axes[0, 0].legend(fontsize=8)

    # ---- (0,1) BMA веса ----
    w_names = list(weights.keys())
    w_vals = [float(v) for v in weights.values()]
    axes[0, 1].bar(w_names, w_vals, color=["#3498db", "#e74c3c", "#27ae60"])
    axes[0, 1].set_ylabel("BMA weight (LOO)")
    axes[0, 1].set_title("Веса моделей (LOO-stacking)")
    axes[0, 1].set_xticks(range(len(w_names)))
    axes[0, 1].set_xticklabels([n[:2] for n in w_names], rotation=15)
    for i, v in enumerate(w_vals):
        axes[0, 1].text(i, v + 0.01, f"{v:.2f}", ha="center")

    # ---- (1,0) компоненты дисперсии случайных эффектов ----
    var_cols = ["1|Exp_sigma", "1|SubID_sigma", "Time|SubID_sigma"]
    var_labels = {"1|Exp_sigma": "σ_Exp", "1|SubID_sigma": "σ_Subj_int",
                  "Time|SubID_sigma": "σ_Subj_Time"}
    hier_names = [n for n in names if "Hier" in n or "covariates" in n]
    if hier_names:
        xx = np.arange(len(var_cols))
        for j, name in enumerate(hier_names):
            means, lbs, ubs = [], [], []
            for vc in var_cols:
                pm = summaries[name]["params"].get(vc)
                if pm is None:
                    means.append(np.nan); lbs.append(np.nan); ubs.append(np.nan)
                    continue
                means.append(float(pm["mean"]))
                lbs.append(float(pm["eti94_lb"]))
                ubs.append(float(pm["eti94_ub"]))
            off = (j - (len(hier_names) - 1) / 2) * 0.18
            axes[1, 0].errorbar(xx + off, means, fmt="o", capsize=4, lw=1.5,
                                label=name[:2],
                                yerr=[np.abs(np.array(means) - np.array(lbs)),
                                      np.abs(np.array(ubs) - np.array(means))])
        axes[1, 0].set_xticks(xx)
        axes[1, 0].set_xticklabels([var_labels[v] for v in var_cols])
        axes[1, 0].set_ylabel("σ")
        axes[1, 0].set_title("Станд. отклонения случайных эффектов (94% ETI)")
        axes[1, 0].legend(fontsize=8)
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(0.1, 0.5, "нет иерархических моделей", transform=axes[1, 0].transAxes)

    # ---- (1,1) ковариаты M3 ----
    cov_names = ["Trait_Anxiety_z", "BMI_z", "Age_z", "Sex"]
    cov_labels = {"Trait_Anxiety_z": "Тревожность(черта)",
                  "BMI_z": "ИМТ", "Age_z": "Возраст", "Sex": "Пол"}
    if "M3_Plus_covariates" in summaries:
        pmap = summaries["M3_Plus_covariates"]["params"]
        present = [c for c in cov_names if c in pmap]
        if present:
            xx = np.arange(len(present))
            means = [float(pmap[c]["mean"]) for c in present]
            lbs = [float(pmap[c]["eti94_lb"]) for c in present]
            ubs = [float(pmap[c]["eti94_ub"]) for c in present]
            axes[1, 1].errorbar(xx, means, fmt="s", color="#e74c3c", capsize=4,
                                lw=1.5,
                                yerr=[np.abs(np.array(means) - np.array(lbs)),
                                      np.abs(np.array(ubs) - np.array(means))])
            axes[1, 1].axhline(0, color="gray", ls="--", lw=1)
            axes[1, 1].set_xticks(xx)
            axes[1, 1].set_xticklabels([cov_labels[c] for c in present],
                                       rotation=15, fontsize=8)
            axes[1, 1].set_ylabel("β")
            if results.get("decision") == "M3_kept":
                axes[1, 1].set_title("Ковариаты M3 (94% ETI) — модель принята")
            else:
                axes[1, 1].set_title("Ковариаты M3 (94% ETI) — из-за расходимостей "
                                     "исключены из отчёта")
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "bayesian_mast.png"), dpi=150)
    print("Сохранено: figures/bayesian_mast.png")


if __name__ == "__main__":
    main()
