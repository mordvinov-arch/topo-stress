# ===== MAST: Конформное предсказание =====
# Split-conformal: предсказание LogAUCg (интегрированный кортизоловый ответ)
# по ковариатам (Trait_Anxiety, BMI, Age, Sex, Anxiety_change, Negative_peak).
# 90%-интервалы, покрытие и ширина, в т.ч. раздельно по группам.

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import PROC_WIDE, RESULTS_DIR, FIGURES_DIR
from topostress.conformal import split_conformal

WIDE = PROC_WIDE

FEATURES = ["Trait_Anxiety", "BMI", "Age", "Anxiety_change", "Negative_peak"]
TARGET = "LogAUCg"


def main():
    df = pd.read_csv(WIDE).dropna(subset=FEATURES + [TARGET, "Gender", "Group"])
    Xraw = df[FEATURES].copy()
    for c in FEATURES:
        Xraw[c] = (Xraw[c] - Xraw[c].mean()) / Xraw[c].std()
    Xraw["Sex_Male"] = (df["Gender"] == "Male").astype(float)
    X = Xraw.values.astype(float)
    y = df[TARGET].values.astype(float)
    g = (df["Group"] == "Stress").values

    alpha = 0.1
    lower, upper, q_hat, cal_idx, model = split_conformal(X, y, alpha=alpha)
    pred = model.predict(X)
    cover_all = np.mean((y >= lower) & (y <= upper))
    width = (upper - lower).mean()

    print(f"Split-conformal: n={len(y)}, alpha={alpha}, q_hat={q_hat:.3f}")
    print(f"Покрытие (все): {cover_all:.1%}  (цель: {1 - alpha:.0%})")
    print(f"Средняя ширина интервала: {width:.3f}")

    res_by_group = {}
    for label, mask in [("Stress", g), ("Control", ~g)]:
        cov = np.mean((y[mask] >= lower[mask]) & (y[mask] <= upper[mask]))
        w = (upper[mask] - lower[mask]).mean()
        res_by_group[label] = {"n": int(mask.sum()), "coverage": float(cov),
                               "mean_width": float(w)}
        print(f"  {label:>7} (n={mask.sum()}): coverage={cov:.1%}, "
              f"width={w:.3f}")

    # вне-калибровочные (validation-подобное) покрытие
    val_mask = np.ones(len(y), dtype=bool)
    val_mask[cal_idx] = False
    cov_val = np.mean((y[val_mask] >= lower[val_mask]) &
                      (y[val_mask] <= upper[val_mask]))
    print(f"Покрытие на не-калибровочных: {cov_val:.1%}")

    out = {
        "n": int(len(y)), "alpha": alpha, "q_hat": float(q_hat),
        "coverage_all": float(cover_all),
        "coverage_non_calibration": float(cov_val),
        "mean_width": float(width), "target": TARGET,
        "features": FEATURES + ["Sex_Male"],
        "by_group": res_by_group,
        "coefs": model.coef_.tolist(), "intercept": float(model.intercept_),
    }
    with open(os.path.join(RESULTS_DIR, "conformal_mast.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Сохранено: results/conformal_mast.json")

    order = np.argsort(pred)
    fig, ax = plt.subplots(figsize=(12, 6))
    xv = np.arange(len(y))
    ax.fill_between(xv, lower[order], upper[order], alpha=0.3, color="#3498db",
                    label=f"{1 - alpha:.0%} интервал (±{q_hat:.2f})")
    ax.plot(xv, pred[order], "b-", lw=2, label="Прогноз")
    ax.scatter(xv, y[order], c=["#e74c3c" if gg else "#2ecc71" for gg in g[order]],
               s=30, zorder=5)
    ax.set_xlabel("Испытуемый (сортировка по прогнозу)")
    ax.set_ylabel("LogAUCg")
    ax.set_title(f"Конформное предсказание LogAUCg (покрытие {cover_all:.1%})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "conformal_mast.png"), dpi=150)
    print("Сохранено: figures/conformal_mast.png")


if __name__ == "__main__":
    main()
