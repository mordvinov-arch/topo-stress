# ===== MAST: HSIC — нелинейные зависимости =====
# HSIC между бинарной группой и физиологическими исходами vs линейная
# (точечно-бисериальная) корреляция. Пермутационный тест.

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import N_PERM, PROC_WIDE, RESULTS_DIR, FIGURES_DIR
from topostress.hsic import hsic_test_median

WIDE = PROC_WIDE
OUTCOMES = ["Cortisol_peak", "LogAUCg", "Slope_logcortisol",
            "Anxiety_peak", "Negative_peak", "Trait_Anxiety"]


def main():
    df = pd.read_csv(WIDE).dropna(subset=OUTCOMES + ["Group"])
    df["Group01"] = (df["Group"] == "Stress").astype(int)
    g = df["Group01"].values.astype(float)

    rows = []
    for var in OUTCOMES:
        y = df[var].values.astype(float)
        hs, p_hsic = hsic_test_median(g, y, n_perm=N_PERM, seed=42)
        r, p_pearson = stats.pointbiserialr(g, y)
        rows.append({
            "variable": var, "hsic": float(hs), "p_hsic": float(p_hsic),
            "r_pearson": float(r), "p_pearson": float(p_pearson),
            "delta_pearson_minus_hsic_p": float(p_pearson - p_hsic),
        })
        print(f"{var:>18}: HSIC={hs:.5f} (p={p_hsic:.3f}) | "
              f"point-biserial r={r:+.3f} (p={p_pearson:.3f})")

    with open(os.path.join(RESULTS_DIR, "hsic_mast.json"), "w",
              encoding="utf-8") as f:
        json.dump({"n": int(len(df)), "n_perm": N_PERM,
                   "results": rows}, f, ensure_ascii=False, indent=2)
    print("Сохранено: results/hsic_mast.json")

    names = [r["variable"] for r in rows]
    xp = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(xp - 0.2, [abs(r["r_pearson"]) for r in rows], 0.4,
                label="|r| Pearson", color="#27ae60")
    axes[0].bar(xp + 0.2, [r["hsic"] for r in rows], 0.4,
                label="HSIC", color="#8e44ad")
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_ylabel("мера зависимости")
    axes[0].legend()

    axes[1].bar(xp - 0.2, [r["p_pearson"] for r in rows], 0.4,
                label="p Pearson", color="#27ae60")
    axes[1].bar(xp + 0.2, [r["p_hsic"] for r in rows], 0.4,
                label="p HSIC", color="#8e44ad")
    axes[1].axhline(0.05, color="red", ls="--")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(names, rotation=20)
    axes[1].set_ylabel("p-value")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hsic_mast.png"), dpi=150)
    print("Сохранено: figures/hsic_mast.png")


if __name__ == "__main__":
    main()
