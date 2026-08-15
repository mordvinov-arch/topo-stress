# ===== MAST: Случайные матрицы (RMT) =====
# Спектр корреляционной матрицы 19 физиологических переменных vs граница
# Марченко-Пастура; значимые компоненты и их нагрузки.

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import PROC_WIDE, RESULTS_DIR, FIGURES_DIR
from topostress.rmt import marchenko_pastur_bound

WIDE = PROC_WIDE

VARS = (
    [f"Cortisol_{i:02d}" for i in range(1, 6)]
    + [f"State_anxiety_{i:02d}" for i in range(2, 6)]
    + [f"Positive_{i:02d}" for i in range(2, 6)]
    + [f"Negative_{i:02d}" for i in range(2, 6)]
    + ["Trait_Anxiety", "BMI", "Age"]
)


def main():
    df = pd.read_csv(WIDE).dropna(subset=VARS)
    df["Group01"] = (df["Group"] == "Stress").astype(int)
    X = df[VARS].values
    Xz = (X - X.mean(0)) / X.std(0)
    n, p = Xz.shape

    C = np.corrcoef(Xz.T)
    evals = np.sort(np.linalg.eigvalsh(C))[::-1]
    lambda_plus = marchenko_pastur_bound(p, n, sigma=1.0)
    q = p / n
    lambda_minus = 1.0 ** 2 * (1 - np.sqrt(q)) ** 2
    evecs = np.linalg.eigh(C)[1][:, ::-1]

    sig_idx = np.where(evals > lambda_plus)[0]
    sig_vals = evals[sig_idx]
    print(f"n={n}, p={p}, q={p/n:.3f}")
    print(f"МП-границы: lambda+={lambda_plus:.3f}, lambda-={lambda_minus:.3f}")
    print(f"Собственные значения: {np.round(evals, 3)}")
    print(f"Значимых (>{lambda_plus:.2f}): {len(sig_vals)} из {p}")
    print(f"  значимые: {np.round(sig_vals, 3)}")

    # интерпретация главной значимой компоненты
    loadings = {}
    if len(sig_idx):
        top = sig_idx[0]
        v = evecs[:, top]
        loadings = {VARS[i]: float(v[i]) for i in range(p)}
        order = sorted(loadings.items(), key=lambda kv: -abs(kv[1]))
        print(f"\nНагрузки главной значимой компоненты (ev={evals[top]:.2f}):")
        for name, load in order[:8]:
            print(f"  {name:>18}: {load:+.3f}")

    # IPC (объяснённая доля): доля следа значимых
    total_trace = evals.sum()
    frac_sig = evals[sig_idx].sum() / total_trace if len(sig_idx) else 0.0

    out = {
        "n": int(n), "p": int(p), "q": float(p / n),
        "lambda_plus": float(lambda_plus), "lambda_minus": float(lambda_minus),
        "eigenvalues": evals.tolist(),
        "n_significant": int(len(sig_vals)),
        "significant_eigenvalues": sig_vals.tolist(),
        "frac_variance_significant": float(frac_sig),
        "top_loadings": loadings,
    }
    with open(os.path.join(RESULTS_DIR, "rmt_mast.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Сохранено: results/rmt_mast.json (доля сигн. дисперсии={frac_sig:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#e74c3c" if e > lambda_plus else "#95a5a6" for e in evals]
    axes[0].bar(range(1, p + 1), evals, color=colors)
    axes[0].axhline(lambda_plus, color="red", ls="--",
                    label=f"$\\lambda_+={lambda_plus:.2f}$")
    axes[0].set_xlabel("Компонента")
    axes[0].set_ylabel("Собственное значение")
    axes[0].set_title(f"RMT-спектр (n={n}, p={p}, q={p/n:.3f}); "
                      f"значимых: {len(sig_vals)} из {p}")
    axes[0].legend()

    if len(sig_idx):
        names = [VARS[i] for i in range(p)]
        loads = [abs(v) for v in evecs[:, sig_idx[0]]]
        axes[1].barh(range(p)[::-1], loads, color="#3498db")
        axes[1].set_yticks(range(p))
        axes[1].set_yticklabels(names[::-1], fontsize=7)
        axes[1].set_xlabel("|нагрузка|")
        axes[1].set_title(f"Нагрузки 1-й значимой компоненты "
                          f"(ev={evals[sig_idx[0]]:.2f})")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "rmt_mast.png"), dpi=150)
    print("Сохранено: figures/rmt_mast.png")


if __name__ == "__main__":
    main()
