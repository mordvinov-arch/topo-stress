# ===== MAST: Функциональный анализ данных (FDA) =====
# log-кортизоловые кривые по 5 сэмплам, интерполяция на сетку,
# функциональная регрессия beta(t) на группу, maxT-пермутационный тест.

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import N_PERM, PROC_WIDE, RESULTS_DIR, FIGURES_DIR
from topostress.fda import interp_curves, pointwise_stats, max_t_stat, N_GRID, T_RAW

WIDE = PROC_WIDE
RNG = np.random.default_rng(7)

LOGCOLS = [f"LogCortisol_{i:02d}" for i in range(1, 6)]


def main():
    df = pd.read_csv(WIDE)
    df = df.dropna(subset=LOGCOLS)
    df["Group01"] = (df["Group"] == "Stress").astype(int)

    X = df[LOGCOLS].values
    g = df["Group01"].values
    grid, Y = interp_curves(X)

    r, p = pointwise_stats(Y, g)
    print(f"Функциональная регрессия: {len(df)} кривых, сетка {N_GRID} точек")

    j_max = int(np.argmax(np.abs(r)))
    print(f"max |r| = {abs(r[j_max]):.3f} на t={grid[j_max]:.2f}, p={p[j_max]:.3f}")

    sig = p < 0.05
    intervals = []
    idx = np.where(sig)[0]
    if len(idx):
        start = idx[0]
        prev = idx[0]
        for j in idx[1:]:
            if j != prev + 1:
                intervals.append((grid[start], grid[prev]))
                start = j
            prev = j
        intervals.append((grid[start], grid[prev]))
    print("Значимые интервалы (p<0.05):", [(round(a, 2), round(b, 2)) for a, b in intervals])

    # maxT-пермутация
    obs = max_t_stat(Y, g)
    perm = np.array([max_t_stat(Y, RNG.permutation(g)) for _ in range(N_PERM)])
    p_maxT = float((perm >= obs).mean())
    print(f"maxT: obs=-log10p={obs:.3f}, p_perm={p_maxT:.3f} (n_perm={N_PERM})")

    mean_s = Y[g == 1].mean(0)
    std_s = Y[g == 1].std(0)
    mean_c = Y[g == 0].mean(0)
    std_c = Y[g == 0].std(0)
    diff = mean_s - mean_c

    out = {
        "n_subjects": int(len(df)), "n_stress": int(g.sum()),
        "n_control": int((1 - g).sum()), "n_grid": N_GRID,
        "max_abs_r": float(abs(r[j_max])), "max_r_time": float(grid[j_max]),
        "max_r_p": float(p[j_max]),
        "sig_intervals": [[float(a), float(b)] for a, b in intervals],
        "maxT": {"stat": float(obs), "p": p_maxT, "n_perm": N_PERM},
        "curve": {
            "grid": grid.tolist(),
            "r": r.tolist(), "p": p.tolist(),
            "mean_stress": mean_s.tolist(), "std_stress": std_s.tolist(),
            "mean_control": mean_c.tolist(), "std_control": std_c.tolist(),
            "diff": diff.tolist(),
        },
    }
    with open(os.path.join(RESULTS_DIR, "fda_mast.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Сохранено: results/fda_mast.json")

    # ---- рисунок ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    t = grid

    for i in range(min(25, len(df))):
        color = "#e74c3c" if g[i] else "#3498db"
        axes[0, 0].plot(t, Y[i], alpha=0.25, lw=1, color=color)
    axes[0, 0].set_xlabel("t (сэмпл)")
    axes[0, 0].set_ylabel("log Cortisol")
    axes[0, 0].set_title("Все log-кортизоловые кривые (красный=Stress)")
    axes[0, 0].set_xticks(T_RAW)

    axes[0, 1].plot(t, mean_s, "r-", lw=2, label=f"Stress (n={int(g.sum())})")
    axes[0, 1].fill_between(t, mean_s - std_s, mean_s + std_s, alpha=0.2, color="red")
    axes[0, 1].plot(t, mean_c, "b-", lw=2, label=f"Control (n={int((1 - g).sum())})")
    axes[0, 1].fill_between(t, mean_c - std_c, mean_c + std_c, alpha=0.2, color="blue")
    axes[0, 1].set_xlabel("t (сэмпл)")
    axes[0, 1].set_ylabel("log Cortisol")
    axes[0, 1].set_title("Средние кривые")
    axes[0, 1].legend()

    axes[0, 2].plot(t, diff, "purple", lw=2)
    axes[0, 2].fill_between(t, 0, diff, where=(p < 0.05), alpha=0.4, color="green",
                            label="p<0.05")
    axes[0, 2].axhline(0, color="gray", ls="--")
    axes[0, 2].set_xlabel("t (сэмпл)")
    axes[0, 2].set_ylabel("d log Cortisol (Stress-Control)")
    axes[0, 2].set_title("Разность кривых")
    axes[0, 2].legend()

    axes[1, 0].plot(t, r, "g-", lw=2)
    axes[1, 0].fill_between(t, 0, r, where=(p < 0.05), alpha=0.3, color="green")
    axes[1, 0].axhline(0, color="gray", ls="--")
    axes[1, 0].set_xlabel("t (сэмпл)")
    axes[1, 0].set_ylabel("r(t)")
    axes[1, 0].set_title("Функциональная регрессия r(t) на группу")

    axes[1, 1].plot(t, -np.log10(p + 1e-300), "orange", lw=1.5)
    axes[1, 1].axhline(-np.log10(0.05), color="red", ls="--", label="p=0.05")
    axes[1, 1].axhline(np.log10(1 / 0.05), color="red", ls="--")
    axes[1, 1].set_xlabel("t (сэмпл)")
    axes[1, 1].set_ylabel("-log10 p")
    axes[1, 1].set_title("Значимость beta(t)")
    axes[1, 1].legend()

    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.9, f"max|r|={abs(r[j_max]):.3f}\nt={grid[j_max]:.2f}\n"
                              f"p={p[j_max]:.3f}\n\nmaxT p={p_maxT:.3f}",
                    fontsize=12, transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fda_mast.png"), dpi=150)
    print("Сохранено: figures/fda_mast.png")


if __name__ == "__main__":
    main()
