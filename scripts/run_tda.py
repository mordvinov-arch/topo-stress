# MAST: топологический анализ (TDA).
# Нормированная топологическая дивергенция d~topo (Stress vs Control) на
# физиологических траекториях; bottleneck-дистанция (beta0), beta1-гомологии,
# d_comb; пермутационные тесты.

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import N_PERM, N_EPS, PROC_WIDE, RESULTS_DIR, FIGURES_DIR
from topostress.topology import (d_topo_normalized, d_combined, persistence_bottleneck,
                                 beta1_curve, d_topo_raw)
from topostress.utils import permutation_test

WIDE = PROC_WIDE

CORTISOL_VARS = [f"LogCortisol_{i:02d}" for i in range(1, 6)]
FULL_VARS = (CORTISOL_VARS
             + [f"State_anxiety_{i:02d}" for i in range(2, 6)]
             + [f"Negative_{i:02d}" for i in range(2, 6)]
             + [f"Positive_{i:02d}" for i in range(2, 6)])


def topo_stat_dbar(X1, X2):
    d, _, _, _ = d_topo_normalized(X1, X2, n_eps=N_EPS)
    return d


def topo_stat_beta1(X1, X2):
    X = np.vstack([X1, X2])
    D = np.max(np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2))
    t_grid = np.linspace(0, 1, N_EPS)
    b1 = beta1_curve(X1, t_grid * D)
    b2 = beta1_curve(X2, t_grid * D)
    return np.trapezoid(np.abs(b1 - b2), t_grid)


def main():
    df = pd.read_csv(WIDE).dropna(subset=FULL_VARS + ["Group"])
    df["Group01"] = (df["Group"] == "Stress").astype(int)
    print(f"Stress={int(df.Group01.sum())}, Control={int((1 - df.Group01).sum())}")

    out = {}
    for fset, vars_ in [("cortisol5d", CORTISOL_VARS), ("full17d", FULL_VARS)]:
        print(f"\n=== TDA на наборе {fset} ({len(vars_)} переменных) ===")
        scaler = StandardScaler()
        X = scaler.fit_transform(df[vars_].values)
        X1 = X[df.Group01 == 1]
        X2 = X[df.Group01 == 0]

        dbar, t_grid, b1, b2 = d_topo_normalized(X1, X2, n_eps=N_EPS)
        print(f"d~topo = {dbar:.4f}")

        # пермутационный тест d~topo
        def stat_pair(Xa, Xb):
            d, _, _, _ = d_topo_normalized(Xa, Xb, n_eps=N_EPS)
            return d

        obs, p_dbar, perms = permutation_test(stat_pair, X1, X2,
                                              n_perm=N_PERM, seed=42)
        print(f"  perm: p = {p_dbar:.3f} (n_perm={N_PERM})")

        # bottleneck
        try:
            dbot = persistence_bottleneck(X1, X2, maxdim=0)
            print(f"bottleneck(beta0) = {dbot:.4f}")
        except Exception as e:
            dbot = float("nan")
            print(f"bottleneck: не удалось ({e})")

        # d_comb
        dcomb, dbar_only, dmean = d_combined(X1, X2, lam1=0.5, lam2=0.5,
                                             n_eps=N_EPS)
        print(f"d_comb = {dcomb:.4f} (d~topo={dbar_only:.4f}, "
              f"|d-means|={dmean:.4f})")

        # beta1
        Xc = np.vstack([X1, X2])
        D = np.max(np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=2))
        db1, _, _, _ = d_topo_normalized(X1, X2, n_eps=N_EPS)
        b1c = beta1_curve(X1, t_grid * D)
        b2c = beta1_curve(X2, t_grid * D)
        dbeta1 = np.trapezoid(np.abs(b1c - b2c), t_grid)
        n_loops1 = int(np.sum(b1c > 0))
        n_loops2 = int(np.sum(b2c > 0))
        print(f"beta1: d(beta1) = {dbeta1:.4f}, число живых петель: "
              f"Stress={n_loops1}, Control={n_loops2}")

        out[fset] = {
            "n_vars": int(len(vars_)),
            "n_stress": int(len(X1)), "n_control": int(len(X2)),
            "d_topo": float(dbar), "p_d_topo": float(p_dbar),
            "d_topo_raw": float(d_topo_raw(X1, X2, n_eps=N_EPS)[0]),
            "bottleneck_beta0": (None if np.isnan(dbot) else float(dbot)),
            "d_comb": float(dcomb), "d_means": float(dmean),
            "d_beta1": float(dbeta1),
            "n_loops_stress": int(n_loops1), "n_loops_control": int(n_loops2),
            "perm_quantiles": {
                "q02.5": float(np.quantile(perms, 0.025)),
                "q50": float(np.quantile(perms, 0.5)),
                "q97.5": float(np.quantile(perms, 0.975)),
            },
            "beta_curves": {
                "t_grid": t_grid.tolist(),
                "beta0_stress": b1.tolist(), "beta0_control": b2.tolist(),
                "beta1_stress": b1c.tolist(), "beta1_control": b2c.tolist(),
            },
            "perms_d_topo": perms.tolist(),
            "n_perm": N_PERM,
        }

        # PCA-проекция для графика
        from sklearn.decomposition import PCA
        if fset == "full17d":
            pca = PCA(n_components=2)
            Xp = pca.fit_transform(X)
            out["pca"] = {"ratio": pca.explained_variance_ratio_.tolist(),
                          "X": Xp.tolist(),
                          "group": df["Group01"].astype(int).tolist()}

    with open(os.path.join(RESULTS_DIR, "tda_mast.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nСохранено: results/tda_mast.json")

    # ---- рисунок ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    f = out["full17d"]
    t = np.array(f["beta_curves"]["t_grid"])
    axes[0, 0].plot(t, f["beta_curves"]["beta0_stress"], "r-", lw=2,
                    label=f"Stress (n={f['n_stress']})")
    axes[0, 0].plot(t, f["beta_curves"]["beta0_control"], "b-", lw=2,
                    label=f"Control (n={f['n_control']})")
    axes[0, 0].fill_between(t, f["beta_curves"]["beta0_stress"],
                            f["beta_curves"]["beta0_control"], alpha=0.2,
                            color="purple")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("$\\bar\\beta_0$(t)")
    axes[0, 0].set_title(f"Кривые $\\bar\\beta_0$ (d~topo={f['d_topo']:.4f})")
    axes[0, 0].legend()

    axes[0, 1].hist(f["perms_d_topo"], bins=40, color="gray", alpha=0.7)
    axes[0, 1].axvline(f["d_topo"], color="red", lw=2,
                       label=f"observed={f['d_topo']:.4f}\np={f['p_d_topo']:.3f}")
    axes[0, 1].set_xlabel("d~topo (perm)")
    axes[0, 1].set_title("Пермутационный тест d~topo")
    axes[0, 1].legend()

    axes[0, 2].plot(t, f["beta_curves"]["beta1_stress"], "r-", lw=2,
                    label="Stress")
    axes[0, 2].plot(t, f["beta_curves"]["beta1_control"], "b-", lw=2,
                    label="Control")
    axes[0, 2].set_xlabel("t")
    axes[0, 2].set_ylabel("$\\beta_1$(t)")
    axes[0, 2].set_title(f"Кривые $\\beta_1$ (d={f['d_beta1']:.4f})")
    axes[0, 2].legend()

    c = out["cortisol5d"]
    axes[1, 0].hist(c["perms_d_topo"], bins=40, color="gray", alpha=0.7)
    axes[1, 0].axvline(c["d_topo"], color="red", lw=2,
                       label=f"d~topo={c['d_topo']:.4f} p={c['p_d_topo']:.3f}")
    axes[1, 0].set_title("d~topo, кортизол 5D")
    axes[1, 0].legend()

    axes[1, 1].bar(["d~topo", "bottleneck", "d_comb"],
                   [f["d_topo"], f["bottleneck_beta0"] or 0, f["d_comb"]],
                   color=["#9b59b6", "#3498db", "#e74c3c"])
    axes[1, 1].set_ylabel("значение")
    axes[1, 1].set_title("Метрики TDA (full17d)")

    axes[1, 2].scatter([p[0] for p in out["pca"]["X"]],
                       [p[1] for p in out["pca"]["X"]],
                       c=["#e74c3c" if gg else "#3498db" for gg in out["pca"]["group"]],
                       s=40, alpha=0.6)
    axes[1, 2].set_xlabel(f"PC1 ({out['pca']['ratio'][0]:.0%})")
    axes[1, 2].set_ylabel(f"PC2 ({out['pca']['ratio'][1]:.0%})")
    axes[1, 2].set_title("PCA физиологического пространства")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "tda_mast.png"), dpi=150)
    print("Сохранено: figures/tda_mast.png")


if __name__ == "__main__":
    main()
