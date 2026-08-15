# MAST: новые анализы (расширение статьи).
# 1) Топология облака траекторий (t, logCortisol) — все точки по субъектам
# 2) d~topo(t): префиксная топология по первым k сэмплам + "топологический AUC"
# 3) Мета-топология: d~topo(Stress vs Control) по экспериментам + объединение Фишера
# 4) d_comb(lambda): выбор lambda максимизирующим отношение сигнал/шум (H0-пермутации)

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

from topostress.config import PROC_WIDE, RESULTS_DIR, FIGURES_DIR
from topostress.topology import d_topo_normalized, d_combined
from topostress.utils import fisher_combine

WIDE = PROC_WIDE

LOGCOLS = [f"LogCortisol_{i:02d}" for i in range(1, 6)]
FULL_VARS = (LOGCOLS
             + [f"State_anxiety_{i:02d}" for i in range(2, 6)]
             + [f"Negative_{i:02d}" for i in range(2, 6)]
             + [f"Positive_{i:02d}" for i in range(2, 6)])

N_PERM = int(os.environ.get("MAST_PERM", 300))
N_EPS = 150
RNG = np.random.default_rng(11)


def dbar_stat(Xa, Xb, n_eps=N_EPS):
    d, _, _, _ = d_topo_normalized(Xa, Xb, n_eps=n_eps)
    return d


def subject_perm_p(X1, X2, stat, n_perm=N_PERM):
    """Пермутация на уровне испытуемых (строки X = субъекты)."""
    X = np.vstack([X1, X2])
    n1 = len(X1)
    obs = stat(X1, X2)
    cnt = 0
    for _ in range(n_perm):
        idx = RNG.permutation(len(X))
        if stat(X[idx[:n1]], X[idx[n1:]]) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def main():
    df = pd.read_csv(WIDE).dropna(subset=FULL_VARS + ["Group"])
    df["Group01"] = (df["Group"] == "Stress").astype(int)
    out = {}

    # ---------- 1. Топология облака траекторий (t, logCortisol) ----------
    t = np.arange(5, dtype=float)
    cloud_s = np.column_stack([np.repeat(t, (df.Group01 == 1).sum()),
                               df.loc[df.Group01 == 1, LOGCOLS].values.ravel()])
    cloud_c = np.column_stack([np.repeat(t, (df.Group01 == 0).sum()),
                               df.loc[df.Group01 == 0, LOGCOLS].values.ravel()])
    obs1, p1 = subject_perm_p(cloud_s, cloud_c, dbar_stat,
                              n_perm=min(N_PERM, 150))
    print(f"[1] Топология облака траекторий: d~topo={obs1:.4f}, p={p1:.3f}",
          flush=True)

    # ---------- 2. d~topo(k) по первым k сэмплам + топологический AUC ----------
    d_k = {}
    for k in range(1, 6):
        cols = LOGCOLS[:k]
        Xs = df.loc[df.Group01 == 1, cols].values
        Xc = df.loc[df.Group01 == 0, cols].values
        sc = StandardScaler().fit(np.vstack([Xs, Xc]))
        obs_k, p_k = subject_perm_p(sc.transform(Xs), sc.transform(Xc), dbar_stat)
        d_k[k] = {"d": float(obs_k), "p": float(p_k)}
        print(f"[2] k={k}: d~topo={obs_k:.4f}, p={p_k:.3f}", flush=True)
    ks = np.array(sorted(d_k))
    topo_auc = float(np.trapezoid([d_k[k]["d"] for k in ks], ks / ks.max()))
    print(f"[2] Топологический AUC = {topo_auc:.4f}")

    # ---------- 3. Мета-топология по экспериментам ----------
    exps = [e for e in df["Exp"].unique() if
            ((df["Exp"] == e) & (df.Group01 == 1)).sum() > 1 and
            ((df["Exp"] == e) & (df.Group01 == 0)).sum() > 1]
    meta = {}
    ps = []
    for e in sorted(exps):
        sub = df[df["Exp"] == e]
        Xs = sub.loc[sub.Group01 == 1, FULL_VARS].values
        Xc = sub.loc[sub.Group01 == 0, FULL_VARS].values
        sc = StandardScaler().fit(np.vstack([Xs, Xc]))
        obs_e, p_e = subject_perm_p(sc.transform(Xs), sc.transform(Xc), dbar_stat)
        meta[e] = {"n_stress": int(len(Xs)), "n_control": int(len(Xc)),
                   "d_topo": float(obs_e), "p": float(p_e)}
        ps.append(p_e)
        print(f"[3] Exp {e}: n={len(Xs)}/{len(Xc)}, d~topo={obs_e:.4f}, "
              f"p={p_e:.3f}", flush=True)
    p_fisher = float(fisher_combine(ps))
    print(f"[3] Объединение Фишера: p_meta={p_fisher:.4f}")

    # ---------- 4. d_comb(lambda) с выбором по сигнал/шум ----------
    Xs = df.loc[df.Group01 == 1, FULL_VARS].values
    Xc = df.loc[df.Group01 == 0, FULL_VARS].values
    sc = StandardScaler().fit(np.vstack([Xs, Xc]))
    Xs, Xc = sc.transform(Xs), sc.transform(Xc)
    X = np.vstack([Xs, Xc])
    n1 = len(Xs)

    grid = np.linspace(0, 1, 11)
    snr = np.zeros(len(grid))
    dcomb_obs = np.zeros(len(grid))
    null = {l: [] for l in grid}
    n_null = min(N_PERM, 150)
    print(f"[4] null-пермутации d_comb: {n_null}", flush=True)
    for _ in range(n_null):
        idx = RNG.permutation(len(X))
        Xa, Xb = X[idx[:n1]], X[idx[n1:]]
        for j, l in enumerate(grid):
            dc, _, _ = d_combined(Xa, Xb, lam1=l, lam2=1 - l, n_eps=N_EPS)
            null[l].append(dc)
    for j, l in enumerate(grid):
        dc, _, _ = d_combined(Xs, Xc, lam1=l, lam2=1 - l, n_eps=N_EPS)
        dcomb_obs[j] = dc
        arr = np.array(null[l])
        snr[j] = (dc - arr.mean()) / (arr.std() + 1e-12)
    j_best = int(np.argmax(snr))
    print(f"[4] d_comb(lambda): лучший lambda={grid[j_best]:.2f} "
          f"(snr={snr[j_best]:.2f}, d_comb={dcomb_obs[j_best]:.4f})")

    out = {
        "1_trajectory_cloud": {"d_topo": float(obs1), "p": float(p1),
                               "n_points_stress": int(len(cloud_s)),
                               "n_points_control": int(len(cloud_c))},
        "2_prefix_topo": {str(k): v for k, v in d_k.items()},
        "2_topo_auc": topo_auc,
        "3_meta_per_exp": meta, "3_fisher_p": p_fisher,
        "4_dcomb": {"lambda_grid": grid.tolist(), "snr": snr.tolist(),
                    "d_comb_obs": dcomb_obs.tolist(),
                    "best_lambda": float(grid[j_best]),
                    "best_snr": float(snr[j_best]),
                    "best_d_comb": float(dcomb_obs[j_best])},
        "n_perm": N_PERM, "n_eps": N_EPS,
    }
    with open(os.path.join(RESULTS_DIR, "novelty_mast.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Сохранено: results/novelty_mast.json")

    # ---- рисунок ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    axes[0, 0].plot(ks, [d_k[k]["d"] for k in ks], "o-", color="#8e44ad")
    axes[0, 0].set_xlabel("первых k сэмплов")
    axes[0, 0].set_ylabel("d~topo(k)")
    axes[0, 0].set_title(f"Префиксная топология (AUC={topo_auc:.3f})")
    for k in ks:
        axes[0, 0].annotate(f"p={d_k[k]['p']:.2f}", (k, d_k[k]["d"]),
                            textcoords="offset points", xytext=(0, 6), fontsize=8)

    names = list(meta.keys())
    axes[0, 1].bar(range(len(names)), [meta[e]["d_topo"] for e in names],
                   color="#3498db")
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names)
    axes[0, 1].set_ylabel("d~topo")
    axes[0, 1].set_title(f"Мета-топология по экспериментам (Fisher p={p_fisher:.3f})")
    for i, e in enumerate(names):
        axes[0, 1].text(i, meta[e]["d_topo"], f"p={meta[e]['p']:.2f}",
                        ha="center", fontsize=8)

    axes[1, 0].plot(grid, snr, "o-", color="#e74c3c")
    axes[1, 0].axvline(grid[j_best], color="gray", ls="--",
                       label=f"lambda*={grid[j_best]:.2f}")
    axes[1, 0].set_xlabel("lambda")
    axes[1, 0].set_ylabel("сигнал/шум (z)")
    axes[1, 0].set_title("Выбор lambda для d_comb")
    axes[1, 0].legend()

    axes[1, 1].plot(grid, dcomb_obs, "o-", color="#27ae60")
    axes[1, 1].set_xlabel("lambda")
    axes[1, 1].set_ylabel("d_comb")
    axes[1, 1].set_title(f"d_comb(lambda); best d_comb={dcomb_obs[j_best]:.4f}")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "novelty_mast.png"), dpi=150)
    print("Сохранено: figures/novelty_mast.png")


if __name__ == "__main__":
    main()
