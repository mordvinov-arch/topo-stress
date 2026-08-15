# ===== MAST: Информационная геометрия / физиотипы =====
# Для каждого испытуемого: бутстреп-распределение наклона log-кортизола
# (индивидуальная реактивность). Матрица расстояний Вассерштейна, MDS,
# Уорд-кластеризация в физиотипы, связь физиотипов с группой и ответом.

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

from topostress.config import N_BOOT, PROC_WIDE, RESULTS_DIR, FIGURES_DIR
from topostress.info_geometry import wasserstein_matrix, mds, ward_clusters

WIDE = PROC_WIDE

LOGCOLS = [f"LogCortisol_{i:02d}" for i in range(1, 6)]
RNG = np.random.default_rng(3)


def ols_slope(x, y):
    return np.polyfit(x, y, 1)[0]


def boot_slopes(y_5):
    t = np.arange(5, dtype=float)
    slopes = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = RNG.integers(0, 5, 5)
        slopes[b] = ols_slope(t, y_5[idx])
    return slopes


def main():
    df = pd.read_csv(WIDE).dropna(subset=LOGCOLS + ["Group", "LogAUCg"])
    df["Group01"] = (df["Group"] == "Stress").astype(int)
    n = len(df)

    samples = [boot_slopes(df.loc[i, LOGCOLS].values.astype(float)) for i in df.index]
    means = np.array([s.mean() for s in samples])
    print(f"Индивидуальные распределения наклона: {n}, boot={N_BOOT}")

    D = wasserstein_matrix(samples)
    print(f"Wasserstein-матрица: {D.shape}, среднее={D[D > 0].mean():.4f}")

    emb, _ = mds(D)
    embed1, embed2 = emb[:, 0], emb[:, 1]
    g = df["Group01"].values
    logauc = df["LogAUCg"].values

    r1, p1 = stats.pearsonr(embed1, logauc)
    r2, p2 = stats.pearsonr(embed2, logauc)
    rb1, pb1 = stats.pointbiserialr(g, embed1)
    print(f"MDS1 vs LogAUCg: r={r1:.3f} p={p1:.3f}")
    print(f"MDS2 vs LogAUCg: r={r2:.3f} p={p2:.3f}")
    print(f"MDS1 vs Group: r={rb1:.3f} p={pb1:.3f}")

    n_clusters = 3
    clusters, Z = ward_clusters(D, n_clusters=n_clusters)
    # переименовываем кластеры по среднему наклону (реактивности)
    cl_means = {c: means[clusters == c].mean() for c in np.unique(clusters)}
    order = sorted(cl_means, key=cl_means.get)
    label_map = {c: f"Физиотип {i + 1}" for i, c in enumerate(order)}
    cl_names = np.array([label_map[c] for c in clusters])

    print(f"\nФизиотипы (по средней реактивности):")
    for i, c in enumerate(order):
        m = clusters == c
        n_c = int(m.sum())
        g_c = g[m]
        n_stress = int(g_c.sum())
        auc_c = logauc[m]
        print(f"  {label_map[c]} (cl {c}): n={n_c}, Stress={n_stress} "
              f"({n_stress / n_c:.0%}), mean slope={cl_means[c]:+.3f}, "
              f"LogAUCg={auc_c.mean():.3f}")

    # связь физиотипа с группой
    ct = pd.crosstab(cl_names, df["Group"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    print(f"\nСвязь физиотип x группа: chi2={chi2:.2f}, p={p_chi:.3f}")

    # ANOVA LogAUCg по физиотипам
    groups_auc = [logauc[clusters == c] for c in np.unique(clusters)]
    f_stat, p_anova = stats.f_oneway(*groups_auc)
    print(f"ANOVA LogAUCg по физиотипам: F={f_stat:.2f}, p={p_anova:.3f}")

    out = {
        "n": int(n), "n_boot": N_BOOT,
        "wasserstein_mean": float(D[D > 0].mean()),
        "mds1_logauc": {"r": float(r1), "p": float(p1)},
        "mds2_logauc": {"r": float(r2), "p": float(p2)},
        "mds1_group": {"r": float(rb1), "p": float(pb1)},
        "physiotypes": {
            label_map[c]: {
                "cluster_id": int(c), "n": int((clusters == c).sum()),
                "n_stress": int(g[clusters == c].sum()),
                "mean_slope": float(cl_means[c]),
                "mean_logaucg": float(logauc[clusters == c].mean()),
            } for c in np.unique(clusters)
        },
        "chi2_group": {"chi2": float(chi2), "p": float(p_chi),
                       "table": ct.to_dict()},
        "anova_logaucg": {"F": float(f_stat), "p": float(p_anova)},
        "embedding": np.column_stack([embed1, embed2]).tolist(),
        "cluster_labels": [label_map[c] for c in clusters],
        "mean_slopes": means.tolist(),
    }
    with open(os.path.join(RESULTS_DIR, "info_geometry_mast.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Сохранено: results/info_geometry_mast.json")

    # ---- рисунок ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    im = axes[0].imshow(D, cmap="viridis", aspect="auto")
    axes[0].set_title("Матрица расстояний Вассерштейна")
    fig.colorbar(im, ax=axes[0])

    cl_colors = {"Физиотип 1": "#2ecc71", "Физиотип 2": "#f39c12",
                 "Физиотип 3": "#e74c3c"}
    for c in np.unique(clusters):
        m = clusters == c
        axes[1].scatter(embed1[m], embed2[m], s=60, alpha=0.7,
                        color=cl_colors[label_map[c]],
                        label=f"{label_map[c]} (n={int(m.sum())})")
    axes[1].set_xlabel("MDS1")
    axes[1].set_ylabel("MDS2")
    axes[1].set_title("Карта физиотипов (MDS по Вассерштейну)")
    axes[1].legend(fontsize=8)

    for i in range(3):
        c = order[i]
        m = clusters == c
        axes[2].hist(means[m], bins=25, alpha=0.5,
                     color=cl_colors[label_map[c]],
                     label=f"{label_map[c]} (n={int(m.sum())})")
    axes[2].axvline(0, color="gray", ls="--")
    axes[2].set_xlabel("Средний наклон log-кортизола")
    axes[2].set_ylabel("count")
    axes[2].set_title("Распределения индивидуальной реактивности")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "info_geometry_mast.png"), dpi=150)
    print("Сохранено: figures/info_geometry_mast.png")


if __name__ == "__main__":
    main()
