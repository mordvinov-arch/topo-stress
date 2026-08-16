# Перегенерация ключевых рисунков с черно-белой читабельностью:
# каждая серия различается не только цветом, но и стилем линии/маркером.
# Рисунки: gdc2_tda, gdc2_evt, gdc2_hsic, gdc2_survival (переиспользуют
# результаты run_gdc_realgroups.py / gdc2_survival.py, расчёт воспроизводим).

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.stats import genextreme

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress import topology  # noqa: E402
from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from topostress.evt import gev_return_level  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
META = os.path.join(PROCESSED_DATA_DIR, "gdc_metadata.csv")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
N_EPS = 100


def main():
    wide = pd.read_csv(WIDE, index_col=0)
    meta = pd.read_csv(META)
    fn2tissue = dict(zip(meta["file_name"], meta["tissue_type"]))
    cols = list(wide.index)
    tissue = np.array([fn2tissue[c] for c in cols])
    X = wide.values
    g1 = tissue == "Tumor"
    g2 = tissue == "Normal"

    # ---- TDA: кривые Бетти-0 с заливкой области d_topo ----
    d_topo, t_grid, b1, b2 = topology.d_topo_normalized(X[g1], X[g2], n_eps=N_EPS)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(t_grid, b1, b2, color="0.7", alpha=0.55, label="d_topo = %.3f" % d_topo)
    ax.plot(t_grid, b1, color="k", ls="-", marker="o", ms=2.5, lw=1.6, label="Tumor (n=542)")
    ax.plot(t_grid, b2, color="0.35", ls="--", marker="s", ms=3, lw=1.6, label="Normal (n=59)")
    ax.set_xlabel("нормализованная шкала t")
    ax.set_ylabel("$\\bar\\beta_0(t)/\\bar n$")
    ax.set_title("TCGA-LUAD: кривые Бетти-0, опухоль против нормы (d_topo = %.3f)" % d_topo)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_tda.png"), dpi=150)
    plt.close(fig)
    print("gdc2_tda.png done (d_topo=%.3f)" % d_topo, flush=True)

    # ---- EVT: GEV-подгонки по группам ----
    maxes = X.max(axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, mask, color, ls, m in [("Tumor", g1, "k", "-", "o"),
                                     ("Normal", g2, "0.4", "--", "s")]:
        xi, mu, sigma = genextreme.fit(maxes[mask])
        rl = gev_return_level(xi, mu, sigma, 0.01)
        xx = np.linspace(maxes[mask].min() - 0.1, maxes[mask].max() + 0.1, 200)
        ax.hist(maxes[mask], bins=30, density=True, alpha=0.30, color=color,
                label="%s (RL99=%.1f)" % (name, rl))
        ax.plot(xx, genextreme.pdf(xx, xi, loc=mu, scale=sigma), color=color,
                ls=ls, marker=m, markevery=12, lw=1.8, label="GEV %s" % name)
    ax.set_xlabel("max log1p(TPM) по 500 генам"); ax.set_ylabel("плотность")
    ax.set_title("TCGA-LUAD: EVT по группам (хвосты Фреше)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_evt.png"), dpi=150)
    plt.close(fig)
    print("gdc2_evt.png done", flush=True)

    # ---- HSIC-иллюстрация: SFTPC x BPIFA1 ----
    genes = list(wide.columns)
    xg = X[:, genes.index("SFTPC")]
    yg = X[:, genes.index("BPIFA1")]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(xg[g1], yg[g1], s=10, c="k", marker="o", alpha=0.55, label="Tumor")
    ax.scatter(xg[g2], yg[g2], s=14, c="0.5", marker="^", alpha=0.8, label="Normal")
    ax.set_xlabel("SFTPC (log1p TPM)"); ax.set_ylabel("BPIFA1 (log1p TPM)")
    ax.set_title("TCGA-LUAD: SFTPC vs BPIFA1 по ткани (HSIC)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_hsic.png"), dpi=150)
    plt.close(fig)
    print("gdc2_hsic.png done", flush=True)

    # ---- Выживаемость по физиотипам (KM) ----
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    df = pd.read_csv(MERGED)
    df = df[df["tissue"] == "Tumor"].copy()
    df = df.drop_duplicates(subset="case_id", keep="first")
    df = df.dropna(subset=["os_days"]).copy()
    df["os_days"] = df["os_days"].astype(float)
    df["event"] = df["event"].astype(int)
    lr = logrank_test(df["os_days"], df["physiotype"].astype(int), df["event"])

    kmf = KaplanMeierFitter()
    styles = {1: ("-", "o", "k"), 2: ("--", "s", "0.4"), 3: (":", "^", "0.7")}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ph in sorted(df["physiotype"].unique()):
        sub = df[df["physiotype"] == ph]
        kmf.fit(sub["os_days"], sub["event"], label="физиотип %d (n=%d)" % (ph, len(sub)))
        ls, m, c = styles[int(ph)]
        kmf.plot_survival_function(ax=ax, lw=2, color=c, ls=ls,
                                   drawstyle="steps-post", ci_show=True, ci_alpha=0.15)
    ax.set_xlabel("время (дни)"); ax.set_ylabel("вероятность общей выживаемости")
    ax.set_title("TCGA-LUAD: OS по физиотипам (лог-ранг p = %.4f)" % lr.p_value)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_survival.png"), dpi=150)
    plt.close(fig)
    print("gdc2_survival.png done", flush=True)


if __name__ == "__main__":
    main()
