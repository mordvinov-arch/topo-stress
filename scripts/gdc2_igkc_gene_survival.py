# GDC2: выживаемость по каждому из 20 генов IGKC-модуля (медианный сплит, log-rank).
#
# Tumor-only, пациенты-дедупликация, os_days/event. Для каждого гена модуля:
# KM по группам high/low, log-rank p, медианы ОС. Итог: таблица + лесной график
# и сетка KM-кривых для генов с p<0.05.
#
# Выход: results/gdc2_igkc_gene_survival.json, figures/gdc2_igkc_gene_survival.png,
# figures/gdc2_igkc_gene_km.png.

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
LMAX = os.path.join(RESULTS_DIR, "gdc2_lmax.json")


def main():
    wide = pd.read_csv(WIDE, index_col=0)
    df = pd.read_csv(MERGED)
    lmax = json.load(open(LMAX, encoding="utf-8"))

    genes = [g for g, _ in lmax["Tumor"]["top20_genes"] if g in wide.columns]
    missing = [g for g, _ in lmax["Tumor"]["top20_genes"] if g not in wide.columns]
    print("IGKC module genes:", len(genes), "missing:", missing, flush=True)

    zwide = (wide.values - wide.values.mean(axis=0)) / (wide.values.std(axis=0) + 1e-12)
    gexp = {g: dict(zip(wide.index, zwide[:, wide.columns.get_loc(g)])) for g in genes}

    df = df[df["tissue"] == "Tumor"].copy()
    df = df.drop_duplicates(subset="case_id", keep="first")
    df = df.dropna(subset=["os_days"]).copy()
    df["os_days"] = df["os_days"].astype(float)
    df["event"] = df["event"].astype(int)
    print("patients:", len(df), "events:", int(df["event"].sum()), flush=True)

    res = {"method": "Per-gene OS (median split, log-rank) — 20 genes of the IGKC (lambda_max) module",
           "n_patients": int(len(df)), "n_events": int(df["event"].sum()),
           "genes_missing": missing}
    km_rows = []
    pvals = []
    for g in genes:
        df[g] = df["sample"].map(gexp[g])
        med = np.median(df[g])
        hi = df[g] > med
        kmf = KaplanMeierFitter()
        stats = {}
        for lbl, m in [("low", ~hi), ("high", hi)]:
            sub = df[m]
            kmf.fit(sub["os_days"], sub["event"])
            stats[lbl] = {"n": int(len(sub)), "events": int(sub["event"].sum()),
                          "median_os_days": float(kmf.median_survival_time_)
                          if np.isfinite(kmf.median_survival_time_) else None}
        lr = logrank_test(df["os_days"][hi], df["os_days"][~hi],
                          event_observed_A=df["event"][hi],
                          event_observed_B=df["event"][~hi])
        meds = (stats["low"]["median_os_days"], stats["high"]["median_os_days"])
        if meds[0] is not None and meds[1] is not None:
            med_ratio = round(meds[0] / meds[1], 3)
        else:
            med_ratio = None
        km_rows.append({"gene": g, "n": int(len(df)), "median_split": round(float(med), 4),
                        "median_os_low": stats["low"]["median_os_days"],
                        "median_os_high": stats["high"]["median_os_days"],
                        "median_ratio_low_high": med_ratio,
                        "logrank_p": round(float(lr.p_value), 4)})
        pvals.append(lr.p_value)
        print("%-12s p=%.4f  med low=%s high=%s ratio=%s"
              % (g, lr.p_value, stats["low"]["median_os_days"],
                 stats["high"]["median_os_days"], med_ratio), flush=True)

    res["per_gene"] = km_rows
    res["n_genes_p<0.05"] = int(np.sum(np.array(pvals) < 0.05))
    res["n_genes_p<0.005"] = int(np.sum(np.array(pvals) < 0.005))
    res["top_gene"] = min(km_rows, key=lambda r: r["logrank_p"])

    with open(os.path.join(RESULTS_DIR, "gdc2_igkc_gene_survival.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ===== лесной график: -log10(p), цвет = медианный ratio =====
    km_df = pd.DataFrame(km_rows).sort_values("logrank_p")
    fig, ax = plt.subplots(figsize=(7.5, 7))
    xs = -np.log10(np.clip(km_df["logrank_p"], 1e-6, 1))
    colors = ["crimson" if p < 0.05 else "steelblue" for p in km_df["logrank_p"]]
    ax.barh(km_df["gene"], xs, color=colors, edgecolor="k", linewidth=0.4)
    ax.axvline(-np.log10(0.05), color="gray", ls="--", lw=1)
    ax.set_xlabel("-log10(log-rank p)")
    ax.set_title("OS by per-gene median split (IGKC module)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_igkc_gene_survival.png"), dpi=150)
    plt.close(fig)

    # ===== KM для генов с p<0.05 =====
    sig = km_df[km_df["logrank_p"] < 0.05]
    if len(sig):
        n = len(sig)
        fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(14, 4 * int(np.ceil(n / 3))))
        axes = np.atleast_1d(axes).ravel()
        for ax, (_, row) in zip(axes, sig.iterrows()):
            g = row["gene"]
            hi = df[g] > row["median_split"]
            kmf = KaplanMeierFitter()
            kmf.fit(df["os_days"][~hi], df["event"][~hi], label="low")
            kmf.plot_survival_function(ax=ax, lw=2, color="seagreen")
            kmf.fit(df["os_days"][hi], df["event"][hi], label="high")
            kmf.plot_survival_function(ax=ax, lw=2, color="crimson")
            ax.set_title("%s (p=%.4f)" % (g, row["logrank_p"]), fontsize=9)
            ax.legend(fontsize=7)
        for ax in axes[len(sig):]:
            ax.axis("off")
        fig.suptitle("OS: per-gene median split (IGKC module, p<0.05)")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(FIGURES_DIR, "gdc2_igkc_gene_km.png"), dpi=150)
        plt.close(fig)
        print("saved figures (%d significant genes)" % len(sig), flush=True)
    print("saved gdc2_igkc_gene_survival.json", flush=True)


if __name__ == "__main__":
    main()
