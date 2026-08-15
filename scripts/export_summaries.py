# Экспорт сводных CSV из JSON-результатов.
# results/bayesian_summary.csv, results/hsic_results.csv, results/physiotypes.csv

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import RESULTS_DIR

FIXED_EFFECTS = ["Intercept", "Time", "Group01", "Time:Group01"]


def bayesian_summary():
    src = os.path.join(RESULTS_DIR, "bayesian_mast.json")
    dst = os.path.join(RESULTS_DIR, "bayesian_summary.csv")
    if not os.path.exists(src):
        print("  пропуск: нет bayesian_mast.json")
        return
    data = json.load(open(src, encoding="utf-8"))
    rows = []
    for model, sm in data["summaries"].items():
        for param, st in sm["params"].items():
            if param in FIXED_EFFECTS or param.endswith("_sigma") or param.endswith("_z"):
                rows.append({
                    "model": model, "param": param,
                    "mean": round(st.get("mean", float("nan")), 4),
                    "eti94_lb": round(st.get("eti94_lb", float("nan")), 4),
                    "eti94_ub": round(st.get("eti94_ub", float("nan")), 4),
                    "r_hat": st.get("r_hat", float("nan")),
                })
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "param", "mean", "eti94_lb", "eti94_ub", "r_hat"])
        w.writeheader()
        w.writerows(rows)
    print(f"  bayesian_summary.csv: {len(rows)} строк")


def hsic_results():
    src = os.path.join(RESULTS_DIR, "hsic_mast.json")
    dst = os.path.join(RESULTS_DIR, "hsic_results.csv")
    if not os.path.exists(src):
        print("  пропуск: нет hsic_mast.json")
        return
    data = json.load(open(src, encoding="utf-8"))
    rows = []
    for r in data["results"]:
        rows.append({
            "variable": r["variable"], "hsic": round(r["hsic"], 6),
            "p_hsic": r["p_hsic"], "r_pearson": round(r["r_pearson"], 4),
            "p_pearson": r["p_pearson"],
            "delta_p_pearson_minus_hsic": round(r["delta_pearson_minus_hsic_p"], 4),
        })
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variable", "hsic", "p_hsic", "r_pearson",
                                          "p_pearson", "delta_p_pearson_minus_hsic"])
        w.writeheader()
        w.writerows(rows)
    print(f"  hsic_results.csv: {len(rows)} строк")


def physiotypes():
    src = os.path.join(RESULTS_DIR, "info_geometry_mast.json")
    dst = os.path.join(RESULTS_DIR, "physiotypes.csv")
    if not os.path.exists(src):
        print("  пропуск: нет info_geometry_mast.json")
        return
    data = json.load(open(src, encoding="utf-8"))
    rows = []
    for label, pt in data["physiotypes"].items():
        n = pt["n"]
        rows.append({
            "physiotype": label, "n": n,
            "n_stress": pt["n_stress"], "pct_stress": round(100 * pt["n_stress"] / n, 1),
            "mean_slope": round(pt["mean_slope"], 4),
            "mean_logaucg": round(pt["mean_logaucg"], 4),
        })
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["physiotype", "n", "n_stress", "pct_stress",
                                          "mean_slope", "mean_logaucg"])
        w.writeheader()
        w.writerows(rows)
    print(f"  physiotypes.csv: {len(rows)} строк")


def main():
    print("Экспорт сводных таблиц:")
    bayesian_summary()
    hsic_results()
    physiotypes()


if __name__ == "__main__":
    main()
