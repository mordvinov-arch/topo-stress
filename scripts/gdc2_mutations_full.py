# GDC2: полный мутационный профиль и TMB для TCGA-LUAD (cBioPortal, luad_tcga_gdc).
#
# Расширяет gdc2_mutations.csv до 11 генов (EGFR, KRAS, TP53, STK11, KEAP1, MET,
# BRAF, ROS1, ALK, NF1, RBM10) и добавляет TMB (TMB_NONSYNONYMOUS) и MUTATION_COUNT
# из клинических данных cBioPortal. Сравнения: χ² физиотип x мутация (tumor-only,
# пациенты), TMB по физиотипам (Kruskal-Wallis).
#
# Выход: data/processed/gdc2_mutations_full.csv, results/gdc2_mutations_full.json,
# figures/gdc2_mutations_heatmap.png, figures/gdc2_tmb_by_physio.png.

import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

STUDY = "luad_tcga_gdc"
PROFILE = f"{STUDY}_mutations"
SAMPLE_LIST = f"{STUDY}_all"
API = "https://www.cbioportal.org/api"
MERGED = os.path.join(PROCESSED_DATA_DIR, "gdc2_clinical_merged.csv")
CACHE_JSON = os.path.join(PROCESSED_DATA_DIR, "cbioportal_luad_gdc_mutations.json")
OUT_CSV = os.path.join(PROCESSED_DATA_DIR, "gdc2_mutations_full.csv")

EXISTING = {"EGFR": 1956, "KRAS": 3845, "TP53": 7157}
NEW_GENES = {"STK11": 6794, "KEAP1": 9817, "MET": 4233, "BRAF": 673, "ROS1": 6098,
             "ALK": 238, "NF1": 4763, "RBM10": 8241}
ALL_GENES = dict(EXISTING)
ALL_GENES.update(NEW_GENES)


def fetch(url, retries=4, timeout=120):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                       "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print("retry %d for %s: %s" % (attempt + 1, url, exc), flush=True)
            time.sleep(3 * (attempt + 1))


def main():
    raw = json.load(open(CACHE_JSON, encoding="utf-8"))
    for gene, entrez in NEW_GENES.items():
        url = (f"{API}/molecular-profiles/{PROFILE}/mutations"
               f"?sampleListId={SAMPLE_LIST}&molecularProfileId={PROFILE}"
               f"&entrezGeneId={entrez}")
        muts = fetch(url)
        raw[gene] = muts
        print(f"{gene}: {len(muts)} событий", flush=True)
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=1)

    # TMB из клинических данных cBioPortal
    clin = fetch(f"{API}/studies/{STUDY}/clinical-data")
    tmb_map, mc_map = {}, {}
    for r in clin:
        attr = r.get("clinicalAttributeId")
        pid = (r.get("patientId") or "").upper()
        if not pid or len(pid) < 12:
            continue
        val = r.get("value")
        try:
            if attr == "TMB_NONSYNONYMOUS":
                tmb_map[pid] = float(val)
            elif attr == "MUTATION_COUNT":
                mc_map[pid] = int(float(val))
        except (TypeError, ValueError):
            pass
    print("TMB_NONSYNONYMOUS кейсов:", len(tmb_map), " MUTATION_COUNT:", len(mc_map), flush=True)

    # статусы по кейсам
    status = {}
    for gene, muts in raw.items():
        cases = set()
        for m in muts:
            sid = m.get("sampleId") or ""
            if len(sid) >= 12:
                cases.add(sid[:12].upper())
        status[gene] = cases

    merged = pd.read_csv(MERGED)
    patients = merged[merged["tissue"] == "Tumor"].drop_duplicates(subset="case_id")
    cases = patients["case_id"].astype(str).str.upper().tolist()

    rows = {"case_id": cases}
    for gene in ALL_GENES:
        rows[gene] = [1 if c in status[gene] else 0 for c in cases]
    rows["TMB"] = [tmb_map.get(c, np.nan) for c in cases]
    rows["mutation_count"] = [mc_map.get(c, np.nan) for c in cases]
    mut = pd.DataFrame(rows)
    mut["physiotype"] = patients["physiotype"].astype(int).values
    mut = mut.sort_values("case_id").reset_index(drop=True)
    mut.to_csv(OUT_CSV, index=False)

    res = {"method": "Full mutation profile (11 genes) + TMB by physiotype",
           "n_patients": len(mut), "tmb_available": int(mut["TMB"].notna().sum())}
    res["gene_frequency_pct"] = {g: round(float(mut[g].mean() * 100), 1) for g in ALL_GENES}

    # χ² физиотип x мутация (по каждому гену с частотой >= 5%)
    from scipy.stats import chi2_contingency
    res["chisq_physio_vs_mutation"] = {}
    heat = np.zeros((len(ALL_GENES), 3))
    gnames = list(ALL_GENES)
    for i, g in enumerate(gnames):
        ct = pd.crosstab(mut["physiotype"], mut[g])
        if ct.shape == (3, 2) and ct[1].min() > 0 and ct[1].sum() >= 10:
            chi2, p, dof, _ = chi2_contingency(ct)
            res["chisq_physio_vs_mutation"][g] = {"chi2": round(float(chi2), 2),
                                                  "p": float(p), "dof": int(dof),
                                                  "freq_by_physio_pct": {str(k): round(float(v), 1)
                                                                         for k, v in ct[1].to_dict().items()}}
        else:
            res["chisq_physio_vs_mutation"][g] = {"skipped": "too rare",
                                                  "n_mut": int(mut[g].sum())}
        for k in [1, 2, 3]:
            sub = mut[mut["physiotype"] == k]
            heat[i, k - 1] = sub[g].mean() * 100
        print("%-6s n=%d  physio freq %%: %s" % (g, int(mut[g].sum()),
                                                 np.round(heat[i], 1).tolist()), flush=True)

    # TMB по физиотипам
    from scipy.stats import kruskal
    tmb_t = mut.dropna(subset=["TMB"])
    res["tmb_by_physio"] = {"n": int(len(tmb_t))}
    groups = [tmb_t.loc[tmb_t["physiotype"] == k, "TMB"].values for k in [1, 2, 3]]
    h, p = kruskal(*groups)
    res["tmb_by_physio"]["kruskal_H"] = round(float(h), 3)
    res["tmb_by_physio"]["p"] = float(p)
    res["tmb_by_physio"]["median_by_physio"] = {str(k): float(np.median(g))
                                                for k, g in zip([1, 2, 3], groups)}
    print("TMB: H=%.3f p=%.4g medians=%s" % (h, p, {k: round(float(np.median(g)), 2)
                                                     for k, g in zip([1, 2, 3], groups)}), flush=True)

    with open(os.path.join(RESULTS_DIR, "gdc2_mutations_full.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ===== рисунки =====
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(3)); ax.set_xticklabels(["PH1", "PH2", "PH3"])
    ax.set_yticks(range(len(gnames))); ax.set_yticklabels(gnames)
    for i in range(len(gnames)):
        for j in range(3):
            ax.text(j, i, "%.1f" % heat[i, j], ha="center", va="center", fontsize=7)
    ax.set_title("Mutation frequency %% by physiotype (TCGA-LUAD tumor patients)")
    fig.colorbar(im, ax=ax, label="% mutated")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_mutations_heatmap.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bp = ax.boxplot(groups, labels=["PH1", "PH2", "PH3"], patch_artist=True)
    for patch, c in zip(bp["boxes"], ["seagreen", "darkorange", "slateblue"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("TMB (nonsynonymous mutations / Mb)")
    ax.set_title("TMB by physiotype (KW p=%.3g)" % p)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_tmb_by_physio.png"), dpi=150)
    plt.close(fig)
    print("saved gdc2_mutations_full.json + figures", flush=True)


if __name__ == "__main__":
    main()
