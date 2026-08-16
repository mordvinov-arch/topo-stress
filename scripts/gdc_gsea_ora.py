# GDC: быстрый путь-обогатительный анализ (ORA, Enrichr API) для физиотипов.
# Топ-200 генов по z-эффекту каждого физиотипа против GO-BP, KEGG, Reactome, Hallmark.

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

COUNTS_NPZ = os.path.join(PROCESSED_DATA_DIR, "gdc_counts.npz")
LABELS = os.path.join(PROCESSED_DATA_DIR, "gdc2_labels.csv")
OUT = os.path.join(RESULTS_DIR, "gdc2_gsea_ora.csv")

LIBS = ["GO_Biological_Process_2021", "KEGG_2021_Human",
        "Reactome_2022", "MSigDB_Hallmark_2020"]


def main():
    d = np.load(COUNTS_NPZ, allow_pickle=True)
    counts = d["counts"].astype(float)
    genes = list(d["genes"])
    samples = list(d["samples"])
    labels = pd.read_csv(LABELS)
    s2p = dict(zip(labels["sample"], labels["physiotype"]))
    physio = np.array([s2p[s] for s in samples])

    cpm = counts * 1e6 / counts.sum(axis=0)
    expr = np.log1p(cpm)
    mean_all = expr.mean(axis=1)
    sd_all = expr.std(axis=1) + 1e-12

    import gseapy
    rows = []
    for cl in sorted(set(physio)):
        m = physio == cl
        eff = (expr[:, m].mean(axis=1) - mean_all) / sd_all
        df = pd.DataFrame({"gene": genes, "z": eff}).dropna()
        df = df.drop_duplicates(subset="gene", keep="first")
        top = df.nlargest(200, "z")["gene"].tolist()
        for lib in LIBS:
            try:
                r = gseapy.enrichr(gene_list=top, gene_sets=lib, organism="human", outdir=None)
                res = r.res2d
                if res is None or len(res) == 0:
                    continue
                res = res.copy()
                res["physiotype"] = cl
                res["library"] = lib
                rows.append(res)
                n = int((res["Adjusted P-value"] < 0.05).sum())
                print("ph%d %s: adj<0.05 = %d" % (cl, lib, n), flush=True)
            except Exception as e:
                print("ERR ph%d %s: %s" % (cl, lib, str(e)[:100]), flush=True)
    if not rows:
        print("no results", flush=True)
        return
    allres = pd.concat(rows, ignore_index=True)
    allres.to_csv(OUT, index=False)
    sig = allres[allres["Adjusted P-value"] < 0.05]
    print("saved:", OUT, "| significant:", len(sig), flush=True)
    for cl in sorted(set(physio)):
        sub = sig[sig["physiotype"] == cl]
        print("\n== physiotype %d (adj p<0.05): %d terms ==" % (cl, len(sub)), flush=True)
        if len(sub):
            print(sub.sort_values("Adjusted P-value")[["library", "Term", "Adjusted P-value"]]
                  .head(10).to_string(), flush=True)


if __name__ == "__main__":
    main()
