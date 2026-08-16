# GDC: GSEA (gseapy.prerank) для каждого физиотипа.
# Ранговая метрика: z-эффект гена в физиотипе vs остальные (log1p CPM).
# Ген-сеты: GO-BP, KEGG, Reactome, Hallmark (скачиваются из Enrichr).

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

COUNTS_NPZ = os.path.join(PROCESSED_DATA_DIR, "gdc_counts.npz")
LABELS = os.path.join(PROCESSED_DATA_DIR, "gdc2_labels.csv")
GSET_DIR = os.path.join(DATA_DIR, "genesets")
OUT = os.path.join(RESULTS_DIR, "gdc2_gsea.csv")

LIBS = {
    "GO_Biological_Process_2021": "GO_Biological_Process_2021",
    "KEGG_2021_Human": "KEGG_2021_Human",
    "Reactome_2022": "Reactome_2022",
    "MSigDB_Hallmark_2020": "MSigDB_Hallmark_2020",
}


def download_gmts():
    os.makedirs(GSET_DIR, exist_ok=True)
    import requests
    out = {}
    for key, lib in LIBS.items():
        path = os.path.join(GSET_DIR, key + ".gmt")
        if not os.path.exists(path):
            url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={lib}"
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(path, "w", encoding="utf-8") as f:
                f.write(r.text)
            print("downloaded", key, len(r.text), "bytes", flush=True)
        else:
            print("cached", key, flush=True)
        out[key] = path
    return out


def main():
    d = np.load(COUNTS_NPZ, allow_pickle=True)
    counts = d["counts"].astype(float)  # genes x samples
    genes = list(d["genes"])
    samples = list(d["samples"])

    labels = pd.read_csv(LABELS)
    s2p = dict(zip(labels["sample"], labels["physiotype"]))
    physio = np.array([s2p[s] for s in samples])
    print("physiotype counts:", pd.Series(physio).value_counts().to_dict(), flush=True)

    cpm = counts * 1e6 / counts.sum(axis=0)
    expr = np.log1p(cpm)  # genes x samples
    print("expr:", expr.shape, flush=True)

    rnks = {}
    mean_all = expr.mean(axis=1)
    sd_all = expr.std(axis=1) + 1e-12
    for cl in sorted(set(physio)):
        m = physio == cl
        eff = (expr[:, m].mean(axis=1) - mean_all) / sd_all
        rnk = pd.DataFrame({"gene": genes, "z": eff}).dropna()
        rnk = rnk[rnk["z"].notna() & (rnk["gene"] != "")]
        rnk = rnk.drop_duplicates(subset="gene", keep="first")
        rnk = rnk.set_index("gene")["z"]
        rnks[cl] = rnk
        print("physiotype", cl, "rnk genes:", len(rnk), flush=True)

    gmt_paths = download_gmts()
    import gseapy
    rows = []
    for cl, rnk in rnks.items():
        for key, path in gmt_paths.items():
            try:
                prerank = gseapy.prerank(rnk=rnk, gene_sets=path,
                                         min_size=5, max_size=500,
                                         permutation_num=int(os.environ.get("GSEA_PERM", 200)),
                                         seed=42,
                                         outdir=None, threads=8,
                                         no_plot=True)
                res = prerank.res2d
                if res is None or len(res) == 0:
                    continue
                res = res.copy()
                res["physiotype"] = cl
                res["library"] = key
                rows.append(res)
                n_sig = int((res["FDR q-val"] < 0.25).sum())
                print("ph%d %s: enriched=%d" % (cl, key, n_sig), flush=True)
            except Exception as e:
                print("ERR ph%d %s: %s" % (cl, key, str(e)[:120]), flush=True)
    if not rows:
        print("no GSEA results", flush=True)
        return
    allres = pd.concat(rows, ignore_index=True)
    allres.to_csv(OUT, index=False)
    print("saved:", OUT, len(allres), "rows", flush=True)

    sig = allres[allres["FDR q-val"] < 0.25]
    for cl in sorted(set(physio)):
        sub = sig[sig["physiotype"] == cl]
        print("\n== physiotype %d (FDR<0.25): %d terms ==" % (cl, len(sub)), flush=True)
        if len(sub):
            print(sub[["library", "Term", "NES", "FDR q-val"]].sort_values("FDR q-val").head(8).to_string(), flush=True)


if __name__ == "__main__":
    main()
