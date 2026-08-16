# GDC: DESeq2 (pydeseq2) на реальных группах Tumor (542) vs Normal (59).
# Загружает сырые counts (unstranded) из 601 TSV, строит матрицу, кэширует,
# затем запускает pydeseq2: size factors, дисперсии, Wald-тест.

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

GDC_DIR = os.path.join(DATA_DIR, "gdc")
COUNTS_NPZ = os.path.join(PROCESSED_DATA_DIR, "gdc_counts.npz")
OUT_RES = os.path.join(RESULTS_DIR, "gdc_deseq2.csv")


def load_counts():
    if os.path.exists(COUNTS_NPZ):
        d = np.load(COUNTS_NPZ, allow_pickle=True)
        return d["counts"], list(d["genes"]), list(d["samples"])
    files = sorted(f for f in os.listdir(GDC_DIR)
                   if f.endswith(".tsv") and "star_gene_counts" in f)
    print("loading raw counts from %d files..." % len(files), flush=True)

    def read_one(fn):
        fp = os.path.join(GDC_DIR, fn)
        df = pd.read_csv(fp, sep="\t", comment="#", usecols=["gene_name", "unstranded"])
        df = df.dropna()
        df = df[df["gene_name"] != ""]
        df = df.drop_duplicates(subset="gene_name", keep="first")
        return df.set_index("gene_name")["unstranded"]

    with ThreadPoolExecutor(max_workers=8) as ex:
        series = list(ex.map(read_one, files))
    mat = pd.concat(series, axis=1, join="inner")
    mat.columns = files
    counts = mat.values.astype(np.int64)  # genes x samples
    genes = list(mat.index)
    samples = list(mat.columns)
    np.savez(COUNTS_NPZ, counts=counts, genes=np.array(genes, dtype=object),
             samples=np.array(samples, dtype=object))
    print("counts cached:", mat.shape, flush=True)
    return counts, genes, samples


def main():
    t0 = time.time()
    counts, genes, samples = load_counts()
    meta = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "gdc_metadata.csv"))
    fn2tissue = dict(zip(meta["file_name"], meta["tissue_type"]))
    tissue = [fn2tissue[s] for s in samples]

    counts_df = pd.DataFrame(counts.T, index=samples, columns=genes)
    metadata = pd.DataFrame({"tissue": tissue}, index=samples)
    print("counts_df:", counts_df.shape, flush=True)
    print(metadata["tissue"].value_counts().to_dict(), flush=True)

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    print("fitting DESeq2...", flush=True)
    dds = DeseqDataSet(counts=counts_df, metadata=metadata, design_factors="tissue",
                       refit_cooks=True, quiet=True)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["tissue", "Tumor", "Normal"], alpha=0.05, quiet=True)
    stats.summary()
    res = stats.results_df.copy()
    res = res.sort_values("padj", na_position="last")
    res.to_csv(OUT_RES)
    n_sig = int((res["padj"] < 0.05).sum())
    n_sig_log2 = int(((res["padj"] < 0.05) & (res["log2FoldChange"].abs() > 1)).sum())
    print("significant (padj<0.05):", n_sig, flush=True)
    print("significant & |log2FC|>1:", n_sig_log2, flush=True)
    print("top up:", list(res[res["log2FoldChange"] > 0].index[:5]), flush=True)
    print("top down:", list(res[res["log2FoldChange"] < 0].index[:5]), flush=True)
    print("saved:", OUT_RES, "in %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
