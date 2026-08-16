# GDC2: устойчивость d_topo к выбору генов (500 HVG / 500 случайных / 500 низковариабельных)
# и интерпретация lambda_max: гены верхнего собственного вектора по группам.

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress import topology, utils  # noqa: E402
from topostress.config import DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
META = os.path.join(PROCESSED_DATA_DIR, "gdc_metadata.csv")
GDC_DIR = os.path.join(DATA_DIR, "gdc")
CACHE = os.path.join(PROCESSED_DATA_DIR, "gdc_tpm.npz")

N_PERM = int(os.environ.get("GDC_ROB_PERM", 199))
N_EPS = int(os.environ.get("GDC_ROB_EPS", 100))
RNG = np.random.default_rng(42)


def save_json(name, payload):
    with open(os.path.join(RESULTS_DIR, name), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print("saved", name, flush=True)


def load_full_tpm():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return d["tpm"], list(d["genes"]), list(d["samples"])
    tsvs = sorted(f for f in os.listdir(GDC_DIR) if f.endswith("star_gene_counts.tsv"))
    meta = pd.read_csv(META)
    valid = set(meta["file_name"])
    tsvs = [f for f in tsvs if f in valid]
    cols = {}
    first = None
    t0 = time.time()
    for i, f in enumerate(tsvs):
        df = pd.read_csv(os.path.join(GDC_DIR, f), sep="\t",
                         usecols=["gene_id", "gene_name", "tpm_unstranded"], comment="#")
        df = df[df["gene_id"].str.startswith("ENSG")].dropna(subset=["tpm_unstranded"])
        if first is None:
            first = df[["gene_id", "gene_name"]].set_index("gene_id")
        cols[f] = df.set_index("gene_id")["tpm_unstranded"]
        if (i + 1) % 150 == 0:
            print("loaded %d/%d (%.0fs)" % (i + 1, len(tsvs), time.time() - t0), flush=True)
    S = pd.DataFrame(cols)  # genes x samples (union)
    S = S.dropna(axis=0)  # complete-case genes
    S = S.loc[np.all(np.isfinite(S.values), axis=1)]
    genes = S.index
    samples = list(S.columns)
    tpm = S.values.astype(np.float64).T  # samples x genes
    np.savez_compressed(CACHE, tpm=tpm, genes=np.array(genes, dtype=object),
                        samples=np.array(samples, dtype=object))
    return tpm, list(genes), samples


def main():
    t0 = time.time()
    meta = pd.read_csv(META)
    fn2tissue = dict(zip(meta["file_name"], meta["tissue_type"]))
    wide = pd.read_csv(WIDE, index_col=0)
    hvg_ref = list(wide.columns)

    print("loading full TPM...", flush=True)
    tpm, genes, samples = load_full_tpm()
    tissue = np.array([fn2tissue[s] for s in samples])
    g1 = tissue == "Tumor"
    g2 = tissue == "Normal"
    print("full:", tpm.shape, "Tumor=%d Normal=%d" % (g1.sum(), g2.sum()), flush=True)

    Xfull = np.log1p(tpm)
    v = Xfull.var(axis=0)
    order = np.argsort(v)[::-1]
    hvg_computed = [genes[i] for i in order[:500]]
    overlap = len(set(hvg_computed) & set(hvg_ref))
    print("top-500 variance overlaps wide HVG: %d/500" % overlap, flush=True)

    lowvar = [genes[i] for i in order[-500:]]
    rnd = [genes[i] for i in RNG.permutation(len(genes))[:500]]
    gene_sets = {
        "HVG_top500": hvg_computed,
        "random_500": rnd,
        "lowest_variance_500": lowvar,
    }

    res = {"method": "d_topo robustness to gene selection", "n_perm": N_PERM,
           "n_eps": N_EPS, "hvg_overlap_with_wide": overlap}
    for name, gs in gene_sets.items():
        idx = [genes.index(g) for g in gs]
        Xs = Xfull[:, idx]
        t1 = time.time()
        d, _, _, _ = topology.d_topo_normalized(Xs[g1], Xs[g2], n_eps=N_EPS)
        _, p, _ = utils.permutation_test(
            lambda A, B: topology.d_topo_normalized(A, B, n_eps=N_EPS)[0],
            Xs[g1], Xs[g2], n_perm=N_PERM, seed=42)
        res[name] = {"d_topo": round(float(d), 4), "p": round(float(p), 4)}
        print("  %-22s d_topo=%.4f p=%.3f (%.0fs)" % (name, d, p, time.time() - t1), flush=True)
    save_json("gdc2_robustness.json", res)

    # ===== lambda_max: верхний собственный вектор по группам =====
    Z = wide.values
    col_genes = list(wide.columns)
    lmax = {"method": "RMT top eigenvector (correlation matrix)", "n_genes": Z.shape[1]}
    for name, mask in [("Tumor", g1), ("Normal", g2)]:
        X = Z[mask]
        C = np.corrcoef(X, rowvar=False)
        w, V = np.linalg.eigh(C)
        k = int(np.argmax(w))
        lmax[name] = {
            "n": int(mask.sum()), "lambda_max": round(float(w[k]), 3),
            "pve_of_trace_pct": round(float(w[k] / len(w) * 100), 2),
            "top20_genes": [(col_genes[i], round(float(V[i, k]), 4))
                            for i in np.argsort(np.abs(V[:, k]))[::-1][:20]],
        }
        print("  %s lambda_max=%.2f (%.1f%% trace) top: %s" %
              (name, w[k], w[k] / len(w) * 100,
               ", ".join(t[0] for t in lmax[name]["top20_genes"][:8])), flush=True)
    save_json("gdc2_lmax.json", lmax)

    print("==== done in %.0fs ====" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
