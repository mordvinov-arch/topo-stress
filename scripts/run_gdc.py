# GDC: прогон конвейера topo-stress на загруженном датасете
# (TCGA RNA-seq STAR gene counts, 601 файлов .tsv, все open-access).
# Группировка для двухгрупповых тестов: знак первой главной компоненты
# (детерминированный сплит "PC1+" vs "PC1-"), т.к. фенотип в манифесте GDC
# отсутствует. Параметры: MAST_PERM, MAST_EPS, MAST_DRAWS, MAST_TUNE.

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from topostress import hsic, rmt, topology, utils  # noqa: E402
from topostress.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR  # noqa: E402

GDC_DIR = os.path.join(DATA_DIR, "gdc")
OUT_WIDE = os.path.join(DATA_DIR, "processed", "gdc_wide.csv")

N_GENES = int(os.environ.get("MAST_N_GENES", 500))
N_PERM = int(os.environ.get("MAST_PERM", 199))
N_EPS = int(os.environ.get("MAST_EPS", 100))
N_BOOT = int(os.environ.get("MAST_BOOT", 50))
DRAWS = int(os.environ.get("MAST_DRAWS", 500))
TUNE = int(os.environ.get("MAST_TUNE", 500))

RNG = np.random.default_rng(42)


def save_json(name, payload):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print("saved", name, flush=True)


def load_expression_matrix():
    files = sorted(f for f in os.listdir(GDC_DIR) if f.endswith(".tsv"))
    print("loading %d files..." % len(files), flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def read_one(fn):
        fp = os.path.join(GDC_DIR, fn)
        df = pd.read_csv(fp, sep="\t", comment="#",
                         usecols=["gene_name", "tpm_unstranded"])
        df = df.dropna()
        df = df[df["gene_name"] != ""]
        df = df.drop_duplicates(subset="gene_name", keep="first")
        return df.set_index("gene_name")["tpm_unstranded"]

    with ThreadPoolExecutor(max_workers=8) as ex:
        series = list(ex.map(read_one, files))
    mat = pd.concat(series, axis=1, join="inner")
    mat.columns = files
    print("matrix:", mat.shape, flush=True)
    return mat  # genes x samples


def main():
    t0 = time.time()
    mat = load_expression_matrix()

    expr = np.log1p(mat.values.T)  # samples x genes
    sample_ids = list(mat.columns)
    gene_names = list(mat.index)

    var = expr.var(axis=0)
    order = np.argsort(var)[::-1][:N_GENES]
    X = expr[:, order]
    hvg = [gene_names[i] for i in order]
    print("reduced to %d HVG; first: %s" % (N_GENES, hvg[:3]), flush=True)

    wide = pd.DataFrame(X, index=sample_ids, columns=hvg)
    wide.index.name = "sample_id"
    os.makedirs(os.path.dirname(OUT_WIDE), exist_ok=True)
    wide.to_csv(OUT_WIDE)
    print("wide table ->", OUT_WIDE, flush=True)

    # Детерминированная группа: знак PC1
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    pc1 = Z @ vt[0]
    group = np.where(pc1 > np.median(pc1), "PC1+", "PC1-")
    g1, g2 = group == "PC1+", group == "PC1-"
    print("groups: PC1+ = %d, PC1- = %d" % (g1.sum(), g2.sum()), flush=True)

    # ===== TDA =====
    print("[1/6] TDA...", flush=True)
    t1 = time.time()
    d_topo, t_grid, b1, b2 = topology.d_topo_normalized(X[g1], X[g2], n_eps=N_EPS)
    d_comb, _, _ = topology.d_combined(X[g1], X[g2], lam1=0.5, lam2=0.5, n_eps=N_EPS)
    obs, p_topo, _ = utils.permutation_test(
        lambda A, B: topology.d_topo_normalized(A, B, n_eps=N_EPS)[0],
        X[g1], X[g2], n_perm=N_PERM, seed=42)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_grid, b1, label="PC1+", lw=2)
    ax.plot(t_grid, b2, label="PC1-", lw=2)
    ax.set_xlabel("normalized scale t")
    ax.set_ylabel("beta0(t) / n")
    ax.set_title("GDC: normalized Betti-0 curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc_tda.png"), dpi=150)
    plt.close(fig)
    save_json("gdc_tda.json", {
        "method": "TDA d_topo (normalized Betti-0)", "n": len(sample_ids),
        "n_genes": N_GENES, "n_perm": N_PERM, "n_eps": N_EPS,
        "d_topo": d_topo, "d_combined": d_comb,
        "p_permutation": float(p_topo),
        "beta1_max": float(max(b1)), "beta2_max": float(max(b2)),
    })
    print("  d_topo=%.4f p=%.3f (%.0fs)" % (d_topo, p_topo, time.time() - t1), flush=True)

    # ===== HSIC =====
    print("[2/6] HSIC...", flush=True)
    t1 = time.time()
    x_g = X[:, 0]
    y_g = X[:, 1]
    hs, p_hsic = hsic.hsic_test_median(x_g, y_g, n_perm=N_PERM, seed=42)
    r_pear, p_pear = utils.pearson_test(x_g, y_g)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x_g, y_g, s=8, c="steelblue", alpha=0.5)
    ax.set_xlabel(hvg[0]); ax.set_ylabel(hvg[1])
    ax.set_title("HSIC: %s vs %s" % (hvg[0], hvg[1]))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc_hsic.png"), dpi=150)
    plt.close(fig)
    save_json("gdc_hsic.json", {
        "method": "HSIC (median bandwidth)", "gene_x": hvg[0], "gene_y": hvg[1],
        "n_perm": N_PERM, "hsic_stat": float(hs), "p_hsic": float(p_hsic),
        "pearson_r": float(r_pear), "pearson_p": float(p_pear),
    })
    print("  HSIC p=%.3f vs Pearson p=%.3f (%.0fs)" % (p_hsic, p_pear, time.time() - t1), flush=True)

    # ===== RMT =====
    print("[3/6] RMT...", flush=True)
    t1 = time.time()
    spec = rmt.correlation_spectrum(X)
    lam_plus = rmt.marchenko_pastur_bound(X.shape[1], X.shape[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(spec, bins=40, density=True, alpha=0.7, color="darkorchid")
    ax.axvline(lam_plus, color="crimson", ls="--", lw=2, label="MP lambda+ = %.2f" % lam_plus)
    ax.legend()
    ax.set_xlabel("eigenvalue"); ax.set_ylabel("density")
    ax.set_title("GDC: correlation spectrum vs Marchenko-Pastur")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc_rmt.png"), dpi=150)
    plt.close(fig)
    frac_above = float((spec > lam_plus).mean())
    save_json("gdc_rmt.json", {
        "method": "RMT: Marchenko-Pastur", "p": X.shape[1], "n": X.shape[0],
        "lambda_plus": lam_plus, "lambda_max": float(spec[0]),
        "fraction_above_bound": frac_above,
    })
    print("  lambda_max=%.2f bound=%.2f above=%.1f%% (%.0fs)" % (spec[0], lam_plus, 100 * frac_above, time.time() - t1), flush=True)

    # ===== EVT =====
    print("[4/6] EVT...", flush=True)
    t1 = time.time()
    from scipy.stats import genextreme
    from topostress.evt import gev_return_level
    maxima = X.max(axis=1)  # максимум log1p-TPM по генам в каждом образце
    xi, mu, sigma = genextreme.fit(maxima)
    rl99 = gev_return_level(xi, mu, sigma, 0.01)
    tail = "Frechet (heavy)" if xi > 0.2 else ("Gumbel" if xi > -0.2 else "Weibull (bounded)")
    fig, ax = plt.subplots(figsize=(6, 4))
    xx = np.linspace(maxima.min() - 0.1, rl99 * 1.05, 200)
    ax.hist(maxima, bins=40, density=True, alpha=0.6, color="teal")
    ax.plot(xx, genextreme.pdf(xx, xi, loc=mu, scale=sigma), "k-", lw=2, label="GEV fit")
    ax.axvline(rl99, color="crimson", ls="--", label="RL99 = %.2f" % rl99)
    ax.legend(); ax.set_xlabel("max log1p(TPM) per sample"); ax.set_ylabel("density")
    ax.set_title("GDC: GEV on sample maxima")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc_evt.png"), dpi=150)
    plt.close(fig)
    save_json("gdc_evt.json", {
        "method": "EVT: GEV on sample maxima", "n": int(len(maxima)),
        "xi": float(xi), "mu": float(mu), "sigma": float(sigma),
        "tail_type": tail, "rl99": float(rl99),
    })
    print("  xi=%.3f (%s) RL99=%.2f (%.0fs)" % (xi, tail, rl99, time.time() - t1), flush=True)

    # ===== Байесовская модель =====
    print("[5/6] Bayesian...", flush=True)
    t1 = time.time()
    import bambi
    y_top = X[:, 0]
    bdata = pd.DataFrame({"y": y_top, "Group": group})
    model = bambi.Model("y ~ 1 + Group", bdata)
    fit = model.fit(draws=DRAWS, tune=TUNE, target_accept=0.99, seed=42,
                    progressbar=False)
    import arviz as az
    eff_var = next(v for v in fit.posterior.data_vars if v.startswith("Group"))
    samples = fit.posterior[eff_var].values.reshape(-1)
    eff_mean = float(samples.mean())
    lo, hi = az.hdi(samples, prob=0.94)
    eff_hdi = [float(lo), float(hi)]
    rhat = float(az.rhat(fit.posterior[eff_var]).max().values)
    div = int(fit.sample_stats.diverging.sum())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=50, color="seagreen", alpha=0.8)
    ax.axvline(0, color="crimson", ls="--")
    ax.set_xlabel("%s effect (log1p TPM)" % eff_var); ax.set_ylabel("posterior density")
    ax.set_title("GDC: Bayesian model, %s" % hvg[0])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc_bayesian.png"), dpi=150)
    plt.close(fig)
    save_json("gdc_bayesian.json", {
        "method": "Bambi gaussian: y ~ 1 + Group", "outcome": hvg[0],
        "draws": DRAWS, "tune": TUNE, "effect_mean": eff_mean,
        "effect_hdi_94": eff_hdi, "rhat": rhat, "divergences": div,
    })
    print("  effect=%.3f [%.3f, %.3f] rhat=%.3f div=%d (%.0fs)"
          % (eff_mean, eff_hdi[0], eff_hdi[1], rhat, div, time.time() - t1), flush=True)

    # ===== Информационная геометрия =====
    print("[6/6] Info geometry...", flush=True)
    t1 = time.time()
    from scipy.stats import chi2_contingency
    from sklearn.manifold import MDS as SKMDS
    from topostress.info_geometry import wasserstein_matrix, ward_clusters
    sub = min(200, X.shape[1])
    dists = (X[:, :sub] - X[:, :sub].min(axis=1, keepdims=True))
    dists = dists / (dists.sum(axis=1, keepdims=True) + 1e-12)
    samples = [dists[i] for i in range(len(dists))]
    D = wasserstein_matrix(samples)
    n_clust = 3
    labels, Z = ward_clusters(D, n_clust)
    ct = pd.crosstab(group, labels)
    chi2, p_chi, _, _ = chi2_contingency(ct)
    mds = SKMDS(n_components=2, dissimilarity="precomputed", random_state=42,
                n_init=5, normalized_stress="auto")
    emb = mds.fit_transform(D)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for cl in sorted(set(labels)):
        m = labels == cl
        ax.scatter(emb[m, 0], emb[m, 1], s=20, label="physiotype %d" % cl, alpha=0.7)
    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2")
    ax.set_title("GDC: physiotypes (Wasserstein + Ward)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc_info_geometry.png"), dpi=150)
    plt.close(fig)
    save_json("gdc_info_geometry.json", {
        "method": "Info geometry: Wasserstein+Ward physiotypes",
        "n_clusters": n_clust, "n_genes_sub": sub,
        "contingency": ct.reset_index().to_dict("records"),
        "chi2": float(chi2), "p_chi2": float(p_chi),
    })
    print("  chi2=%.2f p=%.4f (%.0fs)" % (chi2, p_chi, time.time() - t1), flush=True)

    print("==== GDC pipeline done in %.0fs ====" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
