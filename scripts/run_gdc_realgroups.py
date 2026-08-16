# GDC: полный конвейер topo-stress на РЕАЛЬНЫХ группах (Tumor 542 vs Normal 59).
# Использует широкую таблицу log1p-TPM (500 HVG) + gdc_metadata.csv.
# Ковариаты в Bayesian: plate_id + случайный эффект пациента (matched дизайн).

import json
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress import hsic, rmt, topology, utils  # noqa: E402
from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
META = os.path.join(PROCESSED_DATA_DIR, "gdc_metadata.csv")

N_PERM = int(os.environ.get("MAST_PERM", 199))
N_EPS = int(os.environ.get("MAST_EPS", 100))
DRAWS = int(os.environ.get("MAST_DRAWS", 500))
TUNE = int(os.environ.get("MAST_TUNE", 500))
RNG = np.random.default_rng(42)


def save_json(name, payload):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print("saved", name, flush=True)


def main():
    t0 = time.time()
    wide = pd.read_csv(WIDE, index_col=0)
    meta = pd.read_csv(META)
    fn2tissue = dict(zip(meta["file_name"], meta["tissue_type"]))
    fn2plate = dict(zip(meta["file_name"], meta["plate_id"]))
    fn2case = dict(zip(meta["file_name"], meta["case_id"]))
    cols = list(wide.index)
    tissue = np.array([fn2tissue[c] for c in cols])
    plate = np.array([fn2plate[c] for c in cols])
    caseid = np.array([fn2case[c] for c in cols])
    X = wide.values  # 601 x 500
    print("matrix:", X.shape, flush=True)
    print("tissue:", pd.Series(tissue).value_counts().to_dict(), flush=True)

    g1 = tissue == "Tumor"   # 542
    g2 = tissue == "Normal"  # 59
    print("Tumor=%d Normal=%d" % (g1.sum(), g2.sum()), flush=True)

    # ===== B2: перекрытие PC1-групп с реальными =====
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    pc1 = Z @ vt[0]
    pc1sign = np.where(pc1 > np.median(pc1), "PC1+", "PC1-")
    ct = pd.crosstab(pd.Series(pc1sign, name="PC1"), pd.Series(tissue, name="Tissue"))
    save_json("gdc2_pc1_vs_tissue.json", {
        "crosstab": ct.to_dict(),
        "agreement_pct": float(np.mean((pc1sign == "PC1+") == g1) * 100),
    })
    print("PC1 x Tissue:\n", ct, flush=True)
    print("agreement with Tumor/PC1+ mapping: %.1f%%" % (np.mean((pc1sign == "PC1+") == g1) * 100), flush=True)

    # ===== TDA =====
    print("[1/6] TDA...", flush=True)
    t1 = time.time()
    d_topo, t_grid, b1, b2 = topology.d_topo_normalized(X[g1], X[g2], n_eps=N_EPS)
    d_comb, _, _ = topology.d_combined(X[g1], X[g2], lam1=0.5, lam2=0.5, n_eps=N_EPS)
    obs, p_topo, _ = utils.permutation_test(
        lambda A, B: topology.d_topo_normalized(A, B, n_eps=N_EPS)[0],
        X[g1], X[g2], n_perm=N_PERM, seed=42)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_grid, b1, label="Tumor", lw=2)
    ax.plot(t_grid, b2, label="Normal", lw=2)
    ax.set_xlabel("normalized scale t"); ax.set_ylabel("beta0(t)/n")
    ax.set_title("GDC LUAD: Betti-0 curves (Tumor vs Normal)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_tda.png"), dpi=150)
    plt.close(fig)
    save_json("gdc2_tda.json", {
        "method": "TDA d_topo (Betti-0)", "n_tumor": int(g1.sum()), "n_normal": int(g2.sum()),
        "n_genes": X.shape[1], "n_perm": N_PERM, "n_eps": N_EPS,
        "d_topo": d_topo, "d_combined": d_comb, "p_permutation": float(p_topo),
    })
    print("  d_topo=%.4f p=%.3f (%.0fs)" % (d_topo, p_topo, time.time() - t1), flush=True)

    # ===== RMT по группам =====
    print("[2/6] RMT per group...", flush=True)
    t1 = time.time()
    res_rmt = {"method": "RMT Marchenko-Pastur", "n_genes": X.shape[1]}
    for name, mask in [("Tumor", g1), ("Normal", g2)]:
        spec = rmt.correlation_spectrum(X[mask])
        lam_plus = rmt.marchenko_pastur_bound(X.shape[1], mask.sum())
        frac = float((spec > lam_plus).mean())
        res_rmt[name] = {"n": int(mask.sum()), "q": float(X.shape[1] / mask.sum()),
                         "lambda_plus": float(lam_plus), "lambda_max": float(spec[0]),
                         "fraction_above_bound": round(frac, 4)}
        print("  %s: n=%d q=%.2f lambda_max=%.2f bound=%.2f above=%.1f%%"
              % (name, mask.sum(), X.shape[1] / mask.sum(), spec[0], lam_plus, 100 * frac), flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, mask) in zip(axes, [("Tumor", g1), ("Normal", g2)]):
        spec = rmt.correlation_spectrum(X[mask])
        lam_plus = rmt.marchenko_pastur_bound(X.shape[1], mask.sum())
        ax.hist(spec, bins=40, density=True, alpha=0.7, color="darkorchid")
        ax.axvline(lam_plus, color="crimson", ls="--", lw=2, label="MP+.2f" % lam_plus)
        ax.set_title(name); ax.legend()
        ax.set_xlabel("eigenvalue"); ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_rmt.png"), dpi=150)
    plt.close(fig)
    save_json("gdc2_rmt.json", res_rmt)

    # ===== HSIC =====
    print("[3/6] HSIC...", flush=True)
    t1 = time.time()
    xg = X[:, 0]; yg = X[:, 1]
    res_hsic = {"method": "HSIC median bandwidth", "genes": ["SFTPC", "BPIFA1"]}
    lab = (tissue == "Tumor").astype(float)
    hs, p_h = hsic.hsic_test_median(lab, xg, n_perm=N_PERM, seed=42)
    res_hsic["tissue_vs_SFTPC"] = {"hsic": float(hs), "p": float(p_h)}
    hs, p_h = hsic.hsic_test_median(lab, yg, n_perm=N_PERM, seed=42)
    res_hsic["tissue_vs_BPIFA1"] = {"hsic": float(hs), "p": float(p_h)}
    for name, mask in [("Tumor", g1), ("Normal", g2)]:
        hs, p_h = hsic.hsic_test_median(xg[mask], yg[mask], n_perm=N_PERM, seed=42)
        r_pear, p_pear = utils.pearson_test(xg[mask], yg[mask])
        res_hsic[f"SFTPCxBPIFA1_{name}"] = {"hsic": float(hs), "p": float(p_h),
                                            "pearson_r": float(r_pear), "pearson_p": float(p_pear)}
        print("  %s SFTPCxBPIFA1: HSIC p=%.3f, Pearson r=%.3f p=%.3f"
              % (name, p_h, r_pear, p_pear), flush=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(xg[g1], yg[g1], s=8, c="crimson", alpha=0.4, label="Tumor")
    ax.scatter(xg[g2], yg[g2], s=8, c="steelblue", alpha=0.6, label="Normal")
    ax.set_xlabel("SFTPC (log1p TPM)"); ax.set_ylabel("BPIFA1 (log1p TPM)")
    ax.set_title("HSIC: SFTPC vs BPIFA1 by tissue"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_hsic.png"), dpi=150)
    plt.close(fig)
    save_json("gdc2_hsic.json", res_hsic)

    # ===== EVT по группам =====
    print("[4/6] EVT per group...", flush=True)
    t1 = time.time()
    from scipy.stats import genextreme
    from topostress.evt import gev_return_level
    maxes = X.max(axis=1)
    res_evt = {"method": "EVT GEV on sample maxima", "n_genes": X.shape[1]}
    for name, mask in [("Tumor", g1), ("Normal", g2)]:
        xi, mu, sigma = genextreme.fit(maxes[mask])
        rl = gev_return_level(xi, mu, sigma, 0.01)
        tail = "Frechet (heavy)" if xi > 0.2 else ("Gumbel" if xi > -0.2 else "Weibull (bounded)")
        res_evt[name] = {"n": int(mask.sum()), "xi": float(xi), "mu": float(mu),
                         "sigma": float(sigma), "tail_type": tail, "rl99": float(rl)}
        print("  %s: xi=%.3f (%s) RL99=%.2f" % (name, xi, tail, rl), flush=True)
    obs_rl = res_evt["Tumor"]["rl99"] - res_evt["Normal"]["rl99"]
    rls_t = [gev_return_level(*genextreme.fit(maxes[RNG.choice(np.where(g1)[0], g1.sum(), replace=True)]), 0.01)
             for _ in range(N_PERM)]
    p_rl = float((np.array(rls_t) >= obs_rl).mean())
    res_evt["RL99_Tumor_minus_Normal_perm_p"] = round(p_rl, 4)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, mask, color in [("Tumor", g1, "crimson"), ("Normal", g2, "steelblue")]:
        xi, mu, sigma = genextreme.fit(maxes[mask])
        xx = np.linspace(maxes[mask].min() - 0.1, maxes[mask].max() + 0.1, 200)
        ax.hist(maxes[mask], bins=30, density=True, alpha=0.35, color=color, label=name)
        ax.plot(xx, genextreme.pdf(xx, xi, loc=mu, scale=sigma), color=color, lw=2)
    ax.legend(); ax.set_xlabel("max log1p(TPM)"); ax.set_ylabel("density")
    ax.set_title("EVT per tissue group")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_evt.png"), dpi=150)
    plt.close(fig)
    save_json("gdc2_evt.json", res_evt)
    print("  RL99 diff perm p=%.3f" % p_rl, flush=True)

    # ===== Bayesian: tissue + plate + случайный эффект пациента =====
    print("[5/6] Bayesian...", flush=True)
    t1 = time.time()
    import bambi
    bdata = pd.DataFrame({"y": xg, "tissue": tissue, "plate": plate, "patient": caseid})
    model = bambi.Model("y ~ 1 + tissue + plate + (1|patient)", bdata)
    fit = model.fit(draws=DRAWS, tune=TUNE, target_accept=0.99, seed=42, progressbar=False)
    import arviz as az
    eff_var = next(v for v in fit.posterior.data_vars if v.startswith("tissue"))
    samples = fit.posterior[eff_var].values.reshape(-1)
    eff_mean = float(samples.mean())
    lo, hi = az.hdi(samples, prob=0.94)
    eff_hdi = [float(lo), float(hi)]
    rhat = float(az.rhat(fit.posterior[eff_var]).max().values)
    div = int(fit.sample_stats.diverging.sum())
    res_bayes = {
        "method": "Bambi gaussian: y ~ tissue + plate + (1|patient)",
        "outcome": "SFTPC", "draws": DRAWS, "tune": TUNE,
        "effect_var": eff_var, "effect_mean": eff_mean, "effect_hdi_94": eff_hdi,
        "rhat": rhat, "divergences": div,
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=50, color="seagreen", alpha=0.8)
    ax.axvline(0, color="crimson", ls="--")
    ax.set_xlabel("tissue effect (log1p TPM)"); ax.set_ylabel("posterior density")
    ax.set_title("GDC LUAD: Bayesian model (SFTPC)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_bayesian.png"), dpi=150)
    plt.close(fig)
    save_json("gdc2_bayesian.json", res_bayes)
    print("  effect=%.3f [%.3f, %.3f] rhat=%.3f div=%d (%.0fs)"
          % (eff_mean, eff_hdi[0], eff_hdi[1], rhat, div, time.time() - t1), flush=True)

    # ===== Info geometry / физиотипы =====
    print("[6/6] Info geometry...", flush=True)
    t1 = time.time()
    from scipy.stats import chi2_contingency
    from sklearn.manifold import MDS as SKMDS
    from topostress.info_geometry import wasserstein_matrix, ward_clusters
    sub = min(200, X.shape[1])
    dists = X[:, :sub] - X[:, :sub].min(axis=1, keepdims=True)
    dists = dists / (dists.sum(axis=1, keepdims=True) + 1e-12)
    samples_d = [dists[i] for i in range(len(dists))]
    D = wasserstein_matrix(samples_d)
    labels, Z = ward_clusters(D, 3)
    ct = pd.crosstab(pd.Series(tissue, name="Tissue"), pd.Series(labels, name="Physiotype"))
    chi2, p_chi, _, _ = chi2_contingency(ct)
    mds = SKMDS(n_components=2, dissimilarity="precomputed", random_state=42,
                n_init=5, normalized_stress="auto")
    emb = mds.fit_transform(D)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cl in sorted(set(labels)):
        m = labels == cl
        axes[0].scatter(emb[m, 0], emb[m, 1], s=20, label="physiotype %d" % cl, alpha=0.7)
    axes[0].legend(); axes[0].set_title("by physiotype")
    for tname, color in [("Tumor", "crimson"), ("Normal", "steelblue")]:
        m = tissue == tname
        axes[1].scatter(emb[m, 0], emb[m, 1], s=20, c=color, alpha=0.6, label=tname)
    axes[1].legend(); axes[1].set_title("by tissue")
    for ax in axes:
        ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_info_geometry.png"), dpi=150)
    plt.close(fig)
    save_json("gdc2_info_geometry.json", {
        "method": "Info geometry Wasserstein+Ward", "n_clusters": 3, "n_genes_sub": sub,
        "contingency": ct.reset_index().to_dict("records"),
        "chi2": float(chi2), "p_chi2": float(p_chi),
    })
    print("  chi2=%.2f p=%.2e (%.0fs)" % (chi2, p_chi, time.time() - t1), flush=True)

    print("==== GDC real-groups pipeline done in %.0fs ====" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
