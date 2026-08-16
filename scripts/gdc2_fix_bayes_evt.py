# GDC2: пересчёт Bayesian (правильные настройки) и EVT RL99 (правильная пермутация).
# Перезаписывает gdc2_bayesian.json/.png и gdc2_evt.json.

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

from topostress.config import FIGURES_DIR, PROCESSED_DATA_DIR, RESULTS_DIR  # noqa: E402
from topostress.evt import gev_return_level  # noqa: E402

WIDE = os.path.join(PROCESSED_DATA_DIR, "gdc_wide.csv")
META = os.path.join(PROCESSED_DATA_DIR, "gdc_metadata.csv")
N_PERM = int(os.environ.get("MAST_PERM", 199))
DRAWS = int(os.environ.get("MAST_DRAWS", 500))
TUNE = int(os.environ.get("MAST_TUNE", 500))
RNG = np.random.default_rng(42)


def save_json(name, payload):
    with open(os.path.join(RESULTS_DIR, name), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print("saved", name, flush=True)


def main():
    wide = pd.read_csv(WIDE, index_col=0)
    meta = pd.read_csv(META)
    fn2 = dict(zip(meta["file_name"], meta["tissue_type"]))
    cols = list(wide.index)
    tissue = np.array([fn2[c] for c in cols])
    X = wide.values
    g1, g2 = tissue == "Tumor", tissue == "Normal"

    # ===== EVT: правильная пермутация RL99 =====
    from scipy.stats import genextreme
    maxes = X.max(axis=1)
    res = json.load(open(os.path.join(RESULTS_DIR, "gdc2_evt.json"), encoding="utf-8"))
    rl_t = res["Tumor"]["rl99"]; rl_n = res["Normal"]["rl99"]
    obs_rl = rl_t - rl_n
    n_t, n_n = int(g1.sum()), int(g2.sum())
    pool = maxes
    cnt = 0
    for _ in range(N_PERM):
        perm = RNG.permutation(pool)
        xi_t, mu_t, sg_t = genextreme.fit(perm[:n_t])
        xi_n, mu_n, sg_n = genextreme.fit(perm[n_t:])
        d = (gev_return_level(xi_t, mu_t, sg_t, 0.01)
             - gev_return_level(xi_n, mu_n, sg_n, 0.01))
        if d >= obs_rl:
            cnt += 1
    p_rl = (cnt + 1) / (N_PERM + 1)
    res["RL99_Tumor_minus_Normal_obs"] = float(obs_rl)
    res["RL99_Tumor_minus_Normal_perm_p"] = round(float(p_rl), 4)
    save_json("gdc2_evt.json", res)
    print("EVT: RL99 diff=%.2f perm p=%.4f" % (obs_rl, p_rl), flush=True)

    # ===== Bayesian: правильные настройки =====
    print("Bayesian: retry...", flush=True)
    t1 = time.time()
    import bambi
    import arviz as az
    bdata = pd.DataFrame({
        "y": X[:, 0],
        "tissue": tissue,
        "patient": [meta.set_index("file_name")["case_id"][c] for c in cols],
        "plate": [meta.set_index("file_name")["plate_id"][c] for c in cols],
    })
    model = bambi.Model("y ~ 1 + tissue + (1|patient) + (1|plate)", bdata)
    fit = model.fit(draws=DRAWS, tune=TUNE, target_accept=0.999, seed=42, progressbar=False)
    eff_var = next(v for v in fit.posterior.data_vars if v.startswith("tissue"))
    samples = fit.posterior[eff_var].values.reshape(-1)
    eff_mean = float(samples.mean())
    lo, hi = az.hdi(samples, prob=0.94)
    rhat = float(az.rhat(fit.posterior[eff_var]).max().values)
    div = int(fit.sample_stats.diverging.sum())
    res_bayes = {
        "method": "Bambi gaussian: y ~ tissue + (1|patient) + (1|plate)",
        "outcome": "SFTPC", "draws": DRAWS, "tune": TUNE, "target_accept": 0.999,
        "effect_var": eff_var, "effect_mean": eff_mean, "effect_hdi_94": [float(lo), float(hi)],
        "rhat": rhat, "divergences": div,
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=50, color="seagreen", alpha=0.8)
    ax.axvline(0, color="crimson", ls="--")
    ax.set_xlabel("tissue effect (log1p TPM)"); ax.set_ylabel("posterior density")
    ax.set_title("GDC LUAD: Bayesian (SFTPC, matched + plate)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "gdc2_bayesian.png"), dpi=150)
    plt.close(fig)
    save_json("gdc2_bayesian.json", res_bayes)
    print("  effect=%.3f [%.3f, %.3f] rhat=%.3f div=%d (%.0fs)"
          % (eff_mean, res_bayes["effect_hdi_94"][0], res_bayes["effect_hdi_94"][1],
             rhat, div, time.time() - t1), flush=True)


if __name__ == "__main__":
    main()
