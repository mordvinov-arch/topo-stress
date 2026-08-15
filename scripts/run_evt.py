# MAST: теория экстремальных значений (EVT).
# GEV по блочным максимумам (пик кортизола за 5 сэмплов) по группам,
# Peaks-Over-Threshold (GPD), return levels, пермутационный тест различия хвостов,
# корреляционная сеть как прокси «причинной» структуры.

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from topostress.config import N_PERM, PROC_WIDE, RESULTS_DIR, FIGURES_DIR
from topostress.evt import gev_return_level, gpd_return_level, tail_type

WIDE = PROC_WIDE
RNG = np.random.default_rng(42)


def main():
    df = pd.read_csv(WIDE)
    df = df.dropna(subset=["Cortisol_peak"])
    df["Group01"] = (df["Group"] == "Stress").astype(int)
    print(f"Испытуемых: {len(df)}, Stress={int(df.Group01.sum())}, "
          f"Control={int((1 - df.Group01).sum())}")

    peaks = df["Cortisol_peak"].values
    peaks_s = df.loc[df.Group01 == 1, "Cortisol_peak"].values
    peaks_c = df.loc[df.Group01 == 0, "Cortisol_peak"].values

    out = {}

    # ---------- 1. GEV по группам ----------
    gev_res = {}
    for label, x in [("Stress", peaks_s), ("Control", peaks_c), ("Pooled", peaks)]:
        xi, mu, sigma = stats.genextreme.fit(x)
        q99 = gev_return_level(xi, mu, sigma, 0.99)
        q999 = gev_return_level(xi, mu, sigma, 0.999)
        gev_res[label] = {
            "xi": float(xi), "mu": float(mu), "sigma": float(sigma),
            "tail": tail_type(xi), "q99": float(q99), "q999": float(q999),
        }
        print(f"GEV {label:>7}: xi={xi:+.3f} mu={mu:+.3f} sigma={sigma:+.3f} "
              f"({tail_type(xi)})  RL99={q99:.2f} RL99.9={q999:.2f}")
    out["gev"] = gev_res

    # ---------- 2. GPD (Peaks-Over-Threshold) ----------
    u = np.quantile(peaks, 0.90)
    print(f"\nПорог u = q90(пиков) = {u:.2f}")
    n_over = int((peaks > u).sum())
    gpd_res = {"u": float(u)}
    for label, x in [("Stress", peaks_s), ("Control", peaks_c)]:
        ex = x[x > u]
        if len(ex) < 20:
            print(f"GPD {label}: мало превышений ({len(ex)}), пропуск")
            gpd_res[label] = {"n_exceed": int(len(ex))}
            continue
        xi, _, scale = stats.genpareto.fit(ex - u)
        rl = gpd_return_level(scale, xi, u, len(x), 0.001)
        gpd_res[label] = {
            "n_exceed": int(len(ex)), "xi": float(xi), "scale": float(scale),
            "rl_1e-3": float(rl), "tail": tail_type(xi),
        }
        print(f"GPD {label:>7}: xi={xi:+.3f} scale={scale:+.3f} (n={len(ex)}) "
              f"return(1/1000)={rl:.2f}")
    out["gpd"] = gpd_res

    # ---------- 3. Пермутационный тест различия хвостов ----------
    # статистика: разность RL99 между Stress и Control
    def rl99_diff(x):
        s = x[x["g"] == 1]
        c = x[x["g"] == 0]
        xi1, mu1, sig1 = stats.genextreme.fit(s["v"])
        xi0, mu0, sig0 = stats.genextreme.fit(c["v"])
        return (gev_return_level(xi1, mu1, sig1, 0.99)
                - gev_return_level(xi0, mu0, sig0, 0.99))

    x = np.column_stack([peaks, df["Group01"].values])
    obs = rl99_diff(pd.DataFrame({"v": peaks, "g": df["Group01"].values}))
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        g = RNG.permutation(x[:, 1])
        perm[i] = rl99_diff(pd.DataFrame({"v": x[:, 0], "g": g}))
        if (i + 1) % 200 == 0:
            print(f"  perm {i + 1}/{N_PERM}", flush=True)
    p_two = float(2 * min((perm >= obs).mean(), (perm <= obs).mean()))
    p_gt = float((perm >= obs).mean())
    print(f"\nПермутация RL99 (Stress-Control): obs={obs:+.3f}, "
          f"p_two-sided={p_two:.3f}, p(>obs)={p_gt:.3f}")
    out["permutation"] = {
        "obs_rl99_diff": float(obs), "p_two_sided": p_two,
        "p_gt": p_gt, "n_perm": N_PERM,
        "perm_quantiles": {
            "q02.5": float(np.quantile(perm, 0.025)),
            "q50": float(np.quantile(perm, 0.5)),
            "q97.5": float(np.quantile(perm, 0.975)),
        },
    }

    # ---------- 4. Корреляционная сеть ----------
    net_vars = [
        "Group01", "Age", "BMI", "Trait_Anxiety", "Cortisol_peak",
        "Cortisol_increase", "LogAUCg", "Slope_logcortisol",
        "Anxiety_peak", "Anxiety_change", "Negative_peak", "Positive_peak",
    ]
    dn = df[net_vars].dropna()
    corr = dn.corr(method="pearson")
    G = nx.Graph()
    G.add_nodes_from(net_vars)
    edges = []
    for i, a in enumerate(net_vars):
        for b in net_vars[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.25:
                edges.append((a, b, round(float(r), 2)))
                G.add_edge(a, b, weight=round(float(r), 2))
    print(f"\nРёбра корреляционной сети (|r|>0.25): {len(edges)}")
    for a, b, r in edges:
        print(f"  {a:>20} -- {b:<20} r={r:+.2f}")
    out["network"] = {
        "vars": net_vars, "threshold": 0.25,
        "edges": [{"a": a, "b": b, "r": r} for a, b, r in edges],
    }
    for k in ["Cortisol_peak", "Anxiety_peak", "Slope_logcortisol", "Trait_Anxiety"]:
        r, p = stats.pearsonr(dn["Group01"], dn[k])
        print(f"  corr(Group01,{k:<20}) r={r:+.3f} p={p:.3f}")
        out["network"].setdefault("corr_with_group", {})[k] = {
            "r": float(r), "p": float(p)}

    with open(os.path.join(RESULTS_DIR, "evt_mast.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nСохранено: results/evt_mast.json")

    # ---------- 5. Рисунок ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    xr = np.linspace(peaks.min(), peaks.max() * 1.05, 200)
    for label, x, color in [("Stress", peaks_s, "#e74c3c"),
                            ("Control", peaks_c, "#3498db")]:
        axes[0, 0].hist(x, bins=16, density=True, alpha=0.35, color=color,
                        label=label)
        g = gev_res[label]
        axes[0, 0].plot(xr, stats.genextreme.pdf(
            xr, g["xi"], g["mu"], g["sigma"]), color=color,
            lw=2, label=f"{label} GEV (xi={g['xi']:+.2f})")
    axes[0, 0].axvline(u, color="gray", ls="--", lw=1, label=f"q90={u:.1f}")
    axes[0, 0].set_xlabel("Пик кортизола, мкг/дл")
    axes[0, 0].set_title("GEV-подгонка блочных максимумов")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(perm, bins=40, color="#95a5a6", alpha=0.7)
    axes[0, 1].axvline(obs, color="red", lw=2, label=f"наблюд. RL99 diff={obs:+.2f}")
    axes[0, 1].set_title(f"Пермутация diff RL99 (p={p_two:.3f})")
    axes[0, 1].set_xlabel("Stress RL99 - Control RL99")
    axes[0, 1].legend(fontsize=8)

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=axes[1, 0], node_size=1800,
                           node_color="#ecf0f1", edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=axes[1, 0], font_size=7)
    nx.draw_networkx_edges(G, pos, ax=axes[1, 0], edge_color="gray",
                           width=[abs(d["weight"]) * 3 for _, _, d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G, pos, ax=axes[1, 0],
                                 edge_labels={(a, b): f"{r:+.2f}" for a, b, r in edges},
                                 font_size=6)
    axes[1, 0].set_title("Корреляционная сеть (|r|>0.25)")
    axes[1, 0].axis("off")

    for label, x, color in [("Stress", peaks_s, "#e74c3c"),
                            ("Control", peaks_c, "#3498db")]:
        ex = x[x > u]
        if len(ex) >= 20:
            xi, _, scale = stats.genpareto.fit(ex - u)
            xg = np.linspace(0, ex.max() - u, 100)
            axes[1, 1].plot(u + xg, stats.genpareto.pdf(xg, xi, loc=0, scale=scale),
                            color=color, lw=2, label=f"{label} GPD (xi={xi:+.2f})")
    axes[1, 1].set_title(f"GPD хвосты выше q90={u:.1f}")
    axes[1, 1].set_xlabel("Пик кортизола")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "evt_mast.png"), dpi=150)
    print("Сохранено: figures/evt_mast.png")


if __name__ == "__main__":
    main()
