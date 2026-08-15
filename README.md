# topo-stress

[Русский](README.ru.md) · [Deutsch](README.de.md) · **English**

Multi-method Bayesian-topological analysis of the acute laboratory **MAST** stress test (n = 371).

The project combines **nine analytical methods** on a single dataset in one reproducible pipeline: topological data analysis (TDA), extreme value theory (EVT), functional data analysis (FDA), Bayesian hierarchical models, random matrix theory (RMT), the HSIC test for nonlinear dependencies, conformal prediction, information geometry, and new topological functionals.

## Key results (MAST)

| Method | Result |
|---|---|
| TDA (d̄_topo) | Stress vs Control significantly separated, p < 0.002 (500 permutations); full 17D set: d̄_topo = 0.102 |
| Bayesian model M3 | Time×Group = +0.054 [0.029, 0.079], 0 divergences, R̂ = 1.00; BMA weight 0.87 vs non-hierarchical M1 (−1300) |
| EVT | Fréchet tails; RL99 permutation test: p < 0.001 |
| FDA | maxT permutation: p = 0.002 |
| HSIC | Anxiety_peak p = 0.036 (Pearson p = 0.123) — a nonlinear link invisible to linear methods |
| Info geometry | 3 physiotypes, incl. "super-responders" (22 subjects, 100% Stress, LogAUCg 6.41 vs 3.08), χ² = 14.36, p = 0.0008 |

Full description — in `article/mast_article.md` (and `article/mast_article.docx`).

## Repository structure

```
topo-stress/
├── data/
│   ├── raw/             # MAST_371_2025111002_Raw_Recode.csv (371 subjects)
│   └── processed/       # mast_wide.csv, mast_long_cortisol.csv, mast_long_psych.csv, ...
├── src/topostress/      # Package: topology, bayesian, evt, fda, rmt, hsic, conformal,
│                        #        info_geometry, data, utils, config
├── scripts/             # Runnable scripts (pipeline): run_all.py, run_tda.py, ...
├── figures/             # All figures for the article
├── results/             # JSON/CSV results
├── article/             # Article (markdown + docx)
├── notebooks/           # Jupyter for exploration
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Installation

```bash
# Python 3.10+
pip install -r requirements.txt
pip install -e .            # optional: install as a package
```

More: [docs/installation.md](docs/installation.md).

## Usage

```bash
python scripts/run_all.py          # full pipeline (cached)
FORCE=1 python scripts/run_all.py  # recompute everything
python scripts/run_tda.py          # TDA only
python scripts/run_bayesian.py     # Bayesian analysis only
```

A script is skipped if its artifact already exists in `results/`. Analysis
parameters are overridden by environment variables: `MAST_PERM`, `MAST_DRAWS`,
`MAST_TUNE`, `MAST_BOOT`, `FORCE`. More: [docs/usage.md](docs/usage.md).

## Data

- **MAST**: `MAST_371_2025111002_Raw_Recode.csv` — acute laboratory stress test,
  371 subjects (240 Stress / 131 Control), cortisol (5 samples), psychometrics,
  6 experimental protocols.
- **MMASH** (used only by the regression check of metric transferability):
  `data/processed/MMASH_long.csv` (derived file; raw data and license — see
  [data/README.md](data/README.md)).

> IMPORTANT: the subject identifier in MAST is the column `SubID` (371 levels).
> The column `Subject` repeats across experiments and refers to different people,
> so it is not used for grouping.

## License

MIT — see [LICENSE](LICENSE). The MAST/MMASH data are distributed under their
own licenses (see [data/README.md](data/README.md)).
