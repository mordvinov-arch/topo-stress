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

Full description — in `article/mast_article.md`.

## TCGA-LUAD extension (GDC, n = 601)

The pipeline was transferred unchanged to 601 TCGA-LUAD RNA-seq samples
(542 tumours / 59 normal tissues, NCI GDC open access) and validated against
the real biological grouping.

| Method | Result |
|---|---|
| TDA (d̄_topo) | Tumour vs Normal d̄_topo = 0.600 (p < 0.005) — 22× the unsupervised PC1 split (0.027); robust to gene selection (random 500 genes: d̄_topo = 0.417) |
| RMT | λ_max/λ₊ = 22.7 (tumour) and 5.6 (normal); top eigenvector = immunoglobulin/plasma-cell module |
| HSIC | pooled SFTPC×BPIFA1 dependence was a mixture artifact — within-group p = 0.09 / 0.48 |
| EVT | Fréchet tails; RL99 higher in tumours (18.02 vs 13.14, permutation p = 0.005) |
| Bayesian (matched) | tissue effect on SFTPC = −5.62 [−6.21, −4.95], R̂ = 1.002, 0 divergences |
| DESeq2 | 27,254 DE genes (padj < 0.05); 14,769 with |log2FC| > 1; HVG ∩ top-DE = 6% (Jaccard 0.06) |
| Physiotypes | stage-independent (p = 0.76); agree with TRU/PI/PP subtypes (χ² = 14.44, p = 0.006); OS log-rank p < 0.001 (worst: immune-cold physiotype 3) |
| IGKC signature | 20-gene λ_max module: protective trend only (HR = 0.83, p = 0.15); stage significant within each physiotype, physiotype not within stage |
| External validation (GSE31210) | 3 clusters reproduce (bootstrap ARI = 0.684, IG-enriched cluster), tumour/normal d̄_topo = 0.670 reproduces; survival associations do not (power + GPL570 coverage) |

Articles:
- `article/gdc_article_real.md/.docx` — methods on real groups (Russian)
- `article/gdc_article_biology.md/.docx` — biology: DE, physiotypes, survival (Russian)
- `article/gdc_bioinformatics_ms.md/.docx` — English journal manuscript (Bioinformatics style)

Reproduce (raw GDC downloads ~2.5 GB, **not committed** — place them in `data/gdc/`):

```bash
python scripts/run_gdc_realgroups.py   # full pipeline on real groups (TDA/RMT/HSIC/EVT/Bayesian/InfoGeometry)
python scripts/gdc2_robustness.py      # gene-selection robustness + λ_max module
python scripts/run_gdc_deseq2.py       # DESeq2
python scripts/gdc_umap.py             # UMAP + k-means + physiotypes + top-50 genes
python scripts/gdc_gsea.py             # GSEA prerank (GSEA_PERM=200)
python scripts/gdc2_cmp_deseq2.py      # HVG/PC1 vs DESeq2 comparison
python scripts/gdc2_clinical.py        # stages (UCSC Xena)
python scripts/gdc2_survival.py        # Kaplan–Meier + Cox
python scripts/gdc2_survival_igkc.py   # IGKC signature, stage stratification, mutation Cox
python scripts/gdc2_subtypes.py        # TRU/PI/PP subtypes (Nature 2014) vs physiotypes
python scripts/fetch_geo_gse31210.py   # GSE31210 download/parse (DoH + curl --resolve)
python scripts/geo_validate.py         # external validation of physiotypes on GSE31210
```

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
