# topo-stress

[English](README.md) · [Русский](README.ru.md) · **Deutsch**

> Die deutsche Version der Beschreibung befindet sich hier; die englische Version liegt in [README.md](README.md).

Multimethodale bayesianisch-topologische Analyse des akuten Labortests **MAST** (n = 371).

Das Projekt vereint **neun Analysemethoden** auf denselben Daten in einer reproduzierbaren Pipeline: topologische Datenanalyse (TDA), Extremwerttheorie (EVT), Funktionsdatenanalyse (FDA), bayesianische hierarchische Modelle, Zufallsmatrizen (RMT), den HSIC-Test für nichtlineare Abhängigkeiten, konforme Vorhersage, Informationsgeometrie und neue topologische Funktionale.

## Zentrale Ergebnisse (MAST)

| Methode | Ergebnis |
|---|---|
| TDA (d̄_topo) | Stress vs. Kontrolle signifikant getrennt, p < 0.002 (500 Permutationen); vollständiger 17D-Satz: d̄_topo = 0.102 |
| Bayesianisches Modell M3 | Zeit×Gruppe = +0.054 [0.029, 0.079], 0 Divergenzen, R̂ = 1.00; BMA-Gewicht 0.87 gegenüber nicht-hierarchischem M1 (−1300) |
| EVT | Fréchet-Verteilungsenden; Permutationstest RL99: p < 0.001 |
| FDA | maxT-Permutation: p = 0.002 |
| HSIC | Anxiety_peak p = 0.036 (Pearson p = 0.123) — ein nichtlinearer Zusammenhang, der linearen Methoden verborgen bleibt |
| Informationsgeometrie | 3 Physiotypen, darunter „Super-Reagierer“ (22 Probanden, 100 % Stress, LogAUCg 6.41 vs. 3.08), χ² = 14.36, p = 0.0008 |

Die vollständige Beschreibung — in `article/mast_article.md` (und `article/mast_article.docx`).

## Projektstruktur

```
topo-stress/
├── data/
│   ├── raw/             # MAST_371_2025111002_Raw_Recode.csv (371 Probanden)
│   └── processed/       # mast_wide.csv, mast_long_cortisol.csv, mast_long_psych.csv, ...
├── src/topostress/      # Paket: topology, bayesian, evt, fda, rmt, hsic, conformal,
│                        #        info_geometry, data, utils, config
├── scripts/             # Ausführbare Skripte (Pipeline): run_all.py, run_tda.py, ...
├── figures/             # Alle Abbildungen für den Artikel
├── results/             # JSON/CSV-Ergebnisse
├── article/             # Artikel (Markdown + docx)
├── notebooks/           # Jupyter zur Datenanalyse
├── tests/               # Unit-Tests
└── docs/                # Dokumentation
```

## Installation

```bash
# Python 3.10+
pip install -r requirements.txt
pip install -e .            # optional: als Paket installieren
```

Mehr dazu: [docs/installation.md](docs/installation.md).

## Verwendung

```bash
python scripts/run_all.py          # komplette Pipeline (gecacht)
FORCE=1 python scripts/run_all.py  # alles neu berechnen
python scripts/run_tda.py          # nur TDA
python scripts/run_bayesian.py     # nur bayesianische Analyse
```

Ein Skript wird übersprungen, wenn sein Artefakt bereits in `results/` existiert.
Analyseparameter werden über Umgebungsvariablen überschrieben: `MAST_PERM`,
`MAST_DRAWS`, `MAST_TUNE`, `MAST_BOOT`, `FORCE`. Mehr: [docs/usage.md](docs/usage.md).

## Daten

- **MAST**: `MAST_371_2025111002_Raw_Recode.csv` — akuter Labortest,
  371 Probanden (240 Stress / 131 Kontrolle), Cortisol (5 Proben),
  Psychometrie, 6 experimentelle Protokolle.
- **MMASH** (nur für die Regressionsprüfung der Übertragbarkeit von Metriken):
  `data/processed/MMASH_long.csv` (abgeleitete Datei; Rohdaten und Lizenz —
  siehe [data/README.md](data/README.md)).

> WICHTIG: Die Probanden-ID in MAST ist die Spalte `SubID` (371 Stufen). Die
> Spalte `Subject` wiederholt sich zwischen den Experimenten und bezieht sich auf
> verschiedene Personen — sie wird daher nicht zur Gruppierung verwendet.

## Lizenz

MIT — siehe [LICENSE](LICENSE). Die MAST/MMASH-Daten unterliegen ihren eigenen
Lizenzen (siehe [data/README.md](data/README.md)).
